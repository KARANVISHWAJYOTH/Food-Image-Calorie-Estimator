"""Train the multi-task food CNN."""

import argparse
import csv
from datetime import datetime
import logging
from math import isfinite
from pathlib import Path
import random
import sys
from typing import Any
import warnings

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


ML_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ML_DIR.parent
CONFIG_PATH = ML_DIR / "configs" / "base.yaml"
LOGGER = logging.getLogger("ml.training")


if __package__ in (None, ""):
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ml.data.food_dataset import Food101NutritionDataset
from ml.data.splits import load_or_create_splits
from ml.models.checkpoint import save_checkpoint
from ml.models.multitask_cnn import MultiTaskCNN, freeze_backbone, unfreeze_backbone
from ml.preprocessing.transforms import get_eval_transform, get_train_transform
from ml.training.losses import MultiTaskLoss


class ManifestDataset(Dataset):
    """View global manifest indices with a transform specific to each split."""

    def __init__(
        self,
        samples: list[tuple],
        indices: list[int],
        transform: Any,
    ) -> None:
        self.samples = samples
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        image_path, class_index, nutrition_values = self.samples[self.indices[index]]
        if not image_path.is_file():
            raise FileNotFoundError(f"Image from the split manifest is missing: {image_path}")
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        image_tensor = self.transform(image)
        nutrition_target = torch.tensor(nutrition_values, dtype=torch.float32)
        return image_tensor, class_index, nutrition_target


def _load_config(config_path: Path) -> dict[str, Any]:
    import yaml

    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in config file: {config_path}")
    for section in ("dataset", "model", "training"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Expected a '{section}' mapping in config file: {config_path}")
    return config


def _resolve_ml_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ML_DIR / path


def _make_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if name == "none":
        return None
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, epochs // 3),
            gamma=0.1,
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    raise ValueError(f"Unsupported scheduler: {name!r}")


def _run_epoch(
    model: MultiTaskCNN,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    amp_enabled: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    classification_loss = 0.0
    regression_loss = 0.0
    correct = 0
    num_samples = 0

    for images, class_targets, nutrition_targets in loader:
        images = images.to(device)
        class_targets = class_targets.to(device)
        nutrition_targets = nutrition_targets.to(device)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs = model(images)
                losses = criterion(outputs, class_targets, nutrition_targets)
                batch_total, batch_classification, batch_regression = losses
            if optimizer is not None:
                if amp_enabled:
                    if scaler is None:
                        raise RuntimeError("AMP is enabled for training but no gradient scaler was provided.")
                    scaler.scale(batch_total).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    batch_total.backward()
                    optimizer.step()

        batch_size = images.shape[0]
        total_loss += batch_total.detach().item() * batch_size
        classification_loss += batch_classification.detach().item() * batch_size
        regression_loss += batch_regression.detach().item() * batch_size
        correct += (outputs[0].argmax(dim=1) == class_targets).sum().item()
        num_samples += batch_size

    if num_samples == 0:
        raise ValueError("Cannot compute epoch metrics for an empty data loader.")

    return {
        "loss": total_loss / num_samples,
        "classification_loss": classification_loss / num_samples,
        "regression_loss": regression_loss / num_samples,
        "accuracy": correct / num_samples,
    }


def train(args: argparse.Namespace) -> Path:
    config = _load_config(args.config)
    split_config = config.get("data_splits", {})
    training_config = dict(config["training"])

    configured_stage1_epochs = training_config.get("stage1_epochs", 0)
    configured_stage2_epochs = training_config.get(
        "stage2_epochs",
        training_config.get("num_epochs", 10) - configured_stage1_epochs,
    )
    stage1_epochs = (
        args.freeze_backbone_epochs
        if args.freeze_backbone_epochs is not None
        else configured_stage1_epochs
    )
    if (
        not isinstance(stage1_epochs, int)
        or isinstance(stage1_epochs, bool)
        or stage1_epochs < 0
    ):
        raise ValueError(f"stage1_epochs must be a non-negative integer; got {stage1_epochs!r}.")
    if (
        not isinstance(configured_stage2_epochs, int)
        or isinstance(configured_stage2_epochs, bool)
        or configured_stage2_epochs < 0
    ):
        raise ValueError(
            "stage2_epochs must be a non-negative integer; "
            f"got {configured_stage2_epochs!r}."
        )
    stage2_epochs = configured_stage2_epochs
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError(f"Epoch count must be a positive integer; got {args.epochs!r}.")
        stage1_epochs = min(stage1_epochs, args.epochs)
        stage2_epochs = args.epochs - stage1_epochs
    epochs = stage1_epochs + stage2_epochs
    batch_size = args.batch_size if args.batch_size is not None else config.get("batch_size", 32)
    stage1_learning_rate = training_config.get(
        "stage1_learning_rate",
        training_config.get("learning_rate", 0.001),
    )
    stage2_learning_rate = training_config.get(
        "stage2_learning_rate",
        stage1_learning_rate * 0.1,
    )
    if args.learning_rate is not None:
        stage1_learning_rate = args.learning_rate
        stage2_learning_rate = args.learning_rate
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError(f"Epoch count must be a positive integer; got {epochs!r}.")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError(f"Batch size must be a positive integer; got {batch_size!r}.")
    for rate_name, rate in (
        ("stage1_learning_rate", stage1_learning_rate),
        ("stage2_learning_rate", stage2_learning_rate),
    ):
        if (
            not isinstance(rate, (int, float))
            or isinstance(rate, bool)
            or not isfinite(rate)
            or rate <= 0
        ):
            raise ValueError(f"{rate_name} must be positive; got {rate!r}.")
    use_amp = training_config.get("use_amp", False)
    if not isinstance(use_amp, bool):
        raise ValueError(f"training.use_amp must be a boolean; got {use_amp!r}.")
    if args.num_workers < 0:
        raise ValueError(f"DataLoader worker count cannot be negative; got {args.num_workers}.")
    unfreeze_layers = (
        args.unfreeze_layers
        if args.unfreeze_layers is not None
        else training_config.get("stage2_unfreeze_layers", 1)
    )
    if not isinstance(unfreeze_layers, int) or isinstance(unfreeze_layers, bool) or unfreeze_layers < 1:
        raise ValueError(f"Stage 2 unfreeze layer count must be positive; got {unfreeze_layers!r}.")
    for name, limit in (
        ("max_train_samples", args.max_train_samples),
        ("max_val_samples", args.max_val_samples),
    ):
        if limit is not None and limit < 1:
            raise ValueError(f"{name.replace('_', ' ')} must be positive; got {limit}.")

    random_seed = int(split_config.get("seed", 42))
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    identity_transform = lambda image: image
    train_source = Food101NutritionDataset(
        split="train",
        transform=identity_transform,
        config_path=args.config,
    )
    test_source = Food101NutritionDataset(
        split="test",
        transform=identity_transform,
        config_path=args.config,
    )
    samples = train_source.samples + test_source.samples

    manifest_path = _resolve_ml_path(split_config.get("manifest", "data/splits.json"))
    manifest = load_or_create_splits(
        samples,
        manifest_path,
        train_ratio=split_config.get("train_ratio", 0.70),
        val_ratio=split_config.get("val_ratio", 0.15),
        test_ratio=split_config.get("test_ratio", 0.15),
        seed=random_seed,
    )

    image_size = config.get("image_size", 224)
    train_indices = manifest["split_indices"]["train"]
    val_indices = manifest["split_indices"]["val"]
    subset_randomizer = random.Random(random_seed)
    if args.max_train_samples is not None and args.max_train_samples < len(train_indices):
        train_indices = subset_randomizer.sample(train_indices, args.max_train_samples)
    if args.max_val_samples is not None and args.max_val_samples < len(val_indices):
        val_indices = subset_randomizer.sample(val_indices, args.max_val_samples)

    preprocessing_config = config.get("preprocessing", {})
    if not isinstance(preprocessing_config, dict):
        raise ValueError("The 'preprocessing' config section must be a mapping.")
    augmentation_preset = preprocessing_config.get("augmentation_preset", "standard")
    train_dataset = ManifestDataset(
        samples,
        train_indices,
        get_train_transform(image_size, preset=augmentation_preset),
    )
    val_dataset = ManifestDataset(
        samples,
        val_indices,
        get_eval_transform(image_size),
    )
    generator = torch.Generator().manual_seed(random_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    amp_enabled = use_amp and device.type == "cuda"
    if use_amp and not amp_enabled:
        warnings.warn(
            "Mixed precision was requested but CUDA is unavailable or not selected; "
            "continuing with full precision.",
            RuntimeWarning,
            stacklevel=2,
        )
    model_config = config["model"]
    model = MultiTaskCNN.from_config(config).to(device)
    if stage1_epochs:
        freeze_backbone(model)
    criterion = MultiTaskLoss(config).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    optimizer: torch.optim.Optimizer | None = None
    if stage1_epochs:
        optimizer = torch.optim.AdamW(
            [
                *model.classification_head.parameters(),
                *model.regression_head.parameters(),
            ],
            lr=float(stage1_learning_rate),
        )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = (
        Path(args.log_file)
        if args.log_file is not None
        else ML_DIR
        / "training"
        / "logs"
        / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    training_config.update(
        {
            "num_epochs": epochs,
            "stage1_epochs": stage1_epochs,
            "stage1_learning_rate": float(stage1_learning_rate),
            "stage2_epochs": stage2_epochs,
            "stage2_learning_rate": float(stage2_learning_rate),
            "stage2_unfreeze_layers": unfreeze_layers,
            "use_amp": use_amp,
            "amp_enabled": amp_enabled,
            "batch_size": batch_size,
            "scheduler": args.scheduler,
            "num_workers": args.num_workers,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "augmentation_preset": augmentation_preset,
            "freeze_backbone_epochs": stage1_epochs,
            "unfreeze_layers": unfreeze_layers,
        }
    )
    metadata = {
        "backbone": model_config.get("backbone", "resnet50"),
        "training_config": training_config,
        "version": "foodcnn-v1",
    }

    best_val_loss = float("inf")
    fieldnames = (
        "epoch",
        "train_loss",
        "train_classification_loss",
        "train_regression_loss",
        "train_accuracy",
        "val_loss",
        "val_classification_loss",
        "val_regression_loss",
        "val_accuracy",
    )
    with log_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
        writer.writeheader()
        epoch = 0
        stages = (
            ("stage1", stage1_epochs, float(stage1_learning_rate)),
            ("stage2", stage2_epochs, float(stage2_learning_rate)),
        )
        for stage_name, stage_epochs, stage_learning_rate in stages:
            if stage_epochs == 0:
                continue
            if stage_name == "stage2":
                unfreeze_backbone(model, unfreeze_layers)
                optimizer = torch.optim.AdamW(
                    (
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ),
                    lr=stage_learning_rate,
                )
            if optimizer is None:
                raise RuntimeError(f"Optimizer was not initialized for {stage_name}.")
            scheduler = _make_scheduler(args.scheduler, optimizer, stage_epochs)
            LOGGER.info(
                "%s: %d epochs at learning rate %.6g",
                stage_name,
                stage_epochs,
                stage_learning_rate,
            )
            for _ in range(stage_epochs):
                epoch += 1
                train_metrics = _run_epoch(
                    model,
                    train_loader,
                    criterion,
                    device,
                    optimizer,
                    amp_enabled=amp_enabled,
                    scaler=scaler,
                )
                with torch.no_grad():
                    val_metrics = _run_epoch(
                        model,
                        val_loader,
                        criterion,
                        device,
                        amp_enabled=amp_enabled,
                    )

                if scheduler is not None:
                    scheduler.step()

                row = {"epoch": epoch}
                row.update({f"train_{key}": value for key, value in train_metrics.items()})
                row.update({f"val_{key}": value for key, value in val_metrics.items()})
                writer.writerow(row)
                metrics_file.flush()

                save_checkpoint(
                    model,
                    checkpoint_dir / f"epoch_{epoch:03d}.pt",
                    {
                        **metadata,
                        "epoch": epoch,
                        "stage": stage_name,
                        "val_loss": val_metrics["loss"],
                    },
                )
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    save_checkpoint(
                        model,
                        checkpoint_dir / "best.pt",
                        {
                            **metadata,
                            "epoch": epoch,
                            "stage": stage_name,
                            "val_loss": best_val_loss,
                        },
                    )

                LOGGER.info(
                    "Epoch %d/%d (%s) | train loss %.4f (cls %.4f, reg %.4f), accuracy %.4f "
                    "| val loss %.4f (cls %.4f, reg %.4f), accuracy %.4f",
                    epoch,
                    epochs,
                    stage_name,
                    train_metrics["loss"],
                    train_metrics["classification_loss"],
                    train_metrics["regression_loss"],
                    train_metrics["accuracy"],
                    val_metrics["loss"],
                    val_metrics["classification_loss"],
                    val_metrics["regression_loss"],
                    val_metrics["accuracy"],
                )

    LOGGER.info("Metrics: %s", log_path)
    LOGGER.info("Checkpoints: %s", checkpoint_dir)
    return log_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multi-task food CNN.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="override both configured stage learning rates with this value",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=None,
        help="train only task heads for this many initial epochs, then unfreeze deeper stages",
    )
    parser.add_argument(
        "--unfreeze-layers",
        type=int,
        default=None,
        help="number of deepest backbone stages to unfreeze after frozen epochs",
    )
    parser.add_argument("--scheduler", choices=("none", "step", "cosine"), default="none")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="torch device (defaults to CUDA if available)")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ML_DIR / "training" / "checkpoints",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="CSV output path (defaults to a timestamped file in ml/training/logs/)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    train(args)


if __name__ == "__main__":
    main()
