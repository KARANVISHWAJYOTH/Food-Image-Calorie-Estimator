"""L1 unstructured pruning module for MultiTaskFoodCNN.

Performs:
1. L1 unstructured weight pruning on all convolutional layers (Conv2d) at 20%, 30%, and 50%.
2. Short fine-tuning to recover accuracy lost from weight removal.
3. Removal of pruning reparameterization (torch.nn.utils.prune.remove), permanently baking
   zeros into the weight tensors.
4. Serializes pruned model checkpoints (e.g., best_model_pruned_30.pth).
"""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
import yaml

ML_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ML_DIR.parent
CONFIG_PATH = ML_DIR / "configs" / "base.yaml"
OUTPUT_DIR = ML_DIR / "optimization"
LOGGER = logging.getLogger("ml.optimization.prune")

if __package__ in (None, ""):
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ml.data.food_dataset import Food101NutritionDataset
from ml.data.splits import load_or_create_splits
from ml.models.checkpoint import load_checkpoint, save_checkpoint
from ml.models.multitask_cnn import MultiTaskCNN
from ml.preprocessing.transforms import get_train_transform
from ml.training.losses import MultiTaskLoss
from ml.training.train import ManifestDataset


def get_conv_layers(model: nn.Module) -> list[tuple[str, nn.Conv2d]]:
    """Recursively collect all Conv2d modules with their named paths."""
    conv_layers: list[tuple[str, nn.Conv2d]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))
    return conv_layers


def calculate_conv_sparsity(model: nn.Module) -> dict[str, float]:
    """Calculate the exact percentage of zero-valued weights in all Conv2d layers."""
    convs = [m for _, m in get_conv_layers(model)]
    total_elements = sum(m.weight.numel() for m in convs)
    total_zeros = sum((m.weight == 0).sum().item() for m in convs)
    sparsity_pct = (total_zeros / total_elements * 100.0) if total_elements > 0 else 0.0
    return {
        "total_elements": total_elements,
        "total_zeros": total_zeros,
        "sparsity_percent": round(sparsity_pct, 2),
    }


def apply_l1_unstructured_pruning(
    model: nn.Module,
    amount: float = 0.30,
) -> nn.Module:
    """Apply L1 unstructured pruning to all Conv2d layers in the model.

    Args:
        model: PyTorch MultiTaskCNN module.
        amount: Fraction of weights to prune in [0.0, 1.0) (e.g. 0.2, 0.3, 0.5).

    Returns:
        The pruned model with forward pre-hooks and weight masks attached.
    """
    if not 0.0 < amount < 1.0:
        raise ValueError(f"Pruning amount must be in (0.0, 1.0); got {amount}.")

    conv_layers = get_conv_layers(model)
    LOGGER.info(f"Applying L1 unstructured pruning (amount={amount * 100:.1f}%) to {len(conv_layers)} Conv2d layers...")

    for name, module in conv_layers:
        prune.l1_unstructured(module, name="weight", amount=amount)

    stats = calculate_conv_sparsity(model)
    LOGGER.info(
        f"Pruning applied: {stats['total_zeros']}/{stats['total_elements']} conv weights zeroed "
        f"({stats['sparsity_percent']}% sparsity)."
    )
    return model


def fine_tune_pruned_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 1,
    max_batches: Optional[int] = 50,
    learning_rate: float = 1e-5,
    target_mean: Optional[torch.Tensor] = None,
    target_std: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Run a short fine-tuning schedule to recover accuracy while maintaining zero masks."""
    LOGGER.info(f"Fine-tuning pruned model for {epochs} epoch(s) (max_batches={max_batches}, lr={learning_rate})...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = MultiTaskLoss()

    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        batches_processed = 0

        for batch_idx, (images, class_targets, nutrition_targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device)
            class_targets = class_targets.to(device)
            nutrition_targets = nutrition_targets.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_cuda, dtype=torch.float16):
                logits, nutrition = model(images)
                loss = criterion(logits, class_targets, nutrition, nutrition_targets)

            if use_cuda:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            batches_processed += 1

        avg_loss = (running_loss / batches_processed) if batches_processed > 0 else 0.0
        LOGGER.info(f"Fine-tune Epoch {epoch}/{epochs} | Avg Loss: {avg_loss:.4f} ({batches_processed} batches)")

    model.eval()
    return model


def remove_pruning_reparameterization(model: nn.Module) -> nn.Module:
    """Remove pruning reparameterization, permanently baking zeros into the weights."""
    LOGGER.info("Removing pruning reparameterization hooks and buffers...")
    removed_count = 0

    for name, module in get_conv_layers(model):
        if hasattr(module, "weight_orig") and hasattr(module, "weight_mask"):
            prune.remove(module, "weight")
            removed_count += 1

    stats = calculate_conv_sparsity(model)
    LOGGER.info(
        f"Reparameterization removed from {removed_count} layers. "
        f"Permanent conv sparsity: {stats['sparsity_percent']}% ({stats['total_zeros']} zeros)."
    )
    return model


def prune_and_finetune(
    model: nn.Module,
    amount: float,
    train_loader: Optional[DataLoader],
    device: torch.device,
    fine_tune_epochs: int = 1,
    fine_tune_batches: Optional[int] = 50,
    learning_rate: float = 1e-5,
) -> nn.Module:
    """Full pipeline: clone model, apply L1 pruning, short fine-tune, and remove masks."""
    pruned_model = copy.deepcopy(model).to(device)
    apply_l1_unstructured_pruning(pruned_model, amount=amount)

    if train_loader is not None and len(train_loader) > 0:
        fine_tune_pruned_model(
            pruned_model,
            train_loader,
            device=device,
            epochs=fine_tune_epochs,
            max_batches=fine_tune_batches,
            learning_rate=learning_rate,
        )
    else:
        LOGGER.info("No training data provided for fine-tuning; skipping fine-tune stage.")

    remove_pruning_reparameterization(pruned_model)
    return pruned_model


def _find_checkpoint() -> Optional[Path]:
    candidates = [
        ML_DIR / "training" / "checkpoints" / "best_model.pth",
        REPOSITORY_ROOT / "best_model.pth",
        REPOSITORY_ROOT / "backend" / "app" / "best_model.pth",
        ML_DIR / "training" / "checkpoints" / "best.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="L1 unstructured pruning on MultiTaskFoodCNN.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to trained FP32 model checkpoint (defaults to best_model.pth).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--amounts",
        type=float,
        nargs="+",
        default=[0.20, 0.30, 0.50],
        help="List of pruning ratios (e.g. 0.20 0.30 0.50).",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=1,
        help="Number of epochs for short fine-tuning (default: 1).",
    )
    parser.add_argument(
        "--fine-tune-batches",
        type=int,
        default=50,
        help="Max batches to fine-tune per epoch (default: 50).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Fine-tuning learning rate (default: 1e-5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to save pruned checkpoints.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use ('cpu' or 'cuda').",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    model_config = config.get("model", {})
    backbone = model_config.get("backbone", "resnet50")
    num_classes = model_config.get("num_classes", 101)

    checkpoint_path = args.checkpoint or _find_checkpoint()
    class_names = None
    target_stats = None
    val_metrics = None
    metadata = {
        "backbone": backbone,
        "training_config": config.get("training", {}),
        "version": "1.0.0",
    }

    model = MultiTaskCNN(backbone=backbone, num_classes=num_classes, pretrained=False)

    if checkpoint_path and checkpoint_path.is_file():
        LOGGER.info(f"Loading weights from: {checkpoint_path}")
        try:
            ckpt = load_checkpoint(checkpoint_path)
            model.load_state_dict(ckpt["model_state_dict"])
            class_names = ckpt.get("class_names")
            target_stats = ckpt.get("target_stats")
            val_metrics = ckpt.get("val_metrics")
            if isinstance(ckpt.get("metadata"), dict):
                metadata.update(ckpt["metadata"])
        except Exception:
            raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(raw.get("model_state_dict", raw.get("state_dict", raw)))
    else:
        LOGGER.warning("No checkpoint found. Pruning baseline MultiTaskCNN model.")

    # Setup training loader for fine-tuning if dataset is present
    train_loader = None
    try:
        split_config = config.get("data_splits", {})
        identity_transform = lambda image: image
        train_source = Food101NutritionDataset(split="train", transform=identity_transform, config_path=args.config)
        test_source = Food101NutritionDataset(split="test", transform=identity_transform, config_path=args.config)
        samples = train_source.samples + test_source.samples

        manifest_path = ML_DIR / split_config.get("manifest", "data/splits.json")
        manifest = load_or_create_splits(
            samples,
            manifest_path,
            train_ratio=split_config.get("train_ratio", 0.70),
            val_ratio=split_config.get("val_ratio", 0.15),
            test_ratio=split_config.get("test_ratio", 0.15),
            seed=int(split_config.get("seed", 42)),
        )

        train_indices = manifest["split_indices"]["train"]
        train_transform = get_train_transform(image_size=config.get("image_size", 224))
        train_dataset = ManifestDataset(samples, train_indices, train_transform)
        train_loader = DataLoader(train_dataset, batch_size=config.get("batch_size", 32), shuffle=True)
        LOGGER.info(f"Loaded {len(train_dataset)} training samples for fine-tuning.")
    except Exception as e:
        LOGGER.warning(f"Could not load training dataset for fine-tuning ({e}). Proceeding without fine-tuning.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for amount in args.amounts:
        pct = int(round(amount * 100))
        LOGGER.info(f"\n==========================================")
        LOGGER.info(f"Starting Pruning Pipeline for Amount: {pct}%")
        LOGGER.info(f"==========================================")

        pruned_model = prune_and_finetune(
            model=model,
            amount=amount,
            train_loader=train_loader,
            device=device,
            fine_tune_epochs=args.fine_tune_epochs,
            fine_tune_batches=args.fine_tune_batches,
            learning_rate=args.learning_rate,
        )

        out_path = args.output_dir / f"best_model_pruned_{pct}.pth"
        pruned_meta = {
            **metadata,
            "pruned": True,
            "pruning_method": "l1_unstructured",
            "pruning_amount": amount,
            "conv_sparsity_percent": pct,
        }

        save_checkpoint(
            pruned_model.cpu(),
            out_path,
            metadata=pruned_meta,
            class_names=class_names,
            target_stats=target_stats,
            config=config,
            val_metrics=val_metrics,
        )
        LOGGER.info(f"Saved pruned checkpoint to: {out_path}")

    LOGGER.info("\nAll pruning targets complete.")


if __name__ == "__main__":
    main()
