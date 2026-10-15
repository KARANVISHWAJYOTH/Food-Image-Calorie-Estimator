"""Model evaluation module for Multi-Task Food CNN.

Evaluates trained checkpoints on classification (Top-1/Top-5 accuracy, macro/weighted F1)
and regression (MAE, RMSE, Pearson r, R2 for Calories, Protein, Carbs, Fat).
Outputs metrics to JSON and Markdown reports.
"""

from __future__ import annotations

import argparse
import json
import logging
from math import sqrt
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

ML_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ML_DIR.parent
CONFIG_PATH = ML_DIR / "configs" / "base.yaml"
LOGGER = logging.getLogger("ml.evaluation")

if __package__ in (None, ""):
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ml.data.food_dataset import Food101NutritionDataset
from ml.data.splits import load_or_create_splits
from ml.models.checkpoint import load_checkpoint
from ml.models.multitask_cnn import MultiTaskCNN
from ml.preprocessing.transforms import get_eval_transform
from ml.training.train import ManifestDataset

NUTRITION_TARGET_NAMES = ("calories", "protein", "carbohydrates", "fat")


def compute_regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute MAE, RMSE, Pearson r, and R2 for each nutrition target."""
    metrics: dict[str, dict[str, float]] = {}

    for idx, target_name in enumerate(NUTRITION_TARGET_NAMES):
        pred_col = predictions[:, idx]
        true_col = targets[:, idx]

        mae = float(np.mean(np.abs(pred_col - true_col)))
        mse = float(np.mean((pred_col - true_col) ** 2))
        rmse = float(sqrt(mse))

        # Total sum of squares & R-squared
        ss_tot = float(np.sum((true_col - np.mean(true_col)) ** 2))
        ss_res = float(np.sum((true_col - pred_col) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-8 else 0.0

        # Pearson correlation coefficient
        pred_std = float(np.std(pred_col))
        true_std = float(np.std(true_col))
        if pred_std > 1e-8 and true_std > 1e-8:
            covariance = float(np.mean((pred_col - np.mean(pred_col)) * (true_col - np.mean(true_col))))
            pearson_r = covariance / (pred_std * true_std)
        else:
            pearson_r = 0.0

        metrics[target_name] = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "pearson_r": round(pearson_r, 4),
        }

    # Macro averages across all 4 targets
    metrics["overall"] = {
        "mae": round(float(np.mean([m["mae"] for k, m in metrics.items() if k != "overall"])), 4),
        "rmse": round(float(np.mean([m["rmse"] for k, m in metrics.items() if k != "overall"])), 4),
        "r2": round(float(np.mean([m["r2"] for k, m in metrics.items() if k != "overall"])), 4),
        "pearson_r": round(float(np.mean([m["pearson_r"] for k, m in metrics.items() if k != "overall"])), 4),
    }

    return metrics


def compute_classification_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    top_k: int = 5,
) -> dict[str, Any]:
    """Compute Top-1, Top-k accuracy, macro F1, precision, and recall."""
    num_samples = len(targets)
    if num_samples == 0:
        return {"top1_accuracy": 0.0, "top5_accuracy": 0.0, "macro_f1": 0.0}

    preds = np.argmax(logits, axis=1)
    top1_correct = int(np.sum(preds == targets))
    top1_acc = float(top1_correct / num_samples)

    # Top-K accuracy
    num_classes = logits.shape[1]
    effective_k = min(top_k, num_classes)
    if effective_k > 1:
        top_k_indices = np.argpartition(-logits, effective_k - 1, axis=1)[:, :effective_k]
        top_k_correct = int(np.sum([targets[i] in top_k_indices[i] for i in range(num_samples)]))
        top_k_acc = float(top_k_correct / num_samples)
    else:
        top_k_acc = top1_acc

    # Per-class precision, recall, and F1
    classes = np.unique(targets)
    precisions = []
    recalls = []
    f1s = []

    for c in classes:
        tp = int(np.sum((preds == c) & (targets == c)))
        fp = int(np.sum((preds == c) & (targets != c)))
        fn = int(np.sum((preds != c) & (targets == c)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return {
        "top1_accuracy": round(top1_acc, 4),
        f"top{effective_k}_accuracy": round(top_k_acc, 4),
        "macro_precision": round(float(np.mean(precisions)) if precisions else 0.0, 4),
        "macro_recall": round(float(np.mean(recalls)) if recalls else 0.0, 4),
        "macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 4),
        "total_samples": num_samples,
    }


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Run model inference on the loader and return computed metrics."""
    model.eval()
    all_logits = []
    all_class_targets = []
    all_nutrition_preds = []
    all_nutrition_targets = []

    with torch.no_grad():
        for images, class_targets, nutrition_targets in loader:
            images = images.to(device)
            outputs = model(images)

            if isinstance(outputs, tuple):
                logits, nutrition = outputs
            elif isinstance(outputs, dict):
                logits = outputs["class_logits"]
                nutrition = outputs["macros"] if "macros" in outputs else outputs["calories"]
            else:
                raise ValueError(f"Unexpected model output format: {type(outputs)}")

            all_logits.append(logits.cpu().numpy())
            all_class_targets.append(class_targets.numpy())
            all_nutrition_preds.append(nutrition.cpu().numpy())
            all_nutrition_targets.append(nutrition_targets.numpy())

    concat_logits = np.concatenate(all_logits, axis=0)
    concat_class_targets = np.concatenate(all_class_targets, axis=0)
    concat_nutr_preds = np.concatenate(all_nutrition_preds, axis=0)
    concat_nutr_targets = np.concatenate(all_nutrition_targets, axis=0)

    classification_metrics = compute_classification_metrics(concat_logits, concat_class_targets)
    regression_metrics = compute_regression_metrics(concat_nutr_preds, concat_nutr_targets)

    return {
        "classification": classification_metrics,
        "regression": regression_metrics,
    }


def generate_markdown_report(metrics: dict[str, Any], title: str = "Evaluation Report") -> str:
    """Format evaluation metrics as a clean markdown report."""
    cls_m = metrics.get("classification", {})
    reg_m = metrics.get("regression", {})

    report = [
        f"# 📊 NutriVision AI – {title}",
        "",
        "## 1. Classification Performance",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **Top-1 Accuracy** | {cls_m.get('top1_accuracy', 0.0) * 100:.2f}% |",
        f"| **Top-5 Accuracy** | {cls_m.get('top5_accuracy', 0.0) * 100:.2f}% |",
        f"| **Macro Precision** | {cls_m.get('macro_precision', 0.0):.4f} |",
        f"| **Macro Recall** | {cls_m.get('macro_recall', 0.0):.4f} |",
        f"| **Macro F1 Score** | {cls_m.get('macro_f1', 0.0):.4f} |",
        f"| **Evaluated Samples** | {cls_m.get('total_samples', 0)} |",
        "",
        "## 2. Nutritional Regression Performance",
        "",
        "| Target | MAE | RMSE | Pearson r | R² Score |",
        "|---|---|---|---|---|",
    ]

    for target in ("calories", "protein", "carbohydrates", "fat", "overall"):
        t_data = reg_m.get(target, {})
        name = target.capitalize() if target != "overall" else "**Overall Macro Avg**"
        report.append(
            f"| {name} | {t_data.get('mae', 0.0):.2f} | {t_data.get('rmse', 0.0):.2f} | "
            f"{t_data.get('pearson_r', 0.0):.3f} | {t_data.get('r2', 0.0):.3f} |"
        )

    report.extend([
        "",
        "> **Note:** Regression targets are aligned using Nutrition5k dish proxy distributions.",
    ])

    return "\n".join(report)


def _load_model_for_eval(
    checkpoint_path: Path | None,
    config: dict[str, Any],
    device: torch.device,
) -> MultiTaskCNN:
    """Instantiate and populate the MultiTaskCNN."""
    model_config = config.get("model", {})
    backbone = model_config.get("backbone", "resnet50")
    num_classes = model_config.get("num_classes", 101)
    dropout = model_config.get("dropout", 0.3)

    if checkpoint_path and checkpoint_path.is_file():
        LOGGER.info(f"Loading weights from checkpoint: {checkpoint_path}")
        try:
            checkpoint = load_checkpoint(checkpoint_path)
            state_dict = checkpoint["model_state_dict"]
            metadata = checkpoint.get("metadata", {})
            backbone = metadata.get("backbone", backbone)
        except Exception:
            # Fallback for plain PyTorch checkpoint dictionaries
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))

        model = MultiTaskCNN(
            backbone=backbone,
            num_classes=num_classes,
            dropout=dropout,
            pretrained=False,
        )
        model.load_state_dict(state_dict)
    else:
        LOGGER.warning("No checkpoint file provided or file not found. Evaluating uninitialized model.")
        model = MultiTaskCNN(
            backbone=backbone,
            num_classes=num_classes,
            dropout=dropout,
            pretrained=False,
        )

    model.to(device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MultiTaskFoodCNN checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to trained checkpoint (.pth).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ML_DIR / "evaluation" / "results",
        help="Directory to save evaluation reports and metrics JSON.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size.",
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
    model = _load_model_for_eval(args.checkpoint, config, device)

    # Build dataset split loader
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

    indices = manifest["split_indices"][args.split]
    eval_transform = get_eval_transform(image_size=config.get("image_size", 224))
    dataset = ManifestDataset(samples, indices, eval_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    LOGGER.info(f"Evaluating split '{args.split}' with {len(dataset)} samples...")
    results = evaluate_model(model, loader, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"metrics_{args.split}.json"
    report_path = args.output_dir / f"report_{args.split}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    report_content = generate_markdown_report(results, title=f"Evaluation on {args.split.capitalize()} Set")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print("\n" + report_content + "\n")
    LOGGER.info(f"Metrics saved to {json_path} and {report_path}")


if __name__ == "__main__":
    main()
