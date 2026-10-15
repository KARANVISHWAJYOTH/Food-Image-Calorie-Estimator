"""Model evaluation module for Multi-Task Food CNN.

Evaluates trained checkpoints on the held-out test split, reporting:
- Top-1 and Top-5 accuracy
- Per-class accuracy with worst 10 classes breakdown
- MAE and RMSE for Calories (kcal), Protein (g), Carbohydrates (g), and Fat (g)
- MAPE (Mean Absolute Percentage Error) for Calories
- Visualizations: Confusion-matrix heatmap and Predicted-vs-Actual scatter plot
- Machine-readable metrics.json and report-ready Markdown tables
- Limitations section generator addressing proxy-label noise
"""

from __future__ import annotations

import argparse
import json
import logging
from math import sqrt
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

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
) -> dict[str, Any]:
    """Compute MAE, RMSE, Pearson r, R2, and calorie MAPE for nutrition targets."""
    metrics: dict[str, dict[str, float]] = {}

    for idx, target_name in enumerate(NUTRITION_TARGET_NAMES):
        if predictions.shape[1] <= idx or targets.shape[1] <= idx:
            continue
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

        target_dict: dict[str, float] = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "pearson_r": round(pearson_r, 4),
        }

        # Compute MAPE for calories: |y_true - y_pred| / max(y_true, 1.0) * 100
        if target_name == "calories":
            safe_targets = np.maximum(np.abs(true_col), 1.0)
            mape = float(np.mean(np.abs(pred_col - true_col) / safe_targets) * 100.0)
            target_dict["mape"] = round(mape, 2)

        metrics[target_name] = target_dict

    # Macro averages across all 4 targets
    valid_targets = [k for k in NUTRITION_TARGET_NAMES if k in metrics]
    if valid_targets:
        metrics["overall"] = {
            "mae": round(float(np.mean([metrics[k]["mae"] for k in valid_targets])), 4),
            "rmse": round(float(np.mean([metrics[k]["rmse"] for k in valid_targets])), 4),
            "r2": round(float(np.mean([metrics[k]["r2"] for k in valid_targets])), 4),
            "pearson_r": round(float(np.mean([metrics[k]["pearson_r"] for k in valid_targets])), 4),
        }

    return metrics


def compute_classification_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Compute Top-1, Top-k accuracy, macro precision/recall/F1, and per-class breakdown."""
    num_samples = len(targets)
    if num_samples == 0:
        return {
            "top1_accuracy": 0.0,
            "top5_accuracy": 0.0,
            "macro_f1": 0.0,
            "worst_10_classes": [],
        }

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

    # Per-class precision, recall, F1, and class accuracy
    unique_classes = np.unique(targets)
    precisions = []
    recalls = []
    f1s = []
    per_class_list: list[dict[str, Any]] = []

    for c in unique_classes:
        tp = int(np.sum((preds == c) & (targets == c)))
        fp = int(np.sum((preds == c) & (targets != c)))
        fn = int(np.sum((preds != c) & (targets == c)))
        total_in_class = int(np.sum(targets == c))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        cls_acc = tp / total_in_class if total_in_class > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        c_idx = int(c)
        c_name = class_names[c_idx] if class_names and c_idx < len(class_names) else f"class_{c_idx}"
        per_class_list.append({
            "class_index": c_idx,
            "class_name": c_name,
            "total_samples": total_in_class,
            "correct_samples": tp,
            "accuracy": round(cls_acc, 4),
            "accuracy_percent": round(cls_acc * 100.0, 2),
        })

    # Sort classes by accuracy ascending to identify worst performers
    sorted_classes = sorted(per_class_list, key=lambda x: (x["accuracy"], x["total_samples"]))
    worst_10_classes = sorted_classes[:10]

    return {
        "top1_accuracy": round(top1_acc, 4),
        f"top{effective_k}_accuracy": round(top_k_acc, 4),
        "macro_precision": round(float(np.mean(precisions)) if precisions else 0.0, 4),
        "macro_recall": round(float(np.mean(recalls)) if recalls else 0.0, 4),
        "macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 4),
        "total_samples": num_samples,
        "worst_10_classes": worst_10_classes,
        "per_class_accuracy": {item["class_name"]: item["accuracy"] for item in per_class_list},
    }


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Optional[Sequence[str]] = None,
    target_mean: Optional[torch.Tensor] = None,
    target_std: Optional[torch.Tensor] = None,
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

            # De-normalize regression predictions if target stats are provided
            if target_mean is not None and target_std is not None:
                t_mean = target_mean.to(device)
                t_std = target_std.to(device)
                nutrition = nutrition * t_std + t_mean

            all_logits.append(logits.cpu().numpy())
            all_class_targets.append(class_targets.numpy())
            all_nutrition_preds.append(nutrition.cpu().numpy())
            all_nutrition_targets.append(nutrition_targets.numpy())

    concat_logits = np.concatenate(all_logits, axis=0)
    concat_class_targets = np.concatenate(all_class_targets, axis=0)
    concat_nutr_preds = np.concatenate(all_nutrition_preds, axis=0)
    concat_nutr_targets = np.concatenate(all_nutrition_targets, axis=0)

    classification_metrics = compute_classification_metrics(
        concat_logits,
        concat_class_targets,
        class_names=class_names,
    )
    regression_metrics = compute_regression_metrics(concat_nutr_preds, concat_nutr_targets)

    return {
        "classification": classification_metrics,
        "regression": regression_metrics,
        "predictions": {
            "logits": concat_logits,
            "class_targets": concat_class_targets,
            "nutrition_preds": concat_nutr_preds,
            "nutrition_targets": concat_nutr_targets,
        },
    }


def save_confusion_matrix_plot(
    logits: np.ndarray,
    targets: np.ndarray,
    output_path: Path,
    class_names: Optional[Sequence[str]] = None,
    title: str = "Test Split Confusion Matrix",
) -> None:
    """Generate and save a confusion matrix heatmap to disk."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preds = np.argmax(logits, axis=1)
    num_classes = logits.shape[1]
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(targets, preds):
        cm[t, p] += 1

    # Normalize by true class count for percentage interpretation
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    cax = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", aspect="auto")
    plt.colorbar(cax, label="Normalized Frequency")

    ax.set_title(title, fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel("Predicted Label Index", fontsize=11)
    ax.set_ylabel("True Label Index", fontsize=11)

    # If small number of classes, annotate names
    if num_classes <= 20 and class_names and len(class_names) >= num_classes:
        tick_marks = np.arange(num_classes)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names[:num_classes], rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(class_names[:num_classes], fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(f"Saved confusion matrix plot to {output_path}")


def save_calories_scatter_plot(
    pred_calories: np.ndarray,
    true_calories: np.ndarray,
    output_path: Path,
    mape: Optional[float] = None,
    title: str = "Calorie Estimation: Predicted vs. Actual",
) -> None:
    """Generate and save a Predicted vs. Actual scatter plot for calories."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    # Scatter points with slight transparency
    ax.scatter(true_calories, pred_calories, alpha=0.55, color="#10b981", edgecolors="#047857", s=25, label="Test Predictions")

    # Reference 45-degree line y = x
    min_val = min(float(np.min(true_calories)), float(np.min(pred_calories)), 0.0)
    max_val = max(float(np.max(true_calories)), float(np.max(pred_calories)), 100.0) * 1.05
    ax.plot([min_val, max_val], [min_val, max_val], color="#dc2626", linestyle="--", linewidth=1.8, label="Ideal Line (y = x)")

    mae = float(np.mean(np.abs(pred_calories - true_calories)))
    rmse = float(sqrt(np.mean((pred_calories - true_calories) ** 2)))

    stat_box = f"Calorie MAE: {mae:.1f} kcal\nCalorie RMSE: {rmse:.1f} kcal"
    if mape is not None:
        stat_box += f"\nCalorie MAPE: {mape:.1f}%"

    ax.text(
        0.05,
        0.92,
        stat_box,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cbd5e1", alpha=0.9),
    )

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel("Actual Calories (kcal)", fontsize=11, fontweight="medium")
    ax.set_ylabel("Predicted Calories (kcal)", fontsize=11, fontweight="medium")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(f"Saved calories scatter plot to {output_path}")


def generate_markdown_report(metrics: dict[str, Any], title: str = "Test Split Evaluation Report") -> str:
    """Format evaluation metrics as clean, report-ready markdown tables with limitations."""
    cls_m = metrics.get("classification", {})
    reg_m = metrics.get("regression", {})
    worst_10 = cls_m.get("worst_10_classes", [])
    cal_m = reg_m.get("calories", {})
    mape_str = f"{cal_m.get('mape', 0.0):.2f}%" if "mape" in cal_m else "N/A"

    report = [
        f"# 📊 NutriVision AI – {title}",
        "",
        "## 1. Classification Performance (Test Split)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **Top-1 Accuracy** | {cls_m.get('top1_accuracy', 0.0) * 100:.2f}% |",
        f"| **Top-5 Accuracy** | {cls_m.get('top5_accuracy', 0.0) * 100:.2f}% |",
        f"| **Macro Precision** | {cls_m.get('macro_precision', 0.0):.4f} |",
        f"| **Macro Recall** | {cls_m.get('macro_recall', 0.0):.4f} |",
        f"| **Macro F1 Score** | {cls_m.get('macro_f1', 0.0):.4f} |",
        f"| **Total Test Samples** | {cls_m.get('total_samples', 0)} |",
        "",
        "## 2. Nutritional Regression Performance",
        "",
        "| Nutritional Target | MAE | RMSE | Pearson r | R² Score | MAPE |",
        "|---|---|---|---|---|---|",
    ]

    for target in ("calories", "protein", "carbohydrates", "fat", "overall"):
        t_data = reg_m.get(target, {})
        name = target.capitalize() if target != "overall" else "**Overall Macro Avg**"
        unit = " kcal" if target == "calories" else (" g" if target != "overall" else "")
        mae_val = f"{t_data.get('mae', 0.0):.2f}{unit}"
        rmse_val = f"{t_data.get('rmse', 0.0):.2f}{unit}"
        r_val = f"{t_data.get('pearson_r', 0.0):.3f}"
        r2_val = f"{t_data.get('r2', 0.0):.3f}"
        t_mape = f"{t_data.get('mape'):.2f}%" if "mape" in t_data else ("—" if target != "calories" else mape_str)
        report.append(f"| {name} | {mae_val} | {rmse_val} | {r_val} | {r2_val} | {t_mape} |")

    if worst_10:
        report.extend([
            "",
            "## 3. Per-Class Accuracy: 10 Most Challenging Classes",
            "",
            "| Rank | Dish Category Name | Class Index | Test Samples | Correct | Accuracy |",
            "|:---:|:---|:---:|:---:|:---:|:---:|",
        ])
        for rank, item in enumerate(worst_10, start=1):
            report.append(
                f"| {rank} | **{item.get('class_name')}** | `{item.get('class_index')}` | "
                f"{item.get('total_samples')} | {item.get('correct_samples')} | **{item.get('accuracy_percent'):.1f}%** |"
            )

    report.extend([
        "",
        "## 4. Limitations & Modeling Nuances",
        "",
        "> **Proxy-Label Calorie Noise:**  ",
        "> Food-101 images do not possess direct per-dish laboratory bomb calorimetry or chemical nutrient assays. "
        "> Instead, continuous nutritional regression targets were derived by aligning Food-101 culinary categories "
        "> to cafeteria dish ground-truth distributions from the Nutrition5k dataset using fuzzy text matching and "
        "> manual ingredient heuristics. Consequently, the reported Calorie MAE and MAPE partly reflect proxy-label "
        "> variance, recipe heterogeneity, and culinary preparation differences across real-world food captures, "
        "> rather than solely visual prediction error.",
    ])

    return "\n".join(report)


def _find_default_checkpoint() -> Optional[Path]:
    """Search candidate paths for best_model.pth or best.pt."""
    candidates = [
        ML_DIR / "training" / "checkpoints" / "best_model.pth",
        REPOSITORY_ROOT / "best_model.pth",
        REPOSITORY_ROOT / "backend" / "app" / "best_model.pth",
        ML_DIR / "training" / "checkpoints" / "best.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _load_model_for_eval(
    checkpoint_path: Optional[Path],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[MultiTaskCNN, Optional[list[str]], Optional[dict[str, Any]]]:
    """Instantiate MultiTaskCNN and load weights, returning (model, class_names, target_stats)."""
    model_config = config.get("model", {})
    backbone = model_config.get("backbone", "resnet50")
    num_classes = model_config.get("num_classes", 101)
    dropout = model_config.get("dropout", 0.3)

    resolved_checkpoint = checkpoint_path or _find_default_checkpoint()
    class_names = None
    target_stats = None

    if resolved_checkpoint and resolved_checkpoint.is_file():
        LOGGER.info(f"Loading weights from checkpoint: {resolved_checkpoint}")
        try:
            checkpoint = load_checkpoint(resolved_checkpoint)
            state_dict = checkpoint["model_state_dict"]
            metadata = checkpoint.get("metadata", {})
            backbone = metadata.get("backbone", backbone)
            class_names = checkpoint.get("class_names")
            target_stats = checkpoint.get("target_stats")
        except Exception:
            checkpoint = torch.load(resolved_checkpoint, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))

        model = MultiTaskCNN(
            backbone=backbone,
            num_classes=num_classes,
            dropout=dropout,
            pretrained=False,
        )
        model.load_state_dict(state_dict)
    else:
        LOGGER.warning("No checkpoint found. Evaluating uninitialized baseline MultiTaskCNN model.")
        model = MultiTaskCNN(
            backbone=backbone,
            num_classes=num_classes,
            dropout=dropout,
            pretrained=False,
        )

    model.to(device)
    return model, class_names, target_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MultiTaskFoodCNN checkpoint exclusively on held-out test split.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to trained checkpoint (.pth / .pt). Defaults to best_model.pth in standard locations.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to base.yaml config file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "train"],
        help="Dataset partition to evaluate (default: test split only).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ML_DIR / "evaluation" / "results",
        help="Directory to save metrics.json, plots, and markdown report.",
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
        help="Compute device ('cpu' or 'cuda').",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    model, checkpoint_class_names, target_stats = _load_model_for_eval(args.checkpoint, config, device)

    # Build dataset split loader
    split_config = config.get("data_splits", {})
    identity_transform = lambda image: image
    train_source = Food101NutritionDataset(split="train", transform=identity_transform, config_path=args.config)
    test_source = Food101NutritionDataset(split="test", transform=identity_transform, config_path=args.config)
    samples = train_source.samples + test_source.samples

    resolved_class_names = checkpoint_class_names or train_source.classes

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

    LOGGER.info(f"Evaluating exclusively on '{args.split}' split ({len(dataset)} samples)...")

    target_mean = None
    target_std = None
    if target_stats and "mean" in target_stats and "std" in target_stats:
        target_mean = torch.tensor(target_stats["mean"], dtype=torch.float32)
        target_std = torch.tensor(target_stats["std"], dtype=torch.float32)

    results = evaluate_model(
        model,
        loader,
        device,
        class_names=resolved_class_names,
        target_mean=target_mean,
        target_std=target_std,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_json_path = args.output_dir / "metrics.json"
    report_md_path = args.output_dir / "report.md"
    cm_plot_path = args.output_dir / "confusion_matrix.png"
    scatter_plot_path = args.output_dir / "calories_scatter.png"

    # Separate raw array predictions for clean JSON serialization
    pred_data = results.pop("predictions", None)

    # Save metrics.json
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Generate and save markdown report
    report_markdown = generate_markdown_report(results, title="Held-Out Test Split Evaluation Report")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_markdown)

    # Save visualization plots
    if pred_data is not None:
        save_confusion_matrix_plot(
            pred_data["logits"],
            pred_data["class_targets"],
            cm_plot_path,
            class_names=resolved_class_names,
        )

        cal_mape = results.get("regression", {}).get("calories", {}).get("mape")
        save_calories_scatter_plot(
            pred_data["nutrition_preds"][:, 0],
            pred_data["nutrition_targets"][:, 0],
            scatter_plot_path,
            mape=cal_mape,
        )

    print("\n" + report_markdown + "\n")
    LOGGER.info(f"Summary metrics saved to {metrics_json_path}")
    LOGGER.info(f"Report Markdown saved to {report_md_path}")
    LOGGER.info(f"Plots saved to {cm_plot_path} and {scatter_plot_path}")


if __name__ == "__main__":
    main()
