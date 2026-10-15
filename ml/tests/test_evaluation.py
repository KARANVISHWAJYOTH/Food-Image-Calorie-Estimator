"""Unit tests for the evaluation module."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ml.evaluation.evaluate import (
    compute_classification_metrics,
    compute_regression_metrics,
    evaluate_model,
    generate_markdown_report,
    save_calories_scatter_plot,
    save_confusion_matrix_plot,
)
import tempfile
from pathlib import Path


def test_compute_classification_metrics_exact() -> None:
    # 4 samples, 3 classes
    logits = np.array([
        [10.0, 1.0, 0.0],  # predicts 0, true 0
        [0.0, 5.0, 1.0],   # predicts 1, true 1
        [1.0, 2.0, 8.0],   # predicts 2, true 2
        [9.0, 2.0, 1.0],   # predicts 0, true 1 (misclassified, but true class 1 is in top-2)
    ])
    targets = np.array([0, 1, 2, 1])

    metrics = compute_classification_metrics(logits, targets, class_names=["pizza", "sushi", "tacos"], top_k=2)

    assert metrics["total_samples"] == 4
    assert metrics["top1_accuracy"] == 0.75  # 3 of 4 correct
    assert metrics["top2_accuracy"] == 1.0   # in top 2 for sample 3 as well
    assert "macro_f1" in metrics
    assert "macro_precision" in metrics
    assert "macro_recall" in metrics
    assert "worst_10_classes" in metrics
    assert len(metrics["worst_10_classes"]) == 3
    # sushi (class 1) had 1 of 2 correct = 50%, so it should be worst
    assert metrics["worst_10_classes"][0]["class_name"] == "sushi"
    assert metrics["worst_10_classes"][0]["accuracy"] == 0.5


def test_compute_regression_metrics_exact() -> None:
    # 2 samples, 4 targets: [calories, protein, carbohydrates, fat]
    preds = np.array([
        [500.0, 30.0, 50.0, 20.0],
        [600.0, 40.0, 60.0, 25.0],
    ])
    targets = np.array([
        [520.0, 28.0, 45.0, 22.0],  # error: 20, 2, 5, 2
        [580.0, 42.0, 65.0, 23.0],  # error: 20, 2, 5, 2
    ])

    metrics = compute_regression_metrics(preds, targets)

    assert metrics["calories"]["mae"] == 20.0
    assert metrics["protein"]["mae"] == 2.0
    assert metrics["carbohydrates"]["mae"] == 5.0
    assert metrics["fat"]["mae"] == 2.0
    assert abs(metrics["overall"]["mae"] - ((20.0 + 2.0 + 5.0 + 2.0) / 4.0)) < 1e-4
    assert "rmse" in metrics["calories"]
    assert "pearson_r" in metrics["calories"]
    assert "r2" in metrics["calories"]
    assert "mape" in metrics["calories"]
    # MAPE = mean( [20/520, 20/580] ) * 100 = mean( [0.03846, 0.03448] ) * 100 = ~3.65%
    assert abs(metrics["calories"]["mape"] - 3.65) < 0.1


class DummyMultiTaskModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        # Always predicts class 0 and fixed macros
        class_logits = torch.zeros(batch_size, 3)
        class_logits[:, 0] = 5.0
        macros = torch.tensor([[500.0, 30.0, 50.0, 20.0]]).repeat(batch_size, 1)
        return class_logits, macros


def test_evaluate_model_pipeline() -> None:
    dummy_images = torch.zeros(4, 3, 32, 32)
    dummy_classes = torch.tensor([0, 0, 1, 0])
    dummy_targets = torch.tensor([
        [500.0, 30.0, 50.0, 20.0],
        [500.0, 30.0, 50.0, 20.0],
        [500.0, 30.0, 50.0, 20.0],
        [500.0, 30.0, 50.0, 20.0],
    ])

    dataset = TensorDataset(dummy_images, dummy_classes, dummy_targets)
    loader = DataLoader(dataset, batch_size=2)
    device = torch.device("cpu")

    results = evaluate_model(DummyMultiTaskModel(), loader, device, class_names=["dish_a", "dish_b", "dish_c"])

    assert "classification" in results
    assert "regression" in results
    assert results["classification"]["top1_accuracy"] == 0.75
    assert results["regression"]["calories"]["mae"] == 0.0
    assert results["regression"]["calories"]["mape"] == 0.0
    assert "worst_10_classes" in results["classification"]

    report = generate_markdown_report(results, title="Test Report")
    assert "NutriVision AI – Test Report" in report
    assert "Top-1 Accuracy" in report
    assert "Proxy-Label Calorie Noise" in report


def test_plot_generation() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cm_path = tmp_path / "confusion_matrix.png"
        scatter_path = tmp_path / "calories_scatter.png"

        logits = np.array([[5.0, 1.0], [1.0, 4.0]])
        targets = np.array([0, 1])
        save_confusion_matrix_plot(logits, targets, cm_path, class_names=["pizza", "burger"])
        assert cm_path.is_file()
        assert cm_path.stat().st_size > 0

        pred_cal = np.array([450.0, 620.0])
        true_cal = np.array([440.0, 650.0])
        save_calories_scatter_plot(pred_cal, true_cal, scatter_path, mape=4.2)
        assert scatter_path.is_file()
        assert scatter_path.stat().st_size > 0
