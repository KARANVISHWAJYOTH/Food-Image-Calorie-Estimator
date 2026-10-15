"""Tests for multi-task training and validation metrics."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from ml.training.losses import MultiTaskLoss
from ml.training.train import _run_epoch


class FixedPredictionModel(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = images.shape[0]
        class_logits = images.new_tensor([[2.0, 0.0]]).expand(batch_size, -1)
        nutrition_predictions = images.new_zeros((batch_size, 4))
        return class_logits, nutrition_predictions


def test_run_epoch_reports_accuracy_and_per_target_nutrition_mae() -> None:
    loader = DataLoader(
        TensorDataset(
            torch.zeros(2, 1),
            torch.zeros(2, dtype=torch.long),
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]]),
        ),
        batch_size=1,
    )

    metrics = _run_epoch(
        FixedPredictionModel(),
        loader,
        MultiTaskLoss({"training": {"lambda_cls": 1.0, "lambda_reg": 1.0}}),
        torch.device("cpu"),
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["nutrition_mae"] == 3.5
    assert metrics["calories_mae"] == 2.0
    assert metrics["protein_mae"] == 3.0
    assert metrics["carbohydrates_mae"] == 4.0
    assert metrics["fat_mae"] == 5.0


def test_run_epoch_denormalizes_zscore_targets_to_physical_units() -> None:
    # Normalized z-score targets passed through DataLoader
    target_mean = torch.tensor([200.0, 20.0, 30.0, 10.0])
    target_std = torch.tensor([100.0, 10.0, 10.0, 5.0])

    normalized_targets = torch.tensor([
        [-1.0, -1.0, -1.0, -1.0],  # physical: [100.0, 10.0, 20.0, 5.0]
        [1.0, 1.0, 1.0, 1.0],      # physical: [300.0, 30.0, 40.0, 15.0]
    ])

    loader = DataLoader(
        TensorDataset(
            torch.zeros(2, 1),
            torch.zeros(2, dtype=torch.long),
            normalized_targets,
        ),
        batch_size=1,
    )

    # FixedPredictionModel predicts 0.0 for all regression targets (z-score 0 -> mean)
    metrics = _run_epoch(
        FixedPredictionModel(),
        loader,
        MultiTaskLoss({"training": {"lambda_cls": 1.0, "lambda_reg": 1.0}}),
        torch.device("cpu"),
        target_mean=target_mean,
        target_std=target_std,
    )

    # Physical errors: |200-100|=100, |200-300|=100 -> calories_mae = 100.0
    assert metrics["calories_mae"] == 100.0
    assert metrics["protein_mae"] == 10.0
    assert metrics["carbohydrates_mae"] == 10.0
    assert metrics["fat_mae"] == 5.0
    assert metrics["nutrition_mae"] == 31.25

