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
