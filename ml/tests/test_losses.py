"""Tests for multi-task training losses."""

import torch

from ml.training.losses import MultiTaskLoss


def test_multitask_loss_returns_weighted_total_and_task_losses() -> None:
    batch_size = 3
    outputs = (
        torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]]),
        torch.zeros(batch_size, 4),
    )
    class_targets = torch.tensor([0, 1, 0])
    nutrition_targets = torch.ones(batch_size, 4)
    criterion = MultiTaskLoss(
        {
            "training": {
                "lambda_cls": 2.0,
                "lambda_reg": 0.5,
                "label_smoothing": 0.1,
            }
        }
    )

    total_loss, classification_loss, regression_loss = criterion(
        outputs,
        class_targets,
        nutrition_targets,
    )

    assert torch.isclose(
        total_loss,
        2.0 * classification_loss + 0.5 * regression_loss,
    )
    assert classification_loss.ndim == 0
    assert regression_loss.ndim == 0
    assert classification_loss.item() >= 0.0
    assert regression_loss.item() >= 0.0
