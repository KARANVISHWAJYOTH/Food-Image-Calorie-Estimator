import pytest
import torch

from ml.training.losses import MultiTaskLoss, RegressionLoss


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


def test_regression_loss_rejects_non_4_target_shapes() -> None:
    reg_loss = RegressionLoss()
    valid_pred = torch.randn(2, 4)
    valid_target = torch.randn(2, 4)
    loss = reg_loss(valid_pred, valid_target)
    assert loss.ndim == 0

    # Reject 5 targets (e.g. attempting to predict fiber directly without ground truth)
    pred_5 = torch.randn(2, 5)
    target_5 = torch.randn(2, 5)
    with pytest.raises(ValueError, match="Nutrition predictions and targets must each have four values"):
        reg_loss(pred_5, target_5)

    # Reject 3 targets
    pred_3 = torch.randn(2, 3)
    target_3 = torch.randn(2, 3)
    with pytest.raises(ValueError, match="Nutrition predictions and targets must each have four values"):
        reg_loss(pred_3, target_3)

