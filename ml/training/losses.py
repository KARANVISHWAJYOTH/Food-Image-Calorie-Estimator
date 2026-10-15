"""Loss functions for multi-task food classification and nutrition regression.

Target Consistency:
Regression loss is calculated over the 4 ground-truth continuous targets:
[calories, protein, carbohydrates, fat]. Dietary fiber is excluded from loss
optimization as it is not recorded in Nutrition5k sensor ground truth; it is
deterministically derived downstream at inference time in backend/app/predictor.py.
"""

from collections.abc import Mapping
from math import isfinite
from pathlib import Path
from typing import Any

import torch
from torch import nn


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


class ClassificationLoss(nn.Module):
    """Cross-entropy classification loss with optional label smoothing."""

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        if (
            not isinstance(label_smoothing, (int, float))
            or isinstance(label_smoothing, bool)
            or not isfinite(label_smoothing)
            or not 0.0 <= label_smoothing <= 1.0
        ):
            raise ValueError(
                "label_smoothing must be a finite number in [0, 1]; "
                f"got {label_smoothing!r}."
            )
        self.loss = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))

    def forward(self, class_logits: torch.Tensor, class_targets: torch.Tensor) -> torch.Tensor:
        return self.loss(class_logits, class_targets)


class RegressionLoss(nn.Module):
    """Smooth L1 (Huber) loss for the four nutrition predictions [calories, protein, carbs, fat].

    Dietary fiber is derived downstream in predictor.py from carbohydrates and food class
    rather than predicted here, because Nutrition5k ground truth does not record fiber.
    """

    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        if (
            not isinstance(beta, (int, float))
            or isinstance(beta, bool)
            or not isfinite(beta)
            or beta < 0.0
        ):
            raise ValueError(f"beta must be a finite non-negative number; got {beta!r}.")
        self.loss = nn.SmoothL1Loss(beta=float(beta))

    def forward(
        self,
        nutrition_outputs: torch.Tensor,
        nutrition_targets: torch.Tensor,
    ) -> torch.Tensor:
        if nutrition_outputs.shape[-1] != 4 or nutrition_targets.shape[-1] != 4:
            raise ValueError(
                "Nutrition predictions and targets must each have four values "
                "(calories, protein, carbs, fat). Fiber is derived downstream in predictor.py."
            )
        return self.loss(nutrition_outputs, nutrition_targets)


class MultiTaskLoss(nn.Module):
    """Combine classification and nutrition losses using configured task weights."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            import yaml

            with CONFIG_PATH.open(encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file)

        if not isinstance(config, Mapping):
            raise ValueError("Loss config must be a mapping.")
        training_config = config.get("training", config)
        if not isinstance(training_config, Mapping):
            raise ValueError("The 'training' config section must be a mapping.")

        self.lambda_cls = self._get_weight(training_config, "lambda_cls", 1.0)
        self.lambda_reg = self._get_weight(training_config, "lambda_reg", 1.0)
        label_smoothing = training_config.get("label_smoothing", 0.0)

        self.classification_loss = ClassificationLoss(label_smoothing=label_smoothing)
        self.regression_loss = RegressionLoss()

    @staticmethod
    def _get_weight(config: Mapping[str, Any], name: str, default: float) -> float:
        value = config.get(name, default)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(value)
            or value < 0.0
        ):
            raise ValueError(f"{name} must be a finite non-negative number; got {value!r}.")
        return float(value)

    def forward(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor],
        class_targets: torch.Tensor,
        nutrition_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return total, classification, and regression losses, in that order."""
        class_logits, nutrition_outputs = outputs
        classification_loss = self.classification_loss(class_logits, class_targets)
        regression_loss = self.regression_loss(nutrition_outputs, nutrition_targets)
        total_loss = (
            self.lambda_cls * classification_loss + self.lambda_reg * regression_loss
        )
        return total_loss, classification_loss, regression_loss
