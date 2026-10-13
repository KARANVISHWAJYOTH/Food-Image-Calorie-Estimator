"""Multi-task CNN for food classification and nutrition regression."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
NUTRITION_OUTPUTS = 4


class MultiTaskCNN(nn.Module):
    """Predict a food class and four nutrition values from an image."""

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 101,
        dropout: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(backbone, str):
            raise ValueError(f"Backbone must be a string; got {backbone!r}.")
        if (
            not isinstance(num_classes, int)
            or isinstance(num_classes, bool)
            or num_classes < 1
        ):
            raise ValueError(f"num_classes must be a positive integer; got {num_classes!r}.")
        if (
            not isinstance(dropout, (int, float))
            or isinstance(dropout, bool)
            or not 0.0 <= dropout < 1.0
        ):
            raise ValueError(f"dropout must be a number in [0, 1); got {dropout!r}.")
        if not isinstance(pretrained, bool):
            raise ValueError(f"pretrained must be a boolean; got {pretrained!r}.")

        backbone_name = backbone.lower()
        if backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            network = models.resnet50(weights=weights)
            feature_size = network.fc.in_features
            self.backbone = nn.Sequential(*list(network.children())[:-1], nn.Flatten())
        elif backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            network = models.efficientnet_b0(weights=weights)
            feature_size = network.classifier[-1].in_features
            self.backbone = nn.Sequential(
                network.features,
                network.avgpool,
                nn.Flatten(),
            )
        else:
            raise ValueError(
                f"Unsupported backbone {backbone!r}. "
                "Choose 'resnet50' or 'efficientnet_b0'."
            )

        self.classification_head = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(512, num_classes),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(256, NUTRITION_OUTPUTS),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return class logits and [calories, protein, carbs, fat] predictions."""
        features = self.backbone(x)
        return self.classification_head(features), self.regression_head(features)

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None) -> "MultiTaskCNN":
        """Build a model from the ``model`` section of the base YAML config."""
        if config is None:
            import yaml

            with CONFIG_PATH.open(encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file)

        if not isinstance(config, Mapping):
            raise ValueError("Model config must be a mapping.")
        model_config = config.get("model", config)
        if not isinstance(model_config, Mapping):
            raise ValueError("The 'model' config section must be a mapping.")

        return cls(
            backbone=model_config.get("backbone", "resnet50"),
            num_classes=model_config.get("num_classes", 101),
            dropout=model_config.get("dropout", 0.3),
            pretrained=model_config.get("pretrained", True),
        )
