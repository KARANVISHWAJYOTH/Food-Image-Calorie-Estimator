"""Multi-Task Deep Convolutional Neural Network (CNN) for Food Identification & Nutritional Regression.

Output heads:
- classification_head: Multi-class softmax logits for food category identification (num_classes).
- regression_head: Continuous parameter estimation for the 4 ground-truth nutrition targets:
  [Calories (kcal), Protein (g), Carbohydrates (g), Fat (g)].

Note on Dietary Fiber:
Nutrition5k dataset ground truth only records total calories, protein, carbohydrates, and fat.
Dietary fiber is not recorded by Nutrition5k sensor logs. To avoid training an ungrounded
head on synthetic noise, the neural network strictly predicts the 4 ground-truth targets,
and dietary fiber is derived deterministically in predictor.py from carbohydrates and food category.
"""

from typing import Any, Tuple

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


NUTRITION_OUTPUTS = 4
NUTRITION_TARGET_NAMES = ("calories", "protein", "carbohydrates", "fat")


if TORCH_AVAILABLE:
    class MultiTaskModelOutput(tuple):
        """Tuple subclass supporting both tuple unpacking (logits, nutrition) and dict/attribute access."""
        def __new__(cls, class_logits: torch.Tensor, regression_outputs: torch.Tensor):
            return super().__new__(cls, (class_logits, regression_outputs))

        @property
        def class_logits(self) -> torch.Tensor:
            return self[0]

        @property
        def regression_outputs(self) -> torch.Tensor:
            return self[1]

        @property
        def nutrition(self) -> torch.Tensor:
            return self[1]

        @property
        def calories(self) -> torch.Tensor:
            return self[1][:, 0:1]

        @property
        def macros(self) -> torch.Tensor:
            return self[1][:, 1:]

        def __getitem__(self, key: Any) -> Any:
            if isinstance(key, str):
                if key == "class_logits":
                    return self[0]
                if key in ("nutrition", "regression_outputs"):
                    return self[1]
                if key == "calories":
                    return self[1][:, 0:1]
                if key == "macros":
                    return self[1][:, 1:]
                raise KeyError(f"Invalid key {key!r}. Use 'class_logits', 'nutrition', 'calories', or 'macros'.")
            return super().__getitem__(key)


    class MultiTaskFoodCNN(nn.Module):
        """Unified Multi-Task Food CNN compatible with checkpoints from ml/training/train.py."""

        def __init__(
            self,
            backbone: str = "resnet50",
            num_classes: int = 101,
            dropout: float = 0.3,
            pretrained: bool = True,
            **kwargs: Any,
        ) -> None:
            super().__init__()
            backbone_name = backbone.lower()
            self.backbone_name = backbone_name

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
                    f"Unsupported backbone {backbone!r}. Choose 'resnet50' or 'efficientnet_b0'."
                )

            # Task Head 1: Food Classification Head
            self.classification_head = nn.Sequential(
                nn.Linear(feature_size, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=float(dropout)),
                nn.Linear(512, num_classes),
            )

            # Task Head 2: 4-Target Nutrition Regression Head [Calories, Protein, Carbs, Fat]
            self.regression_head = nn.Sequential(
                nn.Linear(feature_size, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=float(dropout)),
                nn.Linear(256, NUTRITION_OUTPUTS),
            )

        def forward(self, x: torch.Tensor) -> MultiTaskModelOutput:
            features = self.backbone(x)
            class_logits = self.classification_head(features)
            regression_outputs = self.regression_head(features)
            return MultiTaskModelOutput(class_logits, regression_outputs)

    # Alias for interoperability across modules
    MultiTaskCNN = MultiTaskFoodCNN

else:
    class MultiTaskFoodCNN:  # type: ignore[no-redef]
        """Stub fallback if PyTorch environment is missing locally."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    MultiTaskCNN = MultiTaskFoodCNN  # type: ignore[misc]
