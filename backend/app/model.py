"""
Multi-Task Deep Convolutional Neural Network (CNN) for Food Identification & Nutritional Regression
Architecture combines a shared deep convolutional backbone with dual specialized output heads:
- Classification Head: Multi-class softmax logits for food category identification
- Regression Heads: Continuous parameter estimation for Calories (kcal), Protein (g), Carbs (g), and Fat (g)
"""

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class MultiTaskFoodCNN(nn.Module):
        def __init__(self, num_classes: int = 10, pretrained: bool = True, dropout_rate: float = 0.3):
            super(MultiTaskFoodCNN, self).__init__()
            
            # Backbone Feature Extractor (ResNet50 / ResNet18 default)
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            
            # Extract convolutional feature representations
            in_features = backbone.fc.in_features
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            
            # Shared Latent Bottleneck
            self.shared_dense = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate)
            )
            
            # Task Head 1: Food Classification Head
            self.classifier_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate / 2),
                nn.Linear(256, num_classes)
            )
            
            # Task Head 2: Calorie Regression Head (Outputs scalar kcal)
            self.calorie_head = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
                nn.ReLU() # Calories cannot be negative
            )
            
            # Task Head 3: Macronutrient Regression Head (Outputs [Protein, Carbs, Fat, Fiber])
            self.macro_head = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 4),
                nn.ReLU() # Macros cannot be negative
            )
            
        def forward(self, x: torch.Tensor):
            # Extract features from backbone
            features = self.backbone(x)
            latent = self.shared_dense(features)
            
            # Multi-task predictions
            class_logits = self.classifier_head(latent)
            calories = self.calorie_head(latent)
            macros = self.macro_head(latent)
            
            return {
                "class_logits": class_logits,
                "calories": calories,
                "macros": macros
            }
else:
    class MultiTaskFoodCNN:
        """Stub fallback if PyTorch environment is missing locally"""
        def __init__(self, *args, **kwargs):
            pass
