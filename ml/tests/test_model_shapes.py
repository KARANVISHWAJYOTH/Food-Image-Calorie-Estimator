"""Shape checks for the multi-task food CNN."""

import pytest
import torch

from ml.models.multitask_cnn import MultiTaskCNN


@pytest.mark.parametrize("backbone", ["resnet50", "efficientnet_b0"])
def test_multitask_cnn_output_shapes(backbone: str) -> None:
    batch_size = 2
    num_classes = 5
    model = MultiTaskCNN(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=False,
    ).cpu().eval()
    images = torch.randn(batch_size, 3, 64, 64)

    with torch.no_grad():
        class_logits, nutrition_outputs = model(images)

    assert class_logits.shape == (batch_size, num_classes)
    assert nutrition_outputs.shape == (batch_size, 4)
