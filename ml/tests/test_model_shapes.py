"""Shape checks for the multi-task food CNN."""

import pytest
import torch

from ml.models.multitask_cnn import (
    MultiTaskCNN,
    _backbone_layers,
    freeze_backbone,
    unfreeze_backbone,
)


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


@pytest.mark.parametrize("backbone", ["resnet50", "efficientnet_b0"])
def test_backbone_can_be_frozen_then_partially_unfrozen(backbone: str) -> None:
    model = MultiTaskCNN(backbone=backbone, pretrained=False)

    freeze_backbone(model)
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.classification_head.parameters())
    assert all(parameter.requires_grad for parameter in model.regression_head.parameters())

    model.train()
    layers = _backbone_layers(model)
    unfreeze_backbone(model, num_layers=1)

    assert all(not parameter.requires_grad for parameter in layers[0].parameters())
    assert all(parameter.requires_grad for parameter in layers[-1].parameters())
    assert not layers[0].training
