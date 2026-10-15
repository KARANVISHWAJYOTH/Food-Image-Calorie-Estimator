"""Unit tests for ml/optimization modules (quantize, prune, benchmark) and backend MODEL_VARIANT."""

import os
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch
from torch import nn

from ml.models.multitask_cnn import MultiTaskCNN
from ml.optimization.quantize import (
    quantize_dynamic,
    save_quantized_checkpoint,
    get_model_size_mb,
)
from ml.optimization.prune import (
    apply_l1_unstructured_pruning,
    calculate_conv_sparsity,
    remove_pruning_reparameterization,
    get_conv_layers,
)
from ml.optimization.benchmark import (
    benchmark_latency,
    generate_tables,
    load_model_variant,
)
from backend.app.predictor import FoodPredictor


class SimpleTestCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc_cls = nn.Linear(16, 5)
        self.fc_reg = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        feat = self.flatten(self.pool(x))
        return self.fc_cls(feat), self.fc_reg(feat)


def test_quantize_dynamic_and_save():
    model = SimpleTestCNN().eval()
    dummy_input = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        fp32_cls, fp32_reg = model(dummy_input)

    quantized = quantize_dynamic(model)
    with torch.no_grad():
        q_cls, q_reg = quantized(dummy_input)

    assert q_cls.shape == fp32_cls.shape
    assert q_reg.shape == fp32_reg.shape

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "test_int8.pth"
        saved = save_quantized_checkpoint(
            quantized,
            out_path,
            class_names=["a", "b", "c", "d", "e"],
            target_stats={"mean": [500, 25, 40, 15], "std": [150, 10, 15, 8]},
            method="dynamic",
        )
        assert out_path.is_file()
        assert out_path.stat().st_size > 0

        # Verify reload
        loaded_model, stats = load_model_variant(out_path, num_classes=5)
        assert loaded_model is not None
        assert stats is not None
        assert "mean" in stats


def test_l1_pruning_and_reparameterization_removal():
    model = SimpleTestCNN()
    conv_layers = get_conv_layers(model)
    assert len(conv_layers) == 2

    # Prune 30%
    apply_l1_unstructured_pruning(model, amount=0.30)
    stats = calculate_conv_sparsity(model)
    assert 25.0 <= stats["sparsity_percent"] <= 35.0

    # Verify mask buffers exist
    assert hasattr(model.conv1, "weight_mask")
    assert hasattr(model.conv1, "weight_orig")

    # Remove reparameterization
    remove_pruning_reparameterization(model)
    assert not hasattr(model.conv1, "weight_mask")
    assert not hasattr(model.conv1, "weight_orig")

    # Verify zeros are permanently present in weights
    final_stats = calculate_conv_sparsity(model)
    assert 25.0 <= final_stats["sparsity_percent"] <= 35.0


def test_benchmark_latency_and_table_generation():
    model = SimpleTestCNN().eval()
    dummy_input = torch.randn(1, 3, 32, 32)
    device = torch.device("cpu")

    stats = benchmark_latency(model, dummy_input, device, warmup=3, iterations=10)
    assert "mean_ms" in stats
    assert "p95_ms" in stats
    assert "throughput_fps" in stats
    assert stats["mean_ms"] > 0

    dummy_results = [
        {
            "name": "FP32 Baseline",
            "type": "FP32",
            "size_mb": 98.4,
            "cpu_mean_ms": 110.0,
            "cpu_p50_ms": 108.0,
            "cpu_p95_ms": 125.0,
            "cpu_fps": 9.1,
            "gpu_mean_ms": 12.5,
            "top1_acc": 82.5,
            "top5_acc": 95.0,
            "calorie_mae": 68.2,
        },
        {
            "name": "INT8 Quantized",
            "type": "INT8 Dynamic",
            "size_mb": 35.2,
            "cpu_mean_ms": 45.0,
            "cpu_p50_ms": 44.0,
            "cpu_p95_ms": 52.0,
            "cpu_fps": 22.2,
            "gpu_mean_ms": None,
            "top1_acc": 81.9,
            "top5_acc": 94.6,
            "calorie_mae": 69.1,
        },
    ]

    md, csv_rows = generate_tables(dummy_results)
    assert "FP32 Baseline" in md
    assert "INT8 Quantized" in md
    assert "CPU Speedup" in md
    assert len(csv_rows) == 2


def test_predictor_model_variant_env():
    # Test FP32 default variant
    predictor_fp32 = FoodPredictor(variant="fp32")
    assert predictor_fp32.model_variant == "fp32"

    # Test INT8 variant
    predictor_int8 = FoodPredictor(variant="int8")
    assert predictor_int8.model_variant == "int8"
    assert str(predictor_int8.device) == "cpu"

    # Test loading quantized model via FoodPredictor
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = MultiTaskCNN(num_classes=10, pretrained=False).eval()
        quantized = quantize_dynamic(model)
        checkpoint_path = Path(tmp_dir) / "best_model_int8.pth"
        save_quantized_checkpoint(
            quantized,
            checkpoint_path,
            class_names=[f"dish_{i}" for i in range(10)],
            target_stats={"mean": [500, 30, 50, 20], "std": [150, 10, 20, 8]},
            method="dynamic",
        )

        p = FoodPredictor(weights_path=str(checkpoint_path), variant="int8")
        assert p.is_torch_ready is True
        assert p.model_variant == "int8"
        assert len(p.classes) == 10
