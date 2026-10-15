"""Unit tests for inference optimization and quantization."""

from pathlib import Path
import tempfile
import torch
from torch import nn

from ml.inference.export_and_quantize import (
    benchmark_latency,
    export_torchscript,
    quantize_dynamic_model,
)


class SimpleMultiTaskHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(8, 3)
        self.regressor = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.flatten(self.pool(self.conv(x)))
        return self.classifier(features), self.regressor(features)


def test_quantize_dynamic_model() -> None:
    model = SimpleMultiTaskHead().eval()
    quantized = quantize_dynamic_model(model)

    dummy_input = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        orig_cls, orig_reg = model(dummy_input)
        q_cls, q_reg = quantized(dummy_input)

    assert q_cls.shape == orig_cls.shape
    assert q_reg.shape == orig_reg.shape


def test_export_torchscript_and_reload() -> None:
    model = SimpleMultiTaskHead().eval()
    dummy_input = torch.randn(1, 3, 16, 16)

    with tempfile.TemporaryDirectory() as tmpdir:
        ts_path = Path(tmpdir) / "model_traced.pt"
        export_torchscript(model, dummy_input, ts_path)
        assert ts_path.is_file()

        loaded_ts = torch.jit.load(str(ts_path))
        with torch.no_grad():
            cls_out, reg_out = loaded_ts(dummy_input)

        assert cls_out.shape == (1, 3)
        assert reg_out.shape == (1, 4)


def test_benchmark_latency() -> None:
    model = SimpleMultiTaskHead().eval()
    dummy_input = torch.randn(1, 3, 16, 16)

    stats = benchmark_latency(model, dummy_input, warmup=2, iterations=5)

    assert "mean_ms" in stats
    assert "p50_ms" in stats
    assert "p95_ms" in stats
    assert "throughput_fps" in stats
    assert stats["mean_ms"] > 0
