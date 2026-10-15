"""Model quantization, optimization, and export pipeline.

Supports:
1. Dynamic INT8 quantization for CPU deployment (smaller size, 2-3x faster inference).
2. TorchScript tracing for GIL-free C++ / production serving.
3. ONNX export for cross-runtime (ONNX Runtime, WebAssembly, mobile).
4. Latency and memory benchmarking.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import torch
from torch import nn
import yaml

ML_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ML_DIR.parent
CONFIG_PATH = ML_DIR / "configs" / "base.yaml"
LOGGER = logging.getLogger("ml.inference")

if __package__ in (None, ""):
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ml.models.checkpoint import load_checkpoint
from ml.models.multitask_cnn import MultiTaskCNN


def quantize_dynamic_model(model: nn.Module) -> nn.Module:
    """Apply dynamic INT8 quantization to linear layers for fast CPU execution."""
    LOGGER.info("Applying dynamic INT8 quantization to Linear layers...")
    model_cpu = model.cpu().eval()
    quantized = torch.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},
        dtype=torch.qint8,
    )
    return quantized


def export_torchscript(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
) -> Path:
    """Export model via TorchScript tracing."""
    LOGGER.info(f"Tracing model with TorchScript to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_eval = model.cpu().eval()
    traced_model = torch.jit.trace(model_eval, dummy_input.cpu())
    torch.jit.save(traced_model, str(output_path))
    return output_path


def export_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
    opset_version: int = 14,
) -> Path:
    """Export multi-task model to ONNX format."""
    LOGGER.info(f"Exporting model to ONNX at {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_eval = model.cpu().eval()

    torch.onnx.export(
        model_eval,
        dummy_input.cpu(),
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_image"],
        output_names=["class_logits", "nutrition_predictions"],
        dynamic_axes={
            "input_image": {0: "batch_size"},
            "class_logits": {0: "batch_size"},
            "nutrition_predictions": {0: "batch_size"},
        },
    )
    return output_path


def benchmark_latency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    warmup: int = 15,
    iterations: int = 100,
) -> dict[str, float]:
    """Measure inference latency (mean, P50, P95, P99 ms) on CPU."""
    model.eval()
    input_tensor = input_tensor.cpu()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    latencies: list[float] = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(input_tensor)
            elapsed = (time.perf_counter() - start) * 1000.0  # ms
            latencies.append(elapsed)

    latencies.sort()
    mean_lat = float(sum(latencies) / len(latencies))
    p50 = float(latencies[int(len(latencies) * 0.50)])
    p95 = float(latencies[int(len(latencies) * 0.95)])
    p99 = float(latencies[int(len(latencies) * 0.99)])
    fps = round(1000.0 / mean_lat, 1) if mean_lat > 0 else 0.0

    return {
        "mean_ms": round(mean_lat, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "throughput_fps": fps,
    }


def get_model_size_mb(path_or_model: Path | nn.Module) -> float:
    """Calculate file size in megabytes."""
    if isinstance(path_or_model, Path):
        return round(path_or_model.stat().st_size / (1024 * 1024), 2)
    # Estimate size in memory
    total_bytes = sum(p.nelement() * p.element_size() for p in path_or_model.parameters())
    return round(total_bytes / (1024 * 1024), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize and export MultiTaskFoodCNN.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to trained FP32 checkpoint (.pth).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to YAML configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ML_DIR / "inference" / "exported",
        help="Directory to save exported model artifacts.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=True,
        help="Apply dynamic INT8 quantization.",
    )
    parser.add_argument(
        "--torchscript",
        action="store_true",
        default=True,
        help="Export TorchScript traced model.",
    )
    parser.add_argument(
        "--onnx",
        action="store_true",
        default=False,
        help="Export ONNX model.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=True,
        help="Run latency benchmark on CPU.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_config = config.get("model", {})
    backbone = model_config.get("backbone", "resnet50")
    num_classes = model_config.get("num_classes", 101)

    model = MultiTaskCNN(backbone=backbone, num_classes=num_classes, pretrained=False)

    if args.checkpoint and args.checkpoint.is_file():
        LOGGER.info(f"Loading weights from {args.checkpoint}...")
        try:
            ckpt = load_checkpoint(args.checkpoint)
            model.load_state_dict(ckpt["model_state_dict"])
        except Exception:
            raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(raw.get("model_state_dict", raw.get("state_dict", raw)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, 224, 224)

    fp32_size = get_model_size_mb(model)
    LOGGER.info(f"Original FP32 Model Size: ~{fp32_size} MB")

    # Benchmark baseline
    if args.benchmark:
        LOGGER.info("Benchmarking original FP32 model...")
        base_stats = benchmark_latency(model, dummy_input)
        LOGGER.info(f"FP32 Stats: Mean={base_stats['mean_ms']}ms, P95={base_stats['p95_ms']}ms, FPS={base_stats['throughput_fps']}")

    # 1. Quantization
    if args.quantize:
        quantized = quantize_dynamic_model(model)
        quantized_path = args.output_dir / "multitask_cnn_int8.pth"
        torch.save(quantized.state_dict(), quantized_path)
        q_size = get_model_size_mb(quantized_path)
        LOGGER.info(f"Saved INT8 Quantized weights to {quantized_path} ({q_size} MB)")

        if args.benchmark:
            LOGGER.info("Benchmarking INT8 Quantized model...")
            q_stats = benchmark_latency(quantized, dummy_input)
            speedup = round(base_stats['mean_ms'] / max(0.001, q_stats['mean_ms']), 2)
            LOGGER.info(f"INT8 Stats: Mean={q_stats['mean_ms']}ms, P95={q_stats['p95_ms']}ms, Speedup={speedup}x")

    # 2. TorchScript Export
    if args.torchscript:
        ts_path = args.output_dir / "multitask_cnn_traced.pt"
        export_torchscript(model, dummy_input, ts_path)
        LOGGER.info(f"TorchScript artifact ready: {ts_path} ({get_model_size_mb(ts_path)} MB)")

    # 3. ONNX Export
    if args.onnx:
        onnx_path = args.output_dir / "multitask_cnn.onnx"
        export_onnx(model, dummy_input, onnx_path)
        LOGGER.info(f"ONNX artifact ready: {onnx_path} ({get_model_size_mb(onnx_path)} MB)")

    LOGGER.info("Model optimization & export complete.")


if __name__ == "__main__":
    main()
