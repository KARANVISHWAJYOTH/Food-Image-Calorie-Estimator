"""Post-training quantization module for MultiTaskFoodCNN.

Supports:
1. Dynamic INT8 quantization on linear layers (default for CPU deployment).
2. Static INT8 quantization for ResNet-50 when calibration data is available.
3. Produces best_model_int8.pth with full metadata preservation for predictor.py.
"""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
import sys
from typing import Any, Mapping, Optional

import torch
from torch import nn
import yaml

ML_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ML_DIR.parent
CONFIG_PATH = ML_DIR / "configs" / "base.yaml"
DEFAULT_OUTPUT_PATH = ML_DIR / "optimization" / "best_model_int8.pth"
LOGGER = logging.getLogger("ml.optimization.quantize")

if __package__ in (None, ""):
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ml.models.checkpoint import load_checkpoint
from ml.models.multitask_cnn import MultiTaskCNN


def quantize_dynamic(model: nn.Module) -> nn.Module:
    """Apply dynamic INT8 quantization to Linear layers on CPU.

    Converts floating-point weights to signed 8-bit integers (qint8)
    while dynamically quantizing activations during inference.
    """
    LOGGER.info("Applying dynamic INT8 quantization to Linear layers...")
    model_cpu = model.cpu().eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},
        dtype=torch.qint8,
    )
    return quantized_model


def quantize_static(
    model: nn.Module,
    calibration_loader: Optional[torch.utils.data.DataLoader] = None,
    num_calibration_batches: int = 10,
) -> nn.Module:
    """Attempt static post-training quantization on CPU.

    Fuses Conv+BatchNorm+ReLU where possible and quantizes activations
    and weights using representative calibration data. Falls back to
    dynamic quantization if static quantization graph conversion fails.
    """
    LOGGER.info("Attempting static INT8 post-training quantization (PTQ)...")
    model_cpu = copy.deepcopy(model).cpu().eval()

    try:
        # Check for FX graph mode static quantization
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
        from torch.ao.quantization import get_default_qconfig_mapping

        qconfig_mapping = get_default_qconfig_mapping("fbgemm")
        example_inputs = (torch.randn(1, 3, 224, 224),)

        prepared_model = prepare_fx(model_cpu, qconfig_mapping, example_inputs)

        # Run calibration
        if calibration_loader is not None:
            LOGGER.info(f"Calibrating static quantizer over {num_calibration_batches} batches...")
            with torch.no_grad():
                for i, batch in enumerate(calibration_loader):
                    if i >= num_calibration_batches:
                        break
                    images = batch[0] if isinstance(batch, (list, tuple)) else batch
                    prepared_model(images.cpu())
        else:
            LOGGER.info("No calibration loader provided; running calibration on dummy batches...")
            with torch.no_grad():
                for _ in range(5):
                    prepared_model(torch.randn(4, 3, 224, 224))

        quantized_model = convert_fx(prepared_model)
        LOGGER.info("Static FX INT8 quantization succeeded.")
        return quantized_model
    except Exception as exc:
        LOGGER.warning(
            f"Static FX quantization failed ({exc}). Falling back to dynamic INT8 quantization."
        )
        return quantize_dynamic(model)


def get_model_size_mb(path_or_model: Path | nn.Module) -> float:
    """Return model file size or parameter memory in megabytes."""
    if isinstance(path_or_model, Path):
        return round(path_or_model.stat().st_size / (1024 * 1024), 2)
    total_bytes = sum(p.nelement() * p.element_size() for p in path_or_model.parameters())
    return round(total_bytes / (1024 * 1024), 2)


def save_quantized_checkpoint(
    quantized_model: nn.Module,
    output_path: Path,
    class_names: Optional[list[str]] = None,
    target_stats: Optional[Mapping[str, Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
    val_metrics: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    method: str = "dynamic",
) -> Path:
    """Save the quantized model with complete metadata for predictor.py integration."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(metadata or {})
    meta["quantized"] = True
    meta["quantization_method"] = method

    # Save model object directly for seamless PyTorch loading of quantized modules,
    # alongside state_dict and complete metadata
    payload = {
        "model": quantized_model,
        "state_dict": quantized_model.state_dict(),
        "model_state_dict": quantized_model.state_dict(),
        "is_quantized": True,
        "quantization_method": method,
        "class_names": list(class_names) if class_names is not None else None,
        "target_stats": dict(target_stats) if target_stats is not None else None,
        "config": dict(config) if config is not None else None,
        "val_metrics": dict(val_metrics) if val_metrics is not None else None,
        "metadata": meta,
    }

    torch.save(payload, output_path)
    LOGGER.info(f"Saved INT8 quantized model checkpoint to: {output_path}")
    return output_path


def _find_fp32_checkpoint() -> Optional[Path]:
    """Find candidate best_model.pth or best.pt files."""
    candidates = [
        ML_DIR / "training" / "checkpoints" / "best_model.pth",
        REPOSITORY_ROOT / "best_model.pth",
        REPOSITORY_ROOT / "backend" / "app" / "best_model.pth",
        ML_DIR / "training" / "checkpoints" / "best.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize MultiTaskFoodCNN model to INT8.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to FP32 model checkpoint (defaults to best_model.pth).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output path for best_model_int8.pth.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="dynamic",
        choices=["dynamic", "static"],
        help="Quantization technique: 'dynamic' (default) or 'static'.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_config = config.get("model", {})
    backbone = model_config.get("backbone", "resnet50")
    num_classes = model_config.get("num_classes", 101)

    checkpoint_path = args.checkpoint or _find_fp32_checkpoint()
    class_names = None
    target_stats = None
    val_metrics = None
    metadata = {}

    model = MultiTaskCNN(backbone=backbone, num_classes=num_classes, pretrained=False)

    if checkpoint_path and checkpoint_path.is_file():
        LOGGER.info(f"Loading FP32 weights from: {checkpoint_path}")
        try:
            ckpt = load_checkpoint(checkpoint_path)
            model.load_state_dict(ckpt["model_state_dict"])
            class_names = ckpt.get("class_names")
            target_stats = ckpt.get("target_stats")
            val_metrics = ckpt.get("val_metrics")
            metadata = ckpt.get("metadata", {})
        except Exception:
            raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(raw.get("model_state_dict", raw.get("state_dict", raw)))
    else:
        LOGGER.warning("No FP32 checkpoint found. Quantizing uninitialized baseline model.")

    model.eval()
    fp32_size = get_model_size_mb(model)
    LOGGER.info(f"FP32 Model Parameter Footprint: ~{fp32_size} MB")

    if args.method == "static":
        quantized_model = quantize_static(model)
    else:
        quantized_model = quantize_dynamic(model)

    # Save to requested output path
    saved_path = save_quantized_checkpoint(
        quantized_model,
        args.output,
        class_names=class_names,
        target_stats=target_stats,
        config=config,
        val_metrics=val_metrics,
        metadata=metadata,
        method=args.method,
    )

    # Also save a copy to backend/app/best_model_int8.pth if available
    backend_copy = REPOSITORY_ROOT / "backend" / "app" / "best_model_int8.pth"
    try:
        backend_copy.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.load(saved_path, weights_only=False), backend_copy)
        LOGGER.info(f"Synced INT8 model to backend at: {backend_copy}")
    except Exception as e:
        LOGGER.debug(f"Could not copy to backend/app: {e}")

    final_size_mb = get_model_size_mb(saved_path)
    LOGGER.info(f"INT8 Checkpoint File Size: {final_size_mb} MB")

    # Quick forward verification with dummy input
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out_cls, out_reg = quantized_model(dummy_input)
    assert out_cls.shape == (2, num_classes)
    assert out_reg.shape == (2, 4)
    LOGGER.info("Quantized model sanity forward pass verified successfully.")


if __name__ == "__main__":
    main()
