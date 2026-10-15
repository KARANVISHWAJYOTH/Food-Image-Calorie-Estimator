"""Save and load versioned multi-task food model checkpoints.

Ensures checkpoints store:
- state_dict (and model_state_dict)
- class_names: list of food category names
- target_stats: normalization parameters (mean and std for z-score de-normalization)
- config: model and training hyperparameters
- val_metrics: validation performance metrics (loss, accuracy, per-target MAE)
- metadata: version, backbone, timestamp, and provenance metadata
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn


REQUIRED_METADATA = ("backbone", "training_config", "version")


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED_METADATA if key not in metadata]
    if missing:
        raise ValueError(f"Checkpoint metadata is missing required fields: {', '.join(missing)}.")
    if not isinstance(metadata["backbone"], str) or not metadata["backbone"]:
        raise ValueError("Checkpoint metadata 'backbone' must be a non-empty string.")
    if not isinstance(metadata["training_config"], Mapping):
        raise ValueError("Checkpoint metadata 'training_config' must be a mapping snapshot.")
    if not isinstance(metadata["version"], str) or not metadata["version"]:
        raise ValueError("Checkpoint metadata 'version' must be a non-empty string.")


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    metadata: Mapping[str, Any],
    class_names: Optional[list[str]] = None,
    target_stats: Optional[Mapping[str, Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
    val_metrics: Optional[Mapping[str, Any]] = None,
) -> None:
    """Save model weights, class names, target stats, config, and val metrics to ``path``."""
    if not isinstance(metadata, Mapping):
        raise ValueError("Checkpoint metadata must be a mapping.")
    _validate_metadata(metadata)

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    meta_dict = dict(metadata)
    resolved_class_names = list(class_names) if class_names is not None else meta_dict.get("class_names")
    resolved_target_stats = dict(target_stats) if target_stats is not None else meta_dict.get("target_stats")
    resolved_config = dict(config) if config is not None else meta_dict.get("config", meta_dict.get("training_config"))
    resolved_val_metrics = dict(val_metrics) if val_metrics is not None else meta_dict.get("val_metrics")

    state_dict = model.state_dict()
    payload = {
        "state_dict": state_dict,
        "model_state_dict": state_dict,
        "class_names": resolved_class_names,
        "target_stats": resolved_target_stats,
        "config": resolved_config,
        "val_metrics": resolved_val_metrics,
        "metadata": meta_dict,
    }

    torch.save(payload, checkpoint_path)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint onto CPU and unpack weights, class names, target stats, config, and val metrics."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint: expected a mapping payload.")

    model_state = checkpoint.get("state_dict", checkpoint.get("model_state_dict"))
    metadata = checkpoint.get("metadata", {})

    if not isinstance(model_state, Mapping):
        # Raw state dict directly saved
        model_state = checkpoint
        metadata = {
            "backbone": "resnet50",
            "training_config": {},
            "version": "legacy",
        }

    # Extract or fallback for all standard checkpoint fields
    class_names = checkpoint.get("class_names", metadata.get("class_names") if isinstance(metadata, Mapping) else None)
    target_stats = checkpoint.get("target_stats", metadata.get("target_stats") if isinstance(metadata, Mapping) else None)
    config = checkpoint.get("config", metadata.get("config", metadata.get("training_config")) if isinstance(metadata, Mapping) else None)
    val_metrics = checkpoint.get("val_metrics", metadata.get("val_metrics") if isinstance(metadata, Mapping) else None)

    return {
        "state_dict": model_state,
        "model_state_dict": model_state,
        "class_names": class_names,
        "target_stats": target_stats,
        "config": config,
        "val_metrics": val_metrics,
        "metadata": metadata if isinstance(metadata, Mapping) else {},
    }
