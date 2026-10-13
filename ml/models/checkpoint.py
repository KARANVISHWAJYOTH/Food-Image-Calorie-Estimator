"""Save and load versioned multi-task food model checkpoints."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
) -> None:
    """Save model weights and required provenance metadata to ``path``."""
    if not isinstance(metadata, Mapping):
        raise ValueError("Checkpoint metadata must be a mapping.")
    _validate_metadata(metadata)

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": dict(metadata),
        },
        checkpoint_path,
    )


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint onto CPU and validate its weights and metadata fields."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint: expected a mapping payload.")

    model_state = checkpoint.get("model_state_dict")
    metadata = checkpoint.get("metadata")
    if not isinstance(model_state, Mapping):
        raise ValueError("Invalid checkpoint: missing 'model_state_dict' mapping.")
    if not isinstance(metadata, Mapping):
        raise ValueError("Invalid checkpoint: missing 'metadata' mapping.")
    _validate_metadata(metadata)
    return checkpoint
