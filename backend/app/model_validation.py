"""Validation for model checkpoints before they are served by the API.

A checkpoint is only usable when it contains trained weights, class names,
regression normalization statistics, and held-out validation metrics. This
module intentionally rejects checkpoints with default BatchNorm statistics,
which is the signature of a randomly initialized model.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch

TARGET_NAMES = ("calories", "protein", "carbohydrates", "fat")
TARGET_COUNT = len(TARGET_NAMES)


def _extract_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any] | None:
    state = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
    if isinstance(state, Mapping) and state:
        return state
    return None


def _detected_num_classes(state: Mapping[str, Any]) -> int | None:
    weight = state.get("classification_head.3.weight")
    if isinstance(weight, torch.Tensor) and weight.ndim >= 1:
        return int(weight.shape[0])

    packed = state.get("classification_head.3._packed_params._packed_params")
    try:
        packed_weight = packed[0]
    except (TypeError, IndexError, KeyError):
        packed_weight = None
    if isinstance(packed_weight, torch.Tensor) and packed_weight.ndim >= 1:
        return int(packed_weight.shape[0])
    return None


def _detected_regression_outputs(state: Mapping[str, Any]) -> int | None:
    weight = state.get("regression_head.3.weight")
    if isinstance(weight, torch.Tensor) and weight.ndim >= 1:
        return int(weight.shape[0])

    packed = state.get("regression_head.3._packed_params._packed_params")
    try:
        packed_weight = packed[0]
    except (TypeError, IndexError, KeyError):
        packed_weight = None
    if isinstance(packed_weight, torch.Tensor) and packed_weight.ndim >= 1:
        return int(packed_weight.shape[0])
    return None


def _batch_norm_is_default(state: Mapping[str, Any]) -> bool | None:
    """Return True when every BN layer still has untrained 0/1 statistics."""
    means = [
        value
        for key, value in state.items()
        if key.endswith("running_mean") and isinstance(value, torch.Tensor)
    ]
    variances = [
        value
        for key, value in state.items()
        if key.endswith("running_var") and isinstance(value, torch.Tensor)
    ]
    if not means and not variances:
        return None

    means_default = all(
        torch.allclose(value.to(torch.float32), torch.zeros_like(value, dtype=torch.float32), atol=1e-6)
        for value in means
    )
    variances_default = all(
        torch.allclose(value.to(torch.float32), torch.ones_like(value, dtype=torch.float32), atol=1e-6)
        for value in variances
    )
    return means_default and variances_default


def _as_float_list(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != TARGET_COUNT:
        return None
    try:
        numbers = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(number) for number in numbers):
        return None
    return numbers


def validate_checkpoint(
    checkpoint: Mapping[str, Any],
    expected_num_classes: int | None = None,
) -> dict[str, Any]:
    """Validate a checkpoint and return a serializable report.

    The report contains ``valid``, ``errors``, ``warnings``, the resolved
    normalization statistics, and the held-out metrics. It never mutates the
    checkpoint.
    """
    errors: list[str] = []
    warnings: list[str] = []

    state = _extract_state_dict(checkpoint)
    if state is None:
        errors.append("missing_state_dict")

    class_names = checkpoint.get("class_names")
    if not isinstance(class_names, list) or len(class_names) < 2:
        errors.append("missing_class_names")
        class_names = []
    elif any(not isinstance(name, str) or not name.strip() for name in class_names):
        errors.append("invalid_class_names")

    detected_classes = _detected_num_classes(state) if state else None
    if expected_num_classes is not None and detected_classes is not None:
        if detected_classes != expected_num_classes:
            errors.append(
                f"classifier_output_mismatch: checkpoint={detected_classes}, expected={expected_num_classes}"
            )
    if detected_classes is not None and class_names and len(class_names) != detected_classes:
        errors.append(
            f"class_names_length_mismatch: names={len(class_names)}, classifier={detected_classes}"
        )

    regression_outputs = _detected_regression_outputs(state) if state else None
    if regression_outputs is not None and regression_outputs != TARGET_COUNT:
        errors.append(
            f"regression_output_mismatch: checkpoint={regression_outputs}, expected={TARGET_COUNT}"
        )

    target_stats = checkpoint.get("target_stats")
    mean: list[float] | None = None
    std: list[float] | None = None
    if not isinstance(target_stats, Mapping):
        errors.append("missing_target_stats")
    else:
        mean = _as_float_list(target_stats.get("mean"))
        std = _as_float_list(target_stats.get("std"))
        if mean is None or std is None:
            errors.append("invalid_target_stats")
        if std is not None and any(value <= 1e-8 for value in std):
            errors.append("zero_target_std")
        names = target_stats.get("target_names")
        if names is not None and tuple(names) != TARGET_NAMES:
            errors.append("target_name_order_mismatch")

    val_metrics = checkpoint.get("val_metrics")
    if not isinstance(val_metrics, Mapping):
        errors.append("missing_validation_metrics")
        val_metrics = {}
    else:
        for key in ("accuracy", "calories_mae"):
            value = val_metrics.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                errors.append(f"missing_validation_metrics:{key}")
        accuracy = val_metrics.get("accuracy")
        if isinstance(accuracy, (int, float)) and not isinstance(accuracy, bool):
            if not 0.0 <= float(accuracy) <= 1.0:
                errors.append("invalid_validation_accuracy")

    bn_default = _batch_norm_is_default(state) if state else None
    if bn_default is True:
        errors.append("untrained_batch_norm_statistics")
    elif bn_default is None:
        warnings.append("no_batch_norm_statistics_found")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "num_classes": detected_classes or len(class_names),
        "target_stats": {"mean": mean, "std": std, "target_names": list(TARGET_NAMES)}
        if mean is not None and std is not None
        else None,
        "val_metrics": {
            key: float(value)
            for key, value in val_metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        },
    }


def describe_validation_failure(report: Mapping[str, Any]) -> str:
    errors = report.get("errors") or ["unknown_validation_error"]
    return (
        "The configured checkpoint is not a trained, deployable model "
        f"({', '.join(str(error) for error in errors)}). "
        "Train the model on the documented dataset and place the resulting "
        "best checkpoint at backend/app/best_model.pth, or set MODEL_WEIGHTS_PATH."
    )
