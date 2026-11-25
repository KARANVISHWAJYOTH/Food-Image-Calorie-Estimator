"""Fail-fast validation for a trained NutriVision checkpoint.

Usage:
    python ml/training/verify_checkpoint.py --checkpoint backend/app/best_model.pth

Exit code 0 means the checkpoint is deployable. Exit code 1 means it is
incomplete, untrained, or otherwise unsafe to serve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Optional

import torch

ML_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ML_DIR.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from backend.app.model_validation import validate_checkpoint  # noqa: E402


def _default_checkpoint() -> Optional[Path]:
    candidates = [
        REPOSITORY_ROOT / "ml" / "training" / "checkpoints" / "best.pt",
        REPOSITORY_ROOT / "ml" / "training" / "checkpoints" / "best_model.pth",
        REPOSITORY_ROOT / "backend" / "app" / "best_model.pth",
        REPOSITORY_ROOT / "ml" / "optimization" / "best_model_pruned_20.pth",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a NutriVision checkpoint before deployment.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path (.pt/.pth).")
    parser.add_argument("--json", action="store_true", help="Print the raw validation report as JSON.")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or _default_checkpoint()
    if checkpoint_path is None or not checkpoint_path.is_file():
        print("FAIL: no checkpoint found. Expected ml/training/checkpoints/best.pt or backend/app/best_model.pth.")
        return 1

    try:
        payload: Any = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: could not load {checkpoint_path}: {exc}")
        return 1

    if not isinstance(payload, dict):
        print(f"FAIL: {checkpoint_path} is not a checkpoint mapping.")
        return 1

    report = validate_checkpoint(payload)
    if args.json:
        print(json.dumps({"checkpoint": str(checkpoint_path), **report}, indent=2))
    else:
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Valid: {report['valid']}")
        print(f"Classes: {report.get('num_classes')}")
        metrics = report.get("val_metrics") or {}
        if metrics:
            print(
                "Metrics: "
                f"accuracy={metrics.get('accuracy')}, "
                f"calories_mae={metrics.get('calories_mae')}, "
                f"loss={metrics.get('loss')}"
            )
        if report.get("warnings"):
            print(f"Warnings: {', '.join(report['warnings'])}")
        if report.get("errors"):
            print(f"Errors: {', '.join(report['errors'])}")

    if not report["valid"]:
        print(
            "FAIL: this checkpoint must not be deployed. Re-run training and ensure "
            "class_names, target_stats, and val_metrics are saved."
        )
        return 1

    print("PASS: checkpoint is deployable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
