"""Run short multi-task training sweeps over classification/regression weights."""

import argparse
import csv
from math import isfinite
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any

import yaml


ML_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ML_DIR.parent
CONFIG_PATH = ML_DIR / "configs" / "base.yaml"

if __package__ in (None, ""):
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ml.training.train import train


# Edit this list to select candidate weights, or pass one or more --pair options.
WEIGHT_PAIRS = [
    (0.5, 0.5),
    (0.5, 1.0),
    (1.0, 0.5),
    (1.0, 1.0),
    (1.0, 2.0),
    (2.0, 1.0),
    (2.0, 2.0),
]

COMPARISON_FIELDS = (
    "lambda_cls",
    "lambda_reg",
    "epochs",
    "val_accuracy",
    "val_nutrition_mae",
    "val_calories_mae",
    "val_protein_mae",
    "val_carbohydrates_mae",
    "val_fat_mae",
)


def _parse_pair(value: str) -> tuple[float, float]:
    try:
        lambda_cls_text, lambda_reg_text = value.split(",", maxsplit=1)
        pair = (float(lambda_cls_text), float(lambda_reg_text))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Expected a comma-separated lambda_cls,lambda_reg pair; got {value!r}."
        ) from error
    if any(not isfinite(weight) or weight < 0 for weight in pair):
        raise argparse.ArgumentTypeError("Loss weights must be finite, non-negative numbers.")
    return pair


def _last_metrics(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as metrics_file:
        reader = csv.DictReader(metrics_file)
        required = {
            "epoch",
            "val_accuracy",
            "val_nutrition_mae",
            "val_calories_mae",
            "val_protein_mae",
            "val_carbohydrates_mae",
            "val_fat_mae",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Training log {path} is missing columns: {', '.join(sorted(missing))}."
            )
        last_row = None
        for line_number, row in enumerate(reader, start=2):
            last_row = row
            last_line_number = line_number

    if last_row is None:
        raise ValueError(f"Training log contains no metrics: {path}")
    try:
        return {
            name: float(last_row[name])
            for name in COMPARISON_FIELDS
            if name not in {"lambda_cls", "lambda_reg", "epochs"}
        }
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Training log {path}:{last_line_number} contains non-numeric metrics."
        ) from error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare validation metrics for candidate multi-task loss weights."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help=f"training config to copy for each run (default: {CONFIG_PATH})",
    )
    parser.add_argument(
        "--pair",
        action="append",
        type=_parse_pair,
        dest="pairs",
        metavar="CLS,REG",
        help="candidate loss-weight pair; repeat to override WEIGHT_PAIRS",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="number of epochs per candidate (default: 3)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ML_DIR / "training" / "logs" / "loss_weight_comparison.csv",
        help="comparison CSV path",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--device", default=None, help="torch device, e.g. cpu or cuda")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be a positive integer.")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be a positive integer.")
    for option in ("max_train_samples", "max_val_samples"):
        value = getattr(args, option)
        if value is not None and value < 1:
            parser.error(f"--{option.replace('_', '-')} must be a positive integer.")
    args.pairs = args.pairs or WEIGHT_PAIRS
    if not args.pairs:
        parser.error("Provide at least one candidate pair in WEIGHT_PAIRS or with --pair.")
    return args


def _train_args(
    config_path: Path,
    log_path: Path,
    checkpoint_dir: Path,
    args: argparse.Namespace,
) -> argparse.Namespace:
    return argparse.Namespace(
        config=config_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=None,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        freeze_backbone_epochs=None,
        unfreeze_layers=None,
        scheduler="none",
        num_workers=0,
        device=args.device,
        checkpoint_dir=checkpoint_dir,
        log_file=log_path,
    )


def run_sweep(args: argparse.Namespace) -> Path:
    with args.config.open(encoding="utf-8") as config_file:
        config: dict[str, Any] = yaml.safe_load(config_file)
    if not isinstance(config, dict) or not isinstance(config.get("training"), dict):
        raise ValueError(f"Expected a training mapping in config file: {args.config}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as comparison_file:
        writer = csv.DictWriter(comparison_file, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        comparison_file.flush()

        with TemporaryDirectory(prefix="loss-weight-sweep-") as temporary_dir:
            temporary_root = Path(temporary_dir)
            for index, (lambda_cls, lambda_reg) in enumerate(args.pairs, start=1):
                run_dir = temporary_root / f"run_{index:03d}"
                run_dir.mkdir()
                run_config = {
                    **config,
                    "training": {
                        **config["training"],
                        "lambda_cls": lambda_cls,
                        "lambda_reg": lambda_reg,
                    },
                }
                config_path = run_dir / "config.yaml"
                with config_path.open("w", encoding="utf-8") as config_file:
                    yaml.safe_dump(run_config, config_file, sort_keys=False)

                metrics_path = run_dir / "metrics.csv"
                train(
                    _train_args(
                        config_path,
                        metrics_path,
                        run_dir / "checkpoints",
                        args,
                    )
                )
                metrics = _last_metrics(metrics_path)
                writer.writerow(
                    {
                        "lambda_cls": lambda_cls,
                        "lambda_reg": lambda_reg,
                        "epochs": args.epochs,
                        **metrics,
                    }
                )
                comparison_file.flush()
                print(
                    f"[{index}/{len(args.pairs)}] lambda_cls={lambda_cls:g}, "
                    f"lambda_reg={lambda_reg:g}: val_accuracy={metrics['val_accuracy']:.4f}, "
                    f"val_nutrition_mae={metrics['val_nutrition_mae']:.4f}"
                )

    print(f"Saved loss-weight comparison to {args.output_csv}")
    return args.output_csv


def main() -> None:
    run_sweep(_parse_args())


if __name__ == "__main__":
    main()
