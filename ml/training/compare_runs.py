"""Plot validation loss and accuracy from multiple training CSV logs."""

import argparse
import csv
from pathlib import Path


def _read_metrics(path: Path) -> tuple[list[int], list[float], list[float]]:
    with path.open(newline="", encoding="utf-8") as metrics_file:
        reader = csv.DictReader(metrics_file)
        required = {"epoch", "val_loss", "val_accuracy"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {', '.join(sorted(missing))}"
            )

        epochs: list[int] = []
        val_losses: list[float] = []
        val_accuracies: list[float] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                epochs.append(int(row["epoch"]))
                val_losses.append(float(row["val_loss"]))
                val_accuracies.append(float(row["val_accuracy"]))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path}:{line_number}: epoch, val_loss, and val_accuracy "
                    "must be numeric."
                ) from error

    if not epochs:
        raise ValueError(f"No metric rows found in {path}.")
    return epochs, val_losses, val_accuracies


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare validation loss and accuracy across training log CSVs."
    )
    parser.add_argument("logs", nargs="+", type=Path, help="one or more training CSV logs")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml/training/logs/run_comparison.png"),
        help="plot output path (default: ml/training/logs/run_comparison.png)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="optional run labels, one per CSV (defaults to each file name)",
    )
    parser.add_argument("--show", action="store_true", help="also display the plot window")
    args = parser.parse_args()
    if args.labels is not None and len(args.labels) != len(args.logs):
        parser.error("Provide exactly one --labels value for each training log.")

    import matplotlib.pyplot as plt

    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(13, 5))
    labels = args.labels or [path.stem for path in args.logs]
    for path, label in zip(args.logs, labels):
        if not path.is_file():
            parser.error(f"Training log does not exist: {path}")
        epochs, losses, accuracies = _read_metrics(path)
        loss_axis.plot(epochs, losses, marker="o", label=label)
        accuracy_axis.plot(epochs, accuracies, marker="o", label=label)

    loss_axis.set(title="Validation loss", xlabel="Epoch", ylabel="Loss")
    accuracy_axis.set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy")
    for axis in (loss_axis, accuracy_axis):
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(f"Saved comparison plot to {args.output}")
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
