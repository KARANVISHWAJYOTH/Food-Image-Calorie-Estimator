"""Download and extract Food-101 using torchvision."""

import argparse
from pathlib import Path


ML_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ML_DIR / "configs" / "base.yaml"


def configured_data_dir() -> Path:
    import yaml

    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    try:
        configured_path = Path(config["dataset"]["food101_dir"])
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Expected dataset.food101_dir in config file: {CONFIG_PATH}"
        ) from error

    if not configured_path.is_absolute():
        configured_path = ML_DIR / configured_path
    return configured_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract the Food-101 dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="dataset download directory (default: dataset.food101_dir from base.yaml)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir if args.data_dir is not None else configured_data_dir()
    if not data_dir.is_absolute():
        data_dir = ML_DIR / data_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Food-101 into {data_dir}")
    from torchvision.datasets import Food101

    Food101(root=str(data_dir), split="train", download=True)
    print(f"Food-101 is ready at {data_dir / 'food-101'}")


if __name__ == "__main__":
    main()