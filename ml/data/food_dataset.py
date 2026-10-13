"""PyTorch dataset for Food-101 images with class-level nutrition targets."""

from collections import Counter
import logging
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset


ML_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ML_DIR / "configs" / "base.yaml"
NUTRITION_COLUMNS = ("calories", "protein_g", "carbohydrates_g", "fat_g")
LOGGER = logging.getLogger(__name__)


def _resolve_path(value: str | Path, ml_dir: Path = ML_DIR) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ml_dir / path


def _load_config(config_path: Path) -> dict:
    import yaml

    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict) or not isinstance(config.get("dataset"), dict):
        raise ValueError(f"Expected a dataset section in config file: {config_path}")
    return config


def _default_transform(split: str):
    try:
        from ml.preprocessing.transforms import get_eval_transform, get_train_transform
    except ModuleNotFoundError:
        from preprocessing.transforms import get_eval_transform, get_train_transform

    if split == "train":
        return get_train_transform()
    return get_eval_transform()


class Food101NutritionDataset(Dataset):
    """Return ``(image, class_index, nutrition_target)`` for resolved classes.

    Class indices follow Food-101's ``classes.txt`` order. The nutrition target
    is a float32 tensor ordered as calories, protein, carbohydrates, and fat.
    Classes without a complete resolved target are excluded and summarized in a
    warning; image paths and labels are read from the official split manifest.
    """

    def __init__(
        self,
        split: str = "train",
        transform=None,
        data_dir: str | Path | None = None,
        labels_csv: str | Path | None = None,
        config_path: str | Path = CONFIG_PATH,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be either 'train' or 'test'.")

        config_path = Path(config_path)
        config = _load_config(config_path)
        dataset_config = config["dataset"]

        self.split = split
        self.data_dir = _resolve_path(
            data_dir if data_dir is not None else dataset_config["food101_dir"]
        )
        self.food101_dir = (
            self.data_dir
            if self.data_dir.name == "food-101"
            else self.data_dir / "food-101"
        )
        self.labels_csv = _resolve_path(
            labels_csv
            if labels_csv is not None
            else dataset_config["combined_labels_csv"]
        )

        metadata_dir = self.food101_dir / "meta"
        classes_file = metadata_dir / "classes.txt"
        split_file = metadata_dir / f"{split}.txt"
        missing_files = [
            path for path in (classes_file, split_file) if not path.is_file()
        ]
        if missing_files:
            missing = "\n".join(f"  - {path}" for path in missing_files)
            raise FileNotFoundError(
                "Food-101 metadata is missing:\n"
                f"{missing}\n"
                "Run `python ml/data/download_food101.py` to download the dataset."
            )
        if not self.labels_csv.is_file():
            raise FileNotFoundError(
                f"Combined labels CSV not found: {self.labels_csv}. "
                "Run `python ml/data/align_labels.py` first."
            )

        with classes_file.open(encoding="utf-8") as class_file:
            self.classes = [line.strip() for line in class_file if line.strip()]
        if not self.classes or len(self.classes) != len(set(self.classes)):
            raise ValueError(f"Food-101 class list is empty or contains duplicates: {classes_file}")
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}

        import pandas as pd

        labels = pd.read_csv(self.labels_csv)
        required_columns = {"food_class", *NUTRITION_COLUMNS, "match_method"}
        missing_columns = sorted(required_columns - set(labels.columns))
        if missing_columns:
            raise ValueError(
                f"Combined labels CSV is missing required columns: {', '.join(missing_columns)}"
            )
        if labels["food_class"].duplicated().any():
            duplicates = labels.loc[
                labels["food_class"].duplicated(), "food_class"
            ].tolist()
            raise ValueError(
                "Combined labels CSV contains duplicate food_class values: "
                f"{', '.join(duplicates[:10])}"
            )
        labels_by_class = labels.set_index("food_class")

        targets = {}
        skipped_reasons = {}
        for class_name in self.classes:
            if class_name not in labels_by_class.index:
                skipped_reasons[class_name] = "no combined-label row"
                continue

            label = labels_by_class.loc[class_name]
            values = []
            try:
                values = [float(label[column]) for column in NUTRITION_COLUMNS]
            except (TypeError, ValueError):
                pass
            method = str(label["match_method"]).strip().lower()
            if method == "unmatched" or len(values) != len(NUTRITION_COLUMNS):
                skipped_reasons[class_name] = "marked unmatched or missing nutrition values"
                continue
            if not all(math.isfinite(value) and value >= 0 for value in values):
                skipped_reasons[class_name] = "invalid nutrition values"
                continue
            targets[class_name] = tuple(values)

        self.transform = transform if transform is not None else _default_transform(split)
        self.samples = []
        skipped_image_counts = Counter()
        with split_file.open(encoding="utf-8") as manifest:
            for line_number, line in enumerate(manifest, start=1):
                image_id = line.strip()
                if not image_id:
                    continue
                parts = image_id.split("/", maxsplit=1)
                if len(parts) != 2 or parts[0] not in self.class_to_idx:
                    raise ValueError(
                        f"Invalid Food-101 image ID {image_id!r} at "
                        f"{split_file}:{line_number}."
                    )
                class_name, image_stem = parts
                if class_name not in targets:
                    skipped_image_counts[class_name] += 1
                    continue
                image_path = self.food101_dir / "images" / class_name / f"{image_stem}.jpg"
                self.samples.append(
                    (image_path, self.class_to_idx[class_name], targets[class_name])
                )

        if skipped_image_counts:
            skipped_summary = "; ".join(
                f"{name} ({count} images: {skipped_reasons[name]})"
                for name, count in sorted(skipped_image_counts.items())
            )
            LOGGER.warning(
                "Skipped %d %s images whose classes have no resolved nutrition target: %s",
                sum(skipped_image_counts.values()),
                split,
                skipped_summary,
            )
        if not self.samples:
            raise ValueError(
                f"No {split} Food-101 images have resolved nutrition targets. "
                "Review combined_labels.csv and manual overrides."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        from PIL import Image

        image_path, class_index, nutrition_values = self.samples[index]
        if not image_path.is_file():
            raise FileNotFoundError(f"Food-101 image listed in the split manifest is missing: {image_path}")
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        image_tensor = self.transform(image)
        nutrition_target = torch.tensor(nutrition_values, dtype=torch.float32)
        return image_tensor, class_index, nutrition_target