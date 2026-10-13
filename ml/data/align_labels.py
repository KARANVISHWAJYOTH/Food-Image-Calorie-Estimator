"""Align Food-101 classes to Nutrition5k dish labels using text similarity.

These class-level nutrition targets are modeling proxies, not ground truth:
Nutrition5k contains cafeteria dishes, not verified Food-101 class labels.
"""

import argparse
from difflib import SequenceMatcher
import math
from pathlib import Path
import re


ML_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ML_DIR / "configs" / "base.yaml"
NUTRITION_COLUMNS = ("calories", "protein_g", "carbohydrates_g", "fat_g")
OUTPUT_COLUMNS = (
    "food_class",
    *NUTRITION_COLUMNS,
    "match_confidence",
    "match_method",
)
TEXT_COLUMNS = ("dish_name", "category", "ingredient_names")
STRATEGIES = {"fuzzy", "exact"}


def normalize_text(value: str) -> str:
    value = value.replace("_", " ").replace("-", " ").lower()
    return " ".join(re.findall(r"[a-z0-9]+", value))


def candidate_names(record: dict) -> list[str]:
    names = []
    for column in TEXT_COLUMNS:
        value = record.get(column)
        if value is None:
            continue
        try:
            if math.isnan(float(value)):
                continue
        except (TypeError, ValueError):
            pass
        text = str(value).strip()
        if not text:
            continue
        if column == "ingredient_names":
            names.extend(part.strip() for part in re.split(r"[;,|]", text) if part.strip())
        else:
            names.append(text)
    return names


def text_similarity(food_class: str, name: str, strategy: str) -> float:
    normalized_class = normalize_text(food_class)
    normalized_name = normalize_text(name)
    if not normalized_class or not normalized_name:
        return 0.0
    if strategy == "exact":
        return float(normalized_class == normalized_name)
    return SequenceMatcher(None, normalized_class, normalized_name, autojunk=False).ratio()


def load_overrides(path: Path, food_classes: set[str]) -> dict:
    import yaml

    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as override_file:
        data = yaml.safe_load(override_file) or {}
    overrides = data.get("overrides", data) if isinstance(data, dict) else None
    if not isinstance(overrides, dict):
        raise ValueError(f"Expected an 'overrides' mapping in {path}.")

    unknown_classes = sorted(set(overrides) - food_classes)
    if unknown_classes:
        raise ValueError(
            f"Manual overrides in {path} reference unknown Food-101 classes: "
            f"{', '.join(unknown_classes)}"
        )
    for food_class, values in overrides.items():
        if not isinstance(values, dict):
            raise ValueError(f"Override for {food_class!r} must be a mapping of nutrition values.")
        missing = [column for column in NUTRITION_COLUMNS if column not in values]
        if missing:
            raise ValueError(f"Override for {food_class!r} is missing: {', '.join(missing)}")
        for column in NUTRITION_COLUMNS:
            value = float(values[column])
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"Override {food_class!r}.{column} must be finite and non-negative."
                )
    return overrides


def load_config(path: Path) -> dict:
    import yaml

    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict) or not isinstance(config.get("dataset"), dict):
        raise ValueError(f"Expected a dataset section in config file: {path}")
    return config


def resolve_ml_path(value: str | Path, ml_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ml_dir / path


def align_labels(
    food_classes: list[str],
    nutrition_records: list[dict],
    overrides: dict,
    strategy: str,
    threshold: float,
) -> tuple[list[dict], list[str]]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unsupported matching strategy: {strategy}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Confidence threshold must be between 0 and 1.")

    named_records = []
    for record in nutrition_records:
        names = candidate_names(record)
        if names:
            named_records.append((record, names))

    if not named_records and not overrides:
        raise ValueError(
            "The Nutrition5k CSV has no dish_name, category, or ingredient_names text. "
            "Run ml/data/prepare_nutrition5k.py to regenerate it with ingredient names, "
            "or provide manual overrides. Dish IDs are not used as food names."
        )

    aligned = []
    unmatched = []
    for food_class in food_classes:
        if food_class in overrides:
            override = overrides[food_class]
            row = {column: float(override[column]) for column in NUTRITION_COLUMNS}
            confidence = 1.0
            method = "manual_override"
        else:
            matches = []
            for record, names in named_records:
                score = max(
                    text_similarity(food_class, name, strategy) for name in names
                )
                if score >= threshold:
                    matches.append((record, score))

            if matches:
                row = {
                    column: sum(float(record[column]) for record, _ in matches)
                    / len(matches)
                    for column in NUTRITION_COLUMNS
                }
                confidence = sum(score for _, score in matches) / len(matches)
                method = f"{strategy}_name_average"
            else:
                row = {column: float("nan") for column in NUTRITION_COLUMNS}
                confidence = 0.0
                method = "unmatched"
                unmatched.append(food_class)

        aligned.append(
            {
                "food_class": food_class,
                **row,
                "match_confidence": confidence,
                "match_method": method,
            }
        )
    return aligned, unmatched


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create class-level Nutrition5k proxy targets for Food-101. "
            "Text matches are assumptions, not ground-truth class labels."
        )
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--class-file", type=Path, default=None)
    parser.add_argument("--nutrition-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--unmatched-output", type=Path, default=None)
    parser.add_argument("--overrides", type=Path, default=None)
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default=None)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    import pandas as pd

    config = load_config(args.config)
    dataset_config = config["dataset"]
    alignment_config = config.get("label_alignment", {})

    food101_dir = resolve_ml_path(dataset_config["food101_dir"], ML_DIR)
    class_file = args.class_file or food101_dir / "food-101" / "meta" / "classes.txt"
    nutrition_csv = args.nutrition_csv or resolve_ml_path(
        dataset_config["nutrition5k_clean_csv"], ML_DIR
    )
    output_path = args.output or resolve_ml_path(
        dataset_config["combined_labels_csv"], ML_DIR
    )
    overrides_path = args.overrides or resolve_ml_path(
        alignment_config.get("manual_overrides", "data/manual_overrides.yaml"), ML_DIR
    )
    unmatched_path = args.unmatched_output or resolve_ml_path(
        alignment_config.get("unmatched_classes_csv", "data/unmatched_classes.csv"), ML_DIR
    )
    strategy = args.strategy or alignment_config.get("strategy", "fuzzy")
    threshold = args.threshold
    if threshold is None:
        threshold = float(alignment_config.get("confidence_threshold", 0.82))

    if not class_file.is_file():
        parser.error(
            f"Food-101 class list not found: {class_file}. "
            "Run `python ml/data/download_food101.py` first."
        )
    if not nutrition_csv.is_file():
        parser.error(
            f"Nutrition5k clean CSV not found: {nutrition_csv}. "
            "Run `python ml/data/prepare_nutrition5k.py` first."
        )

    with class_file.open(encoding="utf-8") as class_list_file:
        food_classes = [line.strip() for line in class_list_file if line.strip()]
    if len(food_classes) != len(set(food_classes)):
        parser.error(f"Food-101 class list contains duplicate classes: {class_file}")

    nutrition = pd.read_csv(nutrition_csv)
    required_columns = {"dish_id", *NUTRITION_COLUMNS}
    missing_columns = sorted(required_columns - set(nutrition.columns))
    if missing_columns:
        parser.error(
            f"Nutrition CSV is missing required columns: {', '.join(missing_columns)}"
        )
    for column in NUTRITION_COLUMNS:
        nutrition[column] = pd.to_numeric(nutrition[column], errors="coerce")
    if nutrition[list(NUTRITION_COLUMNS)].isna().any().any():
        parser.error("Nutrition CSV contains missing or non-numeric target values.")
    if (nutrition[list(NUTRITION_COLUMNS)] < 0).any().any():
        parser.error("Nutrition CSV contains negative target values.")

    overrides = load_overrides(overrides_path, set(food_classes))
    aligned_rows, unmatched_classes = align_labels(
        food_classes,
        nutrition.to_dict(orient="records"),
        overrides,
        strategy,
        threshold,
    )
    output = pd.DataFrame(aligned_rows, columns=OUTPUT_COLUMNS)
    unmatched_output = pd.DataFrame({"food_class": unmatched_classes})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    unmatched_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    unmatched_output.to_csv(unmatched_path, index=False)

    method_counts = output["match_method"].value_counts().to_dict()
    print(f"Saved {len(output)} class labels to {output_path}")
    print(f"Saved {len(unmatched_classes)} unmatched classes to {unmatched_path}")
    print(f"Match counts: {method_counts}")
    print(
        "Caution: matched nutrition values are class-level proxies averaged from "
        "text-selected cafeteria dishes; they are not Food-101 ground truth."
    )


if __name__ == "__main__":
    main()