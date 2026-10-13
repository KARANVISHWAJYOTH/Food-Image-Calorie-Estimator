"""Prepare dish-level nutrition labels from the Nutrition5k metadata files."""

import argparse
import csv
import math
from pathlib import Path


ML_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ML_DIR / "configs" / "base.yaml"
METADATA_FILES = (
    "metadata/dish_metadata_cafe1.csv",
    "metadata/dish_metadata_cafe2.csv",
)
OUTPUT_COLUMNS = (
    "dish_id",
    "ingredient_names",
    "calories",
    "protein_g",
    "carbohydrates_g",
    "fat_g",
)


def load_config() -> dict:
    import yaml

    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict) or not isinstance(config.get("dataset"), dict):
        raise ValueError(f"Expected a dataset section in config file: {CONFIG_PATH}")
    return config


def extract_ingredient_names(fields: list[str], source: Path, line_number: int) -> str:
    if len(fields) < 7:
        raise ValueError(
            f"{source}:{line_number}: expected num_ingrs after the six dish-level fields."
        )
    try:
        ingredient_count = int(fields[6])
    except ValueError as error:
        raise ValueError(
            f"{source}:{line_number}: num_ingrs must be an integer."
        ) from error
    if ingredient_count < 0:
        raise ValueError(f"{source}:{line_number}: num_ingrs cannot be negative.")
    if ingredient_count == 0:
        return ""

    ingredient_fields = fields[7:]
    if len(ingredient_fields) % ingredient_count != 0:
        raise ValueError(
            f"{source}:{line_number}: ingredient fields do not match num_ingrs."
        )
    fields_per_ingredient = len(ingredient_fields) // ingredient_count
    if fields_per_ingredient < 2:
        raise ValueError(
            f"{source}:{line_number}: ingredient records must include an ingredient name."
        )

    names = [
        ingredient_fields[index * fields_per_ingredient + 1].strip()
        for index in range(ingredient_count)
    ]
    return "; ".join(name for name in names if name)


def read_metadata(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as metadata_file:
        reader = csv.reader(metadata_file)
        for line_number, fields in enumerate(reader, start=1):
            if line_number == 1 and fields and fields[0].strip().lower() == "dish_id":
                continue
            if len(fields) < 7:
                raise ValueError(
                    f"{path}:{line_number}: expected dish fields followed by num_ingrs."
                )

            dish_id = fields[0].strip()
            if not dish_id:
                raise ValueError(f"{path}:{line_number}: dish_id is empty.")

            try:
                calories, fat_g, carbohydrates_g, protein_g = (
                    float(fields[index]) for index in (1, 3, 4, 5)
                )
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: nutrition fields must be numeric."
                ) from error

            values = (calories, protein_g, carbohydrates_g, fat_g)
            if not all(math.isfinite(value) and value >= 0 for value in values):
                raise ValueError(
                    f"{path}:{line_number}: nutrition values must be finite and non-negative."
                )

            ingredient_names = extract_ingredient_names(fields, path, line_number)
            rows.append(
                {
                    "dish_id": dish_id,
                    "ingredient_names": ingredient_names,
                    "calories": calories,
                    "protein_g": protein_g,
                    "carbohydrates_g": carbohydrates_g,
                    "fat_g": fat_g,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine Nutrition5k dish metadata into a clean nutrition CSV."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="raw dataset directory (default: dataset.nutrition5k_raw_dir from base.yaml)",
    )
    args = parser.parse_args()

    config = load_config()
    try:
        data_dir = args.data_dir or Path(config["dataset"]["nutrition5k_raw_dir"])
        output_path = Path(config["dataset"]["nutrition5k_clean_csv"])
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Expected dataset.nutrition5k_raw_dir and "
            "dataset.nutrition5k_clean_csv in ml/configs/base.yaml."
        ) from error

    if not data_dir.is_absolute():
        data_dir = ML_DIR / data_dir
    if not output_path.is_absolute():
        output_path = ML_DIR / output_path

    missing_files = [data_dir / relative_path for relative_path in METADATA_FILES]
    missing_files = [path for path in missing_files if not path.is_file()]
    if missing_files:
        missing_list = "\n".join(f"  - {path}" for path in missing_files)
        parser.error(
            "Nutrition5k dish metadata is missing:\n"
            f"{missing_list}\n"
            "Download Nutrition5k from the official Google Research dataset page "
            "at https://github.com/google-research-datasets/Nutrition5k#download-data "
            "(the full archive is about 181 GB), then extract or copy the files so "
            f"they are under {data_dir / 'metadata'}."
        )

    rows = []
    for relative_path in METADATA_FILES:
        rows.extend(read_metadata(data_dir / relative_path))

    import pandas as pd

    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if frame.empty:
        parser.error("The Nutrition5k metadata files contained no dish rows.")
    if frame["dish_id"].duplicated().any():
        duplicate_ids = frame.loc[frame["dish_id"].duplicated(), "dish_id"].tolist()
        parser.error(
            "Duplicate dish_id values were found across the metadata files: "
            f"{', '.join(duplicate_ids[:10])}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(f"Saved {len(frame):,} dish records to {output_path}")


if __name__ == "__main__":
    main()