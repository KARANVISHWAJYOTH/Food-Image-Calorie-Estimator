"""Tests for the Nutrition5k metadata parser."""

import csv
from pathlib import Path

import pytest

from ml.data.prepare_nutrition5k import extract_ingredient_names, read_metadata


def _write_metadata(path: Path, rows: list[list[str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    return path


def test_read_metadata_parses_official_ingredient_tuples(tmp_path: Path) -> None:
    source = _write_metadata(
        tmp_path / "dish_metadata_cafe1.csv",
        [
            [
                "dish_1", "300.5", "193.0", "12.3", "28.2", "18.6",
                "ingr_0001", "white rice", "120.0", "160.0", "0.4", "35.0", "3.0",
                "ingr_0002", "soy sauce", "10.0", "8.0", "0.0", "1.0", "1.0",
            ],
            ["dish_2", "137.5", "88.0", "8.2", "5.1", "10.2"],
        ],
    )

    rows = read_metadata(source)

    assert len(rows) == 2
    assert rows[0]["dish_id"] == "dish_1"
    assert rows[0]["calories"] == pytest.approx(300.5)
    assert rows[0]["protein_g"] == pytest.approx(18.6)
    assert rows[0]["carbohydrates_g"] == pytest.approx(28.2)
    assert rows[0]["fat_g"] == pytest.approx(12.3)
    assert rows[0]["ingredient_names"] == "white rice; soy sauce"
    assert rows[1]["ingredient_names"] == ""


def test_extract_ingredient_names_rejects_truncated_tuples(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="groups of 7"):
        extract_ingredient_names(
            ["dish_3", "1", "2", "3", "4", "5", "ingr_0001", "rice", "10.0"],
            tmp_path / "metadata.csv",
            2,
        )


def test_read_metadata_skips_a_dish_id_header(tmp_path: Path) -> None:
    source = _write_metadata(
        tmp_path / "dish_metadata_cafe2.csv",
        [
            ["dish_id", "calories", "mass", "fat", "carb", "protein"],
            ["dish_4", "100.0", "50.0", "2.0", "12.0", "4.0"],
        ],
    )

    rows = read_metadata(source)

    assert [row["dish_id"] for row in rows] == ["dish_4"]
