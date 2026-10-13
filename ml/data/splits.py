"""Create and reuse deterministic, class-stratified dataset splits."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile


ML_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ML_DIR / "configs" / "base.yaml"
SPLIT_NAMES = ("train", "val", "test")


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> dict:
    ratios = {
        "train": float(train_ratio),
        "val": float(val_ratio),
        "test": float(test_ratio),
    }
    if any(not math.isfinite(ratio) or ratio <= 0 for ratio in ratios.values()):
        raise ValueError("Each split ratio must be a finite positive number.")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"Split ratios must sum to 1.0; got {sum(ratios.values()):.8f}.")
    return ratios


def _sample_metadata(sample: tuple) -> dict:
    if len(sample) < 2:
        raise ValueError("Each sample must contain an image path and class index.")
    image_path = Path(sample[0])
    if len(image_path.parts) < 2:
        raise ValueError(f"Cannot derive a stable image ID from path: {image_path}")
    return {
        "image_id": "/".join(image_path.parts[-2:]),
        "class_index": int(sample[1]),
    }


def _fingerprint(sample_metadata: list[dict]) -> str:
    encoded = json.dumps(sample_metadata, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _allocate_counts(class_size: int, ratios: dict[str, float]) -> dict[str, int]:
    names = list(SPLIT_NAMES)
    exact = {name: class_size * ratios[name] for name in names}
    counts = {name: math.floor(exact[name]) for name in names}
    remaining = class_size - sum(counts.values())
    remainder_order = sorted(
        names,
        key=lambda name: (exact[name] - counts[name], -names.index(name)),
        reverse=True,
    )
    for name in remainder_order[:remaining]:
        counts[name] += 1

    if class_size >= len(names):
        for name in names:
            if counts[name] == 0:
                donors = [candidate for candidate in names if counts[candidate] > 1]
                donor = max(
                    donors,
                    key=lambda candidate: (counts[candidate], ratios[candidate], -names.index(candidate)),
                )
                counts[donor] -= 1
                counts[name] += 1
    return counts


def create_split_manifest(
    samples: list[tuple],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """Build stratified splits and store both indices and portable sample IDs.

    ``samples`` uses the ``(image_path, class_index, nutrition_target)`` shape
    from :class:`Food101NutritionDataset`. Each class is split independently;
    classes with at least three examples are represented in every split.
    """
    ratios = _validate_ratios(train_ratio, val_ratio, test_ratio)
    if not samples:
        raise ValueError("Cannot split an empty dataset.")

    metadata = [_sample_metadata(sample) for sample in samples]
    image_ids = [sample["image_id"] for sample in metadata]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Duplicate image IDs found; refusing to create overlapping splits.")

    class_indices = {}
    for sample_index, sample in enumerate(metadata):
        class_indices.setdefault(sample["class_index"], []).append(sample_index)

    randomizer = random.Random(seed)
    split_indices = {name: [] for name in SPLIT_NAMES}
    for class_index in sorted(class_indices):
        indices = class_indices[class_index]
        randomizer.shuffle(indices)
        counts = _allocate_counts(len(indices), ratios)
        start = 0
        for split_name in SPLIT_NAMES:
            end = start + counts[split_name]
            split_indices[split_name].extend(indices[start:end])
            start = end

    for split_name in SPLIT_NAMES:
        split_indices[split_name].sort()

    split_samples = {
        split_name: [metadata[index] for index in split_indices[split_name]]
        for split_name in SPLIT_NAMES
    }
    return {
        "format_version": 1,
        "seed": int(seed),
        "ratios": ratios,
        "sample_fingerprint": _fingerprint(metadata),
        "num_samples": len(samples),
        "split_indices": split_indices,
        "split_samples": split_samples,
    }


def _validate_cached_manifest(cached: dict, expected: dict) -> None:
    for key in ("format_version", "seed", "ratios", "sample_fingerprint", "num_samples"):
        if cached.get(key) != expected.get(key):
            raise ValueError(
                "Saved split manifest does not match the current dataset or split settings. "
                "Review the changes and rerun with --regenerate to intentionally replace it."
            )

    split_indices = cached.get("split_indices", {})
    all_indices = [index for name in SPLIT_NAMES for index in split_indices.get(name, [])]
    if sorted(all_indices) != list(range(expected["num_samples"])):
        raise ValueError(
            "Saved split manifest is incomplete or overlapping. "
            "Rerun with --regenerate to rebuild it."
        )


def load_or_create_splits(
    samples: list[tuple],
    output_path: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    regenerate: bool = False,
) -> dict:
    """Load an unchanged saved split manifest or create it once and persist it."""
    output_path = Path(output_path)
    expected = create_split_manifest(
        samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    if output_path.is_file() and not regenerate:
        with output_path.open(encoding="utf-8") as manifest_file:
            cached = json.load(manifest_file)
        _validate_cached_manifest(cached, expected)
        return cached

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            suffix=".tmp",
            delete=False,
        ) as manifest_file:
            temporary_path = Path(manifest_file.name)
            json.dump(expected, manifest_file, indent=2)
            manifest_file.write("\n")
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return expected


def _load_config(config_path: Path) -> dict:
    import yaml

    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict) or not isinstance(config.get("dataset"), dict):
        raise ValueError(f"Expected dataset settings in config file: {config_path}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic, class-stratified Food-101 train/val/test splits."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="replace an existing split manifest after verifying dataset changes",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    dataset_config = config["dataset"]
    split_config = config.get("data_splits", {})
    ratios = {
        "train": (
            args.train_ratio
            if args.train_ratio is not None
            else split_config.get("train_ratio", 0.70)
        ),
        "val": (
            args.val_ratio
            if args.val_ratio is not None
            else split_config.get("val_ratio", 0.15)
        ),
        "test": (
            args.test_ratio
            if args.test_ratio is not None
            else split_config.get("test_ratio", 0.15)
        ),
    }
    seed = args.seed if args.seed is not None else split_config.get("seed", 42)
    output_path = args.output or split_config.get(
        "manifest", "data/splits.json"
    )
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = ML_DIR / output_path

    try:
        from .food_dataset import Food101NutritionDataset
    except ImportError:
        from food_dataset import Food101NutritionDataset

    identity_transform = lambda image: image
    train_dataset = Food101NutritionDataset(split="train", transform=identity_transform)
    test_dataset = Food101NutritionDataset(split="test", transform=identity_transform)
    samples = train_dataset.samples + test_dataset.samples

    manifest = load_or_create_splits(
        samples,
        output_path,
        train_ratio=ratios["train"],
        val_ratio=ratios["val"],
        test_ratio=ratios["test"],
        seed=seed,
        regenerate=args.regenerate,
    )
    print(f"Split manifest: {output_path}")
    for split_name in SPLIT_NAMES:
        print(f"{split_name}: {len(manifest['split_indices'][split_name]):,} samples")


if __name__ == "__main__":
    main()