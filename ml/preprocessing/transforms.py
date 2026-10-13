"""Image transforms for Food-101 model training and evaluation."""

from pathlib import Path

from torchvision import transforms
from torchvision.transforms import InterpolationMode


ML_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ML_DIR / "configs" / "base.yaml"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _get_image_size(image_size: int | None) -> int:
    if image_size is None:
        import yaml

        with CONFIG_PATH.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        image_size = config.get("image_size") if isinstance(config, dict) else None

    if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size <= 0:
        raise ValueError(
            f"Image size must be a positive integer; got {image_size!r}. "
            f"Set image_size in {CONFIG_PATH} or pass it explicitly."
        )
    return image_size


def get_train_transform(image_size: int | None = None) -> transforms.Compose:
    """Build the food-photo augmentation pipeline for training."""
    size = _get_image_size(image_size)
    return transforms.Compose(
        [
            transforms.RandomRotation(
                degrees=8,
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomResizedCrop(
                size,
                scale=(0.8, 1.0),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.1,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_eval_transform(image_size: int | None = None) -> transforms.Compose:
    """Build deterministic resize, center-crop, and normalization for evaluation."""
    size = _get_image_size(image_size)
    return transforms.Compose(
        [
            transforms.Resize(size + 32, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )