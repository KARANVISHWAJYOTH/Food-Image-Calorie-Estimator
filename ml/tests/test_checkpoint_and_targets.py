"""Unit and integration tests for checkpoint persistence, target normalization, and dietary fiber derivation."""

from pathlib import Path
import tempfile
import pytest
import torch

from ml.models.checkpoint import load_checkpoint, save_checkpoint
from ml.models.multitask_cnn import MultiTaskCNN, NUTRITION_OUTPUTS, NUTRITION_TARGET_NAMES
from backend.app.model import MultiTaskFoodCNN, NUTRITION_TARGET_NAMES as BACKEND_TARGET_NAMES
from backend.app.predictor import FoodPredictor, derive_dietary_fiber


def test_target_constants_consistency() -> None:
    """Verify ML and backend models agree on the 4 continuous nutrition targets."""
    assert NUTRITION_OUTPUTS == 4
    assert NUTRITION_TARGET_NAMES == ("calories", "protein", "carbohydrates", "fat")
    assert BACKEND_TARGET_NAMES == ("calories", "protein", "carbohydrates", "fat")


def test_checkpoint_roundtrip_stores_all_required_elements() -> None:
    """Verify save_checkpoint and load_checkpoint store and recover:
    - state_dict / model_state_dict
    - class_names
    - target_stats
    - config
    - val_metrics
    """
    model = MultiTaskCNN(num_classes=3, pretrained=False)
    class_names = ["apple_pie", "caesar_salad", "pizza"]
    target_stats = {
        "mean": [450.0, 20.0, 50.0, 15.0],
        "std": [150.0, 10.0, 20.0, 8.0],
        "target_names": ["calories", "protein", "carbohydrates", "fat"],
    }
    config = {
        "model": {"backbone": "resnet50", "num_classes": 3},
        "training": {"learning_rate": 0.001, "batch_size": 16},
    }
    val_metrics = {
        "loss": 0.42,
        "accuracy": 0.88,
        "calories_mae": 18.5,
        "protein_mae": 2.1,
        "carbohydrates_mae": 3.4,
        "fat_mae": 1.7,
    }
    metadata = {
        "backbone": "resnet50",
        "training_config": config["training"],
        "version": "foodcnn-v1-test",
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "checkpoint.pt"
        save_checkpoint(
            model=model,
            path=ckpt_path,
            metadata=metadata,
            class_names=class_names,
            target_stats=target_stats,
            config=config,
            val_metrics=val_metrics,
        )

        assert ckpt_path.is_file()
        loaded = load_checkpoint(ckpt_path)

        # 1. State dict
        assert "state_dict" in loaded
        assert "model_state_dict" in loaded
        assert isinstance(loaded["state_dict"], dict)

        # 2. Class names
        assert loaded["class_names"] == class_names

        # 3. Target stats
        assert loaded["target_stats"] == target_stats
        assert loaded["target_stats"]["mean"] == [450.0, 20.0, 50.0, 15.0]
        assert loaded["target_stats"]["std"] == [150.0, 10.0, 20.0, 8.0]

        # 4. Config
        assert loaded["config"] == config

        # 5. Validation metrics
        assert loaded["val_metrics"] == val_metrics
        assert loaded["val_metrics"]["accuracy"] == 0.88


def test_predictor_loads_all_checkpoint_fields_and_denormalizes() -> None:
    """Verify backend FoodPredictor loads state_dict, class_names, target_stats,
    config, val_metrics and correctly de-normalizes regression targets."""
    model = MultiTaskFoodCNN(num_classes=3, pretrained=False)
    class_names = ["oatmeal_berry_bowl", "caesar_salad", "pepperoni_pizza"]
    target_stats = {
        "mean": [500.0, 25.0, 60.0, 20.0],
        "std": [100.0, 10.0, 15.0, 5.0],
        "target_names": ["calories", "protein", "carbohydrates", "fat"],
    }
    config = {
        "model": {"backbone": "resnet50", "num_classes": 3},
        "training": {"epochs": 10},
    }
    val_metrics = {"loss": 0.25, "accuracy": 0.92}
    metadata = {
        "backbone": "resnet50",
        "training_config": config["training"],
        "version": "foodcnn-test",
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_model.pt"
        save_checkpoint(
            model=model,
            path=ckpt_path,
            metadata=metadata,
            class_names=class_names,
            target_stats=target_stats,
            config=config,
            val_metrics=val_metrics,
        )

        predictor = FoodPredictor()
        predictor.load_weights(str(ckpt_path))

        assert predictor.is_torch_ready is True
        assert predictor.classes == class_names
        assert predictor.target_stats == target_stats
        assert predictor.config == config
        assert predictor.val_metrics == val_metrics

        # Test de-normalization with z-score inputs:
        # z = [0.0, 1.0, -1.0, 2.0]
        # calories: 0.0 * 100 + 500 = 500.0
        # protein:  1.0 * 10 + 25 = 35.0
        # carbs:   -1.0 * 15 + 60 = 45.0
        # fat:      2.0 * 5 + 20 = 30.0
        z_scores = torch.tensor([0.0, 1.0, -1.0, 2.0])
        denorm = predictor.denormalize_nutrition(z_scores)

        assert torch.isclose(denorm[0], torch.tensor(500.0))
        assert torch.isclose(denorm[1], torch.tensor(35.0))
        assert torch.isclose(denorm[2], torch.tensor(45.0))
        assert torch.isclose(denorm[3], torch.tensor(30.0))

        # Test negative clamping
        extreme_neg_z = torch.tensor([-10.0, -10.0, -10.0, -10.0])
        denorm_clamped = predictor.denormalize_nutrition(extreme_neg_z)
        assert (denorm_clamped >= 0.0).all()
        assert (denorm_clamped == 0.0).all()


def test_derive_dietary_fiber_categories_and_constraints() -> None:
    """Verify explicit deterministic dietary fiber derivation:
    - High-fiber category (e.g. oatmeal, quinoa, salad, berries): ~22% of carbs
    - Moderate category (e.g. pizza, dosa, curry): ~10% of carbs
    - Low-fiber category (e.g. chicken, meat): ~5% of carbs
    - Biological constraints: fiber <= carbs and fiber >= 0.0
    """
    # High fiber: oatmeal
    oat_fiber = derive_dietary_fiber("oatmeal_berry_bowl", carbohydrates=50.0)
    assert oat_fiber == round(50.0 * 0.22, 1)  # 11.0 g

    # High fiber: salad
    salad_fiber = derive_dietary_fiber("caesar_salad", carbohydrates=20.0)
    assert salad_fiber == round(20.0 * 0.22, 1)  # 4.4 g

    # Moderate fiber: pizza
    pizza_fiber = derive_dietary_fiber("pepperoni_pizza", carbohydrates=60.0)
    assert pizza_fiber == round(60.0 * 0.10, 1)  # 6.0 g

    # Lower fiber: grilled chicken
    chicken_fiber = derive_dietary_fiber("grilled_chicken_breast", carbohydrates=10.0)
    assert chicken_fiber == round(10.0 * 0.05, 1)  # 0.5 g

    # Zero carbohydrates
    assert derive_dietary_fiber("steak", carbohydrates=0.0) == 0.0
    assert derive_dietary_fiber("steak", carbohydrates=-5.0) == 0.0

    # Constraint: fiber cannot exceed total carbs
    small_carbs_fiber = derive_dietary_fiber("berry_smoothie", carbohydrates=0.4)
    assert small_carbs_fiber <= 0.4
