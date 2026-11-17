"""Inference engine for NutriVision AI.

Loads trained PyTorch MultiTaskFoodCNN checkpoints, performs torchvision
preprocessing, runs multi-task inference, de-normalizes regression targets
using train-set z-score stats, and deterministically derives dietary fiber.
"""

from __future__ import annotations

import io
import logging
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional
import uuid

from PIL import Image

from .model_validation import describe_validation_failure, validate_checkpoint
from .schemas import PredictionResponse

logger = logging.getLogger("nutrivision.predictor")

# Default classes for zero-config fallback / testing
DEFAULT_CLASS_MAP = {
    0: "grilled_chicken_rice_bowl",
    1: "chicken_biryani",
    2: "masala_dosa",
    3: "paneer_butter_masala",
    4: "salmon_quinoa_bowl",
    5: "avocado_toast",
    6: "caesar_salad",
    7: "pepperoni_pizza",
    8: "oatmeal_berry_bowl",
    9: "greek_salad",
}


# Official Food-101 benchmark dataset classes
FOOD101_CLASSES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters",
    "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare",
    "waffles",
]


def derive_dietary_fiber(food_name: str, carbohydrates: float) -> float:
    """Deterministically derive dietary fiber (g) from predicted carbohydrates and food category.

    Rationale:
    The Nutrition5k ground-truth dataset records calories, protein, carbohydrates, and fat,
    but does NOT measure dietary fiber. To maintain ground-truth integrity without training
    a neural network head on fabricated targets, the model predicts the 4 true macro targets,
    and fiber is estimated with category-based fiber-to-carbohydrate ratios.
    """
    if carbohydrates <= 0.0:
        return 0.0

    name = food_name.lower().replace("-", "_").replace(" ", "_")

    # High fiber categories: oats, berries, salads, legumes, beans, avocado (~18-25% of carbs)
    if any(k in name for k in ("oat", "berry", "salad", "quinoa", "avocado", "bean", "lentil")):
        ratio = 0.22
    # Moderate fiber categories: dosa, rice bowl, pizza, vegetables (~8-14% of carbs)
    elif any(k in name for k in ("bowl", "dosa", "pizza", "masala", "paneer", "soup", "curry")):
        ratio = 0.10
    # Lower fiber categories: refined carbohydrates, meats (~3-6% of carbs)
    else:
        ratio = 0.05

    fiber = round(carbohydrates * ratio, 1)
    # Biological constraint: fiber cannot exceed total carbohydrates
    return float(min(round(carbohydrates, 1), max(0.0, fiber)))


def derive_category(food_name: str) -> str:
    """Map a Food-101 class name to a coarse meal category."""
    name = food_name.lower().replace("-", "_").replace(" ", "_")
    if any(key in name for key in ("cake", "pie", "tiramisu", "pudding", "waffle", "pancake", "donut", "brownie", "cheesecake", "cannoli", "chocolate")):
        return "Dessert"
    if any(key in name for key in ("salad", "sashimi", "seaweed")):
        return "Salad"
    if any(key in name for key in ("soup", "ramen", "pho", "chowder")):
        return "Soup"
    if any(key in name for key in ("pizza", "burger", "sandwich", "taco", "burrito", "hot_dog", "fries")):
        return "Fast Food"
    return "Main Course"


def derive_health_rating(calories: float, protein: float, carbohydrates: float, fat: float) -> str:
    """Derive a coarse nutrition label from the predicted macros."""
    protein_kcal = max(0.0, protein) * 4.0
    carb_kcal = max(0.0, carbohydrates) * 4.0
    fat_kcal = max(0.0, fat) * 9.0
    macro_kcal = protein_kcal + carb_kcal + fat_kcal
    if macro_kcal <= 0.0:
        return "Unknown"
    if protein >= 25.0 and protein_kcal / macro_kcal >= 0.25:
        return "High Protein"
    if 0.0 < calories <= 300.0:
        return "Low Calorie"
    if fat_kcal / macro_kcal >= 0.45:
        return "High Fat"
    if carb_kcal / macro_kcal >= 0.60:
        return "High Carb"
    return "Balanced"


def build_health_tip(food_name: str, calories: float, protein: float, carbohydrates: float, fat: float, is_low_confidence: bool) -> str:
    readable = food_name.replace("_", " ")
    tip = (
        f"{readable} is estimated at {round(calories)} kcal for one standard serving "
        f"({protein:.1f}g protein, {carbohydrates:.1f}g carbs, {fat:.1f}g fat)."
    )
    if is_low_confidence:
        tip += " The image was ambiguous; retake it with better lighting and a clear view of the whole dish."
    return tip


NUTRITION_MAX = {
    "calories": 5000.0,
    "protein": 500.0,
    "carbohydrates": 1000.0,
    "fat": 500.0,
}


def validate_nutrition_output(values: Dict[str, float]) -> None:
    """Reject NaN/infinite or biologically implausible regression outputs."""
    for name, value in values.items():
        if not math.isfinite(value):
            raise RuntimeError(f"Model produced a non-finite {name} value.")
        upper = NUTRITION_MAX[name]
        if value < 0.0 or value > upper:
            raise RuntimeError(
                f"Model produced an implausible {name} value ({value:.2f}); expected 0-{upper:.0f}."
            )
    macro_kcal = values["protein"] * 4.0 + values["carbohydrates"] * 4.0 + values["fat"] * 9.0
    if values["calories"] > 0.0 and macro_kcal > values["calories"] * 1.8 + 50.0:
        raise RuntimeError(
            "Model output is internally inconsistent: macro energy exceeds the calorie estimate."
        )


class FoodPredictor:
    def __init__(self, weights_path: Optional[str] = None, variant: Optional[str] = None) -> None:
        self.device = "cpu"
        self.model_variant = (variant or os.getenv("MODEL_VARIANT", "fp32")).lower().strip()
        self.classes: list[str] = list(DEFAULT_CLASS_MAP.values())
        self.class_map: dict[int, str] = dict(DEFAULT_CLASS_MAP)
        self.target_stats: Optional[Dict[str, Any]] = None
        self.config: Dict[str, Any] = {}
        self.val_metrics: Dict[str, Any] = {}
        self.is_torch_ready = False
        self.is_model_valid = False
        self.checkpoint_path: Optional[str] = None
        self.model_validation: Dict[str, Any] = {
            "valid": False,
            "errors": ["not_initialized"],
            "warnings": [],
        }
        self._init_model(weights_path)

    def load_weights(self, weights_path: str) -> None:
        """Load or reload model weights, class names, target stats, config, and val metrics from checkpoint."""
        self._init_model(weights_path)

    def denormalize_nutrition(self, raw_regression: Any) -> Any:
        """De-normalize continuous regression predictions using train-set z-score stats.

        Formula:
            y_physical = max(0.0, y_normalized * target_std + target_mean)
        """
        import torch

        if not isinstance(raw_regression, torch.Tensor):
            raw_reg = torch.tensor(raw_regression, device=self.device, dtype=torch.float32)
        else:
            raw_reg = raw_regression.to(self.device).float()

        if self.target_stats and "mean" in self.target_stats and "std" in self.target_stats:
            t_mean = torch.tensor(self.target_stats["mean"], device=self.device, dtype=torch.float32)
            t_std = torch.tensor(self.target_stats["std"], device=self.device, dtype=torch.float32)
            denorm = raw_reg * t_std + t_mean
        else:
            denorm = raw_reg

        return torch.clamp(denorm, min=0.0)

    def _resolve_candidate_path(self, explicit_path: Optional[str]) -> Optional[str]:
        if explicit_path and os.path.exists(explicit_path):
            return explicit_path

        env_path = os.getenv("MODEL_WEIGHTS_PATH")
        if env_path and os.path.exists(env_path):
            return env_path

        current_dir = Path(__file__).resolve().parent
        repo_root = current_dir.parents[1]

        primary_candidates = []
        fallback_candidates = []

        if self.model_variant == "int8":
            primary_candidates = [
                current_dir / "best_model_int8.pth",
                repo_root / "ml" / "optimization" / "best_model_int8.pth",
                repo_root / "ml" / "training" / "checkpoints" / "best_model_int8.pth",
                Path("best_model_int8.pth"),
            ]
            fallback_candidates = [
                current_dir / "best_model.pth",
                repo_root / "ml" / "optimization" / "best_model_pruned_20.pth",
                repo_root / "ml" / "training" / "checkpoints" / "best_model.pth",
                repo_root / "best_model.pth",
                Path("best_model.pth"),
            ]
        else:
            primary_candidates = [
                current_dir / "best_model.pth",
                repo_root / "ml" / "optimization" / "best_model_pruned_20.pth",
                repo_root / "ml" / "training" / "checkpoints" / "best_model.pth",
                repo_root / "best_model.pth",
                Path("best_model.pth"),
            ]
            fallback_candidates = [
                current_dir / "best_model_int8.pth",
                repo_root / "ml" / "optimization" / "best_model_int8.pth",
                repo_root / "ml" / "training" / "checkpoints" / "best_model_int8.pth",
                Path("best_model_int8.pth"),
            ]

        for cand in primary_candidates:
            if cand.is_file():
                return str(cand)

        for cand in fallback_candidates:
            if cand.is_file():
                if "int8" in str(cand).lower():
                    self.model_variant = "int8"
                else:
                    self.model_variant = "fp32"
                return str(cand)

        return None

    def _init_model(self, weights_path: Optional[str]) -> None:
        try:
            import torch
            from torch import nn
            from torchvision import transforms
            from .model import MultiTaskFoodCNN, TORCH_AVAILABLE

            resolved_path = self._resolve_candidate_path(weights_path)

            if not TORCH_AVAILABLE or not resolved_path:
                self.model_validation = {
                    "valid": False,
                    "errors": ["checkpoint_not_found"],
                    "warnings": [],
                }
                logger.warning(f"Trained model weights not found for variant '{self.model_variant}'; prediction is disabled.")
                return

            is_int8 = self.model_variant == "int8" or "int8" in resolved_path.lower()
            if is_int8:
                self.model_variant = "int8"
                self.device = torch.device("cpu")
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            logger.info(f"Loading {self.model_variant.upper()} weights from: {resolved_path}")
            checkpoint = torch.load(resolved_path, map_location=self.device, weights_only=False)
            metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint.get("metadata"), dict) else {}
            self.checkpoint_path = resolved_path
            self.model_validation = validate_checkpoint(checkpoint)
            self.is_model_valid = bool(self.model_validation["valid"])
            if self.is_model_valid:
                logger.info(
                    "Checkpoint validation passed: %d classes, accuracy=%s, calories_mae=%s",
                    self.model_validation.get("num_classes"),
                    self.model_validation.get("val_metrics", {}).get("accuracy"),
                    self.model_validation.get("val_metrics", {}).get("calories_mae"),
                )
            else:
                logger.error(
                    "Checkpoint validation failed for %s: %s",
                    resolved_path,
                    ", ".join(self.model_validation.get("errors", [])),
                )

            # 1. State dict
            state_dict = (
                checkpoint.get("state_dict")
                or checkpoint.get("model_state_dict")
                or checkpoint
            )

            # 2. Detect number of output classes directly from state_dict weights
            detected_classes = None
            if isinstance(state_dict, dict):
                if "classification_head.3.weight" in state_dict:
                    detected_classes = state_dict["classification_head.3.weight"].shape[0]
                elif "classification_head.3._packed_params._packed_params" in state_dict:
                    detected_classes = state_dict["classification_head.3._packed_params._packed_params"][0].shape[0]

            # 3. Unpack class names if present, or assign standard Food-101 / fallback mapping
            raw_classes = checkpoint.get("class_names") or metadata.get("class_names")
            if raw_classes and isinstance(raw_classes, list):
                self.classes = list(raw_classes)
            elif detected_classes == 101:
                self.classes = list(FOOD101_CLASSES)
            elif detected_classes and detected_classes != len(self.classes):
                self.classes = [f"Dish {idx + 1}" for idx in range(detected_classes)]

            self.class_map = {idx: name for idx, name in enumerate(self.classes)}
            logger.info(f"Initialized with {len(self.classes)} classes.")

            # 4. Unpack target stats for z-score de-normalization
            self.target_stats = checkpoint.get("target_stats") or metadata.get("target_stats")
            if self.target_stats:
                logger.info(
                    f"Loaded regression target stats (mean={self.target_stats.get('mean')}, "
                    f"std={self.target_stats.get('std')})."
                )

            # 5. Unpack config
            self.config = (
                checkpoint.get("config")
                or metadata.get("config")
                or metadata.get("training_config")
                or {}
            )
            if not isinstance(self.config, dict):
                self.config = {}

            # 6. Unpack val_metrics
            self.val_metrics = (
                checkpoint.get("val_metrics")
                or metadata.get("val_metrics")
                or {}
            )
            if not isinstance(self.val_metrics, dict):
                self.val_metrics = {}
            if self.val_metrics:
                logger.info(f"Loaded checkpoint val_metrics: {self.val_metrics}")

            backbone = "resnet50"
            if isinstance(self.config, dict) and "model" in self.config:
                backbone = self.config["model"].get("backbone", "resnet50")

            # Initialize model
            num_classes = len(self.classes)
            if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], nn.Module):
                self.model = checkpoint["model"]
            else:
                self.model = MultiTaskFoodCNN(
                    num_classes=num_classes,
                    backbone=backbone,
                    pretrained=False,
                )
                if is_int8:
                    self.model = torch.quantization.quantize_dynamic(self.model, {nn.Linear}, dtype=torch.qint8)
                self.model.load_state_dict(state_dict)

            self.model.to(self.device)
            self.model.eval()
            self.is_torch_ready = True

            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            logger.info(f"FoodPredictor successfully initialized ({self.model_variant.upper()} on {self.device}).")
        except Exception as e:
            logger.warning(f"Failed to load PyTorch backend pipeline: {e}. Prediction is disabled.")
            self.is_torch_ready = False
            self.is_model_valid = False
            self.model_validation = {
                "valid": False,
                "errors": [f"checkpoint_load_error: {e}"],
                "warnings": [],
            }

    def predict_image(self, image_bytes: bytes, filename: Optional[str] = None) -> PredictionResponse:
        start_time = time.time()

        if not self.is_torch_ready:
            raise RuntimeError("Trained model weights are not configured. Set MODEL_WEIGHTS_PATH before scanning images.")
        if not self.is_model_valid:
            raise RuntimeError(describe_validation_failure(self.model_validation))

        # 1. Image validation and preprocessing
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid image file format. Ensure it is a valid JPG/PNG: {str(e)}")

        import torch

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            if isinstance(output, tuple):
                class_logits, regression_outputs = output
            else:
                class_logits = output["class_logits"]
                regression_outputs = output["nutrition"]

            probabilities = torch.softmax(class_logits, dim=1)[0]
            top_prob, top_idx = torch.topk(probabilities, 1)
            idx = int(top_idx[0].item())
            predicted_key = self.class_map.get(idx, f"dish_{idx}")
            confidence = float(top_prob[0].item())

            # De-normalize regression predictions using train-set z-score stats
            raw_reg = regression_outputs[0].float()
            denorm_nutr = self.denormalize_nutrition(raw_reg)
            calories = float(denorm_nutr[0].item())
            protein = float(denorm_nutr[1].item())
            carbohydrates = float(denorm_nutr[2].item())
            fat = float(denorm_nutr[3].item())

            # Explicitly derive dietary fiber
            fiber = derive_dietary_fiber(predicted_key, carbohydrates)

        validate_nutrition_output(
            {
                "calories": calories,
                "protein": protein,
                "carbohydrates": carbohydrates,
                "fat": fat,
            }
        )

        # Calculate actual inference runtime
        inference_ms = round((time.time() - start_time) * 1000, 1)
        is_low_conf = confidence < 0.65
        health_rating = derive_health_rating(calories, protein, carbohydrates, fat)

        return PredictionResponse(
            id=f"pred_{uuid.uuid4().hex[:10]}",
            foodClass=predicted_key.replace("_", " ").title(),
            category=derive_category(predicted_key),
            confidenceScore=round(confidence, 3),
            confidencePercentage=round(confidence * 100, 1),
            estimatedCalories=round(calories, 0),
            protein=round(protein, 1),
            carbohydrates=round(carbohydrates, 1),
            fat=round(fat, 1),
            fiber=round(fiber, 1),
            servingSize="1 standard serving (class-average estimate)",
            isLowConfidence=is_low_conf,
            healthRating=health_rating,
            dietaryTags=[health_rating] if health_rating != "Unknown" else [],
            ingredients=[],
            healthTips=build_health_tip(
                predicted_key,
                calories,
                protein,
                carbohydrates,
                fat,
                is_low_conf,
            ),
            inferenceTimeMs=inference_ms,
        )


# Global singleton predictor instance
predictor_instance = FoodPredictor()
