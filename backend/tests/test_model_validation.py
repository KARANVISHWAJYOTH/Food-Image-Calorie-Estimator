"""Unit tests for the checkpoint validation gate.

These tests pin down the exact error codes that decide whether a checkpoint
may be served by the API. They construct synthetic checkpoints in memory and
never depend on real model weights.
"""

import unittest

import torch

from backend.app.model_validation import (
    TARGET_NAMES,
    describe_validation_failure,
    validate_checkpoint,
)

NUM_CLASSES = 8


def _state_dict(num_classes: int = NUM_CLASSES, regression_outputs: int = 4, trained: bool = True):
    running_mean = torch.full((32,), 0.5) if trained else torch.zeros(32)
    running_var = torch.full((32,), 1.5) if trained else torch.ones(32)
    return {
        "backbone.bn1.running_mean": running_mean,
        "backbone.bn1.running_var": running_var,
        "classification_head.3.weight": torch.randn(num_classes, 16),
        "regression_head.3.weight": torch.randn(regression_outputs, 16),
    }


def _checkpoint(**overrides):
    checkpoint = {
        "state_dict": _state_dict(),
        "class_names": [f"class_{index}" for index in range(NUM_CLASSES)],
        "target_stats": {
            "mean": [420.0, 18.0, 40.0, 14.0],
            "std": [180.0, 9.0, 20.0, 7.0],
            "target_names": list(TARGET_NAMES),
        },
        "val_metrics": {"accuracy": 0.72, "calories_mae": 45.3},
    }
    checkpoint.update(overrides)
    return checkpoint


class TestValidateCheckpoint(unittest.TestCase):
    def test_accepts_trained_synthetic_checkpoint(self):
        report = validate_checkpoint(_checkpoint(), expected_num_classes=NUM_CLASSES)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["num_classes"], NUM_CLASSES)
        self.assertEqual(report["target_stats"]["mean"], [420.0, 18.0, 40.0, 14.0])
        self.assertAlmostEqual(report["val_metrics"]["accuracy"], 0.72)

    def test_accepts_model_state_dict_key_and_packed_int8_weights(self):
        packed = _state_dict(trained=True)
        packed.pop("classification_head.3.weight")
        packed.pop("regression_head.3.weight")
        packed["classification_head.3._packed_params._packed_params"] = (
            torch.randn(NUM_CLASSES, 16),
            torch.randn(NUM_CLASSES),
        )
        packed["regression_head.3._packed_params._packed_params"] = (
            torch.randn(4, 16),
            torch.randn(4),
        )
        checkpoint = _checkpoint()
        checkpoint.pop("state_dict")
        checkpoint["model_state_dict"] = packed

        report = validate_checkpoint(checkpoint)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["num_classes"], NUM_CLASSES)

    def test_missing_state_dict(self):
        checkpoint = _checkpoint()
        checkpoint.pop("state_dict")
        report = validate_checkpoint(checkpoint)
        self.assertFalse(report["valid"])
        self.assertIn("missing_state_dict", report["errors"])

    def test_missing_class_names(self):
        report = validate_checkpoint(_checkpoint(class_names=None))
        self.assertIn("missing_class_names", report["errors"])

    def test_invalid_class_names(self):
        names = ["pizza", "  ", "burger", "sushi", "tacos", "ramen", "salad", "steak"]
        report = validate_checkpoint(_checkpoint(class_names=names))
        self.assertIn("invalid_class_names", report["errors"])

    def test_classifier_output_mismatch(self):
        report = validate_checkpoint(_checkpoint(), expected_num_classes=101)
        self.assertIn(
            f"classifier_output_mismatch: checkpoint={NUM_CLASSES}, expected=101",
            report["errors"],
        )

    def test_class_names_length_mismatch(self):
        report = validate_checkpoint(_checkpoint(class_names=["pizza", "burger"]))
        self.assertIn(
            f"class_names_length_mismatch: names=2, classifier={NUM_CLASSES}",
            report["errors"],
        )

    def test_regression_output_mismatch(self):
        report = validate_checkpoint(_checkpoint(state_dict=_state_dict(regression_outputs=5)))
        self.assertIn(
            "regression_output_mismatch: checkpoint=5, expected=4",
            report["errors"],
        )

    def test_missing_target_stats(self):
        report = validate_checkpoint(_checkpoint(target_stats=None))
        self.assertIn("missing_target_stats", report["errors"])
        self.assertIsNone(report["target_stats"])

    def test_invalid_target_stats_shape_and_values(self):
        wrong_length = _checkpoint(target_stats={"mean": [1.0, 2.0], "std": [1.0, 2.0]})
        report = validate_checkpoint(wrong_length)
        self.assertIn("invalid_target_stats", report["errors"])

        not_finite = _checkpoint(
            target_stats={"mean": [1.0, float("nan"), 3.0, 4.0], "std": [1.0, 1.0, 1.0, 1.0]}
        )
        report = validate_checkpoint(not_finite)
        self.assertIn("invalid_target_stats", report["errors"])

    def test_zero_target_std(self):
        stats = {"mean": [1.0, 2.0, 3.0, 4.0], "std": [1.0, 0.0, 1.0, 1.0]}
        report = validate_checkpoint(_checkpoint(target_stats=stats))
        self.assertIn("zero_target_std", report["errors"])

    def test_target_name_order_mismatch(self):
        stats = {
            "mean": [1.0, 2.0, 3.0, 4.0],
            "std": [1.0, 1.0, 1.0, 1.0],
            "target_names": ["calories", "carbohydrates", "protein", "fat"],
        }
        report = validate_checkpoint(_checkpoint(target_stats=stats))
        self.assertIn("target_name_order_mismatch", report["errors"])

    def test_missing_validation_metrics(self):
        report = validate_checkpoint(_checkpoint(val_metrics=None))
        self.assertIn("missing_validation_metrics", report["errors"])
        self.assertEqual(report["val_metrics"], {})

    def test_partial_and_invalid_validation_metrics(self):
        report = validate_checkpoint(_checkpoint(val_metrics={"accuracy": 0.8}))
        self.assertIn("missing_validation_metrics:calories_mae", report["errors"])

        report = validate_checkpoint(_checkpoint(val_metrics={"accuracy": 1.4, "calories_mae": 30.0}))
        self.assertIn("invalid_validation_accuracy", report["errors"])

        report = validate_checkpoint(_checkpoint(val_metrics={"accuracy": True, "calories_mae": 30.0}))
        self.assertIn("missing_validation_metrics:accuracy", report["errors"])

    def test_untrained_batch_norm_statistics_rejected(self):
        report = validate_checkpoint(_checkpoint(state_dict=_state_dict(trained=False)))
        self.assertFalse(report["valid"])
        self.assertIn("untrained_batch_norm_statistics", report["errors"])

    def test_missing_batch_norm_statistics_is_only_a_warning(self):
        state = _state_dict()
        state.pop("backbone.bn1.running_mean")
        state.pop("backbone.bn1.running_var")
        report = validate_checkpoint(_checkpoint(state_dict=state))
        self.assertTrue(report["valid"], report["errors"])
        self.assertIn("no_batch_norm_statistics_found", report["warnings"])

    def test_describe_validation_failure_lists_error_codes(self):
        report = validate_checkpoint(_checkpoint(state_dict=_state_dict(trained=False)))
        message = describe_validation_failure(report)
        self.assertIn("untrained_batch_norm_statistics", message)
        self.assertIn("best_model.pth", message)


if __name__ == "__main__":
    unittest.main()
