import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


class EvaluationControllerTests(unittest.TestCase):
    def test_controller_can_be_constructed_without_loading_model(self):
        from evaluation_controller import EvaluationController
        controller = EvaluationController()
        self.assertFalse(controller.is_loaded)

    def test_cache_key_changes_when_model_file_changes(self):
        from evaluation_controller import EvaluationController
        controller = EvaluationController("m.keras", "n.csv", "a.csv")
        with patch.object(EvaluationController, "_signature", side_effect=[("m",1,1),("n",1,1),("a",1,1),
                                                                           ("m",2,1),("n",1,1),("a",1,1)]):
            first = controller.cache_key()
            second = controller.cache_key()
        self.assertNotEqual(first, second)

    def test_evaluate_uses_loaded_arrays_and_returns_metrics(self):
        from evaluation_controller import EvaluationController
        controller = EvaluationController()
        valid_error = np.zeros((4, 20, 3)); valid_error[2:] = 1
        test_error = np.zeros((4, 20, 3)); test_error[2:] = 1
        controller._bundle = {
            "valid_error": valid_error, "test_error": test_error,
            "y_valid": np.array([0, 0, 1, 1]), "y_test": np.array([0, 0, 1, 1]),
            "cache_hit": True,
        }
        result = controller.evaluate("LAST_STEP_MSE", "PR_INTERSECTION")
        for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1",
                    "specificity", "fpr", "fnr", "tn", "fp", "fn", "tp"):
            self.assertIn(key, result)
        self.assertEqual(result["score_method"], "LAST_STEP_MSE")
        self.assertEqual(result["threshold_method"], "PR_INTERSECTION")
        self.assertEqual(result["cnn_kernel_size"], "unknown")
        controller.model_run = SimpleNamespace(metadata={"cnn_kernel_size": 5})
        self.assertEqual(controller.evaluate("LAST_STEP_MSE", "PR_INTERSECTION")["cnn_kernel_size"], 5)

    def test_existing_prediction_bundle_is_reused(self):
        from evaluation_controller import EvaluationController
        controller = EvaluationController()
        key = ("same-model-and-data",)
        bundle = {"summary": {"valid_samples": 4, "test_samples": 4}}
        EvaluationController._prediction_cache[key] = bundle
        with patch.object(controller, "cache_key", return_value=key):
            summary = controller.load_model_and_predictions()
        self.assertTrue(summary["cache_hit"])
        self.assertIs(controller._bundle, bundle)

    def test_controller_tracks_forecast_model_task(self):
        from evaluation_controller import EvaluationController
        controller = EvaluationController(model_id="LSTM_FORECAST_5")
        self.assertEqual(controller.model_spec.task_type, "FORECAST")
        self.assertEqual(controller.model_spec.forecast_length, 5)


if __name__ == "__main__": unittest.main()
