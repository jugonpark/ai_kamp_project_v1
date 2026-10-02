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
        self.assertEqual(result["cnn_filters"], "unknown")
        self.assertEqual(result["bottleneck_units"], "unknown")
        controller.model_run = SimpleNamespace(metadata={"cnn_filters":16, "cnn_kernel_size":5, "bottleneck_units":64})
        result = controller.evaluate("LAST_STEP_MSE", "PR_INTERSECTION")
        self.assertEqual(result["cnn_filters"], 16)
        self.assertEqual(result["cnn_kernel_size"], 5)
        self.assertEqual(result["bottleneck_units"], 64)

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

    def test_temporal_threshold_uses_processed_validation_and_resets_test(self):
        from evaluation_controller import EvaluationController
        from threshold_methods import ThresholdResult
        controller = EvaluationController()
        controller._bundle = {
            "valid_error": np.zeros((4, 1, 3)), "test_error": np.zeros((4, 1, 3)),
            "y_valid": np.array([0, 0, 1, 1]), "y_test": np.array([0, 0, 1, 1]),
            "valid_metadata": {"segment_ids": np.array([0, 0, 1, 1])},
            "test_metadata": {"segment_ids": np.array([0, 0, 1, 1])},
        }
        outputs = [np.array([1., 3., 9., 11.]), np.array([20., 24., 30., 34.])]
        def threshold(method, scores, labels):
            np.testing.assert_allclose(scores, [1., 2., 9., 10.])
            return ThresholdResult(15., method, method)
        with patch("evaluation_controller.compute_scores", side_effect=outputs), \
             patch("evaluation_controller.calculate_threshold", side_effect=threshold):
            result = controller.evaluate("LAST_STEP_MSE", "NORMAL_P99", "EWMA", .5)
        self.assertEqual(result["threshold"], 15.)
        self.assertEqual(result["fp"], 2)
        self.assertEqual(result["total_error"], result["fp"] + result["fn"])
        self.assertEqual(result["ewma_alpha"], .5)

    def test_none_scores_and_threshold_match_old_path(self):
        from evaluation_controller import EvaluationController
        controller = EvaluationController()
        error = np.broadcast_to(np.arange(8., dtype=float).reshape(8, 1, 1), (8, 1, 3))
        controller._bundle = {"valid_error": error, "test_error": error,
                              "y_valid": np.array([0]*4 + [1]*4),
                              "y_test": np.array([0]*4 + [1]*4)}
        original = controller.evaluate("LAST_STEP_MSE", "NORMAL_P99")
        explicit = controller.evaluate("LAST_STEP_MSE", "NORMAL_P99", "NONE")
        for field in ("threshold", "accuracy", "precision", "recall", "f1", "fp", "fn"):
            self.assertEqual(original[field], explicit[field])
        self.assertEqual(explicit["ewma_alpha"], "")

    def test_sweep_has_baseline_ten_alphas_and_pareto(self):
        from evaluation_controller import EvaluationController
        controller = EvaluationController()
        calls = []
        def fake_evaluate(score, threshold, temporal, alpha, aware):
            calls.append((score, threshold, temporal, alpha, aware))
            index = len(calls)
            return {"fp": index, "fn": 12-index, "total_error": 12,
                    "temporal_method": temporal, "ewma_alpha": alpha if temporal == "EWMA" else ""}
        with patch.object(controller, "evaluate", side_effect=fake_evaluate):
            results = controller.sweep_ewma_alphas("LAST_STEP_MSE", "NORMAL_P99", False)
        self.assertEqual(len(results), 11)
        self.assertEqual(calls[0][2], "NONE")
        self.assertEqual(calls[-1][3], 1.0)
        self.assertTrue(all(row["pareto"] for row in results))
        self.assertEqual(results[-1]["delta_fp"], 10)

    def test_timestamp_toggle_keeps_dataset_reset(self):
        from evaluation_controller import EvaluationController
        controller = EvaluationController()
        controller._bundle = {
            "valid_error": np.zeros((4, 1, 3)), "test_error": np.zeros((4, 1, 3)),
            "y_valid": np.array([0, 0, 1, 1]), "y_test": np.array([0, 0, 1, 1]),
            "valid_metadata": {"segment_ids": np.array([0, 1, 2, 2]),
                               "stream_ids": np.array([0, 0, 1, 1]),
                               "contains_timestamp_gap": np.array([False, True, False, False])},
            "test_metadata": {"segment_ids": np.array([0, 1, 2, 2]),
                              "stream_ids": np.array([0, 0, 1, 1]),
                              "contains_timestamp_gap": np.array([False, True, False, False])},
        }
        scores = np.array([1., 3., 10., 12.])
        captured = []
        def threshold(method, values, labels):
            captured.append(values.copy())
            from threshold_methods import ThresholdResult
            return ThresholdResult(5., method, method)
        with patch("evaluation_controller.compute_scores", return_value=scores), \
             patch("evaluation_controller.calculate_threshold", side_effect=threshold):
            controller.evaluate("LAST_STEP_MSE", "NORMAL_P99", "EWMA", .5, True)
            controller.evaluate("LAST_STEP_MSE", "NORMAL_P99", "EWMA", .5, False)
        np.testing.assert_allclose(captured[0], [1., 3., 10., 11.])
        np.testing.assert_allclose(captured[1], [1., 2., 10., 11.])


if __name__ == "__main__": unittest.main()
