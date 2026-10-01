import unittest

import numpy as np


class ScoreRegistryTests(unittest.TestCase):
    def test_registry_contains_required_eight_methods(self):
        from anomaly_scoring import SCORE_METHODS
        self.assertEqual(set(SCORE_METHODS), {
            "LAST_STEP_MSE", "FULL_WINDOW_MSE", "ROBUST_CHANNEL_FULL",
            "ROBUST_TOPK_10", "ROBUST_TOPK_20", "ROBUST_TOPK_30",
            "TIME_P90", "MAHALANOBIS_ERROR",
        })

    def test_last_step_mse_matches_kamp_formula(self):
        from anomaly_scoring import fit_score_calibration, compute_scores
        error = np.arange(2 * 20 * 3, dtype=float).reshape(2, 20, 3)
        calibration = fit_score_calibration("LAST_STEP_MSE", error[:1])
        actual = compute_scores("LAST_STEP_MSE", error, calibration)
        np.testing.assert_allclose(actual, error[:, -1, :].mean(axis=1))

    def test_robust_zero_mad_remains_finite(self):
        from anomaly_scoring import fit_score_calibration, compute_scores
        normal_error = np.zeros((3, 20, 3), dtype=float)
        calibration = fit_score_calibration("ROBUST_CHANNEL_FULL", normal_error)
        actual = compute_scores("ROBUST_CHANNEL_FULL", np.ones((2, 20, 3)), calibration)
        self.assertTrue(np.isfinite(actual).all())

    def test_mahalanobis_singular_covariance_remains_finite(self):
        from anomaly_scoring import fit_score_calibration, compute_scores
        normal_error = np.ones((4, 20, 3), dtype=float)
        calibration = fit_score_calibration("MAHALANOBIS_ERROR", normal_error)
        actual = compute_scores("MAHALANOBIS_ERROR", normal_error, calibration)
        self.assertTrue(np.isfinite(actual).all())

    def test_scores_accept_forecast_time_dimension(self):
        from anomaly_scoring import fit_score_calibration, compute_scores
        error = np.ones((4, 5, 3), dtype=float)
        for method in ("LAST_STEP_MSE", "FULL_WINDOW_MSE", "ROBUST_TOPK_20", "TIME_P90", "MAHALANOBIS_ERROR"):
            calibration = fit_score_calibration(method, error[:2])
            self.assertEqual(compute_scores(method, error, calibration).shape, (4,))


class ThresholdRegistryTests(unittest.TestCase):
    def test_registry_contains_required_six_methods(self):
        from threshold_methods import THRESHOLD_METHODS
        self.assertEqual(set(THRESHOLD_METHODS), {
            "PR_INTERSECTION", "MAX_F1", "NORMAL_P99", "NORMAL_P995",
            "POT_1PCT", "POT_0P5PCT",
        })

    def test_pr_intersection_is_callable(self):
        from threshold_methods import calculate_threshold
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        result = calculate_threshold("PR_INTERSECTION", scores, labels)
        self.assertTrue(np.isfinite(result.threshold))
        self.assertFalse(result.fallback_used)

    def test_pot_falls_back_with_too_few_excesses(self):
        from threshold_methods import calculate_threshold
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        labels = np.zeros(4, dtype=int)
        result = calculate_threshold("POT_1PCT", scores, labels)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.effective_method, "NORMAL_P99")


if __name__ == "__main__":
    unittest.main()
