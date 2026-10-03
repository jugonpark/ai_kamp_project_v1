import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing_config import default_preprocessing_config
from stage2.preprocessing import fit_and_scale, run_stage2_preprocessing, transform_signal
from stage2.splitting import SegmentSplits


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]


class Stage2PreprocessingTests(unittest.TestCase):
    def test_actual_pipeline_summary_and_window_metadata(self):
        result = run_stage2_preprocessing(ROOT / "press_data_normal.csv",
            ROOT / "outlier_data.csv", default_preprocessing_config())
        summary = result.summary
        self.assertEqual([summary[f"{key}_windows"] for key in
            ("normal_train", "normal_validation", "normal_test", "anomaly_validation", "anomaly_test")],
            [7411, 517, 2050, 129, 147])
        self.assertEqual(result.x_train.shape, (7411, 20, 3))
        self.assertEqual(summary["cross_gap_windows"], 0)
        self.assertEqual(summary["scaler_fit_source"], "normal_train_rows_only")
        self.assertTrue(np.isfinite(result.x_train).all())
        self.assertEqual(result.train_windows.source_original_indices.shape, (7411, 20))
        self.assertEqual(result.normal_cleaned.loc[0, FEATURES].tolist(),
                         pd.read_csv(ROOT / "press_data_normal.csv").loc[0, FEATURES].tolist())

    def test_signal_transforms_do_not_change_input(self):
        raw = pd.DataFrame({"AI0_Vibration": [-2.], "AI1_Vibration": [-3.], "AI2_Current": [-4.]})
        expected = {"ABS_ALL": [2., 3., 4.], "RAW_SIGNED": [-2., -3., -4.],
                    "ABS_VIBRATION_RAW_CURRENT": [2., 3., -4.]}
        for method, values in expected.items():
            with self.subTest(method=method):
                self.assertEqual(transform_signal(raw, method).loc[0, FEATURES].tolist(), values)
        self.assertEqual(raw.loc[0, FEATURES].tolist(), [-2., -3., -4.])

    def test_scaler_fits_only_transformed_normal_train(self):
        train = pd.DataFrame({"AI0_Vibration": [-2., 2.], "AI1_Vibration": [-4., 4.],
                              "AI2_Current": [-10., 10.]})
        extreme = pd.DataFrame({"AI0_Vibration": [100.], "AI1_Vibration": [100.],
                                "AI2_Current": [100.]})
        parts = SegmentSplits(train, extreme, extreme, extreme, extreme)
        minmax, scaler = fit_and_scale(parts, "RAW_SIGNED", "MINMAX")
        np.testing.assert_array_equal(scaler.data_min_, [-2., -4., -10.])
        np.testing.assert_array_equal(scaler.data_max_, [2., 4., 10.])
        self.assertGreater(minmax.normal_validation.AI0_Vibration.iloc[0], 1)
        standardized, standard_scaler = fit_and_scale(parts, "RAW_SIGNED", "STANDARD")
        np.testing.assert_allclose(standard_scaler.mean_, [0., 0., 0.])
        np.testing.assert_allclose(standardized.normal_train[FEATURES].mean(), [0., 0., 0.])
        self.assertEqual(train.AI0_Vibration.iloc[0], -2.)

    def test_config_reaches_backend_and_horizon_mode_is_rejected(self):
        config = {**default_preprocessing_config(), "sequence_length": 15,
                  "signal_transform": "RAW_SIGNED", "scaler": "STANDARD", "gap_threshold_ms": 200}
        result = run_stage2_preprocessing(ROOT / "press_data_normal.csv",
            ROOT / "outlier_data.csv", config)
        self.assertEqual(result.config, config)
        self.assertEqual(result.x_train.shape[1:], (15, 3))
        self.assertEqual(result.summary["signal_transform"], "RAW_SIGNED")
        self.assertEqual(result.summary["scaler"], "STANDARD")
        self.assertEqual(result.summary["gap_threshold_ms"], 200)
        with self.assertRaisesRegex(ValueError, "horizon disabled"):
            run_stage2_preprocessing(ROOT / "press_data_normal.csv",
                ROOT / "outlier_data.csv", {**config, "use_horizon": True})
