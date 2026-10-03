import unittest
from pathlib import Path

from preprocessing_config import default_preprocessing_config
from stage2.data_quality import inspect_csv
from stage2.splitting import split_segments
from stage2.windowing import window_count_summary


ROOT = Path(__file__).resolve().parents[1]


class Stage2SplittingTests(unittest.TestCase):
    def test_actual_whole_segment_split(self):
        config = default_preprocessing_config()
        normal = inspect_csv(ROOT / "press_data_normal.csv", config, 0).frame
        anomaly = inspect_csv(ROOT / "outlier_data.csv", config, 1).frame
        parts = split_segments(normal, anomaly)
        self.assertEqual([parts.normal_train.segment_id.nunique(), parts.normal_validation.segment_id.nunique(),
                          parts.normal_test.segment_id.nunique()], [449, 30, 120])
        self.assertEqual([parts.anomaly_validation.segment_id.nunique(), parts.anomaly_test.segment_id.nunique()], [10, 11])
        self.assertEqual(sum(len(getattr(parts, name)) for name in ("normal_train", "normal_validation", "normal_test")), len(normal))
        normal_sets = [set(getattr(parts, name).segment_id) for name in ("normal_train", "normal_validation", "normal_test")]
        self.assertFalse(normal_sets[0] & normal_sets[1] | normal_sets[0] & normal_sets[2] | normal_sets[1] & normal_sets[2])
        self.assertFalse(set(parts.anomaly_validation.segment_id) & set(parts.anomaly_test.segment_id))
        self.assertLess(parts.normal_train.TimeStamp.max(), parts.normal_validation.TimeStamp.min())
        self.assertLess(parts.normal_validation.TimeStamp.max(), parts.normal_test.TimeStamp.min())
        expected_windows = {10: [11053, 775, 3043, 198, 230],
                            15: [9152, 641, 2527, 163, 184],
                            20: [7411, 517, 2050, 129, 147]}
        names = ("normal_train", "normal_validation", "normal_test", "anomaly_validation", "anomaly_test")
        for length, expected in expected_windows.items():
            with self.subTest(sequence_length=length):
                self.assertEqual([window_count_summary(getattr(parts, name), length)["windows"] for name in names], expected)
