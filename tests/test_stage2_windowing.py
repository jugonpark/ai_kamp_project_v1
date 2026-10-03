import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing_config import default_preprocessing_config
from stage2.data_quality import inspect_csv
from stage2.windowing import generate_windows, window_count_summary


ROOT = Path(__file__).resolve().parents[1]


class Stage2WindowingTests(unittest.TestCase):
    def test_actual_full_dataset_counts_for_three_sequences(self):
        config = default_preprocessing_config()
        normal = inspect_csv(ROOT / "press_data_normal.csv", config, 0).frame
        anomaly = inspect_csv(ROOT / "outlier_data.csv", config, 1).frame
        expected = {10: ((530, 14871), (17, 428)),
                    15: ((480, 12320), (16, 347)),
                    20: ((452, 9978), (13, 276))}
        for length, (normal_expected, anomaly_expected) in expected.items():
            for frame, pair in ((normal, normal_expected), (anomaly, anomaly_expected)):
                with self.subTest(length=length, dataset=len(frame)):
                    report = window_count_summary(frame, length)
                    self.assertEqual((report["eligible_segments"], report["windows"]), pair)
        normal_batch = generate_windows(normal, 20, gap_threshold_ms=150)
        anomaly_batch = generate_windows(anomaly, 20, gap_threshold_ms=150)
        self.assertEqual(normal_batch.X.shape, (9978, 20, 3))
        self.assertEqual(anomaly_batch.X.shape, (276, 20, 3))
        self.assertEqual(normal_batch.cross_segment_windows + anomaly_batch.cross_segment_windows, 0)
        self.assertTrue((normal_batch.y == 0).all())
        self.assertTrue((anomaly_batch.y == 1).all())
        self.assertEqual(normal_batch.source_original_indices.shape, (9978, 20))

    def test_stride_and_gap_validation(self):
        frame = pd.DataFrame({"TimeStamp": pd.date_range("2026-01-01", periods=5, freq="100ms"),
            "AI0_Vibration": [1.] * 5, "AI1_Vibration": [2.] * 5, "AI2_Current": [3.] * 5,
            "Equipment_state": [0] * 5, "segment_id": [0] * 5, "source_row": range(5)})
        batch = generate_windows(frame, 3, stride=2)
        self.assertEqual(batch.X.shape, (2, 3, 3))
        self.assertEqual(batch.source_row_start.tolist(), [0, 2])
        self.assertEqual(batch.source_row_end.tolist(), [2, 4])
        self.assertEqual(len(generate_windows(frame, 10).X), 0)
        frame.loc[3, "TimeStamp"] += pd.Timedelta(seconds=1)
        with self.assertRaisesRegex(ValueError, "timestamp discontinuity"):
            generate_windows(frame, 3)
