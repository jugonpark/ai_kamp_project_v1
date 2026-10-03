"""Synthetic Seq5 windows never cross a recording Segment."""
import unittest
import shutil
import uuid

import numpy as np
import pandas as pd

from preprocessing_config import default_preprocessing_config, validate_preprocessing_config
from stage2.windowing import generate_windows, window_count_summary
from stage2.preprocessing import run_stage2_preprocessing
from train_lstm_ae import FEATURES, LABEL_COLUMN, TIMESTAMP_COLUMN
from train_lstm_ae import OUTPUT_DIR


def frame(lengths):
    rows = []
    for segment_id, length in enumerate(lengths):
        start = pd.Timestamp("2026-01-01") + pd.Timedelta(seconds=segment_id * 10)
        for index in range(length):
            rows.append({**{feature: float(index) for feature in FEATURES},
                         LABEL_COLUMN: 1, TIMESTAMP_COLUMN: start + pd.Timedelta(milliseconds=100 * index),
                         "segment_id": segment_id, "source_row": len(rows)})
    return pd.DataFrame(rows)


class Stage2Seq5Tests(unittest.TestCase):
    def test_lengths_4_5_6_and_stride(self):
        config = validate_preprocessing_config({**default_preprocessing_config(), "sequence_length": 5})
        self.assertEqual(config["sequence_length"], 5)
        for length, expected in ((4, 0), (5, 1), (6, 2)):
            with self.subTest(length=length):
                sample = frame([length])
                batch = generate_windows(sample, 5, stride=1)
                self.assertEqual(len(batch.X), expected)
                self.assertEqual(batch.X.shape[1:], (5, 3))
                self.assertEqual(window_count_summary(sample, 5)["windows"], expected)
        self.assertEqual(len(generate_windows(frame([6]), 5, stride=2).X), 1)

    def test_no_cross_segment_window_or_padding(self):
        sample = frame([4, 4, 5])
        batch = generate_windows(sample, 5)
        self.assertEqual(len(batch.X), 1)
        self.assertEqual(batch.segment_ids.tolist(), [2])
        self.assertEqual(batch.cross_segment_windows, 0)
        np.testing.assert_array_equal(batch.source_original_indices[0], np.arange(8, 13))
        summary = window_count_summary(sample, 5)
        self.assertEqual(summary["eligible_segments"], 1)

    def test_synthetic_full_preprocessing_keeps_segment_splits_and_scaler_policy(self):
        root = OUTPUT_DIR / ".test_artifacts" / f"seq5_preprocessing_{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        try:
            paths = []
            for stream, count, label in (("normal", 480, 0), ("anomaly", 11, 1)):
                rows = []
                for segment in range(count):
                    start = pd.Timestamp("2026-01-01") + pd.Timedelta(seconds=segment)
                    for index in range(5):
                        rows.append({**{feature: float(segment + index) for feature in FEATURES},
                                     LABEL_COLUMN: label,
                                     TIMESTAMP_COLUMN: start + pd.Timedelta(milliseconds=100 * index)})
                path = root / f"{stream}.csv"
                pd.DataFrame(rows).to_csv(path, index=False)
                paths.append(path)
            result = run_stage2_preprocessing(*paths, {**default_preprocessing_config(),
                                                        "sequence_length": 5})
            self.assertEqual(result.x_train.shape, (449, 5, 3))
            self.assertEqual(result.summary["normal_validation_windows"], 30)
            self.assertEqual(result.summary["anomaly_validation_windows"], 10)
            self.assertEqual(result.summary["normal_test_windows"], 1)
            self.assertEqual(result.summary["anomaly_test_windows"], 1)
            self.assertEqual(result.summary["cross_gap_windows"], 0)
            self.assertEqual(result.summary["scaler_fit_source"], "normal_train_rows_only")
            self.assertEqual(result.summary["full_dataset_window_counts"]["seq5"]["anomaly"]["eligible_segments"], 11)
            self.assertEqual(set(result.summary["full_dataset_window_counts"]), {"seq5", "seq10", "seq15", "seq20"})
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__": unittest.main()
