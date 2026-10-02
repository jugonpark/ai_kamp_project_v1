import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from model_data import TaskBundle, create_task_bundle
from model_registry import MODEL_REGISTRY
import train_lstm_ae as core


def _frame(timestamps=None):
    rows = 130
    frame = pd.DataFrame({
        **{name: np.arange(rows, dtype=float) for name in core.FEATURES},
        core.LABEL_COLUMN: np.zeros(rows, dtype=int),
    })
    if timestamps is not None:
        frame[core.TIMESTAMP_COLUMN] = timestamps
    return frame


class TemporalMetadataTests(unittest.TestCase):
    def test_existing_positional_bundle_and_no_timestamp_are_unchanged(self):
        legacy = TaskBundle(np.empty((0,)), np.empty((0,)), np.empty((0,)),
                            "RECONSTRUCTION", "Reconstruction Error")
        self.assertIsNone(legacy.segment_ids)
        bundle = create_task_bundle(_frame(), MODEL_REGISTRY["KAMP_LSTM_AE"])
        self.assertEqual(len(bundle.inputs), 10)
        self.assertIsNone(bundle.sample_timestamps)
        self.assertIsNone(bundle.contains_timestamp_gap)

    def test_label_timestamp_and_gap_within_window(self):
        times = pd.Timestamp("2026-01-01") + pd.to_timedelta(np.arange(130) / 10, unit="s")
        times = times.where(np.arange(130) < 10, times + pd.Timedelta(seconds=5))

        def detect(timestamps):
            self.assertEqual(len(timestamps), 130)
            return np.r_[np.zeros(10, dtype=int), np.ones(120, dtype=int)], 0.15

        with patch.dict(sys.modules, {"score_postprocessing": types.SimpleNamespace(
                detect_timestamp_segments=detect)}):
            bundle = create_task_bundle(_frame(times), MODEL_REGISTRY["KAMP_LSTM_AE"])
        self.assertEqual(len(bundle.inputs), 10)
        self.assertEqual(len(bundle.labels), 10)
        np.testing.assert_array_equal(bundle.sample_timestamps,
                                      times.to_numpy()[120:130])
        np.testing.assert_array_equal(bundle.segment_ids, np.ones(10, dtype=int))
        np.testing.assert_array_equal(bundle.contains_timestamp_gap,
                                      [True] * 10)
        self.assertEqual(bundle.timestamp_gap_threshold, 0.15)

    def test_gap_outside_input_window_is_not_marked(self):
        times = pd.Timestamp("2026-01-01") + pd.to_timedelta(np.arange(130) / 10, unit="s")

        def detect(_):
            return np.r_[np.zeros(50, dtype=int), np.ones(80, dtype=int)], 0.15

        with patch.dict(sys.modules, {"score_postprocessing": types.SimpleNamespace(
                detect_timestamp_segments=detect)}):
            bundle = create_task_bundle(_frame(times), MODEL_REGISTRY["LSTM_FORECAST_5"])
        self.assertFalse(bundle.contains_timestamp_gap.any())
        self.assertEqual(len(bundle.targets), 10)


if __name__ == "__main__":
    unittest.main()
