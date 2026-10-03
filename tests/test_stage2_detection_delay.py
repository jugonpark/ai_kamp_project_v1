import csv
import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from stage2.segment_detection import (build_anomaly_segment_inventory,
    stage2_segment_detection_metrics, save_segment_detail_csv)
from score_postprocessing import apply_temporal_processing
from train_lstm_ae import OUTPUT_DIR
from preprocessing_config import default_preprocessing_config
from evaluation_controller import EvaluationController


BASE = np.datetime64("2026-01-01T00:00:00", "ns")


def at(seconds):
    return BASE + np.timedelta64(round(seconds * 1000), "ms")


def batch(ids, seconds):
    return {"segment_ids": np.asarray(ids),
            "stream_type": np.full(len(ids), "ANOMALY"),
            "window_end_timestamp": np.asarray([at(value) for value in seconds])}


class Stage2SegmentDetectionTests(unittest.TestCase):
    def setUp(self):
        self.inventory = [{"segment_id": 1, "segment_length_samples": 23,
                           "segment_start_timestamp": at(0), "segment_end_timestamp": at(2.2)}]

    def test_first_evaluable_alarm_and_later_alarm(self):
        windows = batch([1, 1], [1.9, 2.2])
        first, details = stage2_segment_detection_metrics(self.inventory, windows, [1, 0], 20)
        self.assertEqual(details[0]["status"], "DETECTED")
        self.assertEqual(details[0]["first_evaluable_timestamp"], at(1.9))
        self.assertEqual(details[0]["first_alarm_timestamp"], at(1.9))
        self.assertAlmostEqual(details[0]["delay_from_segment_start_seconds"], 1.9)
        self.assertEqual(details[0]["delay_after_first_evaluable_seconds"], 0)
        later, details = stage2_segment_detection_metrics(self.inventory, windows, [0, 1], 20)
        self.assertAlmostEqual(details[0]["delay_from_segment_start_seconds"], 2.2)
        self.assertAlmostEqual(details[0]["delay_after_first_evaluable_seconds"], .3)
        self.assertEqual(later["median_post_evaluable_delay_seconds"], .3)
        self.assertEqual(first["segment_detection_rate"], 1)

    def test_missed_non_evaluable_and_zero_denominator(self):
        missed, rows = stage2_segment_detection_metrics(self.inventory, batch([1], [1.9]), [0], 20)
        self.assertEqual(rows[0]["status"], "MISSED")
        self.assertTrue(rows[0]["missed"])
        self.assertIsNone(rows[0]["delay_from_segment_start_seconds"])
        self.assertEqual(missed["missed_segments"], 1)
        self.assertIsNone(missed["median_segment_delay_seconds"])
        short = [{"segment_id": 2, "segment_length_samples": 10,
                  "segment_start_timestamp": at(3), "segment_end_timestamp": at(3.9)}]
        empty = batch([], [])
        metrics, rows = stage2_segment_detection_metrics(short, empty, [], 20)
        self.assertEqual(rows[0]["status"], "NOT_EVALUABLE")
        self.assertFalse(rows[0]["missed"])
        self.assertEqual(metrics["non_evaluable_anomaly_segments"], 1)
        self.assertIsNone(metrics["segment_detection_rate"])

    def test_inventory_uses_raw_segment_start_and_includes_short_test_segment(self):
        frame = pd.DataFrame({"segment_id": [1] * 23 + [2] * 10,
                              "split": ["test"] * 33,
                              "TimeStamp": [str(at(i / 10)) for i in range(23)] +
                                           [str(at(3 + i / 10)) for i in range(10)]})
        inventory = build_anomaly_segment_inventory(frame, {"segment_ids": {"anomaly_test": [1, 2]}})
        self.assertEqual(inventory[0]["segment_start_timestamp"], at(0))
        self.assertEqual(inventory[1]["segment_length_samples"], 10)
        metrics, rows = stage2_segment_detection_metrics(inventory, batch([1, 1], [1.9, 2.2]), [0, 1], 20)
        self.assertEqual([row["status"] for row in rows], ["DETECTED", "NOT_EVALUABLE"])
        self.assertEqual(metrics["segment_detection_rate"], 1)
        self.assertEqual(metrics["anomaly_segments_total"], 2)

    def test_independent_segments_aggregation_and_ewma_alarm_shift(self):
        inventory = self.inventory + [
            {"segment_id": 2, "segment_length_samples": 23, "segment_start_timestamp": at(3), "segment_end_timestamp": at(5.2)},
            {"segment_id": 3, "segment_length_samples": 23, "segment_start_timestamp": at(6), "segment_end_timestamp": at(8.2)},
            {"segment_id": 4, "segment_length_samples": 10, "segment_start_timestamp": at(9), "segment_end_timestamp": at(9.9)}]
        windows = batch([1, 1, 2, 2, 3, 3], [1.9, 2.2, 4.9, 5.2, 7.9, 8.2])
        metrics, rows = stage2_segment_detection_metrics(inventory, windows, [0, 1, 1, 0, 0, 0], 20)
        self.assertEqual([row["status"] for row in rows], ["DETECTED", "DETECTED", "MISSED", "NOT_EVALUABLE"])
        self.assertEqual(metrics["segment_detection_rate"], 2 / 3)
        self.assertEqual(metrics["segment_coverage"], 3 / 4)
        self.assertAlmostEqual(metrics["mean_segment_delay_seconds"], 2.05)
        self.assertAlmostEqual(metrics["median_segment_delay_seconds"], 2.05)
        self.assertAlmostEqual(metrics["max_segment_delay_seconds"], 2.2)
        self.assertAlmostEqual(metrics["mean_post_evaluable_delay_seconds"], .15)
        self.assertAlmostEqual(metrics["median_post_evaluable_delay_seconds"], .15)
        self.assertAlmostEqual(metrics["max_post_evaluable_delay_seconds"], .3)
        raw = np.array([2., 4.])
        smoothed = apply_temporal_processing(raw, "EWMA", .5, segment_ids=[1, 1])
        self.assertEqual(smoothed.tolist(), [2., 3.])
        _, raw_rows = stage2_segment_detection_metrics(self.inventory, batch([1, 1], [1.9, 2.2]), raw > 1.5, 20)
        _, ewma_rows = stage2_segment_detection_metrics(self.inventory, batch([1, 1], [1.9, 2.2]), smoothed > 2.5, 20)
        self.assertLess(raw_rows[0]["delay_after_first_evaluable_seconds"],
                        ewma_rows[0]["delay_after_first_evaluable_seconds"])

    def test_detail_csv_uses_unique_filename(self):
        folder = OUTPUT_DIR / ".test_artifacts" / f"segment_detection_{uuid.uuid4().hex}"
        folder.mkdir(parents=True)
        try:
            run = SimpleNamespace(model_path=folder / "model.keras")
            _, rows = stage2_segment_detection_metrics(self.inventory, batch([1], [1.9]), [1], 20)
            first = save_segment_detail_csv(run, rows)
            second = save_segment_detail_csv(run, rows)
            self.assertNotEqual(first, second)
            with first.open(newline="", encoding="utf-8-sig") as handle:
                self.assertEqual(next(csv.DictReader(handle))["status"], "DETECTED")
        finally:
            shutil.rmtree(folder)

    def test_controller_uses_final_none_or_ewma_predictions(self):
        controller = EvaluationController()
        controller.model_run = SimpleNamespace(metadata={"data_mode": "PROCESSED_DATASET",
            "dataset_id": "synthetic", "preprocessing": default_preprocessing_config()})
        windows = batch([1, 1, 1], [1.9, 2.0, 2.2])
        controller._bundle = {"summary": {"data_mode": "PROCESSED_DATASET"},
            "valid_error": np.zeros((2, 20, 3)), "test_error": np.zeros((3, 20, 3)),
            "y_valid": np.array([0, 1]), "y_test": np.ones(3, dtype=int),
            "valid_metadata": {"segment_ids": np.array([0, 1])},
            "test_metadata": {"segment_ids": np.array([1, 1, 1])},
            "stage2_anomaly_segments": self.inventory,
            "stage2_test_batch": windows, "stage2_sequence_length": 20}
        threshold = SimpleNamespace(threshold=2.5, effective_method="NORMAL_P99",
                                    fallback_used=False, fallback_reason="")
        with patch("evaluation_controller.fit_score_calibration", return_value=None), \
             patch("evaluation_controller.calculate_threshold", return_value=threshold), \
             patch("evaluation_controller.compute_scores", side_effect=[np.zeros(2), np.array([0., 4., 4.])]):
            raw = controller.evaluate("LAST_STEP_MSE", "NORMAL_P99", "NONE")
        raw_details = controller._last_segment_details
        with patch("evaluation_controller.fit_score_calibration", return_value=None), \
             patch("evaluation_controller.calculate_threshold", return_value=threshold), \
             patch("evaluation_controller.compute_scores", side_effect=[np.zeros(2), np.array([0., 4., 4.])]):
            ewma = controller.evaluate("LAST_STEP_MSE", "NORMAL_P99", "EWMA", .5)
        self.assertEqual(raw["detected_segments"], 1)
        self.assertEqual(ewma["detected_segments"], 1)
        self.assertAlmostEqual(raw_details[0]["delay_from_segment_start_seconds"], 2.0)
        self.assertAlmostEqual(controller._last_segment_details[0]["delay_from_segment_start_seconds"], 2.2)


if __name__ == "__main__": unittest.main()
