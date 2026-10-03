"""Synthetic AUTO routing tests; saved models are never loaded or fitted."""
import copy
import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from preprocessing_config import default_preprocessing_config
from stage2.adaptive_sequence import choose_sequence, evaluate_auto_sequence, prepare_auto_pool
from stage2.segment_detection import inventory_from_artifact, stage2_segment_detection_metrics
from train_lstm_ae import OUTPUT_DIR


LENGTHS = {("ANOMALY", index): length for index, length in
           enumerate((4, 5, 9, 10, 14, 15, 19, 20, 30), start=1)}
LENGTHS.update({("NORMAL", 101): 8, ("NORMAL", 102): 25})


class FakeController:
    def select_run(self, run):
        self.run = run

    def _processed_artifact(self):
        return self.run.artifact

    def load_model_and_predictions(self):
        return {"cache_hit": False}

    def evaluate(self, score_method, threshold_method, temporal_method, ewma_alpha, timestamp_aware):
        batch = self.run.artifact.test
        self.last_window_predictions = np.array([
            int(stream == "ANOMALY" and segment_id != 2)
            for stream, segment_id in zip(batch["stream_type"], batch["segment_ids"])])
        _, self._last_segment_details = stage2_segment_detection_metrics(
            inventory_from_artifact(self.run.artifact), batch, self.last_window_predictions,
            self.run.artifact.config["preprocessing"]["sequence_length"])
        return {"threshold": self.run.artifact.config["preprocessing"]["sequence_length"] / 100,
                "effective_threshold_method": threshold_method, "fallback_used": False}


class AdaptiveSequenceTests(unittest.TestCase):
    def setUp(self):
        self.root = OUTPUT_DIR / ".test_artifacts" / f"auto_sequence_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)
        self.runs = [self._run(seq) for seq in (5, 10, 15, 20)]

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _run(self, sequence):
        path = self.root / f"seq{sequence}"
        path.mkdir()
        rows = {"NORMAL": [], "ANOMALY": []}
        test = {"stream_type": [], "segment_ids": [], "y": [], "window_end_timestamp": []}
        for (stream, segment_id), length in LENGTHS.items():
            start = pd.Timestamp("2026-01-01") + pd.Timedelta(seconds=segment_id * 10)
            for index in range(length):
                rows[stream].append({"segment_id": segment_id, "split": "test",
                                     "TimeStamp": start + pd.Timedelta(milliseconds=index * 100)})
            if length >= sequence:
                test["stream_type"].append(stream)
                test["segment_ids"].append(segment_id)
                test["y"].append(int(stream == "ANOMALY"))
                test["window_end_timestamp"].append(start + pd.Timedelta(milliseconds=(sequence - 1) * 100))
        for stream in rows:
            pd.DataFrame(rows[stream]).to_csv(path / f"{stream.lower()}_cleaned.csv", index=False)
        batch = {key: np.asarray(value, dtype="datetime64[ns]" if key == "window_end_timestamp" else None)
                 for key, value in test.items()}
        config = {"preprocessing": {**default_preprocessing_config(), "sequence_length": sequence},
                  "features": ["AI0_Vibration", "AI1_Vibration", "AI2_Current"],
                  "scaler_fit_source": "NORMAL_TRAIN_ONLY",
                  "source": {stream: {"sha256": "a" * 64} for stream in ("normal", "anomaly")}}
        split = {"policy": "chronological_whole_segment", "segment_ids": {
            "normal_train": [], "normal_validation": [], "normal_test": [101, 102],
            "anomaly_validation": [], "anomaly_test": list(range(1, 10))}}
        artifact = SimpleNamespace(path=path, dataset_id=f"seq{sequence}",
                                   config=config, split_manifest=split, test=batch)
        return SimpleNamespace(model_id="CNN_LSTM_AUTOENCODER", run_id=f"run{sequence}",
                               model_path=path / "model.keras",
                               metadata={"data_mode": "PROCESSED_DATASET"}, artifact=artifact)

    def test_router_boundaries_and_short_segments(self):
        expected = {30: 20, 20: 20, 19: 15, 15: 15, 14: 10, 10: 10,
                    9: 5, 5: 5, 4: None}
        for length, chosen in expected.items():
            self.assertEqual(choose_sequence(length, (5, 10, 15, 20)), chosen)
        with self.assertRaises(ValueError):
            choose_sequence(10, (5, 5))

    def test_incompatible_pool_rejected_before_evaluation(self):
        changes = {
            "signal_transform": lambda run: run.artifact.config["preprocessing"].update(signal_transform="RAW_SIGNED"),
            "scaler": lambda run: run.artifact.config["preprocessing"].update(scaler="STANDARD"),
            "gap": lambda run: run.artifact.config["preprocessing"].update(gap_threshold_ms=200),
            "stride": lambda run: run.artifact.config["preprocessing"].update(stride=2),
            "source": lambda run: run.artifact.config["source"]["normal"].update(sha256="b" * 64),
            "split": lambda run: run.artifact.split_manifest.update(policy="other"),
            "split_ids": lambda run: run.artifact.split_manifest["segment_ids"].update(anomaly_test=[1]),
            "family": lambda run: setattr(run, "model_id", "DENOISING_CNN_LSTM_AUTOENCODER"),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                runs = copy.deepcopy(self.runs[:2])
                change(runs[1])
                with self.assertRaisesRegex(ValueError, "compatibility failure"):
                    prepare_auto_pool(runs, FakeController)
        duplicate = copy.deepcopy(self.runs[:2])
        duplicate[1].artifact.config["preprocessing"]["sequence_length"] = 5
        with self.assertRaisesRegex(ValueError, "duplicate Seq5"):
            prepare_auto_pool(duplicate, FakeController)

    def test_one_prediction_tier_per_segment_and_independent_thresholds(self):
        result = evaluate_auto_sequence(self.runs, "ROBUST_TOPK_10", "PR_INTERSECTION",
                                        controller_factory=FakeController)
        self.assertEqual(result["mode"], "AUTO_SEQUENCE_FALLBACK")
        self.assertEqual(result["available_sequences"], [20, 15, 10, 5])
        chosen = {(row["stream"], row["segment_id"]): row["selected_sequence"]
                  for row in result["assignments"]}
        self.assertEqual(chosen[("ANOMALY", 1)], None)
        self.assertEqual(chosen[("ANOMALY", 2)], 5)
        self.assertEqual(chosen[("ANOMALY", 7)], 15)
        self.assertEqual(chosen[("ANOMALY", 9)], 20)
        self.assertEqual(result["test_windows"], 10)
        self.assertEqual((result["tn"], result["fp"], result["fn"], result["tp"]), (2, 0, 1, 7))
        self.assertEqual(result["segment_coverage"], 8 / 9)
        self.assertEqual(result["segment_detection_rate"], 7 / 8)
        self.assertEqual(result["per_tier"]["5"]["threshold"], .05)
        self.assertEqual(result["per_tier"]["20"]["threshold"], .2)
        self.assertEqual(result["per_tier"]["20"]["effective_threshold_method"], "PR_INTERSECTION")
        self.assertTrue(all(info["score_method"] == "ROBUST_TOPK_10" for info in result["per_tier"].values()))
        self.assertIsNone(result["threshold"])
        self.assertEqual(len(result["segment_details"]), 9)


if __name__ == "__main__": unittest.main()
