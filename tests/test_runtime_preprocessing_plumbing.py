import json
import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from preprocessing_config import default_preprocessing_config, preprocessing_from_metadata
from score_postprocessing import detect_timestamp_segments


class RuntimePreprocessingPlumbingTests(unittest.TestCase):
    def test_worker_defaults_to_baseline_and_accepts_gui_config(self):
        from training_engine import TrainingWorker
        baseline = TrainingWorker()
        self.assertEqual(baseline.preprocessing_config["sequence_length"], 20)
        self.assertEqual(baseline.preprocessing_config["prediction_horizon"], 100)
        self.assertEqual(baseline.preprocessing_config["mode"], "KAMP_BASELINE")
        selected = {**default_preprocessing_config(), "sequence_length": 15,
                    "signal_transform": "RAW_SIGNED", "scaler": "STANDARD"}
        worker = TrainingWorker(True, "KAMP_LSTM_AE", {}, selected)
        self.assertEqual(worker.preprocessing_config, selected)
        self.assertIsNot(worker.preprocessing_config, selected)

    def test_explicit_gap_preserves_legacy_default(self):
        times = [0.0, 0.1, 0.28, 0.38]
        default_ids, default_threshold = detect_timestamp_segments(times)
        ids_150, threshold_150 = detect_timestamp_segments(times, gap_threshold_seconds=0.15)
        ids_200, threshold_200 = detect_timestamp_segments(times, gap_threshold_seconds=0.20)
        self.assertAlmostEqual(default_threshold, 0.15)
        self.assertAlmostEqual(threshold_150, 0.15)
        self.assertEqual(threshold_200, 0.20)
        np.testing.assert_array_equal(default_ids, ids_150)
        np.testing.assert_array_equal(ids_150, [0, 0, 1, 1])
        np.testing.assert_array_equal(ids_200, [0, 0, 0, 0])

    def test_task_bundle_optional_dimensions_and_default_path(self):
        from model_data import create_task_bundle
        from model_registry import MODEL_REGISTRY
        from train_lstm_ae import FEATURES, LABEL_COLUMN, TIMESTAMP_COLUMN
        rows = 130
        frame = pd.DataFrame({**{feature: np.arange(rows) for feature in FEATURES},
                              LABEL_COLUMN: np.zeros(rows, dtype=int),
                              TIMESTAMP_COLUMN: pd.date_range("2026-01-01", periods=rows, freq="100ms")})
        spec = MODEL_REGISTRY["KAMP_LSTM_AE"]
        self.assertEqual(create_task_bundle(frame, spec).inputs.shape, (10, 20, 3))
        dynamic = create_task_bundle(frame, spec, sequence_length=15, prediction_horizon=100,
                                     gap_threshold_seconds=.2)
        self.assertEqual(dynamic.inputs.shape, (15, 15, 3))
        self.assertEqual(dynamic.timestamp_gap_threshold, .2)

    def test_metadata_distinguishes_effective_and_requested_config(self):
        import model_artifacts
        from model_registry import MODEL_REGISTRY
        class DummyModel:
            def save(self, path): Path(path).write_bytes(b"model")
            def count_params(self): return 123
        folder = Path.cwd() / "outputs" / ".test_artifacts" / uuid.uuid4().hex
        selected = {**default_preprocessing_config(), "sequence_length": 15,
                    "signal_transform": "RAW_SIGNED"}
        try:
            with patch.object(model_artifacts, "RUNS_ROOT", folder), patch.object(model_artifacts.plot_utils, "save_training_loss"):
                run = model_artifacts.save_model_run(DummyModel(), SimpleNamespace(history={"loss": [.1], "val_loss": [.2]}),
                    MODEL_REGISTRY["KAMP_LSTM_AE"], {"requested_epochs": 1},
                    preprocessing_config=default_preprocessing_config("KAMP_BASELINE"),
                    requested_preprocessing_config=selected)
            saved = json.loads((run.model_path.parent / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["sequence_length"], 20)
            self.assertEqual(saved["preprocessing"]["mode"], "KAMP_BASELINE")
            self.assertEqual(saved["requested_preprocessing"], selected)
            self.assertEqual(preprocessing_from_metadata(saved)["sequence_length"], 20)
            self.assertEqual(preprocessing_from_metadata({"preprocessing": "abs + train-only MinMaxScaler"})["mode"], "KAMP_BASELINE")
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def test_saved_run_preprocessing_wins_over_current_gui_config(self):
        from evaluation_controller import EvaluationController
        from model_artifacts import ModelRun
        selected = {**default_preprocessing_config(), "sequence_length": 15}
        controller = EvaluationController(preprocessing_config=selected)
        legacy = ModelRun("KAMP_LSTM_AE", "legacy", Path("legacy.keras"), {})
        controller.select_run(legacy)
        self.assertEqual(controller.preprocessing_config["mode"], "KAMP_BASELINE")
        self.assertEqual(controller.preprocessing_config["sequence_length"], 20)

    def test_missing_data_mode_uses_legacy_baseline_evaluation(self):
        import model_artifacts
        from evaluation_controller import EvaluationController
        from model_registry import MODEL_REGISTRY
        class DummyModel:
            input_shape = (None, 15, 3)
            def save(self, path): Path(path).write_bytes(b"model")
            def count_params(self): return 123
        folder = Path.cwd() / "outputs" / ".test_artifacts" / uuid.uuid4().hex
        stage2 = {**default_preprocessing_config(), "sequence_length": 15,
                  "gap_threshold_ms": 200, "signal_transform": "RAW_SIGNED", "scaler": "STANDARD"}
        try:
            with patch.object(model_artifacts, "RUNS_ROOT", folder), patch.object(model_artifacts.plot_utils, "save_training_loss"):
                run = model_artifacts.save_model_run(DummyModel(), SimpleNamespace(history={"loss": [.1], "val_loss": [.2]}),
                    MODEL_REGISTRY["KAMP_LSTM_AE"], {"requested_epochs": 1, "dataset_id": "future_dataset"},
                    preprocessing_config=stage2)
            self.assertEqual(run.metadata["sequence_length"], 15)
            self.assertEqual(run.metadata["preprocessing"], stage2)
            controller = EvaluationController()
            controller.select_run(run)
            self.assertEqual(controller.preprocessing_config["mode"], "KAMP_BASELINE")
            with patch.object(controller, "_signature", return_value=("file", 1, 1)):
                key = controller.cache_key()
            self.assertNotIn("future_dataset", key)
            with patch("evaluation_controller.core.load_data", side_effect=RuntimeError("baseline raw CSV path")):
                with self.assertRaisesRegex(RuntimeError, "baseline raw CSV path"):
                    controller.load_model_and_predictions(force=True)
        finally:
            shutil.rmtree(folder, ignore_errors=True)
