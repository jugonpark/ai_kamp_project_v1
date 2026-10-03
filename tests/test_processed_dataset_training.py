import unittest
import json
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from preprocessing_config import default_preprocessing_config
from stage2.dataset_artifacts import load_processed_dataset
from train_lstm_ae import OUTPUT_DIR
from training_engine import TrainingWorker, build_training_model
from model_registry import MODEL_REGISTRY


DATASET_PATH = OUTPUT_DIR / "processed_datasets" / "stage2_exp01_abs_minmax_seq20"


class ProcessedDatasetTrainingTests(unittest.TestCase):
    def test_default_still_uses_baseline_mode(self):
        worker = TrainingWorker()
        self.assertEqual(worker.data_mode, "KAMP_BASELINE")
        self.assertIsNone(worker.processed_dataset_path)

    def test_artifact_labels_and_dynamic_kamp_shapes(self):
        artifact = load_processed_dataset(DATASET_PATH)
        self.assertTrue(np.all(artifact.train["y"] == 0))
        for length in (5, 10, 15, 20):
            config = {**default_preprocessing_config(), "sequence_length": length}
            model = build_training_model(MODEL_REGISTRY["KAMP_LSTM_AE"], {}, config)
            self.assertEqual(model.input_shape, (None, length, 3))
            self.assertEqual(model.output_shape, (None, length, 3))

    def test_processed_worker_uses_exact_windows_and_normal_fit_validation(self):
        artifact = load_processed_dataset(DATASET_PATH)
        class FakeModel:
            input_shape = (None, 20, 3)
            output_shape = (None, 20, 3)
            trainable_weights = []
            def count_params(self): return 1
            def compile(self, **kwargs): pass
            def fit(self, x, targets, **kwargs):
                np.testing.assert_array_equal(x, artifact.train["X"])
                np.testing.assert_array_equal(targets, x)
                valid_x, valid_target = kwargs["validation_data"]
                np.testing.assert_array_equal(valid_x, artifact.validation["X"][artifact.validation["y"] == 0])
                np.testing.assert_array_equal(valid_target, valid_x)
                self_test.fit_seen = True
                return SimpleNamespace(history={"loss": [.2], "val_loss": [.3]})
            def predict(self, x, **kwargs): return np.zeros_like(x)
        self_test = self
        self.fit_seen = False
        worker = TrainingWorker(True, "KAMP_LSTM_AE", {}, data_mode="PROCESSED_DATASET",
                                processed_dataset_path=DATASET_PATH)
        ready, evaluation, errors = [], [], []
        worker.dataset_ready.connect(ready.append)
        worker.evaluation_ready.connect(evaluation.append)
        worker.failed.connect(errors.append)
        saved = SimpleNamespace(model_path=Path("model.keras"), metadata={"experiment_name": "test"})
        def threshold(labels, scores):
            np.testing.assert_array_equal(labels, artifact.validation["y"])
            self.assertEqual(len(scores), 646)
            return .5, .4, .6, None, None, None
        with patch("training_engine.load_processed_dataset", return_value=artifact), \
             patch("training_engine.apply_random_seed"), \
             patch("training_engine.build_training_model", return_value=FakeModel()), \
             patch("training_engine.core.calculate_pr_intersection_threshold", side_effect=threshold), \
             patch("training_engine.save_model_run", return_value=saved) as save, \
             patch("training_engine.append_evaluation_result"), \
             patch("training_engine.core.load_data", side_effect=AssertionError("CSV reload")), \
             patch("training_engine.core.fit_scaler", side_effect=AssertionError("scaler fit")), \
             patch("training_engine.core.transform_features", side_effect=AssertionError("scaler transform")), \
             patch("training_engine.create_task_bundle", side_effect=AssertionError("window rebuild")):
            worker.run()
        self.assertEqual(errors, [])
        self.assertTrue(self.fit_seen)
        self.assertEqual(ready[0]["train_sequences"], 7411)
        self.assertEqual(ready[0]["validation_windows"], 646)
        self.assertEqual(ready[0]["test_windows"], 2197)
        self.assertEqual(evaluation[0]["dataset_id"], artifact.dataset_id)
        self.assertEqual(len(evaluation[0]["test_errors"]), 2197)
        self.assertEqual(save.call_args.kwargs["preprocessing_config"], artifact.config["preprocessing"])
        self.assertEqual(save.call_args.args[3]["dataset_id"], artifact.dataset_id)

    def test_invalid_artifact_fails_before_training(self):
        worker = TrainingWorker(True, "KAMP_LSTM_AE", {}, data_mode="PROCESSED_DATASET",
                                processed_dataset_path=OUTPUT_DIR / "missing_dataset")
        errors = []
        worker.failed.connect(errors.append)
        worker.run()
        self.assertEqual(len(errors), 1)
        self.assertIn("Incomplete processed dataset", errors[0])

    def test_processed_metadata_is_artifact_actual_and_baseline_schema_stays_same(self):
        import model_artifacts
        artifact = load_processed_dataset(DATASET_PATH)
        class DummyModel:
            input_shape = (None, 20, 3)
            def save(self, path): Path(path).write_bytes(b"model")
            def count_params(self): return 1
        root = OUTPUT_DIR / ".test_artifacts" / f"metadata_{uuid.uuid4().hex}"
        info = {"requested_epochs": 1, "train_sequences": 7411, "valid_samples": 646,
                "test_samples": 2197, "data_mode": "PROCESSED_DATASET", "dataset_id": artifact.dataset_id,
                "processed_dataset_path": str(artifact.path), "dataset_config_sha256": "a" * 64,
                "source_csv_sha256": {stream: artifact.config["source"][stream]["sha256"]
                                      for stream in ("normal", "anomaly")}}
        try:
            with patch.object(model_artifacts, "RUNS_ROOT", root), \
                 patch.object(model_artifacts.plot_utils, "save_training_loss"):
                run = model_artifacts.save_model_run(DummyModel(), SimpleNamespace(history={"loss": [.1], "val_loss": [.2]}),
                    MODEL_REGISTRY["KAMP_LSTM_AE"], info, preprocessing_config=artifact.config["preprocessing"])
                denoising_run = model_artifacts.save_model_run(
                    DummyModel(), SimpleNamespace(history={"loss": [.1], "val_loss": [.2]}),
                    MODEL_REGISTRY["DENOISING_CNN_LSTM_AUTOENCODER"],
                    {**info, "noise_clip_requested": True, "noise_clip_effective": False},
                    preprocessing_config={**artifact.config["preprocessing"], "scaler": "STANDARD"})
                baseline = model_artifacts.save_model_run(DummyModel(), SimpleNamespace(history={"loss": [.1], "val_loss": [.2]}),
                    MODEL_REGISTRY["KAMP_LSTM_AE"], {"requested_epochs": 1},
                    preprocessing_config=default_preprocessing_config("KAMP_BASELINE"))
            saved = json.loads((run.model_path.parent / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["preprocessing"], artifact.config["preprocessing"])
            self.assertEqual(saved["artifact_windows"], {"train": 7411, "validation": 646, "test": 2197})
            self.assertEqual(saved["data_mode"], "PROCESSED_DATASET")
            denoising = json.loads((denoising_run.model_path.parent / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(denoising["model_id"], "DENOISING_CNN_LSTM_AUTOENCODER")
            self.assertEqual(denoising["scaler"], "STANDARD")
            self.assertTrue(denoising["denoising_enabled"])
            self.assertTrue(denoising["noise_clip_requested"])
            self.assertFalse(denoising["noise_clip_effective"])
            legacy = json.loads((baseline.model_path.parent / "metadata.json").read_text(encoding="utf-8"))
            self.assertNotIn("data_mode", legacy)
            self.assertNotIn("artifact_windows", legacy)
            self.assertNotIn("noise_clip_effective", legacy)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__": unittest.main()
