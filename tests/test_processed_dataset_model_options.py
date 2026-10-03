"""Mock-only checks for the Stage 2 reconstruction model routes."""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from evaluation_controller import EvaluationController
from model_registry import MODEL_REGISTRY, PROCESSED_DATASET_MODEL_IDS
from preprocessing_config import default_preprocessing_config
from training_config import default_training_config
from training_engine import TrainingWorker, build_training_model
from train_lstm_ae import OUTPUT_DIR
from stage2.dataset_artifacts import load_processed_dataset


DATASET = OUTPUT_DIR / "processed_datasets" / "stage2_exp01_abs_minmax_seq20"


class ProcessedDatasetModelOptionTests(unittest.TestCase):
    def test_allowlist_and_dynamic_shapes(self):
        self.assertEqual(set(PROCESSED_DATASET_MODEL_IDS), {
            "KAMP_LSTM_AE", "CNN_LSTM_AUTOENCODER", "DENOISING_CNN_LSTM_AUTOENCODER"})
        self.assertNotIn("LSTM_FORECAST_5", PROCESSED_DATASET_MODEL_IDS)
        self.assertNotIn("GRU_AUTOENCODER", PROCESSED_DATASET_MODEL_IDS)
        for model_id in PROCESSED_DATASET_MODEL_IDS:
            for length in (5, 10, 15, 20):
                with self.subTest(model=model_id, length=length):
                    model = build_training_model(MODEL_REGISTRY[model_id], default_training_config(),
                        {**default_preprocessing_config(), "sequence_length": length})
                    self.assertEqual(model.input_shape, (None, length, 3))
                    self.assertEqual(model.output_shape, (None, length, 3))

    def _worker_case(self, model_id, scaler, requested_clip):
        artifact = load_processed_dataset(DATASET)
        seq = artifact.train["X"].shape[1]
        clean = np.zeros((4, seq, 3), dtype=np.float32)
        if scaler == "STANDARD":
            clean[:, :, 0] = -2
            clean[:, :, 1] = 2
        else:
            clean[:, :, 0] = 0
            clean[:, :, 1] = 1
        normal = np.zeros((2, seq, 3), dtype=np.float32)
        anomaly = np.ones((2, seq, 3), dtype=np.float32)
        sample = SimpleNamespace(
            path=artifact.path, dataset_id=artifact.dataset_id,
            config={**artifact.config, "preprocessing": {**artifact.config["preprocessing"], "scaler": scaler}},
            summary=artifact.summary,
            train={"X": clean, "y": np.zeros(4, dtype=int)},
            validation={"X": np.concatenate((normal, anomaly)), "y": np.array([0, 0, 1, 1])},
            test={"X": np.concatenate((normal, anomaly)), "y": np.array([0, 0, 1, 1])})
        observed = {}

        class FakeModel:
            input_shape = output_shape = (None, seq, 3)
            trainable_weights = []
            def count_params(self): return 1
            def compile(self, **kwargs): pass
            def fit(self, inputs, targets=None, **kwargs):
                if MODEL_REGISTRY[model_id].denoising:
                    noisy, targets = next(iter(inputs))
                    observed["input"] = noisy.numpy()
                    observed["target"] = targets.numpy()
                else:
                    observed["input"] = inputs
                    observed["target"] = targets
                observed["validation"] = kwargs["validation_data"]
                return SimpleNamespace(history={"loss": [.2], "val_loss": [.3]})
            def predict(self, inputs, **kwargs): return np.zeros_like(inputs)

        worker = TrainingWorker(True, model_id, {"noise_std": .01, "noise_clip": requested_clip},
            data_mode="PROCESSED_DATASET", processed_dataset_path=DATASET)
        errors = []
        worker.failed.connect(errors.append)
        saved = SimpleNamespace(model_path=Path("model.keras"), metadata={"experiment_name": "mock"})
        with patch("training_engine.load_processed_dataset", return_value=sample), \
             patch("training_engine.build_training_model", return_value=FakeModel()), \
             patch("training_engine.apply_random_seed"), \
             patch("training_engine.save_model_run", return_value=saved) as save, \
             patch("training_engine.append_evaluation_result"), \
             patch("training_engine.core.load_data", side_effect=AssertionError("raw reload")), \
             patch("training_engine.core.fit_scaler", side_effect=AssertionError("scaler fit")), \
             patch("training_engine.core.transform_features", side_effect=AssertionError("transform")), \
             patch("training_engine.create_task_bundle", side_effect=AssertionError("window creation")):
            worker.run()
        self.assertEqual(errors, [])
        np.testing.assert_array_equal(observed["target"], clean)
        np.testing.assert_array_equal(observed["validation"][0], normal)
        np.testing.assert_array_equal(observed["validation"][1], normal)
        info = save.call_args.args[3]
        self.assertEqual(info["dataset_id"], artifact.dataset_id)
        self.assertEqual(save.call_args.kwargs["preprocessing_config"]["scaler"], scaler)
        return clean, observed["input"], info

    def test_cnn_uses_clean_windows(self):
        clean, fit_input, info = self._worker_case("CNN_LSTM_AUTOENCODER", "MINMAX", True)
        np.testing.assert_array_equal(fit_input, clean)
        self.assertFalse(info["noise_clip_effective"])

    def test_denoising_minmax_clip(self):
        clean, fit_input, info = self._worker_case("DENOISING_CNN_LSTM_AUTOENCODER", "MINMAX", True)
        self.assertTrue(np.any(fit_input != clean))
        self.assertGreaterEqual(float(fit_input.min()), 0)
        self.assertLessEqual(float(fit_input.max()), 1)
        self.assertTrue(info["noise_clip_effective"])

    def test_denoising_standard_never_clips_signed_values(self):
        clean, fit_input, info = self._worker_case("DENOISING_CNN_LSTM_AUTOENCODER", "STANDARD", True)
        self.assertTrue(np.any(fit_input != clean))
        self.assertLess(float(fit_input[:, :, 0].max()), 0)
        self.assertGreater(float(fit_input[:, :, 1].min()), 1)
        self.assertTrue(info["noise_clip_requested"])
        self.assertFalse(info["noise_clip_effective"])

    def test_cnn_and_denoising_reload_clean_artifact(self):
        artifact = load_processed_dataset(DATASET)
        for model_id in ("CNN_LSTM_AUTOENCODER", "DENOISING_CNN_LSTM_AUTOENCODER"):
            class FakeModel:
                input_shape = output_shape = (None, 20, 3)
                def predict(self, x, **kwargs): return np.zeros_like(x)
            controller = EvaluationController(model_id=model_id)
            controller.model_run = SimpleNamespace(metadata={"data_mode": "PROCESSED_DATASET"})
            with self.subTest(model=model_id), \
                 patch.object(controller, "_processed_artifact", return_value=artifact), \
                 patch.object(controller, "_processed_cache_key", return_value=(model_id,)), \
                 patch("evaluation_controller.tf.keras.models.load_model", return_value=FakeModel()), \
                 patch("evaluation_controller.core.load_data", side_effect=AssertionError("raw reload")), \
                 patch("evaluation_controller.create_task_bundle", side_effect=AssertionError("window creation")):
                summary = controller.load_model_and_predictions(force=True)
            self.assertEqual(summary["model_id"], model_id)
            np.testing.assert_array_equal(controller._bundle["x_valid"], artifact.validation["X"])
            np.testing.assert_array_equal(controller._bundle["x_test"], artifact.test["X"])
            for temporal in ("NONE", "EWMA"):
                result = controller.evaluate("LAST_STEP_MSE", "PR_INTERSECTION", temporal)
                self.assertEqual(result["model_id"], model_id)
                self.assertIn("segment_detection_rate", result)
                self.assertIn("median_segment_delay_seconds", result)


if __name__ == "__main__": unittest.main()
