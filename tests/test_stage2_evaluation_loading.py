import csv
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from evaluation_controller import EvaluationController
from model_artifacts import discover_model_runs
from stage2.dataset_artifacts import load_processed_dataset
from train_lstm_ae import OUTPUT_DIR


DATASET = OUTPUT_DIR / "processed_datasets" / "stage2_exp01_abs_minmax_seq20"
RUN_ID = "20261003_135745_504921"


class Stage2EvaluationLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_run = next(run for run in discover_model_runs("KAMP_LSTM_AE") if run.run_id == RUN_ID)

    def test_actual_saved_run_matches_training_time_evaluation(self):
        controller = EvaluationController()
        controller.select_run(self.model_run)
        summary = controller.load_model_and_predictions(force=True)
        self.assertEqual((summary["valid_samples"], summary["test_samples"]), (646, 2197))
        self.assertEqual((summary["valid_normal"], summary["valid_anomaly"]), (517, 129))
        result = controller.evaluate("LAST_STEP_MSE", "PR_INTERSECTION", "NONE")
        with (self.model_run.model_path.parent / "evaluation_results.csv").open(newline="", encoding="utf-8-sig") as handle:
            saved = next(csv.DictReader(handle))
        self.assertAlmostEqual(result["threshold"], float(saved["threshold"]), delta=1e-7)
        for key in ("tn", "fp", "fn", "tp"):
            self.assertEqual(result[key], int(saved[key]))
        for key in ("precision", "recall", "f1"):
            self.assertAlmostEqual(result[key], float(saved[key]), places=12)
        self.assertEqual(result["dataset_id"], "stage2_exp01_abs_minmax_seq20")
        self.assertEqual(result["data_mode"], "PROCESSED_DATASET")
        artifact = load_processed_dataset(DATASET)
        self.assertEqual(self.model_run.metadata["dataset_id"], artifact.dataset_id)
        for key in ("sequence_length", "stride", "gap_threshold_ms", "signal_transform",
                    "scaler", "segment_aware", "use_horizon"):
            self.assertEqual(self.model_run.metadata["preprocessing"][key], artifact.config["preprocessing"][key])

    def test_saved_path_fallback_and_metadata_mismatch(self):
        metadata = self.model_run.metadata.copy()
        metadata["processed_dataset_path"] = str(OUTPUT_DIR / "moved_dataset")
        moved = SimpleNamespace(model_id=self.model_run.model_id, model_path=self.model_run.model_path, metadata=metadata)
        controller = EvaluationController(); controller.select_run(moved)
        self.assertEqual(controller._processed_artifact().dataset_id, "stage2_exp01_abs_minmax_seq20")
        metadata = {**metadata, "preprocessing": {**metadata["preprocessing"], "sequence_length": 15}}
        controller.select_run(SimpleNamespace(model_id=self.model_run.model_id, model_path=self.model_run.model_path, metadata=metadata))
        with self.assertRaisesRegex(ValueError, "sequence_length mismatch"):
            controller.load_model_and_predictions(force=True)
        wrong_segment = {**self.model_run.metadata, "preprocessing": {
            **self.model_run.metadata["preprocessing"], "segment_aware": False}}
        controller.select_run(SimpleNamespace(model_id=self.model_run.model_id,
            model_path=self.model_run.model_path, metadata=wrong_segment))
        with self.assertRaisesRegex(ValueError, "segment_aware mismatch"):
            controller.load_model_and_predictions(force=True)
        wrong_id = {**self.model_run.metadata, "dataset_id": "different_dataset"}
        controller.select_run(SimpleNamespace(model_id=self.model_run.model_id,
            model_path=self.model_run.model_path, metadata=wrong_id))
        with self.assertRaisesRegex(ValueError, "Dataset ID mismatch"):
            controller.load_model_and_predictions(force=True)

    def test_no_raw_preprocessing_and_artifact_segment_metadata(self):
        controller = EvaluationController(); controller.select_run(self.model_run)
        class Model:
            input_shape = (None, 20, 3)
            output_shape = (None, 20, 3)
            def predict(self, x, **kwargs): return np.zeros_like(x)
        with patch("evaluation_controller.tf.keras.models.load_model", return_value=Model()), \
             patch("evaluation_controller.core.load_data", side_effect=AssertionError("raw CSV")), \
             patch("evaluation_controller.core.validate_data", side_effect=AssertionError("raw validate")), \
             patch("evaluation_controller.core.fit_scaler", side_effect=AssertionError("scaler fit")), \
             patch("evaluation_controller.core.transform_features", side_effect=AssertionError("scaler transform")), \
             patch("evaluation_controller.create_task_bundle", side_effect=AssertionError("window generation")):
            summary = controller.load_model_and_predictions(force=True)
        artifact = load_processed_dataset(DATASET)
        np.testing.assert_array_equal(controller._bundle["x_valid"], artifact.validation["X"])
        np.testing.assert_array_equal(controller._bundle["x_test"], artifact.test["X"])
        metadata = controller._bundle["valid_metadata"]
        self.assertEqual(metadata["dataset_id"], artifact.dataset_id)
        np.testing.assert_array_equal(metadata["source_segment_ids"], artifact.validation["segment_ids"])
        np.testing.assert_array_equal(metadata["sample_timestamps"], artifact.validation["window_end_timestamp"])
        self.assertEqual(metadata["gap_threshold_ms"], 150)
        self.assertNotEqual(metadata["segment_ids"][516], metadata["segment_ids"][517])
        captured = []
        from score_postprocessing import apply_temporal_processing
        def temporal(*args, **kwargs):
            captured.append(args[3])
            return apply_temporal_processing(*args, **kwargs)
        with patch("evaluation_controller.apply_temporal_processing", side_effect=temporal):
            controller.evaluate("LAST_STEP_MSE", "NORMAL_P99", "EWMA", .4, True)
        np.testing.assert_array_equal(captured[0], metadata["segment_ids"])

    def test_shape_mismatch_and_cache_dataset_identity(self):
        controller = EvaluationController(); controller.select_run(self.model_run)
        artifact = load_processed_dataset(DATASET)
        class WrongModel:
            input_shape = (None, 15, 3)
            output_shape = (None, 15, 3)
        with patch("evaluation_controller.tf.keras.models.load_model", return_value=WrongModel()):
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                controller.load_model_and_predictions(force=True)
        other = SimpleNamespace(path=artifact.path, config=artifact.config, dataset_id="other_dataset")
        self.assertNotEqual(controller._processed_cache_key(artifact), controller._processed_cache_key(other))

    def test_unknown_data_mode_is_rejected(self):
        metadata = {**self.model_run.metadata, "data_mode": "UNKNOWN"}
        controller = EvaluationController()
        controller.select_run(SimpleNamespace(model_id=self.model_run.model_id,
            model_path=self.model_run.model_path, metadata=metadata))
        with self.assertRaisesRegex(ValueError, "Unsupported evaluation data_mode"):
            controller.load_model_and_predictions()

    def test_dynamic_sequence_prediction_shapes(self):
        original = load_processed_dataset(DATASET)
        for length in (10, 15, 20):
            config = {**original.config, "preprocessing": {
                **original.config["preprocessing"], "sequence_length": length}}
            valid = {**original.validation, "X": original.validation["X"][:, :length, :]}
            test = {**original.test, "X": original.test["X"][:, :length, :]}
            artifact = replace(original, config=config, validation=valid, test=test)
            class Model:
                input_shape = (None, length, 3)
                output_shape = (None, length, 3)
                def predict(self, x, **kwargs): return np.zeros_like(x)
            controller = EvaluationController(); controller.select_run(self.model_run)
            with patch.object(controller, "_processed_artifact", return_value=artifact), \
                 patch("evaluation_controller.tf.keras.models.load_model", return_value=Model()):
                summary = controller.load_model_and_predictions(force=True)
            self.assertEqual(summary["sequence_length"], length)
            self.assertEqual(controller._bundle["valid_error"].shape[1:], (length, 3))


if __name__ == "__main__": unittest.main()
