import inspect
import csv
import json
import shutil
import uuid
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

class ArtifactAndTrainingTests(unittest.TestCase):
    def test_temporal_results_preserve_historical_csv_and_align_columns(self):
        import model_artifacts
        folder = Path.cwd() / "outputs" / ".test_artifacts" / uuid.uuid4().hex
        try:
            folder.mkdir(parents=True)
            model_path = folder / "model.keras"
            run = SimpleNamespace(model_path=model_path)
            old_path = folder / "evaluation_results.csv"
            old_bytes = b"f1,fp,fn\r\n0.8,3,4\r\n"
            old_path.write_bytes(old_bytes)
            first = model_artifacts.append_evaluation_result(run, {
                "temporal_method": "NONE", "ewma_alpha": .4, "timestamp_aware": True,
                "timestamp_gap_threshold": .15, "f1": .8, "fp": 3, "fn": 4,
                "total_error": 7, "detection_delay_samples": "", "pareto": "",
            })
            second = model_artifacts.append_evaluation_result(run, {
                "temporal_method": "EWMA", "ewma_alpha": .2, "timestamp_aware": True,
                "timestamp_gap_threshold": .15, "f1": .9, "fp": 2, "fn": 3,
                "total_error": 5, "detection_delay_samples": 2,
                "detection_delay_seconds": .2, "pareto": True,
            })
            self.assertEqual(first, second)
            self.assertEqual(first.name, "evaluation_results_v2.csv")
            self.assertEqual(old_path.read_bytes(), old_bytes)
            with first.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["temporal_method"], "NONE")
            self.assertEqual(rows[0]["ewma_alpha"], "")
            self.assertEqual(rows[1]["ewma_alpha"], "0.2")
            self.assertEqual(rows[1]["total_error"], "5")
            self.assertEqual(rows[1]["pareto"], "True")
            self.assertEqual(first.read_bytes().count(b"\xef\xbb\xbf"), 1)
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def test_legacy_result_with_mismatched_header_does_not_corrupt_csv(self):
        import model_artifacts
        folder = Path.cwd() / "outputs" / ".test_artifacts" / uuid.uuid4().hex
        try:
            folder.mkdir(parents=True)
            run = SimpleNamespace(model_path=folder / "model.keras")
            old_path = folder / "evaluation_results.csv"
            old_bytes = b"f1\r\n0.8\r\n"
            old_path.write_bytes(old_bytes)
            path = model_artifacts.append_evaluation_result(run, {"f1": .9, "fp": 2})
            self.assertNotEqual(path, old_path)
            self.assertEqual(old_path.read_bytes(), old_bytes)
            with path.open(newline="", encoding="utf-8-sig") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["fp"], "2")
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def test_legacy_kamp_model_is_discoverable(self):
        from model_artifacts import discover_model_runs
        runs = discover_model_runs("KAMP_LSTM_AE")
        self.assertTrue(any(run.run_id == "LEGACY_KAMP" and run.model_path.name == "kamp_lstm_autoencoder.keras" for run in runs))

    def test_training_worker_accepts_selected_model(self):
        from training_engine import TrainingWorker
        self.assertIn("selected_model_id", inspect.signature(TrainingWorker).parameters)
        self.assertEqual(TrainingWorker(True, "GRU_AUTOENCODER").selected_model_id, "GRU_AUTOENCODER")
        self.assertEqual(TrainingWorker(True, "DENOISING_CNN_LSTM_AUTOENCODER", {"noise_std": .02}).config["noise_std"], .02)

    def test_run_artifacts_include_summary_and_complete_metadata(self):
        import model_artifacts
        from model_registry import MODEL_REGISTRY
        class DummyModel:
            def save(self, path): Path(path).write_bytes(b"model")
            def count_params(self): return 123
        history = SimpleNamespace(history={"loss":[.2,.1], "val_loss":[.3,.15]})
        info = {"requested_epochs":2,"experiment_name":"AUDIT","random_seed":42,"optimizer":"AdamW",
                "weight_decay":.0002,"loss":"Huber","huber_delta":.5,"learning_rate":.001,"batch_size":64,
                "reduce_lr_enabled":True,"reduce_lr_factor":.6,"reduce_lr_patience":3,"min_lr":1e-6,
                "early_stopping_enabled":True,"early_stopping_patience":5,"early_stopping_min_delta":1e-5,
                "restore_best_weights":True,"noise_type":"Gaussian","noise_mean":0,"noise_std":.01,
                "noise_clip":True,"training_duration_seconds":1.25,"cnn_kernel_size":5,
                "cnn_filters":16,"bottleneck_units":64}
        folder = Path.cwd() / "outputs" / ".test_artifacts" / uuid.uuid4().hex
        try:
          with patch.object(model_artifacts, "RUNS_ROOT", folder):
            run = model_artifacts.save_model_run(DummyModel(), history, MODEL_REGISTRY["DENOISING_CNN_LSTM_AUTOENCODER"], info)
            names = {p.name for p in run.model_path.parent.iterdir()}
            self.assertTrue({"model.keras","metadata.json","training_history.csv","training_loss.png","experiment_summary.json"}.issubset(names))
            metadata = json.loads((run.model_path.parent / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["callbacks"]["early_stopping"]["patience"], 5)
            self.assertEqual(metadata["callbacks"]["reduce_lr"]["factor"], .6)
            self.assertEqual(metadata["training_duration_seconds"], 1.25)
            self.assertEqual(metadata["noise_generation"], "dynamic_per_batch")
            self.assertEqual(metadata["noise_std"], .01)
            self.assertEqual(metadata["cnn_kernel_size"], 5)
            self.assertEqual(metadata["cnn_filters"], 16)
            self.assertEqual(metadata["bottleneck_units"], 64)
            summary = json.loads((run.model_path.parent / "experiment_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["cnn_kernel_size"], 5)
            self.assertEqual(summary["cnn_filters"], 16)
            self.assertEqual(summary["bottleneck_units"], 64)
            path = model_artifacts.append_evaluation_result(run, {"f1":.9})
            self.assertTrue(path.is_file())
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def test_existing_saved_models_load_without_training_or_evaluation(self):
        import tensorflow as tf
        from model_artifacts import discover_model_runs
        for model_id in ("CNN_LSTM_AUTOENCODER", "DENOISING_CNN_LSTM_AUTOENCODER", "KAMP_LSTM_AE"):
            runs = discover_model_runs(model_id)
            self.assertTrue(runs, model_id)
            run = next((item for item in runs if item.metadata.get("noise_generation") == "legacy_static"), runs[0]) if model_id == "DENOISING_CNN_LSTM_AUTOENCODER" else runs[0]
            model = tf.keras.models.load_model(run.model_path, compile=False)
            self.assertEqual(model.input_shape[1:], (20, 3))
            self.assertEqual(model.output_shape[1:], (20, 3))
            self.assertEqual(run.metadata.get("noise_generation"),
                             "legacy_static" if model_id == "DENOISING_CNN_LSTM_AUTOENCODER" else "none")

    def test_monitor_best_epoch_is_stored_in_new_metadata(self):
        import model_artifacts
        from model_registry import MODEL_REGISTRY
        class DummyModel:
            def save(self, path): Path(path).write_bytes(b"model")
            def count_params(self): return 123
        history = SimpleNamespace(history={"loss":[.2,.1], "val_loss":[.3,.295]})
        folder = Path.cwd() / "outputs" / ".test_artifacts" / uuid.uuid4().hex
        try:
            with patch.object(model_artifacts, "RUNS_ROOT", folder):
                run = model_artifacts.save_model_run(DummyModel(), history, MODEL_REGISTRY["DENOISING_CNN_LSTM_AUTOENCODER"],
                    {"requested_epochs":2, "monitor_best_epoch":1, "monitor_best_val_loss":.3})
            self.assertEqual(run.metadata["best_epoch"], 1)
            self.assertEqual(run.metadata["best_val_loss"], .3)
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def test_historical_run_without_kernel_metadata_is_unknown(self):
        import model_artifacts
        folder = Path.cwd() / "outputs" / ".test_artifacts" / uuid.uuid4().hex
        run_dir = folder / "CNN_LSTM_AUTOENCODER" / "old_run"
        try:
            run_dir.mkdir(parents=True)
            (run_dir / "model.keras").write_bytes(b"saved model")
            (run_dir / "metadata.json").write_text('{"model_id":"CNN_LSTM_AUTOENCODER"}', encoding="utf-8")
            with patch.object(model_artifacts, "RUNS_ROOT", folder):
                run = model_artifacts.discover_model_runs("CNN_LSTM_AUTOENCODER")[0]
            self.assertEqual(run.metadata["cnn_kernel_size"], "unknown")
            self.assertEqual(run.metadata["cnn_filters"], "unknown")
            self.assertEqual(run.metadata["bottleneck_units"], "unknown")
            self.assertEqual(run.model_path, run_dir / "model.keras")
        finally:
            shutil.rmtree(folder, ignore_errors=True)

if __name__ == "__main__": unittest.main()
