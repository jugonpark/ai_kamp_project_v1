import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class ProcessedDatasetTrainingGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_mode_selection_artifact_source_of_truth_and_stale_independence(self):
        from gui_app import MainWindow
        window = MainWindow()
        try:
            self.assertEqual(window.data_mode_combo.currentData(), "KAMP_BASELINE")
            self.assertEqual(window.training_model_combo.currentData(), "CNN_LSTM_AUTOENCODER")
            self.assertFalse(window.processed_dataset_combo.isEnabled())
            window.data_mode_combo.setCurrentIndex(window.data_mode_combo.findData("PROCESSED_DATASET"))
            self.assertTrue(window.processed_dataset_combo.isEnabled())
            self.assertEqual(window.training_model_combo.count(), 3)
            self.assertEqual(window.training_model_combo.currentData(), "CNN_LSTM_AUTOENCODER")
            self.assertEqual({window.training_model_combo.itemData(i) for i in range(3)},
                             {"KAMP_LSTM_AE", "CNN_LSTM_AUTOENCODER", "DENOISING_CNN_LSTM_AUTOENCODER"})
            window.processed_dataset_combo.setCurrentIndex(
                window.processed_dataset_combo.findText("stage2_exp01_abs_minmax_seq20"))
            self.assertEqual(window.system_labels["Dataset"].text(), "stage2_exp01_abs_minmax_seq20")
            self.assertIn("Train: 7411 | Validation: 646 | Test: 2197", window.processed_dataset_summary_label.text())
            window.sequence_combo.setCurrentIndex(window.sequence_combo.findData(15))
            window.signal_transform_combo.setCurrentIndex(window.signal_transform_combo.findData("RAW_SIGNED"))
            self.assertEqual(window.system_labels["Sequence"].text(), "20")
            self.assertEqual(window.system_labels["Signal Transform"].text(), "ABS_ALL")
            self.assertEqual(window._selected_processed_artifact.config["preprocessing"]["sequence_length"], 20)
            window.data_mode_combo.setCurrentIndex(window.data_mode_combo.findData("KAMP_BASELINE"))
            self.assertFalse(window.processed_dataset_combo.isEnabled())
            self.assertEqual(window.training_model_combo.currentData(), "CNN_LSTM_AUTOENCODER")
        finally:
            window.close()

    def test_invalid_artifact_does_not_start_worker(self):
        from gui_app import MainWindow
        window = MainWindow()
        try:
            window.data_mode_combo.setCurrentIndex(window.data_mode_combo.findData("PROCESSED_DATASET"))
            window.processed_dataset_combo.setItemData(0, "missing_path")
            with patch("gui_app.QMessageBox.warning") as warning, patch("gui_app.training_engine.TrainingWorker") as worker:
                window.start_training()
            warning.assert_called_once()
            worker.assert_not_called()
        finally:
            window.close()

    def test_standard_artifact_disables_denoising_noise_clip(self):
        from gui_app import MainWindow
        window = MainWindow()
        try:
            window.data_mode_combo.setCurrentIndex(window.data_mode_combo.findData("PROCESSED_DATASET"))
            window.training_model_combo.setCurrentIndex(
                window.training_model_combo.findData("DENOISING_CNN_LSTM_AUTOENCODER"))
            original = window._selected_processed_artifact
            window._selected_processed_artifact = SimpleNamespace(
                config={"preprocessing": {**original.config["preprocessing"], "scaler": "STANDARD"}})
            window._update_model_dependent_controls()
            self.assertFalse(window.noise_clip_check.isEnabled())
            self.assertTrue(window.noise_clip_check.isChecked())
            self.assertIn("요청: ON | 실제 적용: OFF", window.noise_clip_policy_label.text())
            window._selected_processed_artifact = original
        finally:
            window.close()

    def test_immediate_evaluation_keeps_dataset_identity_in_comparison(self):
        from gui_app import MainWindow
        window = MainWindow()
        try:
            window.update_evaluation({"data_mode": "PROCESSED_DATASET", "dataset_id": "stage2_exp01_abs_minmax_seq20",
                "model_id": "KAMP_LSTM_AE", "experiment_name": "quick", "sequence_length": 20,
                "signal_transform": "ABS_ALL", "scaler": "MINMAX", "gap_threshold_ms": 150,
                "threshold": .1, "accuracy": .9, "precision": .7, "recall": .6, "f1_score": .65,
                "confusion_matrix": [[10, 2], [3, 4]]})
            self.assertEqual(window.comparison_table.columnCount(), 36)
            self.assertIn("stage2_exp01_abs_minmax_seq20", window.comparison_table.item(0, 2).text())
            self.assertEqual(window.comparison_history[0]["signal_transform"], "ABS_ALL")
            self.assertIn("Gap: 150 ms", window.comparison_table.item(0, 2).toolTip())
        finally:
            window.close()

    def test_switching_saved_datasets_updates_input_shape(self):
        from gui_app import MainWindow
        from preprocessing_config import default_preprocessing_config
        from stage2.preprocessing import run_stage2_preprocessing
        from stage2.dataset_artifacts import save_processed_dataset
        from train_lstm_ae import NORMAL_PATH, OUTLIER_PATH, OUTPUT_DIR
        path15 = OUTPUT_DIR / ".test_artifacts" / "dynamic_processed_dataset" / "seq15"
        if not path15.exists():
            result = run_stage2_preprocessing(NORMAL_PATH, OUTLIER_PATH,
                {**default_preprocessing_config(), "sequence_length": 15})
            save_processed_dataset(result, "seq15", path15.parent)
        path20 = OUTPUT_DIR / "processed_datasets" / "stage2_exp01_abs_minmax_seq20"
        entries = [{"dataset_id": "seq15", "path": str(path15)},
                   {"dataset_id": "stage2_exp01_abs_minmax_seq20", "path": str(path20)}]
        window = MainWindow()
        try:
            with patch("gui_app.discover_processed_datasets", return_value=entries):
                window.data_mode_combo.setCurrentIndex(window.data_mode_combo.findData("PROCESSED_DATASET"))
            self.assertEqual(window.system_labels["Input Shape"].text(), "(15, 3)")
            self.assertEqual(window.system_labels["Dataset"].text(), "seq15")
            self.assertEqual(window._selected_processed_artifact.train["X"].shape[1:], (15, 3))
            window.processed_dataset_combo.setCurrentIndex(1)
            self.assertEqual(window.system_labels["Input Shape"].text(), "(20, 3)")
            self.assertEqual(window.system_labels["Dataset"].text(), "stage2_exp01_abs_minmax_seq20")
        finally:
            window.close()


if __name__ == "__main__": unittest.main()
