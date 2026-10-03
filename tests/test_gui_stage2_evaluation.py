import os
import time
import unittest
import csv
import uuid
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class Stage2EvaluationGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def _wait(self, window):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            self.app.processEvents()
            if window.evaluation_thread is None:
                return
            time.sleep(.01)
        self.fail("Evaluation worker did not finish")

    def test_real_quick_test_run_load_and_evaluate(self):
        from gui_app import MainWindow
        window = MainWindow(); window.show()
        try:
            window.evaluation_model_combo.setCurrentIndex(window.evaluation_model_combo.findData("KAMP_LSTM_AE"))
            index = window.evaluation_run_combo.findText("20261003_135745_504921")
            self.assertGreaterEqual(index, 0)
            window.evaluation_run_combo.setCurrentIndex(index)
            window.score_combo.setCurrentIndex(window.score_combo.findData("LAST_STEP_MSE"))
            window.threshold_combo.setCurrentIndex(window.threshold_combo.findData("PR_INTERSECTION"))
            window.temporal_combo.setCurrentIndex(window.temporal_combo.findData("NONE"))
            window.load_model_button.click(); self._wait(window)
            self.assertIn("DATA MODE: PROCESSED DATASET", window.evaluation_dataset_label.text())
            self.assertIn("stage2_exp01_abs_minmax_seq20", window.evaluation_dataset_label.text())
            self.assertIn("Validation: 646 | Test: 2197", window.evaluation_dataset_label.text())
            self.assertEqual(window.evaluation_gap_label.text(), "150 ms")
            with patch("gui_app.append_evaluation_result", return_value="test-only"):
                window.run_evaluation_button.click(); self._wait(window)
            self.assertEqual(window.eval_result_labels["FP (False Alarm)"].text(), "56")
            self.assertEqual(window.eval_result_labels["FN (Missed Anomaly)"].text(), "79")
            self.assertEqual(window.comparison_history[-1]["dataset_id"], "stage2_exp01_abs_minmax_seq20")
            self.assertEqual(window.comparison_history[-1]["data_mode"], "PROCESSED_DATASET")
        finally:
            window.close()

    def test_comparison_fingerprint_export_and_legacy_import(self):
        from gui_app import MainWindow
        from train_lstm_ae import OUTPUT_DIR
        root = OUTPUT_DIR / ".test_artifacts" / f"stage2_comparison_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        window = MainWindow()
        try:
            result = {"model_id": "KAMP_LSTM_AE", "model": "KAMP LSTM AutoEncoder",
                      "experiment_name": "same", "score_method": "LAST_STEP_MSE",
                      "threshold_method": "PR_INTERSECTION", "threshold": .1,
                      "data_mode": "PROCESSED_DATASET", "dataset_id": "dataset_a",
                      "sequence_length": 20, "stride": 1, "signal_transform": "ABS_ALL",
                      "scaler": "MINMAX", "gap_threshold_ms": 150, "use_horizon": False}
            window._append_comparison(result)
            window._append_comparison({**result, "dataset_id": "dataset_b"})
            self.assertEqual(len(window.comparison_history), 2)
            self.assertIn("dataset_b", window.comparison_table.item(1, 2).text())
            with patch("gui_app.core.OUTPUT_DIR", root):
                window.export_comparison()
            exported = next((root / "gui_comparisons").glob("*.csv"))
            with exported.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[1]["dataset_id"], "dataset_b")
            self.assertEqual(rows[1]["data_mode"], "PROCESSED_DATASET")
            legacy = root / "legacy.csv"
            legacy.write_text("model,score_method,threshold_method,threshold\nKAMP LSTM AutoEncoder,LAST_STEP_MSE,PR_INTERSECTION,0.2\n", encoding="utf-8")
            with patch("gui_app.QFileDialog.getOpenFileName", return_value=(str(legacy), "")):
                window.import_historical_result()
            self.assertEqual(len(window.comparison_history), 3)
            self.assertEqual(window.comparison_history[-1]["dataset_id"], "")
        finally:
            window.close()

    def test_load_failure_shows_error_without_crashing_gui(self):
        from gui_app import MainWindow
        window = MainWindow()
        try:
            with patch("gui_app.QMessageBox.critical") as critical:
                window._evaluation_failed("Dataset missing")
            critical.assert_called_once()
            self.assertIn("DATA MODE: ERROR", window.evaluation_dataset_label.text())
            self.assertIn("ERROR", window.eval_status_label.text())
        finally:
            window.close()


if __name__ == "__main__": unittest.main()
