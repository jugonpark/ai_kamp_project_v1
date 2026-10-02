"""GUI temporal selection and worker forwarding without loading model weights."""
import os
import unittest
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class TemporalGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_temporal_controls_and_pipeline(self):
        from gui_app import MainWindow
        window = MainWindow()
        try:
            self.assertEqual(window.temporal_combo.currentData(), "NONE")
            self.assertFalse(window.ewma_alpha_spin.isEnabled())
            window.temporal_combo.setCurrentIndex(window.temporal_combo.findData("EWMA"))
            self.assertTrue(window.ewma_alpha_spin.isEnabled())
            self.assertEqual(window.ewma_alpha_spin.value(), 0.4)
            self.assertIn("EWMA α=0.40", window.eval_status_label.text())
            window.timestamp_aware_check.setChecked(False)
            self.assertIn("Timestamp Reset OFF", window.eval_status_label.text())
            self.assertIn("원본 센서가 아닌", window.algorithm_description.text())
        finally:
            window.close()

    def test_fingerprint_distinguishes_temporal_settings(self):
        from gui_app import MainWindow
        window = MainWindow()
        try:
            base = {"model_id": "KAMP_LSTM_AE", "experiment_name": "test", "score_method": "LAST_STEP_MSE",
                    "threshold_method": "PR_INTERSECTION", "threshold": 0.1, "created_at": "now"}
            window._append_comparison(base)
            window._append_comparison({**base, "temporal_method": "EWMA", "ewma_alpha": 0.4})
            window._append_comparison({**base, "temporal_method": "EWMA", "ewma_alpha": 0.2})
            self.assertEqual(len(window.comparison_history), 3)
            self.assertIsNone(window.champion_result)
        finally:
            window.close()

    def test_historical_missing_temporal_columns_and_csv_export(self):
        import csv
        import shutil
        import uuid
        from pathlib import Path
        from unittest.mock import patch
        from gui_app import MainWindow
        window = MainWindow()
        folder = Path.cwd() / "outputs" / ".test_artifacts" / uuid.uuid4().hex
        try:
            folder.mkdir(parents=True)
            path = folder / "old.csv"
            path.write_text("model,score_method,threshold_method,fp,fn\nold,LAST_STEP_MSE,PR_INTERSECTION,2,3\n", encoding="utf-8")
            with patch("gui_app.QFileDialog.getOpenFileName", return_value=(str(path), "")):
                window.import_historical_result()
            self.assertEqual(window.comparison_history[0]["temporal_method"], "NONE")
            self.assertEqual(window.comparison_history[0]["ewma_alpha"], "")
            with patch("gui_app.core.OUTPUT_DIR", folder):
                window.export_comparison()
            exported = next((folder / "gui_comparisons").glob("*.csv"))
            with exported.open(encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["temporal_method"], "NONE")
            self.assertTrue({"ewma_alpha", "timestamp_aware", "total_error"}.issubset(row))
        finally:
            shutil.rmtree(folder, ignore_errors=True)
            window.close()


class EvaluationWorkerTests(unittest.TestCase):
    def test_forwards_temporal_settings(self):
        from evaluation_worker import EvaluationWorker
        controller = Mock()
        controller.evaluate.return_value = {}
        controller.compare_all.return_value = []
        controller.sweep_ewma_alphas.return_value = []
        EvaluationWorker(controller, "evaluate", "LAST_STEP_MSE", "POT_1PCT", "EWMA", 0.2, False).run()
        controller.evaluate.assert_called_once_with("LAST_STEP_MSE", "POT_1PCT", "EWMA", 0.2, False)
        EvaluationWorker(controller, "compare_all", "LAST_STEP_MSE", "POT_1PCT", "EWMA", 0.2, False).run()
        args = controller.compare_all.call_args.args
        self.assertTrue(callable(args[0]))
        self.assertEqual(args[1:], ("EWMA", 0.2, False))
        EvaluationWorker(controller, "sweep_ewma", "LAST_STEP_MSE", "POT_1PCT", "EWMA", 0.2, False).run()
        args = controller.sweep_ewma_alphas.call_args.args
        self.assertEqual(args[:3], ("LAST_STEP_MSE", "POT_1PCT", False))
