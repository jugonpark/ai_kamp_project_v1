import csv
import os
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class Stage2DetectionGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_segment_panel_comparison_csv_and_legacy_import(self):
        from gui_app import MainWindow
        window = MainWindow()
        try:
            result = {"model_id": "KAMP_LSTM_AE", "model": "KAMP LSTM AutoEncoder",
                      "score_method": "LAST_STEP_MSE", "threshold_method": "PR_INTERSECTION",
                      "threshold": .5, "accuracy": .5, "balanced_accuracy": .5,
                      "precision": .5, "recall": .5, "f1": .5,
                      "specificity": .5, "fpr": .5, "fnr": .5,
                      "tn": 1, "fp": 1, "fn": 1, "tp": 1,
                      "fallback_used": False, "fallback_reason": "",
                      "data_mode": "PROCESSED_DATASET", "dataset_id": "synthetic",
                      "sequence_length": 20, "temporal_method": "EWMA", "ewma_alpha": .4,
                      "anomaly_segments_total": 3, "evaluable_anomaly_segments": 2,
                      "non_evaluable_anomaly_segments": 1, "detected_segments": 1,
                      "missed_segments": 1, "segment_detection_rate": .5,
                      "median_segment_delay_seconds": 2.2}
            window._evaluation_result_without_history(result)
            self.assertEqual(window.segment_result_labels["segment_detection_rate"].text(), "0.5")
            window._append_comparison(result)
            headers = [window.comparison_table.horizontalHeaderItem(i).text()
                       for i in range(window.comparison_table.columnCount())]
            self.assertIn("Segment Detection Rate", headers)
            self.assertIn("Median Segment-relative Delay (seconds)", headers)
            self.assertEqual(window.comparison_table.item(0, headers.index("Sequence")).text(), "20")
            from train_lstm_ae import OUTPUT_DIR
            folder = OUTPUT_DIR / ".test_artifacts" / f"gui_segment_detection_{uuid.uuid4().hex}"
            folder.mkdir(parents=True)
            try:
                with patch("gui_app.core.OUTPUT_DIR", folder):
                    window.export_comparison()
                exported = next((folder / "gui_comparisons").glob("*.csv"))
                with exported.open(newline="", encoding="utf-8-sig") as handle:
                    row = next(csv.DictReader(handle))
                self.assertEqual(row["segment_detection_rate"], "0.5")
                self.assertEqual(row["median_segment_delay_seconds"], "2.2")
                legacy = folder / "legacy.csv"
                legacy.write_text("model,score_method,threshold_method\nold,LAST_STEP_MSE,PR_INTERSECTION\n", encoding="utf-8")
                with patch("gui_app.QFileDialog.getOpenFileName", return_value=(str(legacy), "")):
                    window.import_historical_result()
                self.assertEqual(window.comparison_history[-1]["segment_detection_rate"], "")
            finally:
                shutil.rmtree(folder)
            window._evaluation_result_without_history({**result, "data_mode": "KAMP_BASELINE",
                **{key: "" for key in ("segment_detection_rate", "median_segment_delay_seconds")}})
            self.assertEqual(window.segment_result_labels["segment_detection_rate"].text(), "-")
        finally:
            window.close()


if __name__ == "__main__": unittest.main()
