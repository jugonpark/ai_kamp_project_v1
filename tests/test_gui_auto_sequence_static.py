import csv
import os
import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class GuiAutoSequenceStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_pool_selection_and_coverage_export(self):
        from PySide6.QtCore import Qt
        from gui_app import MainWindow
        from train_lstm_ae import OUTPUT_DIR
        window = MainWindow()
        root = OUTPUT_DIR / ".test_artifacts" / f"gui_auto_{uuid.uuid4().hex}"
        try:
            model = "CNN_LSTM_AUTOENCODER"
            runs = [SimpleNamespace(model_id=model, run_id=f"run{seq}", model_path=Path(f"seq{seq}.keras"),
                    metadata={"data_mode": "PROCESSED_DATASET", "sequence_length": seq,
                              "dataset_id": f"seq{seq}"}) for seq in (5, 10)]
            runs.append(SimpleNamespace(model_id=model, run_id="baseline", model_path=Path("base.keras"),
                                        metadata={"sequence_length": 20}))
            window.evaluation_model_combo.setCurrentIndex(window.evaluation_model_combo.findData(model))
            with patch("gui_app.discover_model_runs", return_value=runs):
                window.refresh_evaluation_runs()
            self.assertEqual(window.auto_pool_list.count(), 2)
            for index in range(2):
                window.auto_pool_list.item(index).setSelected(True)
            with patch.object(window, "_start_evaluation_worker") as start:
                window.run_auto_evaluation()
            self.assertEqual(start.call_args.args, ("auto_sequence",))
            self.assertEqual({run.run_id for run in start.call_args.kwargs["auto_runs"]}, {"run5", "run10"})
            self.assertEqual(window.auto_pool_list.item(0).data(Qt.UserRole).run_id, "run5")

            result = {"model_id": model, "model": "AUTO CNN", "available_sequences": [10, 5],
                      "signal_transform": "RAW_SIGNED", "scaler": "MINMAX", "gap_threshold_ms": 150,
                      "sequence_length": "AUTO", "test_windows": 6,
                      "score_method": "LAST_STEP_MSE", "threshold_method": "PR_INTERSECTION",
                      "threshold": None, "accuracy": .9, "balanced_accuracy": .9,
                      "precision": .9, "recall": .9, "f1": .9, "specificity": .9,
                      "fpr": .1, "fnr": .1, "tn": 2, "fp": 1, "fn": 1, "tp": 2,
                      "fallback_used": False, "fallback_reason": "",
                      "anomaly_segments_total": 11, "evaluable_anomaly_segments": 10,
                      "non_evaluable_anomaly_segments": 1, "detected_segments": 9,
                      "missed_segments": 1, "segment_coverage": 10 / 11,
                      "segment_detection_rate": .9,
                      "auto_pool": '{"5": "run5", "10": "run10"}',
                      "auto_tier_thresholds": '{"5": 0.05, "10": 0.1}',
                      "auto_tier_methods": '{"5": "PR_INTERSECTION", "10": "PR_INTERSECTION"}'}
            window._auto_evaluation_result(result)
            self.assertEqual(window.eval_result_labels["Threshold"].text(), "시퀀스별")
            self.assertIn("AUTO Sequence pool", window.evaluation_dataset_label.text())
            self.assertEqual(window.segment_result_labels["segment_coverage"].text(), "90.91%")
            self.assertEqual(window.segment_result_labels["segment_detection_rate"].text(), "0.9")
            self.assertEqual(window.comparison_history[-1]["status"], "AUTO")
            window._append_comparison({**result, "auto_pool": '{"5": "other5", "10": "other10"}',
                                       "status": "AUTO"})
            self.assertEqual(len(window.comparison_history), 2)
            with patch("gui_app.core.OUTPUT_DIR", root):
                window.export_comparison()
            exported = next((root / "gui_comparisons").glob("*.csv"))
            with exported.open(newline="", encoding="utf-8-sig") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["segment_coverage"], str(10 / 11))
            self.assertEqual(row["auto_tier_thresholds"], result["auto_tier_thresholds"])
            self.assertEqual(row["auto_tier_methods"], result["auto_tier_methods"])
        finally:
            window.close()
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__": unittest.main()
