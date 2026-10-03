import os
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class Stage2GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def _wait_for_stage2(self, window):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            self.app.processEvents()
            if not window._stage2_busy and window.preprocessing_thread is None:
                return
            time.sleep(.01)
        self.fail("Stage 2 worker did not finish")

    def test_analyze_preview_stale_and_error_flow(self):
        from gui_app import MainWindow
        window = MainWindow(); window.show()
        try:
            window.tabs.setCurrentWidget(window.preprocessing_tab)
            self.assertTrue(window.normal_csv_edit.text().endswith("press_data_normal.csv"))
            self.assertTrue(window.anomaly_csv_edit.text().endswith("outlier_data.csv"))
            window.analyze_data_button.click(); self._wait_for_stage2(window)
            self.assertEqual(window.normal_segments_label.text(), "599")
            self.assertEqual(window.anomaly_segments_label.text(), "21")
            self.assertIn("정리 후 행 수: 19999", window.stage2_analysis_text.toPlainText())
            self.assertIn("정리 후 행 수: 600", window.stage2_analysis_text.toPlainText())
            window.run_preprocessing_button.click(); self._wait_for_stage2(window)
            self.assertEqual(window.stage2_status_label.text(), "전처리 완료")
            self.assertEqual(window.stage2_preprocessing_result.summary["cross_gap_windows"], 0)
            self.assertIn("0 — 통과", window.stage2_cross_gap_label.text())
            self.assertIn("9978 / 276", window.stage2_total_windows_label.text())
            self.assertIn("전체 정상 윈도우 수: 9978", window.stage2_preview_text.toPlainText())
            self.assertIn("전체 이상 윈도우 수: 276", window.stage2_preview_text.toPlainText())
            self.assertIn("[Stage 2 Preprocessing]", window.dataset_text.text())
            window.sequence_combo.setCurrentIndex(window.sequence_combo.findData(15))
            self.assertIsNone(window.stage2_preprocessing_result)
            self.assertIn("결과 만료", window.stage2_status_label.text())
            self.assertEqual(window.normal_segments_label.text(), "599")
            window.run_preprocessing_button.click(); self._wait_for_stage2(window)
            self.assertIn("전체 정상 윈도우 수: 12320", window.stage2_preview_text.toPlainText())
            self.assertIn("전체 이상 윈도우 수: 347", window.stage2_preview_text.toPlainText())
            self.assertEqual(window.stage2_preprocessing_config_snapshot["sequence_length"], 15)
            self.assertIn("12320 / 347", window.stage2_total_windows_label.text())
            window.gap_threshold_spin.setValue(200)
            self.assertIsNone(window.stage2_preprocessing_result)
            self.assertEqual(window.normal_segments_label.text(), "-")
            window.normal_csv_edit.setText(str(window.normal_csv_edit.text()) + ".missing")
            self.assertIn("결과 만료", window.stage2_status_label.text())
            with patch("gui_app.QMessageBox.warning") as warning:
                window.analyze_data_button.click(); self._wait_for_stage2(window)
            self.assertEqual(window.stage2_status_label.text(), "오류")
            self.assertIsNone(window.stage2_preprocessing_result)
            warning.assert_called_once()
        finally:
            window.close()

    def test_selected_signal_and_scaler_reach_backend(self):
        from gui_app import MainWindow
        from preprocessing_config import default_preprocessing_config
        from stage2.preprocessing import run_stage2_preprocessing
        window = MainWindow()
        try:
            window.signal_transform_combo.setCurrentIndex(window.signal_transform_combo.findData("RAW_SIGNED"))
            window.scaler_combo.setCurrentIndex(window.scaler_combo.findData("STANDARD"))
            selected = window._stage2_snapshot()
            self.assertEqual(selected["config"]["signal_transform"], "RAW_SIGNED")
            self.assertEqual(selected["config"]["scaler"], "STANDARD")
            self.assertEqual(selected["config"]["sequence_length"], default_preprocessing_config()["sequence_length"])
            with patch("preprocessing_worker.run_stage2_preprocessing", wraps=run_stage2_preprocessing) as backend:
                window.run_preprocessing_button.click(); self._wait_for_stage2(window)
            backend.assert_called_once_with(selected["normal_path"], selected["anomaly_path"], selected["config"])
            self.assertIn("신호 변환: 원본 부호 유지", window.stage2_preview_text.toPlainText())
            self.assertIn("스케일링: 표준화", window.stage2_preview_text.toPlainText())
        finally:
            window.close()
