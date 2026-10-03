import os
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class ProcessedDatasetGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def _wait(self, window):
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            self.app.processEvents()
            if not window._stage2_busy and window.preprocessing_thread is None:
                return
            time.sleep(.01)
        self.fail("Stage 2 preprocessing did not finish")

    def test_pass_save_stale_and_saved_record(self):
        from gui_app import MainWindow
        from stage2.dataset_artifacts import load_processed_dataset
        from train_lstm_ae import OUTPUT_DIR
        window = MainWindow(); window.show()
        try:
            window.tabs.setCurrentWidget(window.preprocessing_tab)
            self.assertFalse(window.save_processed_dataset_button.isEnabled())
            window.run_preprocessing_button.click(); self._wait(window)
            self.assertEqual(window.stage2_status_label.text(), "전처리 완료")
            self.assertTrue(window.save_processed_dataset_button.isEnabled())
            window.dataset_id_edit.setText("bad id")
            self.assertFalse(window.save_processed_dataset_button.isEnabled())
            window.dataset_id_edit.setText("gui_save_test")
            self.assertTrue(window.save_processed_dataset_button.isEnabled())
            root = OUTPUT_DIR / ".test_artifacts" / "gui_processed_dataset_tests"
            root.mkdir(parents=True, exist_ok=True)
            with patch("gui_app.save_processed_dataset", side_effect=lambda result, dataset_id: __import__(
                    "stage2.dataset_artifacts", fromlist=["save_processed_dataset"]).save_processed_dataset(
                        result, dataset_id, root)):
                if (root / "gui_save_test").exists():
                    # A prior test run already proved the on-disk artifact; use a fresh ID.
                    window.dataset_id_edit.setText(f"gui_save_{int(time.time())}")
                window.save_processed_dataset_button.click()
            self.assertIn("상태: 저장 완료", window.stage2_saved_dataset_label.text())
            self.assertTrue(load_processed_dataset(window.stage2_saved_dataset.path))
            self.assertIn("처리된 데이터셋 모드", window.log_view.toPlainText())
            with patch("gui_app.save_processed_dataset", side_effect=FileExistsError("이미 존재하는 Dataset ID입니다. 다른 이름을 사용하세요.")), \
                    patch("gui_app.QMessageBox.warning") as warning:
                window.save_processed_dataset_button.click()
            warning.assert_called_once()
            self.assertIn("다른 이름", warning.call_args.args[2])
            window.sequence_combo.setCurrentIndex(window.sequence_combo.findData(15))
            self.assertIn("결과 만료", window.stage2_status_label.text())
            self.assertFalse(window.save_processed_dataset_button.isEnabled())
            self.assertIn("상태: 저장 완료", window.stage2_saved_dataset_label.text())
        finally:
            window.close()


if __name__ == "__main__": unittest.main()
