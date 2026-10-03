import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class PreprocessingGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_display_tracks_configuration_without_changing_runtime(self):
        from gui_app import MainWindow
        window = MainWindow()
        try:
            self.assertIn("데이터 전처리", [window.tabs.tabText(i) for i in range(window.tabs.count())])
            self.assertEqual(window.system_labels["Input Shape"].text(), "(20, 3)")
            self.assertEqual(window.system_labels["Horizon"].text(), "OFF")
            self.assertEqual(window.evaluation_gap_label.text(), "150 ms")
            window.sequence_combo.setCurrentIndex(window.sequence_combo.findData(15))
            window.gap_threshold_spin.setValue(175)
            self.assertEqual(window.system_labels["Input Shape"].text(), "(15, 3)")
            self.assertEqual(window.evaluation_gap_label.text(), "175 ms")
            self.assertIn("RepeatVector15", window.architecture_label.text())
            self.assertEqual(window.preprocessing_config["sequence_length"], 15)
            window._set_training_locked(True)
            self.assertFalse(window.sequence_combo.isEnabled())
            self.assertFalse(window.save_preprocessing_button.isEnabled())
            window._set_training_locked(False)
            window.preprocessing_mode_combo.setCurrentIndex(window.preprocessing_mode_combo.findData("KAMP_BASELINE"))
            self.assertEqual(window.system_labels["Horizon"].text(), "100")
            self.assertFalse(window.sequence_combo.isEnabled())
            self.assertFalse(window.preprocessing_config["remove_exact_duplicates"])
        finally:
            window.close()
