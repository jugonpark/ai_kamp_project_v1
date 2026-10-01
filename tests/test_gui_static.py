import inspect
import os
import unittest
from unittest.mock import patch
from pathlib import Path
import shutil
import uuid

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class EvaluationGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_default_selectors_preserve_kamp_baseline(self):
        from gui_app import MainWindow
        window = MainWindow()
        self.assertEqual(window.score_combo.currentData(), "LAST_STEP_MSE")
        self.assertEqual(window.threshold_combo.currentData(), "PR_INTERSECTION")
        self.assertEqual([window.training_model_combo.itemData(i) for i in range(window.training_model_combo.count())], ["CNN_LSTM_AUTOENCODER", "DENOISING_CNN_LSTM_AUTOENCODER"])
        self.assertEqual(window.evaluation_model_combo.currentData(), "KAMP_LSTM_AE")
        window.close()

    def test_loss_graph_can_be_reset(self):
        from gui_app import MainWindow
        window = MainWindow()
        window.loss_canvas.update_data(1, 0.2, 0.3)
        window.reset_loss_graph()
        self.assertEqual(len(window.loss_canvas.train_line.get_xdata()), 0)
        self.assertEqual(len(window.loss_canvas.val_line.get_xdata()), 0)
        window.close()

    def test_full_training_controls_and_config_mapping_exist(self):
        from gui_app import MainWindow
        window = MainWindow(); config = window._training_config_dict()
        required = {"epochs","batch_size","learning_rate","optimizer","weight_decay","loss","huber_delta",
                    "random_seed","reduce_lr_enabled","reduce_lr_factor","reduce_lr_patience","min_lr",
                    "early_stopping_enabled","early_stopping_patience","early_stopping_min_delta",
                    "restore_best_weights","noise_type","noise_mean","noise_std","noise_clip"}
        self.assertTrue(required.issubset(config))
        self.assertEqual(window.comparison_table.columnCount(), 25)
        self.assertEqual(set(window.history_filters), {"Model","Status","Score","Threshold","Loss","Seed"})
        window.close()

    def test_manual_edit_changes_preset_to_custom_and_reset_restores_baseline(self):
        from gui_app import MainWindow
        window = MainWindow(); window.apply_preset("CNN-LSTM BASELINE"); window.preset_combo.setCurrentIndex(window.preset_combo.findData("CNN-LSTM BASELINE"))
        window.learning_rate_spin.setValue(.0008)
        self.assertEqual(window.preset_combo.currentData(), "CUSTOM")
        window.reset_training_config()
        self.assertEqual(window.preset_combo.currentData(), "CNN-LSTM BASELINE")
        self.assertEqual(window.learning_rate_spin.value(), .001)
        window.close()

    def test_evaluation_worker_never_calls_model_fit(self):
        import evaluation_worker, evaluation_controller
        source = inspect.getsource(evaluation_worker) + inspect.getsource(evaluation_controller)
        self.assertNotIn(".fit(", source)
        self.assertNotIn("train_model", source)
        self.assertEqual(inspect.getsource(evaluation_controller).count("model.predict("), 2)

    def test_compare_all_marks_status_completed(self):
        from gui_app import MainWindow
        window = MainWindow()
        result = {"score_method":"LAST_STEP_MSE", "score_name":"KAMP - Last Step MSE",
                  "threshold_method":"PR_INTERSECTION", "threshold_name":"KAMP - PR Intersection",
                  "threshold":.1, "accuracy":1., "balanced_accuracy":1., "precision":1., "recall":1.,
                  "f1":1., "specificity":1., "fpr":0., "fnr":0., "tn":2, "fp":0, "fn":0, "tp":2,
                  "fallback_used":False, "fallback_reason":""}
        window._comparison_results([result])
        self.assertIn("상태: 완료", window.eval_status_label.text())
        window.close()

    def test_duplicate_history_is_rejected_and_manual_baseline_champion_work(self):
        from gui_app import MainWindow
        window = MainWindow()
        result = {"model":"CNN", "model_id":"CNN_LSTM_AUTOENCODER", "experiment_name":"E1", "status":"ACTIVE",
                  "score_method":"MAHALANOBIS_ERROR", "score_name":"Mahalanobis", "threshold_method":"POT_1PCT",
                  "threshold_name":"POT 1%", "threshold":.1, "accuracy":.9, "balanced_accuracy":.9,
                  "precision":.9, "recall":.8, "f1":.85, "tn":9, "fp":1, "fn":2, "tp":8, "created_at":"now"}
        window._append_comparison(result); window._append_comparison(result)
        self.assertEqual(len(window.comparison_history), 1)
        window.set_current_as_baseline(); window.set_current_as_champion()
        self.assertEqual(window.baseline_result["experiment_name"], "E1")
        self.assertEqual(window.champion_result["experiment_name"], "E1")
        window.history_filters["Model"].setCurrentText("CNN"); window.apply_history_filters()
        self.assertFalse(window.comparison_table.isRowHidden(0))
        window.close()

    def test_training_controls_are_visible_and_scrollable_at_small_size(self):
        from gui_app import MainWindow
        from PySide6.QtWidgets import QScrollArea, QFormLayout
        window = MainWindow(); window.resize(1200, 700); window.show(); self.app.processEvents()
        self.assertIsInstance(window.training_scroll, QScrollArea)
        self.assertTrue(window.training_scroll.widgetResizable())
        self.assertGreater(window.training_scroll.verticalScrollBar().maximum(), 0)
        for widget in (window.epochs_spin, window.batch_spin, window.learning_rate_spin,
                       window.optimizer_combo, window.loss_combo, window.seed_spin, window.experiment_name_edit):
            self.assertIsNotNone(widget.parentWidget())
            self.assertIsInstance(widget.parentWidget().layout(), QFormLayout)
            self.assertTrue(widget.isVisible())
            self.assertTrue(widget.isEnabled())
        self.assertTrue(window.start_button.isVisible())
        self.assertFalse(window.denoising_box.isVisible())
        window.training_model_combo.setCurrentIndex(window.training_model_combo.findData("DENOISING_CNN_LSTM_AUTOENCODER"))
        self.app.processEvents()
        self.assertTrue(window.denoising_box.isVisible())
        self.assertTrue(window.noise_std_spin.isVisible())
        self.assertTrue(window.noise_std_spin.isEnabled())
        self.assertIn("Denoising CNN-LSTM AutoEncoder", window.status_label.text())
        window.training_scroll.ensureWidgetVisible(window.noise_std_spin)
        self.app.processEvents()
        self.assertGreater(window.training_scroll.verticalScrollBar().value(), 0)
        window.close()

    def test_dependent_training_controls_and_guide(self):
        from gui_app import MainWindow
        window = MainWindow(); window.show(); self.app.processEvents()
        self.assertFalse(window.weight_decay_spin.isEnabled())
        self.assertFalse(window.huber_delta_spin.isEnabled())
        window.optimizer_combo.setCurrentText("AdamW"); window.loss_combo.setCurrentText("Huber")
        self.assertTrue(window.weight_decay_spin.isEnabled())
        self.assertTrue(window.huber_delta_spin.isEnabled())
        window.reduce_lr_enabled.setChecked(False); window.early_stopping_enabled.setChecked(False)
        self.assertFalse(window.reduce_lr_factor_spin.isEnabled())
        self.assertFalse(window.early_stopping_patience_spin.isEnabled())
        window.show_parameter_guide("Learning Rate")
        self.assertIn("현재 값", window.parameter_guide.text())
        self.assertIn("권장 범위", window.parameter_guide.text())
        window.close()

    def test_training_lock_prevents_edits_but_keeps_help_available(self):
        from gui_app import MainWindow
        window = MainWindow(); window.show(); self.app.processEvents()
        window._set_training_locked(True)
        for widget in (window.training_model_combo, window.preset_combo, window.experiment_name_edit,
                       window.epochs_spin, window.batch_spin, window.learning_rate_spin,
                       window.optimizer_combo, window.loss_combo, window.seed_spin,
                       window.reduce_lr_factor_spin, window.noise_std_spin,
                       window.reset_config_button, window.load_config_button, window.run_mode):
            self.assertFalse(widget.isEnabled(), widget.objectName())
        self.assertTrue(window.stop_button.isEnabled())
        self.assertTrue(window.parameter_help_buttons["Learning Rate"].isEnabled())
        window._set_training_locked(False)
        self.assertTrue(window.epochs_spin.isEnabled())
        self.assertTrue(window.start_button.isEnabled())
        self.assertFalse(window.stop_button.isEnabled())
        window.close()

    def test_wheel_does_not_change_numeric_or_combo_value(self):
        from gui_app import MainWindow
        from PySide6.QtCore import QPoint, QPointF, Qt
        from PySide6.QtGui import QWheelEvent
        from PySide6.QtWidgets import QAbstractSpinBox
        window = MainWindow(); window.show(); self.app.processEvents()
        wheel = QWheelEvent(QPointF(10, 10), QPointF(10, 10), QPoint(0, 0), QPoint(0, 120),
                            Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False)
        before = window.batch_spin.value()
        self.app.sendEvent(window.batch_spin, wheel)
        self.assertEqual(window.batch_spin.value(), before)
        before_model = window.training_model_combo.currentIndex()
        self.app.sendEvent(window.training_model_combo, QWheelEvent(QPointF(10, 10), QPointF(10, 10), QPoint(0, 0), QPoint(0, 120), Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False))
        self.assertEqual(window.training_model_combo.currentIndex(), before_model)
        self.assertEqual(window.batch_spin.buttonSymbols(), QAbstractSpinBox.NoButtons)
        window.close()

    def test_every_training_parameter_has_help_and_launcher_exists(self):
        from gui_app import MainWindow
        window = MainWindow()
        self.assertEqual(len(window.parameter_help_buttons), 21)
        window.show_parameter_guide("Learning Rate")
        self.assertIn("현재 값: 0.001", window.parameter_guide.text())
        self.assertTrue((Path.cwd() / "run_gui.bat").is_file())
        window.close()

    def test_preset_and_config_load_update_actual_widgets(self):
        from gui_app import MainWindow
        from training_config import save_training_config
        window = MainWindow(); window.show(); self.app.processEvents()
        window.preset_combo.setCurrentIndex(window.preset_combo.findData("DENOISING DEFAULT"))
        self.assertEqual(window.training_model_combo.currentData(), "DENOISING_CNN_LSTM_AUTOENCODER")
        self.assertEqual((window.epochs_spin.value(), window.batch_spin.value(), window.learning_rate_spin.value()), (800, 128, .001))
        self.assertEqual((window.optimizer_combo.currentText(), window.loss_combo.currentText(), window.noise_mean_spin.value(), window.noise_std_spin.value()), ("Adam", "MSE", 0, .01))
        window.learning_rate_spin.setValue(.0005); window.batch_spin.setValue(64)
        window.loss_combo.setCurrentText("Huber"); window.noise_std_spin.setValue(.02)
        self.assertEqual(window.preset_combo.currentData(), "CUSTOM")
        self.assertEqual((window._training_config_dict()["learning_rate"], window._training_config_dict()["batch_size"], window._training_config_dict()["loss"], window._training_config_dict()["noise_std"]), (.0005, 64, "Huber", .02))
        folder = Path.cwd() / "outputs" / ".test_artifacts" / uuid.uuid4().hex
        try:
            path = save_training_config(folder / "config.json", window._training_config_dict())
            window.reset_training_config()
            with patch("gui_app.QFileDialog.getOpenFileName", return_value=(str(path), "JSON (*.json)")):
                window.load_training_config()
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        self.assertEqual(window.learning_rate_spin.value(), .0005)
        self.assertEqual(window.batch_spin.value(), 64)
        self.assertEqual(window.loss_combo.currentText(), "Huber")
        self.assertEqual(window.noise_std_spin.value(), .02)
        window.close()

    def test_korean_display_keeps_internal_ids_and_english_config(self):
        from gui_app import MainWindow
        from anomaly_scoring import SCORE_METHODS
        from threshold_methods import THRESHOLD_METHODS
        from training_config import save_training_config
        window = MainWindow(); window.show(); self.app.processEvents()
        self.assertEqual(window.preset_combo.itemText(0), "CNN-LSTM 기준")
        self.assertEqual(window.preset_combo.itemData(0), "CNN-LSTM BASELINE")
        self.assertEqual(window.score_combo.itemText(0), SCORE_METHODS["LAST_STEP_MSE"].display_name)
        self.assertEqual(window.score_combo.itemData(0), "LAST_STEP_MSE")
        self.assertEqual(window.threshold_combo.itemText(0), THRESHOLD_METHODS["PR_INTERSECTION"].display_name)
        self.assertEqual(window.threshold_combo.itemData(0), "PR_INTERSECTION")
        self.assertIn("정밀도 (Precision)", [label.text() for label in window.findChildren(type(window.status_label))])
        folder = Path.cwd() / "outputs" / ".test_artifacts" / uuid.uuid4().hex
        try:
            config = window._training_config_dict()
            config.update(preset="DENOISING DEFAULT", model_id="DENOISING_CNN_LSTM_AUTOENCODER", learning_rate=.0005)
            path = save_training_config(folder / "legacy.json", config)
            with patch("gui_app.QFileDialog.getOpenFileName", return_value=(str(path), "JSON (*.json)")):
                window.load_training_config()
            self.assertEqual(window.preset_combo.currentData(), "DENOISING DEFAULT")
            self.assertEqual(window.training_model_combo.currentData(), "DENOISING_CNN_LSTM_AUTOENCODER")
            self.assertEqual(window._training_config_dict()["learning_rate"], .0005)
        finally:
            shutil.rmtree(folder, ignore_errors=True)
            window.close()

    def test_boolean_controls_and_conditional_rows(self):
        from gui_app import MainWindow
        window = MainWindow(); window.show(); self.app.processEvents()
        toggles = (window.reduce_lr_enabled, window.early_stopping_enabled, window.restore_best_weights, window.noise_clip_check)
        self.assertTrue(all(toggle.text() == "ON" for toggle in toggles))
        window.reduce_lr_enabled.setChecked(False)
        self.assertEqual(window.reduce_lr_enabled.text(), "OFF")
        self.assertTrue(all(not widget.isEnabled() for widget in (window.reduce_lr_factor_spin, window.reduce_lr_patience_spin, window.min_lr_spin)))
        window.early_stopping_enabled.setChecked(False)
        self.assertTrue(all(not widget.isEnabled() for widget in (window.early_stopping_patience_spin, window.early_stopping_min_delta_spin, window.restore_best_weights)))
        self.assertFalse(window.weight_decay_spin.isVisible())
        self.assertFalse(window.huber_delta_spin.isVisible())
        window.optimizer_combo.setCurrentText("AdamW"); window.loss_combo.setCurrentText("Huber")
        self.app.processEvents()
        self.assertTrue(window.weight_decay_spin.isVisible())
        self.assertTrue(window.huber_delta_spin.isVisible())
        self.assertFalse(window.denoising_box.isVisible())
        window.training_model_combo.setCurrentIndex(window.training_model_combo.findData("DENOISING_CNN_LSTM_AUTOENCODER"))
        self.app.processEvents()
        self.assertTrue(window.denoising_box.isVisible())
        window.noise_clip_check.setChecked(False)
        self.assertEqual(window.noise_clip_check.text(), "OFF")
        self.assertFalse(window._training_config_dict()["noise_clip"])
        window.show_parameter_guide("Noise Clip")
        self.assertIn("노이즈 적용 후 범위 제한", window.parameter_guide.text())
        self.assertIn("추천 실험값", window.parameter_guide.text())
        window.close()

    def test_weak_denoising_preset_reaches_training_config(self):
        from gui_app import MainWindow
        from model_data import prepare_fit_data
        import numpy as np
        window = MainWindow(); window.show(); self.app.processEvents()
        window.preset_combo.setCurrentIndex(window.preset_combo.findData("DENOISING WEAK 0.005"))
        config = window._training_config_dict()
        self.assertEqual(config["model_id"], "DENOISING_CNN_LSTM_AUTOENCODER")
        self.assertEqual(config["preset"], "DENOISING WEAK 0.005")
        self.assertEqual(config["noise_mean"], 0.0)
        self.assertEqual(config["noise_std"], .005)
        self.assertTrue(config["noise_clip"])
        self.assertEqual((config["epochs"], config["batch_size"], config["learning_rate"]), (800, 128, .001))
        self.assertIn("매번 새로운 Gaussian 노이즈", window.training_model_description.text())
        window.show_parameter_guide("Noise Std")
        self.assertIn("0.005", window.parameter_guide.text())
        clean = np.full((2, 20, 3), .5, dtype=np.float32)
        dataset, _, _, _ = prepare_fit_data(clean, clean, clean, clean, denoising=True,
            mean=config["noise_mean"], std=config["noise_std"], clip=config["noise_clip"],
            seed=config["random_seed"], batch_size=config["batch_size"])
        noisy = next(iter(dataset))[0].numpy()
        self.assertGreater(float(np.std(noisy - clean)), .003)
        self.assertLess(float(np.std(noisy - clean)), .007)
        window.close()


if __name__ == "__main__": unittest.main()
