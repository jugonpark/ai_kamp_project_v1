import csv
import os
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class TrainingGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def _run(self, folder, model_id, losses, best_epoch=2, status="EARLY_STOPPED"):
        from model_artifacts import ModelRun
        folder.mkdir(parents=True, exist_ok=True)
        model_path = folder / "model.keras"; model_path.write_bytes(b"existing model placeholder")
        with (folder / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("epoch", "loss", "val_loss", "learning_rate"))
            writer.writeheader()
            for epoch, (train, valid, lr) in enumerate(losses, start=1):
                writer.writerow({"epoch":epoch, "loss":train, "val_loss":valid, "learning_rate":lr})
        meta = {"model_id":model_id, "display_name":model_id, "experiment_name":folder.name,
                "best_epoch":best_epoch, "best_val_loss":losses[best_epoch-1][1],
                "random_seed":42, "learning_rate":.001, "batch_size":128, "loss":"MSE",
                "noise_std":.005, "completion_status":status}
        return ModelRun(model_id, folder.name, model_path, meta)

    def test_saved_curves_modes_comparison_zoom_and_export(self):
        from training_graph import TrainingGraphTab, load_training_curve
        from model_artifacts import ModelRun
        root = Path.cwd() / "outputs" / ".test_artifacts" / uuid.uuid4().hex
        try:
            a = self._run(root / "cnn", "CNN_LSTM_AUTOENCODER", [(.5,.6,.001),(.4,.45,.0007),(.3,.43,.0007)])
            b = self._run(root / "denoising", "DENOISING_CNN_LSTM_AUTOENCODER", [(.48,.58,.001),(.38,.42,.0007),(.28,.40,.0007)])
            curve = load_training_curve(a)
            self.assertEqual(curve.lr_reductions, [2])
            self.assertEqual((curve.best_epoch, curve.end_epoch), (2, 3))
            self.assertIsNone(load_training_curve(ModelRun(a.model_id, "missing", root / "missing" / "model.keras", {})))
            panel = TrainingGraphTab()
            panel.current_curve = curve
            panel.render(); self.app.processEvents()
            axis = panel.figure.axes[0]
            self.assertEqual(len(axis.lines[0].get_xdata()), 3)
            self.assertTrue(any(line.get_linestyle() == "--" for line in axis.lines))
            for key in ("lr", "gap", "loss_lr", "comparison"):
                panel.graph_type.setCurrentIndex(panel.graph_type.findData(key))
                panel.render()
                self.assertTrue(panel.figure.axes)
            self.assertEqual(len(panel.figure.axes), 1)
            panel._comparison_curve = load_training_curve(b)
            panel.render()
            self.assertIn("RUN B", panel.summary_label.text())
            self.assertIn("-0.03", panel.summary_label.text())
            self.assertTrue(hasattr(panel.toolbar, "zoom"))
            self.assertTrue(hasattr(panel.toolbar, "pan"))
            self.assertTrue(hasattr(panel.toolbar, "home"))
            with patch("training_graph.core.OUTPUT_DIR", root):
                path = panel.save_png()
            self.assertTrue(path.is_file())
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            panel.close()
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_main_window_links_and_current_epoch_updates(self):
        from gui_app import MainWindow
        window = MainWindow(); window.show(); self.app.processEvents()
        self.assertEqual([window.tabs.tabText(i) for i in range(window.tabs.count())],
                         ["실시간 학습", "학습 그래프", "데이터", "평가 및 비교"])
        window.open_graph_button.click()
        self.assertIs(window.tabs.currentWidget(), window.graph_tab)
        self.assertEqual(window.graph_tab.run_type.currentData(), "current")
        window.update_epoch(1, 3, .5, .6, .001, 1., 0, 1, .6, 0)
        window.update_epoch(2, 3, .4, .45, .0007, 2., 0, 2, .45, 1)
        self.assertEqual(window.graph_tab.current_curve.epochs, [1, 2])
        self.assertEqual(window.graph_tab.current_curve.lr_reductions, [2])
        window.graph_tab.graph_type.setCurrentIndex(window.graph_tab.graph_type.findData("gap"))
        self.assertAlmostEqual(float(window.graph_tab.figure.axes[0].lines[0].get_ydata()[0]), .1)
        window.reset_loss_graph()
        self.assertFalse(window.graph_tab.current_curve.epochs)
        self.assertIn("표시할 학습 기록이 없습니다", window.graph_tab.figure.axes[0].texts[0].get_text())
        root = Path.cwd() / "outputs" / ".test_artifacts" / uuid.uuid4().hex
        try:
            run = self._run(root / "cnn", "CNN_LSTM_AUTOENCODER", [(.5,.6,.001),(.4,.45,.001)])
            window.evaluation_run_combo.addItem("test curve", run)
            window.evaluation_run_combo.setCurrentIndex(window.evaluation_run_combo.count() - 1)
            with patch("training_graph.discover_model_runs", side_effect=lambda model_id: [run] if model_id == run.model_id else []):
                window.show_evaluation_graph()
            self.assertEqual(window.graph_tab.run_type.currentData(), "saved")
            self.assertEqual(window.graph_tab._saved_curve.epochs, [1, 2])
        finally:
            shutil.rmtree(root, ignore_errors=True)
        window.close()

    def test_historical_gru_run_can_open_without_training(self):
        from training_graph import TrainingGraphTab
        root = Path.cwd() / "outputs" / ".test_artifacts" / uuid.uuid4().hex
        try:
            run = self._run(root / "gru", "GRU_AUTOENCODER", [(.5,.55,.001),(.4,.45,.001)])
            with patch("training_graph.discover_model_runs", side_effect=lambda model_id: [run] if model_id == "GRU_AUTOENCODER" else []):
                panel = TrainingGraphTab()
                panel.open_saved_run(run)
                self.assertEqual(panel.model_combo.currentData(), "GRU_AUTOENCODER")
                self.assertEqual(panel.run_type.currentData(), "saved")
                self.assertEqual(panel._saved_curve.epochs, [1, 2])
                panel.close()
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
