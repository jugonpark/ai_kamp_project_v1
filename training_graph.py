"""Read-only training curves and a large Qt/Matplotlib graph workspace."""
import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QPushButton, QSplitter, QVBoxLayout, QWidget)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

import train_lstm_ae as core
from model_artifacts import discover_model_runs
from model_registry import MODEL_REGISTRY


@dataclass
class TrainingCurve:
    epochs: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float | None = None
    end_epoch: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def lr_reductions(self):
        return [self.epochs[i] for i in range(1, len(self.epochs))
                if np.isfinite(self.learning_rate[i-1]) and np.isfinite(self.learning_rate[i])
                and self.learning_rate[i] < self.learning_rate[i-1]]


def load_training_curve(run):
    """Return None for missing/invalid histories without changing an artifact."""
    path = run.model_path.parent / "training_history.csv"
    if not path.is_file():
        return None
    curve = TrainingCurve(metadata=run.metadata.copy())
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                curve.epochs.append(int(row["epoch"]))
                curve.train_loss.append(float(row["loss"]))
                curve.val_loss.append(float(row["val_loss"]))
                curve.learning_rate.append(float(row.get("learning_rate") or row.get("lr") or "nan"))
    except (OSError, KeyError, TypeError, ValueError):
        return None
    if not curve.epochs:
        return None
    curve.best_epoch = int(run.metadata.get("best_epoch") or curve.epochs[int(np.argmin(curve.val_loss))])
    if curve.best_epoch not in curve.epochs:
        curve.best_epoch = curve.epochs[int(np.argmin(curve.val_loss))]
    saved_best = run.metadata.get("best_val_loss")
    curve.best_val_loss = float(saved_best) if saved_best is not None else curve.val_loss[curve.epochs.index(curve.best_epoch)]
    if run.metadata.get("completion_status") == "EARLY_STOPPED":
        curve.end_epoch = curve.epochs[-1]
    return curve


class TrainingGraphTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_curve = TrainingCurve()
        self._saved_curve = None
        self._comparison_curve = None
        self._font = FontProperties(fname=r"C:\Windows\Fonts\malgun.ttf") if Path(r"C:\Windows\Fonts\malgun.ttf").is_file() else None
        self._build_ui()
        self.refresh_runs()
        self.render()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        controls = QGroupBox("그래프 대상 및 종류")
        form = QFormLayout(controls)
        self.run_type = QComboBox(); self.run_type.addItem("현재 학습", "current"); self.run_type.addItem("저장된 실행", "saved")
        self.model_combo = QComboBox()
        for spec in MODEL_REGISTRY.values():
            self.model_combo.addItem(spec.display_name, spec.id)
        self.run_combo = QComboBox()
        self.comparison_combo = QComboBox(); self.comparison_combo.addItem("없음", None)
        self.graph_type = QComboBox()
        for label, key in (("학습 / 검증 손실", "loss"), ("학습률 변화", "lr"),
                           ("일반화 차이", "gap"), ("손실 + 학습률", "loss_lr"),
                           ("실험 간 손실 비교", "comparison")):
            self.graph_type.addItem(label, key)
        row = QHBoxLayout()
        for label, combo in (("대상", self.run_type), ("모델", self.model_combo),
                             ("실험 / 실행", self.run_combo), ("비교 대상", self.comparison_combo),
                             ("그래프 종류", self.graph_type)):
            row.addWidget(QLabel(label)); row.addWidget(combo, 1)
        form.addRow(row)
        actions = QHBoxLayout()
        self.refresh_button = QPushButton("새로고침"); self.refresh_button.clicked.connect(self.refresh_runs)
        self.current_button = QPushButton("현재 학습 보기"); self.current_button.clicked.connect(self.show_current)
        self.saved_button = QPushButton("저장된 실행 불러오기"); self.saved_button.clicked.connect(self.show_saved)
        self.compare_button = QPushButton("비교 실행"); self.compare_button.clicked.connect(self.show_comparison)
        self.save_button = QPushButton("그래프 PNG 저장"); self.save_button.clicked.connect(self.save_png)
        self.train_check = QCheckBox("학습 손실 표시"); self.train_check.setChecked(True)
        self.val_check = QCheckBox("검증 손실 표시"); self.val_check.setChecked(True)
        for widget in (self.refresh_button, self.current_button, self.saved_button, self.compare_button,
                       self.save_button, self.train_check, self.val_check):
            actions.addWidget(widget)
        form.addRow(actions)
        layout.addWidget(controls)
        self.splitter = QSplitter(Qt.Vertical)
        plot_area = QWidget(); plot_layout = QVBoxLayout(plot_area)
        self.figure = Figure(figsize=(11, 6), tight_layout=True, facecolor="#f7f9fc")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        plot_layout.addWidget(self.toolbar); plot_layout.addWidget(self.canvas, 1)
        self.splitter.addWidget(plot_area)
        info_area = QWidget(); info_layout = QHBoxLayout(info_area)
        self.summary_label = QLabel(); self.summary_label.setWordWrap(True)
        self.explanation_label = QLabel(); self.explanation_label.setWordWrap(True)
        info_layout.addWidget(self.summary_label, 1); info_layout.addWidget(self.explanation_label, 2)
        self.splitter.addWidget(info_area); self.splitter.setSizes([680, 170])
        layout.addWidget(self.splitter, 1)
        notice = QLabel("학습·검증 손실은 정상 데이터 학습과 일반화 상태를 보여줍니다. 검증 손실이 낮아도 이상 탐지 성능이 반드시 좋은 것은 아닙니다. 최종 비교는 평가 및 비교 탭의 Precision / Recall / F1 / FP / FN과 함께 확인하세요.")
        notice.setWordWrap(True); layout.addWidget(notice)
        self.run_type.currentIndexChanged.connect(self._selection_changed)
        self.model_combo.currentIndexChanged.connect(self._model_changed)
        self.run_combo.currentIndexChanged.connect(self._selection_changed)
        self.comparison_combo.currentIndexChanged.connect(self._selection_changed)
        self.graph_type.currentIndexChanged.connect(self.render)
        self.train_check.toggled.connect(self.render); self.val_check.toggled.connect(self.render)

    def refresh_runs(self):
        selected = self.run_combo.currentData()
        selected_path = str(selected.model_path) if selected else None
        comparison = self.comparison_combo.currentData()
        comparison_path = str(comparison.model_path) if comparison else None
        all_runs = [run for model_id in MODEL_REGISTRY for run in discover_model_runs(model_id)]
        self.comparison_combo.blockSignals(True); self.comparison_combo.clear(); self.comparison_combo.addItem("없음", None)
        for run in all_runs:
            if (run.model_path.parent / "training_history.csv").is_file():
                self.comparison_combo.addItem(self._run_name(run), run)
        self._select_path(self.comparison_combo, comparison_path)
        self.comparison_combo.blockSignals(False)
        self._populate_model_runs(all_runs, selected_path)
        self._selection_changed()

    @staticmethod
    def _select_path(combo, path):
        if path:
            for i in range(combo.count()):
                item = combo.itemData(i)
                if item and str(item.model_path) == path:
                    combo.setCurrentIndex(i); return

    @staticmethod
    def _run_name(run):
        return f"{MODEL_REGISTRY[run.model_id].display_name} · {run.metadata.get('experiment_name') or run.run_id}"

    def _populate_model_runs(self, all_runs=None, selected_path=None):
        if all_runs is None:
            all_runs = [run for run in discover_model_runs(self.model_combo.currentData())]
        selected = self.run_combo.currentData()
        selected_path = selected_path or (str(selected.model_path) if selected else None)
        self.run_combo.blockSignals(True); self.run_combo.clear()
        for run in all_runs:
            if run.model_id == self.model_combo.currentData():
                self.run_combo.addItem(run.metadata.get("experiment_name") or run.run_id, run)
        self._select_path(self.run_combo, selected_path)
        self.run_combo.blockSignals(False)

    def _model_changed(self):
        self._populate_model_runs()
        self._selection_changed()

    def _selection_changed(self):
        saved = self.run_type.currentData() == "saved"
        self.model_combo.setEnabled(saved); self.run_combo.setEnabled(saved)
        self._saved_curve = load_training_curve(self.run_combo.currentData()) if saved and self.run_combo.currentData() else None
        compare_run = self.comparison_combo.currentData()
        self._comparison_curve = load_training_curve(compare_run) if compare_run else None
        self.render()

    def show_current(self):
        self.run_type.setCurrentIndex(self.run_type.findData("current")); self.render()

    def show_saved(self):
        self.run_type.setCurrentIndex(self.run_type.findData("saved")); self._selection_changed()

    def show_comparison(self):
        self.graph_type.setCurrentIndex(self.graph_type.findData("comparison")); self.render()

    def open_saved_run(self, run):
        self.model_combo.setCurrentIndex(self.model_combo.findData(run.model_id))
        self._populate_model_runs(selected_path=str(run.model_path))
        self.show_saved()

    def reset_current(self, metadata=None):
        self.current_curve = TrainingCurve(metadata=metadata or {})
        if self.run_type.currentData() == "current": self.render()

    def add_current_epoch(self, epoch, train_loss, val_loss, learning_rate, best_epoch, best_val_loss):
        curve = self.current_curve
        curve.epochs.append(int(epoch)); curve.train_loss.append(float(train_loss))
        curve.val_loss.append(float(val_loss)); curve.learning_rate.append(float(learning_rate))
        curve.best_epoch = int(best_epoch); curve.best_val_loss = float(best_val_loss)
        if self.isVisible() and self.run_type.currentData() == "current": self.render()

    def finish_current(self, status, max_epochs):
        if status == "COMPLETED" and self.current_curve.epochs and self.current_curve.epochs[-1] < max_epochs:
            self.current_curve.end_epoch = self.current_curve.epochs[-1]
        if self.isVisible() and self.run_type.currentData() == "current": self.render()

    def _active_curve(self):
        return self.current_curve if self.run_type.currentData() == "current" else self._saved_curve

    def _style(self, axis, title, ylabel):
        axis.set_title(title, fontproperties=self._font, fontsize=13)
        axis.set_xlabel("에포크", fontproperties=self._font)
        axis.set_ylabel(ylabel, fontproperties=self._font)
        axis.grid(True, alpha=.25)

    def _draw_loss(self, axis, curve, prefix="", marker=True):
        if self.train_check.isChecked(): axis.plot(curve.epochs, curve.train_loss, label=f"{prefix}학습 손실", color="#1786c1" if not prefix else "#8696ad")
        if self.val_check.isChecked(): axis.plot(curve.epochs, curve.val_loss, label=f"{prefix}검증 손실", color="#e2783b" if not prefix else "#a54cb9")
        if marker and curve.best_epoch in curve.epochs:
            index = curve.epochs.index(curve.best_epoch)
            axis.axvline(curve.best_epoch, color="#269b66", linestyle="--", alpha=.8, label="최적 검증 에포크")
            axis.scatter([curve.best_epoch], [curve.val_loss[index]], color="#269b66", zorder=4)
        if marker and curve.end_epoch:
            axis.axvline(curve.end_epoch, color="#cc405c", linestyle=":", label="조기 종료")
        if marker:
            for epoch in curve.lr_reductions:
                axis.axvline(epoch, color="#99931f", linestyle=":", alpha=.5)
        self._style(axis, "학습 및 검증 손실", "손실 (Loss)")
        if axis.get_legend_handles_labels()[0]:
            axis.legend(prop=self._font, loc="best")

    def render(self, *_):
        self.figure.clear()
        curve = self._active_curve()
        mode = self.graph_type.currentData()
        if not curve or not curve.epochs:
            axis = self.figure.add_subplot(111); axis.axis("off")
            axis.text(.5, .5, "표시할 학습 기록이 없습니다.", ha="center", va="center", transform=axis.transAxes, fontproperties=self._font, fontsize=16)
            self.summary_label.setText("실행 기록 없음")
            self.explanation_label.setText("저장된 실행을 선택하거나 학습을 시작하면 그래프를 표시합니다.")
            self.canvas.draw_idle(); return
        if mode == "loss_lr":
            loss_axis, lr_axis = self.figure.subplots(2, 1, sharex=True)
            self._draw_loss(loss_axis, curve)
            lr_axis.step(curve.epochs, curve.learning_rate, where="post", color="#7254af")
            self._style(lr_axis, "학습률 변화", "학습률")
        else:
            axis = self.figure.add_subplot(111)
            if mode == "lr":
                axis.step(curve.epochs, curve.learning_rate, where="post", color="#7254af", label="학습률")
                for epoch in curve.lr_reductions: axis.axvline(epoch, color="#99931f", linestyle=":", alpha=.6)
                self._style(axis, "학습률 변화", "학습률"); axis.legend(prop=self._font)
            elif mode == "gap":
                gap = np.asarray(curve.val_loss) - np.asarray(curve.train_loss)
                axis.plot(curve.epochs, gap, color="#ba546b", label="일반화 차이")
                axis.axhline(0, color="#777", linewidth=1)
                self._style(axis, "일반화 차이", "검증 손실 - 학습 손실"); axis.legend(prop=self._font)
            elif mode == "comparison":
                self._draw_loss(axis, curve, prefix="A ", marker=False)
                if self._comparison_curve and self._comparison_curve.epochs:
                    self._draw_loss(axis, self._comparison_curve, prefix="B ", marker=False)
                self._style(axis, "실험 간 손실 비교", "손실 (Loss)")
                if axis.get_legend_handles_labels()[0]: axis.legend(prop=self._font)
            else:
                self._draw_loss(axis, curve)
        self.summary_label.setText(self._summary(curve))
        explanations = {
            "loss": "학습↓ 검증↓: 정상 학습 · 학습↓ 검증 정체: 추가 효과 감소 · 학습↓ 검증↑: 과적합 가능성 · 둘 다 높음: 과소적합 가능성. 그래프만으로 모델 실패를 단정하지 마세요.",
            "gap": "일반화 차이 = 검증 손실 − 학습 손실. 지속적으로 커지면 훈련 데이터에 비해 검증 성능이 떨어지는 과적합 신호일 수 있습니다.",
            "lr": "검증 손실 개선이 멈추면 학습률을 낮추어 더 작은 보폭으로 학습합니다.",
            "loss_lr": "위: 학습·검증 손실. 아래: 학습률 변화. 서로 다른 축으로 표시합니다.",
            "comparison": "두 실행의 학습 곡선을 비교합니다. 검증 손실 차이만으로 이상 탐지 성능의 우열을 판단하지 마세요.",
        }
        self.explanation_label.setText(explanations[mode])
        self.canvas.draw_idle()

    def _summary(self, curve):
        meta = curve.metadata
        def describe(title, item):
            m = item.metadata
            return (f"{title}: {m.get('display_name') or m.get('model_id') or '현재 학습'} / "
                    f"{m.get('experiment_name') or '-'}\n시드 {m.get('random_seed', '-')} · 학습률 {m.get('learning_rate', '-')} · "
                    f"배치 {m.get('batch_size', '-')} · 손실 {m.get('loss', '-')} · 노이즈 {m.get('noise_std', '-')}\n"
                    f"최적 에포크 {item.best_epoch or '-'} · 최적 검증 손실 {item.best_val_loss if item.best_val_loss is not None else '-'}")
        result = describe("RUN A", curve)
        if self.graph_type.currentData() == "comparison" and self._comparison_curve:
            result += "\n" + describe("RUN B", self._comparison_curve)
            if curve.best_val_loss is not None and self._comparison_curve.best_val_loss is not None:
                result += f"\n최적 검증 손실 차이 (B - A): {self._comparison_curve.best_val_loss - curve.best_val_loss:+.6g}"
        return result

    def save_png(self):
        names = {"loss":"loss_curve", "lr":"learning_rate", "gap":"generalization_gap",
                 "loss_lr":"loss_and_learning_rate", "comparison":"comparison"}
        output = core.OUTPUT_DIR / "graphs"
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"{names[self.graph_type.currentData()]}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        self.figure.savefig(path, dpi=160, facecolor=self.figure.get_facecolor())
        self.summary_label.setText(self.summary_label.text() + f"\n그래프 저장: {path}")
        return path
