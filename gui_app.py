"""PySide6 desktop monitor for the KAMP LSTM-AutoEncoder."""
import os
import platform
import sys
import csv
import json
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, Qt, QEvent, Slot
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
    QPlainTextEdit, QProgressBar, QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QSpinBox, QDoubleSpinBox, QLineEdit, QFileDialog,
    QScrollArea, QSplitter, QSizePolicy, QAbstractSpinBox)
import tensorflow as tf

import training_engine
import train_lstm_ae as core
from anomaly_scoring import SCORE_METHODS
from threshold_methods import THRESHOLD_METHODS
from evaluation_controller import EvaluationController
from evaluation_worker import EvaluationWorker
from model_registry import MODEL_REGISTRY
from model_artifacts import discover_model_runs, append_evaluation_result
from training_graph import TrainingGraphTab
from gui_help import PARAMETER_HELP, format_parameter_help
from training_config import load_training_config as load_config_file, save_training_config as save_config_file, validate_training_config
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties


def architecture_signature(result):
    """Show a complete architecture only when every saved dimension is known."""
    values = []
    for key in ("cnn_filters", "cnn_kernel_size", "bottleneck_units"):
        value = result.get(key)
        try:
            number = int(value)
            if number != float(value): return "unknown"
        except (TypeError, ValueError, OverflowError):
            return "unknown"
        values.append(number)
    return f"F{values[0]}-K{values[1]}-B{values[2]}"


class LossCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(7, 4), tight_layout=True)
        super().__init__(self.figure)
        self.axis = self.figure.add_subplot(111)
        korean_font = FontProperties(fname=r"C:\Windows\Fonts\malgun.ttf") if Path(r"C:\Windows\Fonts\malgun.ttf").is_file() else None
        self.axis.set_title("학습 손실", fontproperties=korean_font)
        self.axis.set_xlabel("에포크", fontproperties=korean_font)
        self.axis.set_ylabel("MSE 손실", fontproperties=korean_font)
        self.train_line, = self.axis.plot([], [], label="학습 손실")
        self.val_line, = self.axis.plot([], [], label="검증 손실")
        self.axis.legend(prop=korean_font)
        self.best_marker = None
        self.lr_markers = []

    def update_data(self, epoch, loss, val_loss):
        train_x = list(self.train_line.get_xdata()) + [epoch]
        train_y = list(self.train_line.get_ydata()) + [loss]
        val_y = list(self.val_line.get_ydata()) + [val_loss]
        self.train_line.set_data(train_x, train_y)
        self.val_line.set_data(train_x, val_y)
        self.axis.relim(); self.axis.autoscale_view(); self.draw_idle()

    def clear_data(self):
        self.train_line.set_data([], [])
        self.val_line.set_data([], [])
        if self.best_marker: self.best_marker.remove(); self.best_marker = None
        for marker in self.lr_markers: marker.remove()
        self.lr_markers.clear(); self.axis.relim(); self.axis.autoscale_view(); self.draw_idle()

    def update_markers(self, best_epoch, lr_reduced=False):
        if self.best_marker: self.best_marker.remove()
        self.best_marker = self.axis.axvline(best_epoch, color="green", linestyle="--", alpha=.7)
        if lr_reduced: self.lr_markers.append(self.axis.axvline(len(self.train_line.get_xdata()), color="orange", linestyle=":"))
        self.draw_idle()


class MainWindow(QMainWindow):
    _korean_font_family = None

    def __init__(self):
        super().__init__()
        korean_font_path = Path(r"C:\Windows\Fonts\malgun.ttf")
        if self._korean_font_family is None and korean_font_path.is_file():
            font_id = QFontDatabase.addApplicationFont(str(korean_font_path))
            families = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
            if families:
                MainWindow._korean_font_family = families[0]
        if self._korean_font_family:
            self.setFont(QFont(self._korean_font_family, 10))
        self.setWindowTitle("KAMP 예지보전 AI 학습 시스템")
        self.resize(1280, 820)
        self.thread = None
        self.worker = None
        self.evaluation_thread = None
        self.evaluation_worker = None
        self.evaluation_controller = EvaluationController()
        self.comparison_history = []
        self.baseline_result = None
        self.champion_result = None
        self._build_ui()
        self._log("Application started")

    def _build_ui(self):
        root = QWidget(); outer = QVBoxLayout(root)
        header = QLabel("KAMP 예지보전 AI 학습 시스템")
        header.setObjectName("header"); outer.addWidget(header)
        self.status_label = QLabel("선택한 모델: CNN_LSTM_AUTOENCODER    상태: 준비")
        outer.addWidget(self.status_label)
        tabs = QTabWidget(); self.tabs = tabs; live = QWidget(); live_layout = QHBoxLayout(live)
        self.live_splitter = QSplitter(Qt.Horizontal)
        left_container = QWidget(); left_layout = QVBoxLayout(left_container)
        left_container.setMinimumWidth(420)
        self.training_scroll = QScrollArea(); self.training_scroll.setWidgetResizable(True)
        self.training_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.training_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        settings_content = QWidget(); settings_layout = QVBoxLayout(settings_content)
        settings_layout.setSpacing(12)
        self.training_scroll.setWidget(settings_content)
        left_layout.addWidget(self.training_scroll, 1)
        right_container = QWidget(); right_layout = QVBoxLayout(right_container)
        right_splitter = QSplitter(Qt.Vertical)
        system = QGroupBox("시스템 / 데이터 상태"); system_form = QFormLayout(system)
        self.system_labels = {}
        for key, value in (("Python", platform.python_version()), ("TensorFlow", tf.__version__),
                           ("OS", platform.platform()), ("Device", "CPU"), ("GPU", "Not used"),
                           ("Sequence", "20"), ("Horizon", "100"), ("Input", "(20, 3)")):
            label = QLabel(value); self.system_labels[key] = label; system_form.addRow(key, label)
        system.setMaximumHeight(175)
        right_layout.addWidget(system)
        monitor = QGroupBox("학습 상태"); monitor_form = QFormLayout(monitor)
        self.monitor_labels = {}
        monitor_names = {"Model":"모델", "Experiment Name":"실험 이름", "Epoch":"에포크", "Train Loss":"학습 손실", "Validation Loss":"검증 손실", "Best Epoch":"최적 에포크", "Best Val Loss":"최적 검증 손실", "Generalization Gap":"일반화 차이", "Learning Rate":"학습률", "Elapsed":"경과 시간", "EarlyStopping":"조기 종료 대기", "ReduceLR":"학습률 감소 횟수", "Optimizer":"최적화 알고리즘", "Loss":"손실 함수", "Batch Size":"배치 크기", "Noise Std":"노이즈 표준편차"}
        for key in ("Model", "Experiment Name", "Epoch", "Train Loss", "Validation Loss", "Best Epoch", "Best Val Loss", "Generalization Gap", "Learning Rate", "Elapsed", "EarlyStopping", "ReduceLR", "Optimizer", "Loss", "Batch Size", "Noise Std"):
            label = QLabel("-"); self.monitor_labels[key] = label; monitor_form.addRow(monitor_names[key], label)
        self.progress = QProgressBar(); monitor_form.addRow("진행률", self.progress)
        right_splitter.addWidget(monitor)
        graph_box = QWidget(); graph_layout = QVBoxLayout(graph_box)
        self.loss_canvas = LossCanvas(); graph_layout.addWidget(self.loss_canvas, 3)
        self.log_view = QPlainTextEdit(); self.log_view.setReadOnly(True); self.log_view.setMaximumBlockCount(1000)
        graph_layout.addWidget(self.log_view, 1)
        right_splitter.addWidget(graph_box); right_splitter.setSizes([300, 500])
        right_layout.addWidget(right_splitter)
        model_box = QGroupBox("모델 설정"); model_layout = QVBoxLayout(model_box)
        self.training_model_combo = QComboBox()
        for spec in MODEL_REGISTRY.values():
            if spec.status == "ACTIVE": self.training_model_combo.addItem(spec.display_name, spec.id)
        self.training_model_combo.setCurrentIndex(self.training_model_combo.findData("CNN_LSTM_AUTOENCODER"))
        self.training_model_combo.currentIndexChanged.connect(self.update_training_model)
        model_layout.addWidget(QLabel("학습 모델")); model_layout.addWidget(self.training_model_combo)
        settings_layout.addWidget(model_box)
        training_box = QGroupBox("학습 파라미터"); training_form = QFormLayout(training_box)
        callback_box = QGroupBox("자동 학습 제어"); callback_form = QFormLayout(callback_box)
        self.denoising_box = QGroupBox("노이즈 제거 학습 설정"); denoising_form = QFormLayout(self.denoising_box)
        preset_box = QGroupBox("실험 프리셋"); preset_form = QFormLayout(preset_box)
        self.preset_combo = QComboBox()
        for display, preset_id in (("CNN-LSTM 기준", "CNN-LSTM BASELINE"), ("CNN Bottleneck 16", "CNN BOTTLENECK 16"), ("CNN Filter 16", "CNN FILTER 16"), ("CNN-LSTM 커널 5", "CNN-LSTM KERNEL 5"), ("CNN-LSTM 보수적", "CNN-LSTM CONSERVATIVE"),
                                   ("Denoising 기본", "DENOISING DEFAULT"), ("Denoising 약하게 0.005", "DENOISING WEAK 0.005"), ("사용자 설정", "CUSTOM")):
            self.preset_combo.addItem(display, preset_id)
        self.preset_combo.currentIndexChanged.connect(lambda: self.apply_preset(self.preset_combo.currentData()))
        self.experiment_name_edit = QLineEdit(); self.experiment_name_edit.setPlaceholderText("비워두면 자동 생성")
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 10000); self.epochs_spin.setValue(800); self.epochs_spin.setToolTip("최대 반복 횟수입니다. 조기 종료가 켜져 있으면 더 일찍 끝날 수 있습니다.")
        self.batch_spin = QSpinBox(); self.batch_spin.setRange(1, 4096); self.batch_spin.setValue(128); self.batch_spin.setToolTip("한 번의 가중치 업데이트에 사용할 데이터 개수입니다.")
        self.learning_rate_spin = QDoubleSpinBox(); self.learning_rate_spin.setDecimals(6); self.learning_rate_spin.setRange(0.000001, 1); self.learning_rate_spin.setValue(.001); self.learning_rate_spin.setToolTip("가중치를 한 번에 얼마나 크게 바꿀지 정합니다.")
        self.optimizer_combo = QComboBox(); self.optimizer_combo.addItems(["Adam", "AdamW", "RMSprop"]); self.optimizer_combo.setToolTip("가중치 업데이트 방법입니다.")
        self.loss_combo = QComboBox(); self.loss_combo.addItems(["MSE", "Huber"]); self.loss_combo.setToolTip("MSE는 큰 오차를 강조하고 Huber는 큰 오차에 덜 민감합니다.")
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(0, 999999); self.seed_spin.setValue(42); self.seed_spin.setToolTip("실험 재현을 위한 난수 시작값입니다.")
        self.model_structure_box = QGroupBox("모델 구조 설정"); structure_form = QFormLayout(self.model_structure_box)
        self.cnn_filters_combo = QComboBox()
        self.bottleneck_combo = QComboBox()
        for units in (16, 32, 64):
            self.cnn_filters_combo.addItem(str(units), units)
            self.bottleneck_combo.addItem(str(units), units)
        self.cnn_filters_combo.setCurrentIndex(self.cnn_filters_combo.findData(32))
        self.bottleneck_combo.setCurrentIndex(self.bottleneck_combo.findData(32))
        self.cnn_filters_combo.setToolTip("두 Conv1D 층의 특징 맵 개수입니다.")
        self.bottleneck_combo.setToolTip("Encoder 마지막 및 Decoder 첫 LSTM의 공통 크기입니다.")
        self.cnn_kernel_combo = QComboBox()
        for size in (3, 5, 7): self.cnn_kernel_combo.addItem(str(size), size)
        self.cnn_kernel_combo.setToolTip("Conv1D가 함께 보는 연속 시점 수입니다.")
        self.noise_std_spin = QDoubleSpinBox(); self.noise_std_spin.setDecimals(4); self.noise_std_spin.setRange(0, .1); self.noise_std_spin.setSingleStep(.005); self.noise_std_spin.setValue(.01); self.noise_std_spin.setToolTip("Denoising 입력에 더할 Gaussian 노이즈의 강도입니다."); self.noise_std_spin.setEnabled(False)
        self.huber_delta_spin = QDoubleSpinBox(); self.huber_delta_spin.setRange(.000001, 1000); self.huber_delta_spin.setValue(1.0)
        self.weight_decay_spin = QDoubleSpinBox(); self.weight_decay_spin.setDecimals(6); self.weight_decay_spin.setRange(0, 1); self.weight_decay_spin.setValue(.0001)
        self.reduce_lr_enabled = QCheckBox(); self.reduce_lr_enabled.setChecked(True)
        self.reduce_lr_factor_spin = QDoubleSpinBox(); self.reduce_lr_factor_spin.setDecimals(3); self.reduce_lr_factor_spin.setRange(.001, .999); self.reduce_lr_factor_spin.setValue(.7)
        self.reduce_lr_patience_spin = QSpinBox(); self.reduce_lr_patience_spin.setRange(1, 10000); self.reduce_lr_patience_spin.setValue(50)
        self.min_lr_spin = QDoubleSpinBox(); self.min_lr_spin.setDecimals(8); self.min_lr_spin.setRange(0, 1); self.min_lr_spin.setValue(0)
        self.early_stopping_enabled = QCheckBox(); self.early_stopping_enabled.setChecked(True)
        self.early_stopping_patience_spin = QSpinBox(); self.early_stopping_patience_spin.setRange(1, 10000); self.early_stopping_patience_spin.setValue(120)
        self.early_stopping_min_delta_spin = QDoubleSpinBox(); self.early_stopping_min_delta_spin.setDecimals(8); self.early_stopping_min_delta_spin.setRange(0, 1); self.early_stopping_min_delta_spin.setValue(.00001)
        self.restore_best_weights = QCheckBox(); self.restore_best_weights.setChecked(True)
        self.noise_type_combo = QComboBox(); self.noise_type_combo.addItem("Gaussian")
        self.noise_mean_spin = QDoubleSpinBox(); self.noise_mean_spin.setDecimals(4); self.noise_mean_spin.setRange(-1, 1); self.noise_mean_spin.setValue(0)
        self.noise_clip_check = QCheckBox(); self.noise_clip_check.setChecked(True)
        for toggle in (self.reduce_lr_enabled, self.early_stopping_enabled, self.restore_best_weights, self.noise_clip_check):
            toggle.setText("ON" if toggle.isChecked() else "OFF")
            toggle.toggled.connect(lambda enabled, target=toggle: target.setText("ON" if enabled else "OFF"))
        self.training_model_combo.currentIndexChanged.connect(self._update_model_dependent_controls)
        self.reset_config_button = QPushButton("기준 설정으로 초기화"); self.reset_config_button.clicked.connect(self.reset_training_config)
        self.save_config_button = QPushButton("설정 저장"); self.save_config_button.clicked.connect(self.save_training_config)
        self.load_config_button = QPushButton("설정 불러오기"); self.load_config_button.clicked.connect(self.load_training_config)
        self.parameter_help_buttons = {}
        for label, widget, guide in (("실험 이름",self.experiment_name_edit,"Experiment Name"),("최대 에포크",self.epochs_spin,"Epoch"),("배치 크기 (Batch Size)",self.batch_spin,"Batch Size"),("학습률 (Learning Rate)",self.learning_rate_spin,"Learning Rate"),("최적화 알고리즘 (Optimizer)",self.optimizer_combo,"Optimizer"),("가중치 감쇠",self.weight_decay_spin,"Weight Decay"),("손실 함수 (Loss)",self.loss_combo,"Loss"),("Huber 기준값",self.huber_delta_spin,"Huber Delta"),("랜덤 시드 (Random Seed)",self.seed_spin,"Random Seed")):
            self._add_guided_row(training_form, label, widget, guide)
        self._add_guided_row(structure_form, "CNN 필터 수", self.cnn_filters_combo, "CNN Filters")
        self._add_guided_row(structure_form, "CNN 커널 크기", self.cnn_kernel_combo, "CNN Kernel Size")
        self._add_guided_row(structure_form, "Bottleneck 크기", self.bottleneck_combo, "Bottleneck Units")
        for label, widget, guide in (("학습률 자동 감소",self.reduce_lr_enabled,"ReduceLR Enabled"),("학습률 감소 비율",self.reduce_lr_factor_spin,"ReduceLR Factor"),("학습률 감소 대기 에포크",self.reduce_lr_patience_spin,"ReduceLR Patience"),("최소 학습률",self.min_lr_spin,"Minimum LR"),("조기 종료",self.early_stopping_enabled,"EarlyStopping Enabled"),("조기 종료 대기 에포크",self.early_stopping_patience_spin,"EarlyStopping Patience"),("최소 개선량",self.early_stopping_min_delta_spin,"EarlyStopping Min Delta"),("최적 가중치 복원",self.restore_best_weights,"Restore Best Weights")):
            self._add_guided_row(callback_form, label, widget, guide)
        for label, widget, guide in (("노이즈 종류",self.noise_type_combo,"Noise Type"),("노이즈 평균",self.noise_mean_spin,"Noise Mean"),("노이즈 표준편차",self.noise_std_spin,"Noise Std"),("노이즈 적용 후 범위 제한",self.noise_clip_check,"Noise Clip")):
            self._add_guided_row(denoising_form, label, widget, guide)
        preset_form.addRow("프리셋", self.preset_combo)
        guide_box = QGroupBox("파라미터 설명"); guide_layout = QVBoxLayout(guide_box)
        self.parameter_guide = QLabel(); self.parameter_guide.setWordWrap(True)
        self.parameter_guide.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        guide_layout.addWidget(self.parameter_guide)
        config_buttons = QHBoxLayout(); config_buttons.addWidget(self.reset_config_button); config_buttons.addWidget(self.save_config_button); config_buttons.addWidget(self.load_config_button)
        preset_form.addRow(config_buttons)
        architecture_box = QGroupBox("모델 구조"); architecture_layout = QVBoxLayout(architecture_box)
        self.training_model_description = QLabel(); self.training_model_description.setWordWrap(True)
        self.architecture_label = QLabel(); self.architecture_label.setWordWrap(True)
        self.params_label = QLabel("파라미터 수: -")
        for label in (self.training_model_description, self.architecture_label, self.params_label): architecture_layout.addWidget(label)
        for box in (training_box, self.model_structure_box, callback_box, self.denoising_box, preset_box, guide_box, architecture_box): settings_layout.addWidget(box)
        settings_layout.addStretch()
        self.training_form = training_form
        self.optimizer_combo.currentTextChanged.connect(self._update_optimizer_loss_controls)
        self.loss_combo.currentTextChanged.connect(self._update_optimizer_loss_controls)
        self.reduce_lr_enabled.toggled.connect(self._update_callback_controls)
        self.early_stopping_enabled.toggled.connect(self._update_callback_controls)
        self._update_optimizer_loss_controls()
        self._applying_preset = False
        editable = [self.cnn_filters_combo, self.cnn_kernel_combo, self.bottleneck_combo, self.epochs_spin, self.batch_spin, self.learning_rate_spin, self.optimizer_combo, self.weight_decay_spin,
                    self.loss_combo, self.huber_delta_spin, self.seed_spin, self.reduce_lr_enabled, self.reduce_lr_factor_spin,
                    self.reduce_lr_patience_spin, self.min_lr_spin, self.early_stopping_enabled,
                    self.early_stopping_patience_spin, self.early_stopping_min_delta_spin, self.restore_best_weights,
                    self.noise_type_combo, self.noise_mean_spin, self.noise_std_spin, self.noise_clip_check]
        for widget in editable:
            signal = getattr(widget, "valueChanged", None) or getattr(widget, "currentIndexChanged", None) or getattr(widget, "toggled", None)
            signal.connect(self._manual_config_changed)
            widget.installEventFilter(self)
            if isinstance(widget, QAbstractSpinBox):
                widget.setButtonSymbols(QAbstractSpinBox.NoButtons)
                widget.lineEdit().installEventFilter(self)
        self.preset_combo.installEventFilter(self)
        self.training_model_combo.installEventFilter(self)
        self._wheel_protected_widgets = {self.preset_combo, self.training_model_combo, *editable}
        self._wheel_protected_widgets.update(widget.lineEdit() for widget in editable if isinstance(widget, QAbstractSpinBox))
        self._training_editable_widgets = [self.training_model_combo, self.preset_combo, self.experiment_name_edit, *editable,
                                           self.reset_config_button, self.load_config_button]
        self.experiment_name_edit.textChanged.connect(self._update_experiment_summary)
        self.preset_combo.currentTextChanged.connect(self._update_experiment_summary)
        self.cnn_kernel_combo.currentIndexChanged.connect(self.update_training_model)
        self.cnn_filters_combo.currentIndexChanged.connect(self.update_training_model)
        self.bottleneck_combo.currentIndexChanged.connect(self.update_training_model)
        self.show_parameter_guide("Learning Rate")
        self._update_callback_controls(); self._update_model_dependent_controls()
        self.current_experiment_summary = QLabel(); self.current_experiment_summary.setWordWrap(True)
        self.current_experiment_summary.setObjectName("experimentSummary")
        left_layout.addWidget(self.current_experiment_summary)
        controls = QHBoxLayout(); self.run_mode = QComboBox(); self.run_mode.addItem("전체 학습", "FULL_TRAINING"); self.run_mode.addItem("빠른 시험", "QUICK_TEST")
        controls.addWidget(self.run_mode)
        self.start_button = QPushButton("학습 시작"); self.start_button.clicked.connect(self.start_training); controls.addWidget(self.start_button)
        self.stop_button = QPushButton("학습 중지"); self.stop_button.setEnabled(False); self.stop_button.clicked.connect(self.stop_training); controls.addWidget(self.stop_button)
        left_layout.addLayout(controls)
        utility_controls = QHBoxLayout()
        self.reset_graph_button = QPushButton("그래프 초기화"); self.reset_graph_button.clicked.connect(self.reset_loss_graph); utility_controls.addWidget(self.reset_graph_button)
        self.open_graph_button = QPushButton("학습 그래프 크게 보기"); self.open_graph_button.clicked.connect(self.show_current_graph); utility_controls.addWidget(self.open_graph_button)
        self.open_button = QPushButton("결과 폴더 열기"); self.open_button.clicked.connect(self.open_outputs); utility_controls.addWidget(self.open_button)
        left_layout.addLayout(utility_controls)
        self.live_splitter.addWidget(left_container); self.live_splitter.addWidget(right_container)
        self.live_splitter.setSizes([480, 800]); live_layout.addWidget(self.live_splitter)
        tabs.addTab(live, "실시간 학습")
        self.graph_tab = TrainingGraphTab()
        tabs.addTab(self.graph_tab, "학습 그래프")
        tabs.currentChanged.connect(lambda index: self.graph_tab.render() if tabs.widget(index) is self.graph_tab else None)
        data_tab = QWidget(); data_layout = QFormLayout(data_tab); self.dataset_text = QLabel("실행 기록 없음"); self.dataset_text.setWordWrap(True); data_layout.addRow("데이터셋", self.dataset_text); tabs.addTab(data_tab, "데이터")
        eval_tab = QWidget(); eval_layout = QVBoxLayout(eval_tab)
        selectors = QGroupBox("평가 설정"); selector_form = QFormLayout(selectors)
        self.evaluation_model_combo = QComboBox()
        for spec in MODEL_REGISTRY.values(): self.evaluation_model_combo.addItem(spec.display_name, spec.id)
        self.evaluation_run_combo = QComboBox(); self.evaluation_model_combo.currentIndexChanged.connect(self.refresh_evaluation_runs)
        selector_form.addRow("모델 종류", self.evaluation_model_combo); selector_form.addRow("저장된 실행", self.evaluation_run_combo)
        self.score_combo = QComboBox()
        for method in SCORE_METHODS.values(): self.score_combo.addItem(method.display_name, method.id)
        self.threshold_combo = QComboBox()
        for method in THRESHOLD_METHODS.values(): self.threshold_combo.addItem(method.display_name, method.id)
        self.score_combo.currentIndexChanged.connect(self.update_algorithm_description)
        self.threshold_combo.currentIndexChanged.connect(self.update_algorithm_description)
        self.temporal_combo = QComboBox()
        self.temporal_combo.addItem("사용 안 함", "NONE")
        self.temporal_combo.addItem("EWMA", "EWMA")
        self.ewma_alpha_spin = QDoubleSpinBox()
        self.ewma_alpha_spin.setRange(0.01, 1.00)
        self.ewma_alpha_spin.setSingleStep(0.01)
        self.ewma_alpha_spin.setDecimals(2)
        self.ewma_alpha_spin.setValue(0.40)
        self.ewma_alpha_spin.setEnabled(False)
        self.ewma_alpha_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.timestamp_aware_check = QCheckBox("ON")
        self.timestamp_aware_check.setChecked(True)
        self.temporal_combo.currentIndexChanged.connect(self._temporal_selection_changed)
        self.ewma_alpha_spin.valueChanged.connect(self.update_algorithm_description)
        self.timestamp_aware_check.toggled.connect(self._temporal_selection_changed)
        selector_form.addRow("이상 점수", self.score_combo)
        selector_form.addRow("시간축 점수 처리", self.temporal_combo)
        selector_form.addRow("EWMA α", self.ewma_alpha_spin)
        selector_form.addRow("Timestamp 연속성 고려", self.timestamp_aware_check)
        selector_form.addRow("임계값 방식", self.threshold_combo)
        self.eval_status_label = QLabel("모델: LSTM AutoEncoder    점수: 마지막 시점 MSE    임계값: 정밀도-재현율 균형점    상태: 미불러옴")
        selector_form.addRow("현재 선택", self.eval_status_label)
        self.loaded_architecture_label = QLabel("-")
        selector_form.addRow("불러온 모델 구조", self.loaded_architecture_label)
        eval_layout.addWidget(selectors)
        eval_actions = QHBoxLayout()
        self.load_model_button = QPushButton("모델 불러오기"); self.load_model_button.clicked.connect(self.load_evaluation_model)
        self.run_evaluation_button = QPushButton("평가 실행"); self.run_evaluation_button.clicked.connect(self.run_evaluation)
        self.compare_all_button = QPushButton("전체 방식 비교"); self.compare_all_button.clicked.connect(self.compare_all)
        self.sweep_ewma_button = QPushButton("EWMA α 비교"); self.sweep_ewma_button.clicked.connect(self.sweep_ewma)
        self.compare_models_button = QPushButton("모델 비교"); self.compare_models_button.clicked.connect(self.compare_models)
        self.clear_comparison_button = QPushButton("비교 기록 비우기"); self.clear_comparison_button.clicked.connect(self.clear_comparison)
        self.export_comparison_button = QPushButton("비교 CSV 저장"); self.export_comparison_button.clicked.connect(self.export_comparison)
        self.import_history_button = QPushButton("과거 결과 불러오기"); self.import_history_button.clicked.connect(self.import_historical_result)
        self.set_baseline_button = QPushButton("기준 실험으로 지정"); self.set_baseline_button.clicked.connect(self.set_current_as_baseline)
        self.set_champion_button = QPushButton("최고 실험으로 지정"); self.set_champion_button.clicked.connect(self.set_current_as_champion)
        self.show_history_graph_button = QPushButton("학습 곡선 보기"); self.show_history_graph_button.clicked.connect(self.show_evaluation_graph)
        for button in (self.load_model_button, self.run_evaluation_button, self.compare_all_button, self.sweep_ewma_button, self.compare_models_button, self.clear_comparison_button, self.export_comparison_button, self.import_history_button, self.set_baseline_button, self.set_champion_button, self.show_history_graph_button): eval_actions.addWidget(button)
        eval_layout.addLayout(eval_actions)
        description_box = QGroupBox("평가 방식 설명"); description_layout = QVBoxLayout(description_box)
        self.algorithm_description = QLabel(); self.algorithm_description.setWordWrap(True); description_layout.addWidget(self.algorithm_description); eval_layout.addWidget(description_box)
        result_box = QGroupBox("평가 결과"); result_layout = QGridLayout(result_box)
        self.eval_result_labels = {}
        metric_names = {"Threshold":"임계값", "Accuracy":"정확도 (Accuracy)", "Balanced Accuracy":"균형 정확도", "Precision":"정밀도 (Precision)", "Recall":"재현율 (Recall)", "F1 Score":"F1 점수", "Specificity":"특이도", "FPR":"오경보율 (FPR)", "FNR":"미탐율 (FNR)", "TN":"정상 정확 판정 (TN)", "FP (False Alarm)":"오경보 (FP)", "FN (Missed Anomaly)":"미탐 (FN)", "TP":"이상 정확 탐지 (TP)", "Fallback":"대체 방식"}
        result_keys = ("Threshold", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1 Score", "Specificity", "FPR", "FNR", "TN", "FP (False Alarm)", "FN (Missed Anomaly)", "TP", "Fallback")
        for index, key in enumerate(result_keys):
            label = QLabel("-"); self.eval_result_labels[key] = label
            result_layout.addWidget(QLabel(metric_names[key]), index // 2, (index % 2) * 2); result_layout.addWidget(label, index // 2, (index % 2) * 2 + 1)
        eval_layout.addWidget(result_box)
        self.comparison_summary_label = QLabel("현재 최고 실험: 없음\n기준 실험: 없음"); self.comparison_summary_label.setWordWrap(True); eval_layout.addWidget(self.comparison_summary_label)
        history_box = QGroupBox("실험 기록"); history_layout = QVBoxLayout(history_box)
        self.history_filters = {}
        filter_layout = QHBoxLayout()
        for name in ("Model", "Status", "Score", "Threshold", "Loss", "Seed"):
            combo = QComboBox(); combo.addItem("전체", "ALL"); combo.currentIndexChanged.connect(self.apply_history_filters); self.history_filters[name] = combo
            filter_layout.addWidget(QLabel({"Model":"모델", "Status":"상태", "Score":"점수", "Threshold":"임계값 방식", "Loss":"손실 함수", "Seed":"시드"}[name])); filter_layout.addWidget(combo)
        history_layout.addLayout(filter_layout)
        columns = ["상태", "모델", "실험", "Architecture", "시드", "최적화", "학습률", "손실 함수", "배치", "노이즈 표준편차", "점수", "임계값 방식", "임계값", "정확도", "균형 정확도", "정밀도", "재현율", "F1", "정상 정확 판정 (TN)", "오경보 (FP)", "미탐 (FN)", "이상 정확 탐지 (TP)", "최적 에포크", "최적 검증 손실", "학습 시간", "생성 시각", "Temporal", "Alpha", "Timestamp Reset", "Total Error", "Delay", "Pareto"]
        self.comparison_table = QTableWidget(0, len(columns)); self.comparison_table.setHorizontalHeaderLabels(columns)
        history_layout.addWidget(self.comparison_table); eval_layout.addWidget(history_box, 1)
        self.lock_evaluation_method = QCheckBox("동일 조건 비교: 점수·임계값 방식 잠금"); self.lock_evaluation_method.toggled.connect(lambda locked: (self.score_combo.setEnabled(not locked), self.threshold_combo.setEnabled(not locked))); eval_layout.addWidget(self.lock_evaluation_method)
        self.delta_comparison_label = QLabel("기준 실험과 현재 실험을 선택하면 차이를 표시합니다."); self.delta_comparison_label.setWordWrap(True); eval_layout.addWidget(self.delta_comparison_label)
        self.confusion_comparison_label = QLabel("기준 혼동행렬: -    선택한 혼동행렬: -"); eval_layout.addWidget(self.confusion_comparison_label)
        warning = QLabel("Test 지표는 최종 성능 확인용입니다. Test F1을 반복해서 보고 파라미터를 고르면 Test 데이터에 간접 과적합될 수 있습니다. 현재 분할은 KAMP 호환 방식이며 STRICT_TIME_SPLIT 검증이 추가로 필요합니다. 같은 시드를 써도 환경에 따라 완전히 같은 결과는 보장되지 않습니다.\n다음 권장 실험: Denoising CNN-LSTM에서 CNN 기준과 같은 시드·설정을 유지하고 Gaussian 노이즈 표준편차만 0.01로 변경하세요. MAHALANOBIS_ERROR + POT_1PCT 방식으로 비교하되 각 모델의 임계값 숫자는 검증 데이터에서 따로 계산합니다."); warning.setWordWrap(True); eval_layout.addWidget(warning)
        tabs.addTab(eval_tab, "평가 및 비교"); self.update_training_model(); self.refresh_evaluation_runs(); self.update_algorithm_description()
        outer.addWidget(tabs, 1)
        self.setCentralWidget(root)
        self.setStyleSheet("QMainWindow,QWidget{background:#111820;color:#e7eef5} QGroupBox{background:#16222d;border:1px solid #36536b;margin-top:8px;padding:8px} QGroupBox::title{color:#78c7e8} QLabel,QCheckBox{background:transparent;color:#e7eef5} QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox{background:#1d2d3b;color:#e7eef5;border:1px solid #4b7087;padding:3px;min-height:22px} QComboBox QAbstractItemView{background:#1d2d3b;color:#e7eef5} QScrollArea{background:#111820;border:0} QScrollBar:vertical{background:#14222e;width:12px} QScrollBar::handle:vertical{background:#52758a;min-height:24px} #header{font-size:22px;font-weight:bold;color:#8ad8f5;padding:8px} #experimentSummary{background:#203645;padding:8px;border:1px solid #4b7087} QPlainTextEdit{background:#0b1117;color:#e7eef5;font-family:Consolas} QPushButton{padding:8px;background:#1e536d;color:#e7eef5;border:1px solid #54a7c6}")

    def _log(self, message): self.log_view.appendPlainText(message)

    def _add_guided_row(self, form, label, widget, guide_name):
        title = QWidget(); row = QHBoxLayout(title)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel(label))
        info = QPushButton("ⓘ"); info.setFixedWidth(28)
        info.setToolTip(f"{PARAMETER_HELP[guide_name][0]} 설명 보기")
        info.clicked.connect(lambda checked=False, name=guide_name: self.show_parameter_help(name))
        row.addWidget(info)
        form.addRow(title, widget)
        self.parameter_help_buttons[guide_name] = info

    def show_parameter_help(self, name):
        self.show_parameter_guide(name)
        QMessageBox.information(self, f"{PARAMETER_HELP[name][0]} 설명", self.parameter_guide.text())

    def _set_training_locked(self, locked):
        for widget in self._training_editable_widgets:
            widget.setEnabled(not locked)
        self.run_mode.setEnabled(not locked)
        self.start_button.setEnabled(not locked)
        self.stop_button.setEnabled(locked)
        if not locked:
            self._update_callback_controls()
            self._update_model_dependent_controls()
            self._update_optimizer_loss_controls()

    def _update_optimizer_loss_controls(self, *_):
        adamw = self.optimizer_combo.currentText() == "AdamW"
        huber = self.loss_combo.currentText() == "Huber"
        self.training_form.setRowVisible(self.weight_decay_spin, adamw)
        self.training_form.setRowVisible(self.huber_delta_spin, huber)
        self.weight_decay_spin.setEnabled(adamw)
        self.huber_delta_spin.setEnabled(huber)

    def reset_loss_graph(self):
        self.loss_canvas.clear_data()
        self.graph_tab.reset_current()
        self._log("Training loss graph reset")

    def show_current_graph(self):
        self.graph_tab.show_current()
        self.tabs.setCurrentWidget(self.graph_tab)

    def show_evaluation_graph(self):
        run = self.evaluation_run_combo.currentData()
        if self.comparison_history:
            result = self._selected_history_result()
            model_id = result.get("model_id")
            experiment = result.get("experiment_name")
            if model_id in MODEL_REGISTRY:
                run = next((candidate for candidate in discover_model_runs(model_id)
                            if candidate.metadata.get("experiment_name") == experiment), run)
        if run is not None:
            self.graph_tab.open_saved_run(run)
        else:
            self.graph_tab.show_saved()
        self.tabs.setCurrentWidget(self.graph_tab)

    def update_training_model(self):
        spec = MODEL_REGISTRY[self.training_model_combo.currentData()]
        self.training_model_description.setText(f"{spec.display_name}\n과제: {'예측' if spec.task_type == 'FORECAST' else '정상 패턴 복원'}\n입력: 20 × 3{' + Gaussian 노이즈' if spec.denoising else ''}\n출력: {spec.forecast_length or 20} × 3\n\n{spec.description}")
        cnn = spec.id in {"CNN_LSTM_AUTOENCODER", "DENOISING_CNN_LSTM_AUTOENCODER"}
        kernel_size = self.cnn_kernel_combo.currentData() if cnn else None
        filters = self.cnn_filters_combo.currentData()
        bottleneck = self.bottleneck_combo.currentData()
        architecture = (f"입력 20×3{' + Gaussian 노이즈' if spec.denoising else ''}\n↓\n"
                        f"Conv1D {filters}, k={kernel_size} × 2\n↓\n"
                        f"LSTM64 → B{bottleneck}\n↓\nRepeatVector20\n↓\n"
                        f"LSTM{bottleneck} → LSTM64\n↓\n출력 20×3") if cnn else spec.architecture
        self.architecture_label.setText(architecture)
        model = spec.builder(kernel_size=kernel_size, cnn_filters=filters, bottleneck_units=bottleneck) if cnn else spec.builder()
        self.params_label.setText(f"전체 파라미터 수: {model.count_params()}")
        if self.thread is None or not self.thread.isRunning():
            self.status_label.setText(f"선택한 모델: {spec.display_name}    상태: 준비")
        self._update_experiment_summary()

    def _update_model_dependent_controls(self, *_):
        denoising = self.training_model_combo.currentData() == "DENOISING_CNN_LSTM_AUTOENCODER"
        self.model_structure_box.setVisible(self.training_model_combo.currentData() in {"CNN_LSTM_AUTOENCODER", "DENOISING_CNN_LSTM_AUTOENCODER"})
        self.denoising_box.setVisible(denoising)
        self.noise_std_spin.setEnabled(denoising)
        self._update_experiment_summary()

    def _update_callback_controls(self, *_):
        for widget in (self.reduce_lr_factor_spin, self.reduce_lr_patience_spin, self.min_lr_spin): widget.setEnabled(self.reduce_lr_enabled.isChecked())
        for widget in (self.early_stopping_patience_spin, self.early_stopping_min_delta_spin, self.restore_best_weights): widget.setEnabled(self.early_stopping_enabled.isChecked())

    def _update_experiment_summary(self, *_):
        if not hasattr(self, "current_experiment_summary"): return
        noise = f"Gaussian std {self.noise_std_spin.value():g}" if self.training_model_combo.currentData() == "DENOISING_CNN_LSTM_AUTOENCODER" else "Off"
        cnn = self.training_model_combo.currentData() in {"CNN_LSTM_AUTOENCODER", "DENOISING_CNN_LSTM_AUTOENCODER"}
        architecture = (f"Architecture: F{self.cnn_filters_combo.currentData()} / K{self.cnn_kernel_combo.currentData()} / B{self.bottleneck_combo.currentData()}  "
                        f"CNN Filters: {self.cnn_filters_combo.currentData()}  CNN Kernel: {self.cnn_kernel_combo.currentData()}  Bottleneck: {self.bottleneck_combo.currentData()}  ") if cnn else ""
        self.current_experiment_summary.setText(
            f"현재 실험 설정  |  {self.training_model_combo.currentText()}  |  "
            f"최대 에포크 {self.epochs_spin.value()}  배치 {self.batch_spin.value()}  학습률 {self.learning_rate_spin.value():g}\n"
            f"최적화 {self.optimizer_combo.currentText()}  손실 {self.loss_combo.currentText()}  "
            f"시드 {self.seed_spin.value()}  {architecture}노이즈 {noise}  프리셋 {self.preset_combo.currentText()}  "
            f"실험 {self.experiment_name_edit.text().strip() or '자동 생성'}")

    def reset_training_config(self):
        self.apply_preset("CNN-LSTM BASELINE"); self.preset_combo.setCurrentIndex(self.preset_combo.findData("CNN-LSTM BASELINE")); self.experiment_name_edit.clear()

    def _training_config_dict(self):
        return {"model_id": self.training_model_combo.currentData(), "preset": self.preset_combo.currentData(),
                "experiment_name": self.experiment_name_edit.text().strip(), "epochs": self.epochs_spin.value(),
                "batch_size": self.batch_spin.value(), "learning_rate": self.learning_rate_spin.value(),
                "cnn_kernel_size": self.cnn_kernel_combo.currentData(),
                "cnn_filters": self.cnn_filters_combo.currentData(), "bottleneck_units": self.bottleneck_combo.currentData(),
                "optimizer": self.optimizer_combo.currentText(), "weight_decay":self.weight_decay_spin.value(),
                "loss": self.loss_combo.currentText(), "huber_delta":self.huber_delta_spin.value(),
                "random_seed": self.seed_spin.value(), "reduce_lr_enabled":self.reduce_lr_enabled.isChecked(),
                "reduce_lr_factor":self.reduce_lr_factor_spin.value(), "reduce_lr_patience":self.reduce_lr_patience_spin.value(),
                "min_lr":self.min_lr_spin.value(), "early_stopping_enabled":self.early_stopping_enabled.isChecked(),
                "early_stopping_patience":self.early_stopping_patience_spin.value(),
                "early_stopping_min_delta":self.early_stopping_min_delta_spin.value(),
                "restore_best_weights":self.restore_best_weights.isChecked(), "noise_type":self.noise_type_combo.currentText(),
                "noise_mean":self.noise_mean_spin.value(), "noise_std": self.noise_std_spin.value(),
                "noise_clip":self.noise_clip_check.isChecked()}

    def save_training_config(self):
        default_dir = str(core.BASE_DIR / "configs")
        Path(default_dir).mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(self, "학습 설정 저장", str(Path(default_dir) / "training_config.json"), "JSON (*.json)")
        if not path: return
        save_config_file(path, self._training_config_dict())
        self._log(f"Training config saved: {path}")

    def load_training_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "학습 설정 불러오기", str(core.BASE_DIR / "configs"), "JSON (*.json)")
        if not path: return
        try:
            data = load_config_file(path)
            self._applying_preset = True
            model_id = data.get("model_id")
            index = self.training_model_combo.findData(model_id)
            if index >= 0: self.training_model_combo.setCurrentIndex(index)
            preset = data.get("preset")
            self.preset_combo.blockSignals(True)
            preset_index = self.preset_combo.findData(preset)
            if preset_index >= 0: self.preset_combo.setCurrentIndex(preset_index)
            self.preset_combo.blockSignals(False)
            self.experiment_name_edit.setText(str(data.get("experiment_name", "")))
            self.epochs_spin.setValue(int(data.get("epochs", 800))); self.batch_spin.setValue(int(data.get("batch_size", 128)))
            self.learning_rate_spin.setValue(float(data.get("learning_rate", .001)))
            self.cnn_kernel_combo.setCurrentIndex(self.cnn_kernel_combo.findData(int(data.get("cnn_kernel_size", 3))))
            self.cnn_filters_combo.setCurrentIndex(self.cnn_filters_combo.findData(int(data.get("cnn_filters", 32))))
            self.bottleneck_combo.setCurrentIndex(self.bottleneck_combo.findData(int(data.get("bottleneck_units", 32))))
            self.optimizer_combo.setCurrentText(str(data.get("optimizer", "Adam"))); self.loss_combo.setCurrentText(str(data.get("loss", "MSE")))
            self.weight_decay_spin.setValue(float(data.get("weight_decay", .0001))); self.huber_delta_spin.setValue(float(data.get("huber_delta", 1.0)))
            self.seed_spin.setValue(int(data.get("random_seed", 42))); self.noise_std_spin.setValue(float(data.get("noise_std", .01)))
            self.reduce_lr_enabled.setChecked(bool(data.get("reduce_lr_enabled", True))); self.reduce_lr_factor_spin.setValue(float(data.get("reduce_lr_factor", .7)))
            self.reduce_lr_patience_spin.setValue(int(data.get("reduce_lr_patience", 50))); self.min_lr_spin.setValue(float(data.get("min_lr", 0)))
            self.early_stopping_enabled.setChecked(bool(data.get("early_stopping_enabled", True))); self.early_stopping_patience_spin.setValue(int(data.get("early_stopping_patience", 120)))
            self.early_stopping_min_delta_spin.setValue(float(data.get("early_stopping_min_delta", .00001))); self.restore_best_weights.setChecked(bool(data.get("restore_best_weights", True)))
            self.noise_type_combo.setCurrentText(str(data.get("noise_type", "Gaussian"))); self.noise_mean_spin.setValue(float(data.get("noise_mean", 0))); self.noise_clip_check.setChecked(bool(data.get("noise_clip", True)))
            self._applying_preset = False
            self._update_callback_controls(); self._update_model_dependent_controls(); self.update_training_model(); self._update_experiment_summary()
            self._log(f"Training config loaded: {path}")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self._applying_preset = False
            QMessageBox.critical(self, "설정 오류", f"설정 파일을 불러오지 못했습니다: {exc}")

    def apply_preset(self, preset):
        if preset == "CUSTOM": return
        if not hasattr(self, "epochs_spin"): return
        self._applying_preset = True
        model_id = "DENOISING_CNN_LSTM_AUTOENCODER" if preset in ("DENOISING DEFAULT", "DENOISING WEAK 0.005") else "CNN_LSTM_AUTOENCODER"
        self.training_model_combo.setCurrentIndex(self.training_model_combo.findData(model_id))
        self.cnn_kernel_combo.setCurrentIndex(self.cnn_kernel_combo.findData(5 if preset == "CNN-LSTM KERNEL 5" else 3))
        self.cnn_filters_combo.setCurrentIndex(self.cnn_filters_combo.findData(16 if preset == "CNN FILTER 16" else 32))
        self.bottleneck_combo.setCurrentIndex(self.bottleneck_combo.findData(16 if preset == "CNN BOTTLENECK 16" else 32))
        self.epochs_spin.setValue(800); self.batch_spin.setValue(128); self.optimizer_combo.setCurrentText("Adam"); self.weight_decay_spin.setValue(.0001); self.loss_combo.setCurrentText("MSE"); self.huber_delta_spin.setValue(1.0); self.seed_spin.setValue(42)
        self.reduce_lr_enabled.setChecked(True); self.reduce_lr_factor_spin.setValue(.7); self.reduce_lr_patience_spin.setValue(50); self.min_lr_spin.setValue(0)
        self.early_stopping_enabled.setChecked(True); self.early_stopping_patience_spin.setValue(120); self.early_stopping_min_delta_spin.setValue(.00001); self.restore_best_weights.setChecked(True)
        self.learning_rate_spin.setValue(.0005 if preset == "CNN-LSTM CONSERVATIVE" else .001)
        self.noise_type_combo.setCurrentText("Gaussian"); self.noise_mean_spin.setValue(0); self.noise_std_spin.setValue(.005 if preset == "DENOISING WEAK 0.005" else .01 if preset == "DENOISING DEFAULT" else 0.0); self.noise_clip_check.setChecked(True)
        self._applying_preset = False
        self.update_training_model()
        self._update_experiment_summary()

    def _manual_config_changed(self, *args):
        if not self._applying_preset: self.preset_combo.setCurrentIndex(self.preset_combo.findData("CUSTOM"))
        if hasattr(self, "_selected_parameter_name"):
            self.show_parameter_guide(self._selected_parameter_name)
        self._update_experiment_summary()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel and obj in getattr(self, "_wheel_protected_widgets", ()):
            scrollbar = self.training_scroll.verticalScrollBar()
            scrollbar.setValue(scrollbar.value() - event.angleDelta().y())
            return True
        if event.type() == event.Type.FocusIn:
            labels = {self.cnn_filters_combo:"CNN Filters", self.cnn_kernel_combo:"CNN Kernel Size", self.bottleneck_combo:"Bottleneck Units", self.learning_rate_spin:"Learning Rate", self.batch_spin:"Batch Size", self.epochs_spin:"Epoch",
                      self.optimizer_combo:"Optimizer", self.loss_combo:"Loss", self.huber_delta_spin:"Huber Delta",
                      self.seed_spin:"Random Seed", self.early_stopping_patience_spin:"EarlyStopping Patience",
                      self.early_stopping_min_delta_spin:"EarlyStopping Min Delta", self.reduce_lr_factor_spin:"ReduceLR Factor",
                      self.reduce_lr_patience_spin:"ReduceLR Patience", self.min_lr_spin:"Minimum LR", self.restore_best_weights:"Restore Best Weights",
                      self.weight_decay_spin:"Weight Decay", self.noise_mean_spin:"Noise Mean", self.noise_std_spin:"Noise Std", self.noise_clip_check:"Noise Clip"}
            if obj in labels: self.show_parameter_guide(labels[obj])
        return super().eventFilter(obj, event)

    def show_parameter_guide(self, name):
        self._selected_parameter_name = name
        controls = {"Experiment Name":self.experiment_name_edit, "CNN Filters":self.cnn_filters_combo, "CNN Kernel Size":self.cnn_kernel_combo, "Bottleneck Units":self.bottleneck_combo, "Learning Rate":self.learning_rate_spin, "Batch Size":self.batch_spin, "Epoch":self.epochs_spin,
                    "Optimizer":self.optimizer_combo, "Loss":self.loss_combo, "Huber Delta":self.huber_delta_spin,
                    "Random Seed":self.seed_spin, "EarlyStopping Enabled":self.early_stopping_enabled,
                    "EarlyStopping Patience":self.early_stopping_patience_spin,
                    "EarlyStopping Min Delta":self.early_stopping_min_delta_spin, "ReduceLR Enabled":self.reduce_lr_enabled,
                    "ReduceLR Factor":self.reduce_lr_factor_spin,
                    "ReduceLR Patience":self.reduce_lr_patience_spin, "Minimum LR":self.min_lr_spin,
                    "Restore Best Weights":self.restore_best_weights, "Weight Decay":self.weight_decay_spin,
                    "Noise Type":self.noise_type_combo, "Noise Mean":self.noise_mean_spin,
                    "Noise Std":self.noise_std_spin, "Noise Clip":self.noise_clip_check}
        control = controls[name]
        current = control.text() if isinstance(control, QLineEdit) else control.currentText() if isinstance(control, QComboBox) else ("ON" if isinstance(control, QCheckBox) and control.isChecked() else "OFF" if isinstance(control, QCheckBox) else str(control.value()))
        self.parameter_guide.setText(format_parameter_help(name, current or "자동 생성"))

    def refresh_evaluation_runs(self):
        self.evaluation_run_combo.clear()
        self.loaded_architecture_label.setText("-")
        for run in discover_model_runs(self.evaluation_model_combo.currentData()): self.evaluation_run_combo.addItem(run.run_id, run)
        if hasattr(self, "score_combo"): self.update_algorithm_description()

    def update_algorithm_description(self):
        score_id = self.score_combo.currentData()
        threshold_id = self.threshold_combo.currentData()
        score = SCORE_METHODS[score_id]; threshold = THRESHOLD_METHODS[threshold_id]
        spec = MODEL_REGISTRY[self.evaluation_model_combo.currentData()]
        error_name = "Prediction Error" if spec.task_type == "FORECAST" else "Reconstruction Error"
        description = score.description.replace("Reconstruction Error", error_name).replace("복원 오차", "예측 오차" if spec.task_type == "FORECAST" else "복원 오차")
        temporal = self.temporal_combo.currentData()
        if temporal == "EWMA":
            temporal_description = ("EWMA: 기존 이상점수를 시간 순서대로 지수가중 이동평균하여 순간 변동을 완화합니다.\n"
                "EWMA[t] = alpha * score[t] + (1-alpha) * EWMA[t-1]\n"
                "작은 alpha는 과거 점수를 오래 기억해 오경보를 줄일 수 있지만 탐지가 늦어질 수 있습니다. "
                "큰 alpha는 현재 점수에 빠르게 반응하지만 평활 효과가 작습니다. "
                "원본 센서가 아닌 모델의 이상점수에 적용하는 후처리입니다.\n"
                "Timestamp 연속성 고려: 비정상적으로 긴 시간 간격에서는 EWMA 상태를 초기화합니다.")
        else:
            temporal_description = "시간축 점수 처리 없음: 기존 이상점수를 그대로 임계값 계산에 사용합니다."
        self.algorithm_description.setText(f"{score.display_name} ({'예측 오차' if spec.task_type == 'FORECAST' else '복원 오차'})\n{description}\n\n{temporal_description}\n\n{threshold.display_name}\n{threshold.description}")
        status = "준비" if self.evaluation_controller.is_loaded else "미불러옴"; spec = MODEL_REGISTRY[self.evaluation_model_combo.currentData()]
        self.eval_status_label.setText(f"모델: {spec.display_name}    Pipeline: {score.display_name} → {self._temporal_display()} → {threshold.display_name}    상태: {status}")

    def _temporal_display(self):
        if self.temporal_combo.currentData() != "EWMA":
            return "NONE"
        return f"EWMA α={self.ewma_alpha_spin.value():.2f} → Timestamp Reset {'ON' if self.timestamp_aware_check.isChecked() else 'OFF'}"

    def _temporal_selection_changed(self):
        self.ewma_alpha_spin.setEnabled(self.temporal_combo.currentData() == "EWMA")
        self.timestamp_aware_check.setText("ON" if self.timestamp_aware_check.isChecked() else "OFF")
        self.update_algorithm_description()

    def _set_evaluation_busy(self, busy):
        for button in (self.load_model_button, self.run_evaluation_button, self.compare_all_button, self.sweep_ewma_button, self.compare_models_button): button.setEnabled(not busy)
        for widget in (self.temporal_combo, self.ewma_alpha_spin, self.timestamp_aware_check):
            widget.setEnabled(not busy and (widget is not self.ewma_alpha_spin or self.temporal_combo.currentData() == "EWMA"))

    def _start_evaluation_worker(self, action):
        if self.evaluation_thread and self.evaluation_thread.isRunning():
            self._log("Evaluation is already running")
            return
        if action not in ("load", "compare_models") and not self.evaluation_controller.is_loaded:
            QMessageBox.information(self, "Evaluation", "LOAD MODEL을 먼저 실행하세요.")
            return
        self._set_evaluation_busy(True)
        self.evaluation_thread = QThread(self)
        self.evaluation_worker = EvaluationWorker(self.evaluation_controller, action,
            self.score_combo.currentData(), self.threshold_combo.currentData(),
            self.temporal_combo.currentData(), self.ewma_alpha_spin.value(),
            self.timestamp_aware_check.isChecked())
        self.evaluation_worker.moveToThread(self.evaluation_thread)
        self.evaluation_thread.started.connect(self.evaluation_worker.run)
        self.evaluation_worker.status_changed.connect(self._update_evaluation_status)
        self.evaluation_worker.log_message.connect(self._log)
        self.evaluation_worker.loaded.connect(self._evaluation_loaded)
        self.evaluation_worker.result_ready.connect(self._evaluation_result)
        self.evaluation_worker.comparison_ready.connect(self._comparison_results)
        self.evaluation_worker.failed.connect(self._evaluation_failed)
        self.evaluation_worker.finished.connect(self.evaluation_thread.quit)
        self.evaluation_thread.finished.connect(self._evaluation_thread_finished)
        self.evaluation_thread.start()

    def load_evaluation_model(self):
        run = self.evaluation_run_combo.currentData()
        if run is None: QMessageBox.information(self, "Evaluation", "선택한 모델의 저장 run이 없습니다."); return
        self.evaluation_controller.select_run(run); self._start_evaluation_worker("load")
    def run_evaluation(self): self._start_evaluation_worker("evaluate")
    def compare_all(self): self._start_evaluation_worker("compare_all")
    def sweep_ewma(self): self._start_evaluation_worker("sweep_ewma")
    def compare_models(self): self._start_evaluation_worker("compare_models")

    def _update_evaluation_status(self, status):
        spec = MODEL_REGISTRY[self.evaluation_model_combo.currentData()]
        self.eval_status_label.setText(f"모델: {spec.display_name}    Pipeline: {self.score_combo.currentText()} → {self._temporal_display()} → {self.threshold_combo.currentText()}    상태: {status}")

    def _evaluation_loaded(self, summary):
        self._log(f"Evaluation model ready: {summary}")
        self.update_algorithm_description()
        run = getattr(self.evaluation_controller, "model_run", None)
        if run is not None and run.model_id in {"CNN_LSTM_AUTOENCODER", "DENOISING_CNN_LSTM_AUTOENCODER"}:
            self.loaded_architecture_label.setText(architecture_signature(run.metadata))
        else:
            self.loaded_architecture_label.setText("-")

    def _evaluation_result(self, result):
        values = {"Threshold": f"{result['threshold']:.8g}", "Accuracy": f"{result['accuracy']:.6f}",
            "Balanced Accuracy": f"{result['balanced_accuracy']:.6f}", "Precision": f"{result['precision']:.6f}",
            "Recall": f"{result['recall']:.6f}", "F1 Score": f"{result['f1']:.6f}",
            "Specificity": f"{result['specificity']:.6f}", "FPR": f"{result['fpr']:.6f}", "FNR": f"{result['fnr']:.6f}",
            "TN": str(result['tn']), "FP (False Alarm)": str(result['fp']), "FN (Missed Anomaly)": str(result['fn']),
            "TP": str(result['tp']), "Fallback": result['fallback_reason'] if result['fallback_used'] else "No"}
        for key, value in values.items(): self.eval_result_labels[key].setText(value)
        self.eval_status_label.setText(f"모델: {result['model']}    Pipeline: {SCORE_METHODS[result['score_method']].display_name} → {self._result_temporal_display(result)} → {THRESHOLD_METHODS[result['threshold_method']].display_name}    상태: 완료")
        self._append_comparison(result)
        run = self.evaluation_run_combo.currentData()
        if run is not None:
            path = append_evaluation_result(run, result); self._log(f"Evaluation result saved: {path}")
        if result["fallback_used"]: self._log(f"Threshold fallback: {result['fallback_reason']}")

    def _comparison_results(self, results):
        for result in results: self._append_comparison(result)
        if results:
            self._evaluation_result_without_history(results[-1])
            self.eval_status_label.setText(f"모델: {results[-1].get('model','')}    Pipeline: {SCORE_METHODS[results[-1]['score_method']].display_name} → {self._result_temporal_display(results[-1])} → {THRESHOLD_METHODS[results[-1]['threshold_method']].display_name}    상태: 완료")
        self._log(f"Evaluation comparison completed: {len(results)} configurations")

    @staticmethod
    def _result_temporal_display(result):
        if result.get("temporal_method", "NONE") != "EWMA":
            return "NONE"
        try:
            alpha = f"{float(result.get('ewma_alpha', 0.4)):.2f}"
        except (TypeError, ValueError):
            alpha = "?"
        return f"EWMA α={alpha} → Timestamp Reset {'ON' if result.get('timestamp_aware', True) else 'OFF'}"

    def _evaluation_result_without_history(self, result):
        values = {"Threshold": f"{result['threshold']:.8g}", "Accuracy": f"{result['accuracy']:.6f}",
            "Balanced Accuracy": f"{result['balanced_accuracy']:.6f}", "Precision": f"{result['precision']:.6f}",
            "Recall": f"{result['recall']:.6f}", "F1 Score": f"{result['f1']:.6f}", "Specificity": f"{result['specificity']:.6f}",
            "FPR": f"{result['fpr']:.6f}", "FNR": f"{result['fnr']:.6f}", "TN": str(result['tn']),
            "FP (False Alarm)": str(result['fp']), "FN (Missed Anomaly)": str(result['fn']), "TP": str(result['tp']),
            "Fallback": result['fallback_reason'] if result['fallback_used'] else "No"}
        for key, value in values.items(): self.eval_result_labels[key].setText(value)

    def _append_comparison(self, result):
        fingerprint = (str(result.get("model_id", result.get("model", ""))), str(result.get("experiment_name", "")),
                       str(result.get("score_method", result.get("score_name", ""))), str(result.get("threshold_method", result.get("threshold_name", ""))),
                       str(result.get("threshold", "")), str(result.get("created_at", "")),
                       str(result.get("temporal_method", "NONE")), str(result.get("ewma_alpha", "")),
                       str(result.get("timestamp_aware", True)))
        if any(item.get("_fingerprint") == fingerprint for item in self.comparison_history): return
        result = result.copy(); result["_fingerprint"] = fingerprint
        self.comparison_history.append(result.copy())
        row = self.comparison_table.rowCount(); self.comparison_table.insertRow(row)
        def number(key, digits=6):
            try: return f"{float(result.get(key, '')):.{digits}f}"
            except (TypeError, ValueError): return str(result.get(key, ""))
        score_display = SCORE_METHODS[result["score_method"]].display_name if result.get("score_method") in SCORE_METHODS else result.get("score_name", result.get("score_method", ""))
        threshold_display = THRESHOLD_METHODS[result["threshold_method"]].display_name if result.get("threshold_method") in THRESHOLD_METHODS else result.get("threshold_name", result.get("threshold_method", ""))
        values = (result.get("status", ""), result.get("model", "LSTM AutoEncoder"), result.get("experiment_name", ""), architecture_signature(result), result.get("seed", ""), result.get("optimizer", ""),
                  result.get("learning_rate", ""), result.get("loss", ""), result.get("batch_size", ""), result.get("noise_std", ""), score_display,
                  threshold_display, number("threshold", 8), number("accuracy"), number("balanced_accuracy"), number("precision"), number("recall"), number("f1"),
                  result.get("tn", ""), result.get("fp", ""), result.get("fn", ""), result.get("tp", ""), result.get("best_epoch", ""), result.get("best_val_loss", ""), result.get("training_time", ""), result.get("created_at", ""),
                  result.get("temporal_method", "NONE"), result.get("ewma_alpha", ""), result.get("timestamp_aware", True), result.get("total_error", ""),
                  result.get("detection_delay_samples", ""), result.get("pareto", ""))
        for column, value in enumerate(values): self.comparison_table.setItem(row, column, QTableWidgetItem(str(value)))
        self.comparison_table.item(row, 0).setData(Qt.UserRole, len(self.comparison_history) - 1)
        self._refresh_filter_values(); self.apply_history_filters()
        if hasattr(self, "comparison_summary_label"): self._update_comparison_summary()

    def _refresh_filter_values(self):
        mapping = {"Model":"model", "Status":"status", "Score":"score_method", "Threshold":"threshold_method", "Loss":"loss", "Seed":"seed"}
        for name, key in mapping.items():
            combo = self.history_filters[name]; current = combo.currentData()
            values = sorted({str(item.get(key, "")) for item in self.comparison_history if str(item.get(key, ""))})
            combo.blockSignals(True); combo.clear(); combo.addItem("전체", "ALL")
            for value in values:
                display = SCORE_METHODS[value].display_name if name == "Score" and value in SCORE_METHODS else THRESHOLD_METHODS[value].display_name if name == "Threshold" and value in THRESHOLD_METHODS else value
                combo.addItem(display, value)
            combo.setCurrentIndex(max(0, combo.findData(current))); combo.blockSignals(False)

    def apply_history_filters(self):
        keys = {"Status":"status", "Model":"model", "Seed":"seed", "Loss":"loss", "Score":"score_method", "Threshold":"threshold_method"}
        for row in range(self.comparison_table.rowCount()):
            result_index = self.comparison_table.item(row, 0).data(Qt.UserRole)
            result = self.comparison_history[result_index]
            visible = all(combo.currentData() == "ALL" or str(result.get(keys[name], "")) == combo.currentData() for name, combo in self.history_filters.items())
            self.comparison_table.setRowHidden(row, not visible)

    def clear_comparison(self):
        self.comparison_history.clear(); self.comparison_table.setRowCount(0); self.baseline_result = None; self.champion_result = None; self._update_comparison_summary(); self._log("Comparison history cleared")

    def set_current_as_baseline(self):
        if not self.comparison_history: self._log("Run an evaluation before setting a baseline"); return
        self.baseline_result = self._selected_history_result().copy(); self._update_comparison_summary()

    def set_current_as_champion(self):
        if not self.comparison_history: self._log("Run an evaluation before setting a champion"); return
        self.champion_result = self._selected_history_result().copy(); self._update_comparison_summary()

    def _selected_history_result(self):
        row = self.comparison_table.currentRow()
        if row >= 0 and self.comparison_table.item(row, 0):
            index = self.comparison_table.item(row, 0).data(Qt.UserRole)
            if isinstance(index, int) and 0 <= index < len(self.comparison_history): return self.comparison_history[index]
        return self.comparison_history[-1]

    def _update_comparison_summary(self):
        def label(prefix, result):
            if not result: return f"{prefix}: 없음"
            score = SCORE_METHODS[result['score_method']].display_name if result.get('score_method') in SCORE_METHODS else result.get('score_method', '')
            threshold = THRESHOLD_METHODS[result['threshold_method']].display_name if result.get('threshold_method') in THRESHOLD_METHODS else result.get('threshold_method', '')
            return f"{prefix}: {result.get('model','')} / {result.get('experiment_name','')} / {score} / {threshold} / F1={result.get('f1','')}"
        text = label("현재 최고 실험", self.champion_result) + "\n" + label("기준 실험", self.baseline_result)
        if self.baseline_result and self.comparison_history:
            current = self._selected_history_result()
            def delta(key):
                try: return float(current.get(key,0))-float(self.baseline_result.get(key,0))
                except (TypeError, ValueError): return 0.0
            text += f"\n현재 차이 F1: {delta('f1'):+.6f}  오경보: {delta('fp'):+.0f}  미탐: {delta('fn'):+.0f}"
            metric_names = {"precision":"정밀도", "recall":"재현율", "f1":"F1 점수", "fp":"오경보", "fn":"미탐", "best_val_loss":"최적 검증 손실", "training_time":"학습 시간"}
            rows = [f"{metric_names[key]}: {self.baseline_result.get(key,'')} | {current.get(key,'')} | {delta(key):+.6f}" for key in metric_names]
            self.delta_comparison_label.setText("지표: 기준 | 현재 | 차이\n" + "\n".join(rows))
            self.confusion_comparison_label.setText(f"기준: [[{self.baseline_result.get('tn','')}, {self.baseline_result.get('fp','')}], [{self.baseline_result.get('fn','')}, {self.baseline_result.get('tp','')}]]    선택: [[{current.get('tn','')}, {current.get('fp','')}], [{current.get('fn','')}, {current.get('tp','')}]]")
        self.comparison_summary_label.setText(text)

    def import_historical_result(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Historical Result", str(core.OUTPUT_DIR), "CSV Files (*.csv)")
        if not path: return
        import pandas as pd
        frame = pd.read_csv(path)
        def safe_int(value):
            try: return int(value) if pd.notna(value) else 0
            except (TypeError, ValueError): return 0
        for _, row in frame.iterrows():
            result = {"status":row.get("status", row.get("Status", "HISTORICAL")), "model": row.get("model", row.get("Model", "Historical")), "model_id":row.get("model_id", ""), "experiment_name":row.get("experiment_name", row.get("Experiment", "")), "task_type": row.get("task_type", ""),
                      "cnn_filters":row.get("cnn_filters", ""), "cnn_kernel_size":row.get("cnn_kernel_size", row.get("Kernel", "")), "bottleneck_units":row.get("bottleneck_units", ""), "seed":row.get("seed", ""), "optimizer":row.get("optimizer", ""), "learning_rate":row.get("learning_rate", ""), "loss":row.get("loss", ""), "batch_size":row.get("batch_size", ""), "noise_std":row.get("noise_std", ""),
                      "score_method":row.get("score_method", row.get("Score Method", "")), "threshold_method":row.get("threshold_method", row.get("Threshold Method", "")),
                      "score_name": row.get("score_method", row.get("Score Method", "")), "threshold_name": row.get("threshold_method", row.get("Threshold Method", "")),
                      "threshold": row.get("threshold", ""), "accuracy": row.get("accuracy", ""), "precision": row.get("precision", ""),
                      "temporal_method": row.get("temporal_method", "NONE"), "ewma_alpha": row.get("ewma_alpha", ""),
                      "timestamp_aware": row.get("timestamp_aware", True), "timestamp_gap_threshold": row.get("timestamp_gap_threshold", ""),
                      "total_error": row.get("total_error", ""), "detection_delay_samples": row.get("detection_delay_samples", ""),
                      "detection_delay_seconds": row.get("detection_delay_seconds", ""), "pareto": row.get("pareto", ""),
                      "balanced_accuracy":row.get("balanced_accuracy", ""), "recall": row.get("recall", ""), "f1": row.get("f1", row.get("F1", "")), "tn":safe_int(row.get("tn", row.get("TN", 0))), "fp":safe_int(row.get("fp", row.get("FP", 0))), "fn":safe_int(row.get("fn", row.get("FN", 0))), "tp":safe_int(row.get("tp", row.get("TP", 0))), "best_epoch":row.get("best_epoch", ""), "best_val_loss":row.get("best_val_loss", ""), "training_time":row.get("training_time", ""), "created_at":row.get("created_at", "")}
            self._append_comparison(result)
        self._log(f"Historical result imported: {path}")

    def export_comparison(self):
        if not self.comparison_history:
            QMessageBox.information(self, "Export", "내보낼 Comparison History가 없습니다.")
            return
        output_dir = core.OUTPUT_DIR / "gui_comparisons"; output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
        keys = ["status", "model_id", "model", "experiment_name", "cnn_filters", "cnn_kernel_size", "bottleneck_units", "seed", "optimizer", "learning_rate", "loss", "batch_size", "noise_std",
                "score_method", "threshold_method", "threshold", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "specificity", "fpr", "fnr",
                "tn", "fp", "fn", "tp", "best_epoch", "best_val_loss", "training_time", "created_at",
                "effective_threshold_method", "fallback_used", "fallback_reason", "temporal_method", "ewma_alpha", "timestamp_aware",
                "timestamp_gap_threshold", "total_error", "detection_delay_samples", "detection_delay_seconds", "pareto"]
        with path.open("x", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader()
            for result in self.comparison_history: writer.writerow({key: result.get(key, "") for key in keys})
        self._log(f"Comparison exported: {path}")

    def _evaluation_failed(self, message):
        self._log(message); self._update_evaluation_status("ERROR"); QMessageBox.critical(self, "Evaluation Error", message)

    def _evaluation_thread_finished(self):
        if self.evaluation_worker: self.evaluation_worker.deleteLater()
        if self.evaluation_thread: self.evaluation_thread.deleteLater()
        self.evaluation_worker = None; self.evaluation_thread = None; self._set_evaluation_busy(False)

    def start_training(self):
        quick = self.run_mode.currentData() == "QUICK_TEST"
        try: config = validate_training_config(self._training_config_dict())
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Training Configuration", str(exc)); return
        self.reset_loss_graph()
        self.graph_tab.reset_current({**config, "display_name": self.training_model_combo.currentText()})
        self._set_training_locked(True)
        self.status_label.setText(f"학습 모델: {self.training_model_combo.currentText()}    실행: {'빠른 시험' if quick else '전체 학습'}    상태: 시작 중")
        model_id = self.training_model_combo.currentData()
        self.thread = QThread(self); self.worker = training_engine.TrainingWorker(quick, model_id, config); self.worker.moveToThread(self.thread)
        self.thread.finished.connect(self._thread_finished)
        self.thread.started.connect(self.worker.run); self.worker.status_changed.connect(self.update_status); self.worker.log_message.connect(self._log)
        self.worker.dataset_ready.connect(self.update_dataset); self.worker.model_ready.connect(self.update_model); self.worker.epoch_update.connect(self.update_epoch)
        self.worker.evaluation_ready.connect(self.update_evaluation); self.worker.finished.connect(self.training_finished); self.worker.failed.connect(self.training_failed)
        self.thread.start()

    def stop_training(self):
        if self.worker: self.worker.request_stop(); self._log("Stop requested")

    def update_status(self, status): self.status_label.setText(self.status_label.text().split("상태:")[0] + "상태: " + status)
    def update_dataset(self, data): self.dataset_text.setText("\n".join(f"{k}: {v}" for k, v in data.items())); self._log(f"Dataset ready: {data}")
    def update_model(self, params, trainable): self.params_label.setText(f"전체 파라미터 수: {params}\n학습 파라미터 수: {trainable}")
    @Slot(int, int, float, float, float, float, int, int, float, int)
    def update_epoch(self, epoch, maximum, loss, val_loss, lr, elapsed, since, best_epoch, best_loss, reduce_count):
        previous_lr = getattr(self, "_last_lr", lr); self._last_lr = lr
        self.progress.setMaximum(maximum); self.progress.setValue(epoch); self.loss_canvas.update_data(epoch, loss, val_loss); self.loss_canvas.update_markers(best_epoch, lr < previous_lr)
        values = (("Model", self.training_model_combo.currentData()), ("Experiment Name", self.experiment_name_edit.text().strip() or "AUTO"), ("Epoch", f"{epoch} / {maximum}"), ("Train Loss", f"{loss:.6g}"), ("Validation Loss", f"{val_loss:.6g}"), ("Best Epoch", str(best_epoch)), ("Best Val Loss", f"{best_loss:.6g}"), ("Generalization Gap", f"{val_loss-loss:+.6g}"), ("Learning Rate", f"{lr:.6g}"), ("Elapsed", f"{elapsed:.1f}s"), ("EarlyStopping", f"{since} / {self.early_stopping_patience_spin.value()}"), ("ReduceLR", str(reduce_count)), ("Optimizer", self.optimizer_combo.currentText()), ("Loss", self.loss_combo.currentText()), ("Batch Size", str(self.batch_spin.value())), ("Noise Std", str(self.noise_std_spin.value()) if self.training_model_combo.currentData() == "DENOISING_CNN_LSTM_AUTOENCODER" else "N/A"))
        for key, value in values: self.monitor_labels[key].setText(value)
        self.graph_tab.add_current_epoch(epoch, loss, val_loss, lr, best_epoch, best_loss)
    def update_evaluation(self, data):
        self._log(f"Training baseline evaluation: threshold={data.get('threshold')} accuracy={data.get('accuracy')} f1={data.get('f1_score')}")
    def training_finished(self, status):
        self.graph_tab.finish_current(status, core.QUICK_TEST_EPOCHS if self.run_mode.currentData() == "QUICK_TEST" else self.epochs_spin.value())
        self.graph_tab.refresh_runs()
        self.update_status(status); self._set_training_locked(False); self.refresh_evaluation_runs(); self._finish_thread()
    def training_failed(self, message): QMessageBox.critical(self, "Training Error", message); self.update_status("ERROR"); self._log(message); self._set_training_locked(False); self._finish_thread()
    def _finish_thread(self):
        # Worker completion is emitted before QObject.run returns. Waiting here deadlocks.
        if self.thread and self.thread.isRunning():
            self.thread.quit()

    def _thread_finished(self):
        if self.worker:
            self.worker.deleteLater()
        if self.thread:
            self.thread.deleteLater()
        self.thread = None
        self.worker = None
    def open_outputs(self):
        output_dir = str(core.OUTPUT_DIR)
        if os.name == "nt": os.startfile(output_dir)
        else: self._log(f"Outputs: {output_dir}")

    def closeEvent(self, event):
        if self.worker and self.thread and self.thread.isRunning():
            self._log("Window close requested: stopping training safely")
            self.worker.request_stop()
            self.thread.quit()
            self.thread.wait()
        if self.evaluation_thread and self.evaluation_thread.isRunning():
            self._log("Window close requested: waiting for evaluation worker")
            self.evaluation_thread.quit(); self.evaluation_thread.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    # Do not rely on a stale/hidden window-manager position on Windows.
    window.setGeometry(40, 40, 1280, 820)
    window.show()
    window.showNormal()
    window.raise_()
    window.activateWindow()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
