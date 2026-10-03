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
    QScrollArea, QSplitter, QSizePolicy, QAbstractSpinBox, QListWidget,
    QListWidgetItem, QAbstractItemView)
import tensorflow as tf

import training_engine
import train_lstm_ae as core
from anomaly_scoring import SCORE_METHODS
from threshold_methods import THRESHOLD_METHODS
from evaluation_controller import EvaluationController
from evaluation_worker import EvaluationWorker
from model_registry import MODEL_REGISTRY, PROCESSED_DATASET_MODEL_IDS
from model_artifacts import discover_model_runs, append_evaluation_result
from stage2.segment_detection import METRIC_KEYS, save_segment_detail_csv
from training_graph import TrainingGraphTab
from preprocessing_worker import PreprocessingWorker
from stage2.dataset_artifacts import (save_processed_dataset, validate_dataset_id,
    discover_processed_datasets, load_processed_dataset)
from gui_help import PARAMETER_HELP, format_parameter_help
from training_config import load_training_config as load_config_file, save_training_config as save_config_file, validate_training_config
from preprocessing_config import (DEFAULT_CONFIG_DIR, STAGE2_SEQUENCE_LENGTHS, default_preprocessing_config,
    validate_preprocessing_config, save_preprocessing_config, load_preprocessing_config)


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
        self.preprocessing_config = default_preprocessing_config()
        self.stage2_preprocessing_result = None
        self.stage2_saved_dataset = None
        self._selected_processed_artifact = None
        self.stage2_preprocessing_config_snapshot = None
        self.stage2_preprocessing_paths_snapshot = None
        self.stage2_analysis_snapshot = None
        self.preprocessing_thread = None
        self.preprocessing_worker = None
        self._stage2_busy = False
        self._training_locked = False
        self._stage2_training_dataset = None
        self._stage2_dataset_summary = None
        self._build_ui()
        self._log("Application started")

    def _build_ui(self):
        root = QWidget(); outer = QVBoxLayout(root)
        header = QLabel("KAMP 예지보전 AI 학습 시스템")
        header.setObjectName("header"); outer.addWidget(header)
        self.status_label = QLabel("선택한 모델: CNN_LSTM_AUTOENCODER    상태: 준비")
        outer.addWidget(self.status_label)
        tabs = QTabWidget(); self.tabs = tabs; live = QWidget(); self.live_tab = live; live_layout = QHBoxLayout(live)
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
        system = QGroupBox("시스템 / 데이터 상태"); system_form = QFormLayout(system)
        self.system_labels = {}
        for key, value in (("Python", platform.python_version()), ("TensorFlow", tf.__version__),
                           ("OS", platform.platform()), ("Device", "CPU"), ("GPU", "Not used"),
                           ("Mode", ""), ("Dataset", "-"), ("Segment Gap", ""), ("Signal Transform", ""),
                           ("Scaler", ""), ("Sequence", ""), ("Stride", ""),
                           ("Horizon", ""), ("Input Shape", ""), ("Preprocessing Status", "NOT RUN")):
            label = QLabel(value); self.system_labels[key] = label; system_form.addRow(key, label)
        system.setMaximumHeight(230)
        right_layout.addWidget(system)
        monitor = QGroupBox("학습 상태"); monitor_form = QFormLayout(monitor)
        self.monitor_labels = {}
        monitor_names = {"Model":"모델", "Experiment Name":"실험 이름", "Epoch":"에포크", "Train Loss":"학습 손실", "Validation Loss":"검증 손실", "Best Epoch":"최적 에포크", "Best Val Loss":"최적 검증 손실", "Generalization Gap":"일반화 차이", "Learning Rate":"학습률", "Elapsed":"경과 시간", "EarlyStopping":"조기 종료 대기", "ReduceLR":"학습률 감소 횟수", "Optimizer":"최적화 알고리즘", "Loss":"손실 함수", "Batch Size":"배치 크기", "Noise Std":"노이즈 표준편차"}
        for key in ("Model", "Experiment Name", "Epoch", "Train Loss", "Validation Loss", "Best Epoch", "Best Val Loss", "Generalization Gap", "Learning Rate", "Elapsed", "EarlyStopping", "ReduceLR", "Optimizer", "Loss", "Batch Size", "Noise Std"):
            label = QLabel("-"); self.monitor_labels[key] = label; monitor_form.addRow(monitor_names[key], label)
        self.progress = QProgressBar(); monitor_form.addRow("진행률", self.progress)
        right_layout.addWidget(monitor)
        self.log_view = QPlainTextEdit(); self.log_view.setReadOnly(True); self.log_view.setMaximumBlockCount(1000)
        right_layout.addWidget(self.log_view, 1)
        model_box = QGroupBox("모델 설정"); model_layout = QVBoxLayout(model_box)
        data_mode_row = QHBoxLayout()
        self.data_mode_combo = QComboBox()
        self.data_mode_combo.addItem("KAMP BASELINE", "KAMP_BASELINE")
        self.data_mode_combo.addItem("PROCESSED DATASET", "PROCESSED_DATASET")
        data_mode_row.addWidget(QLabel("DATA MODE")); data_mode_row.addWidget(self.data_mode_combo, 1)
        model_layout.addLayout(data_mode_row)
        dataset_row = QHBoxLayout()
        self.processed_dataset_combo = QComboBox()
        self.refresh_processed_datasets_button = QPushButton("Refresh")
        dataset_row.addWidget(QLabel("Processed Dataset")); dataset_row.addWidget(self.processed_dataset_combo, 1)
        dataset_row.addWidget(self.refresh_processed_datasets_button)
        model_layout.addLayout(dataset_row)
        self.processed_dataset_summary_label = QLabel("KAMP baseline data path")
        self.processed_dataset_summary_label.setWordWrap(True)
        model_layout.addWidget(self.processed_dataset_summary_label)
        self.training_model_combo = QComboBox()
        for spec in MODEL_REGISTRY.values():
            if spec.status == "ACTIVE": self.training_model_combo.addItem(spec.display_name, spec.id)
        self.training_model_combo.setCurrentIndex(self.training_model_combo.findData("CNN_LSTM_AUTOENCODER"))
        self.training_model_combo.currentIndexChanged.connect(self.update_training_model)
        model_layout.addWidget(QLabel("학습 모델")); model_layout.addWidget(self.training_model_combo)
        self.data_mode_combo.currentIndexChanged.connect(self._data_mode_changed)
        self.processed_dataset_combo.currentIndexChanged.connect(self._processed_dataset_changed)
        self.refresh_processed_datasets_button.clicked.connect(self.refresh_processed_datasets)
        self.processed_dataset_combo.setEnabled(False)
        self.refresh_processed_datasets_button.setEnabled(False)
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
        self.noise_clip_policy_label = QLabel()
        self.noise_clip_policy_label.setWordWrap(True)
        denoising_form.addRow(self.noise_clip_policy_label)
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
        self._training_editable_widgets = [self.data_mode_combo, self.processed_dataset_combo,
                                           self.refresh_processed_datasets_button,
                                           self.training_model_combo, self.preset_combo, self.experiment_name_edit, *editable,
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
        self._build_preprocessing_tab(tabs)
        tabs.addTab(live, "실시간 학습")
        self.graph_tab = TrainingGraphTab()
        tabs.addTab(self.graph_tab, "학습 그래프")
        tabs.currentChanged.connect(lambda index: self.graph_tab.render() if tabs.widget(index) is self.graph_tab else None)
        data_tab = QWidget(); data_layout = QFormLayout(data_tab); self.dataset_text = QLabel("실행 기록 없음"); self.dataset_text.setWordWrap(True); data_layout.addRow("데이터셋", self.dataset_text); tabs.addTab(data_tab, "데이터")
        eval_tab = QWidget(); eval_layout = QVBoxLayout(eval_tab)
        self.evaluation_pages = QTabWidget()
        eval_layout.addWidget(self.evaluation_pages)
        settings_page = QWidget(); settings_layout = QVBoxLayout(settings_page)
        results_page = QWidget(); results_layout = QVBoxLayout(results_page)
        self.evaluation_pages.addTab(settings_page, "평가 설정·설명")
        self.evaluation_pages.addTab(results_page, "결과·기록")
        selectors = QGroupBox("평가 설정"); selector_form = QFormLayout(selectors)
        self.evaluation_model_combo = QComboBox()
        for spec in MODEL_REGISTRY.values(): self.evaluation_model_combo.addItem(spec.display_name, spec.id)
        self.evaluation_run_combo = QComboBox(); self.evaluation_model_combo.currentIndexChanged.connect(self.refresh_evaluation_runs)
        self.evaluation_run_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.evaluation_run_combo.setMinimumContentsLength(24)
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
        self.evaluation_gap_label = QLabel()
        selector_form.addRow("Segment Gap Threshold", self.evaluation_gap_label)
        self.evaluation_dataset_label = QLabel("DATA MODE: KAMP BASELINE")
        self.evaluation_dataset_label.setWordWrap(True)
        selector_form.addRow("평가 데이터", self.evaluation_dataset_label)
        selector_form.addRow("임계값 방식", self.threshold_combo)
        self.eval_status_label = QLabel("모델: LSTM AutoEncoder    점수: 마지막 시점 MSE    임계값: 정밀도-재현율 균형점    상태: 미불러옴")
        self.eval_status_label.setWordWrap(True)
        self.eval_status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        selector_form.addRow("현재 선택", self.eval_status_label)
        self.loaded_architecture_label = QLabel("-")
        selector_form.addRow("불러온 모델 구조", self.loaded_architecture_label)
        settings_layout.addWidget(selectors)
        auto_box = QGroupBox("AUTO Sequence Fallback (오프라인 Segment 평가)")
        auto_layout = QVBoxLayout(auto_box)
        self.auto_pool_list = QListWidget()
        self.auto_pool_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.auto_pool_list.setMaximumHeight(110)
        auto_layout.addWidget(QLabel("같은 모델 종류와 전처리 조건으로 학습한 Seq별 실행을 2개 이상 선택하세요."))
        auto_layout.addWidget(self.auto_pool_list)
        self.refresh_auto_pool_button = QPushButton("저장 모델 목록 새로고침")
        self.refresh_auto_pool_button.clicked.connect(self.refresh_auto_pool_runs)
        self.run_auto_button = QPushButton("AUTO Sequence 평가")
        self.run_auto_button.clicked.connect(self.run_auto_evaluation)
        auto_actions = QHBoxLayout()
        auto_actions.addWidget(self.refresh_auto_pool_button)
        auto_actions.addWidget(self.run_auto_button)
        auto_layout.addLayout(auto_actions)
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
        execution_actions = QGridLayout()
        for index, button in enumerate((self.load_model_button, self.run_evaluation_button, self.compare_all_button, self.sweep_ewma_button, self.compare_models_button)):
            execution_actions.addWidget(button, index // 3, index % 3)
        settings_layout.addLayout(execution_actions)
        settings_layout.addWidget(auto_box)
        record_actions = QGridLayout()
        for index, button in enumerate((self.clear_comparison_button, self.export_comparison_button, self.import_history_button, self.set_baseline_button, self.set_champion_button, self.show_history_graph_button)):
            record_actions.addWidget(button, index // 3, index % 3)
        results_layout.addLayout(record_actions)
        description_box = QGroupBox("평가 방식 설명"); description_layout = QVBoxLayout(description_box)
        self.algorithm_description = QLabel(); self.algorithm_description.setWordWrap(True); description_layout.addWidget(self.algorithm_description); settings_layout.addWidget(description_box)
        self.lock_evaluation_method = QCheckBox("동일 조건 비교: 점수·임계값 방식 잠금"); self.lock_evaluation_method.toggled.connect(lambda locked: (self.score_combo.setEnabled(not locked), self.threshold_combo.setEnabled(not locked))); settings_layout.addWidget(self.lock_evaluation_method)
        settings_layout.addStretch()
        result_box = QGroupBox("평가 결과"); result_layout = QGridLayout(result_box)
        self.eval_result_labels = {}
        metric_names = {"Threshold":"임계값", "Accuracy":"정확도 (Accuracy)", "Balanced Accuracy":"균형 정확도", "Precision":"정밀도 (Precision)", "Recall":"재현율 (Recall)", "F1 Score":"F1 점수", "Specificity":"특이도", "FPR":"오경보율 (FPR)", "FNR":"미탐율 (FNR)", "TN":"정상 정확 판정 (TN)", "FP (False Alarm)":"오경보 (FP)", "FN (Missed Anomaly)":"미탐 (FN)", "TP":"이상 정확 탐지 (TP)", "Fallback":"대체 방식"}
        result_keys = ("Threshold", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1 Score", "Specificity", "FPR", "FNR", "TN", "FP (False Alarm)", "FN (Missed Anomaly)", "TP", "Fallback")
        for index, key in enumerate(result_keys):
            label = QLabel("-"); self.eval_result_labels[key] = label
            result_layout.addWidget(QLabel(metric_names[key]), index // 2, (index % 2) * 2); result_layout.addWidget(label, index // 2, (index % 2) * 2 + 1)
        results_layout.addWidget(result_box)
        segment_box = QGroupBox("Segment-level Detection / Segment-relative Detection Delay (seconds)")
        segment_layout = QGridLayout(segment_box)
        self.segment_result_labels = {}
        segment_names = (
            ("anomaly_segments_total", "Anomaly Test Segments"),
            ("evaluable_anomaly_segments", "Evaluable Segments"),
            ("non_evaluable_anomaly_segments", "Non-evaluable Segments"),
            ("detected_segments", "Detected Segments"), ("missed_segments", "Missed Segments"),
            ("segment_coverage", "Segment Coverage"),
            ("segment_detection_rate", "Segment Detection Rate"),
            ("mean_segment_delay_seconds", "Mean Delay From Start (seconds)"),
            ("median_segment_delay_seconds", "Median Delay From Start (seconds)"),
            ("max_segment_delay_seconds", "Max Delay From Start (seconds)"),
            ("mean_post_evaluable_delay_seconds", "Mean Delay After Evaluable (seconds)"),
            ("median_post_evaluable_delay_seconds", "Median Delay After Evaluable (seconds)"),
            ("max_post_evaluable_delay_seconds", "Max Delay After Evaluable (seconds)"),
        )
        for index, (key, title) in enumerate(segment_names):
            label = QLabel("-"); self.segment_result_labels[key] = label
            segment_layout.addWidget(QLabel(title), index // 2, (index % 2) * 2)
            segment_layout.addWidget(label, index // 2, (index % 2) * 2 + 1)
        results_layout.addWidget(segment_box)
        self.comparison_summary_label = QLabel("현재 최고 실험: 없음\n기준 실험: 없음"); self.comparison_summary_label.setWordWrap(True); results_layout.addWidget(self.comparison_summary_label)
        history_box = QGroupBox("실험 기록"); history_layout = QVBoxLayout(history_box)
        self.history_filters = {}
        filter_layout = QHBoxLayout()
        for name in ("Model", "Status", "Score", "Threshold", "Loss", "Seed"):
            combo = QComboBox(); combo.addItem("전체", "ALL"); combo.currentIndexChanged.connect(self.apply_history_filters); self.history_filters[name] = combo
            filter_layout.addWidget(QLabel({"Model":"모델", "Status":"상태", "Score":"점수", "Threshold":"임계값 방식", "Loss":"손실 함수", "Seed":"시드"}[name])); filter_layout.addWidget(combo)
        history_layout.addLayout(filter_layout)
        columns = ["상태", "모델", "실험", "Architecture", "시드", "최적화", "학습률", "손실 함수", "배치", "노이즈 표준편차", "점수", "임계값 방식", "임계값", "정확도", "균형 정확도", "정밀도", "재현율", "F1", "정상 정확 판정 (TN)", "오경보 (FP)", "미탐 (FN)", "이상 정확 탐지 (TP)", "최적 에포크", "최적 검증 손실", "학습 시간", "생성 시각", "Temporal", "Alpha", "Timestamp Reset", "Total Error", "Delay", "Pareto", "Sequence", "Segment Coverage", "Segment Detection Rate", "Median Segment-relative Delay (seconds)"]
        self.comparison_table = QTableWidget(0, len(columns)); self.comparison_table.setHorizontalHeaderLabels(columns)
        history_layout.addWidget(self.comparison_table); results_layout.addWidget(history_box, 1)
        self.delta_comparison_label = QLabel("기준 실험과 현재 실험을 선택하면 차이를 표시합니다."); self.delta_comparison_label.setWordWrap(True); results_layout.addWidget(self.delta_comparison_label)
        self.confusion_comparison_label = QLabel("기준 혼동행렬: -    선택한 혼동행렬: -"); results_layout.addWidget(self.confusion_comparison_label)
        warning = QLabel("Test 지표는 최종 성능 확인용입니다. Test F1을 반복해서 보고 파라미터를 고르면 Test 데이터에 간접 과적합될 수 있습니다. 현재 분할은 KAMP 호환 방식이며 STRICT_TIME_SPLIT 검증이 추가로 필요합니다. 같은 시드를 써도 환경에 따라 완전히 같은 결과는 보장되지 않습니다.\n다음 권장 실험: Denoising CNN-LSTM에서 CNN 기준과 같은 시드·설정을 유지하고 Gaussian 노이즈 표준편차만 0.01로 변경하세요. MAHALANOBIS_ERROR + POT_1PCT 방식으로 비교하되 각 모델의 임계값 숫자는 검증 데이터에서 따로 계산합니다."); warning.setWordWrap(True); results_layout.addWidget(warning)
        tabs.addTab(eval_tab, "평가 및 비교"); self._refresh_preprocessing_display(); self.update_training_model(); self.refresh_evaluation_runs(); self.update_algorithm_description()
        outer.addWidget(tabs, 1)
        self.setCentralWidget(root)
        self.setStyleSheet("QMainWindow,QWidget{background:#111820;color:#e7eef5} QGroupBox{background:#16222d;border:1px solid #36536b;margin-top:8px;padding:8px} QGroupBox::title{color:#78c7e8} QLabel,QCheckBox{background:transparent;color:#e7eef5} QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox{background:#1d2d3b;color:#e7eef5;border:1px solid #4b7087;padding:3px;min-height:22px} QComboBox QAbstractItemView{background:#1d2d3b;color:#e7eef5} QScrollArea{background:#111820;border:0} QScrollBar:vertical{background:#14222e;width:12px} QScrollBar::handle:vertical{background:#52758a;min-height:24px} #header{font-size:22px;font-weight:bold;color:#8ad8f5;padding:8px} #experimentSummary{background:#203645;padding:8px;border:1px solid #4b7087} QPlainTextEdit{background:#0b1117;color:#e7eef5;font-family:Consolas} QPushButton{padding:8px;background:#1e536d;color:#e7eef5;border:1px solid #54a7c6}")

    def _log(self, message): self.log_view.appendPlainText(message)

    def _build_preprocessing_tab(self, tabs):
        page = QWidget(); layout = QVBoxLayout(page)
        self.preprocessing_widgets = []
        source_box = QGroupBox("데이터 원본"); source_form = QFormLayout(source_box)
        self.normal_csv_edit = QLineEdit(str(core.NORMAL_PATH))
        self.anomaly_csv_edit = QLineEdit(str(core.OUTLIER_PATH))
        self.normal_csv_browse = QPushButton("찾아보기")
        self.anomaly_csv_browse = QPushButton("찾아보기")
        self.normal_csv_browse.clicked.connect(lambda: self._browse_stage2_csv(self.normal_csv_edit))
        self.anomaly_csv_browse.clicked.connect(lambda: self._browse_stage2_csv(self.anomaly_csv_edit))
        for label, edit, button in (("정상 데이터 CSV", self.normal_csv_edit, self.normal_csv_browse),
                                    ("이상 데이터 CSV", self.anomaly_csv_edit, self.anomaly_csv_browse)):
            row = QWidget(); row_layout = QHBoxLayout(row); row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(edit, 1); row_layout.addWidget(button)
            source_form.addRow(label, row)
        self.preprocessing_widgets.extend((self.normal_csv_edit, self.anomaly_csv_edit,
                                           self.normal_csv_browse, self.anomaly_csv_browse))
        layout.addWidget(source_box)
        mode_box = QGroupBox("전처리 모드"); mode_form = QFormLayout(mode_box)
        self.preprocessing_mode_combo = QComboBox()
        self.preprocessing_mode_combo.addItem("Stage 2 구간 인식", "STAGE2_SEGMENT_AWARE")
        self.preprocessing_mode_combo.addItem("KAMP 기준선 (고정)", "KAMP_BASELINE")
        mode_form.addRow("모드", self.preprocessing_mode_combo)
        layout.addWidget(mode_box)
        self.preprocessing_widgets.append(self.preprocessing_mode_combo)
        time_box = QGroupBox("시간축"); time_form = QFormLayout(time_box)
        self.expected_interval_spin = QSpinBox(); self.expected_interval_spin.setRange(1, 100000); self.expected_interval_spin.setSuffix(" ms")
        self.gap_threshold_spin = QSpinBox(); self.gap_threshold_spin.setRange(2, 100000); self.gap_threshold_spin.setSuffix(" ms")
        self.remove_duplicates_check = QCheckBox("사용")
        self.segment_aware_check = QCheckBox("사용")
        for label, widget in (("예상 측정 간격", self.expected_interval_spin), ("기록 구간 분리 기준", self.gap_threshold_spin),
                              ("완전 중복 행 제거", self.remove_duplicates_check), ("구간 인식 전처리", self.segment_aware_check)):
            time_form.addRow(label, widget); self.preprocessing_widgets.append(widget)
        layout.addWidget(time_box)
        signal_box = QGroupBox("신호 변환 및 스케일링"); signal_form = QFormLayout(signal_box)
        self.signal_transform_combo = QComboBox()
        for label, value in (("전체 절댓값", "ABS_ALL"), ("원본 부호 유지", "RAW_SIGNED"),
                             ("진동 절댓값 + 전류 원본", "ABS_VIBRATION_RAW_CURRENT")):
            self.signal_transform_combo.addItem(label, value)
        self.scaler_combo = QComboBox()
        for label, value in (("최소·최대 스케일링", "MINMAX"), ("표준화", "STANDARD")):
            self.scaler_combo.addItem(label, value)
        signal_form.addRow("신호 변환", self.signal_transform_combo)
        signal_form.addRow("스케일링", self.scaler_combo)
        signal_form.addRow("스케일러 학습 데이터", QLabel("정상 학습 데이터만"))
        self.preprocessing_widgets.extend((self.signal_transform_combo, self.scaler_combo))
        layout.addWidget(signal_box)
        window_box = QGroupBox("관측 윈도우"); window_form = QFormLayout(window_box)
        self.sequence_combo = QComboBox()
        for length in STAGE2_SEQUENCE_LENGTHS: self.sequence_combo.addItem(str(length), length)
        self.stride_spin = QSpinBox(); self.stride_spin.setRange(1, 100000)
        self.use_horizon_check = QCheckBox("사용 안 함")
        self.baseline_horizon_spin = QSpinBox(); self.baseline_horizon_spin.setRange(1, 100000)
        for label, widget in (("시퀀스 길이", self.sequence_combo), ("이동 간격", self.stride_spin),
                              ("예측 간격 사용", self.use_horizon_check), ("KAMP 기준 예측 간격", self.baseline_horizon_spin)):
            window_form.addRow(label, widget)
        self.preprocessing_widgets.extend((self.sequence_combo, self.stride_spin, self.use_horizon_check, self.baseline_horizon_spin))
        layout.addWidget(window_box)
        segments_box = QGroupBox("기록 구간 분석"); segments_form = QFormLayout(segments_box)
        self.normal_segments_label = QLabel("-"); self.anomaly_segments_label = QLabel("-")
        segments_form.addRow("정상 구간 수", self.normal_segments_label)
        segments_form.addRow("이상 구간 수", self.anomaly_segments_label)
        layout.addWidget(segments_box)
        self.analyze_data_button = QPushButton("데이터 분석")
        self.analyze_data_button.clicked.connect(lambda: self._start_stage2_action("analyze"))
        self.run_preprocessing_button = QPushButton("전처리 실행")
        self.run_preprocessing_button.clicked.connect(lambda: self._start_stage2_action("preprocess"))
        execution = QHBoxLayout(); execution.addWidget(self.analyze_data_button); execution.addWidget(self.run_preprocessing_button)
        layout.addLayout(execution)
        self.preprocessing_widgets.extend((self.analyze_data_button, self.run_preprocessing_button))
        self.stage2_status_label = QLabel("준비")
        layout.addWidget(self.stage2_status_label)
        self.stage2_cross_gap_label = QLabel("구간 경계 횡단 윈도우: -")
        self.stage2_cross_gap_label.setStyleSheet("font-weight:bold;color:#8ad8f5")
        self.stage2_total_windows_label = QLabel("전체 윈도우 (정상 / 이상): - / -")
        layout.addWidget(self.stage2_cross_gap_label)
        layout.addWidget(self.stage2_total_windows_label)
        analysis_box = QGroupBox("데이터 품질 분석"); analysis_layout = QVBoxLayout(analysis_box)
        self.stage2_analysis_text = QPlainTextEdit(); self.stage2_analysis_text.setReadOnly(True)
        self.stage2_analysis_text.setPlainText("데이터 분석을 실행하세요.")
        self.stage2_analysis_text.setMinimumHeight(160)
        analysis_layout.addWidget(self.stage2_analysis_text); layout.addWidget(analysis_box)
        preview_box = QGroupBox("전처리 결과 미리보기"); preview_layout = QVBoxLayout(preview_box)
        self.stage2_preview_text = QPlainTextEdit(); self.stage2_preview_text.setReadOnly(True)
        self.stage2_preview_text.setPlainText("전처리를 실행하세요.")
        self.stage2_preview_text.setMinimumHeight(230)
        preview_layout.addWidget(self.stage2_preview_text); layout.addWidget(preview_box)
        dataset_box = QGroupBox("처리된 데이터셋 저장"); dataset_form = QFormLayout(dataset_box)
        self._suggested_dataset_id = "stage2_abs_minmax_seq20_gap150"
        self.dataset_id_edit = QLineEdit(self._suggested_dataset_id)
        self.dataset_id_edit.setPlaceholderText("영문자, 숫자, _, -")
        self.save_processed_dataset_button = QPushButton("처리된 데이터셋 저장")
        self.save_processed_dataset_button.setEnabled(False)
        self.save_processed_dataset_button.clicked.connect(self._save_processed_dataset)
        self.stage2_saved_dataset_label = QLabel("저장된 데이터셋: -")
        self.stage2_saved_dataset_label.setWordWrap(True)
        dataset_form.addRow("데이터셋 ID", self.dataset_id_edit)
        dataset_form.addRow(self.save_processed_dataset_button)
        dataset_form.addRow(self.stage2_saved_dataset_label)
        layout.addWidget(dataset_box)
        self.preprocessing_widgets.append(self.dataset_id_edit)
        self.dataset_id_edit.textChanged.connect(self._refresh_stage2_save_enabled)
        actions = QHBoxLayout()
        self.save_preprocessing_button = QPushButton("전처리 설정 저장"); self.save_preprocessing_button.clicked.connect(self.save_preprocessing_settings)
        self.load_preprocessing_button = QPushButton("전처리 설정 불러오기"); self.load_preprocessing_button.clicked.connect(self.load_preprocessing_settings)
        actions.addWidget(self.save_preprocessing_button); actions.addWidget(self.load_preprocessing_button)
        self.preprocessing_widgets.extend((self.save_preprocessing_button, self.load_preprocessing_button))
        layout.addLayout(actions); layout.addStretch()
        self.preprocessing_status_label = QLabel("전처리 결과를 저장한 뒤 학습 화면에서 데이터셋을 선택하세요.")
        layout.addWidget(self.preprocessing_status_label)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(page)
        self.preprocessing_tab = scroll; tabs.addTab(scroll, "데이터 전처리")
        self._apply_preprocessing_config(self.preprocessing_config)
        for widget in self.preprocessing_widgets:
            if widget is self.preprocessing_mode_combo: widget.currentIndexChanged.connect(self._preprocessing_mode_changed)
            elif isinstance(widget, QComboBox): widget.currentIndexChanged.connect(self._preprocessing_changed)
            elif isinstance(widget, QSpinBox): widget.valueChanged.connect(self._preprocessing_changed)
            elif isinstance(widget, QCheckBox): widget.toggled.connect(self._preprocessing_changed)
        self.normal_csv_edit.textChanged.connect(self._stage2_inputs_changed)
        self.anomaly_csv_edit.textChanged.connect(self._stage2_inputs_changed)

    def _preprocessing_config_from_ui(self):
        return validate_preprocessing_config({"mode": self.preprocessing_mode_combo.currentData(),
            "expected_interval_ms": self.expected_interval_spin.value(), "gap_threshold_ms": self.gap_threshold_spin.value(),
            "remove_exact_duplicates": self.remove_duplicates_check.isChecked(), "segment_aware": self.segment_aware_check.isChecked(),
            "signal_transform": self.signal_transform_combo.currentData(), "scaler": self.scaler_combo.currentData(),
            "sequence_length": self.sequence_combo.currentData(), "stride": self.stride_spin.value(),
            "use_horizon": self.use_horizon_check.isChecked(), "prediction_horizon": self.baseline_horizon_spin.value()})

    def _apply_preprocessing_config(self, config):
        config = validate_preprocessing_config(config)
        for widget in self.preprocessing_widgets:
            widget.blockSignals(True)
        self.preprocessing_mode_combo.setCurrentIndex(self.preprocessing_mode_combo.findData(config["mode"]))
        self.expected_interval_spin.setValue(config["expected_interval_ms"])
        self.gap_threshold_spin.setValue(config["gap_threshold_ms"])
        self.remove_duplicates_check.setChecked(config["remove_exact_duplicates"])
        self.segment_aware_check.setChecked(config["segment_aware"])
        self.signal_transform_combo.setCurrentIndex(self.signal_transform_combo.findData(config["signal_transform"]))
        self.scaler_combo.setCurrentIndex(self.scaler_combo.findData(config["scaler"]))
        self.sequence_combo.setCurrentIndex(self.sequence_combo.findData(config["sequence_length"]))
        self.stride_spin.setValue(config["stride"])
        self.use_horizon_check.setChecked(config["use_horizon"])
        self.baseline_horizon_spin.setValue(config["prediction_horizon"])
        for widget in self.preprocessing_widgets:
            widget.blockSignals(False)
        self.preprocessing_config = config
        self._update_dataset_id_suggestion()
        self._set_preprocessing_mode_editable()
        self._refresh_preprocessing_display()
        self._stage2_inputs_changed()

    def _set_preprocessing_mode_editable(self):
        editable = self.preprocessing_config["mode"] == "STAGE2_SEGMENT_AWARE"
        for widget in self.preprocessing_widgets:
            if widget not in (self.preprocessing_mode_combo, self.save_preprocessing_button,
                              self.load_preprocessing_button, self.normal_csv_edit, self.anomaly_csv_edit,
                              self.normal_csv_browse, self.anomaly_csv_browse,
                              self.analyze_data_button, self.run_preprocessing_button):
                widget.setEnabled(editable)

    def _preprocessing_mode_changed(self, *_):
        self._apply_preprocessing_config(default_preprocessing_config(self.preprocessing_mode_combo.currentData()))

    def _preprocessing_changed(self, *_):
        try:
            self.preprocessing_config = self._preprocessing_config_from_ui()
        except ValueError as exc:
            self.preprocessing_status_label.setText(str(exc))
            self.save_preprocessing_button.setEnabled(False)
            self._stage2_inputs_changed()
            return
        self.preprocessing_status_label.setText("전처리 결과를 저장한 뒤 학습 화면에서 데이터셋을 선택하세요.")
        self.save_preprocessing_button.setEnabled(True)
        self._update_dataset_id_suggestion()
        self._refresh_preprocessing_display()
        self._stage2_inputs_changed()

    def _update_dataset_id_suggestion(self):
        if not hasattr(self, "dataset_id_edit"):
            return
        config = self.preprocessing_config
        signal = {"ABS_ALL": "abs", "RAW_SIGNED": "raw",
                  "ABS_VIBRATION_RAW_CURRENT": "abs_vibration_raw_current"}[config["signal_transform"]]
        suggested = (f"stage2_{signal}_{config['scaler'].lower()}_"
                     f"seq{config['sequence_length']}_gap{config['gap_threshold_ms']}")
        if not self.dataset_id_edit.text().strip() or self.dataset_id_edit.text() == self._suggested_dataset_id:
            self.dataset_id_edit.setText(suggested)
        self._suggested_dataset_id = suggested

    def _refresh_preprocessing_display(self):
        config = self.preprocessing_config
        if not hasattr(self, "system_labels"): return
        artifact = (self._selected_processed_artifact if hasattr(self, "data_mode_combo") and
                    self.data_mode_combo.currentData() == "PROCESSED_DATASET" else None)
        display = artifact.config["preprocessing"] if artifact is not None else config
        values = {"Mode": "PROCESSED DATASET" if artifact is not None else config["mode"],
                  "Dataset": artifact.dataset_id if artifact is not None else "-",
                  "Segment Gap": f'{display["gap_threshold_ms"]} ms',
                  "Signal Transform": display["signal_transform"], "Scaler": display["scaler"],
                  "Sequence": str(display["sequence_length"]), "Stride": str(display["stride"]),
                  "Horizon": str(display["prediction_horizon"]) if display["use_horizon"] else "OFF",
                  "Input Shape": f'({display["sequence_length"]}, {len(core.FEATURES)})'}
        for key, value in values.items(): self.system_labels[key].setText(value)
        self.use_horizon_check.setText("사용" if config["use_horizon"] else "사용 안 함")
        if hasattr(self, "evaluation_gap_label"):
            evaluation_config = (self.evaluation_controller.preprocessing_config
                if self.evaluation_controller.is_loaded and
                   (getattr(self.evaluation_controller.model_run, "metadata", {}) or {}).get("data_mode") == "PROCESSED_DATASET"
                else config)
            self.evaluation_gap_label.setText(f'{evaluation_config["gap_threshold_ms"]} ms')
        if hasattr(self, "training_model_combo") and hasattr(self, "architecture_label"):
            self._update_model_description_only()

    def _browse_stage2_csv(self, edit):
        path, _ = QFileDialog.getOpenFileName(self, "Stage 2 CSV 선택", edit.text(), "CSV (*.csv)")
        if path:
            edit.setText(path)

    def _stage2_snapshot(self):
        config = self._preprocessing_config_from_ui()
        paths = (self.normal_csv_edit.text().strip(), self.anomaly_csv_edit.text().strip())
        if not all(paths):
            raise ValueError("Normal CSV와 Anomaly CSV 경로를 입력하세요")
        return {"config": config, "normal_path": str(Path(paths[0]).resolve()),
                "anomaly_path": str(Path(paths[1]).resolve())}

    @staticmethod
    def _stage2_analysis_key(snapshot):
        config = snapshot["config"]
        return (snapshot["normal_path"], snapshot["anomaly_path"],
                *(config[key] for key in ("mode", "expected_interval_ms", "gap_threshold_ms",
                                         "remove_exact_duplicates", "segment_aware")))

    def _stage2_inputs_changed(self, *_):
        if not hasattr(self, "stage2_status_label"):
            return
        try: current = self._stage2_snapshot()
        except ValueError: current = None
        if self.stage2_analysis_snapshot is not None and (
                current is None or self._stage2_analysis_key(current) != self._stage2_analysis_key(self.stage2_analysis_snapshot)):
            self.stage2_analysis_snapshot = None
            self.normal_segments_label.setText("-")
            self.anomaly_segments_label.setText("-")
            self.stage2_analysis_text.setPlainText("결과 만료 - 데이터 또는 구간 설정이 변경되었습니다. 다시 분석하세요.")
            if self.stage2_preprocessing_result is None:
                self.stage2_status_label.setText("결과 만료 - 재분석 필요")
        if self.stage2_preprocessing_result is not None and (
                current is None or current["config"] != self.stage2_preprocessing_config_snapshot or
                (current["normal_path"], current["anomaly_path"]) != self.stage2_preprocessing_paths_snapshot):
            self.stage2_preprocessing_result = None
            self.stage2_preview_text.setPlainText("결과 만료 - 입력 또는 설정이 변경되었습니다. 전처리를 다시 실행하세요.")
            self.stage2_status_label.setText("결과 만료 - 전처리 재실행 필요")
            self.stage2_cross_gap_label.setText("구간 경계 횡단 윈도우: -")
            self.stage2_cross_gap_label.setStyleSheet("font-weight:bold;color:#8ad8f5")
            self.stage2_total_windows_label.setText("전체 윈도우 (정상 / 이상): - / -")
            self.system_labels["Preprocessing Status"].setText("STALE")
            self._stage2_dataset_summary = None
            self._refresh_dataset_text()
        self._refresh_stage2_save_enabled()

    def _refresh_stage2_save_enabled(self, *_):
        if not hasattr(self, "save_processed_dataset_button"):
            return
        try:
            validate_dataset_id(self.dataset_id_edit.text().strip())
            current = self._stage2_snapshot()
            matching = (current["config"] == self.stage2_preprocessing_config_snapshot and
                        (current["normal_path"], current["anomaly_path"]) == self.stage2_preprocessing_paths_snapshot)
        except ValueError:
            matching = False
        training_busy = self._training_locked or (self.thread is not None and self.thread.isRunning())
        result = self.stage2_preprocessing_result
        self.save_processed_dataset_button.setEnabled(bool(
            result is not None and matching and
            result.summary["cross_gap_windows"] == 0 and not self._stage2_busy and not training_busy))

    def _save_processed_dataset(self):
        self._refresh_stage2_save_enabled()
        if not self.save_processed_dataset_button.isEnabled():
            return
        dataset_id = self.dataset_id_edit.text().strip()
        self.save_processed_dataset_button.setEnabled(False)
        try:
            artifact = save_processed_dataset(self.stage2_preprocessing_result, dataset_id)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, "처리된 데이터셋 저장 오류", str(exc))
            self._log(f"Processed dataset save error: {exc}")
        else:
            self.stage2_saved_dataset = artifact
            self.stage2_saved_dataset_label.setText(
                f"저장된 데이터셋\n데이터셋 ID: {artifact.dataset_id}\n상태: 저장 완료\n"
                f"생성 시각: {artifact.config['created_at']}\n저장 위치: {artifact.path}")
            self._log(f"Processed dataset saved. ID: {artifact.dataset_id} | Path: {artifact.path}")
            self._log("학습 화면에서 처리된 데이터셋 모드와 이 데이터셋 ID를 선택하세요.")
        finally:
            self._refresh_stage2_save_enabled()

    def _set_stage2_busy(self, busy):
        self._stage2_busy = busy
        training_busy = self.thread is not None and self.thread.isRunning()
        for widget in self.preprocessing_widgets:
            widget.setEnabled(not busy and not training_busy)
        if not busy and not training_busy:
            self._set_preprocessing_mode_editable()
            try: self._stage2_snapshot()
            except ValueError: self.save_preprocessing_button.setEnabled(False)
        if not training_busy:
            self.start_button.setEnabled(not busy)
        self._refresh_stage2_save_enabled()

    def _start_stage2_action(self, action):
        if self._stage2_busy or (self.preprocessing_thread and self.preprocessing_thread.isRunning()):
            return
        if self.thread and self.thread.isRunning():
            QMessageBox.information(self, "Stage 2", "학습이 끝난 뒤 전처리를 실행하세요.")
            return
        try: snapshot = self._stage2_snapshot()
        except ValueError as exc:
            self._stage2_failed(f"ValueError: {exc}", action)
            return
        if snapshot["config"]["mode"] != "STAGE2_SEGMENT_AWARE":
            self._stage2_failed("ValueError: Stage 2 모드를 선택하세요.", action)
            return
        self._stage2_active_snapshot = snapshot
        self._stage2_active_action = action
        if action == "preprocess":
            self.stage2_preprocessing_result = None
            self._stage2_dataset_summary = None
            self._refresh_dataset_text()
            self.system_labels["Preprocessing Status"].setText("PREPROCESSING")
            self.stage2_cross_gap_label.setText("구간 경계 횡단 윈도우: -")
            self.stage2_cross_gap_label.setStyleSheet("font-weight:bold;color:#8ad8f5")
            self.stage2_total_windows_label.setText("전체 윈도우 (정상 / 이상): - / -")
        self._set_stage2_busy(True)
        self.stage2_status_label.setText("데이터 분석 중..." if action == "analyze" else "전처리 중...")
        self._log(f"Stage2 {'analysis' if action == 'analyze' else 'preprocessing'} started")
        self._log(f"Normal path: {snapshot['normal_path']}")
        self._log(f"Anomaly path: {snapshot['anomaly_path']}")
        config = snapshot["config"]
        self._log(f"Preprocessing config: Gap={config['gap_threshold_ms']}ms Signal={config['signal_transform']} "
                  f"Scaler={config['scaler']} Seq={config['sequence_length']} Stride={config['stride']}")
        self.preprocessing_thread = QThread(self)
        self.preprocessing_worker = PreprocessingWorker(action, snapshot["normal_path"],
            snapshot["anomaly_path"], config)
        self.preprocessing_worker.moveToThread(self.preprocessing_thread)
        self.preprocessing_thread.started.connect(self.preprocessing_worker.run)
        self.preprocessing_worker.analysis_ready.connect(self._stage2_analysis_ready)
        self.preprocessing_worker.preprocessing_ready.connect(self._stage2_preprocessing_ready)
        self.preprocessing_worker.failed.connect(self._stage2_worker_failed)
        self.preprocessing_worker.finished.connect(self.preprocessing_thread.quit)
        self.preprocessing_thread.finished.connect(self._stage2_thread_finished)
        self.preprocessing_thread.start()

    @staticmethod
    def _stage2_quality_text(reports):
        labels = (("rows_before", "원본 행 수"), ("rows_after", "정리 후 행 수"),
                  ("exact_duplicates", "완전 중복 행 수"), ("nan_count", "결측값 수"),
                  ("inf_count", "무한값 수"), ("timestamp_backward_count", "시각 역행 수"),
                  ("sampling_interval_ms", "측정 간격 (ms)"),
                  ("segment_count", "구간 수"), ("segment_length_min", "최소 구간 길이"),
                  ("segment_length_mean", "평균 구간 길이"),
                  ("segment_length_median", "중앙 구간 길이"), ("segment_length_max", "최대 구간 길이"))
        lines = []
        for dataset in ("normal", "anomaly"):
            report = reports[dataset]
            lines.append("[정상 데이터]" if dataset == "normal" else "[이상 데이터]")
            lines.extend(f"{title}: {report.get(key, '-')}" for key, title in labels)
            if "selected_sequence_windows" in report:
                counts = report["selected_sequence_windows"]
                lines.append(f"선택 시퀀스 윈도우 수: {counts['windows']}")
                lines.append(f"윈도우 생성 가능 구간 수: {counts['eligible_segments']}")
            lines.append("")
        return "\n".join(lines)

    @Slot(object)
    def _stage2_analysis_ready(self, reports):
        if self._stage2_active_snapshot != self._stage2_snapshot():
            self._stage2_inputs_changed(); return
        self.stage2_analysis_snapshot = self._stage2_active_snapshot.copy()
        self.normal_segments_label.setText(str(reports["normal"]["segment_count"]))
        self.anomaly_segments_label.setText(str(reports["anomaly"]["segment_count"]))
        self.stage2_analysis_text.setPlainText(self._stage2_quality_text(reports))
        if self.stage2_preprocessing_result is None:
            self.stage2_status_label.setText("분석 완료")
        for dataset in ("normal", "anomaly"):
            report = reports[dataset]
            self._log(f"{dataset.title()} cleaned: {report['rows_after']} | segments: {report['segment_count']}")

    @Slot(object)
    def _stage2_preprocessing_ready(self, result):
        if self._stage2_active_snapshot != self._stage2_snapshot():
            self._stage2_inputs_changed(); return
        summary = result.summary
        cross_gap = summary["cross_gap_windows"]
        if cross_gap != 0:
            self.stage2_preprocessing_result = None
            self._stage2_dataset_summary = None
            self._refresh_dataset_text()
            self.stage2_status_label.setText("실패 - 구간 경계 횡단 윈도우 발생")
            self.system_labels["Preprocessing Status"].setText("FAIL")
            self.stage2_cross_gap_label.setText(f"구간 경계 횡단 윈도우: {cross_gap} — 실패")
            self.stage2_cross_gap_label.setStyleSheet("font-weight:bold;color:#ff8c8c")
            self.stage2_preview_text.setPlainText(f"구간 경계 횡단 윈도우: {cross_gap}\n상태: 실패")
            self._log(f"Cross-segment windows: {cross_gap} | FAIL")
            return
        self.stage2_preprocessing_result = result
        self.stage2_preprocessing_config_snapshot = result.config.copy()
        snapshot = self._stage2_active_snapshot
        self.stage2_preprocessing_paths_snapshot = (snapshot["normal_path"], snapshot["anomaly_path"])
        self.stage2_analysis_snapshot = snapshot.copy()
        self.normal_segments_label.setText(str(result.normal_quality_report["segment_count"]))
        self.anomaly_segments_label.setText(str(result.anomaly_quality_report["segment_count"]))
        self.stage2_analysis_text.setPlainText(self._stage2_quality_text({
            "normal": result.normal_quality_report, "anomaly": result.anomaly_quality_report}))
        length = result.config["sequence_length"]
        full = summary["full_dataset_window_counts"][f"seq{length}"]
        signal_name = {"ABS_ALL": "전체 절댓값", "RAW_SIGNED": "원본 부호 유지",
                       "ABS_VIBRATION_RAW_CURRENT": "진동 절댓값 + 전류 원본"}[result.config["signal_transform"]]
        scaler_name = {"MINMAX": "최소·최대 스케일링", "STANDARD": "표준화"}[result.config["scaler"]]
        lines = ["[전처리 설정]", "모드: Stage 2 구간 인식",
            f"예상 측정 간격: {result.config['expected_interval_ms']} ms",
            f"기록 구간 분리 기준: {result.config['gap_threshold_ms']} ms",
            f"신호 변환: {signal_name}", f"스케일링: {scaler_name}",
            f"시퀀스 길이: {length}", f"이동 간격: {result.config['stride']}",
            f"예측 간격: {'사용' if result.config['use_horizon'] else '사용 안 함'}", "",
            "[데이터 품질]",
            f"정상 원본 행 수: {summary['normal_raw_rows']}",
            f"정상 정리 후 행 수: {summary['normal_cleaned_rows']}",
            f"정상 중복 제거 행 수: {summary['normal_duplicates_removed']}",
            f"정상 구간 수: {summary['normal_segments']}",
            f"이상 원본 행 수: {summary['anomaly_raw_rows']}",
            f"이상 정리 후 행 수: {summary['anomaly_cleaned_rows']}",
            f"이상 중복 제거 행 수: {summary['anomaly_duplicates_removed']}",
            f"이상 구간 수: {summary['anomaly_segments']}", "", "[분할]",
            f"정상 학습 구간 수: {summary['normal_train_segments']}",
            f"정상 검증 구간 수: {summary['normal_validation_segments']}",
            f"정상 테스트 구간 수: {summary['normal_test_segments']}",
            f"이상 검증 구간 수: {summary['anomaly_validation_segments']}",
            f"이상 테스트 구간 수: {summary['anomaly_test_segments']}", "", "[관측 윈도우]",
            f"정상 학습 윈도우 수: {summary['normal_train_windows']}",
            f"정상 검증 윈도우 수: {summary['normal_validation_windows']}",
            f"이상 검증 윈도우 수: {summary['anomaly_validation_windows']}",
            f"정상 테스트 윈도우 수: {summary['normal_test_windows']}",
            f"이상 테스트 윈도우 수: {summary['anomaly_test_windows']}",
            f"전체 정상 윈도우 수: {full['normal']['windows']}",
            f"전체 이상 윈도우 수: {full['anomaly']['windows']}",
            f"윈도우 생성 가능 정상 구간 수: {full['normal']['eligible_segments']}",
            f"윈도우 생성 가능 이상 구간 수: {full['anomaly']['eligible_segments']}",
            f"구간 경계 횡단 윈도우: {cross_gap}", "상태: 완료",
            "처리된 데이터셋을 저장한 뒤 학습 화면에서 해당 데이터셋 ID를 선택하세요."]
        self.stage2_preview_text.setPlainText("\n".join(lines))
        self.stage2_status_label.setText("전처리 완료")
        self.stage2_cross_gap_label.setText("구간 경계 횡단 윈도우: 0 — 통과")
        self.stage2_cross_gap_label.setStyleSheet("font-weight:bold;color:#98e6aa")
        self.stage2_total_windows_label.setText(
            f"전체 윈도우 (정상 / 이상): {full['normal']['windows']} / {full['anomaly']['windows']}")
        self.system_labels["Preprocessing Status"].setText("PREPROCESSED")
        self._stage2_dataset_summary = summary
        self._refresh_dataset_text()
        self._log(f"Normal cleaned: {summary['normal_cleaned_rows']} | segments: {summary['normal_segments']}")
        self._log(f"Anomaly cleaned: {summary['anomaly_cleaned_rows']} | segments: {summary['anomaly_segments']}")
        self._log(f"Cross-segment windows: {cross_gap}")
        self._log("Stage 2 preprocessing completed. Save the Processed Dataset before Training.")
        self._refresh_stage2_save_enabled()

    @Slot(str)
    def _stage2_worker_failed(self, message):
        self._stage2_failed(message, self._stage2_active_action)

    def _stage2_failed(self, message, action):
        self._log(f"Stage2 {action} error: {message}")
        self.stage2_status_label.setText("오류")
        if action == "analyze":
            self.stage2_analysis_snapshot = None
            self.normal_segments_label.setText("-"); self.anomaly_segments_label.setText("-")
            self.stage2_analysis_text.setPlainText(f"오류: {message}")
        else:
            self.stage2_preprocessing_result = None
            self._stage2_dataset_summary = None
            self._refresh_dataset_text()
            self.system_labels["Preprocessing Status"].setText("ERROR")
            self.stage2_preview_text.setPlainText(f"오류: {message}")
            self.stage2_cross_gap_label.setText("구간 경계 횡단 윈도우: -")
            self.stage2_cross_gap_label.setStyleSheet("font-weight:bold;color:#8ad8f5")
            self.stage2_total_windows_label.setText("전체 윈도우 (정상 / 이상): - / -")
        QMessageBox.warning(self, "Stage 2 전처리 오류", message)
        self._refresh_stage2_save_enabled()

    @Slot()
    def _stage2_thread_finished(self):
        if self.preprocessing_worker:
            self.preprocessing_worker.deleteLater()
        if self.preprocessing_thread:
            self.preprocessing_thread.deleteLater()
        self.preprocessing_worker = None
        self.preprocessing_thread = None
        self._set_stage2_busy(False)

    def save_preprocessing_settings(self):
        try: config = self._preprocessing_config_from_ui()
        except ValueError as exc:
            QMessageBox.warning(self, "전처리 설정 오류", str(exc)); return
        path, _ = QFileDialog.getSaveFileName(self, "전처리 설정 저장", str(DEFAULT_CONFIG_DIR / "stage2.json"), "JSON (*.json)")
        if path:
            try: save_preprocessing_config(path, config)
            except (ValueError, OSError) as exc: QMessageBox.warning(self, "전처리 설정 오류", str(exc))

    def load_preprocessing_settings(self):
        path, _ = QFileDialog.getOpenFileName(self, "전처리 설정 불러오기", str(DEFAULT_CONFIG_DIR), "JSON (*.json)")
        if path:
            try: self._apply_preprocessing_config(load_preprocessing_config(path))
            except (ValueError, OSError, json.JSONDecodeError) as exc: QMessageBox.warning(self, "전처리 설정 오류", str(exc))

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
        self._training_locked = locked
        for widget in self._training_editable_widgets:
            widget.setEnabled(not locked)
        processed = self.data_mode_combo.currentData() == "PROCESSED_DATASET"
        self.processed_dataset_combo.setEnabled(processed and not locked)
        self.refresh_processed_datasets_button.setEnabled(processed and not locked)
        self.preset_combo.setEnabled(not processed and not locked)
        for widget in self.preprocessing_widgets:
            widget.setEnabled(not locked and not self._stage2_busy)
        self.run_mode.setEnabled(not locked)
        self.start_button.setEnabled(not locked and not self._stage2_busy)
        self.stop_button.setEnabled(locked)
        self._refresh_stage2_save_enabled()
        if not locked and not self._stage2_busy:
            self._set_preprocessing_mode_editable()
            try: self._preprocessing_config_from_ui()
            except ValueError: self.save_preprocessing_button.setEnabled(False)
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
        self._update_model_description_only()
        cnn = spec.id in {"CNN_LSTM_AUTOENCODER", "DENOISING_CNN_LSTM_AUTOENCODER"}
        kernel_size = self.cnn_kernel_combo.currentData() if cnn else None
        filters = self.cnn_filters_combo.currentData()
        bottleneck = self.bottleneck_combo.currentData()
        if self.data_mode_combo.currentData() == "PROCESSED_DATASET" and self._selected_processed_artifact is not None:
            config = self._selected_processed_artifact.config["preprocessing"]
            model = training_engine.build_training_model(spec, self._training_config_dict(), config)
        else:
            model = spec.builder(kernel_size=kernel_size, cnn_filters=filters, bottleneck_units=bottleneck) if cnn else spec.builder()
        self.params_label.setText(f"전체 파라미터 수: {model.count_params()}")
        if self.thread is None or not self.thread.isRunning():
            self.status_label.setText(f"선택한 모델: {spec.display_name}    상태: 준비")
        self._update_experiment_summary()

    def _update_model_description_only(self):
        spec = MODEL_REGISTRY[self.training_model_combo.currentData()]
        sequence = (self._selected_processed_artifact.config["preprocessing"]["sequence_length"]
                    if self.data_mode_combo.currentData() == "PROCESSED_DATASET" and self._selected_processed_artifact is not None
                    else self.preprocessing_config["sequence_length"])
        description = spec.description.replace("20×3", f"{sequence}×3").replace("과거 20 timestep", f"과거 {sequence} timestep")
        self.training_model_description.setText(f"{spec.display_name}\n과제: {'예측' if spec.task_type == 'FORECAST' else '정상 패턴 복원'}\n입력: {sequence} × 3{' + Gaussian 노이즈' if spec.denoising else ''}\n출력: {spec.forecast_length or sequence} × 3\n\n{description}")
        cnn = spec.id in {"CNN_LSTM_AUTOENCODER", "DENOISING_CNN_LSTM_AUTOENCODER"}
        kernel_size = self.cnn_kernel_combo.currentData() if cnn else None
        filters = self.cnn_filters_combo.currentData()
        bottleneck = self.bottleneck_combo.currentData()
        architecture = (f"입력 {sequence}×3{' + Gaussian 노이즈' if spec.denoising else ''}\n↓\n"
                        f"Conv1D {filters}, k={kernel_size} × 2\n↓\n"
                        f"LSTM64 → B{bottleneck}\n↓\nRepeatVector{sequence}\n↓\n"
                        f"LSTM{bottleneck} → LSTM64\n↓\n출력 {sequence}×3") if cnn else spec.architecture.replace("20×3", f"{sequence}×3").replace("RepeatVector20", f"RepeatVector{sequence}")
        self.architecture_label.setText(architecture)

    def _data_mode_changed(self, *_):
        processed = self.data_mode_combo.currentData() == "PROCESSED_DATASET"
        locked = self._training_locked
        self.processed_dataset_combo.setEnabled(processed and not locked)
        self.refresh_processed_datasets_button.setEnabled(processed and not locked)
        self.preset_combo.setEnabled(not processed and not locked)
        current_model = self.training_model_combo.currentData()
        self.training_model_combo.blockSignals(True)
        self.training_model_combo.clear()
        if processed:
            for model_id in PROCESSED_DATASET_MODEL_IDS:
                spec = MODEL_REGISTRY[model_id]
                self.training_model_combo.addItem(spec.display_name, spec.id)
        else:
            for spec in MODEL_REGISTRY.values():
                if spec.status == "ACTIVE": self.training_model_combo.addItem(spec.display_name, spec.id)
        wanted = (current_model if current_model in PROCESSED_DATASET_MODEL_IDS else "CNN_LSTM_AUTOENCODER") if processed else current_model
        index = self.training_model_combo.findData(wanted)
        self.training_model_combo.setCurrentIndex(index if index >= 0 else self.training_model_combo.findData("CNN_LSTM_AUTOENCODER"))
        self.training_model_combo.blockSignals(False)
        if processed:
            self.refresh_processed_datasets()
        else:
            self._selected_processed_artifact = None
            self.processed_dataset_summary_label.setText("KAMP baseline data path")
            self._refresh_preprocessing_display()
        self.update_training_model()
        self._update_model_dependent_controls()

    def refresh_processed_datasets(self):
        previous = self.processed_dataset_combo.currentData()
        self.processed_dataset_combo.blockSignals(True)
        self.processed_dataset_combo.clear()
        for dataset in discover_processed_datasets():
            self.processed_dataset_combo.addItem(dataset["dataset_id"], dataset["path"])
        index = self.processed_dataset_combo.findData(previous)
        if index >= 0:
            self.processed_dataset_combo.setCurrentIndex(index)
        self.processed_dataset_combo.blockSignals(False)
        self._processed_dataset_changed()

    def _processed_dataset_changed(self, *_):
        self._selected_processed_artifact = None
        path = self.processed_dataset_combo.currentData()
        if path and self.data_mode_combo.currentData() == "PROCESSED_DATASET":
            try:
                artifact = load_processed_dataset(path)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                self.processed_dataset_summary_label.setText(f"Invalid Processed Dataset: {exc}")
            else:
                self._selected_processed_artifact = artifact
                config, summary = artifact.config["preprocessing"], artifact.summary
                self.processed_dataset_summary_label.setText(
                    f"Dataset ID: {artifact.dataset_id}\nMode: {config['mode']}\n"
                    f"Signal: {config['signal_transform']} | Scaler: {config['scaler']}\n"
                    f"Sequence: {config['sequence_length']} | Stride: {config['stride']} | "
                    f"Gap Threshold: {config['gap_threshold_ms']} ms | Horizon: OFF\n"
                    f"Train: {summary['train_windows']} | Validation: "
                    f"{summary['validation_normal_windows'] + summary['validation_anomaly_windows']} | "
                    f"Test: {summary['test_normal_windows'] + summary['test_anomaly_windows']} windows")
        elif self.data_mode_combo.currentData() == "PROCESSED_DATASET":
            self.processed_dataset_summary_label.setText("Valid Processed Dataset이 없습니다. Refresh를 확인하세요.")
        self._refresh_preprocessing_display()
        if self.training_model_combo.currentData() and self._selected_processed_artifact is not None:
            self.update_training_model()
        self._update_model_dependent_controls()

    def _update_model_dependent_controls(self, *_):
        denoising = self.training_model_combo.currentData() == "DENOISING_CNN_LSTM_AUTOENCODER"
        self.model_structure_box.setVisible(self.training_model_combo.currentData() in {"CNN_LSTM_AUTOENCODER", "DENOISING_CNN_LSTM_AUTOENCODER"})
        self.denoising_box.setVisible(denoising)
        self.noise_std_spin.setEnabled(denoising and not self._training_locked)
        standard = (denoising and self.data_mode_combo.currentData() == "PROCESSED_DATASET"
                    and self._selected_processed_artifact is not None
                    and self._selected_processed_artifact.config["preprocessing"]["scaler"] == "STANDARD")
        self.noise_clip_check.setEnabled(denoising and not standard and not self._training_locked)
        self.noise_clip_policy_label.setText(
            f"노이즈 범위 제한 요청: {'ON' if self.noise_clip_check.isChecked() else 'OFF'} | 실제 적용: OFF. "
            "StandardScaler의 음수와 1 초과 값을 유지합니다."
            if standard else "")
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
        if self.data_mode_combo.currentData() == "PROCESSED_DATASET": return
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
        if hasattr(self, "auto_pool_list"): self.refresh_auto_pool_runs()
        if hasattr(self, "score_combo"): self.update_algorithm_description()

    def refresh_auto_pool_runs(self):
        selected = {item.data(Qt.UserRole).model_path for item in self.auto_pool_list.selectedItems()}
        self.auto_pool_list.clear()
        model_id = self.evaluation_model_combo.currentData()
        if model_id not in PROCESSED_DATASET_MODEL_IDS:
            return
        for run in discover_model_runs(model_id):
            if (run.metadata or {}).get("data_mode") != "PROCESSED_DATASET":
                continue
            item = QListWidgetItem(f"Seq{run.metadata.get('sequence_length', '?')} | {run.run_id} | {run.metadata.get('dataset_id', '?')}")
            item.setData(Qt.UserRole, run)
            self.auto_pool_list.addItem(item)
            item.setSelected(run.model_path in selected)

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
        for button in (self.load_model_button, self.run_evaluation_button, self.compare_all_button, self.sweep_ewma_button, self.compare_models_button,
                       self.refresh_auto_pool_button, self.run_auto_button): button.setEnabled(not busy)
        self.auto_pool_list.setEnabled(not busy)
        for widget in (self.temporal_combo, self.ewma_alpha_spin, self.timestamp_aware_check):
            widget.setEnabled(not busy and (widget is not self.ewma_alpha_spin or self.temporal_combo.currentData() == "EWMA"))

    def _start_evaluation_worker(self, action, auto_runs=None):
        if self.evaluation_thread and self.evaluation_thread.isRunning():
            self._log("Evaluation is already running")
            return
        if action not in ("load", "compare_models", "auto_sequence") and not self.evaluation_controller.is_loaded:
            QMessageBox.information(self, "Evaluation", "LOAD MODEL을 먼저 실행하세요.")
            return
        self._set_evaluation_busy(True)
        self.evaluation_thread = QThread(self)
        self.evaluation_worker = EvaluationWorker(self.evaluation_controller, action,
            self.score_combo.currentData(), self.threshold_combo.currentData(),
            self.temporal_combo.currentData(), self.ewma_alpha_spin.value(),
            self.timestamp_aware_check.isChecked(), auto_runs=auto_runs)
        self.evaluation_worker.moveToThread(self.evaluation_thread)
        self.evaluation_thread.started.connect(self.evaluation_worker.run)
        self.evaluation_worker.status_changed.connect(self._update_evaluation_status)
        self.evaluation_worker.log_message.connect(self._log)
        self.evaluation_worker.loaded.connect(self._evaluation_loaded)
        self.evaluation_worker.result_ready.connect(self._evaluation_result)
        self.evaluation_worker.comparison_ready.connect(self._comparison_results)
        self.evaluation_worker.auto_ready.connect(self._auto_evaluation_result)
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

    def run_auto_evaluation(self):
        runs = [item.data(Qt.UserRole) for item in self.auto_pool_list.selectedItems()]
        if len(runs) < 2:
            QMessageBox.warning(self, "AUTO Sequence", "서로 다른 Sequence의 저장 모델을 2개 이상 선택하세요.")
            return
        self._start_evaluation_worker("auto_sequence", auto_runs=runs)

    def _update_evaluation_status(self, status):
        spec = MODEL_REGISTRY[self.evaluation_model_combo.currentData()]
        self.eval_status_label.setText(f"모델: {spec.display_name}    Pipeline: {self.score_combo.currentText()} → {self._temporal_display()} → {self.threshold_combo.currentText()}    상태: {status}")

    def _evaluation_loaded(self, summary):
        self._log(f"Evaluation model ready: {summary}")
        self._show_segment_metrics({})
        self.update_algorithm_description()
        if summary.get("data_mode") == "PROCESSED_DATASET":
            self.evaluation_dataset_label.setText(
                f"DATA MODE: PROCESSED DATASET\nDataset: {summary['dataset_id']}\n"
                f"Sequence: {summary['sequence_length']} | Signal: {summary['signal_transform']} | "
                f"Scaler: {summary['scaler']} | Gap: {summary['gap_threshold_ms']} ms\n"
                f"Validation: {summary['valid_samples']} | Test: {summary['test_samples']}")
            self.evaluation_gap_label.setText(f"{summary['gap_threshold_ms']} ms")
        else:
            self.evaluation_dataset_label.setText("DATA MODE: KAMP BASELINE")
            self.evaluation_gap_label.setText(f"{self.preprocessing_config['gap_threshold_ms']} ms")
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
        self._show_segment_metrics(result)
        self.eval_status_label.setText(f"모델: {result['model']}    Pipeline: {SCORE_METHODS[result['score_method']].display_name} → {self._result_temporal_display(result)} → {THRESHOLD_METHODS[result['threshold_method']].display_name}    상태: 완료")
        self._append_comparison(result)
        run = self.evaluation_run_combo.currentData()
        if run is not None:
            path = append_evaluation_result(run, result); self._log(f"Evaluation result saved: {path}")
            if result.get("data_mode") == "PROCESSED_DATASET":
                detail_path = save_segment_detail_csv(run, self.evaluation_controller._last_segment_details)
                self._log(f"Segment details saved: {detail_path}")
        if result["fallback_used"]: self._log(f"Threshold fallback: {result['fallback_reason']}")
        self.evaluation_pages.setCurrentIndex(1)

    def _auto_evaluation_result(self, result):
        self._evaluation_result_without_history(result)
        self.evaluation_dataset_label.setText(
            f"AUTO Sequence pool | Seq {result['available_sequences']}\n"
            f"Signal: {result['signal_transform']} | Scaler: {result['scaler']} | "
            f"Gap: {result['gap_threshold_ms']} ms\nRuns: {result['auto_pool']}")
        self.evaluation_gap_label.setText(f"{result['gap_threshold_ms']} ms")
        self.eval_status_label.setText(
            f"AUTO Sequence 완료 | {result['model_id']} | Seq {result['available_sequences']} | "
            f"각 Sequence별 Validation 임계값 사용")
        self._append_comparison({**result, "status": "AUTO"})
        self._log(f"AUTO Sequence evaluation: {result['test_windows']} selected windows; "
                  f"per-tier thresholds {result['auto_tier_thresholds']}")
        self.evaluation_pages.setCurrentIndex(1)

    def _comparison_results(self, results):
        for result in results: self._append_comparison(result)
        if results:
            self._evaluation_result_without_history(results[-1])
            self.eval_status_label.setText(f"모델: {results[-1].get('model','')}    Pipeline: {SCORE_METHODS[results[-1]['score_method']].display_name} → {self._result_temporal_display(results[-1])} → {THRESHOLD_METHODS[results[-1]['threshold_method']].display_name}    상태: 완료")
            self.evaluation_pages.setCurrentIndex(1)
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
        values = {"Threshold": "시퀀스별" if result["threshold"] is None else f"{result['threshold']:.8g}", "Accuracy": f"{result['accuracy']:.6f}",
            "Balanced Accuracy": f"{result['balanced_accuracy']:.6f}", "Precision": f"{result['precision']:.6f}",
            "Recall": f"{result['recall']:.6f}", "F1 Score": f"{result['f1']:.6f}", "Specificity": f"{result['specificity']:.6f}",
            "FPR": f"{result['fpr']:.6f}", "FNR": f"{result['fnr']:.6f}", "TN": str(result['tn']),
            "FP (False Alarm)": str(result['fp']), "FN (Missed Anomaly)": str(result['fn']), "TP": str(result['tp']),
            "Fallback": result['fallback_reason'] if result['fallback_used'] else "No"}
        for key, value in values.items(): self.eval_result_labels[key].setText(value)
        self._show_segment_metrics(result)

    def _show_segment_metrics(self, result):
        for key in METRIC_KEYS:
            value = result.get(key)
            self.segment_result_labels[key].setText(
                "-" if value is None or value == "" else f"{float(value):.2%}" if key == "segment_coverage" else str(value))

    def _append_comparison(self, result):
        fingerprint = (str(result.get("model_id", result.get("model", ""))), str(result.get("experiment_name", "")),
                       str(result.get("score_method", result.get("score_name", ""))), str(result.get("threshold_method", result.get("threshold_name", ""))),
                       str(result.get("threshold", "")), str(result.get("created_at", "")),
                       str(result.get("temporal_method", "NONE")), str(result.get("ewma_alpha", "")),
                       str(result.get("timestamp_aware", True)),
                       str(result.get("dataset_id", "")), str(result.get("sequence_length", "")),
                       str(result.get("auto_pool", "")),
                       str(result.get("signal_transform", "")), str(result.get("scaler", "")))
        if any(item.get("_fingerprint") == fingerprint for item in self.comparison_history): return
        result = result.copy(); result["_fingerprint"] = fingerprint
        self.comparison_history.append(result.copy())
        row = self.comparison_table.rowCount(); self.comparison_table.insertRow(row)
        def number(key, digits=6):
            value = result.get(key, "")
            if value is None: return ""
            try: return f"{float(value):.{digits}f}"
            except (TypeError, ValueError): return str(result.get(key, ""))
        score_display = SCORE_METHODS[result["score_method"]].display_name if result.get("score_method") in SCORE_METHODS else result.get("score_name", result.get("score_method", ""))
        threshold_display = THRESHOLD_METHODS[result["threshold_method"]].display_name if result.get("threshold_method") in THRESHOLD_METHODS else result.get("threshold_name", result.get("threshold_method", ""))
        experiment_display = result.get("experiment_name", "")
        if result.get("dataset_id"):
            experiment_display = f"{experiment_display} [{result['dataset_id']}]"
        values = (result.get("status", ""), result.get("model", "LSTM AutoEncoder"), experiment_display, architecture_signature(result), result.get("seed", ""), result.get("optimizer", ""),
                  result.get("learning_rate", ""), result.get("loss", ""), result.get("batch_size", ""), result.get("noise_std", ""), score_display,
                  threshold_display, number("threshold", 8), number("accuracy"), number("balanced_accuracy"), number("precision"), number("recall"), number("f1"),
                  result.get("tn", ""), result.get("fp", ""), result.get("fn", ""), result.get("tp", ""), result.get("best_epoch", ""), result.get("best_val_loss", ""), result.get("training_time", ""), result.get("created_at", ""),
                  result.get("temporal_method", "NONE"), result.get("ewma_alpha", ""), result.get("timestamp_aware", True), result.get("total_error", ""),
                  result.get("detection_delay_samples", ""), result.get("pareto", ""),
                  result.get("sequence_length", ""), number("segment_coverage"), number("segment_detection_rate"),
                  number("median_segment_delay_seconds"))
        for column, value in enumerate(values): self.comparison_table.setItem(row, column, QTableWidgetItem(str(value)))
        if result.get("dataset_id"):
            self.comparison_table.item(row, 2).setToolTip(
                f"Data Mode: {result.get('data_mode')}\nDataset: {result['dataset_id']}\n"
                f"Sequence: {result.get('sequence_length')} | Signal: {result.get('signal_transform')} | "
                f"Scaler: {result.get('scaler')} | Gap: {result.get('gap_threshold_ms')} ms")
        elif result.get("auto_pool"):
            self.comparison_table.item(row, 2).setToolTip(
                f"AUTO pool: {result['auto_pool']}\nPer-tier thresholds: {result.get('auto_tier_thresholds', '')}\n"
                f"Effective methods: {result.get('auto_tier_methods', '')}\n"
                f"Signal: {result.get('signal_transform')} | Scaler: {result.get('scaler')} | "
                f"Gap: {result.get('gap_threshold_ms')} ms")
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
                      **{key: row.get(key, "") for key in ("preprocessing_mode", "data_mode", "dataset_id", "sequence_length",
                          "stride", "gap_threshold_ms", "signal_transform", "scaler", "use_horizon", "prediction_horizon")},
                      "total_error": row.get("total_error", ""), "detection_delay_samples": row.get("detection_delay_samples", ""),
                      "detection_delay_seconds": row.get("detection_delay_seconds", ""), "pareto": row.get("pareto", ""),
                      "auto_pool": row.get("auto_pool", ""), "auto_tier_thresholds": row.get("auto_tier_thresholds", ""),
                      "auto_tier_methods": row.get("auto_tier_methods", ""),
                      **{key: row.get(key, "") for key in METRIC_KEYS},
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
                "timestamp_gap_threshold", "total_error", "detection_delay_samples", "detection_delay_seconds", "pareto",
                "preprocessing_mode", "data_mode", "dataset_id", "sequence_length", "stride", "gap_threshold_ms",
                "signal_transform", "scaler", "use_horizon", "prediction_horizon", "auto_pool", "auto_tier_thresholds",
                "auto_tier_methods", *METRIC_KEYS]
        with path.open("x", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader()
            for result in self.comparison_history: writer.writerow({key: result.get(key, "") for key in keys})
        self._log(f"Comparison exported: {path}")

    def _evaluation_failed(self, message):
        self._log(message); self._update_evaluation_status("ERROR")
        if not self.evaluation_controller.is_loaded:
            self.evaluation_dataset_label.setText(f"DATA MODE: ERROR\n{message}")
        QMessageBox.critical(self, "Evaluation Error", message)

    def _evaluation_thread_finished(self):
        if self.evaluation_worker: self.evaluation_worker.deleteLater()
        if self.evaluation_thread: self.evaluation_thread.deleteLater()
        self.evaluation_worker = None; self.evaluation_thread = None; self._set_evaluation_busy(False)

    def start_training(self):
        quick = self.run_mode.currentData() == "QUICK_TEST"
        try: config = validate_training_config(self._training_config_dict())
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Training Configuration", str(exc)); return
        data_mode = self.data_mode_combo.currentData()
        processed_path = None
        if data_mode == "PROCESSED_DATASET":
            model_id = self.training_model_combo.currentData()
            if model_id not in PROCESSED_DATASET_MODEL_IDS or MODEL_REGISTRY[model_id].task_type != "RECONSTRUCTION":
                QMessageBox.warning(self, "Invalid Model", f"지원하지 않는 Processed Dataset 모델: {model_id}"); return
            processed_path = self.processed_dataset_combo.currentData()
            try:
                if not processed_path: raise ValueError("Processed Dataset을 선택하세요.")
                artifact = load_processed_dataset(processed_path)
                preprocessing = artifact.config["preprocessing"]
            except (ValueError, OSError, KeyError, TypeError) as exc:
                QMessageBox.warning(self, "Invalid Processed Dataset", str(exc)); return
        else:
            try: preprocessing = self._preprocessing_config_from_ui()
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid Preprocessing Configuration", str(exc)); return
        self.reset_loss_graph()
        self.graph_tab.reset_current({**config, "display_name": self.training_model_combo.currentText()})
        self._set_training_locked(True)
        self.status_label.setText(f"학습 모델: {self.training_model_combo.currentText()}    실행: {'빠른 시험' if quick else '전체 학습'}    상태: 시작 중")
        model_id = self.training_model_combo.currentData()
        self.thread = QThread(self)
        if data_mode == "PROCESSED_DATASET":
            self.worker = training_engine.TrainingWorker(quick, model_id, config, preprocessing,
                data_mode="PROCESSED_DATASET", processed_dataset_path=processed_path)
        else:
            self.worker = training_engine.TrainingWorker(quick, model_id, config, preprocessing)
        self.worker.moveToThread(self.thread)
        self.thread.finished.connect(self._thread_finished)
        self.thread.started.connect(self.worker.run); self.worker.status_changed.connect(self.update_status); self.worker.log_message.connect(self._log)
        self.worker.dataset_ready.connect(self.update_dataset); self.worker.model_ready.connect(self.update_model); self.worker.epoch_update.connect(self.update_epoch)
        self.worker.evaluation_ready.connect(self.update_evaluation); self.worker.finished.connect(self.training_finished); self.worker.failed.connect(self.training_failed)
        self.thread.start()

    def stop_training(self):
        if self.worker: self.worker.request_stop(); self._log("Stop requested")

    def update_status(self, status): self.status_label.setText(self.status_label.text().split("상태:")[0] + "상태: " + status)
    def _refresh_dataset_text(self):
        sections = []
        if self._stage2_training_dataset is not None:
            heading = "Processed Dataset Training" if self._stage2_training_dataset.get("data_mode") == "PROCESSED_DATASET" else "KAMP Training"
            sections.append(f"[{heading}]\n" + "\n".join(f"{k}: {v}" for k, v in self._stage2_training_dataset.items()))
        if self._stage2_dataset_summary is not None:
            summary = self._stage2_dataset_summary
            full = summary["full_dataset_window_counts"][f"seq{summary['sequence_length']}"]
            sections.append("[Stage 2 Preprocessing]\n"
                f"Normal Cleaned: {summary['normal_cleaned_rows']} | Segments: {summary['normal_segments']}\n"
                f"Anomaly Cleaned: {summary['anomaly_cleaned_rows']} | Segments: {summary['anomaly_segments']}\n"
                f"Sequence: {summary['sequence_length']} | Windows Normal/Anomaly: "
                f"{full['normal']['windows']}/{full['anomaly']['windows']}\n"
                f"Cross-gap: {summary['cross_gap_windows']}\nSave and select this Processed Dataset for Training.")
        self.dataset_text.setText("\n\n".join(sections) if sections else "실행 기록 없음")

    def update_dataset(self, data):
        self._stage2_training_dataset = data.copy()
        if data.get("data_mode") == "PROCESSED_DATASET":
            for key, value in (("Mode", "PROCESSED DATASET"), ("Dataset", data["dataset_id"]),
                               ("Sequence", str(data["sequence_length"])), ("Horizon", "OFF"),
                               ("Input Shape", data["input_shape"]), ("Signal Transform", data["signal_transform"]),
                               ("Scaler", data["scaler"]), ("Segment Gap", f"{data['gap_threshold_ms']} ms")):
                self.system_labels[key].setText(value)
        self._refresh_dataset_text()
        self._log(f"Dataset ready: {data}")
    def update_model(self, params, trainable): self.params_label.setText(f"전체 파라미터 수: {params}\n학습 파라미터 수: {trainable}")
    @Slot(int, int, float, float, float, float, int, int, float, int)
    def update_epoch(self, epoch, maximum, loss, val_loss, lr, elapsed, since, best_epoch, best_loss, reduce_count):
        previous_lr = getattr(self, "_last_lr", lr); self._last_lr = lr
        self.progress.setMaximum(maximum); self.progress.setValue(epoch)
        values = (("Model", self.training_model_combo.currentData()), ("Experiment Name", self.experiment_name_edit.text().strip() or "AUTO"), ("Epoch", f"{epoch} / {maximum}"), ("Train Loss", f"{loss:.6g}"), ("Validation Loss", f"{val_loss:.6g}"), ("Best Epoch", str(best_epoch)), ("Best Val Loss", f"{best_loss:.6g}"), ("Generalization Gap", f"{val_loss-loss:+.6g}"), ("Learning Rate", f"{lr:.6g}"), ("Elapsed", f"{elapsed:.1f}s"), ("EarlyStopping", f"{since} / {self.early_stopping_patience_spin.value()}"), ("ReduceLR", str(reduce_count)), ("Optimizer", self.optimizer_combo.currentText()), ("Loss", self.loss_combo.currentText()), ("Batch Size", str(self.batch_spin.value())), ("Noise Std", str(self.noise_std_spin.value()) if self.training_model_combo.currentData() == "DENOISING_CNN_LSTM_AUTOENCODER" else "N/A"))
        for key, value in values: self.monitor_labels[key].setText(value)
        self.graph_tab.add_current_epoch(epoch, loss, val_loss, lr, best_epoch, best_loss)
    def update_evaluation(self, data):
        self._log(f"Training baseline evaluation: threshold={data.get('threshold')} accuracy={data.get('accuracy')} f1={data.get('f1_score')}")
        if data.get("data_mode") == "PROCESSED_DATASET":
            matrix = data["confusion_matrix"]
            self._append_comparison({"status": "QUICK_TEST" if self.run_mode.currentData() == "QUICK_TEST" else "TRAINED",
                "model": MODEL_REGISTRY[data["model_id"]].display_name, "model_id": data["model_id"],
                "experiment_name": data["experiment_name"], "data_mode": data["data_mode"],
                "dataset_id": data["dataset_id"], "sequence_length": data["sequence_length"],
                "signal_transform": data["signal_transform"], "scaler": data["scaler"],
                "gap_threshold_ms": data["gap_threshold_ms"], "score_method": "LAST_STEP_MSE",
                "threshold_method": "PR_INTERSECTION", "threshold": data["threshold"],
                "accuracy": data["accuracy"], "precision": data["precision"],
                "recall": data["recall"], "f1": data["f1_score"],
                "tn": matrix[0][0], "fp": matrix[0][1], "fn": matrix[1][0], "tp": matrix[1][1]})
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
        if self.preprocessing_thread and self.preprocessing_thread.isRunning():
            self._log("Window close requested: waiting for preprocessing worker")
            self.preprocessing_thread.quit(); self.preprocessing_thread.wait()
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
