"""Qt worker for Stage 2 analysis and preprocessing previews only."""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from preprocessing_config import validate_preprocessing_config
from stage2.data_quality import inspect_csv
from stage2.preprocessing import run_stage2_preprocessing


class PreprocessingWorker(QObject):
    analysis_ready = Signal(object)
    preprocessing_ready = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, action: str, normal_path: str, anomaly_path: str, config: dict):
        super().__init__()
        self.action = action
        self.normal_path = normal_path
        self.anomaly_path = anomaly_path
        self.config = validate_preprocessing_config(config)

    @Slot()
    def run(self):
        try:
            if self.action == "analyze":
                normal = inspect_csv(self.normal_path, self.config, expected_label=0)
                anomaly = inspect_csv(self.anomaly_path, self.config, expected_label=1)
                self.analysis_ready.emit({"normal": normal.report, "anomaly": anomaly.report})
            elif self.action == "preprocess":
                self.preprocessing_ready.emit(run_stage2_preprocessing(
                    self.normal_path, self.anomaly_path, self.config))
            else:
                raise ValueError(f"Unknown preprocessing action: {self.action}")
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.finished.emit()
