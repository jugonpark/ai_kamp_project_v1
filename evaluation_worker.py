"""Qt worker for saved-model evaluation; contains no training path."""
from PySide6.QtCore import QObject, Signal, Slot


class EvaluationWorker(QObject):
    status_changed = Signal(str)
    log_message = Signal(str)
    loaded = Signal(dict)
    result_ready = Signal(dict)
    comparison_ready = Signal(list)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, controller, action, score_method=None, threshold_method=None,
                 temporal_method="NONE", ewma_alpha=0.4, timestamp_aware=True):
        super().__init__()
        self.controller = controller
        self.action = action
        self.score_method = score_method
        self.threshold_method = threshold_method
        self.temporal_method = temporal_method
        self.ewma_alpha = ewma_alpha
        self.timestamp_aware = timestamp_aware

    @Slot()
    def run(self):
        try:
            if self.action == "load":
                self.status_changed.emit("LOADING MODEL / PREDICTIONS")
                summary = self.controller.load_model_and_predictions()
                self.loaded.emit(summary)
                self.log_message.emit("Prediction cache hit" if summary["cache_hit"] else "Model predictions cached in memory")
            elif self.action == "evaluate":
                self.status_changed.emit("EVALUATING")
                self.result_ready.emit(self.controller.evaluate(
                    self.score_method, self.threshold_method, self.temporal_method,
                    self.ewma_alpha, self.timestamp_aware))
            elif self.action == "compare_all":
                self.status_changed.emit("COMPARING ALL")
                self.comparison_ready.emit(self.controller.compare_all(
                    lambda done, total: self.log_message.emit(f"Compare All: {done}/{total}"),
                    self.temporal_method, self.ewma_alpha, self.timestamp_aware))
            elif self.action == "sweep_ewma":
                self.status_changed.emit("COMPARING EWMA ALPHAS")
                self.comparison_ready.emit(self.controller.sweep_ewma_alphas(
                    self.score_method, self.threshold_method, self.timestamp_aware,
                    lambda done, total: self.log_message.emit(f"EWMA sweep: {done}/{total}")))
            elif self.action == "compare_models":
                self.status_changed.emit("COMPARING MODELS")
                self.comparison_ready.emit(self.controller.compare_models(self.score_method, self.threshold_method,
                    lambda done, total: self.log_message.emit(f"Compare Models: {done}/{total}")))
            else:
                raise ValueError(f"Unknown evaluation action: {self.action}")
        except Exception as exc:
            self.failed.emit(f"Evaluation failed: {type(exc).__name__}: {exc}")
        finally:
            self.finished.emit()
