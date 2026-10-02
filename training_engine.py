"""Thread-friendly KAMP training orchestration for the PySide6 monitor."""
import time
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import train_lstm_ae as core
from model_registry import MODEL_REGISTRY
from model_data import create_task_bundle, prepare_fit_data
from model_artifacts import append_evaluation_result, save_model_run
from training_config import apply_random_seed, default_training_config, validate_training_config


def build_optimizer(config):
    if config["optimizer"] == "AdamW":
        return tf.keras.optimizers.AdamW(learning_rate=config["learning_rate"], weight_decay=config["weight_decay"])
    if config["optimizer"] == "RMSprop":
        return tf.keras.optimizers.RMSprop(learning_rate=config["learning_rate"])
    return tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])


def build_loss(config):
    return tf.keras.losses.Huber(delta=config["huber_delta"]) if config["loss"] == "Huber" else "mse"


def build_training_model(spec, config):
    if spec.id in {"CNN_LSTM_AUTOENCODER", "DENOISING_CNN_LSTM_AUTOENCODER"}:
        return spec.builder(kernel_size=config["cnn_kernel_size"])
    return spec.builder()


class GUITrainingCallback(QObject, tf.keras.callbacks.Callback):
    """Emit plain data through Qt signals; never touch widgets from TensorFlow."""

    epoch_finished = Signal(int, int, float, float, float, float, int, int, float, int)
    event = Signal(str)

    def __init__(self, max_epochs, min_delta=0.00001):
        QObject.__init__(self)
        tf.keras.callbacks.Callback.__init__(self)
        self.max_epochs = max_epochs
        self.min_delta = float(min_delta)
        self.stop_requested = False
        self.started_at = None
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.since_improvement = 0
        self.previous_lr = None
        self.reduce_lr_count = 0

    def request_stop(self):
        self.stop_requested = True
        if self.model is not None:
            self.model.stop_training = True

    def on_train_begin(self, logs=None):
        self.started_at = time.monotonic()
        self.event.emit("Training started")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = float(logs.get("loss", float("nan")))
        val_loss = float(logs.get("val_loss", float("nan")))
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        elapsed = time.monotonic() - self.started_at if self.started_at else 0.0
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch + 1
            self.since_improvement = 0
            self.event.emit(f"Epoch {epoch + 1}: validation loss improved")
        else:
            self.since_improvement += 1
        if self.previous_lr is not None and lr < self.previous_lr:
            self.reduce_lr_count += 1
            self.event.emit(f"Learning Rate reduced: {self.previous_lr:.6g} -> {lr:.6g}")
        self.previous_lr = lr
        self.epoch_finished.emit(epoch + 1, self.max_epochs, loss, val_loss, lr, elapsed,
                                 self.since_improvement, self.best_epoch, self.best_loss, self.reduce_lr_count)
        if self.stop_requested:
            self.event.emit("STOP requested at epoch boundary")

    def on_train_end(self, logs=None):
        if self.model and getattr(self.model, "stop_training", False):
            self.event.emit("Training stopped at a safe epoch boundary")
        else:
            self.event.emit("Training ended")


class TrainingWorker(QObject):
    status_changed = Signal(str)
    log_message = Signal(str)
    dataset_ready = Signal(dict)
    model_ready = Signal(int, int)
    epoch_update = Signal(int, int, float, float, float, float, int, int, float, int)
    evaluation_ready = Signal(dict)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, quick_test=False, selected_model_id="KAMP_LSTM_AE", config=None):
        super().__init__()
        self.quick_test = quick_test
        self.selected_model_id = selected_model_id
        self.config = {**default_training_config(), **(config or {})}
        self.callback = None

    def _log(self, message):
        self.log_message.emit(message)

    @Slot()
    def run(self):
        try:
            self.status_changed.emit("LOADING DATA")
            self.config = validate_training_config(self.config)
            apply_random_seed(int(self.config["random_seed"]))
            normal = core.validate_data(core.load_data(core.NORMAL_PATH), "정상", 0)
            anomaly = core.validate_data(core.load_data(core.OUTLIER_PATH), "이상", 1)
            self._log(f"Normal rows: {len(normal)}")
            self._log(f"Anomaly rows: {len(anomaly)}")
            self.status_changed.emit("PREPROCESSING")
            normal[core.FEATURES] = normal[core.FEATURES].abs()
            anomaly[core.FEATURES] = anomaly[core.FEATURES].abs()
            normal_train = normal.iloc[:core.KAMP_TRAIN_ROWS].copy()
            normal_test = normal.iloc[core.KAMP_TRAIN_ROWS:].copy()
            scaler = core.fit_scaler(normal_train)
            normal_train = core.transform_features(normal_train, scaler)
            normal_test = core.transform_features(normal_test, scaler)
            anomaly = core.transform_features(anomaly, scaler)
            spec = MODEL_REGISTRY[self.selected_model_id]
            train_bundle = create_task_bundle(normal_train, spec); normal_bundle = create_task_bundle(normal_test, spec); anomaly_bundle = create_task_bundle(anomaly, spec)
            nv, av = core.KAMP_VALID_NORMAL_SEQUENCES, core.KAMP_VALID_ANOMALY_SEQUENCES
            x_train, target_train = train_bundle.inputs, train_bundle.targets
            x_valid_normal, target_valid_normal, y_valid_normal = normal_bundle.inputs[:nv], normal_bundle.targets[:nv], normal_bundle.labels[:nv]
            x_test_normal, target_test_normal, y_test_normal = normal_bundle.inputs[nv:], normal_bundle.targets[nv:], normal_bundle.labels[nv:]
            x_valid_anomaly, target_valid_anomaly, y_valid_anomaly = anomaly_bundle.inputs[:av], anomaly_bundle.targets[:av], anomaly_bundle.labels[:av]
            x_test_anomaly, target_test_anomaly, y_test_anomaly = anomaly_bundle.inputs[av:], anomaly_bundle.targets[av:], anomaly_bundle.labels[av:]
            x_valid = np.vstack([x_valid_normal, x_valid_anomaly]); target_valid = np.vstack([target_valid_normal, target_valid_anomaly])
            y_valid = np.hstack([y_valid_normal, y_valid_anomaly]); x_test = np.vstack([x_test_normal, x_test_anomaly])
            target_test = np.vstack([target_test_normal, target_test_anomaly]); y_test = np.hstack([y_test_normal, y_test_anomaly])
            self.dataset_ready.emit({
                "normal_rows": len(normal), "anomaly_rows": len(anomaly),
                "train_sequences": len(x_train), "valid_normal": len(x_valid_normal),
                "valid_anomaly": len(x_valid_anomaly), "test_normal": len(x_test_normal),
                "test_anomaly": len(x_test_anomaly), "input_shape": "(20, 3)",
            })
            self._log(f"X_train created: {x_train.shape}")
            self.status_changed.emit("BUILDING MODEL")
            self._log(f"Selected Model: {spec.display_name} | Task: {spec.task_type} | Sequence: {core.SEQUENCE_LENGTH} | Horizon: {core.PREDICTION_HORIZON}")
            model = build_training_model(spec, self.config)
            optimizer = build_optimizer(self.config)
            loss = build_loss(self.config)
            model.compile(optimizer=optimizer, loss=loss)
            trainable_params = sum(int(np.prod(weight.shape)) for weight in model.trainable_weights)
            self.model_ready.emit(model.count_params(), trainable_params)
            max_epochs = core.QUICK_TEST_EPOCHS if self.quick_test else int(self.config["epochs"])
            self.callback = GUITrainingCallback(max_epochs, self.config["early_stopping_min_delta"])
            self.callback.epoch_finished.connect(self.epoch_update.emit)
            self.callback.event.connect(self.log_message.emit)
            callbacks = []
            if self.config["reduce_lr_enabled"]:
                callbacks.append(ReduceLROnPlateau(monitor="val_loss", factor=self.config["reduce_lr_factor"],
                    patience=self.config["reduce_lr_patience"], min_lr=self.config["min_lr"], verbose=0))
            if self.config["early_stopping_enabled"]:
                callbacks.append(EarlyStopping(monitor="val_loss", min_delta=self.config["early_stopping_min_delta"],
                    patience=self.config["early_stopping_patience"], mode="min",
                    restore_best_weights=self.config["restore_best_weights"], verbose=0))
            callbacks.append(self.callback)
            self.status_changed.emit("TRAINING")
            clean_valid_inputs, clean_valid_targets = x_valid[y_valid == 0], target_valid[y_valid == 0]
            fit_inputs, fit_targets, valid_inputs, valid_targets = prepare_fit_data(
                x_train, target_train, clean_valid_inputs, clean_valid_targets, spec.denoising,
                self.config["noise_mean"], self.config["noise_std"], self.config["noise_clip"], self.config["random_seed"],
                self.config["batch_size"])
            training_started = time.monotonic()
            fit_kwargs = {"validation_data": (valid_inputs, valid_targets), "epochs": max_epochs,
                          "shuffle": False, "callbacks": callbacks, "verbose": 0}
            if spec.denoising:
                history = model.fit(fit_inputs, **fit_kwargs)
            else:
                history = model.fit(fit_inputs, fit_targets, batch_size=int(self.config["batch_size"]), **fit_kwargs)
            self.status_changed.emit("EVALUATING")
            valid_prediction = model.predict(x_valid, verbose=0); valid_error = np.square(target_valid - valid_prediction)[:, -1, :].mean(axis=1)
            threshold, val_precision, val_recall, _, _, _ = core.calculate_pr_intersection_threshold(y_valid, valid_error)
            test_prediction = model.predict(x_test, verbose=0); test_error = np.square(target_test - test_prediction)[:, -1, :].mean(axis=1)
            y_pred = (test_error > threshold).astype(int)
            metrics = {"accuracy": float(core.accuracy_score(y_test, y_pred)),
                       "precision": float(core.precision_score(y_test, y_pred, zero_division=0)),
                       "recall": float(core.recall_score(y_test, y_pred, zero_division=0)),
                       "f1_score": float(core.f1_score(y_test, y_pred, zero_division=0)),
                       "confusion_matrix": core.confusion_matrix(y_test, y_pred).tolist()}
            run = save_model_run(model, history, spec, {**self.config, "requested_epochs":max_epochs,
                "completion_status": "STOPPED" if self.callback.stop_requested else "EARLY_STOPPED" if len(history.history.get("loss", [])) < max_epochs else "COMPLETED",
                "monitor_best_epoch": self.callback.best_epoch, "monitor_best_val_loss": self.callback.best_loss,
                "training_duration_seconds": time.monotonic() - training_started,
                "normal_rows":len(normal), "anomaly_rows":len(anomaly), "train_sequences":len(x_train),
                "valid_samples":len(x_valid), "test_samples":len(x_test)})
            append_evaluation_result(run, {"model_id":spec.id, "experiment_name":run.metadata["experiment_name"],
                "score_method":"LAST_STEP_MSE", "threshold_method":"PR_INTERSECTION", "threshold":threshold,
                "accuracy":metrics["accuracy"], "precision":metrics["precision"], "recall":metrics["recall"],
                "f1":metrics["f1_score"], "tn":metrics["confusion_matrix"][0][0], "fp":metrics["confusion_matrix"][0][1],
                "fn":metrics["confusion_matrix"][1][0], "tp":metrics["confusion_matrix"][1][1]})
            self._log(f"Model run saved: {run.model_path}")
            precision_curve, recall_curve, threshold_curve = core.precision_recall_curve(y_valid, valid_error)
            self.evaluation_ready.emit({"threshold": threshold, "val_precision": val_precision,
                                        "val_recall": val_recall, "test_errors": test_error.tolist(),
                                        "test_labels": y_test.tolist(), "precision_curve": precision_curve.tolist(),
                                        "recall_curve": recall_curve.tolist(), "threshold_curve": threshold_curve.tolist(),
                                        **metrics})
            self.finished.emit("STOPPED" if self.callback.stop_requested else "COMPLETED")
        except Exception as exc:  # surface errors in the GUI instead of crashing it
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    def request_stop(self):
        if self.callback:
            self.status_changed.emit("STOPPING")
            self.callback.request_stop()
