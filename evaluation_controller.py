"""Saved-model evaluation controller with validation-safe in-memory prediction cache."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

import train_lstm_ae as core
from model_registry import MODEL_REGISTRY
from model_data import create_task_bundle
from model_artifacts import discover_model_runs
from anomaly_scoring import SCORE_METHODS, compute_scores, fit_score_calibration
from threshold_methods import THRESHOLD_METHODS, calculate_threshold
from score_postprocessing import (TEMPORAL_METHOD_NONE, TEMPORAL_METHOD_EWMA,
                                  apply_temporal_processing, fp_fn_pareto_flags)


EWMA_SWEEP_ALPHAS = (0.02, 0.04, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00)


class EvaluationController:
    _prediction_cache: dict[tuple, dict] = {}

    def __init__(self, model_path=None, normal_path=None, anomaly_path=None, model_id="KAMP_LSTM_AE"):
        self.model_id = model_id; self.model_spec = MODEL_REGISTRY[model_id]
        self.model_path = Path(model_path or (core.MODEL_DIR / "kamp_lstm_autoencoder.keras"))
        self.normal_path = Path(normal_path or core.NORMAL_PATH)
        self.anomaly_path = Path(anomaly_path or core.OUTLIER_PATH)
        self._bundle: dict | None = None

    def select_run(self, model_run):
        self.model_run = model_run
        self.model_id = model_run.model_id; self.model_spec = MODEL_REGISTRY[self.model_id]
        self.model_path = Path(model_run.model_path); self._bundle = None

    @property
    def is_loaded(self): return self._bundle is not None

    @staticmethod
    def _signature(path: Path):
        if not path.is_file(): raise FileNotFoundError(f"Missing evaluation file: {path}")
        stat = path.stat()
        return str(path.resolve()), stat.st_mtime_ns, stat.st_size

    def cache_key(self):
        paths = (self.model_path, self.normal_path, self.anomaly_path)
        signatures = tuple(self._signature(path) for path in paths)
        config = (self.model_id, self.model_spec.task_type, self.model_spec.forecast_length,
                  core.SEQUENCE_LENGTH, core.PREDICTION_HORIZON, core.KAMP_TRAIN_ROWS,
                  core.KAMP_VALID_NORMAL_SEQUENCES, core.KAMP_VALID_ANOMALY_SEQUENCES,
                  tuple(core.FEATURES))
        return signatures + config

    def load_model_and_predictions(self, force=False):
        key = self.cache_key()
        if not force and key in self._prediction_cache:
            self._bundle = self._prediction_cache[key]
            return {"cache_hit": True, **self._bundle["summary"]}

        normal = core.validate_data(core.load_data(self.normal_path), "정상", 0)
        anomaly = core.validate_data(core.load_data(self.anomaly_path), "이상", 1)
        normal[core.FEATURES] = normal[core.FEATURES].abs()
        anomaly[core.FEATURES] = anomaly[core.FEATURES].abs()
        normal_train = normal.iloc[:core.KAMP_TRAIN_ROWS].copy()
        normal_test = normal.iloc[core.KAMP_TRAIN_ROWS:].copy()
        scaler = core.fit_scaler(normal_train)
        normal_test = core.transform_features(normal_test, scaler)
        anomaly = core.transform_features(anomaly, scaler)
        normal_bundle = create_task_bundle(normal_test, self.model_spec); anomaly_bundle = create_task_bundle(anomaly, self.model_spec)
        x_normal, target_normal, y_normal = normal_bundle.inputs, normal_bundle.targets, normal_bundle.labels
        x_anomaly, target_anomaly, y_anomaly = anomaly_bundle.inputs, anomaly_bundle.targets, anomaly_bundle.labels
        nv, av = core.KAMP_VALID_NORMAL_SEQUENCES, core.KAMP_VALID_ANOMALY_SEQUENCES
        x_valid = np.vstack([x_normal[:nv], x_anomaly[:av]])
        target_valid = np.vstack([target_normal[:nv], target_anomaly[:av]])
        y_valid = np.hstack([y_normal[:nv], y_anomaly[:av]]).astype(int)
        x_test = np.vstack([x_normal[nv:], x_anomaly[av:]])
        target_test = np.vstack([target_normal[nv:], target_anomaly[av:]])
        y_test = np.hstack([y_normal[nv:], y_anomaly[av:]]).astype(int)
        def evaluation_metadata(normal_slice, anomaly_slice):
            normal_count = len(normal_slice)
            anomaly_count = len(anomaly_slice)
            normal_segments = normal_bundle.segment_ids
            anomaly_segments = anomaly_bundle.segment_ids
            if normal_segments is None or anomaly_segments is None:
                segment_ids = np.r_[np.zeros(normal_count, dtype=int),
                                    np.ones(anomaly_count, dtype=int)]
            else:
                normal_ids = np.asarray(normal_segments[normal_slice], dtype=np.int64)
                anomaly_ids = np.asarray(anomaly_segments[anomaly_slice], dtype=np.int64)
                segment_ids = np.r_[normal_ids, anomaly_ids + int(normal_ids.max(initial=-1)) + 1]
            gaps = None
            if normal_bundle.contains_timestamp_gap is not None and anomaly_bundle.contains_timestamp_gap is not None:
                gaps = np.r_[normal_bundle.contains_timestamp_gap[normal_slice],
                             anomaly_bundle.contains_timestamp_gap[anomaly_slice]]
            timestamps = None
            if normal_bundle.sample_timestamps is not None and anomaly_bundle.sample_timestamps is not None:
                timestamps = np.r_[normal_bundle.sample_timestamps[normal_slice],
                                   anomaly_bundle.sample_timestamps[anomaly_slice]]
            stream_ids = np.r_[np.zeros(normal_count, dtype=int),
                               np.ones(anomaly_count, dtype=int)]
            return {"segment_ids": segment_ids, "stream_ids": stream_ids,
                    "contains_timestamp_gap": gaps,
                    "sample_timestamps": timestamps,
                    "timestamp_gap_threshold": normal_bundle.timestamp_gap_threshold}
        valid_metadata = evaluation_metadata(slice(None, nv), slice(None, av))
        test_metadata = evaluation_metadata(slice(nv, None), slice(av, None))
        model = tf.keras.models.load_model(self.model_path, compile=False)
        expected = (core.SEQUENCE_LENGTH, len(core.FEATURES))
        if tuple(model.input_shape[1:]) != expected:
            raise ValueError(f"Model input shape {model.input_shape[1:]} does not match {expected}")
        valid_prediction = model.predict(x_valid, verbose=0)
        test_prediction = model.predict(x_test, verbose=0)
        if valid_prediction.shape != target_valid.shape or test_prediction.shape != target_test.shape:
            raise ValueError(f"Prediction/target shape mismatch: {valid_prediction.shape} vs {target_valid.shape}")
        valid_error = np.square(target_valid - valid_prediction)
        test_error = np.square(target_test - test_prediction)
        if not np.isfinite(valid_error).all() or not np.isfinite(test_error).all():
            raise ValueError("Model prediction produced NaN or Inf reconstruction error")
        summary = {"model_path": str(self.model_path.resolve()), "model_id":self.model_id,
                   "task_type":self.model_spec.task_type,"error_name":normal_bundle.error_name,"valid_samples": len(x_valid),
                   "test_samples": len(x_test), "valid_normal": int((y_valid == 0).sum()),
                   "valid_anomaly": int((y_valid == 1).sum())}
        self._bundle = {"x_valid": x_valid, "x_test": x_test,
                        "valid_prediction": valid_prediction, "test_prediction": test_prediction,
                        "valid_error": valid_error, "test_error": test_error,
                        "y_valid": y_valid, "y_test": y_test, "summary": summary,
                        "valid_metadata": valid_metadata, "test_metadata": test_metadata}
        self._prediction_cache[key] = self._bundle
        return {"cache_hit": False, **summary}

    @staticmethod
    def _delay(y_test, prediction, metadata):
        """First detection within an observed anomaly recording; otherwise N/A."""
        timestamps = metadata.get("sample_timestamps")
        segments = metadata.get("segment_ids")
        if timestamps is None or segments is None or not np.any(y_test == 1):
            return "", ""
        timestamps = np.asarray(timestamps)
        segments = np.asarray(segments)
        for start in np.flatnonzero((y_test == 1) & np.r_[True, (y_test[1:] != y_test[:-1]) | (segments[1:] != segments[:-1])]):
            end = start + 1
            while end < len(y_test) and y_test[end] == 1 and segments[end] == segments[start]:
                end += 1
            detections = np.flatnonzero(prediction[start:end] == 1)
            if not len(detections):
                continue
            detected = start + int(detections[0])
            try:
                seconds = float((timestamps[detected] - timestamps[start]) / np.timedelta64(1, 's'))
            except (TypeError, ValueError):
                return "", ""
            if np.isfinite(seconds) and seconds >= 0:
                return detected - start, seconds
        return "", ""

    def evaluate(self, score_method: str, threshold_method: str,
                 temporal_method: str = TEMPORAL_METHOD_NONE, ewma_alpha: float = 0.4,
                 timestamp_aware: bool = True):
        if not self._bundle: raise RuntimeError("Load a model before evaluation")
        if score_method not in SCORE_METHODS: raise KeyError(f"Unknown score method: {score_method}")
        if threshold_method not in THRESHOLD_METHODS: raise KeyError(f"Unknown threshold method: {threshold_method}")
        if temporal_method not in (TEMPORAL_METHOD_NONE, TEMPORAL_METHOD_EWMA):
            raise ValueError(f"Unknown temporal method: {temporal_method}")
        valid_error, test_error = self._bundle["valid_error"], self._bundle["test_error"]
        y_valid, y_test = self._bundle["y_valid"], self._bundle["y_test"]
        calibration = fit_score_calibration(score_method, valid_error[y_valid == 0])
        valid_scores = compute_scores(score_method, valid_error, calibration)
        test_scores = compute_scores(score_method, test_error, calibration)
        if temporal_method != TEMPORAL_METHOD_NONE:
            valid_meta = self._bundle.get("valid_metadata", {})
            test_meta = self._bundle.get("test_metadata", {})
            valid_scores = apply_temporal_processing(
                valid_scores, temporal_method, ewma_alpha,
                valid_meta.get("segment_ids") if timestamp_aware else valid_meta.get("stream_ids"),
                valid_meta.get("contains_timestamp_gap") if timestamp_aware else None)
            test_scores = apply_temporal_processing(
                test_scores, temporal_method, ewma_alpha,
                test_meta.get("segment_ids") if timestamp_aware else test_meta.get("stream_ids"),
                test_meta.get("contains_timestamp_gap") if timestamp_aware else None)
        threshold_result = calculate_threshold(threshold_method, valid_scores, y_valid)
        prediction = (test_scores > threshold_result.threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
        delay_samples, delay_seconds = self._delay(y_test, prediction, self._bundle.get("test_metadata", {}))
        gap_threshold = self._bundle.get("test_metadata", {}).get("timestamp_gap_threshold")
        result = {
            "model": self.model_spec.display_name, "model_id":self.model_id, "task_type":self.model_spec.task_type,
            "error_name": "Prediction Error" if self.model_spec.task_type == "FORECAST" else "Reconstruction Error", "score_method": score_method,
            "score_name": SCORE_METHODS[score_method].display_name,
            "threshold_method": threshold_method,
            "threshold_name": THRESHOLD_METHODS[threshold_method].display_name,
            "threshold": threshold_result.threshold,
            "effective_threshold_method": threshold_result.effective_method,
            "fallback_used": threshold_result.fallback_used,
            "fallback_reason": threshold_result.fallback_reason,
            "accuracy": float(accuracy_score(y_test, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
            "precision": float(precision_score(y_test, prediction, zero_division=0)),
            "recall": float(recall_score(y_test, prediction, zero_division=0)),
            "f1": float(f1_score(y_test, prediction, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
            "fpr": float(fp / (tn + fp)) if tn + fp else 0.0,
            "fnr": float(fn / (fn + tp)) if fn + tp else 0.0,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "total_error": int(fp + fn), "temporal_method": temporal_method,
            "ewma_alpha": float(ewma_alpha) if temporal_method == TEMPORAL_METHOD_EWMA else "",
            "timestamp_aware": bool(timestamp_aware),
            "timestamp_gap_threshold": gap_threshold if gap_threshold is not None else "",
            "detection_delay_samples": delay_samples,
            "detection_delay_seconds": delay_seconds, "pareto": "",
        }
        metadata = getattr(getattr(self, "model_run", None), "metadata", {})
        result.update({
            "status": metadata.get("model_status", self.model_spec.status),
            "experiment_name": metadata.get("experiment_name", ""), "seed": metadata.get("random_seed", ""),
            "optimizer": metadata.get("optimizer", ""), "learning_rate": metadata.get("learning_rate", ""),
            "loss": metadata.get("loss", ""), "batch_size": metadata.get("batch_size", ""),
            "cnn_filters": metadata.get("cnn_filters", "unknown"),
            "cnn_kernel_size": metadata.get("cnn_kernel_size", "unknown"),
            "bottleneck_units": metadata.get("bottleneck_units", "unknown"),
            "noise_std": metadata.get("noise_std", ""), "best_epoch": metadata.get("best_epoch", ""),
            "best_val_loss": metadata.get("best_val_loss", ""),
            "training_time": metadata.get("training_duration_seconds", ""), "created_at": metadata.get("created_at", ""),
        })
        return result

    def compare_all(self, progress=None, temporal_method=TEMPORAL_METHOD_NONE,
                    ewma_alpha=0.4, timestamp_aware=True):
        total = len(SCORE_METHODS) * len(THRESHOLD_METHODS)
        results = []
        for score_id in SCORE_METHODS:
            for threshold_id in THRESHOLD_METHODS:
                results.append(self.evaluate(score_id, threshold_id, temporal_method,
                                             ewma_alpha, timestamp_aware))
                if progress: progress(len(results), total)
        return results

    def sweep_ewma_alphas(self, score_method, threshold_method,
                          timestamp_aware=True, progress=None):
        candidates = [(TEMPORAL_METHOD_NONE, 0.4)] + [
            (TEMPORAL_METHOD_EWMA, alpha) for alpha in EWMA_SWEEP_ALPHAS]
        results = []
        for method, alpha in candidates:
            results.append(self.evaluate(score_method, threshold_method,
                                         method, alpha, timestamp_aware))
            if progress:
                progress(len(results), len(candidates))
        flags = fp_fn_pareto_flags([row["fp"] for row in results],
                                   [row["fn"] for row in results])
        baseline = results[0]
        for row, flag in zip(results, flags):
            row["pareto"] = bool(flag)
            row["delta_fp"] = row["fp"] - baseline["fp"]
            row["delta_fn"] = row["fn"] - baseline["fn"]
            row["delta_total_error"] = row["total_error"] - baseline["total_error"]
        return results

    @classmethod
    def compare_models(cls, score_method, threshold_method, progress=None):
        results = []
        model_ids = list(MODEL_REGISTRY)
        for model_id in model_ids:
            spec = MODEL_REGISTRY[model_id]
            if score_method not in spec.compatible_score_methods: continue
            runs = discover_model_runs(model_id)
            if not runs: continue
            controller = cls(model_path=runs[0].model_path, model_id=model_id)
            controller.model_run = runs[0]
            controller.load_model_and_predictions(); results.append(controller.evaluate(score_method, threshold_method))
            if progress: progress(len(results), len(model_ids))
        return results
