"""Registry-based anomaly scores for KAMP reconstruction errors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ScoreMethod:
    id: str
    display_name: str
    description: str
    compute: Callable[[np.ndarray, dict], np.ndarray]


def _as_error(values: np.ndarray) -> np.ndarray:
    error = np.asarray(values, dtype=np.float64)
    if error.ndim != 3 or error.shape[1] < 1 or error.shape[2] != 3:
        raise ValueError(f"Expected error shape (samples, T, 3) with T >= 1, got {error.shape}")
    if not np.isfinite(error).all() or (error < 0).any():
        raise ValueError("Reconstruction error must contain finite, non-negative values")
    return error


def _last_step(error, _): return error[:, -1, :].mean(axis=1)
def _full_window(error, _): return error.mean(axis=(1, 2))


def _robust_z(error: np.ndarray, calibration: dict) -> np.ndarray:
    return np.maximum((error - calibration["median"]) / calibration["scale"], 0.0)


def _robust_full(error, calibration): return _robust_z(error, calibration).mean(axis=(1, 2))


def _robust_topk(error, calibration, ratio):
    count = max(1, int(np.ceil(error.shape[1] * ratio)))
    time_score = _robust_z(error, calibration).mean(axis=2)
    return np.partition(time_score, -count, axis=1)[:, -count:].mean(axis=1)


def _time_p90(error, calibration):
    return np.percentile(_robust_z(error, calibration).mean(axis=2), 90, axis=1)


def _mahalanobis(error, calibration):
    residual = error.mean(axis=1) - calibration["mu"]
    squared = np.einsum("ij,jk,ik->i", residual, calibration["cov_inv"], residual)
    return np.sqrt(np.maximum(squared, 0.0))


SCORE_METHODS = {
    "LAST_STEP_MSE": ScoreMethod("LAST_STEP_MSE", "마지막 시점 MSE", "20개 시계열 중 마지막 시점의 3개 센서 복원 오차 평균을 사용하는 KAMP 가이드북 기준 방식.", _last_step),
    "FULL_WINDOW_MSE": ScoreMethod("FULL_WINDOW_MSE", "전체 구간 MSE", "20개 timestep과 3개 센서 전체 Reconstruction Error 평균을 이상점수로 사용.", _full_window),
    "ROBUST_CHANNEL_FULL": ScoreMethod("ROBUST_CHANNEL_FULL", "센서별 강건 오차", "정상 Validation의 센서별 median/MAD로 오차를 보정한 뒤 전체 평균을 사용.", _robust_full),
    "ROBUST_TOPK_10": ScoreMethod("ROBUST_TOPK_10", "상위 10% 오차", "센서별 보정 후 timestep 상위 10% 평균을 사용.", lambda e, c: _robust_topk(e, c, .10)),
    "ROBUST_TOPK_20": ScoreMethod("ROBUST_TOPK_20", "상위 20% 오차", "센서별 보정 후 timestep 상위 20% 평균을 사용.", lambda e, c: _robust_topk(e, c, .20)),
    "ROBUST_TOPK_30": ScoreMethod("ROBUST_TOPK_30", "상위 30% 오차", "센서별 보정 후 timestep 상위 30% 평균을 사용.", lambda e, c: _robust_topk(e, c, .30)),
    "TIME_P90": ScoreMethod("TIME_P90", "시간축 상위 90백분위 오차", "센서별 보정 timestep 점수의 sequence 내부 90 percentile을 사용.", _time_p90),
    "MAHALANOBIS_ERROR": ScoreMethod("MAHALANOBIS_ERROR", "Mahalanobis 오차", "3개 센서의 평균 오차가 정상 Validation 오차의 상관관계에서 벗어난 거리를 측정.", _mahalanobis),
}


def fit_score_calibration(method_id: str, valid_normal_error: np.ndarray) -> dict:
    if method_id not in SCORE_METHODS:
        raise KeyError(f"Unknown score method: {method_id}")
    error = _as_error(valid_normal_error)
    if not len(error):
        raise ValueError("Validation normal error is empty")
    if method_id.startswith("ROBUST_") or method_id == "TIME_P90":
        flattened = error.reshape(-1, error.shape[2])
        median = np.median(flattened, axis=0)
        mad = np.median(np.abs(flattened - median), axis=0)
        return {"median": median.reshape(1, 1, -1), "scale": (1.4826 * mad + 1e-12).reshape(1, 1, -1)}
    if method_id == "MAHALANOBIS_ERROR":
        residual = error.mean(axis=1)
        mu = residual.mean(axis=0)
        covariance = np.atleast_2d(np.cov(residual, rowvar=False))
        regularization = max(float(np.trace(covariance)) / 3.0 * 1e-6, 1e-12)
        return {"mu": mu, "cov_inv": np.linalg.pinv(covariance + regularization * np.eye(3))}
    return {}


def compute_scores(method_id: str, reconstruction_error: np.ndarray, calibration: dict | None = None) -> np.ndarray:
    if method_id not in SCORE_METHODS:
        raise KeyError(f"Unknown score method: {method_id}")
    error = _as_error(reconstruction_error)
    scores = np.asarray(SCORE_METHODS[method_id].compute(error, calibration or {}), dtype=np.float64)
    if scores.shape != (len(error),) or not np.isfinite(scores).all():
        raise ValueError(f"Score method {method_id} produced invalid values")
    return scores
