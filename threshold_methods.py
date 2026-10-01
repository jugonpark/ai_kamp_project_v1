"""Validation-only threshold registry for KAMP anomaly scores."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import genpareto
from sklearn.metrics import precision_recall_curve


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    requested_method: str
    effective_method: str
    fallback_used: bool = False
    fallback_reason: str = ""


@dataclass(frozen=True)
class ThresholdMethod:
    id: str
    display_name: str
    description: str
    calculate: Callable[[np.ndarray, np.ndarray], ThresholdResult]


def _validate(scores, labels):
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=int).reshape(-1)
    if len(values) != len(y) or not len(values): raise ValueError("Validation scores and labels must be non-empty and aligned")
    if not np.isfinite(values).all(): raise ValueError("Validation scores must be finite")
    return values, y


def _pr(method, scores, labels, maximize_f1=False):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if not len(thresholds): raise ValueError(f"{method} requires both validation classes")
    p, r = precision[:-1], recall[:-1]
    index = int(np.argmax(2*p*r/(p+r+1e-12))) if maximize_f1 else int(np.argmin(np.abs(p-r)))
    return ThresholdResult(float(thresholds[index]), method, method)


def _percentile(method, scores, labels, percentile):
    normal = scores[labels == 0]
    if not len(normal): raise ValueError(f"{method} requires validation normal scores")
    return ThresholdResult(float(np.percentile(normal, percentile)), method, method)


def _pot(method, scores, labels, probability, fallback_method, fallback_percentile):
    normal = scores[labels == 0]
    fallback = lambda reason: ThresholdResult(float(np.percentile(normal, fallback_percentile)), method, fallback_method, True, reason)
    if not len(normal): raise ValueError(f"{method} requires validation normal scores")
    u = float(np.percentile(normal, 95))
    excess = normal[normal > u] - u
    if len(excess) < 10: return fallback(f"POT fit requires at least 10 excesses; got {len(excess)}")
    try:
        shape, _, scale = genpareto.fit(excess, floc=0)
        tail_fraction = len(excess) / len(normal)
        quantile = 1.0 - probability / tail_fraction
        if not 0 < quantile < 1 or not np.isfinite(scale) or scale <= 0: return fallback("POT fit produced invalid tail parameters")
        threshold = u + float(genpareto.ppf(quantile, shape, loc=0, scale=scale))
        if not np.isfinite(threshold): return fallback("POT threshold is not finite")
        return ThresholdResult(threshold, method, method)
    except Exception as exc:
        return fallback(f"POT fit failed: {type(exc).__name__}: {exc}")


THRESHOLD_METHODS = {
    "PR_INTERSECTION": ThresholdMethod("PR_INTERSECTION", "정밀도-재현율 균형점", "Validation PR curve에서 precision과 recall 차이가 가장 작은 threshold.", lambda s,y: _pr("PR_INTERSECTION",s,y)),
    "MAX_F1": ThresholdMethod("MAX_F1", "최대 F1 기준", "Validation PR curve에서 F1이 최대인 threshold.", lambda s,y: _pr("MAX_F1",s,y,True)),
    "NORMAL_P99": ThresholdMethod("NORMAL_P99", "정상 데이터 상위 1% 기준 (P99)", "정상 Validation score의 99 percentile을 경계로 사용.", lambda s,y: _percentile("NORMAL_P99",s,y,99.0)),
    "NORMAL_P995": ThresholdMethod("NORMAL_P995", "정상 데이터 상위 0.5% 기준 (P99.5)", "정상 Validation score의 99.5 percentile을 경계로 사용.", lambda s,y: _percentile("NORMAL_P995",s,y,99.5)),
    "POT_1PCT": ThresholdMethod("POT_1PCT", "POT 상위 1% 기준", "정상 Validation 상위 5% excess에 GPD를 적합해 1% tail threshold를 계산.", lambda s,y: _pot("POT_1PCT",s,y,.01,"NORMAL_P99",99.0)),
    "POT_0P5PCT": ThresholdMethod("POT_0P5PCT", "POT 상위 0.5% 기준", "정상 Validation 상위 5% excess에 GPD를 적합해 0.5% tail threshold를 계산.", lambda s,y: _pot("POT_0P5PCT",s,y,.005,"NORMAL_P995",99.5)),
}


def calculate_threshold(method_id: str, valid_scores, y_valid) -> ThresholdResult:
    if method_id not in THRESHOLD_METHODS: raise KeyError(f"Unknown threshold method: {method_id}")
    scores, labels = _validate(valid_scores, y_valid)
    return THRESHOLD_METHODS[method_id].calculate(scores, labels)
