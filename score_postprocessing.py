"""Causal processing of anomaly scores after model scoring and before thresholds."""
from __future__ import annotations

import numpy as np


TEMPORAL_METHOD_NONE = "NONE"
TEMPORAL_METHOD_EWMA = "EWMA"
DEFAULT_EWMA_ALPHA = 0.4


class EWMAProcessor:
    """One stream of EWMA state; callers reset at recording boundaries."""

    def __init__(self, alpha: float = DEFAULT_EWMA_ALPHA):
        self.alpha = float(alpha)
        if not np.isfinite(self.alpha) or not 0 < self.alpha <= 1:
            raise ValueError("EWMA alpha must satisfy 0 < alpha <= 1")
        self.reset()

    def reset(self) -> None:
        self._value: float | None = None

    def update(self, score: float) -> float:
        score = float(score)
        if not np.isfinite(score):
            raise ValueError("Anomaly scores must be finite")
        if self._value is None:
            self._value = score
        else:
            self._value = self.alpha * score + (1 - self.alpha) * self._value
        return self._value


def apply_temporal_processing(
    scores,
    method: str = TEMPORAL_METHOD_NONE,
    alpha: float = DEFAULT_EWMA_ALPHA,
    segment_ids=None,
    contains_timestamp_gap=None,
) -> np.ndarray:
    """Process a score vector without carrying state between independent calls.

    A changed segment ID or a gap-containing window starts a fresh EWMA state
    at that sample. NONE preserves input values and dtype exactly.
    """
    values = np.asarray(scores)
    if values.ndim != 1:
        raise ValueError("Expected a one-dimensional score vector")
    if method not in (TEMPORAL_METHOD_NONE, TEMPORAL_METHOD_EWMA):
        raise ValueError(f"Unknown temporal processing method: {method}")
    if method == TEMPORAL_METHOD_NONE:
        return values.copy()

    processor = EWMAProcessor(alpha)
    numeric = np.asarray(values, dtype=np.float64)
    ids = None if segment_ids is None else np.asarray(segment_ids)
    gaps = None if contains_timestamp_gap is None else np.asarray(contains_timestamp_gap)
    if ids is not None and (ids.ndim != 1 or len(ids) != len(values)):
        raise ValueError("segment_ids must match the score vector")
    if gaps is not None and (gaps.ndim != 1 or len(gaps) != len(values)):
        raise ValueError("contains_timestamp_gap must match the score vector")
    result = np.empty(len(values), dtype=np.float64)
    for index, score in enumerate(numeric):
        if index and ((ids is not None and ids[index] != ids[index - 1])
                      or (gaps is not None and bool(gaps[index]))):
            processor.reset()
        result[index] = processor.update(score)
    return result


def detect_timestamp_segments(
    timestamps,
    expected_interval: float | None = None,
    gap_factor: float = 1.5,
) -> tuple[np.ndarray, float | None]:
    """Return segment IDs and gap threshold in seconds.

    The nominal interval is the median of positive consecutive differences.
    Non-increasing and non-finite differences also break continuity. Input may
    contain numeric seconds or datetime-like values.
    """
    values = np.asarray(timestamps)
    if values.ndim != 1:
        raise ValueError("Expected a one-dimensional timestamp vector")
    if not np.isfinite(gap_factor) or gap_factor <= 1:
        raise ValueError("gap_factor must be finite and greater than 1")
    if np.issubdtype(values.dtype, np.datetime64):
        seconds = values.astype("datetime64[ns]").astype(np.int64).astype(np.float64) / 1e9
        seconds[np.isnat(values)] = np.nan
    elif values.dtype == object and len(values) and hasattr(values[0], "timestamp"):
        seconds = np.array([float(value.timestamp()) if value is not None else np.nan
                            for value in values], dtype=np.float64)
    else:
        seconds = np.asarray(values, dtype=np.float64)
    differences = np.diff(seconds)
    positive = differences[np.isfinite(differences) & (differences > 0)]
    if expected_interval is None:
        interval = float(np.median(positive)) if len(positive) else None
    else:
        interval = float(expected_interval)
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError("expected_interval must be finite and positive")
    threshold = None if interval is None else interval * float(gap_factor)
    breaks = ~np.isfinite(differences) | (differences <= 0)
    if threshold is not None:
        breaks |= differences > threshold
    segment_ids = np.zeros(len(values), dtype=np.int64)
    if len(values) > 1:
        segment_ids[1:] = np.cumsum(breaks)
    return segment_ids, threshold


def fp_fn_pareto_flags(fp, fn) -> np.ndarray:
    """Mark configurations not strictly dominated in both FP/FN objectives."""
    false_positives = np.asarray(fp)
    false_negatives = np.asarray(fn)
    if (false_positives.ndim != 1 or false_negatives.ndim != 1
            or len(false_positives) != len(false_negatives)):
        raise ValueError("FP and FN vectors must have equal one-dimensional shape")
    flags = np.ones(len(false_positives), dtype=bool)
    for index in range(len(flags)):
        dominates = ((false_positives <= false_positives[index])
                     & (false_negatives <= false_negatives[index])
                     & ((false_positives < false_positives[index])
                        | (false_negatives < false_negatives[index])))
        flags[index] = not bool(np.any(dominates))
    return flags
