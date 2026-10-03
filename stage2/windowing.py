"""Generate and verify windows wholly inside one timestamp segment."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from train_lstm_ae import FEATURES, LABEL_COLUMN, TIMESTAMP_COLUMN


@dataclass(frozen=True)
class WindowBatch:
    X: np.ndarray
    y: np.ndarray
    segment_ids: np.ndarray
    window_start_timestamp: np.ndarray
    window_end_timestamp: np.ndarray
    source_row_start: np.ndarray
    source_row_end: np.ndarray
    source_original_indices: np.ndarray
    cross_segment_windows: int = 0


def window_count_summary(frame: pd.DataFrame, sequence_length: int, stride: int = 1) -> dict:
    if type(sequence_length) is not int or sequence_length <= 0 or type(stride) is not int or stride <= 0:
        raise ValueError("sequence_length and stride must be positive integers")
    lengths = frame.groupby("segment_id", sort=False).size().to_numpy(dtype=np.int64)
    counts = np.where(lengths >= sequence_length, (lengths - sequence_length) // stride + 1, 0)
    return {"eligible_segments": int((lengths >= sequence_length).sum()),
            "windows": int(counts.sum()), "windows_per_segment": counts.tolist(),
            "segment_lengths": lengths.tolist()}


def generate_windows(frame: pd.DataFrame, sequence_length: int, stride: int = 1,
                     gap_threshold_ms: int = 150, features=FEATURES) -> WindowBatch:
    if type(sequence_length) is not int or sequence_length <= 0 or type(stride) is not int or stride <= 0:
        raise ValueError("sequence_length and stride must be positive integers")
    if gap_threshold_ms <= 0:
        raise ValueError("gap_threshold_ms must be positive")
    required = [*features, LABEL_COLUMN, TIMESTAMP_COLUMN, "segment_id", "source_row"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing window columns: {missing}")
    windows, labels, ids, starts, ends, row_starts, row_ends, original_rows = ([] for _ in range(8))
    for segment_id, group in frame.groupby("segment_id", sort=False):
        values = group[list(features)].to_numpy(dtype=np.float32)
        segment_labels = group[LABEL_COLUMN].to_numpy(dtype=np.int8)
        timestamps = group[TIMESTAMP_COLUMN].to_numpy(dtype="datetime64[ns]")
        source_rows = group["source_row"].to_numpy(dtype=np.int64)
        if len(values) < sequence_length:
            continue
        for start in range(0, len(values) - sequence_length + 1, stride):
            stop = start + sequence_length
            window_times = timestamps[start:stop]
            deltas = np.diff(window_times).astype("timedelta64[ns]").astype(np.int64) / 1e6
            if np.any(deltas <= 0) or np.any(deltas > gap_threshold_ms):
                raise ValueError(f"Window crosses a timestamp discontinuity in segment {segment_id}")
            if not np.all(segment_labels[start:stop] == segment_labels[start]):
                raise ValueError(f"Window contains mixed labels in segment {segment_id}")
            windows.append(values[start:stop]); labels.append(segment_labels[start])
            ids.append(segment_id); starts.append(window_times[0]); ends.append(window_times[-1])
            row_starts.append(source_rows[start]); row_ends.append(source_rows[stop - 1])
            original_rows.append(source_rows[start:stop])
    count = len(windows)
    batch = WindowBatch(
        np.stack(windows).astype(np.float32) if count else np.empty((0, sequence_length, len(features)), dtype=np.float32),
        np.asarray(labels, dtype=np.int8), np.asarray(ids, dtype=np.int64),
        np.asarray(starts, dtype="datetime64[ns]"), np.asarray(ends, dtype="datetime64[ns]"),
        np.asarray(row_starts, dtype=np.int64), np.asarray(row_ends, dtype=np.int64),
        np.stack(original_rows) if count else np.empty((0, sequence_length), dtype=np.int64))
    expected = window_count_summary(frame, sequence_length, stride)["windows"]
    if len(batch.X) != expected:
        raise AssertionError(f"Generated {len(batch.X)} windows; expected {expected}")
    return batch
