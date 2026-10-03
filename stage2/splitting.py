"""Chronological, whole-segment splits for Stage 2 recordings."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SegmentSplits:
    normal_train: pd.DataFrame
    normal_validation: pd.DataFrame
    normal_test: pd.DataFrame
    anomaly_validation: pd.DataFrame
    anomaly_test: pd.DataFrame


def _split(frame: pd.DataFrame, sizes: tuple[int, ...]) -> tuple[pd.DataFrame, ...]:
    if "segment_id" not in frame:
        raise ValueError("Segment IDs are required before splitting")
    segment_ids = frame["segment_id"].drop_duplicates().to_numpy()
    if any(type(size) is not int or size <= 0 for size in sizes):
        raise ValueError("Split sizes must be positive integers")
    if sum(sizes) >= len(segment_ids):
        raise ValueError(f"Need more than {sum(sizes)} segments; found {len(segment_ids)}")
    parts = np.split(segment_ids, np.cumsum(sizes))
    result = tuple(frame.loc[frame["segment_id"].isin(ids)].copy().reset_index(drop=True) for ids in parts)
    sets = [set(part["segment_id"]) for part in result]
    if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i + 1, len(sets))):
        raise AssertionError("Segment overlap across splits")
    if sum(len(part) for part in result) != len(frame):
        raise AssertionError("Split lost source rows")
    return result


def split_segments(normal: pd.DataFrame, anomaly: pd.DataFrame,
                   normal_train_segments: int = 449, normal_validation_segments: int = 30,
                   anomaly_validation_segments: int = 10) -> SegmentSplits:
    normal_train, normal_validation, normal_test = _split(
        normal, (normal_train_segments, normal_validation_segments))
    anomaly_validation, anomaly_test = _split(anomaly, (anomaly_validation_segments,))
    return SegmentSplits(normal_train, normal_validation, normal_test,
                         anomaly_validation, anomaly_test)
