"""Validate raw recordings and mark timestamp-continuous segments."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import numpy as np
import pandas as pd

from preprocessing_config import validate_preprocessing_config
from score_postprocessing import detect_timestamp_segments
from train_lstm_ae import FEATURES, LABEL_COLUMN, TIMESTAMP_COLUMN


REQUIRED_COLUMNS = (TIMESTAMP_COLUMN, *FEATURES, LABEL_COLUMN)
LENGTH_BUCKETS = (("1-9", 1, 9), ("10-19", 10, 19), ("20-29", 20, 29),
                  ("30-39", 30, 39), ("40-49", 40, 49), ("50+", 50, None))


class DataQualityError(ValueError):
    def __init__(self, message: str, counts: dict | None = None):
        super().__init__(message)
        self.counts = counts or {}


@dataclass(frozen=True)
class QualityResult:
    frame: pd.DataFrame
    report: dict


def _drop_exported_row_number(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Ignore only sequential CSV export indexes, not arbitrary source columns."""
    ignored = []
    for column in frame.columns:
        if re.fullmatch(r"Unnamed: \d+", str(column)):
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
            if np.array_equal(values, np.arange(len(frame))):
                ignored.append(column)
    return frame.drop(columns=ignored), ignored


def inspect_dataframe(frame: pd.DataFrame, config: dict, expected_label: int | None = None) -> QualityResult:
    config = validate_preprocessing_config(config)
    if frame.empty:
        raise DataQualityError("Stage 2 dataset is empty")
    raw, ignored = _drop_exported_row_number(frame.copy())
    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise DataQualityError(f"Missing required columns: {missing}")
    if "segment_id" in raw or "delta_t_ms" in raw or "source_row" in raw:
        raise DataQualityError("Input contains reserved Stage 2 metadata columns")
    rows_before = len(raw)
    duplicate_mask = raw.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())
    cleaned = raw.loc[~duplicate_mask if config["remove_exact_duplicates"] else slice(None)].copy()
    cleaned["source_row"] = cleaned.index.to_numpy(dtype=np.int64)
    for feature in FEATURES:
        cleaned[feature] = pd.to_numeric(cleaned[feature], errors="coerce")
    nan_count = int(cleaned[list(FEATURES)].isna().sum().sum())
    inf_count = int(np.isinf(cleaned[list(FEATURES)].to_numpy(dtype=float)).sum())
    if nan_count or inf_count:
        counts = {"nan": nan_count, "inf": inf_count, "duplicates": duplicate_count}
        raise DataQualityError(f"Sensor values contain NaN={nan_count}, Inf={inf_count}; preprocessing stopped", counts)
    timestamps = pd.to_datetime(cleaned[TIMESTAMP_COLUMN], errors="coerce")
    invalid_timestamps = int(timestamps.isna().sum())
    if invalid_timestamps:
        raise DataQualityError(f"TimeStamp parse failed for {invalid_timestamps} rows", {"invalid_timestamps": invalid_timestamps})
    cleaned[TIMESTAMP_COLUMN] = timestamps
    labels = pd.to_numeric(cleaned[LABEL_COLUMN], errors="coerce")
    if labels.isna().any() or not labels.isin((0, 1)).all():
        raise DataQualityError("Equipment_state must contain only 0 or 1")
    if expected_label is not None and not labels.eq(expected_label).all():
        raise DataQualityError(f"Expected Equipment_state={expected_label} for every row")
    cleaned[LABEL_COLUMN] = labels.astype(np.int8)
    delta_ms = timestamps.diff().dt.total_seconds().mul(1000)
    backward = int((delta_ms < 0).sum())
    non_increasing = int((delta_ms <= 0).sum())
    cleaned["delta_t_ms"] = delta_ms
    ids, threshold_seconds = detect_timestamp_segments(
        timestamps.to_numpy(), expected_interval=config["expected_interval_ms"] / 1000,
        gap_threshold_seconds=config["gap_threshold_ms"] / 1000)
    cleaned["segment_id"] = ids
    lengths = cleaned.groupby("segment_id", sort=False).size().to_numpy(dtype=np.int64)
    distribution = {name: int(((lengths >= lower) & ((lengths <= upper) if upper else True)).sum())
                    for name, lower, upper in LENGTH_BUCKETS}
    report = {"rows_before": rows_before, "rows_after": len(cleaned),
              "exact_duplicates": duplicate_count,
              "duplicates_removed": duplicate_count if config["remove_exact_duplicates"] else 0,
              "ignored_export_index_columns": ignored,
              "nan_count": nan_count, "inf_count": inf_count,
              "timestamp_backward_count": backward, "timestamp_non_increasing_count": non_increasing,
              "sampling_interval_ms": float(delta_ms[delta_ms > 0].median()),
              "gap_threshold_ms": float(threshold_seconds * 1000),
              "segment_count": len(lengths), "segment_length_min": int(lengths.min()),
              "segment_length_mean": float(lengths.mean()),
              "segment_length_median": float(np.median(lengths)),
              "segment_length_max": int(lengths.max()), "segment_length_distribution": distribution}
    return QualityResult(cleaned.reset_index(drop=True), report)


def inspect_csv(path: str | Path, config: dict, expected_label: int | None = None) -> QualityResult:
    return inspect_dataframe(pd.read_csv(path), config, expected_label)


def fingerprint_csv_source(path: str | Path) -> dict:
    source = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(source), "sha256": digest.hexdigest(), "size_bytes": source.stat().st_size}
