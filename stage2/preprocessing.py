"""End-to-end Stage 2 preparation without model training or artifact writes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from preprocessing_config import validate_preprocessing_config
from train_lstm_ae import FEATURES
from .data_quality import QualityResult, inspect_csv, fingerprint_csv_source
from .splitting import SegmentSplits, split_segments
from .windowing import WindowBatch, generate_windows, window_count_summary


def transform_signal(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    result = frame.copy(deep=True)
    if method == "ABS_ALL":
        result[list(FEATURES)] = result[list(FEATURES)].abs()
    elif method == "ABS_VIBRATION_RAW_CURRENT":
        result[list(FEATURES[:2])] = result[list(FEATURES[:2])].abs()
    elif method != "RAW_SIGNED":
        raise ValueError(f"Unsupported signal transform: {method}")
    return result


def fit_and_scale(splits: SegmentSplits, signal_transform: str, scaler_name: str):
    transformed = {name: transform_signal(getattr(splits, name), signal_transform)
                   for name in splits.__dataclass_fields__}
    if scaler_name == "MINMAX":
        scaler = MinMaxScaler()
    elif scaler_name == "STANDARD":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unsupported scaler: {scaler_name}")
    scaler.fit(transformed["normal_train"][list(FEATURES)])
    scaled = {}
    for name, frame in transformed.items():
        result = frame.copy(deep=True)
        result[list(FEATURES)] = scaler.transform(frame[list(FEATURES)])
        scaled[name] = result
    return SegmentSplits(**scaled), scaler


@dataclass(frozen=True)
class Stage2PreprocessingResult:
    config: dict
    normal_quality_report: dict
    anomaly_quality_report: dict
    normal_cleaned: pd.DataFrame
    anomaly_cleaned: pd.DataFrame
    normal_train: pd.DataFrame
    normal_validation: pd.DataFrame
    normal_test: pd.DataFrame
    anomaly_validation: pd.DataFrame
    anomaly_test: pd.DataFrame
    train_windows: WindowBatch
    valid_normal_windows: WindowBatch
    valid_anomaly_windows: WindowBatch
    test_normal_windows: WindowBatch
    test_anomaly_windows: WindowBatch
    scaler: MinMaxScaler | StandardScaler
    summary: dict
    source_provenance: dict

    @property
    def x_train(self): return self.train_windows.X
    @property
    def y_train(self): return self.train_windows.y
    @property
    def x_valid_normal(self): return self.valid_normal_windows.X
    @property
    def y_valid_normal(self): return self.valid_normal_windows.y
    @property
    def x_valid_anomaly(self): return self.valid_anomaly_windows.X
    @property
    def y_valid_anomaly(self): return self.valid_anomaly_windows.y
    @property
    def x_test_normal(self): return self.test_normal_windows.X
    @property
    def y_test_normal(self): return self.test_normal_windows.y
    @property
    def x_test_anomaly(self): return self.test_anomaly_windows.X
    @property
    def y_test_anomaly(self): return self.test_anomaly_windows.y


def run_stage2_preprocessing(normal_path: str | Path, anomaly_path: str | Path,
                             preprocessing_config: dict) -> Stage2PreprocessingResult:
    config = validate_preprocessing_config(preprocessing_config)
    if config["mode"] != "STAGE2_SEGMENT_AWARE" or not config["segment_aware"] or config["use_horizon"]:
        raise ValueError("Stage 2 backend requires segment-aware mode with horizon disabled")
    source_provenance = {"normal": fingerprint_csv_source(normal_path),
                         "anomaly": fingerprint_csv_source(anomaly_path)}
    normal: QualityResult = inspect_csv(normal_path, config, expected_label=0)
    anomaly: QualityResult = inspect_csv(anomaly_path, config, expected_label=1)
    if source_provenance != {"normal": fingerprint_csv_source(normal_path),
                             "anomaly": fingerprint_csv_source(anomaly_path)}:
        raise ValueError("Source CSV changed during preprocessing")
    splits = split_segments(normal.frame, anomaly.frame)
    scaled, scaler = fit_and_scale(splits, config["signal_transform"], config["scaler"])
    window_kwargs = {"sequence_length": config["sequence_length"], "stride": config["stride"],
                     "gap_threshold_ms": config["gap_threshold_ms"]}
    batches = {name: generate_windows(getattr(scaled, name), **window_kwargs)
               for name in scaled.__dataclass_fields__}
    cross_gap = sum(batch.cross_segment_windows for batch in batches.values())
    if cross_gap:
        raise AssertionError(f"Cross-segment windows found: {cross_gap}")
    full_counts = {f"seq{length}": {
        "normal": window_count_summary(normal.frame, length, config["stride"]),
        "anomaly": window_count_summary(anomaly.frame, length, config["stride"])}
        for length in (10, 15, 20)}
    summary = {
        "normal_raw_rows": normal.report["rows_before"],
        "normal_cleaned_rows": normal.report["rows_after"],
        "normal_duplicates_removed": normal.report["duplicates_removed"],
        "normal_segments": normal.report["segment_count"],
        "anomaly_raw_rows": anomaly.report["rows_before"],
        "anomaly_cleaned_rows": anomaly.report["rows_after"],
        "anomaly_duplicates_removed": anomaly.report["duplicates_removed"],
        "anomaly_segments": anomaly.report["segment_count"],
        **{f"{name}_segments": int(getattr(scaled, name)["segment_id"].nunique()) for name in scaled.__dataclass_fields__},
        **{f"{name}_windows": len(batch.X) for name, batch in batches.items()},
        "cross_gap_windows": cross_gap,
        "sequence_length": config["sequence_length"], "stride": config["stride"],
        "signal_transform": config["signal_transform"], "scaler": config["scaler"],
        "gap_threshold_ms": config["gap_threshold_ms"],
        "scaler_fit_source": "normal_train_rows_only",
        "full_dataset_window_counts": full_counts,
    }
    return Stage2PreprocessingResult(config, normal.report, anomaly.report,
        normal.frame, anomaly.frame, *(getattr(scaled, name) for name in scaled.__dataclass_fields__),
        batches["normal_train"], batches["normal_validation"], batches["anomaly_validation"],
        batches["normal_test"], batches["anomaly_test"], scaler, summary, source_provenance)
