"""Reproducible, validated Stage 2 dataset artifacts (independent of training)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid

import joblib
import numpy as np
import pandas as pd

from preprocessing_config import validate_preprocessing_config
from train_lstm_ae import FEATURES, OUTPUT_DIR
from .data_quality import fingerprint_csv_source
from .preprocessing import Stage2PreprocessingResult


DEFAULT_DATASET_ROOT = OUTPUT_DIR / "processed_datasets"
REQUIRED_FILES = ("dataset_config.json", "quality_report.json", "split_manifest.json",
                  "normal_cleaned.csv", "anomaly_cleaned.csv", "scaler.joblib",
                  "train.npz", "validation.npz", "test.npz", "dataset_summary.json")
NPZ_KEYS = ("X", "y", "segment_ids", "stream_type", "window_start_timestamp",
            "window_end_timestamp", "source_row_start", "source_row_end", "source_original_indices")


def validate_dataset_id(dataset_id: str) -> str:
    if not isinstance(dataset_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", dataset_id):
        raise ValueError("Dataset ID는 영문자, 숫자, _, -만 사용할 수 있습니다.")
    return dataset_id


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid JSON object: {path.name}")
    return value


def _split_ids(result: Stage2PreprocessingResult) -> dict[str, list[int]]:
    names = ("normal_train", "normal_validation", "normal_test", "anomaly_validation", "anomaly_test")
    return {name: [int(v) for v in getattr(result, name)["segment_id"].drop_duplicates()]
            for name in names}


def _cleaned_with_split(frame: pd.DataFrame, ids: dict, stream: str) -> pd.DataFrame:
    mapped = {}
    for name, segment_ids in ids.items():
        if name.startswith(stream):
            for segment_id in segment_ids:
                if segment_id in mapped:
                    raise ValueError("Split segment overlap")
                mapped[segment_id] = name.removeprefix(stream + "_")
    result = frame.copy()
    result["split"] = result["segment_id"].map(mapped)
    if result["split"].isna().any():
        raise ValueError("Cleaned rows are missing split assignments")
    return result


def _npz_payload(*parts) -> dict[str, np.ndarray]:
    fields = ("X", "y", "segment_ids", "window_start_timestamp", "window_end_timestamp",
              "source_row_start", "source_row_end", "source_original_indices")
    payload = {name: np.concatenate([getattr(batch, name) for batch, _ in parts])
               for name in fields}
    payload["stream_type"] = np.concatenate([
        np.full(len(batch.X), stream, dtype="<U7") for batch, stream in parts])
    for name in ("window_start_timestamp", "window_end_timestamp"):
        payload[name] = payload[name].astype("datetime64[ns]")
    return payload


@dataclass(frozen=True)
class ProcessedDatasetArtifact:
    path: Path
    dataset_id: str
    config: dict
    summary: dict
    quality_report: dict
    split_manifest: dict
    scaler: object
    train: dict[str, np.ndarray]
    validation: dict[str, np.ndarray]
    test: dict[str, np.ndarray]


def _load_and_validate(path: Path, expected_result: Stage2PreprocessingResult | None = None) -> ProcessedDatasetArtifact:
    path = Path(path)
    if not path.is_dir() or not (path / ".complete").is_file():
        raise ValueError("Incomplete processed dataset")
    manifest = _read_json(path / "file_manifest.json")
    if set(manifest) != set(REQUIRED_FILES):
        raise ValueError("Invalid artifact file manifest")
    for name in REQUIRED_FILES:
        if not (path / name).is_file() or _hash(path / name) != manifest[name]:
            raise ValueError(f"Missing or modified artifact file: {name}")
    config = _read_json(path / "dataset_config.json")
    dataset_id = validate_dataset_id(config.get("dataset_id"))
    if path.name != dataset_id and not path.name.startswith(f".{dataset_id}.tmp-"):
        raise ValueError("Artifact path and Dataset ID differ")
    preprocessing = validate_preprocessing_config(config["preprocessing"])
    if preprocessing["mode"] != "STAGE2_SEGMENT_AWARE" or preprocessing["use_horizon"]:
        raise ValueError("Invalid Stage 2 preprocessing mode")
    if config["features"] != list(FEATURES) or config["scaler_fit_source"] != "NORMAL_TRAIN_ONLY":
        raise ValueError("Invalid feature or scaler metadata")
    for stream in ("normal", "anomaly"):
        source = config["source"][stream]
        if not re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) or source["size_bytes"] <= 0:
            raise ValueError("Invalid source provenance")
    quality = _read_json(path / "quality_report.json")
    split = _read_json(path / "split_manifest.json")
    summary = _read_json(path / "dataset_summary.json")
    if summary["dataset_id"] != dataset_id or summary["status"] != "VALID" or summary["cross_segment_windows"] != 0:
        raise ValueError("Invalid dataset summary")
    if summary["sequence_length"] != preprocessing["sequence_length"] or summary["gap_threshold_ms"] != preprocessing["gap_threshold_ms"]:
        raise ValueError("Summary/config mismatch")
    for stream in ("normal", "anomaly"):
        cleaned = pd.read_csv(path / f"{stream}_cleaned.csv")
        if len(cleaned) != summary[f"{stream}_rows"] or len(cleaned) != quality[stream]["rows_after"]:
            raise ValueError("Cleaned row count mismatch")
        if not {"TimeStamp", *FEATURES, "Equipment_state", "segment_id", "delta_t_ms", "split"}.issubset(cleaned.columns):
            raise ValueError("Cleaned CSV columns are missing")
        actual = {key: set(int(v) for v in values) for key, values in split["segment_ids"].items() if key.startswith(stream)}
        if set().union(*actual.values()) != set(int(v) for v in cleaned["segment_id"].unique()):
            raise ValueError("Split IDs do not cover cleaned segments")
        for key, ids in actual.items():
            if set(int(v) for v in cleaned.loc[cleaned["split"] == key.removeprefix(stream + "_"), "segment_id"]) != ids:
                raise ValueError("Cleaned split assignment mismatch")
    if set(split["segment_ids"]["normal_train"]) & set(split["segment_ids"]["normal_validation"]):
        raise ValueError("Overlapping normal segments")
    scaler = joblib.load(path / "scaler.joblib")
    if type(scaler).__name__ != {"MINMAX": "MinMaxScaler", "STANDARD": "StandardScaler"}[preprocessing["scaler"]]:
        raise ValueError("Scaler type mismatch")
    batches = {}
    for name in ("train", "validation", "test"):
        with np.load(path / f"{name}.npz", allow_pickle=False) as loaded:
            if set(loaded.files) != set(NPZ_KEYS):
                raise ValueError(f"Invalid NPZ fields: {name}")
            batch = {key: loaded[key].copy() for key in NPZ_KEYS}
        count = len(batch["X"])
        if batch["X"].shape != (count, preprocessing["sequence_length"], len(FEATURES)):
            raise ValueError(f"Invalid X shape: {name}")
        if any(len(value) != count for value in batch.values()) or batch["source_original_indices"].shape != (count, preprocessing["sequence_length"]):
            raise ValueError(f"NPZ metadata length mismatch: {name}")
        if any(batch[key].dtype != np.dtype("datetime64[ns]") for key in ("window_start_timestamp", "window_end_timestamp")):
            raise ValueError("NPZ timestamps must be datetime64[ns]")
        if np.any(batch["window_end_timestamp"] < batch["window_start_timestamp"]):
            raise ValueError("Invalid window timestamps")
        if name == "train" and (np.any(batch["y"] != 0) or np.any(batch["stream_type"] != "NORMAL")):
            raise ValueError("Train contains non-normal windows")
        if np.any(~np.isin(batch["stream_type"], ("NORMAL", "ANOMALY"))) or np.any(batch["y"] != (batch["stream_type"] == "ANOMALY")):
            raise ValueError("Window labels/stream mismatch")
        if np.any(~np.isfinite(batch["X"])):
            raise ValueError("Non-finite X values")
        batches[name] = batch
    counts = {"train_windows": len(batches["train"]["X"]),
              "validation_normal_windows": int(np.sum(batches["validation"]["y"] == 0)),
              "validation_anomaly_windows": int(np.sum(batches["validation"]["y"] == 1)),
              "test_normal_windows": int(np.sum(batches["test"]["y"] == 0)),
              "test_anomaly_windows": int(np.sum(batches["test"]["y"] == 1))}
    if any(summary[key] != value for key, value in counts.items()):
        raise ValueError("Window count mismatch")
    if expected_result is not None:
        if preprocessing != expected_result.config or config["source"] != expected_result.source_provenance:
            raise ValueError("Artifact does not match preprocessing result")
        expected = {"train": _npz_payload((expected_result.train_windows, "NORMAL")),
                    "validation": _npz_payload((expected_result.valid_normal_windows, "NORMAL"), (expected_result.valid_anomaly_windows, "ANOMALY")),
                    "test": _npz_payload((expected_result.test_normal_windows, "NORMAL"), (expected_result.test_anomaly_windows, "ANOMALY"))}
        for name, fields in expected.items():
            if any(not np.array_equal(value, batches[name][key]) for key, value in fields.items()):
                raise ValueError(f"Artifact differs from preprocessing result: {name}")
    return ProcessedDatasetArtifact(path, dataset_id, config, summary, quality, split, scaler,
                                    batches["train"], batches["validation"], batches["test"])


def validate_processed_dataset(path: str | Path, expected_result: Stage2PreprocessingResult | None = None) -> bool:
    _load_and_validate(Path(path), expected_result)
    return True


def load_processed_dataset(path: str | Path) -> ProcessedDatasetArtifact:
    return _load_and_validate(Path(path))


def save_processed_dataset(result: Stage2PreprocessingResult, dataset_id: str,
                           root: str | Path = DEFAULT_DATASET_ROOT) -> ProcessedDatasetArtifact:
    dataset_id = validate_dataset_id(dataset_id)
    if not isinstance(result, Stage2PreprocessingResult) or result.summary["cross_gap_windows"] != 0:
        raise ValueError("Only PASS results with zero cross-segment windows can be saved")
    if result.config["mode"] != "STAGE2_SEGMENT_AWARE":
        raise ValueError("Only Stage 2 results can be saved")
    for stream in ("normal", "anomaly"):
        if fingerprint_csv_source(result.source_provenance[stream]["path"]) != result.source_provenance[stream]:
            raise ValueError(f"{stream} source CSV changed since preprocessing")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / dataset_id
    if target.exists():
        raise FileExistsError("이미 존재하는 Dataset ID입니다. 다른 이름을 사용하세요.")
    temporary = root / f".{dataset_id}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        ids = _split_ids(result)
        config = {"dataset_id": dataset_id, "created_at": datetime.now(timezone.utc).isoformat(),
                  "preprocessing": result.config, "preprocessing_mode": result.config["mode"],
                  **result.config, "features": list(FEATURES), "source": result.source_provenance,
                  "scaler_fit_source": "NORMAL_TRAIN_ONLY"}
        split = {"policy": "chronological_whole_segment", "segment_ids": ids,
                 "segment_counts": {name: len(values) for name, values in ids.items()},
                 "normal": {key.removeprefix("normal_") + "_segments": len(values) for key, values in ids.items() if key.startswith("normal_")},
                 "anomaly": {"train_segments": 0, **{key.removeprefix("anomaly_") + "_segments": len(values) for key, values in ids.items() if key.startswith("anomaly_")}}}
        full = result.summary["full_dataset_window_counts"][f"seq{result.config['sequence_length']}"]
        summary = {"dataset_id": dataset_id, "normal_rows": result.summary["normal_cleaned_rows"],
                   "anomaly_rows": result.summary["anomaly_cleaned_rows"],
                   "normal_segments": result.summary["normal_segments"], "anomaly_segments": result.summary["anomaly_segments"],
                   "train_windows": result.summary["normal_train_windows"],
                   "validation_normal_windows": result.summary["normal_validation_windows"],
                   "validation_anomaly_windows": result.summary["anomaly_validation_windows"],
                   "test_normal_windows": result.summary["normal_test_windows"],
                   "test_anomaly_windows": result.summary["anomaly_test_windows"],
                   "total_normal_windows": full["normal"]["windows"], "total_anomaly_windows": full["anomaly"]["windows"],
                   "cross_segment_windows": result.summary["cross_gap_windows"],
                   "sequence_length": result.config["sequence_length"], "signal_transform": result.config["signal_transform"],
                   "scaler": result.config["scaler"], "gap_threshold_ms": result.config["gap_threshold_ms"], "status": "VALID"}
        _write_json(temporary / "dataset_config.json", config)
        _write_json(temporary / "quality_report.json", {"normal": result.normal_quality_report, "anomaly": result.anomaly_quality_report})
        _write_json(temporary / "split_manifest.json", split)
        _write_json(temporary / "dataset_summary.json", summary)
        _cleaned_with_split(result.normal_cleaned, ids, "normal").to_csv(temporary / "normal_cleaned.csv", index=False)
        _cleaned_with_split(result.anomaly_cleaned, ids, "anomaly").to_csv(temporary / "anomaly_cleaned.csv", index=False)
        joblib.dump(result.scaler, temporary / "scaler.joblib")
        np.savez_compressed(temporary / "train.npz", **_npz_payload((result.train_windows, "NORMAL")))
        np.savez_compressed(temporary / "validation.npz", **_npz_payload((result.valid_normal_windows, "NORMAL"), (result.valid_anomaly_windows, "ANOMALY")))
        np.savez_compressed(temporary / "test.npz", **_npz_payload((result.test_normal_windows, "NORMAL"), (result.test_anomaly_windows, "ANOMALY")))
        _write_json(temporary / "file_manifest.json", {name: _hash(temporary / name) for name in REQUIRED_FILES})
        (temporary / ".complete").write_text("VALID\n", encoding="ascii")
        validate_processed_dataset(temporary, result)
        if target.exists():
            raise FileExistsError("이미 존재하는 Dataset ID입니다. 다른 이름을 사용하세요.")
        os.rename(temporary, target)
        return load_processed_dataset(target)
    except Exception:
        if temporary.exists() and temporary.parent.resolve() == root:
            shutil.rmtree(temporary)
        raise


def discover_processed_datasets(root: str | Path = DEFAULT_DATASET_ROOT) -> list[dict]:
    root = Path(root)
    if not root.is_dir():
        return []
    found = []
    for path in root.iterdir():
        if not path.is_dir() or not re.fullmatch(r"[A-Za-z0-9_-]+", path.name):
            continue
        try:
            artifact = load_processed_dataset(path)
        except (ValueError, KeyError, OSError, TypeError, json.JSONDecodeError, ImportError):
            continue
        found.append({"dataset_id": artifact.dataset_id, "created_at": artifact.config["created_at"],
                      "sequence": artifact.summary["sequence_length"], "signal": artifact.summary["signal_transform"],
                      "scaler": artifact.summary["scaler"], "window_counts": {
                          key: artifact.summary[key] for key in ("train_windows", "validation_normal_windows",
                          "validation_anomaly_windows", "test_normal_windows", "test_anomaly_windows")},
                      "path": str(path.resolve())})
    return sorted(found, key=lambda item: item["created_at"], reverse=True)
