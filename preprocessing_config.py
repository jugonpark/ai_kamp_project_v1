"""Configuration for the planned Stage 2 pipeline; no processing occurs here."""
from __future__ import annotations

import json
from pathlib import Path


DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent / "configs" / "preprocessing"
MODES = ("STAGE2_SEGMENT_AWARE", "KAMP_BASELINE")
TRANSFORMS = ("ABS_ALL", "RAW_SIGNED", "ABS_VIBRATION_RAW_CURRENT")
SCALERS = ("MINMAX", "STANDARD")
STAGE2_SEQUENCE_LENGTHS = (5, 10, 15, 20)


def default_preprocessing_config(mode="STAGE2_SEGMENT_AWARE") -> dict:
    if mode not in MODES:
        raise ValueError(f"Unsupported preprocessing mode: {mode}")
    return {
        "mode": mode,
        "expected_interval_ms": 100,
        "gap_threshold_ms": 150,
        "remove_exact_duplicates": mode != "KAMP_BASELINE",
        "segment_aware": mode != "KAMP_BASELINE",
        "signal_transform": "ABS_ALL",
        "scaler": "MINMAX",
        "sequence_length": 20,
        "stride": 1,
        "use_horizon": mode == "KAMP_BASELINE",
        "prediction_horizon": 100,
    }


def validate_preprocessing_config(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError("Preprocessing configuration must be an object")
    unknown = set(config) - set(default_preprocessing_config())
    if unknown:
        raise ValueError(f"Unknown preprocessing settings: {sorted(unknown)}")
    mode = config.get("mode", "STAGE2_SEGMENT_AWARE")
    result = {**default_preprocessing_config(mode), **config}
    for key in ("expected_interval_ms", "gap_threshold_ms", "sequence_length", "stride", "prediction_horizon"):
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if result["gap_threshold_ms"] <= result["expected_interval_ms"]:
        raise ValueError("gap_threshold_ms must exceed expected_interval_ms")
    if result["sequence_length"] not in STAGE2_SEQUENCE_LENGTHS:
        raise ValueError("sequence_length must be 5, 10, 15, or 20")
    for key in ("remove_exact_duplicates", "segment_aware", "use_horizon"):
        if type(result[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if result["signal_transform"] not in TRANSFORMS or result["scaler"] not in SCALERS:
        raise ValueError("Unsupported signal transform or scaler")
    if mode == "KAMP_BASELINE" and result != default_preprocessing_config(mode):
        raise ValueError("KAMP baseline configuration is fixed to preserve existing behavior")
    return result


def save_preprocessing_config(path, config: dict) -> Path:
    target = Path(path)
    validated = validate_preprocessing_config(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(validated, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load_preprocessing_config(path) -> dict:
    return validate_preprocessing_config(json.loads(Path(path).read_text(encoding="utf-8")))


def preprocessing_from_metadata(metadata: dict | None) -> dict:
    """Older model files have no structured preprocessing metadata."""
    value = (metadata or {}).get("preprocessing")
    return validate_preprocessing_config(value) if isinstance(value, dict) else default_preprocessing_config("KAMP_BASELINE")
