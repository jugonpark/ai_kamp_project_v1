"""Segment-level detection metrics for saved Stage 2 test artifacts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import csv
from datetime import datetime
from pathlib import Path


METRIC_KEYS = (
    "anomaly_segments_total", "evaluable_anomaly_segments", "non_evaluable_anomaly_segments",
    "detected_segments", "missed_segments", "segment_detection_rate",
    "mean_segment_delay_seconds", "median_segment_delay_seconds", "max_segment_delay_seconds",
    "mean_post_evaluable_delay_seconds", "median_post_evaluable_delay_seconds",
    "max_post_evaluable_delay_seconds",
)


def build_anomaly_segment_inventory(cleaned: pd.DataFrame, split_manifest: dict) -> list[dict]:
    """Include every test anomaly segment, including those with zero windows."""
    ids = split_manifest["segment_ids"]["anomaly_test"]
    rows = []
    for segment_id in ids:
        frame = cleaned.loc[cleaned["segment_id"] == segment_id]
        if frame.empty or not (frame["split"] == "test").all():
            raise ValueError(f"Missing test anomaly segment: {segment_id}")
        timestamps = pd.to_datetime(frame["TimeStamp"], errors="raise").to_numpy(dtype="datetime64[ns]")
        if np.isnat(timestamps).any():
            raise ValueError(f"Invalid test anomaly timestamp: {segment_id}")
        rows.append({"segment_id": int(segment_id), "segment_length_samples": len(frame),
                     "segment_start_timestamp": timestamps[0], "segment_end_timestamp": timestamps[-1]})
    return rows


def inventory_from_artifact(artifact) -> list[dict]:
    return build_anomaly_segment_inventory(
        pd.read_csv(artifact.path / "anomaly_cleaned.csv"), artifact.split_manifest)


def stage2_segment_detection_metrics(inventory: list[dict], test_batch: dict,
                                     prediction, sequence_length: int) -> tuple[dict, list[dict]]:
    """Aggregate final window predictions without altering EWMA or legacy delay."""
    predictions = np.asarray(prediction)
    ids = np.asarray(test_batch["segment_ids"])
    streams = np.asarray(test_batch["stream_type"])
    ends = np.asarray(test_batch["window_end_timestamp"], dtype="datetime64[ns]")
    if any(len(values) != len(predictions) for values in (ids, streams, ends)):
        raise ValueError("Test window metadata/prediction length mismatch")
    known = {row["segment_id"] for row in inventory}
    anomaly_mask = streams == "ANOMALY"
    if set(ids[anomaly_mask].tolist()) - known:
        raise ValueError("Anomaly test windows reference an unknown segment")

    details = []
    for segment in inventory:
        row = segment.copy()
        selected = np.flatnonzero(anomaly_mask & (ids == row["segment_id"]))
        count = len(selected)
        if (row["segment_length_samples"] < sequence_length and count) or (
                row["segment_length_samples"] >= sequence_length and not count):
            raise ValueError(f"Segment/window count mismatch: {row['segment_id']}")
        row.update(evaluable=bool(count), window_count=count, detected=False, missed=False,
                   status="NOT_EVALUABLE", first_evaluable_timestamp=None,
                   first_alarm_timestamp=None, delay_from_segment_start_seconds=None,
                   delay_after_first_evaluable_seconds=None)
        if count:
            first = ends[selected].min()
            if np.isnat(first) or first < row["segment_start_timestamp"] or first > row["segment_end_timestamp"]:
                raise ValueError(f"Invalid first evaluable timestamp: {row['segment_id']}")
            row["first_evaluable_timestamp"] = first
            alarm_positions = selected[predictions[selected] == 1]
            if len(alarm_positions):
                alarm = ends[alarm_positions].min()
                row.update(detected=True, status="DETECTED", first_alarm_timestamp=alarm,
                           delay_from_segment_start_seconds=float((alarm - row["segment_start_timestamp"]) / np.timedelta64(1, "s")),
                           delay_after_first_evaluable_seconds=float((alarm - first) / np.timedelta64(1, "s")))
            else:
                row.update(missed=True, status="MISSED")
        details.append(row)

    detected = [row for row in details if row["detected"]]
    evaluable = sum(row["evaluable"] for row in details)
    result = {"anomaly_segments_total": len(details), "evaluable_anomaly_segments": evaluable,
              "non_evaluable_anomaly_segments": len(details) - evaluable,
              "detected_segments": len(detected), "missed_segments": evaluable - len(detected),
              "segment_detection_rate": len(detected) / evaluable if evaluable else None}
    for source, prefix in (("delay_from_segment_start_seconds", "segment_delay_seconds"),
                           ("delay_after_first_evaluable_seconds", "post_evaluable_delay_seconds")):
        values = [row[source] for row in detected]
        for method in ("mean", "median", "max"):
            result[f"{method}_{prefix}"] = float(getattr(np, method)(values)) if values else None
    return result, details


def save_segment_detail_csv(model_run, details: list[dict]) -> Path:
    """Write one unique per-evaluation detail file beside the saved model."""
    fields = ("segment_id", "segment_length_samples", "window_count", "status",
              "segment_start_timestamp", "segment_end_timestamp", "evaluable",
              "first_evaluable_timestamp", "detected", "missed", "first_alarm_timestamp",
              "delay_from_segment_start_seconds", "delay_after_first_evaluable_seconds")
    directory = Path(model_run.model_path).parent
    while True:
        path = directory / f"segment_detection_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
        try:
            with path.open("x", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for detail in details:
                    writer.writerow({key: detail.get(key) for key in fields})
            return path
        except FileExistsError:
            continue
