"""Offline fixed-model sequence fallback for compatible saved Stage 2 runs.

Every tier keeps its own validation-calibrated threshold. Routing uses complete
test Segment lengths; streaming inference cannot know those lengths in advance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)

from evaluation_controller import EvaluationController
from model_registry import PROCESSED_DATASET_MODEL_IDS
from .segment_detection import inventory_from_artifact, summarize_segment_details


@dataclass(frozen=True)
class AutoTier:
    sequence_length: int
    run: object
    artifact: object
    controller: EvaluationController


def choose_sequence(segment_length: int, available_sequences) -> int | None:
    """Return the longest fixed model that fits, or None for a short Segment."""
    if type(segment_length) is not int or segment_length < 0:
        raise ValueError("segment_length must be a non-negative integer")
    sequences = tuple(available_sequences)
    if not sequences or any(type(seq) is not int or seq <= 0 for seq in sequences) or len(set(sequences)) != len(sequences):
        raise ValueError("available_sequences must contain distinct positive integers")
    return next((seq for seq in sorted(sequences, reverse=True) if seq <= segment_length), None)


def prepare_auto_pool(model_runs, controller_factory=EvaluationController) -> list[AutoTier]:
    """Validate all run/artifact identities before loading any model predictions."""
    runs = list(model_runs)
    if len(runs) < 2:
        raise ValueError("AUTO_SEQUENCE_FALLBACK requires at least two saved sequence models")
    tiers = []
    reference = None
    for run in runs:
        metadata = run.metadata or {}
        if (run.model_id not in PROCESSED_DATASET_MODEL_IDS or
                metadata.get("data_mode") != "PROCESSED_DATASET"):
            raise ValueError("AUTO pool accepts Stage 2 reconstruction runs only")
        controller = controller_factory()
        controller.select_run(run)
        artifact = controller._processed_artifact()
        preprocessing = artifact.config["preprocessing"]
        sequence = preprocessing["sequence_length"]
        identity = {
            "model_family": run.model_id,
            "preprocessing": {key: value for key, value in preprocessing.items() if key != "sequence_length"},
            "features": artifact.config["features"],
            "scaler_fit_source": artifact.config["scaler_fit_source"],
            "source_sha256": {stream: artifact.config["source"][stream]["sha256"] for stream in ("normal", "anomaly")},
            "split_policy": artifact.split_manifest["policy"],
            "split_segment_ids": artifact.split_manifest["segment_ids"],
        }
        if reference is not None:
            for key, value in identity.items():
                if value != reference[key]:
                    raise ValueError(f"AUTO pool compatibility failure: {key} differs")
        else:
            reference = identity
        if any(tier.sequence_length == sequence for tier in tiers):
            raise ValueError(f"AUTO pool has duplicate Seq{sequence} models")
        tiers.append(AutoTier(sequence, run, artifact, controller))
    return sorted(tiers, key=lambda tier: tier.sequence_length, reverse=True)


def _test_segment_lengths(artifact) -> dict[tuple[str, int], int]:
    lengths = {}
    for stream in ("NORMAL", "ANOMALY"):
        frame = pd.read_csv(artifact.path / f"{stream.lower()}_cleaned.csv", usecols=["segment_id", "split"])
        for segment_id, group in frame.loc[frame["split"] == "test"].groupby("segment_id"):
            lengths[(stream, int(segment_id))] = len(group)
    return lengths


def evaluate_auto_sequence(model_runs, score_method="LAST_STEP_MSE", threshold_method="PR_INTERSECTION",
                           temporal_method="NONE", ewma_alpha=0.4, timestamp_aware=True,
                           controller_factory=EvaluationController) -> dict:
    """Evaluate each tier separately, then count each Test Segment exactly once."""
    tiers = prepare_auto_pool(model_runs, controller_factory)
    lengths = _test_segment_lengths(tiers[0].artifact)
    inventory = inventory_from_artifact(tiers[0].artifact)
    anomaly_ids = {row["segment_id"] for row in inventory}
    if anomaly_ids != {segment_id for stream, segment_id in lengths if stream == "ANOMALY"}:
        raise ValueError("AUTO pool anomaly Test inventory mismatch")
    for tier in tiers[1:]:
        if _test_segment_lengths(tier.artifact) != lengths:
            raise ValueError("AUTO pool compatibility failure: Test Segment lengths differ")
        other_inventory = inventory_from_artifact(tier.artifact)
        if [(row["segment_id"], row["segment_length_samples"], str(row["segment_start_timestamp"]),
             str(row["segment_end_timestamp"])) for row in other_inventory] != [
             (row["segment_id"], row["segment_length_samples"], str(row["segment_start_timestamp"]),
              str(row["segment_end_timestamp"])) for row in inventory]:
            raise ValueError("AUTO pool compatibility failure: anomaly Test inventory differs")
    sequences = [tier.sequence_length for tier in tiers]
    assignments = {(stream, segment_id): choose_sequence(length, sequences)
                   for (stream, segment_id), length in lengths.items()}

    by_sequence = {}
    for tier in tiers:
        tier.controller.load_model_and_predictions()
        evaluation = tier.controller.evaluate(score_method, threshold_method, temporal_method,
                                              ewma_alpha, timestamp_aware)
        predictions = tier.controller.last_window_predictions
        if predictions is None or len(predictions) != len(tier.artifact.test["y"]):
            raise ValueError(f"Seq{tier.sequence_length}: missing Test window predictions")
        by_sequence[tier.sequence_length] = (tier, evaluation, predictions)

    labels, predictions, tier_info = [], [], {}
    for sequence, (tier, evaluation, tier_predictions) in by_sequence.items():
        batch = tier.artifact.test
        mask = np.zeros(len(tier_predictions), dtype=bool)
        for (stream, segment_id), chosen in assignments.items():
            if chosen == sequence:
                selected = (batch["stream_type"] == stream) & (batch["segment_ids"] == segment_id)
                if not selected.any():
                    raise ValueError(f"Seq{sequence}: missing Test windows for {stream} Segment {segment_id}")
                mask |= selected
        labels.extend(np.asarray(batch["y"])[mask].astype(int).tolist())
        predictions.extend(np.asarray(tier_predictions)[mask].astype(int).tolist())
        tier_info[str(sequence)] = {
            "run_id": tier.run.run_id, "dataset_id": tier.artifact.dataset_id,
            "score_method": score_method, "threshold_method": threshold_method,
            "effective_threshold_method": evaluation["effective_threshold_method"],
            "threshold": float(evaluation["threshold"]),
            "temporal_method": temporal_method, "ewma_alpha": ewma_alpha if temporal_method == "EWMA" else None,
            "selected_test_windows": int(mask.sum())}
    if not labels:
        raise ValueError("AUTO pool has no evaluable Test windows")
    y_true, y_pred = np.asarray(labels), np.asarray(predictions)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    selected_details = []
    for row in inventory:
        sequence = assignments[("ANOMALY", row["segment_id"])]
        detail_tier = by_sequence[sequence if sequence is not None else sequences[-1]][0]
        detail = next(item for item in detail_tier.controller._last_segment_details
                      if item["segment_id"] == row["segment_id"]).copy()
        if sequence is None and detail["evaluable"]:
            raise ValueError("Short Test Segment unexpectedly has a window")
        detail["selected_sequence"] = sequence
        selected_details.append(detail)
    segment_metrics = summarize_segment_details(selected_details)
    common = tiers[0].artifact.config["preprocessing"]
    return {
        "mode": "AUTO_SEQUENCE_FALLBACK", "data_mode": "PROCESSED_DATASET",
        "model_id": tiers[0].run.model_id, "model": f"AUTO {tiers[0].run.model_id}",
        "experiment_name": "AUTO " + "/".join(f"Seq{sequence}" for sequence in sequences),
        "sequence_length": "AUTO", "available_sequences": sequences,
        "signal_transform": common["signal_transform"], "scaler": common["scaler"],
        "gap_threshold_ms": common["gap_threshold_ms"], "stride": common["stride"],
        "use_horizon": common["use_horizon"],
        "score_method": score_method, "threshold_method": threshold_method,
        "effective_threshold_method": "PER_TIER", "threshold": None,
        "temporal_method": temporal_method, "ewma_alpha": ewma_alpha if temporal_method == "EWMA" else "",
        "timestamp_aware": bool(timestamp_aware),
        "fallback_used": any(evaluation["fallback_used"] for _, evaluation, _ in by_sequence.values()),
        "fallback_reason": "See per-tier threshold methods",
        "per_tier": tier_info,
        "auto_tier_thresholds": json.dumps({sequence: info["threshold"] for sequence, info in tier_info.items()}),
        "auto_tier_methods": json.dumps({sequence: info["effective_threshold_method"] for sequence, info in tier_info.items()}),
        "auto_pool": json.dumps({sequence: info["run_id"] for sequence, info in tier_info.items()}),
        "assignments": [{"stream": stream, "segment_id": segment_id,
                         "segment_length_samples": length, "selected_sequence": assignments[(stream, segment_id)],
                         "status": "NOT_EVALUABLE" if assignments[(stream, segment_id)] is None else "ROUTED"}
                        for (stream, segment_id), length in sorted(lengths.items())],
        "test_windows": len(y_true), "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "total_error": int(fp + fn),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "fpr": float(fp / (tn + fp)) if tn + fp else 0.0,
        "fnr": float(fn / (fn + tp)) if fn + tp else 0.0,
        "segment_details": selected_details, **segment_metrics,
    }
