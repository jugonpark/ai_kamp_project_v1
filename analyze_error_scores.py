"""Offline error-score/threshold experiment for the saved KAMP model.

Loads the existing model once and evaluates sixteen score/threshold pairs;
it never trains or overwrites the original outputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score)
import tensorflow as tf

from train_lstm_ae import (FEATURES, KAMP_TRAIN_ROWS,
    KAMP_VALID_ANOMALY_SEQUENCES, KAMP_VALID_NORMAL_SEQUENCES,
    LABEL_COLUMN, MODEL_DIR, NORMAL_PATH, OUTLIER_PATH, PREDICTION_HORIZON,
    SEQUENCE_LENGTH, fit_scaler, load_data, transform_features, validate_data)

OUT = Path("outputs/experiments/error_score_v1")
SCORE_METHODS = ("LAST_STEP", "FULL_WINDOW", "RECENT_5", "MAX_TIMESTEP")
THRESHOLD_METHODS = ("PR_INTERSECTION", "MAX_F1", "RECALL_85", "NORMAL_P99")


def sequences_with_mapping(data: pd.DataFrame, offset: int):
    values = data[FEATURES].to_numpy(dtype=np.float32)
    labels = data[LABEL_COLUMN].to_numpy(dtype=int)
    n = len(values) - SEQUENCE_LENGTH - PREDICTION_HORIZON
    xs, ys, indices, timestamps = [], [], [], []
    for i in range(n):
        label_i = i + SEQUENCE_LENGTH + PREDICTION_HORIZON
        xs.append(values[i:i + SEQUENCE_LENGTH])
        ys.append(labels[label_i])
        indices.append(offset + label_i)
        timestamps.append(data.iloc[label_i].get("TimeStamp", ""))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=int), np.asarray(indices), np.asarray(timestamps)


def score_values(x, pred, method):
    err = np.square(x - pred)
    if method == "LAST_STEP": return err[:, -1, :].mean(axis=1)
    if method == "FULL_WINDOW": return err.mean(axis=(1, 2))
    if method == "RECENT_5": return err[:, -5:, :].mean(axis=(1, 2))
    if method == "MAX_TIMESTEP": return err.mean(axis=2).max(axis=1)
    raise ValueError(method)


def threshold_for(y, scores, method):
    if method == "NORMAL_P99":
        return float(np.percentile(scores[y == 0], 99.0))
    precision, recall, thresholds = precision_recall_curve(y, scores)
    if not len(thresholds): return float(scores.max())
    p, r = precision[:-1], recall[:-1]
    if method == "PR_INTERSECTION":
        i = int(np.argmin(np.abs(p - r)))
    elif method == "MAX_F1":
        i = int(np.argmax(2 * p * r / (p + r + 1e-12)))
    elif method == "RECALL_85":
        eligible = np.flatnonzero(r >= 0.85)
        i = int(eligible[np.argmax(p[eligible])]) if len(eligible) else int(np.argmax(r))
    else: raise ValueError(method)
    return float(thresholds[i])


def metrics(y, scores, threshold):
    pred = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": threshold, "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0), "specificity": tn / (tn + fp) if tn + fp else 0,
            "fp_rate": fp / (fp + tn) if fp + tn else 0, "fn_rate": fn / (fn + tp) if fn + tp else 0,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "pred": pred}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    normal = validate_data(load_data(NORMAL_PATH), "정상", 0)
    anomaly = validate_data(load_data(OUTLIER_PATH), "이상", 1)
    normal[FEATURES] = normal[FEATURES].abs(); anomaly[FEATURES] = anomaly[FEATURES].abs()
    train = normal.iloc[:KAMP_TRAIN_ROWS].copy(); normal_test = normal.iloc[KAMP_TRAIN_ROWS:].copy()
    scaler = fit_scaler(train)
    train_s, normal_test_s, anomaly_s = [transform_features(d, scaler) for d in (train, normal_test, anomaly)]
    xn, yn, inorm, tnorm = sequences_with_mapping(normal_test_s, KAMP_TRAIN_ROWS)
    xa, ya, ia, ta = sequences_with_mapping(anomaly_s, 0)
    xv_n, xt_n, iv_n, tv_n = xn[:KAMP_VALID_NORMAL_SEQUENCES], xn[KAMP_VALID_NORMAL_SEQUENCES:], inorm[:KAMP_VALID_NORMAL_SEQUENCES], inorm[KAMP_VALID_NORMAL_SEQUENCES:]
    xv_a, xt_a, iv_a, tv_a = xa[:KAMP_VALID_ANOMALY_SEQUENCES], xa[KAMP_VALID_ANOMALY_SEQUENCES:], ia[:KAMP_VALID_ANOMALY_SEQUENCES], ia[KAMP_VALID_ANOMALY_SEQUENCES:]
    xv, yv = np.vstack([xv_n, xv_a]), np.hstack([np.zeros(len(xv_n), int), np.ones(len(xv_a), int)])
    xt, yt = np.vstack([xt_n, xt_a]), np.hstack([np.zeros(len(xt_n), int), np.ones(len(xt_a), int)])
    model_path = MODEL_DIR / "kamp_lstm_autoencoder.keras"
    model = tf.keras.models.load_model(model_path, compile=False)
    pv = model.predict(xv, verbose=0); pt = model.predict(xt, verbose=0)
    rows, predictions = [], {}
    baseline = None
    for sm in SCORE_METHODS:
        sv, st = score_values(xv, pv, sm), score_values(xt, pt, sm)
        for tm in THRESHOLD_METHODS:
            th = threshold_for(yv, sv, tm); m = metrics(yt, st, th); predictions[(sm, tm)] = (st, m["pred"])
            row = {"score_method": sm, "threshold_method": tm, **{k: v for k, v in m.items() if k != "pred"}}
            if sm == "LAST_STEP" and tm == "PR_INTERSECTION": baseline = row.copy()
            rows.append(row)
    for row in rows:
        for k in ("precision", "recall", "f1", "fp", "fn"):
            row["delta_" + k] = row[k] - baseline[k]
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "experiment_results.csv", index=False, encoding="utf-8-sig")
    st, bp = predictions[("LAST_STEP", "PR_INTERSECTION")]
    all_idx = np.r_[iv_n, iv_a, xt_n.shape[0] * 0 + iv_a] if False else np.r_[iv_n, iv_a]
    # Test case rows retain raw and scaled label-timestep values, plus sensor errors.
    raw_normal = normal.iloc[np.r_[KAMP_TRAIN_ROWS + KAMP_VALID_NORMAL_SEQUENCES + np.arange(len(xt_n)) - 0]]
    labels_idx = np.r_[inorm[KAMP_VALID_NORMAL_SEQUENCES:], ia[KAMP_VALID_ANOMALY_SEQUENCES:]]
    raw_frames = []
    for global_i, label in zip(labels_idx, yt):
        source = normal if label == 0 else anomaly; local_i = global_i - (KAMP_TRAIN_ROWS if label == 0 else 0)
        row = {"original_data_index": int(global_i), "TimeStamp": source.iloc[int(local_i)].get("TimeStamp", ""), "label": int(label)}
        for f in FEATURES: row[f + "_raw"] = float(source.iloc[int(local_i)][f])
        raw_frames.append(row)
    cases = pd.DataFrame(raw_frames)
    e_last = np.square(xt[:, -1, :] - pt[:, -1, :])
    for j, f in enumerate(FEATURES): cases[f + "_scaled"] = xt[:, -1, j]; cases[f + "_error"] = e_last[:, j]
    cases["predicted"] = bp; cases["confusion_group"] = np.select([(yt==0)&(bp==0),(yt==0)&(bp==1),(yt==1)&(bp==0)], ["TN","FP","FN"], default="TP")
    cases[cases.confusion_group == "FP"].to_csv(OUT / "baseline_fp_cases.csv", index=False, encoding="utf-8-sig")
    cases[cases.confusion_group == "FN"].to_csv(OUT / "baseline_fn_cases.csv", index=False, encoding="utf-8-sig")
    grp = cases.groupby("confusion_group")[[f + "_error" for f in FEATURES]].agg(["count", "mean", "median"])
    grp.to_csv(OUT / "sensor_error_by_confusion_group.csv", encoding="utf-8-sig")
    for sm in SCORE_METHODS:
        sv, st = score_values(xv, pv, sm), score_values(xt, pt, sm); th = threshold_for(yv, sv, "PR_INTERSECTION")
        fig, ax = plt.subplots(figsize=(8, 4)); ax.hist(st[yt==0], bins=40, alpha=.65, label="Normal"); ax.hist(st[yt==1], bins=40, alpha=.65, label="Anomaly"); ax.axvline(th, color="red", label=f"threshold={th:.5g}"); ax.set_title(sm); ax.legend(); fig.tight_layout(); fig.savefig(OUT / f"score_distribution_{sm.lower()}.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4)); labels = [f"{a}/{b}" for a,b in zip(results.score_method, results.threshold_method)]; x=np.arange(len(labels)); axes[0].bar(x, results.f1); axes[0].set_title("F1"); axes[1].bar(x, results.fp, label="FP"); axes[1].bar(x, results.fn, bottom=results.fp, label="FN"); axes[1].set_title("FP + FN"); axes[1].legend(); [ax.set_xticks(x, labels, rotation=75, ha="right") for ax in axes]; fig.tight_layout(); fig.savefig(OUT / "threshold_comparison.png", dpi=150); plt.close(fig)
    summary = {"model_retrained": False, "baseline": baseline, "shapes": {"valid": list(xv.shape), "test": list(xt.shape)}, "reference_baseline": {"threshold": 0.0205195, "accuracy": 0.96172, "precision": 0.54065, "recall": 0.73889, "f1": 0.62441, "fp": 113, "fn": 47}}
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(results.to_string(index=False)); print("OUTPUT", OUT.resolve())


if __name__ == "__main__": main()
