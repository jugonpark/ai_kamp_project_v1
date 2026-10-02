"""Per-model run storage and discovery, including the legacy KAMP model."""
from dataclasses import dataclass
from datetime import datetime
import csv, hashlib, json, platform
from pathlib import Path
import pandas as pd
import tensorflow as tf
import train_lstm_ae as core
import plot_utils
from model_registry import MODEL_REGISTRY

RUNS_ROOT = core.OUTPUT_DIR / "models"

TEMPORAL_RESULT_FIELDS = (
    "model", "model_id", "task_type", "error_name", "score_method", "score_name",
    "threshold_method", "threshold_name", "threshold", "effective_threshold_method",
    "fallback_used", "fallback_reason", "accuracy", "balanced_accuracy", "precision",
    "recall", "f1", "specificity", "fpr", "fnr", "tn", "fp", "fn", "tp",
    "status", "experiment_name", "seed", "optimizer", "learning_rate", "loss",
    "batch_size", "cnn_filters", "cnn_kernel_size", "bottleneck_units", "noise_std",
    "best_epoch", "best_val_loss", "training_time", "created_at", "temporal_method",
    "ewma_alpha", "timestamp_aware", "timestamp_gap_threshold", "total_error",
    "detection_delay_samples", "detection_delay_seconds", "pareto", "evaluated_at",
)

@dataclass(frozen=True)
class ModelRun:
    model_id: str; run_id: str; model_path: Path; metadata: dict

def discover_model_runs(model_id):
    spec = MODEL_REGISTRY[model_id]; runs = []
    base = RUNS_ROOT / model_id
    if base.is_dir():
        for folder in sorted((p for p in base.iterdir() if p.is_dir()), reverse=True):
            model_path = folder / "model.keras"
            if not model_path.is_file(): continue
            meta_path = folder / "metadata.json"
            metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
            metadata.setdefault("noise_generation", "legacy_static" if spec.denoising else "none")
            metadata.setdefault("cnn_kernel_size", "unknown")
            metadata.setdefault("cnn_filters", "unknown")
            metadata.setdefault("bottleneck_units", "unknown")
            runs.append(ModelRun(model_id, folder.name, model_path, metadata))
    if model_id == "KAMP_LSTM_AE":
        legacy = core.MODEL_DIR / "kamp_lstm_autoencoder.keras"
        if legacy.is_file(): runs.append(ModelRun(model_id, "LEGACY_KAMP", legacy, {"model_id":model_id,"task_type":spec.task_type,"forecast_length":0,"legacy":True,"noise_generation":"none"}))
    return runs

def save_model_run(model, history, spec, training_info):
    created = datetime.now(); run_dir = RUNS_ROOT / spec.id / created.strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    model_path = run_dir / "model.keras"; model.save(model_path)
    frame = pd.DataFrame(history.history); frame.index = frame.index + 1; frame.index.name = "epoch"
    frame.to_csv(run_dir / "training_history.csv")
    plot_utils.save_training_loss(frame.get("loss", []), frame.get("val_loss", []), run_dir / "training_loss.png", frame.get("learning_rate", frame.get("lr")))
    val = list(frame.get("val_loss", [])); best = int(training_info.get("monitor_best_epoch") or 0)
    if not best and val:
        best = int(min(range(len(val)), key=val.__getitem__) + 1)
    best_loss = training_info.get("monitor_best_val_loss")
    if best_loss is None:
        best_loss = float(val[best-1]) if best else None
    metadata = {"model_id":spec.id,"model_status":spec.status,"display_name":spec.display_name,"task_type":spec.task_type,
        "created_at":created.isoformat(),"python_version":platform.python_version(),"tensorflow_version":tf.__version__,
        "sequence_length":core.SEQUENCE_LENGTH,"prediction_horizon":core.PREDICTION_HORIZON,
        "cnn_kernel_size":training_info.get("cnn_kernel_size", 3) if "CNN_LSTM" in spec.id else None,
        "cnn_filters":training_info.get("cnn_filters", 32) if "CNN_LSTM" in spec.id else None,
        "bottleneck_units":training_info.get("bottleneck_units", 32) if "CNN_LSTM" in spec.id else None,
        "forecast_length":spec.forecast_length,"features":core.FEATURES,"preprocessing":"abs + train-only MinMaxScaler",
        "compatible_score_methods":list(spec.compatible_score_methods),
        "optimizer":training_info.get("optimizer", "Adam"),"learning_rate":training_info.get("learning_rate", 0.001),"batch_size":training_info.get("batch_size", core.BATCH_SIZE),
        "weight_decay":training_info.get("weight_decay", 0.0001), "huber_delta":training_info.get("huber_delta", 1.0),
        "requested_epochs":training_info["requested_epochs"],"completed_epochs":len(frame),
        "completion_status":training_info.get("completion_status", "UNKNOWN"),
        "best_epoch":best,"best_val_loss":float(best_loss) if best_loss is not None else None,
        "parameter_count":model.count_params(),"model_file":"model.keras",
        "dataset":{"normal_rows":training_info.get("normal_rows"),"anomaly_rows":training_info.get("anomaly_rows"),
                   "train_sequences":training_info.get("train_sequences"),"valid_samples":training_info.get("valid_samples"),
                   "test_samples":training_info.get("test_samples")},
        "experiment_name":training_info.get("experiment_name") or f"{spec.id}_{run_dir.name}", "random_seed":training_info.get("random_seed", 42),
        "denoising_enabled":spec.denoising, "noise_type":training_info.get("noise_type", "Gaussian"),
        "noise_mean":training_info.get("noise_mean", 0.0), "noise_std":training_info.get("noise_std", 0.01 if spec.denoising else 0.0),
        "noise_clip":training_info.get("noise_clip", True),
        "noise_generation":"dynamic_per_batch" if spec.denoising else "none", "loss":training_info.get("loss", "MSE"),
        "callbacks":{"reduce_lr":{"enabled":training_info.get("reduce_lr_enabled", True),"factor":training_info.get("reduce_lr_factor", .7),
                     "patience":training_info.get("reduce_lr_patience", 50),"min_lr":training_info.get("min_lr", 0.0)},
                     "early_stopping":{"enabled":training_info.get("early_stopping_enabled", True),
                     "patience":training_info.get("early_stopping_patience", 120),"min_delta":training_info.get("early_stopping_min_delta", .00001),
                     "restore_best_weights":training_info.get("restore_best_weights", True)}},
        "training_duration_seconds":training_info.get("training_duration_seconds"),
        "split":"KAMP Guidebook-compatible; future STRICT_TIME_SPLIT validation required"}
    (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: metadata[key] for key in ("model_id", "model_status", "experiment_name", "created_at", "random_seed",
               "optimizer", "learning_rate", "loss", "batch_size", "cnn_filters", "cnn_kernel_size", "bottleneck_units", "requested_epochs", "completed_epochs",
               "best_epoch", "best_val_loss", "training_duration_seconds", "callbacks")}
    summary["denoising"] = {key: metadata[key] for key in ("denoising_enabled", "noise_type", "noise_mean", "noise_std", "noise_clip", "noise_generation")}
    (run_dir / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return ModelRun(spec.id, run_dir.name, model_path, metadata)

def append_evaluation_result(model_run, result):
    """Append one independently calibrated evaluation without overwriting history."""
    row = {**result, "evaluated_at": datetime.now().isoformat()}
    if "temporal_method" in row:
        path = model_run.model_path.parent / "evaluation_results_v2.csv"
        fields = list(TEMPORAL_RESULT_FIELDS)
        if row["temporal_method"] == "NONE":
            row["ewma_alpha"] = ""
    else:
        path = model_run.model_path.parent / "evaluation_results.csv"
        fields = list(row)
    fields.extend(key for key in row if key not in fields)
    if path.exists():
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            existing_fields = next(csv.reader(handle), [])
        if set(row).issubset(existing_fields):
            fields = existing_fields
        elif existing_fields != fields:
            # Preserve the historical CSV: an append under a different header shifts columns.
            signature = hashlib.sha256("\x1f".join(fields).encode("utf-8")).hexdigest()[:12]
            path = path.with_name(f"{path.stem}_{signature}{path.suffix}")
            if path.exists():
                with path.open("r", newline="", encoding="utf-8-sig") as handle:
                    if next(csv.reader(handle), []) != fields:
                        raise ValueError(f"Evaluation CSV schema mismatch: {path}")
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig" if new_file else "utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file or handle.tell() == 0:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})
    return path
