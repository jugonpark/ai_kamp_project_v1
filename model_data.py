"""Task adapters that align reconstruction and forecast samples."""
from dataclasses import dataclass
import numpy as np
import tensorflow as tf
import train_lstm_ae as core

@dataclass
class TaskBundle:
    inputs: np.ndarray; targets: np.ndarray; labels: np.ndarray
    task_type: str; error_name: str
    sample_timestamps: np.ndarray | None = None
    segment_ids: np.ndarray | None = None
    contains_timestamp_gap: np.ndarray | None = None
    timestamp_gap_threshold: float | None = None

def add_training_noise(inputs, mean=0.0, std=0.01, clip=True, seed=42):
    """Legacy one-shot helper; the training worker uses prepare_fit_data instead."""
    if std < 0: raise ValueError("Noise std must be non-negative")
    rng = np.random.default_rng(seed)
    noisy = np.asarray(inputs, dtype=np.float32).copy()
    noisy += rng.normal(mean, std, size=noisy.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0) if clip else noisy

def prepare_fit_data(train_inputs, train_targets, validation_inputs, validation_targets,
                     denoising=False, mean=0.0, std=0.01, clip=True, seed=42, batch_size=128):
    """Keep baseline arrays unchanged; denoising returns a sequential dynamic dataset."""
    clean_inputs = np.asarray(train_inputs, dtype=np.float32).copy()
    clean_targets = np.asarray(train_targets, dtype=np.float32).copy()
    valid_inputs = np.asarray(validation_inputs).copy()
    valid_targets = np.asarray(validation_targets).copy()
    if not denoising:
        return clean_inputs, clean_targets, valid_inputs, valid_targets
    if std < 0:
        raise ValueError("Noise std must be non-negative")
    generator = tf.random.Generator.from_seed(int(seed))
    def noisy_batch(inputs, targets):
        if std == 0:
            return inputs, targets
        noise = generator.normal(tf.shape(inputs), mean=float(mean), stddev=float(std), dtype=inputs.dtype)
        noisy = inputs + noise
        if clip:
            noisy = tf.clip_by_value(noisy, 0.0, 1.0)
        return noisy, targets
    dataset = tf.data.Dataset.from_tensor_slices((clean_inputs, clean_targets)).batch(int(batch_size))
    dataset = dataset.map(noisy_batch, num_parallel_calls=1, deterministic=True)
    return dataset, None, valid_inputs, valid_targets

def create_task_bundle(data, model_spec, sequence_length=None, prediction_horizon=None,
                       gap_threshold_seconds=None):
    sequence_length = core.SEQUENCE_LENGTH if sequence_length is None else sequence_length
    prediction_horizon = core.PREDICTION_HORIZON if prediction_horizon is None else prediction_horizon
    if type(sequence_length) is not int or sequence_length <= 0:
        raise ValueError("sequence_length must be a positive integer")
    if type(prediction_horizon) is not int or prediction_horizon < 0:
        raise ValueError("prediction_horizon must be a non-negative integer")
    values = data[core.FEATURES].to_numpy(dtype=np.float32)
    labels = data[core.LABEL_COLUMN].to_numpy(dtype=int)
    n = len(values) - sequence_length - prediction_horizon
    if n <= 0: raise ValueError("Not enough rows for sequence and horizon")
    inputs = np.stack([values[i:i+sequence_length] for i in range(n)])
    if model_spec.task_type == "RECONSTRUCTION":
        targets = inputs.copy(); error_name = "Reconstruction Error"
    elif model_spec.task_type == "FORECAST":
        length = model_spec.forecast_length
        targets = np.stack([values[i+sequence_length:i+sequence_length+length] for i in range(n)])
        error_name = "Prediction Error"
    else: raise ValueError(f"Unknown task type: {model_spec.task_type}")
    y = np.asarray([labels[i+sequence_length+prediction_horizon] for i in range(n)], dtype=int)
    metadata = {}
    if core.TIMESTAMP_COLUMN in data.columns:
        import pandas as pd
        from score_postprocessing import detect_timestamp_segments

        timestamps = pd.to_datetime(data[core.TIMESTAMP_COLUMN], errors="raise")
        row_segments, gap_threshold = detect_timestamp_segments(timestamps, gap_threshold_seconds=gap_threshold_seconds) if gap_threshold_seconds is not None else detect_timestamp_segments(timestamps)
        row_segments = np.asarray(row_segments, dtype=np.int64)
        label_rows = np.arange(n) + sequence_length + prediction_horizon
        # A gap belongs to the later row. Keep every existing window, including
        # those that cross a recording boundary, and mark it for evaluation.
        row_gaps = np.r_[False, row_segments[1:] != row_segments[:-1]]
        gap_prefix = np.r_[0, np.cumsum(row_gaps, dtype=np.int64)]
        window_end = np.arange(n) + sequence_length
        window_gaps = gap_prefix[window_end] > gap_prefix[np.arange(n) + 1]
        metadata = {
            "sample_timestamps": timestamps.to_numpy()[label_rows],
            "segment_ids": row_segments[label_rows],
            "contains_timestamp_gap": window_gaps,
            "timestamp_gap_threshold": gap_threshold,
        }
    return TaskBundle(inputs.astype(np.float32), targets.astype(np.float32), y,
                      model_spec.task_type, error_name, **metadata)
