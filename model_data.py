"""Task adapters that align reconstruction and forecast samples."""
from dataclasses import dataclass
import numpy as np
import train_lstm_ae as core

@dataclass
class TaskBundle:
    inputs: np.ndarray; targets: np.ndarray; labels: np.ndarray
    task_type: str; error_name: str

def add_training_noise(inputs, mean=0.0, std=0.01, clip=True, seed=42):
    if std < 0: raise ValueError("Noise std must be non-negative")
    rng = np.random.default_rng(seed)
    noisy = np.asarray(inputs, dtype=np.float32).copy()
    noisy += rng.normal(mean, std, size=noisy.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0) if clip else noisy

def prepare_fit_data(train_inputs, train_targets, validation_inputs, validation_targets,
                     denoising=False, mean=0.0, std=0.01, clip=True, seed=42):
    """Return fit arrays without ever mutating clean train/validation arrays."""
    fit_inputs = add_training_noise(train_inputs, mean, std, clip, seed) if denoising else np.asarray(train_inputs).copy()
    return fit_inputs, np.asarray(train_targets).copy(), np.asarray(validation_inputs).copy(), np.asarray(validation_targets).copy()

def create_task_bundle(data, model_spec):
    values = data[core.FEATURES].to_numpy(dtype=np.float32)
    labels = data[core.LABEL_COLUMN].to_numpy(dtype=int)
    n = len(values) - core.SEQUENCE_LENGTH - core.PREDICTION_HORIZON
    if n <= 0: raise ValueError("Not enough rows for sequence and horizon")
    inputs = np.stack([values[i:i+core.SEQUENCE_LENGTH] for i in range(n)])
    if model_spec.task_type == "RECONSTRUCTION":
        targets = inputs.copy(); error_name = "Reconstruction Error"
    elif model_spec.task_type == "FORECAST":
        length = model_spec.forecast_length
        targets = np.stack([values[i+core.SEQUENCE_LENGTH:i+core.SEQUENCE_LENGTH+length] for i in range(n)])
        error_name = "Prediction Error"
    else: raise ValueError(f"Unknown task type: {model_spec.task_type}")
    y = np.asarray([labels[i+core.SEQUENCE_LENGTH+core.PREDICTION_HORIZON] for i in range(n)], dtype=int)
    return TaskBundle(inputs.astype(np.float32), targets.astype(np.float32), y, model_spec.task_type, error_name)
