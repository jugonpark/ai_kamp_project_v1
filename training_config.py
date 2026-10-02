"""Validated, serializable training configuration shared by GUI and worker."""
from __future__ import annotations

import os
import random
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def default_training_config() -> dict:
    return {
        "epochs": 800, "batch_size": 128, "learning_rate": 0.001,
        "optimizer": "Adam", "loss": "MSE", "huber_delta": 1.0,
        "weight_decay": 0.0001, "random_seed": 42, "experiment_name": "",
        "cnn_kernel_size": 3,
        "reduce_lr_enabled": True, "reduce_lr_factor": 0.7,
        "reduce_lr_patience": 50, "min_lr": 0.0,
        "early_stopping_enabled": True, "early_stopping_patience": 120,
        "early_stopping_min_delta": 0.00001, "restore_best_weights": True,
        "noise_type": "Gaussian", "noise_mean": 0.0, "noise_std": 0.01,
        "noise_clip": True,
    }


def validate_training_config(config: dict) -> dict:
    merged = {**default_training_config(), **config}
    kernel_size = merged["cnn_kernel_size"]
    if type(kernel_size) is not int or kernel_size not in {3, 5, 7}:
        raise ValueError("CNN kernel size must be 3, 5, or 7")
    checks = (
        (int(merged["epochs"]) > 0, "Epochs must be greater than 0"),
        (int(merged["batch_size"]) > 0, "Batch size must be greater than 0"),
        (float(merged["learning_rate"]) > 0, "Learning rate must be greater than 0"),
        (float(merged["noise_std"]) >= 0, "Noise std must be non-negative"),
        (float(merged["huber_delta"]) > 0, "Huber delta must be greater than 0"),
        (int(merged["early_stopping_patience"]) >= 1, "EarlyStopping patience must be at least 1"),
        (int(merged["reduce_lr_patience"]) >= 1, "ReduceLR patience must be at least 1"),
        (0 < float(merged["reduce_lr_factor"]) < 1, "ReduceLR factor must be between 0 and 1"),
        (float(merged["min_lr"]) >= 0, "Minimum learning rate must be non-negative"),
        (float(merged["weight_decay"]) >= 0, "Weight decay must be non-negative"),
        (merged["optimizer"] in {"Adam", "AdamW", "RMSprop"}, "Unsupported optimizer"),
        (merged["loss"] in {"MSE", "Huber"}, "Unsupported loss"),
        (merged["noise_type"] == "Gaussian", "Only Gaussian noise is supported"),
    )
    for valid, message in checks:
        if not valid:
            raise ValueError(message)
    return merged


def apply_random_seed(seed: int) -> None:
    seed = int(seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def save_training_config(path, config: dict) -> Path:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    validated = validate_training_config(config)
    target.write_text(json.dumps(validated, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load_training_config(path) -> dict:
    return validate_training_config(json.loads(Path(path).read_text(encoding="utf-8")))
