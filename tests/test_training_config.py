import random
import unittest
import shutil
import uuid
from pathlib import Path

import numpy as np


class TrainingConfigTests(unittest.TestCase):
    def test_defaults_include_callbacks_and_denoising(self):
        from training_config import default_training_config
        config = default_training_config()
        for key in ("reduce_lr_enabled", "reduce_lr_factor", "reduce_lr_patience", "min_lr",
                    "early_stopping_enabled", "early_stopping_patience", "early_stopping_min_delta",
                    "restore_best_weights", "weight_decay", "huber_delta", "noise_type",
                    "noise_mean", "noise_std", "noise_clip"):
            self.assertIn(key, config)
        self.assertEqual(config["cnn_kernel_size"], 3)

    def test_cnn_kernel_size_accepts_only_supported_integer_values(self):
        from training_config import validate_training_config
        for value in (3, 5, 7):
            self.assertEqual(validate_training_config({"cnn_kernel_size": value})["cnn_kernel_size"], value)
        for value in (0, 2, 9, 3.5, "5", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_training_config({"cnn_kernel_size": value})

    def test_invalid_training_values_are_rejected(self):
        from training_config import validate_training_config
        invalid = (("epochs", 0), ("batch_size", 0), ("learning_rate", 0), ("noise_std", -1),
                   ("huber_delta", 0), ("early_stopping_patience", 0), ("reduce_lr_patience", 0),
                   ("reduce_lr_factor", 1), ("min_lr", -1))
        for key, value in invalid:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_training_config({key:value})

    def test_seed_is_applied_to_python_and_numpy(self):
        from training_config import apply_random_seed
        apply_random_seed(73); first = (random.random(), np.random.random())
        apply_random_seed(73); second = (random.random(), np.random.random())
        self.assertEqual(first, second)

    def test_config_save_load_round_trip_preserves_callbacks(self):
        from training_config import default_training_config, load_training_config, save_training_config
        config = default_training_config(); config.update({"experiment_name":"ROUND_TRIP", "reduce_lr_factor":.6, "early_stopping_patience":77, "cnn_kernel_size":5})
        folder = Path.cwd() / "outputs" / ".test_artifacts" / uuid.uuid4().hex
        try:
            path = save_training_config(folder / "config.json", config)
            loaded = load_training_config(path)
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        self.assertEqual(loaded["experiment_name"], "ROUND_TRIP")
        self.assertEqual(loaded["reduce_lr_factor"], .6)
        self.assertEqual(loaded["early_stopping_patience"], 77)
        self.assertEqual(loaded["cnn_kernel_size"], 5)

    def test_legacy_config_defaults_kernel_to_three(self):
        from training_config import validate_training_config
        self.assertEqual(validate_training_config({"epochs": 10})["cnn_kernel_size"], 3)

    def test_optimizer_and_huber_settings_reach_keras_objects(self):
        from training_config import default_training_config
        from training_engine import build_loss, build_optimizer
        config = default_training_config(); config.update({"optimizer":"AdamW", "weight_decay":.002, "learning_rate":.0005,
                                                            "loss":"Huber", "huber_delta":.4})
        optimizer = build_optimizer(config); loss = build_loss(config)
        self.assertEqual(type(optimizer).__name__, "AdamW")
        self.assertAlmostEqual(float(optimizer.learning_rate.numpy()), .0005)
        self.assertAlmostEqual(float(optimizer.weight_decay), .002)
        self.assertAlmostEqual(float(loss._fn_kwargs["delta"]), .4)


if __name__ == "__main__": unittest.main()
