import unittest
from types import SimpleNamespace

import tensorflow as tf
import numpy as np

from training_engine import GUITrainingCallback
from model_data import prepare_fit_data


class DynamicTrainingMonitorTests(unittest.TestCase):
    def test_two_epoch_dummy_fit_consumes_dynamic_batches(self):
        clean = np.full((4, 20, 3), .5, dtype=np.float32)
        train, targets, valid_x, valid_y = prepare_fit_data(clean, clean, clean[:2], clean[:2],
            denoising=True, mean=0.0, std=.005, clip=True, seed=42, batch_size=2)
        self.assertIsNone(targets)
        model = tf.keras.Sequential([tf.keras.layers.Input((20, 3)), tf.keras.layers.Dense(3)])
        model.compile(optimizer="adam", loss="mse")
        history = model.fit(train, validation_data=(valid_x, valid_y), epochs=2, shuffle=False, verbose=0)
        self.assertEqual(len(history.history["loss"]), 2)

    def test_min_delta_and_reduce_lr_counter_follow_config(self):
        callback = GUITrainingCallback(4, min_delta=.02)
        optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
        callback.set_model(SimpleNamespace(optimizer=optimizer, stop_training=False))
        emitted = []
        callback.epoch_finished.connect(lambda *values: emitted.append(values))
        callback.on_train_begin()
        callback.on_epoch_end(0, {"loss":.4, "val_loss":.5})
        callback.on_epoch_end(1, {"loss":.35, "val_loss":.49})
        self.assertEqual(callback.best_epoch, 1)
        self.assertEqual(callback.since_improvement, 1)
        optimizer.learning_rate.assign(.0007)
        callback.on_epoch_end(2, {"loss":.3, "val_loss":.47})
        self.assertEqual(callback.best_epoch, 3)
        self.assertEqual(callback.best_loss, .47)
        self.assertEqual(callback.reduce_lr_count, 1)
        self.assertEqual(emitted[-1][-1], 1)


if __name__ == "__main__":
    unittest.main()
