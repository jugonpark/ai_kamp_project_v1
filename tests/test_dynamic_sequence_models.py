import unittest

import tensorflow as tf

from model_registry import MODEL_REGISTRY
from training_engine import build_training_model
from training_config import default_training_config
from preprocessing_config import default_preprocessing_config


class DynamicSequenceModelsTests(unittest.TestCase):
    def test_reconstruction_builders_accept_sequence_lengths(self):
        config = default_training_config()
        for model_id in ("KAMP_LSTM_AE", "GRU_AUTOENCODER", "CNN_LSTM_AUTOENCODER",
                         "DENOISING_CNN_LSTM_AUTOENCODER"):
            for length in (5, 10, 15, 20):
                with self.subTest(model_id=model_id, length=length):
                    runtime = {**default_preprocessing_config(), "sequence_length": length}
                    model = build_training_model(MODEL_REGISTRY[model_id], config, runtime)
                    self.assertEqual(model.input_shape, (None, length, 3))
                    self.assertEqual(model.output_shape, (None, length, 3))
                    repeat = next(layer for layer in model.layers if isinstance(layer, tf.keras.layers.RepeatVector))
                    self.assertEqual(repeat.n, length)
                    tf.keras.backend.clear_session()

    def test_default_builder_and_forecast_are_preserved(self):
        for model_id in ("KAMP_LSTM_AE", "CNN_LSTM_AUTOENCODER", "LSTM_FORECAST_5"):
            model = build_training_model(MODEL_REGISTRY[model_id], default_training_config())
            self.assertEqual(model.input_shape, (None, 20, 3))
            self.assertEqual(model.output_shape, (None, 5 if model_id == "LSTM_FORECAST_5" else 20, 3))
            tf.keras.backend.clear_session()
