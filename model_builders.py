"""Additional model builders for the KAMP experiment platform."""
import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers

SEQUENCE_LENGTH = 20
FEATURE_COUNT = 3
FORECAST_LENGTH = 5


def _compile(model):
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def build_gru_autoencoder():
    inputs = layers.Input((SEQUENCE_LENGTH, FEATURE_COUNT))
    x = layers.GRU(64, return_sequences=True)(inputs); x = layers.GRU(32)(x)
    x = layers.RepeatVector(SEQUENCE_LENGTH)(x); x = layers.GRU(32, return_sequences=True)(x)
    x = layers.GRU(64, return_sequences=True)(x)
    return _compile(Model(inputs, layers.TimeDistributed(layers.Dense(FEATURE_COUNT))(x), name="gru_autoencoder"))


def build_cnn_lstm_autoencoder(kernel_size=3):
    inputs = layers.Input((SEQUENCE_LENGTH, FEATURE_COUNT))
    x = layers.Conv1D(32, kernel_size, padding="same", activation="relu")(inputs)
    x = layers.Conv1D(32, kernel_size, padding="same", activation="relu")(x)
    x = layers.LSTM(64, return_sequences=True)(x); x = layers.LSTM(32)(x)
    x = layers.RepeatVector(SEQUENCE_LENGTH)(x); x = layers.LSTM(32, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    return _compile(Model(inputs, layers.TimeDistributed(layers.Dense(FEATURE_COUNT))(x), name="cnn_lstm_autoencoder"))


def build_denoising_cnn_lstm_autoencoder(kernel_size=3):
    """Same architecture as CNN-LSTM; denoising is a training-input policy."""
    return build_cnn_lstm_autoencoder(kernel_size=kernel_size)


def build_lstm_forecast_5():
    inputs = layers.Input((SEQUENCE_LENGTH, FEATURE_COUNT))
    x = layers.LSTM(64, return_sequences=True)(inputs); x = layers.LSTM(32)(x)
    x = layers.Dense(64, activation="relu")(x); x = layers.RepeatVector(FORECAST_LENGTH)(x)
    x = layers.LSTM(32, return_sequences=True)(x); x = layers.LSTM(64, return_sequences=True)(x)
    return _compile(Model(inputs, layers.TimeDistributed(layers.Dense(FEATURE_COUNT))(x), name="lstm_forecast_5"))
