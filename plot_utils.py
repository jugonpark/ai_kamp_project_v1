"""Non-GUI static plots. This module never imports Qt or calls pyplot.show()."""
import matplotlib
matplotlib.use("Agg", force=True)

import numpy as np
from matplotlib.figure import Figure


def save_training_loss(loss, val_loss, output_path, learning_rates=None):
    fig = Figure(figsize=(8, 5), tight_layout=True)
    ax = fig.subplots()
    ax.plot(loss, label="Train Loss")
    ax.plot(val_loss, label="Validation Loss")
    if len(val_loss):
        best = int(np.argmin(val_loss))
        ax.axvline(best, color="green", linestyle="--", alpha=.7, label=f"Best epoch {best + 1}")
    if learning_rates is not None:
        rates = np.asarray(learning_rates, dtype=float)
        for index in np.flatnonzero(np.diff(rates) < 0) + 1:
            ax.axvline(index, color="orange", linestyle=":", alpha=.6)
    ax.set(xlabel="Epoch", ylabel="MSE", title="KAMP Reproduction Training Loss")
    ax.legend(); fig.savefig(output_path, dpi=150); fig.clear()


def save_pr_threshold_curve(precision, recall, thresholds, selected_threshold, output_path):
    fig = Figure(figsize=(7, 5), tight_layout=True)
    ax = fig.subplots()
    ax.plot(thresholds, precision[:-1], label="Precision")
    ax.plot(thresholds, recall[:-1], label="Recall")
    ax.axvline(selected_threshold, linestyle="--", label="Selected Threshold")
    ax.set(xlabel="Threshold", ylabel="Score", title="Precision-Recall Threshold Curve")
    ax.legend(); fig.savefig(output_path, dpi=150); fig.clear()


def save_reconstruction_error(errors, labels, threshold, output_path):
    labels = np.asarray(labels)
    errors = np.asarray(errors)
    fig = Figure(figsize=(12, 5), tight_layout=True)
    ax = fig.subplots()
    ax.plot(np.flatnonzero(labels == 0), errors[labels == 0], ".", label="Normal")
    ax.plot(np.flatnonzero(labels == 1), errors[labels == 1], ".", label="Anomaly")
    ax.axhline(threshold, linestyle="--", label="Threshold")
    ax.set(xlabel="Test Sequence", ylabel="Last-timestep MSE", title="KAMP Test Reconstruction Error")
    ax.legend(); fig.savefig(output_path, dpi=150); fig.clear()


def save_confusion_matrix(matrix, output_path):
    matrix = np.asarray(matrix)
    fig = Figure(figsize=(5, 4), tight_layout=True)
    ax = fig.subplots(); ax.imshow(matrix, cmap="Blues")
    ax.set(title="Confusion Matrix", xlabel="Predicted", ylabel="True")
    for (row, column), value in np.ndenumerate(matrix):
        ax.text(column, row, int(value), ha="center", va="center")
    ax.set_xticks([0, 1], ["Normal", "Anomaly"])
    ax.set_yticks([0, 1], ["Normal", "Anomaly"])
    fig.savefig(output_path, dpi=150); fig.clear()
