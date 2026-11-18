"""Saved visual summaries for HAR training runs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_training_figures(
    history: Sequence[Mapping[str, float | int]],
    confusion: np.ndarray,
    class_names: Sequence[str],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write the training-history and test-confusion-matrix PNG files."""
    if not history:
        raise ValueError("training history must not be empty")
    if confusion.shape != (len(class_names), len(class_names)):
        raise ValueError("confusion matrix shape must match the number of classes")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    history_path = destination / "training_history.png"
    confusion_path = destination / "confusion_matrix.png"

    epochs = np.asarray([int(record["epoch"]) for record in history])
    train_loss = np.asarray([float(record["train_loss"]) for record in history])
    validation_loss = np.asarray([float(record["validation_loss"]) for record in history])
    train_acc = np.asarray([float(record["train_accuracy"]) for record in history])
    validation_acc = np.asarray([float(record["validation_accuracy"]) for record in history])
    best_epoch = int(epochs[int(np.argmin(validation_loss))])

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    axes[0].plot(epochs, train_loss, marker="o", label="Train loss")
    axes[0].plot(epochs, validation_loss, marker="o", label="Validation loss")
    axes[0].axvline(best_epoch, color="tab:red", linestyle="--", linewidth=1.2, label=f"Best validation loss: epoch {best_epoch}")
    axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", label="Train accuracy")
    axes[1].plot(epochs, validation_acc, marker="o", label="Validation accuracy")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle("HAR LSTM training history", fontsize=15, fontweight="bold")
    figure.savefig(history_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    normalized = confusion / np.maximum(1, confusion.sum(axis=1, keepdims=True))
    figure, axis = plt.subplots(figsize=(9, 8), layout="constrained")
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(class_names)), class_names, rotation=90, fontsize=7)
    axis.set_yticks(range(len(class_names)), class_names, fontsize=7)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Test confusion matrix (row-normalized)", fontsize=13, fontweight="bold")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.savefig(confusion_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    return {"training_history": str(history_path), "confusion_matrix": str(confusion_path)}
