"""Saved visual summaries for NILM training runs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _metric_value(metrics: Mapping[str, object], name: str) -> float:
    value = metrics.get(name)
    return 0.0 if value is None else float(value)


def generate_training_figures(
    history: Sequence[Mapping[str, float | int]],
    metrics: Mapping[str, Mapping[str, float | int | None]],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write the training-history and per-appliance test-metric PNG files."""
    if not history:
        raise ValueError("training history must not be empty")
    if not metrics:
        raise ValueError("test metrics must not be empty")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    history_path = destination / "training_history.png"
    metrics_path = destination / "test_metrics.png"

    epochs = np.asarray([int(record["epoch"]) for record in history])
    train_loss = np.asarray([float(record["train_loss"]) for record in history])
    validation_loss = np.asarray([float(record["validation_loss"]) for record in history])
    train_f1 = np.asarray([float(record["train_macro_f1"]) for record in history])
    validation_f1 = np.asarray([float(record["validation_macro_f1"]) for record in history])
    best_epoch = int(epochs[int(np.argmin(validation_loss))])

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    axes[0].plot(epochs, train_loss, marker="o", label="Train loss")
    axes[0].plot(epochs, validation_loss, marker="o", label="Validation loss")
    axes[0].axvline(best_epoch, color="tab:red", linestyle="--", linewidth=1.2, label=f"Best validation loss: epoch {best_epoch}")
    axes[0].set(title="Weighted BCE loss", xlabel="Epoch", ylabel="Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_f1, marker="o", label="Train macro F1")
    axes[1].plot(epochs, validation_f1, marker="o", label="Validation macro F1")
    axes[1].set(title="Macro F1 at probability threshold 0.5", xlabel="Epoch", ylabel="Macro F1")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle("NILM training history", fontsize=15, fontweight="bold")
    figure.savefig(history_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    appliance_names = list(metrics)
    f1_scores = [_metric_value(metrics[name], "f1") for name in appliance_names]
    average_precision = [_metric_value(metrics[name], "average_precision") for name in appliance_names]
    positions = np.arange(len(appliance_names))
    bar_height = 0.36
    figure, axis = plt.subplots(figsize=(10, 5.8), layout="constrained")
    axis.barh(positions - bar_height / 2, f1_scores, bar_height, label="F1")
    axis.barh(positions + bar_height / 2, average_precision, bar_height, label="Average precision")
    axis.set_yticks(positions, appliance_names)
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xlabel("Score")
    axis.set_title("Test-set quality by appliance", fontsize=15, fontweight="bold")
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    for index, (f1, ap) in enumerate(zip(f1_scores, average_precision)):
        axis.text(f1 + 0.01, index - bar_height / 2, f"{f1:.3f}", va="center", fontsize=8)
        axis.text(ap + 0.01, index + bar_height / 2, f"{ap:.3f}", va="center", fontsize=8)
    figure.savefig(metrics_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    return {"training_history": str(history_path), "test_metrics": str(metrics_path)}
