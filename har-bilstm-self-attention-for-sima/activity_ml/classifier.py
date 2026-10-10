"""Training, evaluation, and checkpointing for the MuRAL BiLSTM + self-attention HAR classifier."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from activity_ml.config import HARConfig
from activity_ml.model import BiLSTMSelfAttentionClassifier
from activity_ml.plotting import generate_training_figures
from activity_ml.preprocessing import (
    FEATURE_NAMES,
    HARWindowDataset,
    LabelEncoder,
    build_windows,
    encode_features,
    load_mural_dataset,
)

CHECKPOINT_FORMAT_VERSION = 1


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017) with per-class alpha weights.

    Down-weights easy, already-confident predictions (via ``(1 - p_t) ** gamma``)
    so training spends more of its gradient budget on the hard/rare classes,
    on top of (not instead of) the existing per-class ``weight`` term.
    """

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probabilities = F.log_softmax(logits, dim=-1)
        per_sample_log_prob = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        per_sample_prob = per_sample_log_prob.exp()
        focal_term = (1.0 - per_sample_prob) ** self.gamma
        loss = -focal_term * per_sample_log_prob
        if self.weight is not None:
            sample_weight = self.weight[targets]
            return (loss * sample_weight).sum() / sample_weight.sum()
        return loss.mean()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _house_split(df: pd.DataFrame, houses: Sequence[str]) -> np.ndarray:
    return df["house"].isin(houses).to_numpy()


def build_datasets(
    config: HARConfig,
) -> tuple[HARWindowDataset, HARWindowDataset, HARWindowDataset, LabelEncoder, list[dict[str, Any]]]:
    """Load MuRAL, encode features, fit the label vocabulary, and build house-wise window datasets."""
    df = load_mural_dataset(config.mural_dir)
    missing = set(config.all_houses) - set(df["house"].unique())
    if missing:
        raise ValueError(f"configured houses not found in {config.mural_dir}: {sorted(missing)}")

    features = encode_features(df)
    label_encoder = LabelEncoder.fit(df["activity"].tolist())
    label_ids = label_encoder.transform(df["activity"].tolist())
    house_ids = df["house"].to_numpy()

    datasets = []
    test_metadata: list[dict[str, Any]] = []
    for split_name, split_houses in (
        ("train", config.train_houses),
        ("validation", config.validation_houses),
        ("test", config.test_houses),
    ):
        mask = _house_split(df, split_houses)
        split_indices = np.flatnonzero(mask)
        split_features = features[split_indices]
        split_labels = label_ids[split_indices]
        split_houses_arr = house_ids[split_indices]
        starts, _ = build_windows(split_features, split_labels, split_houses_arr, config.window_size, config.stride)
        dataset = HARWindowDataset(split_features, split_labels, starts, config.window_size)
        datasets.append(dataset)
        if split_name == "test":
            event_index = df["event_index"].to_numpy()[split_indices]
            for start in starts:
                end = start + config.window_size
                test_metadata.append(
                    {
                        "sample_id": f"{split_houses_arr[start]}_{event_index[start]}_{event_index[end - 1]}",
                        "house": str(split_houses_arr[start]),
                    }
                )
    return (*datasets, label_encoder, test_metadata)


def build_model(config: HARConfig, input_size: int, num_classes: int) -> BiLSTMSelfAttentionClassifier:
    """Construct the BiLSTM + self-attention classifier from a config."""
    return BiLSTMSelfAttentionClassifier(
        input_size=input_size,
        num_classes=num_classes,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
        num_attention_heads=config.num_attention_heads,
    )


def compute_class_weights(dataset: HARWindowDataset, num_classes: int) -> torch.Tensor:
    counts = dataset.class_counts(num_classes).astype(np.float64)
    if (counts <= 0).any():
        missing = np.flatnonzero(counts <= 0).tolist()
        raise ValueError(f"training split has zero windows for class indices: {missing}")
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss, total_count = 0.0, 0
    all_predictions: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    for sequences, labels in loader:
        sequences, labels = sequences.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(sequences)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        batch_size = labels.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size
        probabilities = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
        all_probabilities.append(probabilities)
        all_predictions.append(probabilities.argmax(axis=1))
        all_labels.append(labels.cpu().numpy())
    return (
        total_loss / total_count,
        np.concatenate(all_predictions),
        np.concatenate(all_labels),
        np.concatenate(all_probabilities),
    )


def multiclass_metrics(labels: np.ndarray, predictions: np.ndarray, class_names: Sequence[str]) -> dict[str, Any]:
    report = classification_report(
        labels, predictions, labels=list(range(len(class_names))), target_names=list(class_names),
        output_dict=True, zero_division=0,
    )
    confusion = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    return {
        "accuracy": float(report["accuracy"]),
        "macro_avg": {k: float(v) for k, v in report["macro avg"].items()},
        "weighted_avg": {k: float(v) for k, v in report["weighted avg"].items()},
        "per_class": {
            name: {k: float(v) for k, v in report[name].items()} for name in class_names
        },
        "confusion_matrix": confusion.tolist(),
    }


def save_checkpoint(path: str | Path, model: BiLSTMSelfAttentionClassifier, metadata: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    reserved = {"format_version", "model_type", "model_config", "state_dict"}
    if reserved & metadata.keys():
        raise ValueError("metadata uses reserved checkpoint keys")
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_type": "bilstm_self_attention",
            "model_config": model.architecture_config(),
            "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            **dict(metadata),
        },
        destination,
    )


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported HAR checkpoint format")
    required = {"model_config", "state_dict", "classes", "feature_names", "window_size"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"checkpoint is missing keys: {sorted(missing)}")
    return payload


def train_from_config(
    config: HARConfig,
    device: str | torch.device | None = None,
    progress: Callable[[dict[str, float | int]], None] | None = None,
) -> dict[str, Any]:
    _set_seed(config.seed)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_data, validation_data, test_data, label_encoder, test_metadata = build_datasets(config)
    num_classes = label_encoder.num_classes
    input_size = len(FEATURE_NAMES)

    train_counts = train_data.class_counts(num_classes)
    class_weights = compute_class_weights(train_data, num_classes) if config.use_class_weights else None

    options = {"batch_size": config.batch_size, "num_workers": config.num_workers}
    train_loader = DataLoader(train_data, shuffle=True, generator=torch.Generator().manual_seed(config.seed), **options)
    validation_loader = DataLoader(validation_data, shuffle=False, **options)
    test_loader = DataLoader(test_data, shuffle=False, **options)

    model = build_model(config, input_size, num_classes).to(selected_device)
    weight_tensor = class_weights.to(selected_device) if class_weights is not None else None
    if config.loss_type == "focal":
        criterion: nn.Module = FocalLoss(weight=weight_tensor, gamma=config.focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_loss, best_state, stale = float("inf"), None, 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.epochs + 1):
        train_loss, train_pred, train_labels, _ = _run_epoch(model, train_loader, criterion, selected_device, optimizer)
        with torch.no_grad():
            validation_loss, val_pred, val_labels, _ = _run_epoch(model, validation_loader, criterion, selected_device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_accuracy": float((train_pred == train_labels).mean()),
            "validation_accuracy": float((val_pred == val_labels).mean()),
        }
        history.append(record)
        if progress is not None:
            progress(record)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("training produced no model state")
    model.load_state_dict(best_state)
    model.to(selected_device)

    with torch.no_grad():
        test_loss, test_pred, test_labels, test_probabilities = _run_epoch(model, test_loader, criterion, selected_device)
    metrics = multiclass_metrics(test_labels, test_pred, label_encoder.classes)

    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.artifact_dir / "best_model.pt"
    metrics_path = config.artifact_dir / "metrics.json"
    predictions_path = config.artifact_dir / "test_predictions.csv"

    confusion = np.asarray(metrics["confusion_matrix"])
    figure_paths = generate_training_figures(history, confusion, label_encoder.classes, config.artifact_dir / "figures")

    predictions_frame = pd.DataFrame(
        {
            "sample_id": [row["sample_id"] for row in test_metadata],
            "house": [row["house"] for row in test_metadata],
            "true_activity": [label_encoder.classes[i] for i in test_labels],
            "predicted_activity": [label_encoder.classes[i] for i in test_pred],
            **{
                f"{name}_prob": test_probabilities[:, i]
                for i, name in enumerate(label_encoder.classes)
            },
        }
    )
    predictions_frame.to_csv(predictions_path, index=False)

    metadata = {
        "classes": list(label_encoder.classes),
        "feature_names": list(FEATURE_NAMES),
        "window_size": config.window_size,
        "house_splits": {
            "train": list(config.train_houses),
            "validation": list(config.validation_houses),
            "test": list(config.test_houses),
        },
        "training_class_counts": {name: int(count) for name, count in zip(label_encoder.classes, train_counts)},
        "history": history,
        "metrics": metrics,
        "figure_paths": figure_paths,
        "training_config": config.to_dict(),
    }
    save_checkpoint(checkpoint_path, model, metadata)

    report = {
        "device": str(selected_device),
        "epochs_completed": len(history),
        "best_validation_loss": best_loss,
        "test_loss": test_loss,
        "test_metrics": metrics,
        "checkpoint_path": str(checkpoint_path),
        "predictions_path": str(predictions_path),
        "figure_paths": figure_paths,
        "num_classes": num_classes,
        "training_class_counts": metadata["training_class_counts"],
    }
    metrics_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return {**report, "metrics_path": str(metrics_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the MuRAL BiLSTM + self-attention HAR classifier")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    result = train_from_config(
        HARConfig.from_json(args.config),
        args.device,
        progress=lambda record: print(json.dumps(record), flush=True),
    )
    print(json.dumps({k: v for k, v in result.items() if k != "test_metrics"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
