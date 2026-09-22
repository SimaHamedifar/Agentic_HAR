"""Training, evaluation, and checkpointing for multi-appliance NILM."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from nilm.cache import load_cached_houses
from nilm.config import APPLIANCE_COLUMNS, DEFAULT_THRESHOLDS, FEATURE_COLUMNS, NILMConfig
from nilm.model import MultiApplianceNILM
from nilm.plotting import generate_training_figures
from nilm.preprocessing import AggregateNormalization, MultiApplianceWindowDataset, fit_aggregate_normalization


CHECKPOINT_FORMAT_VERSION = 2


def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    if logits.shape != targets.shape or logits.shape != mask.shape:
        raise ValueError("logits, targets, and mask must have identical shapes")
    if pos_weight.ndim != 1 or pos_weight.numel() != logits.shape[1]:
        raise ValueError("pos_weight must have one value per appliance")
    denominator = mask.sum()
    if denominator.item() <= 0:
        raise ValueError("masked loss requires at least one observed target")
    element_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    return (element_loss * mask).sum() / denominator


def per_appliance_class_weights(targets: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, list[dict[str, int]]]:
    truth = np.asarray(targets, dtype=np.float32)
    observed_mask = np.asarray(mask, dtype=np.float32)
    if truth.shape != observed_mask.shape or truth.ndim != 2:
        raise ValueError("targets and mask must be equal two-dimensional arrays")
    positives = (truth * observed_mask).sum(axis=0).astype(np.int64)
    observed = observed_mask.sum(axis=0).astype(np.int64)
    negatives = observed - positives
    if (positives <= 0).any() or (negatives <= 0).any():
        bad = np.flatnonzero((positives <= 0) | (negatives <= 0)).tolist()
        raise ValueError(f"each appliance needs observed positive and negative training labels; invalid indices: {bad}")
    counts = [{"positive": int(p), "negative": int(n), "observed": int(o)} for p, n, o in zip(positives, negatives, observed)]
    return negatives.astype(np.float32) / positives, counts


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float | int | None]:
    if labels.size == 0:
        return {"observed": 0, "true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0, "precision": None, "recall": None, "specificity": None, "f1": None, "accuracy": None, "balanced_accuracy": None, "average_precision": None}
    predicted = probabilities >= threshold
    truth = labels.astype(bool)
    tp = int((truth & predicted).sum())
    tn = int((~truth & ~predicted).sum())
    fp = int((~truth & predicted).sum())
    fn = int((truth & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ap = float(average_precision_score(labels, probabilities)) if np.unique(labels).size == 2 else None
    return {"observed": int(labels.size), "true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn, "precision": float(precision), "recall": float(recall), "specificity": float(specificity), "f1": float(f1), "accuracy": float((tp + tn) / labels.size), "balanced_accuracy": float((recall + specificity) / 2), "average_precision": ap}


def multi_appliance_metrics(targets: np.ndarray, probabilities: np.ndarray, mask: np.ndarray, appliance_names: Sequence[str]) -> dict[str, dict[str, float | int | None]]:
    truth = np.asarray(targets)
    scores = np.asarray(probabilities)
    observed = np.asarray(mask).astype(bool)
    if truth.shape != scores.shape or truth.shape != observed.shape or truth.ndim != 2:
        raise ValueError("targets, probabilities, and mask must have identical two-dimensional shapes")
    if truth.shape[1] != len(appliance_names):
        raise ValueError("appliance_names length must match target columns")
    return {name: _binary_metrics(truth[observed[:, index], index], scores[observed[:, index], index]) for index, name in enumerate(appliance_names)}


class StreamingMultiApplianceMetrics:
    """Bounded-memory confusion metrics and binned average precision."""

    def __init__(self, appliance_names: Sequence[str], average_precision_bins: int = 1000) -> None:
        if not appliance_names or average_precision_bins < 2:
            raise ValueError("appliance names and at least two AP bins are required")
        self.appliance_names = tuple(appliance_names)
        self.bins = average_precision_bins
        size = len(self.appliance_names)
        self.tp = np.zeros(size, dtype=np.int64)
        self.tn = np.zeros(size, dtype=np.int64)
        self.fp = np.zeros(size, dtype=np.int64)
        self.fn = np.zeros(size, dtype=np.int64)
        self.positive_hist = np.zeros((size, self.bins), dtype=np.int64)
        self.negative_hist = np.zeros((size, self.bins), dtype=np.int64)

    @property
    def retained_sample_count(self) -> int:
        return 0

    def update(self, targets: torch.Tensor | np.ndarray, probabilities: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray) -> None:
        truth = np.asarray(targets.detach().cpu() if isinstance(targets, torch.Tensor) else targets)
        scores = np.asarray(probabilities.detach().cpu() if isinstance(probabilities, torch.Tensor) else probabilities)
        observed = np.asarray(mask.detach().cpu() if isinstance(mask, torch.Tensor) else mask).astype(bool)
        if truth.shape != scores.shape or truth.shape != observed.shape or truth.ndim != 2 or truth.shape[1] != len(self.appliance_names):
            raise ValueError("streaming metric inputs have incompatible shapes")
        predictions = scores >= 0.5
        labels = truth.astype(bool)
        self.tp += (labels & predictions & observed).sum(axis=0)
        self.tn += (~labels & ~predictions & observed).sum(axis=0)
        self.fp += (~labels & predictions & observed).sum(axis=0)
        self.fn += (labels & ~predictions & observed).sum(axis=0)
        score_bins = np.minimum((np.clip(scores, 0, 1) * (self.bins - 1)).astype(np.int64), self.bins - 1)
        for index in range(len(self.appliance_names)):
            selected = observed[:, index]
            if selected.any():
                np.add.at(self.positive_hist[index], score_bins[selected, index], labels[selected, index].astype(np.int64))
                np.add.at(self.negative_hist[index], score_bins[selected, index], (~labels[selected, index]).astype(np.int64))

    def compute(self) -> dict[str, dict[str, float | int | None]]:
        result = {}
        for index, name in enumerate(self.appliance_names):
            tp, tn, fp, fn = (int(values[index]) for values in (self.tp, self.tn, self.fp, self.fn))
            observed = tp + tn + fp + fn
            if observed == 0:
                result[name] = _binary_metrics(np.array([]), np.array([]))
                continue
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            specificity = tn / (tn + fp) if tn + fp else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            positive_total = int(self.positive_hist[index].sum())
            negative_total = int(self.negative_hist[index].sum())
            average_precision = None
            if positive_total and negative_total:
                cumulative_positive = np.cumsum(self.positive_hist[index, ::-1])
                cumulative_negative = np.cumsum(self.negative_hist[index, ::-1])
                precisions = cumulative_positive / np.maximum(1, cumulative_positive + cumulative_negative)
                recall_increments = self.positive_hist[index, ::-1] / positive_total
                average_precision = float(np.sum(precisions * recall_increments))
            result[name] = {"observed": observed, "true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn, "precision": float(precision), "recall": float(recall), "specificity": float(specificity), "f1": float(f1), "accuracy": float((tp + tn) / observed), "balanced_accuracy": float((recall + specificity) / 2), "average_precision": average_precision}
        return result


def save_checkpoint(path: str | Path, model: MultiApplianceNILM, metadata: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    reserved = {"format_version", "model_config", "state_dict"}
    if reserved & metadata.keys():
        raise ValueError("metadata uses reserved checkpoint keys")
    torch.save({"format_version": CHECKPOINT_FORMAT_VERSION, "model_config": model.architecture_config(), "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()}, **dict(metadata)}, destination)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported NILM checkpoint format")
    required = {"model_config", "state_dict", "appliance_columns", "feature_columns", "thresholds", "normalization"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"checkpoint is missing keys: {sorted(missing)}")
    if tuple(payload["appliance_columns"]) != APPLIANCE_COLUMNS:
        raise ValueError("checkpoint appliance order does not match Sima's contract")
    if tuple(payload["feature_columns"]) != FEATURE_COLUMNS:
        raise ValueError("checkpoint feature order does not match Sima's contract")
    if payload["model_config"].get("input_channels") != 6 or payload["model_config"].get("num_appliances") != 7:
        raise ValueError("checkpoint model must have six inputs and seven appliance outputs")
    if set(payload["thresholds"]) != set(DEFAULT_THRESHOLDS) or any(float(payload["thresholds"][name]) != value for name, value in DEFAULT_THRESHOLDS.items()):
        raise ValueError("checkpoint thresholds do not match Sima's contract")
    return payload


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _dataset(series, config: NILMConfig, normalization: AggregateNormalization) -> MultiApplianceWindowDataset:
    return MultiApplianceWindowDataset(series, normalization, config.window_size, config.threshold_array, config.expected_interval_seconds, config.gap_tolerance, config.stride)


def _run_epoch(model: MultiApplianceNILM, loader: DataLoader, pos_weight: torch.Tensor, device: torch.device, appliance_names: Sequence[str], optimizer: torch.optim.Optimizer | None = None) -> tuple[float, dict[str, dict[str, float | int | None]]]:
    training = optimizer is not None
    model.train(training)
    total_weighted_loss = 0.0
    total_observed = 0.0
    metrics = StreamingMultiApplianceMetrics(appliance_names)
    for windows, targets, mask in loader:
        windows, targets, mask = windows.to(device), targets.to(device), mask.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(windows)
            loss = masked_bce_with_logits(logits, targets, mask, pos_weight)
            if training:
                loss.backward()
                optimizer.step()
        observed = float(mask.sum().item())
        total_weighted_loss += float(loss.detach().cpu()) * observed
        total_observed += observed
        metrics.update(targets, torch.sigmoid(logits), mask)
    return total_weighted_loss / total_observed, metrics.compute()


def train_from_config(config: NILMConfig, device: str | torch.device | None = None, progress: Callable[[dict[str, float | int]], None] | None = None) -> dict[str, Any]:
    _set_seed(config.seed)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_series = load_cached_houses(config.cache_dir, config.train_houses, expected_config=config)
    validation_series = load_cached_houses(config.cache_dir, config.validation_houses, expected_config=config)
    test_series = load_cached_houses(config.cache_dir, config.test_houses, expected_config=config)
    normalization = fit_aggregate_normalization(train_series, config.feature_columns.index("Aggregate"))
    train_data = _dataset(train_series, config, normalization)
    validation_data = _dataset(validation_series, config, normalization)
    test_data = _dataset(test_series, config, normalization)
    positive, negative, observed = train_data.label_counts()
    if (positive <= 0).any() or (negative <= 0).any():
        names = [config.appliance_columns[i] for i in np.flatnonzero((positive <= 0) | (negative <= 0))]
        raise ValueError(f"training houses need positive and negative observed labels for: {names}")
    weights = negative.astype(np.float32) / positive.astype(np.float32)
    counts = [{"positive": int(p), "negative": int(n), "observed": int(o)} for p, n, o in zip(positive, negative, observed)]
    pos_weight = torch.tensor(weights, device=selected_device)
    options = {"batch_size": config.batch_size, "num_workers": config.num_workers, "pin_memory": selected_device.type == "cuda"}
    train_loader = DataLoader(train_data, shuffle=True, generator=torch.Generator().manual_seed(config.seed), **options)
    validation_loader = DataLoader(validation_data, shuffle=False, **options)
    test_loader = DataLoader(test_data, shuffle=False, **options)
    model = MultiApplianceNILM(len(config.feature_columns), len(config.appliance_columns), config.window_size, config.dropout).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_loss, best_state, stale = float("inf"), None, 0
    history = []
    for epoch in range(1, config.epochs + 1):
        train_loss, train_metrics = _run_epoch(model, train_loader, pos_weight, selected_device, config.appliance_columns, optimizer)
        with torch.no_grad():
            validation_loss, validation_metrics = _run_epoch(model, validation_loader, pos_weight, selected_device, config.appliance_columns)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss, "train_macro_f1": float(np.mean([value["f1"] for value in train_metrics.values() if value["f1"] is not None])), "validation_macro_f1": float(np.mean([value["f1"] for value in validation_metrics.values() if value["f1"] is not None]))})
        if progress is not None:
            progress(history[-1])
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
        test_loss, metrics = _run_epoch(model, test_loader, pos_weight, selected_device, config.appliance_columns)
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.artifact_dir / "best_model.pt"
    metrics_path = config.artifact_dir / "metrics.json"
    figure_paths = generate_training_figures(history, metrics, config.artifact_dir / "figures")
    metadata = {"appliance_columns": list(config.appliance_columns), "feature_columns": list(config.feature_columns), "thresholds": dict(config.thresholds), "normalization": asdict(normalization), "house_splits": {"train": list(config.train_houses), "validation": list(config.validation_houses), "test": list(config.test_houses)}, "training_label_counts": {name: count for name, count in zip(config.appliance_columns, counts)}, "positive_class_weights": {name: float(value) for name, value in zip(config.appliance_columns, weights)}, "history": history, "metrics": metrics, "figure_paths": figure_paths, "data_contract": {"window_size": config.window_size, "expected_interval_seconds": config.expected_interval_seconds, "gap_tolerance": config.gap_tolerance}, "training_config": config.to_dict()}
    save_checkpoint(checkpoint_path, model, metadata)
    report = {"device": str(selected_device), "epochs_completed": len(history), "best_validation_loss": best_loss, "test_loss": test_loss, "test_metrics": metrics, "checkpoint_path": str(checkpoint_path), "figure_paths": figure_paths}
    metrics_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return {**report, "metrics_path": str(metrics_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train one seven-output NILM model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    result = train_from_config(
        NILMConfig.from_json(args.config),
        args.device,
        progress=lambda record: print(json.dumps(record), flush=True),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
