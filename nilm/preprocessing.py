"""Target construction and lazy multi-appliance NILM windows."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class HouseSeries:
    features: np.ndarray
    target_power: np.ndarray
    timestamps: np.ndarray
    house_id: str
    source: str

    def __post_init__(self) -> None:
        if np.asarray(self.features).ndim != 2 or np.asarray(self.target_power).ndim != 2:
            raise ValueError("features and target_power must be two-dimensional")
        if np.asarray(self.timestamps).ndim != 1:
            raise ValueError("timestamps must be one-dimensional")
        if not (len(self.features) == len(self.target_power) == len(self.timestamps)) or len(self.timestamps) == 0:
            raise ValueError("house arrays must be non-empty and have equal row counts")
        if not _all_finite(self.features) or not _all_finite(self.timestamps):
            raise ValueError("features and timestamps must be finite")
        if _has_infinity(self.target_power):
            raise ValueError("target_power cannot contain infinity")


def _all_finite(values: np.ndarray, chunk_rows: int = 1_000_000) -> bool:
    array = np.asarray(values)
    return all(np.isfinite(array[start:start + chunk_rows]).all() for start in range(0, len(array), chunk_rows))


def _has_infinity(values: np.ndarray, chunk_rows: int = 1_000_000) -> bool:
    array = np.asarray(values)
    return any(np.isinf(array[start:start + chunk_rows]).any() for start in range(0, len(array), chunk_rows))


@dataclass(frozen=True)
class AggregateNormalization:
    mean: float
    std: float
    index: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean) or not np.isfinite(self.std) or self.std <= 0:
            raise ValueError("normalization mean/std must be finite and std positive")
        if self.index < 0:
            raise ValueError("normalization index must be non-negative")


def activation_targets_and_mask(target_power: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    power = np.asarray(target_power, dtype=np.float32)
    threshold_values = np.asarray(thresholds, dtype=np.float32)
    if power.ndim != 2 or threshold_values.ndim != 1 or power.shape[1] != threshold_values.size:
        raise ValueError("target_power columns must match the threshold vector")
    if not np.isfinite(threshold_values).all() or (threshold_values <= 0).any():
        raise ValueError("thresholds must be finite and positive")
    mask = np.isfinite(power).astype(np.float32)
    targets = np.where(mask.astype(bool), power >= threshold_values, False).astype(np.float32)
    return targets, mask


def valid_window_centres(timestamps: np.ndarray, window_size: int, expected_interval_seconds: float, gap_tolerance: float = 4.0) -> np.ndarray:
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    if expected_interval_seconds <= 0 or gap_tolerance < 1:
        raise ValueError("invalid timestamp interval settings")
    values = np.asarray(timestamps, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("timestamps must be a finite vector")
    if values.size < window_size:
        return np.empty(0, dtype=np.int64)
    invalid = (np.diff(values) <= 0) | (np.diff(values) > expected_interval_seconds * gap_tolerance)
    width = window_size - 1
    prefix = np.concatenate(([0], np.cumsum(invalid, dtype=np.int64)))
    starts = np.flatnonzero(prefix[width:] - prefix[:-width] == 0)
    return (starts + window_size // 2).astype(np.int64)


def fit_aggregate_normalization(series: Sequence[HouseSeries], aggregate_index: int = 0) -> AggregateNormalization:
    if not series:
        raise ValueError("at least one training house is required")
    count = sum(item.features.shape[0] for item in series)
    total = sum(float(item.features[:, aggregate_index].astype(np.float64).sum()) for item in series)
    squared = sum(float(np.square(item.features[:, aggregate_index].astype(np.float64)).sum()) for item in series)
    mean = total / count
    variance = max(0.0, squared / count - mean * mean)
    return AggregateNormalization(mean, float(np.sqrt(variance)), aggregate_index)


class MultiApplianceWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, series: Sequence[HouseSeries], normalization: AggregateNormalization, window_size: int, thresholds: np.ndarray, expected_interval_seconds: float, gap_tolerance: float = 4.0, stride: int = 1) -> None:
        if not series or stride <= 0:
            raise ValueError("houses are required and stride must be positive")
        self.series = tuple(series)
        self.normalization = normalization
        self.window_size = window_size
        self.half = window_size // 2
        self.thresholds = np.asarray(thresholds, dtype=np.float32)
        self._centres: list[np.ndarray] = []
        self._cumulative: list[int] = []
        total = 0
        for item in self.series:
            centres = valid_window_centres(item.timestamps, window_size, expected_interval_seconds, gap_tolerance)[::stride]
            centres = centres[np.isfinite(item.target_power[centres]).any(axis=1)]
            self._centres.append(centres)
            total += centres.size
            self._cumulative.append(total)
        if total == 0:
            raise ValueError("no valid windows remain")

    def __len__(self) -> int:
        return self._cumulative[-1]

    def label_counts(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positive = np.zeros(self.thresholds.size, dtype=np.int64)
        observed = np.zeros(self.thresholds.size, dtype=np.int64)
        for item, centres in zip(self.series, self._centres):
            for start in range(0, centres.size, 1_000_000):
                power = item.target_power[centres[start:start + 1_000_000]]
                targets, mask = activation_targets_and_mask(power, self.thresholds)
                positive += (targets * mask).sum(axis=0).astype(np.int64)
                observed += mask.sum(axis=0).astype(np.int64)
        return positive, observed - positive, observed

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        series_index = bisect_right(self._cumulative, index)
        previous = 0 if series_index == 0 else self._cumulative[series_index - 1]
        centre = int(self._centres[series_index][index - previous])
        start, stop = centre - self.half, centre + self.half + 1
        window = self.series[series_index].features[start:stop].astype(np.float32, copy=True).T
        window[self.normalization.index] = (window[self.normalization.index] - self.normalization.mean) / self.normalization.std
        targets, mask = activation_targets_and_mask(self.series[series_index].target_power[centre:centre + 1], self.thresholds)
        return torch.from_numpy(window), torch.from_numpy(targets[0]), torch.from_numpy(mask[0])
