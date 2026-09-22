"""Validated configuration for Sima's seven-output REFIT NILM model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


FEATURE_COLUMNS = (
    "Aggregate", "Hour_X", "Hour_Y", "DoW_X", "DoW_Y", "is_weekend"
)
APPLIANCE_COLUMNS = (
    "Television Site", "Toaster", "Microwave", "Kettle", "Computer Site",
    "Washing Machine", "Dishwasher",
)
DEFAULT_THRESHOLDS = {
    "Television Site": 15.0,
    "Toaster": 1000.0,
    "Microwave": 200.0,
    "Kettle": 2000.0,
    "Computer Site": 20.0,
    "Washing Machine": 20.0,
    "Dishwasher": 10.0,
}


def _path(value: object, base: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def _houses(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list of house numbers")
    result = tuple(int(item) for item in value)
    if any(item <= 0 for item in result) or len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique positive house numbers")
    return result


@dataclass(frozen=True)
class NILMConfig:
    data_file: Path
    cache_dir: Path
    artifact_dir: Path
    train_houses: tuple[int, ...]
    validation_houses: tuple[int, ...]
    test_houses: tuple[int, ...]
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    appliance_columns: tuple[str, ...] = APPLIANCE_COLUMNS
    thresholds: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    timestamp_column: str = "Time"
    house_column: str = "house"
    window_size: int = 37
    expected_interval_seconds: float = 8.0
    gap_tolerance: float = 4.0
    stride: int = 1
    max_rows_per_house: int | None = None
    batch_size: int = 256
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 5
    num_workers: int = 0
    seed: int = 42
    dropout: float = 0.2

    def __post_init__(self) -> None:
        if self.window_size <= 0 or self.window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        if self.feature_columns != FEATURE_COLUMNS:
            raise ValueError(f"feature_columns must match Sima's fixed order: {FEATURE_COLUMNS}")
        if self.appliance_columns != APPLIANCE_COLUMNS:
            raise ValueError(f"appliance_columns must match Sima's fixed order: {APPLIANCE_COLUMNS}")
        if set(self.thresholds) != set(self.appliance_columns):
            raise ValueError("threshold keys must exactly match appliance_columns")
        if any(not np.isfinite(float(v)) or float(v) <= 0 for v in self.thresholds.values()):
            raise ValueError("all appliance thresholds must be finite and positive")
        splits = {"train": set(self.train_houses), "validation": set(self.validation_houses), "test": set(self.test_houses)}
        names = tuple(splits)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                overlap = splits[left] & splits[right]
                if overlap:
                    raise ValueError(f"house split overlap between {left} and {right}: {sorted(overlap)}")
        if self.expected_interval_seconds <= 0 or self.gap_tolerance < 1:
            raise ValueError("timestamp interval must be positive and gap_tolerance at least 1")
        for name in ("stride", "batch_size", "epochs", "patience"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_rows_per_house is not None and self.max_rows_per_house <= 0:
            raise ValueError("max_rows_per_house must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def all_houses(self) -> tuple[int, ...]:
        return self.train_houses + self.validation_houses + self.test_houses

    @property
    def threshold_array(self) -> np.ndarray:
        return np.asarray([self.thresholds[name] for name in self.appliance_columns], dtype=np.float32)

    @classmethod
    def from_json(cls, path: str | Path) -> "NILMConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload, config_path.parent.resolve())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], base_dir: str | Path = ".") -> "NILMConfig":
        base = Path(base_dir).resolve()
        data = payload.get("data")
        training = payload.get("training", {})
        if not isinstance(data, Mapping):
            raise ValueError("config requires a data object")
        if not isinstance(training, Mapping):
            raise ValueError("training must be an object")
        try:
            data_file = _path(data["data_file"], base)
            cache_dir = _path(data["cache_dir"], base)
            train_houses = _houses(data["train_houses"], "train_houses")
            validation_houses = _houses(data["validation_houses"], "validation_houses")
            test_houses = _houses(data["test_houses"], "test_houses")
        except KeyError as error:
            raise ValueError(f"missing required config field: {error.args[0]}") from error
        feature_columns = tuple(data.get("feature_columns", FEATURE_COLUMNS))
        appliance_columns = tuple(data.get("appliance_columns", APPLIANCE_COLUMNS))
        thresholds = payload.get("thresholds", DEFAULT_THRESHOLDS)
        if not isinstance(thresholds, Mapping):
            raise ValueError("thresholds must be an object")
        artifact_dir = _path(payload.get("artifact_dir", "artifacts/nilm"), base)
        return cls(
            data_file=data_file, cache_dir=cache_dir, artifact_dir=artifact_dir,
            train_houses=train_houses, validation_houses=validation_houses, test_houses=test_houses,
            feature_columns=feature_columns, appliance_columns=appliance_columns,
            thresholds={str(k): float(v) for k, v in thresholds.items()},
            timestamp_column=str(data.get("timestamp_column", "Time")),
            house_column=str(data.get("house_column", "house")),
            window_size=int(data.get("window_size", 37)),
            expected_interval_seconds=float(data.get("expected_interval_seconds", 8.0)),
            gap_tolerance=float(data.get("gap_tolerance", 4.0)),
            stride=int(data.get("stride", 1)),
            max_rows_per_house=None if data.get("max_rows_per_house") is None else int(data["max_rows_per_house"]),
            batch_size=int(training.get("batch_size", 256)), epochs=int(training.get("epochs", 20)),
            learning_rate=float(training.get("learning_rate", 1e-3)), weight_decay=float(training.get("weight_decay", 1e-5)),
            patience=int(training.get("patience", 5)), num_workers=int(training.get("num_workers", 0)),
            seed=int(training.get("seed", 42)), dropout=float(training.get("dropout", 0.2)),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in ("data_file", "cache_dir", "artifact_dir"):
            result[name] = str(result[name])
        result["thresholds"] = dict(self.thresholds)
        return result
