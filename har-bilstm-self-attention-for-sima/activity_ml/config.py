"""Validated configuration for the MuRAL BiLSTM + self-attention HAR classifier."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


def _path(value: object, base: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def _houses(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list of house ids")
    result = tuple(str(item) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique house ids")
    return result


@dataclass(frozen=True)
class HARConfig:
    """Configuration for training and evaluating the BiLSTM + self-attention HAR classifier."""

    mural_dir: Path
    artifact_dir: Path
    train_houses: tuple[str, ...]
    validation_houses: tuple[str, ...]
    test_houses: tuple[str, ...]
    window_size: int = 10
    stride: int = 1
    hidden_size: int = 64
    num_layers: int = 2
    bidirectional: bool = True
    num_attention_heads: int = 4
    dropout: float = 0.2
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 8
    num_workers: int = 0
    seed: int = 42
    use_class_weights: bool = True
    loss_type: str = "focal"
    focal_gamma: float = 1.5

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if self.stride <= 0:
            raise ValueError("stride must be a positive integer")
        splits = {
            "train": set(self.train_houses),
            "validation": set(self.validation_houses),
            "test": set(self.test_houses),
        }
        names = tuple(splits)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                overlap = splits[left] & splits[right]
                if overlap:
                    raise ValueError(f"house split overlap between {left} and {right}: {sorted(overlap)}")
        for name in ("batch_size", "epochs", "patience", "num_layers"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        pooled_size = self.hidden_size * (2 if self.bidirectional else 1)
        if pooled_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size * (2 if bidirectional else 1) ({pooled_size}) must be divisible "
                f"by num_attention_heads ({self.num_attention_heads})"
            )
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.loss_type not in ("cross_entropy", "focal"):
            raise ValueError("loss_type must be 'cross_entropy' or 'focal'")
        if self.focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative")

    @property
    def all_houses(self) -> tuple[str, ...]:
        return self.train_houses + self.validation_houses + self.test_houses

    @classmethod
    def from_json(cls, path: str | Path) -> "HARConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload, config_path.parent.resolve())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], base_dir: str | Path = ".") -> "HARConfig":
        base = Path(base_dir).resolve()
        data = payload.get("data")
        training = payload.get("training", {})
        model = payload.get("model", {})
        if not isinstance(data, Mapping):
            raise ValueError("config requires a data object")
        if not isinstance(training, Mapping):
            raise ValueError("training must be an object")
        if not isinstance(model, Mapping):
            raise ValueError("model must be an object")
        try:
            mural_dir = _path(data["mural_dir"], base)
            train_houses = _houses(data["train_houses"], "train_houses")
            validation_houses = _houses(data["validation_houses"], "validation_houses")
            test_houses = _houses(data["test_houses"], "test_houses")
        except KeyError as error:
            raise ValueError(f"missing required config field: {error.args[0]}") from error
        artifact_dir = _path(payload.get("artifact_dir", "artifacts/activity_ml"), base)
        return cls(
            mural_dir=mural_dir,
            artifact_dir=artifact_dir,
            train_houses=train_houses,
            validation_houses=validation_houses,
            test_houses=test_houses,
            window_size=int(data.get("window_size", 10)),
            stride=int(data.get("stride", 1)),
            hidden_size=int(model.get("hidden_size", 64)),
            num_layers=int(model.get("num_layers", 2)),
            bidirectional=bool(model.get("bidirectional", True)),
            num_attention_heads=int(model.get("num_attention_heads", 4)),
            dropout=float(model.get("dropout", 0.2)),
            batch_size=int(training.get("batch_size", 64)),
            epochs=int(training.get("epochs", 50)),
            learning_rate=float(training.get("learning_rate", 1e-3)),
            weight_decay=float(training.get("weight_decay", 1e-5)),
            patience=int(training.get("patience", 8)),
            num_workers=int(training.get("num_workers", 0)),
            seed=int(training.get("seed", 42)),
            use_class_weights=bool(training.get("use_class_weights", True)),
            loss_type=str(training.get("loss_type", "focal")),
            focal_gamma=float(training.get("focal_gamma", 1.5)),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in ("mural_dir", "artifact_dir"):
            result[name] = str(result[name])
        return result
