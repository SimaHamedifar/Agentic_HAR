"""Inference from one multi-appliance NILM checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from nilm.model import MultiApplianceNILM
from nilm.preprocessing import AggregateNormalization
from nilm.train import load_checkpoint


def load_feature_window_csv(path: str | Path, feature_columns: Sequence[str]) -> np.ndarray:
    frame = pd.read_csv(path)
    missing = [name for name in feature_columns if name not in frame.columns]
    if missing:
        raise ValueError(f"window CSV is missing feature columns: {missing}")
    values = frame[list(feature_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32).T
    if not np.isfinite(values).all():
        raise ValueError("window CSV feature values must be numeric and finite")
    return values


class NILMPredictor:
    def __init__(self, model: MultiApplianceNILM, appliance_columns: Sequence[str], feature_columns: Sequence[str], normalization: AggregateNormalization, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.appliance_columns = tuple(appliance_columns)
        self.feature_columns = tuple(feature_columns)
        self.normalization = normalization
        self.device = device

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device | None = None) -> "NILMPredictor":
        selected = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        payload = load_checkpoint(path, selected)
        model = MultiApplianceNILM(**payload["model_config"])
        model.load_state_dict(payload["state_dict"])
        return cls(model, payload["appliance_columns"], payload["feature_columns"], AggregateNormalization(**payload["normalization"]), selected)

    def predict(self, feature_window: Sequence[Sequence[float]] | np.ndarray) -> dict[str, float]:
        values = np.asarray(feature_window, dtype=np.float32)
        expected = (self.model.input_channels, self.model.window_size)
        if values.shape != expected:
            raise ValueError(f"expected feature window shape {expected}, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("feature window must contain only finite values")
        normalized = values.copy()
        normalized[self.normalization.index] = (normalized[self.normalization.index] - self.normalization.mean) / self.normalization.std
        tensor = torch.from_numpy(normalized).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probabilities = torch.sigmoid(self.model(tensor))[0].cpu().numpy()
        return {name: float(value) for name, value in zip(self.appliance_columns, probabilities)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict seven appliance probabilities from one feature window")
    parser.add_argument("--checkpoint", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--window", help="JSON matrix with shape [6, window_size]")
    source.add_argument("--window-file", help="CSV containing one row per time step and the six feature columns")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    predictor = NILMPredictor.from_checkpoint(args.checkpoint, args.device)
    window = json.loads(args.window) if args.window is not None else load_feature_window_csv(args.window_file, predictor.feature_columns)
    print(json.dumps(predictor.predict(window), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
