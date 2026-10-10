"""Inference from one trained MuRAL BiLSTM + self-attention HAR checkpoint.

This module is the clean prediction interface other components (in
particular the LangGraph agent in ``agent/``) should call. It has no
dependency on agent code, so the model implementation stays decoupled from
the LLM logic; the agent only needs ``HARPredictor.predict_activity``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from activity_ml.classifier import load_checkpoint
from activity_ml.model import BiLSTMSelfAttentionClassifier


def load_feature_window_csv(path: str | Path, feature_names: Sequence[str]) -> np.ndarray:
    """Load a window of encoded feature rows from a CSV (one row per event)."""
    frame = pd.read_csv(path)
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError(f"window CSV is missing feature columns: {missing}")
    values = frame[list(feature_names)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("window CSV feature values must be numeric and finite")
    return values


class HARPredictor:
    """Loads a trained BiLSTM + self-attention HAR checkpoint and predicts activity probabilities."""

    def __init__(
        self,
        model: BiLSTMSelfAttentionClassifier,
        classes: Sequence[str],
        feature_names: Sequence[str],
        window_size: int,
        device: torch.device,
    ) -> None:
        self.model = model.to(device).eval()
        self.classes = tuple(classes)
        self.feature_names = tuple(feature_names)
        self.window_size = window_size
        self.device = device

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device | None = None) -> "HARPredictor":
        selected = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        payload = load_checkpoint(path, selected)
        model = BiLSTMSelfAttentionClassifier(**payload["model_config"])
        model.load_state_dict(payload["state_dict"])
        return cls(model, payload["classes"], payload["feature_names"], payload["window_size"], selected)

    def predict_proba(self, sequence: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """Return the raw probability vector (sums to ~1) for one input window."""
        values = np.asarray(sequence, dtype=np.float32)
        expected = (self.window_size, len(self.feature_names))
        if values.shape != expected:
            raise ValueError(f"expected feature window shape {expected}, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("feature window must contain only finite values")
        tensor = torch.from_numpy(values).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probabilities = self.model.predict_proba(tensor)[0].cpu().numpy()
        return probabilities

    def predict_activity(self, sequence: Sequence[Sequence[float]] | np.ndarray) -> dict[str, object]:
        """Predict the most likely activity and the full probability distribution.

        Returns e.g.::

            {
                "predicted_activity": "watching TV",
                "probabilities": {"watching TV": 0.81, "reading books": 0.09, ...},
            }
        """
        probabilities = self.predict_proba(sequence)
        predicted_index = int(np.argmax(probabilities))
        return {
            "predicted_activity": self.classes[predicted_index],
            "probabilities": {name: float(prob) for name, prob in zip(self.classes, probabilities)},
        }


def predict_activity(
    sequence: Sequence[Sequence[float]] | np.ndarray, checkpoint_path: str | Path, device: str | torch.device | None = None
) -> dict[str, object]:
    """Convenience wrapper: load a checkpoint and predict in one call."""
    return HARPredictor.from_checkpoint(checkpoint_path, device).predict_activity(sequence)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict a MuRAL activity distribution from one feature window")
    parser.add_argument("--checkpoint", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--window", help="JSON matrix with shape [window_size, num_features]")
    source.add_argument("--window-file", help="CSV containing one row per event and the encoded feature columns")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    predictor = HARPredictor.from_checkpoint(args.checkpoint, args.device)
    window = json.loads(args.window) if args.window is not None else load_feature_window_csv(args.window_file, predictor.feature_names)
    print(json.dumps(predictor.predict_activity(window), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
