"""Read the shared HAR checkpoint format for training and inference."""

from pathlib import Path
from typing import Any

import torch

CHECKPOINT_FORMAT_VERSION = 1


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported HAR checkpoint format")
    required = {"model_config", "state_dict", "classes", "feature_names", "window_size"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"checkpoint is missing keys: {sorted(missing)}")
    return payload
