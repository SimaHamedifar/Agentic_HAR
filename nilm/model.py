"""Shared CNN with one output head per REFIT appliance."""

from __future__ import annotations

import torch
from torch import nn


class MultiApplianceNILM(nn.Module):
    def __init__(self, input_channels: int = 6, num_appliances: int = 7, window_size: int = 37, dropout: float = 0.2) -> None:
        super().__init__()
        if input_channels <= 0 or num_appliances <= 0:
            raise ValueError("input_channels and num_appliances must be positive")
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.input_channels = input_channels
        self.num_appliances = num_appliances
        self.window_size = window_size
        self.dropout = dropout
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList(nn.Linear(256, 1) for _ in range(num_appliances))

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        expected = (self.input_channels, self.window_size)
        if windows.ndim != 3 or tuple(windows.shape[1:]) != expected:
            raise ValueError(f"input must have shape [batch, {expected[0]}, {expected[1]}]")
        shared = self.features(windows)
        return torch.cat([head(shared) for head in self.heads], dim=1)

    def architecture_config(self) -> dict[str, int | float]:
        return {"input_channels": self.input_channels, "num_appliances": self.num_appliances, "window_size": self.window_size, "dropout": self.dropout}
