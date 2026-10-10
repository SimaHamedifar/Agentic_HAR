"""BiLSTM + genuine multi-head self-attention classifier for MuRAL HAR.

This was the best-performing of several HAR architectures tried (BiLSTM
with additive attention pooling, a pure Transformer with no recurrence,
and this one); see docs/HAR.md for the full writeup, hyperparameters, and
results. The BiLSTM's output sequence is run through real multi-head
self-attention: query, key, and value are all derived from the same
sequence, so every timestep's representation is updated by attending to
every other timestep in the window (standard Transformer-encoder-sublayer
style), rather than a single pooled summary attending to the sequence.
"""

from __future__ import annotations

import torch
from torch import nn


class BiLSTMSelfAttentionClassifier(nn.Module):
    """BiLSTM + multi-head self-attention classifier over a MuRAL event window.

    Input: ``[batch, window_size, input_size]`` encoded event features.
    Output: ``[batch, num_classes]`` unnormalized class logits (softmax is
    applied by the caller, e.g. via ``predict_proba``, not inside the model,
    so the module can be trained directly with ``nn.CrossEntropyLoss``).

    Pipeline: BiLSTM (contextualizes each event locally) -> multi-head
    self-attention with a residual connection and layer norm (each
    timestep's representation is refined by attending to every other
    timestep in the window) -> mean pooling over the window ->
    classification head.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        num_attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if input_size <= 0 or num_classes <= 0:
            raise ValueError("input_size and num_classes must be positive")
        if hidden_size <= 0 or num_layers <= 0:
            raise ValueError("hidden_size and num_layers must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        num_directions = 2 if bidirectional else 1
        pooled_size = hidden_size * num_directions
        if pooled_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size * num_directions ({pooled_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = num_directions
        self.num_attention_heads = num_attention_heads

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.self_attention = nn.MultiheadAttention(
            embed_dim=pooled_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(pooled_size)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pooled_size, num_classes),
        )

    def forward(self, sequences: torch.Tensor, return_attention: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if sequences.ndim != 3 or sequences.shape[2] != self.input_size:
            raise ValueError(
                f"input must have shape [batch, window_size, {self.input_size}], got {tuple(sequences.shape)}"
            )
        outputs, _ = self.lstm(sequences)  # [batch, window_size, hidden_size * num_directions]
        # Self-attention: query, key, and value all come from the same
        # sequence, so every timestep attends to every other timestep.
        attended, attention_weights = self.self_attention(
            outputs, outputs, outputs, need_weights=return_attention, average_attn_weights=True,
        )
        combined = self.attention_norm(outputs + self.attention_dropout(attended))
        pooled = combined.mean(dim=1)  # [batch, pooled_size]
        logits = self.head(pooled)
        if return_attention:
            return logits, attention_weights  # attention_weights: [batch, window_size, window_size]
        return logits

    def predict_proba(self, sequences: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(sequences)
            return torch.softmax(logits, dim=-1)

    def architecture_config(self) -> dict[str, int | float | bool]:
        return {
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "num_attention_heads": self.num_attention_heads,
        }
