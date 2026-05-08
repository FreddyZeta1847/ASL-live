"""MLP classifier for normalized hand-landmark vectors.

Architecture exactly per feature-3 §3.1:
    Linear(63, 128) -> ReLU -> Dropout(0.2) -> Linear(128, 64) -> ReLU
                  -> Linear(64, 26)

The model returns **raw logits** (no softmax). Reasons:
- Cross-entropy loss expects logits and applies its own log-softmax
  internally — a softmax inside the model would double-apply it.
- ONNX is leaner without it. Softmax is applied at inference time in
  ``recognition/classifier.py``.

~30k parameters total, ~120 KB at FP32. Inference is sub-microsecond on
a laptop CPU, well under the per-frame budget on a Pi 5.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from asl_live.config import LANDMARK_FEATURES, NUM_CLASSES


HIDDEN_1 = 128
HIDDEN_2 = 64
DROPOUT = 0.2


class MLP(nn.Module):
    """4-layer feed-forward classifier (feature-3 §3.1)."""

    def __init__(
        self,
        in_features: int = LANDMARK_FEATURES,
        num_classes: int = NUM_CLASSES,
        hidden_1: int = HIDDEN_1,
        hidden_2: int = HIDDEN_2,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
