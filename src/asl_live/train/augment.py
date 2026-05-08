"""Training-time augmentation for normalized landmark vectors.

Per feature-3 §3.3 — three independent perturbations applied to the train
DataLoader's batches (val/test pass clean data). Each perturbation is
gated *per sample* by an independent Bernoulli(p) draw, so on average
half the samples in any batch get each perturbation.

What's intentionally NOT here
-----------------------------
- **Mirror** — already applied once at ingest (feature-2). Doing it again
  doubles a transformation the dataset has already absorbed and biases
  the model toward symmetry that real ASL doesn't fully have.
- **Rotation** — some ASL letter pairs are distinguished by hand tilt;
  rotation augmentation would conflate them.

Why augment at all
------------------
The dataset is fixed at ~113k vectors and the MLP has ~30k parameters —
plenty of capacity to memorize. Augmentation injects controlled
variation per epoch so the model can't lock onto exact memorized
positions and is forced to learn the shape *family* instead.
"""
from __future__ import annotations

from typing import Optional

import torch

from asl_live.config import LANDMARK_DIMS, LANDMARK_FEATURES, NUM_LANDMARKS


class RandomAugment:
    """Per-batch augmentation: noise + scale + translation, each gated at ``p``.

    Call signature: ``aug(batch)`` where ``batch`` is a ``(B, 63)`` float
    tensor of normalized landmark vectors. Returns a new tensor of the
    same shape; the input is not mutated.

    Defaults match feature-3 §3.3 exactly. Override to disable one
    perturbation by passing ``noise_std=0``, ``scale_range=(1.0, 1.0)``,
    or ``trans_range=0.0``.
    """

    def __init__(
        self,
        *,
        noise_std: float = 0.01,
        scale_range: tuple[float, float] = (0.95, 1.05),
        trans_range: float = 0.02,
        p: float = 0.5,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {noise_std}")
        if scale_range[0] > scale_range[1]:
            raise ValueError(f"scale_range must be (lo, hi) with lo<=hi, got {scale_range}")
        if trans_range < 0:
            raise ValueError(f"trans_range must be >= 0, got {trans_range}")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")

        self.noise_std = float(noise_std)
        self.scale_lo, self.scale_hi = float(scale_range[0]), float(scale_range[1])
        self.trans_range = float(trans_range)
        self.p = float(p)
        self.generator = generator

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim != 2 or batch.shape[1] != LANDMARK_FEATURES:
            raise ValueError(
                f"expected (B, {LANDMARK_FEATURES}) batch, got shape {tuple(batch.shape)}"
            )

        B = batch.shape[0]
        x = batch.view(B, NUM_LANDMARKS, LANDMARK_DIMS).clone().float()

        x = self._maybe_add_noise(x)
        x = self._maybe_scale(x)
        x = self._maybe_translate(x)

        return x.view(B, LANDMARK_FEATURES)

    # ----- per-perturbation helpers --------------------------------------

    def _bernoulli_gate(self, batch_size: int) -> torch.Tensor:
        """Per-sample (B, 1, 1) {0., 1.} mask drawn from Bernoulli(p)."""
        u = torch.rand(batch_size, 1, 1, generator=self.generator)
        return (u < self.p).float()

    def _maybe_add_noise(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_std == 0:
            return x
        gate = self._bernoulli_gate(x.shape[0])
        noise = torch.randn(x.shape, generator=self.generator) * self.noise_std
        return x + gate * noise

    def _maybe_scale(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale_lo == 1.0 and self.scale_hi == 1.0:
            return x
        gate = self._bernoulli_gate(x.shape[0])
        scales = torch.empty(x.shape[0], 1, 1).uniform_(
            self.scale_lo, self.scale_hi, generator=self.generator
        )
        # Where gate=1 use scales; where gate=0 multiply by 1.0 (identity).
        return x * (1.0 + gate * (scales - 1.0))

    def _maybe_translate(self, x: torch.Tensor) -> torch.Tensor:
        if self.trans_range == 0:
            return x
        gate = self._bernoulli_gate(x.shape[0])
        trans = torch.empty(x.shape[0], 1, LANDMARK_DIMS).uniform_(
            -self.trans_range, self.trans_range, generator=self.generator
        )
        # Translation is on x and y only — z is depth, not in the image plane.
        trans[..., 2] = 0
        return x + gate * trans
