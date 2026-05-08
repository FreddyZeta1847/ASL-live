"""Unit tests for training-time augmentation.

Each perturbation (noise, scale, translation) is exercised in isolation
by zeroing the others, then together at p=0 (identity) and p=1 (always
applied) to confirm gating works.
"""
from __future__ import annotations

import pytest
import torch

from asl_live.config import LANDMARK_DIMS, LANDMARK_FEATURES, NUM_LANDMARKS
from asl_live.train.augment import RandomAugment


def _random_batch(B: int = 8, seed: int = 1) -> torch.Tensor:
    """Reproducible (B, 63) batch of plausibly-normalized vectors."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, LANDMARK_FEATURES, generator=g)


# ---------------------------------------------------------------------------
# Shape and contract
# ---------------------------------------------------------------------------


def test_output_has_same_shape_as_input():
    aug = RandomAugment()
    x = _random_batch(8)
    out = aug(x)
    assert out.shape == x.shape
    assert out.dtype == torch.float32


def test_input_is_not_mutated():
    aug = RandomAugment(p=1.0)
    x = _random_batch(4)
    x_before = x.clone()
    aug(x)
    assert torch.equal(x, x_before)


def test_bad_input_shape_raises():
    aug = RandomAugment()
    with pytest.raises(ValueError):
        aug(torch.zeros(LANDMARK_FEATURES))  # unbatched
    with pytest.raises(ValueError):
        aug(torch.zeros(4, 42))  # wrong feature count


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"noise_std": -0.1},
        {"scale_range": (1.1, 0.9)},
        {"trans_range": -0.05},
        {"p": -0.1},
        {"p": 1.1},
    ],
)
def test_bad_constructor_args_raise(kwargs):
    with pytest.raises(ValueError):
        RandomAugment(**kwargs)


# ---------------------------------------------------------------------------
# p=0 → identity
# ---------------------------------------------------------------------------


def test_p_zero_is_identity():
    aug = RandomAugment(p=0.0)
    x = _random_batch(8)
    out = aug(x)
    assert torch.equal(out, x)


def test_disabled_perturbations_are_identity():
    """All three perturbations turned off via parameters → identity at any p."""
    aug = RandomAugment(noise_std=0.0, scale_range=(1.0, 1.0), trans_range=0.0, p=1.0)
    x = _random_batch(8)
    out = aug(x)
    assert torch.equal(out, x)


# ---------------------------------------------------------------------------
# Noise alone
# ---------------------------------------------------------------------------


def test_noise_alone_changes_every_coord_at_p1():
    aug = RandomAugment(noise_std=0.01, scale_range=(1.0, 1.0), trans_range=0.0, p=1.0)
    x = _random_batch(8)
    out = aug(x)
    assert not torch.equal(out, x)


def test_noise_magnitude_is_small_relative_to_std():
    aug = RandomAugment(noise_std=0.01, scale_range=(1.0, 1.0), trans_range=0.0, p=1.0)
    x = torch.zeros(2048, LANDMARK_FEATURES)  # large batch for tight stats
    out = aug(x)
    # Output is purely noise. Empirical std should be near 0.01.
    assert 0.008 < out.std().item() < 0.012


def test_noise_affects_all_three_dims_uniformly():
    """Noise is applied to x, y, AND z — verify by checking variance per dim."""
    aug = RandomAugment(noise_std=0.05, scale_range=(1.0, 1.0), trans_range=0.0, p=1.0)
    x = torch.zeros(2048, LANDMARK_FEATURES)
    out = aug(x).view(-1, NUM_LANDMARKS, LANDMARK_DIMS)
    std_per_dim = out.std(dim=(0, 1))  # (3,)
    assert all(0.045 < s.item() < 0.055 for s in std_per_dim)


# ---------------------------------------------------------------------------
# Scale alone
# ---------------------------------------------------------------------------


def test_scale_alone_within_range_at_p1():
    aug = RandomAugment(noise_std=0.0, scale_range=(0.95, 1.05), trans_range=0.0, p=1.0)
    # Input with no zeros so the ratio is well-defined per element.
    x = torch.full((4, LANDMARK_FEATURES), 1.0)
    out = aug(x)
    ratios = out / x  # all elements in same sample share one scalar ratio
    # Per sample, the ratio must be in [0.95, 1.05]
    for sample_ratios in ratios:
        sample_set = sample_ratios.unique()
        assert sample_set.numel() == 1, "scale should be one ratio per sample"
        r = sample_set.item()
        assert 0.95 - 1e-6 <= r <= 1.05 + 1e-6


def test_scale_applied_uniformly_across_landmarks_and_dims():
    """One scalar multiplier per sample, applied to every (landmark, dim)."""
    aug = RandomAugment(noise_std=0.0, scale_range=(0.9, 1.1), trans_range=0.0, p=1.0)
    # Distinct, non-zero values so ratios are well-defined everywhere.
    x = torch.arange(1, 1 + 4 * LANDMARK_FEATURES, dtype=torch.float32).view(4, LANDMARK_FEATURES)
    out = aug(x)
    ratios = out / x  # (4, 63)
    for sample_ratios in ratios:
        # Same scalar across all 63 entries of this sample.
        assert torch.allclose(sample_ratios, sample_ratios[0].expand_as(sample_ratios), atol=1e-6)


# ---------------------------------------------------------------------------
# Translation alone
# ---------------------------------------------------------------------------


def test_translation_only_shifts_x_and_y_not_z():
    aug = RandomAugment(noise_std=0.0, scale_range=(1.0, 1.0), trans_range=0.02, p=1.0)
    x = torch.full((4, LANDMARK_FEATURES), 1.0)
    out = aug(x)
    delta = (out - x).view(4, NUM_LANDMARKS, LANDMARK_DIMS)
    # z (index 2) should be unchanged.
    assert torch.equal(delta[..., 2], torch.zeros_like(delta[..., 2]))
    # x and y deltas are non-zero (with overwhelming probability) at p=1.
    assert delta[..., 0].abs().sum() > 0
    assert delta[..., 1].abs().sum() > 0


def test_translation_within_range():
    aug = RandomAugment(noise_std=0.0, scale_range=(1.0, 1.0), trans_range=0.02, p=1.0)
    x = torch.zeros(64, LANDMARK_FEATURES)
    out = aug(x).view(64, NUM_LANDMARKS, LANDMARK_DIMS)
    # All landmarks in a sample share the same (dx, dy, 0).
    # Magnitude per dim per sample within [-trans_range, trans_range].
    for sample in out:
        for dim in (0, 1):
            vals = sample[:, dim]
            assert (vals - vals[0]).abs().max() < 1e-6  # uniform across landmarks
            assert vals[0].abs() <= 0.02 + 1e-6


def test_translation_uniform_across_landmarks():
    """Same (dx, dy) applied to every landmark of a given sample."""
    aug = RandomAugment(noise_std=0.0, scale_range=(1.0, 1.0), trans_range=0.05, p=1.0)
    x = torch.zeros(8, LANDMARK_FEATURES)
    out = aug(x).view(8, NUM_LANDMARKS, LANDMARK_DIMS)
    for sample in out:
        # All 21 landmarks share the same dx and the same dy.
        assert torch.allclose(sample[:, 0], sample[0, 0].expand(NUM_LANDMARKS), atol=1e-6)
        assert torch.allclose(sample[:, 1], sample[0, 1].expand(NUM_LANDMARKS), atol=1e-6)


# ---------------------------------------------------------------------------
# Determinism via injected generator
# ---------------------------------------------------------------------------


def test_same_generator_seed_produces_same_output():
    x = _random_batch(8, seed=99)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    out1 = RandomAugment(generator=g1)(x)
    out2 = RandomAugment(generator=g2)(x)
    assert torch.equal(out1, out2)


def test_different_seeds_produce_different_output():
    x = _random_batch(8, seed=99)
    out1 = RandomAugment(generator=torch.Generator().manual_seed(1))(x)
    out2 = RandomAugment(generator=torch.Generator().manual_seed(2))(x)
    assert not torch.equal(out1, out2)
