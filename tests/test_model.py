"""Sanity tests for the MLP architecture."""
from __future__ import annotations

import torch

from asl_live.config import LANDMARK_FEATURES, NUM_CLASSES
from asl_live.train.model import MLP


def test_forward_shape_matches_num_classes():
    model = MLP()
    out = model(torch.randn(4, LANDMARK_FEATURES))
    assert out.shape == (4, NUM_CLASSES)


def test_forward_returns_logits_not_probs():
    """Output should be raw logits — not constrained to sum to 1 or be in [0, 1]."""
    model = MLP()
    out = model(torch.randn(8, LANDMARK_FEATURES))
    # Logits are unbounded and rows almost certainly don't sum to 1.0.
    assert out.dtype == torch.float32
    assert not torch.allclose(out.sum(dim=1), torch.ones(8), atol=0.1)


def test_param_count_matches_architecture():
    """Architecture sanity check.

    Exact: (63*128 + 128) + (128*64 + 64) + (64*26 + 26) = 8192 + 8256 + 1690 = 18138.
    feature-3 §3.1 says '~30k' as a rough figure; the real value is ~18k.
    """
    model = MLP()
    n_params = sum(p.numel() for p in model.parameters())
    expected = (
        LANDMARK_FEATURES * 128 + 128
        + 128 * 64 + 64
        + 64 * NUM_CLASSES + NUM_CLASSES
    )
    assert n_params == expected == 18138
    assert 10_000 < n_params < 100_000


def test_dropout_is_active_in_train_mode():
    """In train mode, dropout zeroes some activations -> outputs differ across calls."""
    torch.manual_seed(0)
    model = MLP(dropout=0.5)
    model.train()
    x = torch.randn(16, LANDMARK_FEATURES)
    out1 = model(x)
    out2 = model(x)
    # With dropout=0.5 in train mode, two forward passes almost certainly differ.
    assert not torch.equal(out1, out2)


def test_dropout_is_deterministic_in_eval_mode():
    """In eval mode, dropout is disabled -> the same input gives the same output."""
    model = MLP()
    model.eval()
    x = torch.randn(16, LANDMARK_FEATURES)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.equal(out1, out2)
