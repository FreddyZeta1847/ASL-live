"""Unit tests for the pure normalization helpers in
``asl_live.recognition.landmarks``.

These tests intentionally avoid MediaPipe and OpenCV — they only
exercise functions over numpy arrays. Run with ``pytest`` from the
repository root.
"""
from __future__ import annotations

import numpy as np
import pytest

from asl_live.config import (
    LANDMARK_DIMS,
    LANDMARK_FEATURES,
    NUM_LANDMARKS,
)
from asl_live.recognition.landmarks import (
    _mirror,
    _normalize,
    _scale_to_unit_max,
    _translate_to_wrist_origin,
)


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _synthetic_coords() -> np.ndarray:
    """A 21x3 array with the wrist at (0.5, 0.5, 0) and landmarks fanning out."""
    return np.array(
        [[0.5 + i * 0.01, 0.5 - i * 0.005, 0.0] for i in range(NUM_LANDMARKS)],
        dtype=np.float32,
    )


def _all_zeros_coords() -> np.ndarray:
    """A degenerate 21x3 array where every landmark coincides with the wrist."""
    return np.zeros((NUM_LANDMARKS, LANDMARK_DIMS), dtype=np.float32)


# ---------------------------------------------------------------------------
# _translate_to_wrist_origin
# ---------------------------------------------------------------------------


def test_translate_puts_wrist_at_origin():
    coords = _synthetic_coords()
    out = _translate_to_wrist_origin(coords)
    assert out.shape == (NUM_LANDMARKS, LANDMARK_DIMS)
    np.testing.assert_allclose(out[0], [0.0, 0.0, 0.0])


def test_translate_does_not_mutate_input():
    coords = _synthetic_coords()
    before = coords.copy()
    _translate_to_wrist_origin(coords)
    np.testing.assert_array_equal(coords, before)


# ---------------------------------------------------------------------------
# _scale_to_unit_max
# ---------------------------------------------------------------------------


def test_scale_makes_max_distance_unit():
    translated = _translate_to_wrist_origin(_synthetic_coords())
    scaled = _scale_to_unit_max(translated)
    assert scaled is not None
    distances = np.linalg.norm(scaled, axis=1)
    assert distances.max() == pytest.approx(1.0)


def test_scale_returns_none_on_degenerate_input():
    assert _scale_to_unit_max(_all_zeros_coords()) is None


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------


def test_normalize_returns_flat_63_dim_float32_vector():
    out = _normalize(_synthetic_coords())
    assert out is not None
    assert out.shape == (LANDMARK_FEATURES,)
    assert out.dtype == np.float32


def test_normalize_returns_none_on_degenerate_input():
    assert _normalize(_all_zeros_coords()) is None


# ---------------------------------------------------------------------------
# _mirror
# ---------------------------------------------------------------------------


def test_mirror_flips_x_around_one():
    coords = np.array(
        [[0.3, 0.5, 0.0], [0.7, 0.5, 0.0]],
        dtype=np.float32,
    )
    out = _mirror(coords)
    np.testing.assert_allclose(out[:, 0], [0.7, 0.3])


def test_mirror_leaves_y_and_z_unchanged():
    coords = np.array(
        [[0.3, 0.5, 0.1], [0.7, 0.4, -0.2]],
        dtype=np.float32,
    )
    out = _mirror(coords)
    np.testing.assert_allclose(out[:, 1:], coords[:, 1:])


def test_mirror_does_not_mutate_input():
    coords = _synthetic_coords()
    before = coords.copy()
    _mirror(coords)
    np.testing.assert_array_equal(coords, before)
