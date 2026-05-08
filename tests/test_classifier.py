"""Unit tests for the runtime Classifier wrapper.

Builds a tiny dummy ONNX inline (no real training) so the pipeline can
be tested end-to-end without any phase-2 training output on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from asl_live.config import LANDMARK_FEATURES
from asl_live.recognition.classifier import Classifier, _softmax
from asl_live.train.export import export_onnx, write_label_map


# ---------------------------------------------------------------------------
# Dummy model fixtures
# ---------------------------------------------------------------------------


class _FixedTopOne(nn.Module):
    """Always emits a logit vector with index 0 strongly preferred."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output shape (B, num_classes). Class 0 gets a +10 bias, others 0.
        batch = x.shape[0]
        logits = torch.zeros(batch, self.num_classes)
        logits[:, 0] = 10.0
        return logits


@pytest.fixture
def dummy_model_files(tmp_path: Path) -> tuple[Path, Path]:
    """Tiny ONNX with a deterministic predicted-class-0 output + matching label map."""
    onnx_path = tmp_path / "mlp.onnx"
    label_path = tmp_path / "label_map.json"
    export_onnx(_FixedTopOne(num_classes=3), onnx_path, in_features=LANDMARK_FEATURES)
    write_label_map({0: "A", 1: "B", 2: "SPACE"}, label_path)
    return onnx_path, label_path


# ---------------------------------------------------------------------------
# Construction + lazy loading
# ---------------------------------------------------------------------------


def test_construction_does_not_load(dummy_model_files: tuple[Path, Path]) -> None:
    onnx_path, label_path = dummy_model_files
    clf = Classifier(model_path=onnx_path, label_map_path=label_path)
    assert clf.is_loaded is False


def test_first_predict_loads_session(dummy_model_files: tuple[Path, Path]) -> None:
    onnx_path, label_path = dummy_model_files
    clf = Classifier(model_path=onnx_path, label_map_path=label_path)
    clf.predict(np.zeros(LANDMARK_FEATURES, dtype=np.float32))
    assert clf.is_loaded is True


# ---------------------------------------------------------------------------
# predict — contract
# ---------------------------------------------------------------------------


def test_predict_returns_str_and_float(dummy_model_files: tuple[Path, Path]) -> None:
    onnx_path, label_path = dummy_model_files
    clf = Classifier(model_path=onnx_path, label_map_path=label_path)
    label, confidence = clf.predict(np.zeros(LANDMARK_FEATURES, dtype=np.float32))
    assert isinstance(label, str)
    assert isinstance(confidence, float)


def test_predict_confidence_in_unit_interval(dummy_model_files: tuple[Path, Path]) -> None:
    onnx_path, label_path = dummy_model_files
    clf = Classifier(model_path=onnx_path, label_map_path=label_path)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.standard_normal(LANDMARK_FEATURES).astype(np.float32)
        _, conf = clf.predict(x)
        assert 0.0 <= conf <= 1.0


def test_predict_label_is_in_label_map(dummy_model_files: tuple[Path, Path]) -> None:
    onnx_path, label_path = dummy_model_files
    clf = Classifier(model_path=onnx_path, label_map_path=label_path)
    rng = np.random.default_rng(1)
    valid_labels = set(json.loads(label_path.read_text()).values())
    for _ in range(10):
        x = rng.standard_normal(LANDMARK_FEATURES).astype(np.float32)
        label, _ = clf.predict(x)
        assert label in valid_labels


def test_predict_returns_top1_for_fixed_model(
    dummy_model_files: tuple[Path, Path],
) -> None:
    """The dummy model always favors class 0 (label 'A') with very high confidence."""
    onnx_path, label_path = dummy_model_files
    clf = Classifier(model_path=onnx_path, label_map_path=label_path)
    label, confidence = clf.predict(np.zeros(LANDMARK_FEATURES, dtype=np.float32))
    assert label == "A"
    # softmax([10, 0, 0]) ≈ [0.99991, 0.000045, 0.000045]
    assert confidence > 0.999


def test_predict_is_deterministic(dummy_model_files: tuple[Path, Path]) -> None:
    """Same input -> same (label, confidence) on repeated calls."""
    onnx_path, label_path = dummy_model_files
    clf = Classifier(model_path=onnx_path, label_map_path=label_path)
    x = np.full(LANDMARK_FEATURES, 0.5, dtype=np.float32)
    out1 = clf.predict(x)
    out2 = clf.predict(x)
    assert out1 == out2


def test_predict_accepts_float64_input(dummy_model_files: tuple[Path, Path]) -> None:
    """Casting to float32 inside predict means the caller can pass float64."""
    onnx_path, label_path = dummy_model_files
    clf = Classifier(model_path=onnx_path, label_map_path=label_path)
    x = np.zeros(LANDMARK_FEATURES, dtype=np.float64)
    label, _ = clf.predict(x)
    assert label == "A"


# ---------------------------------------------------------------------------
# predict — error contract
# ---------------------------------------------------------------------------


def test_predict_rejects_wrong_shape(dummy_model_files: tuple[Path, Path]) -> None:
    onnx_path, label_path = dummy_model_files
    clf = Classifier(model_path=onnx_path, label_map_path=label_path)
    with pytest.raises(ValueError):
        clf.predict(np.zeros(42, dtype=np.float32))
    with pytest.raises(ValueError):
        clf.predict(np.zeros((1, LANDMARK_FEATURES), dtype=np.float32))


def test_missing_model_file_raises(tmp_path: Path) -> None:
    clf = Classifier(
        model_path=tmp_path / "missing.onnx",
        label_map_path=tmp_path / "missing.json",
    )
    with pytest.raises(FileNotFoundError):
        clf.predict(np.zeros(LANDMARK_FEATURES, dtype=np.float32))


def test_missing_label_map_raises(tmp_path: Path) -> None:
    onnx_path = tmp_path / "mlp.onnx"
    export_onnx(_FixedTopOne(), onnx_path, in_features=LANDMARK_FEATURES)
    clf = Classifier(
        model_path=onnx_path,
        label_map_path=tmp_path / "missing.json",
    )
    with pytest.raises(FileNotFoundError):
        clf.predict(np.zeros(LANDMARK_FEATURES, dtype=np.float32))


# ---------------------------------------------------------------------------
# softmax helper
# ---------------------------------------------------------------------------


def test_softmax_sums_to_one():
    out = _softmax(np.array([1.0, 2.0, 3.0]))
    assert out.sum() == pytest.approx(1.0)
    assert (out >= 0).all() and (out <= 1).all()


def test_softmax_numerically_stable_with_large_logits():
    """Without subtracting the max, exp(1000) overflows to inf."""
    out = _softmax(np.array([1000.0, 1000.0, 1000.0]))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, [1 / 3, 1 / 3, 1 / 3])
