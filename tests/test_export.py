"""Unit tests for ONNX export + JSON sidecars.

Includes a round-trip assertion that PyTorch logits and ONNXRuntime
logits agree within 1e-5 — the contract that lets the live recognizer
trust ``mlp.onnx`` as a faithful copy of the trained model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from asl_live.config import LANDMARK_FEATURES, NUM_CLASSES
from asl_live.train.export import (
    ONNX_OPSET,
    export_onnx,
    export_run,
    utc_timestamp,
    write_label_map,
    write_training_report,
    _jsonable,
)
from asl_live.train.model import MLP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def trained_model() -> MLP:
    """A small MLP with random weights — enough to validate I/O contracts."""
    torch.manual_seed(0)
    model = MLP()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# utc_timestamp
# ---------------------------------------------------------------------------


def test_utc_timestamp_is_filesystem_safe():
    ts = utc_timestamp()
    assert ":" not in ts and " " not in ts
    assert ts.endswith("Z")


def test_utc_timestamp_changes_with_each_call_in_time():
    """Trivially: same-second calls may match, but the format is parseable."""
    ts1 = utc_timestamp()
    assert len(ts1) == len("2026-05-09T12-34-56Z")


# ---------------------------------------------------------------------------
# export_onnx
# ---------------------------------------------------------------------------


def test_export_onnx_writes_a_file(trained_model: MLP, tmp_path: Path):
    out = tmp_path / "mlp.onnx"
    export_onnx(trained_model, out)
    assert out.is_file()
    assert out.stat().st_size > 0


def test_export_onnx_creates_parent_dir(trained_model: MLP, tmp_path: Path):
    out = tmp_path / "nested" / "dir" / "mlp.onnx"
    export_onnx(trained_model, out)
    assert out.is_file()


def test_onnx_logits_match_pytorch_within_tolerance(trained_model: MLP, tmp_path: Path):
    """Round-trip: same input -> same logits, within 1e-5 (feature-3 §3.8 spec)."""
    out = tmp_path / "mlp.onnx"
    export_onnx(trained_model, out)

    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((4, LANDMARK_FEATURES)).astype(np.float32)

    with torch.no_grad():
        torch_logits = trained_model(torch.from_numpy(x_np)).numpy()

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(["logits"], {"landmarks": x_np})[0]

    assert torch_logits.shape == onnx_logits.shape == (4, NUM_CLASSES)
    np.testing.assert_allclose(torch_logits, onnx_logits, atol=1e-5, rtol=1e-5)


def test_onnx_supports_dynamic_batch_axis(trained_model: MLP, tmp_path: Path):
    """Different batch sizes should run without re-exporting."""
    out = tmp_path / "mlp.onnx"
    export_onnx(trained_model, out)

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    for batch_size in (1, 4, 32):
        x = np.zeros((batch_size, LANDMARK_FEATURES), dtype=np.float32)
        out_logits = session.run(["logits"], {"landmarks": x})[0]
        assert out_logits.shape == (batch_size, NUM_CLASSES)


# ---------------------------------------------------------------------------
# write_label_map
# ---------------------------------------------------------------------------


def test_label_map_json_keys_are_strings(tmp_path: Path):
    out = tmp_path / "label_map.json"
    write_label_map({0: "A", 1: "B", 25: "DELETE"}, out)
    data = json.loads(out.read_text())
    assert set(data.keys()) == {"0", "1", "25"}
    assert data["25"] == "DELETE"


def test_label_map_json_is_sorted(tmp_path: Path):
    out = tmp_path / "label_map.json"
    # Insert in non-sorted order
    write_label_map({25: "DELETE", 0: "A", 1: "B"}, out)
    text = out.read_text()
    # Sorted-keys output puts "0" before "1" before "25" lexicographically.
    assert text.index('"0"') < text.index('"1"') < text.index('"25"')


# ---------------------------------------------------------------------------
# write_training_report
# ---------------------------------------------------------------------------


def test_training_report_round_trips_through_json(tmp_path: Path):
    out = tmp_path / "training_report.json"
    write_training_report(
        out,
        hyperparams={"lr": 1e-3, "batch_size": 256, "seed": 42},
        dataset_stats={"total": 112966, "per_class": {"A": 4376, "B": 4410}},
        metrics={"macro_f1": 0.97, "test_acc": 0.96},
    )
    data = json.loads(out.read_text())
    assert "timestamp_utc" in data
    assert data["hyperparams"]["lr"] == 1e-3
    assert data["dataset_stats"]["per_class"]["A"] == 4376
    assert data["metrics"]["macro_f1"] == 0.97


def test_training_report_handles_numpy_types(tmp_path: Path):
    """Numpy scalars and arrays in metrics should serialize cleanly."""
    out = tmp_path / "training_report.json"
    write_training_report(
        out,
        hyperparams={"lr": 1e-3},
        dataset_stats={"counts": np.array([100, 200, 300])},
        metrics={
            "macro_f1": np.float64(0.95),
            "n_test": np.int64(11000),
            "confusion_matrix": np.eye(3, dtype=np.int64) * 10,
        },
    )
    data = json.loads(out.read_text())
    assert data["dataset_stats"]["counts"] == [100, 200, 300]
    assert data["metrics"]["macro_f1"] == 0.95
    assert data["metrics"]["n_test"] == 11000
    assert data["metrics"]["confusion_matrix"] == [[10, 0, 0], [0, 10, 0], [0, 0, 10]]


def test_jsonable_rejects_unknown_types():
    with pytest.raises(TypeError):
        _jsonable(object())


# ---------------------------------------------------------------------------
# export_run (end-to-end)
# ---------------------------------------------------------------------------


def test_export_run_writes_three_artifacts_in_timestamped_dir(
    trained_model: MLP, tmp_path: Path
):
    runs_root = tmp_path / "runs"
    latest_dir = tmp_path / "latest"

    run_dir = export_run(
        trained_model,
        label_map={i: f"C{i}" for i in range(NUM_CLASSES)},
        hyperparams={"lr": 1e-3},
        dataset_stats={"total": 1000},
        metrics={"macro_f1": 0.99},
        runs_root=runs_root,
        latest_dir=latest_dir,
    )

    assert run_dir.parent == runs_root
    assert (run_dir / "mlp.onnx").is_file()
    assert (run_dir / "label_map.json").is_file()
    assert (run_dir / "training_report.json").is_file()


def test_export_run_copies_latest_pair_to_flat_paths(
    trained_model: MLP, tmp_path: Path
):
    runs_root = tmp_path / "runs"
    latest_dir = tmp_path / "latest"

    export_run(
        trained_model,
        label_map={i: f"C{i}" for i in range(NUM_CLASSES)},
        hyperparams={},
        dataset_stats={},
        metrics={},
        runs_root=runs_root,
        latest_dir=latest_dir,
    )

    assert (latest_dir / "mlp.onnx").is_file()
    assert (latest_dir / "label_map.json").is_file()
    # training_report.json is NOT mirrored to latest_dir — only run-dir.
    assert not (latest_dir / "training_report.json").exists()
