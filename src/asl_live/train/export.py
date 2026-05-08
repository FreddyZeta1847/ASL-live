"""Export trained MLP to ONNX + label_map.json + training_report.json.

Phase-2 commit 5. Writes three artifacts per run under
``models/runs/<utc-ts>/`` and copies the runtime-needed pair
(``mlp.onnx``, ``label_map.json``) to flat paths under ``models/`` so
the Pi only ever reads two stable filenames.

ONNX export specifics (per feature-3 §3.8):
- Opset 17.
- Dynamic batch axis on dim 0 (``input_names=['landmarks']``,
  ``output_names=['logits']``). The runtime feeds one frame at a time
  but batched inference stays available.
- Constant folding enabled (small wins on a tiny model, no harm).

Run history is timestamped — see ``feature-3-classifier.md`` §3.10 and
the phase-2 plan for the artifact-storage decision rationale.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from asl_live.config import LANDMARK_FEATURES, MODELS_DIR


ONNX_OPSET = 17


# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------


def utc_timestamp() -> str:
    """Filesystem-safe ISO-like timestamp for run directories."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def git_sha() -> Optional[str]:
    """Best-effort current commit SHA; ``None`` if git is unavailable or fails."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


# ---------------------------------------------------------------------------
# ONNX
# ---------------------------------------------------------------------------


def export_onnx(
    model: torch.nn.Module,
    output_path: Path,
    *,
    in_features: int = LANDMARK_FEATURES,
) -> None:
    """Export a trained model to ONNX with a dynamic batch axis.

    Caller is responsible for putting the model into ``eval()`` mode and
    loading the best-checkpoint state dict before calling this.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, in_features, dtype=torch.float32)
    # ``dynamo=False`` selects the legacy tracing exporter, which does
    # not require the ``onnxscript`` optional dependency. Our MLP is
    # trivially traceable so the dynamo-based path buys us nothing.
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["landmarks"],
        output_names=["logits"],
        dynamic_axes={"landmarks": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        dynamo=False,
    )


# ---------------------------------------------------------------------------
# JSON sidecars
# ---------------------------------------------------------------------------


def write_label_map(label_map: dict[int, str], output_path: Path) -> None:
    """Write the integer→class-name lookup as JSON.

    Keys are stringified because the JSON spec doesn't allow integer
    object keys; the runtime should convert back when loading.
    """
    serializable = {str(k): v for k, v in label_map.items()}
    output_path.write_text(json.dumps(serializable, indent=2, sort_keys=True))


def write_training_report(
    output_path: Path,
    *,
    hyperparams: dict[str, Any],
    dataset_stats: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    """Write a per-run summary: timestamp, git SHA, hyperparams, dataset, metrics.

    Numpy scalars and arrays are coerced to Python primitives so the
    file is always readable with stdlib ``json.load``.
    """
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "hyperparams": hyperparams,
        "dataset_stats": dataset_stats,
        "metrics": metrics,
    }
    output_path.write_text(json.dumps(report, indent=2, default=_jsonable))


def _jsonable(obj: Any) -> Any:
    """Default fallback for json.dumps — coerce numpy types to Python primitives."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ---------------------------------------------------------------------------
# Top-level: write a full run
# ---------------------------------------------------------------------------


def export_run(
    model: torch.nn.Module,
    label_map: dict[int, str],
    *,
    hyperparams: dict[str, Any],
    dataset_stats: dict[str, Any],
    metrics: dict[str, Any],
    runs_root: Optional[Path] = None,
    latest_dir: Optional[Path] = None,
) -> Path:
    """Write the three artifacts for one training run.

    Returns the absolute path to the timestamped run directory.

    Layout::

        models/runs/<utc-ts>/
            mlp.onnx
            label_map.json
            training_report.json
        models/
            mlp.onnx              (copy of latest run)
            label_map.json        (copy of latest run)
    """
    runs_root = runs_root if runs_root is not None else MODELS_DIR / "runs"
    latest_dir = latest_dir if latest_dir is not None else MODELS_DIR

    timestamp = utc_timestamp()
    run_dir = runs_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = run_dir / "mlp.onnx"
    label_path = run_dir / "label_map.json"
    report_path = run_dir / "training_report.json"

    export_onnx(model, onnx_path)
    write_label_map(label_map, label_path)
    write_training_report(
        report_path,
        hyperparams=hyperparams,
        dataset_stats=dataset_stats,
        metrics=metrics,
    )

    # Mirror runtime artifacts to flat paths so the Pi only needs two stable filenames.
    latest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, latest_dir / "mlp.onnx")
    shutil.copy2(label_path, latest_dir / "label_map.json")

    return run_dir
