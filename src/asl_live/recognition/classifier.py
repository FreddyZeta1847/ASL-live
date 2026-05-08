"""Runtime classifier wrapping the exported ONNX MLP.

Phase-2 commit 6 (last). Loads ``models/mlp.onnx`` + ``models/label_map.json``
lazily on the first ``predict`` call and answers per-frame queries from
the live recognizer (phase 3).

Design notes
------------
- **Lives in ``recognition/``, not ``train/``** — this code ships to the
  Pi; training does not. The split keeps the Pi install dependency-free
  of torch / onnx (only ``onnxruntime`` is needed at runtime).
- **Lazy loading.** Construction is cheap; the (~1 s) onnxruntime
  session creation only happens on the first predict. Lets callers
  instantiate ``Classifier()`` at module import without paying the cost.
- **Single-frame contract.** ``predict(vec_63)`` takes a 1-D ``(63,)``
  vector. The recognizer worker calls it once per video frame; batched
  inference is supported by the underlying ONNX (dynamic batch axis)
  but not exposed here — keep the API tight to the runtime use case.
- **Softmax happens here.** The trained model returns raw logits; this
  wrapper applies a numerically-stable softmax to produce a confidence
  in [0, 1] that the debounce layer (``feature-4-debounce.md``) can
  threshold against ``MIN_CONF``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from asl_live.config import LANDMARK_FEATURES, MODELS_DIR


DEFAULT_MODEL_PATH = MODELS_DIR / "mlp.onnx"
DEFAULT_LABEL_MAP_PATH = MODELS_DIR / "label_map.json"


class Classifier:
    """Trained-model wrapper. Construction is cheap; loading is lazy."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        label_map_path: Optional[Path] = None,
    ) -> None:
        self.model_path = (
            Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        )
        self.label_map_path = (
            Path(label_map_path) if label_map_path is not None
            else DEFAULT_LABEL_MAP_PATH
        )
        self._session = None  # type: ignore[var-annotated]
        self._label_map: Optional[dict[int, str]] = None
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        """``True`` once the ONNX session and label map have been read from disk."""
        return self._session is not None

    def predict(self, landmarks: np.ndarray) -> tuple[str, float]:
        """Single-frame classification. Returns ``(label, confidence)``.

        ``landmarks`` must be a 1-D ``(63,)`` numpy array — the canonical
        normalized vector produced by ``LandmarkExtractor.extract``.
        Confidence is the softmax probability of the top-1 class, always
        in ``[0, 1]``.
        """
        if landmarks.ndim != 1 or landmarks.shape[0] != LANDMARK_FEATURES:
            raise ValueError(
                f"expected (63,) vector, got shape {tuple(landmarks.shape)}"
            )

        self._ensure_loaded()

        x = landmarks.astype(np.float32, copy=False).reshape(1, LANDMARK_FEATURES)
        logits = self._session.run([self._output_name], {self._input_name: x})[0]
        # logits shape is (1, num_classes); drop the batch dim.
        probs = _softmax(logits[0])

        idx = int(np.argmax(probs))
        assert self._label_map is not None  # populated by _ensure_loaded
        return self._label_map[idx], float(probs[idx])

    # ----- internals --------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """First-call setup: open the ONNX session and read the label map."""
        if self._session is not None:
            return

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                f"Run `python -m asl_live.train.train_mlp` first."
            )
        if not self.label_map_path.is_file():
            raise FileNotFoundError(
                f"Label map not found at {self.label_map_path}. "
                f"Should be written next to mlp.onnx by training."
            )

        # Lazy import: keeps Classifier importable without onnxruntime
        # installed (useful for offline doc generation, etc.).
        import onnxruntime as ort  # noqa: WPS433

        self._session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        raw = json.loads(self.label_map_path.read_text())
        # JSON object keys are always strings; cast back to int.
        self._label_map = {int(k): v for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Softmax (numpy, numerically stable)
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax over the last axis of a 1-D array."""
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()
