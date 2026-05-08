"""Classification metrics derived from a confusion matrix.

Phase-2 commit 4. All metrics — precision, recall, per-class F1,
macro-F1 — fall out of one (n_classes, n_classes) confusion matrix
via row/column arithmetic. The exact formulas and a worked example
live in ``feature-3-classifier.md`` §3.6.

Public API
----------
- ``confusion_matrix``        — (n, n) int64 array, rows=actual, cols=predicted.
- ``per_class_f1``            — ``{class_name: float}`` for each class.
- ``macro_f1``                — unweighted mean of per-class F1.
- ``check_acceptance_bar``    — phase 2 -> phase 3 gate per feature-3 §3.6.
- ``format_confusion_matrix`` — pretty-print helper for the post-training report.

Edge cases (matching sklearn ``zero_division=0`` semantics):
- Class with zero predictions  → precision undefined → F1 = 0.
- Class with zero actuals      → recall undefined    → F1 = 0.
- Both zero                    → F1 = 0.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Build the ``(n_classes, n_classes)`` confusion matrix.

    Convention: rows = actual class, columns = predicted class. So
    ``cm[i, j]`` counts samples whose true label is ``i`` and whose
    predicted label is ``j``. Diagonal cells are correct predictions;
    off-diagonal cells are mistakes broken down by exactly which two
    classes were confused.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape; got {y_true.shape} vs {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("y_true is empty")
    if (y_true < 0).any() or (y_true >= n_classes).any():
        raise ValueError(f"y_true contains labels outside [0, {n_classes})")
    if (y_pred < 0).any() or (y_pred >= n_classes).any():
        raise ValueError(f"y_pred contains labels outside [0, {n_classes})")

    # Vectorized binning: each (true, pred) pair becomes a single integer
    # in [0, n_classes^2), then bincount + reshape gives the matrix.
    flat = y_true.astype(np.int64) * n_classes + y_pred.astype(np.int64)
    counts = np.bincount(flat, minlength=n_classes * n_classes)
    return counts.reshape(n_classes, n_classes)


# ---------------------------------------------------------------------------
# F1
# ---------------------------------------------------------------------------


def _f1_from_cm(cm: np.ndarray) -> np.ndarray:
    """Per-class F1 as a 1-D array of length ``n_classes``.

    For class ``c``:
        TP = cm[c, c]
        row_total = sum(cm[c, :])    # all actual c
        col_total = sum(cm[:, c])    # all predicted c
        precision = TP / col_total
        recall    = TP / row_total
        F1        = 2*P*R / (P+R)

    Uses ``zero_division=0`` semantics: if any denominator is 0, F1 = 0
    for that class.
    """
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"cm must be (n, n); got shape {cm.shape}")

    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    row_totals = cm.sum(axis=1)
    col_totals = cm.sum(axis=0)

    # Suppress divide-by-zero warnings; we replace NaN with 0 below.
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(col_totals > 0, tp / col_totals, 0.0)
        recall = np.where(row_totals > 0, tp / row_totals, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)

    return f1


def per_class_f1(cm: np.ndarray, classes: Iterable[str]) -> dict[str, float]:
    """Return ``{class_name: f1_score}`` for each class."""
    classes = tuple(classes)
    f1 = _f1_from_cm(cm)
    if len(classes) != f1.shape[0]:
        raise ValueError(
            f"classes length ({len(classes)}) does not match cm size ({f1.shape[0]})"
        )
    return {name: float(score) for name, score in zip(classes, f1)}


def macro_f1(cm: np.ndarray) -> float:
    """Unweighted mean of per-class F1 scores."""
    return float(_f1_from_cm(cm).mean())


# ---------------------------------------------------------------------------
# Acceptance bar (feature-3 §3.6)
# ---------------------------------------------------------------------------


# Defaults from feature-3 §3.6.
MACRO_F1_THRESHOLD = 0.95
OFF_DIAGONAL_THRESHOLD = 0.02  # 2 % of the row total


def check_acceptance_bar(
    cm: np.ndarray,
    classes: Iterable[str],
    *,
    macro_f1_threshold: float = MACRO_F1_THRESHOLD,
    off_diagonal_threshold: float = OFF_DIAGONAL_THRESHOLD,
) -> tuple[bool, list[str]]:
    """Phase 2 -> phase 3 acceptance gate.

    Both bars must hold:
      1. macro-F1 >= ``macro_f1_threshold`` (default 0.95)
      2. No off-diagonal cell exceeds ``off_diagonal_threshold`` of its
         row total (default 2 %).

    Returns ``(passed, failures)``. ``failures`` is a list of human-readable
    strings — empty when ``passed`` is True. The specific failures let the
    caller print exactly which rows blocked the gate.
    """
    classes = tuple(classes)
    if cm.shape != (len(classes), len(classes)):
        raise ValueError(
            f"cm shape {cm.shape} does not match classes length {len(classes)}"
        )

    failures: list[str] = []

    score = macro_f1(cm)
    if score < macro_f1_threshold:
        failures.append(
            f"Macro-F1 {score:.4f} < {macro_f1_threshold:.4f} bar"
        )

    cm_f = cm.astype(np.float64)
    row_totals = cm_f.sum(axis=1)
    n = len(classes)
    for i in range(n):
        if row_totals[i] == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            ratio = cm_f[i, j] / row_totals[i]
            if ratio > off_diagonal_threshold:
                failures.append(
                    f"Actual {classes[i]} -> Predicted {classes[j]}: "
                    f"{ratio * 100:.2f}% > {off_diagonal_threshold * 100:.2f}% bar "
                    f"({int(cm[i, j])} / {int(row_totals[i])})"
                )

    return (len(failures) == 0), failures


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------


def format_confusion_matrix(
    cm: np.ndarray,
    classes: Iterable[str],
    *,
    cell_width: int = 5,
) -> str:
    """Render a confusion matrix as a fixed-width text table.

    Header lists predicted classes; each row prefixed with the actual
    class name. Useful for printing alongside the per-class F1 table.
    """
    classes = tuple(classes)
    if cm.shape != (len(classes), len(classes)):
        raise ValueError(
            f"cm shape {cm.shape} does not match classes length {len(classes)}"
        )

    label_width = max(len(c) for c in classes)
    lines: list[str] = []

    header_cells = " ".join(f"{c:>{cell_width}}" for c in classes)
    lines.append(" " * (label_width + 2) + header_cells)

    for i, cls_name in enumerate(classes):
        row_cells = " ".join(f"{int(v):>{cell_width}}" for v in cm[i])
        lines.append(f"{cls_name:>{label_width}}  {row_cells}")

    return "\n".join(lines)
