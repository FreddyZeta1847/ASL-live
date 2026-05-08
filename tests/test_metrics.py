"""Unit tests for confusion-matrix-based metrics.

Validates all metric helpers against ``sklearn.metrics`` ground truth on
synthetic inputs. No real model or data required.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
    f1_score as sk_f1_score,
)

from asl_live.train.metrics import (
    check_acceptance_bar,
    confusion_matrix,
    format_confusion_matrix,
    macro_f1,
    per_class_f1,
)


# ---------------------------------------------------------------------------
# confusion_matrix
# ---------------------------------------------------------------------------


def test_confusion_matrix_shape_and_dtype():
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 2, 2, 1, 0])
    cm = confusion_matrix(y_true, y_pred, n_classes=3)
    assert cm.shape == (3, 3)
    assert cm.dtype == np.int64


def test_confusion_matrix_matches_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, size=200)
    y_pred = rng.integers(0, 5, size=200)
    ours = confusion_matrix(y_true, y_pred, n_classes=5)
    theirs = sk_confusion_matrix(y_true, y_pred, labels=list(range(5)))
    np.testing.assert_array_equal(ours, theirs)


def test_confusion_matrix_perfect_predictions_is_diagonal():
    y = np.array([0, 1, 2, 0, 1, 2])
    cm = confusion_matrix(y, y, n_classes=3)
    assert np.array_equal(cm, np.diag([2, 2, 2]))


def test_confusion_matrix_row_totals_match_actual_counts():
    y_true = np.array([0, 0, 0, 1, 2, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])
    cm = confusion_matrix(y_true, y_pred, n_classes=3)
    np.testing.assert_array_equal(cm.sum(axis=1), [3, 1, 2])


def test_confusion_matrix_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        confusion_matrix(np.array([0, 1]), np.array([0]), n_classes=2)


def test_confusion_matrix_out_of_range_label_raises():
    with pytest.raises(ValueError):
        confusion_matrix(np.array([0, 5]), np.array([0, 0]), n_classes=3)


def test_confusion_matrix_empty_input_raises():
    with pytest.raises(ValueError):
        confusion_matrix(np.array([]), np.array([]), n_classes=3)


# ---------------------------------------------------------------------------
# per_class_f1 / macro_f1
# ---------------------------------------------------------------------------


def test_per_class_f1_keys_match_classes():
    cm = np.eye(3, dtype=np.int64) * 10
    f1 = per_class_f1(cm, classes=("A", "B", "C"))
    assert set(f1.keys()) == {"A", "B", "C"}
    assert all(score == pytest.approx(1.0) for score in f1.values())


def test_macro_f1_perfect_is_one():
    cm = np.eye(4, dtype=np.int64) * 25
    assert macro_f1(cm) == pytest.approx(1.0)


def test_macro_f1_all_wrong_is_zero():
    """Every prediction goes to one wrong class -> per-class F1 = 0 for all."""
    cm = np.zeros((3, 3), dtype=np.int64)
    cm[0, 1] = 10  # actual A, predicted B
    cm[1, 2] = 10  # actual B, predicted C
    cm[2, 0] = 10  # actual C, predicted A
    # TP = 0 for every class -> F1 = 0 for every class.
    assert macro_f1(cm) == pytest.approx(0.0)


def test_macro_f1_matches_sklearn():
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 6, size=500)
    y_pred = rng.integers(0, 6, size=500)
    cm = confusion_matrix(y_true, y_pred, n_classes=6)
    ours = macro_f1(cm)
    theirs = sk_f1_score(y_true, y_pred, average="macro", labels=list(range(6)), zero_division=0)
    assert ours == pytest.approx(theirs, abs=1e-9)


def test_per_class_f1_matches_sklearn_per_label():
    rng = np.random.default_rng(11)
    y_true = rng.integers(0, 4, size=300)
    y_pred = rng.integers(0, 4, size=300)
    cm = confusion_matrix(y_true, y_pred, n_classes=4)
    ours = per_class_f1(cm, classes=("A", "B", "C", "D"))
    theirs = sk_f1_score(y_true, y_pred, average=None, labels=list(range(4)), zero_division=0)
    for name, expected in zip(("A", "B", "C", "D"), theirs):
        assert ours[name] == pytest.approx(float(expected), abs=1e-9)


def test_zero_predicted_class_gives_f1_zero():
    """Class C is never predicted -> precision undefined -> F1 = 0 for C."""
    cm = np.array(
        [
            [10, 0, 0],
            [0, 10, 0],
            [0, 5, 5],  # actual C is misclassified, never predicted as C
        ],
        dtype=np.int64,
    )
    f1 = per_class_f1(cm, classes=("A", "B", "C"))
    # C: TP=5, but col-total for C is 5 (only the diagonal contributes); wait — let me
    # rebuild: column for C is cm[:, 2] = [0, 0, 5], sum=5. TP=cm[2,2]=5, so precision=1.0.
    # row for C: [0,5,5], sum=10. recall = 5/10 = 0.5. F1 = 2*1*0.5/(1+0.5) = 0.667.
    # That's NOT zero. Let me redesign — make C truly never predicted.
    cm2 = np.array(
        [
            [10, 0, 0],
            [0, 10, 0],
            [5, 5, 0],  # actual C never predicted as C; col-total for C = 0
        ],
        dtype=np.int64,
    )
    f1 = per_class_f1(cm2, classes=("A", "B", "C"))
    assert f1["C"] == pytest.approx(0.0)


def test_zero_actual_class_gives_f1_zero():
    """Class C has no actual samples -> recall undefined -> F1 = 0 for C."""
    cm = np.array(
        [
            [10, 0, 0],
            [0, 10, 0],
            [0, 0, 0],  # row C is all zero
        ],
        dtype=np.int64,
    )
    f1 = per_class_f1(cm, classes=("A", "B", "C"))
    assert f1["C"] == pytest.approx(0.0)


def test_per_class_f1_classes_length_must_match_cm():
    cm = np.eye(3, dtype=np.int64)
    with pytest.raises(ValueError):
        per_class_f1(cm, classes=("A", "B"))


# ---------------------------------------------------------------------------
# Acceptance bar
# ---------------------------------------------------------------------------


def test_acceptance_perfect_passes():
    cm = np.eye(4, dtype=np.int64) * 100
    passed, failures = check_acceptance_bar(cm, classes=("A", "B", "C", "D"))
    assert passed is True
    assert failures == []


def test_acceptance_low_macro_f1_fails():
    """Random-ish confusion matrix: macro-F1 well under 0.95."""
    cm = np.full((3, 3), 10, dtype=np.int64)  # uniformly distributed predictions
    passed, failures = check_acceptance_bar(cm, classes=("A", "B", "C"))
    assert passed is False
    assert any("Macro-F1" in f for f in failures)


def test_acceptance_off_diagonal_fail_names_specific_pair():
    cm = np.eye(3, dtype=np.int64) * 100
    cm[2, 0] = 5   # actual C predicted A: 5/105 = 4.76% > 2%
    cm[2, 2] = 100  # row total for C = 105
    passed, failures = check_acceptance_bar(cm, classes=("A", "B", "C"))
    assert passed is False
    assert any("Actual C -> Predicted A" in f for f in failures)


def test_acceptance_collects_all_failures_not_just_first():
    """Two off-diagonal violations -> both reported."""
    cm = np.eye(3, dtype=np.int64) * 100
    cm[0, 1] = 5
    cm[2, 0] = 5
    passed, failures = check_acceptance_bar(cm, classes=("A", "B", "C"))
    assert passed is False
    assert any("Actual A -> Predicted B" in f for f in failures)
    assert any("Actual C -> Predicted A" in f for f in failures)


def test_acceptance_threshold_overrides_take_effect():
    cm = np.eye(3, dtype=np.int64) * 100
    cm[0, 1] = 1  # 1/101 = ~1% — under default 2% bar
    passed, _ = check_acceptance_bar(cm, classes=("A", "B", "C"))
    assert passed is True

    # Tighten the off-diagonal bar to 0.5%; same matrix should now fail.
    passed, failures = check_acceptance_bar(
        cm, classes=("A", "B", "C"), off_diagonal_threshold=0.005
    )
    assert passed is False
    assert any("Actual A -> Predicted B" in f for f in failures)


# ---------------------------------------------------------------------------
# format_confusion_matrix
# ---------------------------------------------------------------------------


def test_format_confusion_matrix_includes_class_labels():
    cm = np.array([[1, 2], [3, 4]], dtype=np.int64)
    text = format_confusion_matrix(cm, classes=("A", "B"))
    assert "A" in text and "B" in text
    assert "1" in text and "4" in text


def test_format_confusion_matrix_shape_mismatch_raises():
    cm = np.eye(3, dtype=np.int64)
    with pytest.raises(ValueError):
        format_confusion_matrix(cm, classes=("A", "B"))
