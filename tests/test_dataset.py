"""Unit tests for the training Dataset, label-map, weights, and split.

Synthetic data only: each test that needs a dataset uses a tmpdir with
fake .npy files — no MediaPipe, no real Kaggle data, sub-second suite.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from asl_live.config import CLASSES, LANDMARK_FEATURES
from asl_live.train.dataset import (
    LandmarkDataset,
    build_label_map,
    compute_class_weights,
    make_splits,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_fake_landmarks(root: Path, samples_per_class: int) -> None:
    """Populate ``root`` with one folder per class, each containing fake .npy files."""
    rng = np.random.default_rng(0)
    for cls in CLASSES:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(samples_per_class):
            vec = rng.standard_normal(LANDMARK_FEATURES).astype(np.float32)
            np.save(cls_dir / f"sample_{i:04d}.npy", vec)


# Module-scoped fixtures: building a fake dataset is hundreds of file ops,
# slow on Windows NTFS. Build once per test module, share across tests.
# Tests that mutate files use their own function-scoped tmp_path instead.


@pytest.fixture(scope="module")
def fake_dataset_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic balanced landmark tree: 26 classes × 20 samples each."""
    root = tmp_path_factory.mktemp("balanced")
    _make_fake_landmarks(root, samples_per_class=20)
    return root


@pytest.fixture(scope="module")
def fake_imbalanced_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Imbalanced tree: every class except SPACE has 20, SPACE has 10."""
    root = tmp_path_factory.mktemp("imbalanced")
    _make_fake_landmarks(root, samples_per_class=20)
    space_dir = root / "SPACE"
    for f in sorted(space_dir.glob("*.npy"))[10:]:
        f.unlink()
    return root


# ---------------------------------------------------------------------------
# build_label_map
# ---------------------------------------------------------------------------


def test_label_map_has_26_entries():
    lm = build_label_map()
    assert len(lm) == 26


def test_label_map_alphabetical_letters_then_controls():
    lm = build_label_map()
    assert lm[0] == "A"
    assert lm[8] == "I"
    assert lm[9] == "K"  # J is skipped
    assert lm[23] == "Y"
    assert lm[24] == "SPACE"
    assert lm[25] == "DELETE"


def test_label_map_deterministic_across_calls():
    assert build_label_map() == build_label_map()


# ---------------------------------------------------------------------------
# LandmarkDataset
# ---------------------------------------------------------------------------


def test_dataset_loads_all_classes(fake_dataset_root: Path) -> None:
    ds = LandmarkDataset(root=fake_dataset_root)
    assert len(ds) == 26 * 20


def test_dataset_getitem_shape_and_label_type(fake_dataset_root: Path) -> None:
    ds = LandmarkDataset(root=fake_dataset_root)
    vec, label = ds[0]
    assert isinstance(vec, torch.Tensor)
    assert vec.shape == (LANDMARK_FEATURES,)
    assert vec.dtype == torch.float32
    assert isinstance(label, int)
    assert 0 <= label < 26


def test_dataset_class_counts(fake_imbalanced_root: Path) -> None:
    ds = LandmarkDataset(root=fake_imbalanced_root)
    counts = ds.class_counts()
    assert counts.shape == (26,)
    # SPACE is index 24 per CLASSES ordering; the rest are 20.
    expected = np.full(26, 20)
    expected[24] = 10
    np.testing.assert_array_equal(counts, expected)


def test_dataset_missing_class_folder_raises(tmp_path: Path) -> None:
    # Create only A; the constructor should fail on the next class.
    (tmp_path / "A").mkdir()
    np.save(tmp_path / "A" / "sample_0000.npy", np.zeros(LANDMARK_FEATURES, dtype=np.float32))
    with pytest.raises(FileNotFoundError):
        LandmarkDataset(root=tmp_path)


def test_dataset_empty_class_folder_raises(tmp_path: Path) -> None:
    for cls in CLASSES:
        (tmp_path / cls).mkdir()
    # All folders exist but no files → first class scan fails.
    with pytest.raises(RuntimeError):
        LandmarkDataset(root=tmp_path)


def test_dataset_bad_vector_shape_raises(tmp_path: Path) -> None:
    _make_fake_landmarks(tmp_path, samples_per_class=2)
    # Overwrite one file with the wrong shape.
    bad_path = next((tmp_path / "A").glob("*.npy"))
    np.save(bad_path, np.zeros(42, dtype=np.float32))
    with pytest.raises(ValueError):
        LandmarkDataset(root=tmp_path, use_cache=False)


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


def test_cache_is_created_on_first_load(tmp_path: Path) -> None:
    _make_fake_landmarks(tmp_path, samples_per_class=4)
    cache = tmp_path / "_cache.npz"
    LandmarkDataset(root=tmp_path, cache_path=cache)
    assert cache.is_file()


def test_cache_reload_returns_same_data(tmp_path: Path) -> None:
    _make_fake_landmarks(tmp_path, samples_per_class=4)
    cache = tmp_path / "_cache.npz"
    ds_fresh = LandmarkDataset(root=tmp_path, cache_path=cache)
    ds_cached = LandmarkDataset(root=tmp_path, cache_path=cache)
    assert torch.equal(ds_fresh._X, ds_cached._X)
    assert torch.equal(ds_fresh._y, ds_cached._y)


def test_cache_invalidated_on_file_addition(tmp_path: Path) -> None:
    _make_fake_landmarks(tmp_path, samples_per_class=4)
    cache = tmp_path / "_cache.npz"
    ds_before = LandmarkDataset(root=tmp_path, cache_path=cache)
    n_before = len(ds_before)
    np.save(tmp_path / "A" / "new_sample.npy", np.zeros(LANDMARK_FEATURES, dtype=np.float32))
    ds_after = LandmarkDataset(root=tmp_path, cache_path=cache)
    assert len(ds_after) == n_before + 1


def test_use_cache_false_does_not_create_cache(tmp_path: Path) -> None:
    _make_fake_landmarks(tmp_path, samples_per_class=3)
    cache = tmp_path / "_cache.npz"
    LandmarkDataset(root=tmp_path, cache_path=cache, use_cache=False)
    assert not cache.exists()


# ---------------------------------------------------------------------------
# compute_class_weights
# ---------------------------------------------------------------------------


def test_class_weights_match_formula():
    # 3 classes: 100, 200, 700. N=1000, K=3. w_c = N / (K * n_c)
    counts = np.array([100, 200, 700])
    weights = compute_class_weights(counts)
    expected = np.array(
        [1000 / (3 * 100), 1000 / (3 * 200), 1000 / (3 * 700)],
        dtype=np.float32,
    )
    assert torch.allclose(weights, torch.from_numpy(expected), atol=1e-6)


def test_class_weights_smaller_class_heavier():
    weights = compute_class_weights(np.array([100, 200, 700]))
    assert weights[0] > weights[1] > weights[2]


def test_class_weights_balanced_dataset_is_uniform():
    weights = compute_class_weights(np.array([500, 500, 500, 500]))
    assert torch.allclose(weights, torch.ones(4))


def test_class_weights_zero_count_raises():
    with pytest.raises(ValueError, match="zero"):
        compute_class_weights(np.array([10, 0, 5]))


def test_class_weights_two_dim_input_raises():
    with pytest.raises(ValueError):
        compute_class_weights(np.array([[10, 20], [30, 40]]))


# ---------------------------------------------------------------------------
# make_splits
# ---------------------------------------------------------------------------


def test_splits_sum_to_dataset_size(fake_dataset_root: Path) -> None:
    ds = LandmarkDataset(root=fake_dataset_root)
    train, val, test = make_splits(ds)
    assert len(train) + len(val) + len(test) == len(ds)


def test_splits_no_overlap(fake_dataset_root: Path) -> None:
    ds = LandmarkDataset(root=fake_dataset_root)
    train, val, test = make_splits(ds)
    train_set, val_set, test_set = set(train), set(val), set(test)
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)


def test_splits_cover_full_index_range(fake_dataset_root: Path) -> None:
    ds = LandmarkDataset(root=fake_dataset_root)
    train, val, test = make_splits(ds)
    assert set(train) | set(val) | set(test) == set(range(len(ds)))


def test_splits_stratified_per_class(fake_dataset_root: Path) -> None:
    """With 20 samples/class and 80/10/10, every class should split exactly 16/2/2."""
    ds = LandmarkDataset(root=fake_dataset_root)
    train, val, test = make_splits(ds)
    labels = ds.labels
    for cls_idx in range(26):
        n_train = int(np.sum(labels[train] == cls_idx))
        n_val = int(np.sum(labels[val] == cls_idx))
        n_test = int(np.sum(labels[test] == cls_idx))
        assert n_train == 16, f"class {cls_idx}: expected 16 train, got {n_train}"
        assert n_val == 2, f"class {cls_idx}: expected 2 val, got {n_val}"
        assert n_test == 2, f"class {cls_idx}: expected 2 test, got {n_test}"


def test_splits_deterministic_same_seed(fake_dataset_root: Path) -> None:
    ds = LandmarkDataset(root=fake_dataset_root)
    s1 = make_splits(ds, seed=42)
    s2 = make_splits(ds, seed=42)
    assert s1 == s2


def test_splits_different_seed_different_assignment(fake_dataset_root: Path) -> None:
    ds = LandmarkDataset(root=fake_dataset_root)
    train_42, _, _ = make_splits(ds, seed=42)
    train_43, _, _ = make_splits(ds, seed=43)
    # Sizes are identical (deterministic from ratios) but the membership differs.
    assert len(train_42) == len(train_43)
    assert sorted(train_42) != sorted(train_43)


def test_splits_imbalanced_classes(fake_imbalanced_root: Path) -> None:
    """SPACE has 10 samples → 80/10/10 = 8/1/1."""
    ds = LandmarkDataset(root=fake_imbalanced_root)
    train, val, test = make_splits(ds)
    labels = ds.labels
    space_idx = 24
    n_train = int(np.sum(labels[train] == space_idx))
    n_val = int(np.sum(labels[val] == space_idx))
    n_test = int(np.sum(labels[test] == space_idx))
    assert n_train == 8
    assert n_val == 1
    assert n_test == 1


def test_splits_bad_ratios_raise(fake_dataset_root: Path) -> None:
    ds = LandmarkDataset(root=fake_dataset_root)
    with pytest.raises(ValueError, match="sum"):
        make_splits(ds, ratios=(0.7, 0.1, 0.1))
    with pytest.raises(ValueError):
        make_splits(ds, ratios=(0.9, 0.2, -0.1))
