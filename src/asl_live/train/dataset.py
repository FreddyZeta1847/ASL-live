"""Dataset loading, label-map, class weights, and stratified split.

Phase 2 / step 1: produce the inputs the training loop in ``train_mlp.py``
will consume. The whole dataset (~28 MB for 113 k vectors) is small enough
to load into RAM up front; we keep one (N, 63) float32 tensor and one
(N,) int64 label tensor and serve indices into them.

Public API
----------
- ``LandmarkDataset``        — PyTorch ``Dataset`` over the on-disk landmarks.
- ``build_label_map``        — ``{int: class_name}`` mirroring ``config.CLASSES``.
- ``compute_class_weights``  — inverse-frequency weights for ``CrossEntropyLoss``.
- ``make_splits``            — stratified 80/10/10 (or arbitrary ratios) split.

Per feature-3 §3.2 (class weighting) and §3.4 (stratified split with
seed=42); the why-we-do-this lives there, the how-we-do-this lives here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from asl_live.config import CLASSES, LANDMARK_FEATURES, LANDMARKS_DIR


# ---------------------------------------------------------------------------
# Label map
# ---------------------------------------------------------------------------


def build_label_map(classes: tuple[str, ...] = CLASSES) -> dict[int, str]:
    """Integer index → class-name lookup, mirroring ``config.CLASSES`` order.

    Used both at training time (to interpret model output) and at deploy
    time (saved next to ``mlp.onnx`` so the runtime knows what each
    output index means).
    """
    return {i: name for i, name in enumerate(classes)}


# ---------------------------------------------------------------------------
# Class weights (feature-3 §3.2)
# ---------------------------------------------------------------------------


def compute_class_weights(counts: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights for class-weighted cross-entropy.

    Formula (feature-3 §3.2):
        w_c = N / (K * n_c)
    where N = total samples, K = number of classes, n_c = samples in class c.

    Smaller classes receive larger weights, so under-represented classes
    (SPACE/DELETE in our dataset) contribute as much to each gradient
    step as the heavily-populated letter classes. Rejects zero-count
    classes loudly — silently weighting them as +inf would corrupt
    training.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1:
        raise ValueError(f"counts must be 1-D, got shape {counts.shape}")
    if (counts <= 0).any():
        zeros = np.where(counts <= 0)[0]
        raise ValueError(f"Classes with zero/negative samples cannot be weighted: {zeros.tolist()}")

    total = counts.sum()
    num_classes = len(counts)
    weights = total / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Stratified split (feature-3 §3.4)
# ---------------------------------------------------------------------------


def make_splits(
    dataset: "LandmarkDataset",
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Stratified split into train / val / test index lists.

    Stratified per feature-3 §3.4: within each class, samples are shuffled
    (deterministic given ``seed``) and split by ``ratios``. This guarantees
    every class is represented in every split — a global random shuffle
    could leave a small class entirely out of test, breaking per-class F1.

    Returns three disjoint lists of indices into ``dataset`` whose union
    equals ``range(len(dataset))``. Rounding remainders go to the test
    set so the train ratio is honored exactly.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0; got {ratios} -> {sum(ratios)}")
    if any(r < 0 for r in ratios):
        raise ValueError(f"ratios must be non-negative; got {ratios}")

    rng = np.random.default_rng(seed)
    labels = dataset.labels  # numpy view, shape (N,)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for cls_idx in range(dataset.num_classes):
        cls_indices = np.where(labels == cls_idx)[0]
        rng.shuffle(cls_indices)

        n = len(cls_indices)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        # Anything left → test. Absorbs rounding remainder cleanly.
        train_idx.extend(cls_indices[:n_train].tolist())
        val_idx.extend(cls_indices[n_train : n_train + n_val].tolist())
        test_idx.extend(cls_indices[n_train + n_val :].tolist())

    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class LandmarkDataset(Dataset):
    """In-memory PyTorch ``Dataset`` over ``data/landmarks/<class>/*.npy``.

    On ``__init__`` it walks every class directory under ``root``, eager-
    loads all ``.npy`` vectors into a single ``(N, 63)`` float32 tensor,
    and records each sample's class index in a ``(N,)`` int64 tensor.
    The dataset is small enough (~28 MB) that this is faster than per-
    sample disk reads on each epoch.

    ``__getitem__`` returns ``(vec_63: torch.Tensor, label: int)``.
    """

    def __init__(
        self,
        root: Path = LANDMARKS_DIR,
        classes: tuple[str, ...] = CLASSES,
    ) -> None:
        self.root = Path(root)
        self.classes = tuple(classes)
        self.num_classes = len(self.classes)
        self.label_map = build_label_map(self.classes)
        self._class_to_idx = {name: i for i, name in enumerate(self.classes)}

        records = self._collect_records()
        self._X, self._y = self._load_records(records)

    # -- private helpers -----------------------------------------------------

    def _collect_records(self) -> list[tuple[Path, int]]:
        """Walk every class directory and gather (file_path, class_idx)."""
        records: list[tuple[Path, int]] = []
        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            if not cls_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing class directory: {cls_dir}. Run phase-1 ingest + "
                    f"collector before training."
                )
            cls_files = sorted(cls_dir.glob("*.npy"))
            if not cls_files:
                raise RuntimeError(f"No .npy files found in {cls_dir}")
            cls_idx = self._class_to_idx[cls_name]
            records.extend((path, cls_idx) for path in cls_files)
        return records

    @staticmethod
    def _load_records(
        records: list[tuple[Path, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Eager-load all records into one (N, 63) float32 tensor."""
        n = len(records)
        X = np.empty((n, LANDMARK_FEATURES), dtype=np.float32)
        y = np.empty(n, dtype=np.int64)
        for i, (path, label) in enumerate(records):
            vec = np.load(path)
            if vec.shape != (LANDMARK_FEATURES,):
                raise ValueError(
                    f"Bad shape in {path}: got {vec.shape}, expected ({LANDMARK_FEATURES},)"
                )
            X[i] = vec
            y[i] = label
        return torch.from_numpy(X), torch.from_numpy(y)

    # -- Dataset protocol ----------------------------------------------------

    def __len__(self) -> int:
        return self._X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # Returns a view of the row, not a copy — DataLoader stacks them.
        return self._X[idx], int(self._y[idx].item())

    # -- public conveniences -------------------------------------------------

    @property
    def labels(self) -> np.ndarray:
        """1-D int array of class indices, length len(self)."""
        return self._y.numpy()

    def class_counts(self) -> np.ndarray:
        """Per-class sample count, shape (num_classes,)."""
        return np.bincount(self.labels, minlength=self.num_classes)
