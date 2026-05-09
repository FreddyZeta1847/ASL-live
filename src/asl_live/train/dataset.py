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

Cache file
----------
Reading 113 k tiny .npy files sequentially on Windows takes ~10 minutes
(the per-file open is throttled by the filesystem / antivirus). After
the first load the dataset is cached as a single ``.npz`` next to
``data/landmarks/`` so subsequent training runs start in <1 s. The cache
manifest is a SHA-1 of the sorted filename list — adding, removing, or
renaming any source file invalidates the cache automatically.
"""
from __future__ import annotations

import hashlib
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
        # Cap n_val to leave at least 0 for test. Without the cap, small-class
        # rounding (e.g. n=5: round(4.0) + round(0.5) = 4 + 1 = 5) can leave
        # the test slice empty, which surfaces later as F1=0 for that class.
        n_val = min(int(round(n * ratios[1])), max(0, n - n_train))
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
        cache_path: Optional[Path] = None,
        use_cache: bool = True,
    ) -> None:
        self.root = Path(root)
        self.classes = tuple(classes)
        self.num_classes = len(self.classes)
        self.label_map = build_label_map(self.classes)
        self._class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self._cache_path = (
            Path(cache_path) if cache_path is not None
            else self.root.parent / "landmarks_cache.npz"
        )

        records = self._collect_records()
        manifest = self._compute_manifest(records)

        if use_cache and self._try_load_cache(manifest, expected_n=len(records)):
            return

        self._X, self._y = self._load_records(records)
        if use_cache:
            self._save_cache(manifest)

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
        *,
        log_every: int = 5_000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Eager-load all records into one (N, 63) float32 tensor."""
        import time

        n = len(records)
        X = np.empty((n, LANDMARK_FEATURES), dtype=np.float32)
        y = np.empty(n, dtype=np.int64)
        # Progress logging matters here because Windows Defender throttles
        # bulk np.load (issue/001): without visible progress a slow load
        # is indistinguishable from a hang.
        t0 = time.monotonic()
        for i, (path, label) in enumerate(records):
            vec = np.load(path)
            if vec.shape != (LANDMARK_FEATURES,):
                raise ValueError(
                    f"Bad shape in {path}: got {vec.shape}, expected ({LANDMARK_FEATURES},)"
                )
            X[i] = vec
            y[i] = label
            if log_every and (i + 1) % log_every == 0:
                elapsed = time.monotonic() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                print(
                    f"  loaded {i + 1:>6}/{n} in {elapsed:6.1f}s ({rate:>5.0f} files/s)",
                    flush=True,
                )
        elapsed = time.monotonic() - t0
        print(f"  loaded {n}/{n} in {elapsed:.1f}s", flush=True)
        return torch.from_numpy(X), torch.from_numpy(y)

    # -- cache ---------------------------------------------------------------

    @staticmethod
    def _compute_manifest(records: list[tuple[Path, int]]) -> str:
        """Cheap fingerprint of the dataset state — SHA-1 of sorted filenames.

        Adding, removing, or renaming any source .npy file changes the
        hash and invalidates the cache automatically. Per-file content
        edits (rare in our pipeline — collect.py and ingest_public.py
        only ever create new files) won't be detected, by design;
        delete the cache file manually if you ever need to rebuild.
        """
        h = hashlib.sha1()
        for path, _label in records:
            h.update(str(path).encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()

    def _try_load_cache(self, manifest: str, *, expected_n: int) -> bool:
        """Return True if a valid cache was loaded into ``self._X`` and ``self._y``."""
        if not self._cache_path.is_file():
            return False
        try:
            with np.load(self._cache_path, allow_pickle=False) as data:
                cached_manifest = str(data["manifest"].item())
                if cached_manifest != manifest:
                    return False
                X = data["X"]
                y = data["y"]
        except (OSError, KeyError, ValueError):
            return False

        if X.shape != (expected_n, LANDMARK_FEATURES) or y.shape != (expected_n,):
            return False
        if X.dtype != np.float32 or y.dtype != np.int64:
            return False

        self._X = torch.from_numpy(X)
        self._y = torch.from_numpy(y)
        return True

    def _save_cache(self, manifest: str) -> None:
        """Atomically write the cache: temp file then rename."""
        # ``np.savez(path, ...)`` auto-appends ``.npz`` to the filename
        # (annoying for temp-file rename), so write to an open handle
        # directly — that path passes through verbatim.
        tmp = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")
        try:
            with tmp.open("wb") as f:
                np.savez(
                    f,
                    X=self._X.numpy(),
                    y=self._y.numpy(),
                    manifest=np.array(manifest),
                )
            tmp.replace(self._cache_path)
        finally:
            if tmp.exists():
                tmp.unlink()

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
