# Phase 2 — Train the MLP classifier

Implementation plan for [PLAN.md §2 / Phase 2](../PLAN.md). Approved 2026-05-08. Authoritative reference until phase 2 ships.

---

## Context

Phase 1 shipped: `LandmarkExtractor`, Kaggle ingest, custom SPACE/DELETE collector. The dataset on disk:

| Class set | Per-class range | Total |
|---|---|---|
| 24 letters | 2,586 (N) – 5,746 (F) | ~110,966 |
| SPACE / DELETE | 1,000 each | 2,000 |
| **Total** | — | **~112,966** |

Phase 2 trains the MLP described in [`feature-3-classifier.md`](../../docs/features/feature-3-classifier.md), validates against the locked acceptance bar (macro-F1 ≥ 0.95, no off-diagonal cell > 2 %), and exports `mlp.onnx` for phase 3's live recognizer.

All design is locked in `feature-3-classifier.md` (architecture, loss, augmentation, split, hyperparams, ONNX format, acceptance metrics). Phase 2 is execution + reporting, not new design — depth on each decision lives in the feature doc; this plan focuses on **how the pieces fit together**.

---

## Decisions for this phase

The genuinely open choices not pinned by feature-3:

| Decision | Choice |
|---|---|
| XGBoost baseline | **Skip in v1**; add only if MLP fails the bar. |
| Artifact storage | **Timestamped runs**: `models/runs/<utc-ts>/{mlp.onnx, label_map.json, training_report.json}`. `models/mlp.onnx` (read at runtime) is overwritten with the latest. |
| Data loading | **PyTorch `Dataset` + `DataLoader`** (conventional torch idiom, useful for learning). |
| Subagent | **`ml-python-expert`** review pass after the metrics module is in place, before ONNX export. |

---

## Training pipeline at a glance

What happens when `python -m asl_live.train.train_mlp` runs, end to end:

1. **Walk** `data/landmarks/<class>/*.npy` → load all ~112,966 vectors into memory as `(N, 63)` float32. Tag each with its class label from the folder name.
2. **Stratified split** → 80 % train / 10 % val / 10 % test, with seed=42 so two runs produce identical splits. *Stratified* means each class gets the same 80/10/10 ratio individually — no class can land entirely in val or test by chance.
3. **DataLoaders.** Each split is wrapped in a `DataLoader` for batching/shuffling. The **train** loader also applies `RandomAugment` per batch (see Modules → augment.py). Val and test loaders pass clean data.
4. **Train the MLP** for up to 100 epochs. Loss = class-weighted cross-entropy (so SPACE/DELETE at 1k samples don't get drowned out by 5k-sample letters). Optimizer = Adam (lr=1e-3). LR schedule halves on plateau. Early stopping watches val loss with patience=10. Best checkpoint is kept in memory.
5. **Evaluate** the best checkpoint on the test set (clean, untouched). Compute per-class F1, macro-F1, and the 26 × 26 confusion matrix.
6. **Acceptance check.** Macro-F1 ≥ 0.95 *and* no off-diagonal confusion cell > 2 % of its row total. If either fails, print the failing rows and exit non-zero — **do not** export a bad model.
7. **Export.** Model → ONNX (opset 17, dynamic batch axis). Write `mlp.onnx`, `label_map.json`, `training_report.json` under `models/runs/<utc-ts>/`, and copy `mlp.onnx` + `label_map.json` to `models/` so the runtime always reads the latest.

Steps 1–4 happen in `train_mlp.py`; the helper modules below isolate the testable pieces.

---

## Modules

### `src/asl_live/train/dataset.py` — load + split

- **`LandmarkDataset(root)`** — PyTorch `Dataset` returning `(vec_63, class_index)`. On init it walks `data/landmarks/<class>/*.npy`, builds a flat list of file paths + class indices, and exposes `__len__` / `__getitem__`. Loading is cheap (each `.npy` is ~252 bytes), so we either mmap on demand or load all into RAM up front.
- **`compute_class_weights(counts)`** — returns a per-class weight `total_n / (num_classes × n_c)` (feature-3 §3.2). The class with the fewest samples gets the highest weight; passed to `nn.CrossEntropyLoss(weight=…)` so each batch's gradient counts SPACE samples as much as letter samples.
  *Why this matters:* without weighting, the model can score 85 % accuracy by ignoring SPACE/DELETE entirely — useless for our UX. The weighted loss makes that shortcut unprofitable.
- **`make_splits(dataset, seed=42)`** — stratified 80/10/10 split. Returns three lists of indices into the dataset. *Stratified* per feature-3 §3.4: every class gets the same ratio, so no class accidentally has zero test samples (which would break per-class F1).
- **`build_label_map(classes)`** — fixed alphabetical ordering with SPACE/DELETE last, identical to `config.CLASSES`. Saved next to the ONNX so the Pi knows "output index 12 means M."

**Tests** (`tests/test_dataset.py`): split sums match dataset size with no overlap, every class is present in every split, class-weight formula matches a hand-computed example, label-map is deterministic across two `build_label_map` calls.

### `src/asl_live/train/augment.py` — training-only augmentation

**Why augmentation at all.** The dataset is fixed at ~113k vectors and the MLP has ~30k parameters — enough capacity to *memorize* training samples instead of *learning* their general shape. Memorized models score perfectly on the training set and collapse on real-world hands that differ even slightly. Augmentation injects controlled variation each epoch so the model can't lock onto exact memorized values.

**What `RandomAugment` does.** Three independent perturbations, each gated at `p=0.5` per sample (so on average half the samples in any batch get each perturbation):

- **Gaussian noise** (std=0.01) — simulates MediaPipe's per-frame detection jitter.
- **Uniform scale** (×[0.95, 1.05]) — simulates the user signing slightly closer or farther from the camera than during collection.
- **Translation** (±0.02 on x and y) — simulates the wrist not landing at the same screen position session-to-session.

These perturbations are **applied as tensor ops on the batch inside the train DataLoader's collate path** — fast, on-the-fly, no precomputed augmented dataset. The transform runs only on the train loader; **val and test see clean data** so their metrics measure model quality, not perturbation tolerance.

**What's NOT applied** (per feature-3 §3.3):
- **No mirror** — already applied once at ingest (feature-2). Doing it again would be redundant.
- **No rotation** — some ASL letter pairs are distinguished by hand tilt; collapsing rotation would conflate them.

**Tests** (`tests/test_augment.py`): bounds are respected for each perturbation, `p=0` is identity, output shape is preserved, z dimension treated identically to x/y.

### `src/asl_live/train/model.py` — the MLP

Architecture exactly per feature-3 §3.1:

```
Linear(63, 128) → ReLU → Dropout(0.2) → Linear(128, 64) → ReLU → Linear(64, 26)
```

~30k parameters, ~120 KB at FP32, < 5 µs inference per frame on a dev PC. Designed to be small enough that a Pi 5 won't notice it. Why MLP and not CNN/RNN/Transformer is in feature-3 §3.1.

The model returns **raw logits** (no softmax). Softmax is applied at inference time in `classifier.py` because:
- training loss (cross-entropy) operates on logits and applies its own log-softmax internally — adding a softmax to the model would double-apply it,
- ONNX export is leaner without it.

### `src/asl_live/train/train_mlp.py` — orchestration + CLI

The entry point. Wires the modules together:

- Loads `LandmarkDataset`, computes class weights, builds splits.
- Builds three `DataLoader`s (`batch_size=256`); attaches `RandomAugment` to the train loader only.
- Constructs `MLP`, `Adam(lr=1e-3, weight_decay=1e-4)`, `ReduceLROnPlateau(factor=0.5, patience=5)`, `CrossEntropyLoss(weight=class_weights)`.
- Trains up to 100 epochs with early stopping (patience=10). Logs `epoch / train_loss / val_loss / val_macro_f1 / lr` per epoch. Holds the best-val-loss state-dict in memory.
- After training: load best state-dict, predict on test set, hand off to `metrics.py` and (if passed) `export.py`.

CLI flags (all default to feature-3 §3.5 values; flags exist for experimentation): `--epochs 100`, `--lr 1e-3`, `--batch-size 256`, `--seed 42`, `--device auto` (cuda if available, else cpu).

### `src/asl_live/train/metrics.py` — score the model

Computes everything from a single 26 × 26 confusion matrix:

- `confusion_matrix(y_true, y_pred, n_classes) → (26, 26) np.ndarray`
- `per_class_f1(cm, classes) → dict[str, float]`
- `macro_f1(cm) → float` — simple unweighted mean of per-class F1
- `check_acceptance_bar(cm, classes) → (passed: bool, failures: list[str])` — returns specific failing rows like `"Actual DELETE → Predicted A: 4.1% > 2.0% bar"` when the bar isn't met

Exact computation (precision/recall directly off rows and columns, edge cases for zero-prediction classes) is documented as a worked example in feature-3 §3.6.

**Tests** (`tests/test_metrics.py`): macro-F1 matches `sklearn.metrics.f1_score(average="macro")` within 1e-9 on synthetic confusion matrices; confusion matrix shape and row totals are correct; acceptance check returns the right `(bool, list)` for hand-built failing matrices.

### `src/asl_live/train/export.py` — make it deployable

PyTorch model → ONNX (opset 17, dynamic batch axis on dim 0 so the Pi can run single-frame *and* batched inference if we ever need it). Three artifacts per run:

- `mlp.onnx` — what the runtime loads.
- `label_map.json` — `{"0": "A", "1": "B", …, "25": "DELETE"}`.
- `training_report.json` — git SHA (`git rev-parse HEAD`), hyperparams used, dataset counts per class, per-class F1, macro-F1, full confusion matrix as a list-of-lists. Lets us reconstruct what produced any saved model without rerunning training.

All three land in `models/runs/<utc-ts>/`. After writing, `mlp.onnx` and `label_map.json` are also copied to `models/` (flat) so the Pi only needs two stable paths.

**Verification at this step:** load the exported ONNX with onnxruntime, run a known sample, assert logits match the PyTorch model within 1e-5. If they don't, something went wrong during export and the run aborts before stamping the latest pointers.

### `src/asl_live/recognition/classifier.py` — runtime wrapper

Sits in `recognition/`, not `train/`, because it ships to the Pi (training does not). Loads `models/mlp.onnx` + `models/label_map.json` lazily on first `predict` call.

```python
class Classifier:
    def predict(self, landmarks: np.ndarray) -> tuple[str, float]:
        # Run onnxruntime, softmax in numpy, return (label, confidence).
```

Used by phase-3's `demo_recognition.py` and (eventually) by the recognizer worker process.

**Tests** (`tests/test_classifier.py`): build a tiny 1-class dummy ONNX inline (~3 lines via `torch.onnx.export`), instantiate `Classifier`, confirm `predict` returns the expected label with confidence ≈ 1.0. No real model required for the test.

---

## Subagent usage

`ml-python-expert` once, after `metrics.py` is implemented and tested, before `export.py`. Reviews: split has no per-class leakage, class-weight formula sign is correct, augmentation bounds match feature-3 §3.3, metrics match sklearn, ONNX export round-trips logits losslessly. Output is a list of findings; address them before commit 5 ships.

---

## Verification (the acceptance gate to phase 3)

```bash
python -m asl_live.train.train_mlp
# Expected tail:
# ✓ Macro-F1: 0.97
# ✓ All off-diagonal cells ≤ 2.0% of row total
# Wrote models/runs/2026-05-08T14-30-22Z/{mlp.onnx, label_map.json, training_report.json}
# Updated models/mlp.onnx -> latest

pytest tests/
# All green.

python -c "
from asl_live.recognition.classifier import Classifier
import numpy as np
print(Classifier().predict(np.random.randn(63).astype(np.float32)))
"
# -> ('SOME_LETTER', 0.xx)
```

If macro-F1 < 0.95 or any off-diagonal cell > 2 %, **do not** advance to phase 3. Failure responses are in feature-3 §3.6 and architecture.md §8 (e.g., DELETE/A confusion → switch DELETE to a pinch gesture).

---

## Out of scope for phase 2

- XGBoost baseline (revisit only if MLP fails)
- Hyperparameter sweeps (manual iteration only — feature-3 §3.5)
- Quantization (deferred — feature-3 §3.9)
- MIN_CONF tuning (phase 7 — feature-3 §3.7)
- Live inference / debounce / Pi deployment (phases 3–4)
