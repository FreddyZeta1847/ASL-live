# Feature 1 — Hand-landmark extraction

**Status:** ✅ LOCKED
**Companion docs:** [`../architecture.md`](../architecture.md) · [`../tech-stack.md`](../tech-stack.md)

**Module:** `src/asl_live/recognition/landmarks.py`

**Public API:**
```python
def extract(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Returns a normalized 63-dim landmark vector for the most confident hand,
    or None if no hand is detected.
    The returned vector is always in canonical "right-hand" form (left-hand
    detections are mirrored).
    """
```

## Decisions

1. **Multi-hand handling.** If MediaPipe detects more than one hand, pick
   the single most-confident detection and ignore the others. No error,
   no warning — silent.

2. **Handedness normalization.** MediaPipe labels each detection as left
   or right. Left-hand detections are mirrored (x → -x in image space)
   *before* normalization, so the classifier only ever sees a canonical
   right-handed sign. Rationale: avoids doubling the dataset / risk of
   overfitting on a left-vs-right split.

3. **No-hand return value.** `extract()` returns `None` when no hand is
   detected. Downstream code (debounce, recognizer worker) treats `None`
   as a "gap" frame. **The classifier is not invoked on `None` frames** —
   this is the efficiency contract.

4. **Normalization scheme.** Two steps, in order:
   - Translate: subtract wrist (landmark 0) from every landmark → wrist at origin.
   - Scale: divide all coordinates by the maximum wrist-to-landmark distance
     in the frame → the largest finger reach becomes magnitude 1.

   **No rotation normalization** and **no distance-only features**.
   Rationale: some ASL letter pairs are distinguished by hand rotation /
   tilt; collapsing rotation away would conflate them.

5. **MediaPipe `model_complexity`.** Default `1` (more accurate). Fall back
   to `0` (faster) only if the Pi can't sustain ≥ 15 fps in phase 3
   testing.

6. **Input resolution.** Camera captures at 640 × 480. MediaPipe internally
   resizes; sending higher res is wasted compute.

## Single source of truth: same preprocessing at training and inference

The `extract()` function above is the **only** path that turns a camera
frame into a 63-dim vector. Both phases of the system call it:

- **Data collection** (`scripts/collect.py`, feature 2) — stores the
  vector to disk as a training sample.
- **Live recognition** (recognizer worker, feature 3) — feeds the
  vector straight to the classifier at inference time.

**Why this matters (train/serve skew).** In ML, if the preprocessing
applied to the training set differs from the preprocessing applied at
inference — even by something as small as a different scale denominator
or a flipped axis — the model sees one kind of input during training and
a subtly different kind in production. The result is silent accuracy
degradation: the model "tested fine" on the held-out set but
under-performs on the device, and the bug is hard to find because the
math itself is correct. Sharing a single function eliminates the risk
structurally: there is no second copy to drift.

**Practical consequence.** If `_normalize()` is ever modified, the
entire dataset must be re-collected (or re-processed from saved raw
landmarks) before retraining — every existing `.npy` was preprocessed
under the *old* scheme and would silently corrupt the next run.

## Out of scope for this feature

- Multi-frame temporal features (would be needed for J/Z motion signs —
  explicitly out of scope per `architecture.md` §1).
- Rotation-invariant feature extraction (rejected above).
- 3D pose estimation beyond MediaPipe's built-in z coordinate.
