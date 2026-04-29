# Sub-feature decisions

Locked design decisions for each sub-feature of ASL-live, captured during
sub-feature-by-sub-feature design review. Companion document to:
- [`discussion.md`](discussion.md) — high-level architecture
- [`plans/PLAN.md`](plans/PLAN.md) — phase-by-phase implementation plan

A sub-feature is "locked" once both parties have signed off on every open
question for it. Locked decisions are not re-litigated unless new evidence
appears (e.g., a confusion matrix shows the choice was wrong).

---

## #1 — Hand-landmark extraction ✅ LOCKED

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

### Decisions

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

### Out of scope for this sub-feature

- Multi-frame temporal features (would be needed for J/Z motion signs —
  explicitly out of scope per discussion §1).
- Rotation-invariant feature extraction (rejected above).
- 3D pose estimation beyond MediaPipe's built-in z coordinate.

---

## #2 — Data collection ✅ LOCKED

**Modules:**
- `scripts/ingest_public.py` — one-shot preprocessor for public dataset.
- `src/asl_live/capture/collect.py` — interactive collection of custom gestures.

### Decisions

1. **Primary dataset: Kaggle ASL Alphabet** (grassknoted). ~87,000 labeled
   images, 29 classes, 200×200 resolution, permissive license (GPL-2.0).
   - Use classes A–Y for letters.
   - Discard J and Z (out of scope per discussion §1).
   - Discard the dataset's "space", "delete", "nothing" classes — their hand
     shapes don't match our chosen gestures.

2. **Two-script collection pipeline.**
   - `ingest_public.py` runs MediaPipe over each Kaggle image, extracts the
     63-dim landmark vector via the same `landmarks.extract()` used at
     runtime, drops frames where no hand was detected, saves as `.npy`
     under `data/landmarks/<class>/`. One-shot, ~1–2 h of CPU.
   - `collect.py` is interactive, only used for SPACE, DELETE, and
     optional user-specialization top-ups.

3. **Custom-collection cadence.** Auto-capture while the hand is detected
   and the landmarks are stable for 5 frames, with a 10-frame cooldown
   between captures. Manual key-press capture not used.

4. **Diversity prompts during custom collection.** Skipped for SPACE and
   DELETE — the gestures are gross enough that pose variation matters
   little. Re-introduce only if a future top-up is needed for fine letter
   distinctions.

5. **Where to collect.** PC, not Pi. Faster iteration. Pi top-up only if
   phase 2 evaluation shows poor real-world accuracy.

6. **Mirror augmentation.** Applies to *both* public ingest and custom
   collection. Each saved sample is also stored mirrored (x → -x in
   normalized coordinates), effectively 2× free data and consistent with
   the runtime extractor's left → right mirroring (sub-feature #1).

7. **Storage format.** Per-sample `.npy` files under
   `data/landmarks/<class>/<source>_<id>.npy` (so we can tell public from
   custom and delete bad samples easily). Consolidated to a single
   `data/dataset.npz` at the start of every training run.

8. **Raw frames.** Not saved by default. `collect.py` accepts a
   `--save-frames` flag for occasional debugging. Public ingest never
   saves frames — landmarks only.

9. **Target class sizes.**
   - Letters A–Y: ~3,000 samples each from public ingest.
   - SPACE, DELETE: ~500 samples each from custom collection.
   - Class imbalance handled at training time via class-weighted
     cross-entropy (sub-feature #3).

10. **License posture (v1).** GPL-2.0 of the source dataset is acceptable
    for the prototype. Documented as a dependency in README. Re-evaluate
    if the project ever moves toward redistribution.

11. **Gitignore.** `data/`, `models/`, `__pycache__/`, `*.pyc`, `.venv/`,
    `.env` — added when implementation starts.

### Out of scope for this sub-feature

- Multi-hand or bimanual sign data.
- Video sequences (would be needed for J/Z motion signs).
- Synthetic data generation.
