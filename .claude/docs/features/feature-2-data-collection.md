# Feature 2 — Data collection

**Status:** ✅ LOCKED
**Companion docs:** [`../architecture.md`](../architecture.md) · [`../tech-stack.md`](../tech-stack.md)

**Modules:**
- `scripts/ingest_public.py` — one-shot preprocessor for public dataset.
- `src/asl_live/capture/collect.py` — interactive collection of custom gestures.

## Decisions

1. **Primary dataset: Kaggle ASL Alphabet** (grassknoted). ~87,000 labeled
   images, 29 classes, 200×200 resolution, permissive license (GPL-2.0).
   - Use classes A–Y for letters.
   - Discard J and Z (out of scope per `architecture.md` §1).
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
   the runtime extractor's left → right mirroring (feature 1).

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
     cross-entropy (feature 3).

10. **License posture (v1).** GPL-2.0 of the source dataset is acceptable
    for the prototype. Documented as a dependency in README. Re-evaluate
    if the project ever moves toward redistribution.

11. **Gitignore.** `data/`, `models/`, `__pycache__/`, `*.pyc`, `.venv/`,
    `.env` — added when implementation starts.

## Out of scope for this feature

- Multi-hand or bimanual sign data.
- Video sequences (would be needed for J/Z motion signs).
- Synthetic data generation.
