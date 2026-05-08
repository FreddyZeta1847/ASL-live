# 001 — Windows Defender throttles bulk np.load on 113 k files

**Date:** 2026-05-09
**Status:** Resolved (cache workaround)
**Touched:** `src/asl_live/train/dataset.py`, `tests/test_dataset.py`

## Problem

Smoke-testing `python -m asl_live.train.train_mlp --epochs 2` on the real
phase-1 dataset (~113 k normalized landmark `.npy` files) hung for 10+
minutes producing zero stdout, with the Python process showing only ~12 MB
working set. Looked like a deadlock or import hang.

## Root cause

Two compounding issues:

1. **Per-file `np.load` is throttled by the OS / antivirus on Windows.**
   Timed sequential load of all files, with progress every 5 000:

   ```
   loaded  5000/112966 in   1.6s
   loaded 10000/112966 in   3.2s
   loaded 15000/112966 in   4.8s
   loaded 20000/112966 in  68.6s   ← cliff
   loaded 25000/112966 in 200.7s
   loaded 30000/112966 in 340.2s
   loaded 35000/112966 in 471.3s
   loaded 40000/112966 in 573.8s
   ```

   First ~15 k files: ~3 000 files/s. Past that: ~40 files/s — an 80×
   slowdown. Pattern is consistent with Windows Defender real-time
   protection scanning each file on open after some warm-up window. Total
   projected time for the full set: ~10 minutes.

2. **Python stdout buffering hid the progress.** When the harness
   redirects stdout to a file (or `subprocess.PIPE`), Python defaults to
   block-buffered output. The script's early `print("Loading dataset…")`
   sat in a buffer instead of appearing — so from the outside, the run
   looked completely silent and likely hung.

## Fix

**Cache the loaded `(X, y)` tensors as one `.npz` next to
`data/landmarks/`** in `dataset.py`:

- `_compute_manifest(records)` — SHA-1 of the sorted filename list.
- `_try_load_cache(manifest, expected_n)` — short-circuit if a valid
  cache exists.
- `_save_cache(manifest)` — atomic write via temp file + rename.

First load is still slow (~10 min, unavoidable on Windows without
disabling Defender for the data folder). Every subsequent training run
loads from the cache in <1 s. Cache invalidates automatically when the
sorted filename list changes (add / remove / rename).

**Sub-bug found and fixed during implementation.** `np.savez(path, …)`
auto-appends `.npz` to the filename, which broke the temp-file rename
pattern (`landmarks_cache.npz.tmp` → `landmarks_cache.npz.tmp.npz` on
disk, then rename of the bare `.tmp` name fails). Fix: pass an open file
handle (`with tmp.open("wb") as f: np.savez(f, …)`) instead of a path —
that bypasses the auto-extension.

**4 cache tests added** in `tests/test_dataset.py`: cache created on
first load, second load returns same data, file addition invalidates,
`use_cache=False` doesn't write a cache.

## Lesson

On Windows, any code that does ≥10 000 sequential file opens needs a
cache, a parallel reader, or a Defender exclusion — sequential per-file
I/O collapses after a few thousand operations. And: when a long-running
script appears silent, suspect stdout buffering before suspecting a
hang. Use `python -u` or `flush=True` on early prints during
development.
