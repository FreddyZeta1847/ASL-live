# Current task

The active task in this session. Update on every session start and
whenever scope shifts. Keep it short — this file is for orientation,
not documentation.

---

## Now

**Phase 1 — shipped.** Awaiting direction on phase 2.

Per the user's plan-before-code workflow, the next step is to draft
[`phases/phase-2-training.md`](phases/phase-2-training.md) with the
implementation plan for PLAN.md phase 2 (train the MLP classifier),
get user approval, *then* code.

## Recently shipped

- 2026-04-30 — **Phase 1 complete (5 commits, all pushed).**
  - `7ca5e99` chore: bootstrap pyproject + package skeleton
  - `9154ee2` feat(recognition): landmark extractor with unit tests
  - `adb2ffe` feat(scripts): Kaggle ASL Alphabet ingest preprocessor
  - `ae4b782` feat(capture): interactive SPACE/DELETE collector
  - `f87b5ca` docs: project overview + collection protocol in README
- 2026-04-29 — Restructured `.claude/` into `docs/` + `plans/` layout.
- 2026-04-29 — Phase 1 implementation plan written and approved.
- All 10 sub-features locked with full rationale (in `docs/features/`).
- Top-level design locked (`docs/architecture.md` + `docs/tech-stack.md`).

## Blocked / waiting on

- User to: kick off phase 2 planning, or run the new ingest /
  collector locally to validate against real hardware (Python env
  setup + Kaggle download + webcam smoke test).
