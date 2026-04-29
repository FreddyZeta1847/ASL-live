# Plans — condensed index

- **[PLAN.md](PLAN.md)** — main implementation plan for ASL-live.
  7 phases: data collection → MLP training → live recognition demo → Pi LCD
  integration → translation/TTS workers → buttons + audio language menu →
  polish (auto-start, BOM, wiring). Each phase ends with a demoable artifact.
  Source-of-truth design: [`../docs/architecture.md`](../docs/architecture.md)
  + [`../docs/tech-stack.md`](../docs/tech-stack.md) +
  [`../docs/features/`](../docs/features/).
- **[current-task.md](current-task.md)** — what's actively being worked on
  in this session. Most-important file for resuming a session.
- **[phases/phase-1-data-collection.md](phases/phase-1-data-collection.md)** —
  implementation plan for PLAN.md phase 1. 5 commits: skeleton →
  landmarks + tests → Kaggle ingest → SPACE/DELETE collector → README.
  Approved 2026-04-29.
