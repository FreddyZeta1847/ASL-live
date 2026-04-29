# Plans — condensed index

- **[PLAN.md](PLAN.md)** — main implementation plan for ASL-live.
  7 phases: data collection → MLP training → live recognition demo → Pi LCD
  integration → translation/TTS workers → buttons + audio language menu →
  polish (auto-start, BOM, wiring). Each phase ends with a demoable artifact.
  Source-of-truth design: [`../discussion.md`](../discussion.md).
- **[phase_1.md](phase_1.md)** — implementation plan for PLAN.md phase 1
  (data collection pipeline). 5 commits: skeleton → landmarks + tests →
  Kaggle ingest → SPACE/DELETE collector → README. Approved 2026-04-29.
