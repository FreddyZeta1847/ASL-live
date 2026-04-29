# CLAUDE.md — ASL-live

Entry point for Claude Code in this project. Read this first.

## What this project is

Offline ASL alphabet → spoken translation device on Raspberry Pi 5.
A mute user signs letters into a USB camera; the Pi recognizes them,
builds words, translates each completed word into the user's chosen
language (IT / ES / FR / EN / DE), and speaks it through a USB speaker.
Fully phone-free and internet-free.

## Directory map

| Path | Purpose |
|---|---|
| `.claude/CLAUDE.md` | This file — project entry point |
| `.claude/docs/architecture.md` | System design: goals, gestures, process layout, UX, scope |
| `.claude/docs/tech-stack.md` | Chosen technologies and rationale |
| `.claude/docs/features/feature-N-<name>.md` | Per-feature locked decisions with full rationale |
| `.claude/docs/decisions/` | ADRs for cross-cutting decisions |
| `.claude/plans/PLAN.md` | Phase-by-phase implementation roadmap |
| `.claude/plans/plan_zip.md` | Condensed index of plan files |
| `.claude/plans/current-task.md` | What's actively being worked on right now |
| `.claude/plans/phases/phase-N-<name>.md` | Per-phase implementation plan |
| `.claude/agents/` | Project-local subagent definitions |
| `tree.md` | Up-to-date filesystem map of the repo |

## Working protocol

1. **Design changes** → update `.claude/docs/architecture.md` and/or the
   relevant `feature-N-<name>.md`. Don't put design in `plans/`.
2. **Starting a new phase** → produce
   `.claude/plans/phases/phase-N-<name>.md` *before* writing any code,
   get user approval, then implement. Auto mode does **not** override
   this — the user wants the plan-before-code beat per phase.
3. **Active session** → keep `.claude/plans/current-task.md` reflecting
   the active task so any session resumes cleanly.
4. **After every significant change** → commit + push (per the
   `git-autopush` skill).
5. **After any file-hierarchy change** → regenerate `tree.md`.

## Project-specific conventions

- Source layout: `src/asl_live/<subpackage>/`. Tests at `tests/`. Scripts
  at `scripts/`.
- Two install profiles in `pyproject.toml`: `[dev]` (PC) and `[pi]`.
- Datasets and trained models are gitignored (`data/`, `models/`).
- Python ≥ 3.11.
- Use the `ml-python-expert` subagent for review of ML-touching code
  (training, classifier, augmentation). Use the main thread for
  everything else.

## Memory

User-specific working preferences live under
`~/.claude/projects/C--Users-Fede-Desktop-Projects-ASL-live/memory/`
and are loaded into context automatically by Claude Code. Don't
duplicate that content here.
