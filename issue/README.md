# Issue log

Captures **non-trivial problems** encountered during development and the
fixes that resolved them. The goal is to save future-us (and reviewers)
the cost of re-discovering the same gotcha.

## When to write an entry

Write one when:

- A problem took non-trivial debugging time to find the root cause.
- The fix is a workaround (not a "fix the typo") that someone reading
  the code later might want to undo, simplify, or revisit.
- The problem is environment- or platform-specific (Windows-only,
  GPU-only, antivirus-only) and likely to recur.
- A subtle behavior of a third-party tool surprised us and the
  surprise isn't obvious from the code alone.

**Skip** routine bugs that get fixed in the same commit they were
introduced — those are caught by `git log` already.

## Naming

`issue/NNN-short-kebab-name.md` — `NNN` zero-padded so files sort
chronologically (`001`, `002`, …, `099`, `100`).

## Entry structure

```markdown
# NNN — Short title

**Date:** YYYY-MM-DD
**Status:** Resolved / Workaround / Open
**Touched:** path/to/file.ext, path/to/other.ext

## Problem
What was observed (symptoms, not theories).

## Root cause
Why it was happening, in one or two paragraphs.

## Fix
What was changed, with file/line references.
Note any workaround vs. proper fix; flag what to revisit later.

## Lesson
One sentence the next person should remember.
```
