# Architecture Decision Records (ADRs)

This folder holds short Markdown files documenting cross-cutting
architectural decisions — choices that affect more than one
sub-feature, are likely to be revisited, or whose rationale is
non-obvious without context.

ADRs differ from per-feature decisions (in
[`../features/`](../features/)): a feature file captures decisions
*scoped to that feature*. An ADR captures decisions that span features
or shape the project as a whole.

## When to add an ADR

Write one when:
- A choice will be questioned later and the rationale isn't trivial
  (e.g., "why MediaPipe instead of a CNN?").
- A choice locks future flexibility (e.g., "v1 has no networked
  features at all").
- A choice was hard to make and the reasoning should survive.

Skip if the decision is purely local to one module — that goes into
the per-feature file instead.

## File format

```
NNN-short-kebab-name.md
```
Three-digit zero-padded number, hyphen-separated short name. Numbers
are assigned in order of creation; gaps are fine if an ADR is
withdrawn.

Each ADR has a small header:

```markdown
# ADR-NNN: Title

**Date:** YYYY-MM-DD
**Status:** Proposed | Accepted | Superseded by ADR-MMM

## Context
What problem are we solving? What are the constraints?

## Decision
What did we choose?

## Consequences
What does this make easy / hard later?

## Alternatives considered
What else did we look at? Why did we reject it?
```

## Index

(none yet — add ADRs as they're written.)
