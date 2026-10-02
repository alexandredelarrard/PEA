---
name: harness-refactor
description: Use when refactoring a part of the codebase and its dependencies
  without changing data aggregation, feature creation, or sanity-check logic,
  driven through the harness stages (spec, plan, implement, validate).
---
# Harness Refactor

Refactor a named part of the codebase and every dependency touching it while keeping its data aggregation, feature creation, and sanity-check logic unchanged. This skill drives `$harness-run` (spec → plan → implement → validate) and adapts the workflow in [refactor.md](../../commands/refactor.md) to the harness audit files.

The canonical rule text lives in [coding standards](../../../wiki/guides/coding-standards.md). Do not restate it; cite it.

## Workflow

1. **Map the subsystem.** Through OpenKnowledge and native code search, list entry points, callers, dependencies, public interfaces, persistence contracts, and existing tests. Record them in `01-research.md`.
2. **Capture goldens before any edit.** Build characterization or replay goldens on real inputs: stored rows, returned frames, and logged counts of the code under refactor. Save them under `_cache/` with the base commit.
3. **Record the baseline.** Run the targeted and broader tests, Ruff lint and format check, Pyright, and the static standards scan on the touched modules. Record existing failures separately in `00-run.md`.
4. **Spec.** `$harness-spec` writes `01-spec.md` with the required items below as acceptance criteria.
5. **Plan.** `$harness-plan` splits the work into behaviour-preserving phases. Each phase names exact files, preserves behaviour, carries its verification command, and ends with a small commit.
6. **Implement.** `$harness-implement` executes phase by phase. Prefer rename, extract function, move code, introduce a seam, or remove verified dead code over broad rewrites. After each phase, rerun the focused tests and the replay.
7. **Validate.** `$harness-validate` reruns the baseline commands, the replay, and the static scan on the final tree and compares them with step 3.

## Required spec items

Every refactor spec carries these acceptance criteria for each touched function and module (rules: [coding standards](../../../wiki/guides/coding-standards.md)):

- [ ] At most 150 lines, one job; a top-level orchestrator only calls step functions in order.
- [ ] At most 3 nesting levels; 2 is the standard.
- [ ] No pass-through functions.
- [ ] Flow ordered load → clean → compute → post-process → save → log / report / sanity checks.
- [ ] Same logic declared once: repeated logic across files or fetchers extracted to the closest utils module and called from every dependent.
- [ ] Computed values passed between functions, never re-declared or recomputed.
- [ ] Lowercase snake_case function names; lowercase, transparent variable names with `df_` for DataFrames.
- [ ] Module constants at the top after imports, and only for parameters, URLs, or user-tunable values.
- [ ] Docstrings a few lines: what the code does, with no history, rationale chronology, or measurements.
- [ ] Large frames pre-filtered to the rows that need computing and vectorised; no `DataFrame.apply` or row loops.
- [ ] Stored data identical to the baseline unless the user approves a change.
- [ ] Risk zones (`context.py`, `utils/step.py`, constants, data store/DDL, configs, `data/`, the PostgreSQL volume, the fingerprint baseline) edited only with explicit approval.

## Evidence

- **Replay:** old versus new outputs on the same real inputs, with 0 diffs in rows, columns, dtypes, and values. Any non-zero diff is a blocking finding unless the user approved it.
- **Static scan:** function length, nesting depth, and pass-throughs on every touched module, before and after.
- **Targeted tests:** each prints a sanity-check conclusion; report only the targeted output.
- **Performance:** wall clock before and after on the same input whenever the change claims a speed gain.

All evidence lands in the harness run directory; nothing exists only in chat.

## Stop and ask

Return to the user when a phase needs a risk-zone edit, changes stored data, changes a public contract, adds a dependency, or when the replay shows a diff whose cause is not proven.
