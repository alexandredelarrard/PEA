---
name: implement
description: Use only when the user explicitly invokes Implement to execute an
  accepted plan with root-cause fixes and auditable test evidence.
---
# Implement compatibility workflow

Use this skill only on explicit invocation.

## Load the source contract

1. Read `.claude/commands/implement.md` through the repository-authorized Markdown tooling.
2. Read the accepted specification, approved plan, and applicable repository documentation.
3. Invoke and follow `$harness-implement` as the canonical implementation workflow.
4. Preserve useful execution checks from the source command, while the current harness governs evidence paths, approval gates, and repository safety.

## Execute the accepted plan

- Confirm the working tree and protect unrelated user changes.
- Implement in the plan's dependency order.
- Use the plan's RED/GREEN checks: reproduce failures first where practical, make the smallest root-cause change, then rerun focused checks.
- Keep changes inside approved scope; stop and return to planning if the contract must change.
- Follow repository architecture, data-store, typing, logging, configuration, test, and documentation rules.
- Prefer simple native solutions and avoid speculative abstractions.
- Update required docs together with behavior.
- Capture command, result, and conclusion for every material check.

## Evidence and handoff

Write `reports/validate/<DATE-slug>/03-implementation.md` with:

- plan item status and changed files;
- root cause and implementation decisions;
- tests added or changed;
- RED/GREEN evidence;
- deviations, residual risks, and deferred work;
- exact validation handoff.

Implementation completion is not validation completion. Hand the candidate to `$harness-validate`.
