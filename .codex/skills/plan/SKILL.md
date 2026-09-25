---
name: plan
description: Use only when the user explicitly invokes Plan to turn accepted
  specifications into a decision-complete, deeply tested implementation plan.
---
# Plan compatibility workflow

Use this skill only on explicit invocation.

## Load the source contract

1. Read `.claude/commands/plan.md` through the repository-authorized Markdown tooling.
2. Read the accepted specification and relevant repository documentation before code.
3. Invoke and follow `$harness-plan` as the canonical planning workflow.
4. Keep useful planning checks from the source command, while the current harness governs report paths, approval gates, and repository rules.

## Plan deeply

- Verify the specification is accepted, bounded, and testable.
- Trace the actual execution path, data contracts, callers, configuration, persistence boundaries, tests, and operational entry points.
- Identify affected files and risk zones precisely; request any required authorization before planning edits there.
- Compare materially different approaches when the trade-off matters.
- Prefer the smallest robust design and record what is deliberately not being built.
- Define ordered implementation increments with RED/GREEN evidence, rollback points, and exact validation commands.
- Define data, modeling, refactor, and documentation checks according to the change type.
- Grill the user on every unresolved decision that could change public behavior, data grain, schema, model interpretation, migration, cost, or acceptance.

## Evidence and gate

Write `reports/validate/<DATE-slug>/02-plan.md` with:

- accepted inputs and decisions;
- current-state call/data flow;
- selected design and rejected alternatives;
- file-level change map;
- ordered implementation and test sequence;
- validation matrix and report lane;
- risks, rollback, and explicit exclusions;
- approval status.

Do not implement until the plan is decision-complete and accepted when acceptance is required.
