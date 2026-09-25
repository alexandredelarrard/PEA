---
name: harness-plan
description: Grill material decisions and turn an approved specification into a
  living, minimal, phase-by-phase plan with exact scope, evidence, recovery, and
  user approval.
---
# Harness Plan

Create an executable plan from an approved specification. This stage resolves material decisions with the user, maps every acceptance criterion to evidence, and stops before product edits.

## Inputs and exact output

Require the approved `01-spec.md` and its `01-research.md`. Reuse their exact run directory:

`reports/validate/<YYYY-MM-DD>-<slug>/`

Produce `02-plan.md` there and obtain one explicit decision: Approve plan and start work, Revise, or Stop. Do not create a second planning directory.

Use the repository-authorized Markdown mechanism. If OpenKnowledge is registered, read and write all Markdown through its MCP tools.

## Ground the plan

1. Read the complete spec and research report, applicable repository instructions, and task-specific docs before source code.
2. Extract current behavior, desired end state, reproduction and root cause for a defect, open questions, dependencies, and affected components.
3. Inspect the real implementation seams, callers, tests, configuration, data contracts, validation commands, and prior reports named by the spec. Do not plan from filenames alone.
4. Capture base revision, branch, dirty-worktree facts, baseline checks, and pre-existing failures without changing the workspace.
5. Reuse existing helpers, registries, platform features, and installed dependencies before proposing new abstractions. Keep the smallest coherent design that satisfies the accepted spec.
6. For data or model work, read the repository's data, modelling, testing, and execution contracts and plan the domain checks at the same time as the implementation.

## Grill the user

Maintain a decision ledger in `02-plan.md`. Ask only questions that can materially change behavior, scope, terminology, interfaces, data semantics, model methodology, validation, risk, recovery, or external effects.

Ask one dependency-ready question at a time. Each question contains:

- The decision to make.
- Two or three real options when alternatives exist.
- A recommendation.
- The trade-off and what changes in the plan.

Challenge unclear terms and conflicting requirements. Do not ask for information already present in the spec or repository. Resolve blocking decisions before finalizing. Record non-blocking unknowns as assumptions with the consequence if wrong.

## Plan document contract

`02-plan.md` is the authoritative, living implementation record. It contains:

1. Objective, approved spec and research paths, base revision, branch strategy, constraints, and approved effects.
2. Current state: observed behavior, relevant architecture and files, limitations, reproduction, and root cause when known.
3. Desired end state: functional, data/model, performance, compatibility, and operational outcomes.
4. Explicit out of scope and deferred work with a trigger for reconsideration.
5. Decision ledger and glossary.
6. Requirement traceability: every `AC-###` maps to one or more planned checks, and every planned check names the criterion it supports.
7. Exact write scope. List risk-zone edits separately and retain their approval gate.
8. Dependencies and migrations, including documentation, configuration, data, and artifact compatibility.
9. Testing strategy: focused, integration, end-to-end, manual, regression, data, model, performance, and recovery checks as applicable.
10. Risks, mitigations, rollback, and conditions that force replanning.
11. Ordered phases and a final completion checklist.

Each phase starts as `⬜` and has:

- Goal and user-visible value.
- Acceptance criteria covered.
- Dependencies and why the ordering matters.
- Exact files or a bounded discovery step.
- Concrete changes and exclusions.
- A failing reproduction or baseline when behavior changes.
- Exact verification command, working directory, expected observation, timeout/cost when material, and possible skip or abstention.
- Rollback or safe stopping point.
- Done conditions that leave the repository functional and independently testable.

Use `🔄` while a phase is active, `✅` only after its checks pass, and `⚠ BLOCKED` when evidence or authority is missing. Harness Implement updates these markers and records deviations without rewriting history.

## Task-specific planning

Adapt phases to the work instead of forcing a template:

- Bug fix: reproduce, isolate the root cause, fix minimally, add regression evidence, and verify sibling paths.
- Data extraction or feature work: schema/grain, source and resume behavior, transformation, storage/integration, live-data validation, then docs.
- Performance work: benchmark the same workload before and after, preserve correctness, and add a regression threshold only when measurements justify it.
- Model or strategy work: temporal split and leakage controls, baseline, training change, out-of-sample diagnostics, artifacts, and downstream portfolio effects.
- Cache work: hit, miss, expiry, invalidation, memory/cost, and stale-data behavior.

Do not invent a fixed coverage percentage, time estimate, optimization, documentation site, or dependency unless the repository or accepted spec requires it. Use repository-native commands from its runbook rather than generic placeholders.

## Size and review

Keep one `02-plan.md` whenever practical. If it would exceed roughly 700 lines, retain `02-plan.md` as the authoritative index and put independently executable detail in `plans/02-phase-<NN>-<slug>.md` inside the same run directory.

Perform a fresh plan review when an independent reviewer is available. Check spec coverage, extra scope, file/interface consistency, real RED/GREEN signals, data/model validation, minimality, recovery, and runnable commands. Record each review finding and disposition in `02-plan.md`.

## Approval and completion

Present the completed plan and ask the user to choose Approve plan and start work, Revise, or Stop. Approval authorizes only the listed scopes, checks, reports, and effects. Material scope expansion returns to this gate.

This stage is complete only when blocking decisions are closed, every phase is testable, every acceptance criterion has credible planned evidence, recovery is explicit, the review is clear, and the user approves implementation.
