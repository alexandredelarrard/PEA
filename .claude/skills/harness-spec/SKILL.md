---
name: harness-spec
description: Research a code, data, or modelling request and turn it into an
  evidence-grounded specification with observable acceptance criteria and fixed
  audit outputs.
---
# Harness Spec

Turn an initial request or an existing draft specification into an evidence-grounded acceptance contract. This stage is research plus specification; it does not plan or edit product code.

## Inputs and exact outputs

Accept an intent, issue, draft spec, or accepted spec path. Reuse the run directory supplied by Harness Run. Otherwise create exactly:

`reports/validate/<YYYY-MM-DD>-<slug>/`

Use a stable lowercase kebab-case slug. Produce these files there and nowhere else:

- `01-research.md`: current behavior, code/data/model flow, sources, constraints, and unknowns.
- `01-spec.md`: the accepted outcome and testable contract.

When orchestrated, add both paths and their status to `00-run.md`. Obtain an explicit user decision: Approve spec for planning, Revise, or Stop.

Use the repository-authorized Markdown mechanism. If OpenKnowledge is registered, read and write all in-scope Markdown through its MCP tools.

## Research

1. Read applicable repository instructions and task-specific docs before source code. Read every user-named file completely in the main context before delegating.
2. Classify the work as code, data, modelling, or mixed. Trace the actual path end to end: entry point, producer, storage/state, consumers, tests, and validation.
3. Prefer repository sources and observed behavior. Search externally only when outside facts are required; record the source and retrieval date.
4. Delegate independent read-only questions when sub-agents are available. Give each a bounded question and source scope. The main agent owns synthesis and verifies material claims.
5. Separate facts, inferences, assumptions, and unresolved questions. Do not turn an implementation idea into a fact about the current system.
6. For a defect, capture a reproducible observation and the best-supported root-cause path.
7. For data work, establish grain, keys, date and freshness semantics, point-in-time constraints, expected versus live population, and downstream use.
8. For modelling work, establish target, split, embargo, metrics, baseline, artifacts, feature availability, and downstream portfolio use.
9. Read prior reports on the same area. Reuse questions and decisions, but re-measure time-sensitive claims.

## Research report contract

`01-research.md` contains:

- Original request and research questions.
- Repository/date/revision and inspected sources.
- Current architecture and end-to-end flow.
- Observed behavior with commands, queries, or named examples.
- Constraints, risk zones, and existing patterns to reuse.
- Facts, inferences, assumptions, unknowns, and conflicts.
- Prior reports reconciled.
- Candidate acceptance boundaries, without choosing implementation prematurely.

Never report an unavailable or skipped inspection as completed.

## Specification

Write the smallest complete behavioral contract with stable identifiers:

- Outcome and user value.
- Current-state summary and evidence.
- In scope and explicitly out of scope.
- Functional requirements as `REQ-001`, `REQ-002`, and so on.
- Acceptance criteria as `AC-001`, `AC-002`, and so on; each is observable and names acceptable evidence.
- Data invariants where applicable: grain, coverage, freshness, null behavior, publication clock, point-in-time integrity, idempotency, and recovery.
- Model invariants where applicable: temporal splits, leakage controls, reproducibility, baseline comparison, diagnostics, and artifact compatibility.
- Security, compatibility, performance, accessibility, and operational constraints only where material.
- Risk zones, external effects, and permissions implementation may require.
- Assumptions, open questions, and deferred work with a trigger for reconsideration.

Do not prescribe code structure unless evidence makes it part of the contract. Prefer behavior and outcomes over implementation taste.

## Questions and approval

Ask only questions whose answers can change scope, behavior, risk, or verification. Ask one dependency-ready question at a time and lead with a recommendation plus trade-off. Do not ask what the request or repository already answers.

Before approval, self-review for placeholders, contradictions, ambiguous terms, unbounded scope, hidden assumptions, and acceptance criteria that cannot be demonstrated. Resolve every blocking unknown or keep it visibly open.

Stop after presenting `01-spec.md`. Planning starts only after the user explicitly approves it.

## Completion

This stage is complete only when research is reproducible, every material requirement has an observable criterion, no blocking unknown is hidden as an assumption, both reports are indexed, and the user approves the spec for planning.
