---
name: research
description: Use only when the user explicitly invokes Research to investigate a
  topic and produce source-backed specifications without implementing.
---
# Research compatibility workflow

Use this skill only on explicit invocation.

## Load the source contract

1. Read `.claude/commands/research.md` through the repository-authorized Markdown tooling.
2. Read the applicable repository instructions and task documentation before code.
3. Invoke and follow `$harness-spec` as the canonical specification and research workflow.
4. When this adapter and the canonical harness differ, preserve the source command's useful research checks but follow the current harness for report paths, gates, and repository safety.

## Execute

- Restate the request, constraints, success criteria, assumptions, and exclusions.
- Inspect the current code and documentation before proposing changes.
- Research external facts only when needed, preferring primary sources and recording links, access dates, and relevant conclusions.
- Separate verified facts, inferences, hypotheses, and open questions.
- Produce a bounded specification; do not plan implementation in detail and do not modify product code.
- Grill the user only on decisions that materially change scope, behavior, risk, or acceptance criteria.

## Evidence

Create or update `reports/validate/<DATE-slug>/01-spec.md` with:

- request and scope;
- current-state evidence;
- external research and sources;
- requirements and acceptance criteria;
- assumptions, exclusions, risks, and unresolved decisions;
- handoff readiness for planning.

Do not claim readiness while material questions remain.
