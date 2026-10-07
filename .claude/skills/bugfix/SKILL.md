---
name: bugfix-cluster
description: Use only when the user explicitly invokes Bugfix Cluster to triage
  and repair a structured cluster of related defects.
---
# Bugfix Cluster compatibility workflow

Use this skill only on explicit invocation.

## Load and verify the input

1. Read `.claude/commands/bugfix-cluster.md` through the repository-authorized Markdown tooling.
2. Read the supplied cluster/report and verify every required field and structured payload is present and valid.
3. Stop with a precise input defect if required fields or JSON are missing; do not guess the cluster contract.
4. Read the relevant repository documentation and the accepted harness plan before editing.

## Isolate and fix

- Reproduce each defect and identify its root cause.
- Group defects only when they share a root cause or atomic change boundary.
- Use one sub-agent per independent bug or coherent bug group when parallel fixes were requested.
- Each worker uses a dedicated branch/worktree, stays within its assigned scope, adds focused RED/GREEN evidence, validates the repair, and returns a clean commit plus handoff.
- The orchestrator reviews commits before integration, merges into one integration branch, resolves conflicts by intent, and reruns affected checks after every resolution.
- Do not let workers merge their own branches or broaden scope.
- Apply a simplification review after correctness: remove accidental complexity without changing accepted behavior.

## Evidence

Record defects, ownership, branch/commit identity, reproduction, root cause, fix, focused validation, integration result, conflicts, and residual risks in the active `reports/validate/<DATE-slug>/` run.

After integration, run `$harness-validate`. If material defects remain, create a new numbered correction cycle and repeat until clean or explicitly blocked.
