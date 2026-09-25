---
name: harness-implement
description: Execute an approved harness plan phase by phase, maintaining living
  state and RED/GREEN evidence while preserving scope and handing off to
  independent validation.
---
# Harness Implement

Implement an approved Harness Plan phase by phase with the smallest correct change, visible state, and current verification. This stage does not perform independent final validation or claim PR readiness.

## Inputs and exact output

Require the approved `01-spec.md` and `02-plan.md`. Reuse their exact run directory:

`reports/validate/<YYYY-MM-DD>-<slug>/`

Maintain the phase markers in `02-plan.md` and produce `03-implementation.md` there. The report is updated after every phase and contains work performed, evidence, deviations, commits when authorized, and the exact handoff to Harness Validate.

Use the repository-authorized Markdown mechanism. If OpenKnowledge is registered, read and write all Markdown through its MCP tools.

## Preflight

1. Read the complete plan, spec, applicable repository instructions, and task-specific docs.
2. Enumerate every phase and its state: `⬜`, `🔄`, `✅`, or `⚠ BLOCKED`. Resume from evidence, not from memory.
3. Confirm repository root, base revision, current branch/worktree, approved write scopes, risk approvals, exact checks, and report path.
4. Inspect status and diff. Preserve all pre-existing user changes. Never stash, reset, discard, overwrite, or absorb unrelated work.
5. Run the plan's baseline checks using the repository's documented interpreter and command forms. Record pre-existing failures, skips, and environmental limits before editing.
6. Use the isolated workspace supplied by Harness Run. Do not create dependencies, branches, commits, network effects, data mutations, or external writes outside the approved plan.

If code or environment materially contradicts the plan, mark the phase `⚠ BLOCKED`, record Expected / Found / Consequence / Recommended adjustment, and return to Harness Plan. A harmless path or symbol correction may proceed only when it preserves approved behavior and is recorded.

## Phase loop

For each approved `⬜` phase:

1. Mark it `🔄` in `02-plan.md`; record start revision, goal, criteria, and baseline in `03-implementation.md`.
2. Read the complete files and every caller, producer, consumer, or contract relevant to the behavior.
3. Reuse an existing helper, registry, standard-library feature, platform capability, or installed dependency when it solves the need.
4. For a behavior change or defect, create or run the smallest real failing check first. Confirm it fails for the expected reason.
5. Implement the root-cause fix. Keep edits inside approved scope; do not bundle opportunistic cleanup or speculative abstractions.
6. Run the phase's focused check and inspect the output. Then run the specified sibling or integration checks.
7. Refactor only while checks stay green. Prefer deletion, direct code, and existing patterns.
8. Record changed paths, exact RED/GREEN commands and observations, sanity-check conclusion, decisions, and deviations in `03-implementation.md`.
9. Mark the phase `✅` only when its done conditions and checks are satisfied and the tree remains functional. Otherwise leave it `🔄` or `⚠ BLOCKED`.
10. Commit at the phase boundary only when the plan or Harness Run authorizes it. Stage owned paths only and record the commit SHA.

After each phase, make the plan and implementation report sufficient to resume in a fresh context without reconstructing state from chat.

## Domain obligations

Apply the relevant checks in addition to ordinary tests:

- Performance: benchmark the same representative workload before and after; report correctness and variance, not only the fastest run.
- Financial/data behavior: use real scoped data, multiple named examples, declared grain and clocks, and point-in-time checks. Synthetic fixtures are for parsing or known-truth mathematics.
- Model/strategy behavior: preserve temporal ordering, embargo, target alignment, reproducibility, and artifact compatibility; run the plan's out-of-sample checks.
- Cache behavior: exercise hit, miss, expiry, invalidation, stale data, and memory/cost constraints.
- Documentation/configuration: update alongside the code when included in scope and run their repository checks.

## Failures and recovery

When a check fails unexpectedly:

- Read the complete error and reproduce it.
- Separate pre-existing, environmental, flaky, and introduced failures with evidence.
- Trace the bad value or behavior to its origin before editing.
- Test one hypothesis at a time.
- Re-run the failed focused check before broader checks.
- After three unsuccessful fixes for the same root cause, stop and return to planning; do not stack a fourth guess.

If a required check cannot run, record why, what claim it prevents, and the next action. Unavailable, skipped, or abstained is not green. Do not skip a phase silently or start a dependent phase from a broken state.

## Implementation report contract

`03-implementation.md` contains:

- Spec, research, and plan identities; base revision; branch/worktree; start and end timestamps.
- Phase table with status, covered criteria, changed paths, start/end revision, and commit.
- Baseline, RED/GREEN evidence, exact commands, decisive output, and sanity conclusions.
- Plan mismatches and approved deviations with impact.
- Pre-existing, introduced, resolved, and remaining failures.
- Performance/data/model/cache evidence when applicable.
- Commits created and owned paths.
- Known limitations and the exact independent-validation handoff.

Do not paste noisy logs; link durable artifacts and preserve concise decisive output.

## Completion

This stage is complete when every approved phase is `✅`, focused and broader implementation checks are current, `03-implementation.md` is current, and no unapproved scope remains. State “ready for independent validation,” never “done” or “PR ready.”
