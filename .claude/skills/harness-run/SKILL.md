---
name: harness-run
description: Run the complete spec-to-PR-ready harness with fixed audit files,
  isolated per-finding sub-agent branches, integrated revalidation, Ponytail
  review, and a final HTML report.
---
# Harness Run

Drive an accepted specification through planning, implementation, validation, isolated corrections, final simplification, and PR-ready handoff. The four stage skills are the contract; this skill coordinates state, Git isolation, sub-agents, integration, and audit artifacts.

## Required input and run directory

Require a specification path or complete specification content. Also accept an optional slug, base branch, and declared validation commands.

Use or create exactly:

`reports/validate/<YYYY-MM-DD>-<slug>/`

Use one stable lowercase kebab-case slug for the entire run. Never split research, planning, implementation, or validation into alternate report roots.

Immediately create `00-run.md` with input identity/digest, repository root, base revision, branch/worktree, approved effects, constraints, unresolved decisions, stage/status table, iteration ledger, finding ledger, and report index.

## Audit file map

Every key step leaves durable evidence here; no stage, fix, merge, review, or final decision exists only in chat.

```text
reports/validate/<YYYY-MM-DD>-<slug>/
├── 00-run.md
├── 01-research.md
├── 01-spec.md
├── 02-plan.md
├── plans/02-phase-<NN>-<slug>.md       # only when the plan needs splitting
├── 03-implementation.md
├── 04-validation-<NN>.md               # every validation attempt
├── defects.md                          # live finding register
├── 05-review-<NN>.md                   # every correctness/simplicity review
├── 05-simplification-plan.md           # only when review changes are accepted
├── fixes/<finding-id>.md               # one per isolated fix branch
├── merges/merge-<NN>.md                # one per integration batch
├── 06-final.md                         # canonical final evidence
├── report.html                         # primary human-readable deliverable
├── _cache/                             # dated data snapshot(s)
├── _out/                               # machine-readable check results
├── _scripts/                           # rerunnable report/table-specific checks
└── plots/                              # referenced figures
```

Use the repository-authorized Markdown mechanism. If OpenKnowledge is registered, all in-scope Markdown goes through its MCP tools; excluded report directories use normal authorized file tools.

## Stage sequence

Load and follow each skill completely; do not replace it with a summary.

1. Harness Spec: use `$harness-spec` to verify the supplied spec against current evidence. Produce `01-research.md` and `01-spec.md`, then obtain spec approval.
2. Harness Plan: use `$harness-plan`. Grill material decisions, produce `02-plan.md`, and obtain Approve plan and start work.
3. Harness Implement: use `$harness-implement`. Maintain phase state, implement the accepted plan, and produce `03-implementation.md`.
4. Harness Validate: use `$harness-validate` on the exact candidate tree. It owns domain validation, `defects.md`, correctness review, Ponytail/simplicity review, `06-final.md`, and `report.html`.

Update `00-run.md` after every stage, user decision, validation/review attempt, fix dispatch, commit, merge, rerun, and terminal decision. Do not start a later stage until the prior completion contract is met.

## Workspace and Git preflight

Before the first mutation:

1. Read repository instructions and determine the true root, current branch, base revision, status, diff, and existing worktrees.
2. Preserve user changes. Never reset, force-checkout, stash, discard, overwrite, or include unrelated paths.
3. Work on an orchestrator branch named `harness/<slug>`. Prefer a host-native isolated workspace; otherwise use an existing ignored worktree convention or a safely verified Git worktree.
4. If relevant uncommitted work is required as the baseline, ask how it enters the orchestrator branch. Isolate from HEAD when unrelated dirty changes can remain untouched.
5. Run and record baseline checks before implementation.

The invocation authorizes the described local branch, worktree, commit, and merge workflow. It does not authorize dependency installation, production data mutation, push, PR creation, deployment, publication, force operations, or branch deletion unless separately requested.

## Correction loop

When Harness Validate returns blocking findings, first group repeated symptoms by root cause. One cause is one finding; unrelated causes stay separate.

For every confirmed bug or blocking review finding:

1. Define a finding contract in `defects.md`: ID, score/severity, evidence, criterion, expected behavior, proven mechanism or hypothesis, exact scope, failing reproduction, focused/broader checks, and base commit.
2. Create a dedicated branch `fix/<slug>/<finding-id>` and isolated worktree from the latest integrated orchestrator commit.
3. Dispatch one fresh sub-agent to that branch. Give it only the finding contract, spec/plan/report paths, repository rules, allowed paths, base commit, branch, and worktree.
4. Require that sub-agent to:
   - Reproduce the finding and prove the root cause before editing.
   - Create or run a check that fails for the reported reason.
   - Implement one minimal root-cause fix.
   - Run focused, sibling, and required data/model checks.
   - Write `fixes/<finding-id>.md` with RED/GREEN evidence, changed paths, limitations, branch, and commit.
   - Commit only owned paths with a message tied to the finding.
   - Return branch, commit SHA, paths, and exact results.
5. Run independent sub-agents concurrently only when write scopes, state, data, and generated artifacts are disjoint. Schedule shared-state findings sequentially, still one branch per finding, from the latest integrated commit.
6. The main orchestrator reviews each diff, report, test evidence, and commit. Reject extra scope, symptom patches, missing RED/GREEN proof, or unowned files.
7. Merge verified branches into `harness/<slug>`, preserving commit provenance. The main orchestrator alone resolves conflicts by reconciling both finding contracts and the accepted plan.
8. Write `merges/merge-<NN>.md` with source branches/commits, target commit, conflict decisions, changed paths, union of checks, results, and remaining findings.
9. Run the union of branch checks and the full applicable integrated checks.
10. Re-run Harness Validate on the integrated tree, incrementing `04-validation-<NN>.md`. Branch-only success never resolves a finding.

For an accepted simplification finding, first write `05-simplification-plan.md`, assign a stable simplification finding ID, and use the same branch, sub-agent, verification, commit, merge, integrated-test, and revalidation protocol.

## Loop bounds and replanning

Continue toward a clear result, but stop and return to the user when:

- A fix needs unapproved scope, a risk-zone edit, dependency, data mutation, or external effect.
- The same root cause survives two integrated correction attempts.
- Three correction rounds finish without `STRONG` or an explicitly accepted `CONDITIONAL`.
- Architecture contradicts the plan or remaining fixes are guesses.
- A destructive, security-sensitive, push, PR, deployment, publication, or branch-deletion action is required.

Record the blocker, evidence, and recommended decision in `00-run.md`, `defects.md`, and the latest validation report.

## Final close

After domain validation has no blocking finding:

1. Run independent correctness review.
2. Run Ponytail Review when available, or the equivalent diff-focused simplification pass.
3. Plan and execute accepted simplifications through the correction loop.
4. Re-run focused checks, broader suite, every affected data/model validation, and both reviews on the exact final tree.
5. Have Harness Validate reconcile `defects.md`, write `06-final.md`, generate `report.html`, and run the HTML/report integrity check.
6. Update `00-run.md` with final candidate identity, verdict, report index, unresolved limitations, and actions not taken.
7. Commit final owned code and audit artifacts only when tracked and authorized. Never force-add ignored reports without approval.
8. Return final branch and HEAD, included fix and merge commits, check summary, verdict, limitations, and the `report.html` path.

The terminal state is a local orchestrator branch whose candidate and audit evidence are current and ready for the user's PR decision. Do not merge to the base branch, push, or submit a PR.

## Success contract

Return `PR READY` only when the approved spec and plan are linked, every acceptance criterion has current evidence, critical/high findings are resolved on the integrated tree, required code/data/model checks measured, no abstention is hidden, correctness and simplification reviews are clear, the HTML report matches the final candidate, and no owned change is unexplained.

Otherwise return `CONDITIONAL` or `NOT READY` with the exact next decision.
