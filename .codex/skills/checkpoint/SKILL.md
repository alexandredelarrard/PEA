---
name: checkpoint
description: Use only when the user explicitly invokes Checkpoint to save a
  concise, resumable record of the current delivery state.
---
# Checkpoint compatibility workflow

Use this skill only on explicit invocation.

## Load the source contract

Read `.claude/commands/checkpoint.md` through the repository-authorized Markdown tooling, then inspect the current run's specification, plan, implementation, validation reports, working-tree state, and unresolved decisions.

## Write a resumable checkpoint

Create or update `reports/sessions/session-<task>-<YYYY-MM-DD>.md` with:

- objective and accepted scope;
- current phase and completed work;
- material decisions and why they were made;
- changed files and branch/commit state;
- tests and validation already run, with outcomes;
- active defects, risks, blockers, and user approvals still needed;
- exact next actions and commands;
- links to the active `reports/validate/<DATE-slug>/` evidence.

Keep it concise, factual, and self-contained. Never claim a check ran when it did not. Do not clear context, commit, push, merge, or change implementation merely because a checkpoint was requested.
