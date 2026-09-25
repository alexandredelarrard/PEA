---
name: data-check
description: Use only when the user explicitly invokes Data Check to validate
  repository data work and publish evidence-backed conclusions.
---
# Data Check compatibility workflow

Use this skill only on explicit invocation.

## Load the source contract

1. Read `.claude/commands/data-check.md` through the repository-authorized Markdown tooling.
2. Read the applicable data schema, live database, data-source, convention, testing, and runbook documentation before execution.
3. Invoke `$harness-validate` and use its data-validation reference as the canonical validation workflow.
4. Preserve useful checks from the source command, while current repository rules and canonical report paths take precedence.

## Validate the data outcome

- Select checks from `src/validate/` before inventing one-off SQL or scripts.
- Keep validation read-only and use the repository's required execution path.
- Verify table identity, grain, primary-key uniqueness, required columns, nulls, types, ranges, categorical domains, date coverage, freshness, duplicates, joins, point-in-time correctness, leakage, and reconciliation as applicable.
- Compare upstream, stored, and downstream row/key/date coverage where possible.
- Exercise restart, idempotency, projection, and resume behavior when the change affects ingestion.
- For model inputs or outputs, add target/feature timing, split, metric, stability, and economic sanity checks.
- Distinguish PASS, WARN, FAIL, and NOT RUN. A successful command is not itself a sound result.
- Every executed test must include its sanity-check conclusion.

## Evidence

Write `reports/validate/<DATE-slug>/04-validation.md` with:

- candidate identity and validation scope;
- environment and commands;
- check-by-check expected versus observed results;
- quantitative evidence and source coverage;
- failures, warnings, limitations, and unrun checks;
- final strength verdict and remediation requirements.

If this is the closing validation stage, update the self-contained `reports/validate/<DATE-slug>/report.html` summary from the stage reports. Do not pass validation with unresolved material failures.
