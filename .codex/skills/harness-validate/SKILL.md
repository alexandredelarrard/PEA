---
name: harness-validate
description: Independently validate code, data, and modelling outcomes; run
  correctness and Ponytail reviews; and publish Markdown evidence plus a
  self-contained HTML readiness report.
---
# Harness Validate

Independently decide whether an implemented change is strong enough for PR handoff. Validate affected code, data, and model behavior; then review correctness and simplicity. Start read-only and return findings for correction rather than hiding them.

## Inputs and exact outputs

Require `01-spec.md`, `02-plan.md`, `03-implementation.md`, the base revision, and the candidate revision or diff. Reuse their exact run directory:

`reports/validate/<YYYY-MM-DD>-<slug>/`

Write only these audit artifacts and their numbered successors:

- `04-validation-<NN>.md`: one domain-validation attempt.
- `defects.md`: live finding register, highest severity first.
- `05-review-<NN>.md`: one post-validation correctness and simplicity review.
- `05-simplification-plan.md`: accepted review fixes, only when changes are required.
- `06-final.md`: canonical final evidence summary.
- `report.html`: self-contained human-readable final deliverable derived from the Markdown evidence.
- `_out/`, `_cache/`, `_scripts/`, and `plots/`: machine evidence where applicable.

Never write generated evidence under source directories. Use the repository-authorized Markdown mechanism for in-scope Markdown; when the run directory is excluded from that system, use the normal repository-authorized file tools.

## Independent validation

Use a fresh reviewer context when available. The implementer may supply facts and commands, but its conclusions are not evidence. Re-run checks against the exact candidate revision and record every exit code.

Select every applicable lane; mixed work uses multiple lanes.

### Code and refactor lane

- Map the diff and tests to every acceptance criterion.
- Inspect affected callers, contracts, errors, compatibility, state transitions, and risk boundaries.
- Run targeted tests, architectural guards, static checks, and the relevant broader suite.
- Exercise the real entry point or a representative smoke path when unit tests cannot prove integration.
- Compare before/after behavior where regression or performance risk is material.
- Record skips, warnings, flakes, environment limits, and the sanity conclusion.

### Data lane

Before any data validation, read `references/data-validation.md` completely and follow it. Measure first and inspect code only to explain measured suspects.

In PEA, run `src/validate` through the documented CLI before ad hoc queries. Pull one dated snapshot, run every applicable generic check, record its JSON and exit code, and name every exit code `3` as ABSTAINED with the missing declaration. A green test suite is not evidence that a populated financial table is sound.

For every material data finding, prove both:

1. Measurement: a rerunnable check/query plus a named entity/date example.
2. Mechanism: the producing source, extraction, aggregation, feature, or validation code path.

### Modelling and strategy lane

- Verify temporal split order, embargo, target alignment, feature availability, point-in-time integrity, and leakage controls.
- Reproduce seeds and artifacts when reproducibility is in scope.
- Compare with the accepted baseline using per-fold and aggregate out-of-sample metrics.
- Inspect prediction distributions, recent-versus-history stability, regimes, feature importance/SHAP where applicable, and artifact compatibility.
- Measure portfolio consequences where applicable: exposure, turnover, costs, concentration, and failure behavior.
- In PEA, enforce the modelling contracts in `docs/modelling.md`, including chronological splits, validation-row SHAP for boosters, printed out-of-fold metrics, and compatible artifacts.

A headline improvement with weaker stability, unexplained row loss, or stale evidence is a finding.

## Findings, scoring, and verdict

Use stable IDs such as `F-001`. Each finding records:

- Score 1–10 and severity.
- Affected acceptance criterion and scope.
- Observed evidence with exact command/query and artifact path.
- Expected behavior.
- Named root-cause mechanism or `not yet proven`.
- Impact and blast radius.
- Minimal required fix.
- A verification that fails before and passes after.
- Status, owner branch/commit when supplied, and final disposition.

Use these bands: 9–10 critical, 7–8 high, 4–6 medium, 1–3 info. Provability comes first, then affected rows/users, persistence, silent-failure risk, and blast radius. Several checks agreeing strengthens a finding; one check repeating the same signal does not create many independent findings.

Verdicts:

- `STRONG`: every required check measured and passed, no open critical/high finding, no unexplained abstention, criteria are evidenced, and final reviews are clear.
- `CONDITIONAL`: no blocking defect, but a named limitation, accepted abstention, or medium risk remains.
- `NOT READY`: a required check failed, a critical/high finding remains, material behavior is unproven, or evidence is stale.

Never infer `STRONG` from “tests pass” alone.

## Post-validation code review

Run only after domain validation has no blocking defect.

Perform two independent read-only passes when possible:

1. Correctness review: spec compliance, edge cases, security, data loss, compatibility, operational failure, and regression risk.
2. Simplicity review: invoke Ponytail Review when available. Otherwise do the same diff-only hunt for deletable code, reusable existing helpers or standard-library features, duplicate logic, speculative abstractions, dead flexibility, and unnecessary dependencies.

Verify review feedback against the repository before accepting it. Critical/high correctness findings and material simplification findings block closure. Put accepted changes in `05-simplification-plan.md`, each with scope, minimal patch, failing evidence when applicable, and focused/broader verification. Harness Run implements them through the same isolated finding-branch loop, then this skill reruns affected checks, domain validation, and both reviews.

## Final reports

After the exact integrated candidate has a current verdict and clear reviews, read `references/final-report.md` completely.

Write `06-final.md` as the canonical evidence source, then generate `report.html` from it and the other run reports using `assets/report-template.html` or a simpler repository-native renderer. The HTML is the primary document the user reads; it summarizes the decision and links to detailed Markdown and machine artifacts.

A product-code change makes domain checks and reviews stale. A report-only correction requires rerunning the report integrity check, not the product suite. Do not alter evidence to improve the verdict.

## Completion

Validation is complete only when the exact candidate tree has current domain evidence and reviews, `defects.md` reconciles every finding, `06-final.md` matches that evidence, and `report.html` passes the integrity checks. Stop before push or PR creation.
