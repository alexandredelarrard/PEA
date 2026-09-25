# Final report contract

The run's primary human deliverable is:

`reports/validate/<YYYY-MM-DD>-<slug>/report.html`

It is a self-contained static summary backed by the Markdown and machine evidence in the same directory. `06-final.md` is the canonical diffable source; the HTML must not invent, soften, or omit material evidence.

## Required source set

Read every existing applicable source before rendering:

- `00-run.md`
- `01-research.md`
- `01-spec.md`
- `02-plan.md` and any `plans/*.md`
- `03-implementation.md`
- all `04-validation-<NN>.md`
- `defects.md`
- all `05-review-<NN>.md`
- `05-simplification-plan.md` when present
- all `fixes/*.md` and `merges/*.md`
- `06-final.md`
- relevant `_out/*.json` and plots

Missing optional files are labelled not applicable. Missing required stage files block `STRONG`.

## 06-final.md

Write this first. Include:

1. Verdict and one-paragraph rationale.
2. Scope, date, repository, base revision, final candidate revision, branch/worktree, and validation timestamp.
3. Stage timeline and user approvals.
4. Acceptance-criterion matrix: criterion, implementation evidence, validation evidence, status, and limitation.
5. Code, data, model/strategy, and operational lane summaries as applicable.
6. Validation attempts with exact command/query, exit, decisive observation, and artifact.
7. Every finding and its final disposition, fix branch/commit, merge, and integrated verification.
8. Correctness and Ponytail/simplicity review results.
9. Abstentions, skips, pre-existing failures, assumptions, and residual risks.
10. Commit and merge ledger.
11. PR handoff recommendation and actions explicitly not taken.
12. Complete relative-path report index.

## report.html

Use `assets/report-template.html` when useful. Prefer existing repository rendering support; otherwise use a small standard-library-only renderer in `_scripts/render_report.py` when repeatability is valuable. Do not add a dependency only to make this report.

The HTML contains:

- A prominent `STRONG`, `CONDITIONAL`, or `NOT READY` banner.
- Scope and immutable base/final candidate identities.
- Executive summary and PR recommendation.
- Stage timeline with links to each stage report.
- Acceptance-criterion matrix.
- Applicable code, data, modelling, and operational evidence.
- Validation command table with exit/status and artifact links.
- Finding/disposition table with severity, mechanism, fix commit, and verification.
- Correctness and simplification review summary.
- Limitations, abstentions, skips, and residual risk.
- Commit/merge ledger.
- Relative links to every detailed Markdown report, JSON result, script, and plot.

Inline CSS and any essential icons. Do not depend on a CDN, network asset, external font, or live JavaScript. Escape evidence before inserting it; put commands and logs in `pre/code`. Keep detail readable in the linked Markdown instead of copying full logs into HTML.

## Integrity check

After generating the HTML, run a deterministic report check and record it in `06-final.md` and `00-run.md`:

- File opens as HTML and contains exactly one final verdict.
- Base and candidate revisions match the final tree.
- Every acceptance criterion from `01-spec.md` appears exactly once in the matrix.
- Every critical/high finding from `defects.md` has a visible disposition.
- Every referenced relative file exists; no link escapes the run directory.
- All required stage reports appear in the index.
- No placeholder token remains.
- No remote resource is required.
- HTML verdict and counts match `06-final.md`.

A report-only correction reruns this integrity check. A code, data, configuration, or model change reruns all affected validation and review before rendering again.
