# FTD on-the-fly availability: final handoff

## Outcome

**FTD code is ready for review. The persisted institutional cube is not clean for model/production readiness.** The candidate removes every product dependency on `sec_ftd_vintages`. Historical availability is derived from the ZIP's stored `period` end plus 15 calendar days, rolled past weekends; the first session on/after that date is the feature `as_of`. Only the globally latest eligible stored period may use an existing ZIP cache mtime when its New York date is today or yesterday. No DB or cache writes were made.

## Acceptance evidence

| Criterion | Outcome |
| --- | --- |
| AC-001: no vintage-table dependency | Extractor write/backfill, cube read, table registry, bootstrap DDL, and quality-script requirement removed. Source search has no runtime table reference. The legacy physical table was not dropped. |
| AC-002: date and cache rule | Known-truth tests pass for `a`/`b`, day-15-in-`b`, weekends, holidays, cache ages 0/1/2, missing/future cache, and partial-universe global latest ZIP. |
| AC-003: real-row point-in-time result | Read-only AAPL sample proves April 15 `202404b` row first publishes May 15; candidate balances and 10 feature names match the captured baseline. |
| AC-004: source integrity | Missing/malformed `period` and conflicting settlement ZIP assignment fail closed. Extraction resume, full replacement, and HTTP rerun checks pass. |
| AC-005: compatibility | 41 focused tests and 5 aggregate regression tests pass; 37 outputs and 1,749 hashes match the protected baseline. |
| AC-006: documentation and review | Six subject wiki pages and append-only log updated through OpenKnowledge. Independent correctness review found and then cleared one partial-universe bug. Ponytail review suggested deleting a distinct RED test; it was retained. Self-contained HTML report generated. |

See [validation details](04-validation-01.md), [correctness review](05-correctness-review.md), [simplicity review](05-ponytail-review.md), and [HTML report](report.html).

## Remaining limits

The saved cube was not rebuilt. Its read-only coverage validator reports stale September 2026 sessions, catalogue reports 40 legacy undocumented columns, and leakage reports a critical beneficial-ownership issue unrelated to FTD. Exact redundancy was stopped at 600,000/3,265,378 rows; clip and timeseries were not run once these saved-data findings were established. Historical `+15` dates are estimates, not verified SEC publication evidence. The physical `sec_ftd_vintages` table can remain in PostgreSQL but is unused by this candidate.
