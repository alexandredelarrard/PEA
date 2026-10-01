# FTD on-the-fly publication clock — harness run

- Input: 2026-10-01 user request to remove `sec_ftd_vintages` dependencies and derive FTD feature as-of dates on the fly. Follow-up: historical ZIPs use period end plus 15 calendar days, advanced past weekends; a newly downloaded latest expected ZIP uses today.
- Input digest: SHA-256 `45c0e30be91077d6296cf6e268ad4c0aba3c8dd6d788d77875ba8f21281a077c` of the normalized intent `FTD on-the-fly publication clock; historical ZIP period end+15 weekend; latest cached ZIP mtime if less than 2 days; remove sec_ftd_vintages`. No input file was supplied.
- Repository: `C:/Users/de larrard alexandre/OneDrive - The Boston Consulting Group, Inc/Documents/repos_github/PEA`.
- Base branch/revision: `dev` at `7ce4ad7a857f973c84201a3599ff5a317f5e90f3`.
- Orchestrator branch/worktree: `harness/ftd-on-the-fly`, `reports/worktrees/ftd-on-the-fly`.
- Approved effects: local research/audit files, the isolated branch/worktree, and the scoped product, test, schema/DDL, and wiki edits in the user-approved plan. Database writes, physical table deletion, saved-cube rebuild, push, and PR were not approved or performed.
- Constraints: preserve unrelated dirty files `src/data_extract/cli.py` and `src/data_extract/utils/fundamentals/fundamentals_employees.py` in the base worktree; no production DB mutation, dependency installation, push, or PR.
- Decisions: user approved `01-spec.md` for planning, including the highest-stored-period-within-60-days rule and leaving the physical table unused. A live table drop remains outside approved effects.

| Stage | Status | Evidence |
| --- | --- | --- |
| Spec research | approved | `01-research.md`, `01-spec.md`; user decision 2026-10-01 |
| Plan | approved and amended | `02-plan.md`; user approved exact scope, then fixture, correctness, and validation-scope amendments were recorded |
| Implement | complete | `03-implementation.md`; 41 focused tests and aggregate regression passed |
| Validate | complete with saved-data findings | `04-validation-01.md`, independent reviews, `06-final.md`, and self-contained `report.html` |

## Iteration ledger

1. 2026-10-01: initialized a clean local harness worktree from `dev`; preserved two unrelated dirty files in the base worktree.
2. 2026-10-01: user clarified historical and newly downloaded latest-ZIP date rules; asked which durable non-table evidence may be used on later rebuilds.
3. 2026-10-01: user selected the cached ZIP timestamp inside a strict two-day freshness window; asked whether live physical table deletion is in scope.
4. 2026-10-01: read-only live probe measured 1,079,359 FTD rows, 413 periods, latest settlement 2026-09-14; a projected 2024 sample confirmed 1,443 day-15 rows in `b` ZIPs. Drafted research and specification.
5. 2026-10-01: defaulted to leaving the physical table unused absent destructive-action authorization; asked which period qualifies for the recent-cache override.
6. 2026-10-01: after allowing time for optional clarification, specified the highest stored period within 60 days as the latest expected FTD ZIP; prepared `01-spec.md` for the required Harness Spec approval gate.
7. 2026-10-01: user replied `approve`; spec approved for planning. The 60-day latest-period definition and no-live-drop default are accepted as written.
8. 2026-10-01: drafted `02-plan.md`, independently reviewed it, and corrected the wiki access method, fixed test paths, exact validator commands, before/after comparison, explicit read-only cache input, and phase gates. Awaiting plan and AGENTS.md risk-zone approval before product edits.
9. 2026-10-01: user approved the exact plan and risk-zone edits. Captured read-only real-data baseline, ran meaningful RED tests, removed code dependencies, and obtained 40/40 focused GREEN tests.
10. 2026-10-01: scoped candidate real-data check passed (42,145 projected 2024 rows; 1,443 day-15 b ZIP rows, no conflicting settlement dates). Broad checks exposed a legacy fingerprint fixture without source `period`; 30 other checks passed, fixture repair and aggregate rerun ongoing.
11. 2026-10-01: repaired only the synthetic fixture and reran aggregate regression: 5 tests passed, all 37 outputs and 1,749 hashes matched. Independent correctness review found a partial-universe latest-ZIP bug; fixed it by using global stored periods and added a targeted test. Re-review found no remaining issue; 41 focused tests passed.
12. 2026-10-01: candidate live-data comparison passed. Read-only saved-cube grain, profile, and bounds passed; coverage, catalogue, and leakage found existing saved-data issues. Exact redundancy was stopped at 600,000/3,265,378 rows; clip and timeseries were not run after the critical unrelated leakage finding. Updated six wiki subject pages and log through OpenKnowledge; completed reviews and final reports.

## Finding ledger

- Resolved: FTD extraction wrote ZIP availability metadata, aggregation loaded it, and feature publication could fall back to settlement-date inference. The candidate derives from persisted `period`, globally scoped latest-period membership, and an optional fresh cache timestamp, without the metadata table.
- Open outside this change: the unchanged saved institutional cube has stale coverage, undocumented legacy columns, and a critical beneficial-ownership leakage finding. See `04-validation-01.md`.

## Report index

- `00-run.md`: this run state.
- `01-research.md`: source and live evidence.
- `01-spec.md`: user-approved specification.
- `02-plan.md`: user-approved implementation plan and dated amendments.
- `03-implementation.md`: implementation evidence.
- `04-validation-01.md`: candidate and read-only saved-cube validation.
- `05-correctness-review.md`, `05-ponytail-review.md`: independent reviews.
- `06-final.md`, `report.html`: final handoff and self-contained report.
