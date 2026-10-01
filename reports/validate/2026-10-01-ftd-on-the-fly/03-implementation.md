# FTD on-the-fly clock — implementation evidence

Status: implementation complete on isolated branch `harness/ftd-on-the-fly`. Approved [plan](./02-plan.md); no PostgreSQL or `data/` write.

## RED and baseline

- Read-only AAPL 2024-04-01..2024-05-31 baseline: 21 FTD rows, including one 2024-04-15 row explicitly tagged `202404b`; 10 declared short-flow feature columns. Old in-memory cumulative balance was absent on 2024-04-29, 3,438 on 2024-04-30 and 2024-05-14, and 5,200 on 2024-05-15/16. Raw JSON: `baseline-ftd.json`.
- `test_ftd_day_15_b_zip_uses_its_period_without_metadata` failed as intended: the old settlement-day fallback published a day-15 `202404b` state before 2024-05-15. Command used the approved Python and `--basetemp=../../pt-ftd-red-20261001` in the isolated worktree; 1 failed.
- New strict cache-window and invalid-period tests failed as intended because `_ftd_available_dates` did not exist. `--basetemp=../../pt-ftd-red-cache-20261001`; 2 failed.

## Changed behavior

- `sec_fails_to_deliver.period` is the publication event key; no settlement-day ZIP inference remains. A helper computes period-end +15 days and rolls weekends, then may use only the globally newest eligible cached ZIP's New York mtime on day 0/1. Two FTD legs share one derived date mapping per build.
- Extractor keeps its existing parse, identity, resume, full replacement, and stored `period` behavior but no longer backfills or writes availability metadata. The cube step reads FTD rows and passes its cache path without loading a second table.
- Removed the vintage table registry entry, its generated SQL block, and the quality script requirement. The legacy live physical table was not touched.
- `scripts.generate_schema_sql` initially rewrote unrelated hand-maintained SQL comments/indexes and live-reflected columns. Restored only that generated file from HEAD and removed exactly the approved vintage-table block. This avoids unrelated DDL drift.
- The synthetic aggregate fingerprint fixture omitted `period`. The plan's test-fixture scope amendment adds the tag to its generated FTD rows; the protected fingerprint baseline file remains unchanged.
- Independent correctness review identified that a ticker-scoped source could select an older cached ZIP as its latest. The step now passes global stored periods from `context.store.distinct`; a regression test proves the older ZIP retains its historical date. Re-review found no remaining issue.

## GREEN and remaining checks

- Focused extractor, short-flow, and institutionals step contract: **41 passed** after the partial-universe fix, with sanity conclusions. The first rerun without a unique pytest temp root hit an external temp-directory cleanup permission error after assertions passed; a unique `--basetemp` rerun exited 0.
- Broader boundary/incremental/registry run: **30 passed, 2 fixture setup errors in 296.37 s**. Both came from the synthetic missing `period`; after fixture correction the full aggregate regression module passed **5 tests**, with 37 output fingerprints and 1,749 hashed columns matching the baseline.
- Read-only real-data candidate: **passed** for 21 AAPL rows. The 2024 projected sample had 42,145 rows, 1,443 day-15 `b` ZIP rows, and zero settlement dates assigned to multiple ZIPs. AAPL 2024-04-15 in `202404b` published on 2024-05-15, with no early `b` state; 10 declared columns and scoped old/new cumulative values match. Latest stored period `202609a` has ZIP mtime 2026-09-29 18:50:19 New York; on 2026-10-01 that is two days old, so estimated availability 2026-09-30 applies. Raw JSON: `candidate-ftd.json`.

## Handoff

The source/wiki reference audit, OpenKnowledge documentation sync, independent correctness/Ponytail reviews, read-only validation, and HTML report are complete. See `04-validation-01.md` for saved-cube findings and incomplete broad scans; those existing data issues are outside this FTD code change.
