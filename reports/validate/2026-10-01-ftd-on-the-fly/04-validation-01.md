# FTD on-the-fly validation

Date: 2026-10-01. Candidate: `harness/ftd-on-the-fly` worktree based on `dev` at `7ce4ad7`. All database reads were read-only. The existing cube part and PostgreSQL vintage table were not changed.

## Candidate behavior

The read-only `_scripts/check_ftd_candidate.py` used `context.store` to project `sec_fails_to_deliver` rows and the existing local ZIP cache path as an explicit read-only input. [Candidate output](candidate-ftd.json) contains:

| Check | Result |
| --- | --- |
| AAPL April–May 2024 | 21 source rows across `202404a`, `202404b`, `202405a`, `202405b`; 10 declared feature columns exactly matched [baseline](baseline-ftd.json). |
| April 15 `202404b` row | The `b` ZIP publishes May 15. AAPL's held balance is 3,438 on May 14 and 5,200 on May 15 and 16; April 29 is unavailable. This matches the captured vintage-table baseline without reading that table in the candidate. |
| 2024 period lineage | 42,145 projected FTD rows; 1,443 day-15 rows belong to `b` ZIPs; zero settlement dates belong to multiple ZIPs. |
| Latest ZIP clock | Globally latest stored period `202609a`; cache mtime `2026-09-29T18:50:19.427469-04:00`. On October 1 it is two New York calendar days old, so the strict `<2` rule rejects it and derives September 30 from period end plus 15 days. |

The known-truth partial-universe regression covers a ticker with only `202608b` rows while the global source also stores `202609a`: a fresh older cache must retain September 15 rather than move to October 1.

## Code and regression checks

| Command/check | Outcome |
| --- | --- |
| Focused pytest: `test_short_flow_features.py`, `test_institutional_step_contract.py`, `test_fails_to_deliver.py`, unique `--basetemp` | **PASS: 41 tests** with sanity conclusions, including 0/1/2-day cache ages, weekend/holiday roll, invalid/ambiguous periods, extraction resume/full replacement, and global latest-period selection. An earlier rerun without isolated `--basetemp` passed assertions but exited on an external pytest temp-directory permission error; the isolated rerun exited 0. |
| Aggregate fingerprint regression | **PASS: 5 tests**; all 37 outputs and 1,749 hashed columns matched the protected baseline. The synthetic FTD source fixture was given its required `period`; the fingerprint baseline was not edited. |
| Other store/cube/registry tests in broad run | **PASS: 30 tests**. The initial two aggregate errors were solely the synthetic fixture's missing `period`, repaired before the dedicated aggregate regression rerun. |
| `git diff --check` | **PASS**. Source reference search found no runtime, script, registry, or SQL reference to `sec_ftd_vintages`; remaining `ftd_vintages` text is the in-memory publication helper name and a negative test assertion. |
| Wiki lint | Six edited subject pages returned zero errors/warnings. `wiki/log.md` returned no problems and reported no checks configured for that reserved log. |

## Read-only persisted-cube validators

The `src validate pull -T cube_part_institutionals` snapshot contains 3,265,378 rows and 491 tickers through September 4, 2026 (400.7 MB Parquet; 754 seconds to pull). Each completed validator used that snapshot except `bounds`, which intentionally reads database float64 values. These checks describe the **unchanged saved cube**, not the candidate in-memory FTD series.

| Validator | Exit/status | Evidence and interpretation |
| --- | --- | --- |
| [grain](pull/_out/grain.json) | 0 / PASS | No findings. |
| [coverage](pull/_out/coverage.json) | 1 / FAIL | 37 findings, worst score 8; saved cube stops September 4 while the price reference has later September trading sessions. Existing saved data is stale. |
| [profile](pull/_out/profile.json) | 0 / PASS | No findings across 120 checked columns. |
| [bounds](pull/_out/bounds.json) | 0 / PASS | No findings on declared numeric bounds. |
| [catalogue](pull/_out/catalogue.json) | 1 / FAIL | 40 undocumented saved columns, worst score 4, including retired FTD legs in the saved cube. Current source emission is unchanged at 10 declared short-flow legs. |
| [leakage](pull/_out/leakage.json) | 1 / FAIL | One critical existing beneficial-ownership finding: `f_ic_bo_holder_count` is non-null before first source event for 232 tickers (worst TKO). The validator tested six persisted FTD legs without an FTD finding. Its label-horizon half abstained because this table declares no `label_pattern`. |
| redundancy | INCOMPLETE | Exact all-row scan was stopped at 600,000 of 3,265,378 rows after several minutes; no result was produced. |
| clip, timeseries | NOT RUN | Further saved-cube scans were stopped after the critical unrelated leakage finding and the already established stale-cube state. They would not evaluate candidate FTD output. |

## Decision and limits

The candidate FTD implementation satisfies the targeted date, cache, extraction, and feature compatibility checks. Independent correctness re-review found no remaining issue. The overall persisted institutional cube is **not clean for production/model readiness** because of the existing coverage, catalogue, and beneficial-ownership leakage findings; this change does not fix or rewrite that cube. The old physical `sec_ftd_vintages` table may remain in PostgreSQL, unused by product code. Historical `+15` dates are estimates rather than verified SEC first-publication timestamps.
