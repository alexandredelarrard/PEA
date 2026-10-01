# Research: derive FTD publication dates on the fly

## Request and questions

The user wants `sec_ftd_vintages` removed from the FTD path. Historical ZIPs should use source-period end plus 15 calendar days, advanced past weekends. A newly downloaded latest expected ZIP should use its cached ZIP timestamp only while that timestamp is less than two calendar days from today. This research asks where the period and cache evidence live, which readers depend on the metadata table, and how to preserve point-in-time behavior.

## Repository and sources

- Repository: PEA; base `dev` at `7ce4ad7a857f973c84201a3599ff5a317f5e90f3`; research date 2026-10-01.
- Repo contracts: `AGENTS.md`; `wiki/architecture/system-overview.md`, `wiki/reference/table-catalog.md`, `wiki/reference/data-sources.md`, `wiki/reference/live-database.md`, `wiki/concepts/source-availability.md`, `wiki/modules/data-extract.md`, `wiki/modules/data-aggregate.md`, `wiki/guides/data-access.md`, `wiki/guides/testing.md`, and `wiki/guides/run-the-pipeline.md` (OpenKnowledge MCP reads).
- Source: `src/data_extract/utils/institutionals/fetch_fails_to_deliver.py`, `src/data_extract/utils/common/bulk_cache.py`, `src/data_store/schema.py`, `src/data_aggregate/transformers/step_cube_institutionals.py`, `src/data_aggregate/utils/institutionals/short_flow_features.py`, `scripts/institutionals_feature_quality.py`, `sql/schema.sql`, and their named tests. A bounded read-only source audit independently enumerated these references.
- Prior evidence: `reports/validate/2026-09-23-ftd-regsho-symbol-repair/report.md` records an earlier FTD rebuild; the 2026-09-30 wiki live-database snapshot records a later 1,079,328-row and 413-period audit. These are dated observations, not a current live measurement.

## Current path and observed behavior

1. `_periods` makes `YYYYMMa/b` ZIP tags. `ensure_zip` streams missing SEC archives to cache and calls `on_download` on HTTP 200. `_parse_ftd` reads settlement rows; extraction stamps the source `period` before saving `Tables.sec_fails_to_deliver` (`fetch_fails_to_deliver.py:59-70,197-276`; `bulk_cache.py:47-93`).
2. The FTD table grain is ticker × settlement date and its projection already includes `period` (`schema.py:243-250`). `ingested_periods` resumes from distinct values of that column (`bulk_cache.py:173-189`). No extraction resume path requires the vintage table.
3. Extraction separately backfills or writes `Tables.sec_ftd_vintages` (`fetch_fails_to_deliver.py:215-219,253-262`). Its `_vintage_frame` chooses either period-end +15/weekend estimate or a recent successful HTTP date (`:128-155`). The registry and generated SQL declare this extra table (`schema.py:251-260`; `sql/schema.sql:193-203`).
4. The institutionals cube step loads both tables and fails if FTD rows exist without vintage metadata (`step_cube_institutionals.py:450-459`). The feature builder passes that metadata to `_publish_ftd_vintages`, which publishes the ZIP's latest cumulative settlement state on the first trading session at or after availability (`short_flow_features.py:144-209,407,430,442,499`). A quality script also enumerates the table (`scripts/institutionals_feature_quality.py:61`).
5. The feature fallback computes a period from settlement day when metadata is absent (`short_flow_features.py:137-190`). This is unsafe for a `b` ZIP containing a day-15 settlement. The dated 2026-09-30 live snapshot counted 36,299 such rows across 141 `b` ZIPs and found no settlement date shared by two periods; source-period identity is required. The current code checks for duplicate date-to-period mapping when `period` exists.
6. Baseline command in the clean harness worktree: `rtk <project Python 3.13> -m pytest tests/data_aggregate/test_short_flow_features.py::test_ftd_observed_zip_date_overrides_historical_estimate tests/data_aggregate/test_short_flow_features.py::test_ftd_publication_uses_persisted_zip_period_at_half_month_boundary -v -s`. Result: 2 passed; printed sanity checks confirm stored-date override and the day-15 `b` ZIP boundary. This verifies the existing behavior, not the requested replacement.
7. A read-only, projected 2024 `context.store.load(Tables.sec_fails_to_deliver, columns=['date','period','ticker'], since='2024-01-01', until='2024-12-31')` returned 42,145 rows; 1,443 had settlement day 15 in a `b` ZIP. Named examples include `AAPL` on 2024-04-15 in `202404b` and `A` on 2024-05-15 in `202405b`. This independently confirms that settlement-day half inference is wrong on real data.

## Constraints and risk zones

- Source code under sibling `src/` packages must not cross-import. Any cache access needed by aggregation must use its own context/path boundary.
- Tabular reads use `context.store`; cube-sized reads are projected and streamed. No new table or secondary SQL access should be introduced.
- `src/data_store/schema.py` and `sql/schema.sql` are risk-zone edits requiring approval under `AGENTS.md`. A live `DROP TABLE` is a separate destructive database action and is excluded by the approved specification.
- The `sec_fails_to_deliver` table's existing `period` is authoritative. `_canonicalise_ftd` carries `period='first'` when it groups on ticker/date; this is a pre-existing ambiguity if two ZIPs share a settlement date. The dated audit found no such live overlap. Preserve the existing fail-closed feature check rather than broadening the extraction task without evidence.
- Historic dates are estimates, not verified SEC posting times. The user-selected two-day cache rule deliberately makes a just-downloaded latest ZIP's as-of date time-dependent: a rebuild after the window can fall back to the estimated date. This consequence needs explicit acceptance in the spec.
- The isolated harness worktree has no copied ignored `data/` ZIP cache. Real-cache validation must point read-only at the existing base-worktree cache or explicitly record that the cache is absent; it must not manufacture a new observation timestamp.
- The base worktree's existing cache contains 413 ZIPs. Its newest file is `cnsfails202609a.zip`, modified 2026-09-29 18:50 New York time; the next newest are August `a`/`b`. On the research date (October 1) that newest timestamp is two calendar days old and fails the requested strict `<2` rule. The newest completed calendar half is September `b`, while the newest actually cached ZIP is September `a`. The specification defaults `latest expected` to the highest stored period within the existing 60-day recency bound; this assumption is visible for approval.
- Existing source tests for metadata backfill, first HTTP observation, and observed-date overrides describe the table-based design and must be replaced with checks for the chosen on-the-fly rule.

## Fact / inference / open decision

- **Fact:** the FTD table already stores each row's ZIP period; the cube loader currently reads the vintage table; the cache contains ZIP files and `ensure_zip` stamps a new download.
- **Fact:** a 2026-10-01 read-only `context.store.row_count`, `distinct(period)`, and `max_date` probe measured 1,079,359 FTD rows, 413 periods, and latest settlement date 2026-09-14. The two focused baseline tests passed on 2026-10-01.
- **Inference:** historical publication dates can be derived without the vintage table from `period` and the trading index while keeping current feature values for periods whose stored vintages were estimated.
- **Decision:** remove code/DDL dependencies while leaving the existing physical local PostgreSQL table unused. The approved specification excludes a destructive drop.
- **Pending measurement:** post-change data validation. The 2026-09-30 wiki snapshot's 36,299 mismatch count is dated and is not represented as a current count.

## Candidate acceptance boundaries

Remove all runtime and source references to `Tables.sec_ftd_vintages`; derive historical dates from the persisted ZIP tag, applying weekend then market-session advancement; apply the user-selected current-ZIP cache timestamp rule; preserve cumulative state, null/zero, and identity behavior; validate against focused tests, real scoped data, read-only institutionals checks, and the aggregate fingerprint without changing its baseline absent approved evidence.
