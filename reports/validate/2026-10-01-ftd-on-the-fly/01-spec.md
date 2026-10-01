# Approved specification: FTD publication without a vintage table

Status: **approved for planning on 2026-10-01**. Research: [01-research.md](./01-research.md). The user's `approve` response accepted this specification, including the highest stored period within 60 days and leaving the physical table unused.

## Outcome

Build FTD features directly from each row's source ZIP period and the existing ZIP cache when its newest relevant file has just arrived. The separate `sec_ftd_vintages` table ceases to be a producer, prerequisite, or consumer.

## Current state

`sec_fails_to_deliver` already stores `period`, but extraction writes a second table and the cube step refuses FTD rows without it. Current feature code can infer a half-month from settlement day when metadata is absent; a real 2024 scoped sample includes 1,443 day-15 rows from `b` ZIPs. The 2026-10-01 live probe measured 1,079,359 FTD rows and 413 source periods. Two current-behavior baseline tests passed (see research report).

## In scope

- FTD extraction, its feature publication and cube loader, table registry/generated DDL, quality-script dependency, affected tests, and targeted wiki guidance.
- Read-only checks against existing PostgreSQL rows and ZIP cache, plus the required harness evidence.
- The existing physical local table is left unused unless the user explicitly authorizes a drop.

## Out of scope

- Re-extracting all SEC ZIP payloads, changing FTD identity resolution or settlement balances, retraining models, editing the aggregate fingerprint baseline without separate approval, and changing unrelated SEC notes or pension clocks.
- Dropping the physical local table, production database mutation, push, PR creation, or deployment unless separately authorized.

## Requirements

- **REQ-001:** No FTD runtime path reads, creates, writes, or requires `sec_ftd_vintages`. Remove its registry and generated DDL declaration and quality-script dependency; retain `sec_fails_to_deliver.period` as the source-period tag.
- **REQ-002:** For a historical FTD ZIP, parse its `YYYYMMa/b` tag, take day 15 for `a` or calendar month end for `b`, add 15 calendar days, and advance a Saturday/Sunday result to Monday. A feature becomes visible on the first trading session on or after that date, including holiday adjustment.
- **REQ-003:** The newest expected FTD ZIP is the highest stored `period` whose period end is between today and 60 calendar days before today. For only that ZIP, use its cached file modification date in New York when that timestamp is no more than one calendar day before the build's New York today (`0 <= today - cache_date < 2 days`). A missing, future, or older cache timestamp uses REQ-002. This defaults to the existing 60-day recent-period bound and prevents an old backfill from receiving a new as-of date.
- **REQ-004:** The source ZIP tag, never settlement day, assigns rows to a publication event. Missing/malformed period or a settlement date mapped to multiple ZIP periods fails visibly. No event can be published before its latest settlement date.
- **REQ-005:** Publish one latest cumulative balance per ZIP; preserve current held-state, unavailable `NaN`, and observed-zero semantics. Existing extraction resume, full rebuild safety, source identity, and peer-relative feature behavior remain valid without vintage metadata.
- **REQ-006:** A feature rebuild does not need the vintage table, and the latest-ZIP two-day rule is documented as time-dependent: after the timestamp ages out, the estimated date may replace the short-lived timestamp date.

## Acceptance criteria and evidence

- **AC-001:** A source reference audit and an absent-vintage-table integration test show no FTD producer, loader, feature API, script, registry, or generated DDL dependency. Focused extraction and cube tests pass.
- **AC-002:** Known-truth cases prove `a`/`b` period-end +15, weekend roll, holiday trading-session roll, and strict 0/1/2-day cache threshold. Each test prints a sanity conclusion.
- **AC-003:** A real, projected FTD sample containing `AAPL` 2024-04-15 in `202404b` proves the feature cannot publish this ZIP's state at the `a` date; no post-availability value appears early. A read-only validator checks current populated data.
- **AC-004:** Missing/invalid `period`, duplicate settlement-to-period mapping, missing cache, and future cache timestamp produce the declared fail-closed/fallback behavior. Extractor full/incremental tests retain resume and replacement safety.
- **AC-005:** The institutionals feature and aggregate regression checks show only explained output changes, with no silent loss of unrelated feature columns or changed fingerprint baseline.
- **AC-006:** A targeted source-grounded wiki update and final self-contained harness report describe the new rule, cache limitation, and exact validation results.

## Data invariants and limits

- Grain remains ticker × settlement date for `sec_fails_to_deliver`; `period` is the ZIP tag, not an added table key.
- Historical dates are estimates, not verified SEC first-publication dates. The cache timestamp is local file metadata and can be changed by copying or touching the file. No claim of immutable first observation is made.
- The strict two-day override can change historical replay after aging out. This is the user's selected rule and will be measured/documented rather than hidden.
- The latest available source row is a cumulative balance; ZIP rows are not summed into a publication event.

## Permissions and open decisions

1. **Newest expected period:** the approved definition is the highest stored ZIP period within the existing 60-day recent-period bound. This is the period actually available to the feature builder; the newest completed calendar half can differ.
2. **Physical table:** retained unused by default. Drop only if the user explicitly authorizes that destructive local DB action.
3. Approval to edit `src/data_store/schema.py`, `sql/schema.sql`, and any needed constant/config is required by `AGENTS.md`; the later Harness Plan will list exact paths and request that approval before implementation.
