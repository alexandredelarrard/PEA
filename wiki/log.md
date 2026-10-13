---
title: Wiki Log
description: Append-only audit trail of wiki generation and refresh runs.
---

# Wiki Log

Append-only audit trail. Add one dated entry per generation or refresh run, recording the profile, the `source_commit` it was anchored to, and the coverage. The codebase-wiki skill describes the entry shape.

## 2026-09-25: generate

- Profile: internal/standard
- source_commit: 054f0e1
- Coverage: overview; five architecture areas; fourteen application, infrastructure, and test modules; five end-to-end flows; seven core concepts; five task guides
- Pages: [Overview](./OVERVIEW.md)
- Architecture: [System overview](./architecture/system-overview.md), [Data platform](./architecture/data-platform.md), [Orchestration and runtime](./architecture/orchestration-and-runtime.md), [Feature-to-portfolio pipeline](./architecture/feature-to-portfolio.md), [Point-in-time and quality controls](./architecture/point-in-time-and-quality.md)
- Modules: [Runtime and shared utilities](./modules/runtime-and-shared-utils.md), [Constants and configuration](./modules/constants-and-configuration.md), [Data store](./modules/data-store.md), [Data extraction](./modules/data-extract.md), [GPT extraction](./modules/gpt-extract.md), [Peer deduction](./modules/data-peers.md), [Cube aggregation](./modules/data-aggregate.md), [Modelling](./modules/modelling.md), [Strategies](./modules/strategies.md), [Portfolio](./modules/portfolio.md), [Validation](./modules/validation.md), [DAGs and infrastructure](./modules/dags-and-infrastructure.md), [Application and scripts](./modules/application-and-scripts.md), [Tests](./modules/tests.md)
- Flows: [Nightly data refresh](./flows/nightly-data-refresh.md), [SEC and LLM extraction](./flows/sec-llm-extraction.md), [Cube build](./flows/cube-build.md), [Model training and daily prediction](./flows/model-training-and-prediction.md), [Portfolio to trade ledger](./flows/portfolio-to-trade-ledger.md)
- Concepts: [Step pattern](./concepts/step-pattern.md), [Table registry and store boundary](./concepts/table-registry-and-store-boundary.md), [Point-in-time data](./concepts/point-in-time-data.md), [Cube parts](./concepts/cube-parts.md), [Peer-relative features](./concepts/peer-relative-features.md), [Source availability](./concepts/source-availability.md), [Strategy sleeves](./concepts/strategy-sleeves.md)
- Guides: [Run the pipeline](./guides/run-the-pipeline.md), [Add a data source](./guides/add-a-data-source.md), [Add a cube feature](./guides/add-a-cube-feature.md), [Add a model or sleeve](./guides/add-a-model-or-sleeve.md), [Validate a change](./guides/validate-a-change.md)

## 2026-09-25: refresh

- Profile: internal/standard
- source_commit: 054f0e1 (unchanged)
- Coverage: fully read and consolidated all 12 legacy `docs/` pages; added dense reference, operations, coding, testing, data-access, and backlog pages; rewired every wiki documentation link
- Primary pages: [Overview](./OVERVIEW.md), [TODO](./TODO.md), [Table catalog](./reference/table-catalog.md), [Data sources](./reference/data-sources.md), [Live database](./reference/live-database.md), [Configuration](./reference/configuration.md), [Modelling and portfolio](./reference/modelling-and-portfolio.md)
- Guides: [Run the pipeline](./guides/run-the-pipeline.md), [Data access](./guides/data-access.md), [Coding standards](./guides/coding-standards.md), [Testing and validation](./guides/testing.md), [Add a data source](./guides/add-a-data-source.md), [Add a model or sleeve](./guides/add-a-model-or-sleeve.md), [Validate a change](./guides/validate-a-change.md)
- Rewired sections: [architecture](./architecture/system-overview.md), [modules](./modules/data-store.md), [flows](./flows/nightly-data-refresh.md), and [concepts](./concepts/source-availability.md)
- Agent entry points: [AGENTS.md](../AGENTS.md), [CLAUDE.md](../CLAUDE.md)
- Retirement state: legacy `docs/` remains present for human review; package metadata and documentation-sync tooling now select the wiki

## 2026-09-25: refresh — documentation retirement audit

- Profile: internal/standard
- source_commit: 054f0e1 (unchanged)
- Coverage: removed active legacy-path references; verified all twelve former documentation subjects against canonical wiki destinations; made the backlog directly discoverable; corrected validation guidance against the inspected CLI; added source-grounded recovery procedures
- Pages: [Overview](./OVERVIEW.md), [TODO](./TODO.md), [Documentation coverage](./reference/documentation-coverage.md), [Run the pipeline](./guides/run-the-pipeline.md), [Large backfills and recovery](./guides/large-backfills-and-recovery.md)
- Agent entry points: [AGENTS.md](../AGENTS.md), [CLAUDE.md](../CLAUDE.md)
- Audit boundary: generated caches, tool-managed local state, secrets, report artifacts, and external provider URLs are not repository-documentation dependencies

## 2026-09-26: refresh — institutionals refactor

- Profile: internal/standard
- source_commit: 0d4ebd0
- Coverage: refreshed the institutional cube-part architecture, ordered build flow, completeness-frontier source references, and the insider outlier diagnostic after extracting input loading and frontier resolution from the step orchestrator
- Architecture: the top-level staged pipeline and persistence boundaries are unchanged; the institutional sub-step now delegates I/O to [inputs.py](../src/data_aggregate/utils/institutionals/inputs.py) and completeness decisions to [frontiers.py](../src/data_aggregate/utils/institutionals/frontiers.py)
- Pages: [Cube aggregation](./modules/data-aggregate.md), [Cube build](./flows/cube-build.md), [Source availability](./concepts/source-availability.md), [Application and scripts](./modules/application-and-scripts.md)
- Public contracts retained: `StepCubeInstitutionals.run()`, `build_panel()`, the ordered seven-panel merge, the shared conditioning sink, exact price/share projections, completeness semantics, and `cube_part_institutionals` persistence

## 2026-09-26: refresh — Ruff and Pyright contract

- Profile: internal/standard
- source_commit: 783e6a5
- Coverage: aligned the repository coding guide with the enforced Ruff 0.16.9 and Pyright 1.1.414 contracts after the institutionals refactor; no architecture, persistence, aggregation, feature, or protocol contract changed
- Naming: lower snake case for functions, methods, arguments, and local variables; exception classes end in `Error`; module constants and classes retain capitals
- Runtime syntax: Python 3.13 unions use `A | B`, including the explicit project convention for `isinstance`; every `zip()` states its strictness
- Tooling: [pyrightconfig.json](../pyrightconfig.json) resolves the project-local `.venv`; matching `pandas-stubs` supports pandas boundaries; Airflow-only suppressions remain file-scoped because DAGs run in the separate Python 3.12 environment
- Page: [Coding standards](./guides/coding-standards.md)

## 2026-09-27: refresh — schema-driven extraction freshness

- Profile: internal/standard
- source_commit: bb65544 (was 783e6a5)
- Coverage: replaced the parallel extraction freshness registry with the canonical table metadata in `src/data_store/schema.py`; documented the three-retry hard gate between extraction and aggregation; removed retired Wikipedia and Google Trends tables from freshness checks
- Pages: [Overview](./OVERVIEW.md), [Data extraction](./modules/data-extract.md), [Nightly data refresh](./flows/nightly-data-refresh.md)

## 2026-09-30: refresh — SEC identity and employee extraction

- Profile: internal/standard
- source_commit: 62dc6d3 (was bb65544)
- Coverage: refreshed SEC registrant discovery, per-ticker identity-scope invalidation, truthful XBRL outcomes, standalone employee repair, Airflow ordering, and recovery guidance
- Pages: [Data extraction](./modules/data-extract.md), [DAGs and infrastructure](./modules/dags-and-infrastructure.md), [Nightly data refresh](./flows/nightly-data-refresh.md), [SEC and LLM extraction](./flows/sec-llm-extraction.md), [Run the pipeline](./guides/run-the-pipeline.md), [Large backfills and recovery](./guides/large-backfills-and-recovery.md), [Data sources](./reference/data-sources.md), [Table catalog](./reference/table-catalog.md)

## 2026-09-30: refresh — financial notes archive availability

- Profile: internal/standard
- source_commit: 91395e7 (was bb65544)
- Coverage: documented monthly SEC Financial Statement and Notes archive availability, the persisted `available_at` clock, monthly freshness, and fail-closed point-in-time projection from the later of filing and archive availability
- Pages: [Table catalog](./reference/table-catalog.md), [Data sources](./reference/data-sources.md), [Overview](./OVERVIEW.md)
- Operational boundary: the code and schema declarations are current; the live database still requires the separately authorized column evolution, metadata repair, and full fundamentals-part rebuild

## 2026-09-30: refresh — historical notes availability audit

- Profile: internal/standard
- source_commit: b91495e
- Coverage: corrected the archive-availability contract after finding that current ZIP `Last-Modified` and local cache mtime do not establish original public release dates; historical repair and full rebuild are on hold for point-in-time use
- Pages: [Overview](./OVERVIEW.md), [Table catalog](./reference/table-catalog.md), [Data sources](./reference/data-sources.md)
- Evidence: the SEC documents later archive corrections; local older ZIPs were acquired in 2026, and their internal build dates differ from SEC dataset update and acquisition dates

## 2026-09-30: refresh — notes estimated and observed date policy

- Profile: internal/standard
- source_commit: 40cba83 (was c679331)
- Coverage: historical notes archives through August 2026 use the next-month 12th, rolled to Monday on weekends; archives from September 2026 use the successful New York download date; existing dates can be corrected by a metadata-only CLI path
- Pages: [Table catalog](./reference/table-catalog.md), [Data sources](./reference/data-sources.md), [Run the pipeline](./guides/run-the-pipeline.md)
- Operational boundary: no live schema migration, notes metadata repair, or fundamentals-part rebuild was run; the historical 12th is an estimate and does not guarantee strict point-in-time provenance

## 2026-09-30: refresh — FTD ZIP availability and settlement lineage

- Profile: internal/standard
- source_commit: 29f81a3 (was b91495e)
- Coverage: added the ZIP-grain `sec_ftd_vintages` table and the estimated/observed availability policy; documented that source `period`, not settlement day, controls FTD publication, including day-15 rows in some `b` ZIPs
- Pages: [Overview](./OVERVIEW.md), [Table catalog](./reference/table-catalog.md), [Data sources](./reference/data-sources.md), [Live database](./reference/live-database.md), [Source availability](./concepts/source-availability.md), [Data extraction](./modules/data-extract.md), [Cube aggregation](./modules/data-aggregate.md)
- Measurement: 1,079,328 stored FTD rows, 413 vintage periods, 36,299 date-inferred-period mismatches, and zero settlement dates shared across stored ZIP periods on 2026-09-30

## 2026-09-30: refresh — institutionals merge and model-readiness backlog

- Profile: internal/standard
- source_commit: 11d8735+838064d (pre-merge parent tips)
- Coverage: reconciled FTD ZIP availability and SEC identity/extraction documentation with current `dev`; added the train/holdout label-boundary and historical-universe P0 gates plus institutionals validation follow-ups
- Pages: [Overview](./OVERVIEW.md), [TODO](./TODO.md), [Cube aggregation](./modules/data-aggregate.md), [Data sources](./reference/data-sources.md)
- Validation: the approved FTD ZIP-publication correction moved only `f_ic_ftd_to_adv20` and `f_ic_ftd_persistence_30d` in the frozen short-flow panel; the updated regression passes all 37 outputs and 1,749 hashed columns

## 2026-10-01: refresh — quarterly pension ZIP availability

- Profile: internal/standard
- source_commit: 2f8f8da (working-tree pension fix not yet committed; previous wiki stamp 11d8735+838064d)
- Coverage: SEC Financial Statement Data Sets now retain quarterly `pension_facts` vintages with an estimated historical quarter-end-plus-12-day clock, an observed future download clock, and a feature start at the later of filing and archive availability
- Pages: [Overview](./OVERVIEW.md), [Table catalog](./reference/table-catalog.md), [Data sources](./reference/data-sources.md), [Run the pipeline](./guides/run-the-pipeline.md)
- Measurement: the table was backed up, recreated, and replayed from 66 CRC-checked ZIPs; 12,742 rows across 66 quarters, no null clocks or duplicate keys, and zero historical date-rule mismatches. The existing fundamentals cube part was not rebuilt and remains stale until a separate full build and assembly.
- Limitation: historical +12 is an estimate, not a verified SEC first-publication date; the generic raw-table leakage validator abstains, while targeted availability/PIT tests and the BDX revision check pass.

## 2026-10-01: refresh — derive FTD ZIP availability during feature builds

- Profile: internal/standard
- source_commit: 7ce4ad7 (uncommitted `harness/ftd-on-the-fly` worktree)
- Coverage: removed the `sec_ftd_vintages` registry, bootstrap DDL, extraction writes, quality-script requirement, and cube read; the feature builder now derives each ZIP date from stored `period` plus 15 days and weekend roll. The globally latest eligible stored ZIP alone may use its local cache date when that New York date is today or yesterday.
- Pages: [Table catalog](./reference/table-catalog.md), [Data sources](./reference/data-sources.md), [Source availability](./concepts/source-availability.md), [Data extraction](./modules/data-extract.md), [Cube aggregation](./modules/data-aggregate.md), [TODO](./TODO.md)
- Validation: 41 focused tests passed, including partial-universe latest-period selection; 37 aggregate regression outputs and 1,749 hashed columns matched the baseline; a read-only AAPL sample confirmed that the April 15 `202404b` row first publishes May 15.
- Operational boundary: the old physical table may still exist in PostgreSQL; this worktree did not drop it or rewrite persisted cube rows. The 15-day historical clock is an estimate, not verified SEC first-publication provenance.

## 2026-10-01: refresh — employee incremental resume

- Profile: internal/standard
- source_commit: 3e955eb (working-tree employee resume fix; wiki-wide freshness anchor remains 2f8f8da)
- Coverage: routine employee extraction skips filing dates already stored in `fundamentals_employees` even when the accession manifest is empty; `--full` remains the explicit replay
- Pages: [Data sources](./reference/data-sources.md), [Run the pipeline](./guides/run-the-pipeline.md)
- Evidence: the local table had 11,257 rows while its manifest had zero accession outcomes; a read-only plan skipped 31 AAPL and 30 CSCO stored dates, and 14 focused tests passed.

## 2026-10-01: refresh — modelling refactor

- Profile: internal/standard
- source_commit: f9ab52d (harness/modelling-transformers-refactor, merged into dev)
- Coverage: `src/modelling` split into `steps/` (StepLongShort), `transformers/` (model families, Monitor, Backtest) and `utils/`; `long_book` and `trend_cta` sleeves removed; new config keys `model.models_dir`, `model.backtest`, per-family `task`; artifact format `transformer-pickle-v1` (portable member pickles; metadata records library versions)
- Pages: [Modelling](./modules/modelling.md), [Strategies](./modules/strategies.md), [Modelling and portfolio](./reference/modelling-and-portfolio.md), [Configuration](./reference/configuration.md), [Table catalog](./reference/table-catalog.md), [Model training and daily prediction](./flows/model-training-and-prediction.md), [Add a model or sleeve](./guides/add-a-model-or-sleeve.md), [Coding standards](./guides/coding-standards.md), [Step pattern](./concepts/step-pattern.md), [Feature-to-portfolio pipeline](./architecture/feature-to-portfolio.md), [TODO](./TODO.md)
- Agent entry points: [AGENTS.md](../AGENTS.md)

## 2026-10-02: refresh — earnings-call extraction on defeatbeta

- Profile: internal/standard
- source_commit: 166a9f2 plus the uncommitted legacy-retirement phase (harness/earnings-call-extraction-gaps)
- Coverage: one transcript source (HuggingFace `defeatbeta/yahoo-finance-data`) replaces ROIC, Motley Fool and the kurry dataset; the Wikipedia pageviews fetcher is gone; `earnings_call_sections` is re-keyed to (ticker, quarter, paragraph) with the call date as `as_of`; CLI `extract-earnings-calls` and one default-pool DAG task; shared speaker-turn split and cleaning in `src/utils/earnings_call_split.py`; config `earnings_calls` in `configs/data.yml`; tone-drift item deferred to the TODO
- Pages: [Data sources](./reference/data-sources.md), [Table catalog](./reference/table-catalog.md), [Live database](./reference/live-database.md), [Configuration](./reference/configuration.md), [Data extraction](./modules/data-extract.md), [Cube aggregation](./modules/data-aggregate.md), [Run the pipeline](./guides/run-the-pipeline.md), [Data access](./guides/data-access.md), [Data platform](./architecture/data-platform.md), [Nightly data refresh](./flows/nightly-data-refresh.md), [TODO](./TODO.md)
- Evidence: `reports/validate/2026-10-01-earnings-call-extraction-gaps/` (full load 33,591 calls in 18.6 min, exact parity with the source index, split `ok` 98.71 %)
- Operational boundary: full FinBERT scoring, the remaining embeddings and the universe `build-text -F` are pending; `cube_part_text` holds 3 sample tickers; the `*_legacy` tables await the user's drop confirmation.

## 2026-10-02: refresh — shared EDGAR extraction driver

- Profile: internal/standard
- source_commit: abd4b53 (branch `harness/edgar-extract-refactor`)
- Coverage: per-ticker EDGAR fetchers declared as `EdgarFetch` and walked by `run_edgar_fetch`; `FilingStamp`; `Identity.filing_scope`; shared `sec_atom.py`, `item_carve.py`, `read_zip_tables` / `pending_periods`; one 13F walk feeding `sec13f_hr` and `sec13f_manager_holdings` plus the per-CIK manager catch-up; one 13D/13G row builder; one insider contract for bulk and live; one Form 3/4/5 pass for both identity tables; superinvestor overrides moved to `configs/sec/`; `gpt.threads` sizes DEF 14A and vote workers; deleted Wikipedia pageview and Google Trends fetchers removed from the docs
- Pages: [Data sources](./reference/data-sources.md), [Data extraction](./modules/data-extract.md), [System overview](./architecture/system-overview.md), [Run the pipeline](./guides/run-the-pipeline.md), [Configuration](./reference/configuration.md), [TODO](./TODO.md)
- Table contracts unchanged; [Live database](./reference/live-database.md) not edited.

## 2026-10-03: refresh — EDGAR extraction validation fixes

- Profile: internal/standard
- source_commit: d86de08 (branch `harness/edgar-extract-refactor`)
- Coverage: `EdgarFetch` has no partial-success flag; `sec_atom.py` owns Atom paging and the entry filter; the 13F listing sorts amendments after same-day originals; the 13F manager catch-up decides per filing from stored (period, filing date) pairs and splits transient from deterministic read failures; review residuals and minor simplifications deferred to the TODO
- Pages: [Data sources](./reference/data-sources.md), [Data extraction](./modules/data-extract.md), [TODO](./TODO.md)
- Evidence: `reports/validate/2026-10-01-edgar-extract-refactor/` (`06-final.md`)

## 2026-10-03: refresh — one insider transactions table

- Profile: internal/standard
- source_commit: 843a4eb (branch `harness/insider-transactions-merge`)
- Coverage: `insider_transactions` is the only Forms 3/4/5 table, keyed (`accession_number`, `security_type`, `row_sequence`) with `source`, `owner_ciks`, `n_reporting_owners`, `original_submission_date`, `footnote_ids`, `acceptance_datetime`, `fetched_at`; EDGAR is authoritative and a zip quarter fills only the filings EDGAR missed, with a per-quarter missing-from-EDGAR warning; identity rejects are logged, not stored; the frontier is the EDGAR run's manifest entry; the cube collapses repeat copies and supersedes Form 4/A cells point in time. Removed: the EDGAR staging, coverage and quarantine tables, the accession overlay, `validate insider-parity`, its `validate.yml` thresholds and the `data.yml` bulk cutover date. Full rebuild = drop the table, then `insider-transactions -F`. TODO: the zip/EDGAR encoding item is resolved (one shared encoding) and removed; the Atom 503 item stays.
- Pages: [Table catalog](./reference/table-catalog.md), [Data sources](./reference/data-sources.md), [Configuration](./reference/configuration.md), [Live database](./reference/live-database.md), [Large backfills and recovery](./guides/large-backfills-and-recovery.md), [Run the pipeline](./guides/run-the-pipeline.md), [Validation](./modules/validation.md), [Data extraction](./modules/data-extract.md), [Source availability](./concepts/source-availability.md), [Cube build](./flows/cube-build.md), [Application and scripts](./modules/application-and-scripts.md), [DAGs and infrastructure](./modules/dags-and-infrastructure.md), [Point-in-time and quality controls](./architecture/point-in-time-and-quality.md), [Nightly data refresh](./flows/nightly-data-refresh.md), [Documentation coverage](./reference/documentation-coverage.md), [TODO](./TODO.md)
- Evidence: `reports/validate/2026-10-03-insider-transactions-merge/` (`03-implementation.md`)
- Operational boundary: the live database still holds the retired tables until the user drops them and runs the refill.

## 2026-10-04: refresh — entity-safe EDGAR filing scope and dated lineage

- Profile: internal/standard
- source_commit: 2d3566c (branch `harness/entity-symbol-lineage`)
- Coverage: `symbol_tenure` evidence partitions (`form345`, `manual`, `dei`); dated `entity_lineage` (`cik_window`, `cik_event`, `symbol` rows, sentinel open start, split change stamps); listing by CIK only with guard skip-and-count, 31-day seam margin and the same-period rule; register-only CIK windows; symbol tapes and S1; `identity-propagate`; `validate identity` and the manual-decision flags; new identity commands, DAG order, cutover and rollback; lineage follow-ups and the database-only resume item in the TODO
- Pages: [Table catalog](./reference/table-catalog.md), [Data sources](./reference/data-sources.md), [Run the pipeline](./guides/run-the-pipeline.md), [Live database](./reference/live-database.md), [TODO](./TODO.md)
- Evidence: `reports/validate/2026-10-02-entity-symbol-lineage/` (`03-implementation.md`)
- Operational boundary: the live identity tables keep the old shape until the user-run cutover.

## 2026-10-04: refresh — extraction resume reads only the database

- Profile: internal/standard
- source_commit: 6080c74 (branch `feat/db-derived-resume`)
- Coverage: every extracted table declares a `Resume` contract and, where a read can find nothing, an empty-filing marker in `schema.py`; `resume.py` derives series windows, EDGAR document work lists (local filing index minus stored accessions) and archive work lists from the tables alone; consumer reads drop markers; the run manifest, the bulk sidecars and their config keys are retired; `sec_io` is the one SEC retry policy; fetchers save what they read, retry in rounds and exit 0; one `price-history` download writes prices, dividends and splits; short interest reads day-level holes, with `--repair-gaps` for per-key gaps; the 13F low watermark and the new-ticker backfill from the SEC 13F data sets; the insider EDGAR window starts at the earlier of the day after the last zip quarter and the run date minus 7 days; the insider, 13D and 13G frontiers are read from the tables; every cube part rewrites at least 7 sessions; DAG gates run on `ALL_DONE`; `extraction-status` reports and exits 0; `predict` raises `StaleInputsError`. TODO: the "Extraction resume reads only the database" item is closed and removed.
- Pages: [Data extraction](./modules/data-extract.md), [Nightly data refresh](./flows/nightly-data-refresh.md), [Configuration](./reference/configuration.md), [Run the pipeline](./guides/run-the-pipeline.md), [Large backfills and recovery](./guides/large-backfills-and-recovery.md), [Table catalog](./reference/table-catalog.md), [Data access](./guides/data-access.md), [Data sources](./reference/data-sources.md), [Cube build](./flows/cube-build.md), [Live database](./reference/live-database.md), [Source availability](./concepts/source-availability.md), [Model training and daily prediction](./flows/model-training-and-prediction.md), [DAGs and infrastructure](./modules/dags-and-infrastructure.md), [Coding standards](./guides/coding-standards.md), [Add a data source](./guides/add-a-data-source.md), [TODO](./TODO.md)
- Evidence: `reports/validate/2026-10-03-db-derived-extraction-resume/` (`02-plan.md`, `03-implementation-phase-*.md`)
- Operational boundary: the live database gets `sp500_tickers.added_on`, the markers and the cache builds only at the cutover.

## 2026-10-04: refresh — verified point-in-time superinvestor roster

- Profile: internal/standard
- source_commit: 548f3a2 (branch `harness/sec13f-superinvestor-roster`)
- Coverage: roster config moved to `configs/superinvestors/` (`dataroma_roster_history.json`, `overrides.json`); quarterly capture-dated Wayback history and its builder `scripts/build_dataroma_roster_history.py`; evidence-backed overrides; manager chains (manager ID = oldest filer CIK, dated windows, returning filer allowed) with `to_manager_books` relabelling; 13F activity gate with EDGAR-listing fallback and dated `inactive` exceptions; daily refresh writes only on change; `superinvestors --seed` rebuilds the table; deferred items in the TODO
- Pages: [Data sources](./reference/data-sources.md), [Table catalog](./reference/table-catalog.md), [Configuration](./reference/configuration.md), [Run the pipeline](./guides/run-the-pipeline.md), [TODO](./TODO.md)
- Evidence: `reports/validate/2026-10-04-sec13f-superinvestor-roster/` (`03-implementation.md`)
- Operational boundary: the live roster was rebuilt from the branch; after merge the user reruns `superinvestors --seed`, unpauses the `data_extraction` DAG and rebuilds the cube.

## 2026-10-05: refresh — extraction resume cutover and refactor

- Profile: internal/standard
- source_commit: f02bbb6 (branch `feat/db-derived-resume`)
- Coverage: the document planner reads each table's stored accessions once and cuts them to each key's listing; `incremental.stored_values` is the one stored-value reader; Item 5.07 votes run as load, plan, tasks, extract and save steps; the 13F backfill treats a missing `added_on` as no new tickers; 13F information-table downloads and the superinvestor 13F listing go through `sec_io`; `sp500_tickers.added_on` exists (cutover rows 2000-01-01); the EDGAR index cache is in `data/sec_edgar_index`; `sec13f_hr` CIKs are 10-digit; `sec_short_interest` starts at 2018-08-01 and a scoped `-F` upserts its own tickers; the manifest and sidecars are retired to `data/_retired/2026-10/`. TODO: the R-07 unpadded-CIK item is removed, the insider 2026q1 ingest is marked done, and an entity-symbol-lineage re-run block is added.
- Pages: [Data extraction](./modules/data-extract.md), [Table catalog](./reference/table-catalog.md), [Data sources](./reference/data-sources.md), [Run the pipeline](./guides/run-the-pipeline.md), [Live database](./reference/live-database.md), [TODO](./TODO.md)
- Evidence: `reports/validate/2026-10-03-db-derived-extraction-resume/` (`05-final-report.md`, `refactor/04-validate.md`, `03-implementation-phase-11-*.md`)
- Operational boundary: the stage E file move, stage D, the cube rebuild and `superinvestors --seed` run after the merge into dev.

## 2026-10-06: refresh — security master, dated event windows and the identity cutover order

- Profile: internal/standard
- source_commit: 09aaaae6 (branch `harness/entity-symbol-lineage`)
- Coverage: `security_master`, `sec_company_tickers` and the per-security tape tables; ticker tapes rebuilt as class sums × conversion ratio; insider `source_symbol`/`economic_date`/`lineage_role`; 8-K/13D/13G by dated CIK windows and the PLD/JCI/DD deferrals; automatic CIK windows; oracle order and one universe ticker per entity; `--every-ticker`; the hard identity gate; the cutover order; identity configs; BRK-A volume scale; lineage TODO (traded-security realignment, F-105 residual, F-110, F-111, F-117 and data findings)
- Pages: [Table catalog](./reference/table-catalog.md), [Data sources](./reference/data-sources.md), [Run the pipeline](./guides/run-the-pipeline.md), [Configuration](./reference/configuration.md), [TODO](./TODO.md)
- Evidence: `reports/validate/2026-10-02-entity-symbol-lineage/` (`p11-runbook.md`, `plans/02-revision-4-q2.md`, `defects.md`)
- Operational boundary: the live database keeps the old identity shape until the user-run cutover.

## 2026-10-07: refresh — remaining legacy 13F P1 parser

- Profile: internal/standard
- source_commit: e1d349d0 (was 2f8f8dad)
- Coverage: shared legacy numeric/column/page/wrap/cover handling and pre-group source guards; finite historical replay and remaining P1 data boundary
- Pages: [Data sources](./reference/data-sources.md), [TODO](./TODO.md), [Overview](./OVERVIEW.md)
- Evidence: local ignored `reports/validate/2026-10-07-superinvestor-legacy-p1/`; exact-source regressions in [test_13f_legacy_fallback.py](../tests/data_extract/institutionals/test_13f_legacy_fallback.py)
- Operational boundary: 4 source-verified replacement books and 62 proposed NULL-amount books are unapplied. Unsupported sources remain unavailable. Production feature certification abstains without `cube_part_prices`; no database write, cube rebuild, P2 correction, push or PR.
- Integrated correction source_commit: a2310a580795e282a180f0ce721913164282bb5a; review R-001 proved matching CUSIPs/cover totals can retain wrong positive shares or a PRN note as common stock. Successful text reads now agree with normalized logical source value, amount, instrument type and put/call. The isolated correction is `2de7d42c`; its provenance is preserved in the local Harness merge.

## 2026-10-07: refresh — legacy 13F parser merged into dev and data backlog prioritized

- Profile: internal/standard
- source_commit: 2c8575ab45f73da9f87711880983660213d289e3 (was e1d349d0)
- Coverage: parser/source-fact fix merged into `dev`; 183 prior scope tests and 74 fresh parser/shared-reader tests on `dev`; catch-up versus roster `-F` semantics; prioritized remaining superinvestor data and output work.
- Pages: [Overview](./OVERVIEW.md), [Data sources](./reference/data-sources.md), [Run the pipeline](./guides/run-the-pipeline.md), [TODO](./TODO.md)
- Evidence: `harness/superinvestor-legacy-p1` at `d25365bd` merged without conflicts; local ignored `reports/validate/2026-10-07-superinvestor-legacy-p1/` (`_out/dev_merge_T1.json`, `_out/baseline.json`, `_out/dispositions.json`, `_out/repair_manifest.json`).
- Remaining priorities: P1 applies 4 verified whole-book replacements and 62 NULL-amount book updates atomically after a live-before check and backup; P2 resolves 15 in-window gaps among 228 never-stored books and source-checks ghost periods, then rebuilds and validates the institutional features; P3 defers further recovery of 90 / 7,369 stored books (1.22%, including 81 before 2011Q3 and 9 in-window), following the user's 2% cutoff.
- Operational boundary: this merge and wiki refresh apply no database repair, source recovery or cube rebuild; the production price-part prerequisite and feature-certification abstention remain. No push or PR was requested.

## 2026-10-08: refresh — employee headcount components and direct read

- Profile: internal/standard
- source_commit: c01ffda6 (branch `harness/employee-headcount-coverage`)
- Coverage: `fundamentals_employees` component shape (components, basis, status, CIK, accession, source document, JSON quotes); no employee roster; ix:header strip, workforce-number windows and annual-report exhibit fallback; scope-first answer schema and per-component quote guard; `employees_sec` removed from the merged history (92 columns) and the DAG edge; the cube reads the table directly (370-day as-of join, FT + α·PT proxy, basis-masked growth); `validate employees`; the universe runbook
- Pages: [Table catalog](./reference/table-catalog.md), [Data sources](./reference/data-sources.md), [Data extraction](./modules/data-extract.md), [Modelling and portfolio](./reference/modelling-and-portfolio.md), [Configuration](./reference/configuration.md), [Run the pipeline](./guides/run-the-pipeline.md), [Large backfills and recovery](./guides/large-backfills-and-recovery.md), [Live database](./reference/live-database.md), [Nightly data refresh](./flows/nightly-data-refresh.md), [DAGs and infrastructure](./modules/dags-and-infrastructure.md), [TODO](./TODO.md)
- Evidence: `reports/validate/2026-10-07-employee-headcount-coverage/` (`02-plan.md`, `03-implementation.md`)
- Operational boundary: the live table was recreated and `employees_sec` dropped on 2026-10-07; the universe run and the cube rebuild are user-run after the merge.

## 2026-10-08: refresh — legacy 13F stored-book repair applied

- Profile: internal/standard
- source_commit: none (database change; no code merged)
- Coverage: `sec13f_manager_holdings` P1 cleanup after the legacy parser fix: 4 source-verified whole-book replacements (Oaktree `0000949509` 2011-03-31, Pabrai `0001549575` 2012-12-31 and 2013-03-31, Vulcan `0001556785` 2012-09-30) and all-nine-amount NULL in 62 structurally corrupt books
- Pages: [TODO](./TODO.md), [Run the pipeline](./guides/run-the-pipeline.md)
- Evidence: local ignored `reports/validate/2026-10-07-superinvestor-legacy-p1/` (`_out/repair_manifest.json`, `_out/apply_repair.sql`, `_scripts/build_apply_sql.py`, table dump `_out/sec13f_manager_holdings_pre_repair_2026-10-08.dump`); the 4 books were re-read live from SEC with the merged parser and matched the manifest's after-rows exactly
- Operational boundary: user-authorized; live rows matched the manifest before-state (9,416 rows, 0 drift), a rolled-back dry run passed, then one psql transaction committed 162 updates, 1 deletion and 9,162 NULL updates (the 66 books now hold 9,415 rows). No source recovery, cube rebuild or push; `cube_part_prices` and `build-institutionals -F` remain the P2 follow-up.
- Documentation note: the OpenKnowledge MCP refused these wiki writes on disk ("document path could not be resolved"), so the three pages were edited directly at the user's request.

## 2026-10-08 - 8-K item FutureWarning silenced
- Change: `fetch_8k_edgar.py` filters edgartools' `CurrentReport[...] found no such item` FutureWarning (absent item = empty text, already handled); test `test_8k_item_absent_from_parsed_body_is_silent_and_empty`.
- Pages: [Data sources](./reference/data-sources.md)
- Documentation note: OpenKnowledge refused the disk write, so the pages were edited directly.

## 2026-10-08: refresh — identity manual review

- Profile: internal/standard
- source_commit: ebc78771 (branch `harness/identity-manual-review`)
- Coverage: the 2026-10-08 `identity_tables` manual items resolved in `configs/sec` (seven `same_entity` acquired-target / LBO verdicts incl. VMRK/AvalonBay, same-CIK symbol renames, 12 evidenced `form345` rejections, five FINRA seam market boundaries); tape-mix flags decided by the stored `security_master` are information, not action
- Pages: [Data sources](./reference/data-sources.md), [TODO](./TODO.md)
- Evidence: `reports/validate/2026-10-08-identity-manual-review/` (`01-research.md` price-continuity audit, `03-implementation.md`)
- Operational boundary: no database write; the next `identity_tables` → `identity_propagate` DAG run applies it.

## 2026-10-08: refresh — dei cover-page symbol parser

- Profile: internal/standard
- source_commit: branch `harness/dei-symbol-parser`
- Coverage: `parse_symbol_field` keeps a slash series code on its issuer (`BDX/26A` → `BDX-26A`, `MS/PL` → `MS-PL`); `notes-download` drops `dei:TradingSymbol` facts dimensioned on `LegalEntityAxis` (subsidiary securities on combined covers). Removes `F` from Ford Motor Credit, `GM` from GM Financial, `PG` from BAC/MS, `PM` from Vornado, `SO` from Southern Power
- Pages: [Data sources](./reference/data-sources.md), [TODO](./TODO.md)
- Evidence: `reports/validate/2026-10-08-dei-symbol-parser/` (offline replay of the 2019+ Notes zips: conflicts 43 → 2, no universe filer loses its own ticker)
- Operational boundary: no database write; a `notes-download --full` recapture applies it.

## 2026-10-09: refresh — EXE fresh start

- Profile: internal/standard
- source_commit: branch `harness/exe-fresh-start`
- Coverage: register kind `fresh_start` dates a ticker's only CIK; the history seam rule also runs for a single filer with a dated start; EXE (Chesapeake, CIK 0000895126) starts at its 2021-02-10 emergence and the old CUSIPs 165167107/165167743 are excluded as cancelled equity (user decision 2026-10-08, like Sharadar)
- Pages: [Configuration](./reference/configuration.md), [TODO](./TODO.md)
- Evidence: `reports/validate/2026-10-08-exe-fresh-start/` (offline: lineage and flags change only EXE's window, 481 FINRA rows re-stamp, EXE history 72 → 24 rows from as_of 2021-03-01)
- Operational boundary: no database write; the next identity and fundamentals-history runs apply it.

- 2026-10-10 — Institutional P1 simpler filing-wide value-unit candidate implemented/tested in isolated worktree. Exact source-plus-market proof: 78.72% of corrected groups / 94.82% of rows; 205 tests passed / 1 skipped. Valid shares survive uncertain dollars. Production activation withheld: complete dollar totals only 1/461, 0/476, 0/482 tickers in three measured quarters. Immutable filing versions and genuine share-flow split changes remain deferred. [Evidence](../reports/validate/2026-10-09-institutional-13f-audit/06-final.md).

## 10 October 2026 — Institutional P1 common-dollar fallback

Updated configuration and table-catalog candidate notes after the user-approved fallback: uncertain common values use valid shares × restored quarter-end prices without supplying unit evidence or overwriting sources. Structural zeros remain valid; nonzero unresolved options/debt stay unknown. Isolated sibling suite: 208 passed / 1 skipped, with independent review clear. Three source-quarter previews restore complete common-dollar totals to 457/461, 472/476 and 436/482 tickers; option/value feature coverage remains material (121/476 unavailable in 2022Q4). Full SEC consensus is recommended for deferral, while PIT filing versions and genuine-split share flows remain required. [Current audit](../reports/validate/2026-10-09-institutional-13f-audit/06-final.md). No production data/code adoption occurred.

## 2026-10-10: refresh — scoped institutional amendment candidate

- Profile: internal/standard.
- source_commit: candidate base `f34684620d868b7d4816a8d744a6dd92c02c8480`, branch `harness/institutional-13f-pit-20261009`; owned P3 content hash `cbd5756cd61af96f3aca87b272497986ad163bf7031d7915e7de68636e15e5a2`. Root user branch remains at `415528a38162966b7646243cc7dbf2a6bd1c8efd`; no merge performed.
- Pages: [Table catalog](./reference/table-catalog.md), [Data sources](./reference/data-sources.md), [Data access](./guides/data-access.md), [Large backfills and recovery](./guides/large-backfills-and-recovery.md), [Testing](./guides/testing.md).
- Coverage: two immutable accession-ledger contracts, atomic completion/resume, report-quarter identity, full replacement versus supplements, latest public prior-quarter baselines, conservative clocks, scoped CLI and independent real-source tests.
- Measured live effects: two approved new ledgers only,24 accessions /118,276 CUSIP-vintage rows; read-only validator PASS18 historical prefixes. No legacy or cube writes. Candidate-only contracts are explicitly marked; wider recovery, P4 split economics and deferred daily refresh remain pending.
- Validation:19 focused tests passed /one opt-in PG skip; broad201-pass regression and separate disposable PG atomicity proof. Independent final per-manager publication review passed. Source/value certification and production activation remain distinct.
- Evidence remains local ignored run artifacts under `reports/validate/2026-10-09-institutional-13f-audit/`; no ignore-policy change was made. New canonical references use plain artifact paths rather than links excluded from the wiki index. The earlier table-catalog P1 ignored-report link warning remains outside this refresh's new links.

- 2026-10-10: P3 copied-book candidate frozen at `25fae2b300950929fd6c35431bde501a1a734ba2189a74ac2163b0c6bba64a27`. Eleven copied NEW books now update reported positions PIT; omitted/unknown typed legs cannot become exits. Source audit133/143 packets and70/75 amendments (93.33%);64 recoverable histories /122 per-book prefix checks. Targeted34pass1skip; broad282pass1skip plus pre-existing legacy5.924pp first-publication failure, reproduced unchanged. Production remains gated. See local ignored `reports/validate/2026-10-09-institutional-13f-audit/p3-amendments.html`.

## 2026-10-10: candidate addendum — institutional P4 and daily cohort preview

- Profile: internal/standard.
- Candidate source base: f34684620d868b7d4816a8d744a6dd92c02c8480 on harness/institutional-13f-pit-20261009; source bytes are frozen in the local ignored P4 manifest/source ZIP. No merge or production adoption.
- Pages: [Testing](./guides/testing.md). The shared wiki overview anchor is preserved: this is a bounded unmerged-candidate addendum, not a refresh of unrelated118-file committed history or the whole wiki.
- Coverage: genuine share events routed to both 13F consumers; scoped complete-book daily comparisons, strict unknowns, observed-prior coverage, prospective active-prior/value revisions and caller-supported stable security identities.
- Validation:139 passes/one opt-in skip/one unchanged legacy5.924pp failure; independent9 focused controls and omission probe pass; read-only source validator14 accessions/94,341 rows/12 prefixes. GE22.121/20.239pp split-only defects removed; genuine NVDA/WST/DD preserved. Local ignored evidence: reports/validate/2026-10-09-institutional-13f-audit/.
- Remaining gates: full-universe versions/identity reporting perimeters/production activation and DAG refresh; day45 predictive significance is unproven. P2 position-guard economic incidence awaits the user's price-cube rerun.

- Final candidate addendum: four-file SHA42182e4321f3ae20f9d1a305fd3ef4b4c34012375e582c86a18f26a4b045a280. The real32-day identical-refiling age defect is fixed; final51 tests/one opt-in skip and independent10 focused checks/probe pass. Real ledger evidence28 complete versions/121,324 rows/35 stock prefixes, plus cutoff validators23 accessions/98,742 rows/12 prefixes PASS. State Street124 supported daily flow cells from day45; three combination/confidential scopes abstain; no checked carry-age mismatch across468 states. These are bounded source/math proofs, not marketwide confidence or production adoption. Current ignored handoff reports/validate/2026-10-09-institutional-13f-audit/p4-daily-preview.html embeds four charts and matches frozen candidate/reviewer hashes. Wiki overview anchor remains unchanged for this unmerged scope; no unrelated wiki regeneration.

## 2026-10-10: institutional backlog clarification

Added the user-requested [13F deferred issues](./TODO.md#13f-deferred-issues-from-the-october-2026-audit): manager reporting components and CIK transitions form one historical/future task; security CUSIP continuity remains separate. Recorded pending price-fallback validation, marketwide day-45 coverage/noise and daily amendment refresh proof; synchronized the earlier audit refresh TODO. Read back the canonical addition. Normal extractor/builder implementation, ten-manager proof and release checks remain unfinished required work. No code, source data, cube or branch merge changed. This bounded backlog update preserves the wiki's shared source anchor.

## 2026-10-10: ordinary institutional code and ten-manager proof completed

Supersedes the earlier unfinished-code backlog note. In the isolated institutional-13f-pit candidate, ordinary SEC/roster/ZIP extraction retains verified accession versions, and both actual broad/elite consumers call daily PIT comparable-book builders. Isolated source recovery140 packets/691,676 grouped rows; all20real full calls pass. Shared expiry collision found in three elite histories and fixed with exact RED/GREEN; fresh independent review clear. Regression318pass/1skip/1 unchanged legacy failure; source validators114prefix controls pass. Updated TODO to distinguish completed Python from required full-universe source recovery and deferred reporting/security/price/noise/DAG contracts. No branch merge or production cube write. Current local ignored report:reports/validate/2026-10-09-institutional-13f-audit/p5-normal-builders.html. Canonical overview source anchor remains unchanged for this unmerged scope.

## 2026-10-10: read-only filing-version and security-master verification

Updated the [13F backlog](./TODO.md#13f-deferred-issues-from-the-october-2026-audit) after checking the live database and all53cached SEC13F ZIP metadata inventories. The live candidate ledgers contain24complete filings/five managers/eleven manager-quarters; the cached ten-manager inventory has603filings,22preserved live and581still to load. Earlier extraction rows exist; the required work is one-time loading of preserved versions from cached source packets before the new cube readers can cover history. Morgan Stanley's August2024original is already preserved. A nested June–August2025ZIP is valid but the ordinary reader expects root member paths; the diagnostic supplied its actual paths without altering the cache. Verified existing security_master dates for DD/GE and Sharadar CUSIP lists; changed the continuity task to reusing those stored mappings in both 13F consumers, with report-period/class/lineage checks and genuine split conversion. GE's570historical positive positions already have dated master coverage; DD's latest987changed CUSIP sets carry99.59%of prior paired shares (nine mixed sets require checks). Evidence is local ignored p6 JSONs and chart; no application source, source cache, database table, cube or branch merge changed. This unmerged-candidate addendum preserves the existing wiki overview source anchor.

Latest user steering: new 13F tables are rejected. Recorded the existing-table/in-memory calculation requirement in the [13F backlog](./TODO.md#13f-deferred-issues-from-the-october-2026-audit), superseding new-ledger population/adoption instructions. The current last-filed keys cannot retain original and amended versions together; preserving versions inside existing tables is a proposed storage change awaiting agreement, with no schema migration, source replay, application refactor or table deletion performed.

## 2026-10-11: insider feature contract addendum

Updated the [aggregate module](./modules/data-aggregate.md#insider-canonical-cleaning-and-feature-contract) for the implemented insider-only three-phase plan: shared canonical title/identity/price cleaning, strictly earlier-day owner comparisons, true120calendar-day surprise contributors, supported known-share account math, and one pooled CEO/CFO/COO60day mean.88targeted tests pass; independent nine-ticker permutation/seven-prefix validation compares4,491,045cells; full paired rebuild produces13fields on3,900,500rows with20scoped edge observations passing. Five material independent bugs are corrected and all eleven integrated files match content identityf63c2ab2e9b43ec0d15257af244e964522ad16e034ca38ed4325029ff75e82d9.

Base415528a38162966b7646243cc7dbf2a6bd1c8efd, internal/standard, uncommitted scoped working-tree documentation; the global overview freshness anchor is unchanged because this is not a repository-wide committed-code refresh. VerdictCONDITIONAL: source/class-price coverage, broad amendment-cell completeness and generic validation/snapshot limits remain. User excluded cleaning configs; none were changed. SEC13F holdings/model work and production source/cube writes were excluded. Evidence is the ignored localreports/validate/2026-10-10-insider-feature-audit/report.html and fixed stage files; unrelated existing workspace/wiki edits are preserved.

## 2026-10-11: refresh — institutional three-finding correction

- Profile: internal/standard
- source_commit: 415528a38162966b7646243cc7dbf2a6bd1c8efd (was 2c8575ab45f73da9f87711880983660213d289e3)
- Coverage: scoped correction of the 13F source candidate addendum only; existing-table storage, acceptance-clock propagation, daily amendment ordering versus classification, and bounded self-validation. Unrelated source changes were not revalidated in this refresh.
- Candidate: isolated unmerged worktree based on f34684620d868b7d4816a8d744a6dd92c02c8480, effective 38-file content identity 2e8902c33b23490e796b58aa798e2d42b043ae167fabff11a0fc6cab75f17f63. No production source/cube rewrite.
- Pages: [Data sources](./reference/data-sources.md), [Overview](./OVERVIEW.md).

## 2026-10-11: insider cumulative-window correction

Updated only the [insider aggregate contract](./modules/data-aggregate.md#insider-canonical-cleaning-and-feature-contract) for the user-approved cumulative revision. Dollar flows sum named windows before current PIT cap division; management/purchase ratios count account holdings once and retain each lot's original disclosure expiry. Issuer-title and cross-account split-unit guards are independently verified. Silent managers and unsupported/class-specific accounts remain incomplete.

100focused tests pass; an overlapping ROOT30-test integration rerun passes. Independent12-ticker permutation/seven-prefix replay agrees on5,988,060cells, and correctness/Ponytail reviews find no material calculation blocker. Final immutable-baseline paired replay covers2,044,497source rows/13fields/2,374,616eligible cells. All13owned ROOT/WORK paths independently match identity249e493a1918b47e71ff6ee8e3d72b0f5b059f99db80f79d9c93d179d6c86828. Purchase/prior coverage78.69%→7.84%; management16.99%→12.19%, reflecting accepted expiry/reconciliation and abstention, not an estimated error rate. Base415528a38162966b7646243cc7dbf2a6bd1c8efd; internal/standard scoped uncommitted addendum. Source/amendment completeness remains conditional; no production cube/source writes, protected edits, models or SEC13F review. Current ignored local evidence isreports/validate/2026-10-10-insider-feature-audit/report.html and07-cumulative-validation.md. Earlier mean/decay results remain historical; unrelated edits and global wiki freshness anchor are preserved.

## 2026-10-11: refresh — governance cleaning stage

- Profile: internal/standard
- source_commit: branch `harness/governance-clean-stage`
- Coverage: one governance cleaning stage (`clean_governance_inputs`) before the features; per-cell provenance (`_source_as_of`, fill kind) so carries expire at true age; CEO stints and stint constants; insider-ownership estimate from holder rows; `f_say_on_pay_support` dropped for the 8-K series plus a proxy fallback; same-filing SCT pay growth; the historical pay backfill lookahead warning
- Pages: [Cube aggregation](./modules/data-aggregate.md), [TODO](./TODO.md)
- Evidence: `reports/validate/2026-10-10-governance-fill-review/` (`03-implementation.md`, `04-handoff.md`)
- Operational boundary: no database write; the user's `build-governance -F` and `assemble-cube` apply it.
- Validation fixes (same day): CEO-person daily values end at a successor's first proxy (`staleness.stint_to_daily`); the stint start year is the majority disclosed value; the ownership estimate drops entity-named holder rows and estimates above 3x the filed maximum; the pay-backfill by-year profile is stated on the module page. Evidence: `defects.md` F-001..F-004.
- Fix round 2 (2026-10-11): the SCT CEO is told apart from relatives sharing the person key; bridged totals feed no cross-filing growth; exact later SCT values replace a carried salary; CEO stints ignore honorifics, same-first-name suffix toggles and one-off foreign proxies; MetLife's board-held trust is not an insider holding; incremental governance builds warn that `build-governance -F` is needed after each proxy season. Evidence: `defects.md` R-C-*, F-008, F-009.

## 2026-10-11: scoped refresh — insider title and quiet-window corrections

- Profile: internal/standard
- source_commit: 415528a38162966b7646243cc7dbf2a6bd1c8efd plus uncommitted insider identity 6754e31aabfa47642aba1f0bfc614d4e2cfcef7939989b4eef308d89f790b5a9
- Coverage: insider subsection only; compound issuer titles, shared confirmed-inactivity zero policy, active uncertainty/version dependencies, final independent validation and source-completeness limits. No SEC13F holdings, source extraction, model or repository-wide refresh.
- Page: [Cube aggregation](./modules/data-aggregate.md)
- Sources: [Insider cleaner](../src/data_aggregate/utils/institutionals/insider_quality.py), [Insider features](../src/data_aggregate/utils/institutionals/insider_features.py)
- Evidence: 124 focused tests, overlapping final-tree 29-test check, full 2,044,497-row paired census, independent correctness/Ponytail and exact final identity checks; conditional source completeness and generic failures retained in the local insider audit report.

## 2026-10-11: scoped refresh — insider branch merged into dev

- Profile: internal/standard
- source_commit: d4ebd98ffe561aee3f51f2c0b910fd2a627ae4f0 (previous overview anchor: 415528a38162966b7646243cc7dbf2a6bd1c8efd)
- Insider branch commit: 0284541fc009a0270f034682c7c6773a8cf72f94. The sole conflict was the explicit loop binding required by Ruff; newer governance code and incremental tests are preserved.
- Coverage: insider aggregate contract and backlog only. Removed completed insider in-memory replay requirements and stale decay wording; retained unresolved extraction/amendment/copy/snapshot tasks and the pending full stored-history rebuild.
- Pages: [Overview](./OVERVIEW.md), [Cube aggregation](./modules/data-aggregate.md#insider-canonical-cleaning-and-feature-contract), [TODO](./TODO.md#insider-transactions-follow-ups)
- Validation: 34 post-merge tests pass in 12.56s, zero live-write attempts, six app sources match the committed branch and tested hashes are stable. Commit hooks including Ruff/formatting pass. Original numerical/independent evidence remains frozen.
- Boundary: scoped refresh only; no production data writes, model run or remote push. Unrelated backlog items are preserved.

## 2026-10-11: refresh — generic 13F cover-total alignment

- Profile: internal/standard
- source_commit: 0ef13cfe8bfc0d2c4c5fb8d917d76eb3060e2971 (previous overview anchor: 27e4f723f2e61a4dffd54d11f4785cee89e2b9ee)
- Coverage: source admission in the shared `reconcile_13f_source` helper. Any finite power of ten may align the cover to raw rows; existing 85–110% count and aligned-money checks remain. Raw values and declared row USD units are unchanged; market-based P1 corrections remain separate.
- Pages: [Overview](./OVERVIEW.md), [Data sources](./reference/data-sources.md), [Table catalog](./reference/table-catalog.md).
- Validation: 114 self-regression tests and 72 overlapping independent focused tests pass, with zero live-write attempts. Actual accession `0002112884-26-000003` has 14/14 rows and a 9.99997 cover ratio; factor 10 is accepted. Both existing-table in-memory fixture validators pass, and the cached ZIP is unchanged.
- Boundary: no production extraction, source-table write, cube rebuild or remote push. Cached-backfill performance measurements and proposals are a separate read-only review, frozen against the prior dev baseline.

## 2026-10-12: refresh

- Profile: internal/standard
- source_commit: 5f8cc3d43f6e71c2acc549b2a80ea3aeabe8e073 (was 0ef13cfe8bfc0d2c4c5fb8d917d76eb3060e2971)
- Coverage: approved cached13F backfill CPU/cache/preparation/resume refactor; immutable existing-table source books, CUSIP scope and ordinary versus forced recovery; targeted parser/writer controls and discovery I/O limits. No production source/cube rebuild was performed.
- Pages: [Overview](./OVERVIEW.md), [Data sources](./reference/data-sources.md), [Recovery](./guides/large-backfills-and-recovery.md), [Testing](./guides/testing.md).
- Evidence: local ignored reports/validate/2026-10-11-13f-backfill-performance/ contains original research, approval, RED/GREEN, actual-source/fixture timings, read-only validation and independent correctness/Ponytail reviews. Cached-download and source-rejection behavior remain disclosed limits.

## 2026-10-12: refresh — material legacy 13F parser repairs

- Profile: internal/standard
- source_commit: 8f3ffae075aa9dec4f7427dc8a8f1f960688f4c3
- Coverage: truthful nullable legacy declarations, literal security identifiers, header/quantity/voting/subtotal parsing and positive complete-source accounting in the existing two holdings destinations. Shared generic economic and metadata rules prevent omitted pages from certifying partial books. The final parser preserves all 99 accepted full payloads across 126 source decisions.
- Pages: [Overview](./OVERVIEW.md), [Data sources](./reference/data-sources.md), [Table catalog](./reference/table-catalog.md), [Data access](./guides/data-access.md), [Recovery guide](./guides/large-backfills-and-recovery.md).
- Evidence: local ignored reports/validate/2026-10-12-13f-legacy-parser-errors/ records 431 passing regression tests, four optional skips, 178 focused independent checks, 74 ordinary saver controls and independent domain validation of 111 accessions/47,259 positions per destination with fourteen causal prefixes and zero findings. Exact pinned checks pass. Failed source-accounting iterations remain preserved.
- Limits: production validation abstains because the scoped old holdings lack verified accessions; no production source or cube rebuild occurred. The five-book cached benchmark adds about 0.052s of source checking and does not establish full-history throughput. Minor unsupported text layouts and independent-oracle gaps remain explicit.
- Planned only: filing-date history from 1999-04-01, HR/HR-A only, EDGAR before ZIP/XML coverage and normal completion of old gaps, missing originals and missing roster copies behind later frontiers. Runtime starts in 2013 until that separate phase is implemented.

## 2026-10-13: refresh — extraction CLI options

- Profile: internal/standard
- source_commit: 95cc128343f1124f8bd7c8d68e9f26c2ee2868f5 (working-tree CLI refactor)
- Coverage: every extraction option is declared in the shared constants module and applied directly as an ARGS/KWARGS pair. Full input processing uses `-F` / `--full`; destructive facts recovery retains `fundamentals --rebuild`. ZIP-only insider replay uses `insider-zip -F`; gap-check output uses `--report`. Source operations retain their existing behavior.
- Pages: [Constants and configuration](./modules/constants-and-configuration.md), [Run the pipeline](./guides/run-the-pipeline.md), [Recovery guide](./guides/large-backfills-and-recovery.md), [Data extraction](./modules/data-extract.md), [Configuration](./reference/configuration.md), [Data sources](./reference/data-sources.md), and [TODO](./TODO.md).
- Validation: 39 focused CLI and DAG tests pass, with zero live-write attempts. All 165 option decorators use the shared declarations; all 43 command help pages render and their option names match callback signatures. Pyright reports zero errors. Ruff is not installed; no production fetch, source-table write or backfill ran.

## 2026-10-13: refresh — aggregation CLI options

- Profile: internal/standard
- source_commit: 639bcbf3656e51783733b341b54d342f73755c4a (working-tree aggregation CLI refactor)
- Coverage: all 18 aggregation option decorators consume shared CONFIG and FULL declarations directly; local dictionary copies and their Any import are removed. Command names, flags, build dispatch, assembly and status behavior remain the same.
- Pages: [Cube aggregation](./modules/data-aggregate.md) and [Constants and configuration](./modules/constants-and-configuration.md).
- Validation: four focused CLI/registry/DAG tests pass, with zero live-write attempts. All ten help pages render; offline dispatch checks preserve default/full modes, config paths, assembly calls and status exit codes. Pyright reports zero errors. Ruff was not run; no cube build or production data write ran.

## 2026-10-13: refresh — SEC extraction form registry

- Profile: internal/standard
- source_commit: 639bcbf3656e51783733b341b54d342f73755c4a (working-tree SEC form refactor)
- Coverage: seven constants form lists, employee and filing-text lists, and every active consumer now read the owning table's Resume.forms. Registrant policy, the EDGAR index, annual subsets, facts-history precedence, employee validation and continuity filters derive from the registry. All former constants forms were already registered. Unregistered 8-K12B is no longer retained during index parsing; transitional 10-KT forms are outside the registered facts continuity filter. Existing cached index files were not rewritten.
- Pages: [Table catalog](./reference/table-catalog.md), [Data sources](./reference/data-sources.md), [Data extraction](./modules/data-extract.md), [Constants and configuration](./modules/constants-and-configuration.md), and [Configuration](./reference/configuration.md).
- Validation: 421 focused extraction/parser/contract/validator/continuity checks pass; three optional 13F integration checks skip (two disposable PostgreSQL checks and one live-source amendment check). No live-write attempt. A 22-file Pyright comparison introduces zero diagnostics against HEAD; its 72 existing diagnostics remain. The employee validator and shared continuity helper pass Pyright. Additional historical-script and employee-test checks expose existing unrelated typing debt. Ruff is unavailable. The first 29 assertions passed but system-temp cleanup failed; the workspace-temp rerun completed normally.
- Evidence: the ignored [registry audit](../reports/validate/2026-10-13-sec-form-registry/registry.json) records 11 form-declaring contracts and 26 distinct forms. The main typing baseline/current diagnostics are in `.cache/sec-form-types/`. No production extraction, source request, data write, DDL application or backfill ran.

## 2026-10-13: refresh — extraction source dates

- Profile: internal/standard
- source_commit: 80a13a87fdbc7e2ad5cb56b76c2278e265eec0a8 (source-date registry refactor; user commits continued during implementation)
- Coverage: extraction history floors come from Resume.source_start, including Sharadar membership, SEC insider/statement/notes/FTD archives, transcripts, roster captures, earnings surprises and the shared EDGAR index. The former first-date/year globals and their test/script imports are removed. EDGAR's existing 1993Q1 boundary is declared on its document tables and the shared index derives the earliest registered floor. Roster reference snapshots have a history contract without a daily freshness claim.
- Date semantics: 13F history keeps its 1999-07-01 source floor. Resume.coverage_start holds the distinct 2013-06-30 broad XML boundary used by the roster activity gate and aggregate level/delta cutoffs. Resume.observed_from holds the 2026-09-01 notes/pension archive-clock cutoff, preserving estimated historical clocks and observed new-download clocks. FTD generation now honors the July 2009 source month and a half-month boundary.
- Pages: [Table catalog](./reference/table-catalog.md), [Data sources](./reference/data-sources.md), and [Data extraction](./modules/data-extract.md).
- Validation: the first extraction run passed 170 checks, skipped one absent cached insider ZIP and deselected one live-source smoke check; its one stale expectation still requested the user-removed 2017-12-29 RegSHO exception. That expectation was corrected. The final affected-area run passed 97 checks with one optional 13F integration check skipped, including the entire short-interest file, table contracts, FTD boundaries, roster gates and institutional aggregate cutoffs. Archive-clock, Sharadar and earnings-surprise paging checks passed in the extraction run. Live tripwires recorded zero blocked attempts. Across the 23 checked Python files, five-file HEAD comparisons match all 46 existing typing diagnostics and introduce none. Ruff is unavailable; git diff --check passes.
- Evidence: the read-only [source-date registry audit](../reports/validate/2026-10-13-source-date-registry/registry.json) validates date metadata for all 29 resume contracts and finds no first-date/year globals in src or scripts. Typing evidence is in `.cache/source-date-types/`. No production source request, data write, DDL application or backfill ran.
