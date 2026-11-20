---
title: Data access and storage
description: Rules for bounded DataStore reads, safe writes, point-in-time data, and incremental processing.
type: guide
tags:
  - wiki
  - guide
  - database
  - point-in-time
source_commit: 77d1016e79b9d08c289c5894b1282f30a84ea733
integration_parents:
  - f06e3781ccb5756a0a1406895182de37329e282b
  - 5f7b40bc4dc8379f2e58e8d32acd82a30143b9d1
---
# Data access and storage

## Goal

Read and write tabular pipeline data without bypassing the SQL boundary, exhausting memory, breaking point-in-time semantics, or creating a second table contract.

## One SQL boundary

The [data-store package](../../src/data_store/store.py) is the only package that may know SQL exists. Outside it, do not import SQLAlchemy, call pandas SQL methods, reach through `store.engine`, open raw connections, or query `information_schema`. [test_store_boundary.py](../../tests/data_store/test_store_boundary.py) enforces this boundary.

If [DataStore](../../src/data_store/store.py) cannot express a required operation, extend its API and test both PostgreSQL and SQLite behavior. Do not add a one-off escape hatch.

## DataStore surface

| Group | Operations |
| --- | --- |
| Introspection | `exists`, `columns`, `row_count`, `bounds`, `max_date`, `max_date_by`, `key_stats`, `distinct` |
| Reads | `load`, `iter_load`, `summarize`, `load_cohort`, `same_rows` |
| Transactions/staging | `transaction`, `temporary` |
| Writes | `insert`, `update`, `save`, `replace`, `append_tail`, `bulk_seed`, `delete`, `drop`, `ensure_columns` |

Pass a `Table` from [schema.py](../../src/data_store/schema.py). New code must not pass a string name even when the resolver accepts legacy strings.

There is no `existing_dates` method. Fetchers plan through their `Resume` contract and [resume.py](../../src/data_extract/utils/common/resume.py), whose reads are `max_date`, `max_date_by`, `key_stats` (first, last and count per key in one `GROUP BY`) and `distinct`.

## Reading safely

Always combine projection and row scope on a large table:

```python
df = store.load(
    Tables.sec13f_hr,
    project=True,
    where={"ticker": ["AAPL", "MSFT"]},
    since="2024-01-01",
)
```

Projection rules:

- use `columns=[...]` for an explicit consumer contract;
- use `project=True` for the registry's `read_columns`;
- never pass both;
- declare tolerated live-schema gaps in `optional_columns`;
- use `iter_load` for cube-sized or vector-heavy reads.

[`iter_load`](../../src/data_store/store.py) requires a projection and enables server-side streaming. Its `order_by` parameter orders the complete result before chunking, allowing consumers to preserve whole calls or other groups across boundaries. Exhaust or explicitly close its generator because it owns a pooled connection for its lifetime. Use vector-sized chunks for embeddings: the [text builder](../../src/data_aggregate/transformers/step_cube_text.py) uses 5,000 rows, carries unfinished calls, and closes the cursor even when feature calculation fails.

Row filters are parameterized and validated against known columns. Scalar values compile to equality, collections to deterministic `IN`, `None` to `IS NULL`, and `store.NOT_NULL` to `IS NOT NULL`. An empty collection matches nothing, never everything. `where` also accepts a DataFrame containing exactly the complete non-null registry primary key, or a flat list of dictionaries/key frames combined with OR. Empty key frames/lists match nothing; an empty dictionary inside OR is rejected. Overlapping branches return each row once. `since` and `until` operate on the registry date column; `until` includes the requested day even for timestamps.

Do not constrain the same date column through both `where` and `since`/`until`; the store raises rather than hide the error behind an empty result.

Identity is read through one object. A `data_extract` step takes CIK windows, symbol intervals, CUSIP lines and fetch bounds from `Identity` (`load_identity(context)` in [identity.py](../../src/data_extract/utils/common/identity.py)), never from its own read of `universe_cik_symbol`, `universe_cusip`, `symbol_tenure` or `universe_tickers`; `test_identity_reads_only.py` enforces it. `data_aggregate`, which cannot import `data_extract`, reads the same published tables through `read_identity` in [identity_tables.py](../../src/utils/identity_tables.py).

## Missing and empty tables

`load` raises `TableMissingError` or `TableEmptyError` by default. This keeps an unrun source visible.

Use `optional=True` only when absence is a legitimate state, such as a cold-start resume probe or an explicitly optional feature source:

```python
df = store.load(Tables.def14a_llm, optional=True)
if df is None:
    ...
```

Branch on `is None`, not `.empty`.

## Marker rows

A table that declares `Table.empty_marker` holds one marker row per filing that was read and holds no data for its key. `load` and `iter_load` drop marker rows unless `markers=True`. Resume and maintenance code use these rows; the 13F feature readers also require them because their markers are verified accession-completion manifests, including source-complete empty books. `distinct`, `max_date`, `max_date_by` and `key_stats` count markers, and so does raw SQL outside the store.

## Writing safely

| Method | Use it for | Contract |
| --- | --- | --- |
| `save` | Normal fetcher output | Upsert on registry PK after table/column checks. |
| `replace` | Full rebuild | Managed tables retain DDL; unmanaged cube parts are dropped and recreated. |
| `append_tail` | Incremental aggregate part | Delete and rewrite from a day boundary; inclusive mode lets forward labels mature. |
| `transaction` | Owner-controlled atomic work | Use the yielded store for the initial guarded reads and every write; no inner commit. |
| `insert` / `update` | Prepared rows / explicit scoped patches | Strict insertion rejects conflicts; updates require a non-empty scope. |
| `bulk_seed` | Later chunks of a streamed replacement | Append to an already-created schema; unknown columns raise. |
| `delete` | Explicit scoped cleanup | A non-empty `where` clause is mandatory. |

Existing managed schemas are assumed current: ordinary writes reject missing fields instead of issuing adoption ALTERs. Fresh databases use `sql/schema.sql` through Docker initialization; explicit generic `ensure_columns` remains available for deliberate administration. Part replacement drops unmanaged tables so retired features disappear physically. Domain-specific source and identity writers live beside their extractors, load each table once at the start of a named phase, then clean, merge, compute, check/report and save through the generic bound store. A fresh locked recheck is its own phase. FINRA symbol inventory/cohort expansion and compact summaries at different grains are bounded planning dependencies.

Tabular data belongs in PostgreSQL. Do not substitute CSV or Parquet as an application table. Models, plots, caches, and the peer dictionary are non-tabular artifacts and belong under `context.paths`. Earnings-call transcripts are tabular: they live only in `earnings_call_sections`, with no file cache.

## Incremental extraction

Plan from the database, never from a file or a full table read. Each extracted table declares a `Resume` contract in [schema.py](../../src/data_store/schema.py), and [resume.py](../../src/data_extract/utils/common/resume.py) turns it into a work list:

- `series_windows` (dated per-key series): each key's own last date minus the overlap, full history for an absent table or for a new key until its stored history reaches the floor (within the table's overlap; after that it resumes like any key; short interest raises the floor to the key's first `prices` date), the table-wide frontier minus the overlap for a rowless key, plus one window per run of sessions missing inside the key's stored span;
- `document_worklist` (EDGAR documents): the local filing-index rows of the table's forms for each key's registrant lineage, from the floor on, minus the accessions already stored, markers included; newest first and capped per run;
- `archive_worklist` (bulk archives): missing or incomplete configured source units. Completed units are reused on normal runs; deliberate historical replay is explicit. The downloader separately reuses validated cached files and downloads missing published files.

Fetch rules:

- A key's frontier is its latest stored date in the contract's frontier column, markers included, so a filing read with no data is not listed again.
- The market-wide 13F walk subtracts stored complete accessions from its configured filing window. A ticker subset also stops at the stored table-wide filing frontier unless an explicit filing window is supplied. A FINRA day file covers every symbol, so short interest reads the union of the days its keys need.
- Save completed entities or durable chunks immediately.
- Catch provider failures per ticker/filing and continue, but re-raise programming errors in repository code.
- A missed or late SEC filing heals on the next run, because the work list is the filing index minus the stored accessions; `-F` only re-reads stored history.

The source grain, not the destination primary key alone, determines the correct frontier.

### 13F work list and progress

The [13F fetcher](../../src/data_extract/utils/institutionals/fetch_13f.py) reads the configured window's metadata from the shared cached SEC filing index, subtracts completed filings with vectorized DataFrame operations, then sorts the pending rows and creates `Filing` objects only for those rows. The extraction loop and `tqdm` receive only pending filings; a run with nothing pending creates no filing objects or progress bar. Logs report listed, already-complete and pending filing units separately, and the saved summary counts successful pending units.

Completion comes from the existing `sec13f_hr` table through `resume.stored_accessions`: one table-wide `SELECT DISTINCT accession_number` filtered to `parse_status = 'complete'`. Filing dates and holdings payload columns are not read for that comparison. When the index lists an accession under several filers, one additional scoped read projects only `cik` and `accession_number` for those filers and preserves completion per filer. Complete empty books count as done; failed reads and `invalid` markers remain pending for the EDGAR walk. Original filings and amendments keep separate accession identities.

Explicit full/recovery mode bypasses the completion subtraction inside the same filing bounds. Completion matching uses the existing table without an added database index or a separate tracking file. The narrow projection limits data returned to Python; PostgreSQL can still scan the table to find complete markers.

## Point-in-time rules

1. Make information available on its publication or filing date, never the period it describes.
2. Keep originals and amendments distinguishable.
3. Join publication snapshots backward, never forward.
4. Apply source-specific reporting lags and completeness frontiers before feature construction.
5. Do not forward-fill across a period where the source was not observable.
6. Preserve the difference between unavailable, missing, and observed zero.

The fundamentals path demonstrates the pattern: accession-grain facts keep original filings, the SEC replay emits complete filing-date snapshots, the Sharadar history reads only Sharadar, and the cube joins three SEC columns on the exact `(ticker, as_of)` key. See [point-in-time data](../concepts/point-in-time-data.md).

Governance later-filing attribute fills and the pooled13F unit/value repair keep their original retrospective evidence rules. Destination publication buffers and bounded writes still apply. These approved exceptions do not establish strict donor-level causality; see [configuration and recovery](../reference/configuration.md#cube-timing-and-recovery).

## Incremental cube parts

[incremental.py](../../src/data_aggregate/utils/common/incremental.py) provides the shared lifecycle. Each part resolves one `WindowPlan` from the Context run, its table policy and the retained market calendar:

```python
calendar = load_trading_calendar(context)
window = plan_window(context, Tables.cube_part_momentum, calendar)
if window is not None:
    # Compute panel with window.read and the required retained source context.
    write_part(context.store, Tables.cube_part_momentum, panel, window, drop_empty=True)
```

[data.yml](../../configs/data.yml) owns source extraction starts, overlap references, source publication columns/lags and explicit cube input-history maps. Raw prices define root output dates; saved price dates/keys define downstream output. Cubes have no independent start/end/PIT block. Each input integer is additional calendar days before actual rewrite start; `null` is full stored history and zero includes the first rewritten date. Missing dependencies fail. The part registry retains binding mathematical look-back checks; it supplies no fallback budget. Substeps reuse explicit resolved Windows and preserve entity/column projections. Insufficient finite budgets refuse instead of widening.

Writes replace only the plan's authorized dates and keys in one transaction. A failed or unavailable input is distinct from a valid empty correction. Invalid candidate keys or unexpected schema changes refuse persistence; widening retained history requires an explicit rebuild. Reading older history or probing historical column support does not authorize a wider write.

SEC13F retains complete sparse books and original calendar ordinals for its matched comparisons, holding duration and public-book age. The approved bounded-memory path retains one broad source stream and two complete-roster passes for consensus and final repair, using identical declared source bounds. Daily projection uses the actual write interval; older projection blocks establish retained column support when necessary. Its normal write overlap is 100 calendar days, as configured in data.yml. A failed previously saved source family or missing required price coverage inside the replacement interval aborts persistence; an untouched older prefix does not force a wider rewrite.

`StepCubeInstitutionals` retains the history needed for cumulative state and computes each family once. Its normal write overlap is 30 calendar days. The root Context run controls full versus incremental mode; there is no separate `run(full)` or lower-level overlap default. A split or a larger read history does not silently widen the authorized write interval.

Governance keeps its original later-filing fills and five-year daily self-history context. Its configured 380-calendar-day write tail leaves older corrections for an explicit `build-governance -F` and subsequent assembly. Text separately uses its configured seven-calendar-day feature tail and required emitter/predecessor calls. Assembly inherits the earliest actual write start among its selected components, resolving mixed units through dates rather than taking a maximum of unlike integers.

The scoped full-versus-rerun checks compare actual values, keys and nulls, deliberately repair a wrong saved cell inside the write interval, and check genuine older witnesses where available. They validate the observed sample and retained calculation context rather than certifying a full production backfill.

## SEC and XBRL handling

- Coalesce a priority-ordered candidate tag set per period.
- Derive missing accounting identities only from explicit components.
- Apply non-negative guards and per-issuer deny rules as rejection layers, not hardcoded tag pins.
- Convert booleans to numeric indicators when they become model features.
- Join entities by CIK/CUSIP/accession, not free-text names.
- Keep identifiers as text.
- Treat multi-class shares, consolidated income, Up-C ownership, amendments, and dimensioned facts as specialized contracts; consult [data sources](../reference/data-sources.md) before changing them.

## Artifact boundary

[Context](../../src/context.py) resolves the root, store and artifact paths from configuration. Archive files are source caches: download planning can reuse validated files and queue missing published units from the configured inventory. Database Resume/completion worklists determine ingestion; a file modification time never supplies a business publication date or an ingestion frontier.

Both `data/` and the PostgreSQL volume are risk zones. Ask before modifying or deleting either.

## Checklist

Before merging a data-path change:

1. confirm the table declaration and grain in the [table catalog](../reference/table-catalog.md);
2. use `Tables.<name>` and the store facade;
3. project and scope every large read;
4. model absence explicitly;
5. choose the frontier that matches provider grain;
6. preserve publication time and amendment history;
7. declare look-backs and test incremental equivalence;
8. run a targeted real-data sanity check; and
9. update [live database state](../reference/live-database.md) only after re-measuring it.

## Related

- [Data platform](../architecture/data-platform.md)
- [Data store module](../modules/data-store.md)
- [Table registry and store boundary](../concepts/table-registry-and-store-boundary.md)
- [Add a data source](./add-a-data-source.md)
- [Add a cube feature](./add-a-cube-feature.md)

## Candidate 13F accession persistence and causal reads

`save_13f_filing(context.store, manifest, holdings)` in [fetch_13f.py](../../src/data_extract/utils/institutionals/fetch_13f.py) is the bounded atomic entry point: one complete accession, any observed cover declarations and the actual typed source payload reconciled, unique CIK/accession/CUSIP grain. A legacy source may omit cover count/total only after the reader positively accounts for its structure and economic rows; missing declarations stay null and cannot be replaced by parsed totals. The existing unit-evidence field carries the source-accounting conclusion; callers cannot waive checks by supplying nulls alone. Holdings and completion manifest commit together. Failed parsing or a rolled-back transaction cannot establish an empty public book. Exact replay returns zero; a changed checksum for an existing accession raises visibly. Resume is by completed accession, so a stored amendment cannot hide a missing original.

`sec13f_hr` is read one report quarter at a time: [read_quarter_lines](../../src/data_aggregate/utils/sec_thirteen/cluster_lines.py) (cube) and `StepManagerProfile._read_lines` (profile) take one projected `context.store.load` per chunk of CIKs on the primary key, keep the completion markers and never read the whole table; the cube caches each report period for one run. The shared candidate resolver `src/utils/institutional_vintages.py` still filters public, complete source versions per CIK and report quarter for its other callers; the cluster features use originals only, filed by the 45-day deadline.

Candidate per-manager publications include active current/prior changes, including value-only corrections. An unrelated manager's event does not emit another manager's unchanged current rows. The SEC 13F step now separates full sparse dependencies from bounded daily projection and guards its inclusive output rewrite. Arbitrarily old delayed corrections, source completeness and downstream cube propagation remain separate refresh follow-ups (`reports/validate/2026-10-09-institutional-13f-audit/todo-13f-refresh.md`, local ignored audit artifact). Scoped replay does not establish production adoption.

A book needs only `book_known` to count its holdings, tenure and pairs; `zero_eligible` (absence supports a zero) decides whether an omitted position is 0 or unknown, so confidential and combination filings list what they hold and leave the rest NaN. `comparison_eligible` no longer gates features. Present known quantities can compare without inventing omitted holdings. Inferred position patches mask previously positive or unknown typed common/call/put/debt legs when the incoming leg is merely structural zero; later inferred packets preserve that unknown. `prior_resolution_status` exposes inferred baselines even when the current book is a clean original. Only a clean original/restatement clears inference. Failed/unavailable source histories are outside the verified complete scope, and must not be certified for production by the successful subset.

## Text output windows

[StepCubeText](../../src/data_aggregate/transformers/step_cube_text.py) separates source acquisition, cached NLP work and feature writes. Routine earnings-call acquisition checks the configured 180-calendar-day source range. NLP stages compare compact source keys within their configured source-prefix bounds and compute only missing or invalid outputs; they reuse valid cached sentiment and embeddings. Normal feature writes start at the stored text frontier minus the configured seven calendar days. Only active emitters and their actual adjacent fiscal predecessors are selected for those feature dates. Paragraph payloads and sentiment/price frames are reused, and vectors stream in ordered bounded chunks. Older source revisions require an explicit full feature rebuild and subsequent assembly, with valid NLP outputs reused.
