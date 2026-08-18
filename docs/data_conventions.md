# Data & DB conventions

Scope: the rules for reading and writing tabular data. Non-negotiable — the first is enforced by a
test. For what each table is, see [data_schema.md](data_schema.md).

## 1. One SQL boundary

`src/data_store/` is the **only** code in the repo that knows SQL exists. Outside it:

- no `import sqlalchemy` / `from sqlalchemy import …`
- no `pd.read_sql`, `.to_sql(`, `engine.connect(`, `raw_connection(`, `store.engine`
- no `information_schema` (it does not exist in SQLite, so such a path cannot be tested offline)

Enforced by [tests/data_store/test_store_boundary.py](../tests/data_store/test_store_boundary.py),
which greps every `src/**/*.py` except the store package and `src/utils/db.py` (the engine factory).

**If the facade cannot express your query, add the capability to `DataStore` — do not bypass it.**

## 2. The `DataStore` surface

One instance lives on `context.store`. Every method takes a `Table` from the registry or its name.

**Introspection** — `exists` · `columns` · `row_count` · `bounds(table, column=None)` ·
`max_date(table, column=None)` · `distinct(table, column, where=, order=, limit=, dropna=)`

**Reads** — `load(...)` · `iter_load(...)`

**Writes** — `save` (upsert on the registry PK) · `replace` (empty + chunked COPY) ·
`append_tail(table, df, cutoff, inclusive=)` · `bulk_seed` (COPY append onto an existing schema) ·
`delete(table, where)` (`where` is **required**) · `drop` · `ensure_columns`

There is **no `existing_dates`** (some older docs claim it). Use `max_date` / `bounds` / `distinct`.

### Filters compose server-side

```python
store.load(Tables.sec13f_hr, project=True, where={"ticker": ["AAPL", "MSFT"]}, since="2024-01-01")
```

`where` values map to predicates by type:
`str`/`bytes` → `=` · list/tuple/set/Series/Index/ndarray → `IN` (sorted, so the SQL is
deterministic; **empty collection → matches nothing**, never a full read) · `None` → `IS NULL` ·
`store.NOT_NULL` → `IS NOT NULL` · a date type → `=` bound as the column's real type.

`since=` / `until=` bound the table's `date_col` (or an explicit `date_col=`). `until` is compiled as
`< until + 1 day`, not `<= until`, so a text-typed date column and an intraday `TIMESTAMP` both
include the day you asked for.

Constraining the same column via both `where` and `since`/`until` **raises** — silently emitting
`date = a AND date >= b` would hide the mistake behind an empty result.

Everything is a **bound parameter**; an unknown column raises `KeyError` before reaching the DB.

### `load` RAISES by default

A missing or empty table is nearly always a real fault, so `load` raises `TableMissingError` /
`TableEmptyError` ([errors.py](../src/data_store/errors.py)) rather than returning a fabricated
empty frame.

```python
df = store.load(Tables.def14a_llm, optional=True)
if df is None:                       # branch on `is None`, NOT on `.empty`
    ...
```

Pass `optional=True` **only** where finding nothing is legitimate: a fetcher's resume check on a
cold DB, a genuinely optional feature source. ~48 call sites in `src/` rely on it.

## 3. Never read a full large table

Always **project** and **scope**:

- `columns=[...]` for an explicit list, or `project=True` for the table's declared `read_columns`.
  Passing both raises.
- `where=` / `since=` / `until=` to scope rows.
- `iter_load` for anything cube-sized. It **requires** `columns=` or `project=True` and raises
  otherwise: streaming an unprojected wide table defeats the purpose. It sets
  `stream_results=True`, without which psycopg2 buffers the whole result client-side and
  `chunksize` does nothing for peak memory.

> `iter_load` holds a pooled connection for the iterator's lifetime. **Exhaust or close it** —
> breaking out early leaks the connection.

Scale, so you know what an unprojected read costs: `sec13f_hr` 21.7M rows / 6.1 GB ·
`earning_calls_embedding` 8.6 GB · `fundamentals_facts` 7.8M rows / 5.2 GB · `cube` ~570 columns,
~26 GB. These are OOMs, not slow queries.

**Three projection declaration sites exist. Prefer the registry.**

| Site | Status |
|---|---|
| `Table.read_columns` in [schema.py](../src/data_store/schema.py) | **canonical** — drives `project=True` |
| `SOURCE_COLUMNS` in [utils/common/sources.py](../src/data_aggregate/utils/common/sources.py) | still live: `StepCubeExtras._load_source` calls its `project_existing`, and `tests/data_aggregate/test_cube_incremental.py` asserts against it |
| `SOURCE_COLUMNS` in [step_cube_extras.py:54](../src/data_aggregate/transformers/step_cube_extras.py#L54) | **dead duplicate** — declared but never read. Do not extend it |

A projection **must** cover every column its builder requires (asserted by
`test_cube_incremental.py`) but must also tolerate an *optional* column the live table lacks —
`read_table` resolves columns via `tbl.c[name]` and raises `KeyError` for an absent one.
`short_interest` is the case that bit: the projection listed `short_interest` / `avg_daily_volume`
unconditionally, the live table has neither, and the read died instead of degrading.

## 4. Writes

| Method | When | Note |
|---|---|---|
| `save` | the normal fetcher write | upsert on the registry PK; runs `ensure_table` then `ensure_columns` |
| `replace` | a full rebuild | managed tables are **DELETE**d (keeping their `schema.sql` DDL); `managed=False` parts are **DROP**ped and recreated, because a rebuild may legitimately *remove* a column and `ensure_columns` only ever adds |
| `append_tail` | incremental part writes | DELETE rows `>= cutoff` (a **day boundary**, never `> cutoff`), then append. Idempotent. `inclusive=True` is what the forward-looking target part needs — a label that was NaN last run matures into a value |
| `bulk_seed` | COPY-append onto a schema `replace` already created (the cube's streaming writer) | does **not** evolve the schema, and **raises** on an unknown column rather than letting COPY drop it silently |
| `delete` | targeted row removal | `where` is required and may not be empty |

**Schema evolution** is automatic: `ensure_columns` issues `ADD COLUMN IF NOT EXISTS` for any frame
column the table lacks (Postgres only; SQLite is dynamically typed). It never drops or retypes.

`replace`'s DROP-for-parts behaviour is load-bearing. With a DELETE, a removed column would persist,
`write_part`'s drift check would fire on every later run, and each incremental build would silently
become a full 15-year rebuild.

## 5. Point-in-time integrity

- **Lag every feature by filing date.** A fundamental is knowable only from the filing that reported
  it, never from the period it describes.
- `fundamentals_facts` keeps ORIGINAL and AMENDED rows as separate rows and never overwrites, so
  "what was known as of D" is answerable without exposing an amendment before its own filing date.
- **Resume from the DB**, per entity: read the stored max date and fetch forward. **Save per
  entity**, so an interrupted run never loses expensive work (LLM calls, 13F zips, API pulls).
- Catch provider errors **per ticker**, `context.log.warning(...)`, continue.
- Cache large downloads (companyfacts JSON, 13F/notes zips, HF parquet) under `data/`; re-download
  only when missing. See [bulk_cache.py](../src/data_extract/utils/common/bulk_cache.py).
- Self-heal cadence: the EDGAR listing fetchers use the manifest's last-run date as `since`, but
  force a full `years_history` relist every `data_extract.manifest_full_rescan_days` (30) so a
  filing missed by a bug, or one EDGAR posts out of order, cannot stay unseen forever
  ([run_manifest.py](../src/data_extract/utils/common/run_manifest.py)).

## 6. Incremental cube parts

One place decides full-vs-incremental:
[utils/common/incremental.py](../src/data_aggregate/utils/common/incremental.py).

```python
window = plan_window(store, Tables.cube_part_extras, full=full,
                     warmup=self._warmup(), trading_index=load_trading_calendar(store))
...
n = write_part(store, Tables.cube_part_extras, panel, window, drop_empty=True)
if n == COLUMNS_CHANGED:
    return self.run(full=True)        # the required response — always handle this
```

- `plan_window` → `PartWindow(last, since)`. `full=True`, a missing part, or no usable calendar →
  full rebuild.
- The window reaches `warmup + extra_back` trading days before the stored max. `warmup` comes from
  `parts.py::CubePart.warmup_trading_days`; `extra_back` is the target step's forward horizon.
- `write_part` returns `COLUMNS_CHANGED` (`-1`) when the stored column set differs from the built
  one. An append into a changed schema would silently misalign, so **the caller must re-run full**.
- `drop_empty=True` (feature parts) drops `(date, ticker)` rows where *every* feature is NaN — the
  merge-based builders left-join onto the full universe grid, so persisting them would store the
  whole 1.85M-cell grid per part regardless of sparsity.

This is only correct because the builders are backward-looking (window ≤ warm-up) and the
cross-sectional standardization is per-day, so a trailing recompute reproduces the full build's tail
exactly. That equivalence is proved on the price builder by
`tests/data_aggregate/test_cube_incremental.py`. **If you add a feature with a longer look-back than
its part's warm-up, you must raise the warm-up** — `test_part_registry.py` checks each warm-up
against its declared `binding_lookbacks`.

## 7. XBRL / SEC specifics

- **Coalesce** across a priority-ordered candidate tag list per period; never take the first present
  tag. Filers split `Revenues` ↔ `RevenueFromContractWithCustomer` at ASC-606,
  `NetIncomeLoss` ↔ `ProfitLoss`, equity ±NCI. The coalesce is deliberately **era-agnostic** so a
  taxonomy migration (ASC 842, CECL, `X`→`XNet`) needs no dates.
- Derive a concept when a filer does not tag it (operating income = gross − SG&A − R&D).
- Two narrow overrides sit on top, both applied in `build_tag_frames` after the tag-map merge:
  `NON_NEGATIVE_STOCK_FIELDS` (a negative debt/asset/share count is a filer defect → the fact is
  inadmissible and the coalesce falls through) and `FIELD_TAG_DENYLIST` (per-issuer escape hatch,
  **deny never pin**, so an unlisted ticker keeps global resolution).
  **A deny-list entry is the conclusion of a diagnosis** — add one only with the evidence written
  beside it, after `fundamentals_tag_ledger` has ranked the case.
- Cast booleans to numeric `1.0`/`0.0` indicators so they are usable as model features.
- Reconcile SEC entities **by identifier** (CIK, CUSIP→ticker via OpenFIGI) — never by free-text name.
- Store identifier-like numeric codes (`cik`, `cusip`, `adsh`) as **TEXT** to preserve leading zeros.
- Multi-class share counts, consolidated-basis income and the Up-C gross-up are a documented,
  subtle area — read [data_sources.md](data_sources.md) before touching `shares_outstanding`.

## 8. Artifacts are not tables

Non-tabular output goes to `context.paths`, never the DB, and never parquet-as-a-table:

| `paths[...]` | Holds |
|---|---|
| `ROOT` | `./` (or `$ROOT_PATH`) |
| `DATA_STORE` | `./data` |
| `OUTPUT_DIR` | `data/output` |
| `MODELS_DIR` | `data/output/models` — boosters (`.txt`), linear members (`.pkl`), `metadata.json` |
| `SECTOR_PEERS_PATH` | `data/output/sector_peers.json` — the peer dict |
| `SEC_13F_INSIDERS_DIR` | cached 13F zips |
| `CUBE_CV_RESULTS_PATH`, `DEF14A_LLM_PATH` | diagnostics parquet; the second only anchors a `_meta.json` sidecar (no parquet is written to it) |

Conversely: **never write tabular data to parquet or CSV.** The Postgres volume and `data/` are both
non-recoverable — treat them as a risk zone.
