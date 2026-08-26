# Phase 1 — Extraction ✅

**Goal**: four Sharadar tables populated with real rows for the **30** entitled DJIA tickers, via a
resumable per-ticker fetcher wired into a new step and the CLI.

**Prerequisite**: phase 0 is DONE (2026-08-26) — `fundamentals_history` is free and is now the
forward-declared name of the phase-4 merged table.
**Read first**: [README.md](README.md) — decisions D1–D13 govern this phase.
**Next**: [phase-2-diagnostics.md](phase-2-diagnostics.md)

---

## What exists today

- No `sharadar` / `nasdaqdatalink` / `quandl` reference anywhere in the repo except the spec file.
- **No new Python dependency is needed.** [src/utils/polite_http.py](../../../../src/utils/polite_http.py)
  covers the whole HTTP surface (`http_get` → curl_cffi with retry, backoff, per-host throttle
  ratchet). The same pattern as `fetch_roic_transcripts.py` and `fetch_cusip_map.py`.
- `Tables` is the single table registry; `ALL` / `MANAGED` / `BY_NAME` are comprehensions over
  `vars(Tables)`, so **declaration order IS the `sql/schema.sql` emission order** and there is no
  second list to update.

---

## Changes

### 1. `.env` — user action

Rename the key (D6):

```
SHARDAR_API_KEY=...   ->   SHARADAR_API_KEY=...
```

The loader reads **only** `SHARADAR_API_KEY` and raises a `RuntimeError` naming the variable if it
is absent. Do not add a silent fallback to the misspelled name — a silent fallback is how the typo
survives forever.

### 2. `src/constants/constants.py`

Literals only (URLs, formats, closed vocabularies). Nothing tunable — tunables go to `configs/`.

- [x] `SHARADAR_BASE_URL = "https://api.sharadar.com/v1.0"`
- [x] `SHARADAR_API_KEY_ENV = "SHARADAR_API_KEY"`
- [x] `SHARADAR_DIMENSIONS = ("ARQ", "ARY", "ART")` — D8. MR* excluded by decision, with the
      one-line reason in a comment: they mutate in place.
- [x] `SHARADAR_SF1_COLUMNS: tuple[str, ...]` — the full 112, in delivered order, as the contract a
      response header is validated against (see the `fields=` trap, README fact 4).
- [x] `SHARADAR_ID_COLUMNS = ("ticker", "dimension", "calendardate", "date", "reportperiod",
      "fiscalperiod", "lastupdated")` — the 7 non-numeric columns. **Everything else in SF1 is a
      value column and must be cast to `float64`.**
- [x] `SHARADAR_ZERO_FILLED_FIELDS: frozenset[str]` — the 41 the vendor documents as zero-filled.
      Phase 1 only records the list; phase 2 measures it and phase 3 acts on it.

### 3. `configs/configs.yml` — risk zone, ask before editing

Under `data_extract:`, a sibling of `years_history` (D3 — depth is a knob, never hard-coded):

```yaml
  # Sharadar's own history knob, separate from `years_history: 15` because the two sources have
  # different entitlements: the SEC walk is limited by patience, Sharadar by subscription tier.
  # The current key grants ~5 years (measured: AAPL ARQ earliest 2021-10-29). Raise to 28 on the
  # Full tier -- it is a config change, not a code change.
  sharadar_years_history: 5
  # Seconds between per-ticker requests. Sharadar documents NO rate limit anywhere; this is
  # deliberate conservatism until measured, not a known constraint.
  sharadar_request_pace: 0.5
```

### 4. `src/data_store/schema.py` — risk zone, ask before editing

Four `Table` entries in a new `# Extract -- fundamentals (Sharadar)` block, placed **immediately
after** the existing fundamentals block so the emission order stays readable.

```python
sharadar_fundamentals = Table(
    "fundamentals_sharadar",
    ("ticker", "dimension", "date", "reportperiod"),
    date_col="date",
    date_type_cols=("date", "reportperiod", "calendardate", "lastupdated"),
    freshness="quarterly",
    read_columns=(...))          # REQUIRED: 112 cols x 3 dimensions is a wide table
```

- [x] `sharadar_fundamentals` as above. `read_columns` must be set — the repo's rule is "never read
      a large table unprojected", and this is the widest extract table in the schema.
      Set it to the identifier columns plus the ~45 the field map consumes; a consumer needing more
      passes `columns=` explicitly.
- [x] `sharadar_tickers` — pk `("table", "permaticker", "ticker")`, no `date_col`,
      `kind=KIND_REFERENCE`, no freshness. Carries `permaticker`, `currency`, `category`,
      `firstquarter`, `isdelisted`.
- [x] `sharadar_actions` — pk `("date", "ticker", "action", "contraticker")`, `date_col="date"`.
      ⚠ CHANGED from the plan's `("date","ticker","name","action")`, which was measured to
      collapse 11 rows into 3. See deviation 1.
- [x] `sharadar_sp500` — pk `("date", "ticker", "action")`, `date_col="date"`.

⚠ **`date` is a reserved-ish column name and it is in three PKs.** Every `store` call must pass the
`Table` object, never a string literal, so the date column travels with the name.

### 5. `sql/schema.sql` — risk zone, ask before editing

Four blocks in the generated format, in the same order as the registry:

```
-- [extract] fundamentals_sharadar  (pk: ticker, dimension, date, reportperiod)

CREATE TABLE IF NOT EXISTS "fundamentals_sharadar" (...);
CREATE INDEX IF NOT EXISTS ix_fundamentals_sharadar_date ON "fundamentals_sharadar" ("date");
```

Note the two spaces before `(pk:`, `NOT NULL` on every PK member, `PRIMARY KEY (...)` last.

⚠ This file is **never applied to the live volume** (initdb only). Tables are created by
`store.ensure_table` on first write. The file edit keeps a *fresh* DB correct; it does not create
anything for you.

### 6. `src/data_extract/utils/fundamentals_sharadar/` — the new package (D10)

```
__init__.py
client.py               # HTTP + paging + response-header validation + entitlement handling
fetch_sharadar.py       # the four fetchers, resumable
```

#### `client.py`

- [x] `_api_key() -> str` — reads `SHARADAR_API_KEY`, raises with a clear message if absent.
- [x] `sharadar_get(context, table: str, **filters) -> pd.DataFrame | None`
  - builds `{SHARADAR_BASE_URL}/data/{table}` with `api_key` + filters
  - **always passes an explicit `date.gte`** where the table has a date column. The API's `from`
    defaults to *"1 year ago"* and `sort` to `date.desc` — omitting either silently truncates.
  - passes `limit=10000` and `sort=date.asc` explicitly.
  - parses CSV into a DataFrame.

- [x] ⚠ **403 handling is the sharpest trap in this phase.** `polite_http.http_get` treats 403 as
      *rate-limited* and retries 4 times with exponential backoff (`141-172`). Every non-entitled
      ticker returns 403, so a naive loop over the S&P 500 would spend **minutes per ticker** doing
      nothing. Call it with `retries=0, log_missing=False` and classify the status yourself:
      `200` → data, `403` → **not entitled, skip and count**, anything else → retry via a second
      call that does use the retry path.
      The run must end with a single summary line: `N entitled, M not entitled (403)`.

- [x] ⚠ **Validate the response header against `SHARADAR_SF1_COLUMNS`.** `fields=` silently drops an
      unavailable field with no warning (README fact 4). A missing expected column is an error, not
      a shrug.

- [x] `cast_value_columns(df) -> pd.DataFrame` — hard-cast every column **not** in
      `SHARADAR_ID_COLUMNS` to `float64` **before the first write**, even for fields the first
      ticker never populates. This is not optional: `ensure_table` infers SQL types from the FIRST
      DataFrame it sees, so an all-`None` object column becomes `TEXT` and every later ticker's real
      number is stored as a string. Measured live on `minorityInterest` / `restrictedCash`
      (VRT created them TEXT; APA's values came back as `'1997000000.0'`).

#### `fetch_sharadar.py`

- [x] `fetch_sharadar_fundamentals(context, tickers, *, years_history, full=False) -> None`
  - resume: `store.max_date_by(Tables.sharadar_fundamentals, "ticker")` → per-ticker
    `date.gte = max_date + 1 day`. On `full=True` (or an empty dict) use
    `today - years_history years`. D13.
  - loop `SHARADAR_DIMENSIONS`; one request per (ticker, dimension).
  - **⚠ USD assertion (D20)**: read `currency` from `sharadar_tickers`; a non-USD filer is
    **not written** and is logged at warning with the ticker and its currency. Only 8 of Sharadar's
    money columns are USD-converted, so a non-USD row mixes units inside itself.
  - `store.save(Tables.sharadar_fundamentals, df)`.
  - **Single-threaded** for now (README risk: no documented rate limits, and `ensure_table` is a
    check-then-create with no lock). If parallelism is added later, serialise the first write per
    table with a `threading.Lock` + a `created` set, exactly as
    [edgar_driver.py:94-107](../../../../src/data_extract/utils/common/edgar_driver.py#L94-L107) does.

- [x] `fetch_sharadar_tickers(context) -> None` — full refresh, `table=fundamentals` filter.
      **Must run before the fundamentals fetch** (the USD assertion depends on it).
- [x] `fetch_sharadar_actions(context, *, years_history, full=False) -> None` — resume on `date`.
- [x] `fetch_sharadar_sp500(context, *, full=False) -> None` — resume on `date`.

### 7. `src/data_extract/transformers/step_extract_fundamentals_sharadar.py` (D11)

`StepExtractFundamentalsSharadar(Step)` with `run(tickers)` as the only public method, calling the
four fetchers in dependency order: `tickers` → `fundamentals` → `actions` → `sp500`.
Logging via `self._log`. Never `print()`.

### 8. `src/data_extract/step_extract_all_data.py`

Instantiate and run it **before** `self._fundamentals` (D11), matching the existing commented-out
style of that file.

### 9. `src/data_extract/cli.py`

One command per source, matching the existing `fundamentals-facts` / `fundamentals-history` pattern
including the `-F/--full` flag:

- [x] `fundamentals-sharadar` — all four fetchers in order.
- [x] `sharadar-tickers`, `sharadar-actions`, `sharadar-sp500` — individually, so the Airflow
      extraction DAG can schedule them separately.

---

## Tests

`tests/data_extract/test_fetch_sharadar.py`. Per the repo's testing rule: **feature/economic tests
use real data**; only parsing math gets synthetic fixtures.

- [x] `test_value_columns_are_float` — synthetic frame where the first ticker has an all-`None`
      column; assert every non-identifier column is `float64` after `cast_value_columns`.
      **This is the regression test for the TEXT-column bug.** Prints the dtype summary.
- [x] `test_response_header_matches_contract` — real single-ticker call; assert the returned columns
      equal `SHARADAR_SF1_COLUMNS` exactly. Prints any diff both ways.
- [x] `test_not_entitled_is_not_a_retry_storm` — real call for a non-DJIA ticker; assert it returns
      "not entitled" and that it took **one** request, not five. Prints the elapsed time.
- [x] `test_resume_is_incremental` — real: run the fetcher twice; assert the second run writes 0 new
      rows and issues no request with a `date.gte` earlier than the stored max. Prints both counts.
- [x] `test_dimensions_stored` — assert the stored set is exactly `{ARQ, ARY, ART}` and that no
      MR* row exists. Prints the per-dimension row counts.

---

## Verification

```bash
PY="$HOME/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe"
rtk "$PY" -m pytest tests/data_extract/test_fetch_sharadar.py -v -s
rtk "$PY" -m src data_extract fundamentals-sharadar -c ./configs
rtk "$PY" -m src data_extract fundamentals-sharadar -c ./configs      # idempotence: 0 new rows
```

```bash
MSYS_NO_PATHCONV=1 docker exec pea_db psql -U alexandre -d pea -c \
 "SELECT dimension, count(*), count(distinct ticker), min(date), max(date)
  FROM fundamentals_sharadar GROUP BY dimension ORDER BY dimension;"
```

**Expected, given the measured entitlement:**

- [x] 29 distinct tickers (not 30 — `DOW` is denied), across all three dimensions.
- [x] ARQ `min(date)` ≈ 2021-10-29, `max(date)` within the last quarter.
- [x] ~20 ARQ rows per ticker (~580 total), fewer for ARY.
- [x] **Zero MR\* rows.**
- [x] The second CLI run writes **0** rows and logs the resume dates.
- [x] `information_schema.columns` shows **no `text` column** among the 105 value columns.
- [x] The run's summary line reads `29 entitled, N not entitled (403)`.

**Stop and report if:** any value column landed as `text`; the response header disagrees with
`SHARADAR_SF1_COLUMNS`; or the second run writes rows.

---

## Measured result (2026-08-26)

`python -m src data_extract fundamentals-sharadar -c ./configs`, S&P 500 universe (491 tickers
after `redundant_ticks`), ~3 min:

| table | rows | detail |
|---|---|---|
| `fundamentals_sharadar` | **1,346** | ARQ 598 / ART 598 / ARY 150, **30 tickers**, 2021-08-27 .. 2026-08-10 |
| `sharadar_tickers` | 17,826 | 16,808 USD / 1,018 non-USD |
| `sharadar_actions` | 594 | 536 dividend, 32 acquisitionof, 12 relation, 4 split, 4 spinoff, 4 spinoffdividend, 2 namechange |
| `sharadar_sp500` | 3,306 | 1992-01-02 .. 2026-08-25 |

- Summary line: `Sharadar SF1: 30 entitled, 461 not entitled (403); 1301 rows written`.
- **Column types: 105 `double precision`, 4 `date`, 3 `text`** — and the 3 text columns are
  exactly `ticker` / `dimension` / `fiscalperiod`. **Zero TEXT value columns**: the
  `cast_value_columns` guard held on the real first write.
- **Zero MR\* rows**; stored dimensions are exactly `{ARQ, ART, ARY}`.
- **Idempotent**: the second run wrote **0** rows, and both side tables logged
  "already current". All four table counts identical before/after.
- Tests: **5/5 pass** (`tests/data_extract/sharadar/test_fetch_sharadar.py`).

---

## Deviations from the plan as written

1. **`sharadar_actions` PK is `(date, ticker, action, contraticker)`**, not
   `(date, ticker, name, action)`. Measured on 1,927 live rows: the planned PK had **8
   collisions**, all `relation` rows where only `contraticker` differs (GS-PD / GS-PA / GS-PC
   are Goldman preferred series; JPM eight more). It would have collapsed 11 rows into 3
   silently. With `contraticker`: 0 duplicates. Verified post-load — all 12 `relation` rows
   are present. **Decided with the user.**

2. **30 entitled, not 29.** The plan's "29 entitled, `DOW` denied" came from a DJIA list that
   still contained `DOW`. The *current* DJIA-30 (`constants.DOW_30_TICKERS`, which carries
   `SHW` in DOW's place) is entitled **in full**; `DOW` is in the S&P 500 but not the DJIA and
   is among the 461 denied. So the expected count is 30 tickers / ~598 ARQ rows, not 29 / ~580.
   HD and NVDA have 19 ARQ rows rather than 20.

3. **`sql/schema.sql` was spliced by hand, not regenerated.** Its header says
   "AUTO-GENERATED ... do not edit by hand", but running
   `python -m scripts.generate_schema_sql` **drops 8 hand-added indexes** that are not
   derivable from the registry (`ix_fundamentals_check_finding`, `_severity`, `_cluster`,
   `_run_id`, `ix_fundamentals_check_run_scope`, `ix_fundamentals_check_status_cluster`,
   `ix_fundamentals_facts_field`, `ix_fundamentals_reason_codes_code`) and rewrites
   `fundamentals_history` as CARRIED OVER. The generated blocks for the four new tables were
   extracted and spliced in instead, so the diff is **178 insertions, 0 deletions**.
   ⚠ The generator and the committed file have drifted; that is worth its own task.

4. **No `ix_*_date` index on any of the four tables, and that is correct.**
   `ddl.table_ddl` emits a date index only `if spec.date_col not in pk`, and `date` is a PK
   member in all four. The plan's expected `CREATE INDEX ix_fundamentals_sharadar_date` line
   therefore never appears. (Note: `date` is the *third* PK column of `fundamentals_sharadar`,
   so the PK btree does not serve a `date`-only scan — immaterial at 1,346 rows, worth
   revisiting at S&P-500 scale.)

5. **`polite_http.get_once` was added** (additive, ~10 lines). The plan said to call
   `http_get(retries=0, log_missing=False)` and "classify the status yourself" — but
   `http_get` collapses *every* non-200 into `None`, so the status is not observable. The new
   helper returns the response whatever its status. Measured: a 403 now costs **1 request and
   0.14s** instead of 5 requests and >45s.

6. **`sharadar_get(context, table, /, ...)` — `table` is positional-only.** The `tickers`
   endpoint has a filter of its own named `table` (`table=fundamentals`), which would
   otherwise collide with the parameter.

7. **`.env`** — the user renamed `SHARDAR_API_KEY` to `SHARADAR_API_KEY` themselves. The
   loader reads only the correct spelling and raises naming the variable if absent.

8. **Config comments** — the user trimmed the two config comments down to one line; both keys
   parse and are read by the step and the CLI.

---

## Phase 0 — DONE (2026-08-26)

Resolved after this phase was written. The user renamed the live table in DBeaver; the code
half was completed in the same session. `Tables.fundamentals_history_sec` reads 3,258 rows;
`Tables.fundamentals_history` is forward-declared for the phase-4 merged table and reads 0.
See the README's phase-0 section for the full record.

---

## Rollback

Everything in this phase is additive. To undo: drop the four tables, delete the package, revert the
schema/constants/config/CLI edits. No existing table is modified.
