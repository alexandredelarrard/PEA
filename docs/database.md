# Live database state

Scope: what is **actually in the local Postgres right now**. For what each table *means*, see
[data_schema.md](data_schema.md). For how to connect, see [runbook.md](runbook.md).

> **Snapshot taken 2026-08-17** by querying the running `pea_db` container. Re-verify before relying
> on a number — `MSYS_NO_PATHCONV=1 docker exec pea_db psql -U alexandre -d pea -c "…"`.

Container `pea_db` (postgres **16.14**), database `pea`, owner role **`alexandre`**, volume
`stock_pick_strat_pgdata`. **24 GB**, **31 tables**, all in schema `public`.

## Read this first: the registry declares 48 tables, the DB has 31

**Missing entirely** — every read of these raises `TableMissingError`:

| Missing | Consequence |
|---|---|
| **`prices`** | The single most load-bearing gap. `StepCubePrices` is its only reader, and every other cube sub-step reads the *part* it produces — so **the whole cube build is blocked**, as are `tests/conftest.py::real_frames` and every fixture derived from it (they call `pytest.skip`). |
| `cube`, `cube_part_*` (all 8) | no features, no training, no prediction |
| `predictions`, `predictions_latest`, `cube_signal` | no model output |
| `strategy`, `trend_asset_returns` | no ledger, no trend sleeve |
| `sec_13d`, `sec_13d_transactions` | 13D activist features unavailable |
| `notes_embedding`, `ticker_descriptions` | `notes_embedding` has no downstream reader anyway |

**Present but not in the registry** (legacy, superseded — do not read, do not extend):
`fundamentals_facts_legacy` (2.3M rows, 817 MB, 19 cols). `fundamentals_history_old` is GONE
(verified absent 2026-08-24) -- the Phase 5 rebuild dropped and recreated the four
`fundamentals_*` tables from `sql/schema.sql`, so there is no longer an old-shape history table
to read by mistake.

## Populated tables

Ordered by size. `tickers` = distinct non-null tickers.

| Table | Rows | Size | Cols | Tickers | Date column | Coverage |
|---|---|---|---|---|---|---|
| `earning_calls_embedding` | 1,375,495 | **8.6 GB** | 13 | 494 | `as_of` | 2005-10-30 → 2026-07-24 |
| `sec13f_hr` | 21,659,435 | **6.1 GB** | 15 | 497 | `period` | 1987-03-31 → 2026-03-31 |
| `fundamentals_facts` | 7,776,870 | **5.2 GB** | 30 | 445 | `filing_date` | 2011-08-12 → 2026-08-12 |
| `earnings_call_sections` | 109,899 | 1.5 GB | 6 | 494 | `as_of` | 2005-10-13 → 2026-07-24 |
| `sec_filing_text` | 34,127 | 1.2 GB | 9 | 498 | `filed` | 2011-07-27 → 2026-08-03 |
| `insider_transactions` | 1,381,478 | 497 MB | 26 | 491 | `transaction_date` | 1990-05-07 → **2026-03-31** |
| `notes_text` | 96,576 | 411 MB | 15 | — | `ddate` | 2006-12-31 → 2026-05-31 |
| `sec_8k` | 95,789 | 137 MB | 14 | 486 | `filing_date` | 2011-08-04 → 2026-08-03 |
| `wiki_pageviews` | 1,699,202 | 136 MB | 3 | 500 | `date` | **2016-07-16** → 2026-07-23 |
| `fails_to_deliver` | 993,775 | 98 MB | 5 | 499 | `date` | 2010-01-04 → **2026-07-14** |
| `short_interest` | 963,115 | 84 MB | 4 | 502 | `date` | **2017-12-29** → 2026-07-31 |
| `fundamentals_facts` | 317,036 | 118 MB | 26 | **54** | `filing_date` | 2009-07-31 → 2026-08-10 |
| `fundamentals_reason_codes` | 76,004 | 12 MB | 5 | **54** | `as_of` | 2009-07-31 → 2026-08-10 |
| `fundamentals_history` | **3,267** | 1.8 MB | **69** | **54** | `as_of` | 2009-07-31 → 2026-08-10 |
| `fundamentals_employees` | 745 | 112 kB | 3 | **54** | `as_of` | 2002-03-20 → 2026-07-29 |
| `fundamentals_check` | 11,926 | 20 MB | 23 | **54** | `run_date` | one run: 2026-08-24 |
| `fundamentals_check_run` | 35 | 104 kB | 17 | — | `run_date` | one run: 2026-08-24 |
| `fundamentals_check_status` | **0** | 32 kB | 7 | — | `decided_at` | nothing decided yet |
| `google_trends` | 388,336 | 32 MB | 3 | 500 | `date` | 2011-07-17 → **2026-07-12** |
| `cusip_ticker_map` | 145,748 | 14 MB | 2 | 19,824 | — | — |
| `notes_num` | 40,587 | 14 MB | 14 | — | `ddate` | 2007-12-31 → **2026-04-30** |
| `def14a_llm` | 6,933 | 13 MB | 45 | 497 | `as_of` | 2011-07-26 → 2026-07-28 |
| `earnings_call_sentiment` | 55,514 | 11 MB | 10 | 494 | `as_of` | 2005-10-30 → 2026-07-24 |
| `earnings_surprises` | 43,383 | 5.3 MB | 5 | 500 | `earnings_date` | 1999-08-02 → **2026-10-29** (forward-dated: scheduled future calls) |
| `dividends` | 22,060 | 3.7 MB | 3 | **413** | `date` | 2011-07-18 → 2026-07-31 |
| `ticker_embeddings` | 502 | 2.9 MB | 2 | 502 | — | — |
| `pension_facts` | 6,244 | 2.5 MB | 13 | — | `ddate` | 2008-10-31 → **2026-02-28** |
| `prices_macro` | see §sanity | — | 3 | `ticker` | `date` | ~1995 → today (per series; `fx_usdeur` from 1999, `gold` from 2000, `breakeven_10y` from 2003) |
| `macro` | 4,175 | 1.0 MB | 10 | — | `date` | 2010-08-02 → 2026-07-31 |
| `sec_def14a_director_comp` | 2,279 | 600 kB | 12 | — | `filing_date` | — |
| `sec_def14a_ownership` | 2,149 | 592 kB | 8 | — | `filing_date` | — |
| `sec_def14a_executive_comp` | 1,395 | 488 kB | 15 | — | `filing_date` | — |
| `sec_def14a_votes` | 1,177 | 368 kB | 8 | — | `filing_date` | — |
| `sec_def14a` | **329** | 160 kB | 46 | **23** | `filing_date` | 2011-09-23 → 2026-05-06 |
| `sp500_tickers` | 500 | 128 kB | 6 | 500 | — | — |

- **The three validator tables hold ONE run** (`3df52ae9af75`, 54 tickers, all tiers). They are
  written only by `src/validate/` and gate nothing. `fundamentals_check` is a LEDGER: nothing is
  ever subtracted from it, so a row-count drop against a later run of the same scope has exactly
  one cause. Runs are comparable only when their `scope_hash` matches, which is why `run_id` is
  in the primary key — two runs of different scope on one day would otherwise collide on every
  ticker they share. `fundamentals_check_status` is empty because no `wontfix` has been recorded;
  that is the only mutable state in the package.

## Coverage gotchas worth knowing before you build a feature

- **`sec_def14a` covers only 23 of 500 tickers** (329 filings). The deterministic proxy path is
  barely seeded; `def14a_llm` (497 tickers) is the one with real coverage. Any `f_ceo_*` /
  governance feature built off `sec_def14a` will be ~95% NaN.
- **The four `fundamentals_*` tables cover 54 tickers, not 500 — this is the Phase 5 rebuild
  scope, not a defect.** All four were dropped and rebuilt from scratch on 2026-08-24
  (`scripts/recreate_fundamentals_tables.py`), so the earlier 491-ticker / 239-column
  `fundamentals_history` and its 445-ticker facts table no longer exist. Their ticker sets are
  now identical by construction. Widening to the full roster is Phase 9's acceptance step; until
  then any cube built off these tables covers 54 names and every coverage rate computed against a
  500-ticker denominator will read ~11%.
- **`fundamentals_history` went 27,602 rows → 3,267 and 239 columns → 69.** Both are deliberate.
  The row count fell because the grain changed from a computed period spine to the
  **publication-event** grain (one row per date on which ≥1 extracted value became newly public)
  AND the scope narrowed to 54 tickers; the column count fell because the contract is now
  enumerated by `Catalogue.history_columns` rather than accreted. Nulls are 36.7% of value cells
  and **every one carries a `fundamentals_reason_codes` row** — the table is honest about what it
  does not know rather than forward-filling a stale value.
- **`fundamentals_facts` is strictly as-filed and keyed on `period_end`, not on fiscal labels.**
  A single filing legitimately reports the same `(fiscal_year, fiscal_period)` more than once
  (AAPL's FY2025 10-K carries FY2023, FY2024 and FY2025 annual revenue), so a label-keyed PK
  silently dropped 18,604 of 337,190 rows. Do not join or dedupe these facts on the fiscal
  labels.
- **Short history is genuinely short for some sources**: `short_interest` starts 2017-12,
  `wiki_pageviews` 2016-07. A 1260-day (5y) rolling window on those covers far less of the panel
  than the same window on prices.
- **`dividends` has 413 tickers**, not 500 — the ~87 non-payers correctly have no rows. Do not read
  a missing dividend row as missing data.
- **`earnings_surprises` extends to 2026-10-29**, past today. Those are *scheduled* future
  earnings dates with no actual. Filter on `eps_actual IS NOT NULL` for realized-surprise features.
- **`insider_transactions` stops 2026-03-31** and `pension_facts` 2026-02-28 — both are quarterly
  bulk-zip sources with a real publication lag, not stale extraction.
- **`sec13f_hr` reaches back to 1987** but the universe is today's S&P 500; survivorship applies.
- Vector columns (`embedding` on `earning_calls_embedding`, `ticker_embeddings`) are Postgres
  `float8[]`. **SQLite's driver refuses to bind a Python list**, which is why
  `tests/conftest.py::FakeStore` exists alongside `sqlite_store`.

## Handling the missing `prices` table

Options, in order of preference:

1. **Re-extract**: `python -m src data_extract price-history` (heavy — 15 years × 500 tickers via
   yfinance; `configs.yml: data_extract.years_history = 15`). This also writes `dividends`.
2. **Work on a stage that does not need it** — fundamentals derivation, text/sentiment, the extract
   layer, anything reading `fundamentals_facts` / `earnings_call_*` directly.
3. **Accept the skips** — real-data fixtures `pytest.skip` cleanly rather than error, so the test
   suite still runs; it just exercises far less. Do not read a green suite as full coverage here.

Do **not** silently substitute a synthetic price frame in a feature or economic test — see
[testing.md](testing.md).

## Stale bind mount

`docker-compose.yml` is **correct** (`./sql → /docker-entrypoint-initdb.d`). The drift is in the
**running container**: `pea_db` was created before the move to the repo root and still binds the
now-nonexistent `./stock_pick_strat/sql` — confirm with
`MSYS_NO_PATHCONV=1 docker inspect pea_db --format '{{range .Mounts}}{{.Source}}{{println}}{{end}}'`.
Harmless while the volume has data (initdb scripts run only on an empty data dir), but recreate the
container from the current compose file **before** you ever rebuild the volume, or the schema will
not be applied.
