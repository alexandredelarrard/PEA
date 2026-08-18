# Data schema — the table registry

Scope: what every table **is** (PK, grain, date column, freshness, projection), derived from
[src/data_store/schema.py](../src/data_store/schema.py). For what is actually **populated right
now**, see [database.md](database.md). For the access rules, see
[data_conventions.md](data_conventions.md).

## The registry is the single source of truth

`src/data_store/schema.py` declares each table **exactly once** as a frozen `Table` dataclass.
48 tables: 40 `managed`, 8 `cube_part_*` (`managed=False`).

```python
from src.data_store.schema import Tables
store.load(Tables.prices, columns=["date", "ticker", "close"], since="2024-01-01")
```

**Never write a table name as a string literal, and never add a `*_TABLE` constant.**
`f"{Tables.prices}"` yields `"prices"` (`Table.__str__`). `resolve()` does accept a bare name — a
few call sites still pass one (e.g. `step_train.py` uses `"predictions"` / `"cube_signal"`) — but
new code must not.

`Table` fields you will actually use:

| Field | Meaning |
|---|---|
| `name`, `pk` | table name; upsert / dedup key |
| `kind` | `reference` / `extract` / `aggregate` / `part` — **DDL grouping only**, distinct from `parts.py`'s `PartKind` |
| `date_col` | the incremental time column: what `since=`/`until=` filter on and `max_date()` reads. `None` for grainless tables |
| `ticker_col` | `None` when there is no per-ticker grain (macro, trend returns) |
| `date_type_cols` | forced to SQL `DATE` (they arrive as strings; `TIMESTAMP` would be a lie) |
| `vector_col` / `vector_prefix` | collapse an `e0..eN` family into one `float8[]` column |
| `managed` | `False` → the owning step creates the table from its own frame's dtypes, it is **excluded from `sql/schema.sql`**, and `replace()` DROPs rather than DELETEs |
| `freshness` / `freshness_date_col` | cadence key into `constants.DATA_FRESHNESS_MAX_AGE_DAYS`; the second is for tables that *publish* on a different clock than the period they describe |
| `read_columns` / `optional_columns` | the `project=True` projection, and which of those may be silently absent |

Derived views are comprehensions, never hand-lists: `ALL`, `BY_NAME`, `MANAGED`, `PARTS`,
`by_kind()`, `freshness_tables()`, `projection()` / `projection_report()`.

**Adding a table:** add one `Table` to `Tables`, then regenerate DDL with
`python -m scripts.generate_schema_sql`. That is the whole change — there is no second list.

## Reference

| Table | PK | Notes |
|---|---|---|
| `sp500_tickers` | `ticker` | THE universe. `name, sector, industry_group, sub_industry, cik`. Also the only ticker→CIK source (the old `cik_mapping` table was dropped). Resolved via [src/utils/universe.py](../src/utils/universe.py)`::load_universe_tickers`, which drops `constants.INSUFFICIENT_HISTORY_TICKERS`. Swap universe by replacing rows only — no step code changes. |

`other_tickers` (`SPY`, `CL=F`, `GC=F`, `USDEUR=X`, `^VIX`) go into `prices` via a plain
`fetch_price_history` call over that list (they are ordinary OHLCV rows) and are **never**
added to the equity universe — nor passed to `fetch_dividends`.

## Extract — prices & market

| Table | PK | date_col | Fresh | Notes |
|---|---|---|---|---|
| `prices` | `ticker, date` | `date` | daily | OHLCV. Read raw **only** by `StepCubePrices`. |
| `dividends` | `ticker, date` | `date` | — | ex-div cash amount. Its **own** fetcher (`fetch_dividends.py`) with its own resume window — ex-dates are quarterly where bars are daily — though it reuses the same yfinance `actions=True` response shape |
| `short_interest` | `ticker, date` | `date` | daily | FINRA RegSHO. Resumes on the table's **global** max date (one day-file covers the whole market, so a per-ticker frontier would only re-fetch days already held). Projection lists `short_interest`/`avg_daily_volume` as **optional** — the live table has neither, and demanding them killed the read instead of degrading it |
| `fails_to_deliver` | `ticker, date` | `date` | biweekly | SEC CNS fails. Separate from `short_interest` so its semi-monthly ~2-month-lagged files don't poison that table's global-max incremental |
| `macro` | `date` | `date` | daily | FRED: 3M/2Y/10Y/30Y yields, 10y-2y & 10y-3m spreads, VIX, BAA spread, 10y breakeven. `ticker_col=None` |
| `macro_asset_prices` | `date` | `date` | daily | long-history (~1995) allocation legs: `equity_tr`, `gold`, `energy`, `bond_10y_tr`, `cash_rate`, `fx_usdeur`, `yield_10y`, `vix`. `ticker_col=None` |
| `cusip_ticker_map` | `cusip` | — | — | CUSIP→ticker via OpenFIGI (+ `constants.CUSIP_TICKER_OVERRIDES`) |

## Extract — fundamentals

| Table | PK | date_col | Fresh | Notes |
|---|---|---|---|---|
| `fundamentals_facts` | `ticker, accession_number, field, fiscal_year, fiscal_period, duration_type` | `filing_date` | quarterly | **the raw backbone**: accession-grain, amendment-aware per-filing XBRL. ORIGINAL and AMENDED (`10-K/A`) rows coexist and are never overwritten, so "what was known as of D" is answerable without leaking an amendment before its own filing date |
| `fundamentals_history` | `ticker, as_of` | `as_of` | quarterly | **derived** from `fundamentals_facts` by `fundamentals_derive.rebuild_fundamentals_history`. 239 columns: statements, ratios, valuation, credit, plus `employees` (parsed from 10-K body text — no XBRL concept exists; the old `employees_history` table is retired) |
| `earnings_surprises` | `ticker, earnings_date` | `earnings_date` | quarterly | consensus vs actual EPS + surprise %. Also the source of the backtestable historical forward P/E (consensus EPS ÷ price) |
| `pension_facts` | `cik, tag, ddate, qtrs` | `ddate` (fresh on `filed`) | quarterly | SEC Financial Statement Data Sets, curated pension tags |
| `notes_num` | `adsh, tag, ddate, qtrs` | `ddate` (fresh on `filed`) | biweekly | SEC Notes sets, undimensioned numeric footnote facts |
| `notes_text` | `adsh, tag, ddate, qtrs` | `ddate` (fresh on `filed`) | biweekly | the narrative TextBlocks; `value` is the text |

## Extract — ownership & institutional

| Table | PK | date_col | Fresh | Notes |
|---|---|---|---|---|
| `sec13f_hr` | `cik, period, ticker, cusip` | `period` | quarterly | **21.7M rows, 6.1 GB — THE reason projections exist.** Long-only quarterly snapshot, 45-day filing lag. Split into stock / call / put / debt legs. Institutional "moves" come from QoQ **share** deltas, not value deltas |
| `insider_transactions` | `accession_number, security_type, transaction_sk` | `transaction_date` (fresh on `filing_date`) | quarterly | Forms 3/4/5 bulk sets; roles, transaction codes, shares, price |
| `sec_13d` | `ticker, accession_number, rp_seq` | `filing_date` | — | one row **per reporting person** per filing (`rp_seq`, not CIK — an RP without a CIK is common). Numeric ownership fields are **NULL, not 0**, when `has_structured_data` is false |
| `sec_13d_transactions` | `ticker, accession_number, trade_seq` | `filing_date` | — | Item 5(c) 60-day trade log; an independent grain from `sec_13d` |

## Extract — governance (DEF 14A) & events

Two complementary paths to the same filings:

- `def14a_llm` (`ticker, accession_number`, `as_of`, yearly) — OpenAI structured extraction.
  45 columns: board composition, CEO age/tenure/pay, ownership, say-on-pay, governance flags.
- `def14a_edgar` (`ticker, accession_number`, `filing_date`) + 4 child tables — deterministic
  edgartools `ProxyStatement`, zero LLM cost:
  `_executive_comp` (`+name, year` — Summary Comp Table, ~3 years/filing),
  `_director_comp` (`+name`), `_ownership` (`+holder_name, holder_type`),
  `_votes` (`+proposal_number` — the **board's recommendation**, not the vote outcome).

> **Only `def14a_edgar`'s XBRL-backed block is trustworthy unconditionally.** edgartools' proxy
> HTML parser emits values that are silently *wrong* rather than absent, so every row passes through
> [def14a_validate.py](../src/data_extract/utils/structure/def14a_validate.py) first: it rescales
> unit-broken fee blocks, NULLs the fabricated `0.5` placeholder for the "*" (= "<1%") footnote,
> undoes Total-column duplication, strips glued titles/footnotes/addresses off name primary keys,
> drops subtotal pseudo-rows and completes the CEO pay-ratio identity. **Rule: never fabricate** —
> a value is written only when deterministically recoverable, else NaN.

| Table | PK | date_col | Notes |
|---|---|---|---|
| `sec_8k` | `ticker, accession_number` | `filing_date` | `items` = raw comma-separated item codes (structured, ~100% fill); `has_earnings`/`has_press_release` are best-effort → **null, not False**, when the typed parse fails |
| `filing_risk_text` | `ticker, accession_number, section` | `filed` | 10-K Item 1A + Item 7 / 10-Q Item 2 raw text. `constants.FILING_TEXT_MIN_CHARS = 1500` rejects TOC stubs |

## Extract — behavioral / text / embeddings

| Table | PK | date_col | Fresh | Notes |
|---|---|---|---|---|
| `google_trends` | `ticker, date` | `date` | weekly | search interest |
| `wiki_pageviews` | `ticker, date` | `date` | daily | pageviews |
| `earnings_call_sections` | `ticker, quarter, tag` | `as_of` | quarterly | transcript prose. **Deliberately not projected**: `text` IS the payload the scoring pass needs |
| `earnings_call_sentiment` | `ticker, quarter, tag` | `as_of` | — | the expensive, call-intrinsic scores (FinBERT-tone probs, `n_words`, LM `uncertainty_ratio`) cached so the GPU pass runs once. Cross-call KPIs are derived cheaply at cube-build time |
| `earning_calls_embedding` | `ticker, quarter, seq` | `None` | — | **8.6 GB** — one row per **speaker turn** with its own `float8[] embedding`, `person`, `tag`, `exchange_idx` (links a question to its answer turns), `answer_idx`. Projection omits `text` |
| `notes_embedding` | `ticker, adsh, tag` | `None` | — | mean-pooled vector per footnote TextBlock. **Populated by extraction but consumed by nothing** — the narrative-drift builder was never wired into a panel |
| `ticker_descriptions` | `ticker` | — | — | business descriptions (input to embeddings) |
| `ticker_embeddings` | `ticker` | — | — | `float8[]` business-similarity vector; feeds `StepDeducePeers` |

## Aggregate — pipeline outputs

| Table | PK | Notes |
|---|---|---|
| `cube` | `ticker, date, target_horizon` | THE feature table. ~570 columns, LONG by horizon. **Reference size ~26 GB / ~5.7M rows** across horizons 30/60/90. Never read unprojected |
| `predictions` | `ticker, date` | backtest-time scores; fully replaced each training run |
| `cube_signal` | `ticker, date` | the blended cross-horizon signal |
| `predictions_latest` | `date, ticker, horizon, model` | **live** predictions, LONG form. Each row carries its own `predicts_for` (as-of + horizon business days), because the h30 and h90 predictions made on one day are about different future dates. `model` ∈ {member name, `ensemble` = that horizon's member average, `blended` = the IR-weighted blend across horizons}. `predicted_at` (when the run produced the row) is deliberately distinct from `date` (the as-of date of the features) |
| `trend_asset_returns` | `date` | multi-asset trend sleeve daily NET returns; `ticker_col=None` |
| `strategy` | `trading_day, sleeve, ticker` | THE trading ledger. **Upserted, not replaced**, so a BUY row written weeks ago gains its `price_sold`/`pnl` on the day its position closes |

## Parts — private plumbing between cube sub-steps

`cube_part_prices`, `_market`, `_targets`, `_betas`, `_fundamentals`, `_momentum`, `_text`,
`_extras`. All `managed=False`: rebuilt wholesale by their owning step, each carrying DDL inferred
from the frame it writes, excluded from `sql/schema.sql`.

PK is `(date, ticker)` **except `cube_part_targets`, which is `(date, ticker, target_horizon)`** —
`_labels_to_long` stamps a horizon per label and concatenates. Declaring the narrower key would let
an upsert path collapse the horizons into one row, and such a path exists (`copy_load` falls back
to an upsert for frames with list-valued cells).

Build orchestration (CLI command, warm-up days, binding look-backs) deliberately lives in
[parts.py](../src/data_aggregate/utils/common/parts.py), not here — that is aggregation policy, not
schema.

## Freshness cadences

17 tables are freshness-checked (`Table.freshness`), gated by
`constants.DATA_FRESHNESS_MAX_AGE_DAYS`:

```
daily 4d · weekly 10d · biweekly 20d · monthly 45d · quarterly 140d · yearly 460d
```

Watched: `prices`, `short_interest`, `macro`, `macro_asset_prices`, `wiki_pageviews` (daily) ·
`google_trends` (weekly) · `fails_to_deliver`, `notes_num`, `notes_text` (biweekly) ·
`fundamentals_history`, `fundamentals_facts`, `earnings_surprises`, `pension_facts`, `sec13f_hr`,
`insider_transactions`, `earnings_call_sections` (quarterly) · `def14a_llm` (yearly).

Four of these measure a **different column** than their period grain (`freshness_date_col`):
`fundamentals_facts`→`filing_date`, `pension_facts`/`notes_num`/`notes_text`→`filed`,
`insider_transactions`→`filing_date`. Freshness must watch *when the fact was filed*, not the
quarter it covers.

Run the gate with `python -m src data_extract check-freshness` (JSON on the last stdout line,
non-zero exit when stale).

## `sql/schema.sql` gap

`ddl.py` generates DDL from the registry **plus live reflection**, and carries over any table it
cannot reflect. Consequence: a `managed` table that has never existed in the live DB has **no DDL**.
Currently missing from `sql/schema.sql`:

```
cube_signal · notes_embedding · predictions · predictions_latest
sec_13d_transactions · strategy · ticker_descriptions · trend_asset_returns
```

This is harmless in practice — `store.save`/`replace` call `ensure_table`, which creates the table
from the frame's dtypes on first write — but it means `schema.sql` is not a complete picture. It
also means `sql/schema.sql` (applied by initdb on an **empty** volume only) will not pre-create them.
