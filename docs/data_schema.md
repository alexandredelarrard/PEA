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

The market / commodity / energy series (`SPY`, `^VIX`, `CL=F`, `GC=F`, `XLE`) do **not** go into
`prices`. They are close-only rows in `prices_macro`, stored under SERIES names (`equity_tr`,
`vix`, `oil`, `gold`, `energy`) by `fetch_macro`, alongside the FRED legs — including FX
(`fx_usdeur` ← `DEXUSEU`, which is quoted USD-per-EUR and reaches back to 1999-01, where
Yahoo's `USDEUR=X` is the reciprocal and only starts 2003-12). So `prices` is the equity
universe and nothing else.

That separation is enforced at the SOURCE, and it is load-bearing. When those tickers were
extra OHLCV rows inside `prices`, every consumer needed a firewall against them — a whole
second part table (`cube_part_market`), three `drop(columns=[market])` guards, a `^`-prefix
filter, a zero-volume trim exemption — because consumers pivot to wide and rank
cross-sectionally, where one stray `SPY` column silently shifts every percentile. All of that
is gone. See [tests/data_extract/test_macro_prices_separation.py](../tests/data_extract/test_macro_prices_separation.py).

## Extract — prices & market

| Table | PK | date_col | Fresh | Notes |
|---|---|---|---|---|
| `prices` | `ticker, date` | `date` | daily | OHLCV. Read raw **only** by `StepCubePrices`. |
| `dividends` | `ticker, date` | `date` | — | ex-div cash amount. Its **own** fetcher (`fetch_dividends.py`) with its own resume window — ex-dates are quarterly where bars are daily — though it reuses the same yfinance `actions=True` response shape |
| `short_interest` | `ticker, date` | `date` | daily | FINRA RegSHO. Resumes on the table's **global** max date (one day-file covers the whole market, so a per-ticker frontier would only re-fetch days already held). Projection lists `short_interest`/`avg_daily_volume` as **optional** — the live table has neither, and demanding them killed the read instead of degrading it |
| `sec_fails_to_deliver` | `ticker, date` | `date` | biweekly | SEC CNS fails. Separate from `short_interest` so its semi-monthly ~2-month-lagged files don't poison that table's global-max incremental |
| `macro` | `date` | `date` | daily | FRED: 3M/2Y/10Y/30Y yields, 10y-2y & 10y-3m spreads, VIX, BAA spread, 10y breakeven. `ticker_col=None` |
| `prices_macro` | `ticker`, `date` | `date` | daily | LONG: one `close` per (series, date). 15 series — yfinance closes (`equity_tr`, `vix`, `oil`, `gold`, `energy`), FRED levels (`yield_2y/10y/30y`, `cash_rate`, `baa_credit_spread`, `breakeven_10y`, `fx_usdeur`) and derived (`yield_curve_10y2y`, `yield_curve_10y3m`, `bond_10y_tr`). Replaced the wide `macro` + `macro_asset_prices` |
| `cusip_ticker_map` | `cusip` | — | — | CUSIP→ticker via OpenFIGI (+ `constants.CUSIP_TICKER_OVERRIDES`) |

## Extract — fundamentals

| Table | PK | date_col | Fresh | Notes |
|---|---|---|---|---|
| `fundamentals_facts` | `ticker, accession_number, field, fiscal_year, fiscal_period, duration_type` | `filing_date` | quarterly | **the raw backbone**: accession-grain, amendment-aware per-filing XBRL. ORIGINAL and AMENDED (`10-K/A`) rows coexist and are never overwritten, so "what was known as of D" is answerable without leaking an amendment before its own filing date |
| `fundamentals_history_sec` | `ticker, as_of` | `as_of` | quarterly | **derived** from `fundamentals_facts` by `fundamentals/build_history.py`, on the **publication-event grain**: `as_of` is a FILING DATE and a row exists for every date on which ≥1 extracted value became newly public. Each row is a COMPLETE snapshot (every column at its latest-known value), so a plain `asof` merge needs no reconstruction. **Append-only, enforced** — the build recomputes every event, diffs against what is stored and REFUSES to overwrite a published row unless `--rebuild-history` is passed. Exactly **69** columns, enumerated by `Catalogue.history_columns`: 4 keys + 52 catalogue fields + 8 derived + `regime` + 4 provenance. The 60 value columns are in **statement order** (`HISTORY_STATEMENT_ORDER`) -- revenue, cost, operating result, bottom line, cash flow, assets, liabilities, equity, share counts -- not the tier-then-name order they are resolved in. `fiscal_quarter` labels which quarter of the ISSUER's own year `fiscal_end` closes, on EVERY row including the TTM and instant ones: a filer's Q4 is not its Q1, and the calendar month of `fiscal_end` does not say which is which for a 52/53-week or non-December filer. `sector`/`industry_group` (join `sp500_tickers`), `revenueGrowth`/`earningsGrowth` (computed by `pit.py` at cube time on a 365-day offset) and `employees` (own table) are deliberately NOT here. ⚠ **This is the SEC REPLAY, not the consumer table** — since the Sharadar integration, consumers read `fundamentals_history` and this feeds 15 of its columns. It remains reason-coded (`fundamentals_reason_codes`), it is **the only fundamentals table `src/validate/` looks at**, and it is the **sole owner of `is_amendment` / `amended_fiscal_end` / `amended_fields`**, which are SEC reconciliation columns and do not cross into the merged table |
| `fundamentals_history` | `ticker, as_of` | `as_of` | quarterly | **THE MERGED TABLE, and the one consumers should read.** Built by `fundamentals_sharadar/merge_history.py`. **Field-block precedence**: Sharadar owns a declared block of columns for ALL history, `fundamentals_history_sec` owns 15, and **no column ever switches source mid-series**. `as_of` is Sharadar's `date` (a FILING date), `fiscal_end` its `reportperiod`; one row per `(ticker, as_of)` after a same-date collapse that keeps the GREATEST `fiscal_end`. Every SEC-sourced column NAME ends in **`_sec`** (`goodwill_sec`, `regime_sec`, …) — that suffix is what replaces a `source` column, because the two producers have DIFFERENT COVERAGE and a bare NULL would not say which one left it. The SEC block is joined **backward as-of**, never exact (the two filing dates agreed on 279/280 measured, so an exact join would drop the whole block on a one-day disagreement) and never forward, capped at `SHARADAR_SEC_ASOF_TOLERANCE_DAYS = 370`. `regime_sec` is the ONLY non-float column. Deliberately does NOT carry `is_amendment` / `amended_fiscal_end` / `amended_fields` — those are pure SEC reconciliation columns, they stay on `fundamentals_history_sec`, and Sharadar ships no amendment events so they would be permanently null here |
| `fundamentals_sharadar` | `ticker, dimension, date, reportperiod` | `date` | quarterly | **the raw vendor layer**, all 112 SF1 columns exactly as delivered, on the three AS-REPORTED dimensions (`ARQ`/`ARY`/`ART`) — `MR*` mutates in place and is never stored. The widest extract table in the schema, so never read it unprojected. `date` is the FILING date (the Direct channel's name; Nasdaq Data Link calls it `datekey`), `reportperiod` the period end, `calendardate` Sharadar's normalisation to the nearest quarter-end. Column types are 105 `double precision` + 4 `date` + 3 `text` (`ticker`, `dimension`, `fiscalperiod` only) — `client.cast_value_columns` forces that BEFORE the first write, because `ensure_table` infers types from the first frame and an all-None column would become TEXT for every later ticker |
| `sharadar_tickers` | `table, permaticker, ticker` | — | on demand | the entity dimension: `permaticker`, `currency`, `category`, `isdelisted`, `lastquarter`. **Full refresh**, not incremental: there is no date column to resume on and `isdelisted`/`lastquarter` MUTATE. Read by the SF1 fetch to assert USD (D20) — the fetch REFUSES to run if this table is empty |
| `sharadar_actions` | `date, ticker, action, contraticker` | `date` | daily | corporate actions: dividends, splits, spinoffs, acquisitions, relations. **Market-wide** — every ticker Sharadar covers, so both consumers filter on `ticker` AND `action`. ⚠ `contraticker` is the literal string `"N/A"`, not NULL, and it is in the PK: read with `keep_default_na=False`. The planned PK `(date, ticker, name, action)` had 8 collisions in 1,927 rows (GS/JPM preferred series differing only in `contraticker`) |
| `sharadar_sp500` | `date, ticker, action` | `date` | on demand | index membership events from 1992. **Ingested only** — `src/utils/universe.py` still resolves the universe from `sp500_tickers`; the survivorship-bias fix that would consume this is a separate task (D27) |
| `fundamentals_reason_codes` | `ticker, as_of, field, dc_code` | `as_of` | quarterly | WHY a `fundamentals_history_sec` cell is null, or why its value is off-basis. **Dense** — one row per null-or-qualified cell at every publication event — so the zero-unexplained-nulls gate is a one-line `LEFT JOIN`. The `dc_code` vocabulary is closed and declared once, in `fundamentals/reason_codes.py`; the build asserts every code against it. Two PAYLOAD columns, each NULL for every code but the one that owns it: `combined_into` names the destination field, and `rejected_value` carries the number a `failed_hard_guard` refused -- a nulled DERIVED value (a TTM, a `derived_identity` total) has no fact row anywhere, so without it the refused number is lost and "did that guard null something CORRECT?" stops being a query |
| `fundamentals_employees` | `ticker, as_of` | `as_of` | quarterly | Headcount, parsed from **10-K body text** (no GAAP concept exists). Its own table because the source is prose: in the wide table one failed regex would fail the whole snapshot. Annual, so consumers forward-fill (`build_history.carry_latest_known`). Produced by `fundamentals_employees.py` inside the facts walk, so no separate fetch |
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
- `sec_def14a` (`ticker, accession_number`, `filing_date`) + 4 child tables — deterministic
  edgartools `ProxyStatement`, zero LLM cost:
  `_executive_comp` (`+name, year` — Summary Comp Table, ~3 years/filing),
  `_director_comp` (`+name`), `_ownership` (`+holder_name, holder_type`),
  `_votes` (`+proposal_number` — the **board's recommendation**, not the vote outcome).

> **Only `sec_def14a`'s XBRL-backed block is trustworthy unconditionally.** edgartools' proxy
> HTML parser emits values that are silently *wrong* rather than absent, so every row passes through
> [def14a_validate.py](../src/data_extract/utils/structure/def14a_validate.py) first: it rescales
> unit-broken fee blocks, NULLs the fabricated `0.5` placeholder for the "*" (= "<1%") footnote,
> undoes Total-column duplication, strips glued titles/footnotes/addresses off name primary keys,
> drops subtotal pseudo-rows and completes the CEO pay-ratio identity. **Rule: never fabricate** —
> a value is written only when deterministically recoverable, else NaN.

| Table | PK | date_col | Notes |
|---|---|---|---|
| `sec_8k` | `ticker, accession_number, item` | `filing_date` | One row **per item code** — an 8-K reports 1..n items and ~75% report more than one. `item_tag` maps the curated high-signal codes; `has_earnings`/`has_press_release` are best-effort → **NaN, not False**, when the typed parse fails |
| `sec_filing_text` | `ticker, accession_number, section` | `filed` | 10-K Item 1A + Item 7 / 10-Q Item 2 raw text. `fetch_filing_text.FILING_TEXT_MIN_CHARS = 1500` rejects TOC stubs |

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
| `fundamentals_check` | `run_date, run_id, check_name, ticker, field, period_key` | the fundamentals validator's **append-only** finding ledger, written by `src/validate/`. **Nothing here gates** — the nightly fundamentals build runs to completion whatever lands in it. `run_date` is IN the key so a re-run appends; what survives across runs is `finding_id`, a hash of `(check_name, ticker, field, period_key)` and deliberately **not** of `run_date`/`severity`/`observed`, so a threshold retune (347 severities moved in one change) cannot re-key a finding and make a delta read as a mass close-and-reopen. `run_id` is in the key because two runs of DIFFERENT scope on one day otherwise collide on every shared ticker -- a `-t MCD` run once lost 269 of its 270 rows to a roster run an hour later. `cluster_id` hashes `(ticker, field)` ALONE: it is the DEFECT, of which each row is one witness, and it is not unique. **Nothing is ever subtracted on the way in**, so a row-count drop between two comparable runs has exactly one cause. `period_key` is TEXT and **polymorphic by grain**: `as_of` \| `period_end` \| `''` (ticker-level) \| `start..end` (series) — one key column, because a PK cannot hold a NULL and a sentinel date would be a lie. The payload is denormalised on purpose: a Tier-2/3 finding on a DERIVED value has no fact row to join back to |
| `fundamentals_check_run` | `run_id, check_name` | WHAT a validation run looked at and what each check did with it: scope (roster, tickers, fields, tiers), `examined` / `queued` / `info`, the declared `ceiling`, and the stored `abstained` / `over_ceiling` flags that drive the report's check-health gate. `run_id` hashes (date, tickers, fields, tiers); `scope_hash` is the same **without** the date and is the COMPARABILITY key -- two runs may only be differenced when it matches, or a 54-ticker baseline vs a one-ticker re-validation would report ~11,800 findings "closed". Scope columns repeat per check row, denormalised deliberately: ~35 rows per run, one read |
| `fundamentals_check_status` | `cluster_id, check_name` | a human's TOLERANCE for a `(ticker, field)` defect. `status` carries **`wontfix` and nothing else** -- `open`, `settled` and `reopened` are DERIVED from the ledger, because a stored `settled` that says so while the check still fires is a suppression list. `check_name` is in the key and `''` means the WHOLE cluster: at cluster grain, tolerating MCD `capex`'s 2 benign `peer_ratio` findings would also silence the eight other checks live on the same defect. `note` must carry a QUANTIFIED cost (the CLI refuses a note with no numeral). `findings_at_decision` records how big THAT ENTRY'S population was when it was assessed, and it **reopens automatically** if it grows past that. Waiving every check still does NOT settle a cluster -- that needs a `fundamentals_check_fix` row |
| `fundamentals_check_fix` | `cluster_id, run_id_after` | an **INTERVENTION**: what was done to a cluster, at which layer, and what it measurably closed. Append-only and a different KIND of thing from the row above -- a fix is an EVENT, a waiver is a STATE -- so a cluster fixed twice has two rows. **No renderer may filter findings using this table**; a test pins `fundamentals_check` row counts as identical with and without it. `run_id_after` is the run that PROVED the fix and both runs must share the pinned `scope_hash`, or the before/after counts are not a comparison. `queued_*` counts exclude `info`, and a row with `queued_after >= queued_before` is recordable but can never settle -- permissive to record, strict to settle. `layer` is closed (`constants.FIX_LAYERS`: check\|catalogue\|extraction\|rows) and describes what the EDIT DID, not which file it lives in; `evidence` is JSON with per-layer required keys (`constants.FIX_EVIDENCE_KEYS`), never prose; `commit_sha` and `test_path` are both verified on write |

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

Each table declares its expected refresh cadence (`Table.freshness`), keyed into
`constants.DATA_FRESHNESS_MAX_AGE_DAYS`:

```
daily 4d · weekly 10d · biweekly 20d · monthly 45d · quarterly 140d · yearly 460d
```

Watched: `prices`, `short_interest`, `prices_macro`, `wiki_pageviews` (daily) ·
`google_trends` (weekly) · `sec_fails_to_deliver`, `notes_num`, `notes_text` (biweekly) ·
`fundamentals_history_sec`, `fundamentals_facts`, `earnings_surprises`, `pension_facts`, `sec13f_hr`,
`insider_transactions`, `earnings_call_sections` (quarterly) · `def14a_llm` (yearly).

Four of these measure a **different column** than their period grain (`freshness_date_col`):
`fundamentals_facts`→`filing_date`, `pension_facts`/`notes_num`/`notes_text`→`filed`,
`insider_transactions`→`filing_date`. Freshness must watch *when the fact was filed*, not the
quarter it covers.

This is **declarative metadata only** — it records each source's expected cadence and is exposed
by `schema.freshness_tables()`. The automated staleness gate that consumed it was removed, so
nothing reads it today; check staleness against [database.md](database.md) by hand, or wire a new
consumer to `freshness_tables()`.

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
