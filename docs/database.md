# Live database state

Scope: what is **actually in the local Postgres right now**. For what each table *means*, see
[data_schema.md](data_schema.md). For how to connect, see [runbook.md](runbook.md).

> **Snapshot re-measured 2026-08-26.** Rows/size/coverage in the tables below were taken on
> **2026-08-17** unless a row says otherwise; the global counts, the missing-table list and the
> whole `fundamentals_*` / `sharadar_*` block were re-queried on 08-26. Re-verify before relying
> on a number — `MSYS_NO_PATHCONV=1 docker exec pea_db psql -U alexandre -d pea -c "…"`.

> ⚠ **THE SHARADAR TABLES BELOW ARE PRE-UPGRADE AND KNOWINGLY STALE.** Every Sharadar row count
> here was measured while the key was on the **free DJIA tier** — 30 tickers, history from 2021.
> The subscription was upgraded on **2026-08-26** and the key now covers the **whole SF1
> universe** (measured: no 403 on any ticker, 5,780 distinct tickers in 2024 alone, history back
> to filing date **1993-12-22**). **A full re-extraction is pending**; until it runs, treat the
> Sharadar and `fundamentals_history` numbers below as a floor, not as coverage. Re-run
> `python -m src data_extract fundamentals-sharadar -F` and re-measure this section.

> **What changed on 2026-08-26:**
> - `fundamentals_history` was **renamed to `fundamentals_history_sec`** (the SEC producer's
>   table). The bare name `fundamentals_history` is now the **merged Sharadar+SEC table**, and it
>   is **built and populated** — `fundamentals_sharadar` + `fundamentals_history_sec` via
>   `merge_history.py`.
> - **Four Sharadar tables added and populated.**
> - The Sharadar subscription was **upgraded from the free DJIA tier to the full universe** (see
>   the warning above).
> - **`fundamentals_facts_legacy` (5.2 GB) and `fundamentals_history_legacy` (36 MB) were
>   DROPPED.** They had no registry entry, no reader, no view and no foreign key. That is where
>   24 GB -> 19 GB came from. ⚠ Their loss is not free: `fundamentals_facts_legacy` was the only
>   445-ticker fundamentals substrate and the declared provenance of every measured rate in
>   `configs/fundamentals/fundamentals_exceptions.json`, which can no longer be re-derived.

Container `pea_db` (postgres **16.14**), database `pea`, owner role **`alexandre`**, volume
`stock_pick_strat_pgdata`. **19 GB**, **41 tables**, all in schema `public` (measured 2026-08-26).

## Read this first: the registry declares 57 tables, the DB has 41

**Nothing live is unregistered** — the 41 live tables are all in `Tables` (verified 2026-08-26), so there is no orphan to read by mistake.

**Missing entirely** — every read of these raises `TableMissingError`:

| Missing | Consequence |
|---|---|
| `cube`, `cube_part_*` (all 7) | no features, no training, no prediction |
| `cube_signal`, `predictions`, `predictions_latest` | no model output |
| `strategy`, `trend_asset_returns` | no ledger, no trend sleeve |
| ~~`fundamentals_history`~~ | **No longer missing** — the merged Sharadar+SEC table is built and populated (51,255 rows / 489 tickers as of 2026-08-31). |
| `notes_embedding`, `ticker_descriptions` | `notes_embedding` has no downstream reader anyway |

**No longer missing** (they were, in the 2026-08-17 snapshot): **`prices`** is populated —
1,777,827 rows, 500 tickers, 2011-08-19 → 2026-08-19 — so the "the whole cube build is blocked"
warning that used to head this list no longer applies. `sec_13d` and `sec_13d_transactions` also
exist now.

**Present but not in the registry**: **none.** All 41 live tables are declared in `Tables`.
`fundamentals_facts_legacy` and `fundamentals_history_legacy` — the last two orphans — were
dropped on 2026-08-26.

## Populated tables

Ordered by size. `tickers` = distinct non-null tickers.

| Table | Rows | Size | Cols | Tickers | Date column | Coverage |
|---|---|---|---|---|---|---|
| `earning_calls_embedding` | 1,375,495 | **8.6 GB** | 13 | 494 | `as_of` | 2005-10-30 → 2026-07-24 |
| `prices` | 1,777,827 | — | — | **500** | `date` | 2011-08-19 → 2026-08-19 *(08-26)* |
| `sec13f_hr` | 21,659,435 | **6.1 GB** | 15 | 497 | `period` | 1987-03-31 → 2026-03-31 |
| `earnings_call_sections` | 109,899 | 1.5 GB | 6 | 494 | `as_of` | 2005-10-13 → 2026-07-24 |
| `sec_filing_text` | 34,127 | 1.2 GB | 9 | 498 | `filed` | 2011-07-27 → 2026-08-03 |
| `insider_transactions` | **1,942,945** | 929 MB | **38** | 498 | `transaction_date` | 1990-05-07 → **2026-03-31** *(09-08)* |
| `insider_footnotes` | **1,860,827** | 713 MB | 3 | — | — | no date column; joins on `accession_number` *(09-08)* |
| `notes_text` | 96,576 | 411 MB | 15 | — | `ddate` | 2006-12-31 → 2026-05-31 |
| `sec_8k` | 95,789 | 137 MB | 14 | 486 | `filing_date` | 2011-08-04 → 2026-08-03 |
| `sec13f_manager_holdings` | **342,501** | 85 MB | 16 | — | `period` | 2011-09-30 → 2026-06-30 *(09-08)* |
| `sec_13g` | **29,176** | 12 MB | 27 | **498** | `filing_date` | 2011-09-09 → 2026-09-08 *(09-09)* |
| `wiki_pageviews` | 1,699,202 | 136 MB | 3 | 500 | `date` | **2016-07-16** → 2026-07-23 |
| `fails_to_deliver` | 993,775 | 98 MB | 5 | 499 | `date` | 2010-01-04 → **2026-07-14** |
| `short_interest` | 963,115 | 84 MB | 4 | 502 | `date` | **2017-12-29** → 2026-07-31 |
| `fundamentals_facts` | 316,245 | 131 MB | 26 | **54** | `filing_date` | 2009-07-31 → 2026-08-10 *(08-26)* |
| `sharadar_tickers` | 17,827 | 15 MB | 28 | 17,827 | — | Sharadar entity dimension *(08-26)* |
| `fundamentals_reason_codes` | 78,239 | 13 MB | 6 | **54** | `as_of` | 2009-07-31 → 2026-08-10 *(08-26)* |
| `fundamentals_history_sec` | **3,258** | 1.8 MB | **69** | **54** | `as_of` | 2009-07-31 → 2026-08-10 *(08-26)* |
| `fundamentals_sharadar` | **116,824** | 130 MB | **112** | **489** | `date` | 1995-09-01 → 2026-08-28 *(08-31)* |
| `fundamentals_history` | **51,255** | 31 MB | **91** | **489** | `as_of` | 1995-09-01 → 2026-08-28 *(08-31)* |
| `sharadar_sp500` | 3,306 | 432 kB | 7 | 30 | `date` | 1992-01-02 → 2026-08-25 *(08-26)* |
| `sharadar_actions` | 594 | 128 kB | 7 | 30 | `date` | 2021-08-27 → 2026-08-25 *(08-26)* |
| `fundamentals_employees` | 754 | 112 kB | 3 | **54** | `as_of` | 2002-03-20 → 2026-07-29 *(08-26)* |
| `fundamentals_check` | 23,656 | 31 MB | 23 | **54** | `run_date` | two runs: 2026-08-24 → 2026-08-25 |
| `fundamentals_check_run` | 70 | 136 kB | 17 | — | `run_date` | two runs: 2026-08-24 → 2026-08-25 |
| `fundamentals_check_status` | 2 | 48 kB | 8 | — | `decided_at` | MCD `capex`: `peer_ratio` + `series_shape` waived |
| `fundamentals_check_fix` | 1 | 64 kB | 16 | — | `decided_at` | MCD `capex` `1c9a517eaa47`, 2026-08-25 |
| `google_trends` | 388,336 | 32 MB | 3 | 500 | `date` | 2011-07-17 → **2026-07-12** |
| `cusip_ticker_map` | 145,748 | 14 MB | 2 | 19,824 | — | — |
| `notes_num` | 40,587 | 14 MB | 14 | — | `ddate` | 2007-12-31 → **2026-04-30** |
| `def14a_llm` | **8,784** | 13 MB | **54** | 497 | `as_of` | 1995-12-19 → 2026-07-28 |
| `earnings_call_sentiment` | 55,514 | 11 MB | 10 | 494 | `as_of` | 2005-10-30 → 2026-07-24 |
| `earnings_surprises` | 43,383 | 5.3 MB | 5 | 500 | `earnings_date` | 1999-08-02 → **2026-10-29** (forward-dated: scheduled future calls) |
| `dividends` | 22,060 | 3.7 MB | 3 | **413** | `date` | 2011-07-18 → 2026-07-31 |
| `ticker_embeddings` | 502 | 2.9 MB | 2 | 502 | — | — |
| `pension_facts` | 6,244 | 2.5 MB | 13 | — | `ddate` | 2008-10-31 → **2026-02-28** |
| `prices_macro` | see §sanity | — | 3 | `ticker` | `date` | ~1995 → today (per series; `fx_usdeur` from 1999, `gold` from 2000, `breakeven_10y` from 2003) |
| `macro` | 4,175 | 1.0 MB | 10 | — | `date` | 2010-08-02 → 2026-07-31 |
| `def14a_ownership` | 6,218 | — | 8 | **14** | `as_of` | 15 smoke tickers only |
| `def14a_executive_comp` | 5,155 | — | 16 | **14** | `as_of` | 15 smoke tickers only |
| `def14a_directors` | 4,650 | — | 11 | **14** | `as_of` | 15 smoke tickers only |
| `def14a_director_comp` | 2,519 | — | 14 | **14** | `as_of` | 15 smoke tickers only |
| `sp500_tickers` | 500 | 128 kB | 6 | 500 | — | — |
| `superinvestor_roster` | 962 | — | 6 | — | `snapshot_date` | 2013-01-01 → today (14 snapshots, 50 → 83 managers) |

- **The four validator tables hold TWO comparable runs** (`3df52ae9af75` → `725bae7bf8ed`,
  54 tickers, all tiers). They are written only by `src/validate/` and gate nothing.
  `fundamentals_check` is a LEDGER: nothing is ever subtracted from it, so a row-count drop
  against a later run of the same scope has exactly one cause. Runs are comparable only when
  their `scope_hash` matches, which is why `run_id` is in the primary key — two runs of
  different scope on one day would otherwise collide on every ticker they share.
- **`fundamentals_check_status` is the only MUTABLE state**, keyed `(cluster_id, check_name)`
  with `''` meaning the whole cluster. Its 2 rows tolerate MCD `capex`'s benign residue one
  check at a time.
- **`fundamentals_check_fix` is APPEND-ONLY** and holds the one backfilled record: cluster
  `1c9a517eaa47` (MCD `capex`), layer `extraction`, commit `2fb6ef2`, findings 55 → 4 and queue
  54 → 3 between the two runs above. It exists because a fix previously had nowhere to be
  recorded — that one left only a commit sha. **Neither table filters a finding**: a waiver is
  applied when a report is RENDERED, and `fundamentals_check` still carries all 4 of that
  cluster's rows with every check firing.

## Coverage gotchas worth knowing before you build a feature

- **⚠ `sec13f_hr` has TWO BROKEN QUARTERS, and counting tickers will never show you.**
  Measured 2026-09-09 by `scripts/source_coverage_report.py`: **2023-12-31 holds 463 distinct
  managers** and **2025-06-30 holds 254**, against ~6,200 and ~7,300 in the quarters either side
  — 13x and 29x short. Ticker counts stay at 492–499 throughout, so every ticker-based coverage
  check passes; only the MANAGER axis exposes it. Both look like interrupted fetches, not SEC
  gaps, since the neighbouring quarters are healthy. **Any feature that differences consecutive
  quarters — breadth, concentration, QoQ deltas, add/reduce/exit inference — will read those
  drops as managers exiting the name.** Four quarterly transitions are affected. Re-fetch those
  quarters or exclude the transitions explicitly. This is separate from, and much sharper than,
  the known structural 2013-06-30 break (41.4 → 555.9 managers per ticker).
- **`sec_13g` has a structured-data CLIFF at 2024-12-17, and it is the same one as `sec_13d`.**
  Before that date the SEC accepted Schedule 13G as unstructured text, so the numeric block is
  effectively empty: `percent_of_class`, `aggregate_amount` and the four voting/dispositive power
  columns are **0.2% filled on 14,713 pre-mandate rows** and **100% filled on 2,903 post-mandate
  rows**. What the pre-mandate era DOES give is the event stream — *who* filed on *whom* and
  *when* (`reporting_person_name` 99.8%, `reporting_person_cik` 99.7%) — so treat it as an event
  source before 2025 and a position source after. Two columns are always NULL by design and
  documented in `schema.py`. `reporting_person_cik` drops to 80.4% post-mandate: those are
  joint-filing co-filers, and 99.2% of filings still resolve at least one CIK.
- **`sec13f_manager_holdings` is the DENOMINATOR table; `sec13f_hr` is a universe-filtered
  slice.** `sec13f_hr` keeps only S&P 500 tickers, so a portfolio weight computed from it is
  inflated by a manager-specific factor — measured across 12 roster managers' 2026Q1 filings, the
  median S&P 500 share is 47% of positions and 52% of value, spanning Atlantic Investment at
  8.3%/13.1% to AltaRock at 100%/100%. Use `sec13f_manager_holdings` for anything that divides.
  ⚠ It covers **87 of the 106 roster CIKs**: the other 19 carry a CIK that files no 13F at all
  (fund-vs-adviser resolution), listed by name in the informed-capital plan README.
- **`insider_footnotes` has no ticker, and does not need one.** Grain is
  `(accession_number, footnote_id)`; **100% of its 653,480 accessions resolve** to a ticker
  through `insider_transactions`. It is attached at FILING grain — the raw zips' 12/19 `_FN`
  pointer columns, which say *which field on which row* a note explains, were deliberately not
  stored (measured: filing-level attribution is already right for 93.3% of multi-row 10b5-1
  filings). Its main use is recovering what the structured columns do not carry: the
  **`is_10b5_1` flag is 0.0% before 2023**, while a footnote naming a 10b5-1 plan runs at
  9.3–13.7% back to 2006 and tracks the structured flag where they overlap (2024: 12.2% vs 13.5%).
- **`insider_transactions`: NULL on the derivative block is a STATEMENT, not a gap.**
  `exercise_price`, `exercise_date`, `expiration_date` and `underlying_shares` are NULL on all
  1,374,374 non-derivative rows by construction. On options they are 99.2% / 50.9% / 98.3% /
  99.5% filled. ⚠ **`exercise_date` is the date first EXERCISABLE (vesting), not the date
  exercised** — it is present on 23.5% of grant rows and 46,300 of those postdate the
  transaction; the exercise event is `transaction_code='M'` on `transaction_date`. Where it is
  NULL on an option, 97.5% of those rows sit on a filing carrying a vesting footnote.
  ⚠ **7,949 rows (0.41%, tickers EA and AVB) predate the 2026-09-08 enrichment and carry none of
  the 12 new columns** — they left the S&P 500 before the re-parse, so the universe filter
  skipped them. Enrichment fill is 100.00% for every in-universe ticker and 0.00% for those two.
- **⚠ `insider_transactions.value_usd` CANNOT BE SUMMED OR AVERAGED AS STORED.** Measured
  2026-09-10, and note carefully **which population each number describes** — they are three
  different ones and mixing them is how this defect reads as smaller than it is:

  | population | rows | sum | mean | median |
  |---|---|---|---|---|
  | whole table, `value_usd` non-null | 1,934,681 | **$8.66e17** | **$447,771,735,138** | $16,516 |
  | non-derivative only | 1,412,708 | $2.18e16 | $15,409,854,524 | $54,173 |
  | market-priced non-deriv `P`/`S`/`F` | 875,361 | $2.14e16 | — | — |
  | **after scope + repair** (what a feature may sum) | 673,746 | **$1,576bn** | **$2,339,662** | $114,116 |

  A **$447bn mean per Form 4 line is not a transaction size, it is one row**: the single
  largest (`NCLH` 2021-03-09, "Exchangeable Senior Notes due 2026", 414m "shares" at
  "$1.03e9") is **49.4% of the whole-table sum on its own**, the top three are **79%**, and
  dropping the top five leaves **0.04%**. Those three are convertible/exchangeable NOTES
  where the filer put the principal amount in `shares`. The median, $16,516, is sane
  throughout — only the tail is broken.

  Within the population a price screen can reach — market-priced non-derivative `P`/`S`/`F`
  rows — **210 rows of 875,361 (0.024%) carry 99.99% of that population's total**, because
  `price_per_share` is wrong on them. `value_usd == shares × price_per_share` holds on 100%
  of them, so the multiplication is right and the price is the corrupt field. 157 of the 210
  are decimal slips (74 at 10⁻¹, 70 at 10⁻², 13 at 10⁻³); the rest are the transaction total
  typed into the per-share box (`AMD` 2017-08-04, 40m shares "@ $525,600,000"). ⚠ The three
  note rows above are **not** among the 210 — they are derivative rows, removed by SCOPE
  rather than by the price screen, which is why a repair alone is not enough and the scope
  cut is not optional.

  Phase 1b reconciled this table against the SEC's own zips with 0 mismatches, so **these are
  the filers' numbers and there is nothing to correct upstream.** Any consumer must scope and
  repair: `data_aggregate/utils/institutionals/insider_quality.py` keeps common-stock,
  open-market, priced rows and repairs the price against a per-ticker, **per-share-class**
  median filed price in a centred 31-day window — split-free, because both sides carry the
  same unknown split basis. Post-repair the open-market tape reads **$142.2bn of purchases
  (median $30,494) against $1,434.1bn of sales (median $120,850)** over 2006–2026, a 10:1
  sell/buy ratio that matches the known stylised fact and is the independent check that the
  repaired figures are the right order of magnitude.
- **⚠ `insider_transactions.ticker` IS RESOLVED SYMBOL-FIRST, so a reused ticker carries another
  company's insiders.** `_filter_universe` keeps a row whose SEC trading symbol is in the
  universe and only falls back to the issuer CIK. Measured against the registrant register:
  **38,910 rows (1.92%), 8,640 of them P/S, across 72 in-universe tickers** hold an `issuer_cik` outside
  their ticker's lineage — 2,075 Trane rows under `IR` (CIK 0001466258, which is `TT`'s own
  universe CIK), 2,046 Weight Watchers under `WTW`, 1,207 CoreSite under `COR`, 1,111 the old
  Constellation Energy under `CEG`, Axovant Sciences under `AXON`. The same screen also names
  ~40 **genuine** predecessors missing from the register (DuPont under `DD`, Chubb Corp under
  `CB`, Avago under `AVGO`), and the two are indistinguishable by date span, so a blanket
  CIK-first cut would delete real history. Unfixed as of 2026-09-10; the reparse that would
  apply a fix needs **no download** (`insider-transactions --reparse`, 81 cached quarters).
- **`sec_def14a` is 2023+ BY REGULATION, and that is correct behaviour rather than a gap.**
  Item 402(v) (Pay-versus-Performance) applies to fiscal years ending on or after 2022-12-16, so a
  proxy covering an earlier year carries no `ecd:` facts and gets NO ROW at all. Measured over the
  23 baseline tickers: 81 rows written, 5 filings correctly skipped, and the PVP columns
  (`peo_name`, `peo_total_comp`, `peo_actually_paid_comp`, TSR, net income) are **100% filled** on
  what is written. Do not build a long-history governance feature off this table — the history is
  `def14a_llm`'s (497 tickers), and every prose field (comp tables, director fees, ownership,
  audit fees, pay ratio) now lives there and in its four child tables.
- **`sec_def14a` and its four `sec_def14a_*` child tables are GONE from Postgres**, dropped at an
  earlier cutover. The retired edgar parser returned values that were silently WRONG rather than
  absent, and the LLM path's own tables replaced them on every measured axis. `ecd.py` still holds
  the XBRL Pay-versus-Performance reader that wrote `sec_def14a`, so **PVP columns
  (`peo_name`, `peo_total_comp`, `peo_actually_paid_comp`, `n_peos`) exist nowhere in the live
  database** — do not look for them on `def14a_llm`, which has never had them.
- **The four `def14a_*` child tables hold 15 tickers, not 497.** They were created by the
  2026-09-03 smoke run over AAPL, JPM, BA, NKE, SBUX, GOOGL, BRK-B, XOM, PG, CAT, PFE, A, AMAT,
  TDG, GE; a universe backfill has not been run. `def14a_llm` still covers all 497, so a join
  from the parent to a child silently narrows to those 15 — project accordingly.
- **`sec_8k_votes` is REGISTERED but still not in Postgres.** Its DDL is in `sql/schema.sql` and
  `fetch_8k_votes_llm` is merged, but nothing has run it, so a query raises `TableMissingError`
  — the intended behaviour (a missing table is a visible fault; an empty one reads as "this
  company discloses nothing").
- **`def14a_llm` is 54 columns now, and only 409 rows carry the newest 12.** The three retired
  technology columns were dropped by `scripts/gpt_refactor_smoke_prep.py` (45 → 42), then the
  smoke run's first write took it 42 → 54 via `store.save`'s `ADD COLUMN` schema evolution. The
  12 added columns (`auditor_name`, the four `audit_fees_*`, `sct_years`, `pct_gender_stated`,
  `n_ownership_rows`, `n_director_comp_rows`, `n_women_directors_vs_inferred`,
  `auditor_since_year`, `auditor_fees_prior`) are **NULL on the 8,375 rows that predate it**.
- **XOM's 14 rows are pre-refactor.** `sp500_tickers` maps XOM to CIK 2115436, which lists **0**
  DEF 14A filings; the real ExxonMobil CIK is 34088 (29 filings). Fix the universe row before
  expecting any EDGAR fetcher to return XOM data.
- **The four `fundamentals_*` tables cover 54 tickers, not 500 — this is the Phase 5 rebuild
  scope, not a defect.** All four were dropped and rebuilt from scratch on 2026-08-24
  (`scripts/recreate_fundamentals_tables.py`), so the earlier 491-ticker / 239-column
  `fundamentals_history_sec` and its 445-ticker facts table no longer exist. Their ticker sets are
  now identical by construction. Widening to the full roster is Phase 9's acceptance step; until
  then any cube built off these tables covers 54 names and every coverage rate computed against a
  500-ticker denominator will read ~11%.
- **`fundamentals_history` (the MERGED table) carries 51,255 rows over 489 tickers, of which 49,280
  hold a whole trailing twelve** — `totalRevenue` is NULL on 1,975 rows (3.9%). Most of those are
  structural: a ticker's first three quarters can never have a four-quarter window, and 220 windows
  hold a quarter whose vendor `revenue` is absent. The SEC block joins on only 5,934 rows over 97
  tickers — the stated coverage asymmetry, not a gap.
  Rebuilt 2026-08-31 after two `build_ttm` window defects were fixed (duplicate ARQ filings and a
  miscalibrated drift gate; see [data_sources.md](data_sources.md)), which recovered **+1,265 whole
  trailing twelves** and removed **353 duplicate `as_of` rows**. AVGO went 0 → 65 rows, KR 0 → 112,
  AZO 0 → 113, COST 2 → 121.
- **`fundamentals_history_sec` went 27,602 rows → 3,258 and 239 columns → 69.** Both are deliberate.
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
  ⚠ It is also **not usable before 2013-06-30**: distinct managers per ticker run 14.0
  (2012-09-30) → 22.1 → 41.4 → **555.9** (2013-06-30). The earlier quarters are the FETCH, not
  the market, and they look perfectly valid. `scripts/source_coverage_report.py` regenerates
  that series on demand.
- **`sec13f_hr` is the S&P 500 SLICE of each manager's book**, not the book — the extraction
  filters to the universe. A portfolio weight computed from it is inflated by a manager-specific
  1.0x–7.6x (measured 2026Q1: Atlantic Investment 8.3% of positions, AltaRock 100%). Use
  `sec13f_manager_holdings` as the denominator.
- **`short_interest` starts 2018-08-01, and that floor MOVES.** FINRA serves the RegSHO files
  from a rolling ~8-year CDN window (probed 2026-09-08: last 403 `20180731`, first 200
  `20180801`). The stored `min(date)` of 2017-12-29 is one anomalous file outside the window,
  not a history start — and rows below the boundary cannot be re-fetched if lost.
- Vector columns (`embedding` on `earning_calls_embedding`, `ticker_embeddings`) are Postgres
  `float8[]`. **SQLite's driver refuses to bind a Python list**, which is why
  `tests/conftest.py::FakeStore` exists alongside `sqlite_store`.

### ⚠ `cik` means THE FILER, not the roster (semantic change, 2026-09-10)

On every SEC-derived table `cik` is the CIK that **actually filed the document**. It used to be a
stamp: the tier-A fetchers resolved by TICKER and then wrote `sp500_tickers.cik` onto each row, so
`sec_8k` held **0 of 491** tickers with more than one distinct CIK and **0** rows where
`sec_8k.cik <> sp500_tickers.cik`. XOM's 526 8-K rows back to 1996 all carried CIK 2115436 — an
entity whose entire archive is 29 filings beginning 2026-07-01. The one column that would have
made a registrant boundary self-announcing instead reported the roster back to itself.

Consequences for anyone reading these tables:

- **Do not join `cik` as an identity key.** A ticker that crossed a registrant boundary carries
  two or more CIKs by design; group on `ticker`.
- **A register ticker showing one CIK means its rows predate the re-fetch**, not that no boundary
  exists.
- `sec_13d` / `sec_13g` always read the ISSUER's CIK off the filing's structured data, which is
  why `sec_13g` alone showed 25 tickers with more than one CIK before this change.
- The register itself lives in `configs/sec/registrant_cutover.json`; see
  [data_sources.md](data_sources.md) for how segments combine per form.

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
