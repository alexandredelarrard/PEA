# Phase 2 — `prices_splits` and the corroborated union ✅

**Parent**: [PLAN.md](PLAN.md) · **Depends on**: P0 · **Blocks**: P3 · **Estimate**: 2-3h

Run concurrently with [P1](PHASE-1-prices-extract.md).

## Goal

A split event list that is actually complete, persisted in its own table.

## What it is NOT for

**Market cap does not use this list.** After P3 the split factor cancels between `close_split`
and `sharesbas`, so the mcap path never reads a split event. This list has exactly three
consumers:

1. `sharesOutstandingPit` (P3) — needs a complete `F(d)`, and today's 9 holes make the PIT counts
   wrong for GOOGL, NVDA, TSLA, AVGO, CMG, ANET, BKNG, MNST, AMCR
2. the split-triggered price re-pull (P1)
3. the P5 validator's corroboration check on the spike-and-revert scan

## The measured gap

`sharadar_actions` is fresh (max date 2026-08-25, 510,016 rows, 705 post-2020 `split` rows) but
holed. yfinance has **every** missing event (measured 2026-09-01 via `yf.Ticker(t).splits`):

| event | `sharadar_actions` | yfinance |
|---|---|---|
| GOOGL 2022-07-18 x20 | absent | present |
| NVDA 2021-07-20 x4 | absent | present |
| TSLA 2022-08-25 x3 | absent | present |
| AVGO 2024-07-15 x10 | absent | present |
| CMG 2024-06-26 x50 | absent | present |
| ANET 2024-12-04 x4 | absent | present |
| BKNG 2026-04-06 x25 | absent | present |
| MNST 2023-03-28 x2, 2026-08-11 x2 | absent | present |
| AMCR 2026-01-15 x0.2 (reverse) | absent | present |
| WTW 2016-01-05 x0.3775 | present | present — genuine |
| SJM 2002-05-30 x0.945 | present | **absent** — false positive, a merger share-issuance factor |

## Changes

### 1. New table `prices_splits`

- [x] `sql/schema.sql`: `(date TIMESTAMP, ticker TEXT, ratio DOUBLE PRECISION, PRIMARY KEY
      (ticker, date))`. Hand-splice; the diff must be purely additive.
- [x] `src/data_store/schema.py`: register beside `dividends`
      (`Table("prices_splits", ("ticker","date"), date_col="date")`).

### 2. `src/data_extract/utils/prices/fetch_splits.py` (new)

Mirror `fetch_dividends.py` exactly — its own fetcher with its own sparse frontier, reusing
`download_ohlcv(..., actions=True)`. The `Stock Splits` column is already in the response and is
thrown away at [fetch_dividends.py:39](src/data_extract/utils/prices/fetch_dividends.py#L39).

- [x] `_extract_splits`: keep **only non-zero** rows. Unlike dividends (where a stored 0 makes the
      refresh idempotent) a zero split is meaningless and would bloat the table by 3.2M rows.
- [x] Empty in, empty out — a total yfinance outage must no-op, not `KeyError`.
- [x] Register a `splits` CLI command with `-F/--full`.

### 3. Union in `split_events`

[field_map.py:467-503](src/data_extract/utils/fundamentals_sharadar/field_map.py#L467-L503)
currently reads `sharadar_actions` only. Add `prices_splits` and apply the corroboration rule:

| case | action |
|---|---|
| in both sources | **keep** (WTW 0.3775 — genuine, and non-integer factors do occur) |
| yfinance only | **keep** — this is the 9-hole fix |
| Sharadar only, integer factor | **keep**, and log at WARNING for review |
| Sharadar only, non-integer factor | **DROP** — merger/exchange artefact (SJM 0.945, CCL 0.0012) |

- [x] Keep the existing spinoff co-dating exclusion on the Sharadar side (the HON trap) — it is
      correct and orthogonal to this rule.
- [x] Where both sources carry an event on nearby dates, prefer the yfinance date (the ex-date
      Yahoo actually adjusted its own prices on) so `F(d)` and `close_split` step on the same day.
- [x] Log a one-line summary per run: kept-both / kept-yf-only / kept-sharadar-only / dropped.

### 4. Investigate CCL

- [x] CCL's merged/SEC ratio spans 0.0012 – 1.4032 — a compounding over-de-adjustment. Confirm it
      is Sharadar-only non-integer rows and that the rule drops them. If it is *not* explained by
      the rule, record what it is; do not widen the rule to make one ticker pass.

## Verification

- [x] `prices_splits` contains all 9 previously-missing events at the ratios in the table above
- [x] `SJM 2002-05-30` is **absent** from the unioned list
- [x] `WTW 2016-01-05 0.3775` is **present**
- [x] CCL's residual is explained in writing, and its unioned events reconcile against SEC
- [x] `forward_split_factor("AAPL", 2020-07-31)` == 4.0 and
      `forward_split_factor("GOOGL", 2022-02-02)` == 20.0 (today it returns 1.0)
- [x] Unit test: the four union cases, on a synthetic fixture — no DB, no network
- [x] Row count is plausible (a few thousand, not a few hundred thousand — catches the zero-row bug)

## Rollback

Drop `prices_splits` and revert the `split_events` union. Nothing else reads it until P3.
