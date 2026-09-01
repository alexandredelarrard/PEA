---
type: DATA
session_id: a16e81b6-02b0-4abc-ab3c-55f16f5e8314
generated_at: 2026-09-01T21:06:25+00:00
baseline: {head_sha: 5538ed90be2594eec0b8061e103297863f4eb8c7}
generator: scripts/dod/data_profile.py@1
---

## 1. Scope

**SAMPLE SCOPE** — a metric without its scope is not a measurement:

- tables: fundamentals_history, prices, prices_splits
- tickers: **all** (no ticker filter)
- since: **no lower bound**
- row limit per table: **none**
- full-scope tables (eligible to set the baseline): prices, prices_splits, fundamentals_history

**What was asked:** implement all six phases of
`reports/planning/active-tasks/2026-09-01-price-shares-basis-fix/PLAN.md` — put price, share
count and market cap on ONE adjustment basis.

The defect: `daily_market_cap` multiplied a split-AND-dividend-adjusted price by a
de-adjusted (point-in-time) share count. Two different bases, so the product was neither the
historical market cap nor a consistent adjusted one, and the error decomposed exactly into
`split_part x dividend_part` — both functions of the FUTURE. The split leg alone was a clean
look-ahead worth +12.4pp/yr on 21.5% of rows.

What shipped: `prices` gained `close_split` (split-only) and `close_total` (total return) and
LOST `close`, re-downloaded in full (3,263,459 rows); `prices_splits` is a new yfinance split
table unioned with `sharadar_actions` under a corroboration rule; `sharesOutstanding` became
the vendor's `sharesbas` verbatim and the point-in-time count moved to
`sharesOutstandingPit`; every consumer was routed to the basis it needs; the labels became
forward COMPOUNDED total returns; and `validate prices` now gates the cube build.

## 2. Gates

| Gate | Check | Verdict | Detail |
|---|---|---|---|
| D1 | declared PK unique over the rows profiled | **PASS** | unique across 3 table(s): prices, prices_splits, fundamentals_history |
| D2 | row count not decreased | **PASS** | 1 table(s) at or above baseline |
| D3 | no column lost | **PASS** | 1 table(s) keep every baseline column |
| D4 | date range covers the expected window | **FAIL** | prices_splits: max 2026-08-11 < 2026-08-28 |
| D5 | per-field null rate not worse | **PASS** | 91 field(s) at or below baseline (+0.5pp slack) |

**1 FAIL** — D4. The work is **NOT done**.

## 3. Metrics

_Observed values only — no verdicts. `rows`, `date_min` and `date_max` are **table-wide** (server-side); every other number is over the **sample** described in §1. Do not compare across the two._

**Tables**

| table | exists | rows | sampled | cols | pk | pk_absent_cols | pk_dupes | date_min | date_max | sample_date_min | sample_date_max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fundamentals_history | yes | 51,255 | 51,255 | 92 | ticker,as_of | — | 0 | 1995-09-01 | 2026-08-28 | 1995-09-01 | 2026-08-28 |
| prices | yes | 3,263,459 | 3,263,459 | 8 | ticker,date | — | 0 | 1995-09-01 00:00:00 | 2026-09-01 00:00:00 | 1995-09-01 | 2026-09-01 |
| prices_splits | yes | 859 | 859 | 3 | ticker,date | — | 0 | 1995-09-05 00:00:00 | 2026-08-11 00:00:00 | 1995-09-05 | 2026-08-11 |

**Fields** (worst null rate first, top 60)

| table | field | dtype | null_% | nunique | mean | std | min | p01 | p50 | p99 | max | mad_outliers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fundamentals_history | longTermDebtCurrentOnly | object | 100 | 0 | — | — | — | — | — | — | — | — |
| fundamentals_history | restrictedCash | object | 100 | 0 | — | — | — | — | — | — | — | — |
| fundamentals_history | shortTermBorrowingsOnly | object | 100 | 0 | — | — | — | — | — | — | — | — |
| fundamentals_history | rentalIncome_sec | float64 | 99.59 | 188 | 2.22865e+09 | 2.87623e+09 | 4.6e+07 | 4.7e+07 | 1.0855e+09 | 1.04742e+10 | 1.11068e+10 | 27 |
| fundamentals_history | realizedInvestmentGains_sec | float64 | 99.54 | 212 | -5.77456e+07 | 1.463e+09 | -1.2717e+10 | -4.68237e+09 | 4.25e+07 | 2.95177e+09 | 5.703e+09 | 20 |
| fundamentals_history | noninterestIncome_sec | float64 | 99.44 | 287 | 2.67174e+10 | 1.59722e+10 | 1.387e+09 | 1.4043e+09 | 2.7111e+10 | 5.91951e+10 | 6.7493e+10 | 0 |
| fundamentals_history | netInvestmentIncome_sec | float64 | 99.24 | 374 | 4.26425e+09 | 5.17221e+09 | 1.2136e+07 | 3.32408e+08 | 3.156e+09 | 2.1002e+10 | 2.7792e+10 | 33 |
| fundamentals_history | premiumsEarned_sec | float64 | 98.95 | 518 | 1.49033e+10 | 1.44213e+10 | 12,000 | 14,000 | 8.68798e+09 | 5.17172e+10 | 6.1005e+10 | 15 |
| fundamentals_history | netInterestIncome_sec | float64 | 98.86 | 558 | 1.20742e+10 | 1.97037e+10 | -5.269e+09 | -4.58675e+09 | 2.928e+09 | 5.8868e+10 | 6.3471e+10 | 131 |
| fundamentals_history | financeLeaseLiability_sec | float64 | 98.76 | 419 | 9.45761e+08 | 4.12498e+09 | 0 | 310,800 | 5.61e+07 | 2.55634e+10 | 2.8434e+10 | 65 |
| fundamentals_history | operatingLeaseLiability_sec | float64 | 96.52 | 1,556 | 2.18335e+09 | 8.4325e+09 | 716,000 | 3.2862e+07 | 4.5705e+08 | 5.82944e+10 | 9.632e+10 | 296 |
| fundamentals_history | ppeGross_sec | float64 | 95.34 | 2,339 | 1.3896e+10 | 2.50075e+10 | 6.63872e+06 | 1.18806e+08 | 5.5345e+09 | 1.13175e+11 | 5.34098e+11 | 277 |
| fundamentals_history | minorityInterest_sec | float64 | 94.93 | 1,621 | 8.22494e+08 | 2.34604e+09 | -2.86e+08 | -2.7e+07 | 1.5212e+08 | 1.08097e+10 | 2.8252e+10 | 437 |
| fundamentals_history | stockholdersEquityInclNci | float64 | 94.93 | 2,554 | 1.97308e+10 | 3.50035e+10 | -2.3562e+10 | -3.49957e+09 | 8.804e+09 | 2.09023e+11 | 2.33021e+11 | 340 |
| fundamentals_history | accumulatedDepreciation_sec | float64 | 93.63 | 3,127 | 7.0139e+09 | 1.19877e+10 | -1.5544e+10 | 4.15408e+07 | 2.71881e+09 | 6.18577e+10 | 1.77073e+11 | 430 |
| fundamentals_history | intangiblesExGoodwill_sec | float64 | 92.91 | 3,342 | 3.71561e+09 | 8.11678e+09 | 0 | 4.98108e+06 | 1.273e+09 | 4.1866e+10 | 8.2876e+10 | 420 |
| fundamentals_history | goodwill_sec | float64 | 89.84 | 4,020 | 6.99757e+09 | 1.23757e+10 | 0 | 2.1939e+07 | 2.5025e+09 | 6.9022e+10 | 9.7873e+10 | 793 |
| fundamentals_history | regime_sec | str | 88.42 | 6 | — | — | — | — | — | — | — | — |
| fundamentals_history | dividendsPerShare | float64 | 47.31 | 4,622 | 1.70339 | 2.35496 | 0.004 | 0.02204 | 1.15 | 10.04 | 105.49 | 1,256 |
| fundamentals_history | dilutedShares | float64 | 33.92 | 29,937 | 8.99905e+08 | 2.21043e+09 | 1.30175e+06 | 2.64441e+07 | 3.3225e+08 | 1.05721e+10 | 2.65189e+10 | 4,083 |
| fundamentals_history | epsDiluted | float64 | 33.92 | 33,852 | 3.73633 | 15.6752 | -755.049 | -6.42471 | 2.03414 | 34.3483 | 506.987 | 2,719 |
| fundamentals_history | optionOverhang | float64 | 33.92 | 30,387 | 0.0208964 | 0.0648781 | -0.182703 | -0.00154546 | 0.00961227 | 0.180787 | 4.31762 | 2,997 |
| fundamentals_history | inventory | float64 | 32.53 | 25,088 | 2.17313e+09 | 4.7974e+09 | 69,200 | 5.20664e+06 | 7.10824e+08 | 1.89587e+10 | 8.9077e+10 | 5,153 |
| fundamentals_history | employees_sec | float64 | 24.32 | 4,306 | 46,266.9 | 138,234 | 100 | 199 | 15,000 | 400,000 | 3.4e+06 | 5,221 |
| fundamentals_history | interestExpense | float64 | 21.72 | 21,943 | 3.84436e+08 | 9.34491e+08 | -3.273e+09 | -836,600 | 1.56e+08 | 3.4436e+09 | 2.6404e+10 | 4,047 |
| fundamentals_history | currentLiabilities | float64 | 18.12 | 35,561 | 6.41886e+09 | 1.34748e+10 | 182,750 | 1.785e+07 | 2.18e+09 | 6.97139e+10 | 3.35735e+11 | 5,743 |
| fundamentals_history | nonCurrentLiabilities | float64 | 18.12 | 37,778 | 1.13488e+10 | 2.20454e+10 | -5.29e+09 | 614,070 | 3.8255e+09 | 1.11044e+11 | 3.08145e+11 | 5,794 |
| fundamentals_history | longTermDebt | float64 | 18.1 | 32,313 | 7.24001e+09 | 1.43536e+10 | 0 | 0 | 2.59272e+09 | 7.34337e+10 | 2.23232e+11 | 5,008 |
| fundamentals_history | shortTermDebt | float64 | 17.86 | 20,938 | 1.12522e+09 | 4.30921e+09 | 0 | 0 | 1.88953e+08 | 1.271e+10 | 1.67737e+11 | 8,854 |
| fundamentals_history | currentAssets | float64 | 17.82 | 36,778 | 8.19464e+09 | 1.6308e+10 | 205,878 | 6.39024e+07 | 3.061e+09 | 8.06375e+10 | 3.43841e+11 | 5,589 |
| fundamentals_history | nonCurrentAssets | float64 | 17.82 | 39,651 | 1.96191e+10 | 3.83966e+10 | 0 | 2.72902e+07 | 7.01e+09 | 1.99638e+11 | 8.46425e+11 | 5,198 |
| fundamentals_history | cash | float64 | 17.81 | 31,055 | 3.11305e+09 | 9.46795e+09 | 0 | 3.8803e+06 | 7.14e+08 | 4.31828e+10 | 2.42474e+11 | 6,607 |
| fundamentals_history | longTermInvestments | float64 | 17.81 | 15,104 | 1.62451e+09 | 8.12347e+09 | 0 | 0 | 0 | 2.82279e+10 | 2.07944e+11 | 1,653 |
| fundamentals_history | shortTermInvestments | float64 | 17.81 | 13,395 | 1.27744e+09 | 7.18888e+09 | 0 | 0 | 0 | 2.94116e+10 | 1.86563e+11 | 1,487 |
| fundamentals_history | capex | float64 | 9.12 | 29,625 | 1.24808e+09 | 3.67144e+09 | 0 | 0 | 3.10301e+08 | 1.53499e+10 | 1.69007e+11 | 7,821 |
| fundamentals_history | exchangeRateEffect | float64 | 4.34 | 12,348 | -1.27501e+07 | 3.70989e+08 | -3.4292e+10 | -4.23838e+08 | 0 | 2.44e+08 | 1.914e+10 | 16,185 |
| fundamentals_history | dividendsPaid | float64 | 4.16 | 21,052 | -7.41727e+08 | 1.71313e+09 | -3.6112e+10 | -9.03952e+09 | -1.89e+08 | 0 | 2.342e+09 | 7,355 |
| fundamentals_history | investmentAcquisitionsNet | float64 | 4.12 | 23,204 | -1.05581e+09 | 1.02946e+10 | -3.51492e+11 | -2.7012e+10 | 0 | 6.16504e+09 | 1.82052e+11 | 15,844 |
| fundamentals_history | ebitda | float64 | 4.1 | 39,143 | 3.86241e+09 | 9.00912e+09 | -9.0836e+10 | -1.1035e+09 | 1.461e+09 | 4.031e+10 | 2.01266e+11 | 6,025 |
| fundamentals_history | stockBasedComp | float64 | 4.1 | 17,782 | 1.63552e+08 | 8.64916e+08 | -1.858e+09 | -1.1324e+07 | 2.26e+07 | 2.19645e+09 | 2.8147e+10 | 8,805 |
| fundamentals_history | businessAcquisitionsNet | float64 | 4.09 | 21,116 | -3.76654e+08 | 2.53728e+09 | -6.9795e+10 | -9.28824e+09 | -1e+07 | 3.04905e+09 | 1.08089e+11 | 14,247 |
| fundamentals_history | debtIssuanceNet | float64 | 4.09 | 30,550 | 5.0464e+08 | 7.11311e+09 | -2.4079e+11 | -9.2418e+09 | 0 | 1.69655e+10 | 2.30811e+11 | 9,130 |
| fundamentals_history | equityIssuanceNet | float64 | 4.09 | 30,679 | -8.10265e+08 | 3.51708e+09 | -1.01109e+11 | -1.15112e+10 | -8.6e+07 | 1.844e+09 | 2.4277e+10 | 10,529 |
| fundamentals_history | financingCashFlow | float64 | 4.09 | 38,228 | -2.77074e+08 | 1.27632e+10 | -1.99568e+11 | -2.07914e+10 | -2.2662e+08 | 2.81298e+10 | 5.96645e+11 | 8,722 |
| fundamentals_history | netCashFlow | float64 | 4.09 | 32,425 | 1.9543e+08 | 6.09948e+09 | -2.08532e+11 | -6.29206e+09 | 1.7e+07 | 8.6254e+09 | 3.43538e+11 | 9,433 |
| fundamentals_history | depAmort | float64 | 4.08 | 30,154 | 1.05978e+09 | 2.5553e+09 | -3.376e+09 | 0 | 3.494e+08 | 1.28911e+10 | 7.52e+10 | 6,250 |
| fundamentals_history | freeCashflow | float64 | 4.08 | 37,552 | 1.96004e+09 | 7.1165e+09 | -3.1207e+11 | -4.39956e+09 | 6.09036e+08 | 2.47358e+10 | 2.12806e+11 | 7,736 |
| fundamentals_history | operatingCashFlow | float64 | 4.08 | 38,242 | 3.15434e+09 | 8.8722e+09 | -3.10373e+11 | -1.87456e+09 | 1.15086e+09 | 3.62933e+10 | 2.14811e+11 | 6,485 |
| fundamentals_history | investingCashFlow | float64 | 4.07 | 37,517 | -2.64658e+09 | 1.14403e+10 | -3.65258e+11 | -3.75488e+10 | -5.8e+08 | 4.75596e+09 | 1.98067e+11 | 8,917 |
| fundamentals_history | profitMargins | float64 | 3.88 | 49,239 | -0.0370433 | 15.1479 | -2,416.28 | -0.596524 | 0.098106 | 0.547627 | 14.1232 | 2,697 |
| fundamentals_history | effectiveTaxRate | float64 | 3.87 | 47,136 | 0.260311 | 4.91717 | -349 | -1.21793 | 0.257363 | 1.20988 | 652 | 2,368 |
| fundamentals_history | netIncome | float64 | 3.87 | 36,612 | 1.95545e+09 | 5.82232e+09 | -9.9289e+10 | -2.88303e+09 | 6.39e+08 | 2.12636e+10 | 2.44205e+11 | 7,303 |
| fundamentals_history | netIncomeCommon | float64 | 3.87 | 36,539 | 1.90349e+09 | 5.76706e+09 | -9.9289e+10 | -2.86462e+09 | 6.17248e+08 | 2.05924e+10 | 2.44119e+11 | 7,311 |
| fundamentals_history | netIncomeDiscontinued | float64 | 3.87 | 4,154 | -1.35933e+07 | 3.68279e+08 | -2.1827e+10 | -5.1684e+08 | 0 | 2.3028e+08 | 8.329e+09 | 1,442 |
| fundamentals_history | netIncomeToNci | float64 | 3.87 | 6,565 | 3.60884e+07 | 3.01514e+08 | -6.096e+09 | -8.677e+07 | 0 | 8.0184e+08 | 1.205e+10 | 1,958 |
| fundamentals_history | preferredDividends | float64 | 3.87 | 2,910 | 1.61907e+07 | 1.51379e+08 | -1.772e+09 | 0 | 0 | 3.4e+08 | 9.622e+09 | 1,669 |
| fundamentals_history | pretaxIncome | float64 | 3.87 | 37,917 | 2.54078e+09 | 7.27761e+09 | -1.07663e+11 | -3.09552e+09 | 8.44696e+08 | 2.71017e+10 | 2.99269e+11 | 7,079 |
| fundamentals_history | researchAndDevelopment | float64 | 3.87 | 12,415 | 5.19668e+08 | 2.83842e+09 | -6.64e+07 | 0 | 0 | 8.98128e+09 | 1.21086e+11 | 2,113 |
| fundamentals_history | returnOnEquity | float64 | 3.87 | 49,238 | 0.141319 | 5.43325 | -306.574 | -2.21453 | 0.144901 | 2.21311 | 678.101 | 5,375 |
| fundamentals_history | sellingGeneralAdmin | float64 | 3.87 | 36,608 | 3.51161e+09 | 8.1741e+09 | -4.247e+09 | 0 | 1.00192e+09 | 4.01637e+10 | 1.53377e+11 | 7,069 |

## 4. Evidence

- baseline file: `reports/baselines/data_profile.json` (2 table(s) recorded)
- `fundamentals_history`: 51,255 rows, 92 cols, 51,255 sampled
- `prices`: 3,263,459 rows, 8 cols, 3,263,459 sampled
- `prices_splits`: 859 rows, 3 cols, 859 sampled

## 5. Regressions, gaps and deliberate omissions

- **D4 FAILS on `prices_splits`, and it should not be read as a freshness failure.**
  `--expect-through 2026-08-28` is a DAILY-series check; `prices_splits` is a sparse EVENT
  table (859 rows / 343 tickers over 31 years). Its max date is 2026-08-11 because that is
  the last day any S&P 500 name actually split (MNST 2:1). There is nothing to backfill. The
  two daily tables in the same run both reach their expected window: `prices` 2026-09-01,
  `fundamentals_history` 2026-08-28. Verified by the fetcher's own incremental frontier,
  which now looks back a full year from `min(table max, today - 1y)` = 2025-09-01.

- ⚠ **The market-cap identity tops out at 87.4%, not the plan's >99% — because the REFERENCE
  is the inconsistent side.** Yahoo back-adjusts prices for SPINOFFS; Sharadar's `sharesbas`
  does not. So for ~226 tickers `sharadar.marketcap` (= `price x sharesbas`) is internally
  inconsistent. HON proves it: `sharesbas` is unchanged across its 2026-06-29 spinoff
  (316,826,560 -> 316,940,010) while its `price` drops 428.68 -> 246.27. **The control
  settles which side is at fault**: `sharadar.price x sharesbas / marketcap` is within 1% on
  **99.82%** of rows, so the identity is sound and the residual is purely a vendor
  disagreement about corporate actions. The cancellation identity removes the split and
  dividend legs but leaves a **spinoff leg**. This was not anticipated by the plan and is the
  single most important thing to carry forward.

- ⚠ **MNST's 97/47 alternation is an UPSTREAM YAHOO DEFECT and is still present.** The plan
  expected the full re-download to clear it. It does not: a freshly emptied table reproduces
  it exactly, and both `yf.download` and `Ticker.history` serve the same alternating series.
  Sharadar prices MNST at 45.18 on 2026-08-07 where Yahoo says 90.36 — exactly 2x — so Yahoo
  never back-adjusted MNST for the 2:1 split its OWN splits feed reports on 2026-08-11.
  `validate prices` flags it (6 unexplained jumps, no corroborating split) and the gate lets
  it through only because it sits inside the measured 1e-4 budget. **MNST's returns,
  momentum, vol and betas are wrong for July–August 2026 and no code change here can fix
  that** — it needs a vendor override or a ticker exclusion, which is out of this task's
  scope.

- **19 of 96 tickers still fail the SEC cover-page check on `sharesOutstandingPit`** (was 24).
  Every INTEGER-factor offender is resolved — AVGO 10.0, ANET 4.0, CMG 50.0, APH 2.0,
  BKNG 25.0, AMCR 0.2 all now agree — and 12 of the 19 remaining have a median ratio of
  exactly 1.0000 (isolated timing rows, not a basis error). The 7 with a genuinely different
  median are non-split causes the plan explicitly scoped out: CCL 1.3528 is Carnival's
  dual-listed CCL+CUK structure (Sharadar counts the combined entity, the SEC cover page
  counts Carnival Corporation only), CMCSA 1.3799 and ACN/AOS are spinoff/dual-class.

- **CCL's residual is NOT what the research recorded.** The research called it
  "over-de-adjusted, badly". It was never de-adjusted at all — `merged == sharesbas` exactly
  on every CCL row, and its single split (1998-06-15 x2) is corroborated by both vendors. Its
  0.0012 outlier is an **SEC-side extraction defect** at 2021-01-26 (932,485,510,000 shares,
  ~1000x too large), not a price/shares basis problem. The union rule was deliberately NOT
  widened to make it pass.

- **A defect in my own first implementation of the union rule, found and fixed.** The plan
  said "yfinance only -> keep" on the premise that yfinance carries no false positives. It
  does: its `Stock Splits` column also carries SPINOFF factors. Trusting it injected BDX
  2022-04-01 x1.025 and 2026-02-10 x1.272, which compound to 1.304 and put BDX's whole PIT
  series 23% off the SEC cover page — 67 bad rows on a ticker that had none before. The shape
  test now applies to BOTH vendors and only corroboration overrides it.

- **Test state, measured. `tests/data_aggregate/`: 202 passed, 9 failed, 0 errors** (from
  182 passed / 25 failed / 8 errors when P4 started), plus `test_aggregate_regression.py`
  going **2 failed -> 4 passed** with `DECLARED_DRIFT` emptied. `tests/strategies/`,
  `tests/data_peers/`, `tests/validate/`: **199 passed, 0 failed**.
  `tests/data_extract/prices/` + the split-union file: **70 passed**, 5 failed.

  **All 14 remaining failures are PRE-EXISTING and were reproduced at HEAD in a git
  worktree** — none is introduced here:
    * `test_part_registry` (2) — `StepCubeMomentum` declares `open`/`high`/`low`, which
      `cube_part_prices` has never stored, so `atr_14`/`gap_21`/`range_21` are silently
      skipped in the real build. A genuine latent defect, unrelated to basis; see §6.
    * `test_fundamental_features` + `test_latest_quarter_features` (7) — including the
      `daily_market_cap(fund_hist, None)` path, which raises identically at HEAD
      (`pit.py:81` there, `pit.py:91` here — my change to that function is a pure rename).
    * the `SimpleNamespace has no attribute config` fixture family in
      `tests/data_extract/prices/` (5).

  **New tests added:** `tests/data_extract/prices/test_adjustment_basis.py` (7, incl. a
  deliberate-corruption test proving the gate FIRES) and
  `tests/data_extract/sharadar/test_split_union.py` (22).

- **The old table is still on disk.** `prices_pre_basis_fix` holds the 3,263,505 pre-fix rows
  for rollback, as the plan's P1 §Rollback specified. It should be dropped once the user's
  cube rebuild is green.

- **NOT done, by design (D6):** the cube rebuild, the label rebuild and the model
  re-baseline. `cube_part_prices` still holds 3,828,833 rows on the OLD single-`close` basis
  and will fail to load against the new `PriceFrames` contract until rebuilt with `-F`.

## 6. Next actions

1. **Rebuild the cube, in order** — this is the user's step and nothing downstream is valid
   until it runs:
   `build-prices -F` -> `build-target` -> `build-betas` -> the feature parts -> merge.
   `cube_part_prices` MUST be rebuilt with `-F`: it still stores a single `close` column and
   `load_price_frames` now asks for `close_split`/`close_total`.

2. **Re-run the leak check on the rebuilt cube** — the single most valuable thing P0 froze.
   Recompute the `split_part` cohort split: pre-fix, rows where the stock splits AFTER the
   observation date earned **27.81% forward 12m against 15.38%** (+12.4pp/yr on 21.5% of
   rows). On the corrected basis the two groups must become **indistinguishable**. If the gap
   survives, something in P4 is still routing the old basis.
   `"$PY" scripts/basis_baseline.py --tag after-rebuild` and diff against `baseline.json`.

3. **Settle the spinoff cluster.** `validate prices` reports 226 tickers where Yahoo and
   Sharadar disagree about a corporate action. Decide per ticker which vendor restated, and
   record the answer as a waived cluster with evidence — do NOT widen `MCAP_TOLERANCE`.
   Start with the largest: FDX, HON, BDX, SPGI, CMCSA, DD, WDC, GE.

4. **Decide what to do about MNST.** Its Yahoo history is corrupt upstream and cannot be
   fixed by re-pulling. Either override it from Sharadar's `price`, or exclude it until Yahoo
   restates.

5. **Drop `prices_pre_basis_fix`** once step 1 and 2 are green.

6. **Fix the pre-existing `open`/`high`/`low` gap** (separate task): `StepCubeMomentum`
   declares them but `cube_part_prices` does not store them, so `atr_14`, `gap_21` and
   `range_21` never get built. Now that the basis routing distinguishes them correctly, this
   is a cheap win.


- 
<!-- At least one bullet. If genuinely nothing: `- None. Checked: <30+ chars>` -->

## 6. Next actions


- 

```json dod-metrics
{
  "baseline_head_sha": "5538ed90be2594eec0b8061e103297863f4eb8c7",
  "content_hash": "sha256:892069bcb8f49cecedb00571896da6212a9b5adba2ff0dffc85062b1389e55b7",
  "gates": {
    "D1": "PASS",
    "D2": "PASS",
    "D3": "PASS",
    "D4": "FAIL",
    "D5": "PASS"
  },
  "generator": "scripts/dod/data_profile.py@1",
  "metrics": {
    "parts_behind": null,
    "stale_sources": null,
    "tables": {
      "fundamentals_history": {
        "columns": [
          "ticker",
          "as_of",
          "fiscal_end",
          "totalRevenue",
          "premiumsEarned_sec",
          "netInterestIncome_sec",
          "noninterestIncome_sec",
          "netInvestmentIncome_sec",
          "realizedInvestmentGains_sec",
          "rentalIncome_sec",
          "costOfRevenue",
          "grossProfit",
          "grossMargins",
          "sellingGeneralAdmin",
          "researchAndDevelopment",
          "depAmort",
          "stockBasedComp",
          "operatingIncome",
          "operatingMargins",
          "ebitda",
          "interestExpense",
          "pretaxIncome",
          "incomeTaxExpense",
          "effectiveTaxRate",
          "netIncome",
          "profitMargins",
          "epsDiluted",
          "revenue_q",
          "netIncome_q",
          "operatingCashFlow",
          "capex",
          "freeCashflow",
          "cash",
          "restrictedCash",
          "shortTermInvestments",
          "accountsReceivable",
          "inventory",
          "currentAssets",
          "ppeGross_sec",
          "accumulatedDepreciation_sec",
          "ppeNet",
          "goodwill_sec",
          "intangiblesExGoodwill_sec",
          "totalAssets",
          "accountsPayable",
          "currentLiabilities",
          "shortTermDebt",
          "shortTermBorrowingsOnly",
          "longTermDebt",
          "longTermDebtCurrentOnly",
          "operatingLeaseLiability_sec",
          "financeLeaseLiability_sec",
          "totalDebt",
          "totalLiabilities",
          "retainedEarnings",
          "minorityInterest_sec",
          "stockholdersEquity",
          "returnOnEquity",
          "debtToEquity",
          "basicShares",
          "dilutedShares",
          "sharesOutstanding",
          "optionOverhang",
          "stockholdersEquityInclNci",
          "employees_sec",
          "regime_sec",
          "cashAndEquivalents",
          "accumulatedOtherComprehensiveIncome",
          "nonCurrentAssets",
          "nonCurrentLiabilities",
          "totalInvestments",
          "longTermInvestments",
          "taxAssets",
          "taxLiabilities",
          "deferredRevenue",
          "deposits",
          "operatingExpenses",
          "netIncomeToNci",
          "netIncomeDiscontinued",
          "netIncomeCommon",
          "preferredDividends",
          "dividendsPerShare",
          "investingCashFlow",
          "financingCashFlow",
          "dividendsPaid",
          "equityIssuanceNet",
          "businessAcquisitionsNet",
          "investmentAcquisitionsNet",
          "debtIssuanceNet",
          "exchangeRateEffect",
          "netCashFlow",
          "sharesOutstandingPit"
        ],
        "date_col": "as_of",
        "date_max": "2026-08-28",
        "date_min": "1995-09-01",
        "exists": true,
        "fields": {
          "accountsPayable": {
            "dtype": "float64",
            "mad_center": 543000000.0,
            "mad_outliers": 9117,
            "mad_scale": 507331000.0,
            "max": 384290000000.0,
            "mean": 3923486636.5828032,
            "min": 0.0,
            "null_rate": 1.95102916788606e-05,
            "nulls": 1,
            "nunique": 35617,
            "p01": 0.0,
            "p25": 130603000.0,
            "p50": 543000000.0,
            "p75": 1935925000.0,
            "p99": 62288020000.00008,
            "std": 17213271036.718967
          },
          "accountsReceivable": {
            "dtype": "float64",
            "mad_center": 862100000.0,
            "mad_outliers": 7271,
            "mad_scale": 776684000.0,
            "max": 230286000000.0,
            "mean": 3341859712.7127247,
            "min": 0.0,
            "null_rate": 1.95102916788606e-05,
            "nulls": 1,
            "nunique": 36101,
            "p01": 0.0,
            "p25": 225800000.0,
            "p50": 862100000.0,
            "p75": 2553015750.0,
            "p99": 47519520000.000015,
            "std": 9577938597.387993
          },
          "accumulatedDepreciation_sec": {
            "dtype": "float64",
            "mad_center": 2718814000.0,
            "mad_outliers": 430,
            "mad_scale": 2231014000.0,
            "max": 177073000000.0,
            "mean": 7013895279.015309,
            "min": -15544000000.0,
            "null_rate": 0.9362793873768412,
            "nulls": 47989,
            "nunique": 3127,
            "p01": 41540800.0,
            "p25": 927357250.0,
            "p50": 2718814000.0,
            "p75": 7155250000.0,
            "p99": 61857749999.99998,
            "std": 11987735838.408333
          },
          "accumulatedOtherComprehensiveIncome": {
            "dtype": "float64",
            "mad_center": -29275500.0,
            "mad_outliers": 13960,
            "mad_scale": 78348000.0,
            "max": 66287000000.0,
            "mean": -664049389.5691849,
            "min": -48298000000.0,
            "null_rate": 1.95102916788606e-05,
            "nulls": 1,
            "nunique": 27981,
            "p01": -13038350000.0,
            "p25": -356000000.0,
            "p50": -29275500.0,
            "p75": 0.0,
            "p99": 2195230000.0000105,
            "std": 2663495477.851273
          },
          "as_of": {
            "dtype": "object",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 6070
          },
          "basicShares": {
            "dtype": "float64",
            "mad_center": 324091625.0,
            "mad_outliers": 5857,
            "mad_scale": 211495375.0,
            "max": 26273779000.0,
            "mean": 817288336.208505,
            "min": 1592750.0,
            "null_rate": 0.03424056189640035,
            "nulls": 1755,
            "nunique": 45168,
            "p01": 27712126.7975,
            "p25": 152488215.75,
            "p50": 324091625.0,
            "p75": 706000000.0,
            "p99": 9719457500.000042,
            "std": 1929509149.6574743
          },
          "businessAcquisitionsNet": {
            "dtype": "float64",
            "mad_center": -10000000.0,
            "mad_outliers": 14247,
            "mad_scale": 57000000.0,
            "max": 108089000000.0,
            "mean": -376653652.1068413,
            "min": -69795000000.0,
            "null_rate": 0.04093259194224954,
            "nulls": 2098,
            "nunique": 21116,
            "p01": -9288240000.0,
            "p25": -235000000.0,
            "p50": -10000000.0,
            "p75": 0.0,
            "p99": 3049054000.000008,
            "std": 2537277950.6267657
          },
          "capex": {
            "dtype": "float64",
            "mad_center": 310301000.0,
            "mad_outliers": 7821,
            "mad_scale": 274327032.0,
            "max": 169007000000.0,
            "mean": 1248082976.545847,
            "min": 0.0,
            "null_rate": 0.09123012389035216,
            "nulls": 4676,
            "nunique": 29625,
            "p01": 0.0,
            "p25": 88501500.0,
            "p50": 310301000.0,
            "p75": 1071000000.0,
            "p99": 15349860000.000015,
            "std": 3671444371.919023
          },
          "cash": {
            "dtype": "float64",
            "mad_center": 714000000.0,
            "mad_outliers": 6607,
            "mad_scale": 628754000.0,
            "max": 242474000000.0,
            "mean": 3113045269.4116893,
            "min": 0.0,
            "null_rate": 0.17814847331967612,
            "nulls": 9131,
            "nunique": 31055,
            "p01": 3880300.0,
            "p25": 197551250.0,
            "p50": 714000000.0,
            "p75": 2297243750.0,
            "p99": 43182829999.99975,
            "std": 9467948073.453297
          },
          "cashAndEquivalents": {
            "dtype": "float64",
            "mad_center": 613167500.0,
            "mad_outliers": 8455,
            "mad_scale": 547181500.0,
            "max": 759869000000.0,
            "mean": 4854234629.153861,
            "min": 0.0,
            "null_rate": 1.95102916788606e-05,
            "nulls": 1,
            "nunique": 36704,
            "p01": 3292949.9999999995,
            "p25": 162746500.0,
            "p50": 613167500.0,
            "p75": 2038000000.0,
            "p99": 87938520000.00014,
            "std": 28423070837.840343
          },
          "costOfRevenue": {
            "dtype": "float64",
            "mad_center": 3309661000.0,
            "mad_outliers": 6644,
            "mad_scale": 3117347500.0,
            "max": 550183000000.0,
            "mean": 12602844760.66524,
            "min": -4323000000.0,
            "null_rate": 0.03853282606574968,
            "nulls": 1975,
            "nunique": 41049,
            "p01": 0.0,
            "p25": 767171000.0,
            "p50": 3309661000.0,
            "p75": 9928650000.0,
            "p99": 165879099999.9999,
            "std": 32958286660.33456
          },
          "currentAssets": {
            "dtype": "float64",
            "mad_center": 3061000000.0,
            "mad_outliers": 5589,
            "mad_scale": 2367822000.0,
            "max": 343840946000.0,
            "mean": 8194643330.417017,
            "min": 205878.0,
            "null_rate": 0.178167983611355,
            "nulls": 9132,
            "nunique": 36778,
            "p01": 63902438.88,
            "p25": 1179348000.0,
            "p50": 3061000000.0,
            "p75": 8009000000.0,
            "p99": 80637539999.9999,
            "std": 16307967101.854563
          },
          "currentLiabilities": {
            "dtype": "float64",
            "mad_center": 2180000000.0,
            "mad_outliers": 5743,
            "mad_scale": 1811900000.0,
            "max": 335734599000.0,
            "mean": 6418863708.5021925,
            "min": 182750.0,
            "null_rate": 0.18115305823822067,
            "nulls": 9285,
            "nunique": 35561,
            "p01": 17849970.0,
            "p25": 685711500.0,
            "p50": 2180000000.0,
            "p75": 6158813750.0,
            "p99": 69713919999.99992,
            "std": 13474780754.441755
          },
          "debtIssuanceNet": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 9130,
            "mad_scale": 296960000.0,
            "max": 230811000000.0,
            "mean": 504639922.69249547,
            "min": -240790000000.0,
            "null_rate": 0.04093259194224954,
            "nulls": 2098,
            "nunique": 30550,
            "p01": -9241800000.0,
            "p25": -121000000.0,
            "p50": 0.0,
            "p75": 540000000.0,
            "p99": 16965520000.000135,
            "std": 7113112435.687316
          },
          "debtToEquity": {
            "dtype": "float64",
            "mad_center": 0.6463520039339611,
            "mad_outliers": 5198,
            "mad_scale": 0.46125783466586456,
            "max": 4787.979182729376,
            "mean": 0.987667238089612,
            "min": -1484.5,
            "null_rate": 1.95102916788606e-05,
            "nulls": 1,
            "nunique": 48426,
            "p01": -10.500940714595226,
            "p25": 0.26204134044326843,
            "p50": 0.6463520039339611,
            "p75": 1.2700559878326876,
            "p99": 13.945360119312186,
            "std": 27.52843986316776
          },
          "deferredRevenue": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 2402,
            "mad_scale": 652088037.3810824,
            "max": 75712000000.0,
            "mean": 652088037.3810824,
            "min": 0.0,
            "null_rate": 1.95102916788606e-05,
            "nulls": 1,
            "nunique": 14168,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 91323250.0,
            "p99": 15282110000.000015,
            "std": 3024535783.081969
          },
          "depAmort": {
            "dtype": "float64",
            "mad_center": 349400000.0,
            "mad_outliers": 6250,
            "mad_scale": 294600000.0,
            "max": 75200000000.0,
            "mean": 1059777621.7133385,
            "min": -3376000000.0,
            "null_rate": 0.04075699931713979,
            "nulls": 2089,
            "nunique": 30154,
            "p01": 0.0,
            "p25": 106676250.0,
            "p50": 349400000.0,
            "p75": 980000000.0,
            "p99": 12891149999.999987,
            "std": 2555298070.550167
          },
          "deposits": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 1659,
            "mad_scale": 14246683358.869728,
            "max": 2713700000000.0,
            "mean": 14246683358.869728,
            "min": 0.0,
            "null_rate": 1.95102916788606e-05,
            "nulls": 1,
            "nunique": 6717,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 346003580000.0,
            "std": 112566957282.20349
          },
          "dilutedShares": {
            "dtype": "float64",
            "mad_center": 332250000.0,
            "mad_outliers": 4083,
            "mad_scale": 219955500.0,
            "max": 26518877000.0,
            "mean": 899904735.2084858,
            "min": 1301750.0,
            "null_rate": 0.3392254414203492,
            "nulls": 17387,
            "nunique": 29937,
            "p01": 26444141.915,
            "p25": 153939812.5,
            "p50": 332250000.0,
            "p75": 766688000.0,
            "p99": 10572082500.0,
            "std": 2210432826.4569235
          },
          "dividendsPaid": {
            "dtype": "float64",
            "mad_center": -189000000.0,
            "mad_outliers": 7355,
            "mad_scale": 189000000.0,
            "max": 2342000000.0,
            "mean": -741726851.877028,
            "min": -36112000000.0,
            "null_rate": 0.041556921275973074,
            "nulls": 2130,
            "nunique": 21052,
            "p01": -9039520000.0,
            "p25": -642402000.0,
            "p50": -189000000.0,
            "p75": -1459000.0,
            "p99": 0.0,
            "std": 1713130215.4224722
          },
          "dividendsPerShare": {
            "dtype": "float64",
            "mad_center": 1.15,
            "mad_outliers": 1256,
            "mad_scale": 0.7699999999999999,
            "max": 105.49,
            "mean": 1.7033941492316238,
            "min": 0.004,
            "null_rate": 0.4731245732123695,
            "nulls": 24250,
            "nunique": 4622,
            "p01": 0.02204000000000002,
            "p25": 0.5,
            "p50": 1.15,
            "p75": 2.133,
            "p99": 10.040000000000001,
            "std": 2.3549568452262406
          },
          "ebitda": {
            "dtype": "float64",
            "mad_center": 1461000000.0,
            "mad_outliers": 6025,
            "mad_scale": 1177070000.0,
            "max": 201266000000.0,
            "mean": 3862414427.7450104,
            "min": -90836000000.0,
            "null_rate": 0.0410496536923227,
            "nulls": 2104,
            "nunique": 39143,
            "p01": -1103500000.0,
            "p25": 512379500.0,
            "p50": 1461000000.0,
            "p75": 3710500000.0,
            "p99": 40310000000.0,
            "std": 9009115392.024591
          },
          "effectiveTaxRate": {
            "dtype": "float64",
            "mad_center": 0.2573628258204853,
            "mad_outliers": 2368,
            "mad_scale": 0.09909635396278936,
            "max": 652.0,
            "mean": 0.2603110342806175,
            "min": -349.0,
            "null_rate": 0.038688908399180565,
            "nulls": 1983,
            "nunique": 47136,
            "p01": -1.217929442717053,
            "p25": 0.15223254074419013,
            "p50": 0.2573628258204853,
            "p75": 0.3538169734411494,
            "p99": 1.2098790424929249,
            "std": 4.917166470005534
          },
          "employees_sec": {
            "dtype": "float64",
            "mad_center": 15000.0,
            "mad_outliers": 5221,
            "mad_scale": 12363.0,
            "max": 3400000.0,
            "mean": 46266.86807775601,
            "min": 100.0,
            "null_rate": 0.24323480636035508,
            "nulls": 12467,
            "nunique": 4306,
            "p01": 199.0,
            "p25": 5000.0,
            "p50": 15000.0,
            "p75": 44000.0,
            "p99": 400000.0,
            "std": 138233.55518987105
          },
          "epsDiluted": {
            "dtype": "float64",
            "mad_center": 2.034136367188589,
            "mad_outliers": 2719,
            "mad_scale": 1.5710929827640132,
            "max": 506.98658628485305,
            "mean": 3.736326373179222,
            "min": -755.0494296577947,
            "null_rate": 0.3392254414203492,
            "nulls": 17387,
            "nunique": 33852,
            "p01": -6.424707065419223,
            "p25": 0.7194519058741558,
            "p50": 2.034136367188589,
            "p75": 4.248448925496725,
            "p99": 34.34826051982026,
            "std": 15.675195286318509
          },
          "equityIssuanceNet": {
            "dtype": "float64",
            "mad_center": -86000000.0,
            "mad_outliers": 10529,
            "mad_scale": 181900000.0,
            "max": 24277000000.0,
            "mean": -810265346.0829519,
            "min": -101109000000.0,
            "null_rate": 0.04085455077553409,
            "nulls": 2094,
            "nunique": 30679,
            "p01": -11511200000.0,
            "p25": -675000000.0,
            "p50": -86000000.0,
            "p75": 10724000.0,
            "p99": 1844000000.0,
            "std": 3517076184.6078534
          },
          "exchangeRateEffect": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 16185,
            "mad_scale": 1559000.0,
            "max": 19140000000.0,
            "mean": -12750132.305539692,
            "min": -34292000000.0,
            "null_rate": 0.043449419568822555,
            "nulls": 2227,
            "nunique": 12348,
            "p01": -423838000.0,
            "p25": -3383250.0,
            "p50": 0.0,
            "p75": 300000.0,
            "p99": 244000000.0,
            "std": 370989105.03110284
          },
          "financeLeaseLiability_sec": {
            "dtype": "float64",
            "mad_center": 56100000.0,
            "mad_outliers": 65,
            "mad_scale": 41100000.0,
            "max": 28434000000.0,
            "mean": 945760838.3045526,
            "min": 0.0,
            "null_rate": 0.9875719442005658,
            "nulls": 50618,
            "nunique": 419,
            "p01": 310800.0,
            "p25": 20000000.0,
            "p50": 56100000.0,
            "p75": 106495000.0,
            "p99": 25563439999.999996,
            "std": 4124984098.250217
          },
          "financingCashFlow": {
            "dtype": "float64",
            "mad_center": -226620000.0,
            "mad_outliers": 8722,
            "mad_scale": 563968000.0,
            "max": 596645000000.0,
            "mean": -277074372.4236285,
            "min": -199568000000.0,
            "null_rate": 0.04085455077553409,
            "nulls": 2094,
            "nunique": 38228,
            "p01": -20791400000.0,
            "p25": -1182597000.0,
            "p50": -226620000.0,
            "p75": 96000000.0,
            "p99": 28129800000.0002,
            "std": 12763179200.294155
          },
          "fiscal_end": {
            "dtype": "object",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2337
          },
          "freeCashflow": {
            "dtype": "float64",
            "mad_center": 609036000.0,
            "mad_outliers": 7736,
            "mad_scale": 598937000.0,
            "max": 212806000000.0,
            "mean": 1960042015.0898864,
            "min": -312070000000.0,
            "null_rate": 0.04083504048385524,
            "nulls": 2093,
            "nunique": 37552,
            "p01": -4399560000.0,
            "p25": 123006750.0,
            "p50": 609036000.0,
            "p75": 1863000000.0,
            "p99": 24735780000.0,
            "std": 7116496025.058163
          },
          "goodwill_sec": {
            "dtype": "float64",
            "mad_center": 2502500000.0,
            "mad_outliers": 793,
            "mad_scale": 1998008000.0,
            "max": 97873000000.0,
            "mean": 6997566734.344987,
            "min": 0.0,
            "null_rate": 0.8984294215198517,
            "nulls": 46049,
            "nunique": 4020,
            "p01": 21939000.0,
            "p25": 915050000.0,
            "p50": 2502500000.0,
            "p75": 6862500000.0,
            "p99": 69022000000.0,
            "std": 12375724442.193584
          },
          "grossMargins": {
            "dtype": "float64",
            "mad_center": 0.4344422820193369,
            "mad_outliers": 62,
            "mad_scale": 0.1716281858154755,
            "max": 12.333505247036847,
            "mean": 0.47804892401424515,
            "min": -139.86091370558376,
            "null_rate": 0.03864988781582285,
            "nulls": 1981,
            "nunique": 45343,
            "p01": 0.04475451258661042,
            "p25": 0.2931554324656097,
            "p50": 0.4344422820193369,
            "p75": 0.6577524349996732,
            "p99": 1.0,
            "std": 0.727234804723718
          },
          "grossProfit": {
            "dtype": "float64",
            "mad_center": 2915263000.0,
            "mad_outliers": 6541,
            "mad_scale": 2196877000.0,
            "max": 393810000000.0,
            "mean": 7964604684.545961,
            "min": -52195000000.0,
            "null_rate": 0.03853282606574968,
            "nulls": 1975,
            "nunique": 42317,
            "p01": 37914270.0,
            "p25": 1146012000.0,
            "p50": 2915263000.0,
            "p75": 7094687750.0,
            "p99": 84575519999.99991,
            "std": 16863223690.753778
          },
          "incomeTaxExpense": {
            "dtype": "float64",
            "mad_center": 204513500.0,
            "mad_outliers": 6896,
            "mad_scale": 195486500.0,
            "max": 55064000000.0,
            "mean": 621470321.1526989,
            "min": -34831000000.0,
            "null_rate": 0.03853282606574968,
            "nulls": 1975,
            "nunique": 29296,
            "p01": -999000000.0,
            "p25": 44000000.0,
            "p50": 204513500.0,
            "p75": 600000000.0,
            "p99": 7591000000.0,
            "std": 1848056927.4250717
          },
          "intangiblesExGoodwill_sec": {
            "dtype": "float64",
            "mad_center": 1273000000.0,
            "mad_outliers": 420,
            "mad_scale": 1038000000.0,
            "max": 82876000000.0,
            "mean": 3715609985.5105915,
            "min": 0.0,
            "null_rate": 0.9290800897473417,
            "nulls": 47620,
            "nunique": 3342,
            "p01": 4981080.000000004,
            "p25": 366650000.0,
            "p50": 1273000000.0,
            "p75": 3305150000.0,
            "p99": 41866039999.99972,
            "std": 8116777418.7388315
          },
          "interestExpense": {
            "dtype": "float64",
            "mad_center": 156000000.0,
            "mad_outliers": 4047,
            "mad_scale": 132582000.0,
            "max": 26404000000.0,
            "mean": 384435877.6408365,
            "min": -3273000000.0,
            "null_rate": 0.2172275875524339,
            "nulls": 11134,
            "nunique": 21943,
            "p01": -836600.0000000001,
            "p25": 48608000.0,
            "p50": 156000000.0,
            "p75": 425000000.0,
            "p99": 3443600000.0000057,
            "std": 934491239.0855061
          },
          "inventory": {
            "dtype": "float64",
            "mad_center": 710824000.0,
            "mad_outliers": 5153,
            "mad_scale": 602376000.0,
            "max": 89077000000.0,
            "mean": 2173132013.634831,
            "min": 69200.0,
            "null_rate": 0.32525607257828504,
            "nulls": 16671,
            "nunique": 25088,
            "p01": 5206640.0,
            "p25": 240401000.0,
            "p50": 710824000.0,
            "p75": 2119250000.0,
            "p99": 18958679999.999992,
            "std": 4797395908.428398
          },
          "investingCashFlow": {
            "dtype": "float64",
            "mad_center": -580000000.0,
            "mad_outliers": 8917,
            "mad_scale": 547954000.0,
            "max": 198067000000.0,
            "mean": -2646575336.160182,
            "min": -365258000000.0,
            "null_rate": 0.04069846844210321,
            "nulls": 2086,
            "nunique": 37517,
            "p01": -37548840000.0,
            "p25": -2047200000.0,
            "p50": -580000000.0,
            "p75": -137370000.0,
            "p99": 4755959999.999999,
            "std": 11440268007.346918
          },
          "investmentAcquisitionsNet": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 15844,
            "mad_scale": 36000000.0,
            "max": 182052000000.0,
            "mean": -1055810279.198189,
            "min": -351492000000.0,
            "null_rate": 0.04116671544239586,
            "nulls": 2110,
            "nunique": 23204,
            "p01": -27012000000.0,
            "p25": -96000000.0,
            "p50": 0.0,
            "p75": 10064000.0,
            "p99": 6165039999.999979,
            "std": 10294601357.688362
          },
          "longTermDebt": {
            "dtype": "float64",
            "mad_center": 2592719000.0,
            "mad_outliers": 5008,
            "mad_scale": 2485767000.0,
            "max": 223232000000.0,
            "mean": 7240013068.01253,
            "min": 0.0,
            "null_rate": 0.18101648619646862,
            "nulls": 9278,
            "nunique": 32313,
            "p01": 0.0,
            "p25": 475000000.0,
            "p50": 2592719000.0,
            "p75": 7949000000.0,
            "p99": 73433720000.0,
            "std": 14353580940.356375
          },
          "longTermDebtCurrentOnly": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 51255,
            "nunique": 0
          },
          "longTermInvestments": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 1653,
            "mad_scale": 1624508123.6245134,
            "max": 207944000000.0,
            "mean": 1624508123.6245134,
            "min": 0.0,
            "null_rate": 0.17814847331967612,
            "nulls": 9131,
            "nunique": 15104,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 570050000.0,
            "p99": 28227879999.99986,
            "std": 8123467784.754668
          },
          "minorityInterest_sec": {
            "dtype": "float64",
            "mad_center": 152120500.0,
            "mad_outliers": 437,
            "mad_scale": 148120500.0,
            "max": 28252000000.0,
            "mean": 822494096.612779,
            "min": -286000000.0,
            "null_rate": 0.9493122622183202,
            "nulls": 48657,
            "nunique": 1621,
            "p01": -27000000.0,
            "p25": 36000000.0,
            "p50": 152120500.0,
            "p75": 585094250.0,
            "p99": 10809730000.000078,
            "std": 2346044763.6001925
          },
          "netCashFlow": {
            "dtype": "float64",
            "mad_center": 17000000.0,
            "mad_outliers": 9433,
            "mad_scale": 169647000.0,
            "max": 343538000000.0,
            "mean": 195429821.86259434,
            "min": -208532000000.0,
            "null_rate": 0.04085455077553409,
            "nulls": 2094,
            "nunique": 32425,
            "p01": -6292060000.0,
            "p25": -99000000.0,
            "p50": 17000000.0,
            "p75": 256000000.0,
            "p99": 8625400000.00001,
            "std": 6099478984.636219
          },
          "netIncome": {
            "dtype": "float64",
            "mad_center": 639000000.0,
            "mad_outliers": 7303,
            "mad_scale": 566425000.0,
            "max": 244205000000.0,
            "mean": 1955448328.2439876,
            "min": -99289000000.0,
            "null_rate": 0.03866939810750171,
            "nulls": 1982,
            "nunique": 36612,
            "p01": -2883028919.9999995,
            "p25": 183500000.0,
            "p50": 639000000.0,
            "p75": 1865000000.0,
            "p99": 21263559999.999996,
            "std": 5822320297.919473
          },
          "netIncomeCommon": {
            "dtype": "float64",
            "mad_center": 617248000.0,
            "mad_outliers": 7311,
            "mad_scale": 550852000.0,
            "max": 244119000000.0,
            "mean": 1903489892.6817527,
            "min": -99289000000.0,
            "null_rate": 0.03866939810750171,
            "nulls": 1982,
            "nunique": 36539,
            "p01": -2864623560.0,
            "p25": 175108000.0,
            "p50": 617248000.0,
            "p75": 1808000000.0,
            "p99": 20592399999.999992,
            "std": 5767059529.748942
          },
          "netIncomeDiscontinued": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 1442,
            "mad_scale": 37149581.24904106,
            "max": 8329000000.0,
            "mean": -13593268.373023765,
            "min": -21827000000.0,
            "null_rate": 0.03866939810750171,
            "nulls": 1982,
            "nunique": 4154,
            "p01": -516839999.99999994,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 230279999.99999884,
            "std": 368279088.279864
          },
          "netIncomeToNci": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 1958,
            "mad_scale": 45214384.985204875,
            "max": 12050000000.0,
            "mean": 36088446.414872244,
            "min": -6096000000.0,
            "null_rate": 0.03866939810750171,
            "nulls": 1982,
            "nunique": 6565,
            "p01": -86770040.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 1000000.0,
            "p99": 801839999.9999965,
            "std": 301513634.7487841
          },
          "netIncome_q": {
            "dtype": "float64",
            "mad_center": 158100000.0,
            "mad_outliers": 7983,
            "mad_scale": 143369000.0,
            "max": 112193000000.0,
            "mean": 496002583.5062533,
            "min": -61659000000.0,
            "null_rate": 0.0062823139205931125,
            "nulls": 322,
            "nunique": 30521,
            "p01": -1057360000.0,
            "p25": 41000000.0,
            "p50": 158100000.0,
            "p75": 471000000.0,
            "p99": 5596480000.000003,
            "std": 1750566739.971981
          },
          "netInterestIncome_sec": {
            "dtype": "float64",
            "mad_center": 2928000000.0,
            "mad_outliers": 131,
            "mad_scale": 3148100000.0,
            "max": 63471000000.0,
            "mean": 12074161564.846416,
            "min": -5269000000.0,
            "null_rate": 0.9885669690761877,
            "nulls": 50669,
            "nunique": 558,
            "p01": -4586750000.0,
            "p25": -156075000.0,
            "p50": 2928000000.0,
            "p75": 9156500000.0,
            "p99": 58868049999.99998,
            "std": 19703680842.253555
          },
          "netInvestmentIncome_sec": {
            "dtype": "float64",
            "mad_center": 3156000000.0,
            "mad_outliers": 33,
            "mad_scale": 2183000000.0,
            "max": 27792000000.0,
            "mean": 4264249254.4987144,
            "min": 12136000.0,
            "null_rate": 0.9924104965369233,
            "nulls": 50866,
            "nunique": 374,
            "p01": 332408000.0,
            "p25": 689532000.0,
            "p50": 3156000000.0,
            "p75": 3971000000.0,
            "p99": 21001960000.000004,
            "std": 5172213684.201319
          },
          "nonCurrentAssets": {
            "dtype": "float64",
            "mad_center": 7010000000.0,
            "mad_outliers": 5198,
            "mad_scale": 6267203000.0,
            "max": 846425000000.0,
            "mean": 19619135127.172615,
            "min": 0.0,
            "null_rate": 0.178167983611355,
            "nulls": 9132,
            "nunique": 39651,
            "p01": 27290200.0,
            "p25": 1761460500.0,
            "p50": 7010000000.0,
            "p75": 21332024000.0,
            "p99": 199638219999.99982,
            "std": 38396566037.38107
          },
          "nonCurrentLiabilities": {
            "dtype": "float64",
            "mad_center": 3825500000.0,
            "mad_outliers": 5794,
            "mad_scale": 3562510500.0,
            "max": 308145000000.0,
            "mean": 11348807921.517323,
            "min": -5290000000.0,
            "null_rate": 0.18115305823822067,
            "nulls": 9285,
            "nunique": 37778,
            "p01": 614070.0,
            "p25": 820206250.0,
            "p50": 3825500000.0,
            "p75": 12391550000.0,
            "p99": 111044129999.99994,
            "std": 22045393115.70688
          },
          "noninterestIncome_sec": {
            "dtype": "float64",
            "mad_center": 27111000000.0,
            "mad_outliers": 0,
            "mad_scale": 14364000000.0,
            "max": 67493000000.0,
            "mean": 26717449477.351917,
            "min": 1387000000.0,
            "null_rate": 0.994400546288167,
            "nulls": 50968,
            "nunique": 287,
            "p01": 1404300000.0,
            "p25": 13187500000.0,
            "p50": 27111000000.0,
            "p75": 42307500000.0,
            "p99": 59195119999.999954,
            "std": 15972190864.672993
          },
          "operatingCashFlow": {
            "dtype": "float64",
            "mad_center": 1150858500.0,
            "mad_outliers": 6485,
            "mad_scale": 944126500.0,
            "max": 214811000000.0,
            "mean": 3154343964.8097515,
            "min": -310373000000.0,
            "null_rate": 0.04083504048385524,
            "nulls": 2093,
            "nunique": 38242,
            "p01": -1874560000.0,
            "p25": 394648250.0,
            "p50": 1150858500.0,
            "p75": 3040000000.0,
            "p99": 36293259999.99998,
            "std": 8872195599.899162
          },
          "operatingExpenses": {
            "dtype": "float64",
            "mad_center": 1671665500.0,
            "mad_outliers": 7346,
            "mad_scale": 1307702500.0,
            "max": 300098000000.0,
            "mean": 5162304728.495941,
            "min": -8486000000.0,
            "null_rate": 0.03853282606574968,
            "nulls": 1975,
            "nunique": 40108,
            "p01": 801980.2700000006,
            "p25": 636000000.0,
            "p50": 1671665500.0,
            "p75": 4510204500.0,
            "p99": 54573409999.999985,
            "std": 11291514942.99157
          },
          "operatingIncome": {
            "dtype": "float64",
            "mad_center": 1015481500.0,
            "mad_outliers": 6464,
            "mad_scale": 853071000.0,
            "max": 197579000000.0,
            "mean": 2802299956.05002,
            "min": -94359000000.0,
            "null_rate": 0.03853282606574968,
            "nulls": 1975,
            "nunique": 37987,
            "p01": -2455561190.0,
            "p25": 325317250.0,
            "p50": 1015481500.0,
            "p75": 2652719500.0,
            "p99": 29777519999.99999,
            "std": 7265079505.397672
          },
          "operatingLeaseLiability_sec": {
            "dtype": "float64",
            "mad_center": 457050000.0,
            "mad_outliers": 296,
            "mad_scale": 293900000.0,
            "max": 96320000000.0,
            "mean": 2183347108.866442,
            "min": 716000.0,
            "null_rate": 0.9652326602282704,
            "nulls": 49473,
            "nunique": 1556,
            "p01": 32862000.0,
            "p25": 230143000.0,
            "p50": 457050000.0,
            "p75": 1103500000.0,
            "p99": 58294360000.0,
            "std": 8432504805.164518
          },
          "operatingMargins": {
            "dtype": "float64",
            "mad_center": 0.15259179444755466,
            "mad_outliers": 1807,
            "mad_scale": 0.07543589790921748,
            "max": 13.846202227121063,
            "mean": 0.045021635787484894,
            "min": -2552.2689001692047,
            "null_rate": 0.03864988781582285,
            "nulls": 1981,
            "nunique": 49249,
            "p01": -0.6135467112420269,
            "p25": 0.0840139211739044,
            "p50": 0.15259179444755466,
            "p75": 0.2383497647518255,
            "p99": 0.6175881986388663,
            "std": 12.66905123853103
          },
          "optionOverhang": {
            "dtype": "float64",
            "mad_center": 0.00961226691974093,
            "mad_outliers": 2997,
            "mad_scale": 0.007212865654294021,
            "max": 4.317618332081142,
            "mean": 0.020896418231184367,
            "min": -0.18270287239051952,
            "null_rate": 0.3392254414203492,
            "nulls": 17387,
            "nunique": 30387,
            "p01": -0.0015454573715426598,
            "p25": 0.0037615234763509475,
            "p50": 0.00961226691974093,
            "p75": 0.020509571217788458,
            "p99": 0.18078716309451376,
            "std": 0.06487814372611224
          },
          "ppeGross_sec": {
            "dtype": "float64",
            "mad_center": 5534500000.0,
            "mad_outliers": 277,
            "mad_scale": 4549000000.0,
            "max": 534098000000.0,
            "mean": 13895965798.425398,
            "min": 6638718.0,
            "null_rate": 0.9534484440542386,
            "nulls": 48869,
            "nunique": 2339,
            "p01": 118805550.0,
            "p25": 1573950000.0,
            "p50": 5534500000.0,
            "p75": 13004500000.0,
            "p99": 113175005000.00006,
            "std": 25007503142.784416
          },
          "ppeNet": {
            "dtype": "float64",
            "mad_center": 1955000000.0,
            "mad_outliers": 9998,
            "mad_scale": 1766113000.0,
            "max": 538789000000.0,
            "mean": 8581864932.341819,
            "min": 99706.0,
            "null_rate": 0.03812310994049361,
            "nulls": 1954,
            "nunique": 41225,
            "p01": 10458000.0,
            "p25": 476805000.0,
            "p50": 1955000000.0,
            "p75": 8383500000.0,
            "p99": 96595000000.0,
            "std": 20301609853.624172
          },
          "preferredDividends": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 1669,
            "mad_scale": 16632067.045359528,
            "max": 9622000000.0,
            "mean": 16190708.797475291,
            "min": -1772000000.0,
            "null_rate": 0.03866939810750171,
            "nulls": 1982,
            "nunique": 2910,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 340000000.0,
            "std": 151379160.09903672
          },
          "premiumsEarned_sec": {
            "dtype": "float64",
            "mad_center": 8687983000.0,
            "mad_outliers": 15,
            "mad_scale": 7395983000.0,
            "max": 61005000000.0,
            "mean": 14903281901.486988,
            "min": 12000.0,
            "null_rate": 0.989503463076773,
            "nulls": 50717,
            "nunique": 518,
            "p01": 14000.0,
            "p25": 3146964000.0,
            "p50": 8687983000.0,
            "p75": 22733000000.0,
            "p99": 51717179999.99999,
            "std": 14421306600.509502
          },
          "pretaxIncome": {
            "dtype": "float64",
            "mad_center": 844696000.0,
            "mad_outliers": 7079,
            "mad_scale": 739848000.0,
            "max": 299269000000.0,
            "mean": 2540783320.5157185,
            "min": -107663000000.0,
            "null_rate": 0.03866939810750171,
            "nulls": 1982,
            "nunique": 37917,
            "p01": -3095519999.9999995,
            "p25": 255370000.0,
            "p50": 844696000.0,
            "p75": 2403000000.0,
            "p99": 27101679999.999992,
            "std": 7277605818.91421
          },
          "profitMargins": {
            "dtype": "float64",
            "mad_center": 0.09810597585532652,
            "mad_outliers": 2697,
            "mad_scale": 0.05676820323178082,
            "max": 14.123242124966135,
            "mean": -0.03704329588909845,
            "min": -2416.2810152284264,
            "null_rate": 0.03878645985757487,
            "nulls": 1988,
            "nunique": 49239,
            "p01": -0.5965243827648521,
            "p25": 0.04871727635083374,
            "p50": 0.09810597585532652,
            "p75": 0.1694472614383349,
            "p99": 0.5476272486030208,
            "std": 15.147913662786937
          },
          "realizedInvestmentGains_sec": {
            "dtype": "float64",
            "mad_center": 42500000.0,
            "mad_outliers": 20,
            "mad_scale": 359500000.0,
            "max": 5703000000.0,
            "mean": -57745602.56410257,
            "min": -12717000000.0,
            "null_rate": 0.9954345917471467,
            "nulls": 51021,
            "nunique": 212,
            "p01": -4682370000.0,
            "p25": -374250000.0,
            "p50": 42500000.0,
            "p75": 357750000.0,
            "p99": 2951769999.9999948,
            "std": 1463002752.296834
          },
          "regime_sec": {
            "dtype": "str",
            "null_rate": 0.8842259291776412,
            "nulls": 45321,
            "nunique": 6
          },
          "rentalIncome_sec": {
            "dtype": "float64",
            "mad_center": 1085500000.0,
            "mad_outliers": 27,
            "mad_scale": 952000000.0,
            "max": 11106800000.0,
            "mean": 2228652264.1509433,
            "min": 46000000.0,
            "null_rate": 0.9958638181640815,
            "nulls": 51043,
            "nunique": 188,
            "p01": 47000000.0,
            "p25": 468500000.0,
            "p50": 1085500000.0,
            "p75": 2799861000.0,
            "p99": 10474185999.999998,
            "std": 2876233905.670229
          },
          "researchAndDevelopment": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 2113,
            "mad_scale": 519679175.62395227,
            "max": 121086000000.0,
            "mean": 519667630.70840013,
            "min": -66400000.0,
            "null_rate": 0.03866939810750171,
            "nulls": 1982,
            "nunique": 12415,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 97242000.0,
            "p99": 8981279999.99997,
            "std": 2838417834.0011945
          },
          "restrictedCash": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 51255,
            "nunique": 0
          },
          "retainedEarnings": {
            "dtype": "float64",
            "mad_center": 2363700000.0,
            "mad_outliers": 8037,
            "mad_scale": 2544520000.0,
            "max": 493371000000.0,
            "mean": 9697350159.129837,
            "min": -34076000000.0,
            "null_rate": 0.03814262023217247,
            "nulls": 1955,
            "nunique": 44503,
            "p01": -10523049840.0,
            "p25": 288907000.0,
            "p50": 2363700000.0,
            "p75": 8764250000.0,
            "p99": 128345600000.00012,
            "std": 27504116206.695934
          },
          "returnOnEquity": {
            "dtype": "float64",
            "mad_center": 0.14490130715335073,
            "mad_outliers": 5375,
            "mad_scale": 0.07398184983476083,
            "max": 678.1010023130301,
            "mean": 0.1413188201202571,
            "min": -306.5742574257426,
            "null_rate": 0.038688908399180565,
            "nulls": 1983,
            "nunique": 49238,
            "p01": -2.214531850185029,
            "p25": 0.080911133729232,
            "p50": 0.14490130715335073,
            "p75": 0.23445217571735302,
            "p99": 2.213108599481596,
            "std": 5.433252628894158
          },
          "revenue_q": {
            "dtype": "float64",
            "mad_center": 1834000000.0,
            "mad_outliers": 6544,
            "mad_scale": 1449000000.0,
            "max": 213386000000.0,
            "mean": 5152086866.070168,
            "min": -23758000000.0,
            "null_rate": 0.006243293337235392,
            "nulls": 320,
            "nunique": 40814,
            "p01": 17708171.52,
            "p25": 650750000.0,
            "p50": 1834000000.0,
            "p75": 4625000000.0,
            "p99": 53637459999.999886,
            "std": 11337833798.608269
          },
          "sellingGeneralAdmin": {
            "dtype": "float64",
            "mad_center": 1001924000.0,
            "mad_outliers": 7069,
            "mad_scale": 917593000.0,
            "max": 153377000000.0,
            "mean": 3511614801.3303027,
            "min": -4247000000.0,
            "null_rate": 0.03866939810750171,
            "nulls": 1982,
            "nunique": 36608,
            "p01": 0.0,
            "p25": 263000000.0,
            "p50": 1001924000.0,
            "p75": 3128000000.0,
            "p99": 40163679999.99988,
            "std": 8174099357.5021515
          },
          "sharesOutstanding": {
            "dtype": "float64",
            "mad_center": 324361773.0,
            "mad_outliers": 6032,
            "mad_scale": 212115267.0,
            "max": 26339376000.0,
            "mean": 815467461.5484928,
            "min": 408397.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 49846,
            "p01": 27556944.0,
            "p25": 152420308.5,
            "p50": 324361773.0,
            "p75": 707468439.0,
            "p99": 9664357056.839994,
            "std": 1926769635.6900592
          },
          "sharesOutstandingPit": {
            "dtype": "float64",
            "mad_center": 236562906.0,
            "mad_outliers": 6164,
            "mad_scale": 162989986.8888889,
            "max": 29206440560.0,
            "mean": 576701494.8398205,
            "min": 1318004.7481203007,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 49857,
            "p01": 10507954.74,
            "p25": 106603893.5,
            "p50": 236562906.0,
            "p75": 532770259.0,
            "p99": 6081073456.819984,
            "std": 1216927142.3773706
          },
          "shortTermBorrowingsOnly": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 51255,
            "nunique": 0
          },
          "shortTermDebt": {
            "dtype": "float64",
            "mad_center": 188953000.0,
            "mad_outliers": 8854,
            "mad_scale": 188953000.0,
            "max": 167736700000.0,
            "mean": 1125218849.5966368,
            "min": 0.0,
            "null_rate": 0.17859721002828993,
            "nulls": 9154,
            "nunique": 20938,
            "p01": 0.0,
            "p25": 7949000.0,
            "p50": 188953000.0,
            "p75": 916000000.0,
            "p99": 12710000000.0,
            "std": 4309211128.76807
          },
          "shortTermInvestments": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 1487,
            "mad_scale": 1277435991.2986422,
            "max": 186563000000.0,
            "mean": 1277435991.2986422,
            "min": 0.0,
            "null_rate": 0.17814847331967612,
            "nulls": 9131,
            "nunique": 13395,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 211352000.0,
            "p99": 29411569999.99987,
            "std": 7188883930.480018
          },
          "stockBasedComp": {
            "dtype": "float64",
            "mad_center": 22600000.0,
            "mad_outliers": 8805,
            "mad_scale": 22600000.0,
            "max": 28147000000.0,
            "mean": 163552108.95693302,
            "min": -1858000000.0,
            "null_rate": 0.040952102233928395,
            "nulls": 2099,
            "nunique": 17782,
            "p01": -11324050.0,
            "p25": 0.0,
            "p50": 22600000.0,
            "p75": 91529250.0,
            "p99": 2196449999.999997,
            "std": 864915831.2294098
          },
          "stockholdersEquity": {
            "dtype": "float64",
            "mad_center": 4272000000.0,
            "mad_outliers": 6120,
            "mad_scale": 3559795500.0,
            "max": 640480000000.0,
            "mean": 11824477065.45852,
            "min": -30996997000.0,
            "null_rate": 1.95102916788606e-05,
            "nulls": 1,
            "nunique": 46028,
            "p01": -2757880000.0,
            "p25": 1329422750.0,
            "p50": 4272000000.0,
            "p75": 11080750000.0,
            "p99": 134669970000.00041,
            "std": 26741594872.88386
          },
          "stockholdersEquityInclNci": {
            "dtype": "float64",
            "mad_center": 8804000000.0,
            "mad_outliers": 340,
            "mad_scale": 5508327000.0,
            "max": 233021000000.0,
            "mean": 19730816346.420322,
            "min": -23562000000.0,
            "null_rate": 0.9493122622183202,
            "nulls": 48657,
            "nunique": 2554,
            "p01": -3499570000.0000005,
            "p25": 4501750000.0,
            "p50": 8804000000.0,
            "p75": 18663500000.0,
            "p99": 209022750000.00003,
            "std": 35003548931.456474
          },
          "taxAssets": {
            "dtype": "float64",
            "mad_center": 1778000.0,
            "mad_outliers": 23714,
            "mad_scale": 1778000.0,
            "max": 37351000000.0,
            "mean": 372135958.63881844,
            "min": 0.0,
            "null_rate": 1.95102916788606e-05,
            "nulls": 1,
            "nunique": 17237,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 1778000.0,
            "p75": 154100000.0,
            "p99": 6037230000.0000105,
            "std": 1523135768.9796786
          },
          "taxLiabilities": {
            "dtype": "float64",
            "mad_center": 122000000.0,
            "mad_outliers": 14560,
            "mad_scale": 122000000.0,
            "max": 67084000000.0,
            "mean": 1442464305.4605105,
            "min": -705000000.0,
            "null_rate": 1.95102916788606e-05,
            "nulls": 1,
            "nunique": 25693,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 122000000.0,
            "p75": 991350000.0,
            "p99": 20079319000.00003,
            "std": 4161109067.928323
          },
          "ticker": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 489
          },
          "totalAssets": {
            "dtype": "float64",
            "mad_center": 13683104500.0,
            "mad_outliers": 7172,
            "mad_scale": 11551958000.0,
            "max": 5015069000000.0,
            "mean": 58598543875.66828,
            "min": 1000.0,
            "null_rate": 1.95102916788606e-05,
            "nulls": 1,
            "nunique": 49199,
            "p01": 127027660.0,
            "p25": 3991750000.0,
            "p50": 13683104500.0,
            "p75": 37231750000.0,
            "p99": 860031970000.0002,
            "std": 213678370229.84927
          },
          "totalDebt": {
            "dtype": "float64",
            "mad_center": 3398800000.0,
            "mad_outliers": 6501,
            "mad_scale": 3171074000.0,
            "max": 1237871000000.0,
            "mean": 14830249528.977972,
            "min": 0.0,
            "null_rate": 1.95102916788606e-05,
            "nulls": 1,
            "nunique": 41170,
            "p01": 0.0,
            "p25": 746035250.0,
            "p50": 3398800000.0,
            "p75": 9931500000.0,
            "p99": 270722200000.0003,
            "std": 57164964870.03927
          },
          "totalInvestments": {
            "dtype": "float64",
            "mad_center": 275645000.0,
            "mad_outliers": 14413,
            "mad_scale": 275645000.0,
            "max": 4191534000000.0,
            "mean": 27053169020.47198,
            "min": 0.0,
            "null_rate": 1.95102916788606e-05,
            "nulls": 1,
            "nunique": 28415,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 275645000.0,
            "p75": 2336899000.0,
            "p99": 627483310000.0015,
            "std": 156643201601.6555
          },
          "totalLiabilities": {
            "dtype": "float64",
            "mad_center": 8081000000.0,
            "mad_outliers": 7767,
            "mad_scale": 7170963000.0,
            "max": 4640471000000.0,
            "mean": 46403760525.606606,
            "min": -3296500000.0,
            "null_rate": 1.95102916788606e-05,
            "nulls": 1,
            "nunique": 48335,
            "p01": 25366877.119999997,
            "p25": 2017079750.0,
            "p50": 8081000000.0,
            "p75": 24358250000.0,
            "p99": 778254790000.0001,
            "std": 193226820320.5081
          },
          "totalRevenue": {
            "dtype": "float64",
            "mad_center": 7377353000.0,
            "mad_outliers": 6357,
            "mad_scale": 5775738000.0,
            "max": 775680000000.0,
            "mean": 20567449445.2112,
            "min": -1860474000.0,
            "null_rate": 0.03853282606574968,
            "nulls": 1975,
            "nunique": 45665,
            "p01": 90419030.05000001,
            "p25": 2682842750.0,
            "p50": 7377353000.0,
            "p75": 18457500000.0,
            "p99": 215024789999.99973,
            "std": 44659785595.44263
          }
        },
        "kind": "extract",
        "pk": [
          "ticker",
          "as_of"
        ],
        "pk_checked_cols": [
          "ticker",
          "as_of"
        ],
        "pk_checked_rows": 51255,
        "pk_complete": true,
        "pk_duplicate_rows": 0,
        "pk_missing_cols": [],
        "rows": 51255,
        "sample_date_max": "2026-08-28",
        "sample_date_min": "1995-09-01",
        "sampled_rows": 51255,
        "scope": {
          "limit": null,
          "since": null,
          "tickers": null
        },
        "table": "fundamentals_history"
      },
      "prices": {
        "columns": [
          "date",
          "open",
          "high",
          "low",
          "close_split",
          "close_total",
          "volume",
          "ticker"
        ],
        "date_col": "date",
        "date_max": "2026-09-01 00:00:00",
        "date_min": "1995-09-01 00:00:00",
        "exists": true,
        "fields": {
          "close_split": {
            "dtype": "float64",
            "mad_center": 40.97999954223633,
            "mad_outliers": 310586,
            "mad_scale": 25.619998931884766,
            "max": 9924.400390625,
            "mean": 79.51253392041949,
            "min": 0.0065100002102553844,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 446027,
            "p01": 1.1120259761810303,
            "p25": 20.43000030517578,
            "p50": 40.97999954223633,
            "p75": 82.76000213623047,
            "p99": 564.5142138671868,
            "std": 191.66080085544837
          },
          "close_total": {
            "dtype": "float64",
            "mad_center": 29.637847900390625,
            "mad_outliers": 386003,
            "mad_scale": 20.61284828186035,
            "max": 9924.400390625,
            "mean": 69.31650949874106,
            "min": 0.0065100002102553844,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2504289,
            "p01": 0.8767521977424622,
            "p25": 13.223114967346191,
            "p50": 29.637847900390625,
            "p75": 70.12639236450195,
            "p99": 548.957258300781,
            "std": 188.6144069273028
          },
          "date": {
            "dtype": "datetime64[us]",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 7800
          },
          "high": {
            "dtype": "float64",
            "mad_center": 41.459999084472656,
            "mad_outliers": 311062,
            "mad_scale": 25.87666606903076,
            "max": 9964.76953125,
            "mean": 80.45017390679156,
            "min": 0.0065100002102553844,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 438611,
            "p01": 1.1354169845581055,
            "p25": 20.719999313354492,
            "p50": 41.459999084472656,
            "p75": 83.66000366210938,
            "p99": 571.0583959960923,
            "std": 193.9722576103157
          },
          "low": {
            "dtype": "float64",
            "mad_center": 40.459999084472656,
            "mad_outliers": 310474,
            "mad_scale": 25.340003967285156,
            "max": 9794.0,
            "mean": 78.53554765345753,
            "min": 0.006184999831020832,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 435959,
            "p01": 1.091541051864624,
            "p25": 20.118769645690918,
            "p50": 40.459999084472656,
            "p75": 81.83000183105469,
            "p99": 557.1900024414062,
            "std": 189.35585595474595
          },
          "open": {
            "dtype": "float64",
            "mad_center": 40.970001220703125,
            "mad_outliers": 310786,
            "mad_scale": 25.610626220703125,
            "max": 9914.169921875,
            "mean": 79.50322711652964,
            "min": 0.0065100002102553844,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 425837,
            "p01": 1.1120259761810303,
            "p25": 20.420000076293945,
            "p50": 40.970001220703125,
            "p75": 82.75,
            "p99": 564.260009765625,
            "std": 191.6613748438514
          },
          "ticker": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 491
          },
          "volume": {
            "dtype": "float64",
            "mad_center": 2071000.0,
            "mad_outliers": 405302,
            "mad_scale": 1535300.0,
            "max": 9230856000.0,
            "mean": 8042224.222047834,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 701781,
            "p01": 35600.0,
            "p25": 847567.5,
            "p50": 2071000.0,
            "p75": 5127600.0,
            "p99": 89230097.9999995,
            "std": 45934870.255892225
          }
        },
        "kind": "extract",
        "pk": [
          "ticker",
          "date"
        ],
        "pk_checked_cols": [
          "ticker",
          "date"
        ],
        "pk_checked_rows": 3263459,
        "pk_complete": true,
        "pk_duplicate_rows": 0,
        "pk_missing_cols": [],
        "rows": 3263459,
        "sample_date_max": "2026-09-01",
        "sample_date_min": "1995-09-01",
        "sampled_rows": 3263459,
        "scope": {
          "limit": null,
          "since": null,
          "tickers": null
        },
        "table": "prices"
      },
      "prices_splits": {
        "columns": [
          "date",
          "ticker",
          "ratio"
        ],
        "date_col": "date",
        "date_max": "2026-08-11 00:00:00",
        "date_min": "1995-09-05 00:00:00",
        "exists": true,
        "fields": {
          "date": {
            "dtype": "datetime64[us]",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 765
          },
          "ratio": {
            "dtype": "float64",
            "mad_center": 2.0,
            "mad_outliers": 19,
            "mad_scale": 0.6809598883271445,
            "max": 50.0,
            "mean": 2.1877452791309,
            "min": 0.03125,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 143,
            "p01": 0.2488961038961039,
            "p25": 1.5,
            "p50": 2.0,
            "p75": 2.0,
            "p99": 10.0,
            "std": 2.860564792214163
          },
          "ticker": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 343
          }
        },
        "kind": "extract",
        "pk": [
          "ticker",
          "date"
        ],
        "pk_checked_cols": [
          "ticker",
          "date"
        ],
        "pk_checked_rows": 859,
        "pk_complete": true,
        "pk_duplicate_rows": 0,
        "pk_missing_cols": [],
        "rows": 859,
        "sample_date_max": "2026-08-11",
        "sample_date_min": "1995-09-05",
        "sampled_rows": 859,
        "scope": {
          "limit": null,
          "since": null,
          "tickers": null
        },
        "table": "prices_splits"
      }
    }
  },
  "scope": {
    "limit": null,
    "since": null,
    "tables": [
      "fundamentals_history",
      "prices",
      "prices_splits"
    ],
    "tickers": [],
    "unknown_tables": []
  },
  "session_id": "a16e81b6-02b0-4abc-ab3c-55f16f5e8314",
  "type": "DATA"
}
```
