# Basis baseline -- `after-p1`

Generated 2026-09-02 00:04:04Z from the live `pea` database by `scripts/basis_baseline.py`.

> The merged `cube` table did not exist when this ran, so **no pre-fix model metrics exist**. `cube_mcap` below is the `daily_market_cap` FORMULA recomputed in-script, not a column read from anywhere. A future session must not hunt for a 'before' IC or Sharpe column -- there never was one.

**Environment**: 3,263,459 price rows (1995-09-01 -> 2026-09-01), 51,255 joined filing rows / 489 tickers. `cube_part_prices`: present.

## mcap error by year

| year | n | median | p05 | min | rows off >10% |
|---|---|---|---|---|---|
| 1995 | 190 | 0.9999 | 0.6044 | 0.3358 | 35 |
| 1998 | 1262 | 0.9999 | 0.6716 | 0.1985 | 180 |
| 2003 | 1436 | 1.0 | 0.718 | 0.1985 | 165 |
| 2013 | 1750 | 1.0 | 0.794 | 0.1985 | 148 |
| 2021 | 1933 | 1.0 | 1.0 | 0.4184 | 65 |
| 2026 | 1438 | 1.0 | 1.0 | 0.5244 | 7 |

## error decomposition -- `mcap_error = split_part x dividend_part`

**The residual must be 1.0000 in every year.** It is the plan's foundation.

| year | n | split_part | dividend_part | product | mcap_error | residual |
|---|---|---|---|---|---|---|
| 1995 | 190 | 1.0 | 0.9999 | 0.9999 | 0.9999 | 1.0 |
| 1998 | 1262 | 1.0 | 0.9999 | 0.9999 | 0.9999 | 1.0 |
| 2003 | 1436 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 2013 | 1750 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 2021 | 1933 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 2026 | 1438 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

## split_part cohorts -- the leak test

| cohort | n | mean fwd 12m | median |
|---|---|---|---|
| `split_part == 1` | 48831 | 17.12% | 11.59% |

## dividend_part quintiles (cross-sectional)

| quintile | n | mean dividend_part | mean fwd 12m |
|---|---|---|---|
| Q1 | 9097 | 0.8906 | 14.97% |
| Q2 | 7931 | 0.9959 | 16.90% |
| Q3 | 8280 | 0.9998 | 15.72% |
| Q4 | 7573 | 1.0006 | 15.50% |
| Q5 | 8448 | 1.0326 | 19.52% |

## combined mcap_error quintiles (the U-shape)

| quintile | n | mean fwd 12m |
|---|---|---|
| Q1 | 9086 | 14.60% |
| Q2 | 7886 | 17.13% |
| Q3 | 8064 | 16.24% |
| Q4 | 7734 | 15.58% |
| Q5 | 8559 | 19.11% |

## de-adjusted rows

0 of 51,255 rows (0.0%) across 0 of 489 tickers. DOWN (forward splits, mcap understated 4-500x): 0 rows / 0 tickers. UP (reverse splits, mcap OVERstated 8-20x): 0 rows / 0 tickers.

## SEC cover-page agreement

Column `sharesOutstandingPit`: 5,412 of 5,553 rows agree within +/-3%; 126 too high, 15 too low. **19 of 96 tickers fail.**

| ticker | bad rows | median ratio merged/SEC |
|---|---|---|
| ACGL | 1 | 1.0 |
| AEP | 3 | 1.0 |
| AIZ | 1 | 1.0 |
| AJG | 2 | 1.0 |
| AMD | 1 | 1.0 |
| ALL | 1 | 1.0 |
| CB | 2 | 1.0 |
| AXON | 1 | 1.0 |
| CAT | 3 | 1.0 |
| BX | 27 | 1.0 |
| CMG | 1 | 1.0 |
| CLX | 1 | 1.0 |
| APP | 2 | 1.0001 |
| ARES | 2 | 1.0001 |
| ALLE | 17 | 1.0132 |
| ACN | 2 | 1.1216 |
| AOS | 5 | 1.1723 |
| CCL | 67 | 1.3528 |
| CMCSA | 2 | 1.3799 |

## spike-and-revert scan

5 events, 3 post-2020 (MNST).

| ticker | date | ret | revert gap |
|---|---|---|---|
| HIG | 2008-11-03 | 57.75% | 6.30% |
| FITB | 2009-02-06 | 60.37% | 1.83% |
| MNST | 2026-07-23 | 95.59% | 0.74% |
| MNST | 2026-08-03 | 94.13% | 2.30% |
| MNST | 2026-08-07 | 91.93% | 0.55% |

## MNST 2026-07-15 -> 2026-08-15

| date | close |
|---|---|
| 2026-07-15 | 97.57 |
| 2026-07-16 | 99.94 |
| 2026-07-17 | 97.5 |
| 2026-07-20 | 47.725 |
| 2026-07-21 | 47.23 |
| 2026-07-22 | 47.835 |
| 2026-07-23 | 93.56 |
| 2026-07-24 | 93.49 |
| 2026-07-27 | 95.33 |
| 2026-07-28 | 97.74 |
| 2026-07-29 | 97.23 |
| 2026-07-30 | 97.65 |
| 2026-07-31 | 48.19 |
| 2026-08-03 | 93.55 |
| 2026-08-04 | 94.18 |
| 2026-08-05 | 94.46 |
| 2026-08-06 | 47.08 |
| 2026-08-07 | 90.36 |
| 2026-08-11 | 45.53 |
| 2026-08-12 | 45.98 |
| 2026-08-13 | 46.68 |
| 2026-08-14 | 46.82 |

## control digests

| digest | value | must be identical after |
|---|---|---|
| `macro_equity_tr_digest` | `c794c4b8e6590101` | **P1** |
| `option_overhang_digest` | `99f56a4553b6e9c0` | **P3** |
