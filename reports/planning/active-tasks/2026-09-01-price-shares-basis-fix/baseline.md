# Basis baseline -- `before`

Generated 2026-09-01 18:37:27Z from the live `pea` database by `scripts/basis_baseline.py`.

> The merged `cube` table did not exist when this ran, so **no pre-fix model metrics exist**. `cube_mcap` below is the `daily_market_cap` FORMULA recomputed in-script, not a column read from anywhere. A future session must not hunt for a 'before' IC or Sharpe column -- there never was one.

**Environment**: 3,263,505 price rows (1995-08-30 -> 2026-08-28), 51,255 joined filing rows / 489 tickers. `cube_part_prices`: present.

## mcap error by year

| year | n | median | p05 | min | rows off >10% |
|---|---|---|---|---|---|
| 1995 | 190 | 0.2081 | 0.0474 | 0.0075 | 189 |
| 1998 | 1262 | 0.2408 | 0.0458 | 0.0042 | 1230 |
| 2003 | 1436 | 0.4186 | 0.0951 | 0.0076 | 1337 |
| 2013 | 1750 | 0.7217 | 0.271 | 0.0232 | 1425 |
| 2021 | 1933 | 0.9105 | 0.7455 | 0.0249 | 886 |
| 2026 | 1438 | 0.9966 | 0.9798 | 0.5174 | 7 |

## error decomposition -- `mcap_error = split_part x dividend_part`

**The residual must be 1.0000 in every year.** It is the plan's foundation.

| year | n | split_part | dividend_part | product | mcap_error | residual |
|---|---|---|---|---|---|---|
| 1995 | 190 | 0.5 | 0.5054 | 0.2081 | 0.2081 | 1.0 |
| 1998 | 1262 | 0.5 | 0.5759 | 0.2408 | 0.2408 | 1.0 |
| 2003 | 1436 | 1.0 | 0.6182 | 0.4186 | 0.4186 | 1.0 |
| 2013 | 1750 | 1.0 | 0.7658 | 0.7217 | 0.7217 | 1.0 |
| 2021 | 1933 | 1.0 | 0.914 | 0.9109 | 0.9105 | 1.0 |
| 2026 | 1438 | 1.0 | 0.9966 | 0.9966 | 0.9966 | 1.0 |

## split_part cohorts -- the leak test

| cohort | n | mean fwd 12m | median |
|---|---|---|---|
| `split_part < 1` (a split WILL occur) | 10870 | 29.29% | 20.32% |
| `split_part == 1` | 37438 | 16.46% | 12.45% |
| `split_part > 1` (reverse split will occur) | 517 | 9.99% | 5.70% |

## dividend_part quintiles (cross-sectional)

| quintile | n | mean dividend_part | mean fwd 12m |
|---|---|---|---|
| Q1 | 9085 | 0.5635 | 12.54% |
| Q2 | 7860 | 0.6785 | 14.20% |
| Q3 | 7914 | 0.7568 | 16.93% |
| Q4 | 7863 | 0.847 | 21.72% |
| Q5 | 8607 | 0.9797 | 28.43% |

## combined mcap_error quintiles (the U-shape)

| quintile | n | mean fwd 12m |
|---|---|---|
| Q1 | 9085 | 20.97% |
| Q2 | 7860 | 16.18% |
| Q3 | 7915 | 16.11% |
| Q4 | 7859 | 18.50% |
| Q5 | 8610 | 21.40% |

## de-adjusted rows

11,597 of 51,255 rows (22.6%) across 290 of 489 tickers. DOWN (forward splits, mcap understated 4-500x): 10,996 rows / 277 tickers. UP (reverse splits, mcap OVERstated 8-20x): 601 rows / 15 tickers.

## SEC cover-page agreement

Column `sharesOutstanding`: 5,141 of 5,553 rows agree within +/-3%; 371 too high, 41 too low. **24 of 96 tickers fail.**

| ticker | bad rows | median ratio merged/SEC |
|---|---|---|
| AMCR | 26 | 0.2 |
| ACGL | 1 | 1.0 |
| AIZ | 1 | 1.0 |
| AEP | 3 | 1.0 |
| ALL | 1 | 1.0 |
| AJG | 2 | 1.0 |
| AXON | 1 | 1.0 |
| AMD | 1 | 1.0 |
| CLX | 1 | 1.0 |
| CB | 2 | 1.0 |
| CAT | 3 | 1.0 |
| BX | 27 | 1.0 |
| APP | 2 | 1.0001 |
| ARES | 2 | 1.0001 |
| ALLE | 17 | 1.0132 |
| ACN | 2 | 1.1216 |
| AOS | 5 | 1.1723 |
| CCL | 67 | 1.3528 |
| CMCSA | 2 | 1.3799 |
| APH | 60 | 2.0 |
| ANET | 42 | 4.0 |
| AVGO | 25 | 10.0 |
| BKNG | 63 | 25.0 |
| CMG | 56 | 50.0 |

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
