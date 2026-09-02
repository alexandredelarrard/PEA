# Spinoff level-basis baseline -- `before`

Generated 2026-09-01 23:17:47Z from the live `pea` database by `scripts/spinoff_level_baseline.py`.

`S(d) = PROD(prices_splits.ratio after d) / PROD(split_events(...).value after d)` -- the price adjustment Yahoo applied that the share count did not.

**Environment**: 51,847 joined filing rows / 489 tickers; 859 yfinance split rows, 8236 genuine ones.

## invariants -- raw vs S-adjusted

| invariant | rows | raw pass | S-adjusted pass | newly passing | newly FAILING |
|---|---|---|---|---|---|
| `market_cap_identity` | 50,762 | 87.44% | 97.63% | +5,176 | 3 (IVZ, OXY) |
| `price_vintage` | 51,264 | 87.33% | 97.51% | +5,224 | 3 (IVZ, OXY) |

## FDX 2020-12-17 -- the landmark row

`close_split` 235.5036, Sharadar `price` 292.26, S = 1.241, shares 265,070,592.

| ours today | ours x S | Sharadar |
|---|---|---|
| $62.425bn | **$77.47bn** | $77.47bn |

## market cap vs Sharadar, spinoff cohort

| ticker | date | S | ours ($bn) | ours x S | Sharadar | err today | err fixed |
|---|---|---|---|---|---|---|---|
| FDX | 2005-03-18 | 1.241 | 23.399 | 29.038 | 29.038 | -19.42% | -0.00% |
| FDX | 2012-03-23 | 1.241 | 23.476 | 29.134 | 29.134 | -19.42% | -0.00% |
| FDX | 2018-03-21 | 1.241 | 53.62 | 66.542 | 66.542 | -19.42% | -0.00% |
| FDX | 2021-03-18 | 1.241 | 56.342 | 69.92 | 69.92 | -19.42% | -0.00% |
| GE | 2005-05-06 | 1.669297 | 227.774 | 380.222 | 380.222 | -40.09% | 0.00% |
| GE | 2012-05-04 | 1.669297 | 122.662 | 204.759 | 204.759 | -40.09% | -0.00% |
| GE | 2018-05-01 | 1.669297 | 73.102 | 122.029 | 122.029 | -40.09% | 0.00% |
| GE | 2021-04-27 | 1.605093 | 73.78 | 118.424 | 118.424 | -37.70% | -0.00% |
| DD | 2005-05-02 | 1.679232 | 8.972 | 15.066 | 45.198 | -80.15% | -66.67% |
| DD | 2012-05-02 | 1.679232 | 7.913 | 13.288 | 39.866 | -80.15% | -66.67% |
| DD | 2018-05-04 | 1.679232 | 29.7 | 49.873 | 149.62 | -80.15% | -66.67% |
| DD | 2021-05-04 | 2.39 | 17.701 | 42.305 | 42.305 | -58.16% | -0.00% |
| T | 2005-05-06 | 1.324 | 58.927 | 78.02 | 78.02 | -24.47% | -0.00% |
| T | 2012-05-04 | 1.324 | 145.512 | 192.658 | 192.658 | -24.47% | 0.00% |
| T | 2018-05-03 | 1.324 | 148.145 | 196.144 | 196.144 | -24.47% | -0.00% |
| T | 2021-05-06 | 1.324 | 174.779 | 231.407 | 231.407 | -24.47% | -0.00% |
| HPQ | 2005-06-08 | 2.202 | 29.401 | 64.74 | 64.74 | -54.59% | 0.00% |
| HPQ | 2012-06-08 | 2.202 | 19.978 | 43.992 | 43.992 | -54.59% | -0.00% |
| HPQ | 2018-06-05 | 1.0 | 37.718 | 37.718 | 37.718 | 0.00% | 0.00% |
| HPQ | 2021-06-04 | 1.0 | 36.794 | 36.794 | 36.794 | -0.00% | -0.00% |
| EXC | 2005-04-26 | 1.402 | 22.411 | 31.42 | 31.42 | -28.67% | -0.00% |
| EXC | 2012-05-10 | 1.402 | 23.675 | 33.193 | 33.193 | -28.67% | 0.00% |
| EXC | 2018-05-02 | 1.402 | 27.805 | 38.982 | 38.982 | -28.67% | 0.00% |
| EXC | 2021-05-05 | 1.402 | 30.34 | 42.536 | 42.536 | -28.67% | -0.00% |
| RTX | 2005-04-26 | 1.589 | 32.626 | 51.843 | 51.843 | -37.07% | 0.00% |
| RTX | 2012-04-30 | 1.589 | 46.824 | 74.403 | 74.403 | -37.07% | -0.00% |
| RTX | 2018-04-27 | 1.589 | 61.658 | 97.975 | 97.975 | -37.07% | -0.00% |
| RTX | 2021-04-27 | 1.0 | 125.465 | 125.465 | 125.465 | -0.00% | -0.00% |

## per-ticker S

### spinoff cohort

| ticker | rows | rows S!=1 | min S | max S | exactly 1.0 |
|---|---|---|---|---|---|
| FDX | 126 | 125 | 1.0 | 1.241 | **NO** |
| GE | 126 | 116 | 1.0 | 1.669297 | **NO** |
| DD | 123 | 119 | 1.0 | 2.39 | **NO** |
| T | 125 | 107 | 1.0 | 1.324 | **NO** |
| HPQ | 129 | 84 | 1.0 | 2.202 | **NO** |
| EXC | 126 | 107 | 1.0 | 1.402 | **NO** |
| RTX | 125 | 99 | 1.0 | 1.589 | **NO** |
| NI | 125 | 80 | 1.0 | 2.545 | **NO** |
| BAX | 122 | 78 | 1.0 | 1.841 | **NO** |
| EQT | 126 | 95 | 1.0 | 1.837 | **NO** |

### control cohort

| ticker | rows | rows S!=1 | min S | max S | exactly 1.0 |
|---|---|---|---|---|---|
| AAPL | 123 | 0 | 1.0 | 1.0 | YES |
| KO | 125 | 0 | 1.0 | 1.0 | YES |
| JNJ | 125 | 0 | 1.0 | 1.0 | YES |
| MSFT | 126 | 0 | 1.0 | 1.0 | YES |
| PG | 127 | 0 | 1.0 | 1.0 | YES |
| XOM | 125 | 0 | 1.0 | 1.0 | YES |

## how much S touches

5,847 of 51,847 panel rows (11.28%) across 83 of 489 tickers.

| ticker | max S |
|---|---|
| LVS | 0.003759 |
| MSI | 0.276393 |
| CBRE | 0.333333 |
| GM | 0.333333 |
| F | 2.740082 |
| NI | 2.545 |
| LDOS | 0.405 |
| DD | 2.39 |
| EBAY | 2.376 |
| ABT | 2.227663 |
| HPQ | 2.202 |
| HLT | 0.487329 |
| IVZ | 2.0 |
| PSKY | 0.5 |
| HWM | 0.5 |

## residual after S -- invariant 1's biggest remaining clusters

The plan scopes out four: MNST, V, the stock-dividend names (APA/HBAN/ORCL) and the as-of join noise. **A fifth here means the plan is wrong.**

| ticker | rows | median ratio |
|---|---|---|
| HON | 123 | 0.5 |
| MNST | 122 | 2.0 |
| IP | 102 | 0.9339 |
| DD | 94 | 0.3333 |
| JCI | 81 | 1.3253 |
| V | 74 | 0.9368 |
| MSI | 61 | 0.1429 |
| WBD | 37 | 1.022 |
| BX | 33 | 0.9814 |
| LDOS | 28 | 0.25 |
| APA | 28 | 0.8658 |
| SJM | 27 | 0.945 |
| HBAN | 19 | 0.7513 |
| BLDR | 18 | 0.841 |
| CCI | 16 | 0.9699 |
| HWM | 13 | 0.7669 |
| HLT | 12 | 0.3333 |
| HAS | 11 | 1.0134 |
| ORCL | 5 | 0.6686 |
| GEN | 4 | 0.9824 |

## the two open questions

### `dividend_leg` -- dividend_yield (yfinance ttm_ps/close_split) / (sharadar dps/price)

**LEGS CANCEL -- on the 2,968 strongly-affected rows the ratio sits 148405.1x closer to 1.0 than to S (median ratio 1.0 vs median S 1.323). The vendor back-adjusted BOTH legs. NO CHANGE NEEDED.**

discriminator on the rows with `|S-1| > 10%`: `{'n': 2968, 'median_S': 1.323, 'median_ratio': 1.0, 'dist_to_1': 0.0, 'dist_to_S': 0.3291}`

| cohort | n | median | p25 | p75 | within 2% of 1.0 |
|---|---|---|---|---|---|
| control | 655 | 1.0 | 1.0 | 1.0 | 95.42% |
| spinoff_cohort | 1,197 | 1.0 | 1.0 | 1.0 | 83.21% |
| affected | 4,745 | 1.0 | 1.0 | 1.0 | 84.78% |
| strongly_affected | 2,968 | 1.0 | 1.0 | 1.0 | 83.56% |
| unaffected | 33,428 | 1.0 | 1.0 | 1.0 | 93.31% |

### `earnings_leg` -- fwd/trailing EPS yield (yfinance eps / close_split) / (sharadar epsdil / price)

**LEGS DO NOT CANCEL -- on the 1,422 strongly-affected rows the ratio sits 3.5x closer to S than to 1.0 (median ratio 1.3831 vs median S 1.275). This consumer IS distorted and needs the level factor.**

discriminator on the rows with `|S-1| > 10%`: `{'n': 1422, 'median_S': 1.275, 'median_ratio': 1.3831, 'dist_to_1': 0.3466, 'dist_to_S': 0.0978}`

| cohort | n | median | p25 | p75 | within 2% of 1.0 |
|---|---|---|---|---|---|
| control | 387 | 1.0 | 1.0 | 1.0676 | 55.56% |
| spinoff_cohort | 517 | 1.4751 | 1.2038 | 2.02 | 3.29% |
| affected | 2,139 | 1.2933 | 1.0944 | 1.6692 | 1.92% |
| strongly_affected | 1,422 | 1.3831 | 1.196 | 1.7524 | 0.56% |
| unaffected | 20,806 | 1.0233 | 1.0 | 1.2206 | 32.77% |

## return controls -- MUST NOT MOVE

| digest | value |
|---|---|
| `prices.close_total` | `000ddd33daae0a38` |
| `ret` from `close_total` | `13acfa1a2cd80947` |
| `cube_part_prices.ret` | `c00b47a452066047` |
| `cube_part_prices.volume` | `1abf3fab2f65ad98` |
