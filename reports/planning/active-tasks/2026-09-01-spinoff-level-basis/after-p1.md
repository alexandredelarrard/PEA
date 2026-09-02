# Spinoff level-basis baseline -- `after-p1`

Generated 2026-09-02 00:05:25Z from the live `pea` database by `scripts/spinoff_level_baseline.py`.

`S(d) = PROD(prices_splits.ratio after d) / PROD(split_events(...).value after d)` -- the price adjustment Yahoo applied that the share count did not.

**Environment**: 51,847 joined filing rows / 489 tickers; 859 yfinance split rows, 9131 genuine ones.

## invariants -- raw vs S-adjusted

| invariant | rows | raw pass | S-adjusted pass | newly passing | newly FAILING |
|---|---|---|---|---|---|
| `market_cap_identity` | 50,762 | 87.44% | 98.30% | +5,516 | 3 (IVZ, OXY) |
| `price_vintage` | 51,264 | 87.33% | 98.18% | +5,568 | 3 (IVZ, OXY) |

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
| DD | 2005-05-02 | 5.037746 | 8.972 | 45.198 | 45.198 | -80.15% | 0.00% |
| DD | 2012-05-02 | 5.037746 | 7.913 | 39.866 | 39.866 | -80.15% | 0.00% |
| DD | 2018-05-04 | 5.037746 | 29.7 | 149.621 | 149.62 | -80.15% | 0.00% |
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
| DD | 123 | 119 | 1.0 | 5.037746 | **NO** |
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

5,955 of 51,847 panel rows (11.49%) across 84 of 489 tickers.

| ticker | max S |
|---|---|
| LVS | 0.003759 |
| DD | 5.037746 |
| HWM | 3.00003 |
| CBRE | 0.333333 |
| GM | 0.333333 |
| F | 2.740082 |
| NI | 2.545 |
| EBAY | 2.376 |
| ABT | 2.227663 |
| HPQ | 2.202 |
| HON | 2.12229 |
| IVZ | 2.0 |
| PSKY | 0.5 |
| DELL | 1.973 |
| MSI | 1.934712 |

## residual after S -- invariant 1's biggest remaining clusters

The plan scopes out four: MNST, V, the stock-dividend names (APA/HBAN/ORCL) and the as-of join noise. **A fifth here means the plan is wrong.**

| ticker | rows | median ratio |
|---|---|---|
| MNST | 122 | 2.0 |
| IP | 102 | 0.9339 |
| JCI | 81 | 1.3253 |
| V | 74 | 0.9368 |
| BX | 33 | 0.9814 |
| APA | 28 | 0.8658 |
| SJM | 27 | 0.945 |
| HBAN | 19 | 0.7513 |
| BLDR | 18 | 0.841 |
| CCI | 16 | 0.9699 |
| HWM | 13 | 0.7669 |
| WBD | 13 | 0.9074 |
| HAS | 11 | 1.0134 |
| ORCL | 5 | 0.6686 |
| NUE | 4 | 1.0037 |
| BMY | 4 | 0.9855 |
| INCY | 4 | 1.0111 |
| TMO | 4 | 1.0201 |
| ADM | 4 | 0.8681 |
| GEN | 4 | 0.9824 |

## the two open questions

### `dividend_leg` -- dividend_yield (yfinance ttm_ps/close_split) / (sharadar dps/price)

**LEGS CANCEL -- on the 3,090 strongly-affected rows the ratio sits 149381.1x closer to 1.0 than to S (median ratio 1.0 vs median S 1.3363). The vendor back-adjusted BOTH legs. NO CHANGE NEEDED.**

discriminator on the rows with `|S-1| > 10%`: `{'n': 3090, 'median_S': 1.3363, 'median_ratio': 1.0, 'dist_to_1': 0.0, 'dist_to_S': 0.3356}`

| cohort | n | median | p25 | p75 | within 2% of 1.0 |
|---|---|---|---|---|---|
| control | 655 | 1.0 | 1.0 | 1.0 | 95.42% |
| spinoff_cohort | 1,197 | 1.0 | 1.0 | 1.0 | 83.21% |
| affected | 4,745 | 1.0 | 1.0 | 1.0 | 84.78% |
| strongly_affected | 3,090 | 1.0 | 1.0 | 1.0 | 84.01% |
| unaffected | 33,428 | 1.0 | 1.0 | 1.0 | 93.31% |

### `earnings_leg` -- fwd/trailing EPS yield (yfinance eps / close_split) / (sharadar epsdil / price)

**LEGS DO NOT CANCEL -- on the 1,488 strongly-affected rows the ratio sits 4.8x closer to S than to 1.0 (median ratio 1.402 vs median S 1.3038). This consumer IS distorted and needs the level factor.**

discriminator on the rows with `|S-1| > 10%`: `{'n': 1488, 'median_S': 1.3038, 'median_ratio': 1.402, 'dist_to_1': 0.3726, 'dist_to_S': 0.077}`

| cohort | n | median | p25 | p75 | within 2% of 1.0 |
|---|---|---|---|---|---|
| control | 387 | 1.0 | 1.0 | 1.0676 | 55.56% |
| spinoff_cohort | 517 | 1.4751 | 1.2038 | 2.02 | 3.29% |
| affected | 2,163 | 1.288 | 1.0875 | 1.6636 | 2.27% |
| strongly_affected | 1,488 | 1.402 | 1.2036 | 1.8513 | 0.54% |
| unaffected | 20,782 | 1.0235 | 1.0 | 1.2208 | 32.77% |

## return controls -- MUST NOT MOVE

| digest | value |
|---|---|
| `prices.close_total` | `000ddd33daae0a38` |
| `ret` from `close_total` | `13acfa1a2cd80947` |
| `cube_part_prices.ret` | `c00b47a452066047` |
| `cube_part_prices.volume` | `1abf3fab2f65ad98` |
