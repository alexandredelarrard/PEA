# Phase 4 — Sharadar vs SEC gap check

Scope: **97 overlapping tickers**, **44 comparable fields**, every shared `as_of`. Flagged when |Δ|/|sec| > 3% **and** |Δ| exceeds the class floor; **systematic** when that holds on ≥ 80% of at least 4 shared dates.

⚠ This is not the validator (D25). It registers no check and writes no `fundamentals_check` row.

## 1. Override candidates — systematic, and NOT a designed-in fork

**These are the findings.** Each is a basis conflict nobody has adjudicated. An override moves the field to a source with the SEC roster's coverage; a ticker outside that roster gets NULL, not a Sharadar fallback.

| ticker | field | class | n_dates | n_flagged | median_pct_gap | min_pct_gap | max_pct_gap | median_abs_gap | inherits_from |
|---|---|---|---|---|---|---|---|---|---|
| CME | shortTermInvestments | money | 68 | 68 | 51193.81% | 6087.73% | 153313.14% | 40,864,050,000 |  |
| CME | debtToEquity | ratio | 30 | 25 | 11250.13% | 752.07% | 910945.65% | 4 | stockholdersEquity |
| CME | totalDebt | money | 30 | 25 | 11250.13% | 751.83% | 866697.50% | 95,161,100,000 |  |
| AEP | capex | money | 18 | 18 | 5397.16% | 207.85% | 32810.00% | 3,603,500,000 |  |
| CMG | sharesOutstanding | shares | 65 | 56 | 4900.00% | 0.00% | 4999980.11% | 1,379,059,185 |  |
| CMG | dilutedShares | shares | 61 | 54 | 4899.96% | 0.00% | 4902.05% | 1,397,615,267 |  |
| CMG | basicShares | shares | 61 | 54 | 4899.94% | 0.00% | 4902.12% | 1,378,443,000 |  |
| BKNG | sharesOutstanding | shares | 65 | 63 | 2400.00% | 0.00% | 2400.00% | 1,139,328,624 |  |
| BKNG | basicShares | shares | 59 | 59 | 2399.97% | 2399.33% | 2400.42% | 1,150,763,750 |  |
| BKNG | dilutedShares | shares | 59 | 59 | 2399.87% | 2389.30% | 2400.39% | 1,170,780,774 |  |
| AMT | costOfRevenue | money | 8 | 8 | 1667.21% | 1279.65% | 5381.28% | 461,346,500 |  |
| APA | costOfRevenue | money | 4 | 4 | 1424.93% | 1331.43% | 1679.41% | 4,709,000,000 |  |
| ADM | shortTermInvestments | money | 21 | 21 | 745.36% | 398.82% | 109525.00% | 3,451,000,000 |  |
| ARES | interestExpense | money | 7 | 7 | 576.49% | 502.12% | 707.64% | 411,361,000 |  |
| BKNG | sellingGeneralAdmin | money | 25 | 25 | 539.58% | 442.22% | 575.74% | 3,541,721,000 |  |
| APP | capex | money | 4 | 4 | 411.03% | 292.59% | 548.52% | 5,441,500 |  |
| BXP | debtToEquity | ratio | 11 | 11 | 374.95% | 212.42% | 560.98% | 1 | stockholdersEquity |
| CCI | costOfRevenue | money | 17 | 17 | 337.19% | 266.67% | 951.92% | 1,487,000,000 |  |
| ANET | basicShares | shares | 40 | 36 | 300.00% | 0.00% | 300.05% | 224,965,401 |  |
| ANET | dilutedShares | shares | 40 | 36 | 300.00% | 0.00% | 300.18% | 238,162,274 |  |
| ANET | sharesOutstanding | shares | 49 | 42 | 300.00% | 0.00% | 300.00% | 226,281,900 |  |
| AJG | stockBasedComp | money | 62 | 62 | 257.30% | 47.55% | 600.83% | 40,700,000 |  |
| BXP | totalDebt | money | 11 | 11 | 240.80% | 141.85% | 382.18% | 7,262,978,000 |  |
| ARES | sellingGeneralAdmin | money | 18 | 18 | 236.75% | 201.88% | 267.71% | 1,565,022,000 |  |
| BXP | capex | money | 20 | 19 | 210.58% | 0.00% | 560.89% | 465,740,000 |  |
| CMCSA | sellingGeneralAdmin | money | 4 | 4 | 189.08% | 188.48% | 190.67% | 15,001,500,000 |  |
| BEN | depAmort | money | 50 | 49 | 174.76% | 0.00% | 427.97% | 137,800,000 |  |
| ARES | returnOnEquity | ratio | 43 | 43 | 152.14% | 101.94% | 3761.87% | 0 | stockholdersEquity |
| AON | depAmort | money | 4 | 4 | 146.80% | 126.01% | 164.55% | 310,500,000 |  |
| BNY | revenue_q | money | 69 | 63 | 146.58% | 0.00% | 44894.47% | 1,953,000,000 |  |

_176 row(s) total; 30 shown._


## 2. Systematic and EXPECTED — the phase-3 basis forks, not defects

Named so they do not drown section 1. `sharadar_field_map.json` states the reason for each.

| ticker | field | class | n_dates | n_flagged | median_pct_gap | min_pct_gap | max_pct_gap | median_abs_gap | inherits_from |
|---|---|---|---|---|---|---|---|---|---|
| ARE | ppeNet | money | 43 | 43 | 77986.89% | 35981.85% | 135451.15% | 20,222,588,000 |  |
| ABNB | accountsPayable | money | 22 | 22 | 4362.57% | 2899.57% | 9144.00% | 6,559,500,000 |  |
| CBOE | accountsPayable | money | 61 | 61 | 1896.70% | 17.62% | 6832.79% | 235,400,000 |  |
| AIG | ppeNet | money | 7 | 7 | 1231.85% | 976.95% | 1440.49% | 40,973,000,000 |  |
| ABBV | shortTermDebt | money | 51 | 43 | 655.45% | 1.05% | 413500.00% | 3,756,000,000 |  |
| CCI | accountsReceivable | money | 66 | 61 | 399.83% | 0.00% | 2277.00% | 1,327,783,500 |  |
| CMI | shortTermDebt | money | 69 | 56 | 373.91% | 0.00% | 5266.67% | 322,000,000 |  |
| CAT | accountsReceivable | money | 67 | 67 | 270.54% | 201.71% | 408.07% | 23,154,000,000 |  |
| AMT | accountsPayable | money | 68 | 56 | 249.44% | 0.00% | 456.01% | 319,700,000 |  |
| CI | accountsPayable | money | 31 | 31 | 245.12% | 193.32% | 338.24% | 16,676,000,000 |  |
| CEG | cash | money | 18 | 18 | 169.03% | 14.13% | 578.27% | 1,324,000,000 |  |
| AMAT | accountsPayable | money | 69 | 65 | 161.58% | 0.00% | 262.63% | 1,442,000,000 |  |
| ANET | cash | money | 49 | 46 | 160.38% | 0.00% | 483.29% | 1,871,353,000 |  |
| AKAM | cash | money | 67 | 67 | 141.83% | 13.53% | 318.39% | 429,932,000 |  |
| BLK | ppeNet | money | 8 | 8 | 139.66% | 132.78% | 155.58% | 1,700,500,000 |  |
| ABNB | ppeNet | money | 11 | 11 | 139.45% | 83.67% | 184.34% | 164,633,000 |  |
| CIEN | shortTermDebt | money | 17 | 17 | 137.93% | 107.05% | 274.68% | 16,138,000 |  |
| ADM | cash | money | 67 | 67 | 130.20% | 17.18% | 1777.25% | 3,857,000,000 |  |
| CAT | shortTermDebt | money | 16 | 16 | 128.59% | 84.38% | 454.04% | 6,507,000,000 |  |
| AAPL | cash | money | 69 | 69 | 127.51% | 47.64% | 345.83% | 26,287,000,000 |  |
| ADBE | cash | money | 69 | 69 | 117.17% | 3.59% | 479.21% | 1,798,045,000 |  |
| APP | ppeNet | money | 6 | 6 | 113.48% | 83.36% | 354.31% | 71,810,000 |  |
| BAX | shortTermDebt | money | 57 | 49 | 108.86% | 0.25% | 31800.00% | 406,000,000 |  |
| BA | accountsReceivable | money | 69 | 69 | 100.85% | 28.99% | 528.70% | 6,432,000,000 |  |
| ADP | shortTermDebt | money | 52 | 46 | 100.00% | 0.00% | 545440.00% | 5,300,000 |  |
| AMZN | shortTermDebt | money | 56 | 56 | 100.00% | 100.00% | 100.00% | 1,208,500,000 |  |
| BX | ppeNet | money | 27 | 27 | 95.07% | 38.37% | 290.83% | 364,982,000 |  |
| AAPL | accountsReceivable | money | 69 | 65 | 87.12% | 0.00% | 132.29% | 13,494,000,000 |  |
| AWK | accountsReceivable | money | 47 | 47 | 83.25% | 53.12% | 418.15% | 263,000,000 |  |
| AEE | accountsReceivable | money | 69 | 69 | 82.10% | 53.22% | 156.00% | 414,000,000 |  |

_90 row(s) total; 30 shown._


## 3. Not systematic — restatements, roundings, one-offs

_3334 (ticker, field) pair(s)._ A gap on 1 of 11 dates is not an override candidate.
