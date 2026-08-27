# Phase 4 — Sharadar vs SEC gap check

Scope: **14 overlapping tickers**, **44 comparable fields**, every shared `as_of`. Flagged when |Δ|/|sec| > 3% **and** |Δ| exceeds the class floor; **systematic** when that holds on ≥ 80% of at least 4 shared dates.

⚠ This is not the validator (D25). It registers no check and writes no `fundamentals_check` row.

## 1. Override candidates — systematic, and NOT a designed-in fork

**These are the findings.** Each is a basis conflict nobody has adjudicated. An override moves the field to a source with the SEC roster's coverage; a ticker outside that roster gets NULL, not a Sharadar fallback.

| ticker   | field                | class   |   n_dates |   n_flagged | median_pct_gap   | min_pct_gap   | max_pct_gap   |   median_abs_gap | inherits_from      |
|:---------|:---------------------|:--------|----------:|------------:|:-----------------|:--------------|:--------------|-----------------:|:-------------------|
| JPM      | debtToEquity         | ratio   |        20 |          20 | 1160.70%         | 752.99%       | 1494.76%      |                2 | stockholdersEquity |
| JPM      | totalDebt            | money   |        20 |          20 | 1160.70%         | 752.99%       | 1494.76%      |  656,099,500,000 |                    |
| GS       | totalDebt            | money   |         8 |           8 | 1081.12%         | 960.02%       | 1182.78%      |  806,112,500,000 |                    |
| GS       | debtToEquity         | ratio   |         8 |           8 | 1081.12%         | 960.02%       | 1182.78%      |                7 | stockholdersEquity |
| UNH      | costOfRevenue        | money   |        17 |          17 | 613.01%          | 565.78%       | 632.71%       |  252,974,000,000 |                    |
| MCD      | depAmort             | money   |        17 |          17 | 404.06%          | 362.98%       | 431.21%       |    1,620,300,000 |                    |
| JPM      | stockBasedComp       | money   |        17 |          17 | 100.00%          | 100.00%       | 100.00%       |    3,295,000,000 |                    |
| UNH      | grossProfit          | money   |        17 |          17 | 73.66%           | 72.19%        | 79.11%        |  252,974,000,000 |                    |
| UNH      | grossMargins         | ratio   |        17 |          17 | 73.66%           | 72.19%        | 79.11%        |                1 |                    |
| GS       | capex                | money   |         8 |           8 | 48.23%           | 28.76%        | 77.14%        |    1,061,500,000 |                    |
| CSCO     | shortTermInvestments | money   |        20 |          20 | 30.10%           | 20.92%        | 42.05%        |    3,408,000,000 |                    |
| CAT      | capex                | money   |        17 |          17 | 24.43%           | 16.52%        | 41.46%        |      750,000,000 |                    |
| MSFT     | debtToEquity         | ratio   |        20 |          20 | 24.10%           | 4.81%         | 55.88%        |                0 | stockholdersEquity |
| MSFT     | totalDebt            | money   |        20 |          20 | 24.10%           | 4.81%         | 55.88%        |   19,133,000,000 |                    |
| WMT      | interestExpense      | money   |        17 |          17 | 20.37%           | 17.32%        | 22.11%        |      445,000,000 |                    |
| CAT      | totalDebt            | money   |         5 |           5 | 17.80%           | 14.71%        | 27.58%        |    6,063,000,000 |                    |
| CAT      | debtToEquity         | ratio   |         5 |           5 | 17.80%           | 14.71%        | 27.58%        |                0 | stockholdersEquity |
| MCD      | sellingGeneralAdmin  | money   |         4 |           4 | 14.88%           | 14.41%        | 15.78%        |      463,000,000 |                    |
| CAT      | freeCashflow         | money   |        17 |          17 | 9.49%            | 7.31%         | 35.94%        |      750,000,000 |                    |
| CSCO     | totalDebt            | money   |        20 |          20 | 8.60%            | 3.63%         | 12.00%        |    1,177,500,000 |                    |
| CSCO     | debtToEquity         | ratio   |        20 |          20 | 8.60%            | 3.63%         | 12.00%        |                0 | stockholdersEquity |
| AXP      | profitMargins        | ratio   |        17 |          15 | 7.86%            | 0.50%         | 9.03%         |                0 |                    |
| WMT      | debtToEquity         | ratio   |        20 |          20 | 7.78%            | 6.29%         | 11.32%        |                0 | stockholdersEquity |
| WMT      | returnOnEquity       | ratio   |        17 |          17 | 7.71%            | 6.29%         | 11.32%        |                0 | stockholdersEquity |
| AXP      | revenue_q            | money   |        20 |          17 | 7.32%            | 0.28%         | 9.10%         |    1,215,500,000 |                    |
| AXP      | totalRevenue         | money   |        17 |          15 | 7.29%            | 0.50%         | 8.28%         |    5,066,000,000 |                    |
| GS       | freeCashflow         | money   |        17 |          14 | 6.12%            | 1.49%         | 145.31%       |    2,706,000,000 |                    |
| UNH      | returnOnEquity       | ratio   |        17 |          17 | 6.05%            | 4.58%         | 6.55%         |                0 | stockholdersEquity |
| UNH      | debtToEquity         | ratio   |        20 |          17 | 5.60%            | 0.28%         | 6.55%         |                0 | stockholdersEquity |


## 2. Systematic and EXPECTED — the phase-3 basis forks, not defects

Named so they do not drown section 1. `sharadar_field_map.json` states the reason for each.

| ticker   | field              | class   |   n_dates |   n_flagged | median_pct_gap   | min_pct_gap   | max_pct_gap   |   median_abs_gap | inherits_from   |
|:---------|:-------------------|:--------|----------:|------------:|:-----------------|:--------------|:--------------|-----------------:|:----------------|
| NVDA     | cash               | money   |        19 |          19 | 369.39%          | 172.41%       | 1398.29%      |   19,218,000,000 |                 |
| BA       | accountsReceivable | money   |        20 |          20 | 350.17%          | 279.93%       | 528.70%       |   10,236,500,000 |                 |
| MSFT     | cash               | money   |        20 |          20 | 300.35%          | 78.93%        | 737.68%       |   64,744,500,000 |                 |
| CAT      | accountsReceivable | money   |        20 |          20 | 251.59%          | 214.54%       | 298.12%       |   23,294,500,000 |                 |
| CSCO     | cash               | money   |        20 |          20 | 147.38%          | 91.53%        | 235.16%       |   14,261,000,000 |                 |
| CAT      | shortTermDebt      | money   |         5 |           5 | 129.13%          | 89.34%        | 188.74%       |    6,665,000,000 |                 |
| UNH      | accountsPayable    | money   |        20 |          20 | 105.96%          | 97.96%        | 115.24%       |   32,669,500,000 |                 |
| AAPL     | accountsReceivable | money   |        20 |          20 | 97.14%           | 69.96%        | 128.11%       |   26,068,000,000 |                 |
| UNH      | accountsReceivable | money   |        20 |          20 | 94.54%           | 63.56%        | 144.50%       |   18,518,000,000 |                 |
| BA       | cash               | money   |        20 |          20 | 89.66%           | 4.89%         | 233.97%       |    6,513,500,000 |                 |
| AAPL     | cash               | money   |        20 |          20 | 75.43%           | 47.64%        | 133.71%       |   24,379,500,000 |                 |
| MCD      | ppeNet             | money   |        20 |          20 | 52.87%           | 50.77%        | 55.38%        |   13,446,500,000 |                 |
| WMT      | longTermDebt       | money   |        20 |          20 | 52.10%           | 43.12%        | 58.92%        |   18,569,500,000 |                 |
| AAPL     | shortTermDebt      | money   |        20 |          20 | 44.31%           | 16.89%        | 106.19%       |    4,496,500,000 |                 |
| MCD      | longTermDebt       | money   |        20 |          20 | 34.40%           | 31.68%        | 37.87%        |   12,937,300,000 |                 |
| MSFT     | longTermDebt       | money   |        20 |          20 | 33.04%           | 20.08%        | 53.21%        |   14,312,000,000 |                 |
| WMT      | shortTermDebt      | money   |        20 |          20 | 31.23%           | 12.53%        | 98.62%        |    2,187,000,000 |                 |
| NVDA     | ppeNet             | money   |        19 |          19 | 29.36%           | 22.80%        | 38.24%        |    1,346,000,000 |                 |
| UNH      | cash               | money   |        20 |          20 | 19.96%           | 10.09%        | 36.95%        |    7,079,500,000 |                 |
| WMT      | ppeNet             | money   |        20 |          20 | 17.90%           | 15.34%        | 19.62%        |   19,440,500,000 |                 |
| MSFT     | ppeNet             | money   |        20 |          20 | 14.46%           | 7.72%         | 18.38%        |   16,884,500,000 |                 |
| MCD      | ebitda             | money   |        17 |          17 | 13.57%           | 13.21%        | 15.95%        |    1,619,300,000 |                 |
| NVDA     | longTermDebt       | money   |        19 |          19 | 13.23%           | 6.77%         | 51.91%        |    1,119,000,000 |                 |
| WMT      | stockholdersEquity | money   |        20 |          20 | 7.21%            | 5.92%         | 10.17%        |    6,518,000,000 |                 |
| UNH      | stockholdersEquity | money   |        20 |          20 | 5.68%            | 4.07%         | 6.15%         |    5,331,500,000 |                 |


## 3. Not systematic — restatements, roundings, one-offs

_469 (ticker, field) pair(s)._ A gap on 1 of 11 dates is not an override candidate.
