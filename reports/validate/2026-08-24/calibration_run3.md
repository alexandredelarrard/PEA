# fundamentals validation -- 2026-08-24

**run_id `3df52ae9af75`** | 54 ticker(s) | tiers 1,2,3 | fields all

11926 finding(s) | 2323 cluster(s) | 56 field family(ies)

severity: critical=7  high=7918  medium=2784  info=1217

*Nothing here gates. The nightly build of `fundamentals_facts` / `fundamentals_history` runs to completion regardless.*

## check health -- read this before the rankings

> **/!\ THE RANKINGS BELOW MAY BE INFLATED.** 1 check(s) fired ABOVE their own declared ceiling (`cross_identity`); 1 check(s) ABSTAINED -- they examined nothing, which is not a pass (`adjustment_unguarded`).
>
> A check over its ceiling has a THRESHOLD BUG until proven otherwise, and it buries every real finding under itself -- DQC_0118: *"inconsistencies reported to filers can be overwhelming as many don't represent real errors."* Clusters carried by such a check are weak evidence and should be treated as a suspected check defect first. A check that abstained examined nothing, so whatever it tests went UNCHECKED on this roster.

| check | tier | substrate | examined | queue | info | rate | ceiling | verdict |
|---|---|---|---|---|---|---|---|---|
| `adjustment_unguarded` | 1 | facts | 0 | 0 | 0 | -- | 100.0% | **ABSTAINED** -- nothing to examine, NOT a pass |
| `amendment_ledger` | 1 | facts | 54 | 0 | 1 | 0.00% | 100.0% | ok |
| `code_vocabulary` | 1 | history | 76,004 | 0 | 0 | 0.00% | 0.0% | ok |
| `column_contract` | 1 | history | 69 | 0 | 0 | 0.00% | 0.0% | ok |
| `coverage_field` | 1 | history | 3,240 | 656 | 0 | 20.25% | 25.0% | ok |
| `coverage_quarters` | 1 | history | 54 | 4 | 0 | 7.41% | 10.0% | ok |
| `coverage_universe` | 1 | history | 54 | 0 | 0 | 0.00% | 0.0% | ok |
| `cross_identity` | 1 | history | 3,267 | 254 | 0 | 7.77% | 3.0% | **THRESHOLD BUG** -- above its own declared ceiling |
| `dimensional_scope` | 1 | facts | 252,001 | 0 | 0 | 0.00% | 0.0% | ok |
| `expected_absent_drift` | 1 | history | 3,240 | 0 | 3 | 0.00% | 100.0% | ok |
| `filing_continuity` | 1 | facts | 54 | 0 | 0 | 0.00% | 10.0% | ok |
| `filing_lag` | 1 | history | 3,267 | 1 | 0 | 0.03% | 1.0% | ok |
| `grain` | 1 | history | 3,267 | 0 | 0 | 0.00% | 0.0% | ok |
| `impossible_value` | 1 | history | 196,020 | 9 | 0 | 0.00% | 1.0% | ok |
| `pit_leak` | 1 | history | 3,267 | 0 | 0 | 0.00% | 0.0% | ok |
| `register_cost` | 1 | history | 2,808 | 0 | 446 | 0.00% | 100.0% | ok |
| `register_coverage` | 1 | history | 54 | 0 | 54 | 0.00% | 100.0% | ok |
| `same_day_collapse` | 1 | facts | 3,273 | 0 | 9 | 0.00% | 100.0% | ok |
| `unexplained_null` | 1 | history | 196,020 | 0 | 0 | 0.00% | 0.0% | ok |
| `basis_step` | 2 | facts | 29,661 | 57 | 0 | 0.19% | 2.0% | ok |
| `level_outlier` | 2 | facts | 29,661 | 1566 | 0 | 5.28% | 6.0% | ok |
| `peer_ratio` | 2 | facts | 83,663 | 2018 | 0 | 2.41% | 3.0% | ok |
| `peer_ratio_abstentions` | 2 | facts | 8 | 0 | 6 | 0.00% | 100.0% | ok |
| `scale` | 2 | facts | 29,661 | 455 | 0 | 1.53% | 2.0% | ok |
| `series_shape` | 2 | facts | 5,616 | 934 | 698 | 16.63% | 18.0% | ok |
| `tag_switch_break` | 2 | facts | 29,661 | 82 | 0 | 0.28% | 2.0% | ok |
| `trend_break` | 2 | facts | 29,661 | 1553 | 0 | 5.24% | 6.0% | ok |
| `annual_footing` | 3 | facts | 11,805 | 186 | 0 | 1.58% | 2.0% | ok |
| `cross_vintage` | 3 | facts | 252,001 | 2477 | 0 | 0.98% | 6.0% | ok |
| `derived_vs_asreported` | 3 | facts | 20,910 | 0 | 0 | 0.00% | 5.0% | ok |
| `duplicate_fact` | 3 | facts | 252,001 | 0 | 0 | 0.00% | 1.0% | ok |
| `holdout_q4` | 3 | facts | 11,805 | 230 | 0 | 1.95% | 2.0% | ok |
| `leaf_vs_total` | 3 | facts | 252,001 | 30 | 0 | 0.01% | 25.0% | ok |
| `q4_footing` | 3 | facts | 11,805 | 197 | 0 | 1.67% | 2.0% | ok |
| `restatement_ledger` | 3 | facts | 1 | 0 | 0 | 0.00% | 100.0% | ok |

*`rate` is QUEUE findings / examined. `info` findings are shown but excluded from the rate -- nothing reads them as work, so they cannot bury anything.*

## delta

*No delta: no earlier run of this exact scope is on record; two runs are comparable only when their scope hash matches.*

*A run is only comparable to one of the SAME scope -- same tickers, same fields, same tiers. Differencing a 54-ticker baseline against a one-ticker re-validation would report ~11,800 findings "closed".*

## field families -- 50 with work in them

*A family spanning >= 5 tickers AND >= 30% of the roster is routed `likely-check-or-catalogue`: forty filers do not fail independently and simultaneously on one field. Both thresholds are constants in `clusters.py`.*

> **The routing hint is NOT discriminating on this roster:** 48 of 50 families are `likely-check-or-catalogue`. Read it as noise here, not as evidence. A hint that says the same thing about every family cannot tell an agent what to challenge first -- most likely a broad statistical check is touching nearly every ticker on nearly every field, which makes every family look wide. The breadth column is still worth reading directly; the label is not.

| field | score | findings | clusters | breadth | routing |
|---|---|---|---|---|---|
| `incomeTaxExpense` | 5,512 | 924 | 53 | 53 of 54 tickers | **likely-check-or-catalogue** |
| `netIncome` | 4,448 | 752 | 53 | 53 of 54 tickers | **likely-check-or-catalogue** |
| `pretaxIncome` | 4,178 | 743 | 53 | 53 of 54 tickers | **likely-check-or-catalogue** |
| `basicShares` | 3,256 | 733 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |
| `dilutedShares` | 3,153 | 713 | 53 | 53 of 54 tickers | **likely-check-or-catalogue** |
| `operatingIncome` | 2,744 | 521 | 44 | 44 of 54 tickers | **likely-check-or-catalogue** |
| `interestExpense` | 2,635 | 510 | 53 | 53 of 54 tickers | **likely-check-or-catalogue** |
| `operatingCashFlow` | 2,060 | 438 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |
| `grossProfit` | 1,856 | 337 | 33 | 33 of 54 tickers | **likely-check-or-catalogue** |
| `totalRevenue` | 1,752 | 440 | 49 | 49 of 54 tickers | **likely-check-or-catalogue** |
| `capex` | 1,690 | 322 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |
| `depAmort` | 1,448 | 348 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |
| `totalDebt` | 1,428 | 403 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |
| `totalAssets` | 1,124 | 196 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |
| `cash` | 1,072 | 307 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |
| `stockBasedComp` | 1,046 | 260 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |
| `minorityInterest` | 992 | 225 | 52 | 52 of 54 tickers | **likely-check-or-catalogue** |
| `ppeNet` | 961 | 227 | 51 | 51 of 54 tickers | **likely-check-or-catalogue** |
| `operatingLeaseLiability` | 918 | 152 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |
| `realizedInvestmentGains` | 896 | 151 | 40 | 40 of 54 tickers | **likely-check-or-catalogue** |
| `longTermDebt` | 730 | 176 | 52 | 52 of 54 tickers | **likely-check-or-catalogue** |
| `shortTermInvestments` | 720 | 167 | 45 | 45 of 54 tickers | **likely-check-or-catalogue** |
| `costOfRevenue` | 711 | 155 | 43 | 43 of 54 tickers | **likely-check-or-catalogue** |
| `netInterestIncome` | 668 | 183 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |
| `researchAndDevelopment` | 628 | 176 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |
| `sellingGeneralAdmin` | 618 | 176 | 37 | 37 of 54 tickers | **likely-check-or-catalogue** |
| `stockholdersEquity` | 618 | 162 | 53 | 53 of 54 tickers | **likely-check-or-catalogue** |
| `financeLeaseLiability` | 618 | 134 | 49 | 49 of 54 tickers | **likely-check-or-catalogue** |
| `shortTermDebt` | 456 | 129 | 50 | 50 of 54 tickers | **likely-check-or-catalogue** |
| `inventory` | 450 | 103 | 38 | 38 of 54 tickers | **likely-check-or-catalogue** |
| `retainedEarnings` | 444 | 128 | 53 | 53 of 54 tickers | **likely-check-or-catalogue** |
| `longTermDebtCurrentOnly` | 391 | 99 | 41 | 41 of 54 tickers | **likely-check-or-catalogue** |
| `shortTermBorrowingsOnly` | 383 | 96 | 46 | 46 of 54 tickers | **likely-check-or-catalogue** |
| `goodwill` | 364 | 108 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |
| `noninterestIncome` | 346 | 86 | 8 | 8 of 54 tickers | **likely-filer** |
| `accountsPayable` | 346 | 89 | 37 | 37 of 54 tickers | **likely-check-or-catalogue** |
| `restrictedCash` | 335 | 95 | 53 | 53 of 54 tickers | **likely-check-or-catalogue** |
| `intangiblesExGoodwill` | 330 | 91 | 50 | 50 of 54 tickers | **likely-check-or-catalogue** |
| `rentalIncome` | 319 | 78 | 46 | 46 of 54 tickers | **likely-check-or-catalogue** |
| `totalLiabilities` | 314 | 89 | 42 | 42 of 54 tickers | **likely-check-or-catalogue** |
| `accountsReceivable` | 304 | 86 | 37 | 37 of 54 tickers | **likely-check-or-catalogue** |
| `accumulatedDepreciation` | 289 | 94 | 50 | 50 of 54 tickers | **likely-check-or-catalogue** |
| `premiumsEarned` | 283 | 77 | 40 | 40 of 54 tickers | **likely-check-or-catalogue** |
| `ppeGross` | 259 | 91 | 49 | 49 of 54 tickers | **likely-check-or-catalogue** |
| `sharesOutstanding` | 233 | 115 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |
| `currentLiabilities` | 170 | 57 | 37 | 37 of 54 tickers | **likely-check-or-catalogue** |
| `netInvestmentIncome` | 164 | 44 | 33 | 33 of 54 tickers | **likely-check-or-catalogue** |
| `currentAssets` | 154 | 57 | 37 | 37 of 54 tickers | **likely-check-or-catalogue** |
| `` | 68 | 69 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |
| `epsDiluted` | 64 | 8 | 1 | 1 of 54 tickers | **likely-filer** |

## clusters -- top 25 of 1939 with work in them

*`score = (sum over findings of w(severity) x w(tier)) x corroboration`, with tier {1: 4, 2: 2, 3: 1}, severity {'critical': 4, 'high': 2, 'medium': 1, 'info': 0}, and each additional agreeing check worth +25% (so 3.25x at ten checks). Those weights are a POLICY, not a fact, and they are module constants in `clusters.py` meant to be retuned once somebody has read a list and disagreed -- which is where the corroboration term came from: volume alone ranked a 62-finding 2-check cluster above a 55-finding 10-check one.*

| # | cluster_id | ticker | field | score | findings | checks | worst | routing |
|---|---|---|---|---|---|---|---|---|
| 1 | `1c9a517eaa47` **<-- B's menu** | MCD | `capex` | 481 | 55 | 10 | high | likely-check-or-catalogue |
| 2 | `919b35844b54` **<-- B's menu** | BA | `incomeTaxExpense` | 390 | 39 | 10 | high | likely-check-or-catalogue |
| 3 | `f38ec1daf240` **<-- B's menu** | MCD | `dilutedShares` | 374 | 61 | 8 | high | likely-check-or-catalogue |
| 4 | `18a8dd2430be` **<-- B's menu** | MCD | `basicShares` | 368 | 60 | 8 | high | likely-check-or-catalogue |
| 5 | `2603621e89ab` **<-- B's menu** | ORCL | `totalRevenue` | 360 | 47 | 7 | high | likely-check-or-catalogue |
| 6 | `a423325138c3` | APA | `netIncome` | 360 | 43 | 7 | high | likely-check-or-catalogue |
| 7 | `2943d47a9b6b` | BA | `operatingIncome` | 358 | 45 | 8 | high | likely-check-or-catalogue |
| 8 | `02926a01f9dd` | APA | `pretaxIncome` | 338 | 47 | 6 | high | likely-check-or-catalogue |
| 9 | `557c69f38a8b` | EQIX | `grossProfit` | 336 | 56 | 3 | high | likely-check-or-catalogue |
| 10 | `9d0ebfbe91bb` | C | `interestExpense` | 335 | 39 | 7 | high | likely-check-or-catalogue |
| 11 | `b09464043fc1` | APA | `operatingIncome` | 320 | 46 | 6 | high | likely-check-or-catalogue |
| 12 | `423bbd467f07` | MET | `incomeTaxExpense` | 310 | 44 | 6 | high | likely-check-or-catalogue |
| 13 | `2474914d45d6` | SMCI | `inventory` | 306 | 53 | 3 | high | likely-check-or-catalogue |
| 14 | `3073bffe9c3f` | HCA | `minorityInterest` | 305 | 62 | 2 | high | likely-check-or-catalogue |
| 15 | `0d8f61c72b85` | TMO | `incomeTaxExpense` | 300 | 47 | 5 | high | likely-check-or-catalogue |
| 16 | `0b47bf08db6e` | AMT | `depAmort` | 298 | 48 | 4 | high | likely-check-or-catalogue |
| 17 | `b5e7c94ccf8f` | MET | `netIncome` | 280 | 42 | 7 | high | likely-check-or-catalogue |
| 18 | `929e5f23bfea` | EQIX | `capex` | 275 | 30 | 8 | high | likely-check-or-catalogue |
| 19 | `5e4107839207` | UNP | `ppeNet` | 265 | 54 | 2 | high | likely-check-or-catalogue |
| 20 | `5dcc2a60db55` | BAC | `pretaxIncome` | 265 | 38 | 7 | high | likely-check-or-catalogue |
| 21 | `3a621a4973ae` | AXP | `ppeNet` | 255 | 51 | 2 | high | likely-check-or-catalogue |
| 22 | `747880a91fe1` | META | `shortTermInvestments` | 255 | 51 | 2 | high | likely-check-or-catalogue |
| 23 | `635d90cacd19` | TMO | `grossProfit` | 255 | 51 | 2 | high | likely-check-or-catalogue |
| 24 | `62815e129657` | SPG | `dilutedShares` | 248 | 54 | 6 | high | likely-check-or-catalogue |
| 25 | `964378397ce8` | CAT | `grossProfit` | 245 | 38 | 4 | high | likely-check-or-catalogue |

*1,914 further cluster(s) carry work and are not listed. Agent B works ONE of the top 5; the rest of this table is context, and the tail is a backlog SIZE rather than a reading list. Query `fundamentals_check` by `cluster_id` for any of them, or widen with `render(clusters=None)`.*

## the packets -- top 25

*Everything needed to start, without a second query. If a packet is not enough to begin on, that is a defect in the CHECK and worth reporting on its own.*

### 1. MCD `capex` -- `1c9a517eaa47`  **<-- B's menu**

- **score 481** from 55 finding(s) across 10 check(s)
- checks agreeing: `scale`x10, `trend_break`x9, `cross_vintage`x8, `leaf_vs_total`x8, `level_outlier`x8, `basis_step`x4, `tag_switch_break`x4, `peer_ratio`x2, `register_cost`x1, `series_shape`x1
- severity: high=36, medium=18, info=1 | tier: T1=1, T2=38, T3=16
- periods: 2009-12-31..2020-03-31
- routing: **likely-check-or-catalogue** (54 of 54 tickers on this field)
- *10 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 55 times.*
- https://www.sec.gov/Archives/edgar/data/63908/000006390816000142/0000063908-16-000142-index.htm
- _the RESOLUTION ROUTE changed at this boundary and the level stepped with it -- the number now means something different, which no cross-vintage test can see_

### 2. BA `incomeTaxExpense` -- `919b35844b54`  **<-- B's menu**

- **score 390** from 39 finding(s) across 10 check(s)
- checks agreeing: `trend_break`x18, `cross_vintage`x5, `scale`x5, `level_outlier`x3, `annual_footing`x2, `holdout_q4`x2, `basis_step`x1, `peer_ratio`x1, `q4_footing`x1, `series_shape`x1
- severity: high=31, medium=8 | tier: T2=29, T3=10
- periods: 2010-12-31..2026-03-31
- routing: **likely-check-or-catalogue** (53 of 54 tickers on this field)
- *10 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 39 times.*
- https://www.sec.gov/Archives/edgar/data/12927/000001292721000040/0000012927-21-000040-index.htm
- _the RESOLUTION ROUTE changed at this boundary and the level stepped with it -- the number now means something different, which no cross-vintage test can see_

### 3. MCD `dilutedShares` -- `f38ec1daf240`  **<-- B's menu**

- **score 374** from 61 finding(s) across 8 check(s)
- checks agreeing: `annual_footing`x11, `holdout_q4`x11, `q4_footing`x11, `scale`x11, `cross_vintage`x5, `level_outlier`x5, `trend_break`x5, `peer_ratio`x2
- severity: high=45, medium=16 | tier: T2=23, T3=38
- periods: 2009-12-31..2026-06-30
- routing: **likely-check-or-catalogue** (53 of 54 tickers on this field)
- *8 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 61 times.*
- https://www.sec.gov/Archives/edgar/data/63908/000006390817000017/0000063908-17-000017-index.htm
- _the only rule that catches a value resolved to an entirely wrong concept without a human noticing first_

### 4. MCD `basicShares` -- `18a8dd2430be`  **<-- B's menu**

- **score 368** from 60 finding(s) across 8 check(s)
- checks agreeing: `annual_footing`x11, `holdout_q4`x11, `q4_footing`x11, `scale`x11, `level_outlier`x5, `trend_break`x5, `cross_vintage`x4, `peer_ratio`x2
- severity: high=44, medium=16 | tier: T2=23, T3=37
- periods: 2009-12-31..2026-06-30
- routing: **likely-check-or-catalogue** (54 of 54 tickers on this field)
- *8 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 60 times.*
- https://www.sec.gov/Archives/edgar/data/63908/000006390817000017/0000063908-17-000017-index.htm
- _the only rule that catches a value resolved to an entirely wrong concept without a human noticing first_

### 5. ORCL `totalRevenue` -- `2603621e89ab`  **<-- B's menu**

- **score 360** from 47 finding(s) across 7 check(s)
- checks agreeing: `basis_step`x10, `level_outlier`x10, `tag_switch_break`x10, `trend_break`x5, `annual_footing`x4, `holdout_q4`x4, `q4_footing`x4
- severity: high=37, medium=10 | tier: T2=35, T3=12
- periods: 2018-05-31..2022-08-31
- routing: **likely-check-or-catalogue** (49 of 54 tickers on this field)
- *7 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 47 times.*
- https://www.sec.gov/Archives/edgar/data/1341439/000156459020030125/0001564590-20-030125-index.htm
- _the RESOLUTION ROUTE changed at this boundary and the level stepped with it -- the number now means something different, which no cross-vintage test can see_

### 6. APA `netIncome` -- `a423325138c3`

- **score 360** from 43 finding(s) across 7 check(s)
- checks agreeing: `trend_break`x29, `cross_vintage`x6, `scale`x3, `level_outlier`x2, `annual_footing`x1, `holdout_q4`x1, `q4_footing`x1
- severity: high=38, medium=5 | tier: T2=34, T3=9
- periods: 2012-09-30..2024-09-30
- routing: **likely-check-or-catalogue** (53 of 54 tickers on this field)
- *7 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 43 times.*
- https://www.sec.gov/Archives/edgar/data/1841666/000173303720000008/0001733037-20-000008-index.htm
- _a flat 3x rule, because for a lumpy field the MAD is wide enough that a real 3x jump scores under 3.5_

### 7. BA `operatingIncome` -- `2943d47a9b6b`

- **score 358** from 45 finding(s) across 8 check(s)
- checks agreeing: `trend_break`x19, `level_outlier`x11, `cross_vintage`x5, `scale`x5, `holdout_q4`x2, `annual_footing`x1, `peer_ratio`x1, `q4_footing`x1
- severity: high=29, medium=16 | tier: T2=36, T3=9
- periods: 2011-12-31..2026-06-30
- routing: **likely-check-or-catalogue** (44 of 54 tickers on this field)
- *8 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 45 times.*
- https://www.sec.gov/Archives/edgar/data/12927/000001292721000094/0000012927-21-000094-index.htm
- _a flat 3x rule, because for a lumpy field the MAD is wide enough that a real 3x jump scores under 3.5_

### 8. APA `pretaxIncome` -- `02926a01f9dd`

- **score 338** from 47 finding(s) across 6 check(s)
- checks agreeing: `trend_break`x26, `cross_vintage`x11, `level_outlier`x5, `scale`x3, `basis_step`x1, `series_shape`x1
- severity: high=39, medium=8 | tier: T2=36, T3=11
- periods: 2010-12-31..2024-09-30
- routing: **likely-check-or-catalogue** (53 of 54 tickers on this field)
- *6 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 47 times.*
- https://www.sec.gov/Archives/edgar/data/1841666/000173303719000004/0001733037-19-000004-index.htm
- _the RESOLUTION ROUTE changed at this boundary and the level stepped with it -- the number now means something different, which no cross-vintage test can see_

### 9. EQIX `grossProfit` -- `557c69f38a8b`

- **score 336** from 56 finding(s) across 3 check(s)
- checks agreeing: `cross_identity`x48, `peer_ratio`x7, `series_shape`x1
- severity: high=8, medium=48 | tier: T1=48, T2=8
- periods: 2010-09-30..2026-07-29
- routing: **likely-check-or-catalogue** (33 of 54 tickers on this field)
- *3 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 56 times.*
- https://www.sec.gov/Archives/edgar/data/1101239/000119312513076572/0001193125-13-076572-index.htm
- _the only rule that catches a value resolved to an entirely wrong concept without a human noticing first_

### 10. C `interestExpense` -- `9d0ebfbe91bb`

- **score 335** from 39 finding(s) across 7 check(s)
- checks agreeing: `peer_ratio`x21, `trend_break`x6, `cross_vintage`x5, `level_outlier`x4, `basis_step`x1, `register_cost`x1, `series_shape`x1
- severity: high=34, medium=4, info=1 | tier: T1=1, T2=33, T3=5
- periods: 2010-12-31..2024-03-31
- routing: **likely-check-or-catalogue** (53 of 54 tickers on this field)
- *7 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 39 times.*
- https://www.sec.gov/Archives/edgar/data/831001/000083100123000097/0000831001-23-000097-index.htm
- _the RESOLUTION ROUTE changed at this boundary and the level stepped with it -- the number now means something different, which no cross-vintage test can see_

### 11. APA `operatingIncome` -- `b09464043fc1`

- **score 320** from 46 finding(s) across 6 check(s)
- checks agreeing: `trend_break`x24, `cross_vintage`x13, `level_outlier`x6, `holdout_q4`x1, `scale`x1, `series_shape`x1
- severity: high=39, medium=7 | tier: T2=32, T3=14
- periods: 2010-12-31..2024-09-30
- routing: **likely-check-or-catalogue** (44 of 54 tickers on this field)
- *6 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 46 times.*
- https://www.sec.gov/Archives/edgar/data/1841666/000167337916000004/0001673379-16-000004-index.htm
- _an interior gap with modal code 'insufficient_quarters' -- present before AND after, so the filer kept reporting and we stopped resolving. A MISSING TAG. Its modal code is `insufficient_quarters`, which does NOT excuse this: the TTM window is a START-of-history condition by its own rationale, and cannot open a hole in the middle of one_

### 12. MET `incomeTaxExpense` -- `423bbd467f07`

- **score 310** from 44 finding(s) across 6 check(s)
- checks agreeing: `trend_break`x23, `cross_vintage`x14, `scale`x3, `level_outlier`x2, `basis_step`x1, `series_shape`x1
- severity: high=39, medium=5 | tier: T2=30, T3=14
- periods: 2010-12-31..2024-09-30
- routing: **likely-check-or-catalogue** (53 of 54 tickers on this field)
- *6 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 44 times.*
- https://www.sec.gov/Archives/edgar/data/1099219/000109921921000177/0001099219-21-000177-index.htm
- _the RESOLUTION ROUTE changed at this boundary and the level stepped with it -- the number now means something different, which no cross-vintage test can see_

### 13. SMCI `inventory` -- `2474914d45d6`

- **score 306** from 53 finding(s) across 3 check(s)
- checks agreeing: `peer_ratio`x48, `cross_vintage`x4, `series_shape`x1
- severity: high=53 | tier: T2=49, T3=4
- periods: 2011-06-30..2026-03-31
- routing: **likely-check-or-catalogue** (38 of 54 tickers on this field)
- *3 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 53 times.*
- https://www.sec.gov/Archives/edgar/data/1375365/000119312512041019/0001193125-12-041019-index.htm
- _the only rule that catches a value resolved to an entirely wrong concept without a human noticing first_

### 14. HCA `minorityInterest` -- `3073bffe9c3f`

- **score 305** from 62 finding(s) across 2 check(s)
- checks agreeing: `peer_ratio`x61, `series_shape`x1
- severity: high=61, info=1 | tier: T2=62
- periods: 2010-12-31..2026-06-30
- routing: **likely-check-or-catalogue** (52 of 54 tickers on this field)
- https://www.sec.gov/Archives/edgar/data/860730/000119312511304797/0001193125-11-304797-index.htm
- _the only rule that catches a value resolved to an entirely wrong concept without a human noticing first_

### 15. TMO `incomeTaxExpense` -- `0d8f61c72b85`

- **score 300** from 47 finding(s) across 5 check(s)
- checks agreeing: `trend_break`x27, `scale`x8, `cross_vintage`x6, `level_outlier`x5, `series_shape`x1
- severity: high=34, medium=13 | tier: T2=41, T3=6
- periods: 2010-12-31..2023-09-30
- routing: **likely-check-or-catalogue** (53 of 54 tickers on this field)
- *5 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 47 times.*
- https://www.sec.gov/Archives/edgar/data/97745/000009774512000039/0000097745-12-000039-index.htm
- _a flat 3x rule, because for a lumpy field the MAD is wide enough that a real 3x jump scores under 3.5_

### 16. AMT `depAmort` -- `0b47bf08db6e`

- **score 298** from 48 finding(s) across 4 check(s)
- checks agreeing: `peer_ratio`x36, `level_outlier`x6, `cross_vintage`x5, `series_shape`x1
- severity: high=42, medium=6 | tier: T2=43, T3=5
- periods: 2010-09-30..2026-06-30
- routing: **likely-check-or-catalogue** (54 of 54 tickers on this field)
- *4 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 48 times.*
- https://www.sec.gov/Archives/edgar/data/1053507/000119312513418850/0001193125-13-418850-index.htm
- _the only rule that catches a value resolved to an entirely wrong concept without a human noticing first_

### 17. MET `netIncome` -- `b5e7c94ccf8f`

- **score 280** from 42 finding(s) across 7 check(s)
- checks agreeing: `cross_vintage`x17, `trend_break`x14, `holdout_q4`x3, `scale`x3, `annual_footing`x2, `level_outlier`x2, `q4_footing`x1
- severity: high=37, medium=5 | tier: T2=19, T3=23
- periods: 2010-12-31..2023-06-30
- routing: **likely-check-or-catalogue** (53 of 54 tickers on this field)
- *7 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 42 times.*
- https://www.sec.gov/Archives/edgar/data/1099219/000109921919000008/0001099219-19-000008-index.htm
- _three numbers the filer published independently, on three different bases, that must reconcile_

### 18. EQIX `capex` -- `929e5f23bfea`

- **score 275** from 30 finding(s) across 8 check(s)
- checks agreeing: `peer_ratio`x12, `basis_step`x5, `tag_switch_break`x5, `cross_vintage`x3, `level_outlier`x2, `register_cost`x1, `scale`x1, `series_shape`x1
- severity: high=25, medium=3, info=2 | tier: T1=1, T2=26, T3=3
- periods: 2011-03-31..2026-03-31
- routing: **likely-check-or-catalogue** (54 of 54 tickers on this field)
- *8 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 30 times.*
- https://www.sec.gov/Archives/edgar/data/1101239/000119312513437940/0001193125-13-437940-index.htm
- _the SOURCE CONCEPT changed at this boundary and the level stepped with it -- an era-aware basis change, which is why comparing each row's tag to a global mode does not work_

### 19. UNP `ppeNet` -- `5e4107839207`

- **score 265** from 54 finding(s) across 2 check(s)
- checks agreeing: `peer_ratio`x53, `series_shape`x1
- severity: high=53, info=1 | tier: T2=54
- periods: 2010-12-31..2024-09-30
- routing: **likely-check-or-catalogue** (51 of 54 tickers on this field)
- https://www.sec.gov/Archives/edgar/data/100885/000119312512170664/0001193125-12-170664-index.htm
- _the only rule that catches a value resolved to an entirely wrong concept without a human noticing first_

### 20. BAC `pretaxIncome` -- `5dcc2a60db55`

- **score 265** from 38 finding(s) across 7 check(s)
- checks agreeing: `cross_vintage`x14, `level_outlier`x6, `peer_ratio`x6, `trend_break`x6, `scale`x3, `tag_switch_break`x2, `basis_step`x1
- severity: high=29, medium=9 | tier: T2=24, T3=14
- periods: 2011-03-31..2025-06-30
- routing: **likely-check-or-catalogue** (53 of 54 tickers on this field)
- *7 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 38 times.*
- https://www.sec.gov/Archives/edgar/data/70858/000007085812000191/0000070858-12-000191-index.htm
- _the RESOLUTION ROUTE changed at this boundary and the level stepped with it -- the number now means something different, which no cross-vintage test can see_

### 21. AXP `ppeNet` -- `3a621a4973ae`

- **score 255** from 51 finding(s) across 2 check(s)
- checks agreeing: `peer_ratio`x50, `series_shape`x1
- severity: high=51 | tier: T2=51
- periods: 2010-12-31..2026-04-17
- routing: **likely-check-or-catalogue** (51 of 54 tickers on this field)
- https://www.sec.gov/Archives/edgar/data/4962/000119312512077400/0001193125-12-077400-index.htm
- _the only rule that catches a value resolved to an entirely wrong concept without a human noticing first_

### 22. META `shortTermInvestments` -- `747880a91fe1`

- **score 255** from 51 finding(s) across 2 check(s)
- checks agreeing: `peer_ratio`x50, `series_shape`x1
- severity: high=51 | tier: T2=51
- periods: 2011-12-31..2025-03-31
- routing: **likely-check-or-catalogue** (45 of 54 tickers on this field)
- https://www.sec.gov/Archives/edgar/data/1326801/000132680124000012/0001326801-24-000012-index.htm
- _the only rule that catches a value resolved to an entirely wrong concept without a human noticing first_

### 23. TMO `grossProfit` -- `635d90cacd19`

- **score 255** from 51 finding(s) across 2 check(s)
- checks agreeing: `cross_identity`x50, `series_shape`x1
- severity: high=1, medium=50 | tier: T1=50, T2=1
- periods: 2012-08-03..2026-07-31
- routing: **likely-check-or-catalogue** (33 of 54 tickers on this field)
- **no `edgar_url`** -- a Tier-1-only cluster reads `fundamentals_history`, which carries no accession. Resolve it manually and say so: this is Phase 7's trigger.
- _a field that goes dark mid-history is almost always a defect -- VLO's capex from 2023-07 tags neither concept undimensioned in 21 of 63 filings, and nothing else in the tier detects a shape_

### 24. SPG `dilutedShares` -- `62815e129657`

- **score 248** from 54 finding(s) across 6 check(s)
- checks agreeing: `level_outlier`x17, `q4_footing`x12, `annual_footing`x11, `holdout_q4`x11, `cross_vintage`x2, `series_shape`x1
- severity: high=37, medium=17 | tier: T2=18, T3=36
- periods: 2009-06-30..2026-06-30
- routing: **likely-check-or-catalogue** (53 of 54 tickers on this field)
- *6 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 54 times.*
- https://www.sec.gov/Archives/edgar/data/1063761/000104746914001416/0001047469-14-001416-index.htm
- _three numbers the filer published independently, on three different bases, that must reconcile_

### 25. CAT `grossProfit` -- `964378397ce8`

- **score 245** from 38 finding(s) across 4 check(s)
- checks agreeing: `cross_identity`x31, `cross_vintage`x5, `level_outlier`x1, `series_shape`x1
- severity: high=6, medium=32 | tier: T1=31, T2=2, T3=5
- periods: 2015-03-31..2026-08-05
- routing: **likely-check-or-catalogue** (33 of 54 tickers on this field)
- *4 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 38 times.*
- https://www.sec.gov/Archives/edgar/data/18230/000001823019000034/0000018230-19-000034-index.htm
- _at least one vintage was DERIVED, so our arithmetic is in play_


## `wontfix` clusters

*None on file.*

*This section is never omitted. A `wontfix` that stops being listed is a suppression, which is precisely what the deleted JSON register was drifting toward.*
