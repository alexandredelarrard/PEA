# fundamentals validation -- 2026-08-25

**run_id `725bae7bf8ed`** | 54 ticker(s) | tiers 1,2,3 | fields all

11730 finding(s) | 2334 cluster(s) | 55 field family(ies)

severity: critical=7  high=7874  medium=2630  info=1219

*Nothing here gates. The nightly build of `fundamentals_facts` / `fundamentals_history` runs to completion regardless.*

## check health -- read this before the rankings

> **/!\ THE RANKINGS BELOW MAY BE INFLATED.** 1 check(s) fired ABOVE their own declared ceiling (`coverage_field`); 1 check(s) ABSTAINED -- they examined nothing, which is not a pass (`adjustment_unguarded`).
>
> A check over its ceiling has a THRESHOLD BUG until proven otherwise, and it buries every real finding under itself -- DQC_0118: *"inconsistencies reported to filers can be overwhelming as many don't represent real errors."* Clusters carried by such a check are weak evidence and should be treated as a suspected check defect first. A check that abstained examined nothing, so whatever it tests went UNCHECKED on this roster.

| check | tier | substrate | examined | queue | info | rate | ceiling | verdict |
|---|---|---|---|---|---|---|---|---|
| `adjustment_unguarded` | 1 | facts | 0 | 0 | 0 | -- | 100.0% | **ABSTAINED** -- nothing to examine, NOT a pass |
| `amendment_ledger` | 1 | facts | 54 | 0 | 1 | 0.00% | 100.0% | ok |
| `catalogue_exclusion_cost` | 1 | facts | 2,808 | 0 | 446 | 0.00% | 100.0% | ok |
| `catalogue_override_coverage` | 1 | facts | 54 | 0 | 54 | 0.00% | 100.0% | ok |
| `code_vocabulary` | 1 | history | 75,627 | 0 | 0 | 0.00% | 0.0% | ok |
| `column_contract` | 1 | history | 69 | 0 | 0 | 0.00% | 0.0% | ok |
| `coverage_field` | 1 | facts | 2,592 | 682 | 0 | 26.31% | 25.0% | **THRESHOLD BUG** -- above its own declared ceiling |
| `coverage_quarters` | 1 | facts | 54 | 4 | 0 | 7.41% | 10.0% | ok |
| `coverage_universe` | 1 | history | 54 | 0 | 0 | 0.00% | 0.0% | ok |
| `cross_identity` | 1 | facts | 26,578 | 83 | 0 | 0.31% | 5.0% | ok |
| `dimensional_scope` | 1 | facts | 251,273 | 0 | 0 | 0.00% | 0.0% | ok |
| `expected_absent_drift` | 1 | facts | 2,592 | 0 | 4 | 0.00% | 100.0% | ok |
| `filing_continuity` | 1 | facts | 54 | 0 | 0 | 0.00% | 10.0% | ok |
| `filing_lag` | 1 | facts | 3,281 | 11 | 0 | 0.34% | 1.0% | ok |
| `grain` | 1 | history | 3,258 | 0 | 0 | 0.00% | 0.0% | ok |
| `impossible_value` | 1 | facts | 251,273 | 1 | 0 | 0.00% | 1.0% | ok |
| `pit_leak` | 1 | history | 3,258 | 0 | 0 | 0.00% | 0.0% | ok |
| `same_day_collapse` | 1 | facts | 3,264 | 0 | 9 | 0.00% | 100.0% | ok |
| `unexplained_null` | 1 | history | 195,480 | 0 | 0 | 0.00% | 0.0% | ok |
| `basis_step` | 2 | facts | 29,582 | 53 | 0 | 0.18% | 2.0% | ok |
| `level_outlier` | 2 | facts | 29,582 | 1559 | 0 | 5.27% | 6.0% | ok |
| `peer_ratio` | 2 | facts | 83,433 | 2020 | 0 | 2.42% | 3.0% | ok |
| `peer_ratio_abstentions` | 2 | facts | 8 | 0 | 6 | 0.00% | 100.0% | ok |
| `scale` | 2 | facts | 29,582 | 445 | 0 | 1.50% | 2.0% | ok |
| `series_shape` | 2 | facts | 5,616 | 933 | 699 | 16.61% | 18.0% | ok |
| `tag_switch_break` | 2 | facts | 29,582 | 78 | 0 | 0.26% | 2.0% | ok |
| `trend_break` | 2 | facts | 29,582 | 1544 | 0 | 5.22% | 6.0% | ok |
| `annual_footing` | 3 | facts | 11,781 | 184 | 0 | 1.56% | 2.0% | ok |
| `cross_vintage` | 3 | facts | 251,273 | 2469 | 0 | 0.98% | 6.0% | ok |
| `derived_vs_asreported` | 3 | facts | 20,834 | 0 | 0 | 0.00% | 5.0% | ok |
| `duplicate_fact` | 3 | facts | 251,273 | 0 | 0 | 0.00% | 1.0% | ok |
| `holdout_q4` | 3 | facts | 11,781 | 228 | 0 | 1.94% | 2.0% | ok |
| `leaf_vs_total` | 3 | facts | 251,273 | 22 | 0 | 0.01% | 25.0% | ok |
| `q4_footing` | 3 | facts | 11,781 | 195 | 0 | 1.66% | 2.0% | ok |
| `restatement_ledger` | 3 | facts | 1 | 0 | 0 | 0.00% | 100.0% | ok |

*`rate` is QUEUE findings / examined. `info` findings are shown but excluded from the rate -- nothing reads them as work, so they cannot bury anything.*

## delta vs 2026-08-24 (54 tickers)

- **1 cluster(s) SETTLED** -- a recorded fix measurably reduced them and nothing unwaived is left:
    - `1c9a517eaa47` MCD `capex` **(3 finding(s) waived across 2 check(s))** -- extraction, route 1 took PaymentsToAcquireProductiveAssets, a total the filer declares BESIDE its own PaymentsToAcquirePropertyPlantAndEquipment leg in the same calculation link, so capex was resolved to the parent rather than the leg, `2fb6ef2` (54 -> 3 queue finding(s))
    - *`(clean)` means zero findings remain. A waived count means the residue was ASSESSED and tolerated -- every one of those rows is still in `fundamentals_check`, still counted and still firing.*
- 12 cluster(s) appear for the first time in this scope

## field families -- 49 with work in them

*A family spanning >= 5 tickers AND >= 30% of the roster is routed `likely-check-or-catalogue`: forty filers do not fail independently and simultaneously on one field. Both thresholds are constants in `clusters.py`.*

> **The routing hint is NOT discriminating on this roster:** 48 of 49 families are `likely-check-or-catalogue`. Read it as noise here, not as evidence. A hint that says the same thing about every family cannot tell an agent what to challenge first -- most likely a broad statistical check is touching nearly every ticker on nearly every field, which makes every family look wide. The breadth column is still worth reading directly; the label is not.

| field | score | findings | clusters | breadth | routing |
|---|---|---|---|---|---|
| `incomeTaxExpense` | 5,500 | 920 | 53 | 53 of 54 tickers | **likely-check-or-catalogue** |
| `netIncome` | 4,475 | 755 | 53 | 53 of 54 tickers | **likely-check-or-catalogue** |
| `pretaxIncome` | 4,138 | 738 | 53 | 53 of 54 tickers | **likely-check-or-catalogue** |
| `basicShares` | 3,321 | 740 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |
| `dilutedShares` | 3,244 | 722 | 53 | 53 of 54 tickers | **likely-check-or-catalogue** |
| `operatingIncome` | 2,813 | 530 | 44 | 44 of 54 tickers | **likely-check-or-catalogue** |
| `interestExpense` | 2,638 | 507 | 53 | 53 of 54 tickers | **likely-check-or-catalogue** |
| `operatingCashFlow` | 2,064 | 439 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |
| `totalRevenue` | 1,740 | 439 | 49 | 49 of 54 tickers | **likely-check-or-catalogue** |
| `totalDebt` | 1,428 | 403 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |
| `depAmort` | 1,428 | 346 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |
| `totalAssets` | 1,124 | 196 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |
| `capex` | 1,104 | 265 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |
| `cash` | 1,072 | 307 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |
| `stockBasedComp` | 1,052 | 263 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |
| `minorityInterest` | 992 | 225 | 52 | 52 of 54 tickers | **likely-check-or-catalogue** |
| `ppeNet` | 956 | 226 | 51 | 51 of 54 tickers | **likely-check-or-catalogue** |
| `realizedInvestmentGains` | 925 | 151 | 40 | 40 of 54 tickers | **likely-check-or-catalogue** |
| `operatingLeaseLiability` | 918 | 152 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |
| `grossProfit` | 910 | 172 | 33 | 33 of 54 tickers | **likely-check-or-catalogue** |
| `netInterestIncome` | 746 | 195 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |
| `longTermDebt` | 730 | 176 | 52 | 52 of 54 tickers | **likely-check-or-catalogue** |
| `costOfRevenue` | 724 | 153 | 43 | 43 of 54 tickers | **likely-check-or-catalogue** |
| `shortTermInvestments` | 720 | 167 | 45 | 45 of 54 tickers | **likely-check-or-catalogue** |
| `stockholdersEquity` | 618 | 162 | 53 | 53 of 54 tickers | **likely-check-or-catalogue** |
| `financeLeaseLiability` | 618 | 134 | 49 | 49 of 54 tickers | **likely-check-or-catalogue** |
| `sellingGeneralAdmin` | 590 | 171 | 37 | 37 of 54 tickers | **likely-check-or-catalogue** |
| `researchAndDevelopment` | 502 | 159 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |
| `shortTermDebt` | 452 | 129 | 50 | 50 of 54 tickers | **likely-check-or-catalogue** |
| `inventory` | 444 | 102 | 38 | 38 of 54 tickers | **likely-check-or-catalogue** |
| `retainedEarnings` | 439 | 127 | 53 | 53 of 54 tickers | **likely-check-or-catalogue** |
| `noninterestIncome` | 408 | 95 | 8 | 8 of 54 tickers | **likely-filer** |
| `longTermDebtCurrentOnly` | 391 | 99 | 41 | 41 of 54 tickers | **likely-check-or-catalogue** |
| `shortTermBorrowingsOnly` | 383 | 96 | 46 | 46 of 54 tickers | **likely-check-or-catalogue** |
| `totalLiabilities` | 368 | 102 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |
| `goodwill` | 364 | 108 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |
| `accountsPayable` | 346 | 89 | 37 | 37 of 54 tickers | **likely-check-or-catalogue** |
| `restrictedCash` | 335 | 95 | 53 | 53 of 54 tickers | **likely-check-or-catalogue** |
| `intangiblesExGoodwill` | 330 | 91 | 50 | 50 of 54 tickers | **likely-check-or-catalogue** |
| `rentalIncome` | 319 | 78 | 46 | 46 of 54 tickers | **likely-check-or-catalogue** |
| `accountsReceivable` | 304 | 86 | 37 | 37 of 54 tickers | **likely-check-or-catalogue** |
| `premiumsEarned` | 296 | 78 | 40 | 40 of 54 tickers | **likely-check-or-catalogue** |
| `accumulatedDepreciation` | 289 | 94 | 50 | 50 of 54 tickers | **likely-check-or-catalogue** |
| `ppeGross` | 259 | 91 | 49 | 49 of 54 tickers | **likely-check-or-catalogue** |
| `sharesOutstanding` | 233 | 115 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |
| `currentLiabilities` | 170 | 57 | 37 | 37 of 54 tickers | **likely-check-or-catalogue** |
| `netInvestmentIncome` | 156 | 43 | 33 | 33 of 54 tickers | **likely-check-or-catalogue** |
| `currentAssets` | 154 | 57 | 37 | 37 of 54 tickers | **likely-check-or-catalogue** |
| `` | 146 | 79 | 54 | 54 of 54 tickers | **likely-check-or-catalogue** |

## clusters -- top 25 of 1951 with work in them

*`score = (sum over findings of w(severity) x w(tier)) x corroboration`, with tier {1: 4, 2: 2, 3: 1}, severity {'critical': 4, 'high': 2, 'medium': 1, 'info': 0}, and each additional agreeing check worth +25% (so 3.25x at ten checks). Those weights are a POLICY, not a fact, and they are module constants in `clusters.py` meant to be retuned once somebody has read a list and disagreed -- which is where the corroboration term came from: volume alone ranked a 62-finding 2-check cluster above a 55-finding 10-check one.*

| # | cluster_id | ticker | field | score | findings | checks | worst | routing |
|---|---|---|---|---|---|---|---|---|
| 1 | `919b35844b54` **<-- B's menu** | BA | `incomeTaxExpense` | 403 | 40 | 10 | high | likely-check-or-catalogue |
| 2 | `f38ec1daf240` **<-- B's menu** | MCD | `dilutedShares` | 363 | 59 | 8 | high | likely-check-or-catalogue |
| 3 | `2603621e89ab` **<-- B's menu** | ORCL | `totalRevenue` | 360 | 47 | 7 | high | likely-check-or-catalogue |
| 4 | `a423325138c3` **<-- B's menu** | APA | `netIncome` | 360 | 43 | 7 | high | likely-check-or-catalogue |
| 5 | `18a8dd2430be` **<-- B's menu** | MCD | `basicShares` | 358 | 58 | 8 | high | likely-check-or-catalogue |
| 6 | `2943d47a9b6b` | BA | `operatingIncome` | 358 | 45 | 8 | high | likely-check-or-catalogue |
| 7 | `02926a01f9dd` | APA | `pretaxIncome` | 338 | 47 | 6 | high | likely-check-or-catalogue |
| 8 | `b09464043fc1` | APA | `operatingIncome` | 320 | 46 | 6 | high | likely-check-or-catalogue |
| 9 | `9d0ebfbe91bb` | C | `interestExpense` | 315 | 37 | 7 | high | likely-check-or-catalogue |
| 10 | `0b47bf08db6e` | AMT | `depAmort` | 312 | 50 | 4 | high | likely-check-or-catalogue |
| 11 | `423bbd467f07` | MET | `incomeTaxExpense` | 310 | 44 | 6 | high | likely-check-or-catalogue |
| 12 | `3073bffe9c3f` | HCA | `minorityInterest` | 305 | 62 | 2 | high | likely-check-or-catalogue |
| 13 | `2474914d45d6` | SMCI | `inventory` | 300 | 52 | 3 | high | likely-check-or-catalogue |
| 14 | `0d8f61c72b85` | TMO | `incomeTaxExpense` | 300 | 47 | 5 | high | likely-check-or-catalogue |
| 15 | `c6c7ccb4f136` | AFL | `realizedInvestmentGains` | 288 | 38 | 5 | high | likely-check-or-catalogue |
| 16 | `b5e7c94ccf8f` | MET | `netIncome` | 280 | 42 | 7 | high | likely-check-or-catalogue |
| 17 | `5e4107839207` | UNP | `ppeNet` | 265 | 54 | 2 | high | likely-check-or-catalogue |
| 18 | `3a621a4973ae` | AXP | `ppeNet` | 255 | 51 | 2 | high | likely-check-or-catalogue |
| 19 | `747880a91fe1` | META | `shortTermInvestments` | 255 | 51 | 2 | high | likely-check-or-catalogue |
| 20 | `929e5f23bfea` | EQIX | `capex` | 253 | 28 | 8 | high | likely-check-or-catalogue |
| 21 | `62815e129657` | SPG | `dilutedShares` | 248 | 54 | 6 | high | likely-check-or-catalogue |
| 22 | `5dcc2a60db55` | BAC | `pretaxIncome` | 245 | 36 | 7 | high | likely-check-or-catalogue |
| 23 | `a237460c993c` | SPG | `basicShares` | 243 | 53 | 6 | high | likely-check-or-catalogue |
| 24 | `b889a901a4ec` | META | `stockBasedComp` | 238 | 31 | 6 | high | likely-check-or-catalogue |
| 25 | `b763bc61638b` | MCD | `operatingLeaseLiability` | 234 | 33 | 4 | high | likely-check-or-catalogue |

*1,926 further cluster(s) carry work and are not listed. Agent B works ONE of the top 5; the rest of this table is context, and the tail is a backlog SIZE rather than a reading list. Query `fundamentals_check` by `cluster_id` for any of them, or widen with `render(clusters=None)`.*

## the packets -- top 25

*Everything needed to start, without a second query. If a packet is not enough to begin on, that is a defect in the CHECK and worth reporting on its own.*

### 1. BA `incomeTaxExpense` -- `919b35844b54`  **<-- B's menu**

- **score 403** from 40 finding(s) across 10 check(s)
- checks agreeing: `trend_break`x18, `cross_vintage`x5, `scale`x5, `level_outlier`x3, `annual_footing`x2, `holdout_q4`x2, `peer_ratio`x2, `basis_step`x1, `q4_footing`x1, `series_shape`x1
- severity: high=32, medium=8 | tier: T2=30, T3=10
- periods: 2010-12-31..2026-03-31
- routing: **likely-check-or-catalogue** (53 of 54 tickers on this field)
- *10 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 40 times.*
- seen in 2 comparable run(s), 2026-08-24 -> 2026-08-25
- https://www.sec.gov/Archives/edgar/data/12927/000001292721000040/0000012927-21-000040-index.htm
- _the RESOLUTION ROUTE changed at this boundary and the level stepped with it -- the number now means something different, which no cross-vintage test can see_

### 2. MCD `dilutedShares` -- `f38ec1daf240`  **<-- B's menu**

- **score 363** from 59 finding(s) across 8 check(s)
- checks agreeing: `scale`x11, `annual_footing`x10, `holdout_q4`x10, `q4_footing`x10, `level_outlier`x6, `cross_vintage`x5, `trend_break`x5, `peer_ratio`x2
- severity: high=42, medium=17 | tier: T2=24, T3=35
- periods: 2010-12-31..2026-06-30
- routing: **likely-check-or-catalogue** (53 of 54 tickers on this field)
- *8 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 59 times.*
- seen in 2 comparable run(s), 2026-08-24 -> 2026-08-25
- https://www.sec.gov/Archives/edgar/data/63908/000006390817000017/0000063908-17-000017-index.htm
- _the only rule that catches a value resolved to an entirely wrong concept without a human noticing first_

### 3. ORCL `totalRevenue` -- `2603621e89ab`  **<-- B's menu**

- **score 360** from 47 finding(s) across 7 check(s)
- checks agreeing: `basis_step`x10, `level_outlier`x10, `tag_switch_break`x10, `trend_break`x5, `annual_footing`x4, `holdout_q4`x4, `q4_footing`x4
- severity: high=37, medium=10 | tier: T2=35, T3=12
- periods: 2018-05-31..2022-08-31
- routing: **likely-check-or-catalogue** (49 of 54 tickers on this field)
- *7 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 47 times.*
- seen in 2 comparable run(s), 2026-08-24 -> 2026-08-25
- https://www.sec.gov/Archives/edgar/data/1341439/000156459020030125/0001564590-20-030125-index.htm
- _the RESOLUTION ROUTE changed at this boundary and the level stepped with it -- the number now means something different, which no cross-vintage test can see_

### 4. APA `netIncome` -- `a423325138c3`  **<-- B's menu**

- **score 360** from 43 finding(s) across 7 check(s)
- checks agreeing: `trend_break`x29, `cross_vintage`x6, `scale`x3, `level_outlier`x2, `annual_footing`x1, `holdout_q4`x1, `q4_footing`x1
- severity: high=38, medium=5 | tier: T2=34, T3=9
- periods: 2012-09-30..2024-09-30
- routing: **likely-check-or-catalogue** (53 of 54 tickers on this field)
- *7 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 43 times.*
- seen in 2 comparable run(s), 2026-08-24 -> 2026-08-25
- https://www.sec.gov/Archives/edgar/data/1841666/000173303720000008/0001733037-20-000008-index.htm
- _a flat 3x rule, because for a lumpy field the MAD is wide enough that a real 3x jump scores under 3.5_

### 5. MCD `basicShares` -- `18a8dd2430be`  **<-- B's menu**

- **score 358** from 58 finding(s) across 8 check(s)
- checks agreeing: `scale`x11, `annual_footing`x10, `holdout_q4`x10, `q4_footing`x10, `level_outlier`x6, `trend_break`x5, `cross_vintage`x4, `peer_ratio`x2
- severity: high=41, medium=17 | tier: T2=24, T3=34
- periods: 2010-12-31..2026-06-30
- routing: **likely-check-or-catalogue** (54 of 54 tickers on this field)
- *8 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 58 times.*
- seen in 2 comparable run(s), 2026-08-24 -> 2026-08-25
- https://www.sec.gov/Archives/edgar/data/63908/000006390817000017/0000063908-17-000017-index.htm
- _the only rule that catches a value resolved to an entirely wrong concept without a human noticing first_

### 6. BA `operatingIncome` -- `2943d47a9b6b`

- **score 358** from 45 finding(s) across 8 check(s)
- checks agreeing: `trend_break`x19, `level_outlier`x11, `cross_vintage`x5, `scale`x5, `holdout_q4`x2, `annual_footing`x1, `peer_ratio`x1, `q4_footing`x1
- severity: high=29, medium=16 | tier: T2=36, T3=9
- periods: 2011-12-31..2026-06-30
- routing: **likely-check-or-catalogue** (44 of 54 tickers on this field)
- *8 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 45 times.*
- seen in 2 comparable run(s), 2026-08-24 -> 2026-08-25
- https://www.sec.gov/Archives/edgar/data/12927/000001292721000094/0000012927-21-000094-index.htm
- _a flat 3x rule, because for a lumpy field the MAD is wide enough that a real 3x jump scores under 3.5_

### 7. APA `pretaxIncome` -- `02926a01f9dd`

- **score 338** from 47 finding(s) across 6 check(s)
- checks agreeing: `trend_break`x26, `cross_vintage`x11, `level_outlier`x5, `scale`x3, `basis_step`x1, `series_shape`x1
- severity: high=39, medium=8 | tier: T2=36, T3=11
- periods: 2010-12-31..2024-09-30
- routing: **likely-check-or-catalogue** (53 of 54 tickers on this field)
- *6 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 47 times.*
- seen in 2 comparable run(s), 2026-08-24 -> 2026-08-25
- https://www.sec.gov/Archives/edgar/data/1841666/000173303719000004/0001733037-19-000004-index.htm
- _the RESOLUTION ROUTE changed at this boundary and the level stepped with it -- the number now means something different, which no cross-vintage test can see_

### 8. APA `operatingIncome` -- `b09464043fc1`

- **score 320** from 46 finding(s) across 6 check(s)
- checks agreeing: `trend_break`x24, `cross_vintage`x13, `level_outlier`x6, `holdout_q4`x1, `scale`x1, `series_shape`x1
- severity: high=39, medium=7 | tier: T2=32, T3=14
- periods: 2010-12-31..2024-09-30
- routing: **likely-check-or-catalogue** (44 of 54 tickers on this field)
- *6 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 46 times.*
- seen in 2 comparable run(s), 2026-08-24 -> 2026-08-25
- https://www.sec.gov/Archives/edgar/data/1841666/000167337916000004/0001673379-16-000004-index.htm
- _an interior gap with modal code 'insufficient_quarters' -- present before AND after, so the filer kept reporting and we stopped resolving. A MISSING TAG. Its modal code is `insufficient_quarters`, which does NOT excuse this: the TTM window is a START-of-history condition by its own rationale, and cannot open a hole in the middle of one_

### 9. C `interestExpense` -- `9d0ebfbe91bb`

- **score 315** from 37 finding(s) across 7 check(s)
- checks agreeing: `peer_ratio`x19, `trend_break`x6, `cross_vintage`x5, `level_outlier`x4, `basis_step`x1, `catalogue_exclusion_cost`x1, `series_shape`x1
- severity: high=32, medium=4, info=1 | tier: T1=1, T2=31, T3=5
- periods: 2010-12-31..2024-03-31
- routing: **likely-check-or-catalogue** (53 of 54 tickers on this field)
- *7 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 37 times.*
- seen in 2 comparable run(s), 2026-08-24 -> 2026-08-25
- https://www.sec.gov/Archives/edgar/data/831001/000083100123000097/0000831001-23-000097-index.htm
- _the RESOLUTION ROUTE changed at this boundary and the level stepped with it -- the number now means something different, which no cross-vintage test can see_

### 10. AMT `depAmort` -- `0b47bf08db6e`

- **score 312** from 50 finding(s) across 4 check(s)
- checks agreeing: `peer_ratio`x38, `level_outlier`x6, `cross_vintage`x5, `series_shape`x1
- severity: high=44, medium=6 | tier: T2=45, T3=5
- periods: 2010-09-30..2026-03-31
- routing: **likely-check-or-catalogue** (54 of 54 tickers on this field)
- *4 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 50 times.*
- seen in 2 comparable run(s), 2026-08-24 -> 2026-08-25
- https://www.sec.gov/Archives/edgar/data/1053507/000119312513418850/0001193125-13-418850-index.htm
- _the only rule that catches a value resolved to an entirely wrong concept without a human noticing first_

### 11. MET `incomeTaxExpense` -- `423bbd467f07`

- **score 310** from 44 finding(s) across 6 check(s)
- checks agreeing: `trend_break`x23, `cross_vintage`x14, `scale`x3, `level_outlier`x2, `basis_step`x1, `series_shape`x1
- severity: high=39, medium=5 | tier: T2=30, T3=14
- periods: 2010-12-31..2024-09-30
- routing: **likely-check-or-catalogue** (53 of 54 tickers on this field)
- *6 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 44 times.*
- seen in 2 comparable run(s), 2026-08-24 -> 2026-08-25
- https://www.sec.gov/Archives/edgar/data/1099219/000109921921000177/0001099219-21-000177-index.htm
- _the RESOLUTION ROUTE changed at this boundary and the level stepped with it -- the number now means something different, which no cross-vintage test can see_

### 12. HCA `minorityInterest` -- `3073bffe9c3f`

- **score 305** from 62 finding(s) across 2 check(s)
- checks agreeing: `peer_ratio`x61, `series_shape`x1
- severity: high=61, info=1 | tier: T2=62
- periods: 2010-12-31..2026-06-30
- routing: **likely-check-or-catalogue** (52 of 54 tickers on this field)
- seen in 2 comparable run(s), 2026-08-24 -> 2026-08-25
- https://www.sec.gov/Archives/edgar/data/860730/000119312511304797/0001193125-11-304797-index.htm
- _the only rule that catches a value resolved to an entirely wrong concept without a human noticing first_

### 13. SMCI `inventory` -- `2474914d45d6`

- **score 300** from 52 finding(s) across 3 check(s)
- checks agreeing: `peer_ratio`x47, `cross_vintage`x4, `series_shape`x1
- severity: high=52 | tier: T2=48, T3=4
- periods: 2011-10-27..2026-03-31
- routing: **likely-check-or-catalogue** (38 of 54 tickers on this field)
- *3 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 52 times.*
- seen in 2 comparable run(s), 2026-08-24 -> 2026-08-25
- https://www.sec.gov/Archives/edgar/data/1375365/000119312512041019/0001193125-12-041019-index.htm
- _the only rule that catches a value resolved to an entirely wrong concept without a human noticing first_

### 14. TMO `incomeTaxExpense` -- `0d8f61c72b85`

- **score 300** from 47 finding(s) across 5 check(s)
- checks agreeing: `trend_break`x27, `scale`x8, `cross_vintage`x6, `level_outlier`x5, `series_shape`x1
- severity: high=34, medium=13 | tier: T2=41, T3=6
- periods: 2010-12-31..2023-09-30
- routing: **likely-check-or-catalogue** (53 of 54 tickers on this field)
- *5 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 47 times.*
- seen in 2 comparable run(s), 2026-08-24 -> 2026-08-25
- https://www.sec.gov/Archives/edgar/data/97745/000009774512000039/0000097745-12-000039-index.htm
- _a flat 3x rule, because for a lumpy field the MAD is wide enough that a real 3x jump scores under 3.5_

### 15. AFL `realizedInvestmentGains` -- `c6c7ccb4f136`

- **score 288** from 38 finding(s) across 5 check(s)
- checks agreeing: `trend_break`x30, `cross_vintage`x4, `scale`x2, `coverage_field`x1, `series_shape`x1
- severity: high=36, medium=2 | tier: T1=1, T2=33, T3=4
- periods: 2010-09-30..2026-06-30
- routing: **likely-check-or-catalogue** (40 of 54 tickers on this field)
- *5 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 38 times.*
- seen in 2 comparable run(s), 2026-08-24 -> 2026-08-25
- https://www.sec.gov/Archives/edgar/data/4977/000162828026054618/0001628280-26-054618-index.htm
- _every filer in this regime resolves this field, so a miss is a defect in OUR extraction_

### 16. MET `netIncome` -- `b5e7c94ccf8f`

- **score 280** from 42 finding(s) across 7 check(s)
- checks agreeing: `cross_vintage`x17, `trend_break`x14, `holdout_q4`x3, `scale`x3, `annual_footing`x2, `level_outlier`x2, `q4_footing`x1
- severity: high=37, medium=5 | tier: T2=19, T3=23
- periods: 2010-12-31..2023-06-30
- routing: **likely-check-or-catalogue** (53 of 54 tickers on this field)
- *7 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 42 times.*
- seen in 2 comparable run(s), 2026-08-24 -> 2026-08-25
- https://www.sec.gov/Archives/edgar/data/1099219/000109921921000270/0001099219-21-000270-index.htm
- _a flat 3x rule, because for a lumpy field the MAD is wide enough that a real 3x jump scores under 3.5_

### 17. UNP `ppeNet` -- `5e4107839207`

- **score 265** from 54 finding(s) across 2 check(s)
- checks agreeing: `peer_ratio`x53, `series_shape`x1
- severity: high=53, info=1 | tier: T2=54
- periods: 2010-12-31..2024-09-30
- routing: **likely-check-or-catalogue** (51 of 54 tickers on this field)
- seen in 2 comparable run(s), 2026-08-24 -> 2026-08-25
- https://www.sec.gov/Archives/edgar/data/100885/000119312512170664/0001193125-12-170664-index.htm
- _the only rule that catches a value resolved to an entirely wrong concept without a human noticing first_

### 18. AXP `ppeNet` -- `3a621a4973ae`

- **score 255** from 51 finding(s) across 2 check(s)
- checks agreeing: `peer_ratio`x50, `series_shape`x1
- severity: high=51 | tier: T2=51
- periods: 2010-12-31..2026-04-17
- routing: **likely-check-or-catalogue** (51 of 54 tickers on this field)
- seen in 2 comparable run(s), 2026-08-24 -> 2026-08-25
- https://www.sec.gov/Archives/edgar/data/4962/000119312512077400/0001193125-12-077400-index.htm
- _an interior gap with modal code 'regime_break' -- present before AND after, so the filer kept reporting and we stopped resolving. A MISSING TAG. Its modal code is `regime_break`, which does NOT excuse this: a standard's adoption is a STEP, not a hole that closes again -- and this code is the modal one over the whole series, so it does not testify about these periods at all_

### 19. META `shortTermInvestments` -- `747880a91fe1`

- **score 255** from 51 finding(s) across 2 check(s)
- checks agreeing: `peer_ratio`x50, `series_shape`x1
- severity: high=51 | tier: T2=51
- periods: 2011-12-31..2025-03-31
- routing: **likely-check-or-catalogue** (45 of 54 tickers on this field)
- seen in 2 comparable run(s), 2026-08-24 -> 2026-08-25
- https://www.sec.gov/Archives/edgar/data/1326801/000132680124000012/0001326801-24-000012-index.htm
- _the only rule that catches a value resolved to an entirely wrong concept without a human noticing first_

### 20. EQIX `capex` -- `929e5f23bfea`

- **score 253** from 28 finding(s) across 8 check(s)
- checks agreeing: `peer_ratio`x10, `basis_step`x5, `tag_switch_break`x5, `cross_vintage`x3, `level_outlier`x2, `catalogue_exclusion_cost`x1, `scale`x1, `series_shape`x1
- severity: high=23, medium=3, info=2 | tier: T1=1, T2=24, T3=3
- periods: 2011-03-31..2026-03-31
- routing: **likely-check-or-catalogue** (54 of 54 tickers on this field)
- *8 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 28 times.*
- seen in 2 comparable run(s), 2026-08-24 -> 2026-08-25
- https://www.sec.gov/Archives/edgar/data/1101239/000119312513178095/0001193125-13-178095-index.htm
- _the RESOLUTION ROUTE changed at this boundary and the level stepped with it -- the number now means something different, which no cross-vintage test can see_

### 21. SPG `dilutedShares` -- `62815e129657`

- **score 248** from 54 finding(s) across 6 check(s)
- checks agreeing: `level_outlier`x17, `q4_footing`x12, `annual_footing`x11, `holdout_q4`x11, `cross_vintage`x2, `series_shape`x1
- severity: high=37, medium=17 | tier: T2=18, T3=36
- periods: 2009-06-30..2026-06-30
- routing: **likely-check-or-catalogue** (53 of 54 tickers on this field)
- *6 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 54 times.*
- seen in 2 comparable run(s), 2026-08-24 -> 2026-08-25
- https://www.sec.gov/Archives/edgar/data/1063761/000104746912001667/0001047469-12-001667-index.htm
- _three numbers the filer published independently, on three different bases, that must reconcile_

### 22. BAC `pretaxIncome` -- `5dcc2a60db55`

- **score 245** from 36 finding(s) across 7 check(s)
- checks agreeing: `cross_vintage`x14, `level_outlier`x6, `trend_break`x6, `peer_ratio`x4, `scale`x3, `tag_switch_break`x2, `basis_step`x1
- severity: high=27, medium=9 | tier: T2=22, T3=14
- periods: 2011-03-31..2025-06-30
- routing: **likely-check-or-catalogue** (53 of 54 tickers on this field)
- *7 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 36 times.*
- seen in 2 comparable run(s), 2026-08-24 -> 2026-08-25
- https://www.sec.gov/Archives/edgar/data/70858/000007085812000191/0000070858-12-000191-index.htm
- _the RESOLUTION ROUTE changed at this boundary and the level stepped with it -- the number now means something different, which no cross-vintage test can see_

### 23. SPG `basicShares` -- `a237460c993c`

- **score 243** from 53 finding(s) across 6 check(s)
- checks agreeing: `level_outlier`x16, `q4_footing`x12, `annual_footing`x11, `holdout_q4`x11, `cross_vintage`x2, `series_shape`x1
- severity: high=37, medium=16 | tier: T2=17, T3=36
- periods: 2009-06-30..2026-06-30
- routing: **likely-check-or-catalogue** (54 of 54 tickers on this field)
- *6 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 53 times.*
- seen in 2 comparable run(s), 2026-08-24 -> 2026-08-25
- https://www.sec.gov/Archives/edgar/data/1063761/000104746914001416/0001047469-14-001416-index.htm
- _the derivation the build performs everywhere, re-run where the answer is independently known_

### 24. META `stockBasedComp` -- `b889a901a4ec`

- **score 238** from 31 finding(s) across 6 check(s)
- checks agreeing: `peer_ratio`x19, `trend_break`x5, `level_outlier`x3, `scale`x2, `catalogue_exclusion_cost`x1, `series_shape`x1
- severity: high=24, medium=5, info=2 | tier: T1=1, T2=30
- periods: 2011-06-30..2026-03-31
- routing: **likely-check-or-catalogue** (54 of 54 tickers on this field)
- *6 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 31 times.*
- seen in 2 comparable run(s), 2026-08-24 -> 2026-08-25
- https://www.sec.gov/Archives/edgar/data/1326801/000119312512325997/0001193125-12-325997-index.htm
- _the only rule that catches a value resolved to an entirely wrong concept without a human noticing first_

### 25. MCD `operatingLeaseLiability` -- `b763bc61638b`

- **score 234** from 33 finding(s) across 4 check(s)
- checks agreeing: `peer_ratio`x30, `coverage_field`x1, `cross_vintage`x1, `series_shape`x1
- severity: high=33 | tier: T1=1, T2=31, T3=1
- periods: 2011-09-30..2026-06-30
- routing: **likely-check-or-catalogue** (54 of 54 tickers on this field)
- *4 INDEPENDENT checks agree here. That is a far stronger prior than one check firing 33 times.*
- seen in 2 comparable run(s), 2026-08-24 -> 2026-08-25
- https://www.sec.gov/Archives/edgar/data/63908/000006390826000035/0000063908-26-000035-index.htm
- _every filer in this regime resolves this field, so a miss is a defect in OUR extraction_


## recorded fixes

| cluster_id | ticker | field | layer | root cause | queue before -> after | commit | test |
|---|---|---|---|---|---|---|---|
| `1c9a517eaa47` | MCD | `capex` | extraction | route 1 took PaymentsToAcquireProductiveAssets, a total the filer declares BESIDE its own PaymentsToAcquirePropertyPlantAndEquipment leg in the same calculation link, so capex was resolved to the parent rather than the leg | 54 -> 3 | `2fb6ef2` | `tests/data_extract/test_linkbase_sibling_total_1c9a517eaa47.py` |

*`fundamentals_check_fix` is APPEND-ONLY and NOTHING here filters a finding. A fix row records what was done and what it measurably closed; it never subtracts a row, which is what keeps a row-count drop usable as proof. `root_cause` and `evidence` are in the table -- `validate fix show <cluster_id>`.*

## `wontfix` clusters

*None on file.*

*This section is never omitted. A `wontfix` that stops being listed is a suppression, which is precisely what the deleted JSON register was drifting toward.*
