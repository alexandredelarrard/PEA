# Phase 3b audit: is linkbase-driven resolution actually right across 26 tickers × 2011-2026?

**Date**: 2026-08-22
**Mode**: research only — no fixes applied
**Substrate**: 26 tickers × every 10-K and 10-Q since 2011-01-01, amendments excluded.
**1,544 filings · 144,190 ledger rows · 109,000 valued rows**, all 49 extracted fields
resolved per filing. Ledger cached at `scratchpad/shards/*.parquet`.

---

## Verdict

**The numbers are not right yet.** The architecture works — where the linkbase is actually
read, it resolves the filer's own total correctly and beats the tag list decisively. But three
defects make a material share of the output wrong or absent, and **one of them silently
disables the entire rebuild for four of the fifteen years**.

Phase 3's own measurement could not have found any of them: it read 6 filings, latest-only,
annual periods, nothing before FY2023.

| # | severity | defect | blast radius |
|---|---|---|---|
| 1 | **critical** | `menucat` is NULL on every pre-2015 filing → the whole linkbase is discarded | **418 of 1,544 filings (27.1%)**; linkbase share of resolutions is **0.9% in 2011-2014** vs ~70% from 2016 |
| 2 | **critical** | `discover_root` signature 2 picks an arbitrary parentless root | **74 revenue rows resolved to `Assets`, `LiabilitiesAndStockholdersEquity`, a cash-flow total, `ComprehensiveIncomeNetOfTax`, even `NoninterestExpense`** — 4 tickers |
| 3 | **high** | a filer tagging `Revenues = 0` still wins the priority walk | ETN 14 rows, VRT 24 rows still zero — **the plan's headline acceptance criterion is not met** |
| 4 | **high** | bank revenue resolves to a post-provision or single-leg concept | MTB: 110 rows on `InterestIncomeExpenseAfterProvisionForLoanLoss`, 6 on `NoninterestIncome` |
| 5 | **medium** | lease subtraction over-subtracts | **158 negative `shortTermDebt`** across 10 tickers, worst −$893M |
| 6 | medium | `totalLiabilities` never resolves for 5 tickers | elective tag; the assets−equity derivation is unbuilt |
| 7 | low | `sharesOutstanding` absent for META | upstream edgartools #691, confirmed live |
| 8 | low | by-ticker walk silently truncates at a CIK change | APA has **22 filings, none before 2021** |

---

## 1. `menucat` is null pre-2015 — the biggest defect, and it is our filter, not missing data

`statement_arcs()` keeps `arcs[arcs.menucat == "Statements"]`. That column comes from
**FilingSummary.xml**, which edgartools cannot categorise on older filings.

Measured: **418 of 1,544 filings return `menucat = None` for 100% of their arcs** — every
filing from 2011 through mid-2015, uniformly ~18 per ticker:

| year | filings with no usable linkbase |
|---|---|
| 2011 | 88 | 2012 | 91 | 2013 | 96 | 2014 | 96 | 2015 | 47 |

Consequence, measured on the resolution mix by year:

| year | linkbase routes | `tag_primary` | `tag_fallback` |
|---|---|---|---|
| 2011 | **0.9%** | 88.9% | 7.8% |
| 2012 | **0.9%** | 89.2% | 7.6% |
| 2013 | **0.9%** | 89.0% | 7.8% |
| 2014 | **0.8%** | 88.9% | 8.0% |
| 2015 | 37.6% | 52.6% | 7.7% |
| 2016 | 71.4% | 18.4% | 8.0% |
| 2020 | 69.7% | 22.1% | 5.8% |
| 2025 | 69.1% | 22.3% | 5.9% |

**For 2011-2014 the rebuild IS the old tag-list architecture.** Every claim about reading the
filer's own roll-up is false for that period, and the pooled 50% linkbase rate hides it.

**The data is present.** AAPL's FY2013 10-K carries 149 arcs including the complete face
statements — `StatementOfIncome` (10), `StatementOfFinancialPositionClassified` (25),
`StatementOfCashFlowsIndirect` (27), `StatementOfOtherComprehensiveIncome` (9) — with correct
signed weights: `GrossProfit ← +1 SalesRevenueNet, −1 CostOfGoodsAndServicesSold`. We threw all
of it away.

**A role-URI test recovers it, and agrees exactly with `menucat` where `menucat` works** —
7 of 7 filings tested, identical arc counts:

| ticker | year | arcs | `menucat=="Statements"` | role-URI | verdict |
|---|---|---|---|---|---|
| AAPL | 2012 | 130 | 0 | **63** | recovers |
| AAPL | 2014 | 157 | 0 | **74** | recovers |
| AAPL | 2016 | 145 | 76 | 76 | identical |
| AAPL | 2024 | 191 | 70 | 70 | identical |
| JPM | 2012 | 315 | 0 | **104** | recovers |
| JPM | 2014 | 424 | 0 | **104** | recovers |
| JPM | 2016 | 449 | 101 | 101 | identical |
| JPM | 2024 | 438 | 107 | 107 | identical |
| XOM | 2012/2014 | 186/198 | 0/0 | **88/88** | recovers |
| XOM | 2016/2024 | 196/186 | 86/81 | 86/81 | identical |
| MET | 2012/2014 | 321/307 | 0/0 | **139/145** | recovers |
| MET | 2016 | 294 | 139 | 139 | identical |

Filer role URIs are self-describing (`…/role/StatementOfIncome` vs `…/role/Disclosure*`,
`*Details`, `*Parenthetical`).

---

## 2. `discover_root` signature 2 is dangerously under-constrained

When no revenue anchor yields a candidate, `discover_root` falls back to *"a parentless,
reported root whose children are all positive-weight"*. On a real filing there are **several**
such roots — `Assets`, `LiabilitiesAndStockholdersEquity`, the cash-flow
period-increase total, `ComprehensiveIncomeNetOfTax` — and the code takes whichever comes
first out of a dict.

Measured `totalRevenue` concept census, the wrong tail:

| concept resolved as `totalRevenue` | rows | tickers |
|---|---|---|
| `CashCashEquivalentsRestrictedCash…PeriodIncreaseDecreaseIncludingExchangeRateEffect` | 21 | APA, DTE |
| `Assets` | 18 | DTE, USB |
| `LiabilitiesAndStockholdersEquity` | 16 | USB |
| `ComprehensiveIncomeNetOfTax` | 14 | GS |
| `CashCashEquivalents…ExcludingExchangeRateEffect` | 3 | GS |
| `NoninterestExpense` | 2 | USB |
| **total garbage** | **74** | **4 tickers** |

APA is the clearest case: `apa:RevenuesAndOther` resolves correctly for 62 rows
($4.308B–$12.132B), but 3 annual rows land on the cash-flow total, producing **revenue of
−$467M**.

This route contributed only **0.62%** of all resolutions, and a meaningful fraction of that
0.62% is wrong.

---

## 3. A filer tagging `Revenues = 0` still wins — ETN and VRT are NOT fixed

The plan's headline acceptance criterion is *"APA / ETN / VRT `totalRevenue` non-zero and
non-null"*. Measured:

| ticker | valued rows | **zeros** | window | concept |
|---|---|---|---|---|
| APA | 69 | 4 | 2021-2022 | the two bad ones above |
| **ETN** | 177 | **14** | 2012, 2014-2017 | `us-gaap:Revenues`, `SalesRevenueNet` |
| **VRT** | 116 | **24** | 2018-2020 | `us-gaap:Revenues` |

The filer tags `Revenues` as literally `0` while its real top line sits under a different
concept. The priority walk sees a *reported* first candidate and takes it — value zero is
still a value. This is the identical failure mode to the original APA defect, and the
research anticipated it (`PARTIAL_REVENUE_MATERIALITY` in the old stack existed for exactly
this and could not fire).

---

## 4. Bank revenue lands on the wrong basis for MTB

| ticker | concepts resolved as `totalRevenue` | rows | window |
|---|---|---|---|
| JPM | `Revenues` → `RevenuesNetOfInterestExpense` | 275 | clean 2015 handover |
| BAC | `Revenues` | all | fine (bank `Revenues` = net revenue) |
| USB | `Revenues` + 3 garbage concepts | — | see defect 2 |
| **MTB** | `Revenues` 2014-2024, **`InterestIncomeExpenseAfterProvisionForLoanLoss` 2015-2026 (110 rows)**, **`NoninterestIncome` 2025-2026 (6)** | | overlapping |

`InterestIncomeExpenseAfterProvisionForLoanLoss` is net interest income **after** the credit
provision — a different basis from Rule 9-04 caption 10, and not comparable to JPM's
`RevenuesNetOfInterestExpense`. `NoninterestIncome` alone is one leg of two.

---

## 5. Sign violations — the lease subtraction over-subtracts

**166 negative values on `sign: non_negative` fields** (0.25% of them):

| field | negatives | tickers | worst |
|---|---|---|---|
| `shortTermDebt` | **158** | 10 | −$893,000,000 |
| `accumulatedDepreciation` | 2 | 1 | −$450.6M |
| `interestExpense` | 2 | 2 | −$13.4M |
| `longTermDebtCurrentOnly` | 2 | 1 | −$765M |
| `longTermDebt` | 1 | 1 | −$22.1M |
| `stockBasedComp` | 1 | 1 | −$89M |

`shortTermDebt` dominates and its `total_adjustment` subtracts
`FinanceLeaseLiabilityCurrent` + `OperatingLeaseLiabilityCurrent` **unconditionally** — with
no linkbase test that those legs were ever *inside* the resolved total. `ppeNet` has exactly
that guard (`_only_when`, the sibling test); `shortTermDebt` has none.

---

## 6-8. Coverage gaps

- **`totalLiabilities` never resolves for APA, DTE, EOG, ETN, VLO.** Rule 5-02 has no "Total
  liabilities" caption, so `us-gaap:Liabilities` is elective. The plan already specifies the
  repair (`totalAssets − equityInclNCI`) but assigns it to the history layer, so it is
  currently a hole in a **Tier-1** field.
- **`sharesOutstanding` never resolves for META** — edgartools #691 (META/GOOG/BRK-B/STZ
  publish zero undimensioned shares-outstanding facts), confirmed live.
- **Filing-count truncation.** A `Company(ticker).get_filings()` walk gives no signal that
  history is missing:

| ticker | filings since 2011 | first | cause |
|---|---|---|---|
| **APA** | **22** | 2021-05 | CIK change — pre-2021 is Apache Corp, a different CIK |
| VRT | 33 | 2018-08 | 2018 listing |
| KR | 54 | 2011-03 | 8 filings lost to defect 9 below |
| ETN | 56 | 2012-11 | Cooper merger / Irish domestication |
| META | 57 | 2012-07 | 2012 IPO |
| all others | 62-63 | 2011 | complete |

---

## 9. Concept flapping *within* a year

The ASC-606 handover is clean where expected (AAPL, CSCO, META, KR, ETN, VRT all switch
`SalesRevenueNet` → `RevenueFromContractWithCustomer…` at 2018). But several tickers carry
**two concepts live in the same year**, meaning the basis changes filing-to-filing:

| ticker | overlap |
|---|---|
| AMT | `SalesRevenueNet`[2011-18] · `RevenueFromContract…`[2018] · `Revenues`[2018-26] |
| SWKS | `RevenueFromContract…`[2019-23] · `Revenues`[2019-26] — 5 years of overlap |
| XOM | `Revenues`[2012-26] · `xom:TotalRevenuesAndOtherIncome`[2015-17] |
| USB | `Revenues`[2013-26] + 3 garbage concepts [2020-23] |
| DTE | 5 concepts, incl. `Assets` and a cash-flow total |
| GS | `RevenuesNetOfInterestExpense`[2015-26] + 3 garbage [2019-20] |

Some of this is defect 2 leaking in; the AMT/SWKS/XOM cases are genuine dual-tagging where
the priority walk is not deterministic across filings.

---

## What is working, and should not be disturbed

- **The extension-total repair holds.** APA's `apa:RevenuesAndOther` resolves for 62 rows,
  $4.308B–$12.132B, via `linkbase_root` — a concept no tag list can name.
- **Regimes are stable**: exactly one regime per ticker across 15 years, AMT correctly
  `industrial` (the tower-REIT trap), BRK-B `hybrid`, GS `broker_dealer`.
- **Quarters exist now**: 21,553 `quarterly` + 8,420 `ytd6` + 7,953 `ytd9` + 36,364 `instant`
  from 10-Qs, versus zero quarterly periods in the Phase 3 measurement.
- **Only 4 `other`-shaped duration facts** in 144k rows — the day-count bands are well chosen.
- **`tag_fallback` is 6.87%** overall, comfortably under the plan's 20% gate — though that
  number will move once defect 1 is fixed and the early years actually use the linkbase.
- **10-Q linkbases are not worse than 10-K's** (52.0% vs 47.0% `linkbase_total`), refuting the
  concern that quarterly resolution would degrade.

---

## Open question this audit could not settle

Whether the 2011-2014 numbers, once the linkbase is actually read, *agree* with what the tag
list produced there. That is a before/after comparison that can only run after defect 1 is
fixed, and it is the single best test of whether the architecture change is an improvement or
merely a different answer. It should be measured, not assumed.
