# Research: how to extract SEC fundamentals consistently across the S&P 500

**Date**: 2026-08-21
**Research Phase**: 1 of 3 (FIC workflow)
**Next Phase**: Planning (`/plan`)
**Request**: [specs/2026-08-21_research_how to extract_Edgartools.md](../../../specs/2026-08-21_research_how%20to%20extract_Edgartools.md)

---

## Research Question

Extract every key financial KPI from SEC filings so that each KPI is (a) comparable
quarter-to-quarter within a ticker, (b) equivalent between tickers, (c) NULL when genuinely
absent, and (d) on one accounting definition. Biggest question asked: **what is the best
definition to adopt so a KPI means the same thing for every S&P 500 constituent** — specifically
for capex, assets, debt, cash and FCF. Plus: shortlist the ~40 KPIs to build first.

Constraints set by the request: no implementation, no plan, no code changes, no
Wikipedia-grade definitions, and raise trade-offs rather than deciding them.

---

## Summary

The current pipeline is not failing because of missing data or weak plumbing. It is failing
because of **one architectural assumption**: that a KPI can be resolved by taking the
highest-priority hit from an ordered list of candidate us-gaap concepts, on the premise that
those candidates are substitutes for one another.

Measured on this repo's own 7.8M-fact dump, **they are substitutes only 30-56% of the time**.
And FASB's published element definitions say why: several candidate pairs currently sitting in
the same list are, by definition, a *superset and its subset*, or two *disjoint components* of a
total, or an *ex-lease* and an *incl-lease* measure of the same thing.

Three consequences, all measured:

1. **Only 52 of 443 tickers (12%) use one revenue concept across their history.** 56% use three
   or more. Two thirds of the switches have no overlap period, so there is nothing to reconcile
   against — the level simply steps.
2. **`shortTermDebt` discards a real component.** Its candidates `LongTermDebtCurrent` and
   `ShortTermBorrowings` are disjoint legs whose sum is `DebtCurrent`. For 2,017 (ticker, period)
   cells across 111 tickers both legs are tagged with no total, the resolver keeps one, and the
   discarded leg is the *larger* one 54% of the time.
3. **The user's own check #3 cannot fail.** All 203,798 Q4 rows in the table are derived as
   FY−(Q1+Q2+Q3), so the footing identity passes 99.73% by construction and validates nothing.

The way out is visible and verified: **the filer already declares its own roll-up.** The XBRL
calculation linkbase names each filer's own total and gives signed weights to foot it. It
resolved cleanly for all six hard accounting regimes I tested (bank, insurer, REIT/Up-C,
integrated oil, E&P, utility). It is available in bulk from the SEC as `cal.tsv` — no per-filing
parsing — alongside the presentation linkbase, the dimension table, and the SEC's own documented
deduplication priority and period-error metrics.

The vendors converge on the same answer, from a different direction: all of them use a fixed
target schema plus **industry-specific statement templates** (Compustat INDL/FS + 5 financial
sub-models, FactSet 4 profiles, Worldscope 4 templates, Morningstar 6), and all of them gate
publication on footing identities. None of them claims a single universal definition — because
there isn't one.

---

## Part 1 — What exists today

### 1.1 The extraction path

| Stage | Where | Note |
|---|---|---|
| CLI `fundamentals` | [cli.py:149-156](../../../src/data_extract/cli.py#L149-L156) | `fetch_fundamentals_edgartools` then `rebuild_fundamentals_history` |
| Per-ticker walk | [fetch_fundamentals_edgar.py:1142-1232](../../../src/data_extract/utils/fundamentals/fetch_fundamentals_edgar.py#L1142-L1232) | 8 threads (`DEFAULT_WORKERS`, hardcoded), per-ticker `store.save` |
| Per-filing parse | `:985-1139` | `Company(t).get_filings()` then `filing.xbrl()` per filing, 3 retries |
| Tag resolution | `build_tag_frames`, `:308-718` | vectorised; flattens candidate lists to `(field, tag, priority)` and merges once |
| Period logic | [fundamentals_periods.py](../../../src/data_extract/utils/fundamentals/fundamentals_periods.py) (1,140 lines) | YTD decumulation, Q4 derivation, fiscal-calendar re-keying |
| Tag registry | [fundamentals_tags.py](../../../src/data_extract/utils/fundamentals/fundamentals_tags.py) (1,433 lines) | **213 fields, 376 concepts, 6 dicts**; ~80% of the file is evidence commentary |
| Long→wide | `_derive_history` in [fetch_fundamentals.py](../../../src/data_extract/utils/fundamentals/fetch_fundamentals.py) | full rebuild per ticker, no projection (`SELECT *` filtered by ticker) |
| Cube | [step_cube_fundamentals.py](../../../src/data_aggregate/transformers/step_cube_fundamentals.py) | 5 builders share one `PitFrames` |

Only edgartools import in the fetcher is `Company` ([:37](../../../src/data_extract/utils/fundamentals/fetch_fundamentals_edgar.py#L37)). The library's own
standardization layer is deliberately unused — a decision this research validates (§3.1).

### 1.2 Resolution semantics

Strict per-period priority: highest-priority (earliest-listed) candidate that reported *that
period* wins, never first-non-null over the series and never the largest. Winner selection at
[:670-672](../../../src/data_extract/utils/fundamentals/fetch_fundamentals_edgar.py#L670-L672);
the rule is stated at [fundamentals_tags.py:6-10](../../../src/data_extract/utils/fundamentals/fundamentals_tags.py#L6-L10).
Admissibility gates run *before* the priority pick, so a rejected fact falls through to the next
candidate. **No candidate list is ever summed** — the only two additive exceptions are the SG&A
companion add and the multi-class cover-page sum.

A field-level undimensioned override ([:657-666](../../../src/data_extract/utils/fundamentals/fetch_fundamentals_edgar.py#L657-L666))
makes the priority ordering safe against a shaky dimensioned admission.

### 1.3 Q4 and period handling

`Q4 = FY − (Q1_discrete + Q2_discrete + Q3_discrete)`, gated by five preconditions
([fundamentals_periods.py:492-537](../../../src/data_extract/utils/fundamentals/fundamentals_periods.py#L492-L537)).
Native Q4 labels are **never trusted** (`:363-366`) — proven unreliable in two opposite ways
(Q4-shaped context carrying an FY-sized value; a genuinely-earlier quarter labelled Q4).

Notable engineering that is *right* and should survive any rebuild:
- FY anchor must be annual-**shaped**, not merely FY-**labelled** (Skyworks FY2020 tags both a
  370-day and a 97-day fact as `fp='FY'`).
- `_relabel_by_chronological_rank` uses relative order, not a day-count divisor — calendar
  quarters are 90/91/92 days, never a uniform 91.
- Fiscal years are keyed off the issuer's own 10-K period-ends, immune to 52/53-week drift.
- `_fy_matches_quarterly_run_rate` uses the sum of |quarters|, not |sum of quarters|, so
  offsetting quarters don't collapse the denominator.
- Provenance always travels with the value's own row, never re-joined.

### 1.4 Live state does not match the code

| table | rows | tickers | PK | what it is |
|---|---|---|---|---|
| `fundamentals_facts` | 7,776,870 | 445 | `(ticker, accession, **concept, period_key, dim_key**)` | raw tag-agnostic fact dump with a `dim_admission` verdict. **No repo code writes it.** |
| `fundamentals_facts_legacy` | 2,371,881 | 491 | matches `_FACTS_COLS` | the actual output of `fetch_fundamentals_edgartools`, renamed aside |
| `fundamentals_history` | 27,602 | 491 | `(ticker, as_of)` | 239 cols, `as_of` = filing date, ~60 rows/ticker 2011→2026 |

The fetcher's `pk=[... field ... duration_type]` cannot upsert against the live PK. Consistent
with the fundamentals step being commented out at
[step_extract_all_data.py:45-52](../../../src/data_extract/step_extract_all_data.py#L45-L52).
`sql/schema.sql:328-359` is internally inconsistent — its PK names `field` and `duration_type`,
neither of which appears in its own column list.

Silver lining: that 7.8M-row raw dump is an excellent offline audit substrate, and most of the
measurements in this document came from it.

### 1.5 What the 239 columns are actually for

- **130 consumed** downstream; **109 read by nothing**.
- 2 (`shortTermBorrowingsOnly`, `longTermDebtCurrentOnly`) are 100% NULL and produced by no code
  — but are named exactly after the two legs §2.3 shows must be separated.
- The unconsumed set is dominated by split legs whose *reconstructed total* is what gets used
  (leases, ARO, deferred revenue, regulatory, restricted cash), the per-year maturity ladders
  (only `debtMaturity1y` + derived `debtMaturity5yTotal` are read), and the 6 pension components
  (only derived `nonServicePensionCost` is read).

### 1.6 Validation apparatus — less than recorded

`fundamentals_identities.py`, `fundamentals_quality.py`, `classify_kinks`, the
`fundamentals-quality` CLI and `fundamentals_history_provenance` **have never existed in this
tree** (verified with `git log --all --diff-filter=A`). What does exist:

| file | role |
|---|---|
[fundamentals_validation.py](../../../src/validate/fundamentals_validation.py) | `apply_plausibility_guards` (13 mutating rules) + `reconcile_fundamentals_facts` (6 flag-only checks) |
[analyze_history.py](../../../src/validate/analyze_history.py) + [outliers.py](../../../src/utils/outliers.py) | MAD modified-z, threshold 3.5 (Iglewicz & Hoaglin) |
[fundamentals_tag_ledger.py](../../../src/utils/fundamentals_tag_ledger.py) | `detect_tag_switch_breaks` — the closest thing to a kink explainer |
[fundamentals_audit.py](../../../src/validate/fundamentals_audit.py) | Tiingo→Yahoo→uncovered fallback, 7-source ranked queue |

No CLI; all are `python -m` `__main__` blocks with no checkpointing.
`data/sec_bulk_cache/` and `data/gaps/` **do not exist**, which darkens ~28 tests, including the
entire "real-data coverage" half that [docs/testing.md:25-27](../../../docs/testing.md#L25-L27) mandates.
[test_fundamentals_point_in_time.py](../../../tests/data_extract/test_fundamentals_point_in_time.py) is *deliberately red* —
its docstring records that both point-in-time invariants are currently violated (493 tickers,
13.9% of rows; lags out to 1,884 days) and calls itself the acceptance criteria for the fix.

---

## Part 2 — Measured evidence

All figures from the live `pea` DB and the cached 7.8M-fact dump, 2026-08-21.

### 2.1 Candidate concepts are not substitutes

Ratio A/B where **both** concepts are tagged undimensioned for the **exact same period**:

| pair | n pairs | tickers | agree ±1% | off >50% | median |
|---|---|---|---|---|---|
| `ProfitLoss` / `NetIncomeLoss` | 18,085 | 331 | 48.3% | 4.5% | 1.006 |
| `CashInclRestricted` / `CashAndEquiv` | 10,170 | 379 | 56.2% | 9.8% | 1.005 |
| `LongTermDebtNoncurrent` / `LongTermDebt` | 6,005 | 267 | **34.4%** | 3.7% | 0.964 |
| `ASC606Excl` / `Revenues` | 3,109 | 148 | 39.9% | **24.2%** | 0.981 |
| `DDA` / `DepAmort` | 1,600 | 84 | 48.1% | 15.5% | 1.000 |
| `PaymentsToAcquireProductiveAssets` / `...PP&E` | 615 | 52 | 42.9% | **37.1%** | 1.000 |
| `CostOfGoodsAndServicesSold` / `CostOfRevenue` | 116 | 10 | 30.2% | **45.7%** | 0.969 |
| `ASC606Excl` / `SalesRevenueNet` | 79 | 37 | **73.4%** | 6.3% | 1.000 |

### 2.2 Concept switching, and why it produces a step

Distinct undimensioned concepts per (ticker, KPI family) over full history:

| family | tickers | 1 concept | 2 | 3+ | max | % of ticker-YEARS with 2+ co-tagged |
|---|---|---|---|---|---|---|
| revenue | 443 | **52 (12%)** | 141 | **250 (56%)** | 7 | **34.3%** |
| cash | 445 | 19 (4%) | 362 | 64 | 3 | 50.9% |
| ltd | 416 | 97 | 207 | 112 | 3 | 52.1% |
| dep_amort | 438 | 111 | 233 | 94 | 4 | 55.5% |
| cogs | 327 | 141 | 123 | 63 | 4 | 12.8% |
| net_income | 445 | 102 | 343 | 0 | 2 | 53.4% |
| capex | 430 | 289 (67%) | 109 | 32 | 5 | 12.1% |
| sga | 363 | 286 (79%) | 67 | 10 | 3 | 15.1% |

Low co-tagging = sequential switch = no overlap year to reconcile = the level steps. Revenue,
capex, COGS and SG&A are all in that regime.

The ASC-606 cutover is one dated, universe-wide taxonomy event, not filer noise. FASB deprecated
`SalesRevenueNet`, `SalesRevenueGoodsNet`, `SalesRevenueServicesNet`, `CostOfGoodsSold` and
`CostOfServices` on **2018-01-31**:

| FY | `SalesRevenueNet` | `Revenues` | ASC606-ExclAssessed |
|---|---|---|---|
| 2017 | 185 | 197 | 9 |
| **2018** | 158 | **274** | **192** |
| 2019 | 16 | 240 | 252 |
| 2020 | 0 | 230 | 270 |

### 2.3 The `shortTermDebt` component-split defect

Which tags actually win (`fundamentals_facts_legacy`, instant):

| field | winning tag | % rows | tickers | leases? |
|---|---|---|---|---|
| `longTermDebt` | `LongTermDebtNoncurrent` | 58.9% | 333 | **excl** |
| | `LongTermDebtAndCapitalLeaseObligations` | 21.0% | 156 | **incl** |
| | `LongTermDebt` | 19.7% | 234 | **excl** |
| `shortTermDebt` | `LongTermDebtCurrent` | 43.3% | 281 | current maturities of LTD **only** |
| | `DebtCurrent` | 29.8% | 175 | **all** current debt, **incl** leases |
| | `ShortTermBorrowings` | 21.1% | 173 | revolver/CP **only** |

FASB's verbatim definitions (2025 documentation linkbase, 14,899 elements parsed locally):

- `LongTermDebt`, `LongTermDebtNoncurrent` — "…of long-term debt. **Excludes lease obligation.**"
- `LongTermDebtAndCapitalLeaseObligations` — "long-term debt **and lease obligation**, noncurrent"
- `DebtCurrent` — "Amount of debt **and lease obligation**, classified as current."

So `LongTermDebtCurrent` and `ShortTermBorrowings` are **disjoint legs whose sum is
`DebtCurrent`**. The priority order `DebtCurrent > LongTermDebtCurrent > ShortTermBorrowings`
keeps one leg and discards the other whenever the total is untagged.

**Measured: 2,017 (ticker, period) cells across 111 tickers** tag both legs with no total.
Median `ShortTermBorrowings / LongTermDebtCurrent` = **1.103**. The discarded leg exceeds 25% of
the kept leg in **86.1%** of cells and is outright larger in **54.4%**.

Inherited by `debtToEquity`, `capital.net_debt`, `net_debt_to_ebitda`, `cash_to_debt`,
`interest_coverage`, `refinancing_risk`, `altman_z`.

Separately, `totalDebt = longTermDebt + shortTermDebt` adds an ex-lease long leg (79% of rows) to
an incl-lease short leg — a taxonomy asymmetry, not a coding bug.

### 2.4 Two more definitional traps, from FASB's own strings

- **`PaymentsToAcquireProductiveAssets` is a superset of `PaymentsToAcquirePropertyPlantAndEquipment`**
  — officially "property, plant and equipment (capital expenditures), **software, and other
  intangible assets**". They sit at priority 1 and 0 of the same `capex` field. Hence the 37.1%
  disagreement above. FASB's calculation linkbase makes it the total-capex node:
  PP&E + Software + Intangibles + MineralRights + CryptoAsset + EquipmentOnLease + Other.
- **`DepreciationAndAmortization` is officially non-production D&A only** — "long-lived, physical
  assets **not used in production**". `DepreciationDepletionAndAmortization` is "the **aggregate**".
  The repo's priority order is correct; the `MCD` deny-list entry forces the *lower*-priority
  element, against the taxonomy definition.

### 2.5 The staircase: a TTM that does not move

Consecutive `(ticker, as_of)` pairs where `totalRevenue` is *exactly* unchanged — a TTM should
move at nearly every filing, so an exact repeat means the annual fallback
(`ttm_a` → `<field>_ann`.ffill(limit=4)) supplied it:

**1,622 of 26,242 pairs = 6.2% frozen**, across 442 of 491 tickers.
Worst: **APA 18/18 = 100%**, **XOM 21/58 = 36%**, ETN 33%, MTB 28%, TROW 26%.

XOM worked example — no `duration_type='quarterly'` revenue row exists before FY2018:

```
as_of        fiscal_end   rev_ttm($bn)  revenue_q
2014-02-26   2013-12-31      420.8        NULL
2014-05-07   2014-03-31      420.8        NULL
2014-08-06   2014-06-30      420.8        NULL
2014-11-05   2014-09-30      420.8        NULL
2015-02-25   2014-12-31      394.1        NULL
...
2018-05-03   2018-03-31      237.2        65.4   <- first real quarter
```

`revenueGrowth = pct_change(4)` is therefore exactly 0 for three quarters in four, then steps.

**Do not confuse this with banks/REITs.** `revenue_q` is structurally NULL for banks (the facts
layer deliberately nulls the bank top line and `_derive_history` rebuilds the TTM from
NII + noninterest income). Verified TFC and EQR TTMs move smoothly every quarter. The
frozen-value test above is the correct detector; `revenue_q IS NULL` is not.

### 2.6 APA — a closed root-cause chain

APA's own calculation linkbase declares **`RevenuesAndOther`** as its pretax revenue parent.
`RevenuesAndOther` is **absent from `fundamentals_tags.py`** (verified by grep). The resolver
therefore fell through to `RevenueFromContractWithCustomerIncludingAssessedTax`, which APA tags
as literally **$0.00** — every quarter, every year. `PARTIAL_REVENUE_MATERIALITY` could not fire
because it compares against `Revenues`, which APA does not tag undimensioned.

Three tickers carry `totalRevenue = 0`: APA (19 rows), ETN (16), VRT (5).

### 2.7 Quarter completeness and the vacuous Q4 check

- **All 203,798 Q4 rows are derived.** Only 28 (ticker, field, year) groups universe-wide carry
  an as-reported Q4, and 92.9% of *those* fail the footing test.
- So `Q1+Q2+Q3+Q4 == FY` passes **99.73%** at 2% tolerance on derived groups. **It validates
  nothing** — the identity is how Q4 was constructed. An identity must not take a derived
  quantity as an input.
- Completeness per (ticker, field, fiscal_year): 4 quarters **81.2%**, 3q 2.6%, 2q 9.3%, 1q 7.0%.
- Per-field complete-year rate clusters tightly at 84-88% for every core P&L/CF field. The
  uniform ~11.5% residual ≈ 2 boundary years out of ~15 (first partial + current in-progress) —
  largely benign. Spot-checked: MAA/JPM/MET/AFL/C/DTE/REG all show the same 14-of-16 pattern.
- Genuinely episodic (filer omits the line in no-activity quarters): `debtIssued` 58.9% complete,
  `acquisitions` 60.8%, `debtRepaid` 69.1%, `buybacks` 80.0% — handled by `CHARGE_FLOWS` 0-fill.

### 2.8 Field absence is structural, not a gap

Tickers with **zero** rows for a field, by GICS sector:

| sector | n | currentAssets | grossProfit | COGS | R&D | operatingIncome | inventory | SG&A |
|---|---|---|---|---|---|---|---|---|
| Financials | 76 | **48** | **59** | **61** | 68 | **38** | 72 | 38 |
| Real Estate | 31 | **23** | 13 | 14 | 30 | 0 | 28 | 1 |
| Utilities | 31 | 0 | 12 | 12 | 29 | 0 | 14 | 26 |
| Industrials | 77 | 3 | 10 | 11 | 31 | 0 | 18 | 12 |
| Info Tech | 72 | 0 | 0 | 0 | 7 | 0 | 19 | 0 |
| Health Care | 57 | 0 | 4 | 4 | 15 | 0 | 6 | 2 |
| Cons Disc | 47 | 4 | 3 | 3 | 35 | 0 | 11 | 1 |
| Cons Staples | 33 | 0 | 0 | 0 | 13 | 0 | 1 | 6 |
| Materials | 26 | 0 | 0 | 0 | 9 | 0 | 0 | 0 |
| Energy | 21 | 0 | 3 | 3 | 14 | 0 | 3 | 1 |
| Comm Svcs | 20 | 0 | 8 | 8 | 15 | 1 | 8 | 1 |

**This is correct filing behaviour, not missing data.** FASB ships
`StatementOfFinancialPositionUnclassified-DepositBasedOperations` and
`...-SecuritiesBasedOperations` roles; the bank balance sheet contains **zero** occurrences of
`AssetsCurrent` while still carrying `Assets`. The 78 tickers with no `AssetsCurrent` are filing
completely.

Implication for design: **expected absence and coverage regression currently look identical.**
A structural regime flag would separate them; a null-count never can.

### 2.9 The Financials top-line rebuild is under-fed

`_derive_history` builds `bank_rev = (nii.fillna(0) + noni.fillna(0)).where(nii.notna() | noni.notna())`
and `insurer_rev = premiums + netInvestmentIncome + realizedGains`. Across the 76 Financials:

| | count |
|---|---|
| has `netInterestIncome` | 34 |
| has `noninterestIncome` | **23** |
| **has NII but NO noninterest income → revenue = NII alone** | **11** |
| has `premiumsEarned` | 25 |
| **has premiums but NO net investment income** | **6** |
| **has NEITHER leg → falls back to the ASC-606 candidate list** | **17** |

Noninterest income is typically 30-45% of a US bank's net revenue, so those 11 tickers'
revenue is understated by roughly a third. Real Estate: 2 of 31 have NII, 0 have noninterest income.

---

## Part 3 — External evidence

### 3.1 edgartools (installed 5.44.1; PyPI 5.51.0)

**Reuse:**
- **`edgar/ttm/` quarterization is genuinely good.** Real YTD decumulation and
  Q4 = FY − YTD_9M with a fallback to FY − (Q1+Q2+Q3), selecting inputs **by calendar period,
  not `fiscal_period` label**. Guards refuse instants, shares, ratios and per-share units.
  Verified: AAPL Q4-2025 revenue = 102.466 B, matching the actual quarter.
- `xbrl.calculation_linkbase()` — the strongest primitive in the library (§3.2).
- `entity/data/industry_extensions/*.json` — 20 files, a free data-derived inventory of
  bank/insurer/REIT line items with occurrence rates.

**Do NOT adopt `standardization/exclusions.py` as a denylist** (an earlier draft of this document
recommended it — that was wrong, and reading the file is what corrected it). It holds **272** tags,
not the 276 its own stale comment claims, and **160 of them (59%) are per-share/EPS tags**. Its
docstring is explicit that these are marked *"DropThisItem"* because they *"don't map cleanly to
standard concepts"* — a statement about edgartools' standardization goal, **not** about data
quality.

Measured against this repo's 376 candidate concepts, **16 collide**, several load-bearing:
`EarningsPerShareBasic`/`Diluted`/`BasicAndDiluted` (→ `epsBasic`, `epsDiluted`),
`IncreaseDecreaseInAccountsReceivable`/`AccountsPayable`/`Inventories` (→ the whole
`changeIn*` working-capital family that `dso`/`dpo`/`dio`/`cash_conversion_cycle` and the Beneish
M-score are built from), `CashAndCashEquivalentsPeriodIncreaseDecrease` (→ `cashPeriodChange`),
`ComprehensiveIncomeNetOfTax`, `OtherComprehensiveIncomeLossNetOfTax`, `IncomeTaxesPaid`,
`AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount`,
`CommonStockSharesAuthorized`, `PreferredStockSharesAuthorized`.

What IS worth taking is the *principle* behind the 59%: **a per-share figure is not additive.**
This repo already encodes that in `LATEST_DURATION_TAGS` (share counts and `effectiveTaxRate` are
routed through the instant path so `ttm_a` cannot sum them) — but `epsDiluted`/`epsBasic` sit in
`EXTRA_FLOW_TAGS` (`fundamentals_tags.py:502-503`) and therefore *are* TTM-summed. Summing four
quarterly EPS is a defensible convention, but it drifts from annual EPS as share count moves, and
it is the same non-additivity edgartools hit from the other side in
[#690](https://github.com/dgunning/edgartools/issues/690) (its naive Q4-EPS derivation matched AAPL
in only 5 of 17 years). Worth a decision rather than an accident.

**Do not trust:**
- **`standard_concept` is not a safe join key.** `gaap_mappings.json` ships
  `avg_confidence: 0.506`. Open bug [#914](https://github.com/dgunning/edgartools/issues/914)
  (unfixed through 5.51.0): line-item tags map to balance-sheet *total* concepts — on Citizens
  Financial, `standard_concept == 'Assets'` returns 3 rows including bank-owned life insurance.
  Independently reproduced: the `Assets` concept ingests `BankOwnedLifeInsurance`,
  `FederalFundsSold`, `FederalHomeLoanBankStock`.
- SG&A lumps the total **and** both components (observed double-count on AAPL); debt and D&A are
  each split 3 ways.
- The FF48 industry-override path is **inert**: `Company("AAPL").latest("10-K").xbrl().standardization.industry`
  returns `None`, because industry is only set if the SGML header happens to be loaded already.
  And `sic_to_fama_french` is a first-match scan over overlapping ranges, so SIC 3571 (AAPL)
  resolves to `Mach`, not `Comps` — the 230 `Comps` overrides are unreachable.
- **No FCF, no EBITDA, no restatement vintage.** `is_restated` is declared and never set.
- **[#691](https://github.com/dgunning/edgartools/issues/691)** — META, GOOG, BRK-B, STZ return
  0 non-dimensioned shares-outstanding facts 2012-2026. This is the repo's 36-NULL-ticker
  finding, upstream.
- Default `to_dataframe()` view is DETAILED: dimensional breakdown rows carry the **same**
  `concept` as the parent, so grouping by concept double-counts.

Worth upgrading for 5.45.0 (filings **over 10 MB were losing most of their text** — hits
bank/insurer 10-Ks directly) and 5.50.0/5.50.1 (companyfacts and the HTTP cache were effectively
not caching). Note 5.45.0 changes extracted text for nearly every filing.

### 3.2 The filer declares its own roll-up — verified across all six regimes

`xbrl().calculation_linkbase()` returns `concept, parent_concept, weight, role_uri, role_short,
menucat, is_abstract, label` — 465 arcs for JPM's latest 10-K. Select `menucat == "Statements"`.

| ticker | regime | what the FILER declares as its top line |
|---|---|---|
| JPM | bank | `RevenuesNetOfInterestExpense` ← `NoninterestIncome` + …; − provision − `NoninterestExpense` − tax |
| MET | insurer | root `NetIncomeLossAvailableToCommonStockholdersBasic`; preferred-dividend bridge explicit |
| MAA | REIT / Up-C | same root; `ProfitLoss` → NCI → `PreferredStockDividendsIncomeStatementImpact` |
| XOM | integrated oil | **`Revenues` − `CostsAndExpenses`** |
| APA | E&P | **`RevenuesAndOther` − `CostsAndExpenses`** |
| DTE | utility | `RegulatedAndUnregulatedOperatingRevenue`; `OperatingIncomeLoss` − `CostsAndExpenses` |

FASB publishes the same structure for the standard statements. Verified from
`us-gaap-stm-scf-indir-cal-2025.xml` (766 arcs): exactly one arc feeds a bottom-line income
concept into operating cash flow — **`NetCashProvidedByUsedInOperatingActivities ← +1 ProfitLoss`**.
So OCF officially reconciles from the *consolidated* figure. The repo's `netIncome` already
prefers `ProfitLoss`, so `accruals` compares like with like — **a validated existing decision**,
worth recording so nobody "fixes" it.

### 3.3 Available in bulk — no per-filing cost

Downloaded and inspected:

- **FSDS**, quarterly, `.../financial-statement-data-sets/2025q4.zip` — 66 MB → 676 MB.
  `pre.txt` header: `adsh report line stmt inpth rfile tag version plabel negating`.
  `num.txt` header: `adsh tag version ddate qtrs uom **segments** coreg value footnote` —
  post-Dec-2024 reprocessing, so **dimensioned facts are present**, unlike companyfacts.
- **FSNDS**, now **monthly**, `.../financial-statement-notes-data-sets/2025_10_notes.zip`
  (quarterly `*q*_notes.zip` is 404) — 93 MB → 705 MB. Ships
  `sub tag **dim** ren **cal** pre num txt`.
  - `cal.tsv`: `adsh grp arc **negative ptag pversion ctag cversion**` — the calculation
    linkbase for every filer, in bulk.
  - `num.tsv`: `adsh tag version ddate qtrs uom **dimh iprx** value footnote footlen **dimn**
    coreg **durp datp dcml**`. `dimn=0`/`dimh=0x00000000` selects the consolidated undimensioned
    fact; `iprx` is the SEC's own **documented dedup priority**; `durp`/`datp` quantify how far a
    fact's duration/date is from a clean quarter/month-end; `dcml` is the scale-error detector.

What `pre` gives that `companyfacts` cannot: statement placement (`stmt` ∈ BS/IS/CF/EQ/CI/UN/CP,
so a cash-flow `Cash` is distinguishable from a balance-sheet `Cash`), the filer's own
presentation order, the filer's own label, and a parenthetical flag.

The repo already holds the URL template at
[constants.py:113-115](../../../src/constants/constants.py#L113-L115) with
`SEC_FINNOTES_FIRST_YEAR = 2009`, and `notes_num`/`notes_text`/`pension_facts` exist in Postgres.
But `notes_num` is a 40,587-row filtered subset that **drops `dimn`, `iprx`, `durp`, `datp`,
`dcml`, `coreg`, `segments`**, there is no `cal`/`pre` table, and no `src/` code references any
of them any more.

Cost: ~190 monthly zips for 15 years of complete coverage, versus ~30k individual
`filing.xbrl()` fetch-and-parse calls (measured at 6.5-10 s/filing for a large filer).

### 3.4 How the vendors actually solve this

Every vendor uses a fixed target schema **plus industry statement templates**, and none claims a
single universal definition.

| vendor | templates | original/restated in schema? | published dictionary? | validation |
|---|---|---|---|---|
| Compustat | INDL / FS + 5 financial balancing sub-models | NA overwrites quarterly restatements; Global keeps `HIST_STD` + `RST_STD` | Yes (*Data Definitions*) | **14,000 checks/company** |
| FactSet | 4 (Commercial, Bank, Insurance, Other Financial) | **Yes** — `ANN`/`ANN_R`, `QTR`/`QTR_R`, `RP`/`RF` preliminary/final, `/point-in-time` | Yes (`/metrics`) | not found |
| Worldscope | 4 (industrials, banks, insurance, other financial) | advertises PIT; vintage semantics not found | Yes (~737 pp) | **2,300 balance/magnitude/correlation/alpha tests, no-publish rule** |
| Morningstar | **6** (`N` Normal, `M` Mining, `U` Utility, `T` Transportation, `B` Bank, `I` Insurance) — exposed as a queryable field | **Yes** — First/Last-Known, Final/Preliminary, `IsBestKnownReport` | **Yes, the best public one** | auditor name + opinion code + GAAP style as fields |
| Bloomberg | asserted, not published | CoFi PIT product | **No** (terminal only) | **13,000 checks** |
| MSCI/Barra | none — buys Compustat | not found | descriptor formulas only | 3-bucket outlier rule (drop / trim 3σ / keep) |

Load-bearing details:

- **Compustat's key is compound**: `(gvkey, datadate, indfmt, consol, popsrc, datafmt)`. `INDL`
  is the *universal baseline*; `FS` is **annual-only and supplements rather than replaces** it —
  so both records exist and a naive pull returns duplicate rows. Canonical research filter:
  `indfmt='INDL', datafmt='STD', consol='C', popsrc='D'`.
- **Compustat's item definitions carry explicit industry overrides inside a single item.** `CHE`:
  "For banks and savings and loans this includes cash and due from banks and federal funds."
  `SALE` includes "Banks and savings and loans' interest income and fee revenue". `XRD` and
  `CAPX` each end with "**This item is not available for banks**".
- **`XSGA` is not comparable by Compustat's own design**: "If a company allocates any of these
  expenses to Cost of Goods Sold, **we will not include them** in Selling, General, and
  Administrative Expense."
- **Reason codes beside every value.** `*_DC` companion column: `4` = **combined figure** (rolled
  into another item — "do not treat as zero"), `8` = insignificant. Plus a published
  `Combined Data Item → Data Item(s) Combined with` **destination** map. WRDS's own caveat: only
  ~1.2% of missing `XRD` is coded, so most blanks are unexplained even at Compustat.
- **Worldscope draws the line explicitly**: it normalizes *presentation*, never *valuation*.
  "We do not believe this is a feasible or desirable aim… We simply scrutinize exactly what
  components each reported figure is made up of, and if necessary, **rebuild the accounts using
  the same components the company originally used**." And every reversible adjustment stays
  reversible — impairment is removed from operating profit *and* stored in its own field.
- **Vendors disagree on the same problem**: Worldscope annualizes the income statement to 12
  months when a fiscal year changes; Compustat explicitly does not restate the prior annual year.
- **Fiscal-year convention**: Compustat assigns FYE Jan-May to the prior calendar year, Jun-Dec
  to the current; plus a day-of-month rule (prior month if the year ends on day 1-14). The modern
  availability-lag convention (Jensen/Kelly/Pedersen) is **4 months after fiscal period end**.

**Vintage instability is the biggest trap.** Compustat's common products *overwrite* on every
re-standardization. Lyle, Siano & Yohn measure: the same firm-period is revised ≥5 times on
average, items move by 9% of total assets, **the sign of earnings flips ~14% of the time**, and
~50% of 35 anomalies "yield materially different inferences across data vintages". Their
conclusion: "precise replication of prior studies using common Compustat products is nearly
impossible."

⚠️ **Citation correction.** The most-quoted paper on standardization-induced distortion —
Du, Huddart & Jiang, "Lost in standardization", *JAE* 76(1) 2023 — **is RETRACTED**
(verified via OpenAlex: `is_retracted: true`, canonical title prefixed "RETRACTED:"), and
retracted specifically for a methodological error in how the authors standardized the data.
Do not cite its magnitudes. Defensible substitutes: Ulbricht & Weiner (2005) on
Worldscope-vs-Compustat coverage and size bias; García Lara et al. (2006), *Abacus* — same
Ohlson design across 7 databases and 14 EU countries, results varied considerably and disparities
persisted even after matching observations; Ljungqvist, Malloy & Marston (2009), *JF*
"Rewriting History" on I/B/E/S vintage instability; Boritz & No (2020), *JIS* — as-reported XBRL
matches the 10-K to within **0.01%** while aggregators disagree at **6.5-7.7%**, but 48-63% of
XBRL line items are absent from the aggregators.

**Free asset worth adopting: the XBRL US Data Quality Committee rules** — 196 approved rules,
ruleset 30.0.3 (Aug 2026), free, versioned, executable via Arelle+Xule against your own
extracted instances. The families that map onto this exact problem: sign
(`DQC_0013/0014/0015/0174`), scale (`0095/0139/0192/0222`), footing
(`0117/0118/0126/0213/0214/0227`), cross-element identities (`0004/0009/0011/0128/0228/0231`),
and `0084` durational aggregation. Note Debreceny et al. (2010) found the **dominant** cause of
XBRL arithmetic failure is "inappropriate treatment… of underlying debit/credit assumptions" —
i.e. sign convention. `TAG.crdr` is the sign oracle.

### 3.5 Extension elements — smaller problem than feared

SEC DERA's own series (avg % of a filer's custom line-item tags on 10-K, face statements only):
11.26% (2012) → 9.27% (2017) → 12.61% (2022) → **11.80% (2025)**. Including notes: 21.44% (2012)
→ **17.04% (2025)**. Worst 2025 cohorts: BDCs 15.41%, Finance/Insurance/Real Estate 14.65%.
**Flat-to-declining over 13 years, not exploding.** And Debreceny et al. (2011) found **40% of
extensions were unnecessary** — a semantically equivalent us-gaap element already existed
(cf. DQC rule `0215`).

---

## Part 4 — The KPI shortlist, tiered by measured universality

`n_tick` = tickers with ≥1 non-null row (of 491). `cov` = mean per-ticker share of its own rows
that are non-null. **Flows carry a structural ~4/60 warm-up deficit** (a TTM needs 4 quarters),
so cov ≈ 0.96 is *full* coverage for a flow.

### Tier 1 — true universals (all 491 tickers, every accounting regime). 11 fields.

| field | n_tick | cov | note |
|---|---|---|---|
| `totalAssets` | 491 | .995 | `Assets` — the one balance-sheet total with no regime variant |
| `totalLiabilities` | 491 | .995 | derived from footing where untagged |
| `stockholdersEquity` | 491 | .995 | incl-NCI (consolidated) basis chosen |
| `cash` | 491 | .994 | 3 competing definitions |
| `totalDebt` | 491 | .981 | ⚠ §2.3 component-split defect |
| `sharesOutstanding` | 491 | .977 | cover-page tag is the only summable one for multi-class |
| `totalRevenue` | 490 | .964 | ⚠ 3 sector rebuilds + ASC-606 break + APA/ETN/VRT = 0 |
| `netIncome` | 490 | .960 | `ProfitLoss` preferred — matches OCF's basis (§3.2) |
| `operatingCashFlow` | 490 | .965 | |
| `ebitda` | 490 | .967 | **non-GAAP — no us-gaap element exists**; derived |
| `freeCashflow` | 490 | .965 | non-GAAP; OCF − capex |

### Tier 2 — near-universal (94-99%). 12 fields.
`pretaxIncome` (487/.952) · `incomeTaxExpense` (487/.948) · `longTermDebt` (487/.967 — ⚠ 21% of
rows incl leases) · `dilutedShares` (485/.960) · `epsDiluted` (484/.932) · `depAmort` (482/.941 —
use `DepreciationDepletionAndAmortization`) · `shortTermDebt` (481/.927 — ⚠ the defect lives
here) · `retainedEarnings` (479/.963) · `stockBasedComp` (478/.899) · `effectiveTaxRate`
(473/.816 — a ratio, never TTM-summed) · `capex` (470/.903 — ⚠ superset problem) ·
`employees` (490/.959 — 10-K body text, not XBRL)

### Tier 3 — regime-conditional (75-95%). Absence is structural. 13 fields.
`ppeNet` (466) · `interestExpense` (460) · `goodwill` (457) · **`operatingIncome` (452 — absent
for 38 of 76 Financials)** · `ppeGross` (449) · `accumulatedDepreciation` (441) ·
`intangiblesExGoodwill` (441) · **`currentAssets` (413) / `currentLiabilities` (414) — absent for
the same 78 unclassified-balance-sheet tickers** · `sellingGeneralAdmin` (403) ·
`accountsReceivable` / `accountsPayable` (392) · **`grossProfit` (379) / `costOfRevenue` (375)** ·
`inventory` (311) · `minorityInterest` (335 — needed to reconcile the two equity bases)

### Tier 4 — sector-exclusive, must be regime-gated, never universe-z-scored. 7 fields.
`researchAndDevelopment` (225 — universal only in Info Tech 65/72, Materials 17/26, Health Care
42/57) · `netInterestIncome` (113) · `rentalIncome` (96) · `provisionForCreditLosses` (88) ·
`premiumsEarned` (33) · `netInvestmentIncome` (25) · **`noninterestIncome` (23 — see §2.9)**

### Deliberately excluded from a first pass
Leases: `operatingLeaseLiability` reaches 486 tickers but cov is only **.513** — ASC-842-era only,
so any 2011-2026 series has a structural 2019 discontinuity. Off-balance-sheet debt needs the
pre-2019 commitment-footnote ladder to be comparable; that ladder (`leaseMaturity*`, 486 tickers)
exists and is currently consumed by nothing.

---

## Open questions for the planning phase

These are decisions for the user, not for me.

**1. Breadth vs comparability.** Three coherent packages:
- **A. Universal-only (11-23 fields, Tiers 1-2).** Every field means the same thing for every
  ticker; ratios comparable universe-wide with no gating. Loses margins, gross profit, working
  capital and R&D — i.e. most of the quality/profitability factor set.
- **B. Universal + regime-gated (36-45 fields, Tiers 1-3 with a regime column).** What every
  vendor actually does. Requires a template/regime field on every row and peer-relative scoring
  *within* regime, not across.
- **C. B + Tier 4 scored only inside its own cohort.** Maximum signal, maximum plumbing. The
  machinery already exists (`SECTOR_KPI_SCOPE`, `sector_features.py`, 57 `SECTOR_KPI_COLS`).

**2. Resolution mechanism.** Keep priority-ordered candidate lists, or drive roll-ups from the
filer's own calculation linkbase (§3.2) with the tag list as fallback? The linkbase resolved all
six hard regimes and would have caught APA. It is a different architecture, not a bigger tag list.

**3. Substrate.** Per-filing `filing.xbrl()` (current) or SEC bulk FSDS/FSNDS (§3.3)? Bulk gives
`cal` + `pre` + `dim` + `iprx`/`durp`/`datp` and removes ~30k network round-trips, at the cost of
a new ingest and a monthly cadence.

**4. Debt definition.** Gross debt ex-lease, ex-lease + finance leases, or + operating leases?
Whatever is chosen, the long and short legs must be on the *same* basis — today they are not (§2.3).

**FASB answered the operating-lease question explicitly, three times**, in ASU 2016-02's Basis for
Conclusions. **BC264**, verbatim: *"While both types of lease liabilities are financial liabilities,
**finance lease liabilities are the equivalent of debt, and operating lease liabilities are not
'debt like' but, rather, operating in nature.**"* FASB's stated basis is **bankruptcy treatment**,
not cash-flow certainty. Restated at **BC14(c)** (*"Topic 842 characterizes operating lease
liabilities as operating liabilities, rather than debt"*) and **BC383(c)(3)**.

Three caveats that keep this a decision rather than a mandate:
- It lives only in the Basis for Conclusions, which is **not authoritative GAAP**. No codified ASC
  842 paragraph says it. The codified hook is only `842-20-45-3`, which *prohibits* finance and
  operating leases sharing a line item.
- FASB did **not** call them non-financial — BC264 says both *are* financial liabilities.
- **BC48(c) is the counter-authority, and it is strong**: *"The majority of financial statement
  users engaged throughout the project, **including most credit analysts, supported a
  single-approach lessee accounting model**"* — because it matches *"the adjustments they make to
  lessees' financial statements … to approximate capital lease accounting."*

The practitioners split the same way, and it is **not** agency-agnostic:

| agency | operating leases in adjusted debt? | EBITDA |
|---|---|---|
| **S&P** (Apr 2019 criteria ¶44) | **Yes**, all sectors, at the reported liability, with a 3×-next-12-months floor override | **grossed up** (interest + depreciation add-back) |
| **Moody's** | **Yes**, reported liability + qualitative overlay | **grossed up** |
| **Fitch** (Dec 2024 update) | **No for most sectors**; reported liability only where adjusted | lease cost stays operating; uses **EBITDAR** |

Also correcting a common belief: the **8×-rent multiple was Moody's, not S&P**, and it varied by
**sector, not region** (5×/6×/8×/10×, revised down in June 2015). S&P used present value from at
least 2008 — the Nov 2013 change was rate *standardisation* to a fixed 7%, not multiples→PV.

**5. Cash definition.** `CashAndCashEquivalentsAtCarryingValue` (BS), the ASU-2016-18
restricted-inclusive total, or cash + short-term investments? Note `CashAndDueFromBanks` is a
bank's *first* line, not its total.

**ASU 2016-18 makes the restricted-inclusive total the cash-flow-statement anchor.** `230-10-45-4`:
the statement of cash flows *"shall explain the change during the period in the total of cash, cash
equivalents, and amounts generally described as restricted cash or restricted cash equivalents."*
And `230-10-45-5`: transfers between them *"are not part of the entity's operating, investing, and
financing activities."* Effective for public filers **FY2018**, applied **retrospectively**. FASB
deliberately **did not define restricted cash**.

Verified consequence for the repo: `CashAndCashEquivalentsPeriodIncreaseDecrease` is **absent from
us-gaap 2025** (deprecated), while
`CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect`
is the current root — which is exactly the order the repo's `cashPeriodChange` candidate list
already uses. `230-10-50-8` requires filers to reconcile the total back to the balance-sheet line
items, so the split is recoverable.

**6. Vintage policy.** Append-only vintages (Compustat Snapshot's design) or overwrite (current)?
The measured cost of overwriting elsewhere: earnings sign flips ~14% of the time and ~50% of
anomalies change inference across vintages. `test_fundamentals_point_in_time.py` is already red
on exactly this.

**7. Where does zero-vs-missing get decided?** Adopt a Compustat-style `_DC` reason code per
value, with a "combined into" destination? Today a blank is ambiguous between not-disclosed,
not-applicable-for-regime, rolled-into-another-line, and failed-a-guard.

**8. Does the live/code table split get resolved by rename or rebuild?** (§1.4)

---

## Corrections to prior recorded knowledge

- The `fundamentals_identities.py` / `fundamentals_quality.py` / `classify_kinks` quality gate
  recorded as built on 2026-08-12 **has never existed in this tree**. Memory corrected.
- Du, Huddart & Jiang (2023) is **retracted**; do not cite its magnitudes.
- `pre.txt` **does** carry a `negating` column (a source had this as not found).
- FSNDS is **monthly** now; the quarterly `*q*_notes.zip` pattern 404s.

## Gaps this research did not close

- **Compustat's exact `CAPX` include/exclude list** — the reference definition the factor
  literature actually used. Not established. Any capex-based factor ported from a paper is
  approximate until this is reconciled.
- Whether Compustat `COGS` includes `DP`, and whether `XSGA` includes `XRD`. The us-gaap model
  puts R&D *outside* SG&A; if Compustat does the opposite, RMW numerators differ.
- Exact formulas from Novy-Marx (2013), Sloan (1996), Piotroski (2000), Ball et al. (2016).
- MSCI/Barra descriptor treatment of financials.
- Compustat's `REVT` definition under `INDFMT=FS` for a bank — could be gross interest income +
  noninterest income, or net revenue. They differ by the whole interest-expense leg. Do not
  assume it equals `InterestIncomeExpenseNet + NoninterestIncome` without checking.
- ASC 606-10-15-2's exact scope-exclusion text for insurance (Codification is login-walled); the
  *consequence* is measured and solid.
- ASU 2018-12 (LDTI) substantive changes and effective dates (FASB project page 403s); the
  *evidence of the break* is measured and solid.

⚠️ **Caveat on one artifact.** `us-gaap-doc-2025.xml` as fetched holds **14,899** documentation
labels against **17,326** elements in `elts/us-gaap-2025.xsd`. Every definition quoted in this
document was read from an entry that *was* present, so the strings are sound. But **absence from
that file is not evidence of absence from the taxonomy** — existence claims were re-checked
against the schema itself.

⚠️ **Do not cite these ASC paragraph numbers without checking FASB directly.** The Codification is
login-walled behind a reCAPTCHA'd SPA and cannot be verified programmatically, so the following
numbers are inferred from position or from secondary reproductions, even though the underlying
*text* is verbatim from FASB's own FAS/ASU PDFs:
- `ASC 210-10-45-1` / `45-2` — which one carries *"A total of current assets shall be presented in
  classified balance sheets"* is unresolved (its liabilities twin is attributed to `45-5`).
- `ASC 210-10-05-1` vs `05-4` — two Deloitte renderings conflict.
- `ASC 835-20-25-1` / `25-2`, and the `25-4`/`25-5` split — **zero EDGAR citations exist**, so no
  public quotation is available. The source text is FAS 34 ¶6-7. Verified separately: the
  capitalization-period paragraph is **`25-3`**, not `25-1`.
- `ASC 985-20-25-1` / `25-3` — only `25-2` is confirmed (FASB's own ASU 2025-06 **BC46** cites
  `985-20-25-2(a)(3)`, which maps to FAS 86 ¶4(a)(3)).
- `ASC 840-10-50-2` — obtained only as its predecessor, FAS 13 ¶16; SEC staff letters cite the
  subtopic inconsistently (`840-20-50` vs `840-10-50-2`).
- `CashAndCashEquivalentsPeriodIncreaseDecrease` deprecation **year** — secondary only (Calcbench
  says 2022). Its absence from us-gaap 2025 is verified; the year is not.

Also unreached in primary form: **Moody's current financial-statement-adjustments methodology**
(Cloudflare-blocked; its post-IFRS-16 treatment rests on one high-quality secondary source, and the
document's vintage is unconfirmed), Moody's "Guideline Rent Expense Multiples" table, and Fitch's
December 2024 "Appendix 1" criteria update.

---

## Part 5 — Industry regimes: how the "universal" KPIs actually break

### 5.1 The taxonomy publishes the industry mapping you need

Two machine-readable files answer the regime question deterministically, instead of by heuristic:

- **`us-gaap-ref-2025.xml`** (reference linkbase) maps every element to its ASC topic *and, for
  SEC-materials sections, to the exact Reg S-X caption number*. So `us-gaap:InterestIncomeExpenseNet`
  is *formally declared* to be Reg S-X Rule 9-04 caption 10:
  ```xml
  <codification-part:Topic>942</codification-part:Topic>
  <ref:Subparagraph>(SX 210.9-04(9))</ref:Subparagraph>
  ```
- **`stm/`** holds 26 statement modules — and critically, **every one imports the single shared
  element schema** (`../elts/us-gaap-std-2025.xsd`). Only the presentation, calculation and
  definition linkbases differ.

**That is the structural answer to §2.8.** `AssetsCurrent` *exists* for a bank; the bank simply
never presents it, because its role is `108000 – Statement of Financial Position, Unclassified –
Deposit Based Operations`. Which is exactly why your extractor sees "no `AssetsCurrent` ever"
rather than a validation error.

The regime-relevant roles: `sfp-dbo` 108000 (banks) · `sfp-ibo` 108200 (insurers) ·
`sfp-sbo` 112000 (broker-dealers) · `sfp-clreo` 110000 / `sfp-ucreo` 110200 (real estate,
**both** exist) · `soi-int` 132001 (interest-based revenue) · `soi-ins` 136000 ·
`soi-reit` 145000 · `scf-dbo` 160000 · `scf-inv` 164000 · `scf-re` 170000.

Element counts by ASC topic — this is why REITs hurt most:

| ASC topic | elements referencing it |
|---|---|
| 944 Insurance | **1,082** |
| 942 Depository & Lending | **818** |
| 946 Investment Companies | 624 |
| 932 Extractive – Oil & Gas | 251 |
| 980 Regulated Operations | 125 |
| 970 Real Estate – General | 65 |
| **974 Real Estate – REITs** | **6** |

Banks and insurers get a purpose-built element set anchored in Reg S-X Articles 9 and 7. REITs get
general-purpose elements plus a Nareit non-GAAP layer with **no taxonomy representation at all** —
`FundsFromOperations`, `AdjustedFundsFromOperations`, `NetOperatingIncome`, `EBITDAre` are all
verified absent from us-gaap 2025.

### 5.2 Banks — and the single most dangerous comparability bug found

Reg S-X **Rule 9-01**: Article 9 *replaces* Article 5 for bank holding companies. Rule 9-04
caption order, with FASB-declared element mapping: total interest income (5) →
`InterestAndDividendIncomeOperating` · total interest expense (9) → `InterestExpense` ·
**net interest income (10)** → `InterestIncomeExpenseNet` · provision (11) →
`ProvisionForLoanLeaseAndOtherLosses` · other income (13) → `NoninterestIncome` ·
other expenses (14) → `NoninterestExpense`.

**`us-gaap:Revenues` is populated for banks but means something different.** Measured, CY2024
frames: JPM `Revenues` = **$177,556 M**, USB = **$27,455 M** — both equal to reported *total net
revenue*, i.e. already net of interest expense. XOM ($349,585 M), PGR ($75,372 M), SO ($26,724 M),
BRK ($371,433 M) are all gross. So the same tag silently switches basis by regime.

**Your 11-of-34 finding, explained.** Rule 9-04 captions 13/14 are "Other income"/"Other expenses"
— a *menu*, not a mandate. A filer presenting fee income as several disaggregated lines (service
charges, card income, trust fees, investment-banking fees) may tag each line and **never tag the
subtotal**. Frames CY2024 counts: `InterestIncomeExpenseNet` **912** entities vs `NoninterestIncome`
**429** — the gap is exactly that population.

Recommended build, with the branch recorded:
`coalesce(Revenues, InterestIncomeExpenseNet + NoninterestIncome, InterestIncomeExpenseNet)`.
**Never fall back to `InterestAndDividendIncomeOperating`** — that is gross interest income and
would inflate revenue by the entire interest-expense leg.

**Deposits are not debt.** `us-gaap:Deposits` is declared against `SX 210.9-03(12)`. Bank debt =
`LongTermDebt` (9-03(16)) + `ShortTermBorrowings` (9-03(13)) + `AdvancesFromFederalHomeLoanBanks`
(ASC 942-470-45-1) + `SubordinatedDebt` + `SecuritiesSoldUnderAgreementsToRepurchase`.

**Bank capex is not reliably reconstructible.** JPM, BAC and USB do **not appear at all** in the
`PaymentsToAcquirePropertyPlantAndEquipment` CY2024 frame; large banks bury premises purchases in
"all other investing activities". So bank FCF is not computable from XBRL.

**Regulatory capital vanishes from the API.** `companyconcept` HTTP status:

| | `CommonEquityTierOneCapitalRatio` | `CommonEquityTierOneCapital` | `TierOneRiskBasedCapitalToRiskWeightedAssets` |
|---|---|---|---|
| JPM | **404** | **404** | 200 |
| USB | **404** | **404** | **404** |
| BAC | **404** | **404** | **404** |

Because ASC 942-505-50-1 disclosure is given for the holding company *and* each significant bank
subsidiary, every fact is qualified by `dei:LegalEntityAxis` — and `companyfacts`/`frames` publish
only dimensionally-unqualified facts. Confirms the repo's existing "companyfacts drops dimensioned
facts" note, with a named consequence.

**A live restatement trap.** `us-gaap:Revenues` for Bank of America (CIK 70858):
FY2023 = 98,581 M as filed in 2024 and 2025, but **102,769 M** as re-presented in the FY2025 10-K
— and the `frames` API returns the *restated* figure. FY2024 likewise 101,887 → **105,856 M**.
So a frames-based pull and an as-filed pull disagree by ~4% on the same ticker-year.

Meaningless for banks: gross margin (`GrossProfit`/`CostOfRevenue` are declared against
`220-10-S99-2`/`235-10-S99-1`, i.e. Article 5 SEC materials), EBITDA, FCF, net debt/EBITDA,
working capital. Used instead: NIM, efficiency ratio, PPNR, ROTCE, CET1 — **none of which exist as
us-gaap elements** (verified absent: `EfficiencyRatio`, `NetInterestMargin`,
`PreProvisionNetRevenue`, `ReturnOnTangibleCommonEquity`, `TangibleCommonEquity`).

### 5.3 Insurers

Rule 7-04 captions → elements: premiums (1) → `PremiumsEarnedNet` · net investment income (2) →
`NetInvestmentIncome` (*"Excludes realized gain (loss) on investments"*) · realized gains (3) →
`RealizedInvestmentGainsLosses` · benefits/claims (5) → `PolicyholderBenefitsAndClaimsIncurredNet`.
Rule 7-03 has **no current/noncurrent split**.

**The ASC-606 fee slice, quantified.** MetLife CY2024: `Revenues` = **$70,986 M**,
`RevenueFromContractWithCustomerExcludingAssessedTax` = **$2,245 M** — the ASC 606 element captures
**3.2%**. Any pipeline preferring the contract element over `Revenues` understates a life insurer
by ~30×. Of PGR/MET/AFL, only MET appears in that frame at all.

**LDTI is a hard 2021 break.** MetLife's `MarketRiskBenefitLiabilityAmount` facts begin at
`end: 2021-01-01` — the transition date. So `LiabilityForFuturePolicyBenefits`, DAC and net income
are on **two incompatible bases** either side of 1 Jan 2021. A pre-2021 vs post-2021 panel for a
life insurer is a level break, not a trend.

**Unusually, the ratios are tagged concepts** (denominator = non-life net premiums earned, units
`pure` not `USD`): `LossRatio`, `UnderwritingExpenseRatio`, `GeneralAndAdministrativeExpenseRatio`,
`CombinedRatio`. Note the exact amortization element is
`DeferredPolicyAcquisitionCostAmortizationExpense` (singular "Cost"), with near-identical decoys.

### 5.4 REITs

**Nareit FFO, verbatim** ([2018 White Paper Restatement](https://www.reit.com/sites/default/files/2018-FFO-white-paper-(11-27-18).pdf),
effective 2018-12-15): net income (GAAP) excluding — depreciation and amortization related to real
estate; gains and losses from the sale of certain real estate assets; gains and losses from change
in control; impairment write-downs of certain real estate assets and investments in entities where
the impairment is directly attributable to decreases in the value of depreciable real estate.

What is **in** the add-back: depreciation of real estate, amortization of capitalized leasing
expenses, tenant allowances/improvements, in-place-lease intangibles, and depreciation on operating
properties on ground-leased land. What is **out**: *"amortization of above and below market leases
… and amortization of deferred financing costs and premiums and discounts on debt"*, plus
non-real-estate D&A (software, office improvements, furniture).

Nareit is explicit that FFO *"is not intended to be used as a measure of the cash generated by a
REIT nor of its dividend-paying capacity."*

**AFFO is not standardized, by Nareit's own refusal**: *"Nareit believes that there is not adequate
consensus among preparers and users … to allow agreement on a single definition of FAD, CAD, or
AFFO."* ⇒ compute it yourself from components; never compare two companies' reported AFFO.

**EBITDAre** ([Nareit, Sept 2017](https://www.reit.com/sites/default/files/EBITDAre_Whitepaper(9-18-17).pdf)):
net income + interest + tax + D&A ± losses/gains on disposition of depreciated property +
impairments of depreciated property, ± share of unconsolidated affiliates. Two differences from
FFO that matter: the D&A add-back is **all** D&A, not just real-estate; and it includes **100%** of
consolidated affiliates' EBITDAre, deliberately, so it pairs with consolidated debt.
⇒ **net debt / EBITDAre** is the REIT leverage metric, and neither term is a tagged element.

**REIT capex is a four-way split** (Nareit: *"corporate items, existing properties, development of
new properties, and acquisitions"*), and the taxonomy makes **no semantic distinction** — every
payment element carries the same generic ASC 230-10-45-13 reference. `PaymentsForCapitalImprovements`
has only **138** filers vs `PaymentsToAcquirePropertyPlantAndEquipment` **3,937**.

MAA measured, FY2025 10-K instance:
```
PaymentsToAcquirePropertyPlantAndEquipment                0
PaymentsToAcquireRealEstate                              0
PaymentsToDevelopRealEstateAssets                        0
PaymentsForCapitalImprovements                           6   "Capital improvements and other"
PaymentsToAcquireInProcessResearchAndDevelopment         6   "Development costs" ($272,030K)  ← !
PaymentsToAcquireResidentialRealEstate                  10
maa:PaymentsToAcquireRealEstateAndOtherAssets            6   company extension
AssetsCurrent / LiabilitiesCurrent / GrossProfit / CostOfRevenue   0
```
**MAA tags $272 M of development capex as `PaymentsToAcquireInProcessResearchAndDevelopment`.** A
name-keyed capex extractor either misses it or books a multifamily REIT as having R&D spend.

**The Up-C / dual-registrant answer.** MAA's FY2025 instance carries **two**
`dei:EntityCentralIndexKey` facts (0000912595 parent, 0001581776 LP), but **all 1,417
`xbrli:identifier` values are the parent's CIK**. The LP is scoped by `dei:LegalEntityAxis` with a
**company-extension member** `maa:LimitedPartnershipMember`, carrying the LP's full primary
statements — not a footnote.

⇒ **Take the dimensionally-unqualified (default-member) facts, and filter on the AXIS, not the
member** (a fixed us-gaap member list cannot catch an extension member). One rule fixes both this
and the bank holdco/subsidiary case — but it is also exactly what makes CET1 unreachable, so
regulatory capital needs a deliberate exception.

Worse at **Southern Company**: six registrant CIKs in one instance, 3,579 `LegalEntityAxis`
occurrences, all identifiers = parent. And four of those SEC registrants that file 10-Ks return
**404 from `companyconcept`** — Alabama Power, Georgia Power, Mississippi Power, Southern Company
Gas have no `companyfacts` at all. By-CIK universe construction yields silent nulls, not an error.

**Practice varies on the classified balance sheet**: AMT reports `AssetsCurrent`; MAA and SPG do
not. FASB ships both the classified (110000) and unclassified (110200) real-estate roles, so both
are correct. Your 23-of-31 is the expected majority, and the 8 exceptions are also right.

### 5.5 Oil & gas

**Successful efforts is the S&P 500 norm.** Read from filings: EOG *"successful efforts method of
accounting"*; DVN likewise. Frames CY2024Q4I: `OilAndGasPropertyFullCostMethodNet` **28** entities
vs `…SuccessfulEffortMethodNet` **41** — and the full-cost list is entirely small/micro-cap
(US Energy, Evolution Petroleum, Zion, PEDEVCO, W&T Offshore, SandRidge, Vital, Viper, Kimbell…).
**No S&P 500 constituent appears.**

The difference in one line: under successful efforts, unsuccessful exploration hits the income
statement as `ExplorationExpense`; under full cost it is capitalized and returns only via DD&A or
a Rule 4-10(c)(4)(i) ceiling-test write-down (PV of proved reserves at a **10% discount**, prices
= *"unweighted arithmetic average of the first-day-of-the-month price"* over 12 months). Same
drilling programme ⇒ different opex, capex, DD&A and book equity.

Detection: `OilAndGasAccountingMethodFullCostOrSuccessfulEffortsExtensibleEnumeration` exists —
**but EOG doesn't use it** (0 occurrences). Fall back to element presence.

E&P capex is per-company: XOM uses `PaymentsToAcquirePropertyPlantAndEquipment`; EOG uses
`PaymentsToAcquireOilAndGasPropertyAndEquipment` + `…OilAndGasProperty` +
`…OtherPropertyPlantAndEquipment`; DVN uses `PaymentsToAcquireProductiveAssets` **plus a company
extension** `dvn:PaymentsToPropertyAndEquipmentAcquisition`. All share the same generic ASC
reference, so the taxonomy gives no help disambiguating.

**SMOG and PV-10 are effectively untagged** — not merely "dimensioned and dropped". Verified
absent from us-gaap 2025: any element matching `Standardized*`, `FutureNetCashFlow*`,
`DiscountedFuture*`. EOG's FY2025 10-K contains the phrase in text, tags
`OilAndGasExplorationAndProductionIndustriesDisclosuresTextBlock` **0 times**, and tags only two
reserve-related facts. Reserve data needs HTML/PDF table parsing or a reserves vendor.

AROs (`AssetRetirementObligation`, ASC 410-20) are debt-like — discounted, non-cancellable, with
interest-like accretion. Rating agencies add them to adjusted debt.

### 5.6 Utilities

**AFUDC-equity is non-cash income.** `PublicUtilitiesAllowanceForFundsUsedDuringConstructionCapitalizedCostOfEquity`
(18 facts in SO's 10-K) lets the utility book a return on construction-work-in-progress before the
asset is in service — raising net income and EPS with **zero cash**, and capitalizing into the
asset base. During a heavy build cycle this is a material share of reported earnings.
⇒ **strip AFUDC-equity before using utility net income in any quality or cash-conversion signal.**

**Regulatory assets distort book equity.** Under ASC 980 a cost the regulator has agreed to allow
in future rates is capitalized rather than expensed, inflating assets and equity relative to an
unregulated peer. So ROE and P/B are not comparable to an industrial's, and debt/total-capital is
distorted. SO's dominant element is `NetRegulatoryAssets` (395 facts), not `RegulatoryAssets`
(24) — and the `RegulatoryAssets` CY2024Q4I frame has only **55** entities, with **SO present but
NEE absent**. Not reliably retrievable from a single tag.

Revenue is clean ordinary ASC 606; margins are not comparable, because fuel and purchased power
are pass-through under fuel-adjustment clauses, the allowed return is set on rate base, and AFUDC
inflates the numerator. Use rate-base growth, allowed vs earned ROE, and FFO/debt.

### 5.7 GICS mapping traps

Four, all verified against the [MSCI GICS Methodology, March 2023](https://www.msci.com/documents/1296102/11185224/GICS+Methodology+2023.pdf):

1. **Mortgage REITs are `40204010`, under Financials (40), not Real Estate (60).** FFO is
   meaningless for them; book value and net interest spread are the metrics.
2. **The equity-REIT industry codes are not a contiguous run**: `601010, 601025, 601030, 601040,
   601050, 601060, 601070, 601080`. Old `601020` was retired and **Industrial REITs sits at
   601025**. A "step by 10" enumeration silently drops Industrial REITs.
3. **Insurance Brokers (40301010)** — MMC, AON, AJG — sit inside GICS Insurance but are **Article 5
   fee businesses**. Combined-ratio logic produces nulls; they have normal revenue, gross margin
   and EBITDA.
4. **Financial Exchanges & Data (40203040)** and **Transaction & Payment Processing (40201060)** —
   ICE, CME, NDAQ, V, MA, FIS — are Financials by GICS but **Article 5 companies with ordinary ASC
   606 revenue and real gross margins**. Mastercard reports `Revenues`, `OperatingIncomeLoss`,
   `AssetsCurrent` and `PaymentsToAcquirePropertyPlantAndEquipment` — it behaves exactly like an
   industrial. **Do not route GICS 40 to a bank template.**

Also: **Telecom Tower REITs (60108030)** — AMT/CCI/SBAC — file like industrials (AMT reports
`AssetsCurrent`, `OperatingIncomeLoss`, PP&E capex *and* the ASC-606 revenue element). Tower,
data-center and timber REITs are the most likely to break a blanket "Real Estate ⇒ REIT template"
rule.

**Hybrids.** Berkshire is the canonical case: reports `Revenues` and PP&E capex and
`OperatingLeaseLeaseIncome`, but **no `AssetsCurrent`** — so it fails an industrial template on
the balance sheet and an insurance template on the income statement, while its GICS is
Financials/Insurance. Compustat's answer is the `INDFMT` key, which is why a hybrid can appear
twice for the same `gvkey`/`datadate`. Note WRDS defaults to `INDL` only, which **silently drops
financial companies from a query**.

### 5.8 The matrix — 22 KPIs × 6 regimes

`=` same definition and elements as industrials · `≠` different elements · `✗` not meaningful.

| # | KPI | Industrials | Banks | Insurers | REITs | E&P | Utilities |
|---|---|---|---|---|---|---|---|
| 1 | Revenue | ASC-606 elements / `Revenues` | **≠** `Revenues` means **NET** revenue. Build `InterestIncomeExpenseNet` + `NoninterestIncome` | **≠** `PremiumsEarnedNet` + `NetInvestmentIncome` + `RealizedInvestmentGainsLosses` + fees. ASC-606 elt = 3.2% of MET | **≠** usually `Revenues`; else `OperatingLeaseLeaseIncome`. `RealEstateRevenueNet` deprecated | **=** `Revenues` | **=** clean, but fuel is pass-through |
| 2 | Gross profit / margin | `GrossProfit`, `CostOfRevenue` | **✗** use efficiency ratio | **✗** use loss / expense / `CombinedRatio` | **✗** use NOI (not an element) | **✗** use cash margin per boe | **✗** fuel pass-through |
| 3 | Operating income | `OperatingIncomeLoss` | **✗** absent 38/76. Use PPNR | **≠** rarely tagged. Use underwriting result | **=** AMT/SPG yes, MAA no | **=** | **=** |
| 4 | EBITDA | computed; no element | **✗** interest is revenue | **✗** use ROE, combined ratio | **✗** use **EBITDAre** | **≠** DD&A differs by method | **≠** use FFO/debt |
| 5 | D&A | `DepreciationDepletionAndAmortization` | **=** immaterial | **≠** dominated by DAC amortization | **≠** must split real-estate vs non-real-estate vs above/below-market | **≠** `DepletionOfOilAndGasProperties` | **=** + regulatory amortization |
| 6 | Net income | `NetIncomeLoss` / `ProfitLoss` | **=** but pair with ROTCE | **≠** LDTI break at 2021-01-01 | **✗** nearly useless — use FFO | **≠** ceiling-test / method distortion | **≠** AFUDC-equity inflated |
| 7 | EPS | standard | **=** | **=** | **✗** use FFO/share, **add OP units** | **=** | **≠** AFUDC-inflated |
| 8 | Cash | `CashAndCashEquivalentsAtCarryingValue` | **≠** `CashAndDueFromBanks` + `InterestBearingDepositsInBanks` + fed funds sold | **≠** `Cash` (7-03(2)); portfolio ≠ cash | **=** | **=** | **=** |
| 9 | Total debt | `LongTermDebt` (excl leases) + `DebtCurrent` | **≠ deposits are NOT debt**. + FHLB advances, sub debt, repo | **≠** only 7-03(16). Policy & separate-account liabilities are not debt | **≠** secured/mortgages + unsecured + revolver + CP | **≠** add AROs | **≠** add decommissioning; reg. liabilities are not debt |
| 10 | Net debt / leverage | net debt ÷ EBITDA | **✗** CET1, Tier 1 leverage, SLR | **✗** financial-leverage ratio, RBC | **≠** net debt ÷ **EBITDAre** | **≠** net debt ÷ EBITDAX | **≠** **FFO/debt** |
| 11 | Working capital | `AssetsCurrent` (5-02 cap. 9) | **✗ structurally absent** (Rule 9-03) | **✗ absent** (Rule 7-03) | **⚠ varies** — MAA ✗, SPG ✗, AMT ✓ | **=** | **=** |
| 12 | Total assets | `Assets` | **=** | **=** | **=** | **=** | **≠** inflated by regulatory assets |
| 13 | Book equity / BVPS | `StockholdersEquity` | **≠** use tangible common equity | **≠** use BVPS **ex-AOCI** | **✗** book ≈ depreciated cost; use NAV | **≠** method-dependent | **≠** distorted by regulatory assets |
| 14 | Capex | `PaymentsToAcquirePropertyPlantAndEquipment` | **✗ not reliably available** | **✗** not used | **≠ four-way split**; PP&E elt usually absent | **≠ per-company** + extensions; accrual ≠ cash | **=** but strip AFUDC-equity |
| 15 | Free cash flow | CFO − capex | **✗** meaningless | **✗** use statutory dividend capacity | **✗** use AFFO (non-standardized) | **≠** compute on accrual capex | **≠** structurally negative in build cycles |
| 16 | Interest expense | financing cost | **✗ it is a cost of revenue** — never add back | **≠** debt service only | **=** exclude deferred-financing amort. from FFO | **=** watch capitalized interest | **≠** partly recovered in rates; AFUDC-debt |
| 17 | Effective tax rate | `IncomeTaxExpenseBenefit` ÷ pretax | **=** watch muni income | **=** | **✗ near-zero by REIT election** | **=** | **≠** excess deferred taxes flow to customers |
| 18 | ROE / ROA | NI ÷ equity | **≠ ROTCE** | **=** ROE is *the* metric | **✗** FFO ÷ equity | **=** method-dependent | **≠** earned vs allowed ROE |
| 19 | Credit / provision | n/a | **≠** `ProvisionForLoanLeaseAndOtherLosses`, NCO, ACL/loans | **≠** `PolicyholderBenefitsAndClaimsIncurredNet` + reserve development | n/a | n/a | n/a |
| 20 | Shares / dilution | `WeightedAverageNumberOfDilutedSharesOutstanding` | **=** | **=** | **≠ add OP units** (extension element) | **=** | **=** |
| 21 | Entity selection | single registrant | **⚠** axis splits holdco vs bank sub — why CET1 404s | **⚠** axis splits holdco vs insurance subs | **⚠⚠** MAA 2 CIKs, LP via **extension member** | **=** | **⚠⚠** Southern 6 registrants, 4 CIKs 404 |
| 22 | Headline metric | revenue growth, margin, FCF conversion | NIM, efficiency, PPNR, ROTCE, CET1 — **no elements** | combined/loss/expense ratio (**elements exist**), BVPS ex-AOCI, float | FFO, AFFO, NOI, EBITDAre, NAV — **no elements** | production/boe, reserve replacement, PV-10 (**untagged**), F&D cost | rate-base growth, allowed vs earned ROE, FFO/debt |

---

## Part 6 — The regulatory layer: why some "missing" tags were never required

Primary text from the **2025 annual CFR edition** (govinfo XML, `CFR-2025-title17-vol3`) and from
FASB's own ASU PDFs.

### 6.1 Gross profit and operating income were never required line items — for anyone

Run against the text of **Rule 5-03** (17 CFR 210.5-03, "Statements of comprehensive income",
which governs *all* commercial and industrial registrants):

- `"gross profit"` → **0 occurrences**
- `"operating income"` → **1 occurrence, inside caption 7 "Non-operating income"**

The enumerated captions run 1–9 (revenue and expense) straight into **caption 10, "Income or loss
before income tax expense and appropriate items below"**. There is no intervening gross-profit or
operating-income subtotal, and no rule prescribing where one would go.

**But cost of revenue *is* required.** Captions 1 (Net sales and gross revenues) and 2 (Costs and
expenses applicable to sales and revenues) are a matched five-way pair — 1(a)–(e) ↔ 2(a)–(e) — and
5-03(b) enforces the pairing: if revenue subcaptions are combined, *"related costs and expenses as
described under § 210.5-03.2 **shall be combined in the same manner**."*

⇒ `us-gaap:GrossProfit` and `us-gaap:OperatingIncomeLoss` are **elective tags for every filer**,
not just for banks. Their absence is not a filing defect and must never be read as missing data. It
must be reconstructed from captions 1–9. That is the authoritative basis for §2.8's measured
`operatingIncome` gap and for `grossProfit`/`costOfRevenue` sitting at only 379/375 tickers.

### 6.2 The classified balance sheet is conditional, and there is no "Total liabilities" caption

**Rule 5-02** captions, verbatim, including the rule's own inconsistent capitalisation:
`Current Assets, when appropriate` … **`9. Total current assets, when appropriate.`** …
`Current Liabilities, When Appropriate` … **`21. Total current liabilities, when appropriate.`**

That "when appropriate" is the legal mechanism by which an in-scope Article 5 registrant — a REIT,
say — lawfully files an unclassified balance sheet. (Note: the common secondary claim that
"S-X 5-02 *requires* a classified balance sheet" is a simplification the primary text does not
support. Cite the CFR.) `ASC 210-10-15-3` agrees: the classified-balance-sheet guidance *"applies
only when an entity is preparing a classified balance sheet."*

**The citable authority for "this industry doesn't classify" is `17 CFR 210.1-02(bb)(1)(i)`**, which
requires summarized financial information to give current/noncurrent splits *"(**for specialized
industries in which classified balance sheets are normally not presented**, information shall be
provided as to the nature and amount of the majority components of assets and liabilities)"*. Only
6 filers invoke it corpus-wide on EDGAR, and two are exactly the cohorts measured in §2.8:
- **Insurance** — AmTrust Financial, CORRESP 2018-04-30: *"as is the case with other companies in
  the insurance industry, the Company does not characterize assets and liabilities as current or
  noncurrent."*
- **REIT** — Aimco Properties LP, CORRESP 2010-10-22: *"in accordance with Section 1-02(bb)(1) of
  Regulation S-X, **classified balance sheets are normally not presented for real estate
  companies**."*

This is the cleanest primary basis for encoding expected absence as a **structural flag** rather
than a null count — the design implication flagged in §2.8.

**Rule 5-02 has no "Total liabilities" caption** — only 18 (Total assets) and 32 (Total liabilities
and equity). ⇒ `us-gaap:Liabilities` is elective, which is why `_derive_history` has to fall back to
`totalAssets − stockholdersEquity`. That fallback is correct by construction, not a workaround.

Scope confirmation, **Rule 5-01**: Articles 5-01→5-04 apply to all persons **except** registered
investment companies, employee plans, insurance companies (Art. 7), and bank holding companies and
banks (Art. 9). **Real estate entities and utilities are Article 5 filers** — the utility "flavour"
is FERC Uniform System of Accounts *caption vocabulary* accommodated inside the Article 5 format by
5-02.13(b), not a different presentation regime. Measured: 13 of 13 major US utilities tag
`AssetsCurrent` continuously.

Also note **banks and insurers do get a debt-tenor split** — Rule 9-03 caption 13 (Short-term
borrowing) vs 16 (Long-term debt); Rule 7-03.16(a) requires short-term and long-term separately. So
short/long debt KPIs are computable for them even though working capital never is.

### 6.3 Non-GAAP measures: what the SEC actually permits

**EBIT and EBITDA are the only two non-GAAP measures the SEC names by exception.**
Item 10(e)(1)(ii)(A) prohibits excluding cash-settled charges from non-GAAP *liquidity* measures
*"other than the measures earnings before interest and taxes (EBIT) and earnings before interest,
taxes, depreciation, and amortization (EBITDA)"* — carved out, per the adopting release,
*"because of their wide and recognized existing use."*

**Free cash flow — C&DI 102.07, verbatim on the point that matters:** deducting capex from
operating cash flow does not violate Item 10(e), *"However, companies should be aware that **this
measure does not have a uniform definition and its title does not describe how it is calculated**…
Also, free cash flow is a **liquidity** measure that **must not be presented on a per share
basis**."*

⇒ The SEC's own position is that FCF has no uniform definition. Whatever this pipeline picks is a
house convention that must be written down, and it is a liquidity measure — so per-share FCF is
out of bounds for anything published.

Two more C&DIs that bear on a KPI catalogue: **100.04** treats adjustments that change GAAP
*recognition and measurement* as individually tailored and potentially misleading (including
*"changing the basis of accounting for revenue or expenses … from an accrual basis … to a cash
basis"*). **100.05** flags labelling a computed measure the same as a GAAP subtotal — *"such as
'Gross Profit' or 'Sales'"* — as a violation.

### 6.4 Two cash-flow traps in capex and interest

**Capitalized interest is inside investing capex, and inside capitalized software.**
`ASC 230-10-45-13(c)` puts in investing outflows the payments to acquire PP&E and other productive
assets *"**including interest capitalized as part of the cost of those assets**"*, and
`230-10-50-2` requires interest paid to be disclosed *"**(net of amounts capitalized)**"*.
`ASC 350-40-30-1(c)` puts capitalized interest inside internal-use software too.

Recovering gross interest is harder than it looks. `ASC 835-20-50-1(b)` requires disclosure of total
interest cost incurred whenever any is capitalized — but measured element counts (CY2023):
**`InterestCostsCapitalized` 437 facts vs `InterestCostsIncurred` 151 facts.** Roughly 3× more
filers tag the capitalized amount than the gross. So gross interest generally has to be computed as
`InterestExpense + InterestCostsCapitalized`.

**`45-13(c)` also pushes seller-financed principal into *financing*** — so PP&E additions ≠
investing outflow whenever a purchase is vendor-financed. Combined with the `230-10-50-3`/`50-4`
noncash-disclosure requirement, this means **`PaymentsToAcquirePropertyPlantAndEquipment` is not
gross PP&E additions.** Accrued-but-unpaid capex is deliberately excluded and disclosed as noncash.

**Capitalized software splits across two cash-flow sections.** Internal-use software development →
**investing**, under `230-10-45-13(c)`'s "other productive assets", usually buried inside
"Purchases of property and equipment". Cloud-computing-arrangement implementation costs → follow the
hosting fee, in practice **operating**, under `ASC 350-40-45-3`. Confirmed by two SEC staff
exchanges (Robert Half 2022, ServiceTitan 2024). ⇒ a single "capitalized software" concept silently
mixes operating and investing cash flow.

### 6.5 Effective-date discontinuities to encode as regime boundaries

| standard | public filers | all others | note |
|---|---|---|---|
| **ASU 2016-18** restricted cash | FY2018 | FY2019 | **retrospective** |
| **ASC 842** leases | **FY2019** | FY2022 (deferred twice) | ROU assets/liabilities appear ~3 years apart |
| **ASU 2018-15** CCA implementation costs | FY2020 | FY2021 | |
| **ASU 2025-06** internal-use software | FY2028 | FY2028 | removes project stages; **does not touch ASC 985-20** |
| ASC 606 revenue | FY2018 | — | the measured cliff in §2.2 |
| LDTI (ASU 2018-12) | 2021-01-01 transition | — | measured on MET in §5.3 |
