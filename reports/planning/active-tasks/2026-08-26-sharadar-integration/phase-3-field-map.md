# Phase 3 — The field map and basis translation ✅

**Goal**: turn `fundamentals_sharadar` (112 vendor columns, ARQ grain) into a repo-vocabulary frame
on the repo's TTM/instant contract, with every basis fork implemented as decided.

**Prerequisite**: phase 2 findings read and approved; `configs/sharadar/sharadar_zero_rules.json` exists.
**Read first**: [README.md](README.md) — D14–D21 and the basis-forks table.
**Next**: [phase-4-merge.md](phase-4-merge.md)

This phase **writes no table**. It produces a pure, testable transform plus its config.

---

## ✅ Phase-2 hand-back, and what it added to this phase (2026-08-26)

**The 4 `null` rules are APPROVED** — `intexp`, `inventory`, `ppnenet`, `dps`. The approval is
recorded as an `_APPROVED` block in `configs/sharadar/sharadar_zero_rules.json`, and
**`apply_zero_rules` must refuse to run against a file that lacks one**: a regenerated proposal
is byte-identical to a reviewed decision otherwise, and the whole point of the file is that a
human looked at those four.

Phase 2 also found three defects that the zero rule **cannot express**, because none of them is
a zero. They are the reason for §0 below.

| defect | measured | why the zero rule cannot reach it |
|---|---|---|
| `capex` positive | 13 / 1,346 rows (BA, CVX, GS ×11, IBM) | the cells are positive, not zero |
| `intexp` net-basis | NKE 14 negative quarters, **0 zeros**; MMM 1 | the cells are negative, not zero |
| `sharesbas` split-adjusted | NVDA 10×, WMT 3× on pre-split dates | the cells are correct numbers on the *wrong basis* |

### 0. `configs/sharadar/sharadar_corrections.json` — the correction register ✅

**Decided 2026-08-26: build a proper correction mechanism in the Sharadar layer**, rather than
scattering `if ticker == "GS"` guards through the field map. Same governance as D22 —
machine-proposed, human-approved, and the transform only *reads* it, so the map stays
deterministic given the register.

Grain is **(field, ticker)** — deliberately not (field), which is the zero rule's grain, and not
(field, ticker, date), which would be an unmaintainable cell-level patch list. Proposed shape:

```json
{
  "intexp": {"NKE": {"action": "null", "reason": "filer tags 'Interest expense (income), net'; basis is net, the repo column is gross", "evidence": "ebt - ebit == -intexp on 20/20 rows"}},
  "capex":  {"GS":  {"action": "null_if_positive", "reason": "Sharadar's bank cash-flow mapping; 11 positive rows"}}
}
```

Every entry needs an `action`, a `reason` and an **`evidence`** field naming what was measured
or which filing was read — this repo has been burned by fallbacks with no stated authority.
Actions stay a **closed vocabulary** (`null`, `null_if_positive`, …); a free-form expression
field would become code in a config file.

Applied **in the Sharadar layer, before the field map**, so `fundamentals_sharadar` stays a
faithful record of what the vendor sent (D7: a mapping mistake must be re-derivable without
refetching) and every correction is one auditable, reversible step.

### 0b. `intexp` — INVESTIGATE BEFORE RULING ✅

**Do not write an `intexp` correction from the phase-2 numbers alone.** They establish *that*
NKE is on a net basis, not *how many filers are*, and a rule built on one ticker will not
survive the S&P 500. The investigation, before any entry is written:

- [x] How many of the 30 tickers tag a **net** interest line? `ebt - ebit == -intexp` holds for
      NKE on 20/20 — run that identity across the whole roster and see who else it fits.
- [x] Does the SEC layer resolve **gross** interest expense for the same filers? It declares
      `interestExpense` gross and `non_negative`, and its regime machinery already distinguishes
      bank from industrial caption chains. If it does, the answer is "SEC owns this column",
      not a correction — but NKE is **not** on the 54-ticker SEC roster, so measure coverage
      before assuming a fallback exists.
- [x] Is a negative `intexp` always net-basis, or sometimes a genuine credit (a capitalised-
      interest reversal)? Read MMM's single negative quarter against its 10-Q, since it is the
      one case that does not fit the NKE pattern.
- [x] Decide between: NULL the negatives; carry the net line as a **separate column**
      (`interestExpenseNet`, one basis per column, the repo's rule); or move `interestExpense`
      to SEC-owned in D18.

Only then write the register entry. **This is a blocker for `interestExpense` specifically, not
for the rest of the field map** — the other 59 columns are unaffected.

---

## The contract it must produce

The repo's vocabulary is `HISTORY_STATEMENT_ORDER` in
[kpi_catalogue.py:199](../../../../src/data_extract/utils/fundamentals/kpi_catalogue.py#L199) —
60 names, in statement order. `Catalogue.history_columns` asserts it. Read that list before writing
a single mapping line; it is the authority, not this document.

Two hard rules carried from D17:
- a **duration** field (income statement, cash flow) is a **TTM sum of four discrete ARQ quarters**;
- an **instant** field (balance sheet, share counts) is the **period-end value**, not an average.

---

## Changes

### 1. `configs/sharadar/sharadar_field_map.json` — the map, as data not code

⚠ These JSON registers are **hand-formatted**; a `json.dumps` round-trip reformats the whole file.
Use a validated emitter or a text splice.

Three entry kinds:

```json
{
  "totalRevenue":  {"from": "revenue",  "kind": "direct"},
  "capex":         {"from": "capex",    "kind": "direct", "negate": "if_non_positive"},
  "ebitda":        {"kind": "derived",  "formula": "opinc + depamor"},
  "goodwill":      {"kind": "sec"}
}
```

⚠ `"negate": true` was the plan's original spelling and phase 2 killed it: 13 of 1,346 stored
rows carry a POSITIVE `capex`, so an unconditional flip writes a negative into a column the SEC
catalogue declares `non_negative`. `"if_non_positive"` flips where the sign convention holds and
NULLs where it does not, which must be **counted and logged**, never silent.

### 2. The map itself

**Direct, no basis question** — map and move on:

| repo | Sharadar | | repo | Sharadar |
|---|---|---|---|---|
| `costOfRevenue` | `cor` | | `totalAssets` | `assets` |
| `grossProfit` | `gp` | | `currentAssets` | `assetsc` |
| `sellingGeneralAdmin` | `sgna` | | `totalLiabilities` | `liabilities` |
| `researchAndDevelopment` | `rnd` | | `currentLiabilities` | `liabilitiesc` |
| `depAmort` | `depamor` | | `retainedEarnings` | `retearn` |
| `stockBasedComp` | `sbcomp` | | `inventory` | `inventory` |
| `operatingIncome` | `opinc` | | `shortTermInvestments` | `investmentsc` |
| `incomeTaxExpense` | `taxexp` | | `basicShares` | `shareswa` |
| `epsDiluted` | `epsdil` | | `dilutedShares` | `shareswadil` |
| `operatingCashFlow` | `ncfo` | | `totalDebt` | `debt` ✅ *(both lease-inclusive — this one genuinely agrees)* |

**Direct, but carrying a known basis fork** — map, and record the fork in the docstring so the next
reader does not "fix" it:

| repo | Sharadar | the fork |
|---|---|---|
| `totalRevenue` | `revenue` | ⚠ filer's own caption. JPM matches exactly; AXP is 6.6–8.1% low (post-provision). Resolved per-ticker in phase 4's override register, **not here**. |
| `accountsReceivable` | `receivables` | trade **and non-trade**; the repo uses `AccountsReceivableNetCurrent`. Affects `dso`, `cash_conversion_cycle`, Beneish DSRI. |
| `accountsPayable` | `payables` | trade **and non-trade**. Affects `dpo`. |
| `ppeNet` | `ppnenet` | **includes operating ROU assets**; the repo *subtracts* finance-lease ROU where detectable. |
| `shortTermDebt` | `debtc` | lease-**inclusive**; the repo's is lease-exclusive. |
| `longTermDebt` | `debtnc` | lease-**inclusive**. |
| `pretaxIncome` | `ebt` | Sharadar tags `ebt` as `[Metrics]`, not `[Income Statement]`. |
| `interestExpense` | `intexp` | **16.6% zero-filled** (not 25.4% — phase 2 re-measured on the full table) **and provably wrong on 58/58 judgeable cells** (AXP, GS, JPM). Ruled `"null"` in `sharadar_zero_rules.json`. ⚠ **The null rule does not cover this column's second defect, because that defect is never a zero.** `intexp` is whatever single interest line the filer tags, and for NKE that line is **"Interest expense (income), net"** — negative in 14 of 20 quarters, and 0 zeros, so the null rule never inspects a single NKE row. Verified: `ebt - ebit == -intexp` exactly on all 20. This is a **BASIS** mismatch, not a sign convention — unlike `capex`, negating is *not* lossless, because it would report an $8m interest expense NKE never incurred against a column defined as **gross** ("Total interest expense for the period", `sign: non_negative`). NKE is not on the 54-ticker SEC roster, so there is no fallback value. Decide: NULL the negatives, or carry the net line under a different name. Do **not** negate. Census across 30 tickers: NKE 14 neg / 0 zero, MMM 1 neg / 9 zero, AAPL 11 zero, CRM 19 zero, AXP+GS+JPM 20 zero each. |
| `sharesOutstanding` | ⚠ **NOT `sharesbas`** | **Phase 2 answered this, and the answer is no.** `sharesbas` does not sum share classes — 12/14 overlap tickers match the SEC cover page at ratio exactly 1.0 — but it **is retroactively SPLIT-ADJUSTED**: NVDA's 2021 rows carry ~25bn shares against the ~2.5bn then outstanding (10-for-1, June 2024), WMT the same at 3x. `sharefactor` is `1.0` on every one and does not flag it. **`sharesbas` is not point-in-time.** Take this column from the SEC layer on the overlap, or de-adjust with `sharadar_actions` (ingested, carries the splits) — and pin whichever you choose with a test on NVDA's 2021 rows. |

**Sign flip — measured, and the easiest silent bug in this phase:**

| repo | Sharadar | |
|---|---|---|
| `capex` | `capex` | The repo declares `capex` as **`sign: non_negative`** ([fundamentals_kpis.json]). Sharadar stores it negative. ⚠ **`negate: true` IS NOT SAFE UNCONDITIONALLY** — phase 2 measured **13 of 1,346 rows with a POSITIVE `capex`** (11 of them GS; also BA, CVX, IBM), so a blind flip writes a *negative* into a `non_negative` column. **Flip where `capex <= 0`, NULL the rest**, and count what you nulled. |
| `freeCashflow` | `fcf` | No transform. ✅ Confirmed by phase 2 on **all 1,346 stored rows**, zero violations: `fcf == ncfo + capex_sharadar` **exactly**, which is `ncfo − capex_repo`. Consistent by construction. |

**Derived — the basis decisions, implemented:**

| repo | formula | why |
|---|---|---|
| `ebitda` | `opinc + depamor` | **Top-down** (D). Not Sharadar's `ebitda`, which is bottom-up `netinc + taxexp + intexp + depamor` and therefore contaminated by every non-operating item. |
| `cash` | `cashneq + investmentsc` | The repo's widening, best available. Restricted cash absent and accepted. |
| `netIncome` | `consolinc` | **Not `netinc`.** Measured on JPM's 11 dates: the repo's `netIncome` equals `consolinc` on all 11 and differs from `netinc` on all 11. `consolinc` includes NCI, which is the repo's validated basis. |
| `stockholdersEquity` | `equity` | Parent-only, universal, **one basis for the whole cohort**. |
| `stockholdersEquityInclNci` | `equity + <SEC minorityInterest>` | **A new column** (not in the 60). 54-ticker coverage by construction — which is exactly why it is a second column and not a modification of the first. |
| `grossMargins` | `gp / revenue` | Recomputed. **Never** Sharadar's `grossmargin` (D21). |
| `operatingMargins` | `opinc / revenue` | Recomputed. |
| `profitMargins` | `consolinc / revenue` | Recomputed on the repo's net-income basis. ⚠ Sharadar's own `netmargin` uses `netinccmn` (after preferred) — a **different** basis. |
| `effectiveTaxRate` | `taxexp / ebt` | Recomputed. |
| `returnOnEquity` | `consolinc / equity` | Recomputed. **Never** Sharadar's `roe`. |
| `debtToEquity` | `debt / equity` | Recomputed. ⚠ Sharadar's `de` is **liabilities/equity** despite being named "Debt to Equity Ratio" — using it would be wrong by the whole non-debt liability stack. |
| `optionOverhang` | per the existing catalogue formula | Do not invent one; read the catalogue. |
| `revenue_q`, `netIncome_q` | the **discrete ARQ** values, pre-TTM | These two columns exist precisely to expose the single quarter next to the TTM line cut from it. |

**SEC-owned — `{"kind": "sec"}`, no Sharadar source (D18), 15 columns:**

`goodwill`, `intangiblesExGoodwill`, `ppeGross`, `accumulatedDepreciation`, `minorityInterest`,
`operatingLeaseLiability`, `financeLeaseLiability`, `premiumsEarned`, `netInterestIncome`,
`noninterestIncome`, `netInvestmentIncome`, `realizedInvestmentGains`, `rentalIncome`, `employees`,
**`regime`**.

**No source anywhere — permanently NULL, and all four are read by nothing downstream:**

`restrictedCash`, `shortTermBorrowingsOnly`, `longTermDebtCurrentOnly`.
Keep the columns (the contract asserts by list equality) and document them as free losses.

**Sharadar extras kept under their own names** (D16 — no repo counterpart, so no rename). These
revive eight currently-dead cube inputs:

`deferredrev`, `deposits`, `ncfdiv`, `ncfcommon`, `accoci`, `netincnci`, `netincdis` — plus
`assetsnc`, `liabilitiesnc`, `investments`, `investmentsnc`, `taxassets`, `taxliabilities`, `opex`,
`prefdivis`, `netinccmn`, `dps`, and the cash-flow decomposition `ncfi`, `ncff`, `ncfbus`, `ncfinv`,
`ncfdebt`, `ncfx`, `ncf`.

**Excluded entirely (D21)** — vendor ratios, which stay in the raw table:
`pe`, `pe1`, `ps`, `ps1`, `pb`, `roe`, `roa`, `roic`, `de`, `ev`, `evebit`, `evebitda`, `marketcap`,
`price`, `divyield`, `netmargin`, `grossmargin`, `ebitdamargin`, `payoutratio`, `bvps`, `tbvps`,
`fcfps`, `sps`, `currentratio`, `workingcapital`, `assetturnover`, `ros`, `invcap`, `invcapavg`,
`assetsavg`, `equityavg`, `tangibles`, `sharefactor`, `fxusd`, and the 8 `*usd` twins.

### 3. `regime` — decided: carried from SEC (D18b)

`fundamentals_history` carries a `regime` column (`HISTORY_REGIME`), the filing's resolution regime,
stamped per filing by the SEC facts layer. **Sharadar has no regime concept.**

**Decision (2026-08-26): `regime` is the 15th SEC-owned column.** `{"kind": "sec"}`, same as the
other 14, with the SEC roster's coverage.

⚠ Do **not** derive it from GICS via `sp500_tickers` instead. GICS already reaches the cube by that
route, and substituting it here would silently change what the column *means* — from "how this
filing's KPIs were resolved" to "what sector this company is in". Those are different facts that
happen to correlate.

⚠ Do **not** confuse `regime` with the **six regime top-line legs** (`premiumsEarned`,
`netInterestIncome`, `noninterestIncome`, `netInvestmentIncome`, `realizedInvestmentGains`,
`rentalIncome`). They are also SEC-owned, but they are value columns; `regime` is a label.

### 4. `src/data_extract/utils/fundamentals_sharadar/field_map.py`

- [x] `load_field_map() -> FieldMap` — loads and validates the JSON. **Fails loudly** if any of the
      60 `HISTORY_STATEMENT_ORDER` names is unmapped, or if a mapped Sharadar column is absent from
      `SHARADAR_SF1_COLUMNS`.
- [x] `apply_zero_rules(df) -> df` — reads `sharadar_zero_rules.json`; for every field ruled
      `"null"`, replaces `0.0` with `NaN`. **Fails loudly** on a field missing from the register.
- [x] `translate(df_arq) -> pd.DataFrame` — direct maps, negations, derived formulas, in that order.
      Zero rules apply **before** any derived formula, so a nulled input propagates instead of
      silently contributing a zero to a sum.

### 5. `src/data_extract/utils/fundamentals_sharadar/build_ttm.py`

- [x] `build_ttm(df_arq) -> pd.DataFrame` — per ticker, per `date`: duration fields become the sum of
      the four most recent discrete quarters; instant fields take the period-end value.
- [x] Reuse the SEC path's staleness contract rather than inventing one: fewer than four available
      quarters → **NULL**, matching `insufficient_quarters`. Read
      [build_history.py](../../../../src/data_extract/utils/fundamentals/build_history.py) for the
      45-day cap and mirror it.
- [x] ⚠ Do **not** use Sharadar's `ART` (D17). It is documented as *not* equal to the sum of four
      ARQ, and it would silently redefine every duration column.

---

## Tests

`tests/data_extract/test_sharadar_field_map.py` — parsing math gets synthetic known-truth fixtures;
the basis decisions get real data. Each prints its conclusion.

- [x] `test_every_history_column_is_mapped` — all 60 names resolve to direct / derived / sec / null.
      Prints any unmapped.
- [x] `test_capex_sign_is_flipped` — real AAPL row: Sharadar negative in, repo-positive out, and
      `freeCashflow == operatingCashFlow - capex` on the repo's signs. Prints all three numbers.
- [x] `test_netincome_is_consolinc_not_netinc` — real JPM: assert the mapped `netIncome` equals
      `consolinc` and differs from `netinc` on every date. Prints both series.
- [x] `test_ebitda_is_top_down` — assert `ebitda == opinc + depamor` and that it **differs** from
      Sharadar's own `ebitda` column on at least one ticker. Prints the gap distribution.
- [x] `test_debt_to_equity_is_not_vendor_de` — assert the recomputed value differs from `de`.
      Prints both, because this is the trap most likely to be "helpfully" reverted later.
- [x] `test_ttm_is_four_discrete_quarters` — synthetic four-quarter fixture with known sum; assert
      the TTM equals it and that three quarters yields NULL. Prints both cases.
- [x] `test_zero_rules_propagate_into_derived` — a field ruled `"null"` and used in a derived
      formula produces NaN, not a zero-contaminated result. Prints the before/after.

---

## Verification

```bash
PY="$HOME/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe"
rtk "$PY" -m pytest tests/data_extract/test_sharadar_field_map.py -v -s
```

- [x] All 60 contract columns resolve; the loader fails loudly on a gap.
- [x] `capex` positive out, `freeCashflow` identity holds on real data.
- [x] `netIncome == consolinc` on JPM, all dates.
- [x] `ebitda` measurably differs from the vendor column.
- [x] TTM equals the four-quarter sum; three quarters is NULL.
- [x] `regime` resolves as `{"kind": "sec"}`, not as a GICS derivation.

---

# ✅ DONE 2026-08-26 — what was actually implemented

**Verification**: `tests/data_extract/sharadar/test_sharadar_field_map.py` — **20 passed**.
Every test prints its conclusion; the numbers below are that output, not estimates.

## Files

| file | what it is |
|---|---|
| `configs/sharadar/sharadar_corrections.json` | §0's register — 3 approved entries, each with `evidence` |
| `configs/sharadar/sharadar_field_map.json` | the map: 60 contract columns + 3 added + 25 extras + 42 excluded |
| `src/data_extract/utils/fundamentals_sharadar/field_map.py` | loader, both registers, the cleaning stages, `translate`, `apply_derived` |
| `src/data_extract/utils/fundamentals_sharadar/build_ttm.py` | the TTM/instant contract + `ttm_coverage` |
| `src/constants/constants.py` | the two filenames, the closed vocabularies, the split-action names |

Measured on all 598 stored ARQ rows, 30 tickers:

```
zero-rule NULLs         : 375 over 4 field(s) {dps 96, intexp 99, inventory 140, ppnenet 40}
correction NULLs        : 24 {intexp/NKE 20, intexp/MMM 1, capex/GS 3}
sign-guard NULLs        : 3  {capex}
split de-adjusted cells : 89 {basicShares 24, dilutedShares 24, sharesOutstanding 24, dps 17}
splits applied          : AMZN x20, WMT x3, NVDA x10     splits rejected: HON x0.5
rows out 598 | totalRevenue non-null 508 (= 598 - 3 per ticker cold start) | totalAssets 598
```

## §0b `intexp` — INVESTIGATED AND SETTLED, and the plan's proposed evidence was wrong

⚠ **`ebt - ebit == -intexp` is a TAUTOLOGY.** The plan proposed it as the evidence that NKE is
on a net basis. Sharadar *defines* `ebit = ebt + intexp`, so it holds on **598 of 598 ARQ rows
across all 30 tickers** — for every filer, on either basis. It distinguishes nothing, and
`test_the_ebt_minus_ebit_identity_is_a_tautology` records that so nobody re-derives a decision
from it. The four questions, answered from stored data:

1. **How many filers tag a net line?** Only NKE shows the signature. Sign census over 20 ARQ
   quarters each: NKE 14 neg / 0 zero, MMM 1 neg / 9 zero, AAPL 0 / 11, CRM 0 / 19,
   AXP+GS+JPM 0 / 20 each. The other 23 tickers are clean.
2. **Does the SEC layer cover the same filers?** Not usefully. **NKE has 0 rows** on
   `fundamentals_history_sec`, and of the 14 overlap tickers **CAT and BA have 0 non-null
   `interestExpense` in 60 rows each**. There is no fallback to fall back to.
3. **Is MMM's negative a net line or a genuine credit?** *Neither* — it is Sharadar's
   Q4-by-subtraction meeting a zero-filled annual row, and the arithmetic closes exactly:
   MMM FY2023 Q1..Q3 = 123 + 144 + 304 = 571M, ARY FY2023 = **0**, so Q4 = 0 − 571 = **−571M**,
   the stored value to the dollar. MMM's ARY `intexp` is 0 for FY2023/24/25 (against 488M and
   462M for FY2021/22), which is also why every quarter from FY2024-Q2 on is 0. **No filing
   needed to be read.**
4. **Decision — NULL, and `interestExpense` stays Sharadar-owned.** NKE gets a ticker-level
   `"null"`: the whole series is net, not just the negatives. Evidence is the one measurement
   that separates the bases — `intexp` slides +57M → −8M and **crosses zero** while `debt` is
   flat at 12.8bn → 11.0bn (−14%); a *gross* expense on flat debt cannot cross zero, a net line
   can as interest income rises with the rate cycle. MMM gets `null_if_negative`.
   Moving the column to SEC-owned was rejected on coverage: it would drop 30 tickers to 14,
   two of which carry no value at all. Result: **401 of 598 rows survive (67%)**, every removed
   cell removed for a stated, measured reason.

## Deviations from the plan, each with the measurement behind it

| # | plan said | implemented | why |
|---|---|---|---|
| 1 | `epsDiluted ← epsdil`, direct | **derived**, `netIncome / dilutedShares` | `epsdil` is on the `netinccmn` basis: over 549 rows it matches `netinccmn/shareswadil` on 426, `netinc` on 412, `consolinc` on only **296** — and `netinccmn ≠ consolinc` on **208 rows (37.9%)**. A direct map imports a second net-income basis into the same row as `netIncome ← consolinc`. Matches the SEC path's `_FORMULAS`. |
| 2 | `sharesbas` is split-adjusted | **the whole share block is** — `sharesbas`, `shareswa`, `shareswadil`, `eps`, `epsdil`, `dps` | Phase 2 only saw `sharesbas` because it cross-checked 14 SEC-roster tickers. Measured: NVDA `shareswadil` 25.35bn (ART) vs SEC 2.535bn — exactly 10x; `epsdil` 0.097 vs 0.97 as filed; `dps` 0.004 vs $0.04. **AMZN is 20x adjusted from 2021 and phase 2 never saw it** (not on the SEC roster). So the plan's "Direct, no basis question" table is wrong for `basicShares`, `dilutedShares` and (had it stayed direct) `epsDiluted`. |
| 3 | "take from SEC **or** de-adjust" | **de-adjust**, via `sharadar_actions` | 30/30 ticker coverage instead of 14/30, and the same mechanism is needed anyway for the three sibling columns SEC cannot supply for 16 tickers. Pinned: de-adjusted `sharesOutstanding` matches the SEC cover page on **39 of 39 overlap rows**, 38 at ratio exactly 1.00000 and one at 1.00080 (Sharadar's own 4-significant-figure rounding). |
| 4 | — | ⚠ **the HON trap**, new | A `split` row is **not** always a share split. `sharadar_actions` has 4: AMZN 20, WMT 3, NVDA 10 and **HON 0.5**, and HON's is co-dated with `spinoff=1` / `spinoffdividend=221.01` (Honeywell Aerospace) — it is the spinoff's *price* factor. HON's own cover page proves it: `sharesbas` is 316,826,560 on 2026-04-23 and 316,940,010 on 2026-07-23, **unchanged**. Applying it would have DOUBLED every HON share count — a 100% error on 19 of 20 rows manufactured from a correct number. A candidate is used only when no `spinoff` shares its `(ticker, date)`. |
| 5 | two bases (duration / instant) | **three** — `duration`, `instant`, `mean` | `basicShares`/`dilutedShares` are weighted averages the catalogue declares `not_additive`; summing four gives 4x the year's average. `_basis_for` reads that flag off the catalogue, so the two layers cannot drift. |
| 6 | `"formula": "opinc + depamor"` | `op` + `inputs` execute; `formula` is prose, **asserted against them** by the loader | An eval-able expression string is code in a config file. The loader recomputes the prose from `op`/`inputs` and refuses a mismatch, so the comment cannot drift. |
| 7 | `cash` = `cashneq + investmentsc` (Sharadar names) | `cashneq` **carried as a 25th extra**; `cash = cashneq + shortTermInvestments` | Otherwise the file needs two evaluation spaces (one formula in vendor names, the rest in repo names). `cashneq` genuinely has no repo counterpart — the repo's `cash` is the wider concept — so D16 already covers carrying it. **Every derived formula now runs in one pass on the TTM frame.** |
| 8 | `tests/data_extract/test_sharadar_field_map.py` | `tests/data_extract/`**`sharadar/`**`test_sharadar_field_map.py` | Matches where phase 1 and 2 put their tests. |
| 9 | "permanently NULL … **all four**" | there are **three** | `restrictedCash`, `shortTermBorrowingsOnly`, `longTermDebtCurrentOnly`. The prose said four and listed three; the list is right. |

## A gap the plan did not anticipate

⚠ **`shareswadil` is missing for 4 of the 30 tickers** — HON 0/20, PG 6/20, CVX 15/20, BA 10/20
— so `dilutedShares` reaches **440 of 598 rows against `basicShares`' 508**, and `epsDiluted`
and `optionOverhang` inherit that ceiling. It is a vendor gap, not a transform defect
(`test_diluted_share_count_gaps_are_the_vendors` prints it). **Phase 4 decides** whether the SEC
layer supplies it for the overlap — BA and PG are on that roster, HON and CVX are not — which
would be a per-ticker override, not a field-block switch (D14).

## Two residuals, stated rather than discovered later

- **`freeCashflow` is kept where `capex` was nulled.** `fcf == ncfo + capex_vendor` exactly on
  all 1,346 rows, so the two are consistent by construction — but on the 6 ARQ rows whose
  `capex` the sign guard or the GS correction removed, `freeCashflow` survives and the identity
  is no longer checkable there. Not corrected: phase 2 measured `fcf` at **zero violations**,
  and nulling a validated column to match an invalidated one needs its own evidence.
- **`stockholdersEquityInclNci` is 0/598 here by construction.** Its NCI leg is SEC-owned, so it
  populates at the phase-4 merge, on the 54-ticker roster. That is the point of it being a
  second column rather than a modification of the first.

## Verification block — results

- [x] All 60 contract columns resolve (32 direct, 12 derived, 13 sec, 3 null); the loader fails
      loudly on a gap, on a `from` SF1 does not deliver, on a derived input it cannot produce,
      and on a register with no `_APPROVED` block. **All 112 SF1 columns are accounted for** —
      mapped, extra, excluded or identifier; none silently dropped.
- [x] `capex` positive out (AAPL: vendor −3,223M → repo +3,223M), `freeCashflow` identity holds
      to 0.00, and all 6 vendor rows with a positive `capex` are NULL rather than negative.
- [x] `netIncome == consolinc` on JPM, **20 of 20 dates**, and differs from `netinc` on all 20.
- [x] `ebitda` measurably differs from bottom-up on the **same trailing twelve**: >1% on 282 of
      401 rows over 22 tickers, median 3.03%, p90 20.03%. ⚠ Comparing against the vendor's own
      `ebitda` COLUMN is a trap — on ARQ it is a *quarter*, so the ~4x period gap swamps the
      definitional one; the test builds the bottom-up leg from the TTM frame instead.
- [x] TTM equals the four-quarter sum (1000.0 from 100/200/300/400); three quarters is NULL;
      a window with a missing quarter is NULL, **not** the 4-row splice.
- [x] `debtToEquity` differs from vendor `de` on >90% of rows, and `de` reproduces
      `liabilities/equity` on **598 of 598** — it is not a debt ratio.
- [x] `regime` resolves as `{"kind": "sec"}`, one of the 15 SEC-owned columns, not a GICS
      derivation.

---

## ⚠ Amendment 2026-08-26 — `deadjust_splits` MOVED to `build_ttm`

Deviation 3 above (de-adjust rather than take from SEC) stands. Its **stage** did not.

Running the de-adjustment on the DISCRETE quarters put two split bases inside one
four-quarter window, overstating `epsDiluted` by up to **3.48x** for the three filings after
every split (AMZN 3.96 → 1.13, NVDA 6.56 → 2.13, WMT 4.01 → 2.01). `translate()` no longer
de-adjusts and no longer takes `actions=`; `build_ttm(..., actions=...)` does it between the
aggregation and `apply_derived`.

Full measurement, before/after table and the two new tests:
[phase-4-merge.md](phase-4-merge.md) — "the split de-adjustment ran at the wrong STAGE".
