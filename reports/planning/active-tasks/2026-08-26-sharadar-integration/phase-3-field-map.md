# Phase 3 — The field map and basis translation ⬜

**Goal**: turn `fundamentals_sharadar` (112 vendor columns, ARQ grain) into a repo-vocabulary frame
on the repo's TTM/instant contract, with every basis fork implemented as decided.

**Prerequisite**: phase 2 findings read and approved; `configs/sharadar/sharadar_zero_rules.json` exists.
**Read first**: [README.md](README.md) — D14–D21 and the basis-forks table.
**Next**: [phase-4-merge.md](phase-4-merge.md)

This phase **writes no table**. It produces a pure, testable transform plus its config.

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
  "capex":         {"from": "capex",    "kind": "direct", "negate": true},
  "ebitda":        {"kind": "derived",  "formula": "opinc + depamor"},
  "goodwill":      {"kind": "sec"}
}
```

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
| `interestExpense` | `intexp` | **25.4% zero-filled and provably wrong for banks** (JPM, GS). Governed by `sharadar_zero_rules.json`. |
| `sharesOutstanding` | `sharesbas` | whether it sums share classes is undocumented; phase 2 measured it. SF1 covers the **primary class only**. |

**Sign flip — measured, and the easiest silent bug in this phase:**

| repo | Sharadar | |
|---|---|---|
| `capex` | `capex` | The repo declares `capex` as **`sign: non_negative`** ([fundamentals_kpis.json]). Sharadar stores it **negative** (`-2,455,000,000` on AAPL). **`negate: true`.** |
| `freeCashflow` | `fcf` | No transform. Measured: `fcf == ncfo + capex_sharadar` **exactly**, which is `ncfo − capex_repo`. Consistent by construction. |

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

- [ ] `load_field_map() -> FieldMap` — loads and validates the JSON. **Fails loudly** if any of the
      60 `HISTORY_STATEMENT_ORDER` names is unmapped, or if a mapped Sharadar column is absent from
      `SHARADAR_SF1_COLUMNS`.
- [ ] `apply_zero_rules(df) -> df` — reads `sharadar_zero_rules.json`; for every field ruled
      `"null"`, replaces `0.0` with `NaN`. **Fails loudly** on a field missing from the register.
- [ ] `translate(df_arq) -> pd.DataFrame` — direct maps, negations, derived formulas, in that order.
      Zero rules apply **before** any derived formula, so a nulled input propagates instead of
      silently contributing a zero to a sum.

### 5. `src/data_extract/utils/fundamentals_sharadar/build_ttm.py`

- [ ] `build_ttm(df_arq) -> pd.DataFrame` — per ticker, per `date`: duration fields become the sum of
      the four most recent discrete quarters; instant fields take the period-end value.
- [ ] Reuse the SEC path's staleness contract rather than inventing one: fewer than four available
      quarters → **NULL**, matching `insufficient_quarters`. Read
      [build_history.py](../../../../src/data_extract/utils/fundamentals/build_history.py) for the
      45-day cap and mirror it.
- [ ] ⚠ Do **not** use Sharadar's `ART` (D17). It is documented as *not* equal to the sum of four
      ARQ, and it would silently redefine every duration column.

---

## Tests

`tests/data_extract/test_sharadar_field_map.py` — parsing math gets synthetic known-truth fixtures;
the basis decisions get real data. Each prints its conclusion.

- [ ] `test_every_history_column_is_mapped` — all 60 names resolve to direct / derived / sec / null.
      Prints any unmapped.
- [ ] `test_capex_sign_is_flipped` — real AAPL row: Sharadar negative in, repo-positive out, and
      `freeCashflow == operatingCashFlow - capex` on the repo's signs. Prints all three numbers.
- [ ] `test_netincome_is_consolinc_not_netinc` — real JPM: assert the mapped `netIncome` equals
      `consolinc` and differs from `netinc` on every date. Prints both series.
- [ ] `test_ebitda_is_top_down` — assert `ebitda == opinc + depamor` and that it **differs** from
      Sharadar's own `ebitda` column on at least one ticker. Prints the gap distribution.
- [ ] `test_debt_to_equity_is_not_vendor_de` — assert the recomputed value differs from `de`.
      Prints both, because this is the trap most likely to be "helpfully" reverted later.
- [ ] `test_ttm_is_four_discrete_quarters` — synthetic four-quarter fixture with known sum; assert
      the TTM equals it and that three quarters yields NULL. Prints both cases.
- [ ] `test_zero_rules_propagate_into_derived` — a field ruled `"null"` and used in a derived
      formula produces NaN, not a zero-contaminated result. Prints the before/after.

---

## Verification

```bash
PY="$HOME/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe"
rtk "$PY" -m pytest tests/data_extract/test_sharadar_field_map.py -v -s
```

- [ ] All 60 contract columns resolve; the loader fails loudly on a gap.
- [ ] `capex` positive out, `freeCashflow` identity holds on real data.
- [ ] `netIncome == consolinc` on JPM, all dates.
- [ ] `ebitda` measurably differs from the vendor column.
- [ ] TTM equals the four-quarter sum; three quarters is NULL.
- [ ] `regime` resolves as `{"kind": "sec"}`, not as a GICS derivation.
