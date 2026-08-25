---
type: DATA
session_id: 6ea3721c-02be-4c23-97e4-584ba5be6f14
generated_at: 2026-08-24T21:40:04+00:00
baseline: {head_sha: ebca051681d4cf8eedac443cf9899026aab57a61}
generator: scripts/dod/data_profile.py@1
---

## 1. Scope

**SAMPLE SCOPE** — a metric without its scope is not a measurement:

- tables: fundamentals_employees, fundamentals_facts, fundamentals_history, fundamentals_reason_codes
- tickers: **all** (no ticker filter)
- since: **no lower bound**
- row limit per table: 500,000
- full-scope tables (eligible to set the baseline): none

**What was asked:** Phase 5 of the fundamentals rebuild -- build `fundamentals_history` on a
PUBLICATION-EVENT grain (`as_of` IS a filing date, one row per date on which >=1 extracted value
became newly public, each row a complete snapshot), plus the dense `fundamentals_reason_codes`
side table and `fundamentals_employees`, over the 52-ticker rebuild roster. Mid-phase the user
added two requirements: a **fiscal-quarter label (Q1-Q4) on every row including the TTM and
instant ones**, and the value columns **reordered into statement order** (revenue -> cost -> net
revenue, then debt/assets, then shares).

All four tables were dropped and rebuilt from `sql/schema.sql` on 2026-08-24 because a
primary-key defect was found mid-phase (see section 5), so every number here is from a
from-scratch build: facts fetched in 9 chunks of 6 tickers with `-F`, then a 4h39m history
replay, then `scripts/verify_fundamentals_history.py` (all 8 gates PASS).

## 2. Gates

| Gate | Check | Verdict | Detail |
|---|---|---|---|
| D1 | declared PK unique over the rows profiled | **PASS** | unique across 4 table(s): fundamentals_facts, fundamentals_history, fundamentals_reason_codes, fundamentals_employees |
| D2 | row count not decreased | **N/A** | no full-scope baseline to compare against (scoped run: fundamentals_facts, fundamentals_history, fundamentals_reason_codes, fundamentals_employees) — this run records one |
| D3 | no column lost | **N/A** | no baseline columns recorded yet |
| D4 | date range covers the expected window | **FAIL** | fundamentals_employees: max 2026-07-29 < 2026-07-31 |
| D5 | per-field null rate not worse | **N/A** | no full-scope baseline null rates to compare against |

**1 FAIL** — D4. The work is **NOT done**.

## 3. Metrics

_Observed values only — no verdicts. `rows`, `date_min` and `date_max` are **table-wide** (server-side); every other number is over the **sample** described in §1. Do not compare across the two._

**Tables**

| table | exists | rows | sampled | cols | pk | pk_absent_cols | pk_dupes | date_min | date_max | sample_date_min | sample_date_max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fundamentals_employees | yes | 745 | 745 | 3 | ticker,as_of | — | 0 | 2002-03-20 | 2026-07-29 | 2002-03-20 | 2026-07-29 |
| fundamentals_facts | yes | 317,036 | 317,036 | 26 | ticker,accession_number,field,duration_type,period_end | — | 0 | 2009-07-31 | 2026-08-10 | 2009-07-31 | 2026-08-10 |
| fundamentals_history | yes | 3,267 | 3,267 | 69 | ticker,as_of | — | 0 | 2009-07-31 | 2026-08-10 | 2009-07-31 | 2026-08-10 |
| fundamentals_reason_codes | yes | 76,004 | 76,004 | 5 | ticker,as_of,field,dc_code | — | 0 | 2009-07-31 | 2026-08-10 | 2009-07-31 | 2026-08-10 |

**Fields** (worst null rate first, top 60)

| table | field | dtype | null_% | nunique | mean | std | min | p01 | p50 | p99 | max | mad_outliers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fundamentals_reason_codes | combined_into | object | 100 | 0 | — | — | — | — | — | — | — | — |
| fundamentals_history | amended_fields | str | 99.36 | 14 | — | — | — | — | — | — | — | — |
| fundamentals_history | amended_fiscal_end | object | 99.36 | 21 | — | — | — | — | — | — | — | — |
| fundamentals_facts | adjustment | str | 99.34 | 377 | — | — | — | — | — | — | — | — |
| fundamentals_facts | root_anchor | str | 97.69 | 9 | — | — | — | — | — | — | — | — |
| fundamentals_facts | roll_up_children | str | 97.02 | 57 | — | — | — | — | — | — | — | — |
| fundamentals_history | realizedInvestmentGains | float64 | 93.17 | 142 | 3.45556e+07 | 5.91768e+08 | -1.9122e+09 | -1.553e+09 | 7.8e+07 | 1.797e+09 | 2.7689e+09 | 9 |
| fundamentals_history | netInvestmentIncome | float64 | 92.93 | 229 | 6.92296e+09 | 7.57084e+09 | 3.895e+08 | 3.9981e+08 | 3.423e+09 | 2.25185e+10 | 2.407e+10 | 57 |
| fundamentals_history | rentalIncome | float64 | 91.86 | 223 | 2.82633e+09 | 3.3567e+09 | 3.7e+07 | 3.7e+07 | 9.62e+08 | 1.05291e+10 | 1.11068e+10 | 52 |
| fundamentals_history | premiumsEarned | float64 | 90.3 | 281 | 6.09519e+10 | 7.66306e+10 | 6.62e+08 | 6.62e+08 | 3.6803e+10 | 3.38055e+11 | 3.53256e+11 | 38 |
| fundamentals_history | financeLeaseLiability | float64 | 85.98 | 360 | 2.57749e+09 | 7.28731e+09 | 4e+06 | 5e+06 | 6.055e+08 | 4.22059e+10 | 6.6594e+10 | 68 |
| fundamentals_history | noninterestIncome | float64 | 85.95 | 456 | 3.21518e+10 | 1.89802e+10 | 1.53521e+09 | 1.76679e+09 | 3.205e+10 | 8.52443e+10 | 9.957e+10 | 1 |
| fundamentals_history | restrictedCash | float64 | 79.64 | 552 | 5.65285e+09 | 9.75006e+09 | 0 | 0 | 3.45e+08 | 4.098e+10 | 4.71e+10 | 215 |
| fundamentals_facts | dc_code | str | 79.49 | 6 | — | — | — | — | — | — | — | — |
| fundamentals_history | netInterestIncome | float64 | 77.53 | 669 | 1.89144e+10 | 2.30802e+10 | -2.609e+09 | -2.34867e+09 | 6.99087e+09 | 9.25977e+10 | 9.9838e+10 | 143 |
| fundamentals_history | grossMargins | float64 | 73.4 | 861 | 0.367 | 0.22291 | -0.097751 | 0.00241924 | 0.396585 | 0.740393 | 0.762218 | 0 |
| fundamentals_history | grossProfit | float64 | 72.94 | 594 | 2.87618e+10 | 4.10346e+10 | -5.685e+09 | 1.03187e+08 | 1.1328e+10 | 1.91275e+11 | 2.27123e+11 | 113 |
| fundamentals_history | researchAndDevelopment | float64 | 72.6 | 857 | 8.03723e+09 | 1.08928e+10 | 6.4223e+07 | 8.54536e+07 | 4.591e+09 | 5.10652e+10 | 7.1634e+10 | 82 |
| fundamentals_history | longTermDebtCurrentOnly | float64 | 71.59 | 712 | 2.21917e+09 | 2.45728e+09 | -1.82e+07 | 0 | 1.4595e+09 | 1.09427e+10 | 1.4009e+10 | 29 |
| fundamentals_history | shortTermInvestments | float64 | 68.9 | 952 | 1.14949e+10 | 2.45351e+10 | 0 | 1.015e+06 | 1.87495e+09 | 1.18641e+11 | 1.86563e+11 | 229 |
| fundamentals_history | operatingLeaseLiability | float64 | 67.34 | 990 | 4.61139e+09 | 5.86198e+09 | 0 | 2.38205e+07 | 1.469e+09 | 2.18834e+10 | 3.019e+10 | 259 |
| fundamentals_history | shortTermBorrowingsOnly | float64 | 66.51 | 943 | 1.43669e+10 | 2.98656e+10 | 0 | 0 | 2.037e+09 | 1.20513e+11 | 2.52927e+11 | 264 |
| fundamentals_history | costOfRevenue | float64 | 65.17 | 1,021 | 6.82508e+10 | 9.92242e+10 | 4.06e+07 | 4.06e+07 | 2.83215e+10 | 4.75946e+11 | 5.4415e+11 | 139 |
| fundamentals_facts | period_days | float64 | 57.51 | 46 | 194.227 | 112.725 | 81 | 83 | 180 | 365 | 1,095 | 2 |
| fundamentals_facts | period_start | object | 57.51 | 627 | — | — | — | — | — | — | — | — |
| fundamentals_history | ppeGross | float64 | 56.93 | 1,385 | 4.97099e+10 | 6.5359e+10 | 1.24344e+08 | 2.26013e+08 | 3.3297e+10 | 4.08452e+11 | 5.70059e+11 | 82 |
| fundamentals_history | sellingGeneralAdmin | float64 | 54.09 | 1,444 | 1.42244e+10 | 2.242e+10 | 5.518e+07 | 9.40738e+07 | 6.64795e+09 | 1.20791e+11 | 1.50972e+11 | 110 |
| fundamentals_history | inventory | float64 | 52.77 | 1,470 | 7.0067e+09 | 1.02846e+10 | 3.91e+07 | 4.981e+07 | 3.209e+09 | 5.60889e+10 | 6.5354e+10 | 134 |
| fundamentals_history | intangiblesExGoodwill | float64 | 52.4 | 1,453 | 6.8047e+09 | 9.65265e+09 | 3.47e+07 | 5.2e+07 | 3.075e+09 | 4.80222e+10 | 5.4942e+10 | 184 |
| fundamentals_history | accumulatedDepreciation | float64 | 49.8 | 1,604 | 1.90733e+10 | 2.7442e+10 | -1.9086e+10 | 5.0585e+07 | 1.19235e+10 | 1.15973e+11 | 2.7851e+11 | 111 |
| fundamentals_history | minorityInterest | float64 | 45.24 | 1,243 | 1.38838e+09 | 2.17944e+09 | -1.87e+08 | -2.8e+07 | 4.35e+08 | 8.67232e+09 | 1.1871e+10 | 346 |
| fundamentals_facts | role_uri | str | 41.54 | 1,849 | — | — | — | — | — | — | — | — |
| fundamentals_history | accountsPayable | float64 | 39.36 | 1,879 | 8.18786e+09 | 1.36047e+10 | 126,022 | 4.10432e+07 | 2.423e+09 | 6.25622e+10 | 7.7088e+10 | 310 |
| fundamentals_history | stockBasedComp | float64 | 38.93 | 1,478 | 1.44669e+09 | 3.22164e+09 | 2.049e+06 | 2.232e+06 | 2.92e+08 | 1.75364e+10 | 2.8147e+10 | 454 |
| fundamentals_history | accountsReceivable | float64 | 38.08 | 1,898 | 7.43749e+09 | 9.79474e+09 | 1e+07 | 1.46185e+08 | 3.98e+09 | 4.77711e+10 | 8.0876e+10 | 203 |
| fundamentals_history | ebitda | float64 | 37.65 | 1,970 | 1.65107e+10 | 2.54139e+10 | -1.0521e+10 | -2.08728e+09 | 8.84e+09 | 1.36383e+11 | 1.98685e+11 | 162 |
| fundamentals_history | operatingMargins | float64 | 34.71 | 2,118 | 0.193499 | 0.239534 | -4.25585 | -0.155722 | 0.199087 | 0.527951 | 0.803395 | 16 |
| fundamentals_history | currentAssets | float64 | 34.01 | 2,128 | 3.38228e+10 | 4.05885e+10 | 515,881 | 7.40116e+08 | 1.60535e+10 | 1.7407e+11 | 3.43524e+11 | 238 |
| fundamentals_history | currentLiabilities | float64 | 33.98 | 2,126 | 2.4706e+10 | 2.95648e+10 | 511,697 | 2.16156e+08 | 1.3349e+10 | 1.26726e+11 | 1.76392e+11 | 239 |
| fundamentals_history | operatingIncome | float64 | 33.46 | 1,881 | 1.23928e+10 | 2.15883e+10 | -2.5494e+10 | -6.71144e+09 | 5.24291e+09 | 1.19749e+11 | 1.62285e+11 | 208 |
| fundamentals_history | freeCashflow | float64 | 31.47 | 2,142 | 9.24391e+09 | 1.79034e+10 | -9.0387e+10 | -1.58412e+10 | 3.452e+09 | 9.20106e+10 | 1.36683e+11 | 276 |
| fundamentals_history | shortTermDebt | float64 | 31.25 | 1,839 | 9.08349e+09 | 2.17225e+10 | -6.34e+08 | 0 | 2.1025e+09 | 1.03938e+11 | 2.52927e+11 | 342 |
| fundamentals_history | capex | float64 | 31.04 | 2,044 | 5.78182e+09 | 9.82575e+09 | 5.001e+06 | 3.87222e+07 | 2.716e+09 | 4.6288e+10 | 1.32402e+11 | 270 |
| fundamentals_history | goodwill | float64 | 26.32 | 1,837 | 1.98714e+10 | 2.32116e+10 | 0 | 2.0098e+07 | 9.842e+09 | 9.99828e+10 | 1.19661e+11 | 233 |
| fundamentals_history | ppeNet | float64 | 23.02 | 2,418 | 2.64476e+10 | 4.5481e+10 | 8.1289e+07 | 1.93934e+08 | 1.12686e+10 | 2.50815e+11 | 3.13076e+11 | 278 |
| fundamentals_facts | source_concept | str | 22.22 | 140 | — | — | — | — | — | — | — | — |
| fundamentals_facts | decimals | str | 20.51 | 14 | — | — | — | — | — | — | — | — |
| fundamentals_facts | value | float64 | 20.51 | 68,183 | 5.23205e+10 | 9.60129e+12 | -2.67506e+11 | -6.75e+08 | 3.2584e+09 | 4.33685e+11 | 4.81906e+15 | 57,019 |
| fundamentals_facts | unit | str | 20.39 | 190 | — | — | — | — | — | — | — | — |
| fundamentals_history | longTermDebt | float64 | 19.8 | 2,526 | 4.10498e+10 | 6.46775e+10 | 0 | 2.40697e+07 | 1.87815e+10 | 2.87171e+11 | 3.98965e+11 | 270 |
| fundamentals_history | interestExpense | float64 | 19.19 | 1,881 | 3.72324e+09 | 1.19535e+10 | 11,000 | 11,000 | 5.95684e+08 | 8.13552e+10 | 1.0135e+11 | 442 |
| fundamentals_history | effectiveTaxRate | float64 | 15.61 | 2,741 | 0.198807 | 0.34789 | -9.10667 | -0.477701 | 0.204437 | 0.944718 | 5.12308 | 126 |
| fundamentals_history | pretaxIncome | float64 | 14.42 | 2,624 | 1.38268e+10 | 2.33746e+10 | -3.0576e+10 | -6.0912e+09 | 6.0035e+09 | 1.19002e+11 | 2.99269e+11 | 323 |
| fundamentals_history | debtToEquity | float64 | 14.02 | 2,786 | 0.754449 | 12.838 | -560.411 | -14.0816 | 0.635964 | 9.63818 | 167.483 | 296 |
| fundamentals_history | totalDebt | float64 | 13.8 | 2,721 | 4.74378e+10 | 7.56715e+10 | 0 | 3.48123e+07 | 2.08785e+10 | 3.46489e+11 | 5.01315e+11 | 276 |
| fundamentals_history | depAmort | float64 | 12.67 | 2,441 | 3.61579e+09 | 4.73527e+09 | 6.364e+06 | 2.89111e+07 | 2.163e+09 | 2.2935e+10 | 4.6463e+10 | 304 |
| fundamentals_history | epsDiluted | float64 | 10.41 | 2,906 | 32,091.1 | 613,228 | -62.2434 | -8.24033 | 5.08529 | 50.4062 | 1.23129e+07 | 258 |
| fundamentals_history | optionOverhang | float64 | 9.03 | 2,720 | 0.0114455 | 0.015154 | -0.0337281 | 0 | 0.00691148 | 0.0744953 | 0.19963 | 307 |
| fundamentals_history | dilutedShares | float64 | 8.54 | 2,778 | 1.70289e+09 | 2.57606e+09 | 713.639 | 4.33928e+07 | 6.88135e+08 | 1.22673e+10 | 2.4804e+10 | 679 |
| fundamentals_history | basicShares | float64 | 7.28 | 2,813 | 2.18962e+09 | 3.00551e+10 | 711.013 | 1.64305e+06 | 6.53e+08 | 1.21921e+10 | 1.64989e+12 | 690 |

## 4. Evidence

- baseline file: `reports/baselines/data_profile.json` (0 table(s) recorded)
- `fundamentals_employees`: 745 rows, 3 cols, 745 sampled
- `fundamentals_facts`: 317,036 rows, 26 cols, 317,036 sampled
- `fundamentals_history`: 3,267 rows, 69 cols, 3,267 sampled
- `fundamentals_reason_codes`: 76,004 rows, 5 cols, 76,004 sampled

## 5. Regressions, gaps and deliberate omissions

- **D4's FAIL is my flag, not the data.** `--expect-through 2026-07-31` was applied uniformly, but
  `fundamentals_employees` is parsed from 10-K BODY TEXT and is therefore **annual**: its max
  `as_of` (2026-07-29) is the most recent 10-K across the 54 tickers and has no reason to land
  within two days of the most recent 10-Q. The three quarterly tables all reach 2026-08-10. The
  gate as invoked cannot pass for an annual table alongside quarterly ones, and `data_profile.py`
  has no per-table `--expect-through`. Not a data gap.
- **`fundamentals_history` shrank 27,602 -> 3,267 rows and 239 -> 69 columns, deliberately, and
  the shrink is declared.** The row count fell because the grain changed (a computed period spine
  became one row per publication event) AND the scope narrowed to 54 tickers; the columns fell
  because the contract is now enumerated by `Catalogue.history_columns` instead of accreted. The
  old 491-ticker table no longer exists.
- **SCOPE: 54 tickers, not 500.** All four tables cover the same 54 names by construction. Any
  coverage rate computed against a 500-ticker denominator will read ~11%. Widening is Phase 9.
- **Nulls are 36.7% of value cells (71,857 of 196,020) and that is UP, on purpose.** The previous
  builder forward-filled the last computable TTM, which froze 1,622 of 26,242 consecutive
  `totalRevenue` pairs into a staircase. A refused TTM now stays NULL with its reason. **Every
  null carries a `fundamentals_reason_codes` row -- 0 unexplained.**
- **`reason_codes.combined_into` is 100% NULL.** The register's `combined_into` cells never fire on
  this 54-ticker roster. The column is not dead (the accessor works and is tested); it simply has
  no applicable filer here. Re-measure when the roster widens.
- **`not_disclosed` is 51,887 of 76,004 codes (68%) and is a VERDICT, not evidence.** It means
  "the resolver walked this filing's calculation linkbase and found no concept it recognises",
  which is a statement about our concept MAP. An unresolved fact row carries NULL in
  `source_concept`, `roll_up_children`, `root_anchor` and `role_uri`, so an absence has none of
  the supporting evidence a resolved value does. Anatomy measured on AAPL: 66% are lines the filer
  structurally cannot have, 16% pre-date the filer's own first disclosure (`totalDebt` NULL for
  exactly the 16 events before Apple's first bond issue on 2013-04-30), 16% pre-date the
  accounting standard (`operatingLeaseLiability` first appears at Apple's ASC 842 adoption,
  2020-01-29). Only 5 cells were genuinely mis-coded, and those are fixed.
- **A register change was approved by the user and then WITHDRAWN on measurement.** The proposal
  was to mark the cross-regime top lines `expected_absent` for `industrial`. Measured: of 27
  industrial filers only `noninterestIncome` is 0/27; `premiumsEarned` (UNH $72bn, CVS $34bn),
  `rentalIncome` (AMT $3.5bn, CAT $549M) and `netInterestIncome` (WMT -$990M) are REAL,
  filer-tagged and material. The edit would have been a false claim, and the payoff had collapsed
  from 552 codes to 69 (~0.1%). Withdrawn rather than narrowed, because the `industrial` block's
  own `_authority` states *"Nothing is structurally excused here"*.
- **The finding that survives it: the register's `by_regime` blocks never covered the cross-regime
  top-line family for ANY regime.** `scripts/audit_absence_evidence.py` (new) reports 7
  unregistered structural fields for `energy` and 5 for `utility`. Closing this honestly needs
  more than the 4 filers those regimes have in a 54-ticker roster -- Phase 5b.
- **38 rows still have NULL `totalLiabilities`, and all 38 lack the inputs.** All 38 are missing
  `stockholdersEquity` (29 also missing `totalAssets`); **0** have both present. Nothing refuses
  for want of an inference any more; what remains has nothing to compute from.
- **152 of 901 derived `totalLiabilities` cells (17%) rest on an inference about absence**, carrying
  their own `derived_identity_nci_assumed_zero` code so a consumer can drop them separately from
  the 749 that are evidence-backed. Bounded where observable: EOG's two equity bases agree to the
  dollar on 6 of 7 overlap dates, TMO's differ by 0.02-0.12% of equity.
- **9 pre-existing `tests/data_aggregate` failures are NOT from this phase** -- verified identical
  on a stashed tree. Phase 6's backlog (the cube reads a `fundamentals_history` this rebuild
  re-scoped).

## 6. Next actions

- **Phase 6 -- repair the 9 `tests/data_aggregate` failures.** The cube still reads the old
  `fundamentals_history` shape; `ebitda_q` / `freeCashflow_q` / `capexGlobal` are declared
  casualties needing reconciliation, and `revenueGrowth` / `earningsGrowth` must now be computed
  by `pit.py` on a 365-DAY `as_of` offset rather than a 4-ROW one (an amendment row makes four
  rows ~9 months, not 12).
- **Phase 5b -- the validator toolkit**, with two concrete inputs from this phase: register the
  cross-regime top-line family per regime (needs >4 filers per regime), and work the MIXED
  population `scripts/audit_absence_evidence.py` identifies -- 31 of 48 industrial fields, where
  no config rule can decide whether an absence is legitimate and only the filing can.
- **Consider giving an ABSENCE the evidence a presence has.** A resolved fact row stores
  `source_concept` / `role_uri` / `roll_up_children`; an unresolved one stores nothing. Recording
  what the filer DID tag on the relevant statement would turn `not_disclosed` from a verdict into
  a checkable claim, and is the durable fix for the question this phase could only answer by peer
  comparison.
- **Phase 9 -- widen from 54 to the full roster**, at which point every rate in section 3 must be
  re-measured; none should be quoted against a 500-ticker denominator until then.
- **Re-run `data_profile.py` with a per-table date expectation** (or split the annual
  `fundamentals_employees` into its own invocation) so D4 measures something true.

```json dod-metrics
{
  "baseline_head_sha": "ebca051681d4cf8eedac443cf9899026aab57a61",
  "content_hash": "sha256:546a2860b081d117ce6387bc7a6a364cfa7e5aaa3af9293076aff9e36f5e6678",
  "gates": {
    "D1": "PASS",
    "D2": "N/A",
    "D3": "N/A",
    "D4": "FAIL",
    "D5": "N/A"
  },
  "generator": "scripts/dod/data_profile.py@1",
  "metrics": {
    "parts_behind": null,
    "stale_sources": null,
    "tables": {
      "fundamentals_employees": {
        "columns": [
          "ticker",
          "as_of",
          "employees"
        ],
        "date_col": "as_of",
        "date_max": "2026-07-29",
        "date_min": "2002-03-20",
        "exists": true,
        "fields": {
          "as_of": {
            "dtype": "object",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 411
          },
          "employees": {
            "dtype": "float64",
            "mad_center": 40600.0,
            "mad_outliers": 85,
            "mad_scale": 36300.0,
            "max": 2300000.0,
            "mean": 122151.70201342282,
            "min": 1000.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 598,
            "p01": 1241.64,
            "p25": 9922.0,
            "p50": 40600.0,
            "p75": 125000.0,
            "p99": 2200000.0,
            "std": 313734.7554381544
          },
          "ticker": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 54
          }
        },
        "kind": "extract",
        "pk": [
          "ticker",
          "as_of"
        ],
        "pk_checked_cols": [
          "ticker",
          "as_of"
        ],
        "pk_checked_rows": 745,
        "pk_complete": true,
        "pk_duplicate_rows": 0,
        "pk_missing_cols": [],
        "rows": 745,
        "sample_date_max": "2026-07-29",
        "sample_date_min": "2002-03-20",
        "sampled_rows": 745,
        "scope": {
          "limit": 500000,
          "since": null,
          "tickers": null
        },
        "table": "fundamentals_employees"
      },
      "fundamentals_facts": {
        "columns": [
          "ticker",
          "accession_number",
          "field",
          "duration_type",
          "period_end",
          "fiscal_year",
          "fiscal_period",
          "cik",
          "form",
          "filing_date",
          "is_amendment",
          "period_of_report",
          "regime",
          "period_start",
          "period_days",
          "value",
          "unit",
          "decimals",
          "resolution_method",
          "source_concept",
          "roll_up_children",
          "root_anchor",
          "adjustment",
          "role_uri",
          "is_extension",
          "dc_code"
        ],
        "date_col": "filing_date",
        "date_max": "2026-08-10",
        "date_min": "2009-07-31",
        "exists": true,
        "fields": {
          "accession_number": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 3290
          },
          "adjustment": {
            "dtype": "str",
            "null_rate": 0.9933887634211888,
            "nulls": 314940,
            "nunique": 377
          },
          "cik": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 57
          },
          "dc_code": {
            "dtype": "str",
            "null_rate": 0.7948655673172763,
            "nulls": 252001,
            "nunique": 6
          },
          "decimals": {
            "dtype": "str",
            "null_rate": 0.20513443268272372,
            "nulls": 65035,
            "nunique": 14
          },
          "duration_type": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 6
          },
          "field": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 48
          },
          "filing_date": {
            "dtype": "object",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1306
          },
          "fiscal_period": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 8
          },
          "fiscal_year": {
            "dtype": "int64",
            "mad_center": 2018.0,
            "mad_outliers": 0,
            "mad_scale": 4.0,
            "max": 2027.0,
            "mean": 2018.0656770839905,
            "min": 2006.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 22,
            "p01": 2010.0,
            "p25": 2014.0,
            "p50": 2018.0,
            "p75": 2022.0,
            "p99": 2026.0,
            "std": 4.463800095186333
          },
          "form": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 4
          },
          "is_amendment": {
            "dtype": "bool",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2
          },
          "is_extension": {
            "dtype": "bool",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2
          },
          "period_days": {
            "dtype": "float64",
            "mad_center": 180.0,
            "mad_outliers": 2,
            "mad_scale": 91.0,
            "max": 1095.0,
            "mean": 194.22736890558528,
            "min": 81.0,
            "null_rate": 0.5751491944132527,
            "nulls": 182343,
            "nunique": 46,
            "p01": 83.0,
            "p25": 90.0,
            "p50": 180.0,
            "p75": 273.0,
            "p99": 365.0,
            "std": 112.72509719795912
          },
          "period_end": {
            "dtype": "object",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1759
          },
          "period_of_report": {
            "dtype": "object",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 600
          },
          "period_start": {
            "dtype": "object",
            "null_rate": 0.5751491944132527,
            "nulls": 182343,
            "nunique": 627
          },
          "regime": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 8
          },
          "resolution_method": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 8
          },
          "role_uri": {
            "dtype": "str",
            "null_rate": 0.41543231683468124,
            "nulls": 131707,
            "nunique": 1849
          },
          "roll_up_children": {
            "dtype": "str",
            "null_rate": 0.9701516547016743,
            "nulls": 307573,
            "nunique": 57
          },
          "root_anchor": {
            "dtype": "str",
            "null_rate": 0.9769237562926608,
            "nulls": 309720,
            "nunique": 9
          },
          "source_concept": {
            "dtype": "str",
            "null_rate": 0.22218927818922773,
            "nulls": 70442,
            "nunique": 140
          },
          "ticker": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 54
          },
          "unit": {
            "dtype": "str",
            "null_rate": 0.2038695920967966,
            "nulls": 64634,
            "nunique": 190
          },
          "value": {
            "dtype": "float64",
            "mad_center": 3258400000.0,
            "mad_outliers": 57019,
            "mad_scale": 3081400000.0,
            "max": 4819056000000000.0,
            "mean": 52320496979.597115,
            "min": -267506000000.0,
            "null_rate": 0.20513443268272372,
            "nulls": 65035,
            "nunique": 68183,
            "p01": -675000000.0,
            "p25": 655288000.0,
            "p50": 3258400000.0,
            "p75": 16155000000.0,
            "p99": 433685000000.0,
            "std": 9601286851052.781
          }
        },
        "kind": "extract",
        "pk": [
          "ticker",
          "accession_number",
          "field",
          "duration_type",
          "period_end"
        ],
        "pk_checked_cols": [
          "ticker",
          "accession_number",
          "field",
          "duration_type",
          "period_end"
        ],
        "pk_checked_rows": 317036,
        "pk_complete": true,
        "pk_duplicate_rows": 0,
        "pk_missing_cols": [],
        "rows": 317036,
        "sample_date_max": "2026-08-10",
        "sample_date_min": "2009-07-31",
        "sampled_rows": 317036,
        "scope": {
          "limit": 500000,
          "since": null,
          "tickers": null
        },
        "table": "fundamentals_facts"
      },
      "fundamentals_history": {
        "columns": [
          "ticker",
          "as_of",
          "fiscal_end",
          "fiscal_quarter",
          "totalRevenue",
          "premiumsEarned",
          "netInterestIncome",
          "noninterestIncome",
          "netInvestmentIncome",
          "realizedInvestmentGains",
          "rentalIncome",
          "costOfRevenue",
          "grossProfit",
          "grossMargins",
          "sellingGeneralAdmin",
          "researchAndDevelopment",
          "depAmort",
          "stockBasedComp",
          "operatingIncome",
          "operatingMargins",
          "ebitda",
          "interestExpense",
          "pretaxIncome",
          "incomeTaxExpense",
          "effectiveTaxRate",
          "netIncome",
          "profitMargins",
          "epsDiluted",
          "revenue_q",
          "netIncome_q",
          "operatingCashFlow",
          "capex",
          "freeCashflow",
          "cash",
          "restrictedCash",
          "shortTermInvestments",
          "accountsReceivable",
          "inventory",
          "currentAssets",
          "ppeGross",
          "accumulatedDepreciation",
          "ppeNet",
          "goodwill",
          "intangiblesExGoodwill",
          "totalAssets",
          "accountsPayable",
          "currentLiabilities",
          "shortTermDebt",
          "shortTermBorrowingsOnly",
          "longTermDebt",
          "longTermDebtCurrentOnly",
          "operatingLeaseLiability",
          "financeLeaseLiability",
          "totalDebt",
          "totalLiabilities",
          "retainedEarnings",
          "minorityInterest",
          "stockholdersEquity",
          "returnOnEquity",
          "debtToEquity",
          "basicShares",
          "dilutedShares",
          "sharesOutstanding",
          "optionOverhang",
          "regime",
          "publication_form",
          "is_amendment",
          "amended_fiscal_end",
          "amended_fields"
        ],
        "date_col": "as_of",
        "date_max": "2026-08-10",
        "date_min": "2009-07-31",
        "exists": true,
        "fields": {
          "accountsPayable": {
            "dtype": "float64",
            "mad_center": 2423000000.0,
            "mad_outliers": 310,
            "mad_scale": 2066572000.0,
            "max": 77088000000.0,
            "mean": 8187863544.01363,
            "min": 126022.0,
            "null_rate": 0.3936333027242118,
            "nulls": 1286,
            "nunique": 1879,
            "p01": 41043200.00000001,
            "p25": 870000000.0,
            "p50": 2423000000.0,
            "p75": 8563000000.0,
            "p99": 62562200000.0,
            "std": 13604742741.09392
          },
          "accountsReceivable": {
            "dtype": "float64",
            "mad_center": 3980000000.0,
            "mad_outliers": 203,
            "mad_scale": 2774000000.0,
            "max": 80876000000.0,
            "mean": 7437486074.641622,
            "min": 10000000.0,
            "null_rate": 0.3807774716865626,
            "nulls": 1244,
            "nunique": 1898,
            "p01": 146185420.0,
            "p25": 1589000000.0,
            "p50": 3980000000.0,
            "p75": 8907500000.0,
            "p99": 47771059999.99998,
            "std": 9794742866.65197
          },
          "accumulatedDepreciation": {
            "dtype": "float64",
            "mad_center": 11923500000.0,
            "mad_outliers": 111,
            "mad_scale": 7908000000.0,
            "max": 278510000000.0,
            "mean": 19073290953.048782,
            "min": -19086000000.0,
            "null_rate": 0.49801040710131617,
            "nulls": 1627,
            "nunique": 1604,
            "p01": 50585050.0,
            "p25": 5534500000.0,
            "p50": 11923500000.0,
            "p75": 21897750000.0,
            "p99": 115973479999.9993,
            "std": 27442007287.736443
          },
          "amended_fields": {
            "dtype": "str",
            "null_rate": 0.9935720844811754,
            "nulls": 3246,
            "nunique": 14
          },
          "amended_fiscal_end": {
            "dtype": "object",
            "null_rate": 0.9935720844811754,
            "nulls": 3246,
            "nunique": 21
          },
          "as_of": {
            "dtype": "object",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1302
          },
          "basicShares": {
            "dtype": "float64",
            "mad_center": 653000000.0,
            "mad_outliers": 690,
            "mad_scale": 347460273.9726027,
            "max": 1649891000000.0,
            "mean": 2189621979.323375,
            "min": 711.012602739726,
            "null_rate": 0.07284970921334558,
            "nulls": 238,
            "nunique": 2813,
            "p01": 1643053.1095890412,
            "p25": 394800000.0,
            "p50": 653000000.0,
            "p75": 1961871780.8219178,
            "p99": 12192123178.082182,
            "std": 30055089095.34326
          },
          "capex": {
            "dtype": "float64",
            "mad_center": 2716000000.0,
            "mad_outliers": 270,
            "mad_scale": 1913000000.0,
            "max": 132402000000.0,
            "mean": 5781818576.120728,
            "min": 5001000.0,
            "null_rate": 0.310376492194674,
            "nulls": 1014,
            "nunique": 2044,
            "p01": 38722240.0,
            "p25": 965000000.0,
            "p50": 2716000000.0,
            "p75": 6025000000.0,
            "p99": 46288040000.00007,
            "std": 9825754864.668869
          },
          "cash": {
            "dtype": "float64",
            "mad_center": 4261500000.0,
            "mad_outliers": 537,
            "mad_scale": 3798500000.0,
            "max": 759869000000.0,
            "mean": 33240728518.17414,
            "min": 3.4,
            "null_rate": 0.05417814508723599,
            "nulls": 177,
            "nunique": 2942,
            "p01": 46000000.0,
            "p25": 1024248500.0,
            "p50": 4261500000.0,
            "p75": 14501750000.0,
            "p99": 527962100000.0004,
            "std": 93945093234.68248
          },
          "costOfRevenue": {
            "dtype": "float64",
            "mad_center": 28321500000.0,
            "mad_outliers": 139,
            "mad_scale": 25657612500.0,
            "max": 544150000000.0,
            "mean": 68250815525.48331,
            "min": 40600000.0,
            "null_rate": 0.6516681971227426,
            "nulls": 2129,
            "nunique": 1021,
            "p01": 40600000.0,
            "p25": 7116925000.0,
            "p50": 28321500000.0,
            "p75": 82139750000.0,
            "p99": 475945559999.99927,
            "std": 99224182499.47453
          },
          "currentAssets": {
            "dtype": "float64",
            "mad_center": 16053500000.0,
            "mad_outliers": 238,
            "mad_scale": 12754872500.0,
            "max": 343524000000.0,
            "mean": 33822758597.39471,
            "min": 515881.0,
            "null_rate": 0.3400673400673401,
            "nulls": 1111,
            "nunique": 2128,
            "p01": 740116300.0,
            "p25": 5526000000.0,
            "p50": 16053500000.0,
            "p75": 50071750000.0,
            "p99": 174069749999.99997,
            "std": 40588502206.90965
          },
          "currentLiabilities": {
            "dtype": "float64",
            "mad_center": 13349000000.0,
            "mad_outliers": 239,
            "mad_scale": 10586000000.0,
            "max": 176392000000.0,
            "mean": 24705961909.23783,
            "min": 511697.0,
            "null_rate": 0.3397612488521579,
            "nulls": 1110,
            "nunique": 2126,
            "p01": 216156480.0,
            "p25": 4407000000.0,
            "p50": 13349000000.0,
            "p75": 31493000000.0,
            "p99": 126725680000.00008,
            "std": 29564762466.611664
          },
          "debtToEquity": {
            "dtype": "float64",
            "mad_center": 0.6359641201613389,
            "mad_outliers": 296,
            "mad_scale": 0.43311383709761575,
            "max": 167.48275862068965,
            "mean": 0.7544490415557497,
            "min": -560.4109589041096,
            "null_rate": 0.14018977655341291,
            "nulls": 458,
            "nunique": 2786,
            "p01": -14.081631582149855,
            "p25": 0.30185938928833983,
            "p50": 0.6359641201613389,
            "p75": 1.3987623452250604,
            "p99": 9.638178322788407,
            "std": 12.838026077704999
          },
          "depAmort": {
            "dtype": "float64",
            "mad_center": 2163000000.0,
            "mad_outliers": 304,
            "mad_scale": 1201000000.0,
            "max": 46463000000.0,
            "mean": 3615786059.9369087,
            "min": 6364000.0,
            "null_rate": 0.12672176308539945,
            "nulls": 414,
            "nunique": 2441,
            "p01": 28911120.0,
            "p25": 1130000000.0,
            "p50": 2163000000.0,
            "p75": 3856000000.0,
            "p99": 22934960000.000008,
            "std": 4735270549.723417
          },
          "dilutedShares": {
            "dtype": "float64",
            "mad_center": 688134800.0,
            "mad_outliers": 679,
            "mad_scale": 367594808.7431694,
            "max": 24804000000.0,
            "mean": 1702889361.2955813,
            "min": 713.6391780821918,
            "null_rate": 0.08539944903581267,
            "nulls": 279,
            "nunique": 2778,
            "p01": 43392789.04109589,
            "p25": 402539699.1269005,
            "p50": 688134800.0,
            "p75": 2074155205.4794521,
            "p99": 12267258520.54795,
            "std": 2576055159.9415946
          },
          "ebitda": {
            "dtype": "float64",
            "mad_center": 8840000000.0,
            "mad_outliers": 162,
            "mad_scale": 5503000000.0,
            "max": 198685000000.0,
            "mean": 16510692760.432009,
            "min": -10521000000.0,
            "null_rate": 0.37649219467401285,
            "nulls": 1230,
            "nunique": 1970,
            "p01": -2087280000.0000002,
            "p25": 4017000000.0,
            "p50": 8840000000.0,
            "p75": 16545000000.0,
            "p99": 136383239999.99966,
            "std": 25413909042.946198
          },
          "effectiveTaxRate": {
            "dtype": "float64",
            "mad_center": 0.20443724040117517,
            "mad_outliers": 126,
            "mad_scale": 0.07186456618756021,
            "max": 5.123076923076923,
            "mean": 0.19880681875445444,
            "min": -9.106666666666667,
            "null_rate": 0.1561065197428834,
            "nulls": 510,
            "nunique": 2741,
            "p01": -0.47770122149489,
            "p25": 0.13357204016501906,
            "p50": 0.20443724040117517,
            "p75": 0.2781146098181107,
            "p99": 0.9447177391259292,
            "std": 0.34789021128761494
          },
          "epsDiluted": {
            "dtype": "float64",
            "mad_center": 5.085287846481877,
            "mad_outliers": 258,
            "mad_scale": 2.394230574977268,
            "max": 12312945.070664234,
            "mean": 32091.10170695948,
            "min": -62.24338624338624,
            "null_rate": 0.10407101316192226,
            "nulls": 340,
            "nunique": 2906,
            "p01": -8.2403349059042,
            "p25": 3.002737320511798,
            "p50": 5.085287846481877,
            "p75": 8.078901572211345,
            "p99": 50.406228547081106,
            "std": 613227.5323293922
          },
          "financeLeaseLiability": {
            "dtype": "float64",
            "mad_center": 605500000.0,
            "mad_outliers": 68,
            "mad_scale": 580200000.0,
            "max": 66594000000.0,
            "mean": 2577488414.847162,
            "min": 4000000.0,
            "null_rate": 0.8598102234465871,
            "nulls": 2809,
            "nunique": 360,
            "p01": 5000000.0,
            "p25": 38250000.0,
            "p50": 605500000.0,
            "p75": 2128448500.0,
            "p99": 42205940000.000046,
            "std": 7287306257.944171
          },
          "fiscal_end": {
            "dtype": "object",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 600
          },
          "fiscal_quarter": {
            "dtype": "float64",
            "mad_center": 2.0,
            "mad_outliers": 0,
            "mad_scale": 1.0,
            "max": 4.0,
            "mean": 2.4998425196850396,
            "min": 1.0,
            "null_rate": 0.028160391796755432,
            "nulls": 92,
            "nunique": 4,
            "p01": 1.0,
            "p25": 1.0,
            "p50": 2.0,
            "p75": 4.0,
            "p99": 4.0,
            "std": 1.126770807024502
          },
          "freeCashflow": {
            "dtype": "float64",
            "mad_center": 3452000000.0,
            "mad_outliers": 276,
            "mad_scale": 3601448000.0,
            "max": 136683000000.0,
            "mean": 9243912147.833855,
            "min": -90387000000.0,
            "null_rate": 0.3146617692072238,
            "nulls": 1028,
            "nunique": 2142,
            "p01": -15841160000.0,
            "p25": 643219500.0,
            "p50": 3452000000.0,
            "p75": 12503000000.0,
            "p99": 92010599999.99973,
            "std": 17903389606.629517
          },
          "goodwill": {
            "dtype": "float64",
            "mad_center": 9842000000.0,
            "mad_outliers": 233,
            "mad_scale": 8615000000.0,
            "max": 119661000000.0,
            "mean": 19871446826.7553,
            "min": 0.0,
            "null_rate": 0.26323844505662686,
            "nulls": 860,
            "nunique": 1837,
            "p01": 20098000.0,
            "p25": 3191098500.0,
            "p50": 9842000000.0,
            "p75": 26563500000.0,
            "p99": 99982760000.00038,
            "std": 23211571672.107265
          },
          "grossMargins": {
            "dtype": "float64",
            "mad_center": 0.3965847174827021,
            "mad_outliers": 0,
            "mad_scale": 0.2233034366969434,
            "max": 0.762218032364391,
            "mean": 0.3669998419849311,
            "min": -0.0977509542969153,
            "null_rate": 0.734006734006734,
            "nulls": 2398,
            "nunique": 861,
            "p01": 0.0024192388900475042,
            "p25": 0.1570253059874297,
            "p50": 0.3965847174827021,
            "p75": 0.5964209152763538,
            "p99": 0.7403932362165543,
            "std": 0.22291009869395248
          },
          "grossProfit": {
            "dtype": "float64",
            "mad_center": 11328000000.0,
            "mad_outliers": 113,
            "mad_scale": 9719650000.0,
            "max": 227123000000.0,
            "mean": 28761813566.74208,
            "min": -5685000000.0,
            "null_rate": 0.7294153657790021,
            "nulls": 2383,
            "nunique": 594,
            "p01": 103186740.00000003,
            "p25": 3817750000.0,
            "p50": 11328000000.0,
            "p75": 31329500000.0,
            "p99": 191275179999.99988,
            "std": 41034610567.3028
          },
          "incomeTaxExpense": {
            "dtype": "float64",
            "mad_center": 1311000000.0,
            "mad_outliers": 332,
            "mad_scale": 1173000000.0,
            "max": 55064000000.0,
            "mean": 2810869549.4955897,
            "min": -23516000000.0,
            "null_rate": 0.0630547903275176,
            "nulls": 206,
            "nunique": 2595,
            "p01": -2111654599.9999993,
            "p25": 300000000.0,
            "p50": 1311000000.0,
            "p75": 2983000000.0,
            "p99": 23945000000.000023,
            "std": 4836968751.164555
          },
          "intangiblesExGoodwill": {
            "dtype": "float64",
            "mad_center": 3075000000.0,
            "mad_outliers": 184,
            "mad_scale": 2690400000.0,
            "max": 54942000000.0,
            "mean": 6804695933.118971,
            "min": 34700000.0,
            "null_rate": 0.5240281603917968,
            "nulls": 1712,
            "nunique": 1453,
            "p01": 52000000.0,
            "p25": 895500000.0,
            "p50": 3075000000.0,
            "p75": 7464000000.0,
            "p99": 48022180000.000015,
            "std": 9652653266.868633
          },
          "interestExpense": {
            "dtype": "float64",
            "mad_center": 595683500.0,
            "mad_outliers": 442,
            "mad_scale": 441683500.0,
            "max": 101350000000.0,
            "mean": 3723241287.878788,
            "min": 11000.0,
            "null_rate": 0.1919191919191919,
            "nulls": 627,
            "nunique": 1881,
            "p01": 11000.0,
            "p25": 275000000.0,
            "p50": 595683500.0,
            "p75": 1843000000.0,
            "p99": 81355160000.0,
            "std": 11953506746.628355
          },
          "inventory": {
            "dtype": "float64",
            "mad_center": 3209000000.0,
            "mad_outliers": 134,
            "mad_scale": 2745507000.0,
            "max": 65354000000.0,
            "mean": 7006697164.6143875,
            "min": 39100000.0,
            "null_rate": 0.5277012549739822,
            "nulls": 1724,
            "nunique": 1470,
            "p01": 49810000.0,
            "p25": 1119850000.0,
            "p50": 3209000000.0,
            "p75": 8639000000.0,
            "p99": 56088919999.99994,
            "std": 10284591524.197693
          },
          "is_amendment": {
            "dtype": "bool",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2
          },
          "longTermDebt": {
            "dtype": "float64",
            "mad_center": 18781500000.0,
            "mad_outliers": 270,
            "mad_scale": 12323000000.0,
            "max": 398965000000.0,
            "mean": 41049772804.580154,
            "min": 0.0,
            "null_rate": 0.1980410162228344,
            "nulls": 647,
            "nunique": 2526,
            "p01": 24069720.000000007,
            "p25": 8239000000.0,
            "p50": 18781500000.0,
            "p75": 38444750000.0,
            "p99": 287170609999.99994,
            "std": 64677497970.18588
          },
          "longTermDebtCurrentOnly": {
            "dtype": "float64",
            "mad_center": 1459500000.0,
            "mad_outliers": 29,
            "mad_scale": 1363000000.0,
            "max": 14009000000.0,
            "mean": 2219174827.586207,
            "min": -18200000.0,
            "null_rate": 0.7159473523109887,
            "nulls": 2339,
            "nunique": 712,
            "p01": 0.0,
            "p25": 249000000.0,
            "p50": 1459500000.0,
            "p75": 3270000000.0,
            "p99": 10942660000.0,
            "std": 2457281821.5224547
          },
          "minorityInterest": {
            "dtype": "float64",
            "mad_center": 435000000.0,
            "mad_outliers": 346,
            "mad_scale": 428000000.0,
            "max": 11871000000.0,
            "mean": 1388375649.5248742,
            "min": -187000000.0,
            "null_rate": 0.4524028160391797,
            "nulls": 1478,
            "nunique": 1243,
            "p01": -28000000.0,
            "p25": 33000000.0,
            "p50": 435000000.0,
            "p75": 1660000000.0,
            "p99": 8672319999.999996,
            "std": 2179435150.1092396
          },
          "netIncome": {
            "dtype": "float64",
            "mad_center": 4502000000.0,
            "mad_outliers": 372,
            "mad_scale": 3257000000.0,
            "max": 244205000000.0,
            "mean": 10408223303.677326,
            "min": -23528000000.0,
            "null_rate": 0.05234159779614325,
            "nulls": 171,
            "nunique": 2913,
            "p01": -5061550000.0,
            "p25": 1781250000.0,
            "p50": 4502000000.0,
            "p75": 11001250000.0,
            "p99": 97154350000.00003,
            "std": 18279824354.62165
          },
          "netIncome_q": {
            "dtype": "float64",
            "mad_center": 1129000000.0,
            "mad_outliers": 392,
            "mad_scale": 877500000.0,
            "max": 112193000000.0,
            "mean": 2678164704.3374233,
            "min": -49697000000.0,
            "null_rate": 0.0021426385062748698,
            "nulls": 7,
            "nunique": 2750,
            "p01": -2736009999.9999986,
            "p25": 405000000.0,
            "p50": 1129000000.0,
            "p75": 2863500000.0,
            "p99": 27581209999.999958,
            "std": 5696382897.954879
          },
          "netInterestIncome": {
            "dtype": "float64",
            "mad_center": 6990868000.0,
            "mad_outliers": 143,
            "mad_scale": 7539868000.0,
            "max": 99838000000.0,
            "mean": 18914414818.80109,
            "min": -2609000000.0,
            "null_rate": 0.7753290480563207,
            "nulls": 2533,
            "nunique": 669,
            "p01": -2348670000.0,
            "p25": -115000000.0,
            "p50": 6990868000.0,
            "p75": 43479000000.0,
            "p99": 92597740000.0,
            "std": 23080237183.794983
          },
          "netInvestmentIncome": {
            "dtype": "float64",
            "mad_center": 3423000000.0,
            "mad_outliers": 57,
            "mad_scale": 2215000000.0,
            "max": 24070000000.0,
            "mean": 6922964069.26407,
            "min": 389500000.0,
            "null_rate": 0.9292929292929293,
            "nulls": 3036,
            "nunique": 229,
            "p01": 399810000.0,
            "p25": 1808150000.0,
            "p50": 3423000000.0,
            "p75": 6709000000.0,
            "p99": 22518500000.0,
            "std": 7570836458.132638
          },
          "noninterestIncome": {
            "dtype": "float64",
            "mad_center": 32050000000.0,
            "mad_outliers": 1,
            "mad_scale": 11765000000.0,
            "max": 99570000000.0,
            "mean": 32151773481.48148,
            "min": 1535209000.0,
            "null_rate": 0.859504132231405,
            "nulls": 2808,
            "nunique": 456,
            "p01": 1766785560.0,
            "p25": 22103500000.0,
            "p50": 32050000000.0,
            "p75": 43862500000.0,
            "p99": 85244320000.00002,
            "std": 18980210013.124973
          },
          "operatingCashFlow": {
            "dtype": "float64",
            "mad_center": 6815000000.0,
            "mad_outliers": 420,
            "mad_scale": 4819000000.0,
            "max": 185675000000.0,
            "mean": 13512846343.918129,
            "min": -162534000000.0,
            "null_rate": 0.05785123966942149,
            "nulls": 189,
            "nunique": 2948,
            "p01": -34125910000.0,
            "p25": 3119073750.0,
            "p50": 6815000000.0,
            "p75": 15116500000.0,
            "p99": 112425000000.00002,
            "std": 24612240210.565613
          },
          "operatingIncome": {
            "dtype": "float64",
            "mad_center": 5242911500.0,
            "mad_outliers": 208,
            "mad_scale": 4038961500.0,
            "max": 162285000000.0,
            "mean": 12392811117.75529,
            "min": -25494000000.0,
            "null_rate": 0.33455769819406184,
            "nulls": 1093,
            "nunique": 1881,
            "p01": -6711437670.0,
            "p25": 2346075000.0,
            "p50": 5242911500.0,
            "p75": 13413000000.0,
            "p99": 119749389999.99998,
            "std": 21588347564.136307
          },
          "operatingLeaseLiability": {
            "dtype": "float64",
            "mad_center": 1469000000.0,
            "mad_outliers": 259,
            "mad_scale": 1290100000.0,
            "max": 30190000000.0,
            "mean": 4611392377.69447,
            "min": 0.0,
            "null_rate": 0.6734006734006734,
            "nulls": 2200,
            "nunique": 990,
            "p01": 23820460.0,
            "p25": 519014000.0,
            "p50": 1469000000.0,
            "p75": 7715850000.0,
            "p99": 21883419999.999996,
            "std": 5861980970.751634
          },
          "operatingMargins": {
            "dtype": "float64",
            "mad_center": 0.19908692991951804,
            "mad_outliers": 16,
            "mad_scale": 0.10956570660906245,
            "max": 0.8033951437730371,
            "mean": 0.1934994485074233,
            "min": -4.25584765058994,
            "null_rate": 0.34710743801652894,
            "nulls": 1134,
            "nunique": 2118,
            "p01": -0.1557224491909984,
            "p25": 0.08474038314596616,
            "p50": 0.19908692991951804,
            "p75": 0.30106042542511136,
            "p99": 0.5279510183693497,
            "std": 0.23953390833117094
          },
          "optionOverhang": {
            "dtype": "float64",
            "mad_center": 0.006911481557092092,
            "mad_outliers": 307,
            "mad_scale": 0.003862309429276123,
            "max": 0.19963031423290212,
            "mean": 0.011445512699033662,
            "min": -0.03372806321342636,
            "null_rate": 0.09029690847872666,
            "nulls": 295,
            "nunique": 2720,
            "p01": 0.0,
            "p25": 0.0038107276775788956,
            "p50": 0.006911481557092092,
            "p75": 0.013085689224669073,
            "p99": 0.07449526752781742,
            "std": 0.015153977033972851
          },
          "ppeGross": {
            "dtype": "float64",
            "mad_center": 33297000000.0,
            "mad_outliers": 82,
            "mad_scale": 23269000000.0,
            "max": 570059000000.0,
            "mean": 49709853118.69225,
            "min": 124344000.0,
            "null_rate": 0.5693296602387512,
            "nulls": 1860,
            "nunique": 1385,
            "p01": 226012780.0,
            "p25": 11809500000.0,
            "p50": 33297000000.0,
            "p75": 60528000000.0,
            "p99": 408452220000.0008,
            "std": 65359021992.801445
          },
          "ppeNet": {
            "dtype": "float64",
            "mad_center": 11268600000.0,
            "mad_outliers": 278,
            "mad_scale": 9115400000.0,
            "max": 313076000000.0,
            "mean": 26447638161.0338,
            "min": 81289000.0,
            "null_rate": 0.23018059381695746,
            "nulls": 752,
            "nunique": 2418,
            "p01": 193933620.0,
            "p25": 3625000000.0,
            "p50": 11268600000.0,
            "p75": 24859300000.0,
            "p99": 250815200000.00003,
            "std": 45481016789.06146
          },
          "premiumsEarned": {
            "dtype": "float64",
            "mad_center": 36803000000.0,
            "mad_outliers": 38,
            "mad_scale": 21344000000.0,
            "max": 353256000000.0,
            "mean": 60951905362.776024,
            "min": 662000000.0,
            "null_rate": 0.9029690847872666,
            "nulls": 2950,
            "nunique": 281,
            "p01": 662000000.0,
            "p25": 16098000000.0,
            "p50": 36803000000.0,
            "p75": 76132000000.0,
            "p99": 338055079999.9997,
            "std": 76630628987.47688
          },
          "pretaxIncome": {
            "dtype": "float64",
            "mad_center": 6003500000.0,
            "mad_outliers": 323,
            "mad_scale": 4569420500.0,
            "max": 299269000000.0,
            "mean": 13826781532.146639,
            "min": -30576000000.0,
            "null_rate": 0.14416896235078053,
            "nulls": 471,
            "nunique": 2624,
            "p01": -6091200000.0,
            "p25": 2299250000.0,
            "p50": 6003500000.0,
            "p75": 14365000000.0,
            "p99": 119002300000.00002,
            "std": 23374575687.065514
          },
          "profitMargins": {
            "dtype": "float64",
            "mad_center": 0.1507632894402761,
            "mad_outliers": 34,
            "mad_scale": 0.08737811360293721,
            "max": 0.8428179410653303,
            "mean": 0.15446890711054825,
            "min": -3.589868782422948,
            "null_rate": 0.06458524640342822,
            "nulls": 211,
            "nunique": 3041,
            "p01": -0.12207680513002368,
            "p25": 0.06759684780875444,
            "p50": 0.1507632894402761,
            "p75": 0.2414150804229587,
            "p99": 0.5207239415701385,
            "std": 0.18931090425133404
          },
          "publication_form": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 4
          },
          "realizedInvestmentGains": {
            "dtype": "float64",
            "mad_center": 78000000.0,
            "mad_outliers": 9,
            "mad_scale": 293000000.0,
            "max": 2768900000.0,
            "mean": 34555605.38116592,
            "min": -1912200000.0,
            "null_rate": 0.9317416590143863,
            "nulls": 3044,
            "nunique": 142,
            "p01": -1553000000.0,
            "p25": -140500000.0,
            "p50": 78000000.0,
            "p75": 399500000.0,
            "p99": 1796998000.0000002,
            "std": 591768081.7429997
          },
          "regime": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 8
          },
          "rentalIncome": {
            "dtype": "float64",
            "mad_center": 962000000.0,
            "mad_outliers": 52,
            "mad_scale": 901000000.0,
            "max": 11106800000.0,
            "mean": 2826327327.0676694,
            "min": 37000000.0,
            "null_rate": 0.9185797367615549,
            "nulls": 3001,
            "nunique": 223,
            "p01": 37000000.0,
            "p25": 141500000.0,
            "p50": 962000000.0,
            "p75": 5216365750.0,
            "p99": 10529070000.000004,
            "std": 3356704369.239549
          },
          "researchAndDevelopment": {
            "dtype": "float64",
            "mad_center": 4591000000.0,
            "mad_outliers": 82,
            "mad_scale": 3703000000.0,
            "max": 71634000000.0,
            "mean": 8037229317.318436,
            "min": 64223000.0,
            "null_rate": 0.7260483624119988,
            "nulls": 2372,
            "nunique": 857,
            "p01": 85453620.0,
            "p25": 1358212500.0,
            "p50": 4591000000.0,
            "p75": 9627000000.0,
            "p99": 51065219999.99992,
            "std": 10892811495.410599
          },
          "restrictedCash": {
            "dtype": "float64",
            "mad_center": 345000000.0,
            "mad_outliers": 215,
            "mad_scale": 332747000.0,
            "max": 47100000000.0,
            "mean": 5652847189.473684,
            "min": 0.0,
            "null_rate": 0.7964493419038874,
            "nulls": 2602,
            "nunique": 552,
            "p01": 0.0,
            "p25": 64000000.0,
            "p50": 345000000.0,
            "p75": 6200000000.0,
            "p99": 40980000000.00001,
            "std": 9750058331.75627
          },
          "retainedEarnings": {
            "dtype": "float64",
            "mad_center": 23914300000.0,
            "mad_outliers": 344,
            "mad_scale": 21405700000.0,
            "max": 798959000000.0,
            "mean": 56095589241.94871,
            "min": -34076000000.0,
            "null_rate": 0.039179675543311906,
            "nulls": 128,
            "nunique": 3114,
            "p01": -10469480000.0,
            "p25": 6721650000.0,
            "p50": 23914300000.0,
            "p75": 66778000000.0,
            "p99": 457578239999.99976,
            "std": 93013842136.9081
          },
          "returnOnEquity": {
            "dtype": "float64",
            "mad_center": 0.13867400783612766,
            "mad_outliers": 328,
            "mad_scale": 0.06883703915829104,
            "max": 43.58620689655172,
            "mean": -58.428302117328606,
            "min": -180657.64023210833,
            "null_rate": 0.05723905723905724,
            "nulls": 187,
            "nunique": 3075,
            "p01": -2.354283957283657,
            "p25": 0.0881872007079158,
            "p50": 0.13867400783612766,
            "p75": 0.2493609718221879,
            "p99": 3.2034789892607276,
            "std": 3255.229732931302
          },
          "revenue_q": {
            "dtype": "float64",
            "mad_center": 9856000000.0,
            "mad_outliers": 339,
            "mad_scale": 7984000000.0,
            "max": 190656000000.0,
            "mean": 20002454663.48816,
            "min": -34644000000.0,
            "null_rate": 0.004897459442913988,
            "nulls": 16,
            "nunique": 3075,
            "p01": 243487000.0,
            "p25": 4186278000.0,
            "p50": 9856000000.0,
            "p75": 22274000000.0,
            "p99": 129686500000.0,
            "std": 27467050567.201405
          },
          "sellingGeneralAdmin": {
            "dtype": "float64",
            "mad_center": 6647950000.0,
            "mad_outliers": 110,
            "mad_scale": 5147000000.0,
            "max": 150972000000.0,
            "mean": 14224399748.666666,
            "min": 55180000.0,
            "null_rate": 0.5408631772268135,
            "nulls": 1767,
            "nunique": 1444,
            "p01": 94073790.0,
            "p25": 2412500000.0,
            "p50": 6647950000.0,
            "p75": 19069250000.0,
            "p99": 120790949999.99995,
            "std": 22420013949.119213
          },
          "sharesOutstanding": {
            "dtype": "float64",
            "mad_center": 688587083.0,
            "mad_outliers": 673,
            "mad_scale": 360174378.0,
            "max": 4819056000000000.0,
            "mean": 1576244575964.1614,
            "min": 0.0,
            "null_rate": 0.0630547903275176,
            "nulls": 206,
            "nunique": 3031,
            "p01": 47021870.6,
            "p25": 407989898.0,
            "p50": 688587083.0,
            "p75": 1984267239.0,
            "p99": 14970324800.000008,
            "std": 87102403196624.72
          },
          "shortTermBorrowingsOnly": {
            "dtype": "float64",
            "mad_center": 2037000000.0,
            "mad_outliers": 264,
            "mad_scale": 1871279000.0,
            "max": 252927000000.0,
            "mean": 14366870494.51554,
            "min": 0.0,
            "null_rate": 0.665136210590756,
            "nulls": 2173,
            "nunique": 943,
            "p01": 0.0,
            "p25": 461000000.0,
            "p50": 2037000000.0,
            "p75": 9463500000.0,
            "p99": 120513499999.99977,
            "std": 29865562218.489952
          },
          "shortTermDebt": {
            "dtype": "float64",
            "mad_center": 2102500000.0,
            "mad_outliers": 342,
            "mad_scale": 1987000000.0,
            "max": 252927000000.0,
            "mean": 9083488134.9065,
            "min": -634000000.0,
            "null_rate": 0.3125191307009489,
            "nulls": 1021,
            "nunique": 1839,
            "p01": 0.0,
            "p25": 504500000.0,
            "p50": 2102500000.0,
            "p75": 6743665750.0,
            "p99": 103938000000.00023,
            "std": 21722520057.33973
          },
          "shortTermInvestments": {
            "dtype": "float64",
            "mad_center": 1874950000.0,
            "mad_outliers": 229,
            "mad_scale": 1762900000.0,
            "max": 186563000000.0,
            "mean": 11494876346.456694,
            "min": 0.0,
            "null_rate": 0.6890113253749618,
            "nulls": 2251,
            "nunique": 952,
            "p01": 1015000.0,
            "p25": 302500000.0,
            "p50": 1874950000.0,
            "p75": 8142500000.0,
            "p99": 118641000000.00002,
            "std": 24535091162.542263
          },
          "stockBasedComp": {
            "dtype": "float64",
            "mad_center": 292000000.0,
            "mad_outliers": 454,
            "mad_scale": 221000000.0,
            "max": 28147000000.0,
            "mean": 1446685901.2531328,
            "min": 2049000.0,
            "null_rate": 0.38934802571166205,
            "nulls": 1272,
            "nunique": 1478,
            "p01": 2232000.0,
            "p25": 118495000.0,
            "p50": 292000000.0,
            "p75": 1078500000.0,
            "p99": 17536379999.99996,
            "std": 3221635182.386452
          },
          "stockholdersEquity": {
            "dtype": "float64",
            "mad_center": 24426000000.0,
            "mad_outliers": 466,
            "mad_scale": 19661000000.0,
            "max": 750177000000.0,
            "mean": 60423355267.968414,
            "min": -23562000000.0,
            "null_rate": 0.011631466176920723,
            "nulls": 38,
            "nunique": 3194,
            "p01": -7439920000.0,
            "p25": 10815000000.0,
            "p50": 24426000000.0,
            "p75": 68095000000.0,
            "p99": 389752759999.9992,
            "std": 88280750132.55583
          },
          "ticker": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 54
          },
          "totalAssets": {
            "dtype": "float64",
            "mad_center": 94670000000.0,
            "mad_outliers": 520,
            "mad_scale": 63454685000.0,
            "max": 5015069000000.0,
            "mean": 337361279558.96356,
            "min": 4776348.0,
            "null_rate": 0.008876645240281604,
            "nulls": 29,
            "nunique": 3223,
            "p01": 1689505230.0,
            "p25": 39291375000.0,
            "p50": 94670000000.0,
            "p75": 229618750000.0,
            "p99": 3253585600000.0015,
            "std": 669596764482.5135
          },
          "totalDebt": {
            "dtype": "float64",
            "mad_center": 20878500000.0,
            "mad_outliers": 276,
            "mad_scale": 14718000000.0,
            "max": 501315000000.0,
            "mean": 47437796443.892044,
            "min": 0.0,
            "null_rate": 0.13804713804713806,
            "nulls": 451,
            "nunique": 2721,
            "p01": 34812300.0,
            "p25": 8531250000.0,
            "p50": 20878500000.0,
            "p75": 48823075000.0,
            "p99": 346488999999.9997,
            "std": 75671454449.64613
          },
          "totalLiabilities": {
            "dtype": "float64",
            "mad_center": 61323000000.0,
            "mad_outliers": 503,
            "mad_scale": 44768000000.0,
            "max": 4640471000000.0,
            "mean": 277671104144.1533,
            "min": -8237223652.0,
            "null_rate": 0.011631466176920723,
            "nulls": 38,
            "nunique": 3206,
            "p01": 327851960.0,
            "p25": 25114000000.0,
            "p50": 61323000000.0,
            "p75": 158782000000.0,
            "p99": 2970025119999.999,
            "std": 611494156962.015
          },
          "totalRevenue": {
            "dtype": "float64",
            "mad_center": 39234500000.0,
            "mad_outliers": 323,
            "mad_scale": 31971447500.0,
            "max": 725305000000.0,
            "mean": 79660222038.13834,
            "min": -194700000.0,
            "null_rate": 0.0529537802265075,
            "nulls": 173,
            "nunique": 3035,
            "p01": 1548629490.0,
            "p25": 17297750000.0,
            "p50": 39234500000.0,
            "p75": 88946250000.0,
            "p99": 510284050000.0003,
            "std": 107728003537.75468
          }
        },
        "kind": "extract",
        "pk": [
          "ticker",
          "as_of"
        ],
        "pk_checked_cols": [
          "ticker",
          "as_of"
        ],
        "pk_checked_rows": 3267,
        "pk_complete": true,
        "pk_duplicate_rows": 0,
        "pk_missing_cols": [],
        "rows": 3267,
        "sample_date_max": "2026-08-10",
        "sample_date_min": "2009-07-31",
        "sampled_rows": 3267,
        "scope": {
          "limit": 500000,
          "since": null,
          "tickers": null
        },
        "table": "fundamentals_history"
      },
      "fundamentals_reason_codes": {
        "columns": [
          "ticker",
          "as_of",
          "field",
          "dc_code",
          "combined_into"
        ],
        "date_col": "as_of",
        "date_max": "2026-08-10",
        "date_min": "2009-07-31",
        "exists": true,
        "fields": {
          "as_of": {
            "dtype": "object",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1302
          },
          "combined_into": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 76004,
            "nunique": 0
          },
          "dc_code": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 16
          },
          "field": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 60
          },
          "ticker": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 54
          }
        },
        "kind": "extract",
        "pk": [
          "ticker",
          "as_of",
          "field",
          "dc_code"
        ],
        "pk_checked_cols": [
          "ticker",
          "as_of",
          "field",
          "dc_code"
        ],
        "pk_checked_rows": 76004,
        "pk_complete": true,
        "pk_duplicate_rows": 0,
        "pk_missing_cols": [],
        "rows": 76004,
        "sample_date_max": "2026-08-10",
        "sample_date_min": "2009-07-31",
        "sampled_rows": 76004,
        "scope": {
          "limit": 500000,
          "since": null,
          "tickers": null
        },
        "table": "fundamentals_reason_codes"
      }
    }
  },
  "scope": {
    "limit": 500000,
    "since": null,
    "tables": [
      "fundamentals_employees",
      "fundamentals_facts",
      "fundamentals_history",
      "fundamentals_reason_codes"
    ],
    "tickers": [],
    "unknown_tables": []
  },
  "session_id": "6ea3721c-02be-4c23-97e4-584ba5be6f14",
  "type": "DATA"
}
```

