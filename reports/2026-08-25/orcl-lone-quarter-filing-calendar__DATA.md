---
type: DATA
session_id: 7aff37c0-3151-4721-9e1d-090217de16ad
generated_at: 2026-08-25T22:20:17+00:00
baseline: {head_sha: 6d119bf4bf9e63d57014710b92935ccdbb6a0c4a}
generator: scripts/dod/data_profile.py@1
---

## 1. Scope

**SAMPLE SCOPE** — a metric without its scope is not a measurement:

- tables: fundamentals_facts, fundamentals_history
- tickers: **all** (no ticker filter)
- since: **no lower bound**
- row limit per table: **none**
- full-scope tables (eligible to set the baseline): fundamentals_facts, fundamentals_history

**What was asked:** settle validator cluster `2603621e89ab` -- ORCL `totalRevenue`, 47
findings across 7 agreeing checks -- end to end: challenge the checks, read the filings, fix
the layer that is wrong, rebuild, and prove it with a measured row-count delta at the ORIGINAL
54-ticker scope. One cluster, not a sweep.

Oracle stamps its full-year `us-gaap:Revenues` into a 91-day fourth-quarter context in its
fiscal 2020, 2021 and 2022 10-Ks -- 9 rows spanning fiscal 2018-2022, fiscal 2022 reading
$42,440M where the true fourth quarter is $11,840M. The checks were challenged before the
data and are sound: the same filing carries the correct figure under
`RevenueFromContractWithCustomerExcludingAssessedTax` on a 364-day window at the same
$42,440M, so the filer contradicts itself inside one document. None of the 7 checks was over
its declared ceiling in this run, and the `likely-check-or-catalogue` routing hint was
discounted because the report itself declares that hint non-discriminating on this roster.

The profile above is the WHOLE of both tables (no ticker filter, no lower bound, no row cap),
because the fix had to be shown not to cost the other 53 tickers anything. The change itself
removes rows for one ticker.

## 2. Gates

| Gate | Check | Verdict | Detail |
|---|---|---|---|
| D1 | declared PK unique over the rows profiled | **PASS** | unique across 2 table(s): fundamentals_facts, fundamentals_history |
| D2 | row count not decreased | **N/A** | no full-scope baseline to compare against — this run records one |
| D3 | no column lost | **N/A** | no baseline columns recorded yet |
| D4 | date range covers the expected window | **PASS** | every dated table reaches 2026-06-30 |
| D5 | per-field null rate not worse | **N/A** | no full-scope baseline null rates to compare against |

**All gates pass** (N/A gates are stated above, not skipped).

## 3. Metrics

_Observed values only — no verdicts. `rows`, `date_min` and `date_max` are **table-wide** (server-side); every other number is over the **sample** described in §1. Do not compare across the two._

**Tables**

| table | exists | rows | sampled | cols | pk | pk_absent_cols | pk_dupes | date_min | date_max | sample_date_min | sample_date_max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fundamentals_facts | yes | 316,128 | 316,128 | 26 | ticker,accession_number,field,duration_type,period_end | — | 0 | 2009-07-31 | 2026-08-10 | 2009-07-31 | 2026-08-10 |
| fundamentals_history | yes | 3,258 | 3,258 | 69 | ticker,as_of | — | 0 | 2009-07-31 | 2026-08-10 | 2009-07-31 | 2026-08-10 |

**Fields** (worst null rate first, top 60)

| table | field | dtype | null_% | nunique | mean | std | min | p01 | p50 | p99 | max | mad_outliers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fundamentals_history | amended_fields | str | 99.36 | 14 | — | — | — | — | — | — | — | — |
| fundamentals_history | amended_fiscal_end | object | 99.36 | 21 | — | — | — | — | — | — | — | — |
| fundamentals_facts | adjustment | str | 99.32 | 380 | — | — | — | — | — | — | — | — |
| fundamentals_facts | root_anchor | str | 97.67 | 9 | — | — | — | — | — | — | — | — |
| fundamentals_facts | roll_up_children | str | 96.99 | 57 | — | — | — | — | — | — | — | — |
| fundamentals_history | realizedInvestmentGains | float64 | 93.16 | 142 | 3.45556e+07 | 5.91768e+08 | -1.9122e+09 | -1.553e+09 | 7.8e+07 | 1.797e+09 | 2.7689e+09 | 9 |
| fundamentals_history | netInvestmentIncome | float64 | 92.91 | 229 | 6.92296e+09 | 7.57084e+09 | 3.895e+08 | 3.9981e+08 | 3.423e+09 | 2.25185e+10 | 2.407e+10 | 57 |
| fundamentals_history | rentalIncome | float64 | 91.84 | 223 | 2.82633e+09 | 3.3567e+09 | 3.7e+07 | 3.7e+07 | 9.62e+08 | 1.05291e+10 | 1.11068e+10 | 52 |
| fundamentals_history | premiumsEarned | float64 | 90.27 | 281 | 6.09519e+10 | 7.66306e+10 | 6.62e+08 | 6.62e+08 | 3.6803e+10 | 3.38055e+11 | 3.53256e+11 | 38 |
| fundamentals_history | financeLeaseLiability | float64 | 85.94 | 360 | 2.57749e+09 | 7.28731e+09 | 4e+06 | 5e+06 | 6.055e+08 | 4.22059e+10 | 6.6594e+10 | 68 |
| fundamentals_history | noninterestIncome | float64 | 85.91 | 456 | 3.21518e+10 | 1.89802e+10 | 1.53521e+09 | 1.76679e+09 | 3.205e+10 | 8.52443e+10 | 9.957e+10 | 1 |
| fundamentals_history | restrictedCash | float64 | 79.59 | 552 | 5.65285e+09 | 9.75006e+09 | 0 | 0 | 3.45e+08 | 4.098e+10 | 4.71e+10 | 215 |
| fundamentals_facts | dc_code | str | 79.48 | 7 | — | — | — | — | — | — | — | — |
| fundamentals_history | netInterestIncome | float64 | 77.47 | 669 | 1.89144e+10 | 2.30802e+10 | -2.609e+09 | -2.34867e+09 | 6.99087e+09 | 9.25977e+10 | 9.9838e+10 | 143 |
| fundamentals_history | grossMargins | float64 | 73.33 | 861 | 0.367 | 0.22291 | -0.097751 | 0.00241924 | 0.396585 | 0.740393 | 0.762218 | 0 |
| fundamentals_history | grossProfit | float64 | 72.87 | 594 | 2.87618e+10 | 4.10346e+10 | -5.685e+09 | 1.03187e+08 | 1.1328e+10 | 1.91275e+11 | 2.27123e+11 | 113 |
| fundamentals_history | researchAndDevelopment | float64 | 72.53 | 857 | 8.03723e+09 | 1.08928e+10 | 6.4223e+07 | 8.54536e+07 | 4.591e+09 | 5.10652e+10 | 7.1634e+10 | 82 |
| fundamentals_history | longTermDebtCurrentOnly | float64 | 71.79 | 704 | 2.2381e+09 | 2.46164e+09 | -1.82e+07 | 0 | 1.484e+09 | 1.09464e+10 | 1.4009e+10 | 29 |
| fundamentals_history | shortTermInvestments | float64 | 68.82 | 952 | 1.14949e+10 | 2.45351e+10 | 0 | 1.015e+06 | 1.87495e+09 | 1.18641e+11 | 1.86563e+11 | 229 |
| fundamentals_history | operatingLeaseLiability | float64 | 67.25 | 990 | 4.61139e+09 | 5.86198e+09 | 0 | 2.38205e+07 | 1.469e+09 | 2.18834e+10 | 3.019e+10 | 259 |
| fundamentals_history | shortTermBorrowingsOnly | float64 | 66.42 | 943 | 1.43669e+10 | 2.98656e+10 | 0 | 0 | 2.037e+09 | 1.20513e+11 | 2.52927e+11 | 264 |
| fundamentals_history | costOfRevenue | float64 | 65.07 | 1,021 | 6.82508e+10 | 9.92242e+10 | 4.06e+07 | 4.06e+07 | 2.83215e+10 | 4.75946e+11 | 5.4415e+11 | 139 |
| fundamentals_facts | period_days | float64 | 57.52 | 46 | 194.286 | 112.74 | 81 | 83 | 180 | 365 | 1,095 | 2 |
| fundamentals_facts | period_start | object | 57.52 | 622 | — | — | — | — | — | — | — | — |
| fundamentals_history | ppeGross | float64 | 57.09 | 1,376 | 4.98126e+10 | 6.55565e+10 | 1.24344e+08 | 2.25219e+08 | 3.3295e+10 | 4.09988e+11 | 5.70059e+11 | 82 |
| fundamentals_history | sellingGeneralAdmin | float64 | 54.24 | 1,435 | 1.4296e+10 | 2.24686e+10 | 5.518e+07 | 9.37479e+07 | 6.683e+09 | 1.21222e+11 | 1.50972e+11 | 110 |
| fundamentals_history | inventory | float64 | 52.92 | 1,464 | 7.04719e+09 | 1.03011e+10 | 3.91e+07 | 4.9765e+07 | 3.298e+09 | 5.61631e+10 | 6.5354e+10 | 121 |
| fundamentals_history | intangiblesExGoodwill | float64 | 52.27 | 1,453 | 6.8047e+09 | 9.65265e+09 | 3.47e+07 | 5.2e+07 | 3.075e+09 | 4.80222e+10 | 5.4942e+10 | 184 |
| fundamentals_history | accumulatedDepreciation | float64 | 49.94 | 1,595 | 1.91114e+10 | 2.75128e+10 | -1.9086e+10 | 5.03425e+07 | 1.1925e+10 | 1.16601e+11 | 2.7851e+11 | 110 |
| fundamentals_history | minorityInterest | float64 | 45.09 | 1,243 | 1.38838e+09 | 2.17944e+09 | -1.87e+08 | -2.8e+07 | 4.35e+08 | 8.67232e+09 | 1.1871e+10 | 346 |
| fundamentals_facts | role_uri | str | 41.55 | 1,849 | — | — | — | — | — | — | — | — |
| fundamentals_history | accountsPayable | float64 | 39.44 | 1,871 | 8.21845e+09 | 1.36238e+10 | 126,022 | 4.05389e+07 | 2.436e+09 | 6.25671e+10 | 7.7088e+10 | 310 |
| fundamentals_history | stockBasedComp | float64 | 39.04 | 1,470 | 1.45284e+09 | 3.22763e+09 | 2.049e+06 | 2.232e+06 | 2.975e+08 | 1.76015e+10 | 2.8147e+10 | 449 |
| fundamentals_history | accountsReceivable | float64 | 37.91 | 1,898 | 7.43749e+09 | 9.79474e+09 | 1e+07 | 1.46185e+08 | 3.98e+09 | 4.77711e+10 | 8.0876e+10 | 203 |
| fundamentals_history | ebitda | float64 | 37.75 | 1,961 | 1.65434e+10 | 2.54655e+10 | -1.0521e+10 | -2.11446e+09 | 8.824e+09 | 1.36625e+11 | 1.98685e+11 | 162 |
| fundamentals_history | operatingMargins | float64 | 34.75 | 2,111 | 0.193115 | 0.239834 | -4.25585 | -0.156692 | 0.198475 | 0.528138 | 0.803395 | 16 |
| fundamentals_history | currentAssets | float64 | 34.1 | 2,119 | 3.39489e+10 | 4.06266e+10 | 515,881 | 7.37673e+08 | 1.6258e+10 | 1.74089e+11 | 3.43524e+11 | 231 |
| fundamentals_history | currentLiabilities | float64 | 34.07 | 2,117 | 2.4796e+10 | 2.95938e+10 | 511,697 | 2.15722e+08 | 1.3388e+10 | 1.26851e+11 | 1.76392e+11 | 237 |
| fundamentals_history | operatingIncome | float64 | 33.49 | 1,874 | 1.24083e+10 | 2.16215e+10 | -2.5494e+10 | -6.71801e+09 | 5.214e+09 | 1.1983e+11 | 1.62285e+11 | 210 |
| fundamentals_history | shortTermDebt | float64 | 31.34 | 1,831 | 9.11888e+09 | 2.1759e+10 | -6.34e+08 | 0 | 2.117e+09 | 1.0405e+11 | 2.52927e+11 | 341 |
| fundamentals_history | freeCashflow | float64 | 30.6 | 2,164 | 9.1852e+09 | 1.78249e+10 | -9.0387e+10 | -1.56832e+10 | 3.527e+09 | 9.1465e+10 | 1.36683e+11 | 275 |
| fundamentals_history | capex | float64 | 30.17 | 2,066 | 5.75962e+09 | 9.77908e+09 | 5.001e+06 | 3.89889e+07 | 2.716e+09 | 4.5458e+10 | 1.32402e+11 | 273 |
| fundamentals_history | goodwill | float64 | 26.4 | 1,828 | 1.99366e+10 | 2.32307e+10 | 0 | 2.0098e+07 | 9.908e+09 | 1.00444e+11 | 1.19661e+11 | 228 |
| fundamentals_history | ppeNet | float64 | 23.08 | 2,409 | 2.64651e+10 | 4.55617e+10 | 8.1289e+07 | 1.93764e+08 | 1.12326e+10 | 2.50839e+11 | 3.13076e+11 | 279 |
| fundamentals_facts | source_concept | str | 22.22 | 140 | — | — | — | — | — | — | — | — |
| fundamentals_facts | decimals | str | 20.52 | 14 | — | — | — | — | — | — | — | — |
| fundamentals_facts | value | float64 | 20.52 | 67,982 | 5.24565e+10 | 9.6154e+12 | -2.67506e+11 | -6.88e+08 | 3.268e+09 | 4.37241e+11 | 4.81906e+15 | 56,780 |
| fundamentals_facts | unit | str | 20.39 | 190 | — | — | — | — | — | — | — | — |
| fundamentals_history | longTermDebt | float64 | 19.86 | 2,517 | 4.11541e+10 | 6.47644e+10 | 0 | 2.36478e+07 | 1.88805e+10 | 2.87232e+11 | 3.98965e+11 | 270 |
| fundamentals_history | interestExpense | float64 | 19.24 | 1,873 | 3.73437e+09 | 1.19724e+10 | 11,000 | 11,000 | 5.97e+08 | 8.13602e+10 | 1.0135e+11 | 441 |
| fundamentals_history | effectiveTaxRate | float64 | 15.38 | 2,741 | 0.198765 | 0.347879 | -9.10667 | -0.477701 | 0.204437 | 0.944718 | 5.12308 | 126 |
| fundamentals_history | pretaxIncome | float64 | 14.18 | 2,624 | 1.38268e+10 | 2.33746e+10 | -3.0576e+10 | -6.0912e+09 | 6.0035e+09 | 1.19002e+11 | 2.99269e+11 | 323 |
| fundamentals_history | debtToEquity | float64 | 14.06 | 2,777 | 0.75432 | 12.8586 | -560.411 | -14.0841 | 0.635519 | 9.64941 | 167.483 | 296 |
| fundamentals_history | totalDebt | float64 | 13.84 | 2,712 | 4.75544e+10 | 7.57646e+10 | 0 | 3.47743e+07 | 2.0936e+10 | 3.46815e+11 | 5.01315e+11 | 276 |
| fundamentals_history | depAmort | float64 | 12.71 | 2,432 | 3.62304e+09 | 4.741e+09 | 6.364e+06 | 2.88926e+07 | 2.1655e+09 | 2.29689e+10 | 4.6463e+10 | 304 |
| fundamentals_history | epsDiluted | float64 | 10.37 | 2,899 | 32,168 | 613,960 | -62.2434 | -8.25405 | 5.09099 | 50.4288 | 1.23129e+07 | 258 |
| fundamentals_history | optionOverhang | float64 | 9.02 | 2,712 | 0.0114407 | 0.0151741 | -0.0337281 | 0 | 0.00688815 | 0.0745063 | 0.19963 | 308 |
| fundamentals_history | dilutedShares | float64 | 8.53 | 2,770 | 1.70456e+09 | 2.57931e+09 | 713.639 | 4.33928e+07 | 6.87643e+08 | 1.22716e+10 | 2.4804e+10 | 680 |
| fundamentals_history | basicShares | float64 | 7.27 | 2,805 | 2.19259e+09 | 3.00948e+10 | 711.013 | 1.64305e+06 | 6.49913e+08 | 1.21962e+10 | 1.64989e+12 | 691 |
| fundamentals_history | profitMargins | float64 | 6.41 | 3,034 | 0.154354 | 0.189513 | -3.58987 | -0.122107 | 0.150372 | 0.520995 | 0.842818 | 34 |

## 4. Evidence

- baseline file: `reports/baselines/data_profile.json` (0 table(s) recorded)
- `fundamentals_facts`: 316,128 rows, 26 cols, 316,128 sampled
- `fundamentals_history`: 3,258 rows, 69 cols, 3,258 sampled

## 5. Regressions, gaps and deliberate omissions

- **D2, D3 and D5 came back N/A because no full-scope baseline existed** -- this run records
  the first one. "Row count not decreased" and "null rate not worse" are therefore UNPROVEN by
  the generator here, not passed. Checked by hand instead: `fundamentals_facts` went
  316,137 -> 316,128 (-9, the intended drop), and no ticker other than ORCL lost a row, sized
  by replaying the guard's predicate over the table in SQL before any code was written.
- **The fix does NOT repair Oracle's point-in-time fourth quarters, and no check can see the
  hole.** `fundamentals_history.revenue_q` is `_latest(quarters, source)` -- the newest
  discrete quarter we could compute -- and it carries no period of its own, while the row's
  `fiscal_end` / `fiscal_quarter` come from the filing event. At `as_of` 2020-06-22,
  2021-06-21 and 2022-06-21 ORCL's Q4 row therefore holds the preceding Q3 (9,796 / 10,085 /
  10,513 $M), because at those dates no annual window had been published yet. Table-wide the
  signature -- `revenue_q` exactly equal to the previous `as_of` -- is **60 rows across 17
  tickers**, 1.89% of non-null `revenue_q`, MAA worst at 21. None of the six history-substrate
  checks (`grain`, `column_contract`, `code_vocabulary`, `unexplained_null`, `pit_leak`,
  `coverage_universe`) tests value repetition, so this is invisible to the validator. Scoped
  out deliberately: different defect, different layer, and this cluster's 47 findings are all
  facts-substrate.
- **Oracle's correct FY2020 annual revenue ($39,068M) is still not captured.** It sits in all
  three filings under the ASC 606 element, but `totalRevenue` resolves to `us-gaap:Revenues`
  and resolution is period-agnostic by design. FY2021 and FY2022 are recovered later from the
  FY2023/FY2024 10-Ks, whose filing agent windows them correctly; FY2020 is recovered nowhere.
  A resolution-layer fix was scoped out: its candidate set spans all 9 quarterly-only
  (ticker, field) pairs -- including four filers' legitimate ASC 270 `grossProfit` tables --
  and cannot be sized from the stored table, which keeps only the CHOSEN tag.
- **ORCL was rebuilt with `--rebuild`, so some of its movement is refetch drift, not this
  fix.** `series_shape` -8 and `coverage_field` +3 on ORCL are not attributable to the guard;
  all 15 of its `coverage_field` findings were re-keyed because ORCL's old series key
  (`2011-09-23..2026-06-22`, filing dates) became `2011-08-31..2026-05-31`, period-ends, the
  shape every other ticker already had. Attribution is clean only at cluster and per-check
  level, where every drop equals that check's cluster count exactly.
- **Two pre-existing repo defects surfaced and were NOT touched.** (a) `.gitignore:77`'s
  Rust-style `target/` rule hides `src/data_aggregate/utils/target/`, a real Python package,
  from version control -- it exists only on this machine, and two tests error at setup in any
  clean checkout. (b) 10 `data_aggregate` tests fail on `dev` independently of this work,
  verified by running each on a clean worktree at `6d119bf` with the change absent; they
  include a cube-fingerprint guard whose baseline the file itself documents as deliberately
  not regenerated.
- **`validate fix record` could not derive the run pair for a cluster closed to zero.**
  `_resolve_runs` defaults `after` to "the latest run that saw the cluster", and a fully
  closed cluster is not seen by the run that proved it -- so the derivation picked the BEFORE
  run and refused. `--after` was passed explicitly (the CLI's own sibling message directs
  this); the `scope_hash` guard and the count derivation both still applied and returned
  47 -> 0 unaided. `validate fix show` has the same blind spot cosmetically: it renders the
  cluster "as of" run `3df52ae9af75` and lists the closed 47 as "still pending".
- **A `git stash pop` was run without checking which stash was on top**, which half-applied a
  pre-existing stash (`b492640`) into the working tree and left conflicts across 11 untouched
  files. The pop failed, so that stash was never dropped; all 11 files were restored to HEAD
  and verified byte-identical, and the stash is still on the list. No work was lost, but the
  tree was briefly inconsistent, and the cause was bundling `stash push`, a long test run and
  `stash pop` into one command that a timeout could cut in half.
- **This report was generated twice.** The first fill sliced the document to replace sections
  5 and 6 and so deleted the hash-checked `dod-metrics` block that the generator appends after
  section 6. Regenerated and re-filled by exact marker replacement; the block below is the
  generator's own.

## 6. Next actions

- **Give the stale-quarter defect its own cluster and a check that can see it.** The cheapest
  honest detector is a history-substrate rule on `revenue_q` / `netIncome_q` equal to the
  previous `as_of`'s value for the same ticker: 60 rows, 17 tickers, and currently zero checks
  can fire on it. Better still, carry the quarter's own `period_end` beside the value so the
  column stops being silently unaligned with `fiscal_end`.
- **Fix `.gitignore:77`.** Narrow `target/` to the Rust build directory it was meant for, or
  add `!src/data_aggregate/utils/target/`, and commit the package. Until then CI and every
  fresh clone are missing a source module. Repo-root config change -- proposed, not applied.
- **Decide whether the 10 failing `data_aggregate` tests are accepted drift or a broken gate.**
  A cube-fingerprint guard that has been failing long enough for its own baseline to be
  documented as stale is not protecting anything.
- **Teach `_resolve_runs` about a cluster that closed completely** -- prefer the newest run at
  the cluster's `scope_hash` over the newest run that saw the cluster, so a 100% close records
  without `--after` and `fix show` stops reporting closed findings as pending.
- **Read the 7 rows the unconditional union would also drop** (DTE, EQIX, META, VLO). They
  share the prose-aside shape but were left alone because nobody has checked them at filing
  level; that reading is small, bounded work and would either widen the guard or justify the
  current scoping permanently.
- **Re-run this profile with `--update-baseline`** once someone is happy with the shape, so
  D2/D3/D5 stop coming back N/A for the next person.

```json dod-metrics
{
  "baseline_head_sha": "6d119bf4bf9e63d57014710b92935ccdbb6a0c4a",
  "content_hash": "sha256:48b5483596081fddbf4d6401a69a100207f3d06b7f5446a29696caec828eca33",
  "gates": {
    "D1": "PASS",
    "D2": "N/A",
    "D3": "N/A",
    "D4": "PASS",
    "D5": "N/A"
  },
  "generator": "scripts/dod/data_profile.py@1",
  "metrics": {
    "parts_behind": null,
    "stale_sources": null,
    "tables": {
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
            "nunique": 3281
          },
          "adjustment": {
            "dtype": "str",
            "null_rate": 0.9931704879036339,
            "nulls": 313969,
            "nunique": 380
          },
          "cik": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 57
          },
          "dc_code": {
            "dtype": "str",
            "null_rate": 0.7948109626480413,
            "nulls": 251262,
            "nunique": 7
          },
          "decimals": {
            "dtype": "str",
            "null_rate": 0.2051890373519587,
            "nulls": 64866,
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
            "nunique": 1301
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
            "mean": 2018.090254580423,
            "min": 2006.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 22,
            "p01": 2010.0,
            "p25": 2014.0,
            "p50": 2018.0,
            "p75": 2022.0,
            "p99": 2026.0,
            "std": 4.446025970022345
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
            "mean": 194.28563768785094,
            "min": 81.0,
            "null_rate": 0.5752290211559874,
            "nulls": 181846,
            "nunique": 46,
            "p01": 83.0,
            "p25": 90.0,
            "p50": 180.0,
            "p75": 273.0,
            "p99": 365.0,
            "std": 112.7402050764663
          },
          "period_end": {
            "dtype": "object",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1758
          },
          "period_of_report": {
            "dtype": "object",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 600
          },
          "period_start": {
            "dtype": "object",
            "null_rate": 0.5752290211559874,
            "nulls": 181846,
            "nunique": 622
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
            "null_rate": 0.4154646219252961,
            "nulls": 131340,
            "nunique": 1849
          },
          "roll_up_children": {
            "dtype": "str",
            "null_rate": 0.9699488814657354,
            "nulls": 306628,
            "nunique": 57
          },
          "root_anchor": {
            "dtype": "str",
            "null_rate": 0.9767404342544792,
            "nulls": 308775,
            "nunique": 9
          },
          "source_concept": {
            "dtype": "str",
            "null_rate": 0.22222643992306915,
            "nulls": 70252,
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
            "null_rate": 0.2039205638222492,
            "nulls": 64465,
            "nunique": 190
          },
          "value": {
            "dtype": "float64",
            "mad_center": 3268000000.0,
            "mad_outliers": 56780,
            "mad_scale": 3091000000.0,
            "max": 4819056000000000.0,
            "mean": 52456486257.73154,
            "min": -267506000000.0,
            "null_rate": 0.2051890373519587,
            "nulls": 64866,
            "nunique": 67982,
            "p01": -688000000.0,
            "p25": 656000000.0,
            "p50": 3268000000.0,
            "p75": 16181000000.0,
            "p99": 437240659999.99554,
            "std": 9615395625269.033
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
        "pk_checked_rows": 316128,
        "pk_complete": true,
        "pk_duplicate_rows": 0,
        "pk_missing_cols": [],
        "rows": 316128,
        "sample_date_max": "2026-08-10",
        "sample_date_min": "2009-07-31",
        "sampled_rows": 316128,
        "scope": {
          "limit": null,
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
            "mad_center": 2436000000.0,
            "mad_outliers": 310,
            "mad_scale": 2088000000.0,
            "max": 77088000000.0,
            "mean": 8218451485.398378,
            "min": 126022.0,
            "null_rate": 0.3944137507673419,
            "nulls": 1285,
            "nunique": 1871,
            "p01": 40538879.99999999,
            "p25": 874700000.0,
            "p50": 2436000000.0,
            "p75": 8574000000.0,
            "p99": 62567080000.0,
            "std": 13623803294.262405
          },
          "accountsReceivable": {
            "dtype": "float64",
            "mad_center": 3980000000.0,
            "mad_outliers": 203,
            "mad_scale": 2774000000.0,
            "max": 80876000000.0,
            "mean": 7437486074.641622,
            "min": 10000000.0,
            "null_rate": 0.37906691221608346,
            "nulls": 1235,
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
            "mad_center": 11925000000.0,
            "mad_outliers": 110,
            "mad_scale": 8032000000.0,
            "max": 278510000000.0,
            "mean": 19111391025.751072,
            "min": -19086000000.0,
            "null_rate": 0.49938612645794966,
            "nulls": 1627,
            "nunique": 1595,
            "p01": 50342500.0,
            "p25": 5417000000.0,
            "p50": 11925000000.0,
            "p75": 21948500000.0,
            "p99": 116600600000.00032,
            "std": 27512821724.06878
          },
          "amended_fields": {
            "dtype": "str",
            "null_rate": 0.9935543278084714,
            "nulls": 3237,
            "nunique": 14
          },
          "amended_fiscal_end": {
            "dtype": "object",
            "null_rate": 0.9935543278084714,
            "nulls": 3237,
            "nunique": 21
          },
          "as_of": {
            "dtype": "object",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1297
          },
          "basicShares": {
            "dtype": "float64",
            "mad_center": 649912602.7397261,
            "mad_outliers": 691,
            "mad_scale": 344750958.90410966,
            "max": 1649891000000.0,
            "mean": 2192594160.9738536,
            "min": 711.012602739726,
            "null_rate": 0.072744014732965,
            "nulls": 237,
            "nunique": 2805,
            "p01": 1643053.1095890412,
            "p25": 394762803.23450136,
            "p50": 649912602.7397261,
            "p75": 1963292328.7671232,
            "p99": 12196236712.328777,
            "std": 30094815232.16779
          },
          "capex": {
            "dtype": "float64",
            "mad_center": 2716000000.0,
            "mad_outliers": 273,
            "mad_scale": 1892000000.0,
            "max": 132402000000.0,
            "mean": 5759620198.681318,
            "min": 5001000.0,
            "null_rate": 0.30171884591774095,
            "nulls": 983,
            "nunique": 2066,
            "p01": 38988880.0,
            "p25": 1016000000.0,
            "p50": 2716000000.0,
            "p75": 5956500000.0,
            "p99": 45457979999.99911,
            "std": 9779083367.975649
          },
          "cash": {
            "dtype": "float64",
            "mad_center": 4292000000.0,
            "mad_outliers": 534,
            "mad_scale": 3823000000.0,
            "max": 759869000000.0,
            "mean": 33331752295.085392,
            "min": 3.4,
            "null_rate": 0.05432780847145488,
            "nulls": 177,
            "nunique": 2934,
            "p01": 46000000.0,
            "p25": 1020000000.0,
            "p50": 4292000000.0,
            "p75": 14582000000.0,
            "p99": 528250999999.9994,
            "std": 94067125675.31577
          },
          "costOfRevenue": {
            "dtype": "float64",
            "mad_center": 28321500000.0,
            "mad_outliers": 139,
            "mad_scale": 25657612500.0,
            "max": 544150000000.0,
            "mean": 68250815525.48331,
            "min": 40600000.0,
            "null_rate": 0.6507059545733579,
            "nulls": 2120,
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
            "mad_center": 16258000000.0,
            "mad_outliers": 231,
            "mad_scale": 12966000000.0,
            "max": 343524000000.0,
            "mean": 33948867273.39683,
            "min": 515881.0,
            "null_rate": 0.34100675260896257,
            "nulls": 1111,
            "nunique": 2119,
            "p01": 737673160.0,
            "p25": 5614500000.0,
            "p50": 16258000000.0,
            "p75": 50275500000.0,
            "p99": 174089100000.0,
            "std": 40626635092.15376
          },
          "currentLiabilities": {
            "dtype": "float64",
            "mad_center": 13388000000.0,
            "mad_outliers": 237,
            "mad_scale": 10625000000.0,
            "max": 176392000000.0,
            "mean": 24796015474.03445,
            "min": 511697.0,
            "null_rate": 0.3406998158379374,
            "nulls": 1110,
            "nunique": 2117,
            "p01": 215721510.0,
            "p25": 4467784750.0,
            "p50": 13388000000.0,
            "p75": 31544250000.0,
            "p99": 126851410000.00027,
            "std": 29593806496.693394
          },
          "debtToEquity": {
            "dtype": "float64",
            "mad_center": 0.6355194413498737,
            "mad_outliers": 296,
            "mad_scale": 0.4338905795706002,
            "max": 167.48275862068965,
            "mean": 0.7543198678206274,
            "min": -560.4109589041096,
            "null_rate": 0.14057704112952732,
            "nulls": 458,
            "nunique": 2777,
            "p01": -14.084131958214734,
            "p25": 0.30064138123083034,
            "p50": 0.6355194413498737,
            "p75": 1.4025332443493026,
            "p99": 9.649414315858035,
            "std": 12.858649075051709
          },
          "depAmort": {
            "dtype": "float64",
            "mad_center": 2165500000.0,
            "mad_outliers": 304,
            "mad_scale": 1205500000.0,
            "max": 46463000000.0,
            "mean": 3623038020.042194,
            "min": 6364000.0,
            "null_rate": 0.1270718232044199,
            "nulls": 414,
            "nunique": 2432,
            "p01": 28892580.0,
            "p25": 1126250000.0,
            "p50": 2165500000.0,
            "p75": 3871376750.0,
            "p99": 22968890000.00006,
            "std": 4740999726.702334
          },
          "dilutedShares": {
            "dtype": "float64",
            "mad_center": 687643358.9041096,
            "mad_outliers": 680,
            "mad_scale": 365927057.53424656,
            "max": 24804000000.0,
            "mean": 1704557887.299979,
            "min": 713.6391780821918,
            "null_rate": 0.08532842234499693,
            "nulls": 278,
            "nunique": 2770,
            "p01": 43392789.04109589,
            "p25": 402288736.26373625,
            "p50": 687643358.9041096,
            "p75": 2087257739.7260275,
            "p99": 12271614027.397263,
            "std": 2579310007.824016
          },
          "ebitda": {
            "dtype": "float64",
            "mad_center": 8824000000.0,
            "mad_outliers": 162,
            "mad_scale": 5523500000.0,
            "max": 198685000000.0,
            "mean": 16543390805.226824,
            "min": -10521000000.0,
            "null_rate": 0.3775322283609576,
            "nulls": 1230,
            "nunique": 1961,
            "p01": -2114460000.0000002,
            "p25": 3998750000.0,
            "p50": 8824000000.0,
            "p75": 16615250000.0,
            "p99": 136625430000.00005,
            "std": 25465481775.003635
          },
          "effectiveTaxRate": {
            "dtype": "float64",
            "mad_center": 0.20443724040117517,
            "mad_outliers": 126,
            "mad_scale": 0.07172470824866645,
            "max": 5.123076923076923,
            "mean": 0.19876505480390835,
            "min": -9.106666666666667,
            "null_rate": 0.15377532228360957,
            "nulls": 501,
            "nunique": 2741,
            "p01": -0.47770122149489,
            "p25": 0.13357204016501906,
            "p50": 0.20443724040117517,
            "p75": 0.2776686313032089,
            "p99": 0.9447177391259292,
            "std": 0.34787882976844303
          },
          "epsDiluted": {
            "dtype": "float64",
            "mad_center": 5.0909917910278235,
            "mad_outliers": 258,
            "mad_scale": 2.3961186185801298,
            "max": 12312945.070664234,
            "mean": 32168.02129612416,
            "min": -62.24338624338624,
            "null_rate": 0.10374462860650706,
            "nulls": 338,
            "nunique": 2899,
            "p01": -8.254045158745948,
            "p25": 3.000705815923207,
            "p50": 5.0909917910278235,
            "p75": 8.104045946132153,
            "p99": 50.428811331733236,
            "std": 613960.3613800086
          },
          "financeLeaseLiability": {
            "dtype": "float64",
            "mad_center": 605500000.0,
            "mad_outliers": 68,
            "mad_scale": 580200000.0,
            "max": 66594000000.0,
            "mean": 2577488414.847162,
            "min": 4000000.0,
            "null_rate": 0.8594229588704727,
            "nulls": 2800,
            "nunique": 360,
            "p01": 5000000.0,
            "p25": 38250000.0,
            "p50": 605500000.0,
            "p75": 2128448500.0,
            "p99": 42205940000.000046,
            "std": 7287306257.944172
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
            "mean": 2.4998421218819074,
            "min": 1.0,
            "null_rate": 0.027931246163290364,
            "nulls": 91,
            "nunique": 4,
            "p01": 1.0,
            "p25": 1.0,
            "p50": 2.0,
            "p75": 4.0,
            "p99": 4.0,
            "std": 1.126792797802272
          },
          "freeCashflow": {
            "dtype": "float64",
            "mad_center": 3527000000.0,
            "mad_outliers": 275,
            "mad_scale": 3613793000.0,
            "max": 136683000000.0,
            "mean": 9185198097.74436,
            "min": -90387000000.0,
            "null_rate": 0.3060159607120933,
            "nulls": 997,
            "nunique": 2164,
            "p01": -15683199999.999998,
            "p25": 676700000.0,
            "p50": 3527000000.0,
            "p75": 12411000000.0,
            "p99": 91465000000.00023,
            "std": 17824924730.907692
          },
          "goodwill": {
            "dtype": "float64",
            "mad_center": 9908000000.0,
            "mad_outliers": 228,
            "mad_scale": 8718200000.0,
            "max": 119661000000.0,
            "mean": 19936618895.746456,
            "min": 0.0,
            "null_rate": 0.2639656230816452,
            "nulls": 860,
            "nunique": 1828,
            "p01": 20098000.0,
            "p25": 3309750000.0,
            "p50": 9908000000.0,
            "p75": 26579500000.0,
            "p99": 100443890000.00026,
            "std": 23230659239.955715
          },
          "grossMargins": {
            "dtype": "float64",
            "mad_center": 0.3965847174827021,
            "mad_outliers": 0,
            "mad_scale": 0.2233034366969434,
            "max": 0.762218032364391,
            "mean": 0.3669998419849311,
            "min": -0.0977509542969153,
            "null_rate": 0.7332719459791283,
            "nulls": 2389,
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
            "null_rate": 0.7286678944137508,
            "nulls": 2374,
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
            "mad_center": 1302000000.0,
            "mad_outliers": 334,
            "mad_scale": 1167271500.0,
            "max": 55064000000.0,
            "mean": 2812403830.6048493,
            "min": -23516000000.0,
            "null_rate": 0.06322897483118478,
            "nulls": 206,
            "nunique": 2587,
            "p01": -2154462559.999999,
            "p25": 297250000.0,
            "p50": 1302000000.0,
            "p75": 2994000000.0,
            "p99": 23967949999.999943,
            "std": 4844024641.99814
          },
          "intangiblesExGoodwill": {
            "dtype": "float64",
            "mad_center": 3075000000.0,
            "mad_outliers": 184,
            "mad_scale": 2690400000.0,
            "max": 54942000000.0,
            "mean": 6804695933.118971,
            "min": 34700000.0,
            "null_rate": 0.5227133210558625,
            "nulls": 1703,
            "nunique": 1453,
            "p01": 52000000.0,
            "p25": 895500000.0,
            "p50": 3075000000.0,
            "p75": 7464000000.0,
            "p99": 48022180000.000015,
            "std": 9652653266.868631
          },
          "interestExpense": {
            "dtype": "float64",
            "mad_center": 597000000.0,
            "mad_outliers": 441,
            "mad_scale": 443900000.0,
            "max": 101350000000.0,
            "mean": 3734371189.6617255,
            "min": 11000.0,
            "null_rate": 0.19244935543278086,
            "nulls": 627,
            "nunique": 1873,
            "p01": 11000.0,
            "p25": 275000000.0,
            "p50": 597000000.0,
            "p75": 1847000000.0,
            "p99": 81360199999.99998,
            "std": 11972424007.739656
          },
          "inventory": {
            "dtype": "float64",
            "mad_center": 3298000000.0,
            "mad_outliers": 121,
            "mad_scale": 2810734500.0,
            "max": 65354000000.0,
            "mean": 7047186717.731421,
            "min": 39100000.0,
            "null_rate": 0.529158993247391,
            "nulls": 1724,
            "nunique": 1464,
            "p01": 49765000.0,
            "p25": 1138250000.0,
            "p50": 3298000000.0,
            "p75": 8694500000.0,
            "p99": 56163080000.00006,
            "std": 10301094507.346048
          },
          "is_amendment": {
            "dtype": "bool",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2
          },
          "longTermDebt": {
            "dtype": "float64",
            "mad_center": 18880500000.0,
            "mad_outliers": 270,
            "mad_scale": 12408500000.0,
            "max": 398965000000.0,
            "mean": 41154147279.96936,
            "min": 0.0,
            "null_rate": 0.19858809085328422,
            "nulls": 647,
            "nunique": 2517,
            "p01": 23647800.000000007,
            "p25": 8222500000.0,
            "p50": 18880500000.0,
            "p75": 38552500000.0,
            "p99": 287231900000.00006,
            "std": 64764421932.99553
          },
          "longTermDebtCurrentOnly": {
            "dtype": "float64",
            "mad_center": 1484000000.0,
            "mad_outliers": 29,
            "mad_scale": 1382600000.0,
            "max": 14009000000.0,
            "mean": 2238098302.5027204,
            "min": -18200000.0,
            "null_rate": 0.7179251074278699,
            "nulls": 2339,
            "nunique": 704,
            "p01": 0.0,
            "p25": 260500000.0,
            "p50": 1484000000.0,
            "p75": 3282000000.0,
            "p99": 10946439999.999998,
            "std": 2461643262.291334
          },
          "minorityInterest": {
            "dtype": "float64",
            "mad_center": 435000000.0,
            "mad_outliers": 346,
            "mad_scale": 428000000.0,
            "max": 11871000000.0,
            "mean": 1388375649.5248742,
            "min": -187000000.0,
            "null_rate": 0.450890116635973,
            "nulls": 1469,
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
            "mad_center": 4482000000.0,
            "mad_outliers": 372,
            "mad_scale": 3255518000.0,
            "max": 244205000000.0,
            "mean": 10420476966.068306,
            "min": -23528000000.0,
            "null_rate": 0.05187231430325353,
            "nulls": 169,
            "nunique": 2906,
            "p01": -5063720000.0,
            "p25": 1779000000.0,
            "p50": 4482000000.0,
            "p75": 11037000000.0,
            "p99": 97164639999.99998,
            "std": 18298712960.408035
          },
          "netIncome_q": {
            "dtype": "float64",
            "mad_center": 1125000000.0,
            "mad_outliers": 392,
            "mad_scale": 878000000.0,
            "max": 112193000000.0,
            "mean": 2682153163.9926176,
            "min": -49697000000.0,
            "null_rate": 0.002148557397176182,
            "nulls": 7,
            "nunique": 2741,
            "p01": -2768500000.0,
            "p25": 402729000.0,
            "p50": 1125000000.0,
            "p75": 2870000000.0,
            "p99": 27606500000.0,
            "std": 5703756874.218373
          },
          "netInterestIncome": {
            "dtype": "float64",
            "mad_center": 6990868000.0,
            "mad_outliers": 143,
            "mad_scale": 7539868000.0,
            "max": 99838000000.0,
            "mean": 18914414818.80109,
            "min": -2609000000.0,
            "null_rate": 0.7747084100675261,
            "nulls": 2524,
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
            "null_rate": 0.929097605893186,
            "nulls": 3027,
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
            "null_rate": 0.8591160220994475,
            "nulls": 2799,
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
            "mad_scale": 4834702000.0,
            "max": 185675000000.0,
            "mean": 13533229405.858585,
            "min": -162534000000.0,
            "null_rate": 0.058011049723756904,
            "nulls": 189,
            "nunique": 2940,
            "p01": -34235440000.0,
            "p25": 3100000000.0,
            "p50": 6815000000.0,
            "p75": 15126000000.0,
            "p99": 112497000000.00014,
            "std": 24645415251.746616
          },
          "operatingIncome": {
            "dtype": "float64",
            "mad_center": 5214000000.0,
            "mad_outliers": 210,
            "mad_scale": 4024000000.0,
            "max": 162285000000.0,
            "mean": 12408307508.07568,
            "min": -25494000000.0,
            "null_rate": 0.33486801718845915,
            "nulls": 1091,
            "nunique": 1874,
            "p01": -6718012140.0,
            "p25": 2337371000.0,
            "p50": 5214000000.0,
            "p75": 13435500000.0,
            "p99": 119830380000.00017,
            "std": 21621467980.436344
          },
          "operatingLeaseLiability": {
            "dtype": "float64",
            "mad_center": 1469000000.0,
            "mad_outliers": 259,
            "mad_scale": 1290100000.0,
            "max": 30190000000.0,
            "mean": 4611392377.69447,
            "min": 0.0,
            "null_rate": 0.6724984653161449,
            "nulls": 2191,
            "nunique": 990,
            "p01": 23820460.0,
            "p25": 519014000.0,
            "p50": 1469000000.0,
            "p75": 7715850000.0,
            "p99": 21883419999.999996,
            "std": 5861980970.751635
          },
          "operatingMargins": {
            "dtype": "float64",
            "mad_center": 0.19847473101476037,
            "mad_outliers": 16,
            "mad_scale": 0.10882961829395416,
            "max": 0.8033951437730371,
            "mean": 0.19311467270012794,
            "min": -4.25584765058994,
            "null_rate": 0.3474524248004911,
            "nulls": 1132,
            "nunique": 2111,
            "p01": -0.15669189904211372,
            "p25": 0.08444133120261774,
            "p50": 0.19847473101476037,
            "p75": 0.2999220773793403,
            "p99": 0.5281375845142218,
            "std": 0.23983399420441354
          },
          "optionOverhang": {
            "dtype": "float64",
            "mad_center": 0.0068881465802715924,
            "mad_outliers": 308,
            "mad_scale": 0.0038384656702050712,
            "max": 0.19963031423290212,
            "mean": 0.011440742319462606,
            "min": -0.03372806321342636,
            "null_rate": 0.09023941068139964,
            "nulls": 294,
            "nunique": 2712,
            "p01": 0.0,
            "p25": 0.0038030115960814848,
            "p50": 0.0068881465802715924,
            "p75": 0.01302525292564688,
            "p99": 0.07450626504343834,
            "std": 0.015174105623014258
          },
          "ppeGross": {
            "dtype": "float64",
            "mad_center": 33295000000.0,
            "mad_outliers": 82,
            "mad_scale": 23434500000.0,
            "max": 570059000000.0,
            "mean": 49812623274.67811,
            "min": 124344000.0,
            "null_rate": 0.570902394106814,
            "nulls": 1860,
            "nunique": 1376,
            "p01": 225218860.0,
            "p25": 11722599250.0,
            "p50": 33295000000.0,
            "p75": 60867250000.0,
            "p99": 409987589999.9994,
            "std": 65556524297.6668
          },
          "ppeNet": {
            "dtype": "float64",
            "mad_center": 11232598500.0,
            "mad_outliers": 279,
            "mad_scale": 9054901500.0,
            "max": 313076000000.0,
            "mean": 26465128641.260975,
            "min": 81289000.0,
            "null_rate": 0.23081645181092694,
            "nulls": 752,
            "nunique": 2409,
            "p01": 193764150.0,
            "p25": 3607000000.0,
            "p50": 11232598500.0,
            "p75": 24973050000.0,
            "p99": 250839499999.99994,
            "std": 45561680771.32906
          },
          "premiumsEarned": {
            "dtype": "float64",
            "mad_center": 36803000000.0,
            "mad_outliers": 38,
            "mad_scale": 21344000000.0,
            "max": 353256000000.0,
            "mean": 60951905362.776024,
            "min": 662000000.0,
            "null_rate": 0.9027010435850215,
            "nulls": 2941,
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
            "null_rate": 0.141804788213628,
            "nulls": 462,
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
            "mad_center": 0.1503723404255319,
            "mad_outliers": 34,
            "mad_scale": 0.0873947855980351,
            "max": 0.8428179410653303,
            "mean": 0.15435434713500637,
            "min": -3.589868782422948,
            "null_rate": 0.06414978514426029,
            "nulls": 209,
            "nunique": 3034,
            "p01": -0.12210677043422084,
            "p25": 0.06734776027168152,
            "p50": 0.1503723404255319,
            "p75": 0.24154243119266056,
            "p99": 0.5209953905646604,
            "std": 0.18951301804648774
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
            "null_rate": 0.9315531000613874,
            "nulls": 3035,
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
            "null_rate": 0.9183548189073051,
            "nulls": 2992,
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
            "null_rate": 0.7252915899324739,
            "nulls": 2363,
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
            "null_rate": 0.7958870472682628,
            "nulls": 2593,
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
            "mad_center": 23864500000.0,
            "mad_outliers": 344,
            "mad_scale": 21456027000.0,
            "max": 798959000000.0,
            "mean": 56163799594.401596,
            "min": -34076000000.0,
            "null_rate": 0.039287906691221605,
            "nulls": 128,
            "nunique": 3105,
            "p01": -10469840000.0,
            "p25": 6659896250.0,
            "p50": 23864500000.0,
            "p75": 67015250000.0,
            "p99": 457758420000.00006,
            "std": 93138754871.91226
          },
          "returnOnEquity": {
            "dtype": "float64",
            "mad_center": 0.1385962125624327,
            "mad_outliers": 328,
            "mad_scale": 0.06875933861090588,
            "max": 43.58620689655172,
            "mean": -58.5622091223641,
            "min": -180657.64023210833,
            "null_rate": 0.05678330263965623,
            "nulls": 185,
            "nunique": 3068,
            "p01": -2.3550356906648613,
            "p25": 0.0877987156542794,
            "p50": 0.1385962125624327,
            "p75": 0.24899083440186162,
            "p99": 3.206687069479839,
            "std": 3258.9351695696723
          },
          "revenue_q": {
            "dtype": "float64",
            "mad_center": 9944450000.0,
            "mad_outliers": 335,
            "mad_scale": 8058376500.0,
            "max": 190656000000.0,
            "mean": 20041092292.10364,
            "min": -34644000000.0,
            "null_rate": 0.004910988336402701,
            "nulls": 16,
            "nunique": 3066,
            "p01": 243487000.0,
            "p25": 4181500000.0,
            "p50": 9944450000.0,
            "p75": 22289000000.0,
            "p99": 129690010000.0,
            "std": 27495346768.34543
          },
          "sellingGeneralAdmin": {
            "dtype": "float64",
            "mad_center": 6683000000.0,
            "mad_outliers": 110,
            "mad_scale": 5188000000.0,
            "max": 150972000000.0,
            "mean": 14296045823.608316,
            "min": 55180000.0,
            "null_rate": 0.5423572744014733,
            "nulls": 1767,
            "nunique": 1435,
            "p01": 93747900.0,
            "p25": 2432650000.0,
            "p50": 6683000000.0,
            "p75": 19099000000.0,
            "p99": 121222499999.99956,
            "std": 22468580217.30003
          },
          "sharesOutstanding": {
            "dtype": "float64",
            "mad_center": 687839228.0,
            "mad_outliers": 674,
            "mad_scale": 359356875.5,
            "max": 4819056000000000.0,
            "mean": 1580889610504.9116,
            "min": 0.0,
            "null_rate": 0.06322897483118478,
            "nulls": 206,
            "nunique": 3022,
            "p01": 46967078.96,
            "p25": 407742604.0,
            "p50": 687839228.0,
            "p75": 1996476965.75,
            "p99": 14978087029.99998,
            "std": 87230736139056.23
          },
          "shortTermBorrowingsOnly": {
            "dtype": "float64",
            "mad_center": 2037000000.0,
            "mad_outliers": 264,
            "mad_scale": 1871279000.0,
            "max": 252927000000.0,
            "mean": 14366870494.51554,
            "min": 0.0,
            "null_rate": 0.6642111724984653,
            "nulls": 2164,
            "nunique": 943,
            "p01": 0.0,
            "p25": 461000000.0,
            "p50": 2037000000.0,
            "p75": 9463500000.0,
            "p99": 120513499999.99977,
            "std": 29865562218.489956
          },
          "shortTermDebt": {
            "dtype": "float64",
            "mad_center": 2117000000.0,
            "mad_outliers": 341,
            "mad_scale": 2000000000.0,
            "max": 252927000000.0,
            "mean": 9118879057.21949,
            "min": -634000000.0,
            "null_rate": 0.31338244321669734,
            "nulls": 1021,
            "nunique": 1831,
            "p01": 0.0,
            "p25": 509000000.0,
            "p50": 2117000000.0,
            "p75": 6816000000.0,
            "p99": 104049599999.99985,
            "std": 21759001217.418728
          },
          "shortTermInvestments": {
            "dtype": "float64",
            "mad_center": 1874950000.0,
            "mad_outliers": 229,
            "mad_scale": 1762900000.0,
            "max": 186563000000.0,
            "mean": 11494876346.456694,
            "min": 0.0,
            "null_rate": 0.6881522406384285,
            "nulls": 2242,
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
            "mad_center": 297500000.0,
            "mad_outliers": 449,
            "mad_scale": 225128000.0,
            "max": 28147000000.0,
            "mean": 1452837498.9929507,
            "min": 2049000.0,
            "null_rate": 0.39042357274401474,
            "nulls": 1272,
            "nunique": 1470,
            "p01": 2232000.0,
            "p25": 121075000.0,
            "p50": 297500000.0,
            "p75": 1100750000.0,
            "p99": 17601450000.000065,
            "std": 3227630476.9832773
          },
          "stockholdersEquity": {
            "dtype": "float64",
            "mad_center": 24559000000.0,
            "mad_outliers": 464,
            "mad_scale": 19797700000.0,
            "max": 750177000000.0,
            "mean": 60553297844.804344,
            "min": -23562000000.0,
            "null_rate": 0.011663597298956415,
            "nulls": 38,
            "nunique": 3185,
            "p01": -7454410000.0,
            "p25": 10773025000.0,
            "p50": 24559000000.0,
            "p75": 68163000000.0,
            "p99": 390113479999.99976,
            "std": 88369789350.52191
          },
          "ticker": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 54
          },
          "totalAssets": {
            "dtype": "float64",
            "mad_center": 95131000000.0,
            "mad_outliers": 519,
            "mad_scale": 63755252000.0,
            "max": 5015069000000.0,
            "mean": 338216091951.6643,
            "min": 4776348.0,
            "null_rate": 0.008901166359729895,
            "nulls": 29,
            "nunique": 3214,
            "p01": 1687824120.0,
            "p25": 39630000000.0,
            "p50": 95131000000.0,
            "p75": 230238000000.0,
            "p99": 3254658399999.9976,
            "std": 670333441900.6166
          },
          "totalDebt": {
            "dtype": "float64",
            "mad_center": 20936000000.0,
            "mad_outliers": 276,
            "mad_scale": 14789000000.0,
            "max": 501315000000.0,
            "mean": 47554444847.16779,
            "min": 0.0,
            "null_rate": 0.13842848373235114,
            "nulls": 451,
            "nunique": 2712,
            "p01": 34774320.0,
            "p25": 8524000000.0,
            "p50": 20936000000.0,
            "p75": 48973000000.0,
            "p99": 346814800000.0002,
            "std": 75764607696.90126
          },
          "totalLiabilities": {
            "dtype": "float64",
            "mad_center": 61442500000.0,
            "mad_outliers": 501,
            "mad_scale": 44885500000.0,
            "max": 4640471000000.0,
            "mean": 278400412447.6618,
            "min": -8237223652.0,
            "null_rate": 0.011663597298956415,
            "nulls": 38,
            "nunique": 3197,
            "p01": 327608330.0,
            "p25": 25266000000.0,
            "p50": 61442500000.0,
            "p75": 159379250000.0,
            "p99": 2970533259999.9995,
            "std": 612192509857.1149
          },
          "totalRevenue": {
            "dtype": "float64",
            "mad_center": 39531000000.0,
            "mad_outliers": 323,
            "mad_scale": 32053000000.0,
            "max": 725305000000.0,
            "mean": 79785376056.3654,
            "min": -194700000.0,
            "null_rate": 0.052486187845303865,
            "nulls": 171,
            "nunique": 3028,
            "p01": 1547474980.0,
            "p25": 17273199500.0,
            "p50": 39531000000.0,
            "p75": 89075500000.0,
            "p99": 510404099999.99976,
            "std": 107817991067.95793
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
        "pk_checked_rows": 3258,
        "pk_complete": true,
        "pk_duplicate_rows": 0,
        "pk_missing_cols": [],
        "rows": 3258,
        "sample_date_max": "2026-08-10",
        "sample_date_min": "2009-07-31",
        "sampled_rows": 3258,
        "scope": {
          "limit": null,
          "since": null,
          "tickers": null
        },
        "table": "fundamentals_history"
      }
    }
  },
  "scope": {
    "limit": null,
    "since": null,
    "tables": [
      "fundamentals_facts",
      "fundamentals_history"
    ],
    "tickers": [],
    "unknown_tables": []
  },
  "session_id": "7aff37c0-3151-4721-9e1d-090217de16ad",
  "type": "DATA"
}
```

