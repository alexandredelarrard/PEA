---
type: DATA
session_id: 4d204d4b-c056-4b30-a83d-5ca4fd25bf76
generated_at: 2026-08-26T03:19:49+00:00
baseline: {head_sha: 0cc2b755cdeca3de4405d8be38584e914f8c77de}
generator: scripts/dod/data_profile.py@1
---

## 1. Scope

**SAMPLE SCOPE** — a metric without its scope is not a measurement:

- tables: fundamentals_facts, fundamentals_history, fundamentals_reason_codes
- tickers: ORCL
- since: **no lower bound**
- row limit per table: **none**
- full-scope tables (eligible to set the baseline): none

**What was asked:** settle validator cluster `876ab8a57bd8` — ORCL `grossProfit`, reported as
"value is almost always the same, and only available since fiscal_end = 2017-05-31". Two checks
agreed (`series_shape` tier-2 high `late_start`; `coverage_field` tier-1 medium MIXED, 8 of 60
periods). Both were right and both under-reported: the periods that *did* resolve were also wrong.

The profile above is **ORCL only**, which is the cluster's scope — it is not a universe run and
must not be read as one. The roster-wide history rebuild that commit `816c4c5` and `8c6fab1`
imply was still in flight when this was written (4 of 49 tickers), so no full-scope baseline was
recorded and D2/D3/D5 are N/A by construction rather than by omission.

## 2. Gates

| Gate | Check | Verdict | Detail |
|---|---|---|---|
| D1 | declared PK unique over the rows profiled | **PASS** | unique across 3 table(s): fundamentals_facts, fundamentals_history, fundamentals_reason_codes |
| D2 | row count not decreased | **N/A** | no full-scope baseline to compare against (scoped run: fundamentals_facts, fundamentals_history, fundamentals_reason_codes) — this run records one |
| D3 | no column lost | **N/A** | no baseline columns recorded yet |
| D4 | date range covers the expected window | **N/A** | no --expect-through given |
| D5 | per-field null rate not worse | **N/A** | no full-scope baseline null rates to compare against |

**All gates pass** (N/A gates are stated above, not skipped).

## 3. Metrics

_Observed values only — no verdicts. `rows`, `date_min` and `date_max` are **table-wide** (server-side); every other number is over the **sample** described in §1. Do not compare across the two._

**Tables**

| table | exists | rows | sampled | cols | pk | pk_absent_cols | pk_dupes | date_min | date_max | sample_date_min | sample_date_max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fundamentals_facts | yes | 316,245 | 5,659 | 26 | ticker,accession_number,field,duration_type,period_end | — | 0 | 2009-07-31 | 2026-08-10 | 2011-09-23 | 2026-06-22 |
| fundamentals_history | yes | 3,258 | 60 | 69 | ticker,as_of | — | 0 | 2009-07-31 | 2026-08-10 | 2011-09-23 | 2026-06-22 |
| fundamentals_reason_codes | yes | 75,829 | 1,339 | 6 | ticker,as_of,field,dc_code | — | 0 | 2009-07-31 | 2026-08-10 | 2011-09-23 | 2026-06-22 |

**Fields** (worst null rate first, top 60)

| table | field | dtype | null_% | nunique | mean | std | min | p01 | p50 | p99 | max | mad_outliers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fundamentals_history | amended_fields | object | 100 | 0 | — | — | — | — | — | — | — | — |
| fundamentals_history | amended_fiscal_end | object | 100 | 0 | — | — | — | — | — | — | — | — |
| fundamentals_history | longTermDebtCurrentOnly | object | 100 | 0 | — | — | — | — | — | — | — | — |
| fundamentals_history | netInterestIncome | object | 100 | 0 | — | — | — | — | — | — | — | — |
| fundamentals_history | netInvestmentIncome | object | 100 | 0 | — | — | — | — | — | — | — | — |
| fundamentals_history | noninterestIncome | object | 100 | 0 | — | — | — | — | — | — | — | — |
| fundamentals_history | premiumsEarned | object | 100 | 0 | — | — | — | — | — | — | — | — |
| fundamentals_history | realizedInvestmentGains | object | 100 | 0 | — | — | — | — | — | — | — | — |
| fundamentals_history | rentalIncome | object | 100 | 0 | — | — | — | — | — | — | — | — |
| fundamentals_history | restrictedCash | object | 100 | 0 | — | — | — | — | — | — | — | — |
| fundamentals_history | shortTermBorrowingsOnly | object | 100 | 0 | — | — | — | — | — | — | — | — |
| fundamentals_reason_codes | combined_into | object | 100 | 0 | — | — | — | — | — | — | — | — |
| fundamentals_reason_codes | rejected_value | object | 100 | 0 | — | — | — | — | — | — | — | — |
| fundamentals_facts | adjustment | str | 99.58 | 6 | — | — | — | — | — | — | — | — |
| fundamentals_history | longTermDebt | float64 | 98.33 | 1 | 0 | NaN | 0 | 0 | 0 | 0 | 0 | 0 |
| fundamentals_facts | root_anchor | str | 90.86 | 5 | — | — | — | — | — | — | — | — |
| fundamentals_history | financeLeaseLiability | float64 | 90 | 6 | 4.463e+09 | 2.41549e+09 | 9e+08 | 1.0017e+09 | 4.4965e+09 | 7.62845e+09 | 7.701e+09 | 0 |
| fundamentals_facts | roll_up_children | str | 87.42 | 10 | — | — | — | — | — | — | — | — |
| fundamentals_facts | dc_code | str | 79.66 | 5 | — | — | — | — | — | — | — | — |
| fundamentals_history | operatingLeaseLiability | float64 | 76.67 | 14 | 9.91493e+09 | 8.97111e+09 | 1.904e+09 | 1.91557e+09 | 6.197e+09 | 2.90356e+10 | 3.019e+10 | 1 |
| fundamentals_history | accumulatedDepreciation | float64 | 75 | 15 | 8.24e+09 | 5.71415e+09 | 2.398e+09 | 2.4407e+09 | 6.823e+09 | 2.17613e+10 | 2.2694e+10 | 0 |
| fundamentals_history | ppeGross | float64 | 75 | 15 | 2.41985e+10 | 3.08282e+10 | 5.419e+09 | 5.46618e+09 | 1.3075e+10 | 1.13817e+11 | 1.22651e+11 | 2 |
| fundamentals_history | effectiveTaxRate | float64 | 68.33 | 19 | 0.288231 | 0.185502 | 0.16766 | 0.16842 | 0.222164 | 0.707324 | 0.708223 | 3 |
| fundamentals_history | pretaxIncome | float64 | 68.33 | 19 | 1.24636e+10 | 9.53763e+08 | 1.1407e+10 | 1.14133e+10 | 1.2426e+10 | 1.39199e+10 | 1.3941e+10 | 0 |
| fundamentals_history | intangiblesExGoodwill | float64 | 65 | 21 | 6.50981e+09 | 8.47259e+08 | 4.943e+09 | 4.9726e+09 | 6.64e+09 | 7.8516e+09 | 7.899e+09 | 0 |
| fundamentals_facts | period_days | float64 | 52.96 | 8 | 200.714 | 112.824 | 89 | 89 | 182 | 365 | 365 | 0 |
| fundamentals_facts | period_start | object | 52.96 | 49 | — | — | — | — | — | — | — | — |
| fundamentals_history | shortTermInvestments | float64 | 48.33 | 30 | 5.87826e+09 | 9.14456e+09 | 2.07e+08 | 2.334e+08 | 6.77e+08 | 3.45899e+10 | 3.8567e+10 | 12 |
| fundamentals_history | inventory | float64 | 41.67 | 32 | 2.63714e+08 | 8.27862e+07 | 1.42e+08 | 1.4404e+08 | 2.43e+08 | 4.756e+08 | 4.96e+08 | 0 |
| fundamentals_history | debtToEquity | float64 | 38.33 | 37 | 0.346415 | 1.05429 | -1.28415 | -0.821859 | 0.0443872 | 4.372 | 5.72622 | 11 |
| fundamentals_history | shortTermDebt | float64 | 38.33 | 28 | 3.46254e+09 | 2.54074e+09 | 0 | 3.5964e+08 | 2.499e+09 | 1.03141e+10 | 1.0605e+10 | 2 |
| fundamentals_history | totalDebt | float64 | 38.33 | 28 | 5.49516e+09 | 8.22486e+09 | 0 | 3.5964e+08 | 2.95e+09 | 3.73734e+10 | 4.509e+10 | 4 |
| fundamentals_facts | role_uri | str | 34.88 | 160 | — | — | — | — | — | — | — | — |
| fundamentals_facts | source_concept | str | 20.45 | 46 | — | — | — | — | — | — | — | — |
| fundamentals_facts | decimals | str | 20.34 | 4 | — | — | — | — | — | — | — | — |
| fundamentals_facts | value | float64 | 20.34 | 2,098 | 1.08118e+12 | 7.17743e+13 | -3.4076e+10 | -1.25612e+10 | 4.278e+09 | 1.3018e+11 | 4.81906e+15 | 707 |
| fundamentals_facts | unit | str | 19.28 | 6 | — | — | — | — | — | — | — | — |
| fundamentals_history | epsDiluted | float64 | 16.67 | 50 | 3.10807 | 1.06304 | 0.890091 | 1.45989 | 3.0545 | 5.79663 | 5.93995 | 0 |
| fundamentals_history | netIncome | float64 | 16.67 | 50 | 1.06311e+10 | 2.3403e+09 | 3.769e+09 | 4.85876e+09 | 1.0672e+10 | 1.68606e+10 | 1.7309e+10 | 3 |
| fundamentals_history | profitMargins | float64 | 16.67 | 50 | 0.246184 | 0.0502036 | 0.0954854 | 0.115199 | 0.254145 | 0.345052 | 0.346033 | 0 |
| fundamentals_history | returnOnEquity | float64 | 16.67 | 50 | 0.180956 | 2.27598 | -12.5062 | -8.57137 | 0.246336 | 4.48696 | 5.57134 | 9 |
| fundamentals_history | costOfRevenue | float64 | 13.33 | 51 | 1.01747e+10 | 4.16872e+09 | 6.856e+09 | 6.8764e+09 | 8.0145e+09 | 2.20367e+10 | 2.3021e+10 | 14 |
| fundamentals_history | grossMargins | float64 | 13.33 | 52 | 0.774643 | 0.0439127 | 0.658229 | 0.664665 | 0.796796 | 0.817707 | 0.819108 | 14 |
| fundamentals_history | grossProfit | float64 | 13.33 | 52 | 3.34628e+10 | 3.95101e+09 | 2.9263e+10 | 2.92732e+10 | 3.16785e+10 | 4.3648e+10 | 4.4337e+10 | 5 |
| fundamentals_history | netIncome_q | float64 | 6.67 | 55 | 2.47546e+09 | 1.61465e+09 | -4.024e+09 | -4.024e+09 | 2.4985e+09 | 5.5223e+09 | 6.135e+09 | 5 |
| fundamentals_history | basicShares | float64 | 5 | 57 | 3.63399e+09 | 7.99522e+08 | 2.6791e+09 | 2.67924e+09 | 3.634e+09 | 4.98745e+09 | 5.015e+09 | 0 |
| fundamentals_history | capex | float64 | 5 | 56 | 6.00761e+09 | 1.09974e+10 | 5.78e+08 | 5.7912e+08 | 1.676e+09 | 5.15117e+10 | 5.5663e+10 | 13 |
| fundamentals_history | depAmort | float64 | 5 | 53 | 3.78881e+09 | 1.68401e+09 | 2.281e+09 | 2.30004e+09 | 2.931e+09 | 8.64944e+09 | 9.294e+09 | 17 |
| fundamentals_history | dilutedShares | float64 | 5 | 57 | 3.71615e+09 | 8.04666e+08 | 2.73487e+09 | 2.74213e+09 | 3.732e+09 | 5.06521e+09 | 5.095e+09 | 0 |
| fundamentals_history | ebitda | float64 | 5 | 57 | 1.80662e+10 | 3.3649e+09 | 1.3954e+10 | 1.39848e+10 | 1.7232e+10 | 2.8681e+10 | 2.9899e+10 | 5 |
| fundamentals_history | fiscal_quarter | float64 | 5 | 4 | 2.52632 | 1.13555 | 1 | 1 | 3 | 4 | 4 | 0 |
| fundamentals_history | freeCashflow | float64 | 5 | 57 | 9.53728e+09 | 8.12904e+09 | -2.4736e+10 | -2.4148e+10 | 1.2372e+10 | 1.46668e+10 | 1.4729e+10 | 9 |
| fundamentals_history | incomeTaxExpense | float64 | 5 | 57 | 2.28625e+09 | 2.12312e+09 | -1.646e+09 | -1.20976e+09 | 1.997e+09 | 9.01e+09 | 9.066e+09 | 4 |
| fundamentals_history | interestExpense | float64 | 5 | 56 | 2.19039e+09 | 1.04309e+09 | 7.62e+08 | 7.6368e+08 | 2.047e+09 | 4.34084e+09 | 4.599e+09 | 0 |
| fundamentals_history | operatingCashFlow | float64 | 5 | 57 | 1.55449e+10 | 3.59482e+09 | 9.539e+09 | 9.93996e+09 | 1.4789e+10 | 2.72377e+10 | 3.1977e+10 | 8 |
| fundamentals_history | operatingIncome | float64 | 5 | 57 | 1.42774e+10 | 1.95456e+09 | 1.0123e+10 | 1.05732e+10 | 1.3896e+10 | 2.00316e+10 | 2.0605e+10 | 2 |
| fundamentals_history | operatingMargins | float64 | 5 | 57 | 0.335508 | 0.0416072 | 0.229245 | 0.245047 | 0.34232 | 0.393723 | 0.394944 | 0 |
| fundamentals_history | optionOverhang | float64 | 5 | 57 | 0.0233518 | 0.00463439 | 0.0151931 | 0.0152215 | 0.0241092 | 0.0309512 | 0.0318519 | 0 |
| fundamentals_history | researchAndDevelopment | float64 | 5 | 55 | 6.74158e+09 | 1.66392e+09 | 4.523e+09 | 4.60756e+09 | 6.099e+09 | 1.02906e+10 | 1.0313e+10 | 4 |
| fundamentals_history | sellingGeneralAdmin | float64 | 5 | 56 | 9.35432e+09 | 6.36642e+08 | 8.133e+09 | 8.16772e+09 | 9.371e+09 | 1.03612e+10 | 1.0411e+10 | 0 |

## 4. Evidence

- baseline file: `reports/baselines/data_profile.json` (0 table(s) recorded)
- `fundamentals_facts`: 316,245 rows, 26 cols, 5,659 sampled
- `fundamentals_history`: 3,258 rows, 69 cols, 60 sampled
- `fundamentals_reason_codes`: 75,829 rows, 6 cols, 1,339 sampled

## 5. Regressions, gaps and deliberate omissions

- **The cluster is NOT settled, and the delta so far is small.** Re-validated at the original
  scope (`run 8090b885946f`, `scope_hash 4c12c4065d45`, same as baseline `10cc745356bc`):
  queue severities **2 → 1**. `series_shape` high 1 → 0; `coverage_field` medium stays at 1;
  8 new `info` rows from `adjustment_unguarded` recording the refusal. That run was taken
  BEFORE the derivation (`816c4c5`) and the config (`88552ab`) landed, so it understates the
  current state — but a fix is not settled until a re-run says so, and that re-run has not
  happened. **No `validate fix record` row has been written yet.**
- **41 of 54 tickers still carry frozen history.** The staleness cap only takes effect on a
  rebuild. 13 are done (WMT, META, GOOGL, COST, ORCL + AAPL, ADM, AFL, AMT and the rest of the
  in-flight batch); the remainder still hold whatever their frozen runs are hiding.
- **The GOOGL defect was found by accident and its class is unmeasured.** `dilutedShares` was
  frozen at 688,134,800 — a pre-split count — from 2021-07-28 to 2024-04-26, straight through
  Alphabet's 20-for-1 split of July 2022, so stored `epsDiluted` read **$107.24** for FY2023
  against an actual **$5.80** (18.5x). It surfaced only because GOOGL happened to be in the
  sample I rebuilt to test the derivation. Interior-run scanning finds more candidates
  (`netIncome` TMO 32 rows, `interestExpense` CB 37, `depAmort` BAC 27, `basicShares` SPG 28,
  `operatingCashFlow`/`capex` MAA 24 each) but none is confirmed until its ticker is rebuilt.
- **Nothing in the validator can see a frozen series.** `frozen_series` exists only in two
  docstrings as the rationale for preferring the facts grain; there is no such check. Every
  tier-2 check scores a change, and a constant series has none. This is why an 18.5x EPS error
  survived on a mega-cap.
- **ORCL `pretaxIncome` has no fix and is now visibly NULL** for 31 rows (was silently frozen at
  $12,847M since 2018). ORCL moved to `orcl:IncomeLossFromContinuingOperationsIncludingNon
  controllingInterestBeforeIncomeTaxesExtraordinaryItems` in FY2019; FY2026's real figure is
  $19,554M. There is **no per-ticker `fallback_concepts` mechanism** — `by_ticker` carries
  `leaves`/`not_leaves` only — so unlike `costOfRevenue` this cannot be closed by config. It
  needs either a new register key or `linkbase_root_discovery` extended beyond `totalRevenue`.
  `effectiveTaxRate` is null for the same 31 rows, being derived from it.
- **`configs/` was edited, which is normally forbidden.** Commit `88552ab` applied
  `fundamentals_kpis.json` and `fundamentals_exceptions.json` under explicit user authorisation
  given mid-task. Blast radius was measured *before* applying: 11 control tickers × 6 filings =
  66 filing-level resolutions, zero changed. Both files were text-spliced and re-parsed, with an
  assertion that no key outside the two intended ones moved.
- **`operatingIncome`'s `derived_fallback` is deliberately left unwired.** Measured, its formula
  lands within 1% of the filed figure in **0.5% of 550 rows** (mean abs error 29.3%, signed bias
  −18.1%). Wiring it would inject the same class of plausible-but-wrong number this cluster
  exists to remove. The catalogue still declares it, as prose.
- **CAT and COST get no derived `grossProfit`.** Their own filings contradict the identity
  (+22.5% on 24 rows, +20.3% on 6), meaning our `costOfRevenue` is short for them. That is a real
  open coverage question, not a solved one.
- **One pre-existing test failure, unrelated and untouched:**
  `test_linkbase_resolution.py::test_apa_revenue_is_a_real_number_and_comes_from_an_extension`
  calls `.values()` on `_materialise`'s return, which is annotated `tuple[dict, dict]` on HEAD.
  It fails identically without any change of mine.
- The `--expect-through`, `--parts` and `--update-baseline` flags were **not** used: a
  single-ticker run must never set a baseline, and D4 has no meaningful window for a
  publication-event grain scoped to one filer.
- 
<!-- At least one bullet. If genuinely nothing: `- None. Checked: <30+ chars>` -->

## 6. Next actions

1. **Let the roster history rebuild finish** (~04:10, background, no network), then re-validate
   at `scope_hash 4c12c4065d45` and read the real delta. Only then record
   `validate fix record 876ab8a57bd8 --layer extraction` with the commit trio and this test file.
2. **Audit what the rebuild nulls.** Every cell the staleness cap removes was previously being
   served as live. Diff `fundamentals_history` against the snapshot taken before the run and
   look for another GOOGL — a frozen value that spans a corporate action.
3. **Add a `frozen_series` check on the HISTORY grain, for duration fields only.** The docstring
   rejecting it was right about instants (a balance-sheet level is legitimately carried) and
   wrong to generalise: a duration column that repeats to the cent across quarters is a
   forward-fill, and nothing else currently looks.
4. **Decide ORCL `pretaxIncome`.** Either add a per-ticker `fallback_concepts` key to the
   register, or extend `linkbase_root_discovery`. 31 rows and `effectiveTaxRate` depend on it.
5. **Re-check the cube.** `grossMargins` changed for ORCL (0.360 → 0.658 at FY2026) and gained
   160 rows across WMT/META/GOOGL; `epsDiluted` and `optionOverhang` changed for GOOGL by 18.5x.
   Any model trained on the old values was trained on those numbers.
- 

```json dod-metrics
{
  "baseline_head_sha": "0cc2b755cdeca3de4405d8be38584e914f8c77de",
  "content_hash": "sha256:4d9d1f57c7b1d94fdca0aadac3021243e1ad0ac562b299a50a316b7df7b6bf22",
  "gates": {
    "D1": "PASS",
    "D2": "N/A",
    "D3": "N/A",
    "D4": "N/A",
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
            "nunique": 60
          },
          "adjustment": {
            "dtype": "str",
            "null_rate": 0.9957589680155504,
            "nulls": 5635,
            "nunique": 6
          },
          "cik": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          },
          "dc_code": {
            "dtype": "str",
            "null_rate": 0.7966071744124403,
            "nulls": 4508,
            "nunique": 5
          },
          "decimals": {
            "dtype": "str",
            "null_rate": 0.20339282558755964,
            "nulls": 1151,
            "nunique": 4
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
            "nunique": 60
          },
          "fiscal_period": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 7
          },
          "fiscal_year": {
            "dtype": "int64",
            "mad_center": 2018.0,
            "mad_outliers": 0,
            "mad_scale": 4.0,
            "max": 2027.0,
            "mean": 2018.3656122989928,
            "min": 2009.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 19,
            "p01": 2011.0,
            "p25": 2015.0,
            "p50": 2018.0,
            "p75": 2022.0,
            "p99": 2026.0,
            "std": 4.3675785036331485
          },
          "form": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2
          },
          "is_amendment": {
            "dtype": "bool",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          },
          "is_extension": {
            "dtype": "bool",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2
          },
          "period_days": {
            "dtype": "float64",
            "mad_center": 182.0,
            "mad_outliers": 0,
            "mad_scale": 92.0,
            "max": 365.0,
            "mean": 200.71412471825695,
            "min": 89.0,
            "null_rate": 0.5295988690581375,
            "nulls": 2997,
            "nunique": 8,
            "p01": 89.0,
            "p25": 90.0,
            "p50": 182.0,
            "p75": 273.0,
            "p99": 365.0,
            "std": 112.8239466362414
          },
          "period_end": {
            "dtype": "object",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 127
          },
          "period_of_report": {
            "dtype": "object",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 60
          },
          "period_start": {
            "dtype": "object",
            "null_rate": 0.5295988690581375,
            "nulls": 2997,
            "nunique": 49
          },
          "regime": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          },
          "resolution_method": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 7
          },
          "role_uri": {
            "dtype": "str",
            "null_rate": 0.34882488072097545,
            "nulls": 1974,
            "nunique": 160
          },
          "roll_up_children": {
            "dtype": "str",
            "null_rate": 0.8741827177946634,
            "nulls": 4947,
            "nunique": 10
          },
          "root_anchor": {
            "dtype": "str",
            "null_rate": 0.908641102668316,
            "nulls": 5142,
            "nunique": 5
          },
          "source_concept": {
            "dtype": "str",
            "null_rate": 0.20445308358367204,
            "nulls": 1157,
            "nunique": 46
          },
          "ticker": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          },
          "unit": {
            "dtype": "str",
            "null_rate": 0.19279024562643576,
            "nulls": 1091,
            "nunique": 6
          },
          "value": {
            "dtype": "float64",
            "mad_center": 4278000000.0,
            "mad_outliers": 707,
            "mad_scale": 3577000000.0,
            "max": 4819056000000000.0,
            "mean": 1081176776999.5564,
            "min": -34076000000.0,
            "null_rate": 0.20339282558755964,
            "nulls": 1151,
            "nunique": 2098,
            "p01": -12561250000.0,
            "p25": 1653000000.0,
            "p50": 4278000000.0,
            "p75": 12595000000.0,
            "p99": 130180200000.00053,
            "std": 71774296409966.02
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
        "pk_checked_rows": 5659,
        "pk_complete": true,
        "pk_duplicate_rows": 0,
        "pk_missing_cols": [],
        "rows": 316245,
        "sample_date_max": "2026-06-22",
        "sample_date_min": "2011-09-23",
        "sampled_rows": 5659,
        "scope": {
          "limit": null,
          "since": null,
          "tickers": [
            "ORCL"
          ]
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
            "mad_center": 601000000.0,
            "mad_outliers": 12,
            "mad_scale": 180500000.0,
            "max": 10977000000.0,
            "mean": 1476033333.3333333,
            "min": 361000000.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 57,
            "p01": 364540000.0,
            "p25": 478500000.0,
            "p50": 601000000.0,
            "p75": 1144000000.0,
            "p99": 10483169999.999996,
            "std": 2362912810.0762835
          },
          "accountsReceivable": {
            "dtype": "float64",
            "mad_center": 4582000000.0,
            "mad_outliers": 3,
            "mad_scale": 840000000.0,
            "max": 10719000000.0,
            "mean": 5362833333.333333,
            "min": 3407000000.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 60,
            "p01": 3432370000.0,
            "p25": 3988500000.0,
            "p50": 4582000000.0,
            "p75": 6201000000.0,
            "p99": 10521939999.999998,
            "std": 1824547124.3680027
          },
          "accumulatedDepreciation": {
            "dtype": "float64",
            "mad_center": 6823000000.0,
            "mad_outliers": 0,
            "mad_scale": 3214000000.0,
            "max": 22694000000.0,
            "mean": 8240000000.0,
            "min": 2398000000.0,
            "null_rate": 0.75,
            "nulls": 45,
            "nunique": 15,
            "p01": 2440700000.0,
            "p25": 3852500000.0,
            "p50": 6823000000.0,
            "p75": 10781500000.0,
            "p99": 21761319999.999996,
            "std": 5714152980.601262
          },
          "amended_fields": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 60,
            "nunique": 0
          },
          "amended_fiscal_end": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 60,
            "nunique": 0
          },
          "as_of": {
            "dtype": "object",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 60
          },
          "basicShares": {
            "dtype": "float64",
            "mad_center": 3634000000.0,
            "mad_outliers": 0,
            "mad_scale": 799180821.9178085,
            "max": 5015000000.0,
            "mean": 3633994060.7751126,
            "min": 2679098630.1369863,
            "null_rate": 0.05,
            "nulls": 3,
            "nunique": 57,
            "p01": 2679238246.5753427,
            "p25": 2805528767.1232877,
            "p50": 3634000000.0,
            "p75": 4326016438.356164,
            "p99": 4987449836.065574,
            "std": 799522476.0830871
          },
          "capex": {
            "dtype": "float64",
            "mad_center": 1676000000.0,
            "mad_outliers": 13,
            "mad_scale": 992000000.0,
            "max": 55663000000.0,
            "mean": 6007614035.087719,
            "min": 578000000.0,
            "null_rate": 0.05,
            "nulls": 3,
            "nunique": 56,
            "p01": 579120000.0,
            "p25": 1391000000.0,
            "p50": 1676000000.0,
            "p75": 5981000000.0,
            "p99": 51511719999.999985,
            "std": 10997432099.21945
          },
          "cash": {
            "dtype": "float64",
            "mad_center": 17590000000.0,
            "mad_outliers": 0,
            "mad_scale": 4160000000.0,
            "max": 38455000000.0,
            "mean": 18382066666.666668,
            "min": 6813000000.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 60,
            "p01": 7642540000.0,
            "p25": 13597250000.0,
            "p50": 17590000000.0,
            "p75": 21733000000.0,
            "p99": 37737559999.99999,
            "std": 6980659713.048512
          },
          "costOfRevenue": {
            "dtype": "float64",
            "mad_center": 8014500000.0,
            "mad_outliers": 14,
            "mad_scale": 595000000.0,
            "max": 23021000000.0,
            "mean": 10174653846.153847,
            "min": 6856000000.0,
            "null_rate": 0.13333333333333333,
            "nulls": 8,
            "nunique": 51,
            "p01": 6876400000.0,
            "p25": 7579500000.0,
            "p50": 8014500000.0,
            "p75": 12564250000.0,
            "p99": 22036700000.000004,
            "std": 4168723418.1043463
          },
          "currentAssets": {
            "dtype": "float64",
            "mad_center": 44328000000.0,
            "mad_outliers": 0,
            "mad_scale": 12674000000.0,
            "max": 78545000000.0,
            "mean": 44896783333.333336,
            "min": 17561000000.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 59,
            "p01": 18230650000.0,
            "p25": 31664500000.0,
            "p50": 44328000000.0,
            "p75": 56970750000.0,
            "p99": 78065330000.0,
            "std": 17010486867.245583
          },
          "currentLiabilities": {
            "dtype": "float64",
            "mad_center": 18689000000.0,
            "mad_outliers": 0,
            "mad_scale": 4567500000.0,
            "max": 41764000000.0,
            "mean": 20360733333.333332,
            "min": 11072000000.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 60,
            "p01": 11402400000.0,
            "p25": 14491750000.0,
            "p50": 18689000000.0,
            "p75": 24167500000.0,
            "p99": 41158070000.0,
            "std": 7879809338.798701
          },
          "debtToEquity": {
            "dtype": "float64",
            "mad_center": 0.04438721136767318,
            "mad_outliers": 11,
            "mad_scale": 0.024170105920608573,
            "max": 5.726221079691516,
            "mean": 0.34641474295350855,
            "min": -1.2841539528432733,
            "null_rate": 0.38333333333333336,
            "nulls": 23,
            "nunique": 37,
            "p01": -0.821858529819695,
            "p25": 0.03178283136973887,
            "p50": 0.04438721136767318,
            "p75": 0.18060317811451534,
            "p99": 4.372000887041105,
            "std": 1.0542863439874457
          },
          "depAmort": {
            "dtype": "float64",
            "mad_center": 2931000000.0,
            "mad_outliers": 17,
            "mad_scale": 122000000.0,
            "max": 9294000000.0,
            "mean": 3788807017.5438595,
            "min": 2281000000.0,
            "null_rate": 0.05,
            "nulls": 3,
            "nunique": 53,
            "p01": 2300040000.0,
            "p25": 2861000000.0,
            "p50": 2931000000.0,
            "p75": 4570000000.0,
            "p99": 8649439999.999998,
            "std": 1684012347.66635
          },
          "dilutedShares": {
            "dtype": "float64",
            "mad_center": 3732000000.0,
            "mad_outliers": 0,
            "mad_scale": 818000000.0,
            "max": 5095000000.0,
            "mean": 3716148377.0058427,
            "min": 2734871232.8767123,
            "null_rate": 0.05,
            "nulls": 3,
            "nunique": 57,
            "p01": 2742131287.671233,
            "p25": 2872545205.479452,
            "p50": 3732000000.0,
            "p75": 4421498630.136987,
            "p99": 5065208306.010929,
            "std": 804665835.7833074
          },
          "ebitda": {
            "dtype": "float64",
            "mad_center": 17232000000.0,
            "mad_outliers": 5,
            "mad_scale": 1169000000.0,
            "max": 29899000000.0,
            "mean": 18066245614.035088,
            "min": 13954000000.0,
            "null_rate": 0.05,
            "nulls": 3,
            "nunique": 57,
            "p01": 13984800000.0,
            "p25": 16386000000.0,
            "p50": 17232000000.0,
            "p75": 18588000000.0,
            "p99": 28680999999.999996,
            "std": 3364904378.1823545
          },
          "effectiveTaxRate": {
            "dtype": "float64",
            "mad_center": 0.22216395735011363,
            "mad_outliers": 3,
            "mad_scale": 0.013229572345685503,
            "max": 0.708223447614481,
            "mean": 0.28823073980698033,
            "min": 0.16766014608345228,
            "null_rate": 0.6833333333333333,
            "nulls": 41,
            "nunique": 19,
            "p01": 0.16842001873974813,
            "p25": 0.20347811487530795,
            "p50": 0.22216395735011363,
            "p75": 0.23007925674341262,
            "p99": 0.7073240523619011,
            "std": 0.18550217435477234
          },
          "epsDiluted": {
            "dtype": "float64",
            "mad_center": 3.0544973068287495,
            "mad_outliers": 0,
            "mad_scale": 0.7539310189249608,
            "max": 5.939945092656143,
            "mean": 3.1080733128586746,
            "min": 0.8900912617150474,
            "null_rate": 0.16666666666666666,
            "nulls": 10,
            "nunique": 50,
            "p01": 1.4598876351002505,
            "p25": 2.2895275700460314,
            "p50": 3.0544973068287495,
            "p75": 3.7496619019727557,
            "p99": 5.796634574468651,
            "std": 1.0630447184326521
          },
          "financeLeaseLiability": {
            "dtype": "float64",
            "mad_center": 4496500000.0,
            "mad_outliers": 0,
            "mad_scale": 1658000000.0,
            "max": 7701000000.0,
            "mean": 4463000000.0,
            "min": 900000000.0,
            "null_rate": 0.9,
            "nulls": 54,
            "nunique": 6,
            "p01": 1001700000.0,
            "p25": 3204750000.0,
            "p50": 4496500000.0,
            "p75": 5931500000.0,
            "p99": 7628450000.0,
            "std": 2415487859.625877
          },
          "fiscal_end": {
            "dtype": "object",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 60
          },
          "fiscal_quarter": {
            "dtype": "float64",
            "mad_center": 3.0,
            "mad_outliers": 0,
            "mad_scale": 1.0,
            "max": 4.0,
            "mean": 2.526315789473684,
            "min": 1.0,
            "null_rate": 0.05,
            "nulls": 3,
            "nunique": 4,
            "p01": 1.0,
            "p25": 2.0,
            "p50": 3.0,
            "p75": 4.0,
            "p99": 4.0,
            "std": 1.1355499479153377
          },
          "freeCashflow": {
            "dtype": "float64",
            "mad_center": 12372000000.0,
            "mad_outliers": 9,
            "mad_scale": 1101000000.0,
            "max": 14729000000.0,
            "mean": 9537280701.754387,
            "min": -24736000000.0,
            "null_rate": 0.05,
            "nulls": 3,
            "nunique": 57,
            "p01": -24148000000.0,
            "p25": 9542000000.0,
            "p50": 12372000000.0,
            "p75": 13164000000.0,
            "p99": 14666840000.0,
            "std": 8129039382.1744
          },
          "goodwill": {
            "dtype": "float64",
            "mad_center": 43762000000.0,
            "mad_outliers": 0,
            "mad_scale": 10027000000.0,
            "max": 62274000000.0,
            "mean": 43520250000.0,
            "min": 21831000000.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 58,
            "p01": 21927170000.0,
            "p25": 34121500000.0,
            "p50": 43762000000.0,
            "p75": 61502500000.0,
            "p99": 62266330000.0,
            "std": 13120343618.413897
          },
          "grossMargins": {
            "dtype": "float64",
            "mad_center": 0.7967960269993142,
            "mad_outliers": 14,
            "mad_scale": 0.008805083491451593,
            "max": 0.819107675259228,
            "mean": 0.7746425667174572,
            "min": 0.6582291635737403,
            "null_rate": 0.13333333333333333,
            "nulls": 8,
            "nunique": 52,
            "p01": 0.6646653488149279,
            "p25": 0.7408408059949121,
            "p50": 0.7967960269993142,
            "p75": 0.8023695095656093,
            "p99": 0.8177070301569536,
            "std": 0.043912703121231716
          },
          "grossProfit": {
            "dtype": "float64",
            "mad_center": 31678500000.0,
            "mad_outliers": 5,
            "mad_scale": 1579500000.0,
            "max": 44337000000.0,
            "mean": 33462807692.307693,
            "min": 29263000000.0,
            "null_rate": 0.13333333333333333,
            "nulls": 8,
            "nunique": 52,
            "p01": 29273200000.0,
            "p25": 30952750000.0,
            "p50": 31678500000.0,
            "p75": 35892750000.0,
            "p99": 43647990000.0,
            "std": 3951011124.1894875
          },
          "incomeTaxExpense": {
            "dtype": "float64",
            "mad_center": 1997000000.0,
            "mad_outliers": 4,
            "mad_scale": 751000000.0,
            "max": 9066000000.0,
            "mean": 2286245614.0350876,
            "min": -1646000000.0,
            "null_rate": 0.05,
            "nulls": 3,
            "nunique": 57,
            "p01": -1209760000.0,
            "p25": 1274000000.0,
            "p50": 1997000000.0,
            "p75": 2810000000.0,
            "p99": 9010000000.0,
            "std": 2123120986.7340996
          },
          "intangiblesExGoodwill": {
            "dtype": "float64",
            "mad_center": 6640000000.0,
            "mad_outliers": 0,
            "mad_scale": 527000000.0,
            "max": 7899000000.0,
            "mean": 6509809523.809524,
            "min": 4943000000.0,
            "null_rate": 0.65,
            "nulls": 39,
            "nunique": 21,
            "p01": 4972600000.0,
            "p25": 5955000000.0,
            "p50": 6640000000.0,
            "p75": 7152000000.0,
            "p99": 7851600000.0,
            "std": 847258615.7158638
          },
          "interestExpense": {
            "dtype": "float64",
            "mad_center": 2047000000.0,
            "mad_outliers": 0,
            "mad_scale": 792000000.0,
            "max": 4599000000.0,
            "mean": 2190385964.9122806,
            "min": 762000000.0,
            "null_rate": 0.05,
            "nulls": 3,
            "nunique": 56,
            "p01": 763680000.0,
            "p25": 1344000000.0,
            "p50": 2047000000.0,
            "p75": 3014000000.0,
            "p99": 4340839999.999999,
            "std": 1043085100.5274478
          },
          "inventory": {
            "dtype": "float64",
            "mad_center": 243000000.0,
            "mad_outliers": 0,
            "mad_scale": 57000000.0,
            "max": 496000000.0,
            "mean": 263714285.7142857,
            "min": 142000000.0,
            "null_rate": 0.4166666666666667,
            "nulls": 25,
            "nunique": 32,
            "p01": 144040000.0,
            "p25": 211000000.0,
            "p50": 243000000.0,
            "p75": 313000000.0,
            "p99": 475599999.9999998,
            "std": 82786249.0103887
          },
          "is_amendment": {
            "dtype": "bool",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          },
          "longTermDebt": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 0,
            "mad_scale": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "min": 0.0,
            "null_rate": 0.9833333333333333,
            "nulls": 59,
            "nunique": 1,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 0.0,
            "std": NaN
          },
          "longTermDebtCurrentOnly": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 60,
            "nunique": 0
          },
          "minorityInterest": {
            "dtype": "float64",
            "mad_center": 484000000.0,
            "mad_outliers": 0,
            "mad_scale": 51500000.0,
            "max": 737000000.0,
            "mean": 493833333.3333333,
            "min": 347000000.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 53,
            "p01": 347000000.0,
            "p25": 430750000.0,
            "p50": 484000000.0,
            "p75": 532000000.0,
            "p99": 723429999.9999999,
            "std": 87850637.23887743
          },
          "netIncome": {
            "dtype": "float64",
            "mad_center": 10672000000.0,
            "mad_outliers": 3,
            "mad_scale": 1094500000.0,
            "max": 17309000000.0,
            "mean": 10631140000.0,
            "min": 3769000000.0,
            "null_rate": 0.16666666666666666,
            "nulls": 10,
            "nunique": 50,
            "p01": 4858760000.0,
            "p25": 9345000000.0,
            "p50": 10672000000.0,
            "p75": 11187000000.0,
            "p99": 16860649999.999998,
            "std": 2340303704.4323096
          },
          "netIncome_q": {
            "dtype": "float64",
            "mad_center": 2498500000.0,
            "mad_outliers": 5,
            "mad_scale": 429500000.0,
            "max": 6135000000.0,
            "mean": 2475464285.714286,
            "min": -4024000000.0,
            "null_rate": 0.06666666666666667,
            "nulls": 4,
            "nunique": 55,
            "p01": -4024000000.0,
            "p25": 2195500000.0,
            "p50": 2498500000.0,
            "p75": 3183250000.0,
            "p99": 5522300000.000003,
            "std": 1614648860.9465044
          },
          "netInterestIncome": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 60,
            "nunique": 0
          },
          "netInvestmentIncome": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 60,
            "nunique": 0
          },
          "noninterestIncome": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 60,
            "nunique": 0
          },
          "operatingCashFlow": {
            "dtype": "float64",
            "mad_center": 14789000000.0,
            "mad_outliers": 8,
            "mad_scale": 993000000.0,
            "max": 31977000000.0,
            "mean": 15544894736.842106,
            "min": 9539000000.0,
            "null_rate": 0.05,
            "nulls": 3,
            "nunique": 57,
            "p01": 9939960000.0,
            "p25": 13796000000.0,
            "p50": 14789000000.0,
            "p75": 15542000000.0,
            "p99": 27237719999.99998,
            "std": 3594817652.261032
          },
          "operatingIncome": {
            "dtype": "float64",
            "mad_center": 13896000000.0,
            "mad_outliers": 2,
            "mad_scale": 912000000.0,
            "max": 20605000000.0,
            "mean": 14277438596.491228,
            "min": 10123000000.0,
            "null_rate": 0.05,
            "nulls": 3,
            "nunique": 57,
            "p01": 10573240000.0,
            "p25": 13374000000.0,
            "p50": 13896000000.0,
            "p75": 14849000000.0,
            "p99": 20031559999.999996,
            "std": 1954564794.613075
          },
          "operatingLeaseLiability": {
            "dtype": "float64",
            "mad_center": 6197000000.0,
            "mad_outliers": 1,
            "mad_scale": 4248500000.0,
            "max": 30190000000.0,
            "mean": 9914928571.428572,
            "min": 1904000000.0,
            "null_rate": 0.7666666666666667,
            "nulls": 46,
            "nunique": 14,
            "p01": 1915570000.0,
            "p25": 2269000000.0,
            "p50": 6197000000.0,
            "p75": 15580000000.0,
            "p99": 29035599999.999992,
            "std": 8971105004.095398
          },
          "operatingMargins": {
            "dtype": "float64",
            "mad_center": 0.34231995381178304,
            "mad_outliers": 0,
            "mad_scale": 0.03509558119615325,
            "max": 0.3949435180204411,
            "mean": 0.3355080384301321,
            "min": 0.22924498392137324,
            "null_rate": 0.05,
            "nulls": 3,
            "nunique": 57,
            "p01": 0.2450472420430028,
            "p25": 0.30438171405333964,
            "p50": 0.34231995381178304,
            "p75": 0.3713009491903964,
            "p99": 0.3937232360186893,
            "std": 0.04160722536189667
          },
          "optionOverhang": {
            "dtype": "float64",
            "mad_center": 0.02410915509964906,
            "mad_outliers": 0,
            "mad_scale": 0.0037061989632245496,
            "max": 0.0318518518518518,
            "mean": 0.023351759595363314,
            "min": 0.015193124677893577,
            "null_rate": 0.05,
            "nulls": 3,
            "nunique": 57,
            "p01": 0.015221525948789134,
            "p25": 0.020071989339827345,
            "p50": 0.02410915509964906,
            "p75": 0.027608461814270457,
            "p99": 0.030951215279759532,
            "std": 0.004634392256676103
          },
          "ppeGross": {
            "dtype": "float64",
            "mad_center": 13075000000.0,
            "mad_outliers": 2,
            "mad_scale": 6599000000.0,
            "max": 122651000000.0,
            "mean": 24198533333.333332,
            "min": 5419000000.0,
            "null_rate": 0.75,
            "nulls": 45,
            "nunique": 15,
            "p01": 5466180000.0,
            "p25": 7695500000.0,
            "p50": 13075000000.0,
            "p75": 24174000000.0,
            "p99": 113817419999.99997,
            "std": 30828181876.94654
          },
          "ppeNet": {
            "dtype": "float64",
            "mad_center": 6100000000.0,
            "mad_outliers": 9,
            "mad_scale": 2969000000.0,
            "max": 92493000000.0,
            "mean": 12633516666.666666,
            "min": 2900000000.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 59,
            "p01": 2917700000.0,
            "p25": 3598250000.0,
            "p50": 6100000000.0,
            "p75": 12797750000.0,
            "p99": 83680759999.99995,
            "std": 17869409790.307983
          },
          "premiumsEarned": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 60,
            "nunique": 0
          },
          "pretaxIncome": {
            "dtype": "float64",
            "mad_center": 12426000000.0,
            "mad_outliers": 0,
            "mad_scale": 909000000.0,
            "max": 13941000000.0,
            "mean": 12463631578.947369,
            "min": 11407000000.0,
            "null_rate": 0.6833333333333333,
            "nulls": 41,
            "nunique": 19,
            "p01": 11413300000.0,
            "p25": 11519500000.0,
            "p50": 12426000000.0,
            "p75": 13297500000.0,
            "p99": 13919940000.0,
            "std": 953762736.074935
          },
          "profitMargins": {
            "dtype": "float64",
            "mad_center": 0.25414453550360727,
            "mad_outliers": 0,
            "mad_scale": 0.03280104397176897,
            "max": 0.34603330068560234,
            "mean": 0.24618425908352506,
            "min": 0.09548540737738144,
            "null_rate": 0.16666666666666666,
            "nulls": 10,
            "nunique": 50,
            "p01": 0.11519898445751413,
            "p25": 0.21419018399955395,
            "p50": 0.25414453550360727,
            "p75": 0.2811985475244473,
            "p99": 0.3450517999212128,
            "std": 0.0502036314872178
          },
          "publication_form": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2
          },
          "realizedInvestmentGains": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 60,
            "nunique": 0
          },
          "regime": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          },
          "rentalIncome": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 60,
            "nunique": 0
          },
          "researchAndDevelopment": {
            "dtype": "float64",
            "mad_center": 6099000000.0,
            "mad_outliers": 4,
            "mad_scale": 741000000.0,
            "max": 10313000000.0,
            "mean": 6741578947.368421,
            "min": 4523000000.0,
            "null_rate": 0.05,
            "nulls": 3,
            "nunique": 55,
            "p01": 4607560000.0,
            "p25": 5640000000.0,
            "p50": 6099000000.0,
            "p75": 8032000000.0,
            "p99": 10290600000.0,
            "std": 1663919586.6920502
          },
          "restrictedCash": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 60,
            "nunique": 0
          },
          "retainedEarnings": {
            "dtype": "float64",
            "mad_center": 1911500000.0,
            "mad_outliers": 0,
            "mad_scale": 22666000000.0,
            "max": 28586000000.0,
            "mean": 2502983333.3333335,
            "min": -34076000000.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 60,
            "p01": -33527890000.0,
            "p25": -17787250000.0,
            "p50": 1911500000.0,
            "p75": 24767000000.0,
            "p99": 28414900000.0,
            "std": 22708373816.86538
          },
          "returnOnEquity": {
            "dtype": "float64",
            "mad_center": 0.2463358362011564,
            "mad_outliers": 9,
            "mad_scale": 0.2658297541680623,
            "max": 5.57133676092545,
            "mean": 0.18095627537643588,
            "min": -12.506194690265486,
            "null_rate": 0.16666666666666666,
            "nulls": 10,
            "nunique": 50,
            "p01": -8.571370589106527,
            "p25": 0.18437421918594057,
            "p50": 0.2463358362011564,
            "p75": 0.7287261505049146,
            "p99": 4.4869607695432885,
            "std": 2.275979319615947
          },
          "revenue_q": {
            "dtype": "float64",
            "mad_center": 9942500000.0,
            "mad_outliers": 4,
            "mad_scale": 979000000.0,
            "max": 19184000000.0,
            "mean": 10909550000.0,
            "min": 8181000000.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 59,
            "p01": 8293690000.0,
            "p25": 9202000000.0,
            "p50": 9942500000.0,
            "p75": 11948750000.0,
            "p99": 18007539999.999992,
            "std": 2397999253.7918854
          },
          "sellingGeneralAdmin": {
            "dtype": "float64",
            "mad_center": 9371000000.0,
            "mad_outliers": 0,
            "mad_scale": 515000000.0,
            "max": 10411000000.0,
            "mean": 9354315789.473684,
            "min": 8133000000.0,
            "null_rate": 0.05,
            "nulls": 3,
            "nunique": 56,
            "p01": 8167720000.0,
            "p25": 8799000000.0,
            "p50": 9371000000.0,
            "p75": 9800000000.0,
            "p99": 10361160000.0,
            "std": 636641807.7328753
          },
          "sharesOutstanding": {
            "dtype": "float64",
            "mad_center": 3503286500.0,
            "mad_outliers": 1,
            "mad_scale": 721802500.0,
            "max": 4819056000000000.0,
            "mean": 80321145745233.33,
            "min": 2664926000.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 60,
            "p01": 2666832290.0,
            "p25": 2802414500.0,
            "p50": 3503286500.0,
            "p75": 4343825250.0,
            "p99": 1975815936315163.5,
            "std": 622136988902927.1
          },
          "shortTermBorrowingsOnly": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 60,
            "nunique": 0
          },
          "shortTermDebt": {
            "dtype": "float64",
            "mad_center": 2499000000.0,
            "mad_outliers": 2,
            "mad_scale": 1249000000.0,
            "max": 10605000000.0,
            "mean": 3462540540.5405407,
            "min": 0.0,
            "null_rate": 0.38333333333333336,
            "nulls": 23,
            "nunique": 28,
            "p01": 359640000.0,
            "p25": 1516000000.0,
            "p50": 2499000000.0,
            "p75": 4491000000.0,
            "p99": 10314120000.0,
            "std": 2540737989.843312
          },
          "shortTermInvestments": {
            "dtype": "float64",
            "mad_center": 677000000.0,
            "mad_outliers": 12,
            "mad_scale": 307000000.0,
            "max": 38567000000.0,
            "mean": 5878258064.5161295,
            "min": 207000000.0,
            "null_rate": 0.48333333333333334,
            "nulls": 29,
            "nunique": 30,
            "p01": 233400000.0,
            "p25": 458000000.0,
            "p50": 677000000.0,
            "p75": 8205000000.0,
            "p99": 34589899999.99999,
            "std": 9144561604.100155
          },
          "stockBasedComp": {
            "dtype": "float64",
            "mad_center": 1617000000.0,
            "mad_outliers": 0,
            "mad_scale": 725000000.0,
            "max": 4908000000.0,
            "mean": 2088140350.877193,
            "min": 660000000.0,
            "null_rate": 0.05,
            "nulls": 3,
            "nunique": 56,
            "p01": 685200000.0,
            "p25": 985000000.0,
            "p50": 1617000000.0,
            "p75": 3046000000.0,
            "p99": 4853680000.0,
            "std": 1347931662.089796
          },
          "stockholdersEquity": {
            "dtype": "float64",
            "mad_center": 38807500000.0,
            "mad_outliers": 0,
            "mad_scale": 13178500000.0,
            "max": 56366000000.0,
            "mean": 28682083333.333332,
            "min": -9658000000.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 60,
            "p01": -8804270000.0,
            "p25": 10014250000.0,
            "p50": 38807500000.0,
            "p75": 46333000000.0,
            "p99": 56327060000.0,
            "std": 20308655558.31203
          },
          "ticker": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          },
          "totalAssets": {
            "dtype": "float64",
            "mad_center": 114492000000.0,
            "mad_outliers": 2,
            "mad_scale": 18446500000.0,
            "max": 261759000000.0,
            "mean": 120873100000.0,
            "min": 72910000000.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 60,
            "p01": 73469910000.0,
            "p25": 98722750000.0,
            "p50": 114492000000.0,
            "p75": 134535750000.0,
            "p99": 252012789999.99994,
            "std": 36492029713.34973
          },
          "totalDebt": {
            "dtype": "float64",
            "mad_center": 2950000000.0,
            "mad_outliers": 4,
            "mad_scale": 1537000000.0,
            "max": 45090000000.0,
            "mean": 5495162162.162162,
            "min": 0.0,
            "null_rate": 0.38333333333333336,
            "nulls": 23,
            "nunique": 28,
            "p01": 359640000.0,
            "p25": 1516000000.0,
            "p50": 2950000000.0,
            "p75": 4494000000.0,
            "p99": 37373400000.000015,
            "std": 8224859108.936873
          },
          "totalLiabilities": {
            "dtype": "float64",
            "mad_center": 86778000000.0,
            "mad_outliers": 0,
            "mad_scale": 36885000000.0,
            "max": 218703000000.0,
            "mean": 92191016666.66667,
            "min": 30643000000.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 60,
            "p01": 30909680000.0,
            "p25": 56867250000.0,
            "p50": 86778000000.0,
            "p75": 130184500000.0,
            "p99": 211319739999.99997,
            "std": 44254863544.07617
          },
          "totalRevenue": {
            "dtype": "float64",
            "mad_center": 39531000000.0,
            "mad_outliers": 12,
            "mad_scale": 2102000000.0,
            "max": 67358000000.0,
            "mean": 43136263157.89474,
            "min": 36928000000.0,
            "null_rate": 0.05,
            "nulls": 3,
            "nunique": 56,
            "p01": 36994640000.0,
            "p25": 37901000000.0,
            "p50": 39531000000.0,
            "p75": 46073000000.0,
            "p99": 65520639999.99999,
            "std": 7878915789.54852
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
        "pk_checked_rows": 60,
        "pk_complete": true,
        "pk_duplicate_rows": 0,
        "pk_missing_cols": [],
        "rows": 3258,
        "sample_date_max": "2026-06-22",
        "sample_date_min": "2011-09-23",
        "sampled_rows": 60,
        "scope": {
          "limit": null,
          "since": null,
          "tickers": [
            "ORCL"
          ]
        },
        "table": "fundamentals_history"
      },
      "fundamentals_reason_codes": {
        "columns": [
          "ticker",
          "as_of",
          "field",
          "dc_code",
          "combined_into",
          "rejected_value"
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
            "nunique": 60
          },
          "combined_into": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 1339,
            "nunique": 0
          },
          "dc_code": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 11
          },
          "field": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 49
          },
          "rejected_value": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 1339,
            "nunique": 0
          },
          "ticker": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
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
        "pk_checked_rows": 1339,
        "pk_complete": true,
        "pk_duplicate_rows": 0,
        "pk_missing_cols": [],
        "rows": 75829,
        "sample_date_max": "2026-06-22",
        "sample_date_min": "2011-09-23",
        "sampled_rows": 1339,
        "scope": {
          "limit": null,
          "since": null,
          "tickers": [
            "ORCL"
          ]
        },
        "table": "fundamentals_reason_codes"
      }
    }
  },
  "scope": {
    "limit": null,
    "since": null,
    "tables": [
      "fundamentals_facts",
      "fundamentals_history",
      "fundamentals_reason_codes"
    ],
    "tickers": [
      "ORCL"
    ],
    "unknown_tables": []
  },
  "session_id": "4d204d4b-c056-4b30-a83d-5ca4fd25bf76",
  "type": "DATA"
}
```

