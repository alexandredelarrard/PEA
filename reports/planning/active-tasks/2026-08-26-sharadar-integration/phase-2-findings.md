# Phase 2 — Sharadar acceptance gates, measured from the database

**Generated**: 2026-08-26  
**Scope**: 30 ticker(s), 2021-08-27..2026-08-10  
**Source**: `fundamentals_sharadar` (read-only) cross-checked against `fundamentals_history_sec` on 14 overlapping ticker(s).

> ⚠ This is **not** the SEC check scheme (D25). No check was registered, no `fundamentals_check` row was written, and `src/validate/` was neither imported nor invoked. This is a standalone read-only diagnostic whose only consumers are the Full-tier purchase decision and phase 3's `sharadar_zero_rules.json`.

---

## Gate 1 — completeness

Expected quarter count is measured against **each ticker's own observed window**, not a global start: on a 5-year entitlement a ticker whose history begins late is not a gap. Quarters are Sharadar's own `calendardate`, already normalised to the nearest calendar quarter-end, so 52/53-week retail calendars need no special handling.

| ticker | first_quarter | last_quarter | n_rows | n_quarters | expected_quarters | n_missing | missing_quarters | n_duplicate_quarters |
|---|---|---|---|---|---|---|---|---|
| AAPL | 2021Q3 | 2026Q2 | 20 | 20 | 20 | 0 | - | 0 |
| AMGN | 2021Q3 | 2026Q2 | 20 | 20 | 20 | 0 | - | 0 |
| AMZN | 2021Q3 | 2026Q2 | 20 | 20 | 20 | 0 | - | 0 |
| AXP | 2021Q3 | 2026Q2 | 20 | 20 | 20 | 0 | - | 0 |
| BA | 2021Q3 | 2026Q2 | 20 | 20 | 20 | 0 | - | 0 |
| CAT | 2021Q3 | 2026Q2 | 20 | 20 | 20 | 0 | - | 0 |
| CRM | 2021Q2 | 2026Q1 | 20 | 20 | 20 | 0 | - | 0 |
| CSCO | 2021Q2 | 2026Q1 | 20 | 20 | 20 | 0 | - | 0 |
| CVX | 2021Q3 | 2026Q2 | 20 | 20 | 20 | 0 | - | 0 |
| DIS | 2021Q3 | 2026Q2 | 20 | 20 | 20 | 0 | - | 0 |
| GS | 2021Q3 | 2026Q2 | 20 | 20 | 20 | 0 | - | 0 |
| HD | 2021Q3 | 2026Q1 | 19 | 19 | 19 | 0 | - | 0 |
| HON | 2021Q3 | 2026Q2 | 20 | 20 | 20 | 0 | - | 0 |
| IBM | 2021Q3 | 2026Q2 | 20 | 20 | 20 | 0 | - | 0 |
| JNJ | 2021Q3 | 2026Q2 | 20 | 20 | 20 | 0 | - | 0 |
| JPM | 2021Q3 | 2026Q2 | 20 | 20 | 20 | 0 | - | 0 |
| KO | 2021Q3 | 2026Q2 | 20 | 20 | 20 | 0 | - | 0 |
| MCD | 2021Q3 | 2026Q2 | 20 | 20 | 20 | 0 | - | 0 |
| MMM | 2021Q3 | 2026Q2 | 20 | 20 | 20 | 0 | - | 0 |
| MRK | 2021Q3 | 2026Q2 | 20 | 20 | 20 | 0 | - | 0 |
| MSFT | 2021Q3 | 2026Q2 | 20 | 20 | 20 | 0 | - | 0 |
| NKE | 2021Q3 | 2026Q2 | 20 | 20 | 20 | 0 | - | 0 |
| NVDA | 2021Q3 | 2026Q1 | 19 | 19 | 19 | 0 | - | 0 |
| PG | 2021Q3 | 2026Q2 | 20 | 20 | 20 | 0 | - | 0 |
| SHW | 2021Q3 | 2026Q2 | 20 | 20 | 20 | 0 | - | 0 |
| TRV | 2021Q3 | 2026Q2 | 20 | 20 | 20 | 0 | - | 0 |
| UNH | 2021Q3 | 2026Q2 | 20 | 20 | 20 | 0 | - | 0 |
| V | 2021Q3 | 2026Q2 | 20 | 20 | 20 | 0 | - | 0 |
| VZ | 2021Q3 | 2026Q2 | 20 | 20 | 20 | 0 | - | 0 |
| WMT | 2021Q2 | 2026Q1 | 20 | 20 | 20 | 0 | - | 0 |


## Gate 2 — implausible quarters

Replaces the spec's acceptance check #3, which is dead (see gate 4). Sharadar CONSTRUCTS Q4 as `ARY - Σ(Q1..Q3)`; the identity therefore cannot fail, but the construction can produce absurd levels — the legacy Quandl documentation shows it yielding ABT 2011 Q4 revenue of **-$7.1bn**. Magnitude threshold is `data_extract.fundamentals_periods.max_opposite_sign_q4_ratio` = **3.0**, reused from the SEC path rather than reinvented.

**239 flagged cell(s)**: 17 negative, 222 magnitude.

**Every negative — these are the findings that matter.** A negative in a field with no negative reading is a level error the annual row still absorbs, so no identity check can ever see it:

| ticker | field | fiscal_year | fiscal_position | calendardate | value | reason | max_other_abs | ratio_vs_other |
|---|---|---|---|---|---|---|---|---|
| IBM | cor | 2021 | Q4 | 2021-12-31 | -2,849,000,000 | negative |  |  |
| MMM | intexp | 2023 | Q4 | 2023-12-31 | -571,000,000 | negative |  |  |
| NKE | intexp | 2023 | Q3 | 2023-03-31 | -7,000,000 | negative |  |  |
| NKE | intexp | 2023 | Q4 | 2023-06-30 | -28,000,000 | negative |  |  |
| NKE | intexp | 2024 | Q1 | 2023-09-30 | -34,000,000 | negative |  |  |
| NKE | intexp | 2024 | Q2 | 2023-12-31 | -22,000,000 | negative |  |  |
| NKE | intexp | 2024 | Q3 | 2024-03-31 | -52,000,000 | negative |  |  |
| NKE | intexp | 2024 | Q4 | 2024-06-30 | -53,000,000 | negative |  |  |
| NKE | intexp | 2025 | Q1 | 2024-09-30 | -43,000,000 | negative |  |  |
| NKE | intexp | 2025 | Q2 | 2024-12-31 | -24,000,000 | negative |  |  |
| NKE | intexp | 2025 | Q3 | 2025-03-31 | -18,000,000 | negative |  |  |
| NKE | intexp | 2025 | Q4 | 2025-06-30 | -22,000,000 | negative |  |  |
| NKE | intexp | 2026 | Q1 | 2025-09-30 | -18,000,000 | negative |  |  |
| NKE | intexp | 2026 | Q2 | 2025-12-31 | -9,000,000 | negative |  |  |
| NKE | intexp | 2026 | Q3 | 2026-03-31 | -15,000,000 | negative |  |  |
| NKE | intexp | 2026 | Q4 | 2026-06-30 | -8,000,000 | negative |  |  |
| MMM | opex | 2022 | Q3 | 2022-09-30 | -265,000,000 | negative |  |  |


**Magnitude outliers, by field and by fiscal position.** Read these two tables before the row dump: they say the threshold is measuring lumpiness, not error.

| field | n_flagged |
|---|---|
| ncfbus | 48 |
| ncfdebt | 31 |
| ncfinv | 20 |
| ncfi | 20 |
| ncf | 14 |
| ncfx | 13 |
| ncff | 11 |
| ncfcommon | 8 |
| sbcomp | 5 |
| netincnci | 4 |
| eps | 4 |
| epsdil | 4 |
| consolinc | 4 |
| netinc | 4 |
| netinccmn | 4 |
| fcf | 3 |
| ebt | 3 |
| netincdis | 2 |
| investmentsc | 2 |
| debtc | 2 |
| taxexp | 2 |
| opinc | 2 |
| ebitda | 2 |
| ebit | 2 |
| ncfo | 1 |
| investments | 1 |
| deferredrev | 1 |
| cashneq | 1 |
| cashnequsd | 1 |
| capex | 1 |
| intexp | 1 |
| intangibles | 1 |


| fiscal_position | n_flagged |
|---|---|
| Q3 | 60 |
| Q1 | 59 |
| Q4 | 52 |
| Q2 | 51 |


Worst 20 by ratio:

| ticker | field | fiscal_year | fiscal_position | calendardate | value | reason | max_other_abs | ratio_vs_other |
|---|---|---|---|---|---|---|---|---|
| MMM | ncfdebt | 2024 | Q1 | 2024-03-31 | 5,509,000,000 | magnitude | 3,000,000 | 1,836 |
| AMGN | ncfinv | 2023 | Q1 | 2023-03-31 | 1,674,000,000 | magnitude | 2,000,000 | 837 |
| WMT | ncfbus | 2025 | Q4 | 2024-12-31 | -1,899,000,000 | magnitude | 3,000,000 | 633 |
| HD | ncfbus | 2025 | Q2 | 2024-06-30 | -17,570,000,000 | magnitude | 43,000,000 | 408.6 |
| TRV | ncfbus | 2024 | Q1 | 2024-03-31 | -381,000,000 | magnitude | 1,000,000 | 381 |
| BA | ncfbus | 2025 | Q4 | 2025-12-31 | 9,302,000,000 | magnitude | 35,000,000 | 265.8 |
| CRM | ncfbus | 2022 | Q2 | 2021-06-30 | -14,356,000,000 | magnitude | 60,000,000 | 239.3 |
| JNJ | netincdis | 2023 | Q3 | 2023-09-30 | -21,719,000,000 | magnitude | 108,000,000 | 201.1 |
| MCD | ncfinv | 2024 | Q1 | 2024-03-31 | -1,820,000,000 | magnitude | 17,000,000 | 107.1 |
| JNJ | ncfbus | 2022 | Q4 | 2022-12-31 | -16,909,000,000 | magnitude | 205,000,000 | 82.48 |
| NKE | ncfdebt | 2023 | Q4 | 2023-06-30 | -508,000,000 | magnitude | 7,000,000 | 72.57 |
| GS | ncfbus | 2023 | Q4 | 2023-12-31 | 495,000,000 | magnitude | 8,000,000 | 61.88 |
| V | ncfbus | 2025 | Q1 | 2024-12-31 | -906,000,000 | magnitude | 19,000,000 | 47.68 |
| CVX | ncfbus | 2022 | Q2 | 2022-06-30 | -2,845,000,000 | magnitude | 62,000,000 | 45.89 |
| BA | ncfcommon | 2023 | Q1 | 2023-03-31 | 44,000,000 | magnitude | 1,000,000 | 44 |
| NVDA | ncfdebt | 2024 | Q2 | 2023-06-30 | -1,261,000,000 | magnitude | 30,000,000 | 42.03 |
| MSFT | ncfbus | 2024 | Q2 | 2023-12-31 | -65,029,000,000 | magnitude | 1,575,000,000 | 41.29 |
| DIS | ncf | 2026 | Q3 | 2026-06-30 | -490,000,000 | magnitude | 13,000,000 | 37.69 |
| MMM | ncfbus | 2022 | Q3 | 2022-09-30 | 478,000,000 | magnitude | 13,000,000 | 36.77 |
| CRM | ncfdebt | 2026 | Q4 | 2025-12-31 | 5,854,000,000 | magnitude | 179,000,000 | 32.7 |

_20 of 239 rows shown._


## Gate 3 — zero-fill prevalence, per field

`n_zero_mixed` is the Sharadar-internal signal: zeros belonging to a ticker that reports the **same field non-zero in another quarter**. It is the only evidence available for the 21 fields with no SEC counterpart, and it needs no basis reconciliation.

The SEC columns are **basis-matched** before comparison: a duration field is judged at the TTM level (`fundamentals_history_sec` is TTM, so a non-zero there says nothing about one quarter — the Sharadar side of that comparison is its own ART dimension), an instant field point-in-time. A `sec_wider` counterpart can only produce `sec_suspect`, never `sec_contradicted`: `totalDebt` carries lease liabilities Sharadar's `debt` does not, and `cash` carries restricted cash and short-term investments its `cashneq` does not.

| field | n_rows | n_zero | pct_zero | n_tickers | n_tickers_all_zero | n_zero_mixed | sec_basis | sec_overlap_zeros | sec_checked | sec_agrees | sec_absent | sec_contradicted | sec_suspect | sec_inconclusive | sec_contradicted_tickers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| netincdis | 598 | 574 | 0.9599 | 30 | 24 | 96 | no SEC counterpart | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| prefdivis | 598 | 571 | 0.9548 | 30 | 28 | 13 | no SEC counterpart | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| deposits | 598 | 518 | 0.8662 | 30 | 26 | 0 | no SEC counterpart | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| taxassets | 598 | 382 | 0.6388 | 30 | 19 | 3 | no SEC counterpart | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| deferredrev | 598 | 379 | 0.6338 | 30 | 19 | 0 | no SEC counterpart | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| netincnci | 598 | 363 | 0.607 | 30 | 17 | 25 | no SEC counterpart | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| rnd | 598 | 305 | 0.51 | 30 | 15 | 6 | researchAndDevelopment (exact, TTM) | 140 | 0 | 0 | 140 | 0 | 0 | 0 |  |
| investmentsnc | 518 | 216 | 0.417 | 26 | 10 | 17 | no SEC counterpart | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| ncfbus | 598 | 236 | 0.3946 | 30 | 2 | 196 | no SEC counterpart | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| investmentsc | 518 | 186 | 0.3591 | 26 | 8 | 27 | shortTermInvestments (exact, instant) | 80 | 0 | 0 | 80 | 0 | 0 | 0 |  |
| taxliabilities | 598 | 180 | 0.301 | 30 | 8 | 21 | no SEC counterpart | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| inventory | 598 | 140 | 0.2341 | 30 | 7 | 0 | inventory (exact, instant) | 80 | 4 | 0 | 76 | 4 | 0 | 0 | UNH |
| ncfinv | 598 | 126 | 0.2107 | 30 | 1 | 107 | no SEC counterpart | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| investments | 598 | 112 | 0.1873 | 30 | 5 | 13 | no SEC counterpart | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| intexp | 598 | 99 | 0.1656 | 30 | 3 | 39 | interestExpense (exact, TTM) | 71 | 58 | 0 | 13 | 58 | 0 | 0 | AXP, GS, JPM |
| dps | 598 | 96 | 0.1605 | 30 | 2 | 56 | no SEC counterpart | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| ncfx | 598 | 92 | 0.1538 | 30 | 4 | 13 | no SEC counterpart | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| ncfcommon | 598 | 82 | 0.1371 | 30 | 0 | 82 | no SEC counterpart | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| divyield | 598 | 73 | 0.1221 | 30 | 2 | 33 | no SEC counterpart | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| ncfdiv | 598 | 66 | 0.1104 | 30 | 2 | 26 | no SEC counterpart | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| cor | 598 | 60 | 0.1003 | 30 | 3 | 0 | costOfRevenue (exact, TTM) | 60 | 0 | 0 | 60 | 0 | 0 | 0 |  |
| intangibles | 598 | 58 | 0.09699 | 30 | 2 | 18 | goodwill+intangiblesExGoodwill (sec_wider, instant) | 58 | 25 | 0 | 33 | 0 | 25 | 0 |  |
| ncfdebt | 598 | 56 | 0.09365 | 30 | 0 | 56 | no SEC counterpart | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| debtc | 518 | 36 | 0.0695 | 26 | 1 | 16 | shortTermDebt (exact, instant) | 8 | 5 | 5 | 3 | 0 | 0 | 0 |  |
| capex | 598 | 40 | 0.06689 | 30 | 2 | 0 | capex (exact, TTM) | 20 | 0 | 0 | 20 | 0 | 0 | 0 |  |
| ppnenet | 598 | 40 | 0.06689 | 30 | 2 | 0 | ppeNet (exact, instant) | 20 | 12 | 0 | 8 | 12 | 0 | 0 | GS |
| depamor | 598 | 20 | 0.03344 | 30 | 1 | 0 | depAmort (exact, TTM) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| debtusd | 598 | 0 | 0 | 30 | 0 | 0 | totalDebt (sec_wider, instant) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| debtnc | 518 | 0 | 0 | 26 | 0 | 0 | longTermDebt (exact, instant) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| cashneq | 598 | 0 | 0 | 30 | 0 | 0 | cash (sec_wider, instant) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| accoci | 598 | 0 | 0 | 30 | 0 | 0 | no SEC counterpart | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| debt | 598 | 0 | 0 | 30 | 0 | 0 | totalDebt (sec_wider, instant) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| cashnequsd | 598 | 0 | 0 | 30 | 0 | 0 | cash (sec_wider, instant) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| ncff | 598 | 0 | 0 | 30 | 0 | 0 | no SEC counterpart | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| ncfi | 598 | 0 | 0 | 30 | 0 | 0 | no SEC counterpart | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| payables | 598 | 0 | 0 | 30 | 0 | 0 | accountsPayable (exact, instant) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| receivables | 598 | 0 | 0 | 30 | 0 | 0 | accountsReceivable (exact, instant) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| revenueusd | 598 | 0 | 0 | 30 | 0 | 0 | totalRevenue (exact, TTM) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| revenue | 598 | 0 | 0 | 30 | 0 | 0 | totalRevenue (exact, TTM) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| sgna | 598 | 0 | 0 | 30 | 0 | 0 | sellingGeneralAdmin (exact, TTM) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| taxexp | 598 | 0 | 0 | 30 | 0 | 0 | incomeTaxExpense (exact, TTM) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |


### The proposed rule

Two rules only: `null` (treat 0 as unknown) and `keep` (0 is a real value). Machine-proposed here, **human-approved** in `configs/sharadar/sharadar_zero_rules.json`. Written to `configs/sharadar/sharadar_zero_rules.json`.

| field | rule | pct_zero | reason |
|---|---|---|---|
| netincdis | keep | 0.9599 | discrete event: a zero means no transaction that quarter, so the 96/574 zeros in mixed tickers are expected, not a fill |
| prefdivis | keep | 0.9548 | 13/571 zeros are in mixed tickers, below the 50% bar; no SEC counterpart, so this rests on Sharadar alone |
| deposits | keep | 0.8662 | structural: every zero belongs to one of 26 tickers that never report this field |
| taxassets | keep | 0.6388 | 3/382 zeros are in mixed tickers, below the 50% bar; no SEC counterpart, so this rests on Sharadar alone |
| deferredrev | keep | 0.6338 | structural: every zero belongs to one of 19 tickers that never report this field |
| netincnci | keep | 0.607 | 25/363 zeros are in mixed tickers, below the 50% bar; no SEC counterpart, so this rests on Sharadar alone |
| rnd | keep | 0.51 | 6/305 zeros are in mixed tickers, below the 50% bar; the SEC layer has no value either on all 140 overlap zeros, which is agreement -- it stores NULL where it finds nothing, never 0 |
| investmentsnc | keep | 0.417 | 17/216 zeros are in mixed tickers, below the 50% bar; no SEC counterpart, so this rests on Sharadar alone |
| ncfbus | keep | 0.3946 | discrete event: a zero means no transaction that quarter, so the 196/236 zeros in mixed tickers are expected, not a fill |
| investmentsc | keep | 0.3591 | 27/186 zeros are in mixed tickers, below the 50% bar; the SEC layer has no value either on all 80 overlap zeros, which is agreement -- it stores NULL where it finds nothing, never 0 |
| taxliabilities | keep | 0.301 | 21/180 zeros are in mixed tickers, below the 50% bar; no SEC counterpart, so this rests on Sharadar alone |
| ncfinv | keep | 0.2107 | discrete event: a zero means no transaction that quarter, so the 107/126 zeros in mixed tickers are expected, not a fill |
| investments | keep | 0.1873 | 13/112 zeros are in mixed tickers, below the 50% bar; no SEC counterpart, so this rests on Sharadar alone |
| ncfx | keep | 0.1538 | discrete event: a zero means no transaction that quarter, so the 13/92 zeros in mixed tickers are expected, not a fill |
| ncfcommon | keep | 0.1371 | discrete event: a zero means no transaction that quarter, so the 82/82 zeros in mixed tickers are expected, not a fill |
| divyield | keep | 0.1221 | 33/73 zeros are in mixed tickers, below the 50% bar; no SEC counterpart, so this rests on Sharadar alone |
| ncfdiv | keep | 0.1104 | 26/66 zeros are in mixed tickers, below the 50% bar; no SEC counterpart, so this rests on Sharadar alone |
| cor | keep | 0.1003 | structural: every zero belongs to one of 3 tickers that never report this field; SEC agrees or is equally absent on 60/60 |
| intangibles | keep | 0.097 | 18/58 zeros are in mixed tickers, below the 50% bar; SEC contradicts 0/25 of the zeros it could judge, and is equally absent on 33 more |
| ncfdebt | keep | 0.0936 | discrete event: a zero means no transaction that quarter, so the 56/56 zeros in mixed tickers are expected, not a fill |
| debtc | keep | 0.0695 | 16/36 zeros are in mixed tickers, below the 50% bar; SEC contradicts 0/5 of the zeros it could judge, and is equally absent on 3 more |
| capex | keep | 0.0669 | structural: every zero belongs to one of 2 tickers that never report this field; SEC agrees or is equally absent on 20/20 |
| depamor | keep | 0.0334 | structural: every zero belongs to one of 1 tickers that never report this field |
| accoci | keep | 0 | never zero in 598 stored ARQ rows |
| cashneq | keep | 0 | never zero in 598 stored ARQ rows |
| cashnequsd | keep | 0 | never zero in 598 stored ARQ rows |
| debt | keep | 0 | never zero in 598 stored ARQ rows |
| debtnc | keep | 0 | never zero in 518 stored ARQ rows |
| debtusd | keep | 0 | never zero in 598 stored ARQ rows |
| ncff | keep | 0 | never zero in 598 stored ARQ rows |
| ncfi | keep | 0 | never zero in 598 stored ARQ rows |
| payables | keep | 0 | never zero in 598 stored ARQ rows |
| receivables | keep | 0 | never zero in 598 stored ARQ rows |
| revenue | keep | 0 | never zero in 598 stored ARQ rows |
| revenueusd | keep | 0 | never zero in 598 stored ARQ rows |
| sgna | keep | 0 | never zero in 598 stored ARQ rows |
| taxexp | keep | 0 | never zero in 598 stored ARQ rows |
| inventory | null | 0.2341 | provably wrong: the SEC layer reports a non-zero inventory (exact, instant) on 4/4 of the zeros it could judge (UNH) |
| intexp | null | 0.1656 | provably wrong: the SEC layer reports a non-zero interestExpense (exact, TTM) on 58/58 of the zeros it could judge (AXP, GS, JPM) |
| dps | null | 0.1605 | 56/96 zeros sit in tickers that report this field non-zero in another quarter, which is a fill and not a fact |
| ppnenet | null | 0.0669 | provably wrong: the SEC layer reports a non-zero ppeNet (exact, instant) on 12/12 of the zeros it could judge (GS) |


## Gate 4 — the Q4 identity is tautological (evidence, not a check)

ΣARQ vs ARY per (ticker, fiscal year, field) over the 33 duration fields, on 3,532 comparable triples.

- **exactly zero: 3,395 / 3,532 (96.1%)** — not "small", *exactly* 0.0
- float noise (0 < dev ≤ 0.01%): 79
- materially non-zero: **58** (1.6%), max 135.98%

This is the number that kills the spec's acceptance check #3. Sharadar builds Q4 by subtraction, so wherever the identity holds it holds **exactly**, and the check carries no information about the quality of the quarters it is supposedly testing. Recorded here so it is not re-proposed as a gate later.

⚠ **But the exceptions are not noise, and they are not what the plan expected.** The plan predicted `+0.000%` everywhere. The residual triples are fiscal years Sharadar **restated between publishing the quarters and publishing the year** — a real event. They cluster hard, which is what tells you it is not arithmetic drift:

| ticker | fiscal_year | n_fields |
|---|---|---|
| MMM | 2024 | 14 |
| MCD | 2024 | 9 |
| IBM | 2022 | 5 |
| MCD | 2025 | 4 |
| IBM | 2024 | 3 |
| NVDA | 2024 | 3 |
| PG | 2024 | 3 |
| PG | 2026 | 3 |
| MRK | 2025 | 3 |
| SHW | 2024 | 3 |
| MSFT | 2026 | 2 |
| JPM | 2024 | 1 |
| JPM | 2025 | 1 |
| MRK | 2022 | 1 |
| IBM | 2025 | 1 |
| MRK | 2023 | 1 |
| MRK | 2024 | 1 |


`MMM` FY2024 is the worst, with **14 fields** moving together — its quarters sum to 26,562,000,000 of revenue against an annual row of 24,575,000,000, a gap of 8.1%. A whole income statement moving in one direction at once is a **restated year**, not arithmetic drift — the quarters were published before the reclassification and the annual row after it. Phase 4's gap check will see these same years, and should not read them as extraction defects.

Worst 20 triples:

| ticker | fiscal_year | field | sum_arq | ary | abs_diff | pct_dev |
|---|---|---|---|---|---|---|
| MMM | 2024 | netincdis | 59,000,000 | -164,000,000 | 223,000,000 | 1.36 |
| MMM | 2024 | rnd | 1,268,000,000 | 1,085,000,000 | 183,000,000 | 0.1687 |
| MMM | 2024 | opex | 6,097,000,000 | 5,306,000,000 | 791,000,000 | 0.1491 |
| MMM | 2024 | sgna | 4,829,000,000 | 4,221,000,000 | 608,000,000 | 0.144 |
| MSFT | 2026 | depamor | 43,448,000,000 | 38,534,000,000 | 4,914,000,000 | 0.1275 |
| MMM | 2024 | gp | 11,271,000,000 | 10,128,000,000 | 1,143,000,000 | 0.1129 |
| MRK | 2025 | netincnci | 8,000,000 | 9,000,000 | 1,000,000 | 0.1111 |
| MMM | 2024 | taxexp | 889,000,000 | 804,000,000 | 85,000,000 | 0.1057 |
| MMM | 2024 | ebit | 5,447,000,000 | 4,977,000,000 | 470,000,000 | 0.09443 |
| MMM | 2024 | revenue | 26,562,000,000 | 24,575,000,000 | 1,987,000,000 | 0.08085 |
| MMM | 2024 | revenueusd | 26,562,000,000 | 24,575,000,000 | 1,987,000,000 | 0.08085 |
| MMM | 2024 | ebitda | 6,810,000,000 | 6,340,000,000 | 470,000,000 | 0.07413 |
| MMM | 2024 | opinc | 5,174,000,000 | 4,822,000,000 | 352,000,000 | 0.073 |
| MMM | 2024 | cor | 15,291,000,000 | 14,447,000,000 | 844,000,000 | 0.05842 |
| IBM | 2025 | netincdis | -23,000,000 | -22,000,000 | 1,000,000 | 0.04545 |
| MSFT | 2026 | ebitda | 212,433,000,000 | 207,519,000,000 | 4,914,000,000 | 0.02368 |
| MMM | 2024 | ebt | 5,062,000,000 | 4,977,000,000 | 85,000,000 | 0.01708 |
| PG | 2024 | netincnci | 94,000,000 | 95,000,000 | 1,000,000 | 0.01053 |
| PG | 2026 | netincnci | 99,000,000 | 98,000,000 | 1,000,000 | 0.0102 |
| MCD | 2024 | ncfx | -102,000,000 | -101,000,000 | 1,000,000 | 0.009901 |


## Sign conventions (a stop condition for phase 3)

`capex <= 0` throughout and `fcf == ncfo + capex` to the cent. The phase-3 field map maps `freeCashflow <- fcf` with **no reconstruction**, and flips `capex`'s sign because the SEC catalogue declares it non-negative while Sharadar stores it negative.

| dimension | rows | capex > 0 | capex max | fcf rows | max abs fcf residual | violations | worst row |
|---|---|---|---|---|---|---|---|
| ARQ | 598 | 6 | 2,746,000,000 | 598 | 0 | 0 | AAPL 2021-10-29 |
| ART | 598 | 6 | 1,371,000,000 | 598 | 0 | 0 | AAPL 2021-10-29 |
| ARY | 150 | 1 | 962,000,000 | 150 | 0 | 0 | AAPL 2021-10-29 |


Every row with a positive `capex`:

| ticker | dimension | date | fiscalperiod | capex |
|---|---|---|---|---|
| CVX | ARQ | 2025-02-21 | 2024-Q4 | 2,746,000,000 |
| GS | ARQ | 2024-02-23 | 2023-Q4 | 1,581,000,000 |
| GS | ARQ | 2021-11-01 | 2021-Q3 | 1,493,000,000 |
| GS | ART | 2024-08-02 | 2024-Q2 | 1,371,000,000 |
| GS | ART | 2024-11-04 | 2024-Q3 | 1,228,000,000 |
| GS | ART | 2024-05-03 | 2024-Q1 | 1,044,000,000 |
| GS | ARY | 2024-02-23 | 2023-FY | 962,000,000 |
| GS | ART | 2024-02-23 | 2023-Q4 | 962,000,000 |
| GS | ARQ | 2023-02-24 | 2022-Q4 | 680,000,000 |
| GS | ART | 2022-08-04 | 2022-Q2 | 131,000,000 |
| BA | ARQ | 2021-10-27 | 2021-Q3 | 89,000,000 |
| GS | ART | 2023-11-03 | 2023-Q3 | 61,000,000 |
| IBM | ARQ | 2024-10-30 | 2024-Q3 | 55,000,000 |


- `fcf == ncfo + capex`: **HOLDS** — so `freeCashflow <- fcf` needs no reconstruction, as decided.
- `capex <= 0`: **DOES NOT HOLD**. 13 of 1346 rows (0.97%) carry a POSITIVE capex, on BA, CVX, GS, IBM — and 10 of the 13 are GS alone.

**Consequence for phase 3, stated precisely:** the identity is not universal, so an unconditional `capex = -sharadar.capex` would write a *negative* value into a column the SEC catalogue declares `non_negative`, on those rows. The fix is a guard, not a different mapping — flip the sign where `capex <= 0` and NULL the rest, recording them. This does not invalidate the field map; it invalidates doing it blind.

## `sharesbas` cross-check (D-decision `sharesOutstanding <- sharesbas`)

Whether Sharadar SUMS MULTIPLE SHARE CLASSES is undocumented. The SEC column is a known **consolidated** basis — this repo built it for 36 multi-class tickers by summing the cover-page `dei:EntityCommonStockSharesOutstanding` across classes — so a systematic ratio, not noise, is the answer.

| ticker | n_dates | median_ratio | min_ratio | max_ratio | median_sharefactor | ratio_span | verdict |
|---|---|---|---|---|---|---|---|
| NVDA | 19 | 10 | 1 | 10.01 | 1 | 10.01 | SPLIT-ADJUSTED history (not as-filed) |
| WMT | 20 | 2 | 1 | 3 | 1 | 3 | SPLIT-ADJUSTED history (not as-filed) |
| CAT | 20 | 1 | 1 | 1.022 | 1 | 1.022 | agrees with the SEC cover page |
| BA | 20 | 1 | 1 | 1 | 1 | 1 | agrees with the SEC cover page |
| AXP | 20 | 1 | 1 | 1 | 1 | 1 | agrees with the SEC cover page |
| AAPL | 20 | 1 | 1 | 1 | 1 | 1 | agrees with the SEC cover page |
| GS | 20 | 1 | 1 | 1 | 1 | 1 | agrees with the SEC cover page |
| CSCO | 20 | 1 | 1 | 1 | 1 | 1 | agrees with the SEC cover page |
| JNJ | 20 | 1 | 1 | 1 | 1 | 1 | agrees with the SEC cover page |
| JPM | 20 | 1 | 1 | 1 | 1 | 1 | agrees with the SEC cover page |
| MSFT | 20 | 1 | 1 | 1 | 1 | 1 | agrees with the SEC cover page |
| MCD | 20 | 1 | 1 | 1 | 1 | 1 | agrees with the SEC cover page |
| PG | 20 | 1 | 1 | 1 | 1 | 1 | agrees with the SEC cover page |
| UNH | 20 | 1 | 1 | 1 | 1 | 1 | agrees with the SEC cover page |


🚨 **The share-class question is not the finding here.** 12 of 14 tickers sit at exactly 1.0, so Sharadar is not carrying one class of a multi-class filer. What the `ratio_span` column exposes instead is that **`sharesbas` is retroactively SPLIT-ADJUSTED**: NVDA's 2021-11-22 row reports 25.0bn shares against the ~2.5bn actually outstanding before its June 2024 10-for-1, and WMT shows the same at 3x for its February 2024 split. `sharefactor` is `1.0` on every one of those rows and does **not** flag it.

That makes `sharesbas` **not point-in-time**, which is a different and more serious property than the one D-decision asked about. Anything that multiplies it by an as-filed price — a market cap, a per-share book value — is wrong by the split factor for every date before the split. Phase 3 must either take `sharesOutstanding` from the SEC layer on the overlap, or de-adjust `sharesbas` using `sharadar_actions`, which is already ingested and carries the split events.

---

## The decision this phase hands back

1. **Completeness — clean.** No gaps at all, over 30 ticker(s) measured against each ticker's own observed window, and 0 duplicate normalised quarter(s). There is nothing structural or random to distinguish, because there is nothing.
2. **Implausible quarters — 17 real, the rest is lumpiness.** The 17 negative value(s) in a field that has no negative reading are the only LEVEL errors: intexp (15), cor (1), opex (1). The 222 magnitude outlier(s) beyond 3.0x the largest other quarter are **not** a Q4 construction artefact — only 23% sit in the Q4 position (chance is 25%), and 76% of them are `ncf*` / `fcf` legs, where one acquisition or one bond issue legitimately dwarfs the year. The threshold was calibrated on the SEC path's income-statement fields and does not transfer to event-driven cash-flow lines.
3. **Zero rule — 4 of 41 fields must be NULL-ruled** (dps, intexp, inventory, ppnenet), converting 375 of 24,198 measured cells (1.55%) from 0 to NULL.

**Recommendation**: **Buy the Full tier — but do not let phase 3 map a single field blind.**

The data clears the gates that would have been disqualifying: no missing quarters, `fcf == ncfo + capex` exactly, and every remaining defect has a measured size and a mechanical fix. What this phase actually bought you is the list of those defects, and it is longer than the plan assumed:

- `capex` is positive on 13 of 1346 rows (BA, CVX, GS, IBM) — guard the sign flip and NULL the exceptions
- 17 negative value(s) in non-negative fields — enumerated above, so each can be settled against its filing
- `sharesbas` is retroactively split-adjusted (NVDA, WMT), so it is not point-in-time and must not be multiplied by an as-filed price
- 4 field(s) zero-fill wrongly and are NULL-ruled

The residual risk is the one no gate here can close. Sharadar constructs Q4 by subtraction, so the identity is tautological wherever it holds and this window can detect a bad LEVEL but never a bad CONSTRUCTION. A 5-year DJIA-30 window is also not a 20-year S&P 500 window — the three CIK-cutover tickers (D19) are not in it at all, and that test is written but skipped. **Re-run this diagnostic against the full history on day one of the new entitlement**, before any of it reaches the cube.
