# Phase 4 — The merged `fundamentals_history` ✅

**Goal**: build the new `fundamentals_history` — Sharadar-first, SEC for the declared gap columns —
plus the gap check that proposes overrides and the register that records your decisions.

**Prerequisite**: phase 3 tests pass.
**Read first**: [README.md](README.md) — D14, D15, D18, D19, D22, D23, D24.
**Next**: [phase-5-docs-dod.md](phase-5-docs-dod.md)

---

## The table

`fundamentals_history`, PK `(ticker, as_of)`, publication-event grain — the same grain as before, so
`src/data_aggregate/` keeps reading the name it always read (README phase-0 group B).

**Columns** = the 60 contract names (phase 3) + `stockholdersEquityInclNci` + the ~23 Sharadar
extras + `regime` (SEC-owned, D18b) + the 2 keys.

**Absent by decision (D15)**: `publication_form`, `is_amendment`, `amended_fiscal_end`,
`amended_fields`, and any source column. These are **pure SEC reconciliation columns** — they stay
on `fundamentals_history_sec`, where the amendment grain is real and the validator uses them.
Sharadar has no amendment events, so carrying them here would produce four permanently-null columns
that lie about what the table knows.

### `as_of`

`as_of` ← Sharadar's **`date`** (the SEC filing date for AR dimensions). This is not an assumption:
measured on 14 tickers × 5 years, `ARQ.date` vs `fundamentals_history_sec.as_of` matched **279 of
280** (99.64%), with `reportperiod == fiscal_end` on **279/279** (100%). The single mismatch was a
GS 10-K/A on 2024-02-28 that Sharadar has no row for — an amendment, i.e. exactly the class D15
drops on purpose.

⚠ **Same-date collapse.** Sharadar's AR dimensions *"may include multiple observations in a
quarter"*, and there is **no form column** to resolve a same-day 10-K + 10-Q the way
`FORM_PRECEDENCE` does for SEC. Use the vendor's own documented rule: on a duplicate
`(ticker, date)`, keep the row with the **greatest `reportperiod`**. Log every collapse.

---

## Changes

### 1. `src/data_store/schema.py` and `sql/schema.sql` — risk zones, ask first

- [x] A new `fundamentals_history` `Table(...)`, PK `("ticker", "as_of")`, `date_col="as_of"`,
      `date_type_cols=("as_of", "fiscal_end")`, `freshness="quarterly"`, with `read_columns` set.
- [x] The matching `-- [aggregate]`/`-- [extract]` block in `sql/schema.sql`.
- [x] A comment on `fundamentals_reason_codes` recording that it points at
      `fundamentals_history_sec` and **not** at this table (D24).

### 2. `src/data_extract/utils/fundamentals_sharadar/merge_history.py`

- [x] `build_merged_history(context, tickers, *, full=False) -> None`

**Step order, and each step's rule:**

1. **Load Sharadar ARQ**, projected. Apply `translate()` and `build_ttm()` from phase 3.
2. **Collapse same-date rows** per the rule above.
3. **Load the SEC gap columns** from `fundamentals_history_sec`, projected to the **15** SEC-owned
   names (including `regime`) + `(ticker, as_of)`.
4. **Join.** `pd.merge_asof(direction="backward")` per ticker on `as_of` — *not* an exact join.
   Exact would drop the SEC block whenever the two dates differ by a day; backward gives the latest
   SEC snapshot **knowable at** the Sharadar publication date, which is the correct point-in-time
   semantics and matches the repo's existing `carry_latest_known` practice for `fundamentals_employees`.
   **Never join forward.** Cap the lookback so a stale SEC row cannot be carried indefinitely.
5. **Apply the override register** (below): for each `(ticker, field)` marked `sec`, replace the
   Sharadar value with the SEC one. This is the *only* place a Sharadar-owned column takes a SEC
   value, and it happens by explicit registered decision, never by a runtime heuristic.
6. **`employees`** comes from `fundamentals_employees`, forward-filled — it is annual 10-K prose and
   was never on the filing cadence.
7. **Assert the column contract** — build the frame from the declared list and assert its length,
   the way `build_history` already does. A silent column drift here is invisible downstream:
   `pit.py:62` returns an empty frame for an unknown column rather than raising.
8. **Hard-cast every value column to `float64` before the write** — same `ensure_table` trap as
   phase 1. ⚠ **`regime` is TEXT, not a value column**, as are `ticker`, `as_of` and `fiscal_end`.
   Exclude them explicitly from the cast rather than by a "looks numeric" heuristic; a cast that
   catches `regime` turns every regime label into `NaN` silently.

### 3. The gap check — `src/data_extract/utils/fundamentals_sharadar/gap_check.py`

- [x] `measure_gaps(context, tickers) -> pd.DataFrame`

Scope: the **14 overlapping tickers**, every field both sources carry, every shared `as_of`.
Threshold (D23): flag when **|Δ| / |sec| > 3% AND |Δ| > an absolute floor**. Put the floor in
`configs/configs.yml` next to the other fundamentals guards, not in code.

Report per `(ticker, field)`: `n_dates`, `n_flagged`, `median_pct_gap`, `min`/`max`, and — the
column that actually decides it — **`is_systematic`**: does the gap hold on *most* dates?
AXP was 6.6–8.1% low on **all 11** dates; that persistence is what distinguishes a **basis conflict**
from a one-off restatement. A gap on 1 of 11 dates is not an override candidate.

Expected findings, as a sanity check on the implementation:
- **JPM `totalRevenue`: no gap.** Sharadar matched the repo exactly on all 11 dates.
- **AXP `totalRevenue`: ~6.6–8.1%, systematic.** The gap is AXP's provision for credit losses.
- `stockholdersEquity`, `ppeNet`, `shortTermDebt`, `longTermDebt`, `accountsReceivable`,
  `accountsPayable`, `cash`, `ebitda` should all show gaps — those are the **known** basis forks
  from phase 3, not defects. The report must **name them as expected** so they do not drown the
  signal. Anything gapping that is *not* on that list is the real finding.

### 4. `configs/sharadar/sharadar_source_overrides.json` — D22

Machine-**proposed**, human-**approved**. The merge only reads it; it never decides at runtime.

```json
{
  "AXP": {
    "totalRevenue": {
      "source": "sec",
      "reason": "Sharadar takes AXP's own TotalRevenuesNetOfInterestExpenseAfterProvisionsForLosses caption (post-provision); the repo bans that basis by name. Gap = AXP's provision for credit losses.",
      "measured_gap_pct": 0.079,
      "n_dates": 11,
      "approved": "2026-XX-XX"
    }
  }
}
```

- [x] `--propose` writes candidate entries with `approved: null`.
- [x] The merge **ignores any entry with `approved: null`** and logs how many are awaiting decision.
      An unapproved proposal must never silently change data.

⚠ An override moves a field to a source with **54-ticker coverage**. For a ticker outside that
roster the override yields **NULL**, not a fallback to Sharadar — that is the point of field-block
precedence (D14). The report must state the coverage cost of each approved override.

### 5. `src/data_extract/transformers/step_extract_fundamentals_sharadar.py`

Append `build_merged_history` after the four fetchers — the same "never on its own schedule"
reasoning the SEC step already documents: a snapshot is only as fresh as the rows it reads.

### 6. `src/data_extract/cli.py`

- [x] `fundamentals-history-merged` — build only, with `-F/--full`.
- [x] `sharadar-gap-check` — measure and `--propose`.

---

## Tests

`tests/data_extract/test_sharadar_merge.py` — real data; each prints its conclusion.

- [x] `test_as_of_matches_sec` — reproduce the 279/280 measurement on the 14 overlapping tickers.
      Assert ≥ 99% and that every mismatch is SEC-only. Prints the mismatch list.
- [x] `test_column_contract` — the built frame's columns equal the declared list, in order, and the
      4 amendment columns are **absent**. Prints the diff both ways.
- [x] `test_no_amendment_columns` — explicit, because their absence is a *decision* (D15) and a
      future reader will otherwise assume it is an oversight.
- [x] `test_sec_block_is_asof_backward` — assert no SEC value is dated **after** its `as_of`.
      This is the no-leakage property; it is the single most important test in the phase.
- [x] `test_unapproved_override_is_ignored` — an entry with `approved: null` does not change data.
      Prints the count ignored.
- [x] `test_axp_revenue_gap_is_detected` — real: the gap check flags AXP `totalRevenue` as
      systematic and does **not** flag JPM's. Prints both.
- [x] `test_value_columns_are_float` — the `ensure_table` TEXT-column regression, again, on this table.
- [x] `test_cik_cutover_continuity` — APA / GOOGL / ETN, `skipif` not in roster (D19 is
      **unverifiable on DJIA-29**). Prints the skip reason.

---

## Verification

```bash
PY="$HOME/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe"
rtk "$PY" -m pytest tests/data_extract/test_sharadar_merge.py -v -s
rtk "$PY" -m src data_extract sharadar-gap-check -c ./configs --propose
rtk "$PY" -m src data_extract fundamentals-history-merged -c ./configs
```

```bash
MSYS_NO_PATHCONV=1 docker exec pea_db psql -U alexandre -d pea -c \
 "SELECT count(*) rows, count(distinct ticker) tickers, min(as_of), max(as_of),
         count(goodwill) sec_covered, count(\"totalRevenue\") shar_covered
  FROM fundamentals_history;"
```

- [x] 29 tickers, ~580 rows, `as_of` spanning the entitled window.
- [x] `shar_covered` ≈ all rows; `sec_covered` ≈ the 14 overlapping tickers' rows **only** — the
      expected, stated coverage asymmetry, not a bug.
- [x] No `text` columns among the value columns.
- [x] The gap check's proposals are written, and **you have adjudicated them** before the register
      is marked approved.
- [x] The no-leakage test passes.

---

## Rollback

The merged table is a fresh build from two read-only sources; drop and rebuild costs one CLI run and
no network. `fundamentals_history_sec` and `fundamentals_sharadar` are never written by this phase.

---

# ✅ DONE 2026-08-26 — what was actually implemented

**Verification**: `tests/data_extract/sharadar/test_sharadar_merge.py` — **18 passed, 3 skipped**
(the 3 skips are D19's CIK-cutover tickers, which the entitled roster does not contain).
Every test prints its conclusion; the numbers below are that output, not estimates.

## Files

| file | what it is |
|---|---|
| `src/data_extract/utils/fundamentals_sharadar/merge_history.py` | the override register, the 4 join steps, the contract assertion, the cast, `build_merged_history` |
| `src/data_extract/utils/fundamentals_sharadar/gap_check.py` | `measure_gaps`, `candidates`, `propose`, the markdown report |
| `configs/sharadar/sharadar_source_overrides.json` | the register — **29 proposals, 0 approved** |
| `configs/configs.yml` | `data_extract.sharadar_gap_floor` (3 classes) |
| `src/data_store/schema.py` | `fundamentals_history.read_columns` — the 91-column contract |
| `sql/schema.sql` | the `-- [extract] fundamentals_history` block (+147 lines, **purely additive**) |
| `src/constants/constants.py` | the phase-4 block: register filename, thresholds, tolerance, expected forks |
| `src/data_extract/cli.py` | `fundamentals-history-merged`, `sharadar-gap-check` |
| `src/data_extract/transformers/step_extract_fundamentals_sharadar.py` | step 5, after the four fetchers |

## The measurements the plan asked for, reproduced exactly

```
as_of agreement over 14 overlapping ticker(s): 279/280 = 99.64%
mismatches (1): [('GS', 2024-02-28, 'sec-only')]          <- the 10-K/A, exactly as predicted
SEC block joined on 279 of 598 row(s); lag min 0.0, median 0.0, max 0.0 days
rows carried from an EARLIER SEC filing: 0
AXP totalRevenue: 15/17 flagged, median 7.29%, max 8.28%  -> is_systematic TRUE
JPM totalRevenue:  0/17 flagged, median 0.00%             -> is_systematic FALSE
```

⚠ **The as-of tolerance never fired.** All 279 joins are exact same-day matches, so on this
roster the backward join is behaving as an exact one. That is the *reason* to keep it backward
rather than an argument for switching to an exact join: the one date the two sources disagree
on today is a SEC-only amendment, and the next one may not be.

## The table, live

```
598 rows | 30 tickers | 2021-08-27 .. 2026-08-10 | 91 columns
totalRevenue non-null      508  (= 598 - 3 per ticker cold start, unchanged from phase 3)
totalAssets  non-null      598  (30 tickers)
regime       non-null      279  (14 tickers -- the SEC overlap, i.e. the stated asymmetry)
employees    non-null      279  (14 tickers)
stockholdersEquityInclNci  101  (6 tickers)
TEXT columns among the 88 values: 0.  `regime` is TEXT, `ticker` is TEXT. Nothing else.
```

⚠ **The plan predicted "29 tickers, ~580 rows"; it is 30 and 598.** No same-date collapse
occurred at all on this roster — every `(ticker, date)` in the entitled ARQ set is already
unique. The collapse is still implemented and still tested (synthetically), because the vendor
documents the duplicate as possible and the roster is going to widen.

## Deviations from the plan, each with the reason

| # | plan said | implemented | why |
|---|---|---|---|
| 1 | load the **15** SEC-owned names from `fundamentals_history_sec` | **14 from there + `employees` from `fundamentals_employees`** | `employees` is not a column of that table. It was retired to its own table because the source is 10-K PROSE and one failed regex must not fail a 91-column snapshot. Asking for it in a projection RAISES, it does not return NULLs. |
| 2 | `apply_derived(frame, field_map)` | gained **`only=`** | `stockholdersEquityInclNci`'s NCI leg is SEC-owned, so it is uncomputable until the merge. A blanket re-run would also recompute a column somebody deliberately overrode. One evaluator, two callers, so the formulas cannot drift. |
| 3 | — | ⚠ **an override RIPPLES**, new | Moving `totalRevenue` to SEC leaves `grossMargins` / `operatingMargins` / `profitMargins` as ratios whose numerator and denominator now come from different sources. `rederive` recomputes exactly the derived columns that read a changed input, and `test_approved_override_takes_the_sec_value` asserts the ripple set is **neither wider nor narrower**. |
| 4 | the register carries an `_APPROVED` block like the other two | **per-ENTRY `approved` only** | The other two registers are all-or-nothing inputs a transform cannot run without, so an unsigned file must be fatal. This one is a decision LIST: a fresh proposal must be INERT, not fatal, or `--propose` breaks the pipeline until a human is available — which is an incentive to approve without reading. |
| 5 | `--propose` writes candidates | writes them through a **stable one-line-per-entry emitter** | `json.dumps(indent=2)` reformats every line it touches, so a 2-entry proposal reads as a whole-file diff and the review this register exists for becomes impossible. `test_reproposing_is_byte_identical` pins it; a second `--propose` reports **0 new**. |
| 6 | "an absolute floor" (one) | **three**, by field class | The 44 comparable fields are not all in dollars. A $1m bar is meaningless on `grossMargins` and 0.005 is meaningless on `totalAssets`. The class is read off `sharadar_field_map.json` (`op: ratio*`, `split_basis: count`), never from a name pattern. |
| 7 | — | ⚠ **`inherits_from`** column, new | `returnOnEquity` and `debtToEquity` read `stockholdersEquity`, a designed-in fork, so 6 flagged rows are the same decision one level down. Informational only — it does not reclassify, and those rows are still candidates. |
| 8 | — | **`__sec_as_of__`** carried through the join | The no-leakage property has to be *measurable*, not argued in a comment. Dropped before the write; a leak of any `__sec__*` column fails the contract assertion. |
| 9 | `tests/data_extract/test_sharadar_merge.py` | `tests/data_extract/`**`sharadar/`**`test_sharadar_merge.py` | Where phases 1-3 put theirs (phase-3 deviation 8). |
| 10 | 8 tests | **21** | The plan's 8 are all present. The extra 13 cover the tolerance, the collapse rule, the grain, the coverage asymmetry, the NCI re-derivation, the employees carry, the register's file format and its closed source vocabulary. |

## Four findings the gap check produced, and none of them is a Sharadar defect alone

**29 candidates over 9 tickers, all written unapproved.** The four worth naming:

1. ⚠ **A gap does not say which side is wrong, and the proposer cannot tell.** **MCD `depAmort`**
   gaps by ~480% because *the SEC layer* stores ~460M against MCD's own ~2.2bn annual D&A
   (4.63x-4.86x over 6 consecutive dates — the TTM-vs-quarter signature). **Sharadar is right
   there**, and approving that proposal would import the defect. This is now stated in the
   register's own `_README`.
2. ⚠ **`sbcomp` has NO zero rule, because it is not one of the 41 fields Sharadar documents as
   zero-filled** — so `apply_zero_rules` never inspects it. **JPM is 0 on 20 of 20 ARQ rows**
   while AAPL, AXP and GS are 0 on none. The merged table currently carries JPM
   `stockBasedComp = 0`, a fill wearing a value's clothes. Phase-2/3 scope, surfaced here.
3. ⚠ **The plan's expected-8 list does not cover the ROLL-UPS and RATIOS that read them.**
   `shortTermDebt` and `longTermDebt` are expected forks; `totalDebt` and `debtToEquity` are
   not on the list and so report as findings — JPM at 1160%, GS at 1081%. Deliberately left
   as findings rather than quietly added to the expected list: JPM's SEC `longTermDebt` is
   **NULL** while Sharadar's `debt` is 1.24tn, and both sides deserve a look.
4. **`stockholdersEquityInclNci` is 101/598 over 6 tickers**, exactly tracking
   `minorityInterest`'s non-null count. Working as designed — a `sum` propagates the NaN the
   SEC layer stores where a filer never tagged NCI — but a consumer expecting it to fall back
   to `stockholdersEquity` for a firm with no NCI will not get that.

## Verification block — results

- [x] 30 tickers, 598 rows, `as_of` spanning the entitled window (not the predicted 29/~580 —
      see above; nothing collapsed).
- [x] Sharadar-owned columns cover all 598 rows; the SEC block covers 279 rows over the 14
      overlapping tickers **only** — the expected, stated coverage asymmetry.
- [x] **No `text` columns among the value columns**, asserted both in-frame (87 float64) and
      against the live Postgres types.
- [x] The gap check's proposals are written — **29, every one `approved: null`**, and the merge
      logs all 29 as ignored. ⚠ **They are NOT adjudicated.** That is the one remaining human
      step in this phase, and finding 1 above is why it cannot be automated.
- [x] The no-leakage test passes, on a synthetic fixture that puts a SEC filing squarely in the
      future of two rows and asserts they stay empty.

---

## Amendment 2026-08-26 — the column NAMING convention (requested after the first build)

The 91 columns are unchanged in number, meaning and source. Only their **names** changed, and
the change makes the source readable off the name.

### 1. Every SEC-sourced column now ends in `_sec`

```
premiumsEarned_sec  netInterestIncome_sec  noninterestIncome_sec  netInvestmentIncome_sec
realizedInvestmentGains_sec  rentalIncome_sec  ppeGross_sec  accumulatedDepreciation_sec
goodwill_sec  intangiblesExGoodwill_sec  operatingLeaseLiability_sec
financeLeaseLiability_sec  minorityInterest_sec  employees_sec  regime_sec
```

Exactly 15, verified against `information_schema`. This is what replaces the `source` column
D15 refuses: the two producers have **different coverage** — Sharadar spans all 30 entitled
tickers, the SEC block only the 14 both sources have — so a bare `goodwill` beside a bare
`totalRevenue` said nothing about which producer left a NULL. Precedence stays per-COLUMN;
the name now says so.

⚠ The suffix is applied **after `rederive`**, because `stockholdersEquityInclNci` reads
`minorityInterest` under the name the field map declares. Renaming any earlier breaks the one
formula that spans both sources.

### 2. All 25 Sharadar extras are now repo camelCase

| vendor | repo | | vendor | repo |
|---|---|---|---|---|
| `cashneq` | `cashAndEquivalents` | | `dps` | `dividendsPerShare` |
| `accoci` | `accumulatedOtherComprehensiveIncome` | | `ncfi` | `investingCashFlow` |
| `assetsnc` | `nonCurrentAssets` | | `ncff` | `financingCashFlow` |
| `liabilitiesnc` | `nonCurrentLiabilities` | | `ncfdiv` | `dividendsPaid` |
| `investments` | `totalInvestments` | | `ncfcommon` | `equityIssuanceNet` |
| `investmentsnc` | `longTermInvestments` | | `ncfbus` | `businessAcquisitionsNet` |
| `taxassets` | `taxAssets` | | `ncfinv` | `investmentAcquisitionsNet` |
| `taxliabilities` | `taxLiabilities` | | `ncfdebt` | `debtIssuanceNet` |
| `deferredrev` | `deferredRevenue` | | `ncfx` | `exchangeRateEffect` |
| `deposits` | `deposits` *(unchanged)* | | `ncf` | `netCashFlow` |
| `opex` | `operatingExpenses` | | `netincnci` | `netIncomeToNci` |
| `netincdis` | `netIncomeDiscontinued` | | `netinccmn` | `netIncomeCommon` |
| `prefdivis` | `preferredDividends` | | | |

D16 said to keep Sharadar's own name where there is nothing to rename to; that had been read
as keeping its own **spelling**, which left `ncfx`, `prefdivis` and `accoci` in a table whose
other 63 columns are camelCase.

Three of these were judgement calls worth stating:

- **`investments` → `totalInvestments`.** A bare `investments` sitting next to
  `shortTermInvestments` reads as a different concept; it is the total.
- **`investmentsnc` → `longTermInvestments`**, not `nonCurrentInvestments`, so it mirrors the
  existing `shortTermInvestments` (← `investmentsc`).
- **`ncfi`/`ncff` → `investingCashFlow`/`financingCashFlow`**, matching the contract's own
  `operatingCashFlow` (← `ncfo`). The three cash-flow legs now read as one family.
- `deposits` was left alone: already a plain camelCase word, and unambiguous.

### How it is enforced

`sharadar_field_map.json`'s extras are now **keyed by the vendor column and emitted under a
required `to`**, so the file still reads as a map *from* `fundamentals_sharadar`. The loader
refuses a `to` that collides with a contract column (two sources would write one column and
the later one would win silently) and a `to` used twice (the dict would keep only the last).
The SF1-coverage assertion moved to `spec.source`, since the emitted name is no longer an SF1
column.

### Blast radius, measured

⚠ **All 15 SEC-owned names are referenced in `src/data_aggregate/` and/or
`configs/build_cube.yml`** — `goodwill` in 7 files, `rentalIncome` and `employees` in 9 each,
the rest in 3-6. Those readers now find nothing, and
**`pit.fundamentals_to_daily` returns an EMPTY FRAME for a missing column rather than raising**,
so the failure is silent by construction.

This is inside the plan's declared scope — *"Not a `data_aggregate` update… the cube is not
expected to stay green"* — and re-pointing those readers belongs to that separate task. It is
recorded here so the next person does not discover it as an empty feature panel.

No stale vendor names remain anywhere in `src/` outside `fundamentals_sharadar`'s own
`read_columns`, where the vendor spelling is correct and must stay.

### Verified after the rename

```
DROP TABLE + rebuild -> 598 rows | 30 tickers | 91 columns
_sec columns in information_schema           : 15  (exactly the 15 SEC-owned)
TEXT among the value columns                 :  0  (regime_sec and ticker only)
cashAndEquivalents non-null                  : 598
exchangeRateEffect non-null                  : 508
goodwill_sec / regime_sec non-null           : 244 / 279
tests/data_extract/sharadar                  : 50 passed, 4 skipped
```

Two phase-3 tests needed updating and both were the test, not the code:
`test_every_sf1_column_is_accounted_for` compared the *emitted* extras names against
`SHARADAR_SF1_COLUMNS` (now compares `spec.source`), and
`test_a_zero_denominator_is_null_not_infinity` built its fixture with the vendor name
`cashneq` (now `cashAndEquivalents`).

---

## Amendment 2026-08-26 (2) — the split de-adjustment ran at the wrong STAGE

Found while explaining the de-adjustment mechanism, not by a check. **Neither ingredient was
wrong; the order was.**

### The defect

`deadjust_splits` ran inside `translate()`, i.e. per DISCRETE QUARTER. Sharadar stores the
whole share block on **one** basis (today's), so `as_filed = adjusted / F` where `F` is the
factor at that row's filing date. De-adjusting each quarter first put **two bases inside one
four-quarter window** whenever the window straddled a split, and the mean of those is on no
basis at all:

```
NVDA window ending at the 2024-08-28 filing (split 2024-06-10 x10):
  stored          : 24.9bn  24.9bn  24.9bn  24.9bn     <- one basis
  ÷ F per quarter :  ÷10     ÷10     ÷10      ÷1
  mean            : 8.08bn                             <- on NO basis
```

`epsDiluted = netIncome_TTM / dilutedShares` inherited it. Measured, before → after:

| | `dilutedShares` before | after | `epsDiluted` before | after | as-filed |
|---|---|---|---|---|---|
| AMZN 2022-07-29 | 2,929,250,000 | **10,253,750,000** | 3.96 | **1.13** | ~1.14 |
| NVDA 2024-08-28 | 8,081,250,000 | **24,904,500,000** | 6.56 | **2.13** | ~2.16 |
| WMT 2024-03-15 | 4,053,750,000 | **8,108,750,000** | 4.01 | **2.01** | ~2.02 |

Worst error **3.48x** (AMZN), decaying over the next two quarters and self-correcting on the
4th. **9 rows** on this roster — 3 per split, 3 splits.

### The fix

Aggregate first, de-adjust the RESULT once with `F` at the row's own filing date. Every
quarter in the window then shares the vendor's single basis, so the mean is coherent, and
dividing by `F` maps it to the basis in force at that date — which is exactly what a filer
does when it restates comparatives.

Where a window does **not** straddle a split the two orders are algebraically identical
(`mean(x)/F ≡ mean(x/F)`), which is why only 9 of 59 rows moved and
`sharesOutstanding` — an INSTANT, so its window is one quarter — is **identical on 59/59**.

`deadjust_splits` itself is unchanged; it already keys on `frame["date"]` and each column's
`split_basis`, and the TTM frame carries `date`. Only the call site moved:

- `translate()` no longer de-adjusts, and **no longer takes `actions=`** — deleted rather
  than deprecated, so a caller still passing it fails loudly instead of being ignored.
- `build_ttm()` takes `actions=` and `report=`, and calls `deadjust_splits` **between** the
  rolling aggregation and the closing `apply_derived` — after the window so the window is on
  one basis, before the formulas so `epsDiluted` reads a corrected denominator.
- Three call sites pass `actions` one level down: `merge_history.build_frame`,
  `gap_check.sharadar_history`, and the phase-3 `ttm` fixture.

`dividendsPerShare` gets the same correction for free — a `duration` **sum** has the identical
straddle problem, fixed by `× F(row.date)`.

### Tests

- `test_a_ttm_window_straddling_a_split_stays_on_one_basis` — **synthetic**, because the
  property is about an order of operations and real data can only show that two orders differ,
  not which is right. Sharadar stores 200 on every quarter, the split is 2-for-1, so pre-split
  rows must be 100 and post-split rows 200, and **nothing may ever be 125** — which is what
  the old order produced.
- `test_post_split_share_counts_are_not_a_hybrid_basis` — **real**, on
  `dilutedShares / sharesOutstanding`. A weighted average and a period-end count differ by a
  few percent, never by a factor of three. Measured range after the fix: **0.994 – 1.275**.
  The 1.27 tail is V (Visa) and is a MULTI-CLASS effect, not a split one — `sharesbas` is the
  Class-A cover-page count while `shareswadil` is the as-converted diluted total.
- `test_share_block_is_deadjusted_against_the_sec_cover_page` moved from the `translated`
  fixture to `ttm`. Its assertion is unchanged and still passes: `sharesOutstanding` is an
  instant, so its value is identical at either stage.

`split de-adjusted cells` fell from 89 to **59** — not a loss. The MEAN and SUM columns are
structurally NULL for each ticker's first three quarters, and de-adjustment only touches
non-null cells, so the same correction now lands on fewer cells because it lands *after* the
window contract has applied.

### ⚠ A caveat that applies either way

Both orders assume the **entire stored series shares one basis**. If a split occurs between
incremental fetches, already-stored rows sit on the old basis while new ones use the new one,
and `F` is wrong for the old rows. Pre-existing, not introduced here — but it means **any new
split requires a `fundamentals-sharadar --full` refetch** before the next merge. Belongs in
the phase-5 docs.

### ⚠ Unrelated, found in the same run: THE ENTITLEMENT HAS WIDENED

`test_not_entitled_is_not_a_retry_storm` began failing — not from this change (the file does
not import `translate`, `build_ttm` or `deadjust_splits`). Its probe ticker **ADBE, measured
as a 403 "Exceeds free tier" when phase 1 was written, now returns 23 rows.**

This is not a test to "fix" — it is a signal that the subscription changed, and it touches
several things this plan deferred:

- the roster may be far wider than 30, so `fundamentals-sharadar` is worth re-running;
- **D19's CIK-cutover continuity** (APA / GOOGL / ETN) may now be verifiable rather than
  skipped;
- the phase-2 acceptance gates (D28) were measured to decide whether to *buy* the Full tier.

Left alone deliberately, pending a decision.
