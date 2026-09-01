# Implementation Plan: Sharadar TTM window defects — duplicate quarters + the 45-day gate

**Date Created**: 2026-08-31
**Planning Phase**: 2 of 3 (FIC Workflow)
**Based on Research**: this session's live diagnosis (DB + Sharadar API + SEC EDGAR); no prior research doc
**Next Phase**: Implementation (`/implement`)

## Overview

`build_ttm` is silently refusing **1,285 trailing-twelve rows** it should publish, and publishing
**353 duplicate `as_of` rows** it should not. Two independent root causes, both in
`src/data_extract/utils/fundamentals_sharadar/build_ttm.py`, both fixable transform-side with no
schema change and no refetch:

* **Defect 2 — duplicate quarters.** Sharadar emits one ARQ row *per filing*, amendments
  included. A repeated quarter fails `_window_is_whole` exactly as a missing one does, so
  **543 duplicate groups over 316 tickers** each null three trailing twelves.
* **Defect 3 — the 45-day sanity gate is miscalibrated.** `_normalisation_is_sane` drops
  **239 ARQ rows over 4 tickers**, leaving **AVGO with no fundamentals at all** and KR/AZO/COST
  at 100%/100%/97.6% NULL revenue.

A third defect (AAPL's absent 2006-Q3, a late-filed 10-Q Sharadar never emitted an AR\* row for)
is diagnosed but **explicitly out of scope** — see *Out of Scope*.

## Current State Analysis

### The code

One file, three functions, called from two production sites:

| Location | Role |
|---|---|
| `build_ttm.py:65-76` | `_window_is_whole` — ordinal contiguity on `calendardate` |
| `build_ttm.py:79-90` | `_normalisation_is_sane` — the 45-day absolute-drift gate |
| `build_ttm.py:129-137` | `build_ttm` — sorts, applies the gate, computes `whole` |
| `merge_history.py:413` | production caller |
| `gap_check.py:124` | diagnostic caller |

### Defect 2 — measured

Sharadar's ARQ grain is one row per FILING. EDGAR confirms all three named pairs are
original + amendment on the same `reportDate`:

| ticker | original | amendment | reportDate |
|---|---|---|---|
| IBM | 10-Q 2004-10-28 | 10-Q/A 2004-11-01 | 2004-09-30 |
| KO | 10-K 2002-03-11 | 10-K/A 2002-03-13 | 2001-12-31 |
| GOOGL | 10-Q 2007-05-09 | 10-Q/A 2007-05-10 | 2007-03-31 |

With a duplicated ordinal, `ordinal - ordinal.shift(3) == 3` fails for exactly 3 rows — the
same signature as a gap. Table-wide census of `(ticker, calendardate)` duplicate groups in ARQ:

| class | groups | extra rows | tickers | meaning |
|---|---|---|---|---|
| **B** same `reportperiod`, identical values | 439 | 485 | 287 | pure re-publication |
| **C** same `reportperiod`, values **differ** | 97 | 106 | 79 | genuine restatement |
| **A** **different** `reportperiod`, one `calendardate` | 7 | 8 | 4 | two REAL quarters colliding on the vendor's normalisation (BBY, GPN, OKE, KR) |
| total | **543** | **599** | **316** | |

Class A is why the dedup key must be `reportperiod`, not `calendardate`.

### Defect 3 — measured

`_normalisation_is_sane` rejects any row whose `calendardate` sits >45 days from its own
`reportperiod`. That is not a data error for off-calendar filers — it is Sharadar's convention:

```
AVGO  reportperiod 2024-02-04 -> calendardate 2024-03-31   drift 56   correct, REJECTED
COST  reportperiod 2024-05-12 -> calendardate 2024-06-30   drift 49   correct, REJECTED
COST  reportperiod 2024-09-01 -> calendardate 2024-09-30   drift 29   correct, accepted
```

Consequence:

| ticker | ARQ rows | dropped by gate | `fundamentals_history` rows | % NULL revenue |
|---|---|---|---|---|
| **AVGO** | 69 | **69** | **0** | absent entirely |
| KR | 128 | 78 | 50 | **100%** |
| AZO | 121 | 52 | 69 | **100%** |
| COST | 125 | 40 | 85 | **97.6%** |

`build_ttm`'s own docstring predicted this: the *0 missing / 0 duplicate* measurement was taken
on 30 tickers with the note "the test is here because the roster is going to widen." It widened
to ~500, and both claims are now false (180 gap events, 543 duplicate groups).

### Rules evaluated for the replacement

| candidate | rows wrongly rejected | verdict |
|---|---|---|
| 45-day absolute drift (today) | 239 | miscalibrated |
| `calendardate` == end of quarter *containing* `reportperiod` | **6,083** | **measurably wrong** — Sharadar maps WMT 1995-07-31 *backward* to 1995-06-30 and AVGO 2024-02-04 *forward* to 2024-03-31; not a containment rule at all |
| per-ticker median drift ±45d | 59 | false-positives BBY's 2012 fiscal-calendar change |
| **drop the gate; guard on `reportperiod` span** | **0** | **chosen** |
| `fiscalperiod` as sequencing key instead of `calendardate` | — | rejected: 187 gaps / 42 duplicates vs 180 / 7 |

### Measured size of the fix

Simulated in SQL against the live table:

| variant | output rows | whole TTM |
|---|---|---|
| status quo (gate on, no dedup) | 51,608 | 48,215 |
| candidate (dedup + span guard) | 51,255 | **49,500** |

**+1,285 recovered TTM rows, −353 spurious duplicate `as_of` rows.** NULL rate on
`totalRevenue` falls from ~6.6% to ~3.5%.

Per-ticker, including regression checks on the class-A tickers:

| ticker | whole today | whole after | recovered |
|---|---|---|---|
| COST | 2 | 121 | **+119** |
| AZO | 0 | 113 | **+113** |
| KR | 0 | 112 | **+112** |
| AVGO | 0 | 66 | **+66** |
| GPN | 95 | 99 | +4 |
| GOOGL | 83 | 87 | +4 |
| IBM | 119 | 121 | +2 |
| KO | 119 | 121 | +2 |
| AAPL | 117 | 117 | 0 (defect 1 out of scope) |
| BBY | 118 | 118 | 0 (**no regression**, class A) |
| OKE | 119 | 119 | 0 (**no regression**, class A) |

## Desired End State

* `_normalisation_is_sane` is gone; no row is dropped for an absolute drift.
* One ARQ row per `(ticker, reportperiod)` reaches the window maths — the earliest filing.
* `_window_is_whole` asserts contiguity **and** that the window spans a real twelve months,
  measured on the filer's own `reportperiod`.
* AVGO, KR, AZO, COST have a populated TTM line; GOOGL/IBM/KO lose their mid-history holes.
* `diagnostics.gate_completeness` reports 0 duplicate normalised quarters.
* No schema change, no refetch, no change to `fundamentals_sharadar` contents.

## Out of Scope

* **Defect 1 — AAPL's absent 2006-Q3.** Recoverable exactly (`ARY` FY2006 minus the three known
  ARQ quarters pins revenue to 4,370M and netinc to 472M, both matching Sharadar's own MRQ row
  to the dollar; `ARY` carries the true publication date 2006-12-29, so it is PIT-clean). Left
  out because it is the only option that **synthesises a row Sharadar never published**, and it
  buys 3 rows on 1 ticker. Revisit separately if wanted; the ARY-minus-ARQ identity is recorded
  here so the work is not re-derived.
* Class-A collisions (7 groups, 4 tickers). They degrade gracefully — the ordinal check nulls
  rather than splices, and BBY/OKE show 0 regression. Not worth a special case.
* Adopting Sharadar's `ART` (still refused, D17).
* `MRQ`/`MRT` ingestion — their `date` is the period end used as a placeholder, which is a
  leakage trap, not a fix.
* Backfilling the SEC path (`fundamentals_history_sec` starts 2009 for AAPL).
* The survivorship-bias / `sharadar_sp500` work (D27).

## Implementation Approach

### Phase 1: Baseline snapshot ✅

**Goal**: make the recovery provable rather than asserted. Nothing here changes behaviour.

**Changes**:

1. Baseline note (scratchpad or `reports/planning/active-tasks/2026-08-31-sharadar-baseline.md`):
   - [x] Record, from the live DB *before* any code change:
     - total `fundamentals_history` rows and NULL-`totalRevenue` count (expect 51,608 / 3,593)
     - the 11-ticker table above, `whole today` column
     - `AVGO` row count in `fundamentals_history` (expect 0)
   - [x] Save the exact SQL used, so the after-measurement is the same query.

**Verification**:
- [x] Baseline numbers match the figures in *Current State Analysis*. If they do not, the DB has
      moved since 2026-08-31 and every target number in this plan must be re-measured first.

---

### Phase 2: De-duplicate the quarter grain ✅

**Goal**: one ARQ row per `(ticker, reportperiod)` reaches the window maths. Fixes GOOGL, IBM,
KO and 313 other tickers.

**Changes**:

1. `src/data_extract/utils/fundamentals_sharadar/build_ttm.py`:
   - [x] Add `_one_row_per_quarter`, keyed on `reportperiod` and keeping the earliest `date`:

   ```python
   def _one_row_per_quarter(frame: pd.DataFrame) -> pd.DataFrame:
       """One ARQ row per (ticker, reportperiod) -- the EARLIEST filing.

       Sharadar's ARQ grain is one row per FILING, amendments included. IBM's quarter ended
       2004-09-30 arrives twice, as the 10-Q of 2004-10-28 and the 10-Q/A of 2004-11-01;
       verified on EDGAR for KO (10-K/10-K/A, 2002-03-11/13) and GOOGL (10-Q/10-Q/A,
       2007-05-09/10) too. A REPEATED quarter fails `_window_is_whole` exactly as a MISSING one
       does, so 543 duplicate groups over 316 tickers were nulling three trailing twelves apiece.

       The EARLIEST filing wins because it is what the market knew on the day: AR* is
       as-reported and immutable, and taking the amendment would file a later restatement under
       an earlier publication date. 439 of the 543 groups carry identical values, so the choice
       only bites on the 97 that were genuinely restated.

       Keyed on `reportperiod`, NOT `calendardate`: 7 groups over 4 tickers (BBY, GPN, OKE, KR)
       are two REAL quarters whose fiscal ends normalise onto one calendar quarter, and keying
       on the normalisation would DELETE one of them.
       """
       return (frame.sort_values(["ticker", "reportperiod", "date"])
                    .drop_duplicates(["ticker", "reportperiod"], keep="first"))
   ```

   - [x] In `build_ttm`, replace the sort at line 129 so dedup runs before anything else and the
         sequence follows the filer's own period ends (deterministic where two rows share a
         `calendardate`):

   ```python
   out = (_one_row_per_quarter(frame)
          .sort_values(["ticker", "reportperiod"])
          .reset_index(drop=True))
   ```

   - [x] Log the drop count at INFO when non-zero — a silent 353-row filter is the failure mode
         this repo keeps re-learning.

2. `tests/data_extract/sharadar/test_sharadar_field_map.py`:
   - [x] `test_amended_filing_does_not_break_the_window` — synthetic 5 quarters, one duplicated
         under a second `date` with identical values; assert the TTM at the 4th quarter is the
         sum of four and that no row is nulled.
   - [x] `test_dedup_keeps_the_earliest_filing` — duplicate with DIFFERENT values; assert the
         retained value is the earlier filing's.
   - [x] `test_two_real_quarters_on_one_calendardate_both_survive` — class A: two rows, same
         `calendardate`, different `reportperiod`; assert both rows survive dedup (this is the
         guard against deleting a real quarter).
   - [x] Each test prints a sanity-check conclusion (AGENTS.md).

**Verification**:
- [x] `"$PY" -m pytest tests/data_extract/sharadar/test_sharadar_field_map.py -v -s`
- [x] Existing gap test at `test_sharadar_field_map.py:500-501` still passes — a genuine gap must
      still null.
- [x] Ad-hoc: run `build_ttm` on live GOOGL/IBM/KO ARQ; assert their mid-history NULL runs are gone.

---

### Phase 3: Retire the 45-day gate, add the span guard ✅

**Goal**: stop discarding correct rows from off-calendar filers. Recovers AVGO, KR, AZO, COST.

**Changes**:

1. `src/data_extract/utils/fundamentals_sharadar/build_ttm.py`:
   - [x] **Delete** `_normalisation_is_sane` (lines 79-90) and its call site (lines 130-135).
   - [x] **Remove** the now-unused import of `TTM_STALENESS_DAYS` (line 49). Confirm no other
         symbol from `build_history` is used here.
   - [x] Add the band constant. It stays module-local, not in `constants.py` — one consumer
         (the *Constants placement rule*: `constants.py` is for 2+ non-test `src` consumers):

   ```python
   #: A trailing twelve must span three quarter-steps of the FILER'S OWN calendar. Measured over
   #: the 49,500 windows the ordinal check accepts: min 240 days, median 274 (= 39 weeks, exactly
   #: 3 x 13), max 315. The band is the observed envelope plus a margin, so it is a TRIPWIRE
   #: against a spliced window rather than a filter on today's data -- it rejects nothing now.
   #:
   #: It replaces a 45-day cap on `calendardate` vs `reportperiod`, which measured the wrong
   #: thing: Sharadar's normalisation legitimately drifts 56-59 days for a filer whose quarters
   #: end early in the calendar quarter (AVGO), and that cap silently deleted 239 correct rows
   #: over 4 tickers -- every one of AVGO's, leaving it absent from `fundamentals_history`.
   TTM_SPAN_DAYS: tuple[int, int] = (240, 320)
   ```

   - [x] Rewrite `_window_is_whole` to assert both contiguity and span:

   ```python
   def _window_is_whole(frame: pd.DataFrame) -> pd.Series:
       """Is each row the end of FOUR CONSECUTIVE quarters spanning a real twelve months?

       Two independent tests, and both are needed:

         * `ordinal - ordinal.shift(3) == 3` on `calendardate` -- the vendor's own quarter
           LABELS are contiguous. A gap, a duplicate quarter or a short history all fail it, and
           all three must: a "TTM" spliced across a missing quarter is a 15-month number wearing
           a 12-month label.
         * `reportperiod - reportperiod.shift(3)` inside `TTM_SPAN_DAYS` -- the ECONOMICS really
           do span a year, measured on the FILER'S OWN period ends. This trusts no vendor
           normalisation, so it survives both a 52/53-week calendar and a mid-history fiscal
           calendar change, which every drift-based test measured here does not.

       Both shifts are per TICKER, so one issuer's first quarters can never borrow the previous
       issuer's last ones.
       """
       ordinals = quarter_ordinal(frame["calendardate"])
       by_ticker = frame["ticker"]
       contiguous = (ordinals - ordinals.groupby(by_ticker, sort=False).shift(TTM_QUARTERS - 1)
                     ) == (TTM_QUARTERS - 1)
       reported = pd.to_datetime(frame["reportperiod"], errors="coerce")
       span = (reported - reported.groupby(by_ticker, sort=False).shift(TTM_QUARTERS - 1)).dt.days
       low, high = TTM_SPAN_DAYS
       return contiguous & span.between(low, high)
   ```

   - [x] In `build_ttm`, log at WARNING when `contiguous` holds but the span check fails — that
         is the tripwire firing, and it must not be silent. Behaviour is **null the window**
         (never raise, never publish): the repo's NULL-over-guess contract.
   - [x] Rewrite the module docstring:
     - the "45-day cap" paragraph (lines 34-40) → the span guard and why drift was the wrong test
     - the "0 missing and 0 duplicate over all 30 tickers" claim (lines 25-32) → the measured
       reality (180 gap events; 543 duplicate groups over 316 tickers; the roster did widen)
     - keep the measurements, drop the chronology (per the repo docstring rule)

**Verification**:
- [x] `"$PY" -m pytest tests/data_extract/sharadar/ -v -s`
- [x] New test `test_span_guard_nulls_a_spliced_window` — synthetic 4 contiguous ordinals whose
      `reportperiod`s span ~15 months; assert NULL and assert the warning is logged.
- [x] New real-data test `test_off_calendar_filers_have_a_ttm_line` over AVGO, KR, AZO, COST:
      assert each yields >60 whole TTM rows and print the count per ticker.
- [x] Grep confirms `TTM_STALENESS_DAYS` is no longer imported in this module and `build_history`
      is otherwise untouched.

---

### Phase 4: Rebuild and prove the delta ✅

**Goal**: `fundamentals_history` reflects the fix, and the recovery is measured, not assumed.

**Changes**:

1. Rebuild the merged table (Sharadar block only; the SEC block is unchanged):
   ```bash
   rtk "$PY" -m src data_extract fundamentals-sharadar -c ./configs -F
   ```
   - [x] `-F` so `build_merged_history` DELETEs before it rebuilds — a partial overlay would
         leave the old NULL rows in place.
   - [x] Note: this re-runs the four fetchers too. For a fetch-free rebuild, call
         `build_merged_history(context, tickers=..., full=True)` directly instead.

2. Re-run the diagnostic gate:
   - [x] `diagnostics.gate_completeness` — assert `n_rows == n_quarters` for every ticker
         (0 duplicate normalised quarters). This gate already counts duplicates as a distinct
         defect, so it is the natural instrument.
   - [x] `gap_check.run_gap_check` — confirm the Sharadar-vs-SEC gap report does not regress.

3. `tests/data_extract/sharadar/test_sharadar_ttm_roster.py` (**new file**) — the ROSTER-WIDE
   regression test. This is the one that covers all 316 affected tickers rather than the 11 named
   ones, and it must FAIL on the pre-fix code:

   - [x] `test_no_window_is_nulled_except_by_a_genuine_missing_quarter` — over the WHOLE live ARQ
         table, recompute wholeness INDEPENDENTLY by set membership and assert it agrees with
         `build_ttm`'s own answer for every ticker:

   ```python
   # Independent predicate: a row at quarter ordinal q is whole iff the four ordinals
   # {q-3, q-2, q-1, q} are all PRESENT in that ticker's observed set. Set membership cannot
   # see a duplicate (a repeated quarter does not change the set) and cannot see an absolute
   # drift, so it is exactly the two defects' blind spot -- which is what makes it a valid
   # cross-check rather than a restatement of `_window_is_whole`'s shift arithmetic.
   observed = set(quarter_ordinal(group["calendardate"]).dropna().astype(int))
   expected_whole = {q for q in observed if {q - 3, q - 2, q - 1, q} <= observed}
   ```

     Assert `set(built.loc[built[FIELD].notna(), "_ordinal"]) == expected_whole` per ticker, for a
     duration field with no zero-rule nulls (pick one and state why). Report every disagreeing
     ticker by name, not just a count — a bare `assert len(bad) == 0` gives the next reader
     nothing to work with.

   - [x] **Pre-fix falsification check**: record in the test docstring that this assertion fails
         for **316 tickers** before the fix (543 duplicate groups) and for **AVGO/KR/AZO/COST**
         (the gate). A regression test nobody has watched fail is not known to test anything.

   - [x] **Class-A carve-out, pinned not hidden**: BBY, GPN, OKE, KR have two REAL quarters
         normalising onto one `calendardate`, so the observed SET collapses them and the predicate
         claims whole where the shift correctly refuses. Carve these 4 out by name with the reason
         inline, and assert the carve-out list is EXACTLY those 4 — so a 5th such ticker appearing
         later fails the test rather than being silently absorbed.

   - [x] `test_ttm_coverage_does_not_regress` — aggregate floors, so a future change that quietly
         drops rows is caught even if the set predicate still agrees:
     - total whole TTM rows `>= 49,400` (measured 49,500; margin for roster drift)
     - tickers with 0 whole TTM rows == 0 (this is the assertion AVGO fails today)
     - duplicate `(ticker, reportperiod)` pairs reaching the window maths == 0
     - rows with duplicate `(ticker, calendardate)` surviving dedup == 8, over exactly the 4
       class-A tickers
   - [x] Both tests print a sanity-check conclusion: per-ticker recovery table for the named 11,
         plus the roster totals (AGENTS.md).

   **Note on cost**: this reads the full ARQ table (~51.8k rows, ~90 columns). Project the
   columns and run `build_ttm` once, shared across both tests via a module-scoped fixture — do
   not rebuild per ticker.

**Verification** (re-run Phase 1's exact SQL):
- [x] `fundamentals_history` NULL-`totalRevenue` count falls ~3,593 → ~2,300 (−1,285)
- [x] total rows falls 51,608 → ~51,255 (−353 duplicate `as_of` rows)
- [x] AVGO: 0 → ~66 rows with revenue
- [x] KR 0→112, AZO 0→113, COST 2→121
- [x] GOOGL 83→87, IBM 119→121, KO 119→121
- [x] BBY, OKE, GPN, AAPL: **no regression** (0, 0, +4, 0)
- [x] Spot-check one recovered row against the filing: AVGO's TTM revenue at a known `as_of` must
      equal the sum of its four discrete quarters, and the split de-adjustment must still be
      correct (it runs after aggregation — `deadjust_splits`).
- [x] Aggregate fingerprint: the cube reads `fundamentals_history`, so the baseline WILL move.
      Ask before updating it (AGENTS.md risk zone).

---

### Phase 5: Docs, memory and the DoD report 🔄

**Goal**: the next reader does not re-derive any of this.

**Changes**:
1. - [x] `docs/data_sources.md` — Sharadar quirks: ARQ is one row **per filing** (amendments
        included), and `calendardate` is a per-ticker fiscal offset, not a bounded normalisation.
2. - [x] `docs/database.md` — refresh the `fundamentals_history` coverage figures.
3. - [x] Correct the stored memory `sharadar-sec-merge-key-measured`, which says Sharadar
        "drops amendments entirely". It does not — 543 duplicate groups are original+amendment
        pairs, EDGAR-confirmed on IBM/KO/GOOGL.
4. - [x] New memory: `calendardate` is a per-ticker fiscal offset (AVGO +56-59d, WMT −31d);
        absolute-drift and containment tests both fail on it; sequence on ordinals, validate on
        `reportperiod` span.
5. - [x] DoD report via the `dod-data-report` skill →
        `reports/<YYYY-MM-DD>/sharadar-ttm-window-defects__DATA.md`.

**Verification**:
- [x] `rtk "$PY" -m pytest tests/ -q` — full suite green.
- [x] AGENTS.md still ≤70 lines (no change expected).

## Testing Strategy

### Unit (synthetic, known-truth — parsing/window maths)
- duplicate quarter, identical values → window whole
- duplicate quarter, differing values → earliest retained
- class A (two `reportperiod`s, one `calendardate`) → both rows survive dedup
- genuine gap → still nulls (regression guard on existing behaviour)
- contiguous ordinals, 15-month span → nulled + warning

### Roster-wide regression (the primary proof — `test_sharadar_ttm_roster.py`)
The named-ticker checks below are spot checks; they would pass even if the fix regressed the
other 305 tickers. The roster test is what actually covers all 316:
- wholeness recomputed by INDEPENDENT set membership must agree with `build_ttm` for every
  ticker, with disagreeing tickers named
- known to FAIL pre-fix on 316 tickers (duplicates) + AVGO/KR/AZO/COST (the gate)
- class-A carve-out is exactly {BBY, GPN, OKE, KR} — a 5th such ticker fails the test
- floors: total whole `>= 49,400`; tickers with 0 whole rows == 0; duplicate
  `(ticker, reportperiod)` == 0; surviving duplicate `(ticker, calendardate)` == 8

### Real-data spot checks (economic — per AGENTS.md, real data)
- AVGO, KR, AZO, COST each yield >60 whole TTM rows
- GOOGL/IBM/KO mid-history NULL runs are gone
- BBY/OKE unchanged
- AVGO TTM revenue == sum of its four discrete quarters at a known `as_of`

### Manual
- `diagnostics.gate_completeness` shows 0 duplicate normalised quarters
- Phase 1 SQL re-run reproduces every target number

## Risk Mitigation

1. **Issue**: `-F` re-runs the four fetchers; a Sharadar outage mid-rebuild leaves a partial table.
   **Mitigation**: call `build_merged_history(..., full=True)` directly to rebuild without
   refetching; `fundamentals_sharadar` is untouched by this plan, so the input is already local.

2. **Issue**: the sort key changes from `calendardate` to `reportperiod`, altering output row order.
   **Mitigation**: `merge_history` joins as-of on `date`, so order is not load-bearing — but assert
   it in Phase 4 by diffing a stable ticker (AAPL) row-for-row before/after.

3. **Issue**: keeping the earliest filing discards 97 genuine restatements.
   **Mitigation**: intended — it is the AR\*-immutable, no-leakage contract. If the restatement
   magnitudes are later wanted, they are recoverable from `fundamentals_sharadar`, which this plan
   leaves lossless. That is the whole reason the fix is transform-side.

4. **Issue**: the span band (240-320) rejects nothing today, so it is untested against real data.
   **Mitigation**: cover it with the synthetic 15-month test; log-on-fire means a future trip is
   visible rather than silent.

5. **Issue**: the cube's aggregate fingerprint baseline will move.
   **Mitigation**: expected — 1,285 new rows. Ask before updating the baseline (risk zone).

### Rollback
Single-file, transform-side, no schema or vendor-table change: `git revert` the `build_ttm.py`
commit and re-run Phase 4's rebuild. `fundamentals_sharadar` never changes, so the old output is
exactly reproducible.

## Dependencies
- No new libraries.
- Touches only `build_ttm.py` + tests + docs.
- `build_history.TTM_STALENESS_DAYS` loses one importer; the constant itself stays (SEC path uses it).
- Downstream: `merge_history`, `gap_check`, `diagnostics`, and the cube (via `fundamentals_history`).
- **No** risk-zone edits (`constants/`, `data_store/`, `sql/schema.sql`, `configs/`) — except the
  fingerprint baseline in Phase 4, which needs explicit approval.

## Success Criteria
- [x] +1,285 whole TTM rows; −353 duplicate `as_of` rows
- [x] AVGO present in `fundamentals_history` (0 → ~66 rows)
- [x] KR, AZO no longer 100% NULL; COST no longer 97.6% NULL
- [x] GOOGL/IBM/KO mid-history holes closed
- [x] BBY/OKE/AAPL unregressed
- [x] `gate_completeness` reports 0 duplicate normalised quarters
- [x] **Roster-wide regression test passes over all ~500 tickers**, and was WATCHED TO FAIL on the
      pre-fix code for 316 tickers + AVGO/KR/AZO/COST
- [x] No ticker has 0 whole TTM rows
- [x] Full suite green; every new test prints a sanity-check conclusion
- [x] `_normalisation_is_sane` deleted, not merely widened

## Estimated Effort
- Phase 1 (baseline): 0.5h
- Phase 2 (dedup + tests): 2h
- Phase 3 (gate + span guard + tests): 2.5h
- Phase 4 (roster regression test + rebuild + prove): 3h + rebuild wall time
- Phase 5 (docs/memory/DoD): 1h
- **Total: ~9h** plus rebuild

Sequencing note: write the roster test BEFORE Phase 2/3 land if you want to watch it fail — that
is the only way to know it tests anything. Stash the fix, run the test, confirm 316 tickers +
AVGO/KR/AZO/COST fail, then unstash. (`stash@{0}` is the user's long-lived stash — use a worktree
at HEAD instead of stashing.)

## Notes for Implementation
- Do **not** widen the 45-day cap to ~62 days. It is the same miscalibrated test with a new magic
  number and it breaks on the next off-calendar filer. Delete it.
- Do **not** use a containment test (`calendardate` == end of the quarter containing
  `reportperiod`). Measured: **6,083 false rejects** across HPQ, WMT, DE, TGT, HD, CSCO and ~100
  other filers. Sharadar maps WMT 1995-07-31 *backward* to 1995-06-30 and AVGO 2024-02-04
  *forward* to 2024-03-31 — it is not containment.
- Do **not** switch the sequencing key to `fiscalperiod`: 187 gaps / 42 duplicates vs 180 / 7.
- Dedup on `reportperiod`, never `calendardate` — the class-A carve-out depends on it.
- Split de-adjustment must still run AFTER aggregation and BEFORE `apply_derived`
  (`build_ttm.py:186-190`). Do not reorder while editing nearby.
- `rolling(4)` without `min_periods` already nulls a window holding a NaN. That is the contract;
  leave it.
- Keep the measurement, drop the chronology, in every docstring touched.

---

## Implementation results (2026-08-31)

Measured, not predicted. Baseline SQL and its re-run: `2026-08-31-sharadar-baseline.md`.

### The delta on `fundamentals_history`

| metric | before | after | plan predicted |
|---|---|---|---|
| total rows | 51,608 | **51,255** | 51,255 ✅ |
| NULL `totalRevenue` | 3,593 | **1,975** | ~2,300 |
| whole trailing twelves | 48,015 | **49,280** | 49,500 |
| tickers | 488 | **489** (AVGO joins) | — |

**+1,265 whole TTM rows, −353 duplicate `as_of` rows.** The recovery is 1,265 rather than the
predicted 1,285, and the whole-window total 49,280 rather than 49,500: the plan's SQL simulation
counted CONTIGUOUS windows, of which there are indeed exactly 49,500, but 220 of them hold a quarter
whose vendor `revenue` is absent and `rolling(4)` nulls those by contract. Both numbers are right;
they count different things.

### Per ticker

| ticker | before | after | target | |
|---|---|---|---|---|
| AVGO | **0 rows at all** | 65 | ~66 | ✅ present |
| AZO | 0 | 113 | 113 | ✅ |
| KR | 0 | 112 | 112 | ✅ |
| COST | 2 | 121 | 121 | ✅ |
| GPN | 94 | 98 | +4 | ✅ |
| GOOGL | 83 | 87 | 87 | ✅ |
| IBM | 119 | 121 | 121 | ✅ |
| KO | 119 | 121 | 121 | ✅ |
| AAPL | 117 | 117 | 117 | ✅ no regression |
| BBY | 118 | 118 | 118 | ✅ no regression |
| OKE | 119 | 119 | 119 | ✅ no regression |

⚠ GPN measured **94** at baseline, not the plan's 95, so its target moved to 98. The +4 recovery is
what was asserted and what landed.

### Corrections to the plan's figures

1. **Surviving `(ticker, calendardate)` duplicates are 7, not 8.** There are 7 class-A groups, each
   with exactly 2 distinct `reportperiod`s. KR's 1998-12-31 group holds 3 ROWS over those 2 periods —
   an amendment on top of a collision — which is why the RAW extra-row count is 8 and the
   POST-DEDUP survivor count is 7.
2. **The roster test fails pre-fix on 274 tickers, not the predicted 320.** A duplicate inside a
   ticker's first three quarters, or beside an already-missing one, nulls nothing that was not
   already null. 274 was watched to fail before either fix landed.
3. **TMUS is a third duplicate class the plan did not name**: its 2006-Q4 is filed twice and the
   EARLIEST filing carries no `revenue` at all. The earliest-filing rule correctly refuses that
   window. This is why the roster test applies `_one_row_per_quarter` as a stated precondition
   rather than folding value-selection into the window predicate.
4. **The span band's low edge is not "envelope plus margin".** Measured min is 240 days and the band
   is (240, 320), so the margin is one-sided — recorded in the constant's docstring rather than
   quietly widened.

### Verification performed

- **Roster regression test watched to FAIL pre-fix** (274 tickers; 48,015 < floor), then pass.
- `gate_completeness`: duplicate normalised quarters **316 → 4 tickers**, and those 4 are exactly
  the class-A collisions {BBY, GPN, KR, OKE}, which are not amendments. Missing quarters unchanged
  at 324 — the dedup created no gaps.
- `gap_check`, like-for-like on the same scope: 3,535 pairs / 96 tickers / 264 systematic → 3,600 /
  97 / 266. The top-60 finding set is unchanged bar one swap (APP capex in, BX returnOnEquity out).
  **No regression**; the extra ticker is AVGO entering the overlap at all.
- **AVGO TTM revenue == the sum of its four discrete quarters, diff 0** on all 5 most recent rows
  (e.g. fiscal_end 2025-08-03 → 59,926M = 15,952 + 15,004 + 14,916 + 14,054).
- **AAPL is identical row-for-row** against a reconstruction of the old code path (123/123 rows,
  index equal, 0 differing cells over totalRevenue/netIncome/totalAssets/dilutedShares) — the sort
  key change from `calendardate` to `reportperiod` is not load-bearing, as risk 2 required.

### Pre-existing failures, NOT caused by this work

Confirmed by running them against `HEAD`'s `build_ttm.py`, where they fail identically:

- `test_sharadar_field_map.py`: `test_every_sf1_column_is_accounted_for`,
  `test_share_block_is_deadjusted_against_the_sec_cover_page`,
  `test_a_spinoff_priced_split_row_is_rejected`, `test_interest_expense_corrections_are_applied`,
  `test_coverage_of_the_built_frame`, `test_post_split_share_counts_are_not_a_hybrid_basis`
- `test_sharadar_merge.py`: `test_as_of_matches_sec`, `test_axp_revenue_gap_is_detected`
- `tests/data_aggregate/test_cube_target_factor_panel.py` fails to COLLECT —
  `MACRO_CUBE_FACTORS` / `MACRO_MARKET_SERIES` are absent from `src/constants/constants.py`.
  Unrelated to this change (`constants.py` was never touched).

### Still open

- **The aggregate fingerprint baseline** — the cube reads `fundamentals_history` and 1,265 new rows
  will move it. NOT regenerated; it is a declared risk zone and needs explicit approval.
