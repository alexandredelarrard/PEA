# Phase 4 — The merged `fundamentals_history` ⬜

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

- [ ] A new `fundamentals_history` `Table(...)`, PK `("ticker", "as_of")`, `date_col="as_of"`,
      `date_type_cols=("as_of", "fiscal_end")`, `freshness="quarterly"`, with `read_columns` set.
- [ ] The matching `-- [aggregate]`/`-- [extract]` block in `sql/schema.sql`.
- [ ] A comment on `fundamentals_reason_codes` recording that it points at
      `fundamentals_history_sec` and **not** at this table (D24).

### 2. `src/data_extract/utils/fundamentals_sharadar/merge_history.py`

- [ ] `build_merged_history(context, tickers, *, full=False) -> None`

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

- [ ] `measure_gaps(context, tickers) -> pd.DataFrame`

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

- [ ] `--propose` writes candidate entries with `approved: null`.
- [ ] The merge **ignores any entry with `approved: null`** and logs how many are awaiting decision.
      An unapproved proposal must never silently change data.

⚠ An override moves a field to a source with **54-ticker coverage**. For a ticker outside that
roster the override yields **NULL**, not a fallback to Sharadar — that is the point of field-block
precedence (D14). The report must state the coverage cost of each approved override.

### 5. `src/data_extract/transformers/step_extract_fundamentals_sharadar.py`

Append `build_merged_history` after the four fetchers — the same "never on its own schedule"
reasoning the SEC step already documents: a snapshot is only as fresh as the rows it reads.

### 6. `src/data_extract/cli.py`

- [ ] `fundamentals-history-merged` — build only, with `-F/--full`.
- [ ] `sharadar-gap-check` — measure and `--propose`.

---

## Tests

`tests/data_extract/test_sharadar_merge.py` — real data; each prints its conclusion.

- [ ] `test_as_of_matches_sec` — reproduce the 279/280 measurement on the 14 overlapping tickers.
      Assert ≥ 99% and that every mismatch is SEC-only. Prints the mismatch list.
- [ ] `test_column_contract` — the built frame's columns equal the declared list, in order, and the
      4 amendment columns are **absent**. Prints the diff both ways.
- [ ] `test_no_amendment_columns` — explicit, because their absence is a *decision* (D15) and a
      future reader will otherwise assume it is an oversight.
- [ ] `test_sec_block_is_asof_backward` — assert no SEC value is dated **after** its `as_of`.
      This is the no-leakage property; it is the single most important test in the phase.
- [ ] `test_unapproved_override_is_ignored` — an entry with `approved: null` does not change data.
      Prints the count ignored.
- [ ] `test_axp_revenue_gap_is_detected` — real: the gap check flags AXP `totalRevenue` as
      systematic and does **not** flag JPM's. Prints both.
- [ ] `test_value_columns_are_float` — the `ensure_table` TEXT-column regression, again, on this table.
- [ ] `test_cik_cutover_continuity` — APA / GOOGL / ETN, `skipif` not in roster (D19 is
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

- [ ] 29 tickers, ~580 rows, `as_of` spanning the entitled window.
- [ ] `shar_covered` ≈ all rows; `sec_covered` ≈ the 14 overlapping tickers' rows **only** — the
      expected, stated coverage asymmetry, not a bug.
- [ ] No `text` columns among the value columns.
- [ ] The gap check's proposals are written, and **you have adjudicated them** before the register
      is marked approved.
- [ ] The no-leakage test passes.

---

## Rollback

The merged table is a fresh build from two read-only sources; drop and rebuild costs one CLI run and
no network. `fundamentals_history_sec` and `fundamentals_sharadar` are never written by this phase.
