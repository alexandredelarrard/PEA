# Phase 3 — Efficiency B: memoise the period engine, vectorise the instant lookup ⬜

**Goal**: stop recomputing a field's quarters when that field's visible facts have not changed —
with a written proof that the memo key is exact, not a hope. Then vectorise the one remaining
hot primitive, and **measure** whether the deeper vintage redesign is worth doing at all.

**Gate**: Phase 0 harness, 0 differing cells. Plus a dedicated invalidation test, because a memo
bug is exactly the class of "data bug for the sake of speed" the user ruled out.

---

## 3.1 Why "compute once and slice" is WRONG (settle this first)

`build_history.py:814` suggests slicing is intended, and it is tempting: compute
`build_periods` once on the whole ticker, then take the rows whose `known_from <= as_of`. It does
not work, for two independent reasons.

**Reason 1 — restatements would leak backwards.** `_latest_per_window` (`periods.py:196`) keeps
**one row per window, the latest filing winning**. Run on the full frame, it returns the
*restated* value; run on the prefix, the *as-filed* one. `us-gaap:Revenues` for BAC FY2023 is
$98,581M as filed and **$102,769M** as re-presented in the FY2025 10-K
(`periods.py:199-215`). Slicing a full-frame result by `known_from` would publish the FY2025
number at a 2024 `as_of`. That is look-ahead leakage in the training data — the worst possible
failure of this pipeline.

**Reason 2 — the ladder's basis choice is visible-set dependent, so slicing would also
under-produce.** `quarterize:606-608` ranks as-reported above derived and keeps *one* row per
`period_end`. `_ladder` prefers `Q4 = FY - YTD9` and falls back to `Q4 = FY - (Q1+Q2+Q3)`
(`:629, :643, :647, :658, :662`). At an early `as_of` where YTD9 was not yet visible, the true
replay emits the **fallback** row; the full-frame run emits the **preferred** row with a later
`known_from`, and a `known_from <= as_of` slice would therefore emit *nothing* at that event.
Missing values, not wrong ones — but still a changed history.

**Reason 3 — refusals are visible-set dependent too.** `_is_ambiguous_duration` reads the
`ends`/`values` arrays of the *visible* set (`periods.py:232+`), and `_drop_annual_masquerading_as_quarter`
compares against the largest quarter *observed so far*. A window's refusal code can therefore
differ between events. Any slicing scheme must vintage the refusals as well as the values.

**Conclusion**: the safe win is memoisation on an exact key, not slicing. The slicing family of
designs (make `quarterize` emit every candidate basis with its own `known_from`, and defer the
dedup to an as-of slice) is a real O(E) design, but it is a **redesign** and it is out of scope
here by decision 1. Section 3.5 measures whether it would be worth its risk.

---

## 3.2 The memoisation, and its proof

### The claim

Let `P_i` be the set of facts visible at event `i`. For field `f`, let `n_f(i) = |P_i ∩ f|` and
`Y_i = fiscal_year_ends(durations(P_i))`. Then:

> `(n_f(i), Y_i) == (n_f(i-1), Y_{i-1})` ⟹ `quarterize` and `trailing_twelve` return
> **identical** frames and identical refusals for field `f` at events `i` and `i-1`.

### The proof

1. `_normalise_facts` (`build_history.py:914`, sorts at `:930`) leaves `frame` sorted by
   `filing_date`. `build_ticker:812-814` slices `frame.iloc[:upto]` with
   `upto = filed.searchsorted(as_of, side="right")`. So `P_i` is a **prefix** of the sorted
   frame and `P_1 ⊆ P_2 ⊆ ... ⊆ P_E`. (The code already states this at `:810-813`.)
2. `build_periods:966` partitions on `duration_type`, then `:970` does
   `durations.groupby("field", sort=True)`. pandas preserves within-group row order, so
   `group_f(P_i)` is **the first `n_f(i)` rows of `group_f(P_E)`** — identical rows, identical
   order.
3. `quarterize(group, spec, guards, year_ends, refusals)` is a pure function of those five
   arguments: it `.copy()`s before mutating (`:566`), touches no module state, reads no clock,
   and its only global is `_QUARTER_COLUMNS`. `spec` and `guards` are constant across the
   replay. Therefore its output depends on `(n_f(i), Y_i)` only.
4. `trailing_twelve(quarters_f, spec, annual=group_f, guards)` (`:793`) reads only its four
   arguments; `annual` is `group_f` again. Same conclusion.
5. Hence equal keys ⟹ equal outputs. ∎

### What the key must include, and what it must not

- **Include `n_f(i)`** — not `i`, and not a hash of the frame. The count is sufficient because
  the visible set is a prefix (step 1); a hash would be correct but pointlessly expensive.
- **Include `Y_i`** as a tuple of `Timestamp`s. `year_ends` is **ticker-wide** — built from every
  annual-shaped fact any field reported (`build_periods:967`) — and it reaches
  `label_fiscal_periods` (`quarterize:610-611`). One new annual fact from *any* field therefore
  relabels *every* field. This is the cross-field coupling that makes a naive per-field memo
  wrong, and it is the single most important line in this phase.
- **Do NOT include** the `refusals` list — it is an out-parameter, not an input. Cache the
  refusal rows *alongside* the frames and re-emit **copies** on every hit, because `_snapshot`
  writes one reason-code row per `as_of` and the cached list must not be aliased into the
  caller's accumulator.

### Implementation

- [ ] New `periods.PeriodCache` (or a small dataclass held by `build_ticker`):
      `dict[str, tuple[int, tuple[Timestamp, ...], DataFrame, DataFrame, list[dict]]]` keyed by
      field name, holding `(n_f, year_ends, quarters, ttm, refusals)`.
- [ ] `build_periods` gains an optional `cache: PeriodCache | None = None`. Absent, behaviour is
      byte-identical to today — this keeps every existing test and every other caller untouched.
- [ ] The per-field loop becomes: compute `n_f = len(group)`; on a hit, reuse; on a miss,
      recompute and store. Return copies of the cached frames if any downstream code mutates
      them (check `_snapshot`; if it does not, return them directly and add a comment saying so).
- [ ] `instant_stock(facts)` (`periods.py:907`) is recomputed per event over the whole prefix.
      Memoise it on the **count of instant rows** by the same argument. `_instant(instants,
      field, as_of)` still runs per event — it is an as-of lookup, and cheap.
- [ ] `fiscal_year_ends` gains a cache keyed on the annual-shaped row count, so computing `Y_i`
      to *test* the key is not itself O(prefix) per event.

### Expected win — state it honestly

On a filer that reports all 48 fields in every 10-Q, `n_f` changes at every event and the memo
buys **nothing**. The win comes from three real populations:

- annual-only fields, whose `n_f` moves on ~1 event in 4;
- fields the filer never reports (already skipped by `groupby`, but their absence is now free
  rather than re-established);
- amendment events, which typically restate a handful of fields.

**Measure it on both sample tiers and report the number, whatever it is.** If it is under 20 %,
say so and keep the change only if it is otherwise clean — the parallel pool in Phase 4 is the
bigger lever, and an unhelpful cache is complexity.

### Tests (these are the point of the phase)

- [ ] `test_period_cache_hit_is_identical` — replay a ticker with and without the cache; assert
      `assert_frame_equal(check_exact=True)` on quarters, ttm, instants **and** equal refusal
      sets, at **every** event, not just the last.
- [ ] `test_period_cache_invalidates_on_new_year_end` — synthetic ticker where event N adds an
      annual fact that shifts a fiscal year end. Assert the cached and uncached
      `fiscal_quarter`/`fiscal_year` labels match for **all** fields, including fields with no
      new facts at event N. This is the test that fails if `year_ends` is left out of the key.
- [ ] `test_period_cache_invalidates_on_restatement` — synthetic ticker where event N restates
      a window with a different value. Assert the pre-N events keep the as-filed value and event
      N flips to the restated one.
- [ ] `test_period_cache_refusals_are_not_aliased` — assert the refusal rows written at event
      `i` and event `i+1` are distinct objects and that mutating one does not change the other.
- [ ] Each prints its differing-cell count (expected 0) as the sanity conclusion.

---

## 3.3 Vectorise `carry_latest_known`

`build_history.py:224-256` builds a **1-row DataFrame and calls `merge_asof`** to answer a single
as-of lookup, once per instant field per event: 312 calls at E=12, **12.8 % of the whole
profile** (`cumtime` 27.28 s of 213.52 s).

- [ ] Replace with, per (field, ticker): a `numpy` array of `period_end` (sorted, already
      guaranteed) and a parallel value array, plus `np.searchsorted(ends, as_of, side="right")-1`.
- [ ] Keep the tie-break rule exactly as `merge_asof` implements it (last row wins on an exact
      match, `direction="backward"`), and keep `_collapse_same_day` upstream doing the same-day
      collapse it does today.
- [ ] The arrays are built **once per ticker**, not per event — they are a prefix structure, so
      the as-of bound is a `searchsorted` on the filing-date axis too.
- [ ] Test: for a synthetic ticker with same-day duplicates, a gap, and an exact-match date,
      assert the vectorised and `merge_asof` answers are identical at 50 random as-of dates.
      **Keep the `merge_asof` implementation in the test as the oracle** — do not delete it into
      the void.

`merge_history._asof_join:272-290` is the same primitive with the same docstring reasoning. Phase
6 unifies them; do not do it here, so this phase's diff stays inside `fundamentals/`.

---

## 3.4 The reads

- [ ] `build_history.py:1061` loads `fundamentals_history_sec` with **`columns=None`** -> `SELECT *`
      over 69 columns. `AGENTS.md` forbids an unprojected read. Pass
      `columns=list(catalogue.history_columns)` explicitly.
      **Honest note**: this is *not* a perf win — `diff_against_stored` needs every value column,
      so the projection is the whole table. It is a convention fix and it makes the read fail
      loudly if the table and the 69-column contract ever diverge, instead of silently handing
      the diff an extra column.
- [ ] `build_history.py:1053` (facts, per ticker, projected, `where=`) is already correct. Leave it.
- [ ] `fetch_fundamentals_sec.py:877` reads `sp500_tickers` and `edgar_driver.py:74` ->
      `sec_utils.py:131` reads it **again, unprojected**, in the same run. Collapse to one
      projected read passed down. (Also fixes an `AGENTS.md` violation.)
- [ ] `fetch_fundamentals_sec.py:883` reads the whole `fundamentals_employees` table with no
      `where=`. Add one, or state in the docstring why the whole table is needed (it seeds a
      median) — but do not leave an unprojected, unfiltered read undocumented.

---

## 3.5 The restatement census — the measurement that decides the next plan

Deliverable, not code. Answers "would the vintage redesign (3.1's rejected family) actually pay?"

**Scope it to the 8 sample tickers, at full history.** `fundamentals_facts` is being written by
the in-flight walk, so a full-table census is both a moving target and a heavy read; an 8-ticker
census is a read of ~55k rows and is enough to tell "rare" from "common". Mark the number
**indicative** and re-run it over the whole table after the walk — that is in
[post-run-checklist.md](post-run-checklist.md).

- [ ] Per `(ticker, field, window)` where a window is `_latest_per_window`'s identity (end within
      `_SAME_PERIOD_DAYS`):
      - number of vintages (distinct `filing_date` reporting that window);
      - whether any two vintages carry **different values** beyond float noise;
      - the fiscal-year span between the first and the value-changing vintage.
- [ ] Report: `% of windows with >1 vintage`, `% with a value-changing vintage`, distribution by
      field, and the **worst of the 8 filers**. (BAC is in the sample precisely because it is
      known to restate; treat it as the upper bound, not the average.)
- [ ] Same census for **refusal flips**: how often does `_is_ambiguous_duration` /
      `_drop_annual_masquerading_as_quarter` change its verdict on a window as the prefix grows?
      This is the number that decides whether vintaged refusals are tractable.
- [ ] Write to
      `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/restatement-census.md`
      with an explicit recommendation: **redesign** or **leave it**.

If value-changing restatements are rare (say < 2 % of windows), a hybrid becomes attractive:
slice for clean fields, replay for dirty ones. That is the next plan's problem, and it needs
this measurement to be written before anyone argues about it.

---

## Verification

- [ ] Phase 0 harness, **tier A frozen** mode: 0 differing cells, 0 dtype changes, 0 code deltas.
- [ ] Phase 0 harness, **tier B** (full history on the 8) — mandatory in this phase, not
      optional. The memo's invalidation only gets exercised once enough events have accumulated
      for a field's visible count to *stop* changing, which a 16-filing truncation barely reaches.
- [ ] Phase 0 harness, **db** mode: same.
- [ ] `rtk "$PY" -m pytest tests/data_extract/fundamentals -v -s` — all pass, plus 5 new tests.
- [ ] Tier A and tier B wall clock vs `baseline.md` and vs the Phase 2 number, per ticker with its
      E, so the memo's effect is visible. Report the hit rate (`hits / (hits+misses)`) explicitly,
      per ticker — a 69-filing filer and a 34-filing one will not agree.
- [ ] `restatement-census.md` exists and ends in a one-line recommendation.

## Risks

| Risk | Mitigation |
|---|---|
| The memo key misses a dependency nobody listed | The proof in 3.2 enumerates every argument of both functions. Re-read those two signatures at implementation time and confirm nothing was added since. If a new parameter exists, it goes in the key or the phase stops. |
| `groupby` order assumption is wrong on some pandas version | Assert it once in the cache: on a miss, check `group.index.is_monotonic_increasing`. Cheap, and it turns an assumption into a check. |
| Cached frames are mutated downstream, corrupting later events | Audit `_snapshot`'s use of `quarters`/`ttm`/`instants`; if any mutation exists, return `.copy()` and eat the cost. Prove with `test_period_cache_hit_is_identical` at every event. |
| Refusal rows aliased across events, so one `as_of`'s codes appear under another | `test_period_cache_refusals_are_not_aliased`. |
| The vectorised `carry_latest_known` diverges from `merge_asof` on ties | The oracle test keeps `merge_asof` as the reference at 50 random dates. |
| The memo turns out not to help | Report the number and drop the change rather than keeping dead complexity. That is an acceptable outcome of this phase. |
