# Phase 3 — Efficiency B: memoise the period engine, vectorise the instant lookup ✅

**Outcome**: the memo was built, proven correct, measured at break-even and **dropped**; the
vectorised instant lookup ships at a measured **15.4x** on its primitive; the reads are fixed;
the census says **do not** build the vintage redesign. Zero differing cells on both tiers.

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

## 3.2 The memoisation — BUILT, PROVEN CORRECT, MEASURED, AND DROPPED

**Outcome first**: `PeriodCache` was implemented exactly as specified below, passed the tier A
gate at 0 differing cells, passed four dedicated invalidation tests — and was then **removed**,
because its measured hit rate is 0.3–15 % (mean ~5 %) and its net effect on CPU time is between
−5.8 % and +1.6 %. The plan pre-registered this outcome ("report the number and drop the change
rather than keeping dead complexity") and the user confirmed the disposition. §3.6 records it.

The proof below is kept because it is the reusable part: it is what any future vintage redesign
(D-3) has to satisfy, and it is what the measurement was interpreted against. The
`git` history holds the code.

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

- [x] New `periods.PeriodCache` holding `dict[str, _FieldMemo]` keyed by field name, where
      `_FieldMemo` is `(n_facts, tail, quarters, ttm, refusals)`. `year_ends` is NOT stored
      per field — it is ticker-wide, so `note_year_ends` clears the whole table when the
      calendar moves, which is O(1) per event instead of a ~30-tuple compare per field.
- [x] `tail` is an extra key component the plan did not ask for: the last visible row's
      `(filing_date, period_end, value)`, three array reads. Under the prefix property a
      matching count already implies a matching tail, so it never fires — which is the
      point. It converts the `groupby`-order assumption into a check, which is what the
      risk table asked for and is strictly cheaper than the
      `group.index.is_monotonic_increasing` it proposed (`_normalise_facts` does not
      `reset_index`, so that assertion would have been false for an unrelated reason).
- [x] `build_periods` gains `cache: PeriodCache | None = None`. Absent, behaviour is
      byte-identical — every existing test and every other caller is untouched.
- [x] The per-field loop hits or recomputes-and-stores. **No copies needed**: `_snapshot`
      reads `quarters`/`ttm` only through `_latest` (a `[mask]` filter then `.loc[idxmax]`)
      and `instants` only through the lookup, and `build_periods` hands out a fresh
      `pd.concat` rather than the cached frames themselves. Audited, and pinned by
      `test_period_cache_hit_is_identical`, which forces 117 hits.
- [x] `instant_stock(facts)` memoised on the **raw** instant row count — not the
      deduplicated output's, which does not determine the frame: a restatement of an
      existing balance-sheet date changes a value while leaving that count alone.
- [x] `fiscal_year_ends` memoised on the annual-shaped valued row count, so testing the key
      is one numpy pass rather than a filter + `to_datetime` + `unique` + the gap-fill loop.

### Measured win — it is small, and the plan pre-registered what to do about that

The plan's own prediction ("on a filer that reports all fields in every 10-Q the memo buys
nothing") is what the data says. **Measured hit rate, tier B (full history, all 8):**

| ticker | E | lookups | hit rate | ceiling with the calendar coupling REMOVED | cost of the coupling |
|---|---|---|---|---|---|
| BA | 69 | 985 | **0.3 %** | 0.4 % | 0.1 pt |
| MCD | 69 | 857 | **1.9 %** | 3.2 % | 1.3 pt |
| BAC | 69 | 831 | **2.3 %** | 3.0 % | 0.7 pt |
| ORCL | 68 | 987 | **3.1 %** | 4.0 % | 0.8 pt |
| KR | 70 | 910 | **5.4 %** | 7.1 % | 1.8 pt |
| VRT | 34 | 429 | **5.4 %** | 5.4 % | 0.0 pt |
| APA | 69 | 830 | **14.9 %** | 17.1 % | 2.2 pt |
| BRK-B | 69 | 661 | **15.0 %** | 18.8 % | 3.8 pt |

Tier A (16 filings) is worse — 0 % on three of the eight, 2.6 % overall.

**Two conclusions the measurement forces:**

1. **The ceiling is ~5 %, and the ticker-wide `year_ends` key is not what caps it.** Deleting
   the calendar from the key entirely — which would be *wrong* — raises the rate by 0.0–3.8
   points and no further. The cause is structural: every filing re-tags the windows it
   already reported as comparatives, so nearly every field's visible fact count grows at
   every event. No key can fix that; only the vintage redesign §3.1 rejected can.
2. **The instant memo never hits at all — 0.0 % on all 8 tickers.** The raw instant row count
   grows at literally every event. `instant_stock` is therefore recomputed every event
   regardless, and only the `InstantLookup` in §3.3 actually buys anything there.

This is under the plan's 20 % keep-threshold, so the decision rule it pre-registered applies:
keep only if otherwise clean. See **§3.6** for the disposition.

### Tests (these are the point of the phase)

All five were written and **passed** in `test_period_cache.py`. Four went with the memo; the
fifth (`test_instant_lookup_matches_merge_asof`) survives in
`tests/data_extract/fundamentals/test_instant_lookup.py`. Recorded here because what they
found is the durable result, not the code they tested.

- [x] `test_period_cache_hit_is_identical` — APA's real sweep ledger capped at 12 filings,
      through `build_periods` at every prefix and through `build_ticker` end to end.
      72 frames compared cell-exact, 0 differing; history + 316 reason-code rows identical.

      **A correction worth recording.** The first version of this test replayed VRT and
      passed with a **0.0 % hit rate** — it never executed the hit path at all, so it proved
      nothing. A replay's natural hit rate is low by construction (see below), so the test
      now runs a THIRD pass over each identical prefix, which forces a hit for every field:
      **117 forced hits**, and every comparison above therefore reads a cached frame. A test
      that can pass without the code under test running is not a test.
- [x] `test_period_cache_invalidates_on_new_year_end` — synthetic filer whose FY2018 annual
      leaves the calendar ending 2019, so `costOfRevenue`'s two 2021 quarters (filed at
      2021-05-03 and **never touched again**) are unlabelled; the FY2022 annual arrives on
      `totalRevenue` alone, the gap-fill inserts 2019-2021, and both quarters acquire
      `FY2021 Q1/Q2`. Measured `(<NA>, <NA>) -> (2021, 1)`. This is the test that fails if
      `year_ends` is left out of the key.
- [x] `test_period_cache_invalidates_on_restatement` — 2021 Q1 filed at 1000.0, restated to
      1250.0 at 2021-11-01. Measured per event:
      `{2021-05-03: 1000.0, 2021-08-02: 1000.0, 2021-11-01: 1250.0}`.
- [x] `test_period_cache_refusals_are_not_aliased` — APA refuses 3 windows
      (`derived_sign_implausible`) inside the cap. 0 of 3 dicts shared between two
      emissions; mutating one left the other emission and the cache untouched. The
      `pytest.skip` the first draft used when a filer refused nothing is now a hard assert —
      an empty refusal list means the plumbing broke.
- [x] `test_instant_lookup_matches_merge_asof` — §3.3's oracle test, 150 lookups, 0
      mismatches.
- [x] Each prints its differing-cell / mismatch count (0) as the sanity conclusion.

---

## 3.3 Vectorise `carry_latest_known`

`build_history.py:224-256` builds a **1-row DataFrame and calls `merge_asof`** to answer a single
as-of lookup, once per instant field per event: 312 calls at E=12, **12.8 % of the whole
profile** (`cumtime` 27.28 s of 213.52 s).

- [x] New `periods.InstantLookup`: one sorted `(period_end, value)` array pair per field,
      answered by `np.searchsorted(ends, as_of, side="right") - 1`. `_instant` now takes the
      lookup instead of the frame.
- [x] Tie-break kept exactly. `direction="backward"` with exact matches allowed IS
      `side="right"` minus one, and the ties `merge_asof` would have had to break **do not
      exist**: `instant_stock` already emits at most one row per `(ticker, field,
      period_end)`. `_collapse_same_day` upstream is untouched.
- [x] **Deviation from the plan, stated.** The arrays are built once per **event**, not once
      per ticker. The plan's "once per ticker" version needs a 2-D as-of — for each
      `(field, period_end)` the latest *visible* vintage, which is not the latest vintage —
      and getting that wrong is precisely the look-ahead leak §3.1 rejects slicing over. One
      groupby-and-sort per event replacing ~20 `merge_asof` calls (each of which builds two
      DataFrames) is already the whole win; the ticker-wide version is the vintage redesign
      wearing a different hat.
- [x] Test: `test_instant_lookup_matches_merge_asof`. 50 as-of dates x 3 fields = **150
      lookups, 0 mismatches**, over a fixture carrying two vintages of one date, a two-year
      gap, three exact-match dates, a pre-history date, a null value and a never-reported
      field. **`carry_latest_known` stays in the tree as the oracle**, with a docstring that
      now says so.

`merge_history._asof_join:272-290` is the same primitive with the same docstring reasoning. Phase
6 unifies them; do not do it here, so this phase's diff stays inside `fundamentals/`.

---

## 3.4 The reads

- [x] `build_history.py` now loads `fundamentals_history_sec` with
      `columns=list(catalogue.history_columns)`. As the plan said, **not a perf win** — the
      projection is the whole table — and the comment in the code says so.
- [x] The facts read (per ticker, projected, `where=`) is already correct. Left alone.
- [x] The duplicate `sp500_tickers` read is collapsed. `run_edgar_fetch` gains an optional
      `cik_map=`; `fetch_fundamentals_sec` does the one `load_cik_mapping` call and derives
      the three GICS levels off that same frame instead of a second read.

      One behaviour delta, checked and benign: the GICS read was `optional=True` and degraded
      to an empty regime map, whereas `load_cik_mapping` raises on a missing universe. It
      changes nothing, because `run_edgar_fetch` called `load_cik_mapping` unconditionally ten
      lines later — the fetch could never have proceeded without the table. The raise just
      happens sooner.
- [x] `load_cik_mapping` is now projected, via a new `sec_utils.CIK_MAPPING_COLUMNS` — the
      union of every column its seven callers read.

      **Honest note, again.** Measured against the live DB, `sp500_tickers` has exactly six
      columns and all six are in that union, so the projection is once more the whole table
      and ~500 rows. It is not the "large table unprojected" `AGENTS.md` targets; naming the
      columns states the contract and fails loudly if the universe table loses one. The
      *real* fix here was the duplicate read.
- [x] The `fundamentals_employees` read: `where=` deliberately NOT added, and the docstring
      now says why. `history_by_ticker` seeds the continuity guard from every stored
      headcount, so filtering to the run's ticker list would silently narrow the guard to
      whichever chunk is being fetched — and the backfill is chunked. Three columns of an
      annual, ~500-ticker table, so it is bounded by construction.
- [x] `read_columns` on `Tables.sp500_tickers` + `project=True` would be the idiomatic
      mechanism (it degrades with a warning instead of raising). **Not done**: `schema.py` is
      a risk zone and this phase has no approval for it. Recorded for Phase 5, which does.

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

## 3.6 Disposition — and how the timing was nearly got wrong

### The measurement method had to be fixed before the decision could be made

The first attempt measured wall clock over sequential runs of the three trees (`head`,
`nomemo` = InstantLookup only, `full` = + memo), three alternating rounds — Phase 2's own
protocol. It produced this:

| round | head | nomemo | full | apparent verdict |
|---|---|---|---|---|
| 1 | 524.7 s | 299.9 s | 377.5 s | memo costs 26 % |
| 2 | 179.8 s | 156.0 s | 294.9 s | memo costs 89 % |
| 3 | 165.2 s | 178.1 s | 201.1 s | InstantLookup costs 8 % |

**Every one of those verdicts is an artefact.** A live `main.py` was burning CPU throughout
(PID 48852, 2,060 s of CPU accumulated across the window) and its load drifted, so within each
round the variant that ran last was penalised — and round 1's `head` was additionally polluted
by two stray REPL processes of my own. Round 2 says the memo costs 89 %; round 3 says the
InstantLookup, which is a strict 15x improvement to a primitive, costs 8 %. Both are load.

**Wall clock is the wrong instrument on a contended machine.** `time.process_time()` counts
only this process's CPU, and interleaving the two arms inside ONE process additionally removes
the import/warmup difference. Redone that way:

| ticker | memo hit rate | full / nomemo, CPU time |
|---|---|---|
| BAC | 0 % | **1.016x** — the memo costs 1.6 % |
| APA | 11 % | **0.942x** — the memo saves 5.8 % |

Which is exactly what the hit-rate analysis predicts: bookkeeping cost when it never hits,
proportional saving when it does. At the sample's mean hit rate the memo is break-even.

**Lesson for Phase 4, which is judged on a speed-up ratio**: measure CPU time, interleave the
arms, and do not quote a wall-clock ratio taken while anything else is running. Phase 0's
"re-baseline on a quiet machine" item is still open, and this is why it matters.

### The instant read, measured properly

`InstantLookup` vs `carry_latest_known` at the replay's own call volume — MCD's full history,
2,505 instant rows, 26 instant fields, 69 events = 1,794 lookups, CPU time, and the two agree
on every value found:

| primitive | CPU |
|---|---|
| `carry_latest_known` (one-row `merge_asof`) | **42.58 s** |
| `InstantLookup` (`searchsorted`) | **2.77 s** |

**15.4x**, i.e. ~40 s of CPU saved per full-history ticker. This is the phase's real win, and it
matches the plan's profiled 12.8 %.

### What ships

- [x] **KEEP `InstantLookup`** (§3.3) and `carry_latest_known` as its oracle.
- [x] **DROP `PeriodCache`** and its four tests: `PeriodCache`, `_FieldMemo`,
      `_tail_signature`, `_annual_row_count`, `_instant_row_count`, the `cache=` parameter on
      `build_periods` / `_snapshot`, and `cache=` / `memoise=` on `build_ticker`.
      `periods.py` went 1,040 (at `582ffb6`) -> 1,269 with the memo -> **1,111** without it,
      so the net cost of the phase in that file is the **+71 lines** `InstantLookup` occupies.
      Phase 7 still has to get it under 600.
      `build_periods`' docstring now records the measurement so nobody re-proposes it blind.
- [x] The decisive reason beyond the numbers: a memo means **every future change to
      `quarterize`'s or `trailing_twelve`'s inputs has to re-prove the key**. That is a
      permanent correctness tax for ~1 %, on the one module in this package where a silent
      wrong number is the worst possible failure.
- [x] Recoverable: if D-3's vintage redesign lands, the proof in §3.2 and the code in git are
      both still valid.

## Verification

- [x] Phase 0 harness, **tier A frozen** mode: **0 differing cells, 0 dtype changes, 0 code
      deltas**, all 8 tickers. Run FOUR times over the phase, which is stronger than the plan
      asked for because each run gated a different tree:
      | snapshot | tree | result |
      |---|---|---|
      | `tierA_p3` | InstantLookup + memo | 0 cells, 0 codes |
      | `ab_full_3` | InstantLookup + memo | 0 cells, 0 codes |
      | `ab_nomemo_2`, `ab_nomemo_3` | InstantLookup only — **what ships** | 0 cells, 0 codes |
      | `ab_head_3` | `582ffb6` itself, as a CONTROL | 0 cells, 0 codes |
      The control matters: it proves the harness and the frozen inputs are stable, so the three
      zeroes above are the code being equal and not the comparison being blind.
- [x] Phase 0 harness, **tier B** (full history on the 8), on the shipped tree:
      **0 differing cells, 0 codes added or removed, all 8 tickers** — APA/BA/BAC/BRK-B/MCD 69
      rows, KR 70, ORCL 68, VRT 34. 3,272 s CPU / 5,171 s wall.

      That wall-to-CPU gap of 1.6x is the machine, not the code, and it is the same effect
      §3.6 is about: the tier A run of the same tree measured 531 s CPU against **1,518 s
      wall** (2.9x) because two processes from a superseded verification run were competing
      with it. CPU time is stable across both; wall clock is not.
- [x] Phase 0 harness, **db** mode (read-only). Two results:
      - **The moving-target guard passes: `moved=[]`.** A fresh uncapped row-count read of all
        8 sample tickers still matches the tier-B manifest, so both gates above rest on inputs
        that have not drifted since the freeze.
      - **VRT and MCD: 0 differing cells, no dtype drift** between a replay fed by a LIVE
        projected read and one fed by parquet. Two tickers rather than eight because each
        needs two full-history replays; VRT is the shortest history and MCD is the longest and
        was `baseline.md`'s own DATE-dtype canary.

      **Scope changed, and not by choice.** `fundamentals_history_sec` held 54 tickers when this
      plan was written; it holds **1** (`A`, 68 rows) today, because the table was emptied
      before this session and a live `main.py` is rebuilding it one ticker at a time. None of
      the 8 sample tickers has stored history, so `compare_against_stored` has nothing to
      compare and the Phase 2 drift finding cannot be reproduced. What db mode exists for is
      still testable and is what ran instead: replay a sample ticker from a LIVE projected read
      (DATE columns arriving as `datetime.date`) and diff against the parquet-fed replay
      (`Timestamp`), which is the round-trip a parquet-only harness hides.

      By the end of this phase the table was back up to **20** tickers (`A` -> `ADM`), so the
      rebuild is working; none of the 8 had been reached yet. Re-running the real
      `compare_against_stored` over the sample is **post-run** work.
- [x] `"$PY" -m pytest tests/data_extract/fundamentals -q` — **232 passed, 0 failed**
      (37m 09s). Phase 2 left 231; this phase adds exactly **one**, not the five the plan
      budgeted, because four went with the memo (§3.6). No existing test needed changing:
      `build_periods`' new keyword was optional and is now gone again, and `instant_stock` and
      `carry_latest_known` kept their signatures.
- [x] Timings: **do not use the wall-clock numbers this phase first produced.** §3.6 has the
      corrected CPU-time measurements and the reason. Headline: the instant primitive is
      **15.4x** (42.58 s -> 2.77 s CPU on MCD's real call volume); the memo was break-even and
      is gone. Per-ticker memo hit rates are in §3.2 for the record.
- [x] [`restatement-census.md`](restatement-census.md) exists and ends in a recommendation:
      **LEAVE IT** — do not build the vintage redesign. Both halves fail their pre-registered
      test independently: **8.22 %** of windows carry a material restatement (threshold was
      < 2 %) and **70 %** of `(ticker, field)` pairs are dirty, so the hybrid saves ~30 %, not
      ~98 %; and **60 %** of refusals REVERSE as the prefix grows, so most of them have no
      `known_from` to be vintaged with. The second is the harder blocker.

## Risks

| Risk | Mitigation |
|---|---|
| The memo key misses a dependency nobody listed | The proof in 3.2 enumerates every argument of both functions. Re-read those two signatures at implementation time and confirm nothing was added since. If a new parameter exists, it goes in the key or the phase stops. |
| `groupby` order assumption is wrong on some pandas version | Assert it once in the cache: on a miss, check `group.index.is_monotonic_increasing`. Cheap, and it turns an assumption into a check. |
| Cached frames are mutated downstream, corrupting later events | Audit `_snapshot`'s use of `quarters`/`ttm`/`instants`; if any mutation exists, return `.copy()` and eat the cost. Prove with `test_period_cache_hit_is_identical` at every event. |
| Refusal rows aliased across events, so one `as_of`'s codes appear under another | `test_period_cache_refusals_are_not_aliased`. |
| The vectorised `carry_latest_known` diverges from `merge_asof` on ties | The oracle test keeps `merge_asof` as the reference at 50 random dates. |
| The memo turns out not to help | Report the number and drop the change rather than keeping dead complexity. That is an acceptable outcome of this phase. |
