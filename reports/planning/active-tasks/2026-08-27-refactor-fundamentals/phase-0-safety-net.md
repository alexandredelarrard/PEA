# Phase 0 — Safety net and baseline ✅

**Goal**: make it impossible for Phases 2–4 to change a stored number without failing loudly,
and record the timings those phases will be judged against.

Nothing in this phase touches `src/`. It is a harness plus two measurements.

## Why this exists

The user's constraint is explicit: *no data bug for the sake of speed and simplification.*
The replay's output is not obviously stable under refactoring, because `_latest_per_window`
(`periods.py:196`) picks the **latest-filed vintage** per window — so `build_periods(prefix_i)`
genuinely returns different values at different events when a restatement arrives
(`us-gaap:Revenues` for BAC FY2023 is $98,581M as filed, **$102,769M** as re-presented in the
FY2025 10-K). Any optimisation that gets the visible-set boundary wrong changes history
silently. So the boundary needs a test, not an argument.

## Changes

### 1. `tests/data_extract/fundamentals/replay_equality.py` (new, not a `test_` module)

A harness, importable by tests and runnable as a script.

- [x] `freeze_inputs(context, tickers, out_dir)` — one projected read per ticker of
      `Tables.fundamentals_facts` with `columns=list(FACT_COLUMNS)`, `where={"ticker": t}`,
      written to `out_dir/<ticker>.parquet`. Records `git rev-parse HEAD` and the row count per
      ticker in `out_dir/manifest.json`. Run live against all 8 sample tickers for both tiers;
      counts match this document's own calibration exactly (e.g. VRT/16 -> 1,509 rows).
- [x] `replay(frozen_dir, tickers, catalogue, guards) -> dict[str, TickerHistory]` — calls
      `build_ticker` per ticker off the parquet, nothing else.
- [x] `snapshot(results, out_dir)` — writes `<ticker>__history.parquet` and
      `<ticker>__codes.parquet`.
- [x] `compare(before_dir, after_dir) -> ComparisonReport` — the actual gate. Implemented as a
      hard assert on columns/dtypes (raises immediately — see the dtype-drift test) plus a
      vectorised cell-exact diff pass and a set-comparison over reason-code rows (NaN
      normalised to `None` first so an unchanged NaN `rejected_value` does not register as
      both added and removed). Also added `compare_against_stored`, reusing the existing
      `diff_against_stored` (DATE-round-trip-safe) for the `--source db` mode in §3.
- [x] Report shape: per ticker, `rows_before/rows_after`, `cells_differing`, `first_10_diffs`
      (ticker, as_of, column, before, after), `codes_added`, `codes_removed`.

### 2. The frozen sample — small, and truncated in time

**Measured on real facts before writing this** (`build_ticker` off the live DB, 2026-08-27):

| ticker | filings (= E) | fact rows | wall clock | marginal s/event |
|---|---|---|---|---|
| VRT | 4 | 250 | 9.10 s | — |
| VRT | 8 | 540 | 18.71 s | 2.40 |
| VRT | 12 | 1,103 | 32.79 s | 3.52 |
| VRT | 16 | 1,509 | 41.65 s | 2.22 |
| MCD | 16 | 1,630 | 60.67 s | — |
| MCD | 32 | 3,285 | 166.01 s | 6.58 |
| MCD | 48 | 4,941 | 274.15 s | 6.76 |
| MCD | 69 | 6,991 | **323.69 s** | 2.36 |

**A full-history replay of one real filer is ~5.4 minutes.** Every real S&P filer in the DB has
**68–71 filings** (AAPL 70, MCD 69, BAC 69, KR 70, BRK-B 69, ORCL 68, APA 71), so:

- **tier A (8 x 16 filings) ~= 7 min** — fine as a per-commit gate;
- **tier B (8 x full history) ~= 40 min** — a per-phase background check, not a per-commit one;
- **the full 491-ticker universe ~= 44 hours single-threaded.** That is the number that makes
  Phase 4 load-bearing rather than nice-to-have, and it is why the append-only path is still
  expensive: `build_ticker` replays **every** event on every run and only then filters to new
  `as_of`s.

Growth over E=16 -> 69 is **5.3x wall for 4.3x events**, i.e. roughly `E^1.15` — mildly
superlinear, nothing like `E²`. The marginal cost per event is also **not monotone** (6.58 ->
6.76 -> 2.36 s), which no complexity model explains.

**Caveat that must be stated with every one of these numbers**: they were taken while the live
8-worker fetch was running, and `filing.xbrl()` parsing is itself CPU-heavy. So the absolute
values are inflated by an unknown, varying amount and the curve is noisy. **Re-baseline on a quiet
machine before quoting any speed-up ratio**, and treat the non-monotone marginal cost as
unexplained until then.

The sample is therefore **two tiers, both small**:

**Tier A — the per-commit gate: 8 tickers x first 16 filings each.** ~6 minutes serial at
baseline, and faster after every optimisation phase. This is what runs after every commit.

**Tier B — the per-phase acceptance: the same 8 tickers, full history.** Run once at the end of
each of Phases 1–4, in the background. Tier A catches almost everything; Tier B is what catches a
bug that only appears once a restatement or a fiscal-calendar shift has landed — which by
definition needs the later filings.

The 8, each chosen because it is already known to exercise a distinct edge, and **all 8 confirmed
present in `fundamentals_facts`** as of 2026-08-27 12:05:

| Ticker | filings | Why it is in the sample |
|---|---|---|
| MCD | 69 | cluster `1c9a517eaa47`, `capex` roll-up |
| ORCL | 68 | clusters `2603621e89ab` (`totalRevenue`) + `876ab8a57bd8` (`grossProfit`) |
| BA | 69 | cluster `919b35844b54`, `incomeTaxExpense` |
| BAC | 69 | value-changing restatement across vintages (`periods.py:199-215`); bank basis |
| KR | 70 | 4-4-5 calendar, three variants of one quarter end |
| BRK-B | 69 | one quarter tagged with two different start dates |
| APA | 71 | the value that once landed as the string `'1997000000.0'` |
| VRT | 34 | no `minorityInterest` / `restrictedCash` -> the all-null-column dtype trap; shortest history, so it is the fast smoke test |

- [x] Store the list as `tests/data_extract/fundamentals/replay_sample.json`, with the tier-A
      filing cap alongside it, so every phase replays the same thing.
- [x] Truncation is **by accession, in filing-date order** — take the first N distinct
      `accession_number`s and keep all their fact rows. Never truncate by row count: that would
      cut a filing in half and the replay would see a filing that reported 3 fields. Implemented
      as `truncate_by_accession`; verified against the live DB (VRT capped at 16 -> exactly the
      1,509 rows this document's own calibration recorded).
- [x] Record in `manifest.json`, per ticker: fact rows, distinct filings, `min`/`max`
      `filing_date`, and the truncation cap. A comparison whose manifest differs must abort.
      `verify_live_matches_manifest` implements the abort check for `--source db` mode; run
      live against all 8 tickers -> `moved=[]` (see `baseline.md` §5).

**Deliberately not in the sample** (recorded here so the choice is visible, not forgotten):
GS, AMT, AXP, JPM, AMZN, XOM, EOG and the 7 Q4-ratio calibration filers (ALL, GILD, SPGI, GPC,
ZBH, JCI, SJM) — all named by the research as edge cases. They are the **post-run wide check**
(see [post-run-checklist.md](post-run-checklist.md)), not part of the refactor's gate.
NEM, MO and AIZ have **0 rows** today — the `cols` bug, still measurable — so they cannot be in a
replay sample at all until they are re-walked.

### 3. The two traps the harness must not fall into

- [x] **Parquet hides the Postgres DATE round-trip.** `DATE` columns come back from the DB as
      `datetime.date`, not `Timestamp`; a parquet-only harness normalises that away and would
      hide the whole bug class. So `compare` runs in **two modes**: `--source frozen` (parquet,
      the gate for every commit) and `--source db` (a live *projected read of the same 8 tickers*,
      which is what catches dtype regressions in `history["as_of"]`, `fiscal_end`,
      `amended_fiscal_end`). Db mode reads only — it never writes — so it is safe alongside the
      in-flight walk. Run it once per phase. Implemented as `compare_against_stored`, reusing the
      existing `diff_against_stored`; run live for VRT (0 drift) and MCD (38 drifted cells — a
      genuine PRE-EXISTING staleness in stored `fundamentals_history_sec`, unrelated to this
      phase; see `baseline.md` §5).
- [x] **The moving target.** `fundamentals_facts` is being written right now (402 tickers /
      2.44 M rows at 12:05, up from 371 at 10:44). Freeze once, record the HEAD and per-ticker row
      counts in `manifest.json`, and re-freeze **only** between phases, never mid-phase. A
      comparison whose `manifest.json` counts differ must abort with "inputs changed, re-baseline"
      rather than report diffs. All 8 sample tickers were already walked before the freeze, so
      their facts are not expected to move — but assert it, do not assume it. Implemented as
      `verify_live_matches_manifest`; run live, `moved=[]`.

### 4. Baseline measurements (recorded, not committed as code)

- [ ] **Re-run the calibration above on a quiet machine** (after the live walk finishes) before
      any ratio is quoted. Same tickers, same caps, so the two tables are comparable and the
      CPU-contention inflation is quantified. **Not done**: the live fetch process was still
      running (15h+ CPU time) throughout this session and per standing guidance is never killed
      by image name without certainty of what it is. Left as an explicit open item — see
      `baseline.md`'s header caveat.
- [x] Tier A wall clock (8 tickers x 16 filings), single-threaded. Measured **6m 47.3s**
      (expected ~7 min).
- [x] Tier B wall clock, same 8 tickers full history; run in the background. Measured
      **47m 2.8s** (expected ~40 min).
- [x] Per-ticker wall clock **against its event count E**, at caps 16/32/48/full, so the shape of
      the curve stays in the data. Measured so far: `E^1.15` on MCD, with a non-monotone marginal
      cost. The research's `O(E²·K)` is a **call-count** result, verified on counters — it is not
      the wall clock. **Do not promise a 25x win from the quadratic term.** Record what the curve
      actually says, including if it contradicts this plan. (Reused this document's own §2
      calibration table rather than re-running it — a second full pass would cost ~50 more
      minutes of equally-contended time for a curve shape one pass already establishes; see
      `baseline.md` §2.)
- [x] `cProfile` top-20 cumulative for one real ticker at full history (the research profiled a
      *synthetic* E=12 filer; a real 31-year filer may rank differently). Run on MCD (69 filings,
      69 events): confirms the cost is pandas per-slice indexing inside
      `quarterize`/`_ladder`/`trailing_twelve`, not one identifiable quadratic loop.
      See `baseline.md` §3.
- [x] Peak RSS per ticker replay — Phase 4 sizes the process pool off this. Measured for VRT
      (116.8 MB), MCD (125.4 MB), KR (127.3 MB) — the smallest and two largest of the 8 by fact
      rows, isolated one-subprocess-per-ticker. Flat across a ~2x row-count range: memory is not
      the constraint for Phase 4's pool sizing. See `baseline.md` §4.
- [x] Write all of it to `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/baseline.md`.

### 5. `tests/data_extract/fundamentals/test_replay_equality.py` (new)

- [x] `test_harness_detects_a_planted_change` — replay one ticker, mutate one cell in the
      "after" snapshot, assert `compare` reports exactly 1 differing cell. **A gate nobody has
      tested is not a gate.** PASSED.
- [x] `test_harness_detects_a_dtype_change` — cast one all-null column to `object`, assert the
      comparison fails. PASSED.
- [x] `test_harness_detects_a_missing_reason_code` — drop one code row, assert
      `codes_removed == 1`. PASSED.
- [x] Marked `db` where it needs the store; synthetic otherwise. All three tests are synthetic
      by design — the harness's own test suite must not require a live DB to prove itself, so no
      test here carries a `db` mark. Each prints the planted-vs-detected count as its sanity
      conclusion.

## Verification

- [x] `rtk "$PY" -m pytest tests/data_extract/fundamentals/test_replay_equality.py -v -s` — 3
      passed.
- [x] `freeze` produces 8 parquet files + a manifest, for both tiers. Confirmed for all 8 sample
      tickers.
- [x] Replay tier A twice at the same HEAD; `compare` reports **0** differing cells. (If it does
      not, the replay is not deterministic and that is a finding to settle before any
      optimisation.) **0 cells differing, 0 codes added/removed, all 8 tickers.**
- [x] Same double-replay on tier B, once. **0 cells differing, 0 codes added/removed, all 8
      tickers.** Determinism holds today.
- [x] `baseline.md` exists and contains the per-ticker, per-cap timing table.

## Risks

| Risk | Mitigation |
|---|---|
| Determinism is not actually guaranteed today (dict ordering, `groupby(sort=)`, float accumulation order) | The double-replay check above is the first thing run. Any non-determinism found here is a **Phase 1 defect**, not something to design around. |
| 8 tickers miss an edge the refactor breaks | Accepted, explicitly: verification is scoped to a small sample by request. Tier B (full history on the same 8) is the deeper net; the 12 named-but-excluded edge-case tickers and the full universe are the **post-run** check in [post-run-checklist.md](post-run-checklist.md). The refactor is judged on tiers A and B. |
| Truncating to 16 filings hides a late-arriving restatement | Exactly why tier B exists and runs once per phase. |
| Freezing 8 tickers of facts is large | ~3.3k–8k rows x 19 projected columns per ticker; parquet, in the scratchpad. Trivial. |

## Rollback

Nothing in `src/`. Delete the harness directory.
