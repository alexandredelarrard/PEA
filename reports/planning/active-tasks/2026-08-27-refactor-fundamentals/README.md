# Implementation Plan: Refactor `src/data_extract/utils/fundamentals`

**Date Created**: 2026-08-27
**Planning Phase**: 2 of 3 (FIC workflow)
**Research**: [2026-08-27-refactor-fundamentals.md](../../../research/codebase/2026-08-27-refactor-fundamentals.md)
**Spec**: [specs/2026-08-26/refactor-fundamentals.md](../../../../specs/2026-08-26/refactor-fundamentals.md)
**Branch**: `feature/refactor-fundamentals` @ `e8740ad`
**Next Phase**: `/implement reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-N-*.md`

## Overview

`fundamentals/` is 6,593 LOC over 12 files against `prices/`'s 1,640 over 10; it has 5 files
over 500 LOC, 9 functions over 80 LOC, 124 module globals, a 1.39:1 prose-to-code ratio, one
live `NameError` that silently zeroed 3 tickers, and a replay whose docstring claims
`O(filings)` while measuring `O(E²·K)`. This plan fixes the defects, makes the replay faster
**without changing a single stored number**, moves run tracking into the DB, gives `Context`
the config dir and the EDGAR identity it should always have owned, and cuts the prose.

The organising constraint, stated by the user: **no data bug for the sake of speed or
simplification.** Every efficiency phase is therefore gated on a cell-exact replay-equality
harness built in Phase 0, and on the append-only guard (`diff_against_stored`) that production
already enforces.

## Current State (measured, from the research phase)

| | `prices/` (reference) | `fundamentals/` |
|---|---|---|
| LOC / files | 1,640 / 10 | 6,593 / 12 |
| Files > 500 LOC | 0 | 5 (max 1,516) |
| Functions >= 80 LOC | 0 of 60 | 9 of 186 (max 202) |
| Module globals | 28 | 124 |
| `record_run` sites | 11, in 7/10 files | 4, in 2/12 — both main fetchers: zero |
| prose : code | — | 1.39 : 1 (`xbrl_linkbase.py` 1.76 : 1) |

Cost model: `fetch_fundamentals_sec` is 8-worker network-bound (~30k `filing.xbrl()` at 1.4–5.8 s);
`build_fundamentals_history` is **single-threaded** and **quadratic** (cProfile: 213.5 s for a
12-event synthetic ticker, 99.5 % in `_snapshot`, 79 % in `build_periods`, 61 % in `quarterize`).
A real S&P 500 filer has E ~ 60, i.e. ~25x the E=12 cost.

## Decisions taken in planning

These were open questions in the research doc. They are settled — do not re-litigate them.

| # | Question | Decision |
|---|---|---|
| 1 | How far to push the replay speed-up | **Safe-only + measured census.** Bit-identical semantics; constant-factor removal, per-field memoisation, cross-ticker parallelism. Measure the restatement census so the vintage redesign becomes a *later, data-backed* decision. |
| 2 | Parallelising the replay | **Yes**, across tickers, process pool, parent owns all store I/O. Phase 4. |
| 3 | `cols` blast radius | **Fix + regression test now; remediation decided later**, once the running walk's coverage is measured. |
| 4 | `config_dir` on `Context` | **Yes — plus the EDGAR identity/session.** Risk-zone edits to `context.py` and `utils/step.py` are approved for this scope. Phase 5. |
| 5 | `record_run` generalisation | **DB table, not JSON.** New `extraction_run`, PK `(table_name, run_id)`, seeded once from the 17 JSON entries, then the JSON is deleted. Phase 5. |
| 6 | `constants.py` | **New rule**: a symbol lives in `constants.py` only when 2+ *non-test* `src/` files read it. Delete the 46 with zero consumers; relocate the 45 exclusive to `fundamentals/**`; leave `prices/`, `sharadar/`, `modelling/` to a later sweep. Phase 6. |
| 7 | De-verbosing | **Drop the chronology, keep the measurement**, rewritten in present tense. Delete outright — git holds the rest. Phase 7. |
| 8 | `full=True` hardcode | **Do not touch.** A full run is in flight and that is intended. |
| 9 | `notes_text` | **Keep extracting.** The consumer is not wired yet; that is a separate task. |
| 10 | Test tripwires | Convert the **3 silent-skip string pins** to hard imports; add `tests/data_extract/conftest.py` and a pytest config with markers. Leave the 26 private-symbol pins alone; where a rename is forced, update the test in the same commit. |

## Out of Scope

- **`data_aggregate/` and the cube.** Explicitly excluded by the user. This means live bug #2 —
  `step_cube_fundamentals.py:179-180` passes the history frame as `headcount_history` and
  `employee_features.py:40` reads column `employees` while the table declares `employees_sec`,
  so `employee_growth`, `revenue_per_employee` and the whole workforce panel are silently
  empty — **is NOT fixed here**. It is recorded in [deferred.md](deferred.md) and must be
  picked up by the aggregation phase.
- Restoring the 3 commented-out sub-steps or the `full=True` hardcode in `step_extract_all_data.py`.
- Splitting the `full` flag's two meanings (fetcher window vs `rebuild_history`) — user declined.
- Removing the `notes_text` extraction.
- The repo-wide `constants.py` relocation sweep (35 one-consumer symbols outside `fundamentals/`).
- The vintage redesign of `quarterize` (emit all bases + as-of slice). Phase 3 produces the
  measurement that decides it; the work itself is a separate plan.
- Backfilling the tickers the `cols` bug zeroed (decision 3).

## Phases

| Phase | File | Goal | Depends on |
|---|---|---|---|
| 0 | [phase-0-safety-net.md](phase-0-safety-net.md) | Replay-equality harness + the 8-ticker frozen sample + baseline timings | — |
| 1 | [phase-1-defects-dead-code.md](phase-1-defects-dead-code.md) | 2 live bugs, ~40 dead symbols, name collisions, silent-skip pins, `Step._log` | 0 (for the equality gate) |
| 2 | [phase-2-efficiency-constant-factor.md](phase-2-efficiency-constant-factor.md) | Delete redundant work: duplicate `to_datetime`/sorts/`fiscal_year_ends`, catalogue views, `calculation_linkbase` x2, `_leaf_sum` prologue | 0, 1 |
| 3 | [phase-3-efficiency-memoisation.md](phase-3-efficiency-memoisation.md) | Per-field memoisation of the period engine, vectorised `carry_latest_known`, projected reads, restatement census | 0, 2 |
| 4 | [phase-4-efficiency-parallel-replay.md](phase-4-efficiency-parallel-replay.md) | Cross-ticker process pool; parent owns store I/O | 0, 3 |
| 5 | [phase-5-wiring-context-ledger.md](phase-5-wiring-context-ledger.md) | `config_dir` + EDGAR identity on `Context`; `extraction_run` table; the 8 unrecorded writers; `fetch_financial_statements` on the chain (**seed + JSON delete are post-run**) | 1 |
| 6 | [phase-6-constants-generalization.md](phase-6-constants-generalization.md) | `constants.py` -46 dead / -45 relocated; merge the duplicated helpers into `utils/` | 1, 5 |
| 7 | [phase-7-structure-prose-tests.md](phase-7-structure-prose-tests.md) | Split the 9 big functions and `xbrl_linkbase.py`; drop the chronology; conftest + pytest config; docs + DoD report | all |

Two supporting documents, not phases: [deferred.md](deferred.md) (13 findings deliberately not
fixed, each with evidence and an owner) and
[post-run-checklist.md](post-run-checklist.md) (what waits for the walk to finish).

**Why this order.** The safety net comes first because everything after it is judged by it.
Defects before optimisation, so the harness baseline is not built on a `NameError`. Efficiency
in three ascending-risk steps (delete redundancy -> memoise -> parallelise), each independently
revertable. Wiring is orthogonal and may proceed alongside 2–4. The textual churn — function
splits and de-verbosing — is **last**, because a 2,000-line prose diff landing mid-stream would
bury the equality-gated numerical diffs.

## Verification is scoped to a small sample

By request: **refactor and verify on a small sample.** The full walk runs until tomorrow, so
nothing in Phases 0–7 depends on it, and no phase writes to a table the walk is writing.

**The sample: 8 tickers, two tiers** (defined in [phase-0](phase-0-safety-net.md#2-the-frozen-sample--small-and-truncated-in-time)):

- **Tier A** — 8 tickers x first 16 filings. ~6 min serial at baseline. Runs after every commit.
- **Tier B** — the same 8 at full history. Runs once at the end of each of Phases 1–4.

MCD, ORCL, BA, BAC, KR, BRK-B, APA, VRT — each one already known to exercise a distinct edge, and
all 8 confirmed present in `fundamentals_facts` at 2026-08-27 12:05.

Measured before writing this plan (`build_ticker` off the live DB — full table in
[phase-0](phase-0-safety-net.md)):

| what | measured |
|---|---|
| MCD, full history (E=69, 6,991 rows) | **323.69 s** |
| MCD at E=16 / 32 / 48 | 60.67 / 166.01 / 274.15 s |
| VRT at E=4 / 8 / 12 / 16 | 9.10 / 18.71 / 32.79 / 41.65 s |
| growth E=16 -> 69 | 5.3x wall for 4.3x events, i.e. **`E^1.15`** |
| tier A (8 x 16 filings) | ~7 min |
| tier B (8 x full history) | ~40 min |
| **full 491-ticker universe** | **~44 hours single-threaded** |

Two things follow. First, the tier split is not arbitrary — full history on the 8 is 40 minutes,
which is a per-phase check, not a per-commit one. Second, **Phase 4's process pool is the
load-bearing change, not an optimisation**: 44 h -> ~11 h at 4 workers. The append-only path does
not help, because `build_ticker` replays every event on every run and only then filters to new
`as_of`s.

Note on the headline: the research's `O(E²·K)` is a **call-count** result, verified on counters.
The wall clock is `E^1.15`, and its marginal cost per event is not even monotone. **Do not promise
a 25x win from the quadratic term.** All of these numbers were also taken while the live 8-worker
fetch was competing for CPU (`filing.xbrl()` parsing is CPU-heavy), so they are inflated by an
unknown amount — Phase 0 re-baselines on a quiet machine before any ratio is quoted.

### What is NOT verified here

Deliberately deferred to [post-run-checklist.md](post-run-checklist.md): the full-universe replay
acceptance, the ledger seed and JSON cutover, the whole-table restatement census, the 12 named
edge-case tickers left out of the sample, and the `cols` coverage remediation.

## Sequencing constraints: the live run

A full `main.py` walk started **2026-08-27 00:06**, was still appending at **11:29**, and runs
**until tomorrow** (`.log/output_2026-08-27_00.log`). It will run `fetch_fundamentals_sec`, then
`build_fundamentals_history`, with the **current** code.

- Editing `.py` files cannot disturb it — the modules are already imported. Phase 1's one-word
  `cols` fix is safe to land immediately, but **will not help this run**.
- **Do not run a live `build_fundamentals_history` or `--rebuild-history` while it is in flight.**
  Phase 0–4 verification replays frozen parquet inputs; db-mode comparisons are read-only.
- **Do not delete or edit `data/extraction_manifest.json`.** The running process will
  read-modify-write it on completion using the old `record_run`; if the file is gone by then,
  `_load_manifest` returns `{}` and the rewrite **destroys the other 15 tables' history**, which
  means a full rescan for every one of them. A snapshot has already been taken to
  `manifest-snapshot.json` (17 entries).
- **Do not run any fetcher** between the Phase 5 code cutover and the post-run seed — that is the
  only window in which `manifest_window` would read an empty table.
- The history this run writes becomes a **free production-grade gate** afterwards: a non-rebuild
  re-run must append 0 rows and raise nothing, and `diff_against_stored`
  (`build_history.py:994`) raises on any changed cell. `fundamentals_history_sec` held only **54
  tickers** at 12:05, so most of that gate does not exist yet — hence the post-run checklist.

## Risk zones — ask before editing

`AGENTS.md` requires approval for these. Approval status for this plan:

| Path | Phase | Approved? |
|---|---|---|
| `src/context.py` | 5 | **Yes** — decision 4 (`config_dir` + EDGAR identity) |
| `src/utils/step.py` | 1, 5 | **Yes** — decision 4 + `Step._log` attribution |
| `src/constants/constants.py` | 6 | **Yes** — decision 6 |
| `src/data_store/schema.py` | 5 | **Yes** — `extraction_run` table |
| `sql/schema.sql` | 5 | **Yes**, by hand-spliced additive block only. The generator DROPS 8 hand-added indexes; the diff must be purely additive. |
| `configs/configs.yml` | 4 | **Confirm at implementation time** — one new knob, `fundamentals_replay_workers` |
| `data/` + the Postgres volume | 0, 4 | Reads only, until Phase 4's acceptance run |
| the aggregate fingerprint baseline | — | **Not touched.** `docs/testing.md:154-169` forbids regenerating it in a commit that touches `src/`. |

## Verification spine

Three independent nets, in order of strength:

1. **Cell-exact replay equality** (Phase 0), on the 8-ticker sample. Frozen facts ->
   `build_ticker` -> both output frames compared cell-by-cell including dtypes, NaN placement and
   reason-code sets. Tier A after every commit, tier B per phase. This is the gate for Phases 2, 3
   and 4. A single differing cell blocks the phase.
2. **`diff_against_stored`** against the live DB — restricted here to whichever of the 8 sample
   tickers already have `fundamentals_history_sec` rows. It compares against numbers published by
   the **old** code, which makes it the strongest check available; the full-universe version is
   post-run.
3. **The 201 existing test functions** plus the 4 hash-suffixed cluster regressions
   (`1c9a517eaa47` MCD capex, `2603621e89ab` ORCL totalRevenue, `919b35844b54` BA
   incomeTaxExpense, `876ab8a57bd8` ORCL grossProfit). **Never rename those files** —
   `validate/cli.py:509-519` only checks that the path exists, so a rename silently invalidates
   the recorded `test_path` in `fundamentals_check_fix` with no failing test.

Every test must print a sanity-check conclusion (`docs/testing.md:76-102`) and run with `-s`
from the repo root:

```bash
PY="$HOME/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe"
rtk "$PY" -m pytest tests/data_extract/fundamentals -v -s
```

## Success Criteria

- [ ] Zero changed cells in `fundamentals_history_sec` / `fundamentals_reason_codes` across
      Phases 2–4, proven by the Phase 0 harness on **both tiers of the 8-ticker sample**, in
      frozen and db mode, **and** by a live non-rebuild re-run over whichever of the 8 already
      have published history.
- [ ] `build_fundamentals_history` wall clock on tier B **<= 40 %** of the quiet-machine baseline
      single-threaded (Phases 2+3), and **<= 15 %** of it with the pool at 4 workers (Phase 4).
      Deliberately looser than a quadratic-term story would allow: the measured curve is `E^1.15`,
      so the win must come from the constant factor and the pool, not an asymptotic collapse.
      In absolute terms the bar is **MCD full history under ~130 s** (from 324 s) serial.
- [ ] No file in `fundamentals/` over **600 LOC**; no function over **80 LOC**.
- [ ] Module globals in `fundamentals/` under **60** (from 124); `constants.py` under **720**
      lines (from 1,058) with **0** zero-consumer symbols.
- [ ] prose : code under **0.7 : 1** in every `fundamentals/` file; **0** occurrences of
      `Phase N` / `§N` / `decision #N` / `D<number>` / `used to` / `previously` / `an earlier
      version` under `src/data_extract/utils/fundamentals/`.
- [ ] `record_run` fires for every fundamentals-family writer and `extraction_run` exists with a
      row per table per run. **The seed and the JSON delete are post-run** — see
      [post-run-checklist.md](post-run-checklist.md).
- [ ] `-c /some/other/configs` demonstrably reaches the catalogue loader.
- [ ] All 201 existing tests pass, the 3 silent-skip pins are hard imports, and the new branch
      tests (empty linkbase, memo invalidation, ledger seeding) pass with printed conclusions.
- [ ] `tests/data_store/test_store_boundary.py` still passes (no `sqlalchemy` leak into the
      process-pool worker).

## Estimated Effort

| Phase | Estimate |
|---|---|
| 0 Safety net | 0.5 day |
| 1 Defects + dead code | 0.5 day |
| 2 Constant-factor efficiency | 1 day |
| 3 Memoisation + census | 1.5 days |
| 4 Parallel replay | 1 day |
| 5 Wiring (Context + ledger) | 1.5 days |
| 6 constants + generalization | 1 day |
| 7 Structure + prose + tests | 1.5 days |
| **Total** | **~8.5 days** |
