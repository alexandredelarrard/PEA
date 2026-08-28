# Phase 4 — Efficiency C: parallelise the replay across tickers ⬜

**Goal**: turn `build_fundamentals_history`'s `for ticker in tickers:` (`build_history.py:1052`)
into a bounded **process** pool, while every database read, write, drift check and log line stays
in the parent. This is the largest single wall-clock lever in the plan and the one with the
smallest semantic surface — the per-ticker computation is untouched.

**Measured stake**: one real filer's full-history replay is **323.69 s** (MCD, E=69), so the
491-ticker universe is **~44 hours single-threaded**. At 4 workers that is ~11 h; at 8, ~5.5 h.
Phases 2 and 3 shave the constant factor; this phase is what turns the number from "a day and a
half" into "overnight". It is not an optimisation, it is the fix.

**Gate**: Phase 0 harness (0 differing cells) **plus** a same-result-as-serial test, because a
pool changes ordering, not values.

## Why a process pool and not a thread pool

`parallel_fetch.run_per_ticker` uses `ThreadPoolExecutor` and its docstring explains exactly why
that is right *there*: the EDGAR walk is network-bound and `edgartools` serialises request starts
through a shared ~9 req/s limiter, so a single-threaded walk is latency-bound with one request in
flight.

The replay is the opposite. `periods.py` has **zero** DB access and `xbrl_linkbase.py` is entirely
offline — no `requests`, no `urllib`, no `httpx`, no `Company`, no `get_filings`. The profile is
pure Python and pandas: 31.7 % in `DataFrame.__getitem__`, 19.0 % in `_getitem_bool_array`, i.e.
GIL-held work. Threads would buy close to nothing. So: `ProcessPoolExecutor`.

`parallel_fetch.run_per_ticker` is therefore **not** reused. Its contract ("worker must catch its
own exceptions", `cik_map` of `(ticker, cik)` rows) does not fit either. Add a sibling —
`parallel_cpu.run_per_ticker_cpu` in `utils/common/` — rather than overloading it, and cross-
reference both docstrings so the next reader knows which is which and why.

## Design

```
parent (has Context, store, log)                    worker process (pure CPU, no store)
--------------------------------                    -----------------------------------
for each ticker:
  store.load(facts, columns=, where=)  ---------->  build_ticker(ticker, facts,
                                                                 catalogue, guards)
                                       <----------  TickerHistory(history, reason_codes)
  store.load(history_sec, columns=)
  diff_against_stored(...)  -> raise on drift
  filter to new as_of
  store.save(history_sec) ; store.save(reason_codes)
  log.info(...)
```

- [ ] Worker signature: `_replay_one(ticker: str, facts: pd.DataFrame) -> tuple[str, pd.DataFrame, pd.DataFrame]`
      at **module level** in `build_history.py` (Windows uses `spawn`; a closure or a local
      function cannot be pickled).
- [ ] Catalogue and guards are **not** pickled per task. A pool `initializer` calls
      `load_catalogue(config_dir)` / `load_guards(config_dir)` once per process; both are
      `@cache`d, so every task in that process reuses them. Pass `config_dir` as a plain string
      through `initargs` (Phase 5 makes it a `Context` property; until then pass the resolved
      path).
- [ ] **The `Context` is never pickled.** Only the facts frame crosses the boundary. This keeps
      `tests/data_store/test_store_boundary.py` green by construction: the worker module must not
      import `store`, `sqlalchemy`, or anything that opens a connection.
- [ ] Bounded submission window: do not submit 491 tasks at once. Submit `2 x workers` and top up
      as futures complete, so at most `~2 x workers` facts frames are in flight. Phase 0 recorded
      peak RSS per replay; size the window off that number, and record the chosen bound in the
      docstring with the measurement behind it.
- [ ] One shared `tqdm` bar, as `run_per_ticker` does today.
- [ ] `workers=1` must take the **serial path** (no pool at all), so profiling and debugging are
      unchanged and a broken pool is one config flip from bypassed.

## The `ensure_table` race — why it does not apply here, and the one case where it would

`store.ensure_table` is a check-then-create with no lock; threaded writers on a **cold** table can
silently lose rows. That risk is real and is the reason to be explicit:

- In this design **only the parent writes**, serially. There is no concurrent `ensure_table`.
- The two target tables are already warm in the live DB.
- [ ] Belt and braces: before the pool starts, the parent asserts both tables exist (or creates
      them from the first ticker's frame on the serial path). Write that as a one-line
      precondition with a comment naming the race, so nobody later "optimises" the writes into
      the workers.
- [ ] Add a comment at the write site: **writes stay in the parent** — this is a correctness
      constraint, not a style preference.

## Failure handling

Phase 1 established the principle: a programming error must not read as a data problem.

- [ ] A worker exception is **not** swallowed. The parent catches it per future, logs
      `ERROR ticker=<t>` with the traceback, records the ticker in a `failed` list, and continues
      the remaining tickers.
- [ ] At the end, if `failed` is non-empty, log the full list and **raise** — the run must not
      report success with 3 tickers missing, which is precisely how the `cols` bug survived.
- [ ] `diff_against_stored`'s `ValueError` (`build_history.py:1077-1082`) keeps its current
      behaviour: it aborts the run. It fires in the parent, so nothing changes.
- [ ] The failed-ticker list feeds `extraction_run.tickers_failed` (Phase 5).

## Config

- [ ] `configs/configs.yml`: add `data_extract.fundamentals_replay_workers` next to the existing
      `fundamentals_workers`, with a comment explaining the difference — one is a **network**
      width capped by SEC's ~9 req/s, the other a **CPU** width capped by cores and RSS.
      Default: `min(8, os.cpu_count() - 1)`.
      **Risk zone — confirm before editing `configs/`.**
- [ ] `build_fundamentals_history(context, tickers, *, rebuild_history=False, workers=None)`;
      `None` reads the config. The step passes it explicitly, following
      `step_extract_prices.py:12-14`'s stated rule that the step resolves parameters and passes
      them **in**.

## Tests

- [ ] `test_replay_pool_matches_serial` — 4 synthetic tickers, `workers=1` vs `workers=3`,
      assert identical history and reason-code frames per ticker (sorted by `(ticker, as_of)`
      before comparing, since completion order differs). Prints the differing-cell count.
- [ ] `test_replay_pool_reports_a_failing_ticker` — one worker raises; assert the other 3 tickers
      are still written, the failure is logged, and the call raises at the end with that ticker
      named.
- [ ] `test_replay_worker_imports_no_store` — a grep-style architectural guard in the spirit of
      `tests/data_store/test_store_boundary.py:125-136`: assert the worker's module-level imports
      contain no `store`/`sqlalchemy`/`context`. Cheap, and it is what keeps the boundary from
      eroding later.

## Verification

- [ ] Phase 0 harness, **tier A** with `workers=1` and `workers=4`: **0 differing cells** in both,
      and identical between them. 8 tickers is enough to saturate 4 workers.
- [ ] Phase 0 harness, **tier B** with `workers=1` and `workers=4`: same.
- [ ] `rtk "$PY" -m pytest tests/data_extract/fundamentals tests/data_store/test_store_boundary.py -v -s`
- [ ] **Sample acceptance run** — a non-rebuild `build_fundamentals_history` restricted to the 8
      sample tickers via `-t`, against the live DB. Only for tickers that already have rows in
      `fundamentals_history_sec` (**54 tickers today**, so check which of the 8 qualify).
      Expected: **0 rows appended, 0 drift raised**. `diff_against_stored` is doing the work here
      — this is the strongest single check in the plan, because it compares against numbers that
      were published by the *old* code.
- [ ] Record tier A and tier B wall clock at `workers=1` and at the configured width, plus peak
      RSS per worker.
- [ ] **Full-universe acceptance is NOT part of this phase** — see
      [post-run-checklist.md](post-run-checklist.md). The walk runs until tomorrow and the replay
      that follows it will publish history for the remaining ~348 tickers under the *old* code;
      the wide check belongs after that.

## Risks

| Risk | Mitigation |
|---|---|
| Windows `spawn` re-imports the module and re-runs import-time work per process | Keep the worker module's import side effects to nothing (the catalogue load is in the `initializer`, not at import). Measure process start-up once; if it dominates for small samples, fall back to serial below a ticker-count threshold. |
| Pickling cost of facts frames | ~6k rows x 19 projected columns per ticker. Measure it once and record; if it is material, pass the parquet path instead of the frame and let the worker read it. |
| Memory blow-up | Bounded submission window sized off Phase 0's per-ticker peak RSS. The live fetch process was observed at **4.0 GB** — do not run the pool alongside it. |
| A worker inherits a half-initialised logging config and logs nothing | Workers should not log. All logging is in the parent. Assert this in review. |
| Ordering-dependent output somewhere in `build_ticker` | `test_replay_pool_matches_serial` compares after an explicit sort; the Phase 0 double-replay determinism check already proved per-ticker determinism. |
| Pool masks the real progress signal | Keep the single `tqdm` bar and add a parent `INFO` line per completed ticker with rows written — the live run's **12 INFO lines in 10.6 h** is the failure mode to avoid. |
