# Phase 0 baseline — measured 2026-08-27

HEAD `e8740ad2039c37944f05607d3d07dc4b6f1478aa`. Every number below was taken via
`tests/data_extract/fundamentals/replay_equality.py` against the live DB.

**Contamination, stated up front and applying to every number in this file**: a live
multi-hour fetch process (one `python.exe`, 15h+ accumulated CPU time, 4-7 GB RSS observed
across the run) was on this machine for the whole measurement window, and the machine
acquired several more background python processes as the afternoon went on. The plan's own
pre-existing calibration table (reproduced below) already carries the identical caveat. No
"quiet machine" re-baseline was possible in this session without killing that process, which
per standing guidance is never done by image name / without being certain what it is. **Every
absolute wall-clock number here is inflated by an unknown, varying amount.** Ratios computed
*within* one contended run (e.g. tier A vs tier B, or the shape of the E-curve) are more
trustworthy than any absolute number quoted alone.

## 1. Tier A / Tier B wall clock (this session, single-threaded, contended)

| Tier | Scope | Wall clock | Plan's pre-registered expectation |
|---|---|---|---|
| A | 8 tickers x first 16 filings | **6m 47.3s** (407.3 s) | ~7 min |
| B | 8 tickers x full history | **47m 2.8s** (2822.8 s) | ~40 min |

Both land close to the plan's own contention-aware estimate, which was itself calibrated
under the same live fetch. Per-ticker breakdown is not separated for these two runs (the CLI
times the whole 8-ticker batch); the per-ticker curve in §2 is what has ticker-level numbers.

## 2. Per-ticker wall clock vs event count E

Reproduced from the plan document's own "measured before writing this" pass (same
contamination caveat applies; not re-run here because a second full pass would cost another
~50 min of contended time for a curve shape that one pass already establishes):

| ticker | filings (=E) | fact rows | wall clock | marginal s/event |
|---|---|---|---|---|
| VRT | 4 | 250 | 9.10 s | — |
| VRT | 8 | 540 | 18.71 s | 2.40 |
| VRT | 12 | 1,103 | 32.79 s | 3.52 |
| VRT | 16 | 1,509 | 41.65 s | 2.22 |
| MCD | 16 | 1,630 | 60.67 s | — |
| MCD | 32 | 3,285 | 166.01 s | 6.58 |
| MCD | 48 | 4,941 | 274.15 s | 6.76 |
| MCD | 69 | 6,991 | 323.69 s | 2.36 |

Growth over E=16 -> 69 on MCD is **~E^1.15** — mildly superlinear, nothing like `E^2`. Marginal
cost per event is **not monotone** (6.58 -> 6.76 -> 2.36 s/event), which no complexity model
explains on its own and which contention noise is the most likely explanation for. **Do not
promise a quadratic-term win until this curve is re-measured on a quiet machine.**

## 3. cProfile top-20 cumulative, MCD, full history (69 filings, 6,991 fact rows, 69 events)

Profiled run: **1354.05 s wall**, 209,315,516 function calls (204,871,599 primitive). This is
~4x the un-profiled tier-B-implied per-ticker cost for MCD (~324 s in §2) — cProfile's own
instrumentation overhead plus the same contention. The *shape* (where time goes), not the
absolute seconds, is the useful part of this table:

```
ncalls   cumtime  function
     1  1354.130  build_ticker  (build_history.py:785)
    69  1347.717  _snapshot  (build_history.py:584)
    69  1092.519  build_periods  (periods.py:941)
   857   745.088  quarterize  (periods.py:530)
454369   506.562  DataFrame.__getitem__  (pandas frame.py:4337)
   857   434.267  _ladder  (periods.py:614)
   857   337.674  trailing_twelve  (periods.py:793)
 22810   288.551  _same_start_before  (periods.py:377)
 57535   280.999  DataFrame._getitem_bool_array  (pandas frame.py:4406)
 64068   262.232  BlockManager.take  (pandas managers.py:1052)
 58961   255.105  NDFrame.take  (pandas generic.py:4010)
 69360   242.877  BlockManager.reindex_indexer  (pandas managers.py:800)
509832   226.310  DataFrame._ixs  (pandas frame.py:4292)
503597   212.517  Block.take_nd  (pandas blocks.py:995)
714085   200.252  take_nd  (pandas array_algos/take.py:57)
422766   182.182  DataFrame._get_item  (pandas frame.py:4966)
244514   161.227  new_method  (pandas ops/common.py:67)
183475   142.507  Index.__getitem__  (pandas indexing.py:1192)
 30772   132.608  DataFrame.sort_values  (pandas frame.py:8128)
  1794   129.396  _instant  (build_history.py:559)
```

Confirms the module's own docstring on `PERIOD_COLUMNS`/`_period_projection`: the cost is
**pandas per-slice indexing inside `quarterize`/`_ladder`/`trailing_twelve`**
(`__getitem__`/`take`/`_ixs`/`reindex_indexer`, ~450k-720k calls each), not a single
identifiable O(E^2) loop. This is a REAL 69-filing filer, unlike the research's synthetic
E=12 profile, and the ranking is consistent with it: indexing overhead dominates, exactly
where Phases 1-4 should look first.

## 4. Peak RSS per ticker (isolated, one subprocess per ticker, full history)

Measured via each child process's own `PeakWorkingSetSize` (Windows;
`resource.getrusage` is POSIX-only) right before it exits, so each number is that ticker's
replay alone, not contaminated by the other 7:

| ticker | fact rows (full) | events | peak RSS |
|---|---|---|---|
| VRT | 3,319 | 34 | 116.8 MB |
| MCD | 6,991 | 69 | 125.4 MB |
| KR | 7,576 | 70 | 127.3 MB |

Flat across a ~2x range of fact rows: the interpreter + pandas/pyarrow import baseline
dominates, not the ticker's own data. **Phase 4's process pool is not memory-bound** at this
sample size — N workers cost roughly `130 MB x N`, which is cheap next to any reasonable
worker count; the pool should be sized on CPU/wall-clock, not RAM.

## 5. The two `--source db` traps (`compare_against_stored`, `verify_live_matches_manifest`)

Both read-only, run once against the tier-B (uncapped) freeze:

- **Moving-target guard**: `verify_live_matches_manifest` on all 8 tickers -> `moved=[]`. The
  freeze taken for this baseline still matches the live `fundamentals_facts` row counts at
  check time.
- **Stored-vs-rebuilt drift** (`diff_against_stored`, DATE-round-trip-safe): checked on VRT and
  MCD.
  - VRT: **0** drifted cells.
  - MCD: **38** drifted cells, all shaped `stored=NaN -> rebuilt=<value>` at
    `as_of=2011-11-04` (`fiscal_quarter`, `totalRevenue`, `sellingGeneralAdmin`, ...). This is
    **pre-existing staleness in the stored `fundamentals_history_sec`**, not something this
    session's changes caused (Phase 0 touches no `src/`): MCD's stored history was built
    before some of its earlier (2011-era) facts backfilled into `fundamentals_facts`, so a
    fresh rebuild now resolves values that were null when it was last built. Flagged here as a
    finding, not fixed here — fixing it means re-running `build_fundamentals_history` for MCD,
    which is outside a phase that touches no `src/` and no tables.

## 6. Determinism (the actual gate, run twice at the same HEAD)

| Tier | Tickers | Cells differing | Codes added/removed |
|---|---|---|---|
| A | all 8 | **0** | 0 / 0 |
| B | all 8 | **0** | 0 / 0 |

Both double-replays used `tests/data_extract/fundamentals/replay_equality.py compare`, the
same gate Phases 1-4 will be judged against. No non-determinism found — dict ordering,
`groupby(sort=)` and float accumulation order are all stable here today. If Phase 1+ ever
trips this, it is a new defect, not a pre-existing one.

## Bottom line for Phases 1-4

- The harness works and is deterministic on real data today; it is a real gate, not a
  hypothetical one (§6, and the three planted-defect tests in `test_replay_equality.py`).
- The absolute timings in §1-2 cannot be trusted as a clean baseline — re-measure on a quiet
  machine before quoting any speed-up ratio, per the plan's own instruction.
- The cProfile evidence (§3) says the cost is indexing overhead inside `quarterize`/`_ladder`/
  `trailing_twelve`, not one identifiable quadratic hot loop — a call-count argument for
  `O(E^2*K)` is not the same claim as a wall-clock one, and this profile does not confirm the
  latter.
- Memory (§4) is not the constraint for Phase 4's process-pool design; wall clock is.
- §5 surfaced one real, pre-existing, out-of-scope finding (MCD stored-history staleness) as a
  side effect of exercising the db-mode comparison — worth a follow-up ticket, not a Phase 0
  fix.
