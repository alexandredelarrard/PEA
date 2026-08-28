# Research: Refactor / simplify `src/data_extract/utils/fundamentals` and its dependencies

**Date**: 2026-08-27
**Research Phase**: 1 of 3 (FIC Workflow)
**Next Phase**: Planning (`/plan`)
**Branch**: `feature/refactor-fundamentals` @ `e8740ad`; working tree clean except the untracked spec
**Spec**: `specs/2026-08-26/refactor-fundamentals.md`

## Research Question

How to refactor `src/data_extract/utils/fundamentals` and all its dependencies to be faster, more
efficient and cleaner — fewer globals, smaller functions — against three stated goals:
readability/simplification, performance/efficiency, maintainability. Plus a specific question:
`fetch_fundamentals_sec` is slow and emits many connection / not-found messages — **is there a bug?**

Method: 8 parallel documentarian agents over the 12 fundamentals modules, `constants.py`,
`run_manifest.py`, the `Step`/`Context` layer, the sibling SEC fetchers and the test suite;
plus direct measurement of the live run log, the DB, the extraction manifest and the installed
edgartools package. Every claim carries a `file:line` or a measured number.

---

## Summary

**Yes, there are bugs — three are live and one is silently destroying data.**

1. **A `NameError` kills whole tickers.** `xbrl_linkbase.py:514` returns
   `pd.DataFrame(columns=cols)`; `cols` is undefined (the three sibling early-returns at 439, 441
   and 504 correctly use `ARC_COLUMNS`). In the run that started 2026-08-27 00:06 it fired for
   **NEM, MO and AIZ**, which produced **zero facts** while the run reported success — because
   `run_per_ticker` requires the worker to swallow its own exceptions
   (`parallel_fetch.py:41-44`) and `filing_rows`' bare `except Exception` (`:660-663`) turns it
   into "unreadable XBRL → `[]`". No test drives that branch.
2. **The cube's whole workforce block is dead.** `step_cube_fundamentals.py:179-180` passes the
   `fundamentals_history` frame as `headcount_history`; `employee_features.py:40` reads column
   `"employees"`; the merged table declares **`employees_sec`** (`schema.py:210`) after the
   phase-0 rename. `fundamentals_to_daily` returns empty for a missing column instead of raising.
3. **`fetch_earnings_surprises` records its run twice** — `:149` records 0 rows on the empty
   branch with **no `return`**, then falls through to `:154-156`.

**The messages the user is seeing are 97% benign library noise, but the walk is genuinely slow —
and the slowness is not where the warnings are.** The 380 warnings in the live log are
edgartools' own (`edgar/sgml/sgml_common.py:238`, `edgar/_filings.py:2013`); 301 are cosmetic.
The 76 `SGML fetch failed … falling back to homepage` lines each cost up to **7 extra GETs**.
But the dominant cost is elsewhere, and it is architectural:

| Stage | Cost | Parallel? |
|---|---|---|
| `fetch_fundamentals_sec` | `filing.xbrl()` at 1.4–5.8 s × ~30k filings | **8 workers** |
| `build_fundamentals_history` | **O(E²·K)** — profiled 213.5 s for a 12-event synthetic ticker | **single-threaded** |

`build_ticker` rebuilds **the entire period engine from scratch on every publication event**.
Its own docstring claims "The replay is O(filings)" (`build_history.py:789`); measured, it is
quadratic. 99.5 % of the time is in `_snapshot`, 79 % in `build_periods`, 61 % in `quarterize`.

**The `full=True` hardcode makes every `main.py` run a full 31-year rescan.**
`step_extract_all_data.py:55-56` passes `full=True` literally, `run()` takes no `full` parameter,
and 3 of the 5 sub-steps are commented out. `full=True` bypasses `manifest_window` entirely
(`edgar_driver.py:76-86`).

**On the three stated goals, measured against `StepExtractPrices` as the reference:**

| | `prices/` (reference) | `fundamentals/` |
|---|---|---|
| LOC / files | 1,640 / 10 | **6,593 / 12** |
| Files > 500 LOC | **0** | **5** (max 1,516) |
| Functions ≥ 80 LOC | **0** of 60 | **9** of 186 (max **202**) |
| Module globals | 28 | **124** |
| Files reading config themselves | 1/10 | 4/12 (+4 reading config *files*) |
| `record_run` call sites | 11, in 7/10 files | **4, in 2/12** — both main fetchers: zero |
| Comment+docstring : code | — | **1.39 : 1** (`xbrl_linkbase.py` **1.76 : 1**) |

**`constants.py` is not just big, it is 30 % dead.** Of 154 symbols, **46 have no `src/` consumer
at all** and **80 have exactly one**. Meanwhile `fundamentals/` defines **89 of its own**
module-level constants against **12** imported from `constants.py`.

**The `./configs` literal is a symptom, not the disease.** `get_config_context` accepts
`config_path` and then hardcodes `read_config(path="./configs")` at `context.py:129`, using the
argument only in an error message. The CLI's `-c` **never reaches the config reader**. Neither
`Context` nor `Step` exposes a config dir at all, which the code already records at
`fetch_fundamentals_sec.py:869-871`.

---

## Part A — Live defects and dead code

### A.1 The three live bugs

| # | Defect | Evidence | Impact |
|---|---|---|---|
| 1 | `cols` is undefined | `xbrl_linkbase.py:514` vs `ARC_COLUMNS` at `:439, :441, :504` | 3 tickers (NEM/MO/AIZ) produced 0 facts in the 2026-08-27 run; logged as a warning, run reported success |
| 2 | `employees` vs `employees_sec` | `step_cube_fundamentals.py:179-180` → `employee_features.py:40`; `schema.py:210` | `employee_growth`, `revenue_per_employee` and the whole workforce panel silently empty |
| 3 | `record_run` fall-through | `fetch_earnings_surprises.py:146-156` — `:149` has no `return` | manifest entry written twice per call |

### A.2 Dead code, by module

| Symbol | Location | Evidence |
|---|---|---|
| `ONLY_WHEN_SIBLING` | `xbrl_linkbase.py:839` | no reference in `src/`, `tests/`, `scripts/`; only `ONLY_WHEN_DESCENDANT` is compared (`:890`) |
| `us_gaap_only` | `entity_scope.py:142` | sole occurrence is its own `def` |
| `dimensioned_facts` | `entity_scope.py:232` | sole occurrence is its own `def` |
| `ENTITY_AXES` | `entity_scope.py:39` | never read; its own comment (`:36-38`) says the default path does not consult it, and its only stated user (`dimensioned_facts`) is itself dead |
| `Catalogue.regime_for_sub_industry` | `kpi_catalogue.py:365` | sole occurrence is its own `def` |
| `COMBINED_INTO` | `reason_codes.py:68` | no producer; only self-reference at `:184`. `Catalogue.combined_into` (`:514`) returns `None` for every cell today, stated at `:524` |
| `BASIS_EX_IPRD` | `reason_codes.py:97` | the *string* is emitted from `fundamentals_kpis.json:602`, never from the constant |
| `Kind`, `Sign`, `EXTRACTED_KINDS`, `SCORED_TIERS` | `kpi_catalogue.py:61, 62, 66, 71` | never imported anywhere |
| `_VALUE_KEY`, `_QUARTER_COLUMNS`, `_RATIOS`, `_EQUITY_INCL_NCI`, `_CODE_COLUMNS`, `PERIOD_COLUMNS`, `FACT_COLUMNS` | `build_history.py:92, 116, 121, 386, 868, 895, 988` | no external importer |
| `AS_REPORTED`, `Q2_FROM_YTD6`, `Q3_FROM_YTD9`, `FY_MINUS_YTD9`, `FY_MINUS_QUARTERS`, `TTM_MIN_DAYS`, `TTM_MAX_DAYS`, `_SAME_PERIOD_DAYS`, `_CONTIGUOUS_DAYS`, `_QUARTER_COLUMNS`, `_YEAR_END_LABEL`, `_DURATION_BANDS` | `periods.py:118-122, 171, 175, 177, 232, 904, 87` | no external importer |
| `DEF14A_LLM_PATH`, `SEC_13F_INSIDERS_DIR` | `context.py:45, 49` | **0 reads anywhere**; the `_meta.json` sidecar they exist to anchor (`context.py:34-49`) **is not implemented by any fetcher** |
| 46 symbols in `constants.py` | see Part D | 0 `src/` consumers |
| 10 `Catalogue` members, 7 `Catalogue` methods | see Part I.4 | reachable only from `tests/`, or from nothing |

**Name collisions:** `_QUARTER_COLUMNS` exists in both `build_history.py:116` (dict, 2 keys) and
`periods.py:177` (tuple, 12 items) — two modules that import each other. `_latest_per_window`
exists in `periods.py:196` and `validate/.../tier3_internal.py:508` with **different window
identities**. `_normalise` exists 4× (`build_history.py:914`, `ledger.py:401`,
`substrate.py:228`, `refactor_metrics.py:125`).

### A.3 Orchestration defects

- **3 of 5 sub-steps commented out**, `full=True` hardcoded: `step_extract_all_data.py:53-57`.
  `run()` takes no `full` (`:50`), so `main.py` can only do a full rebuild.
- **One `full` flag drives two different semantics**: `full=full` on the fetcher (skip the
  manifest window) and `rebuild_history=full` on the replay (DELETE then rebuild) —
  `step_extract_fundamentals.py:40, 48`.
- **`fetch_financial_statements` is off the Step chain.** It writes `pension_facts` (`:167`) and
  is reached only by `cli.py:371-373` and `dag_data_extraction.py:89`; the five siblings are
  wired at `step_extract_fundamentals.py:19-23`. It also records no run.
- **`Step._log` is bound to `src.utils.step`** (`step.py:19`), so every step's log lines are
  attributed to `step.py`. `self._log` is used **twice** in all of `src/data_extract/`
  (`step_extract_all_data.py:45, :47`); the 5 transformer steps log **nothing**.
- **`notes_text` has no reader anywhere** (`fetch_financial_notes.py:316` writes it), and the
  download is the heaviest in the repo — "~300-450MB EACH … ~26GB back-fill"
  (`fetch_financial_notes.py:44-46`). `constants.py:536-539` and `schema.py:515-517` both record
  that the consumer "was never wired into any panel and has been removed".

---

## Part B — Efficiency: the measured cost model

### B.1 The replay is quadratic (the single biggest finding)

`build_ticker` (`build_history.py:785-861`) drives `for _, event in events.iterrows():` (`:809`)
and hands `_snapshot` the **whole visible prefix** each time (`:814-816`). `_snapshot:597` then
calls `build_periods`, which re-runs `quarterize` + `trailing_twelve` for all 22 duration fields
over that entire prefix.

Measured call counts (synthetic filer, 4 filings/yr, `catalogue=./configs`, K=22 duration fields,
K_inst=26 instant fields):

| callee | E=8 | E=12 | E=16 | scaling |
|---|---|---|---|---|
| `_snapshot` / `build_periods` | 8 | 12 | 16 | E |
| `fiscal_year_ends` | 16 | 24 | 32 | **2·E** (twice per event: `periods.py:967` and `build_history.py:604`) |
| `quarterize` / `_ladder` / `trailing_twelve` / `label_fiscal_periods` | 176 | 264 | 352 | E·K |
| `_latest_per_window` | 880 | 1320 | 1760 | **5·E·K** |
| `carry_latest_known` / `_instant` | 208 | 312 | 416 | E·K_inst |
| **`_is_ambiguous_duration`** | 594 | 1320 | 2310 | **E²·K** |
| **`_same_start_before`** | 528 | 1188 | 2112 | **E²·K** |
| **`_derived`** | 462 | 1034 | 1870 | **E²·K** |

Wall clock, uninstrumented: 28.85 s (E=8) → 40.67 s (E=12) → **89.03 s (E=16)**.

cProfile at E=12 (213.5 s total, 26.0 M calls):

| function | ncalls | cumtime | % |
|---|---|---|---|
| `build_ticker` | 1 | 213.52 | 100 |
| `_snapshot` (`build_history.py:584`) | 12 | 212.44 | **99.5** |
| `build_periods` (`periods.py:941`) | 12 | 168.85 | **79.1** |
| `quarterize` (`periods.py:530`) | 264 | 131.08 | **61.4** |
| `_ladder` (`periods.py:614`) | 264 | 38.97 | 18.3 |
| `_shape` (`periods.py:225`) | 1,056 | 35.97 | 16.8 |
| `trailing_twelve` (`periods.py:793`) | 264 | 35.62 | 16.7 |
| `_latest_per_window` (`periods.py:196`) | 1,320 | 33.21 | 15.6 |
| `carry_latest_known` (`build_history.py:224`) | 312 | 27.28 | **12.8** |
| pandas `DataFrame.__getitem__` | 36,563 | 67.58 | 31.7 |
| pandas `_getitem_bool_array` | 7,153 | 40.49 | 19.0 |

A real S&P 500 filer over 31 years has E ≈ 60, i.e. ~25× the E=12 cost. The code already knows
the order of magnitude: `build_history.py:589-592` records a prior state at
**"~14 minutes a ticker"**, and `periods.py:335-341` records that one function was
">50% of the entire period engine".

**And it is single-threaded.** `build_fundamentals_history:1052` is a plain
`for ticker in tickers:` — no `run_per_ticker`, no pool, while the sibling fetch uses 8 workers
(`fetch_fundamentals_sec.py:901`).

### B.2 Repeated work inside loops

| What | Site | Loop | Independent of the loop variable? |
|---|---|---|---|
| The whole period engine | `build_history.py:597` | event loop `:809` | every earlier event's quarters are recomputed identically |
| `fiscal_year_ends` twice per event | `build_history.py:604`, `periods.py:967` | event loop | yes |
| `visible.sort_values("filing_date")` | `build_history.py:599` | event loop | `_normalise:930` already sorted it |
| `pd.to_datetime(period_of_report)` over the prefix | `build_history.py:552` | event loop | `_normalise:921` already coerced it |
| `pd.to_datetime` on 3 date cols | `periods.py:569-571` | E·K | `_normalise:920-921` already coerced them |
| `_latest_per_window` re-sorts the group | `periods.py:219` | 5× per (event, field) | `quarterize:569-571` already sorted |
| `_fiscal_bounds` re-`sorted()`s `year_ends` | `periods.py:719` | E·K + E | `fiscal_year_ends:685` already sorted |
| `_annual_by_end` re-derives the annual shape | `periods.py:860` | E·K | `quarterize:587` already computed it |
| `carry_latest_known` builds a 1-row DataFrame + `merge_asof` **for a single date** | `build_history.py:242-256` | 26·E | 12.8 % of the profile for 312 single-date lookups |
| `_facts_code` / `_has_valued_fact` / `_qualifiers` each re-filter `visible["field"] == field` | `build_history.py:304, 355, 360` | E×52 | one `groupby("field")` would serve all three |
| `_contradicts_gross_profit` full `pivot_table` | `build_history.py:413` | per event | recomputed E times over a near-identical frame |
| `load_guards()` | `periods.py:564, 816` | E·K | masked by `@lru_cache(maxsize=4)` at `:69` |
| **`calculation_linkbase()` twice per filing** | `fetch_fundamentals_sec.py:697` and again via `statement_arcs` → `xbrl_linkbase.py:502` | per filing | `edgar/xbrl/xbrl.py:304` has **no** cache decorator. Honest size: 0.003–0.006 s × 23k filings ≈ 70–140 s (`xbrl_linkbase.py:106-108`) — cleanliness, not the bottleneck |
| `catalogue.extracted_fields` (`sorted()` over 53 fields) | `fetch_fundamentals_sec.py:713` | per filing | answer never varies (48) |
| `_candidates` rebuilt | `xbrl_linkbase.py:1065, 1161, 1452` | up to **7×** per (filing, field) | yes |
| `spec.never_use(regime)` builds a fresh merged dict | `xbrl_linkbase.py:815, 1165, 1451` | up to **10×** per (filing, field) | yes |
| `_leaf_sum` prologue + `catalogue.filer_leaves()` | `xbrl_linkbase.py:1198-1200` | all 48 fields | only **3** (`capex`, `costOfRevenue`, `depAmort`) declare the `roll_up.any_of` that makes route 3b applicable |
| `getattr(filing, "period_of_report")` | `fetch_fundamentals_sec.py:625` | **per ROW** (hundreds/filing) | plain `@property` calling `self.sgml()` each access (memoised after the first) |
| `pd.Timestamp(filing.filing_date)` | `fetch_fundamentals_sec.py:618, :623` | twice per row | yes |

`Catalogue.field(name)` is O(1) (`kpi_catalogue.py:332`), but **every derived view is an uncached
linear scan + sort**: `history_fields` (`:295`), `side_table_fields` (`:283`), `history_columns`
(`:319`), `scored_fields`/`input_fields`/`extracted_fields`/`unverified_fields`, `by_tier`,
`never_use` (fresh dict per call), `filer_leaves` (fresh frozenset per call). The docstring at
`:258` claims "with lookups precomputed"; only `fields` is. `history_fields` is re-evaluated
once per publication event per ticker (`build_history.py:619`).

### B.3 Network cost

`xbrl_linkbase.py` is **entirely offline** — no `requests`/`urllib`/`httpx`/`Company`/`get_filings`.
All network on this path goes through the `filing` object.

**Request cost for one ticker with N new filings** (P = pagination files in the submissions JSON,
C = 2 if the ticker has a `cik_cutover` entry else 1, `h_f` = homepage needed for
`period_of_report`, `b_f` = SGML fell back to homepage, `a_f ∈ [0,6]` = attachments downloaded
individually):

```
requests(ticker) = C·(1 + P) + Σ_f [ 1 + h_f + b_f + a_f ]
nominal:          1 + P + N          (1 request per filing)
worst observed:   1 + P + 8·N
```

The single request is `filing.xbrl()` (`fetch_fundamentals_sec.py:661`) → `XBRL.from_filing` →
`filing.attachments` → `filing.sgml()` → **1 GET of `{base_dir}/{accession}.txt`**. All six
linkbase/instance attachments are in-memory slices of that one download. On fallback, the
homepage-built `FilingSGML` has "valid URLs but without in-memory content"
(`edgar/_filings.py:1970-1972`), so each attachment becomes its own GET.

Rate ceiling: **9 req/s process-global** (`edgar/httpclient.py:434`), shared across the 8
configured workers (`configs/configs.yml:26`). **The repo sets no retry, backoff, timeout,
throttle or `EDGAR_*` env var anywhere** — 0 matches repo-wide, including `.env*`.
`common/rate_limit.py::call_with_retries` exists but is used only by the yfinance/Trends paths.

### B.4 Caching

| Cache | Where | Keyed on | Invalidated? |
|---|---|---|---|
| edgartools disk cache | `~/.edgar/_tcache/` (`edgar/httpclient.py:120-122`) — **94,468 files / 49 GB measured** | request URL | `/Archives/edgar/data` → **forever** (the `.txt` submission *and* the `-index.html`); `data.sec.gov/submissions` → 30 s |
| `load_catalogue` | `@cache`, `kpi_catalogue.py:592` | `config_dir` **argument object** | never |
| `load_cutovers` | `@cache`, `cik_cutover.py:76` | same | never |
| `load_guards` | `@lru_cache(4)`, `periods.py:69` | same | never |
| `ArcGraph` 6 × `cached_property` | `xbrl_linkbase.py:547, 557, 580, 592, 603, 622` | instance (1/filing) | dies with the filing |
| `Filing.html()` | `@lru_cache(maxsize=4)`, edgartools | the Filing object | **holds up to 4 Filings + their full SGML alive** |
| **Not cached** | `Filing.xbrl()` (`_filings.py:1859`), `XBRL.calculation_linkbase()` (`xbrl/xbrl.py:304`) | — | — |
| `load_field_map` | `field_map.py:315` — **not cached**, and it calls `load_catalogue(config_dir)` internally at `:326` | — | — |
| `fundamentals_rosters.json` | 3 independent uncached readers: `validate/cli.py:133`, `sweep_fundamentals_resolution.py:66`, `report_fundamentals_sweep.py:598` | — | — |

**A `functools.cache` subtlety with a real cost**: `f()` and `f("./configs")` are **distinct cache
keys**. Both conventions exist — no-arg at `fetch_fundamentals_sec.py:872`,
`build_history.py:796, :1050`; explicit at `field_map.py:326`, `validator.py:244`. So one
`StepExtractAllData.run()` parses the 169 KB catalogue (3 JSON files) **twice** and re-runs all
six validation passes each time.

Because `/Archives` is cached forever, **a re-run is parse-bound, not network-bound** — which is
why the replay's quadratic CPU cost dominates any realistic re-run.

The cache has **no TTL, no size cap and no eviction**, and nothing in the repo clears it
(`EDGAR_LOCAL_DATA_DIR` / `EDGAR_USE_LOCAL_DATA` / `use_local_storage` / any cache-clear → **0**
matches repo-wide, `.env*` included), so it grows monotonically — hence 49 GB. It lives at
`~/.edgar` only because `EDGAR_LOCAL_DATA_DIR` is unset (`edgar/core.py:322-327`).

Two further library defaults in force, both relevant to the `incomplete chunked read` failures:
`edgar/httpclient.py:71-73` sets `httpx.Timeout(get_edgar_http_timeout(), connect=10.0)`,
overridable only via `EDGAR_HTTP_TIMEOUT` (unset here); and `:56-61` sets **`http2=False`** with
the library's own comment naming the failure mode it avoids — "a mid-stream reset from cloud
egress fails all in-flight requests at once (InvalidBodyLengthError / ConnectionTerminated)" —
i.e. the same class of mid-stream failure now appearing on HTTP/1.1.

### B.5 DB access

| Site | Table | `columns=` | `where=` | In a per-ticker loop? |
|---|---|---|---|---|
| `build_history.py:1053` | `fundamentals_facts` | **yes** (19 of ~28) | yes | yes |
| **`build_history.py:1061`** | `fundamentals_history_sec` | **NO — `columns=None`** → `SELECT *` over 69 columns | yes | yes |
| `build_history.py:1065, :1066` | delete both tables | – | yes | only under `rebuild_history` |
| `build_history.py:1088, :1090` | save both | – | – | yes |
| `fetch_fundamentals_sec.py:877` | `sp500_tickers` | yes | **no** (all tickers) | no |
| `fetch_fundamentals_sec.py:883` | `fundamentals_employees` | yes | **no** (whole table) | no |
| `edgar_driver.py:74` → `sec_utils.py:131` | `sp500_tickers` | **no** (all columns) | yes | no — **the second read of the same table in one run** |
| `edgar_driver.py:91` → `sec_utils.py:87` | `fundamentals_facts` | `distinct(accession_number)`, **no `where`** | – | no |

`since=` / `until=` / `project=` / `order_by=` are **never used** in either file, though
`store.load` supports them (`store.py:435-442`) and `fundamentals_history_sec` declares
`date_col="as_of"` (`schema.py:156`). `iter_load` has **0 occurrences** in `src/data_extract`.
`periods.py` has **zero** DB access.

### B.6 The warnings, precisely

Both are emitted by **edgartools 5.51.0** (`poetry.lock:678`), not repo code — 0 grep hits for
either string in `src/`, `tests/`, `scripts/`.

| Message | Emitter | Guard |
|---|---|---|
| `SGML header declares %d public document(s) but only %d were parsed…` | `edgar/sgml/sgml_common.py:238-242` | `if declared_count and 0 < parsed_count < declared_count - 1` (`:237`) — a deficit of exactly 1 and `parsed_count == 0` are deliberately tolerated (`:225-234`) |
| `SGML fetch failed for {accession}, falling back to homepage: {e}` | `edgar/_filings.py:2013-2016` | `RemoteProtocolError` is **not** in the re-raise list at `:2005-2011`, so "incomplete chunked read" lands in the transient branch → warn + `from_homepage` |

Census of `.log/output_2026-08-27_00.log` (00:06:41 → 10:44:12, 392 lines, 380 warnings, **12 INFO**):

| Count | Shape |
|---|---|
| 301 | `SGML header declares …` (cosmetic) |
| 50 | `SGML fetch failed … SEC returned HTML or XML instead of expected SGML` |
| 15 | `… peer closed connection (incomplete chunked read)` |
| 7 | `… [WinError] connection forcibly closed` |
| 1 | `… peer closed connection (received N bytes, expected N)` |
| **3** | **`fundamentals (linkbase): {NEM,MO,AIZ} failed (name 'cols' is not defined)`** |

Nothing suppresses them and the config surfaces them: `configs/logging.yml:51-53` sets root to
`INFO` with `[console, file_handler]`, `disable_existing_loggers: False` (`:3`), and there are
**0** edgar-specific `getLogger`/`setLevel`/`addFilter` entries in `src/`.

### B.7 Observability of the live run (measured mid-flight)

- Run started `00:06:41`, still appending at `10:44:12` — **10.6 h**; `python.exe` PID 30368 at
  **4.0 GB RSS**.
- **12 INFO lines in 10.6 h** — seed, dotenv, 5 × `starting step`, universe size, CIK cutovers,
  edgar identity, ReverseIndex. tqdm goes to console only; nothing persists progress.
- `data/extraction_manifest.json` still reads `last_run_date: 2026-08-25`,
  **`ticker_count: 1`** for `fundamentals_facts` and `fundamentals_employees` — because
  `record_run` only fires when the walk completes (`edgar_driver.py:143-144`).
  Per `manifest_window:89`, that `ticker_count: 1` **guarantees a full rescan** on the next run.
- DB mid-run: `fundamentals_facts` = 2,249,866 rows / 371 tickers / 23,147 filings /
  `filing_date` 2009-04-15 → 2026-08-26, against a 491-ticker universe.

---

## Part C — Structure and size

### C.1 Per-file, `fundamentals/` vs the `prices/` reference

| file | LOC | funcs | max func LOC (name) | globals | reads config? | records run? |
|---|---|---|---|---|---|---|
| `xbrl_linkbase.py` | **1516** | 40 | **202** (`_resolve_once`) | **25** | no | 0 |
| `build_history.py` | **1092** | 31 | **130** (`_snapshot`) | 13 | no | **0** |
| `periods.py` | **984** | 24 | **102** (`_drop_annual_masquerading_as_quarter`) | **21** | no (calls `read_config`, `:73`) | 0 |
| `fetch_fundamentals_sec.py` | **901** | 20 | **130** (`rows_from_xbrl`) | 5 | yes (`:901`) | **0** (delegates) |
| `kpi_catalogue.py` | **680** | 38 | 89 (`load_catalogue`) | 13 | reads a file (`:599`) | 0 |
| `fetch_financial_notes.py` | 323 | 10 | 47 | 13 | yes (`:288`) | 2 |
| `entity_scope.py` | 239 | 8 | 39 | 5 | no | 0 |
| `reason_codes.py` | 188 | **0** | — | **15** | no | 0 |
| `fundamentals_employees.py` | 179 | 5 | 59 | 4 | no | 0 |
| `fetch_financial_statements.py` | 172 | 3 | 45 | 5 | yes (`:137`) | **0** |
| `cik_cutover.py` | 163 | 4 | 51 | 2 | reads a file (`:90`) | 0 |
| `fetch_earnings_surprises.py` | 156 | 3 | 49 | 3 | yes (`:121`) | 2 (one is a fall-through) |
| **total** | **6593** | **186** | **202** | **124** | **4/12** | **4 sites, 2/12 files** |

`prices/`: 1,640 LOC over 10 files; **no file over 260 LOC**, **no function over 73 LOC**, 28
globals, 1/10 reads config, `record_run` in 7/10 files. The step resolves both time windows and
passes them **into** the fetchers, documented as a rule at `step_extract_prices.py:12-14`.

### C.2 The nine functions ≥ 80 LOC

| LOC | function | file:line |
|---|---|---|
| **202** | `_resolve_once` | `xbrl_linkbase.py:1132` |
| **130** | `_snapshot` | `build_history.py:584` |
| **130** | `rows_from_xbrl` | `fetch_fundamentals_sec.py:669` |
| **126** | `_leaf_sum` | `xbrl_linkbase.py:1349` |
| **102** | `_drop_annual_masquerading_as_quarter` | `periods.py:273` |
| 89 | `load_catalogue` | `kpi_catalogue.py:592` |
| 84 | `resolve_field` | `xbrl_linkbase.py:1013` |
| 82 | `quarterize` | `periods.py:530` |
| 82 | `_total_liabilities_identity` | `build_history.py:452` |

Plus two large classes: `ArcGraph` **186 lines** (`xbrl_linkbase.py:536`, 6 `cached_property` +
12 methods) and `Resolution` **118 lines** (`:300`, 15 dataclass fields carrying 70 `#:` lines).

**Nesting ≥ 4:** `_resolve_once` at `:1304`; `rows_from_xbrl` at `:747`;
`_drop_annual_masquerading_as_quarter` at `:356`; `facts_frame_from_companyfacts` at `:959`;
`build_fundamentals_history` at `:1073`.

### C.3 Branch counts in the period engine

44 branch points across 9 functions, producing 5 quarter bases, 3 TTM bases and 5 refusal codes:
`quarterize` 6 terminal paths (`:565, :576, :593, :595, :610`), `_ladder` 5
(`:629, :643, :647, :658, :662`), `_derived` 4 (`:497, :511, :517, :520`), `_scale_agrees` 4,
`_is_coherent` 5, `_drop_annual_masquerading_as_quarter` 6, `_is_ambiguous_duration` 3,
`trailing_twelve` 5 (`:815, :824, :831, :835, :842`), `build_periods` 6.

`PeriodGuards` (`periods.py:58-67`) holds 3 floats; **two of the three field names differ from
their YAML keys** (`max_opposite_sign_ratio` ← `max_opposite_sign_q4_ratio`,
`concept_switch_scale_max` ← `q4_tag_mismatch_fy_max`), with the mapping existing only in the
constructor at `:75-77`. Source: `configs/configs.yml:41-66`.

---

## Part D — Globals and `constants.py`

### D.1 `constants.py` (1,058 lines, 154 symbols)

Composition: 612 comment lines (57.8 %), 95 blank, 6 docstring, 1 import, **344 lines carrying a
value** (32.5 %). Growth over the Sharadar work: 837 → 1,058. `docs/coding_standard.md:18` still
says "(927 lines)" — stale by 131.

| Bucket | Count | Share |
|---|---|---|
| **0 `src/` consumers** | **46** | 29.9 % |
| Exactly 1 `src/` consumer file | **80** | 51.9 % |
| 2+ consumer files | 28 | 18.2 % |
| Used only by tests | 0 | — |
| Fully dead (0 src, 0 tests, 0 scripts, no self-ref) | **38** | — |

Verified independently: no `import *`, no `getattr(constants, …)` anywhere, and spot-checks of
`SHARES_OUTSTANDING_MIN`, `EPS_ABS_MAX`, `Q4_RECONCILIATION_TOLERANCE`, `NOTES_THEME_TAGS`,
`EFFECTIVE_TAX_RATE_MIN` return **0** `.py` references outside their own definition.

The 46-symbol dead bucket, grouped:

| Group | Count | Lines | Note |
|---|---|---|---|
| Plausibility / tolerance scalars | 23 | `843-1023` (194 lines, 148 of them comment) | appear only in `docs/coding_standard.md:24` and two `reports/planning/` files |
| `GICS_SECTOR_*` / `GICS_GROUP_*` | 10 | `794-804` | 7 feed only `SECTOR_KPI_SCOPE` in the same file; **all 10 names are re-typed as dict keys in `common/gics.py:19-85`**, which is the module actually imported |
| `NOTES_*` narrative | 3 | `541-578` (33 lines) | the section's own comment says the consumer "has been removed" |
| Sharadar zero-rule | 4 | `370-394` | declared for `diagnostics.py`, which does not reference them |
| Data-freshness | 2 | `764-770` | the header says the gate that read these was removed; `schema.py::freshness_tables()` also documents "**NO caller**" |
| SEC URLs | 2 | `42-43` | `sec_utils.py:124` documents the derived table was dropped |
| Other | 2 | `740, 929` | plus `FUNDAMENTALS_ROSTERS_FILENAME` (`:112`), used only in `scripts/` |

**The fundamentals slice:** 47 symbols (120 of 365 definition lines), **45 of them exclusive** to
`fundamentals/**` + `fundamentals_sharadar/**`. `src/validate/**` consumes exactly **2**, both in
`validate/cli.py:79`; nothing under `validate/fundamentals/` (2,891 LOC) imports from
`src.constants`. `field_map.py` is the sole consumer of **16** symbols.

`constants.py` contains **no** functions, classes, comprehensions, or I/O at import time — only 2
computed expressions (`MACRO_ALL_SERIES:732`, `EFFECTIVE_TAX_RATE_MIN:1004`).

### D.2 Duplication inside and against `constants.py`

**Same literal under multiple names** (10 groups) — e.g. `5.0` = `OPERATING_MARGIN_ABS_MAX`,
`PROFIT_MARGIN_ABS_MAX`, `HEADCOUNT_CONTINUITY_MAX`, `FUNDAMENTALS_DISCONTINUITY_MAX`;
`"text-embedding-3-small"` = `EARNINGS_CALL_EMBED_MODEL` + `NOTES_EMBED_MODEL`.
`FUNDAMENTALS_DISCONTINUITY_MIN/MAX:954` self-declares as a clone of `HEADCOUNT_CONTINUITY_*`.

**Six hand-maintained subsets of one 112-name vocabulary** (91 definition lines):
`SHARADAR_SF1_COLUMNS:141` ⊃ `SHARADAR_ID_COLUMNS:168` (literally its first 7 elements),
`SHARADAR_ZERO_FILLED_FIELDS:187`, `SHARADAR_FLOW_FIELDS:288`,
`SHARADAR_NON_NEGATIVE_FIELDS:308`, `SHARADAR_EVENT_FIELDS:370`,
`SHARADAR_DIAGNOSTIC_EXTRA_COLUMNS:278`.

**Reverse violations — literals hardcoded where the standard says `constants.py`:**

| Kind | Sites |
|---|---|
| `DATE_FORMAT = "%Y-%m-%d"` re-typed inline | **11** sites incl. `store.py:161`, `step_train.py:558, :759` |
| `DATE_FORMAT_COMPACT = "%Y%m%d"` re-typed inline | **8** sites incl. `fetch_financial_notes.py:190, :203, :219`, `fetch_financial_statements.py:75, :92` |
| SEC URL templates defined locally | `fetch_financial_notes.py:73`, `validate/fundamentals/finding.py:76`, `fetch_fails_to_deliver.py:35-36` |
| Form lists restated | `build_history.py:49 FORM_PRECEDENCE` (= `FUNDAMENTALS_FORMS`), `fetch_fundamentals_sec.py:176 _ANNUAL_FORMS`, `fundamentals_employees.py:46 HEADCOUNT_FORMS` (byte-identical to the previous) |
| Bounds/tolerances local while `constants.py` twins sit unused | `build_history.py:74-79 HARD_GUARDS` (vs unused `SHARES_OUTSTANDING_MIN/MAX`), `:393 GROSS_PROFIT_IDENTITY_TOLERANCE`, `:53 MAX_AMENDMENT_LAG_DAYS`, `:269 TTM_STALENESS_DAYS`, `periods.py:172, :175, :232` |
| **Table names as constants** (explicitly forbidden) | `fetch_financial_notes.py:70-71 _NUM_TABLE/_TXT_TABLE`, `fetch_financial_statements.py:42 _TABLE = "pension_facts"` — while `Tables.pension_facts` exists at `schema.py:282` |

**Scale:** 89 module-level `UPPER`/`_UPPER` assignments across `fundamentals/**` against the
**12** symbols it imports from `constants.py`.

**Dead doc reference:** `constants.py:895` anchors `Q4_RECONCILIATION_TOLERANCE` on
"`_TO_COMMON_TOL` (0.02, fetch_fundamentals.py)". Neither exists.

---

## Part E — `config_dir`, `Step`, `Context`

### E.1 The root cause

`get_config_context(config_path, …)` accepts the path and then calls
`read_config(path="./configs")` at **`context.py:129`**, hardcoded; `config_path` is used only in
the error message at `:133`. **The CLI's `-c` never reaches the config reader.** The code already
records this at `fetch_fundamentals_sec.py:869-871`:

> "`Context` exposes no config directory (it is a named risk zone, and the CLI's `-c` never
> reaches it), so the catalogue loads from its own default"

Neither `Context` (`context.py:64-123`, 4 properties: `config`, `use_cache`, `save`,
`random_state`) nor `Step` (`step.py:14-24`, 5 attributes) holds a config dir. `config_dir` has
**0** occurrences in either file.

### E.2 Five independent declarations of the same string

`kpi_catalogue.py:51 DEFAULT_CONFIG_DIR`, `cik_cutover.py:77` (bare literal),
`validator.py:235` (bare literal, though it imports from `kpi_catalogue` at `:57`),
`command_line_interface.py:12`, `context.py:129`. Plus `app/app.py:46`, `main.py:10`,
4 `scripts/` argparse defaults, and 13 test modules at module scope.

**17 functions repo-wide take a `config_dir` parameter.** Two conventions coexist in sibling
steps: `StepExtractFundamentalsSharadar.run` threads `config_dir` explicitly
(`step_extract_fundamentals_sharadar.py:37, :79`), while the SEC path calls `load_catalogue()`
with no argument (`fetch_fundamentals_sec.py:872`). `StepExtractAllData` forwards nothing
(`step_extract_all_data.py:56`). **20 distinct production sites** would need to change for one
source of truth.

The Airflow chain works only by coincidence: `dag_data_extraction.py:65` passes
`-c /opt/airflow/project/configs` with `cwd=/opt/airflow/project`, so the absolute flag and the
relative `"./configs"` default resolve to the same directory — but the flag itself is still
discarded at `context.py:129`.

### E.3 Clients and connections re-derived rather than inherited

| Resource | Verdict |
|---|---|
| DB engine / store | **Clean.** `get_engine()`/`DataStore(...)` appear only at `context.py:14, :82` and `utils/db.py:41`; **0** occurrences in `src/data_extract/` |
| EDGAR identity | `set_identity` at `edgar_driver.py:41`, called per fetch-run from `:73` and `fetch_13f.py:125`. `SEC_USER_AGENT` is read in **2 files** with 2 hand-written `RuntimeError`s (`edgar_driver.py:34-40`, `sec_utils.py:33-40`) |
| HTTP sessions | **9 sessionless `requests`/`curl_cffi` call sites**; `sec_utils.py:64` re-reads `os.getenv` on **every** request. `src/utils/polite_http.py` and `crawler.py` are both sessionless too and unreachable from `Context` |
| Sharadar key | `os.getenv` per `sharadar_get` call (`client.py:60, :134`) |
| OpenAI | **3 independent constructions** with 2 different key-precedence orders: `llm_extractor.py:43`, `data_peers/utils/embeddings.py:35`, `utils/openai_embeddings.py:25` |
| FRED | `Fred(api_key=os.getenv(...))` **per `_fred_frame` call** (`fetch_macro.py:133`) |

**28 `os.getenv`/`os.environ` sites, 21 distinct variables.** Env loading happens once at
`context.py:100-107`, but **no reader goes through `Context`** — and
`fetch_google_trends.py:58-59` reads at module-import time, before any `Context` exists.

### E.4 Paths

`context.paths` provides 8 keys. `SEC_13F_INSIDERS_DIR` and `DEF14A_LLM_PATH` have **0 reads**;
`ROOT`'s only apparent hit is a false positive (`ssl_setup.py:48` is a Windows cert-store name).
Meanwhile **9 cache directories** are created by string key with no `paths` entry
(`bulk_cache.py:45-52` falls back to `DATA_STORE / key`), and two modules carry a **date-stamped
report path literal in code**: `diagnostics.py:75` and `gap_check.py:61` — two copies of
`reports/planning/active-tasks/2026-08-26-sharadar-integration/…`.

### E.5 Logging

| Directory | `self._log` | `context.log` | `getLogger(__name__)` | `print(` |
|---|---|---|---|---|
| `src/data_extract` (3 files) | **2** | 3 | 0 | 0 |
| `transformers` (5 files) | **0** | **0** | 0 | 0 |
| `utils/prices` | 0 | 7 | **9** | 0 |
| `utils/fundamentals` | 0 | 11 | 2 | 0 |
| `utils/fundamentals_sharadar` | 0 | **28** | 4 | 0 |
| `utils/common` | 0 | 4 | 8 | 0 |
| `utils/behavioral` | 0 | 5 | 4 | **4** |
| **total** | **2** | **63** | **28** | **4** |

Three names are used for the module logger: `logger` (22 sites), `log` (4), and inline
`log or logging.getLogger(__name__)` (5, all in `bulk_cache.py`). **14 `print(` remain in
`src/`** — the 4 in `fetch_wiki_pageviews.py:172-190` are operational output.

---

## Part F — `record_run`

### F.1 What it is

`record_run(context, table, ticker_count, rows_added, is_full_rescan=False, run_date=None)` at
`run_manifest.py:111-118`. It writes **a JSON file, not a table**:
`Path(context.paths["DATA_STORE"]) / "extraction_manifest.json"` (`:47, :50-51`) →
`./data/extraction_manifest.json`, **git-ignored** (`.gitignore:120`).

Per-table entry: `last_run_date`, `last_full_rescan_date`, `ticker_count`, `rows_added`,
`updated_at`. Read-modify-write with **whole-entry overwrite**; `prior` is consulted for exactly
one field (`last_full_rescan_date`, `:129-133`). No lock, no atomic rename, no fsync (`:142`).
A parse failure logs a warning and returns `{}` (`:60-62`) — **a corrupt manifest silently
discards every table's history**.

Readers: `get_entry` (`:65`), `manifest_window` (`:73-108`) → `edgar_driver.py:88-90` and
`fetch_def14a_llm.py:600-603, :525`. **Nothing else** — no CLI command, no docs generator, no
freshness gate. `rows_added` and `updated_at` are write-only in production.

**Live content: 17 entries.** Absent entirely: every `sharadar_*` table, `fundamentals_history`,
`fundamentals_history_sec`, `fundamentals_reason_codes`, `notes_num`, `notes_text`,
`pension_facts`, `earnings_surprises`, `insider_transactions`, `def14a_llm`, `google_trends`,
`wiki_pageviews`, `earnings_call_sections`, `cusip_ticker_map`, `sp500_tickers`.

### F.2 The gap table — fundamentals writers with no run recorded

| Writer | Table | Write site | Records? |
|---|---|---|---|
| `build_fundamentals_history` | `fundamentals_history_sec` | `build_history.py:1088` | **NO** |
| `build_fundamentals_history` | `fundamentals_reason_codes` | `build_history.py:1090` | **NO** |
| `fetch_financial_statements` | `pension_facts` | `fetch_financial_statements.py:167` | **NO** |
| `fetch_sharadar_tickers` | `sharadar_tickers` | `fetch_sharadar.py:100` | **NO** |
| `fetch_sharadar_fundamentals` | `sharadar_fundamentals` | `fetch_sharadar.py:166` | **NO** |
| `_fetch_dated_table` ×2 | `sharadar_actions`, `sharadar_sp500` | `fetch_sharadar.py:196` | **NO** |
| `build_merged_history` | `fundamentals_history` | `merge_history.py:546` | **NO** |
| `fetch_fundamentals_sec` | `fundamentals_facts`, `fundamentals_employees` | via `edgar_driver.py:103, :106` | yes, at `edgar_driver.py:144` |
| `fetch_financial_notes` | `notes_num`, `notes_text` | `:312, :316` | yes ×2 |
| `fetch_earnings_surprises` | `earnings_surprises` | `:154` | yes, **twice** (bug) |

The entire `fundamentals_sharadar` package contains **zero** `record_run` references — which is
why `StepExtractFundamentalsSharadar`, orchestrating all five stages, leaves no trace.

### F.3 Inconsistencies across the 21 call sites

- **6 pass a raw string** instead of `Tables.<name>`, though all five tables are registered:
  `fetch_google_trends.py:375, :430`, `fetch_wiki_pageviews.py:182, :191`,
  `fetch_financial_notes.py:321, :322`. `name_of` tolerates it by design (`schema.py:752-760`).
- **`ticker_count` means four different things**: requested tickers, resolved CIKs, macro
  *series* (15, `fetch_macro.py:242`) with a hardcoded `0` on the skip path (`:224`), and a
  filter on market-wide data. `fetch_insider_transactions.py:259` passes `len(tickers)` where
  `tickers` was rebound to a **set** at `:231`.
- **`is_full_rescan` passed by 2/21**; **`run_date` by 0/21** (tests only).
- **Early returns that record nothing**: `fetch_def14a_llm.py:581-585` (same-day skip) and
  `:609-612` (no `OPENAI_API_KEY`).
- **Write and record in different functions**, with the CLI bypassing the recording wrapper:
  `fetch_earnings_calls.py` writes at `:426`, records at `:497`, and `cli.py:491` calls
  `ingest_all_earnings_calls` directly.

### F.4 The `_meta.json` sidecar does not exist

`context.py:34-49` documents it in detail. Grep: `_meta.json` appears **only** in
`context.py:35, :36, :48` and `docs/data_conventions.md:207` — **zero fetchers**. `with_name(`
appears only in that docstring plus `tests/data_aggregate/aggregate_fingerprint.py:54-55`. No
`*_HISTORY_PATH` key is even defined. The live sidecar convention is a different one:
`<cache_dir>/<table>_universe.json` (`sec_utils.py:97-113`).

### F.5 Overlapping run/state mechanisms already in the tree

Three independent per-table state stores coexist: `data/extraction_manifest.json` (extract, JSON,
untracked), `part_status_report` (aggregate, DB-derived, no file — `part_status.py:35-87`), and
`reports/baselines/data_profile.json` (DoD, git-tracked — `scripts/dod/baseline.py:48-52`,
per-table `{recorded_at, rows, columns, null_rate, date_min, date_max, scope}`).

**The closest structural precedent for what the spec asks is already built**: `src/validate/`
writes `fundamentals_check_run` (`schema.py:623-627`, PK `(run_id, check_name)`,
`run_id` = hash of `(run_date, tickers, fields, tiers)`, `scope_hash` = same without the date),
with `ValidationRun`/`check_run_frame`/`write()` at `validator.py:65-328` and a full `Ledger`
reader at `ledger.py:88-356`. `schema.py:592-598` records why `run_id` is in the PK: two same-day
runs of different scope once clobbered 269 of 270 rows.

Ten distinct "already in the DB?" mechanisms exist: `ingested_periods` (col `period`),
`bulk_ingested_quarters` (col `quarter`), `existing_filings`, `load/save_processed_universe`,
`load_existing`, `resume_since`, `manifest_window`, `store.max_date` ×2, `_plan_fetch`,
`_is_up_to_date`. `incremental.py:28-30` explicitly **declines** to merge the last of these.

---

## Part G — Duplication and generalization gaps

### G.1 Inside the fundamentals family

| Helper | Duplicate | Evidence |
|---|---|---|
| `build_history.carry_latest_known:224-256` | `merge_history._asof_join:272-290` | both docstrings give the same reason verbatim; `merge_history.py:45` names the primitive |
| `build_history._collapse_same_day:204-219` | `merge_history.collapse_same_date:255-269` | `merge_history.py:259` states the relationship |
| `build_history` dtype-pinning `:825-849` | `merge_history._cast:462-478` | near line-for-line: `:470-471`≡`:825-827`, `:474`≡`:836-837`, `:475-477`≡`:846-848` |
| `periods.trailing_twelve:793-854` (python row loop) | `build_ttm.py:155-161` (vectorised `rolling`) | `build_ttm.py:17` says so explicitly |
| `periods._window_is_contiguous:887-897` (python loop) | `build_ttm._window_is_whole:64-75` (vectorised) | same contract, two implementations |
| `periods._latest_per_window:196` | `tier3_internal._latest_per_window:508` | **same name**, different window identity |
| `cik_cutover.cutover_filings:130` | `edgar_driver.new_filings:44` | `cik_cutover.py:141`: "**Mirrors** `edgar_driver.new_filings` — same dedup, same `since` filter, same ordering" |
| `kpi_catalogue._data_items:571-573` | `field_map._entries:174-176` | byte-for-byte the same comprehension |
| `periods.TTM_QUARTERS = 4` | `src/utils/quarters.py:21 QUARTERS_PER_YEAR = 4` | `periods.py` imports nothing from `utils/quarters.py`, which exists precisely because "three unrelated subfolders need it" |
| `periods.TTM_MIN_DAYS/MAX_DAYS = 330, 400` | the `ANNUAL` band in `_DURATION_BANDS:91` | same two literals, same file, restated not derived |

`TTM_STALENESS_DAYS` is the one place the pattern is done right — `build_ttm.py:49` imports it and
`:34` says "IMPORTED rather than restated".

### G.2 Across `data_extract` — capability duplication

| Capability | Implementations | Shared one exists? |
|---|---|---|
| HTTP GET + retry/backoff | `polite_http.http_get:155`, `rate_limit.call_with_retries:33`, `sec_utils.sec_get:59` (**no retry**), `bulk_cache.py:74` (**no retry**), + 5 bare `requests.get` | Yes, **two** — and **every SEC download path uses neither** |
| unzip + read member | shared `read_zip_member:103`, `read_zip_members:126`, `read_zip_text:153` | **3 of 4 bulk fetchers open `zipfile.ZipFile` directly** (`fetch_financial_notes.py:258-269`, `fetch_insider_transactions.py:203-213`, `fetch_financial_statements.py:106-119`), each re-implementing case-insensitive member lookup — two with `.lower()`, one with `.upper()` |
| corrupt-zip self-heal | shared `bulk_cache._drop_corrupt:92-100`, whose docstring claims closure | **3 private copies remain**: `fetch_financial_notes.py:270-273`, `fetch_insider_transactions.py:214-217`, `fetch_financial_statements.py:120-123` |
| CIK zero-padding | shared `src/utils/string.py:16 pad_cik` | **ignored by 7 sites**; 3 of them drop the `.0` strip or the empty-string guard (`sec_utils.py:137` is the only complete one) |
| cik→ticker inverse dict | — | **byte-identical 3 lines** at `fetch_financial_notes.py:286-287`, `fetch_insider_transactions.py:226-227`, `fetch_financial_statements.py:135-136` |
| quarter/period generation | shared `bulk_cache.quarter_periods:195-203` | `fetch_financial_notes._generate_periods:152-165` and `fetch_fails_to_deliver._periods:42-57` each re-derive the same `max(FIRST_YEAR, today.year - years)` clamp |
| "already ingested?" | `ingested_periods` (col `period`) vs `bulk_ingested_quarters` (col `quarter`) | the split is purely the column name the writer stamped |
| chunked ticker batching | 5 ad-hoc loops | **no shared helper**; `_CHUNK = 500_000` duplicated in 2 files |
| JSON config loading | **6 inline implementations** with 4 different missing-file behaviours (raise / `{}` / uncaught / explicit `exists()`) | none |
| ticker normalisation | **5 copies** of `strip().upper()` | none |

`common/form_registry.py` (115 LOC) is imported by **nothing** in `src/`; its own docstring
(`:9-17`) says it is "consulted by tests and future orchestration work".

---

## Part H — Verbosity and history narration

### H.1 Measured ratios

| file | total | comment | docstring | code | prose : code |
|---|---|---|---|---|---|
| `xbrl_linkbase.py` | 1516 | 334 | 563 | **511** | **1.76 : 1** |
| `periods.py` | 984 | 121 | 365 | 416 | 1.17 : 1 |
| `fetch_fundamentals_sec.py` | 901 | 98 | 317 | 431 | 0.96 : 1 |
| `build_history.py` | 1092 | 180 | 293 | 518 | 0.91 : 1 |
| `reason_codes.py` | 188 | 105 | 33 | **31** | **4.45 : 1** |
| `entity_scope.py` | 239 | 40 | 95 | 78 | 1.73 : 1 |
| `fundamentals_employees.py` | 179 | 14 | 76 | 71 | 1.27 : 1 |
| `common/parallel_fetch.py` | 51 | 1 | 29 | **13** | **2.31 : 1** |
| `common/incremental.py` | 89 | 1 | 50 | **23** | **2.22 : 1** |

Module docstrings: `xbrl_linkbase.py` **lines 1–109** (7.2 % of the file);
`fetch_fundamentals_sec.py` 1–27; `reason_codes.py` 1–33 (a file with **0 functions and 0
classes** — 15 constants and 138 lines of prose).

Prose-heavier-than-code functions: `_gross_profit_identity` 19 doc / **2** code;
`_one_share_basis` 16 / **2**; `_inclusive_days` 8 / **1**; `_latest_per_window` 20 / **5**;
`bare` 10 / **2**; `is_note_only` 25 / **5**; `_leaf_sum` 65 / 61; `sibling_leg` 56 / 14.

### H.2 The history-narration corpus

Roughly **250 comment/docstring blocks** narrate what the code *used to* do, what was measured,
what was rejected, or which plan phase owns something. Volume by file (regex over
`used to|previously|before the fix|no longer|an earlier version|until Phase|tried and reverted|
REJECTED|the plan|Phase N|§N|decision N|Replaces the|drifted|measured`):

| file | narrating lines |
|---|---|
| `xbrl_linkbase.py` | 76 |
| `fetch_fundamentals_sec.py` | 53 |
| `build_history.py` | ~60 blocks |
| `periods.py` | ~55 blocks |

Dangling or stale references a de-verbosing pass must resolve:

| Reference | Problem |
|---|---|
| `build_history.py:460` cites `scripts/measure_total_liabilities_legs.py` | producer script no longer in the tree; `data/total_liabilities_legs.json` still read at `:460` |
| `constants.py:895` cites `_TO_COMMON_TOL` in `fetch_fundamentals.py` | neither exists |
| `cik_cutover.py:81-88` cites `tests/data_extract/test_cik_cutover.py` | actual path is `tests/data_extract/fundamentals/test_cik_cutover.py` |
| `fundamentals_employees.py:89-93` says the median is seeded from `fundamentals_facts` | `fetch_fundamentals_sec.py:883` reads `Tables.fundamentals_employees`, and `:882` says headcount "no longer lives" there |
| `fetch_financial_statements.py:17-19` says "once the … Notes sets are wired" | they are — `fetch_financial_notes.py`, whose `:5` cross-references this module |
| `run_manifest.py:6` cites "the fetchers in `step_extract_all_data.py`" | that file has no fetchers |
| `form_registry.py:4` cites `schema_registry.TableSpec` | superseded per `schema.py:5` |
| `bulk_cache.py:96-97` says all six fetchers now get the shared self-heal | three private copies remain |
| `build_history.py:789` "The replay is O(filings)" | measured O(E²·K) |
| `kpi_catalogue.py:258` "with lookups precomputed" | only `fields` is |
| `xbrl_linkbase.py:1196-1197` `_leaf_sum` "is free when it does not apply" | its prologue + `filer_leaves` run for all 48 fields; only 3 apply |
| `edgar_extract.py:2`, `edgar_fillings.py:2`, `gics.py:2`, `llm_extractor.py:2`, `rate_limit.py:2` | **5 wrong self-paths** in line 2, all predating the move into `common/` |
| `docs/coding_standard.md:18` | says `constants.py` is 927 lines; it is 1,058 |
| `step_extract_all_data.py:4-11` | docstring says "four sub-steps"; `__init__` builds five, and the order differs from both `__init__` and `run` |
| `step_extract_fundamentals.py:9` | cites `reports/planning/active-tasks/2026-08-23-fundamentals-rebuild-plan-v2.md` |

Cross-references to plan artefacts that are not current behaviour: `Phase 1/4b/5/5b/6 §6.1/7/10`,
`plan-5b`, `§5.0`, `§5.1`, `§B.5`, `§B.6.6`, `4c.1`, `4c.8`, `register item 7/8/9`,
`decision #9/24/28/30/31/32/33/34/35/37/40/46`,
`D1/D1b/D7/D8/D11/D14/D15/D17/D18/D20/D21/D23/D24/D25/D27`.

---

## Part I — Test blast radius

### I.1 Inventory

| | Count |
|---|---|
| Test files importing `utils/fundamentals/**` | **27** |
| Test functions in `tests/data_extract/fundamentals/` | **201** across 6,737 LOC |
| Test files importing `fundamentals_sharadar/**` | 4 (53 tests, 1,771 LOC) |
| Test files importing `validate/fundamentals/**` | 6 (85 tests, 2,260 LOC) |
| Test files importing `src/utils/step.py` | **0** |
| `conftest.py` in the whole tree | **3** — and **none** under `tests/data_extract/` |
| pytest config (`[tool.pytest.ini_options]`, `pytest.ini`, `tox.ini`, `setup.cfg`) | **none** → no markers, no `-m "not network"` |

Gating, per test function: 163/201 (81 %) synthetic, **31 network** (all
`if not os.getenv("SEC_USER_AGENT"): pytest.skip`), 4 DB, 3 committed-parquet
(`data/fundamentals_sweep`, which is git-ignored — so they skip on a clean clone).
`test_linkbase_history.py` is 7/7 network.

### I.2 The tripwires — 26 private symbols pinned by tests

**17 imported directly.** Highest-traffic: `_drop_note_only_quarter`
(`fetch_fundamentals_sec.py:290`, 13 test functions), `_filing_annual_windows` (`:275`, 6),
`_hard_guard` (`build_history.py:716`, 5), `_gross_profit_identity` (`:426`, 4),
`_is_stale` (`:283`, 4), `_contradicts_gross_profit` (`:396`, 3), `_plan_fetch`
(`fetch_earnings_surprises.py:71`, 3), plus `_compose`, `_materialise`, `_period_frame`,
`_FORMULAS`, `_adjustment_json`, `_linkbase_weights`, `_resolve_subtractions`,
`_total_liabilities_identity`, `_RECENT_LIMIT`, `_values_by_period`.

**9 more via module alias**: `bh._snapshot`, `fe._download_one`, `fe.yf`, `fn._read_notes`,
`fn._period_year`, `fn._notes_periods`, `fn._join_notes_num`, `fn._generate_periods`,
`fin._join_pension`, `fin._read_pension_facts` — and `fn._scrape_available_periods` is
monkeypatched **by string name** (`test_financial_notes.py:168, :174`).

**Module-alias pins expose whole surfaces**: `P` = `periods` (15 attributes,
`test_periods_q4.py:26`), `rc` = `reason_codes` (12), `scope` = `entity_scope` (7).

**3 silent-skip pins** — `pytest.importorskip` with a **dotted string**: a rename turns the test
into a *skip*, not a failure: `test_fundamentals_employees.py:62, :170`,
`test_fundamentals_point_in_time.py:173`.

**Also pinned**: `test_step_extract_fundamentals.py:25-31` pins five module attribute *names* on
`step_extract_fundamentals` **and their call order** via `setattr` — a rename fails at `setattr`.
`test_periods_q4.py:33` pins the three exact `PeriodGuards` field names.

### I.3 Collection-time cost

**9 files call `load_catalogue("./configs")` at module import**: `test_amendment_grain.py:29`,
`test_build_history.py:26`, `test_leaf_sum_resolution.py:34`, `test_linkbase_history.py:37`,
`test_linkbase_resolution.py:24`, `test_linkbase_sibling_total_1c9a517eaa47.py:35`,
`test_periods_q4.py:29`, `test_segment_margin_876ab8a57bd8.py:66`,
`test_statement_role_routes.py:33`. Because `@cache` keys on the argument and
`test_kpi_catalogue.py:55` passes an **absolute** path, a run of that directory pays **2 full
parses** (6 JSON reads) with all six validation passes each. An invalid catalogue is a
**collection error in 9 modules at once**, even for a `-k`-filtered run.

**12 ad-hoc `get_config_context(...)` sites** in `tests/`, and the 4 sharadar files each carry a
**byte-for-byte duplicated** `context()` fixture (`test_fetch_sharadar.py:46-56`,
`test_sharadar_diagnostics.py:51-61`, `test_sharadar_field_map.py:71-82`,
`test_sharadar_merge.py:56-68`).

### I.4 What is unguarded

Public symbols in `fundamentals/**` whose name appears **nowhere** in `tests/`:
`build_history.TickerHistory:127`, `facts_frame_from_companyfacts:933`, `cik_cutover.Cutover:54`,
`entity_scope.us_gaap_only:142`, `entity_scope.dimensioned_facts:232`,
`fetch_fundamentals_sec.build_ticker_fundamentals:801` (the per-ticker orchestrator),
`fundamentals_employees.filing_body_text:105`, `xbrl_linkbase.qualify:522`,
`xbrl_linkbase.is_income_statement_role:927`, and `xbrl_linkbase.bare:796` (25 apparent hits,
**all 25 prose or a local DataFrame column** — never called).

`Catalogue` methods with 0 test references: `regime_for_sub_industry`, `regime_for_gics`,
`regime_for_role_uris`, `expected_absent`, `periodicity_shapes`, `combined_into`,
`regime_break_effective`. `fundamentals_sharadar/**` has **23** unreferenced public symbols.

**The aggregate fingerprint cannot catch fundamentals-extraction changes.**
`tests/data_aggregate/aggregate_fingerprint.py:55` reads a **committed**
`aggregate_fingerprint_fundamentals.parquet` whenever it exists; the DB path (`:67-87`) runs only
when the parquet is absent.

### I.5 The four hash-suffixed regression tests

`cluster_id(ticker, field) = sha256(f"{ticker}\x1f{field}").hexdigest()[:12]`
(`validate/fundamentals/finding.py:107-127`). All four verified by recomputation:

| file | cluster | (ticker, field) |
|---|---|---|
| `test_linkbase_sibling_total_1c9a517eaa47.py` | `1c9a517eaa47` | MCD `capex` |
| `test_note_only_quarter_2603621e89ab.py` | `2603621e89ab` | ORCL `totalRevenue` |
| `test_note_only_quarter_919b35844b54.py` | `919b35844b54` | BA `incomeTaxExpense` |
| `test_segment_margin_876ab8a57bd8.py` | `876ab8a57bd8` | ORCL `grossProfit` |

The convention is in `src/validate/README.md:336-338`, **not** `docs/testing.md`.
`validate/cli.py:509-519` checks only `Path(test_path).exists()` — it never parses the name — so
**renaming one silently invalidates the recorded `test_path` in `fundamentals_check_fix` with no
failing test**.

### I.6 Rules a refactor must respect (`docs/testing.md`)

`:20-27` parsing/derivation math may use synthetic known-truth fixtures **paired with a real-data
coverage check** — this is the rule the 31 network-gated tests satisfy.
`:76-102` **every test must print a sanity-check conclusion**; `:102` "Do not say 'tests pass'
without this printed conclusion". `:110-121` run from the repo root with the Poetry venv python
and **`-s`**. `:125-136` architectural guards — `tests/data_store/test_store_boundary.py` greps
every `src/**/*.py` for `sqlalchemy`/`read_sql`/`to_sql`/`engine.connect`/`store.engine`.
`:154-169` the fingerprint baseline "may be regenerated **only** in a commit that touches no
`src/` file". `:171-175` CI is pylint-only on Python 3.8/3.9/3.10 and stale — **there is no
pytest job**.

---

## Code References

**The live bugs to fix first**
- `src/data_extract/utils/fundamentals/xbrl_linkbase.py:514` — `cols` undefined; should be `ARC_COLUMNS`
- `src/data_aggregate/transformers/step_cube_fundamentals.py:179-180` + `src/data_aggregate/utils/fundamentals/employee_features.py:40` — reads `employees`, table has `employees_sec`
- `src/data_extract/utils/fundamentals/fetch_earnings_surprises.py:146-156` — missing `return`

**The efficiency hot path**
- `src/data_extract/utils/fundamentals/build_history.py:809-816` — the `iterrows` event loop handing the whole prefix
- `src/data_extract/utils/fundamentals/build_history.py:597` — the period engine rebuilt per event
- `src/data_extract/utils/fundamentals/build_history.py:1052` — sequential per-ticker loop, no pool
- `src/data_extract/utils/fundamentals/build_history.py:1061` — unprojected 69-column read
- `src/data_extract/utils/fundamentals/build_history.py:224-256` — 1-row DataFrame + `merge_asof` per instant field per event
- `src/data_extract/utils/fundamentals/periods.py:530, :941, :967` — `quarterize`, `build_periods`, the duplicate `fiscal_year_ends`
- `src/data_extract/utils/fundamentals/fetch_fundamentals_sec.py:697-698` — `calculation_linkbase()` twice
- `src/data_extract/utils/fundamentals/fetch_fundamentals_sec.py:625` — `period_of_report` per row
- `src/data_extract/utils/fundamentals/xbrl_linkbase.py:1198-1200` — `_leaf_sum` prologue for 48 fields, 3 apply
- `src/data_extract/utils/fundamentals/kpi_catalogue.py:283-326` — uncached scan+sort views

**The wiring**
- `src/data_extract/step_extract_all_data.py:50-57` — 3 steps commented out, `full=True` hardcoded
- `src/context.py:129` — `read_config(path="./configs")` discards `config_path`
- `src/data_extract/utils/common/edgar_driver.py:76-90, :143-144` — the `full` bypass, `manifest_window`, `record_run`
- `src/data_extract/utils/common/run_manifest.py:111-143` — the JSON writer
- `src/data_extract/transformers/step_extract_prices.py:12-14, :37-38` — the reference pattern
- `src/utils/step.py:14-24` — 5 attributes, no paths, no clients, no config dir

**The reference precedent for a run ledger**
- `src/data_store/schema.py:592-627` — `fundamentals_check_run`, `run_id` in the PK and why
- `src/validate/fundamentals/validator.py:65-328`, `ledger.py:88-356`

## Architecture Documentation

`fundamentals/` violates the folder pattern `prices/` follows. In `prices/`, each file is one
fetcher with a `(context, tickers=..., years_history=...)` signature, the step resolves the
windows and passes them in, and `record_run` fires in the fetcher. In `fundamentals/`, 5 of 12
files export no `context`-taking function at all (`entity_scope`, `kpi_catalogue`, `periods`,
`reason_codes`, `xbrl_linkbase`) — they are a domain library living inside a fetcher folder — and
`reason_codes.py` has no functions whatsoever.

The two-layer split (`fundamentals_facts` → `fundamentals_history_sec`) is sound and documented,
and `build_history`'s append-only guard (`diff_against_stored:994`) is genuinely load-bearing.
What is not sound is that the second layer re-derives the *entire* period history at every
publication event rather than extending it.

## Key Data Flows

```
StepExtractFundamentals.run(tickers, full)
 |- fetch_fundamentals_sec(years_history, full)      8 workers, network
 |    `- run_edgar_fetch(tables=(facts, employees))  manifest_window + existing_filings
 |         `- per ticker: new_filings | cutover_filings
 |              `- per filing: filing.xbrl()  [1 GET; up to 8 on fallback]
 |                   `- rows_from_xbrl: scope -> 2x calculation_linkbase -> 48x resolve_field
 |                        `- per field: up to 3x _resolve_once (routes 1,2,3,3b,4,5)
 |    -> fundamentals_facts, fundamentals_employees   [record_run x2]
 |- build_fundamentals_history(rebuild_history=full) SINGLE-THREADED, no network
 |    `- per ticker: load facts (projected) + load history (UNPROJECTED)
 |         `- build_ticker: per publication event
 |              `- _snapshot(whole visible prefix)
 |                   `- build_periods -> 22x quarterize + trailing_twelve   [O(E^2*K)]
 |    -> fundamentals_history_sec, fundamentals_reason_codes   [NO record_run]
 |- fetch_earnings_surprises   -> earnings_surprises      [record_run x2, one is a bug]
 |- fetch_insider_transactions -> insider_transactions    [record_run]
 `- fetch_financial_notes      -> notes_num, notes_text   [record_run x2; notes_text has no reader]
    (fetch_financial_statements -> pension_facts is NOT here: CLI + DAG only, no record_run)
```

## Dependencies

External on this path: `edgartools 5.51.0` (constraint `>=5.44.1,<6.0`), `pandas`, `numpy`,
`omegaconf`, `tqdm`, `requests`, `curl_cffi`, `yfinance` (surprises only). Internal:
`common/{edgar_driver, parallel_fetch, run_manifest, sec_utils, bulk_cache, edgar_extract}`,
`utils/{string.pad_cik, quarters (unused by periods.py), polite_http (unused by the SEC path)}`,
`data_store/{schema, store}`, `constants/constants` (12 symbols).

## Test Coverage

See Part I. 201 test functions over `tests/data_extract/fundamentals/`, 81 % synthetic,
26 private-symbol tripwires, 3 silent-skip string pins, no shared fixtures, no pytest config,
and no fingerprint or CI net beneath any of it.

## Related Documentation

- `AGENTS.md` — the two-fundamentals-tables rule, the `store` boundary, risk zones
- `docs/coding_standard.md` — constants-first, function size, logging, "docstrings carry the reasoning"
- `docs/testing.md:76-102` (sanity print), `:154-169` (fingerprint gating)
- `src/validate/README.md:336-338` — the `cluster_id` test-naming convention
- `docs/architecture.md:88-110` — the pipeline and the stated extraction order
- `specs/2026-08-26/refactor-fundamentals.md` — this task's brief

## Open Questions for the Planning Phase

1. **Incrementalising the replay.** `_snapshot` needs "everything filed by `as_of`". Can
   `build_periods` be computed once per ticker and *sliced* by `as_of`, or must each event's
   quarters be re-derived because a later filing restates an earlier period?
   `build_history.py:814` suggests slicing is intended; `periods.py:199-215` (BAC FY2023 as-filed
   vs re-presented) suggests it is not safe in general. **This one question decides whether the
   fix is O(E) or a memoisation.**
2. **Parallelising the replay.** It is pure CPU with no network and no shared state beyond the
   catalogue. Blocker: `store.ensure_table` is a check-then-create race (`edgar_driver.py:94-97`)
   — but both target tables are already warm.
3. **The `cols` fix and its blast radius.** One-word fix; but how many stored tickers are missing
   facts from *previous* runs that hit it? Needs a replay count before deciding whether a
   targeted refetch is required.
4. **`config_dir`.** Adding it to `Context` touches two named risk zones (`context.py`,
   `utils/step.py`) and needs approval per `AGENTS.md`. The alternative — threading it through 20
   sites — is what `fundamentals_sharadar` already does and `fundamentals` already refuses.
5. **`record_run` generalisation.** Extend the JSON, or adopt the `fundamentals_check_run` table
   pattern (`schema.py:623`) that already solves scope-hashing and same-day clobbering? The spec
   asks for "a clear json", but the DB precedent exists and is better.
6. **`constants.py`.** Deleting 46 dead symbols is safe and mechanical. Relocating the 80
   one-consumer symbols conflicts with `docs/coding_standard.md:12-18` ("never hardcode a global
   literal inline") — needs an explicit convention decision, and `constants.py` is a risk zone.
7. **De-verbosing vs the standard.** `docs/coding_standard.md:45-55` says these docstrings are
   "unusually load-bearing … several explicitly say the duplication is deliberate", while the
   spec asks to remove the history narration. The ~250 narrating blocks encode *measurements*
   that are not reproducible (e.g. `kpi_catalogue.py:549-556` — the source table was dropped
   2026-08-26). Proposal to settle in planning: keep the measurement, drop the chronology; move
   the plan-phase and decision-number references out of `src/` and into a report.
8. **The `full=True` hardcode.** Restoring `run(full=...)` changes `main.py`'s behaviour from
   full-rebuild to incremental. Intended?
9. **`notes_text`.** ~26 GB of cached ZIPs feeding a table with no reader. Stop extracting, or
   wire the consumer?
10. **Test tripwires.** 26 private symbols and 3 string-path pins mean any rename breaks tests —
    and the 3 string pins break them *silently*. Decide up front: promote the pinned privates to
    public, or rewrite those tests against public entry points.
