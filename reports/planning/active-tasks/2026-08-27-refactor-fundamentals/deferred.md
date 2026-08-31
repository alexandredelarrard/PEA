# Deferred — found in research, deliberately not fixed in this refactor

Each entry says **what**, **the evidence**, **why it is not here**, and **who should take it**.
Nothing on this list is "won't do"; it is all "not in this scope".

---

## D-1. The cube's whole workforce block is dead — LIVE BUG, HIGHEST PRIORITY

`step_cube_fundamentals.py:179-180` passes the `fundamentals_history` frame as
`headcount_history`; `employee_features.py:40` reads column `"employees"`; the merged table
declares **`employees_sec`** (`schema.py:210`) after the phase-0 rename. `fundamentals_to_daily`
returns **empty for a missing column instead of raising**, so nothing fails.

**Impact**: `employee_growth`, `revenue_per_employee` and the whole workforce panel are silently
empty in the cube — and therefore in every model trained on it.

**Why not here**: the user excluded `data_aggregate/` and cube creation from this refactor
("another phase, do not touch").

**Owner**: the aggregation phase. Fix the column name **and** make `fundamentals_to_daily` raise
on a missing column rather than returning empty — the silent-empty is the reason a rename went
unnoticed. Also check the other callers of that helper for the same pattern.

---

## D-2. Remediation for the `cols` bug's missing facts

Decision 3: fix + regression test land in Phase 1; the **backfill** decision waits until the
in-flight walk's coverage is measured. NEM, MO and AIZ produced zero facts in the 2026-08-27 run;
earlier runs may have hit the same branch on other tickers.

Detector, to run after the walk finishes:

```sql
SELECT t.ticker FROM sp500_tickers t
LEFT JOIN (SELECT DISTINCT ticker FROM fundamentals_facts) f USING (ticker)
WHERE f.ticker IS NULL;
```

Plus the weaker signal — tickers whose `filing_date` max is far behind the roster median.

**Owner**: whoever runs the post-walk review. The Phase 7 report carries the detector output.

---

## D-3. The vintage redesign of `quarterize` (the O(E) replay)

Phase 3.1 shows why "compute once and slice" is unsafe as written, and sketches the design that
*would* be safe: emit every candidate basis row with its own `known_from` and rank, vintage the
refusals, and defer the as-reported-beats-derived dedup to an as-of slice.

**Why not here**: decision 1 — safe-only. The refusal-vintaging half is genuinely hard
(`_is_ambiguous_duration` and `_drop_annual_masquerading_as_quarter` both read the visible set).

**Gate**: `restatement-census.md` (Phase 3.5). If value-changing restatements and refusal flips
are both rare, a hybrid becomes attractive and this deserves its own plan.

**GATE RESULT (2026-08-28): FAILED — recommendation is LEAVE IT.** Measured on the 8 sample
tickers at full history, 9,345 windows / 517 events:

- **8.22 %** of windows carry a materially changed vintage (> 0.1 %), against the < 2 % the
  gate asked for. Worst filer APA 15.5 %, best MCD 3.2 %. Median lag from first vintage to the
  changed one is **364 days**, so a truncated sample if anything UNDERSTATES it.
- The hybrid's actual unit, the `(ticker, field)` pair, is **70 % dirty** — only 32 of 108 are
  never materially restated. Ticker-agnostic it is worse: **18 of 19 fields** restate on at
  least one filer.
- **60 % of refusals REVERSE** as the prefix grows (37 of 62 triples; `split_basis_mismatch`
  72 %, `derived_basis_mismatch` 62 %, `derived_sign_implausible` 46 %). This is the harder
  blocker — a verdict that changes has no single `known_from` to be vintaged with.

Do Phase 4's process pool instead (4x, no semantic risk). Revisit only if a whole-table census
comes in under ~2 % on both measures. See [restatement-census.md](restatement-census.md).

---

## D-4. SEC network reliability: no retry, no backoff, no timeout, no `EDGAR_*` env var

Measured: **0** matches repo-wide (including `.env*`) for any retry/backoff/timeout/throttle
setting or `EDGAR_*` variable. `common/rate_limit.call_with_retries:33` exists but is used only by
the yfinance/Trends paths. Every SEC download path uses neither it nor `polite_http.http_get:155`.

In the live log: 50 `SGML fetch failed … SEC returned HTML or XML`, 15 `peer closed connection
(incomplete chunked read)`, 7 `[WinError] connection forcibly closed`, 1 short read. Each fallback
costs up to **7 extra GETs**, because the homepage-built `FilingSGML` has "valid URLs but without
in-memory content" (`edgar/_filings.py:1970-1972`) so every attachment becomes its own request.

Relevant library defaults in force: `httpx.Timeout(get_edgar_http_timeout(), connect=10.0)`
(`edgar/httpclient.py:71-73`, overridable only via the unset `EDGAR_HTTP_TIMEOUT`) and
**`http2=False`** (`:56-61`), whose own comment names the mid-stream-reset failure mode it avoids
— the same class now appearing on HTTP/1.1.

**Why not here**: adding retry changes failure behaviour under load. That is an operational
change, not a refactor, and it deserves measurement of its own.

**Owner**: a network-reliability task. Cheap first step: one line in `configs/logging.yml`
silencing the **301 cosmetic** `SGML header declares …` warnings
(`edgar/sgml/sgml_common.py:238`), which are 79 % of all log lines.

---

## D-5. The 49 GB edgartools cache with no eviction

`~/.edgar/_tcache/` measured at **94,468 files / 49 GB**. `/Archives/edgar/data` responses are
cached **forever** (both the `.txt` submission and the `-index.html`); `data.sec.gov/submissions`
for 30 s. No TTL, no size cap, no eviction, and nothing in the repo clears it
(`EDGAR_LOCAL_DATA_DIR` / `EDGAR_USE_LOCAL_DATA` / `use_local_storage` / any cache-clear -> **0**
matches repo-wide). It lives at `~/.edgar` only because `EDGAR_LOCAL_DATA_DIR` is unset
(`edgar/core.py:322-327`).

Upside worth keeping in mind: because `/Archives` is cached forever, **a re-run is parse-bound,
not network-bound** — which is exactly why the replay's CPU cost dominates any realistic re-run.

**Owner**: an ops task. Decide a location and a retention policy; do not silently delete 49 GB.

---

## D-6. `constants.py` outside `fundamentals/` — 35 one-consumer symbols

Decision 6 scoped the relocation to the fundamentals slice. The remaining one-consumer symbols
live in `prices/`, `sharadar/` and `modelling/` and touch ~19 test files.

Also deferred: deriving the **6 hand-maintained Sharadar subsets** (`SHARADAR_SF1_COLUMNS:141` ⊃
`SHARADAR_ID_COLUMNS:168`, `SHARADAR_ZERO_FILLED_FIELDS:187`, `SHARADAR_FLOW_FIELDS:288`,
`SHARADAR_NON_NEGATIVE_FIELDS:308`, `SHARADAR_EVENT_FIELDS:370`,
`SHARADAR_DIAGNOSTIC_EXTRA_COLUMNS:278` — 91 definition lines over one 112-name vocabulary) from a
single declaration. Real improvement, real risk, needs its own verification.

And the repo-wide reverse violations left alone: `DATE_FORMAT = "%Y-%m-%d"` re-typed inline at
**11** sites (incl. `store.py:161`, `step_train.py:558, :759`), the 3 remaining
`DATE_FORMAT_COMPACT` sites, `fetch_fails_to_deliver.py:35-36`'s local URL templates, and
`fetch_fails_to_deliver._periods:42-57`'s third copy of the quarter clamp.

---

## D-7. Clients still constructed per call, outside the SEC path

Phase 5.2 puts the EDGAR identity and one SEC session on `Context` and stops there. Still
outstanding:

- **OpenAI: 3 independent constructions with 2 different key-precedence orders** —
  `llm_extractor.py:43`, `data_peers/utils/embeddings.py:35`, `utils/openai_embeddings.py:25`.
  Two precedence orders is a latent bug, not just duplication.
- **FRED**: `Fred(api_key=os.getenv(...))` per `_fred_frame` call (`fetch_macro.py:133`).
- **Sharadar**: `os.getenv` per `sharadar_get` call (`client.py:60, :134`).
- **28 `os.getenv`/`os.environ` sites across 21 distinct variables**; env loading happens once at
  `context.py:100-107` but **no reader goes through `Context`** — and
  `fetch_google_trends.py:58-59` reads at **module-import time**, before any `Context` exists.
- 9 sessionless `requests`/`curl_cffi` call sites outside the SEC path.
- `src/utils/polite_http.py` and `crawler.py` are both sessionless and unreachable from `Context`.

---

## D-8. `notes_text` has no reader

`fetch_financial_notes.py:316` writes it; nothing reads it anywhere. The download is the heaviest
in the repo — "~300-450MB EACH … ~26GB back-fill" (`:44-46`). `constants.py:536-539` and
`schema.py:515-517` both record that the consumer "was never wired into any panel and has been
removed".

Decision 9: **keep extracting**. Wiring the consumer is a separate task; so is stopping.

---

## D-9. The three commented-out sub-steps and the `full=True` hardcode

`step_extract_all_data.py:53-57` — 3 of 5 sub-steps commented out, `full=True` passed literally,
and `run()` takes no `full` parameter (`:50`), so `main.py` can only do a full rebuild.
`full=True` bypasses `manifest_window` entirely (`edgar_driver.py:76-86`).

Decision 8: **do not touch** — a full run is in flight and that is intended. Only the docstring
("four sub-steps" vs five built) is corrected, in Phase 7.5.

Related and also deferred (user declined): splitting `full`'s two meanings — `full=full` on the
fetcher (skip the manifest window) and `rebuild_history=full` on the replay (DELETE then rebuild),
`step_extract_fundamentals.py:40, :48`. Two very different blast radii on one flag.

---

## D-10. The aggregate fingerprint cannot catch fundamentals-extraction changes

`tests/data_aggregate/aggregate_fingerprint.py:55` reads a **committed**
`aggregate_fingerprint_fundamentals.parquet` whenever it exists; the DB path (`:67-87`) runs only
when the parquet is absent. `docs/testing.md:154-169` allows regenerating the baseline **only** in
a commit that touches no `src/` file — so this cannot be addressed from inside this refactor.

Also: CI is **pylint-only on Python 3.8/3.9/3.10** and stale — **there is no pytest job**
(`docs/testing.md:171-175`). Every test gate in this plan is therefore a local gate. Worth its own
task.

---

## D-11. `store.ensure_table` is a check-then-create race

No lock. Threaded writers on a **cold** table can silently lose rows, and `sqlite_store` cannot
reproduce it. Phase 4 avoids it by construction (only the parent writes, serially) and asserts the
tables exist before the pool starts — but the underlying race is untouched and still live for any
future threaded writer.

---

## D-12. Ten overlapping "already in the DB?" mechanisms

`ingested_periods` (column `period`), `bulk_ingested_quarters` (column `quarter`),
`existing_filings`, `load/save_processed_universe`, `load_existing`, `resume_since`,
`manifest_window`, `store.max_date` x2, `_plan_fetch`, `_is_up_to_date`. The
`ingested_periods` / `bulk_ingested_quarters` split is **purely the column name the writer
stamped**. `incremental.py:28-30` explicitly declines to merge the last of these.

Phase 5 unifies the *run ledger*; it does not unify these. A follow-up should decide how many of
the ten are genuinely different questions.

---

## D-13. Two more shared-capability gaps

- **chunked ticker batching**: 5 ad-hoc loops, no shared helper, `_CHUNK = 500_000` duplicated in
  2 files.
- **`common/form_registry.py`** (115 LOC) is imported by nothing in `src/`; its docstring
  (`:9-17`) says it is "consulted by tests and future orchestration work" and `:4` cites
  `schema_registry.TableSpec`, superseded per `schema.py:5`. Phase 6.5 requires a
  wire-it-or-delete-it decision; if that decision is "wire it", the work lands here.

## CPAY's two pre-XBRL 2011 filings make the "0 facts" tripwire cry wolf every run

**Found**: 2026-08-27, from a live `full=False` run, while Phase 2 was being verified.
**Owner**: whoever next touches `fetch_fundamentals_sec`'s logging. **Not a Phase 2 defect** —
measured on both trees, the behaviour is identical.

```
CPAY: 0 facts from 2 filing(s) (0 unreadable) -- the ticker's whole history is missing, not empty
```

The two filings are `0001193125-11-078175` (10-K, filed 2011-03-25) and `0001193125-11-140813`
(10-Q, filed 2011-05-16) — FleetCor's first two filings after its December 2010 IPO. **Neither
ships XBRL at all**: `filing.xbrl()` returns `None`, which is the clean early return in
`filing_rows`, which is why the count of unreadable filings is correctly **0**. Nothing failed.

Three consequences, none of them a data defect:

1. **The message is false in the incremental case.** CPAY holds **6,353 facts from 62
   accessions** (2011-08-15 to 2026-08-10). Its history is neither missing nor empty. The
   tripwire (`fetch_fundamentals_sec.py:913`) was written for the full-rescan case, where
   `filings` really is the whole history; on a resumed run `filings` is only what is *not
   already stored*, and a filer's pre-XBRL era is permanently in that set.
2. **They are re-walked on every run, forever.** They can never enter `done_accessions`,
   because a filing that yields no facts writes no row to key it by.
3. **It blunts the tripwire.** This line exists to catch the `cols` `NameError` class of bug in
   hour one. A recurring false positive on a known-benign ticker is exactly how that stops
   working.

The fix is small and belongs with the tripwire, not here: count filings whose `xbrl()` is `None`
separately, and when EVERY walked filing is in that set, say so at WARNING ("N pre-XBRL
filing(s), no facts expected") instead of asserting at ERROR that the history is missing.
Reserve the ERROR for the case it was written for — filings that HAD XBRL and still produced
nothing.

Worth a sweep at the same time: how many tickers have pre-XBRL filings permanently in the
incremental window. XBRL was phased in over 2009-2011, so it is unlikely to be only CPAY.

---

## D-14. `sp500_tickers` has no `read_columns`, so `project=True` cannot be used on it

`sec_utils.load_cik_mapping` read the universe with no projection at all. Phase 3.4 gave it an
explicit `columns=list(CIK_MAPPING_COLUMNS)` — the union of what its seven callers consume —
which is correct but is the *manual* form of the mechanism this repo already has:
`Table(read_columns=(...))` + `store.load(project=True)`, which additionally degrades with a
logged warning when a column is missing instead of raising.

**Evidence**: the live table has exactly six columns (`ticker, name, sector, industry_group,
sub_industry, cik`), all six are in the union, so today the projection is the whole table and
~500 rows. This is a convention fix, not a byte saving.

**Why not here**: `src/data_store/schema.py` is a named risk zone and Phase 3 has no approval
for it. **Owner**: Phase 5, which does.

---

## D-15. `_latest` re-masks the whole quarters/ttm frame once per field per event

`build_history._snapshot` calls `_latest(ttm, field)` and `_latest(quarters, source)` inside the
field loop, and each call does `frame[frame["field"] == field]` over the CONCATENATED frame for
every field in the catalogue. On MCD that is ~120 full-frame boolean masks per event over ~2,600
rows, ~21 M element comparisons across a 69-event replay.

The fix is the same shape as `_split_by_field`, which already exists two functions away: group
once per event into a `dict[str, DataFrame]` and index it, rather than re-scanning per field.

**Why not here**: this is a Phase 2 constant-factor item and Phase 2 is closed; Phase 3's
scope is §3.2–§3.5. It is also **unmeasured** — noticed while reading, not profiled — so it
needs a CPU-time measurement before anyone spends effort on it. **Owner**: whoever reopens the
constant-factor work, or Phase 7 if the function split makes it free.

**Method note for whoever takes it**: measure CPU time, not wall clock, and interleave the arms
in one process. Phase 3.6 records how sequential wall-clock timing produced three mutually
contradictory verdicts on this machine.
