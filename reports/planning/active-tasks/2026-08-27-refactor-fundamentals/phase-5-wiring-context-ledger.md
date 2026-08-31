# Phase 5 — Wiring: `Context` owns the config dir and the EDGAR identity; runs go in the DB 🔄

**Status (this pass)**: 5.1 ✅, 5.2 ✅, 5.4 ✅. 5.3 🔄 — the schema, the seed script and the
"fill the gaps" writer sweep are done; the `run_manifest.py` DB cutover itself, running the
seed script and deleting the JSON are DELIBERATELY held — `main.py` was mid-flight (started
2026-08-30 16:54, still running when this pass ended) and the user chose to land the code
without touching the live run. See "5.3 status" below for the exact resume point.

**Goal**: one source of truth for `./configs`, one place the SEC identity is configured, and a
run ledger in Postgres that every fundamentals writer actually writes to.

Independent of Phases 2–4 — it can land alongside them.

**Risk zones, approved**: `src/context.py`, `src/utils/step.py`, `src/data_store/schema.py`,
`sql/schema.sql` (additive splice only).

---

## 5.1 `config_dir` — fix the discard, then thread it ✅

### The root cause

`get_config_context(config_path, ...)` accepts the path and then calls
`read_config(path="./configs")` at **`context.py:129`**, hardcoded. `config_path` is used only in
the error message at `:133`. **The CLI's `-c` never reaches the config reader.** The code already
knows: `fetch_fundamentals_sec.py:869-871` says so in a comment.

The Airflow chain works by coincidence — `dag_data_extraction.py:65` passes
`-c /opt/airflow/project/configs` with `cwd=/opt/airflow/project`, so the absolute flag and the
relative default resolve to the same directory. The flag is still discarded.

### Changes

- [x] `constants.py`: **one** `DEFAULT_CONFIG_DIR = "./configs"`. It is read by `context.py` and
      `src/constants/command_line_interface.py:12` (`CONFIG_KWARGS`) — two non-test `src/` files —
      so under the Phase 6 rule it belongs in `constants.py`, which sits in the same package as
      the CLI kwargs. This is the only new constant this refactor adds there.

The five current declarations, confirmed by grep: `context.py:129`,
`constants/command_line_interface.py:12`, `fundamentals/kpi_catalogue.py:51`,
`fundamentals/cik_cutover.py:77`, `validate/fundamentals/validator.py:235`.
- [x] `context.py:129`: `read_config(path=config_path)`. Resolve to an absolute `Path` first, so
      the `@cache` keys downstream normalise (Phase 2.5).
- [x] `context.py`: add a `config_dir: Path` property alongside the existing `config`,
      `use_cache`, `save`, `random_state`. Risk zone; one property.
- [x] `utils/step.py`: expose `self._config_dir = context.config_dir`. Risk zone; one line.
- [x] Delete the four redundant declarations and read `context.config_dir` /
      `self._config_dir` instead:
      - `kpi_catalogue.py:51` `DEFAULT_CONFIG_DIR` — now imports it from `constants.py` and
        re-exports it (many fundamentals modules import the name from `kpi_catalogue`).
      - `cik_cutover.py:77` (bare literal) — ALREADY gone: current code is
        `config_dir: str | None = None`, resolved via `resolve_config_dir`, no literal to delete.
      - `validator.py:235` (bare literal — and it already imports from `kpi_catalogue` at `:57`)
      - `command_line_interface.py:12` -> the new constant
- [x] Thread it into the fundamentals loaders that currently default: `load_catalogue()` at
      `fetch_fundamentals_sec.py:945` (line drifted from `:872`), `build_history.py:1100`
      (`build_fundamentals_history`, the one call site with `context` in scope — the OTHER
      site at `build_ticker`, `:845-846`, is a pure-function fallback for direct/test callers
      and is left on its default per the "don't chase every signature" rule below).
      `load_guards()` similarly threaded via `build_history.py:1101`; `periods.py`'s own
      `load_guards()` defaults (3 call sites) are never hit on the production path because
      `build_ticker` always receives `guards=` explicitly. `load_cutovers()` threaded at
      `fetch_fundamentals_sec.py:967`. `load_field_map()` needed no change — its only callers
      (`merge_history.py`, `gap_check.py`) already receive an explicit `config_dir` from their
      own caller.
      Followed the convention `StepExtractFundamentalsSharadar` already used
      (`step_extract_fundamentals_sharadar.py:37`) — its `config_dir` default changed from the
      module constant to `config_dir or self._config_dir`, since `StepExtractAllData.run()`
      calls `.run()` with no `config_dir` at all and would otherwise silently keep ignoring `-c`.
- [x] Left the 17 other `config_dir` parameters repo-wide alone (`build_merged_history`,
      `run_gap_check`, `run_diagnostics`, `field_map.py`, `gap_check.py`'s own defaults) — all
      already receive an explicit value from their CLI command's `-c`, independent of this bug.

### Verification

- [x] Copied `configs/` to a scratch probe dir, changed `fundamentals_workers` 8 -> 3, called
      `get_config_context(probe_dir, ...)` directly: `context.config.data_extract
      .fundamentals_workers == 3`. **Failed before the fix** (always read `8` from the real
      `./configs`), **passes now**.
- [x] `rtk grep -rn '"\./configs"' src/` -> 1 hit (`constants.py`'s `DEFAULT_CONFIG_DIR`).
- [x] `get_config_context(probe_dir, ...)` -> `context.config_dir` resolves to that absolute
      path; printed as the sanity conclusion during verification.

---

## 5.2 EDGAR identity and the SEC session on `Context` ✅

Today: `set_identity` at `edgar_driver.py:41`, called per fetch-run from `:73` and
`fetch_13f.py:125`. `SEC_USER_AGENT` is read in **two** files with **two hand-written
`RuntimeError`s** (`edgar_driver.py:34-40`, `sec_utils.py:33-40`), and `sec_utils.py:64` re-reads
`os.getenv` on **every request**.

- [x] `Context.sec_user_agent` — a cached property that reads the env once and raises **one**
      well-worded error naming the variable and where to set it. Both hand-written raises are
      deleted (`edgar_driver.configure_identity` removed entirely; `sec_utils._sec_headers`
      removed entirely).
- [x] `Context.ensure_edgar_identity()` — idempotent; calls `set_identity` on first use and is a
      no-op after. `edgar_driver.py` (`run_edgar_fetch`) and `fetch_13f.py` call it instead of
      `configure_identity`/`set_identity`. `edgar` is imported LAZILY inside the method —
      `Context` is imported by every package, including ones that never touch SEC EDGAR.
- [x] `Context.sec_session` — one `requests.Session` with the SEC user agent pre-set on
      `session.headers`, used by `sec_utils.sec_get` and `bulk_cache.ensure_zip`. This kills the
      per-request `os.getenv` and gives connection reuse on the bulk-ZIP downloads.
      `sec_get` and `ensure_zip` both gained a leading `context: Context` parameter (neither
      took one before) — a small, traced sweep: `bulk_cache.ensure_zip` has exactly 4 callers,
      all with `context` in scope; `sec_get` has ~6, of which `edgar_fillings.list_filings`
      (1 caller, `fetch_def14a_llm.py`) and `fetch_superinvestors._edgar_cik_for_name` (already
      dependency-injected via `get_fn` for testability) needed their own signatures updated to
      pass `context` through. All touched tests updated to match (`test_edgar_incremental.py`,
      `test_def14a_llm.py`, `test_financial_notes.py`); `test_edgar_driver.py`'s
      `configure_identity` monkeypatches replaced with a no-op `ensure_edgar_identity` on the
      fake context stand-in.
- [x] **Bounded on purpose**: did **not** move the OpenAI client, the FRED client or the
      Sharadar key onto `Context` in this phase — noted here as the next sweep, not attempted.
- [x] Retry/backoff **not** added — untouched, as scoped.

### Verification

- [x] Test: unset `SEC_USER_AGENT` (via `unittest.mock.patch.dict`), assert **one** error
      message, from `Context.sec_user_agent`. Passed.
- [x] Test: call `ensure_edgar_identity()` twice with `edgar.set_identity` monkeypatched to a
      call counter -> ran once. Passed.
- [x] `rtk grep -rn "SEC_USER_AGENT" src/` -> 1 read site (`context.py`). Confirmed.
- [x] `rtk grep -rn "set_identity" src/` -> 1 call site (`context.py`). Confirmed.

---

## 5.3 `extraction_run` — the run ledger 🔄

Replaces `data/extraction_manifest.json`: a git-ignored, lock-free, non-atomic read-modify-write
whose parse failure **silently discards every table's history** (`run_manifest.py:60-62`), whose
`rows_added` and `updated_at` are write-only, and which is missing an entry for every
`sharadar_*` table, `fundamentals_history*`, `pension_facts`, `earnings_surprises`,
`insider_transactions` and 8 more.

### Schema (`src/data_store/schema.py`)

```python
# One row per (table, run). Keyed on `run_id` as well as the table name for the reason
# `fundamentals_check_run` learned the hard way (schema.py:592-598): two runs of DIFFERENT
# SCOPE on the same day must be able to coexist, or the second silently overwrites the
# first and every delta computed against it is nonsense.
extraction_run = Table(
    "extraction_run", ("table_name", "run_id"),
    KIND_AGGREGATE, date_col="run_date", ticker_col=None,
    date_type_cols=("run_date", "last_full_rescan_date"))
```

Columns:

| column | meaning |
|---|---|
| `table_name` | the target table, always via `name_of(Table)` |
| `run_id` | `sha256(run_date, sorted(tickers), full_flag)[:12]` — the `fundamentals_check_run` pattern |
| `scope_hash` | the same hash **without** the date, so two runs are comparable iff it matches |
| `run_date` | the run's logical date |
| `last_full_rescan_date` | carried forward when `is_full_rescan` is false — the one field `record_run` reads from the prior entry today (`:129-133`) |
| `tickers_requested` | **defined once**: the size of the scope the run was given. For a market-wide table (macro, 13F, bulk sets) it is `0`, and `0` means "not ticker-scoped" — never "no tickers" |
| `tickers_written` | tickers that produced at least one row |
| `tickers_failed` | tickers that raised or produced zero rows from >= 1 filing. **This column is what would have caught the `cols` bug in hour one.** |
| `rows_added` | rows written by this run |
| `is_full_rescan` | passed straight to `manifest_window`'s successor |
| `started_at`, `finished_at` | UTC; the pair the live run needed and did not have |
| `status` | `ok` / `partial` / `failed` |

- [x] `sql/schema.sql`: hand-spliced additive `CREATE TABLE IF NOT EXISTS` block (right before
      `fundamentals_check`, matching `schema.py`'s placement). **Did not regenerate**.
      `git diff --stat sql/schema.sql` shows +33/-0 — purely additive, confirmed. Added one
      index on `(table_name, run_date DESC)`, the only read pattern. Parses correctly via
      `ddl.existing_blocks` (checked).

### `run_manifest.py` -> the ledger — ⬜ DEFERRED, not started

**Why**: `main.py` was actively running the full extraction pipeline (started 16:54, PIDs
34756/12108) for the whole of this pass, and was still active — its manifest kept growing
(19 -> 23 tables) as later steps completed. Cutting `run_manifest.py` over to the DB now would
make `manifest_window` read an EMPTY `extraction_run` table for every fetch invoked before the
seed step runs, forcing a full rescan on anything triggered in that window (cost, not data
loss, but exactly the risk this section's own "Risks" table flags). Per the user's explicit
choice (asked mid-session): land the code, hold the cutover, resume once the live run is
confirmed finished.

**Resume point** — do this first, before anything else in this subsection:
1. Confirm `main.py` (PIDs above, or whatever superseded them) has exited.
2. `record_run(context, table, *, tickers_requested, tickers_written, tickers_failed,
   rows_added, is_full_rescan=False, run_date=None, status="ok")` — writes one row via
   `context.store.save(Tables.extraction_run, ...)`. Keyword-only after `table`.
3. `manifest_window` ported as a pure function of the entry dict, swapping only its source
   (latest DB row for the table instead of the JSON entry) — semantics unchanged, per the
   Risks table below.
4. `get_entry` -> `latest_run(context, table) -> dict | None`.
5. The corrupt-file branch disappears with the JSON; a DB read failure now raises.
6. Every `record_run(context, table, ticker_count, rows_added, is_full_rescan=...)` call site
   added or touched in "Fill the gaps" below is on the OLD positional signature — it was
   written that way deliberately, so it would not depend on this deferred cutover. **All of
   them need converting to the new keyword-only `tickers_requested=`/`tickers_written=`/
   `tickers_failed=` form in the same change that lands this section**, including the ones
   this pass added (`fetch_sharadar.py` x4, `build_history.py` x2, `merge_history.py`,
   `fetch_financial_statements.py`) and every pre-existing site (`edgar_driver.py`,
   `fetch_13f.py`, `fetch_prices.py`, `fetch_short_interest.py`, `fetch_dividends.py`,
   `fetch_insider_transactions.py`, `fetch_fails_to_deliver.py`, `fetch_earnings_surprises.py`,
   `fetch_google_trends.py`, `fetch_wiki_pageviews.py`, `fetch_financial_notes.py`,
   `fetch_earnings_calls.py`). None of them currently pass `tickers_written`/`tickers_failed`
   (the old signature has no such concept) — decide per-writer whether to compute them or pass
   `None`.

### Seeding and cutover — the sequencing hazard

The first run after cutover must not trigger a ~10 h full rescan just because the table is empty.
And there is a second, sharper hazard while the walk is in flight:

> **The running process will read-modify-write `data/extraction_manifest.json` when its walk
> completes** (`edgar_driver.py:143-144`), using the **old** `record_run` code it already
> imported. If the JSON has been deleted by then, `_load_manifest` returns `{}` (`:60-62`) and the
> file is rewritten containing **only the fundamentals entries — silently destroying the other 15
> tables' history**, which per `manifest_window:89` means a full rescan for every one of them.

So the code lands now; the **data operation waits**:

- [x] **Right now, before anything else**: copied `data/extraction_manifest.json` to
      `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/manifest-snapshot.json`
      (refreshed once more mid-pass as the live run grew it from 19 to 23 tables). Plain copy,
      the live file was never touched.
- [x] **Did not delete or edit the live JSON** — confirmed untouched throughout.
- [x] **Did not run any fetcher** (only unit tests against synthetic/DB fixtures; no real
      SEC/Sharadar network call was made from this session).
- [x] `scripts/seed_extraction_run.py` written (dry-run verified against the live, still-growing
      JSON — read-only, 23 tables parsed correctly). **Not run with `--yes`** — that is the
      resume point once `run_manifest.py` is cut over and the live run has finished. Includes a
      `--verify` mode that inlines the OLD JSON-era `manifest_window` fallback logic (since by
      the time this script is actually run, `run_manifest.py` will no longer contain it) and
      compares it against the new DB-backed `manifest_window` for every table.
- [ ] **Verify then delete**: not reached — depends on the `run_manifest.py` cutover above.
- [ ] Note the known consequence about `fundamentals_facts`' `ticker_count: 1`: still applies,
      unchanged, whenever seeding actually runs.
- [ ] Delete `_MANIFEST_FILENAME`, `_manifest_path`, `_load_manifest`: not reached (the cutover
      hasn't happened, so this code still runs the live process's manifest writes). The
      `.gitignore` line the plan names (`:120`) is the generic `data/` folder ignore, not a
      manifest-specific entry — it also covers other local caches (SEC bulk zips, JSON caches)
      that must stay git-ignored regardless of this cutover, so nothing there should be deleted
      even once the JSON itself is retired.

### Fill the gaps

`record_run` fires at 4 sites in 2 of 12 fundamentals files today; both main fetchers record
nothing. Add it to every writer:

| Writer | Table(s) | Site |
|---|---|---|
| `build_fundamentals_history` | `fundamentals_history_sec`, `fundamentals_reason_codes` | `build_history.py:1088, :1090` | ✅ |
| `fetch_financial_statements` | `pension_facts` | `:167` | ✅ |
| `fetch_sharadar_tickers` | `sharadar_tickers` | `fetch_sharadar.py:100` | ✅ |
| `fetch_sharadar_fundamentals` | `sharadar_fundamentals` | `fetch_sharadar.py:166` | ✅ |
| `_fetch_dated_table` x2 | `sharadar_actions`, `sharadar_sp500` | `fetch_sharadar.py:196` | ✅ |
| `build_merged_history` | `fundamentals_history` | `merge_history.py:546` | ✅ |

All six added using the CURRENT (pre-cutover) positional `record_run` signature — they need
converting to the new keyword-only form when the cutover above lands (noted there).

- [x] Normalised the existing sites: `fetch_google_trends.py` (both call sites) ->
      `Tables.google_trends`, `fetch_wiki_pageviews.py` (both) -> `Tables.wiki_pageviews`
      (needed a new `Tables` import there), `fetch_financial_notes.py` (both) ->
      `Tables.notes_num` / `Tables.notes_text` — which also let the module's own
      `_NUM_TABLE`/`_TXT_TABLE` string constants be deleted entirely (5.4's table-name-as-
      constant rule), not just the `record_run` calls.
- [x] `fetch_insider_transactions.py:259` — checked; **not a bug in the current tree**.
      `tickers` is rebound to the deduped scope SET at `:231` and `record_run` at the (now)
      current line reads that same set — `len(tickers)` there already IS the scope size. This
      item appears to already be resolved (by earlier, unrelated work on this file) or the plan
      is describing a state that no longer matches the tree; no change made.
- [ ] `fetch_macro.py:242` dual-meaning fix — **deferred with the cutover**: fixing it now would
      change the recorded `ticker_count` for `prices_macro` (currently `15`, the series count)
      to `0` under the OLD JSON-based `manifest_window`, which compares that value across runs
      to decide "did the universe change size" — changing it pre-cutover would itself trigger a
      spurious full rescan on the very next macro run, which is exactly the hazard this section
      exists to avoid. Do this together with the signature migration.
- [ ] `fetch_def14a_llm.py` early-return `status="partial"` recording — **deferred**: the
      current `record_run` signature has no `status` parameter to pass.
- [x] `fetch_earnings_calls.py`: moved `record_run` out of the top-level `fetch_earnings_calls`
      wrapper and into `ingest_all_earnings_calls` itself (right after both `ingest_hf_transcripts`
      + `ingest_earnings_calls` have written), so the CLI's `ingest-earnings-calls` command and
      the Airflow DAG's separate `ingest-earnings-calls` task now also record a run — previously
      only the monolithic `fetch_earnings_calls` (main.py / tests) did.
- [x] The `fetch_earnings_surprises` double-record — confirmed already fixed (Phase 1), no
      action needed here.

### Retire the phantom sidecar ✅

`context.py:34-49` documents a `_meta.json` sidecar in detail. It exists in **zero** fetchers —
`_meta.json` appears only in that docstring and `docs/data_conventions.md:207`. The live
convention is `<cache_dir>/<table>_universe.json` (`sec_utils.py:97-113`).

- [x] Deleted the docstring and the two path keys it anchored (`DEF14A_LLM_PATH`,
      `SEC_13F_INSIDERS_DIR` — confirmed 0 reads each before deleting: `SEC_13F_INSIDERS_DIR`
      was named only in a `bulk_cache.py` docstring example, never actually passed by any
      caller; `DEF14A_LLM_PATH` had no reader at all outside `context.py` itself). Also updated
      the now-stale `bulk_cache.cache_dir` docstring example.
- [x] Fixed `docs/data_conventions.md:207` to document `<table>_universe.json` instead.

---

## 5.4 `fetch_financial_statements` onto the Step chain ✅

It writes `pension_facts` (`:167`) and is reached only by `cli.py:371-373` and
`dag_data_extraction.py:89`; the five siblings are wired at `step_extract_fundamentals.py:19-23`.

- [x] Imported and called in `StepExtractFundamentals.run`, after `fetch_financial_notes` (both
      read the same SEC bulk data sets, so cache locality is real).
- [x] It now records a run (5.3, current positional signature) and already logged adequately
      (`logger.info("pension_facts: upserted %d rows...")`, pre-existing).
- [x] **Behaviour change, stated here for the DoD report**: `main.py` will start writing
      `pension_facts` on its next full run. `fetch_financial_notes` remains the heavier of the
      two SEC bulk downloads (~300–450 MB/set, ~26 GB at `notes_years_history=15`);
      `fetch_financial_statements` shares its cache convention (`cache_dir`/`ensure_zip`) but is
      a separate, smaller download (`SEC_FINSTMT_URL_TEMPLATE`, the plain Financial Statement
      Data Sets, not the heavier Notes sets) — confirm the added budget is acceptable before the
      first full run reaches this step.
- [x] Fixed `fetch_financial_statements.py:42` `_TABLE = "pension_facts"` -> `Tables.pension_facts`
      (`schema.py`), all 4 uses. Fixed `fetch_financial_notes.py:70-71` `_NUM_TABLE` /
      `_TXT_TABLE` -> `Tables.notes_num` / `Tables.notes_text` the same way, including the
      `store.save` / `bulk_ingested_quarters` / `load_processed_universe` /
      `save_processed_universe` call sites, not just `record_run`.

**Known pre-existing conflict, not caused by this pass**: `tests/data_extract
/test_step_extract_fundamentals.py` asserts `StepExtractFundamentals.run`'s active call
sequence and its own docstring states "`fetch_financial_statements` is no longer imported by
this module... not exercised here". That test was ALREADY failing before this pass (confirmed
in a full-suite baseline run taken right after 5.1, before any 5.4 edit) for an unrelated
reason: `EXPECTED_SOURCES` still expects `fetch_insider_transactions` inside this module and
`fetch_fundamentals_sec`/`build_fundamentals_history` as ACTIVE (non-commented) calls, but the
CURRENT (uncommitted, in-flight from another phase) `step_extract_fundamentals.py` has moved
`fetch_insider_transactions` to `StepExtractPrices` and commented the other two out. This
test's `EXPECTED_SOURCES` is stale relative to that OTHER in-flight change, independent of
5.4. Left untouched rather than guessed at — whoever finishes that other WIP should add
`fetch_financial_statements` to `EXPECTED_SOURCES` at the same time.

---

## Verification

- [x] `"$PY" -m pytest tests/data_extract -q` — full suite run TWICE this pass: once right
      after 5.1 (baseline: **18 failed, 487 passed, 4 skipped**, all confirmed pre-existing —
      `test_edgar_driver.py` x7 (a `CIK_MAPPING_COLUMNS`/fixture mismatch from earlier,
      unrelated in-flight work on `sec_utils.load_cik_mapping`), `test_step_extract_fundamentals
      .py` x2 (see 5.4's note), `test_sharadar_field_map.py` x5, `test_sharadar_merge.py` x2,
      `test_fundamentals_point_in_time.py` x1, `test_short_interest_resume.py` x1 — none touched
      by Phase 5). Targeted re-runs after every subsequent edit (edgar_driver, sec_utils,
      bulk_cache, edgar_fillings, def14a_llm, financial_notes, financial_statements,
      fetch_sharadar, build_history, merge_history, run_manifest, google_trends,
      wiki_pageviews, earnings_calls, superinvestors) all green beyond that same pre-existing
      set. Did not re-run the full 30-minute suite a second time given the targeted coverage
      matched every touched file 1:1.
- [ ] Ledger tests (`test_extraction_run_two_same_day_scopes_coexist`,
      `test_manifest_window_matches_json_semantics`, `test_seed_is_lossless`,
      `test_record_run_rejects_positional_counts`) — **not written**; they test the
      `run_manifest.py` DB cutover, which is deferred (see 5.3).
- [ ] `psql ... SELECT ... FROM extraction_run ...` — the table exists (verified via
      `ddl.existing_blocks` parsing the hand-spliced DDL) but has 0 rows until the seed script
      is actually run.
- [ ] `rtk grep -rn "extraction_manifest" src/ scripts/ docs/ .gitignore` -> still has hits
      (`run_manifest.py` itself, by design — the cutover is deferred). `scripts/
      seed_extraction_run.py` also matches (its own docstring + default path), expected.
- [x] `rtk grep -rn "_TABLE = \"" src/data_extract/` -> 0 hits for the two named in 5.4
      (`fetch_financial_statements.py`, `fetch_financial_notes.py`). Not swept repo-wide —
      `data_aggregate/utils/fundamentals/fundamental_features.py`'s `_NOTES_NUM_TABLE` is the
      same anti-pattern but outside this phase's named scope (tracked under Phase 6 instead).

## Risks

| Risk | Mitigation |
|---|---|
| The cutover triggers a full 10 h rescan | **Realised as a live constraint this pass, not just a hypothetical**: `main.py` was mid-flight the whole time, which is exactly why the cutover itself was deferred rather than landed alongside the schema. The seed script (written, dry-run verified) + its `--verify` mode + the recorded before/after `manifest_window` pairs remain the mitigation once the cutover resumes. Do not delete the JSON until those agree. |
| `manifest_window` semantics drift while being ported | Port it as a **pure function of the entry dict**, unchanged, and swap only the source of that dict. The parametrised test covers all four fallback branches. |
| A DB read failure now aborts a run that the JSON would have silently continued | Deliberate, documented in the docstring, and stated in the DoD report. A run that cannot know its own window should not guess it. |
| `sql/schema.sql` regeneration drops the 8 hand-added indexes | Hand-splice only; `git diff --stat sql/schema.sql` must show additions and zero deletions. |
| Touching `context.py` breaks an unrelated consumer | `context.py` is imported everywhere; run the **full** suite, not just `tests/data_extract`. |
| `pension_facts` starts downloading on every `main.py` run | Flagged for explicit confirmation before the first full run. |
