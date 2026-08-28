# Phase 5 — Wiring: `Context` owns the config dir and the EDGAR identity; runs go in the DB ⬜

**Goal**: one source of truth for `./configs`, one place the SEC identity is configured, and a
run ledger in Postgres that every fundamentals writer actually writes to.

Independent of Phases 2–4 — it can land alongside them.

**Risk zones, approved**: `src/context.py`, `src/utils/step.py`, `src/data_store/schema.py`,
`sql/schema.sql` (additive splice only).

---

## 5.1 `config_dir` — fix the discard, then thread it

### The root cause

`get_config_context(config_path, ...)` accepts the path and then calls
`read_config(path="./configs")` at **`context.py:129`**, hardcoded. `config_path` is used only in
the error message at `:133`. **The CLI's `-c` never reaches the config reader.** The code already
knows: `fetch_fundamentals_sec.py:869-871` says so in a comment.

The Airflow chain works by coincidence — `dag_data_extraction.py:65` passes
`-c /opt/airflow/project/configs` with `cwd=/opt/airflow/project`, so the absolute flag and the
relative default resolve to the same directory. The flag is still discarded.

### Changes

- [ ] `constants.py`: **one** `DEFAULT_CONFIG_DIR = "./configs"`. It is read by `context.py` and
      `src/constants/command_line_interface.py:12` (`CONFIG_KWARGS`) — two non-test `src/` files —
      so under the Phase 6 rule it belongs in `constants.py`, which sits in the same package as
      the CLI kwargs. This is the only new constant this refactor adds there.

The five current declarations, confirmed by grep: `context.py:129`,
`constants/command_line_interface.py:12`, `fundamentals/kpi_catalogue.py:51`,
`fundamentals/cik_cutover.py:77`, `validate/fundamentals/validator.py:235`.
- [ ] `context.py:129`: `read_config(path=config_path)`. Resolve to an absolute `Path` first, so
      the `@cache` keys downstream normalise (Phase 2.5).
- [ ] `context.py`: add a `config_dir: Path` property alongside the existing `config`,
      `use_cache`, `save`, `random_state`. Risk zone; one property.
- [ ] `utils/step.py`: expose `self._config_dir = context.config_dir`. Risk zone; one line.
- [ ] Delete the four redundant declarations and read `context.config_dir` /
      `self._config_dir` instead:
      - `kpi_catalogue.py:51` `DEFAULT_CONFIG_DIR`
      - `cik_cutover.py:77` (bare literal)
      - `validator.py:235` (bare literal — and it already imports from `kpi_catalogue` at `:57`)
      - `command_line_interface.py:12` -> the new constant
- [ ] Thread it into the fundamentals loaders that currently default: `load_catalogue()` at
      `fetch_fundamentals_sec.py:872`, `build_history.py:796` and `:1050`. `load_guards()` at
      `periods.py:73`. `load_cutovers()`. `load_field_map()`.
      Follow the convention `StepExtractFundamentalsSharadar` already uses
      (`step_extract_fundamentals_sharadar.py:37, :79`) — the step resolves it and passes it in,
      exactly as `step_extract_prices.py:12-14` documents for time windows.
- [ ] Leave the 17 other `config_dir` parameters repo-wide alone. This phase makes the value
      *correct and available*; it does not chase every signature.

### Verification

- [ ] `cp -r configs /tmp/configs_probe`, change one knob (e.g. `fundamentals_workers`), run
      `-c /tmp/configs_probe`, and assert the changed value is the one in force. **Today this
      test fails** — that is the point.
- [ ] `rtk grep -rn '"\./configs"' src/` -> 1 hit (the constant). Tests may keep their own.
- [ ] Test: `get_config_context("/tmp/configs_probe")` -> `context.config_dir` is that path and
      `read_config` was called with it. Prints the resolved path as the sanity conclusion.

---

## 5.2 EDGAR identity and the SEC session on `Context`

Today: `set_identity` at `edgar_driver.py:41`, called per fetch-run from `:73` and
`fetch_13f.py:125`. `SEC_USER_AGENT` is read in **two** files with **two hand-written
`RuntimeError`s** (`edgar_driver.py:34-40`, `sec_utils.py:33-40`), and `sec_utils.py:64` re-reads
`os.getenv` on **every request**.

- [ ] `Context.sec_user_agent` — a cached property that reads the env once and raises **one**
      well-worded error naming the variable and where to set it. Both hand-written raises are
      deleted.
- [ ] `Context.ensure_edgar_identity()` — idempotent; calls `set_identity` on first use and is a
      no-op after. `edgar_driver.py:73` and `fetch_13f.py:125` call it instead of `set_identity`.
- [ ] `Context.sec_session` — one `requests.Session` with the SEC user agent pre-set on
      `session.headers`, used by `sec_utils.sec_get` and `bulk_cache`. This kills the per-request
      `os.getenv` and gives connection reuse on the bulk-ZIP downloads.
- [ ] **Bounded on purpose**: do **not** move the OpenAI client (3 constructions, 2 key-precedence
      orders), the FRED client (constructed per `_fred_frame` call) or the Sharadar key onto
      `Context` in this phase. They are real duplication, they are recorded in the research, and
      they are not on the fundamentals path. Note them in the Phase 7 report as the next sweep.
- [ ] Retry/backoff is **not** added here. The repo sets no retry, timeout, throttle or `EDGAR_*`
      env var anywhere, and `common/rate_limit.call_with_retries` exists but is used only by the
      yfinance/Trends paths. Wiring it into the SEC path changes failure behaviour under load and
      deserves its own task — see the Phase 2.6 note.

### Verification

- [ ] Test: unset `SEC_USER_AGENT`, assert **one** error message, from `Context`.
- [ ] Test: call `ensure_edgar_identity()` twice, assert `set_identity` ran once (monkeypatch).
- [ ] `rtk grep -rn "SEC_USER_AGENT" src/` -> 1 read site.
- [ ] `rtk grep -rn "set_identity" src/` -> 1 call site.

---

## 5.3 `extraction_run` — the run ledger

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

- [ ] `sql/schema.sql`: hand-spliced additive `CREATE TABLE IF NOT EXISTS` block. **Do not
      regenerate** — the generator drops 8 hand-added indexes. The diff must be purely additive.
      Add one index on `(table_name, run_date DESC)`, which is the only read pattern.

### `run_manifest.py` -> the ledger

- [ ] `record_run(context, table, *, tickers_requested, tickers_written, tickers_failed,
      rows_added, is_full_rescan=False, run_date=None, status="ok")` writes one row via
      `context.store.save`. Keyword-only after `table`, so the four different things
      `ticker_count` means today cannot be passed positionally by accident.
- [ ] `manifest_window` keeps **exactly its current semantics** — this is load-bearing and must
      not drift: fall back to `fallback_since` with `is_full_rescan=True` when there is no
      recorded run, when `tickers_requested` differs from the current universe size, or when the
      last full rescan is `>= full_rescan_days` old; otherwise return the last `run_date`,
      inclusive. It now reads the **latest row for that table** rather than the JSON entry.
      Keep the docstring's explanation of *why* the self-heal exists (a filing missed by a bug,
      or one EDGAR posts out of order, would otherwise stay missing forever).
- [ ] `get_entry` -> `latest_run(context, table) -> dict | None`.
- [ ] The corrupt-file branch (`:60-62`) disappears with the JSON. Note in the docstring that a
      DB read failure now **raises** rather than silently resetting every table's history — that
      is a deliberate behaviour change and it is the right one.

### Seeding and cutover — the sequencing hazard

The first run after cutover must not trigger a ~10 h full rescan just because the table is empty.
And there is a second, sharper hazard while the walk is in flight:

> **The running process will read-modify-write `data/extraction_manifest.json` when its walk
> completes** (`edgar_driver.py:143-144`), using the **old** `record_run` code it already
> imported. If the JSON has been deleted by then, `_load_manifest` returns `{}` (`:60-62`) and the
> file is rewritten containing **only the fundamentals entries — silently destroying the other 15
> tables' history**, which per `manifest_window:89` means a full rescan for every one of them.

So the code lands now; the **data operation waits**:

- [ ] **Right now, before anything else**: copy `data/extraction_manifest.json` to
      `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/manifest-snapshot.json`.
      It is git-ignored and about to be overwritten by the in-flight run anyway. This costs
      nothing and is the only copy of those 17 entries.
- [ ] **Do not delete or edit the live JSON while the walk is running.**
- [ ] **Do not run any fetcher before the seed.** `manifest_window` and `record_run` are only
      reached from the fetch path; the sample replays in Phases 0–4 never touch them. So the
      window between "code cut over to the DB" and "table seeded" is safe *as long as no fetch
      runs in it*. Write that as a one-line precondition at the top of the seed script.
- [ ] `scripts/seed_extraction_run.py` (one-shot, **post-run**): read the JSON's entries, write one
      `extraction_run` row each with `run_id = "seed"`, `status="ok"`,
      `tickers_written`/`tickers_failed` = `NULL` (unknown, and honestly so),
      `tickers_requested` = the JSON's `ticker_count`. Idempotent — re-running must upsert, not
      duplicate.
- [ ] **Verify then delete**: after seeding, assert `manifest_window` returns the same
      `(since, is_full_rescan)` pair from the DB that it returned from the JSON, for every table.
      Only then delete the JSON. Write the before/after pairs into the Phase 7 report — that table
      is the proof the cutover was lossless.
- [ ] Note the known consequence: `fundamentals_facts` currently carries `ticker_count: 1`
      because `record_run` only fires when the walk completes, and per `manifest_window:89` that
      **guarantees** a full rescan next run. Seeding preserves that — correctly. The in-flight run
      overwrites it on completion, which is the value that should be seeded.
- [ ] Delete `_MANIFEST_FILENAME`, `_manifest_path`, `_load_manifest`, and the
      `.gitignore:120` entry — in the same commit as the verified delete, not before.

### Fill the gaps

`record_run` fires at 4 sites in 2 of 12 fundamentals files today; both main fetchers record
nothing. Add it to every writer:

| Writer | Table(s) | Site |
|---|---|---|
| `build_fundamentals_history` | `fundamentals_history_sec`, `fundamentals_reason_codes` | `build_history.py:1088, :1090` |
| `fetch_financial_statements` | `pension_facts` | `:167` |
| `fetch_sharadar_tickers` | `sharadar_tickers` | `fetch_sharadar.py:100` |
| `fetch_sharadar_fundamentals` | `sharadar_fundamentals` | `fetch_sharadar.py:166` |
| `_fetch_dated_table` x2 | `sharadar_actions`, `sharadar_sp500` | `fetch_sharadar.py:196` |
| `build_merged_history` | `fundamentals_history` | `merge_history.py:546` |

- [ ] Also normalise the existing sites: 6 pass a **raw string** instead of `Tables.<name>`
      (`fetch_google_trends.py:375, :430`, `fetch_wiki_pageviews.py:182, :191`,
      `fetch_financial_notes.py:321, :322`). `name_of` tolerates it by design
      (`schema.py:752-760`), but the registry is the convention.
- [ ] `fetch_insider_transactions.py:259` passes `len(tickers)` where `tickers` was rebound to a
      **set** at `:231`. Fix to the scope size.
- [ ] `fetch_macro.py:242` passes 15 macro **series** as `ticker_count`, with a hardcoded `0` on
      the skip path (`:224`). Under the new definition it passes `tickers_requested=0` and puts
      the series count in `rows_added`'s sibling — or logs it. Do not overload one column with
      two meanings again.
- [ ] Early returns that record nothing: `fetch_def14a_llm.py:581-585` (same-day skip) and
      `:609-612` (no `OPENAI_API_KEY`). Record a row with `status="partial"` and
      `rows_added=0`, so "we chose not to run" is distinguishable from "we never ran".
- [ ] `fetch_earnings_calls.py` writes at `:426` and records at `:497`, and `cli.py:491` calls
      `ingest_all_earnings_calls` **directly**, bypassing the recording wrapper. Move the
      `record_run` next to the write.
- [ ] The `fetch_earnings_surprises` double-record is fixed in Phase 1.

### Retire the phantom sidecar

`context.py:34-49` documents a `_meta.json` sidecar in detail. It exists in **zero** fetchers —
`_meta.json` appears only in that docstring and `docs/data_conventions.md:207`. The live
convention is `<cache_dir>/<table>_universe.json` (`sec_utils.py:97-113`).

- [ ] Delete the docstring and the two path keys it anchors (`DEF14A_LLM_PATH`,
      `SEC_13F_INSIDERS_DIR`, 0 reads each). Coordinate with Phase 1 item 2 — do it once.
- [ ] Fix `docs/data_conventions.md:207` to document `<table>_universe.json` instead.

---

## 5.4 `fetch_financial_statements` onto the Step chain

It writes `pension_facts` (`:167`) and is reached only by `cli.py:371-373` and
`dag_data_extraction.py:89`; the five siblings are wired at `step_extract_fundamentals.py:19-23`.

- [ ] Import and call it in `StepExtractFundamentals.run`, after `fetch_financial_notes` (both
      read the same SEC bulk data sets, so cache locality is real).
- [ ] It now records a run (5.3) and logs (Phase 1 item 5).
- [ ] **State the behaviour change plainly in the DoD report**: `main.py` starts writing
      `pension_facts`. Confirm the cache and the download budget are acceptable before the first
      full run — `fetch_financial_notes` is already the heaviest download in the repo
      (~300–450 MB per set, ~26 GB back-fill at `notes_years_history=15`), and this shares its
      cache dir.
- [ ] Fix `fetch_financial_statements.py:42` `_TABLE = "pension_facts"` — a table name as a
      module constant is explicitly forbidden; `Tables.pension_facts` exists at `schema.py:282`.
      Same for `fetch_financial_notes.py:70-71` `_NUM_TABLE` / `_TXT_TABLE`.

---

## Verification

- [ ] `rtk "$PY" -m pytest tests/data_extract -v -s`
- [ ] Ledger tests, each printing its conclusion:
      - `test_extraction_run_two_same_day_scopes_coexist` — two runs, different ticker scopes,
        same day: **2 rows**, neither overwritten. This is the regression the validator paid for.
      - `test_manifest_window_matches_json_semantics` — parametrised over the 4 fallback
        branches, asserting the same `(since, is_full_rescan)` the JSON version returned.
      - `test_seed_is_lossless` — build a fixture manifest, seed, assert `manifest_window` agrees
        for all 17 tables.
      - `test_record_run_rejects_positional_counts` — the keyword-only signature.
- [ ] `MSYS_NO_PATHCONV=1 docker exec pea_db psql -U alexandre -d pea -c "SELECT table_name, run_date, tickers_requested, tickers_written, tickers_failed, rows_added, status FROM extraction_run ORDER BY run_date DESC LIMIT 20;"`
- [ ] `rtk grep -rn "extraction_manifest" src/ scripts/ docs/ .gitignore` -> 0 hits after cutover.
- [ ] `rtk grep -rn "_TABLE = \"" src/data_extract/` -> 0 hits.

## Risks

| Risk | Mitigation |
|---|---|
| The cutover triggers a full 10 h rescan | The seed script + `test_seed_is_lossless` + the recorded before/after `manifest_window` pairs. Do not delete the JSON until those agree. |
| `manifest_window` semantics drift while being ported | Port it as a **pure function of the entry dict**, unchanged, and swap only the source of that dict. The parametrised test covers all four fallback branches. |
| A DB read failure now aborts a run that the JSON would have silently continued | Deliberate, documented in the docstring, and stated in the DoD report. A run that cannot know its own window should not guess it. |
| `sql/schema.sql` regeneration drops the 8 hand-added indexes | Hand-splice only; `git diff --stat sql/schema.sql` must show additions and zero deletions. |
| Touching `context.py` breaks an unrelated consumer | `context.py` is imported everywhere; run the **full** suite, not just `tests/data_extract`. |
| `pension_facts` starts downloading on every `main.py` run | Flagged for explicit confirmation before the first full run. |
