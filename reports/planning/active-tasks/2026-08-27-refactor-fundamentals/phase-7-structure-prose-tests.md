# Phase 7 — Structure, prose, test infrastructure, docs ⬜

**Goal**: the shape the spec asked for — small single-purpose functions, files that fit in a head,
prose that describes today. Last, deliberately: this is the largest textual diff in the plan and
it must not obscure the equality-gated numerical work in Phases 2–4.

---

## 7.1 Split the nine functions over 80 LOC

`prices/` — the stated reference — has **no function over 73 LOC** across 60 functions.
`fundamentals/` has 9 over 80 across 186.

| LOC | function | file:line | Split along |
|---|---|---|---|
| **202** | `_resolve_once` | `xbrl_linkbase.py:1132` | the 5 resolution routes it dispatches (1, 2, 3, 3b, 4, 5) — one function per route, one dispatcher. Nesting hits 4 at `:1304`. |
| **130** | `_snapshot` | `build_history.py:584` | (a) build the row skeleton + regime + fiscal label, (b) the per-field value/reason loop, (c) the `totalLiabilities` identity bridge, (d) the guard/qualifier codes |
| **130** | `rows_from_xbrl` | `fetch_fundamentals_sec.py:669` | (a) scope, (b) linkbase load (once — Phase 2 item 12), (c) the per-field resolve loop, (d) row assembly. Nesting hits 4 at `:747`. |
| **126** | `_leaf_sum` | `xbrl_linkbase.py:1349` | the prologue (gated in Phase 2 item 18), the candidate walk, the sum-and-check |
| **102** | `_drop_annual_masquerading_as_quarter` | `periods.py:273` | its 6 terminal paths; nesting hits 4 at `:356` |
| 89 | `load_catalogue` | `kpi_catalogue.py:592` | read -> validate (6 passes) -> index. The 6 validation passes are already separable. |
| 84 | `resolve_field` | `xbrl_linkbase.py:1013` | route selection vs route execution |
| 82 | `quarterize` | `periods.py:530` | as-reported extraction, ladder invocation, share-day transform in/out, labelling |
| 82 | `_total_liabilities_identity` | `build_history.py:452` | the declared-legs read vs the fallback arithmetic |

Plus the two large classes: `ArcGraph` **186 lines** (`xbrl_linkbase.py:536`, 6 `cached_property`
+ 12 methods) and `Resolution` **118 lines** (`:300`, 15 dataclass fields carrying **70** `#:`
lines — most of the prose reduction in 7.3 lands here).

- [ ] Also fix the remaining nesting >= 4: `facts_frame_from_companyfacts:959` and
      `build_fundamentals_history:1073` (the latter simplifies naturally under Phase 4).
- [ ] **Rule while splitting**: extracted helpers keep the **same names** where a test pins them.
      26 private symbols are pinned by tests; the highest-traffic are `_drop_note_only_quarter`
      (13 test functions), `_filing_annual_windows` (6), `_hard_guard` (5),
      `_gross_profit_identity` (4), `_is_stale` (4), `_contradicts_gross_profit` (3), `_plan_fetch`
      (3). Where a rename is unavoidable, update the test **in the same commit** — never in a
      follow-up.
- [ ] `test_step_extract_fundamentals.py:25-31` pins five module attribute *names* on
      `step_extract_fundamentals` **and their call order** via `setattr`. A rename fails at
      `setattr`. Phase 5.4 adds a sixth call (`fetch_financial_statements`) — update this test
      there or here, once.

## 7.2 Split the oversized files

Target: **no file over 600 LOC** (`prices/` has none over 260).

- [ ] `xbrl_linkbase.py` **1,516 LOC / 40 functions / 25 globals** ->
      - `linkbase_graph.py` — `ArcGraph`, `ARC_COLUMNS`, `statement_arcs`, `qualify`,
        `is_income_statement_role`, `bare`
      - `resolve_routes.py` — the per-route resolvers extracted in 7.1
      - `resolve_field.py` — `Resolution`, `resolve_field`, the dispatcher
      - `leaf_sum.py` — `_leaf_sum` and its helpers
- [ ] `build_history.py` **1,092 LOC** ->
      - `history_snapshot.py` — `_snapshot` and its extracted parts
      - `history_identities.py` — `_total_liabilities_identity`, `_gross_profit_identity`,
        `_contradicts_gross_profit`, `_FORMULAS`
      - `build_history.py` — `build_ticker`, `build_fundamentals_history`, `_normalise_facts`,
        `diff_against_stored`, the dtype pinning
- [ ] `periods.py` **984 LOC** ->
      - `period_shapes.py` — `_shape`, `_latest_per_window`, `_DURATION_BANDS`,
        `_drop_annual_masquerading_as_quarter`, `_is_ambiguous_duration`
      - `period_ladder.py` — `quarterize`, `_ladder`, `_derived`, the share-day transform
      - `period_calendar.py` — `fiscal_year_ends`, `_fiscal_bounds`, `fiscal_quarter_of_end`,
        `label_fiscal_periods`
      - `periods.py` — `build_periods`, `trailing_twelve`, `PeriodGuards`, `load_guards`
- [ ] **Circular-import watch**: `build_history` and `periods` already import each other's
      private names (the `_QUARTER_COLUMNS` collision fixed in Phase 1 is the symptom). Draw the
      dependency graph before moving anything; shared primitives go to `src/utils/` per
      `docs/coding_standard.md`, never a cross-import between subfolders.
- [ ] `reason_codes.py` — **188 LOC, 0 functions, 0 classes, 15 constants, 138 lines of prose**
      (4.45 : 1). After 7.3 it should be ~50 lines. Keep it as a module — it is a vocabulary, and
      `rc` is pinned by tests as a module alias (12 attributes).
- [ ] Keep the module count sane: `fundamentals/` goes from 12 files to ~19. That is still fewer
      files per LOC than `prices/` (10 files / 1,640 LOC).

## 7.3 De-verbose: drop the chronology, keep the measurement

Decision 7. Roughly **250 comment/docstring blocks** narrate what the code used to do, what was
rejected, or which plan phase owns something.

| file | total | comment | docstring | code | prose : code | narrating lines |
|---|---|---|---|---|---|---|
| `xbrl_linkbase.py` | 1516 | 334 | 563 | 511 | **1.76 : 1** | 76 |
| `periods.py` | 984 | 121 | 365 | 416 | 1.17 : 1 | ~55 blocks |
| `fetch_fundamentals_sec.py` | 901 | 98 | 317 | 431 | 0.96 : 1 | 53 |
| `build_history.py` | 1092 | 180 | 293 | 518 | 0.91 : 1 | ~60 blocks |
| `reason_codes.py` | 188 | 105 | 33 | **31** | **4.45 : 1** | — |
| `entity_scope.py` | 239 | 40 | 95 | 78 | 1.73 : 1 | — |

### The rule, by example

Before (`configs/configs.yml`-adjacent style, `periods.py`):

> "Raised from 1.0 after the 1.0 bar was measured rejecting real, as-filed quarters at 1.03-2.6x:
> Allstate FY2023 (1.10x), Gilead FY2017 (1.26x, the TCJA writedown), S&P Global FY2014 (2.41x,
> the $1.6B legal settlement), Genuine Parts 2025 (2.39x), Zimmer Biomet 2018 (2.59x), Johnson
> Controls 2012 (2.59x), J.M. Smucker 2023 (1.99x)."

After:

> "3.0 clears every confirmed as-filed quarter measured on both rosters — the worst is Zimmer
> Biomet 2018 at 2.59x, where a one-off charge legitimately dwarfs the run-rate. Magnitude alone
> cannot separate that from a data error; the `non_negative` sign test does the real work and this
> only backstops signed fields."

The number survives, the evidence survives, the reasoning survives. The **history** goes.

### Mechanics

- [ ] Delete, everywhere under `src/data_extract/utils/fundamentals/`: `used to`, `previously`,
      `before the fix`, `no longer`, `an earlier version`, `until Phase`, `tried and reverted`,
      `REJECTED`, `the plan`, `Phase N`, `§N`, `decision #N`, `D<number>`, `register item N`,
      `plan-5b`, `4c.1`, `Replaces the`, `drifted`.
- [ ] Full list of plan cross-references to remove (from the research):
      `Phase 1/4b/5/5b/6 §6.1/7/10`, `plan-5b`, `§5.0`, `§5.1`, `§B.5`, `§B.6.6`, `4c.1`, `4c.8`,
      `register item 7/8/9`, `decision #9/24/28/30/31/32/33/34/35/37/40/46`,
      `D1/D1b/D7/D8/D11/D14/D15/D17/D18/D20/D21/D23/D24/D25/D27`.
- [ ] Where a decision number is the **only** identifier of a rule (e.g. "decision #9 defines
      `epsDiluted` as `netIncome_ttm / dilutedShares_ttm`"), restate the rule and drop the number.
- [ ] Module docstrings: `xbrl_linkbase.py` **lines 1–109** (7.2 % of the file) and
      `fetch_fundamentals_sec.py` 1–27. Target <= 20 lines each — what the module does, its
      inputs/outputs, and the one non-obvious constraint.
- [ ] The prose-heavier-than-code functions get one line each: `_gross_profit_identity` 19 doc /
      **2** code, `_one_share_basis` 16 / **2**, `_inclusive_days` 8 / **1**, `_latest_per_window`
      20 / **5**, `bare` 10 / **2**, `is_note_only` 25 / **5**, `sibling_leg` 56 / **14**.
      Exception: `_latest_per_window` and `_inclusive_days` encode a *non-obvious* invariant (the
      window identity is its end within a few days; `period_days` is one short and not additive).
      Keep 3–5 lines each, not 20 and not 1.
- [ ] Correct the claims that are now false — do not just shorten them:
      - `build_history.py:789` "The replay is O(filings)" -> state the measured complexity.
      - `kpi_catalogue.py:258` "with lookups precomputed" -> true after Phase 2.4; verify.
      - `xbrl_linkbase.py:1196-1197` `_leaf_sum` "is free when it does not apply" -> true after
        Phase 2.18; verify.
      - `bulk_cache.py:96-97` "all six fetchers now get the shared self-heal" -> true after Phase
        6.5; verify.
      - `_snapshot`'s `narrow` docstring cites "~14 minutes a ticker" as a prior state — keep the
        *reason* `narrow` exists (filters copy the whole frame), drop the before/after.
- [ ] `AGENTS.md` and `docs/coding_standard.md:45-55` both say these docstrings are "unusually
      load-bearing … several explicitly say the duplication is deliberate". That is not
      contradicted by this phase and it stays: **the test is whether a sentence describes today's
      behaviour or yesterday's diff.** Add that sentence to the standard so the rule is written
      down rather than re-argued next time.

## 7.4 Test infrastructure

Decision 10. Today: **no `conftest.py` under `tests/data_extract/`** (3 in the whole tree, none
here), **no pytest config anywhere** (no `[tool.pytest.ini_options]`, `pytest.ini`, `tox.ini` or
`setup.cfg`), so no markers and no `-m "not network"`.

- [ ] `pyproject.toml`: add `[tool.pytest.ini_options]` with `markers = ["network", "db", "slow"]`
      and `testpaths`. **Do not** add `-x` or coverage gates — CI is pylint-only on Python
      3.8/3.9/3.10 and has no pytest job; changing that is a separate task.
- [ ] `tests/data_extract/conftest.py`:
      - a **session-scoped** `catalogue` fixture. **9 files call `load_catalogue("./configs")` at
        module import** (`test_amendment_grain.py:29`, `test_build_history.py:26`,
        `test_leaf_sum_resolution.py:34`, `test_linkbase_history.py:37`,
        `test_linkbase_resolution.py:24`, `test_linkbase_sibling_total_1c9a517eaa47.py:35`,
        `test_periods_q4.py:29`, `test_segment_margin_876ab8a57bd8.py:66`,
        `test_statement_role_routes.py:33`), and because `test_kpi_catalogue.py:55` passes an
        **absolute** path, a run of that directory pays **2 full parses** (6 JSON reads) with all
        six validation passes each. Phase 2.5's key normalisation already fixes the double parse;
        the fixture also removes the **collection error in 9 modules at once** that an invalid
        catalogue causes today, even for a `-k`-filtered run.
      - one `context` fixture, replacing the **12 ad-hoc `get_config_context(...)` sites** and the
        **byte-for-byte duplicated** fixture in the 4 sharadar files
        (`test_fetch_sharadar.py:46-56`, `test_sharadar_diagnostics.py:51-61`,
        `test_sharadar_field_map.py:71-82`, `test_sharadar_merge.py:56-68`).
      - a `frozen_sample` fixture reading Phase 0's parquet, so the equality harness is available
        to any test.
- [ ] Mark the **31 network tests** (all currently gated by
      `if not os.getenv("SEC_USER_AGENT"): pytest.skip`) with `@pytest.mark.network` **in addition
      to** the existing skip — the skip is what makes a clean clone work, the marker is what makes
      `-m "not network"` work. `test_linkbase_history.py` is 7/7 network.
- [ ] Mark the 4 DB tests `db` and the 3 committed-parquet tests (`data/fundamentals_sweep`,
      which is **git-ignored**, so they skip on a clean clone) `slow` — and add a one-line comment
      saying they skip on a fresh clone by design.
- [ ] Cover the genuinely unguarded public surface (names appearing **nowhere** in `tests/`):
      `build_history.TickerHistory:127`, `facts_frame_from_companyfacts:933`,
      `cik_cutover.Cutover:54`, `fetch_fundamentals_sec.build_ticker_fundamentals:801` (**the
      per-ticker orchestrator**), `fundamentals_employees.filing_body_text:105`,
      `xbrl_linkbase.qualify:522`, `xbrl_linkbase.is_income_statement_role:927`,
      `xbrl_linkbase.bare:796` (25 apparent hits, all prose or a DataFrame column — never called).
      Prioritise `build_ticker_fundamentals` — it is the orchestrator this whole refactor moves
      around.
- [ ] **Never rename** the 4 hash-suffixed regression files. `cluster_id(ticker, field) =
      sha256(f"{ticker}\x1f{field}").hexdigest()[:12]` (`validate/fundamentals/finding.py:107-127`);
      `validate/cli.py:509-519` only checks that the path exists, so a rename silently invalidates
      the recorded `test_path` in `fundamentals_check_fix` **with no failing test**. The convention
      lives in `src/validate/README.md:336-338`.
      - [ ] Move that convention into `docs/testing.md` too, and add a check that every
            `test_*_<12hex>.py` name recomputes to a `(ticker, field)` that appears in the file.
            That turns a silent invalidation into a failing test.
- [ ] Note, do not fix: `tests/data_aggregate/aggregate_fingerprint.py:55` reads a **committed**
      `aggregate_fingerprint_fundamentals.parquet` whenever it exists, so the DB path (`:67-87`)
      never runs and **the fingerprint cannot catch fundamentals-extraction changes**.
      `docs/testing.md:154-169` forbids regenerating the baseline in a commit that touches `src/`,
      so this cannot be addressed inside this refactor. Record it in the report as a gap.

## 7.5 Docs

- [ ] `docs/coding_standard.md` — the new constants rule (Phase 6.1), the de-verbosing test
      sentence (7.3), and the function/file size targets this plan met.
- [ ] `docs/data_schema.md` + `docs/database.md` — `extraction_run`.
- [ ] `docs/data_conventions.md:207` — replace the phantom `_meta.json` with
      `<table>_universe.json`.
- [ ] `docs/architecture.md:88-110` — the extraction order now includes
      `fetch_financial_statements`.
- [ ] `docs/testing.md` — markers, the conftest fixtures, the `cluster_id` naming convention.
- [ ] `docs/config.md` — `fundamentals_replay_workers`; and that `-c` now actually reaches the
      config reader.
- [ ] `docs/runbook.md` — the seed script, and the `extraction_run` query for "what did the last
      run do?".
- [ ] `AGENTS.md` — **cap 70 lines**. Only if a hard rule changed: the `-c` fix and the
      `extraction_run` table are worth one line each; take one out if needed.
- [ ] `step_extract_all_data.py:4-11` — the docstring says "four sub-steps"; `__init__` builds
      five and the order differs from both `__init__` and `run`. Fix the **docstring** only; the
      commented-out steps and `full=True` are out of scope by decision 8.
- [ ] `step_extract_fundamentals.py:9` cites
      `reports/planning/active-tasks/2026-08-23-fundamentals-rebuild-plan-v2.md` — drop the
      citation, keep the two-layer explanation.
- [ ] `diagnostics.py:75` and `gap_check.py:61` carry **date-stamped report path literals in
      code** (two copies of `reports/planning/active-tasks/2026-08-26-sharadar-integration/...`).
      Move to a parameter or `context.paths`.

## 7.6 Final report (DoD)

- [ ] Classification is **DATA** — a new table, a schema change, changed fetchers, files under
      `src/data_extract` and `sql/`. Use the `dod-data-report` skill; the structural half of the
      work is summarised inside it rather than filed as a second report.
- [ ] Must contain, as measured numbers not claims:
      - the Phase 0 -> Phase 4 wall-clock table, per ticker with its event count, for **both
        sample tiers**, plus the measured shape of the E-curve against the research's `O(E²·K)`
        call-count claim;
      - the memo hit rate per ticker, and whether the memo was kept;
      - **0 differing cells** across Phases 2–4, on both tiers, in both harness modes, plus the
        sample-scoped live non-rebuild re-run result;
      - LOC / max-function-LOC / globals / prose:code before and after, per file;
      - `constants.py` line count and the consumer census before and after;
      - the `extraction_run` rows written by the sample runs;
      - `restatement-census.md`'s **indicative** (8-ticker) recommendation on the vintage redesign;
      - the deferred list (see [deferred.md](deferred.md)) and the outstanding
        [post-run-checklist.md](post-run-checklist.md) items, stated as outstanding rather than
        quietly omitted.
- [ ] **Do not claim the wide checks that have not run.** The ledger seed / JSON cutover, the
      full-universe replay acceptance, the whole-table census, the 12 excluded edge-case tickers
      and the `cols` coverage remediation are all post-run. The report says so explicitly.

## Verification

- [ ] `rtk "$PY" -m pytest tests/ -v -s` — full suite.
- [ ] `rtk "$PY" -m pytest tests/ -m "not network and not db" -v -s` — the marker split works.
- [ ] Phase 0 harness: **0 differing cells**. A 2,000-line prose diff must not move a number; if
      it does, a docstring was carrying code.
- [ ] Metrics gate: no file > 600 LOC, no function > 80 LOC, `fundamentals/` globals < 60,
      prose:code < 0.7 : 1 per file.
- [ ] `rtk grep -rnE "used to|previously|an earlier version|Phase [0-9]|decision #[0-9]" src/data_extract/utils/fundamentals/` -> **0 hits**.
- [ ] `wc -l AGENTS.md` -> <= 70.
- [ ] pylint on 3.8/3.9/3.10 (what CI actually runs) is clean on the changed files.
