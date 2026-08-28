---
type: REFACTOR
session_id: 806e2210-ff8b-49f6-acfb-2174c833db49
generated_at: 2026-08-28T03:17:22+00:00
baseline: {head_sha: e8740ad2039c37944f05607d3d07dc4b6f1478aa}
generator: scripts/dod/refactor_metrics.py@1
---

## 1. Scope

**Files written (57):** `docs/coding_standard.md`, `docs/runbook.md`, `reports/2026-08-27/fundamentals-refactor-phase0-safety-net__REFACTOR.md`, `reports/2026-08-27/phase-1-fundamentals-defects-dead-code__REFACTOR.md`, `reports/2026-08-27/phase-2-fundamentals-constant-factor__REFACTOR.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/README.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/baseline.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/deferred.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/manifest-snapshot.json`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-0-safety-net.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-1-defects-dead-code.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-2-efficiency-constant-factor.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-3-efficiency-memoisation.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-4-efficiency-parallel-replay.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-5-wiring-context-ledger.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-6-constants-generalization.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-7-structure-prose-tests.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/post-run-checklist.md`, `reports/research/codebase/2026-08-27-refactor-fundamentals.md`, `specs/2026-08-26/refactor-fundamentals.md` … +37 more

**Sample scope:** whole repository working tree vs `e8740ad2039c` (a refactor's scope is the diff, not a data sample).

**What was asked:** Phase 2 of `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/`
-- delete work that is provably independent of the loop variable it sits inside. 18 numbered
items, no algorithm change and no semantic change, each one "this is computed N times and the
answer never varies". The organising constraint is the user's: **no data bug for the sake of
speed**, so the phase is gated on Phase 0's cell-exact replay-equality harness reporting **zero**
differing cells -- not "within tolerance".

**Phase 2 itself touched 7 `src/` files** -- `fundamentals/{periods,build_history,kpi_catalogue,
xbrl_linkbase,fetch_fundamentals_sec,cik_cutover}.py` and `fundamentals_sharadar/field_map.py` --
plus 2 test files updated and 2 new test modules. The 56-file scope above is wider than that
because Phases 0 and 1 were left **uncommitted** in the working tree, so the generator's baseline
(`e8740ad`) sees all three phases at once. Sections 2 and 3 therefore describe Phases 0+1+2
together; section 5 separates them.

## 2. Gates

| Gate | Check | Verdict | Detail |
|---|---|---|---|
| G1 | targeted tests green (2 file(s)) | **PASS** | 8 passed in 0.60s |
| G2 | store boundary test green | **N/A** | no `src/data_store/` file touched |
| G3 | no new `print(` under src/ | **PASS** | none added |
| G4 | public API stable or call sites updated | **FAIL** | tests/data_extract/fundamentals/test_fundamentals_employees.py::pytest (still referenced) |
| G5 | docs moved with the code | **PASS** | 24 src file(s); docs touched: docs/coding_standard.md, docs/runbook.md |
| G6 | docstring lines did not shrink | **FAIL** | src/data_extract/utils/fundamentals/entity_scope.py 95->88 |
| G7 | AGENTS.md <= 70 lines | **PASS** | 70 lines |

**2 FAIL** — G4, G6. The work is **NOT done**.

## 3. Metrics

_Observations only — no verdicts. LOC is never a target (see [definition_of_done.md](../../docs/definition_of_done.md))._

**Per touched Python file**

| file | status | loc_before | loc_after | code | docstring | comment | public_api |
|---|---|---|---|---|---|---|---|
| src/constants/constants.py | modified | 1,058 | 1,059 | 345 | 6 | 613 | 1 |
| src/data_extract/step_extract_all_data.py | modified | 59 | 59 | 32 | 12 | 4 | 12 |
| src/data_extract/transformers/step_extract_fundamentals.py | modified | 60 | 96 | 43 | 22 | 16 | 12 |
| src/data_extract/utils/common/edgar_driver.py | modified | 144 | 163 | 90 | 25 | 29 | 18 |
| src/data_extract/utils/common/edgar_extract.py | modified | 210 | 210 | 107 | 32 | 53 | 5 |
| src/data_extract/utils/common/edgar_fillings.py | modified | 111 | 111 | 71 | 21 | 4 | 9 |
| src/data_extract/utils/common/gics.py | modified | 103 | 103 | 80 | 14 | 1 | 3 |
| src/data_extract/utils/common/llm_extractor.py | modified | 75 | 75 | 39 | 25 | 2 | 8 |
| src/data_extract/utils/common/parallel_fetch.py | modified | 51 | 56 | 16 | 34 | 0 | 8 |
| src/data_extract/utils/common/rate_limit.py | modified | 66 | 66 | 40 | 20 | 0 | 5 |
| src/data_extract/utils/common/run_manifest.py | modified | 143 | 143 | 76 | 48 | 0 | 14 |
| src/data_extract/utils/fundamentals/build_history.py | modified | 1,092 | 1,132 | 535 | 307 | 185 | 26 |
| src/data_extract/utils/fundamentals/cik_cutover.py | modified | 163 | 170 | 90 | 57 | 5 | 13 |
| src/data_extract/utils/fundamentals/entity_scope.py | modified | 239 | 210 | 63 | 88 | 37 | 8 |
| src/data_extract/utils/fundamentals/fetch_earnings_surprises.py | modified | 156 | 161 | 97 | 37 | 8 | 11 |
| src/data_extract/utils/fundamentals/fetch_financial_statements.py | modified | 172 | 173 | 115 | 28 | 12 | 17 |
| src/data_extract/utils/fundamentals/fetch_fundamentals_sec.py | modified | 901 | 973 | 471 | 336 | 104 | 47 |
| src/data_extract/utils/fundamentals/fundamentals_employees.py | modified | 179 | 181 | 71 | 78 | 14 | 12 |
| src/data_extract/utils/fundamentals/kpi_catalogue.py | modified | 680 | 740 | 330 | 259 | 73 | 18 |
| src/data_extract/utils/fundamentals/periods.py | modified | 984 | 1,040 | 433 | 381 | 137 | 20 |
| src/data_extract/utils/fundamentals/xbrl_linkbase.py | modified | 1,516 | 1,538 | 523 | 575 | 332 | 20 |
| src/data_extract/utils/fundamentals_sharadar/field_map.py | modified | 679 | 688 | 381 | 204 | 38 | 44 |
| src/utils/step.py | modified | 36 | 39 | 29 | 0 | 3 | 8 |
| src/validate/fundamentals/checks/tier3_internal.py | modified | 645 | 650 | 367 | 193 | 27 | 26 |
| tests/data_extract/common/test_edgar_driver.py | modified | 256 | 289 | 180 | 24 | 14 | 22 |
| tests/data_extract/fundamentals/replay_equality.py | new | — | 304 | 199 | 43 | 11 | 26 |
| tests/data_extract/fundamentals/test_build_history.py | modified | 572 | 577 | 378 | 72 | 57 | 33 |
| tests/data_extract/fundamentals/test_config_dir_cache.py | new | — | 63 | 34 | 17 | 1 | 11 |
| tests/data_extract/fundamentals/test_fetch_earnings_surprises.py | modified | 47 | 86 | 55 | 13 | 2 | 8 |
| tests/data_extract/fundamentals/test_filing_rows_error_classes.py | new | — | 98 | 51 | 23 | 0 | 9 |
| tests/data_extract/fundamentals/test_fundamentals_employees.py | modified | 192 | 185 | 96 | 37 | 23 | 13 |
| tests/data_extract/fundamentals/test_fundamentals_point_in_time.py | modified | 210 | 207 | 115 | 67 | 0 | 8 |
| tests/data_extract/fundamentals/test_linkbase_empty_arcs.py | new | — | 94 | 51 | 22 | 3 | 7 |
| tests/data_extract/fundamentals/test_per_filing_reuse.py | new | — | 107 | 61 | 19 | 3 | 7 |
| tests/data_extract/fundamentals/test_periods_q4.py | modified | 922 | 927 | 555 | 233 | 31 | 41 |
| tests/data_extract/fundamentals/test_replay_equality.py | new | — | 124 | 90 | 12 | 0 | 9 |

**Totals**

| files_touched | python_files_touched | loc_after_total | docstring_lines_after_total | docstring_lines_before_total |
|---|---|---|---|---|
| 57 | 36 | 12,897 | 3,384 | 3,117 |

**Duplication** (shingle = 6 normalised code lines): 21 of 9,067 (0.2%) recur. Some duplication in this repo is deliberate and documented — read the docstring before removing any.

## 4. Evidence

- baseline: `e8740ad2039c`
- tests run: tests/data_extract/fundamentals/test_config_dir_cache.py, tests/data_extract/fundamentals/test_per_filing_reuse.py
- non-Python files touched (21): docs/coding_standard.md, docs/runbook.md, reports/2026-08-27/fundamentals-refactor-phase0-safety-net__REFACTOR.md, reports/2026-08-27/phase-1-fundamentals-defects-dead-code__REFACTOR.md, reports/2026-08-27/phase-2-fundamentals-constant-factor__REFACTOR.md, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/README.md, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/baseline.md, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/deferred.md, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/manifest-snapshot.json, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-0-safety-net.md, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-1-defects-dead-code.md, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-2-efficiency-constant-factor.md
- pytest_summary: 8 passed in 0.60s
- pytest_targets: ['tests/data_extract/fundamentals/test_config_dir_cache.py', 'tests/data_extract/fundamentals/test_per_filing_reuse.py']
- g6_note: A shrink is ALLOWED but must be justified in §5 -- say which docstring you removed and why it was not load-bearing.

## 5. Regressions, gaps and deliberate omissions

- **Both gate failures belong to Phase 1, not to this phase, and are already justified in
  Phase 1's own report** (`phase-1-fundamentals-defects-dead-code__REFACTOR.md` section 5, which
  carries the identical G4/G6 lines). They reappear here only because Phase 1 is uncommitted, so
  this run's baseline predates it. **G4** is a false positive: the removed name is `pytest`, the
  unused `import pytest` left in `test_fundamentals_employees.py` once its two `importorskip`
  silent-skip pins became hard imports; grepping that file for `pytest` now returns nothing.
  **G6** is `entity_scope.py` 95->88 docstring lines, from deleting `ENTITY_AXES`,
  `us_gaap_only` and `dimensioned_facts` -- symbols the plan named as dead. Phase 2 shrank no
  docstring anywhere: its totals go **up** (repo-wide 3,117 -> 3,384).

- **G1 covers only the 2 new modules (8 tests, 1.3 s), which badly understates what was run.**
  The real gate for this phase is the replay-equality harness, which is not a pytest target.
  Measured: `pytest tests/data_extract/fundamentals` -- **231 passed, 0 failed** (15m 49s); plus
  `tests/validate tests/data_store tests/data_extract/sharadar` -- 210 passed, 11 skipped,
  **5 failed**, and all 5 **fail identically at HEAD `e8740ad`** in a clean worktree, so they
  are pre-existing and outside this phase (`test_sharadar_field_map.py` x4,
  `test_sharadar_merge.py::test_as_of_matches_sec`).

- **Two regressions were introduced; the suite caught both and both are fixed.** (a)
  `functools.cached_property` writes into the instance `__dict__`, so `FieldSpec(**real.__dict__)`
  -- the clone idiom in `test_periods_q4.py:40` -- began failing with `unexpected keyword
  argument '_never_use_by_regime'` as soon as anything asked for `never_use`. The test now uses
  `dataclasses.replace`. The failure was **order-dependent**, so it would otherwise have appeared
  at a random later date. (b) This phase's own edit to the NCI-bridge test broke a `pd.concat`
  that shared the same fixture helper.

- **Three items were not implemented as the plan wrote them, and one needed no change.** Item 7's
  prescribed fix was unsafe: `quarterize`'s `annual` frame carries **share-days**, not the
  as-reported twelve-month average, for a non-additive field, so passing it into
  `trailing_twelve` would have multiplied every annual share count by ~366. What shipped removes
  strictly more work with no semantic surface -- `_annual_by_end` is now built only on the
  `not is_additive` branch that reads it. Item 8 was already free (`guards or load_guards()`
  short-circuits and `build_ticker` resolves them before the loop). Item 18's gate is `any_of`
  **OR** a filer register entry, because route 3b's `groups` is their union and gating on the
  declaration alone would silently disable the route for a register-only filer. Item 10 needed
  no change: `_contradicts_gross_profit` has one caller and fires at most once per event, not
  once per field -- the call count was checked before touching it, as the plan instructed.

- **The measured speed-up is 1.19x, not the 2-3x the plan pre-registered.** HEAD 262/246/238 s
  against Phase 2 214/208/207 s, alternating on an idle machine. Stated rather than hunted,
  per the plan's own "if the gain is under 1.5x, say so plainly and move on to Phase 3". Two
  single readings taken earlier in the session -- a 789 s "baseline" and a 156 s "after" -- were
  machine-noise outliers; quoting either would have produced a fabricated 5x. **On this machine
  a single timing reading is worthless**; only paired, alternating, repeated runs are usable.

- **Items 12-18 are not covered by the equality harness.** It replays `build_ticker` off frozen
  facts and never calls the resolver, so the fetch path's gate is the test suite plus the three
  call-count pins in `test_per_filing_reuse.py`. A fetch-path acceptance belongs with the
  post-run wide check.

- **Item 3 was met only as a dtype guard.** `_latest_period_known` still scans the whole visible
  prefix once per event, which is the quadratic shape; removing it is the same vectorisation
  Phase 3 does for `carry_latest_known`.

- **Per-item commits were not made**, though the plan asked for 18. Phases 0 and 1 are
  uncommitted and their edits sit in the same files and functions, so splitting would have meant
  reconstructing another session's uncommitted work by hand. The bisection this was for is moot:
  every gate passed first time.

- **db-mode drift is real but pre-existing and provably not this phase's.** 31-49 drifted cells
  on 6 of the 8 sample tickers, all `stored=NaN -> rebuilt=<value>` at each ticker's earliest
  2011 `as_of`. `diff_against_stored` is a pure function of (stored, rebuilt), and the tier-B
  check shows this phase's `rebuilt` is cell-identical to HEAD's -- so the drift is a property of
  the stored table. It is `baseline.md` section 5's MCD finding, now visible on six more tickers.

- **One live-run finding, unrelated to this phase, recorded in `deferred.md`.** A `full=False`
  run logs `CPAY: 0 facts from 2 filing(s) (0 unreadable) -- the ticker's whole history is
  missing, not empty`. The two filings are FleetCor's first two after its 2010 IPO
  (`0001193125-11-078175`, `0001193125-11-140813`); **neither ships XBRL**, so `filing.xbrl()`
  returns `None` and `filing_rows` takes its clean early return. CPAY in fact holds 6,353 facts
  from 62 accessions. Phase 1's tripwire states something false on the incremental path, and
  will do so on every run forever, because a fact-less filing can never enter `done_accessions`.
  Not fixed here -- it is Phase 1's line and outside this phase's scope.

## 6. Next actions

- **Phase 3** (`phase-3-efficiency-memoisation.md`) is the one that has to deliver the wall
  clock: per-field memoisation of the period engine, a vectorised `carry_latest_known` (which
  also closes item 3 properly), projected reads, and the restatement census. The constant-factor
  work is done and it bought 1.19x; the irreducible cost is the per-window pandas indexing
  inside `quarterize`/`_ladder`/`trailing_twelve`, which is exactly where Phase 3 aims.
- **Decide whether to commit Phases 0-2.** They sit uncommitted in one working tree, which is
  why this report's scope and both gate failures span three phases. Committing them separately
  now, while the boundaries are still known, costs little and makes Phase 3's report clean.
- **Fix the `0 facts` tripwire's wording** (`fetch_fundamentals_sec.py:913`): count filings whose
  `xbrl()` is `None` separately and, when every walked filing is one, log at WARNING as
  "N pre-XBRL filing(s)" instead of asserting the history is missing. Then sweep for how many
  other tickers carry pre-XBRL filings permanently in the incremental window -- XBRL phased in
  over 2009-2011, so CPAY is unlikely to be alone.
- **The 5 pre-existing sharadar failures** need an owner. They are red at `e8740ad` and nothing
  in this plan's remaining phases touches them.
- **The MCD/ORCL/BA/BAC/BRK-B/APA stored-history staleness** needs a `build_fundamentals_history`
  re-run for those tickers; it is on the post-run checklist, not blocking.

```json dod-metrics
{
  "baseline_head_sha": "e8740ad2039c37944f05607d3d07dc4b6f1478aa",
  "content_hash": "sha256:0362d227e1e04494914923a1aea808ffb84450c17c429e6c5263725b9121eff6",
  "gates": {
    "G1": "PASS",
    "G2": "N/A",
    "G3": "PASS",
    "G4": "FAIL",
    "G5": "PASS",
    "G6": "FAIL",
    "G7": "PASS"
  },
  "generator": "scripts/dod/refactor_metrics.py@1",
  "metrics": {
    "duplication": {
      "duplicate_ratio": 0.0023160913201720527,
      "duplicated_shingles": 21,
      "shingles": 9067,
      "top_sites": [
        {
          "at": [
            "src/validate/fundamentals/checks/tier3_internal.py:192",
            "src/validate/fundamentals/checks/tier3_internal.py:240",
            "src/validate/fundamentals/checks/tier3_internal.py:287"
          ],
          "count": 3
        },
        {
          "at": [
            "src/validate/fundamentals/checks/tier3_internal.py:98",
            "src/validate/fundamentals/checks/tier3_internal.py:143"
          ],
          "count": 2
        },
        {
          "at": [
            "src/validate/fundamentals/checks/tier3_internal.py:99",
            "src/validate/fundamentals/checks/tier3_internal.py:144"
          ],
          "count": 2
        },
        {
          "at": [
            "src/validate/fundamentals/checks/tier3_internal.py:100",
            "src/validate/fundamentals/checks/tier3_internal.py:145"
          ],
          "count": 2
        },
        {
          "at": [
            "src/validate/fundamentals/checks/tier3_internal.py:202",
            "src/validate/fundamentals/checks/tier3_internal.py:250"
          ],
          "count": 2
        }
      ],
      "window_lines": 6
    },
    "per_file": {
      "src/constants/constants.py": {
        "code": 345,
        "comment": 613,
        "docstring_after": 6,
        "docstring_before": 6,
        "loc_after": 1059,
        "loc_before": 1058,
        "public_api_count": 1,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/step_extract_all_data.py": {
        "code": 32,
        "comment": 4,
        "docstring_after": 12,
        "docstring_before": 12,
        "loc_after": 59,
        "loc_before": 59,
        "public_api_count": 12,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/transformers/step_extract_fundamentals.py": {
        "code": 43,
        "comment": 16,
        "docstring_after": 22,
        "docstring_before": 13,
        "loc_after": 96,
        "loc_before": 60,
        "public_api_count": 12,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/common/edgar_driver.py": {
        "code": 90,
        "comment": 29,
        "docstring_after": 25,
        "docstring_before": 25,
        "loc_after": 163,
        "loc_before": 144,
        "public_api_count": 18,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/common/edgar_extract.py": {
        "code": 107,
        "comment": 53,
        "docstring_after": 32,
        "docstring_before": 32,
        "loc_after": 210,
        "loc_before": 210,
        "public_api_count": 5,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/common/edgar_fillings.py": {
        "code": 71,
        "comment": 4,
        "docstring_after": 21,
        "docstring_before": 21,
        "loc_after": 111,
        "loc_before": 111,
        "public_api_count": 9,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/common/gics.py": {
        "code": 80,
        "comment": 1,
        "docstring_after": 14,
        "docstring_before": 14,
        "loc_after": 103,
        "loc_before": 103,
        "public_api_count": 3,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/common/llm_extractor.py": {
        "code": 39,
        "comment": 2,
        "docstring_after": 25,
        "docstring_before": 25,
        "loc_after": 75,
        "loc_before": 75,
        "public_api_count": 8,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/common/parallel_fetch.py": {
        "code": 16,
        "comment": 0,
        "docstring_after": 34,
        "docstring_before": 29,
        "loc_after": 56,
        "loc_before": 51,
        "public_api_count": 8,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/common/rate_limit.py": {
        "code": 40,
        "comment": 0,
        "docstring_after": 20,
        "docstring_before": 20,
        "loc_after": 66,
        "loc_before": 66,
        "public_api_count": 5,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/common/run_manifest.py": {
        "code": 76,
        "comment": 0,
        "docstring_after": 48,
        "docstring_before": 48,
        "loc_after": 143,
        "loc_before": 143,
        "public_api_count": 14,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/fundamentals/build_history.py": {
        "code": 535,
        "comment": 185,
        "docstring_after": 307,
        "docstring_before": 293,
        "loc_after": 1132,
        "loc_before": 1092,
        "public_api_count": 26,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/fundamentals/cik_cutover.py": {
        "code": 90,
        "comment": 5,
        "docstring_after": 57,
        "docstring_before": 55,
        "loc_after": 170,
        "loc_before": 163,
        "public_api_count": 13,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/fundamentals/entity_scope.py": {
        "code": 63,
        "comment": 37,
        "docstring_after": 88,
        "docstring_before": 95,
        "loc_after": 210,
        "loc_before": 239,
        "public_api_count": 8,
        "public_api_removed": [
          "dimensioned_facts",
          "us_gaap_only"
        ],
        "status": "modified"
      },
      "src/data_extract/utils/fundamentals/fetch_earnings_surprises.py": {
        "code": 97,
        "comment": 8,
        "docstring_after": 37,
        "docstring_before": 37,
        "loc_after": 161,
        "loc_before": 156,
        "public_api_count": 11,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/fundamentals/fetch_financial_statements.py": {
        "code": 115,
        "comment": 12,
        "docstring_after": 28,
        "docstring_before": 27,
        "loc_after": 173,
        "loc_before": 172,
        "public_api_count": 17,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/fundamentals/fetch_fundamentals_sec.py": {
        "code": 471,
        "comment": 104,
        "docstring_after": 336,
        "docstring_before": 317,
        "loc_after": 973,
        "loc_before": 901,
        "public_api_count": 47,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/fundamentals/fundamentals_employees.py": {
        "code": 71,
        "comment": 14,
        "docstring_after": 78,
        "docstring_before": 76,
        "loc_after": 181,
        "loc_before": 179,
        "public_api_count": 12,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/fundamentals/kpi_catalogue.py": {
        "code": 330,
        "comment": 73,
        "docstring_after": 259,
        "docstring_before": 230,
        "loc_after": 740,
        "loc_before": 680,
        "public_api_count": 18,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/fundamentals/periods.py": {
        "code": 433,
        "comment": 137,
        "docstring_after": 381,
        "docstring_before": 365,
        "loc_after": 1040,
        "loc_before": 984,
        "public_api_count": 20,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/fundamentals/xbrl_linkbase.py": {
        "code": 523,
        "comment": 332,
        "docstring_after": 575,
        "docstring_before": 563,
        "loc_after": 1538,
        "loc_before": 1516,
        "public_api_count": 20,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/fundamentals_sharadar/field_map.py": {
        "code": 381,
        "comment": 38,
        "docstring_after": 204,
        "docstring_before": 201,
        "loc_after": 688,
        "loc_before": 679,
        "public_api_count": 44,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/utils/step.py": {
        "code": 29,
        "comment": 3,
        "docstring_after": 0,
        "docstring_before": 0,
        "loc_after": 39,
        "loc_before": 36,
        "public_api_count": 8,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/validate/fundamentals/checks/tier3_internal.py": {
        "code": 367,
        "comment": 27,
        "docstring_after": 193,
        "docstring_before": 188,
        "loc_after": 650,
        "loc_before": 645,
        "public_api_count": 26,
        "public_api_removed": null,
        "status": "modified"
      },
      "tests/data_extract/common/test_edgar_driver.py": {
        "code": 180,
        "comment": 14,
        "docstring_after": 24,
        "docstring_before": 16,
        "loc_after": 289,
        "loc_before": 256,
        "public_api_count": 22,
        "public_api_removed": null,
        "status": "modified"
      },
      "tests/data_extract/fundamentals/replay_equality.py": {
        "code": 199,
        "comment": 11,
        "docstring_after": 43,
        "docstring_before": null,
        "loc_after": 304,
        "loc_before": null,
        "public_api_count": 26,
        "public_api_removed": null,
        "status": "new"
      },
      "tests/data_extract/fundamentals/test_build_history.py": {
        "code": 378,
        "comment": 57,
        "docstring_after": 72,
        "docstring_before": 71,
        "loc_after": 577,
        "loc_before": 572,
        "public_api_count": 33,
        "public_api_removed": null,
        "status": "modified"
      },
      "tests/data_extract/fundamentals/test_config_dir_cache.py": {
        "code": 34,
        "comment": 1,
        "docstring_after": 17,
        "docstring_before": null,
        "loc_after": 63,
        "loc_before": null,
        "public_api_count": 11,
        "public_api_removed": null,
        "status": "new"
      },
      "tests/data_extract/fundamentals/test_fetch_earnings_surprises.py": {
        "code": 55,
        "comment": 2,
        "docstring_after": 13,
        "docstring_before": 7,
        "loc_after": 86,
        "loc_before": 47,
        "public_api_count": 8,
        "public_api_removed": null,
        "status": "modified"
      },
      "tests/data_extract/fundamentals/test_filing_rows_error_classes.py": {
        "code": 51,
        "comment": 0,
        "docstring_after": 23,
        "docstring_before": null,
        "loc_after": 98,
        "loc_before": null,
        "public_api_count": 9,
        "public_api_removed": null,
        "status": "new"
      },
      "tests/data_extract/fundamentals/test_fundamentals_employees.py": {
        "code": 96,
        "comment": 23,
        "docstring_after": 37,
        "docstring_before": 36,
        "loc_after": 185,
        "loc_before": 192,
        "public_api_count": 13,
        "public_api_removed": [
          "pytest"
        ],
        "status": "modified"
      },
      "tests/data_extract/fundamentals/test_fundamentals_point_in_time.py": {
        "code": 115,
        "comment": 0,
        "docstring_after": 67,
        "docstring_before": 67,
        "loc_after": 207,
        "loc_before": 210,
        "public_api_count": 8,
        "public_api_removed": null,
        "status": "modified"
      },
      "tests/data_extract/fundamentals/test_linkbase_empty_arcs.py": {
        "code": 51,
        "comment": 3,
        "docstring_after": 22,
        "docstring_before": null,
        "loc_after": 94,
        "loc_before": null,
        "public_api_count": 7,
        "public_api_removed": null,
        "status": "new"
      },
      "tests/data_extract/fundamentals/test_per_filing_reuse.py": {
        "code": 61,
        "comment": 3,
        "docstring_after": 19,
        "docstring_before": null,
        "loc_after": 107,
        "loc_before": null,
        "public_api_count": 7,
        "public_api_removed": null,
        "status": "new"
      },
      "tests/data_extract/fundamentals/test_periods_q4.py": {
        "code": 555,
        "comment": 31,
        "docstring_after": 233,
        "docstring_before": 228,
        "loc_after": 927,
        "loc_before": 922,
        "public_api_count": 41,
        "public_api_removed": null,
        "status": "modified"
      },
      "tests/data_extract/fundamentals/test_replay_equality.py": {
        "code": 90,
        "comment": 0,
        "docstring_after": 12,
        "docstring_before": null,
        "loc_after": 124,
        "loc_before": null,
        "public_api_count": 9,
        "public_api_removed": null,
        "status": "new"
      }
    },
    "totals": {
      "docstring_lines_after_total": 3384,
      "docstring_lines_before_total": 3117,
      "files_touched": 57,
      "loc_after_total": 12897,
      "python_files_touched": 36
    }
  },
  "scope": {
    "baseline_sha": "e8740ad2039c37944f05607d3d07dc4b6f1478aa",
    "tests": [
      "tests/data_extract/fundamentals/test_config_dir_cache.py",
      "tests/data_extract/fundamentals/test_per_filing_reuse.py"
    ],
    "touched": [
      "docs/coding_standard.md",
      "docs/runbook.md",
      "reports/2026-08-27/fundamentals-refactor-phase0-safety-net__REFACTOR.md",
      "reports/2026-08-27/phase-1-fundamentals-defects-dead-code__REFACTOR.md",
      "reports/2026-08-27/phase-2-fundamentals-constant-factor__REFACTOR.md",
      "reports/planning/active-tasks/2026-08-27-refactor-fundamentals/README.md",
      "reports/planning/active-tasks/2026-08-27-refactor-fundamentals/baseline.md",
      "reports/planning/active-tasks/2026-08-27-refactor-fundamentals/deferred.md",
      "reports/planning/active-tasks/2026-08-27-refactor-fundamentals/manifest-snapshot.json",
      "reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-0-safety-net.md",
      "reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-1-defects-dead-code.md",
      "reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-2-efficiency-constant-factor.md",
      "reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-3-efficiency-memoisation.md",
      "reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-4-efficiency-parallel-replay.md",
      "reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-5-wiring-context-ledger.md",
      "reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-6-constants-generalization.md",
      "reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-7-structure-prose-tests.md",
      "reports/planning/active-tasks/2026-08-27-refactor-fundamentals/post-run-checklist.md",
      "reports/research/codebase/2026-08-27-refactor-fundamentals.md",
      "specs/2026-08-26/refactor-fundamentals.md",
      "src/constants/constants.py",
      "src/data_extract/step_extract_all_data.py",
      "src/data_extract/transformers/step_extract_fundamentals.py",
      "src/data_extract/utils/common/edgar_driver.py",
      "src/data_extract/utils/common/edgar_extract.py",
      "src/data_extract/utils/common/edgar_fillings.py",
      "src/data_extract/utils/common/gics.py",
      "src/data_extract/utils/common/llm_extractor.py",
      "src/data_extract/utils/common/parallel_fetch.py",
      "src/data_extract/utils/common/rate_limit.py",
      "src/data_extract/utils/common/run_manifest.py",
      "src/data_extract/utils/fundamentals/build_history.py",
      "src/data_extract/utils/fundamentals/cik_cutover.py",
      "src/data_extract/utils/fundamentals/entity_scope.py",
      "src/data_extract/utils/fundamentals/fetch_earnings_surprises.py",
      "src/data_extract/utils/fundamentals/fetch_financial_statements.py",
      "src/data_extract/utils/fundamentals/fetch_fundamentals_sec.py",
      "src/data_extract/utils/fundamentals/fundamentals_employees.py",
      "src/data_extract/utils/fundamentals/kpi_catalogue.py",
      "src/data_extract/utils/fundamentals/periods.py",
      "src/data_extract/utils/fundamentals/xbrl_linkbase.py",
      "src/data_extract/utils/fundamentals_sharadar/field_map.py",
      "src/utils/step.py",
      "src/validate/fundamentals/checks/tier3_internal.py",
      "tests/data_extract/common/test_edgar_driver.py",
      "tests/data_extract/fundamentals/replay_equality.py",
      "tests/data_extract/fundamentals/replay_sample.json",
      "tests/data_extract/fundamentals/test_build_history.py",
      "tests/data_extract/fundamentals/test_config_dir_cache.py",
      "tests/data_extract/fundamentals/test_fetch_earnings_surprises.py",
      "tests/data_extract/fundamentals/test_filing_rows_error_classes.py",
      "tests/data_extract/fundamentals/test_fundamentals_employees.py",
      "tests/data_extract/fundamentals/test_fundamentals_point_in_time.py",
      "tests/data_extract/fundamentals/test_linkbase_empty_arcs.py",
      "tests/data_extract/fundamentals/test_per_filing_reuse.py",
      "tests/data_extract/fundamentals/test_periods_q4.py",
      "tests/data_extract/fundamentals/test_replay_equality.py"
    ]
  },
  "session_id": "806e2210-ff8b-49f6-acfb-2174c833db49",
  "type": "REFACTOR"
}
```

