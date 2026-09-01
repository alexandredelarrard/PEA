---
type: REFACTOR
session_id: a71e5b62-404b-441f-a135-b98a87481acf
generated_at: 2026-08-27T23:01:27+00:00
baseline: {head_sha: e8740ad2039c37944f05607d3d07dc4b6f1478aa}
generator: scripts/dod/refactor_metrics.py@1
---

## 1. Scope

**Files written (50):** `docs/coding_standard.md`, `docs/runbook.md`, `reports/2026-08-27/fundamentals-refactor-phase0-safety-net__REFACTOR.md`, `reports/2026-08-27/phase-1-fundamentals-defects-dead-code__REFACTOR.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/README.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/baseline.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/deferred.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/manifest-snapshot.json`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-0-safety-net.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-1-defects-dead-code.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-2-efficiency-constant-factor.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-3-efficiency-memoisation.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-4-efficiency-parallel-replay.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-5-wiring-context-ledger.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-6-constants-generalization.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-7-structure-prose-tests.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/post-run-checklist.md`, `reports/research/codebase/2026-08-27-refactor-fundamentals.md`, `specs/2026-08-26/refactor-fundamentals.md`, `src/constants/constants.py` … +30 more

**Sample scope:** whole repository working tree vs `e8740ad2039c` (a refactor's scope is the diff, not a data sample).

**What was asked:** implement Phase 1 of the fundamentals refactor
([plan](../planning/active-tasks/2026-08-27-refactor-fundamentals/phase-1-defects-dead-code.md)):
fix three live defects, delete dead code, break three name collisions, un-pin three silent-skip
tests, fix `Step._log` attribution and instrument the fundamentals step, and correct eight stale
citations — with **zero** behaviour change proven by the Phase 0 replay harness, so that Phases
2-4 optimise against a correct baseline rather than a `NameError`.

The scope table above is the whole working tree because Phase 0's untracked harness and the seven
plan documents were already uncommitted when this phase started; the **22 `src/` files** are this
phase's actual edit.

## 2. Gates

| Gate | Check | Verdict | Detail |
|---|---|---|---|
| G1 | targeted tests green (4 file(s)) | **PASS** | test: 100%\|##########\| 1/1 [00:00<00:00, 59.45it/s] |
| G2 | store boundary test green | **N/A** | no `src/data_store/` file touched |
| G3 | no new `print(` under src/ | **PASS** | none added |
| G4 | public API stable or call sites updated | **FAIL** | tests/data_extract/fundamentals/test_fundamentals_employees.py::pytest (still referenced) |
| G5 | docs moved with the code | **PASS** | 22 src file(s); docs touched: docs/coding_standard.md, docs/runbook.md |
| G6 | docstring lines did not shrink | **FAIL** | src/data_extract/utils/fundamentals/entity_scope.py 95->88; src/data_extract/utils/fundamentals/kpi_catalogue.py 230->229 |
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
| src/data_extract/utils/fundamentals/build_history.py | modified | 1,092 | 1,096 | 522 | 295 | 178 | 25 |
| src/data_extract/utils/fundamentals/cik_cutover.py | modified | 163 | 163 | 87 | 55 | 5 | 12 |
| src/data_extract/utils/fundamentals/entity_scope.py | modified | 239 | 210 | 63 | 88 | 37 | 8 |
| src/data_extract/utils/fundamentals/fetch_earnings_surprises.py | modified | 156 | 161 | 97 | 37 | 8 | 11 |
| src/data_extract/utils/fundamentals/fetch_financial_statements.py | modified | 172 | 173 | 115 | 28 | 12 | 17 |
| src/data_extract/utils/fundamentals/fetch_fundamentals_sec.py | modified | 901 | 946 | 456 | 329 | 103 | 46 |
| src/data_extract/utils/fundamentals/fundamentals_employees.py | modified | 179 | 181 | 71 | 78 | 14 | 12 |
| src/data_extract/utils/fundamentals/kpi_catalogue.py | modified | 680 | 676 | 303 | 229 | 73 | 15 |
| src/data_extract/utils/fundamentals/xbrl_linkbase.py | modified | 1,516 | 1,515 | 515 | 563 | 329 | 20 |
| src/utils/step.py | modified | 36 | 39 | 29 | 0 | 3 | 8 |
| src/validate/fundamentals/checks/tier3_internal.py | modified | 645 | 650 | 367 | 193 | 27 | 26 |
| tests/data_extract/common/test_edgar_driver.py | modified | 256 | 289 | 180 | 24 | 14 | 22 |
| tests/data_extract/fundamentals/replay_equality.py | new | — | 304 | 199 | 43 | 11 | 26 |
| tests/data_extract/fundamentals/test_fetch_earnings_surprises.py | modified | 47 | 86 | 55 | 13 | 2 | 8 |
| tests/data_extract/fundamentals/test_filing_rows_error_classes.py | new | — | 98 | 51 | 23 | 0 | 9 |
| tests/data_extract/fundamentals/test_fundamentals_employees.py | modified | 192 | 185 | 96 | 37 | 23 | 13 |
| tests/data_extract/fundamentals/test_fundamentals_point_in_time.py | modified | 210 | 207 | 115 | 67 | 0 | 8 |
| tests/data_extract/fundamentals/test_linkbase_empty_arcs.py | new | — | 94 | 51 | 22 | 3 | 7 |
| tests/data_extract/fundamentals/test_replay_equality.py | new | — | 124 | 90 | 12 | 0 | 9 |

**Totals**

| files_touched | python_files_touched | loc_after_total | docstring_lines_after_total | docstring_lines_before_total |
|---|---|---|---|---|
| 50 | 30 | 9,338 | 2,395 | 2,252 |

**Duplication** (shingle = 6 normalised code lines): 15 of 6,337 (0.2%) recur. Some duplication in this repo is deliberate and documented — read the docstring before removing any.

## 4. Evidence

- baseline: `e8740ad2039c`
- tests run: tests/data_extract/fundamentals/test_linkbase_empty_arcs.py, tests/data_extract/fundamentals/test_filing_rows_error_classes.py, tests/data_extract/fundamentals/test_fetch_earnings_surprises.py, tests/data_extract/common/test_edgar_driver.py
- non-Python files touched (20): docs/coding_standard.md, docs/runbook.md, reports/2026-08-27/fundamentals-refactor-phase0-safety-net__REFACTOR.md, reports/2026-08-27/phase-1-fundamentals-defects-dead-code__REFACTOR.md, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/README.md, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/baseline.md, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/deferred.md, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/manifest-snapshot.json, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-0-safety-net.md, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-1-defects-dead-code.md, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-2-efficiency-constant-factor.md, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-3-efficiency-memoisation.md
- pytest_summary: test: 100%|##########| 1/1 [00:00<00:00, 59.45it/s]
- pytest_targets: ['tests/data_extract/fundamentals/test_linkbase_empty_arcs.py', 'tests/data_extract/fundamentals/test_filing_rows_error_classes.py', 'tests/data_extract/fundamentals/test_fetch_earnings_surprises.py', 'tests/data_extract/common/test_edgar_driver.py']
- g6_note: A shrink is ALLOWED but must be justified in §5 -- say which docstring you removed and why it was not load-bearing.

## 5. Regressions, gaps and deliberate omissions

**The two standing gate FAILs**

- **G4 is a false positive by construction.** The removed name is `pytest` — the unused
  `import pytest` left behind in `test_fundamentals_employees.py` once its two `importorskip`
  pins went. G4 then runs `git grep -w pytest -- src tests scripts app`, which of course hits
  every other test file in the repo. Nothing references that import *from that module*; the file
  no longer contains the token at all. Restoring the import to turn the gate green would leave a
  dead import in a phase whose whole subject is dead code.
- **G6 is the deletion this phase was asked to make.** Docstrings shrank in exactly two files,
  and only inside symbols the plan named as dead: `entity_scope.py` 95->88 (`us_gaap_only`,
  `dimensioned_facts` and `ENTITY_AXES`' comment) and `kpi_catalogue.py` 230->229
  (`regime_for_sub_industry`). Nothing load-bearing was lost — that method's measurement (the
  four forced GICS overrides, 37 live tickers) was **moved onto `regime_for_gics`**, the accessor
  that superseded it, which is why a whole deleted docstring nets −1 line. Repo-wide, docstring
  lines went **2,252 -> 2,395**.

**Where the plan was wrong, and what I did instead** (each is recorded in the plan file too)

- **The swallow was not where the plan said.** `filing_rows`' `try` wraps only `filing.xbrl()`
  and logs nothing; the resolver runs outside it. The `NameError` was actually caught by
  `edgar_driver._worker`'s blanket `except Exception`, logged `"fundamentals: NEM failed"` at
  WARNING, and cost the **whole ticker** — which is why NEM/MO/AIZ have zero rows rather than one
  gap each. Fixing only `filing_rows` would have left the live swallow in place, so the edit
  widened to `edgar_driver.py` (shared by the 8-K / 13D / DEF 14A / filing-text fetchers, which
  inherit the new behaviour) and `parallel_fetch.py`'s contract docstring.
- **Five of the eleven "dead" symbols were live.** `Kind`/`Sign` annotate `FieldSpec`;
  `EXTRACTED_KINDS` and `SCORED_TIERS` are read by `is_extracted`/`is_scored`; `COMBINED_INTO`
  and `BASIS_EX_IPRD` are members of `ALL_CODES`, the closed set `build_history` asserts every
  written row against. Deleting `BASIS_EX_IPRD` would have made a code the catalogue **does**
  emit illegal and failed that assertion — a live defect, not a cleanup. All five kept.
- **`build_history.py:460` reads nothing.** The plan said to "keep the read" of
  `data/total_liabilities_legs.json`; there is no read, only a docstring citing a deleted script.
  The citation now names the JSON as the artefact of record.
- **`test_financial_notes.py`'s monkeypatches were never a tripwire.** `monkeypatch.setattr`
  defaults to `raising=True`. Measured, not assumed: patching a renamed attribute raises
  `AttributeError`. Left unchanged; the proposed `fn._scrape_available_periods.__name__` fix
  would have been the same string one step later.

**Deliberately not done**

- `context.py`'s `DEF14A_LLM_PATH` / `SEC_13F_INSIDERS_DIR` — deferred to Phase 5, taking the
  option the plan's own table offers. It is a named risk zone and Phase 5 rewrites the file.
- `docs/coding_standard.md:18`'s stale "927 lines" — the plan itself defers it to Phase 6.
- The 19 non-`cols` zero-fact tickers. A walk is still running (472 -> 478 tickers during this
  phase), so the detector must be re-run when it stops before anything is concluded.

**Side effects I caused**

- Two of the three planted-`NameError` CLI runs completed normally and so called `record_run`:
  `data/extraction_manifest.json` now reads `ticker_count: 1, rows_added: 0` for
  `fundamentals_facts` and `fundamentals_employees`. `manifest_window` compares that count to the
  universe size, so the **next** run takes the full years-history window and marks itself a full
  rescan — more work, no data loss, and the in-flight walk overwrites it on completion.
- The third run (the one that failed, as designed) wrote nothing and recorded nothing:
  AIZ/MO/NEM are still at 0 rows, verified after the fact.

**Known gaps**

- The new log lines use `%s` lazy formatting, matching every other call in
  `edgar_driver.py` / `fetch_fundamentals_sec.py`. `docs/coding_standard.md` says to prefer
  f-strings in log calls; I followed the neighbours rather than churn five call sites, and the
  `%s` form is also the one that does not format on a suppressed level.
- `PROGRAMMING_ERRORS` includes `AttributeError`/`TypeError`/`KeyError`, which a sufficiently
  strange filing could in principle raise out of `rows_from_xbrl` and now abort a run rather than
  skip one filing. The `filing.xbrl()` boundary still swallows everything, which is where a
  malformed submission actually blows up, so the exposure is narrow — but it is real and the
  in-flight full walk is the first thing that will test it.
- Nothing is committed. All 22 `src/` files are in the working tree.

## 6. Next actions

- **Review and commit.** The plan asks Phases 2-4 to land one item per commit so a harness failure
  bisects; Phase 1 should be one commit before that starts.
- **Phase 2** (`phase-2-efficiency-constant-factor.md`) is unblocked: the harness is proven on a
  no-op change, and the frozen tier-A/tier-B inputs are reusable as-is
  (`<scratchpad>/frozen_a`, `frozen_b`, with `before_a` / `before_b` now being **Phase 1's**
  output, i.e. the correct baseline for Phase 2).
- **Re-run the zero-fact detector when the in-flight walk stops**, then decide remediation for
  whatever remains. `-F` is required and only works on genuinely empty tickers (see the new
  `docs/runbook.md` bullet).
- **Watch the first full walk under the new error contract.** A programming error now fails the
  run; if the in-flight one dies on an `AttributeError` from a strange filing, that is the
  narrow-exposure case named in §5 and the class list is where to look.

```json dod-metrics
{
  "baseline_head_sha": "e8740ad2039c37944f05607d3d07dc4b6f1478aa",
  "content_hash": "sha256:0cdd463363ce78d4c5f3a40935b73975d16233aa5447ef9ec15d7b8501472013",
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
      "duplicate_ratio": 0.0023670506548840145,
      "duplicated_shingles": 15,
      "shingles": 6337,
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
            "src/data_extract/utils/fundamentals/cik_cutover.py:24",
            "src/data_extract/utils/fundamentals/kpi_catalogue.py:30"
          ],
          "count": 2
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
        "code": 522,
        "comment": 178,
        "docstring_after": 295,
        "docstring_before": 293,
        "loc_after": 1096,
        "loc_before": 1092,
        "public_api_count": 25,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/fundamentals/cik_cutover.py": {
        "code": 87,
        "comment": 5,
        "docstring_after": 55,
        "docstring_before": 55,
        "loc_after": 163,
        "loc_before": 163,
        "public_api_count": 12,
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
        "code": 456,
        "comment": 103,
        "docstring_after": 329,
        "docstring_before": 317,
        "loc_after": 946,
        "loc_before": 901,
        "public_api_count": 46,
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
        "code": 303,
        "comment": 73,
        "docstring_after": 229,
        "docstring_before": 230,
        "loc_after": 676,
        "loc_before": 680,
        "public_api_count": 15,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/fundamentals/xbrl_linkbase.py": {
        "code": 515,
        "comment": 329,
        "docstring_after": 563,
        "docstring_before": 563,
        "loc_after": 1515,
        "loc_before": 1516,
        "public_api_count": 20,
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
      "docstring_lines_after_total": 2395,
      "docstring_lines_before_total": 2252,
      "files_touched": 50,
      "loc_after_total": 9338,
      "python_files_touched": 30
    }
  },
  "scope": {
    "baseline_sha": "e8740ad2039c37944f05607d3d07dc4b6f1478aa",
    "tests": [
      "tests/data_extract/fundamentals/test_linkbase_empty_arcs.py",
      "tests/data_extract/fundamentals/test_filing_rows_error_classes.py",
      "tests/data_extract/fundamentals/test_fetch_earnings_surprises.py",
      "tests/data_extract/common/test_edgar_driver.py"
    ],
    "touched": [
      "docs/coding_standard.md",
      "docs/runbook.md",
      "reports/2026-08-27/fundamentals-refactor-phase0-safety-net__REFACTOR.md",
      "reports/2026-08-27/phase-1-fundamentals-defects-dead-code__REFACTOR.md",
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
      "src/data_extract/utils/fundamentals/xbrl_linkbase.py",
      "src/utils/step.py",
      "src/validate/fundamentals/checks/tier3_internal.py",
      "tests/data_extract/common/test_edgar_driver.py",
      "tests/data_extract/fundamentals/replay_equality.py",
      "tests/data_extract/fundamentals/replay_sample.json",
      "tests/data_extract/fundamentals/test_fetch_earnings_surprises.py",
      "tests/data_extract/fundamentals/test_filing_rows_error_classes.py",
      "tests/data_extract/fundamentals/test_fundamentals_employees.py",
      "tests/data_extract/fundamentals/test_fundamentals_point_in_time.py",
      "tests/data_extract/fundamentals/test_linkbase_empty_arcs.py",
      "tests/data_extract/fundamentals/test_replay_equality.py"
    ]
  },
  "session_id": "a71e5b62-404b-441f-a135-b98a87481acf",
  "type": "REFACTOR"
}
```

