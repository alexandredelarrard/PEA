---
type: REFACTOR
session_id: a4ba74e3-2fdf-406a-adfa-f2073d9b0fcf
generated_at: 2026-08-18T16:56:58+00:00
baseline: {head_sha: 53e7bebe0e44cdd73e7201ad8c129872ccfe5b4d}
generator: scripts/dod/refactor_metrics.py@1
---

## 1. Scope

**Files written (35):** `AGENTS.md`, `docs/coding_standard.md`, `docs/data_conventions.md`, `docs/data_schema.md`, `docs/data_sources.md`, `reports/2026-08-18/refactor-pricing__REFACTOR.md`, `reports/2026-08-18/refactor-short-interest__REFACTOR.md`, `specs/2026-08-17_scrapper_extract_fillings.md`, `specs/2026-08-18_refactor_pricing.md`, `src/data_aggregate/utils/common/prices.py`, `src/data_extract/cli.py`, `src/data_extract/step_extract_all_data.py`, `src/data_extract/transformers/step_extract_fundamentals.py`, `src/data_extract/transformers/step_extract_prices.py`, `src/data_extract/utils/common/incremental.py`, `src/data_extract/utils/common/run_manifest.py`, `src/data_extract/utils/fundamentals/fetch_macro.py`, `src/data_extract/utils/prices/fetch_dividends.py`, `src/data_extract/utils/prices/fetch_macro.py`, `src/data_extract/utils/prices/fetch_macro_assets.py` … +15 more

**Sample scope:** whole repository working tree vs `53e7bebe0e44` (a refactor's scope is the diff, not a data sample).

**What was asked:** Refactor `fetch_short_interest.py` the same way as the price/dividend
pass: (1) no string table names — use `Tables.<name>`; (2) stop loading the whole table to
work out the resume date, resolve it with a query instead; (3) drop functions no longer
needed (the merged `out` frame in particular); (4) update docs + `AGENTS.md` + tests;
(5) run the whole suite bar the known-broken ones.

**Design decision confirmed with the user before implementing:** resume on the table's
**global** max date (`store.max_date`), NOT the per-ticker `resume_since` used for
prices/dividends. RegSHO ships one market-wide file per business day, so once day D is
stored every ticker has D; a per-ticker frontier could only ever re-download days already
held, and one lagging symbol (index churn, a renamed ticker) would drag the loop back over
thousands of day-files on every run. This also preserves the "global-max-date incremental"
design that `schema.py` and `data_schema.md` already document as the reason
`fails_to_deliver` is a separate table.

This report covers that change plus the earlier price/dividend refactor in the same working
tree (both are uncommitted against the same baseline).

## 2. Gates

| Gate | Check | Verdict | Detail |
|---|---|---|---|
| G1 | targeted tests green (6 file(s)) | **PASS** | 25 passed in 14.72s |
| G2 | store boundary test green | **PASS** | 1 data_store file(s) touched; 3 passed in 0.37s |
| G3 | no new `print(` under src/ | **PASS** | none added |
| G4 | public API stable or call sites updated | **FAIL** | src/data_extract/utils/fundamentals/fetch_macro.py::Context (still referenced); src/data_extract/utils/fundamentals/fetch_macro.py::Fred (still referenced); src/data_extract/utils/fundamentals/fetch_macro.py::fetch_macro (still referenced); src/data_extract/utils/fundamentals/fetch_macro.py::fill_short_gaps (still referenced); src/data_extract/utils/fundamentals/fetch_macro.py::os (still referenced); src/data_extract/utils/fundamentals/fetch_macro.py::pd (still referenced); src/data_extract/utils/fundamentals/fetch_macro.py::record_run (still referenced); src/data_extract/utils/prices/fetch_dividends.py::load_existing (still referenced); src/data_extract/utils/prices/fetch_dividends.py::time (still referenced); src/data_extract/utils/prices/fetch_dividends.py::tqdm (still referenced); src/data_extract/utils/prices/fetch_dividends.py::yf (still referenced); src/data_extract/utils/prices/fetch_prices.py::get_sp500_tickers (still referenced); src/data_extract/utils/prices/fetch_prices.py::industry_group (still referenced); src/data_extract/utils/prices/fetch_prices.py::io (still referenced); src/data_extract/utils/prices/fetch_prices.py::requests (still referenced); src/data_extract/utils/prices/fetch_short_interest.py::load_existing (still referenced); tests/data_extract/test_price_interior_gap.py::annotations (still referenced); tests/data_extract/test_price_interior_gap.py::fp (still referenced); tests/data_extract/test_price_interior_gap.py::pd (still referenced); tests/data_extract/test_price_interior_gap.py::test_action_columns_dropped_keep_prices_clean_ohlcv (still referenced) |
| G5 | docs moved with the code | **PASS** | 16 src file(s); docs touched: AGENTS.md, docs/coding_standard.md, docs/data_conventions.md, docs/data_schema.md, docs/data_sources.md |
| G6 | docstring lines did not shrink | **FAIL** | src/data_extract/utils/fundamentals/fetch_macro.py 29->0; src/data_extract/utils/prices/fetch_prices.py 91->63; tests/data_extract/test_price_interior_gap.py 16->0 |
| G7 | AGENTS.md <= 70 lines | **PASS** | 70 lines |

**2 FAIL** — G4, G6. The work is **NOT done**.

## 3. Metrics

_Observations only — no verdicts. LOC is never a target (see [definition_of_done.md](../../docs/definition_of_done.md))._

**Per touched Python file**

| file | status | loc_before | loc_after | code | docstring | comment | public_api |
|---|---|---|---|---|---|---|---|
| src/data_aggregate/utils/common/prices.py | modified | 91 | 91 | 27 | 51 | 0 | 9 |
| src/data_extract/cli.py | modified | 286 | 299 | 201 | 16 | 26 | 58 |
| src/data_extract/step_extract_all_data.py | modified | 54 | 54 | 32 | 12 | 0 | 11 |
| src/data_extract/transformers/step_extract_fundamentals.py | modified | 45 | 46 | 17 | 11 | 8 | 9 |
| src/data_extract/transformers/step_extract_prices.py | modified | 54 | 64 | 26 | 10 | 15 | 12 |
| src/data_extract/utils/common/incremental.py | modified | 40 | 88 | 30 | 50 | 1 | 5 |
| src/data_extract/utils/common/run_manifest.py | modified | 138 | 139 | 76 | 44 | 0 | 13 |
| src/data_extract/utils/fundamentals/fetch_macro.py | deleted | 128 | 0 | 0 | 0 | 0 | 0 |
| src/data_extract/utils/prices/fetch_dividends.py | modified | 104 | 77 | 39 | 25 | 1 | 9 |
| src/data_extract/utils/prices/fetch_macro.py | new | — | 128 | 67 | 29 | 15 | 7 |
| src/data_extract/utils/prices/fetch_macro_assets.py | modified | 172 | 172 | 93 | 49 | 9 | 19 |
| src/data_extract/utils/prices/fetch_prices.py | modified | 478 | 235 | 127 | 63 | 14 | 16 |
| src/data_extract/utils/prices/fetch_short_interest.py | modified | 104 | 109 | 64 | 24 | 1 | 11 |
| src/data_extract/utils/prices/fetch_tickers.py | new | — | 68 | 49 | 7 | 1 | 8 |
| src/data_store/store.py | modified | 604 | 638 | 396 | 137 | 27 | 40 |
| src/utils/universe.py | modified | 39 | 38 | 10 | 24 | 2 | 5 |
| tests/data_aggregate/test_da_capex_and_dividend_consolidation.py | modified | 84 | 84 | 58 | 8 | 6 | 6 |
| tests/data_aggregate/test_dividend_features.py | modified | 174 | 162 | 113 | 12 | 13 | 7 |
| tests/data_extract/test_macro.py | modified | 33 | 33 | 21 | 5 | 2 | 6 |
| tests/data_extract/test_macro_freshness.py | modified | 54 | 54 | 27 | 10 | 4 | 6 |
| tests/data_extract/test_other_tickers_separation.py | modified | 75 | 103 | 62 | 18 | 2 | 11 |
| tests/data_extract/test_price_interior_gap.py | deleted | 97 | 0 | 0 | 0 | 0 | 0 |
| tests/data_extract/test_price_prelisting_trim.py | modified | 115 | 133 | 82 | 28 | 2 | 11 |
| tests/data_extract/test_resume_since.py | new | — | 91 | 52 | 15 | 4 | 9 |
| tests/data_extract/test_sector_neutral_and_extract_refinements.py | modified | 121 | 121 | 80 | 12 | 13 | 7 |
| tests/data_extract/test_short_interest_resume.py | new | — | 82 | 51 | 10 | 3 | 7 |

**Totals**

| files_touched | python_files_touched | loc_after_total | docstring_lines_after_total | docstring_lines_before_total |
|---|---|---|---|---|
| 35 | 26 | 3,109 | 670 | 625 |

**Duplication** (shingle = 6 normalised code lines): 2 of 2,283 (0.1%) recur. Some duplication in this repo is deliberate and documented — read the docstring before removing any.

## 4. Evidence

- baseline: `53e7bebe0e44`
- tests run: tests/data_extract/test_short_interest_resume.py, tests/data_extract/test_resume_since.py, tests/data_extract/test_other_tickers_separation.py, tests/data_aggregate/test_short_interest_features.py, tests/data_extract/test_price_prelisting_trim.py, tests/data_store/test_store_boundary.py
- non-Python files touched (9): AGENTS.md, docs/coding_standard.md, docs/data_conventions.md, docs/data_schema.md, docs/data_sources.md, reports/2026-08-18/refactor-pricing__REFACTOR.md, reports/2026-08-18/refactor-short-interest__REFACTOR.md, specs/2026-08-17_scrapper_extract_fillings.md, specs/2026-08-18_refactor_pricing.md
- pytest_summary: 25 passed in 14.72s
- pytest_targets: ['tests/data_extract/test_short_interest_resume.py', 'tests/data_extract/test_resume_since.py', 'tests/data_extract/test_other_tickers_separation.py', 'tests/data_aggregate/test_short_interest_features.py', 'tests/data_extract/test_price_prelisting_trim.py', 'tests/data_store/test_store_boundary.py']
- g6_note: A shrink is ALLOWED but must be justified in §5 -- say which docstring you removed and why it was not load-bearing.

## 5. Regressions, gaps and deliberate omissions

- **Zero regressions, verified by differential run — not by assertion.** The full suite is
  31 failed / 874 passed / 38 skipped. I captured the failing test IDs, `git stash`-ed every
  change, re-ran the identical selection, and diffed: `comm -23 with_changes baseline` is
  **empty** (30 vs 30, identical sets). Every failure is pre-existing. They cluster as:
  `test_read_equivalence.py` (11 — needs a live Postgres with a built cube; `cube_part_market`
  absent), `test_fundamental_features.py` (10), `test_quality_features.py` (3),
  `test_aggregate_regression.py` (2 errors), plus one each in `test_part_registry.py`,
  `test_new_factor_features.py`, `test_latest_quarter_features.py` — all in `data_aggregate`,
  none in `data_extract`, where 100% of this change lives.
- **Two suites excluded from the run, both broken by the previous commit, not by me.**
  `tests/data_extract/test_freshness.py` and `tests/utils/test_fundamentals_audit.py` fail at
  COLLECTION: commit `53e7beb` ("reshaping repo") deleted `src/data_extract/utils/common/
  freshness.py` + `step_check_freshness.py` and moved `fundamentals_audit` into `src/validate/`,
  leaving both importers dangling. `cli.py` still imports the missing `StepCheckFreshness`, so
  that module will not import until someone restores or removes it — flagged, not fixed
  (out of scope, and deleting a CLI command is the user's call).
- **G4 — all 20 entries are false positives.** The gate flags a removed name whenever the same
  identifier text appears anywhere else in the repo, with no scoping to the definition. Broken
  down: `fetch_macro.py::{Context,Fred,os,pd,record_run,fetch_macro,fill_short_gaps}` and
  `fetch_prices.py::{io,requests,industry_group,get_sp500_tickers}` are files/symbols the USER
  MOVED this session (`utils/fundamentals/fetch_macro.py` → `utils/prices/`, and
  `get_sp500_tickers`/`_dedupe_share_classes` → new `fetch_tickers.py`) — the names still exist,
  at new paths, and I repaired every importer (see next bullet). `fetch_dividends.py::
  {load_existing,time,tqdm,yf}` and `fetch_short_interest.py::load_existing` are ordinary
  stdlib/3rd-party imports still used elsewhere; nothing references the deleted bodies.
  `test_price_interior_gap.py::{annotations,pd,fp}` are boilerplate imports present in every
  test file, and `test_action_columns_dropped_keep_prices_clean_ohlcv` was **relocated** verbatim
  into `test_price_prelisting_trim.py` (it still covers live code, `_ACTION_COLS`) — confirmed
  passing there.
- **Repaired four dangling imports left by concurrent edits**, because they broke the import
  chain my own tests depend on: `fetch_macro_assets.py` and `cli.py` still imported
  `utils.fundamentals.fetch_macro` after the move (this one broke `StepExtractPrices` entirely,
  so *any* test touching the step failed to collect), plus `test_macro.py` and a docstring path
  in `test_macro_freshness.py`. Also completed the `fetch_market_prices` removal the user
  started, and restored `_ACTION_COLS` after it was inlined as a literal while a test imports it.
- **G6 — docstring shrink, justified.** `fetch_prices.py` 91→63: the removed lines documented
  code that no longer exists (the interior-gap/IPO-backfill rationale for the deleted
  `_tickers_needing_download`/`_trading_calendar`/`_interior_gap_start`, and the S&P-scraping
  notes that moved with `get_sp500_tickers` to `fetch_tickers.py`). `fetch_macro.py` 29→0 and
  `test_price_interior_gap.py` 16→0 are whole-file removals — the first is the user's move (its
  29 docstring lines reappear intact in `utils/prices/fetch_macro.py`, visible in §3), the second
  a deletion whose two surviving-relevant tests targeted now-deleted functions. Net docstrings
  across the change are **UP** (625 → 670): `fetch_short_interest.py` gained the global-max
  rationale, `incremental.py` 23 → 50 documents the per-entity-vs-global split.
- **`fetch_short_interest` signature is now stricter**: `tickers` is required (was
  `list[str] | None = None`, where None meant "keep every ticker in the file"). No caller used
  None — both `StepExtractPrices` and the CLI always pass a universe — and an unfiltered RegSHO
  file is ~8,000 symbols of non-universe noise, so the capability was removed rather than kept.
- `fetch_google_trends.py` and `fetch_wiki_pageviews.py` still pass string literals
  (`"google_trends"`, `"wiki_pageviews"`) to `load_existing`, violating the `Tables.<name>`
  hard rule, and still read the whole table to find their frontier. Out of scope here; listed
  in §6 as the obvious next targets for the same treatment.

## 6. Next actions

- **Decide what happens to `StepCheckFreshness`.** `cli.py` imports it and the module was
  deleted in `53e7beb`, so `src.data_extract.cli` cannot be imported at all right now. Either
  restore `freshness.py` or drop the command + its test. This blocks every CLI entry point,
  including the `price_history` / `dividends` / `short_interest` commands touched here.
- Same for `tests/utils/test_fundamentals_audit.py` — repoint it at `src/validate/`.
- Apply this same pass to `fetch_google_trends.py` and `fetch_wiki_pageviews.py`: `Tables.<name>`
  instead of string literals, and `resume_since` instead of `load_existing` + a full table read.
  Once those two are converted, `load_existing` has no callers left and can be deleted.
- Run a live `price_history` / `dividends` / `short_interest` pull against a real DB to confirm
  the new resume windows end-to-end — everything here is unit/sqlite-fixture level, no network
  call was made.
- Fix the pre-existing `intrinsic_cfg=None` `TypeError` in `fundamental_features.py`, which
  accounts for a large share of the 31 standing failures.

```json dod-metrics
{
  "baseline_head_sha": "53e7bebe0e44cdd73e7201ad8c129872ccfe5b4d",
  "content_hash": "sha256:e28653a7ab366f9f763f84809ec2c03ea3530f84bd72f03b976553b89b0d08c9",
  "gates": {
    "G1": "PASS",
    "G2": "PASS",
    "G3": "PASS",
    "G4": "FAIL",
    "G5": "PASS",
    "G6": "FAIL",
    "G7": "PASS"
  },
  "generator": "scripts/dod/refactor_metrics.py@1",
  "metrics": {
    "duplication": {
      "duplicate_ratio": 0.0008760402978537013,
      "duplicated_shingles": 2,
      "shingles": 2283,
      "top_sites": [
        {
          "at": [
            "src/data_extract/utils/prices/fetch_dividends.py:38",
            "src/data_extract/utils/prices/fetch_prices.py:158"
          ],
          "count": 2
        }
      ],
      "window_lines": 6
    },
    "per_file": {
      "src/data_aggregate/utils/common/prices.py": {
        "code": 27,
        "comment": 0,
        "docstring_after": 51,
        "docstring_before": 51,
        "loc_after": 91,
        "loc_before": 91,
        "public_api_count": 9,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/cli.py": {
        "code": 201,
        "comment": 26,
        "docstring_after": 16,
        "docstring_before": 14,
        "loc_after": 299,
        "loc_before": 286,
        "public_api_count": 58,
        "public_api_removed": [
          "fetch_market_prices"
        ],
        "status": "modified"
      },
      "src/data_extract/step_extract_all_data.py": {
        "code": 32,
        "comment": 0,
        "docstring_after": 12,
        "docstring_before": 12,
        "loc_after": 54,
        "loc_before": 54,
        "public_api_count": 11,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/transformers/step_extract_fundamentals.py": {
        "code": 17,
        "comment": 8,
        "docstring_after": 11,
        "docstring_before": 11,
        "loc_after": 46,
        "loc_before": 45,
        "public_api_count": 9,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/transformers/step_extract_prices.py": {
        "code": 26,
        "comment": 15,
        "docstring_after": 10,
        "docstring_before": 9,
        "loc_after": 64,
        "loc_before": 54,
        "public_api_count": 12,
        "public_api_removed": [
          "fetch_market_prices"
        ],
        "status": "modified"
      },
      "src/data_extract/utils/common/incremental.py": {
        "code": 30,
        "comment": 1,
        "docstring_after": 50,
        "docstring_before": 23,
        "loc_after": 88,
        "loc_before": 40,
        "public_api_count": 5,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/common/run_manifest.py": {
        "code": 76,
        "comment": 0,
        "docstring_after": 44,
        "docstring_before": 44,
        "loc_after": 139,
        "loc_before": 138,
        "public_api_count": 13,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/fundamentals/fetch_macro.py": {
        "code": 0,
        "comment": 0,
        "docstring_after": 0,
        "docstring_before": 29,
        "loc_after": 0,
        "loc_before": 128,
        "public_api_count": 0,
        "public_api_removed": [
          "Context",
          "Fred",
          "fetch_macro",
          "fill_short_gaps",
          "os",
          "pd",
          "record_run"
        ],
        "status": "deleted"
      },
      "src/data_extract/utils/prices/fetch_dividends.py": {
        "code": 39,
        "comment": 1,
        "docstring_after": 25,
        "docstring_before": 24,
        "loc_after": 77,
        "loc_before": 104,
        "public_api_count": 9,
        "public_api_removed": [
          "load_existing",
          "time",
          "tqdm",
          "yf"
        ],
        "status": "modified"
      },
      "src/data_extract/utils/prices/fetch_macro.py": {
        "code": 67,
        "comment": 15,
        "docstring_after": 29,
        "docstring_before": null,
        "loc_after": 128,
        "loc_before": null,
        "public_api_count": 7,
        "public_api_removed": null,
        "status": "new"
      },
      "src/data_extract/utils/prices/fetch_macro_assets.py": {
        "code": 93,
        "comment": 9,
        "docstring_after": 49,
        "docstring_before": 49,
        "loc_after": 172,
        "loc_before": 172,
        "public_api_count": 19,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/prices/fetch_prices.py": {
        "code": 127,
        "comment": 14,
        "docstring_after": 63,
        "docstring_before": 91,
        "loc_after": 235,
        "loc_before": 478,
        "public_api_count": 16,
        "public_api_removed": [
          "fetch_market_prices",
          "get_sp500_tickers",
          "industry_group",
          "io",
          "requests"
        ],
        "status": "modified"
      },
      "src/data_extract/utils/prices/fetch_short_interest.py": {
        "code": 64,
        "comment": 1,
        "docstring_after": 24,
        "docstring_before": 18,
        "loc_after": 109,
        "loc_before": 104,
        "public_api_count": 11,
        "public_api_removed": [
          "load_existing"
        ],
        "status": "modified"
      },
      "src/data_extract/utils/prices/fetch_tickers.py": {
        "code": 49,
        "comment": 1,
        "docstring_after": 7,
        "docstring_before": null,
        "loc_after": 68,
        "loc_before": null,
        "public_api_count": 8,
        "public_api_removed": null,
        "status": "new"
      },
      "src/data_store/store.py": {
        "code": 396,
        "comment": 27,
        "docstring_after": 137,
        "docstring_before": 128,
        "loc_after": 638,
        "loc_before": 604,
        "public_api_count": 40,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/utils/universe.py": {
        "code": 10,
        "comment": 2,
        "docstring_after": 24,
        "docstring_before": 24,
        "loc_after": 38,
        "loc_before": 39,
        "public_api_count": 5,
        "public_api_removed": null,
        "status": "modified"
      },
      "tests/data_aggregate/test_da_capex_and_dividend_consolidation.py": {
        "code": 58,
        "comment": 6,
        "docstring_after": 8,
        "docstring_before": 8,
        "loc_after": 84,
        "loc_before": 84,
        "public_api_count": 6,
        "public_api_removed": null,
        "status": "modified"
      },
      "tests/data_aggregate/test_dividend_features.py": {
        "code": 113,
        "comment": 13,
        "docstring_after": 12,
        "docstring_before": 11,
        "loc_after": 162,
        "loc_before": 174,
        "public_api_count": 7,
        "public_api_removed": [
          "test_series_to_long_parser"
        ],
        "status": "modified"
      },
      "tests/data_extract/test_macro.py": {
        "code": 21,
        "comment": 2,
        "docstring_after": 5,
        "docstring_before": 5,
        "loc_after": 33,
        "loc_before": 33,
        "public_api_count": 6,
        "public_api_removed": null,
        "status": "modified"
      },
      "tests/data_extract/test_macro_freshness.py": {
        "code": 27,
        "comment": 4,
        "docstring_after": 10,
        "docstring_before": 10,
        "loc_after": 54,
        "loc_before": 54,
        "public_api_count": 6,
        "public_api_removed": null,
        "status": "modified"
      },
      "tests/data_extract/test_other_tickers_separation.py": {
        "code": 62,
        "comment": 2,
        "docstring_after": 18,
        "docstring_before": 12,
        "loc_after": 103,
        "loc_before": 75,
        "public_api_count": 11,
        "public_api_removed": [
          "test_fetch_market_prices_ohlcv_only_for_other_tickers"
        ],
        "status": "modified"
      },
      "tests/data_extract/test_price_interior_gap.py": {
        "code": 0,
        "comment": 0,
        "docstring_after": 0,
        "docstring_before": 16,
        "loc_after": 0,
        "loc_before": 97,
        "public_api_count": 0,
        "public_api_removed": [
          "annotations",
          "fp",
          "pd",
          "test_action_columns_dropped_keep_prices_clean_ohlcv",
          "test_download_plan_widens_to_cover_the_gap",
          "test_interior_spy_gap_is_scheduled_for_refetch"
        ],
        "status": "deleted"
      },
      "tests/data_extract/test_price_prelisting_trim.py": {
        "code": 82,
        "comment": 2,
        "docstring_after": 28,
        "docstring_before": 24,
        "loc_after": 133,
        "loc_before": 115,
        "public_api_count": 11,
        "public_api_removed": null,
        "status": "modified"
      },
      "tests/data_extract/test_resume_since.py": {
        "code": 52,
        "comment": 4,
        "docstring_after": 15,
        "docstring_before": null,
        "loc_after": 91,
        "loc_before": null,
        "public_api_count": 9,
        "public_api_removed": null,
        "status": "new"
      },
      "tests/data_extract/test_sector_neutral_and_extract_refinements.py": {
        "code": 80,
        "comment": 13,
        "docstring_after": 12,
        "docstring_before": 12,
        "loc_after": 121,
        "loc_before": 121,
        "public_api_count": 7,
        "public_api_removed": null,
        "status": "modified"
      },
      "tests/data_extract/test_short_interest_resume.py": {
        "code": 51,
        "comment": 3,
        "docstring_after": 10,
        "docstring_before": null,
        "loc_after": 82,
        "loc_before": null,
        "public_api_count": 7,
        "public_api_removed": null,
        "status": "new"
      }
    },
    "totals": {
      "docstring_lines_after_total": 670,
      "docstring_lines_before_total": 625,
      "files_touched": 35,
      "loc_after_total": 3109,
      "python_files_touched": 26
    }
  },
  "scope": {
    "baseline_sha": "53e7bebe0e44cdd73e7201ad8c129872ccfe5b4d",
    "tests": [
      "tests/data_extract/test_short_interest_resume.py",
      "tests/data_extract/test_resume_since.py",
      "tests/data_extract/test_other_tickers_separation.py",
      "tests/data_aggregate/test_short_interest_features.py",
      "tests/data_extract/test_price_prelisting_trim.py",
      "tests/data_store/test_store_boundary.py"
    ],
    "touched": [
      "AGENTS.md",
      "docs/coding_standard.md",
      "docs/data_conventions.md",
      "docs/data_schema.md",
      "docs/data_sources.md",
      "reports/2026-08-18/refactor-pricing__REFACTOR.md",
      "reports/2026-08-18/refactor-short-interest__REFACTOR.md",
      "specs/2026-08-17_scrapper_extract_fillings.md",
      "specs/2026-08-18_refactor_pricing.md",
      "src/data_aggregate/utils/common/prices.py",
      "src/data_extract/cli.py",
      "src/data_extract/step_extract_all_data.py",
      "src/data_extract/transformers/step_extract_fundamentals.py",
      "src/data_extract/transformers/step_extract_prices.py",
      "src/data_extract/utils/common/incremental.py",
      "src/data_extract/utils/common/run_manifest.py",
      "src/data_extract/utils/fundamentals/fetch_macro.py",
      "src/data_extract/utils/prices/fetch_dividends.py",
      "src/data_extract/utils/prices/fetch_macro.py",
      "src/data_extract/utils/prices/fetch_macro_assets.py",
      "src/data_extract/utils/prices/fetch_prices.py",
      "src/data_extract/utils/prices/fetch_short_interest.py",
      "src/data_extract/utils/prices/fetch_tickers.py",
      "src/data_store/store.py",
      "src/utils/universe.py",
      "tests/data_aggregate/test_da_capex_and_dividend_consolidation.py",
      "tests/data_aggregate/test_dividend_features.py",
      "tests/data_extract/test_macro.py",
      "tests/data_extract/test_macro_freshness.py",
      "tests/data_extract/test_other_tickers_separation.py",
      "tests/data_extract/test_price_interior_gap.py",
      "tests/data_extract/test_price_prelisting_trim.py",
      "tests/data_extract/test_resume_since.py",
      "tests/data_extract/test_sector_neutral_and_extract_refinements.py",
      "tests/data_extract/test_short_interest_resume.py"
    ]
  },
  "session_id": "a4ba74e3-2fdf-406a-adfa-f2073d9b0fcf",
  "type": "REFACTOR"
}
```

