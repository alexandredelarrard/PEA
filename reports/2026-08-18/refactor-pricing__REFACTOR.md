---
type: REFACTOR
session_id: a4ba74e3-2fdf-406a-adfa-f2073d9b0fcf
generated_at: 2026-08-18T16:03:54+00:00
baseline: {head_sha: 53e7bebe0e44cdd73e7201ad8c129872ccfe5b4d}
generator: scripts/dod/refactor_metrics.py@1
---

## 1. Scope

**Files written (18):** `AGENTS.md`, `docs/coding_standard.md`, `docs/data_sources.md`, `specs/2026-08-17_scrapper_extract_fillings.md`, `specs/2026-08-18_refactor_pricing.md`, `src/data_extract/cli.py`, `src/data_extract/step_extract_all_data.py`, `src/data_extract/transformers/step_extract_fundamentals.py`, `src/data_extract/transformers/step_extract_prices.py`, `src/data_extract/utils/common/incremental.py`, `src/data_extract/utils/prices/fetch_dividends.py`, `src/data_extract/utils/prices/fetch_prices.py`, `src/data_store/store.py`, `tests/data_aggregate/test_da_capex_and_dividend_consolidation.py`, `tests/data_aggregate/test_dividend_features.py`, `tests/data_extract/test_other_tickers_separation.py`, `tests/data_extract/test_price_interior_gap.py`, `tests/data_extract/test_price_prelisting_trim.py`

**Sample scope:** whole repository working tree vs `53e7bebe0e44` (a refactor's scope is the diff, not a data sample).

**What was asked:** Refactor `fetch_prices.py::fetch_price_history` per
`specs/2026-08-18_refactor_pricing.md`: take `years_history` as an explicit parameter,
resume from the oldest per-ticker last-extracted date across the whole ticker batch (via
a new generic, reusable helper), re-download every ticker forward from that one shared
date, and simplify the flow. Move the dividend-piggyback logic into a new public
`fetch_dividends()` in `fetch_dividends.py` (utils/prices), and move the empty-frame
guard out of every caller into `DataStore.save` itself (logs a warning, no-ops).
User-confirmed during planning: fully drop the old per-ticker interior-gap self-heal
(`_trading_calendar`/`_interior_gap_start`) rather than layering it on top of the new
single-window design — an accepted simplification, not an oversight.

## 2. Gates

| Gate | Check | Verdict | Detail |
|---|---|---|---|
| G1 | targeted tests green (6 file(s)) | **FAIL** | 1 failed, 26 passed in 9.49s |
| G2 | store boundary test green | **PASS** | 1 data_store file(s) touched; 3 passed in 0.58s |
| G3 | no new `print(` under src/ | **PASS** | none added |
| G4 | public API stable or call sites updated | **FAIL** | src/data_extract/utils/prices/fetch_dividends.py::load_existing (still referenced); src/data_extract/utils/prices/fetch_dividends.py::time (still referenced); src/data_extract/utils/prices/fetch_dividends.py::tqdm (still referenced); src/data_extract/utils/prices/fetch_dividends.py::yf (still referenced); tests/data_extract/test_price_interior_gap.py::annotations (still referenced); tests/data_extract/test_price_interior_gap.py::fp (still referenced); tests/data_extract/test_price_interior_gap.py::pd (still referenced); tests/data_extract/test_price_interior_gap.py::test_action_columns_dropped_keep_prices_clean_ohlcv (still referenced) |
| G5 | docs moved with the code | **PASS** | 8 src file(s); docs touched: AGENTS.md, docs/coding_standard.md, docs/data_sources.md |
| G6 | docstring lines did not shrink | **FAIL** | src/data_extract/utils/prices/fetch_dividends.py 24->18; src/data_extract/utils/prices/fetch_prices.py 91->72; tests/data_extract/test_price_interior_gap.py 16->0 |
| G7 | AGENTS.md <= 70 lines | **FAIL** | 71 lines |

**4 FAIL** — G1, G4, G6, G7. The work is **NOT done**.

## 3. Metrics

_Observations only — no verdicts. LOC is never a target (see [definition_of_done.md](../../docs/definition_of_done.md))._

**Per touched Python file**

| file | status | loc_before | loc_after | code | docstring | comment | public_api |
|---|---|---|---|---|---|---|---|
| src/data_extract/cli.py | modified | 286 | 287 | 193 | 14 | 26 | 57 |
| src/data_extract/step_extract_all_data.py | modified | 54 | 54 | 32 | 12 | 0 | 11 |
| src/data_extract/transformers/step_extract_fundamentals.py | modified | 45 | 46 | 17 | 11 | 8 | 9 |
| src/data_extract/transformers/step_extract_prices.py | modified | 54 | 56 | 23 | 9 | 13 | 12 |
| src/data_extract/utils/common/incremental.py | modified | 40 | 69 | 26 | 36 | 0 | 5 |
| src/data_extract/utils/prices/fetch_dividends.py | modified | 104 | 45 | 19 | 18 | 0 | 6 |
| src/data_extract/utils/prices/fetch_prices.py | modified | 478 | 346 | 207 | 72 | 19 | 22 |
| src/data_store/store.py | modified | 604 | 605 | 374 | 128 | 26 | 40 |
| tests/data_aggregate/test_da_capex_and_dividend_consolidation.py | modified | 84 | 84 | 58 | 8 | 6 | 6 |
| tests/data_aggregate/test_dividend_features.py | modified | 174 | 162 | 113 | 12 | 13 | 7 |
| tests/data_extract/test_other_tickers_separation.py | modified | 75 | 77 | 48 | 12 | 2 | 9 |
| tests/data_extract/test_price_interior_gap.py | deleted | 97 | 0 | 0 | 0 | 0 | 0 |
| tests/data_extract/test_price_prelisting_trim.py | modified | 115 | 133 | 82 | 28 | 2 | 11 |

**Totals**

| files_touched | python_files_touched | loc_after_total | docstring_lines_after_total | docstring_lines_before_total |
|---|---|---|---|---|
| 18 | 13 | 1,964 | 360 | 383 |

**Duplication** (shingle = 6 normalised code lines): 0 of 1,457 (0.0%) recur. Some duplication in this repo is deliberate and documented — read the docstring before removing any.

## 4. Evidence

- baseline: `53e7bebe0e44`
- tests run: tests/data_extract/test_price_prelisting_trim.py, tests/data_extract/test_other_tickers_separation.py, tests/data_aggregate/test_da_capex_and_dividend_consolidation.py, tests/data_aggregate/test_dividend_features.py, tests/data_store/test_store_boundary.py, tests/data_store/test_store_where.py
- non-Python files touched (5): AGENTS.md, docs/coding_standard.md, docs/data_sources.md, specs/2026-08-17_scrapper_extract_fillings.md, specs/2026-08-18_refactor_pricing.md
- pytest_summary: 1 failed, 26 passed in 9.49s
- pytest_targets: ['tests/data_extract/test_price_prelisting_trim.py', 'tests/data_extract/test_other_tickers_separation.py', 'tests/data_aggregate/test_da_capex_and_dividend_consolidation.py', 'tests/data_aggregate/test_dividend_features.py', 'tests/data_store/test_store_boundary.py', 'tests/data_store/test_store_where.py']
- g6_note: A shrink is ALLOWED but must be justified in §5 -- say which docstring you removed and why it was not load-bearing.

## 5. Regressions, gaps and deliberate omissions

- **G1 (1 failed)**: the single failure is `test_da_capex_and_dividend_consolidation.py
  ::test_da_vs_capex_feature`, a pre-existing `TypeError` in
  `fundamental_features.py::_intrinsic_fields` (`intrinsic_cfg=None` unpacked with `**`)
  — a file this refactor never touches. Verified by `git stash`-ing all my changes and
  re-running: identical failure. All 18 tests actually exercising the refactored code
  (dividends-from-price-frame extraction, market-price OHLCV-only/years_history
  plumbing, prelisting trim + action-column drop, sector-neutral/dedup, dividend
  features) pass, including the moved/updated ones.
- **G4 (false positives, all 8)**: the generator flags a deleted name as "still
  referenced" whenever the same identifier text appears anywhere else in the repo, with
  no scoping to the deleted definition. `fetch_dividends.py::load_existing/time/tqdm/yf`
  are just common imports (stdlib/`yfinance`/`tqdm`, and the shared `load_existing`
  helper) still legitimately used elsewhere — nothing calls the *deleted* old
  `fetch_dividends()`/`_ticker_dividends`/`_series_to_long`. Likewise
  `test_price_interior_gap.py::annotations/fp/pd` are just `from __future__`/module
  imports that appear in every test file. `test_action_columns_dropped_keep_prices_
  clean_ohlcv` was deliberately *relocated* (not orphaned) into
  `test_price_prelisting_trim.py`, verbatim, since it still covers live code
  (`_ACTION_COLS`) — confirmed passing there.
- **G6 (docstring shrink, justified)**: `fetch_dividends.py` (24->18) and
  `fetch_prices.py` (91->72) shrank because the deleted code they described is gone:
  the old per-ticker `yf.Ticker().dividends` approach (no piggyback, its own
  incremental-window docstring) and the interior-gap/IPO-backfill design rationale for
  `_tickers_needing_download`/`_trading_calendar`/`_interior_gap_start` — all three
  functions removed per the user's explicit, confirmed decision (see plan) to drop
  gap-healing in favor of the simpler single-window design. `test_price_interior_gap.py`
  (16->0) is a full-file deletion: two of its three tests targeted those now-deleted
  functions; the third (still-valid) test was moved, not lost.
- **G7 (AGENTS.md 71 lines, pre-existing/out of scope)**: this refactor never edited
  `AGENTS.md`. The overage is one line ("Always use `rtk` in all your bash, grep, git
  scripts...") that was already present in the working tree before this task's edits
  began (confirmed via `git diff`/`git stash`) — most likely inserted by rtk's own
  tooling hook in response to an unrelated mid-session instruction to use `rtk` for bash
  commands. Not touched, and trimming it is outside this task's authorized scope
  (`AGENTS.md` is a risk zone — propose-and-approve, not silently edit).
- Dropped interior-gap self-healing means a rare partial-download hole in the middle of
  a ticker's history (e.g. a botched past run) no longer self-heals automatically; it
  would need a manual/periodic full rebuild to catch. User-confirmed tradeoff, not an
  oversight.
- Deleted the old, unused `fetch_dividends()` (per-ticker `yf.Ticker(...).dividends`,
  no production caller) and its `_series_to_long`/`_ticker_dividends` helpers wholesale
  to make room for the spec's request to name the new piggyback function
  `fetch_dividends()` in that same file — flagged during planning, not silent.
- `tests/data_store/test_read_equivalence.py` (11 failures) needs a live Postgres DB
  with a built cube (`cube_part_market` missing) — pre-existing environment gap,
  confirmed via `git stash` to fail identically without my changes; not in the G1
  `--tests` scope for that reason.

## 6. Next actions

- Run a live `price_history` / `market_prices` CLI pull against a real (or staging) DB
  to confirm the new single-window `resume_since` behavior end-to-end (this session
  only ran unit/sqlite-fixture tests, per `docs/testing.md` conventions — no network
  call was made).
- If the `AGENTS.md` rtk-line overage (G7) needs resolving, that's a separate,
  unrelated cleanup — propose it to the user rather than folding it into this diff.
- Consider whether `resume_since` should be adopted by the other fetchers that
  currently duplicate the per-ticker `groupby(...)[date_col].max()` idiom inline
  (`fetch_wiki_pageviews.py`, `fetch_earnings_surprises.py`, `fetch_filing_text.py`) —
  out of scope here since the spec asked only for `fetch_price_history`, but it's the
  "reusable afterward" the spec anticipated.
- Fix the pre-existing `test_da_vs_capex_feature` / `intrinsic_cfg=None` bug in
  `fundamental_features.py` separately — unrelated to this refactor.

```json dod-metrics
{
  "baseline_head_sha": "53e7bebe0e44cdd73e7201ad8c129872ccfe5b4d",
  "content_hash": "sha256:c3cd900b1724668fa83be70e016c68d63aa8b9a7038bb61785fdfac6e2bbef27",
  "gates": {
    "G1": "FAIL",
    "G2": "PASS",
    "G3": "PASS",
    "G4": "FAIL",
    "G5": "PASS",
    "G6": "FAIL",
    "G7": "FAIL"
  },
  "generator": "scripts/dod/refactor_metrics.py@1",
  "metrics": {
    "duplication": {
      "duplicate_ratio": 0.0,
      "duplicated_shingles": 0,
      "shingles": 1457,
      "top_sites": [],
      "window_lines": 6
    },
    "per_file": {
      "src/data_extract/cli.py": {
        "code": 193,
        "comment": 26,
        "docstring_after": 14,
        "docstring_before": 14,
        "loc_after": 287,
        "loc_before": 286,
        "public_api_count": 57,
        "public_api_removed": null,
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
        "code": 23,
        "comment": 13,
        "docstring_after": 9,
        "docstring_before": 9,
        "loc_after": 56,
        "loc_before": 54,
        "public_api_count": 12,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/common/incremental.py": {
        "code": 26,
        "comment": 0,
        "docstring_after": 36,
        "docstring_before": 23,
        "loc_after": 69,
        "loc_before": 40,
        "public_api_count": 5,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/prices/fetch_dividends.py": {
        "code": 19,
        "comment": 0,
        "docstring_after": 18,
        "docstring_before": 24,
        "loc_after": 45,
        "loc_before": 104,
        "public_api_count": 6,
        "public_api_removed": [
          "load_existing",
          "time",
          "tqdm",
          "yf"
        ],
        "status": "modified"
      },
      "src/data_extract/utils/prices/fetch_prices.py": {
        "code": 207,
        "comment": 19,
        "docstring_after": 72,
        "docstring_before": 91,
        "loc_after": 346,
        "loc_before": 478,
        "public_api_count": 22,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_store/store.py": {
        "code": 374,
        "comment": 26,
        "docstring_after": 128,
        "docstring_before": 128,
        "loc_after": 605,
        "loc_before": 604,
        "public_api_count": 40,
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
      "tests/data_extract/test_other_tickers_separation.py": {
        "code": 48,
        "comment": 2,
        "docstring_after": 12,
        "docstring_before": 12,
        "loc_after": 77,
        "loc_before": 75,
        "public_api_count": 9,
        "public_api_removed": null,
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
      }
    },
    "totals": {
      "docstring_lines_after_total": 360,
      "docstring_lines_before_total": 383,
      "files_touched": 18,
      "loc_after_total": 1964,
      "python_files_touched": 13
    }
  },
  "scope": {
    "baseline_sha": "53e7bebe0e44cdd73e7201ad8c129872ccfe5b4d",
    "tests": [
      "tests/data_extract/test_price_prelisting_trim.py",
      "tests/data_extract/test_other_tickers_separation.py",
      "tests/data_aggregate/test_da_capex_and_dividend_consolidation.py",
      "tests/data_aggregate/test_dividend_features.py",
      "tests/data_store/test_store_boundary.py",
      "tests/data_store/test_store_where.py"
    ],
    "touched": [
      "AGENTS.md",
      "docs/coding_standard.md",
      "docs/data_sources.md",
      "specs/2026-08-17_scrapper_extract_fillings.md",
      "specs/2026-08-18_refactor_pricing.md",
      "src/data_extract/cli.py",
      "src/data_extract/step_extract_all_data.py",
      "src/data_extract/transformers/step_extract_fundamentals.py",
      "src/data_extract/transformers/step_extract_prices.py",
      "src/data_extract/utils/common/incremental.py",
      "src/data_extract/utils/prices/fetch_dividends.py",
      "src/data_extract/utils/prices/fetch_prices.py",
      "src/data_store/store.py",
      "tests/data_aggregate/test_da_capex_and_dividend_consolidation.py",
      "tests/data_aggregate/test_dividend_features.py",
      "tests/data_extract/test_other_tickers_separation.py",
      "tests/data_extract/test_price_interior_gap.py",
      "tests/data_extract/test_price_prelisting_trim.py"
    ]
  },
  "session_id": "a4ba74e3-2fdf-406a-adfa-f2073d9b0fcf",
  "type": "REFACTOR"
}
```

