---
type: REFACTOR
session_id: e78468c8-5c9e-4dff-b353-1c636a571958
generated_at: 2026-08-28T19:43:00+00:00
baseline: {head_sha: 582ffb6ed435230d4b9a4746ae760bb7bab45e61}
generator: scripts/dod/refactor_metrics.py@1
---

## 1. Scope

**Files written (11):** `docs/data_schema.md`, `reports/2026-08-28/phase-3-fundamentals-instant-lookup__REFACTOR.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/deferred.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-3-efficiency-memoisation.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/restatement-census.md`, `src/data_extract/utils/common/edgar_driver.py`, `src/data_extract/utils/common/sec_utils.py`, `src/data_extract/utils/fundamentals/build_history.py`, `src/data_extract/utils/fundamentals/fetch_fundamentals_sec.py`, `src/data_extract/utils/fundamentals/periods.py`, `tests/data_extract/fundamentals/test_instant_lookup.py`

**Sample scope:** whole repository working tree vs `582ffb6ed435` (a refactor's scope is the diff, not a data sample).

**What was asked:** Phase 3 of the fundamentals refactor
([plan](../planning/active-tasks/2026-08-27-refactor-fundamentals/phase-3-efficiency-memoisation.md)):
make the point-in-time replay faster **without changing a single stored number**, by (§3.2)
memoising the period engine per field, (§3.3) vectorising the as-of instant read, (§3.4) fixing
four reads, and (§3.5) measuring a restatement census to decide whether a later vintage
redesign is worth its risk. The plan pre-registered that the memo should be **dropped** if it
measured under a 20 % hit rate and was not otherwise free.

**What was delivered, against that:**

| § | asked | delivered |
|---|---|---|
| 3.2 | per-field memo | built, proven cell-exact by 4 tests, measured **0.3–15 %** hit rate and ±6 % CPU, **DROPPED** (user-confirmed) |
| 3.3 | vectorise `carry_latest_known` | `periods.InstantLookup`, **15.4x** on the primitive (42.58 s -> 2.77 s CPU, MCD's 1,794 real lookups) |
| 3.4 | 4 read fixes | all 4; two are contract fixes not byte savings, and the employees read is documented rather than filtered — see §5 |
| 3.5 | restatement census | [restatement-census.md](../planning/active-tasks/2026-08-27-refactor-fundamentals/restatement-census.md) — recommendation **LEAVE IT** |

**Equality gates (the phase's real acceptance, beyond G1).** The Phase 0 harness, cell-exact
including dtypes, NaN placement and reason-code sets, against Phase 2's committed output:

| gate | scope | result |
|---|---|---|
| tier A | 8 tickers x 16 filings | **0 differing cells, 0 codes** |
| tier B | 8 tickers x full history, 517 events | **0 differing cells, 0 codes** |
| db mode, moving-target guard | all 8 vs live `fundamentals_facts` | `moved=[]` |
| db mode, live-read vs parquet | VRT, MCD | **0 cells, no dtype drift** |
| full suite | `tests/data_extract/fundamentals` | **232 passed, 0 failed** (37m 09s) |
| control | `582ffb6` replayed by the same harness | **0 differing cells** |

## 2. Gates

| Gate | Check | Verdict | Detail |
|---|---|---|---|
| G1 | targeted tests green (1 file(s)) | **PASS** | 1 passed in 7.70s |
| G2 | store boundary test green | **N/A** | no `src/data_store/` file touched |
| G3 | no new `print(` under src/ | **PASS** | none added |
| G4 | public API stable or call sites updated | **PASS** | no public name removed |
| G5 | docs moved with the code | **PASS** | 5 src file(s); docs touched: docs/data_schema.md |
| G6 | docstring lines did not shrink | **PASS** | no touched file lost docstring lines |
| G7 | AGENTS.md <= 70 lines | **PASS** | 70 lines |

**All gates pass** (N/A gates are stated above, not skipped).

## 3. Metrics

_Observations only — no verdicts. LOC is never a target (see [definition_of_done.md](../../docs/definition_of_done.md))._

**Per touched Python file**

| file | status | loc_before | loc_after | code | docstring | comment | public_api |
|---|---|---|---|---|---|---|---|
| src/data_extract/utils/common/edgar_driver.py | modified | 163 | 169 | 92 | 29 | 29 | 18 |
| src/data_extract/utils/common/sec_utils.py | modified | 140 | 152 | 70 | 47 | 12 | 18 |
| src/data_extract/utils/fundamentals/build_history.py | modified | 1,132 | 1,147 | 534 | 316 | 192 | 27 |
| src/data_extract/utils/fundamentals/fetch_fundamentals_sec.py | modified | 973 | 980 | 471 | 336 | 111 | 48 |
| src/data_extract/utils/fundamentals/periods.py | modified | 1,040 | 1,111 | 469 | 404 | 143 | 21 |
| tests/data_extract/fundamentals/test_instant_lookup.py | new | — | 90 | 57 | 19 | 3 | 8 |

**Totals**

| files_touched | python_files_touched | loc_after_total | docstring_lines_after_total | docstring_lines_before_total |
|---|---|---|---|---|
| 11 | 6 | 3,649 | 1,151 | 1,093 |

**Duplication** (shingle = 6 normalised code lines): 0 of 2,660 (0.0%) recur. Some duplication in this repo is deliberate and documented — read the docstring before removing any.

## 4. Evidence

- baseline: `582ffb6ed435`
- tests run: tests/data_extract/fundamentals/test_instant_lookup.py
- non-Python files touched (5): docs/data_schema.md, reports/2026-08-28/phase-3-fundamentals-instant-lookup__REFACTOR.md, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/deferred.md, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-3-efficiency-memoisation.md, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/restatement-census.md
- pytest_summary: 1 passed in 7.70s
- pytest_targets: ['tests/data_extract/fundamentals/test_instant_lookup.py']

## 5. Regressions, gaps and deliberate omissions

- **G1 is narrower than it looks.** It ran the one new test file (1 passed, 8.73 s). The real
  acceptance for this phase is the six-row table in §1 — in particular the full
  `tests/data_extract/fundamentals` suite at **232 passed** and the two replay-equality tiers
  at **0 differing cells**. Those were run against this exact tree, before any doc edits, but
  the generator only knows about `--tests`.
- **A live `main.py` picked up mid-phase code, and I did not intend that.** A pipeline run
  started at 08:52 while `periods.py`/`build_history.py` were part-edited, so it executed
  Phase 3's `build_ticker` against the live DB before the gate had passed. It completed with
  no drift error and the gate has since passed cell-exact, so the output is provably identical
  to Phase 2's — but the sequencing was luck, not control. The plan's own rule ("do not run a
  live `build_fundamentals_history` while a walk is in flight") needs to cover *editing while
  a scheduled run may start*, not just launching one.
- **`fundamentals_history_sec` was already empty when this phase began** — 1 ticker / 68 rows,
  down from the 54 the plan recorded. **Not caused by this work**: the 08:37 run *added* `A`'s
  68 rows, which means it found none stored, so the table was emptied before my first edit.
  The live run rebuilt it to 20 tickers by the end of the session. Consequence for this
  report: the plan's `--source db` check against stored history could not run on the sample,
  and was replaced by a live-read-vs-parquet diff (§1). The real one is post-run.
- **The db-mode check covers 2 of 8 tickers, not 8.** Each ticker needs two full-history
  replays; all eight would have been ~2 h for a dtype question that VRT (shortest) and MCD
  (longest, and `baseline.md`'s DATE canary) settle. Deliberate, and stated in the plan.
- **§3.4's two "projection" fixes save no bytes and I have said so in the code.** Both
  `fundamentals_history_sec` (69 of 69 columns) and `sp500_tickers` (6 of 6, ~500 rows) project
  to the whole table. They are contract fixes. The only real saving in §3.4 is collapsing the
  duplicated `sp500_tickers` read.
- **The `fundamentals_employees` read is still unfiltered, on purpose.** `history_by_ticker`
  seeds the headcount continuity guard from every stored row, so a `where=` on the run's ticker
  list would silently narrow the guard to whichever chunk is being fetched — and the backfill
  is chunked. Documented in the docstring rather than "fixed".
- **One behaviour delta, checked and benign.** The GICS read was `optional=True` (degrading to
  an empty regime map); `load_cik_mapping` raises on a missing universe. Nothing changes,
  because `run_edgar_fetch` called `load_cik_mapping` unconditionally ten lines later — the
  fetch could never have proceeded without the table. The raise just happens sooner.
- **I nearly drew the wrong conclusion from bad measurements.** Sequential wall-clock A/B gave
  three contradictory verdicts across three rounds (memo "costs 26 %", "costs 89 %",
  `InstantLookup` "costs 8 %") — all artefacts of a competing `main.py` and, in round 1, two
  stray REPLs of my own. Only CPU time with interleaved arms gave the stable answer. **Phase 4
  is judged on a speed-up ratio and will hit this exact trap**; the method is written up in
  §3.6 of the plan and repeated in D-15.
- **I lost an hour of census replay to my own bug** — `json.dumps` on a dict with tuple keys,
  after the expensive work was done. Re-run with string keys, per-ticker incremental writes,
  and the serialisation smoke-tested on one ticker first.
- **Docstring lines grew by 58** (1,093 -> 1,151) and `periods.py` grew 1,040 -> 1,111. That is
  against Phase 7's "no file over 600 LOC" target, which now has 71 more lines to remove in
  `periods.py` than it did at `582ffb6`. Named here so it is not a surprise later.
- **The census is 8 tickers, and two of them (BAC, APA) are in the sample *because* they
  restate.** The 8.22 % material-restatement rate is therefore an upper-middle estimate, not a
  universe mean. It would have to fall below ~2 % to flip the recommendation, and the cleanest
  filer in the sample is already at 3.2 %.

## 6. Next actions

1. **Commit.** Nothing is committed yet; the working tree holds the whole phase.
2. **Phase 4 (process pool)** is now the load-bearing change — the census killed the vintage
   redesign, so cross-ticker parallelism is the only remaining 4x. **Measure it in CPU time
   with interleaved arms**, not sequential wall clock (§3.6).
3. **Post-run**: re-run `compare_against_stored` over the 8 sample tickers once the live
   rebuild reaches them (it was at `ADM` at end of session), and re-run the census over the
   whole `fundamentals_facts` table now that the walk has finished.
4. **Phase 5** owns [D-14](../planning/active-tasks/2026-08-27-refactor-fundamentals/deferred.md):
   give `Tables.sp500_tickers` a `read_columns` so `project=True` replaces the hand-written
   column list added here. `schema.py` is a risk zone Phase 3 had no approval for.
5. **[D-15](../planning/active-tasks/2026-08-27-refactor-fundamentals/deferred.md) is unmeasured**
   — `_latest` re-masks the whole frame per field per event. Measure before acting on it.

```json dod-metrics
{
  "baseline_head_sha": "582ffb6ed435230d4b9a4746ae760bb7bab45e61",
  "content_hash": "sha256:ec2e43c078045caab3438f73696eb1da04ffbd08a4138b92557eb91d80bc735d",
  "gates": {
    "G1": "PASS",
    "G2": "N/A",
    "G3": "PASS",
    "G4": "PASS",
    "G5": "PASS",
    "G6": "PASS",
    "G7": "PASS"
  },
  "generator": "scripts/dod/refactor_metrics.py@1",
  "metrics": {
    "duplication": {
      "duplicate_ratio": 0.0,
      "duplicated_shingles": 0,
      "shingles": 2660,
      "top_sites": [],
      "window_lines": 6
    },
    "per_file": {
      "src/data_extract/utils/common/edgar_driver.py": {
        "code": 92,
        "comment": 29,
        "docstring_after": 29,
        "docstring_before": 25,
        "loc_after": 169,
        "loc_before": 163,
        "public_api_count": 18,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/common/sec_utils.py": {
        "code": 70,
        "comment": 12,
        "docstring_after": 47,
        "docstring_before": 44,
        "loc_after": 152,
        "loc_before": 140,
        "public_api_count": 18,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/fundamentals/build_history.py": {
        "code": 534,
        "comment": 192,
        "docstring_after": 316,
        "docstring_before": 307,
        "loc_after": 1147,
        "loc_before": 1132,
        "public_api_count": 27,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/fundamentals/fetch_fundamentals_sec.py": {
        "code": 471,
        "comment": 111,
        "docstring_after": 336,
        "docstring_before": 336,
        "loc_after": 980,
        "loc_before": 973,
        "public_api_count": 48,
        "public_api_removed": null,
        "status": "modified"
      },
      "src/data_extract/utils/fundamentals/periods.py": {
        "code": 469,
        "comment": 143,
        "docstring_after": 404,
        "docstring_before": 381,
        "loc_after": 1111,
        "loc_before": 1040,
        "public_api_count": 21,
        "public_api_removed": null,
        "status": "modified"
      },
      "tests/data_extract/fundamentals/test_instant_lookup.py": {
        "code": 57,
        "comment": 3,
        "docstring_after": 19,
        "docstring_before": null,
        "loc_after": 90,
        "loc_before": null,
        "public_api_count": 8,
        "public_api_removed": null,
        "status": "new"
      }
    },
    "totals": {
      "docstring_lines_after_total": 1151,
      "docstring_lines_before_total": 1093,
      "files_touched": 11,
      "loc_after_total": 3649,
      "python_files_touched": 6
    }
  },
  "scope": {
    "baseline_sha": "582ffb6ed435230d4b9a4746ae760bb7bab45e61",
    "tests": [
      "tests/data_extract/fundamentals/test_instant_lookup.py"
    ],
    "touched": [
      "docs/data_schema.md",
      "reports/2026-08-28/phase-3-fundamentals-instant-lookup__REFACTOR.md",
      "reports/planning/active-tasks/2026-08-27-refactor-fundamentals/deferred.md",
      "reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-3-efficiency-memoisation.md",
      "reports/planning/active-tasks/2026-08-27-refactor-fundamentals/restatement-census.md",
      "src/data_extract/utils/common/edgar_driver.py",
      "src/data_extract/utils/common/sec_utils.py",
      "src/data_extract/utils/fundamentals/build_history.py",
      "src/data_extract/utils/fundamentals/fetch_fundamentals_sec.py",
      "src/data_extract/utils/fundamentals/periods.py",
      "tests/data_extract/fundamentals/test_instant_lookup.py"
    ]
  },
  "session_id": "e78468c8-5c9e-4dff-b353-1c636a571958",
  "type": "REFACTOR"
}
```

