---
type: REFACTOR
session_id: 81563f42-5b71-4ae6-9ddb-b7788f3c75c3
generated_at: 2026-08-27T20:21:51+00:00
baseline: {head_sha: e8740ad2039c37944f05607d3d07dc4b6f1478aa}
generator: scripts/dod/refactor_metrics.py@1
---

## 1. Scope

**Files written (19):** `reports/2026-08-27/fundamentals-refactor-phase0-safety-net__REFACTOR.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/README.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/baseline.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/deferred.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/manifest-snapshot.json`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-0-safety-net.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-1-defects-dead-code.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-2-efficiency-constant-factor.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-3-efficiency-memoisation.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-4-efficiency-parallel-replay.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-5-wiring-context-ledger.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-6-constants-generalization.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-7-structure-prose-tests.md`, `reports/planning/active-tasks/2026-08-27-refactor-fundamentals/post-run-checklist.md`, `reports/research/codebase/2026-08-27-refactor-fundamentals.md`, `specs/2026-08-26/refactor-fundamentals.md`, `tests/data_extract/fundamentals/replay_equality.py`, `tests/data_extract/fundamentals/replay_sample.json`, `tests/data_extract/fundamentals/test_replay_equality.py`

**Sample scope:** whole repository working tree vs `e8740ad2039c` (a refactor's scope is the diff, not a data sample).

**What was asked:** Implement Phase 0 of the fundamentals-replay refactor plan
(`reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-0-safety-net.md`): a
safety-net harness plus baseline measurements for Phases 1-4, touching no `src/` file. Concretely:
(1) `tests/data_extract/fundamentals/replay_equality.py` — freeze/replay/snapshot/compare, a
cell-exact gate over `build_ticker` output, plus a `--source db` mode reusing
`diff_against_stored` for the Postgres DATE-round-trip trap and a moving-target manifest guard;
(2) the 8-ticker frozen sample (`replay_sample.json`, tier-A cap = 16 filings); (3) three
synthetic tests proving the gate actually detects a planted cell change, an all-null dtype
drift, and a dropped reason-code row; (4) baseline measurements (tier A/B wall clock, the
per-ticker E-curve, cProfile, peak RSS) written to `baseline.md`.

## 2. Gates

| Gate | Check | Verdict | Detail |
|---|---|---|---|
| G1 | targeted tests green (1 file(s)) | **PASS** | 3 passed in 16.35s |
| G2 | store boundary test green | **N/A** | no `src/data_store/` file touched |
| G3 | no new `print(` under src/ | **PASS** | none added |
| G4 | public API stable or call sites updated | **PASS** | no public name removed |
| G5 | docs moved with the code | **N/A** | no `src/` file touched |
| G6 | docstring lines did not shrink | **PASS** | no touched file lost docstring lines |
| G7 | AGENTS.md <= 70 lines | **PASS** | 70 lines |

**All gates pass** (N/A gates are stated above, not skipped).

## 3. Metrics

_Observations only — no verdicts. LOC is never a target (see [definition_of_done.md](../../docs/definition_of_done.md))._

**Per touched Python file**

| file | status | loc_before | loc_after | code | docstring | comment | public_api |
|---|---|---|---|---|---|---|---|
| tests/data_extract/fundamentals/replay_equality.py | new | — | 304 | 199 | 43 | 11 | 26 |
| tests/data_extract/fundamentals/test_replay_equality.py | new | — | 124 | 90 | 12 | 0 | 9 |

**Totals**

| files_touched | python_files_touched | loc_after_total | docstring_lines_after_total | docstring_lines_before_total |
|---|---|---|---|---|
| 19 | 2 | 428 | 55 | 0 |

**Duplication** (shingle = 6 normalised code lines): 0 of 331 (0.0%) recur. Some duplication in this repo is deliberate and documented — read the docstring before removing any.

## 4. Evidence

- baseline: `e8740ad2039c`
- tests run: tests/data_extract/fundamentals/test_replay_equality.py
- non-Python files touched (17): reports/2026-08-27/fundamentals-refactor-phase0-safety-net__REFACTOR.md, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/README.md, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/baseline.md, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/deferred.md, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/manifest-snapshot.json, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-0-safety-net.md, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-1-defects-dead-code.md, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-2-efficiency-constant-factor.md, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-3-efficiency-memoisation.md, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-4-efficiency-parallel-replay.md, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-5-wiring-context-ledger.md, reports/planning/active-tasks/2026-08-27-refactor-fundamentals/phase-6-constants-generalization.md
- pytest_summary: 3 passed in 16.35s
- pytest_targets: ['tests/data_extract/fundamentals/test_replay_equality.py']

## 5. Regressions, gaps and deliberate omissions

- **No "quiet machine" re-baseline.** A live multi-hour fetch process (15h+ accumulated CPU
  time, several GB RSS) was on this machine for the entire session and could not be killed
  (standing guidance: never kill `python.exe` by image name without certainty of what it is).
  Every wall-clock number in `baseline.md` is therefore contention-inflated by an unknown
  amount — stated explicitly in the report's header and repeated at each affected number, per
  the plan's own instruction not to promise a speed-up ratio from noisy numbers.
- **Per-ticker E-curve (caps 16/32/48/full) reused rather than re-measured.** The plan document
  already carried this table (VRT 4/8/12/16, MCD 16/32/48/69) from a prior pass under the same
  contention; a second full pass would have cost another ~50 minutes of equally contended time
  for a curve shape one pass already establishes, so `baseline.md` §2 cites it instead of
  re-running it.
- **Peak RSS measured on 3 of 8 tickers** (VRT, MCD, KR — the shortest history and the two
  longest by fact-row count), not all 8. Isolating each ticker in its own subprocess makes this
  ~1 full tier-B pass per ticker measured; the 3 chosen bracket the sample's size range and the
  number was flat across them (116.8-127.3 MB), so the marginal information from the other 5
  is low relative to the cost.
- **`compare_against_stored` (db mode) exercised on 2 of 8 tickers** (VRT, MCD), not the full
  sample, for the same reason (each call rebuilds full history, ~1 tier-B-per-ticker cost). It
  surfaced a genuine, PRE-EXISTING finding — 38 drifted cells for MCD, all
  `stored=NaN -> rebuilt=<value>` at `as_of=2011-11-04` — that is stored `fundamentals_history_sec`
  staleness unrelated to this session's changes (Phase 0 touches no `src/` and no table), not
  fixed here, and worth a follow-up ticket.
- **`freeze_inputs`/`replay`/`snapshot`/`compare` have no unit test of the CLI (`main()`)
  itself** — only the library functions are unit-tested (synthetically) and integration-tested
  (live, against the real DB, in this session's transcript, not as a committed test). The CLI
  is a thin argparse wrapper over those functions; adding a CLI-level test was judged lower
  value than the three planted-defect tests that actually exercise the gate's detection logic.

## 6. Next actions

- Re-run the tier A/B wall-clock and per-ticker E-curve measurements once the machine is
  quiet, before any Phase 1-4 speed-up ratio is quoted against this baseline.
- File a follow-up on the MCD `fundamentals_history_sec` staleness found in §5
  (`compare_against_stored`'s 38 drifted cells) — rebuild MCD's stored history from the
  now-more-complete `fundamentals_facts`.
- Phase 1 can now proceed: it is gated on this harness (`compare` reporting **0** cells
  differing on both tiers, confirmed) rather than on an argument.

```json dod-metrics
{
  "baseline_head_sha": "e8740ad2039c37944f05607d3d07dc4b6f1478aa",
  "content_hash": "sha256:1d203dbe692f76c9ce3fad0498c4dc49e51f0f1fbd89c8ac3ed3585a7eed7fc8",
  "gates": {
    "G1": "PASS",
    "G2": "N/A",
    "G3": "PASS",
    "G4": "PASS",
    "G5": "N/A",
    "G6": "PASS",
    "G7": "PASS"
  },
  "generator": "scripts/dod/refactor_metrics.py@1",
  "metrics": {
    "duplication": {
      "duplicate_ratio": 0.0,
      "duplicated_shingles": 0,
      "shingles": 331,
      "top_sites": [],
      "window_lines": 6
    },
    "per_file": {
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
      "docstring_lines_after_total": 55,
      "docstring_lines_before_total": 0,
      "files_touched": 19,
      "loc_after_total": 428,
      "python_files_touched": 2
    }
  },
  "scope": {
    "baseline_sha": "e8740ad2039c37944f05607d3d07dc4b6f1478aa",
    "tests": [
      "tests/data_extract/fundamentals/test_replay_equality.py"
    ],
    "touched": [
      "reports/2026-08-27/fundamentals-refactor-phase0-safety-net__REFACTOR.md",
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
      "tests/data_extract/fundamentals/replay_equality.py",
      "tests/data_extract/fundamentals/replay_sample.json",
      "tests/data_extract/fundamentals/test_replay_equality.py"
    ]
  },
  "session_id": "81563f42-5b71-4ae6-9ddb-b7788f3c75c3",
  "type": "REFACTOR"
}
```

