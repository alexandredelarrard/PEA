# Harness run — earnings-call-extraction-gaps

| Field | Value |
|---|---|
| Slug | `earnings-call-extraction-gaps` |
| Run dir | `reports/validate/2026-10-01-earnings-call-extraction-gaps/` |
| Repo root | `C:/Users/de larrard alexandre/OneDrive - The Boston Consulting Group, Inc/Documents/repos_github/PEA` |
| Base | `dev` @ `71a9ecd` (dirty, unrelated: 5 files under `src/data_extract/utils/{institutionals,schemas,structure}`) |
| Orchestrator branch | `harness/earnings-call-extraction-gaps` (to create at implementation) |
| Worktree | `reports/worktrees/earnings-call-extraction-gaps` (repo convention) |
| In-flight harness branches | `edgar-extract-refactor`, `modelling-transformers-refactor`, `13f-managers-cusip`: no overlap with the write scope (verified by `git diff --stat` against the base) |
| Long-lived stash | `stash@{0}` belongs to the user — never pop |

## Approved effects

None yet. The plan lists the effects requested at its approval gate.

## Constraints

- Follow the AGENTS.md hard rules.
- Ask before editing: constants, data store/DDL, configs, `data/`, the PG volume, the fingerprint baseline.
- Never kill `python.exe` by image name; stop processes by PID only.

## Stages

| Stage | File | Status |
|---|---|---|
| Research | [01-research](01-research.md) | ✅ incl. defeatbeta follow-up |
| Spec | [01-spec](01-spec.md) | v2 DRAFT — OD-1/OD-2 open |
| Plan | [02-plan](02-plan.md) | DRAFT — awaiting approval |
| Implement | 03-implementation.md | ⬜ |
| Validate | [04-validation-01](04-validation-01.md), [defects](defects.md), [05-review-01](05-review-01.md), [05-simplification-plan](05-simplification-plan.md) | ✅ P9 2026-10-02/03 — verdict **NOT READY** (1 high F-001, 6 medium incl. F-016) |
| Final | [06-final](06-final.md), `report.html` | ✅ report integrity PASS (`_scripts/p9_report_check.py` → `_out/p9-report-check.txt`, exit 0: 10 ACs, 45 links, 1 high finding dispositioned) |

## Decisions log

- **2026-10-01**: user chose defeatbeta as the sole source, accepted the licence for personal research, put the split and features in aggregation, asked for faster extraction, and asked to retire ROIC/Fool/kurry (spec U1–U6).
- **2026-10-02**: user decisions:
  - **U7** — re-scheme `earnings_call_sections` as the raw table, keeping the same table count;
  - **U8** — one-day lag only: `as_of` = call date, and the existing `side="right"` rule gives the next session;
  - **U9** — delete the Wikipedia pageviews fetcher and its dependencies;
  - **U10** — defer new features;
  - execution as one sub-agent per phase, stopped after hand-back, with `rtk` throughout.

  Spec v3 and plan v2 were written.

## Iteration ledger

Empty.

## Finding ledger

Empty.

## Report index

- `_out/`:
  - `coverage_by_quarter.csv`
  - `ec_coverage_by_quarter.png`
  - `split_prototype_sample.csv`
- `_scripts/`:
  - `coverage.py`
  - `fiscal_check.py`
  - `roic_probe.py`
  - `defeatbeta_*.py`
  - `split_prototype.py`
- `_cache/`:
  - ROIC payloads
  - defeatbeta index and sample
  - DB extracts

## Baseline checks (2026-10-01, base `71a9ecd`, main tree)

Earnings-call test subset: **66 passed, 1 failed**. The failure is pre-existing and unrelated: `test_wiki_incremental.py::test_wiki_incremental_reads_last_date_per_ticker`, `UnknownTableError` because `wiki_pageviews` is missing from `schema.py`.

The subset covered:

- `tests/data_extract/behavioral`
- `tests/data_aggregate/test_earnings_call_*`
- `tests/validate/test_earnings_calls.py`
- `tests/utils/test_text_metrics.py`
- `tests/dags/test_extraction_dag.py`

The full-suite baseline is captured in P0, in the worktree.
