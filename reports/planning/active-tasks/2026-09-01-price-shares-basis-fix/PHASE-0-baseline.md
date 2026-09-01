# Phase 0 — Baseline: record the "before" numbers ⬜

**Parent**: [PLAN.md](PLAN.md) · **Depends on**: — · **Blocks**: P1, P2 · **Estimate**: ~1h

## Goal

Turn every measurement in the research document into ONE rerunnable script, and freeze its
output. Without this there is no fix-delta to report at P6, and no way to tell a genuine
improvement from a coincidence.

## Why a script and not a notebook

The same script is rerun at P1, P3, P4 and P6 verification. It is the spine of the whole task.
It lives in `scripts/` (throwaway diagnostics), **not** in `src/` — the permanent version of
invariants 1–3 is the P5 validator.

## Changes

### 1. `scripts/basis_baseline.py` (new)

- [ ] Connect via the psycopg2 creator pattern (the DB password contains `!`, which breaks a
      SQLAlchemy URL string — see the `db-access` memory). Ask for the password; do not hardcode.
- [ ] Emit a single JSON blob to `reports/planning/active-tasks/2026-09-01-price-shares-basis-fix/baseline.json`
      plus a markdown summary to `baseline.md`.

Measurements, each keyed so P6 can diff them:

| key | measurement | expected "before" value |
|---|---|---|
| `mcap_error_by_year` | median / p05 / min of `cube_mcap / sharadar.marketcap`, and rows off by >10%, per year | 1995: 0.208, 2003: 0.419, 2013: 0.722, 2021: 0.910, 2026: 0.997 |
| `error_decomposition` | median `split_part`, median `dividend_part`, median product, median residual, per year | residual must be **1.0000** in every year — if it is not, the decomposition claim is wrong and the plan needs revisiting |
| `split_part_cohorts` | n, mean and median forward-12m return for `split_part` <1 / ==1 / >1 | 10,812 rows @ 27.81% / 37,370 @ 15.38% / 513 @ 9.61% |
| `dividend_part_quintiles` | n, mean `dividend_part`, mean fwd-12m per cross-sectional quintile | Q1 0.606 / 13.64%, Q3 0.779 / 17.20%, Q5 0.990 / 27.62% |
| `deadjusted_rows` | rows with `sharesOutstanding != sharesbas`, split down vs up, and distinct tickers | 10,996 / 51,255 rows (21.5%), 277 of 489 tickers, 601 upward |
| `sec_cover_page_agreement` | on the `fundamentals_history_sec` overlap: rows within ±3%, too high, too low, and the failing ticker list with its ratio | 5,141 agree / 371 high / 41 low; **24 of 96 tickers fail** |
| `spike_revert_scan` | days with `abs(move) > 55%` reversing the next day, per ticker and year | 3 days on MNST in 2026, 1 in 2001, 1 in 1998 |
| `mnst_window` | raw `close` for MNST 2026-07-15 → 2026-08-15 | the 97 / 47 alternation, verbatim |
| `macro_equity_tr_digest` | a hash of `prices_macro.equity_tr` over its full history | **must be identical at P6** — the macro leg is not supposed to change |

### 2. Capture the current model metrics

- [ ] Copy the latest recorded IC / Sharpe per model and per composite into `baseline.md`, with
      the run id and date they came from. After P6 these become non-comparable, so this is the
      only surviving record.
- [ ] Note explicitly which composites are expected to *degrade*: `value` and `value_rerating`
      (10/10 and 8/8 members affected). A fall there is the fix working, not a regression.

## Verification

- [ ] `baseline.json` exists and every key above is populated
- [ ] `error_decomposition.residual` is 1.0000 (±1e-4) in every year — **if not, stop and
      re-open the research**, the multiplicative decomposition is the plan's foundation
- [ ] The numbers reproduce the research document's tables. A discrepancy means the DB moved
      since 2026-09-01; record the new numbers and note the drift.
- [ ] Re-running the script twice gives byte-identical JSON (no nondeterminism, no `now()`
      leaking into a filter)

## Rollback

Read-only. Nothing to roll back.
