# Phase 0 — Baseline: record the "before" numbers ✅

**Parent**: [PLAN.md](PLAN.md) · **Depends on**: — · **Blocks**: P1, P2 · **Estimate**: ~1h

## Goal

Turn every measurement in the research document into ONE rerunnable script, and freeze its
output. Without this there is no fix-delta, and no way to tell a genuine improvement from a
coincidence.

## ⚠ There is no cube data — and this phase does not need any

The cube is **empty today**. That is fine: every measurement below is computed from the
extract-layer tables (`prices`, `fundamentals_history`, `fundamentals_sharadar`,
`fundamentals_history_sec`, `sharadar_actions`, `prices_macro`), which is exactly how the
research measured them.

In particular `mcap_error_by_year` **replicates the `daily_market_cap` formula inside the
script** — `close x sharesOutstanding`, ffilled to the daily grid — rather than reading a
`marketCap` column from anywhere. "cube mcap" in the research tables is the name of that
formula's output, not the name of a table.

So P0 is runnable as-is. What the empty cube *does* remove is the model-metrics snapshot (§2).

## Why a script and not a notebook

The same script is rerun at P1, P3 and P4 verification, and again by the user after the manual
cube rebuild. It is the spine of the whole task. It lives in `scripts/` (throwaway diagnostics),
**not** in `src/` — the permanent version of invariants 1–3 is the P5 validator.

## Changes

### 1. `scripts/basis_baseline.py` (new)

- [x] Connect via the psycopg2 creator pattern (the DB password contains `!`, which breaks a
      SQLAlchemy URL string — see the `db-access` memory). Ask for the password; do not hardcode.
- [x] Emit a single JSON blob to `reports/planning/active-tasks/2026-09-01-price-shares-basis-fix/baseline.json`
      plus a markdown summary to `baseline.md`.

Measurements, each keyed so the post-rebuild run can diff them:

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
| `macro_equity_tr_digest` | a hash of `prices_macro.equity_tr` over its full history | **must be identical after P1** — the macro leg is not supposed to change |
| `option_overhang_digest` | a hash of `fundamentals_history.optionOverhang` | **must be identical after P3** — it is split-invariant (both legs carry the same factor), so it is the control that proves the shares change touched only what it should |

### 2. ~~Capture the current model metrics~~ — DROPPED

There is no cube and there are no current model metrics, so there is nothing to snapshot.

This removes the whole non-comparability problem that D6 was worried about: there is no pre-fix
IC or Sharpe to invalidate. **The first rebuild after the fix simply becomes the baseline** —
clean, and measured on a correct basis from the start.

- [x] Record in `baseline.md` that the cube was empty on the day P0 ran, and that no model
      metrics predate the fix. A future session must not go hunting for a "before" column that
      never existed.

## Verification

- [x] `baseline.json` exists and every key above is populated
- [x] `error_decomposition.residual` is 1.0000 (±1e-4) in every year — **if not, stop and
      re-open the research**, the multiplicative decomposition is the plan's foundation
- [x] The numbers reproduce the research document's tables. A discrepancy means the DB moved
      since 2026-09-01; record the new numbers and note the drift.
- [x] Re-running the script twice gives byte-identical JSON (no nondeterminism, no `now()`
      leaking into a filter)

## Rollback

Read-only. Nothing to roll back.
