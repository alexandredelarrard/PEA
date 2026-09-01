# Phase 5 — The prices validator and the permanent gate ⬜

**Parent**: [PLAN.md](PLAN.md) · **Depends on**: P4 · **Blocks**: P6 · **Estimate**: 3-4h

## Goal

The invariant that would have caught all of this runs nightly and blocks the cube build.

## Why this phase is not optional

`src/validate/` has **no price validator** — [cli.py:17](src/validate/cli.py#L17) names one as
future work. `tests/` has **zero** occurrences of `auto_adjust`, `adj_close`, `unadjusted`, or
any split-ratio assertion. The bug survived because nothing checked, and the check is free:
Sharadar publishes `marketcap` on exactly the basis P1+P3 establish.

## Changes

### 1. `src/validate/prices.py` (new) + `validate prices` CLI command

Sibling to the fundamentals validator, following its existing conventions: read-only against
every table but the three the validator owns, `--roster`, `--since`, `--report`, `--no-write`.

**Invariant 1 — the market-cap identity (the primary gate)**

```
| close_split(d) x sharesOutstanding(d) / sharadar.marketcap(d) - 1 |  <  1%
```

- [ ] Join `prices` to `fundamentals_history` to `fundamentals_sharadar` ARQ on `(ticker, date)`.
      ~51,255 rows. P0 recorded a median row off by 15% and 1995 rows off by 5x.
- [ ] Cluster failures by ticker so a single bad ticker reads as one finding, not 60.
- [ ] Report only. Sharadar's `marketcap` can itself be wrong for a multi-class name; the
      validator must never auto-correct.

**Invariant 2 — price vintage freshness**

```
| close_split(d) / sharadar.price(d) - 1 |  <  0.5%     on filing dates
```

- [ ] Two independent vendors on the same basis, so this catches a stale adjustment vintage the
      day it appears. This is the check that would have flagged MNST in July 2026 instead of an
      audit finding it in September.

**Invariant 3 — spike-and-revert**

- [ ] No `|move| > 50%` reversing the next day without a corroborating row in the P2 unioned
      split list. Genuine events must pass: 2020-03-09's oil crash (APA, OXY, FANG, TRGP), PCG's
      bankruptcy, CVNA 2022 — verify each is not flagged.

### 2. Unit tests (`tests/data_extract/prices/test_adjustment_basis.py`, new)

Fixture-based, no network, no DB:

- [ ] **The stored basis is split-adjusted, not dividend-adjusted**: AAPL 2020-07-31
      `close_split` == 106.26 (not 102.80); KO 2004-02-27 == 24.98 (not 12.79).
- [ ] **`close_split == close_total` for a non-payer** (AMZN) on every row. This is the clean
      regression guard — it is exactly 0, not approximately.
- [ ] **`close_total / close_split` is monotone non-decreasing and terminates at 1.0** for a
      payer. Encodes the definition of `D(d)` as a test.
- [ ] **The macro leg is total-return**: `SPY`/`equity_tr` must NOT change basis. Assert
      `auto_adjust=True` on the macro call path — this is the P1 trap, pinned.
- [ ] **`forward_compound(ret, h)` != `forward_return(close_split, h)` for a payer**, and they
      agree for a non-payer. Pins hazard 4c so it cannot regress.

### 3. Wire into the DoD gate (D7)

- [ ] `validate prices` runs before the cube build; invariant 1 failing above a threshold blocks it.
- [ ] Pick the threshold from P0's measured distribution, not a round number. Suggested: block if
      >1% of joined rows fail invariant 1, warn below that. A vendor hiccup on a handful of
      tickers must warn, not halt the nightly.
- [ ] Record known-bad tickers as waived clusters with quantified evidence, following the
      fundamentals validator's `fix record` convention — **not** by widening the tolerance.

## Verification

- [ ] `validate prices` runs clean end-to-end on the post-P4 database
- [ ] Invariant 1 passes on **>99%** of the 51,255 rows
- [ ] Invariant 2 passes on all filing dates; any failure names a specific ticker and date
- [ ] Invariant 3 returns 0 unexplained spikes post-2020 (P0 recorded 3, all MNST)
- [ ] Deliberately corrupt one ticker's `close_split` by 2x in a scratch copy and confirm all
      three invariants fire on it — **a gate that has never failed has never been tested**
- [ ] The five unit tests pass, and the AMZN identity is exact

## Rollback

Self-contained and read-only. Remove the command and the gate wiring.
