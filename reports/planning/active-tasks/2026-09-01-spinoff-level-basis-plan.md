# Implementation Plan: the spinoff level-basis gap

**Date Created**: 2026-09-01
**Planning Phase**: 2 of 3 (FIC Workflow)
**Based on**: this session's measurements against the live `pea` DB; the shipped work in
`reports/planning/active-tasks/2026-09-01-price-shares-basis-fix/PLAN.md` (§OUTCOME)
**Next Phase**: `/implement`

---

## Overview

The 2026-09-01 basis fix put price and share count on one basis for **splits**. It did not, and
could not, do so for **spinoffs** — and the residual is measured, bounded and fully explained:

* Yahoo's `Close` back-adjusts history for spinoffs (correct for RETURNS: it keeps the series
  continuous). Sharadar's `price` does not (correct for LEVELS: it is what the stock traded at).
* A spinoff does **not** change the parent's share count — verified against the SEC cover page
  on 8 events, `sharesbas`/SEC = 1.0000 on *both* sides of every one.
* So the price leg carries a factor the share leg does not, and nothing cancels.
  `close_split × sharesOutstanding` **understates** market cap by that factor.

FDX on 2020-12-17, from the live DB:

| | value |
|---|---|
| Sharadar `price` (what FDX traded at) | 292.26 |
| our `close_split` (= 292.26 ÷ 1.241) | 235.50 |
| `sharesOutstanding` (both vendors agree) | 265,070,592 |
| **truth** 292.26 × 265,070,592 | **$77.5bn** |
| **what `daily_market_cap` returns today** | **$62.4bn** — 19.4% low |

This plan corrects the LEVEL at the point of use. It does **not** touch `prices`.

**Decisions settled with the user (2026-09-01), do not re-litigate:**

| # | decision |
|---|---|
| D1 | Fix inside `data_aggregate`, **not** the extract layer. `prices.close_split` is never rewritten. |
| D2 | The `sharesOutstandingPit` reverse-split veto fix is **Phase 1** of this plan — it is not independent (see §Phase 1). |
| D3 | `dollar_volume_63` / `amihud_63` are fixed too. |
| D4 | `daily_market_cap`'s new parameter is **required, keyword-only, no default**. |
| D5 | `S(d)` is computed once in `StepCubePrices` and **stored as a column on `cube_part_prices`**, for auditability and to avoid recomputing it in three steps. |
| D6 | The validator applies `S(d)` **in memory only**. `src/validate/prices.py` stays write-free. |

---

## Current State Analysis

### The factor, and where it comes from

```
S(d)  =  Π{ prices_splits.ratio    : date > d }
       ÷ Π{ split_events(...).value : date > d }
```

A **ratio of two products**, not a set subtraction — that matters for HON, where both sources
carry an event on 2026-06-29 with *different* values (yfinance 0.9535, Sharadar 0.5).

**yfinance is the source, not `sharadar_actions`**, and this is not a preference:

1. The factor must be *exactly what Yahoo applied to its own series*, because its only job is
   to undo that. Sharadar's `spinoff` value is a **share ratio** (0.1 = one child share per ten
   parent), not a price factor — it cannot undo a price adjustment.
2. Measured: **162 of the 226 affected tickers have no `sharadar_actions` row at all** at the
   break — FDX, SPGI, WDC, LEN, J, LH, FLEX, T, ZBH, DHR, CMCSA, GE…
3. `1/step` matched `prices_splits` to four decimals on every one of them: FDX 1.2410 = 1.241,
   LH 1.1640 = 1.164, T 1.3240 = 1.324, FLEX 1.3270 = 1.327, GE 1.6692 = 1.253 × 1.332,
   BDX 1.3038 = 1.025 × 1.272.

`sharadar_actions` participates only through `split_events`, which decides which
`prices_splits` rows are *genuine splits* — those must appear in the denominator, because a
genuine split already cancels against `sharesbas`.

Worked check, HON, a 1996 row:
```
Π(prices_splits after d) = 2(1997) × 1.00533 × 1.011 × 1.032 × 1.061 × 0.9535 = 2.12228
Π(genuine after d)       = 2(1997) × 0.5(2026)                                = 1.00000
S(d)                     = 2.12228        # measured leg_price = 0.4712 = 1/2.1223  ✓
```

### Verified consumers

**Through `daily_market_cap` — 7 call sites, all fixed by one change:**

| # | site | feeds |
|---|---|---|
| 1 | `utils/target/factors.py:94` | `size` = −log(mcap) → risk projection → the neutralized/epsilon target |
| 2 | `transformers/step_cube_target.py:260` | `market_cap` size characteristic |
| 3 | `utils/fundamentals/fundamental_features.py:1452` | earnings_yield, sales_yield, book_yield, fcf_yield |
| 4 | `utils/fundamentals/intrinsic.py:94` | intrinsic-value features |
| 5 | `utils/extras/institutional_features.py:159` | 13F value/mcap weight, net $ flow |
| 6 | `utils/extras/superinvestor_features.py:188` | elite 13F value/mcap weight |
| 7 | `utils/extras/insider_features.py:104` | insider size-scaled conviction |

plus `utils/common/pit.py:269` `PitFrames.market_cap`, the memoized funnel several share.

✓ The extras pass the **whole** `fundamentals_history` frame; `daily_market_cap` selects
`sharesOutstanding` while `inst_ownership_pct` separately selects `sharesOutstandingPit`
(`institutional_features.py:150`). Both bases are already right and stay right.

**Distorted the same way, NOT through `daily_market_cap`:**

* `utils/momentum/features.py:197` — `dollar_vol = split × volume`. Its docstring reasons only
  about splits ("the split factor cancels between the two"). For a spinoff the price is divided
  by S and volume is untouched, so **nothing cancels**. Affects `dollar_volume_63` and
  `amihud_63`.

**Suspected, to be MEASURED in Phase 0 before any edit:**

* `utils/fundamentals/dividend_features.py:73` — `dividend_yield = ttm_ps / close_split`, where
  `ttm_ps` is yfinance's `Dividends`. If Yahoo back-adjusts dividends for spinoffs the way it
  does the quote, the legs cancel and there is nothing to do.
* `utils/fundamentals/earnings_features.py` — `fwd_eps_yield`, `forward_earnings_yield`, both
  `eps / close_split` where eps is an analyst estimate.

**Reads `close_split` and must NOT change:**
`step_cube_prices` (trading calendar, universe), `strategies/step_ls.py`,
`step_eq_long_only.py`, `step_super_investors.py` (blotter share counts, marking a 13F mirror —
these want a continuous quote), `factors.py:86` momentum (already `close_total`).

---

## Desired End State

* `daily_market_cap` returns the historical market cap on ONE basis, for every ticker.
* `cube_part_prices` carries an auditable `level_factor` column (1.0 for ~89% of rows).
* `sharesOutstandingPit` is the actual share count for the 26 tickers a real reverse split was
  wrongly vetoed on.
* `validate prices` measures the corrected identity, and `prices` is byte-identical throughout.
* Measured targets:

| check | today | after |
|---|---|---|
| invariant 1 (`close_split·S·shares / marketcap`) | 87.44% | **≈98.21%** |
| invariant 2 (`close_split·S / sharadar.price`) | 87.33% | **≈98.09%** |
| rows that pass today and fail after | — | **≤1** (OXY, measured) |
| control cohort output (AAPL/KO/JNJ/MSFT) | — | **bit-identical** |

---

## Out of Scope

* **MNST.** `S(d)` does not fix it: Yahoo *publishes* the 2026-08-11 x2 but never applied it, so
  Π(yf) and Π(genuine) both contain it and it cancels to 1.0. MNST needs the anchored per-bar
  vintage repair prototyped this session (anchor error 1.0000 → 0.0005, window max |return|
  95.59% → 4.04%), which is a separate task. 122 rows.
* **Visa** — `sharadar.price × sharesbas ≠ marketcap`, a vendor-internal multi-class defect
  (`sharesbas` is Class A, `marketcap` is as-converted; the leg flips 1.206 → 0.913 around
  2016-07). 74 rows. Not ours to fix.
* **Stock dividends Sharadar's price ignores** — APA 1.1×1.05, HBAN 1.1⁵, ORCL. ~60 rows.
* **The as-of join noise** — 527 rows / 179 tickers, median 2 rows/ticker, median offset 2.6%,
  70% pre-2003. Not a basis error.
* Any change to `prices`, `close_split`, `close_total`, or the extract price path.
* The cube/label rebuild and the model re-baseline — the user's, as before.

---

## Implementation Approach

### Phase 0: Freeze the before ⬜

**Goal**: a rerunnable measurement that makes every later claim checkable, taken BEFORE any edit.

**Changes**:

1. `scripts/spinoff_level_baseline.py` (new):
   - [ ] `level_factor(tickers, dates, prices_splits, genuine)` — the reference implementation of
         `S(d)`, so Phase 2 can be diffed against it
   - [ ] emit `before.json` + `before.md` next to this plan with:
         - invariant 1 / invariant 2 pass rates, raw and S-adjusted
         - per-ticker `S` for the spinoff cohort and proof that `S ≡ 1.0` for the control cohort
         - `marketCap` for FDX/GE/DD/T/HPQ/EXC/RTX at 4 dates each, ours vs Sharadar's
         - **the two open questions, MEASURED**:
           `dividend_yield` (yfinance `ttm_ps` / `close_split`) vs Sharadar `dps / price`, and
           `fwd_eps_yield` — for spinoff vs control tickers. A ratio of 1.0 on the spinoff
           cohort means the legs cancel and that consumer needs no change.
   - [ ] control digests that MUST NOT move: `ret` and `close_total` from `cube_part_prices`

**Verification**:
- [ ] `"$PY" scripts/spinoff_level_baseline.py --tag before`
- [ ] `S ≡ 1.0` exactly on AAPL, KO, JNJ, MSFT (never spun anything off)
- [ ] invariant 1 raw reproduces **87.44%** and S-adjusted reproduces **≈97.64%**
      (pre-Phase-1, i.e. before the reverse splits are un-vetoed)
- [ ] Record the dividend/earnings-yield verdict IN THE PLAN before Phase 3 touches them

---

### Phase 1: Un-veto the real reverse splits ⬜

**Goal**: `sharesOutstandingPit` becomes the actual share count on 26 tickers, and `S(d)` gets
the right denominator. **This must precede Phase 2** — un-vetoing HON's `split=0.5` changes
which rows `split_events` returns, which changes `S(d)`.

**The bug**: `field_map.split_events` drops a Sharadar `split` co-dated with a `spinoff`. It was
justified on HON with *"`sharesbas` is unchanged across the date (316,826,560 → 316,940,010)"*.
**That argument is void** — `sharesbas` is retroactively restated, so it is continuous across a
real split *by construction*. Continuity proves nothing.

Deep history discriminates, and says the split is real:

| HON | Sharadar | actual |
|---|---|---|
| 2010 `sharesbas` | 390,086,318 | ~780M — halved |
| 2010 `price` | 94.52 | ~47 — doubled |
| 2015 `dps` | 1.03 | ~0.5175 — doubled |
| 2015 `epsdil` | 3.20 | ~1.60 — doubled |

Four fields restated 2×, with `marketcap` correct ($36.9bn in 2010) because the two cancel.
All 27 vetoed rows are split-shaped; 24 are reciprocals of small integers — the reverse-split
signature: EXPE/ITT/KSU/HON 0.5, TYC/LDOS/PRSU 0.25, DD/HLT/RRD/SBRA 1/3, MSI/CCEC/UNTD 1/7,
T1/HSH 0.2.

**Changes**:

1. `src/data_extract/utils/fundamentals_sharadar/field_map.py`:
   - [ ] **Delete the spinoff veto** in `split_events` (the `spinoffs` set and the
         `spinoff_dropped` branch). The shape test is the only filter, as for every other
         candidate.
   - [ ] `SPLIT_INTEGER_TOL` **1e-6 → 1e-4**, with the margin recorded in the comment:
         Sharadar publishes to 5 dp, so `0.33333` vs `1/3` is 3.33e-6 and `0.14286` vs `1/7` is
         2.86e-6 — both currently REJECTED. The nearest false positive is BDX `1.272` vs `14/11`
         at **7.27e-4**, and SJM `0.945` vs `17/18` at 5.6e-4. 1e-4 sits 30× above the rounding
         error and 7× below the nearest false positive.
   - [ ] **Conflict resolution in `union_split_sources`**: when the two vendors match on date but
         the ratios differ materially, keep the **split-shaped** one, not unconditionally
         yfinance's. Required for DD (yfinance 0.4725 vs Sharadar 0.33333 — 0.33333 is the split)
         and HON (0.9535 vs 0.5). Keep the yfinance DATE. Warn on every such resolution.
   - [ ] Log `CNX 2017-11-29 x1.2` for manual review: it is the only newly-admitted row whose
         ratio is split-shaped (6/5) but which is *not* a reciprocal of a small integer.
         CDSCY 0.64 and PRY1 1.30591 are rejected by the shape test and need no action.

2. `tests/data_extract/sharadar/test_split_union.py`:
   - [ ] **Invert** `test_a_spinoff_co_dated_row_is_still_rejected` → `..._is_now_kept`, with
         the deep-history evidence in the docstring
   - [ ] Add: Sharadar 5-dp rounding is accepted (0.33333, 0.14286, 0.16667) while 1.272, 0.945,
         1.025, 0.3775 are still rejected — the tolerance-margin test
   - [ ] Add: the DD conflict — yfinance 0.4725 + Sharadar 0.33333 on one date → **0.33333** kept
   - [ ] Add: the HON conflict — yfinance 0.9535 + Sharadar 0.5 → **0.5** kept

3. `src/data_extract/utils/fundamentals_sharadar/merge_history.py`:
   - [ ] No code change expected; confirm `deadjust_splits` still touches only
         `sharesOutstandingPit` (the only column with `split_basis` in
         `configs/sharadar/sharadar_field_map.json`)

**Verification**:
- [ ] `"$PY" -m pytest tests/data_extract/sharadar/test_split_union.py -q` → all pass
- [ ] Rebuild `fundamentals_history`; assert `sharesOutstanding == sharesbas` still **51,255 /
      51,255** (this fix must not touch it)
- [ ] `sharesOutstandingPit / sharesbas` becomes the split factor (2.0 for HON, 3.0 for DD,
      7.0 for MSI, 4.0 for LDOS, 3.0 for HLT) on pre-event rows; 1.0 after
- [ ] SEC cover-page agreement (`scripts/basis_baseline.py`) must **not regress** from
      5,412/5,553 — none of the 26 tickers is in the 99-ticker SEC table, so this is a
      no-change assertion
- [ ] `option_overhang_digest` unchanged (`99f56a4553b6e9c0`)

**Rollback**: revert the three edits; `deadjust_splits` is deterministic given the config, so a
`fundamentals_history` rebuild restores the previous state exactly.

---

### Phase 2: Compute and store `level_factor` ⬜

**Goal**: one auditable column on `cube_part_prices`, computed once.

**Changes**:

1. `src/data_aggregate/utils/common/level_basis.py` (new, pure — no I/O):
   - [ ] ```python
         def level_factor(index: pd.DatetimeIndex, universe: Sequence[str],
                          yf_splits: pd.DataFrame, genuine_splits: pd.DataFrame) -> pd.DataFrame:
             """S(d) = Π(all yfinance factors after d) / Π(genuine splits after d), wide.

             A ratio of two PRODUCTS, not a set difference: HON carries an event on
             2026-06-29 in both sources with different values (0.9535 and 0.5), and only the
             ratio gives the right answer (2.12228, measured).
             """
         ```
   - [ ] Snap `|S − 1| < 1e-12` to exactly `1.0`, so a ticker whose two event sets agree is
         **bit-identical** to today rather than 1.0000000000000002
   - [ ] Both products are computed right-to-left over dates sorted ascending, so a ticker
         present in both sets cancels term-by-term

2. `src/data_aggregate/transformers/step_cube_prices.py`:
   - [ ] Load `Tables.prices_splits` and `Tables.sharadar_actions` (projected) via
         `self._context.store`
   - [ ] Call `split_events(actions, yf_splits)` for the denominator — the SAME function the
         extract layer uses, so the two layers cannot drift
   - [ ] Add `level_factor` to the written frame
   - [ ] Log: rows with `S ≠ 1`, distinct tickers, and the top 10 by `|log S|`

3. `src/data_aggregate/utils/common/price_frames.py`:
   - [ ] `ALL_FIELDS += ("level_factor",)`
   - [ ] `level_factor: pd.DataFrame | None = None` on `PriceFrames`
   - [ ] Docstring: it is a FACTOR, not a price; multiply a LEVEL by it, never a return

4. `tests/data_aggregate/test_level_basis.py` (new):
   - [ ] `S ≡ 1.0` exactly when the two sources agree (the AAPL case) — assert `is` equality to
         1.0, not `approx`
   - [ ] FDX: one yfinance-only non-split factor → `S = 1.241` before, `1.0` after
   - [ ] GE: two stacked factors → `S = 1.253 × 1.332 = 1.6692`
   - [ ] HON: both sources on one date with different values → `S = 2.12228`
   - [ ] MNST: identical sets → `S = 1.0` (documents that this plan does NOT fix MNST)
   - [ ] empty `prices_splits` → `S ≡ 1.0`, no crash

**Verification**:
- [ ] `"$PY" -m pytest tests/data_aggregate/test_level_basis.py -q`
- [ ] `build-prices -F`; then `SELECT count(*) FROM cube_part_prices WHERE level_factor <> 1`
      → ≈11% of rows, ≈72 tickers
- [ ] `close_split`, `close_total`, `ret`, `volume` digests in `cube_part_prices` **unchanged**
      vs Phase 0 — the new column must be purely additive

---

### Phase 3: Consume it ⬜

**Goal**: every LEVEL is on the corrected basis; no RETURN moves.

**Changes**:

1. `src/data_aggregate/utils/common/pit.py`:
   - [ ] ```python
         def daily_market_cap(fundamentals_history, close_split, *, level_factor):
         ```
         **Required, keyword-only, no default** (D4) — the pattern P1 used for `auto_adjust`,
         so a future caller cannot silently inherit the wrong basis.
   - [ ] `mcap = close_split[cols].mul(shares[cols]).mul(level_factor[cols])`, aligned on the
         same column subset
   - [ ] Docstring: why the factor exists, with the FDX 2020-12-17 numbers
   - [ ] `PitFrames` takes and forwards `level_factor`

2. The 7 call sites — each passes `frames.level_factor` (or the frame it already holds):
   - [ ] `utils/target/factors.py:94` — add `level_factor` to `build_characteristics`
   - [ ] `transformers/step_cube_target.py:260`
   - [ ] `utils/fundamentals/fundamental_features.py:1452`
   - [ ] `utils/fundamentals/intrinsic.py:94`
   - [ ] `utils/extras/institutional_features.py:159`
   - [ ] `utils/extras/superinvestor_features.py:188`
   - [ ] `utils/extras/insider_features.py:104`
   - [ ] `transformers/step_cube_extras.py` ×3 and `step_cube_fundamentals.py` ×3 — thread
         `level_factor=frames.level_factor` alongside the existing `stock_close=frames.close_split`

3. `src/data_aggregate/utils/momentum/features.py` (D3):
   - [ ] `dollar_vol = split × volume × level_factor`
   - [ ] Correct the module docstring: it currently claims the factor cancels between price and
         volume. That is true for splits and **false for spinoffs** — volume is not
         spinoff-adjusted.
   - [ ] `compute_raw_features(..., level_factor=None)`; `None` → 1.0 so the synthetic
         fingerprint harness is unaffected

4. `dividend_features.py` / `earnings_features.py`:
   - [ ] **Only if Phase 0 measured them broken.** If the yfinance dividend leg carries the same
         spinoff adjustment as the quote, they cancel and nothing changes — record the measured
         verdict here either way.

5. Tests:
   - [ ] Update the ~6 files calling `daily_market_cap` (`test_pit_cache.py`,
         `test_institutional_features.py`, `test_target_exposure_neutral.py`,
         `aggregate_fingerprint.py`, `test_fundamental_features.py`) to pass `level_factor`
   - [ ] New: `daily_market_cap` with `S ≡ 1` is **bit-identical** to the old two-argument result
   - [ ] New: FDX 2020-12-17 → `$77.5bn`, matching Sharadar within 1% (today: `$62.4bn`)

**Verification**:
- [ ] `"$PY" -m pytest tests/data_aggregate/ -q` — no NEW failures against the 202p/9f baseline
- [ ] `TypeError` when `level_factor` is omitted — the required-kwarg contract is live
- [ ] `tests/data_aggregate/test_aggregate_regression.py` — record which digests move and why.
      `label.*` and `panel.betas` must **NOT** move (returns are untouched); `panel.fundamentals`
      and the mcap-derived families SHOULD.

---

### Phase 4: Validator ⬜

**Goal**: `validate prices` measures the identity the cube actually builds. Read-only (D6).

**Changes**:

1. `src/validate/prices.py`:
   - [ ] `load_panel` computes `S(d)` in memory from `prices_splits` + `sharadar_actions`, reusing
         `level_basis.level_factor` — **no write, ever**
   - [ ] `invariant_market_cap`: `close_split × S × sharesOutstanding / marketcap`
   - [ ] `invariant_price_vintage`: `close_split × S / sharadar.price`, so what remains is a
         genuine stale vintage rather than a convention difference
   - [ ] Report BOTH raw and S-adjusted rates, so the spinoff wedge stays visible
   - [ ] Update `MCAP_BLOCK_SHARE`'s comment with the new measured rate. **Keep it `None`** —
         raising a new gate is a separate decision, not this plan's.
   - [ ] Rewrite the module docstring: the "12.6% fails on a CORRECT table because Sharadar is the
         inconsistent side" paragraph is **wrong** and must be replaced with the measured
         decomposition (leg_shares 100.00%, leg_vendor 99.82%, leg_price 87.59%).

2. `tests/data_extract/prices/test_adjustment_basis.py`:
   - [ ] Add: a synthetic spinoff ticker fails invariant 1 without `S` and passes with it

**Verification**:
- [ ] `validate prices` → invariant 1 **≈98.21%**, invariant 2 **≈98.09%**
- [ ] The residual clusters are exactly the named ones: MNST 122, V 74, APA/HBAN/ORCL ~60,
      join noise ~527. **No new cluster.**
- [ ] Invariant 3 (spike-and-revert) unchanged — it never reads `S`

---

### Phase 5: Prove it, and write it down ⬜

**Goal**: the before/after the user asked for, on both cohorts.

**Changes**:

1. `scripts/spinoff_level_baseline.py --tag after`:
   - [ ] **Spinoff cohort** (FDX, GE, DD, T, HPQ, EXC, RTX, NI, BAX, EQT): `marketCap` within 1%
         of Sharadar's `marketcap`, where today it is off by 1.2× to 5.0×
   - [ ] **Control cohort** (AAPL, KO, JNJ, MSFT, PG, XOM): `marketCap` **bit-identical** to the
         Phase 0 `before.json`. This is the test that proves the change is targeted.
   - [ ] **Aggregate**: 87.44% → ≈98.21%, and **at most 1 row** that passed before now fails
         (measured: OXY, a single as-of-join row)
   - [ ] **Return controls**: `ret` and `close_total` digests byte-identical
   - [ ] Cross-sectional impact, so the user can size the model effect: ~7.2% of all rows change
         size decile, ~7.3% change earnings-yield quintile (28.97% / 30.40% among affected rows)

2. Docs:
   - [ ] `docs/data_schema.md` — the `level_factor` column
   - [ ] `src/validate/README.md` — the corrected invariant-1 story
   - [ ] `reports/2026-09-01/...__DATA.md` — supersede the "87.4% is the ceiling / Sharadar is the
         inconsistent side" finding with the measured decomposition
   - [ ] This plan: §OUTCOME with the measured numbers

**Verification**:
- [ ] Both cohorts pass their assertions
- [ ] `"$PY" -m pytest tests/ -q` — no new failures vs the recorded baseline

---

## Testing Strategy

**Unit** — `test_level_basis.py` (new, 6 cases), `test_split_union.py` (4 new + 1 inverted),
`test_pit_cache.py` + 5 others updated for the required kwarg.

**Integration** — `S ≡ 1` ⇒ bit-identical `daily_market_cap`; the fingerprint regression with an
explicit statement of which digests move; `validate prices` end-to-end.

**Manual** — FDX 2020-12-17 = $77.5bn against Sharadar; HON `sharesOutstandingPit` doubles
pre-2026-06-29; `SELECT ... WHERE level_factor <> 1` returns ≈72 tickers.

---

## Risk Mitigation

| # | risk | mitigation |
|---|---|---|
| 1 | **Phase 1 admits a false positive.** The shape test at 1e-4 is the only filter once the veto is gone. | The margin is measured (7.27e-4 nearest false positive vs 3.33e-6 largest true rounding error). Every newly-admitted row is logged; CNX 1.2 is called out for manual review. |
| 2 | **`S` silently ≠ 1 for a clean ticker** from float noise, moving a control digest. | Snap `|S−1| < 1e-12` to exactly 1.0; the control cohort must be **bit**-identical, which will catch it. |
| 3 | **A call site is missed.** | The required keyword-only argument makes a missed site a `TypeError`, not a wrong number. |
| 4 | **`cube_part_prices` column change.** | Not in `sql/schema.sql` (parts are dynamic), and `StepCubePrices` already self-heals on `COLUMNS_CHANGED` with `run(full=True)`. |
| 5 | **A return-shaped feature moves.** | `ret` / `close_total` digests are explicit Phase 0 controls and are asserted in Phase 5. |
| 6 | **Phase 1 and Phase 2 interact** (un-vetoing changes `S`). | Enforced ordering; Phase 0 records the pre-Phase-1 S-adjusted rate (97.64%) separately from the post (98.21%), so the two effects stay distinguishable. |

**Rollback**: each phase is independently revertible. Phase 1 needs a `fundamentals_history`
rebuild; Phases 2–4 are code-only plus a `build-prices -F`. `prices` is never written, so there
is nothing to restore there.

---

## Success Criteria

- [ ] invariant 1 ≥ 98%, invariant 2 ≥ 98%, ≤1 newly-failing row
- [ ] Control cohort `marketCap` bit-identical; `ret` / `close_total` digests byte-identical
- [ ] FDX 2020-12-17 `marketCap` = $77.5bn (Sharadar: $77.5bn); today $62.4bn
- [ ] `sharesOutstandingPit` carries the reverse split on all 26 tickers
- [ ] `daily_market_cap` cannot be called without stating a basis
- [ ] `level_factor` queryable in `cube_part_prices`
- [ ] No new test failures against the recorded baseline
- [ ] The residual is exactly the four named out-of-scope clusters — no new one

---

## Estimated Effort

| phase | estimate |
|---|---|
| 0 — freeze the before | 1.5 h (incl. the 2 measured open questions) |
| 1 — un-veto | 2 h + a `fundamentals_history` rebuild |
| 2 — compute + store `S` | 2 h + a `build-prices -F` |
| 3 — consume (7 sites + momentum + ~6 test files) | 3 h |
| 4 — validator | 1.5 h |
| 5 — prove + docs | 1.5 h |
| **total** | **≈11.5 h** plus two rebuilds |

---

## Notes for Implementation

* **`split_events` is shared** between the extract layer and `StepCubePrices`. That is deliberate
  — it is what stops the numerator and denominator of `S` drifting apart. Do not fork it.
* **`S` multiplies a LEVEL, never a RETURN.** A return computed from `close_split × S` would be
  wrong at the spinoff date, which is exactly what Yahoo's back-adjustment exists to prevent.
* **Do not widen `MCAP_TOLERANCE`** to make a cluster pass. Four residual clusters are named and
  scoped out; a fifth appearing means something in this plan is wrong.
* **Never `print()`** in `src/` — `self._log` / `context.log`.
* All tabular I/O through `self._context.store`; table names only via `Tables.<name>`.
* The user's long-lived `stash@{0}` (b492640) must remain untouched.
