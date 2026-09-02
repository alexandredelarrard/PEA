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

  **MEASURED 2026-09-01 across all 853 events in `prices_splits`, not reasoned:**

  | | median volume ratio after/before | median event factor |
  |---|---|---|
  | 746 genuine splits | **0.869** | 2.000 |
  | 107 non-split (spinoff) factors | **1.103** | 1.122 |

  *Splits*: volume IS retroactively adjusted. Were it raw, BKNG's 25:1 would show ~25 and
  AMZN's 20:1 ~20; they show **0.804** and **0.826** (AVGO 10:1 → 0.677, AAPL 4:1 → 1.095).
  The 0.87 is the ordinary post-event fade in interest. Median dollar-volume ratio **0.8745**
  ≈ the volume ratio **0.8690**, so price ÷ R and volume × R cancel exactly — the docstring is
  RIGHT for splits and only needs the spinoff caveat added.

  *Spinoffs*: volume is NOT adjusted, which is correct — the share count does not change.
  Were it adjusted, a 2.5× factor would show ~1/2.5 ≈ 0.4. Measured: NI 2.545 → **1.107**,
  DD 2.390 → **1.576**, HPQ 2.202 → **0.976**, ABT 2.084 → **0.883**, DELL 1.973 → **0.925**,
  BAX 1.841 → **0.906**. Price ratio across the event is 1.02 (continuous, i.e. back-adjusted).

  So `dollar_volume_63` is UNDERSTATED by `S(d)` before a spinoff and `amihud_63` is
  OVERSTATED by it — those 72 names read as more illiquid than they were.

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

### Phase 0: Freeze the before ✅

**Goal**: a rerunnable measurement that makes every later claim checkable, taken BEFORE any edit.

**Changes**:

1. `scripts/spinoff_level_baseline.py` (new):
   - [x] `level_factor(tickers, dates, prices_splits, genuine)` — the reference implementation of
         `S(d)`, so Phase 2 can be diffed against it
   - [x] emit `before.json` + `before.md` in
         `reports/planning/active-tasks/2026-09-01-spinoff-level-basis/`
   - [x] control digests that MUST NOT move: `close_total` and the return derived from it.
         Taken from **`prices`, not `cube_part_prices`** — the live part table is a build
         behind (it still carries the pre-fix `close` column and has no `close_split` /
         `close_total`), so it cannot serve as a control across this change. The script
         digests the part table too when the columns appear.

**Verification** — all met:
- [x] `"$PY" scripts/spinoff_level_baseline.py --tag before`
- [x] `S ≡ 1.0` **exactly** on all 6 control tickers (`exactly_one: true`, `==` not `approx`)
- [x] invariant 1 raw **87.44%** (predicted 87.44%), S-adjusted **97.63%** (predicted ≈97.64%);
      invariant 2 raw **87.33%**, S-adjusted **97.51%**
- [x] FDX 2020-12-17: **$62.425bn → $77.47bn**, Sharadar **$77.47bn** — exact

**MEASURED RESULTS (Phase 0, pre-Phase-1)**

| | value |
|---|---|
| panel | 51,847 filing rows / 489 tickers; 859 yfinance split rows, 8,236 genuine |
| `S ≠ 1` | 5,847 rows (11.28%) / 83 tickers |
| invariant 1 | 87.44% → 97.63% (+5,176 pass, **3 newly fail**: IVZ, OXY) |
| invariant 2 | 87.33% → 97.51% (+5,224 pass, **3 newly fail**: IVZ, OXY) |

Residual clusters after `S`, pre-Phase-1 — **no fifth family**. The top five are precisely the
reverse splits Phase 1 un-vetoes, which is independent confirmation of the phase ordering:

| cluster | rows | median ratio | disposition |
|---|---|---|---|
| HON 123, DD 94, MSI 61, LDOS 28, HLT 12 | 318 | 0.5 / 0.3333 / 0.1429 / 0.25 / 0.3333 | **Phase 1 fixes** |
| MNST | 122 | 2.0 | out of scope (named) |
| V | 74 | 0.9368 | out of scope (named) |
| APA 28, SJM 27, HBAN 19, ORCL 5 | 79 | 0.87 / 0.945 / 0.75 / 0.67 | out of scope (stock dividends) |
| IP, JCI, WBD, BX, BLDR, CCI, HWM, HAS, GEN | 285 | 0.93–1.33 | as-of join noise |

**THE TWO OPEN QUESTIONS — ANSWERED. Do not re-litigate.**

Both were scored as a YIELD ratio against Sharadar's own level-basis yield, on the 
`|S−1| > 10%` rows where the two hypotheses (`ratio == 1` = legs cancel, `ratio == S` = 
broken) are far apart. A definitional wedge between the vendors inflates both distances 
equally, so the ratio of the distances is the call:

| consumer | measured | verdict |
|---|---|---|
| `dividend_features.dividend_yield` (`ttm_ps / close_split`) | 2,968 rows, median ratio **1.0** vs median S 1.323; **148,405× closer to 1.0 than to S** | **LEGS CANCEL.** yfinance back-adjusts `Dividends` for spinoffs exactly as it does the quote. **NO CHANGE.** |
| `earnings_features` (`eps / close_split`) | 1,422 rows, median ratio **1.3831** vs median S 1.275; **3.5× closer to S than to 1.0** | **DISTORTED.** yfinance's earnings history is on the LEVEL basis. **Phase 3 must fix it.** |

**ONE CONSUMER THE PLAN DID NOT LIST**, found while measuring the dividend leg:
`dividend_features.py:92` builds its own `mcap = shares × close` inline. Where the ex-date
history has the dividend, `total / mcap` reduces algebraically to `ttm_ps / close` and is the
same number as the primary leg — but where it falls back to the SEC dollar total
(`dividendsPaid`, a LEVEL figure) it is a market cap and is `S` too low, making that yield `S`
too high. Narrow, but it is a market cap, and every market cap goes on one basis. Fixed in
Phase 3.

---

### Phase 1: Un-veto the real reverse splits ✅

**Goal**: `sharesOutstandingPit` becomes the actual share count, and `S(d)` gets the right
denominator. **Must precede Phase 2** — un-vetoing HON's `split=0.5` changes which rows
`split_events` returns, which changes `S(d)`. Phase 0 confirmed the ordering empirically: the
five largest residual clusters after `S` but before Phase 1 were exactly HON/DD/MSI/LDOS/HLT.

**The bug**: `field_map.split_events` dropped a Sharadar `split` co-dated with a `spinoff`,
justified on HON with *"`sharesbas` is unchanged across the date"*. **That argument is void** —
`sharesbas` is retroactively restated, so it is continuous across a real split *by
construction*. Deep history discriminates and says the split is real (HON 2010 `sharesbas`
390M vs an actual ~780M; `price` 94.52 vs ~47; 2015 `dps` 1.03 vs 0.5175; `epsdil` 3.20 vs
1.60 — four fields restated 2×, `marketcap` correct because the legs cancel).

**Changes — all done**:

1. `src/data_extract/utils/fundamentals_sharadar/field_map.py`:
   - [x] Spinoff veto **deleted**. The shape test is the only filter, as for every other
         candidate. `spinoff` rows are still read, but only to NAME the rows the veto used to
         drop (`_log_codated`).
   - [x] `SPLIT_INTEGER_TOL` **1e-6 → 1e-4**, sitting in a measured gap: 30× above the largest
         true rounding error (`0.33333` vs 1/3 = 3.33e-6) and 7× below the nearest false
         positive (BDX `1.272` vs 14/11 = 7.27e-4).
   - [x] **Conflict resolution** (`_resolve_ratio_conflict`, `SPLIT_RATIO_CONFLICT_TOL` = 1%):
         on a corroborated event the DATE is always yfinance's, but where the RATIOS differ
         materially the **split-shaped** one wins. Fires on 6 dates, every one correct:
         DD 0.4725→**0.33333**, HLT 0.4873→**0.33333**, HON 0.9535→**0.5**,
         LDOS 0.405→**0.25**, MSI 0.2474→**0.14286**, WBD 1.957→**2.0**.
   - [x] Review log. ⚠ **Deviation from the plan, deliberately**: the plan asked to log CNX
         2017-11-29 x1.2. A generic "split-shaped but not n:1" rule prints **~300 lines** of
         long-standing, twice-corroborated 3:2 splits and 5% stock dividends, and a warning
         that long stops being read. `_log_codated` instead names exactly the rows the veto
         used to drop — the entire behavioural change — and flags the odd-shaped ones among
         them, which is where CNX lands.

2. `tests/data_extract/sharadar/test_split_union.py` — 32 pass:
   - [x] `test_a_spinoff_co_dated_row_is_still_rejected` → **`..._is_now_kept`**
   - [x] `test_the_tolerance_sits_in_a_measured_gap` — the 5-dp rounding cases accepted
         (0.33333 / 0.14286 / 0.16667) and 1.272 / 0.945 / 1.067 / 1.025 / 0.3775 still
         rejected, each asserting its measured distance sits on the right side of the tolerance
   - [x] `test_the_split_shaped_ratio_wins_a_material_conflict` — DD and HON, parametrized
   - [x] The GOOGL test extended to be the conflict rule's NEGATIVE case (1.998 vs 2.0 is
         0.1% apart, inside the band, so rounding never "resolves")

3. ⚠ **Two more tests the plan did not list** asserted the same reversed behaviour, in
   `tests/data_extract/sharadar/test_sharadar_field_map.py`. Both corrected:
   - [x] `test_a_spinoff_priced_split_row_is_rejected` → **`..._is_kept`**
   - [x] `test_hon_share_count_is_unchanged_across_its_split_row` →
         **`test_hon_continuity_proves_nothing_about_the_split`**. The measurement is kept
         (the step IS 1.0004) but its meaning is inverted: continuity is a property of the
         vendor's storage, not of the company.

4. `merge_history.py` / `gap_check.py`: `deadjust_splits` confirmed to touch only
   `sharesOutstandingPit` (the one column with `split_basis` in the config); comments corrected
   to say the `spinoff` rows are now diagnostic rather than a veto.

**Verification — all met** (`fundamentals-history-merged -F`, 51,255 rows / 489 tickers):

| check | before | after | verdict |
|---|---|---|---|
| `sharesOutstanding == sharesbas` | 51,255 / 51,255 | **51,255 / 51,255** | ✅ untouched |
| SEC cover-page agreement | 5,412 / 5,553 (19/96 tickers fail) | **5,412 / 5,553 (19/96)** | ✅ no regression, no change |
| `option_overhang_digest` | `99f56a4553b6e9c0` | **`99f56a4553b6e9c0`** | ✅ identical |
| `macro_equity_tr_digest` | `c794c4b8e6590101` | **`c794c4b8e6590101`** | ✅ identical |
| invariant 1, S-adjusted | 97.63% | **98.30%** | ✅ beats the ≈98.21% target |
| invariant 2, S-adjusted | 97.51% | **98.18%** | ✅ beats the ≈98.09% target |

`sharesOutstandingPit / sharesbas` either side of each un-vetoed event — every one an exact
integer:

| ticker | event | before | after | step |
|---|---|---|---|---|
| HON | 2026-06-29 | 2.0000 | 1.0000 | **2×** |
| DD | 2019-06-03 | 9.0001 | 3.0000 | **3×** |
| MSI | 2011-01-04 | 6.9999 | 1.0000 | **7×** |
| LDOS | 2013-09-30 | 4.0000 | 1.0000 | **4×** |
| HLT | 2017-01-04 | 3.0000 | 1.0000 | **3×** |

**Two measured corrections to the plan's numbers:**

* **"26 tickers"** is the count across Sharadar's whole 8,923-ticker universe. Inside the S&P
  500 roster this repo actually builds, **6 tickers** carry a newly de-adjusted
  `sharesOutstandingPit`. The 27 vetoed rows are mostly delisted/foreign names (SDH1, PRY1,
  NTLS, LEAF, HTZGQ, WINMQ, CDSCY, T1, HSH…).
* **Newly-failing rows are 3, not ≤1** — IVZ and OXY, both as-of-join rows. Measured, stable
  across Phase 0 and Phase 1, and set against +5,516 newly passing.

**Pre-existing test failures, NOT caused by this work**: 7 in
`tests/data_extract/sharadar/{test_sharadar_field_map,test_sharadar_merge}.py`
(`ImportError: SHARADAR_ID_COLUMNS`, `KeyError: 'JPM'`, a market-wide `actions` fixture).
Verified identical on HEAD by restoring the HEAD source files and re-running. This phase took
that count from 9 to **7** by fixing the two that were in scope.

**Also fixed to unblock the work**: `pyproject.toml` declared `[tool.ruff.lint]` **twice**,
which made *every* `pytest` invocation abort with
`Cannot declare ('tool','ruff','lint') twice`. Present in HEAD, unrelated to this plan. The two
tables are merged; no key changed value.

---

### Phase 2: Compute and store `level_factor` ✅

**Goal**: one auditable column on `cube_part_prices`, computed once.

**Changes — all done**:

1. `src/data_aggregate/utils/common/level_basis.py` (new):
   - [x] `level_factor(index, universe, yf_splits, genuine_splits) -> pd.DataFrame`, wide
   - [x] Snap `|S − 1| < 1e-12` (`LEVEL_SNAP_TOL`) to exactly `1.0`
   - [x] Both products are SUFFIX products (`_suffix_factor`: `cumprod` reversed +
         `searchsorted(side="right")`), formed right-to-left over dates sorted ascending, so
         two identical event lists cancel to **bit-exact** 1.0. `searchsorted` rather than a
         loop over events also makes it O(events + dates) instead of O(events × dates).
   - [x] `describe()` — the `StepCubePrices` log line, ranked by `|log S|`

2. `src/data_aggregate/transformers/step_cube_prices.py`:
   - [x] `_level_factor()` loads `Tables.prices_splits` and `Tables.sharadar_actions`
         (projected, filtered to `split`/`spinoff`) and calls the SAME `split_events` the
         extract layer uses
   - [x] ⚠ **Masked to `close_split.notna()`.** Not cosmetic: `S` is 1.0 everywhere and never
         NaN, so an unmasked column makes `frames_to_long`'s "drop rows where every value is
         NULL" a no-op and materialises a row for every (date, ticker) pair in the grid,
         including years before a ticker listed. **Verified additive**: 0 rows exist where
         only `level_factor` is non-null, and `level_factor` is NULL on exactly the rows
         `close_split` is.

3. `src/data_aggregate/utils/common/price_frames.py`:
   - [x] `ALL_FIELDS += ("level_factor",)`, the `PriceFrames` field, and the load wiring
   - [x] Docstrings state it is a FACTOR: multiply a LEVEL, never a RETURN

4. `tests/data_aggregate/test_level_basis.py` (new, **9 pass**):
   - [x] AAPL — `== 1.0` bit-exactly on all 8,349 dates, not `approx`
   - [x] FDX — `S = 1.241`, and 235.5036 × 265,070,592 × S = **$77.47bn** = Sharadar's
   - [x] GE — `S = 1.669297` and `1.605093`, both reproducing `before.md` to 6 dp
   - [x] HON — both sources on one date, different values → `S = 2.12229`, `1/S = 0.4712`
   - [x] MNST — identical sets → `S = 1.0`, documenting that this plan does NOT fix it
   - [x] empty sources, the snap, the log ranking, and the frame's shape/alignment

**⚠ ONE ARCHITECTURAL EXCEPTION, made deliberately.** AGENTS.md forbids cross-imports between
`src/` subfolders, and this needs `data_extract`'s `split_events`. The plan forbids forking it
("it is what stops the numerator and denominator drifting apart"). Both are right, so the rule
is bent **exactly once**, in `level_basis.py`, which re-exports it as `genuine_splits`;
`StepCubePrices` imports from `level_basis` and never names `data_extract`. The comment at the
import states why. Moving `split_events` to `src/utils/` would satisfy the letter of the rule
but drags `TranslationReport`, both registers and the whole corroboration rule with it — a
refactor of the extract layer this plan has no mandate for. **Worth raising with the user.**

**Verification**:
- [x] `"$PY" -m pytest tests/data_aggregate/test_level_basis.py -q` → 9 passed
- [x] `build-prices -F` → 7,798 dates × 491 tickers, 3,827,534 rows
- [x] `SELECT count(*) FROM cube_part_prices WHERE level_factor <> 1` → **361,996 rows
      (9.46%) across 80 tickers**. Plan estimated ≈11% / ≈72 tickers.
- [x] Purely additive (see the mask note above)

**Found while verifying — the mechanism behind the 3 newly-failing rows.** `split_events`
keeps *uncorroborated Sharadar-only* events (real but unvouched). Where Yahoo never applied
one, it lands in the denominator alone and `S` becomes `1/value`. Mostly harmless — LVS
x266 → 0.0038, VRSK x50 → 0.0200, GM/CBRE x3 → 0.3333 all sit on **pre-listing cells** and are
masked away. But **IVZ 2007-11-01 x0.5 → S = 2.0 reaches real rows**, and IVZ is exactly one
of the two tickers whose rows the fix breaks. IVZ's 2007 event was a Bermuda re-domiciliation,
not a 1:2 reverse split, so it is a **pre-existing Sharadar false positive that already
mis-de-adjusts `sharesOutstandingPit` by 2×** — `S` only makes an existing defect visible.
Already logged by the "uncorroborated, kept and warned" branch. Out of scope here; worth its
own cluster.

---

### Phase 3: Consume it ✅

**Goal**: every LEVEL is on the corrected basis; no RETURN moves.

**Changes — all done**:

1. `pit.daily_market_cap(fundamentals_history, close_split, *, level_factor)` — **required,
   keyword-only, no default** (D4). `PitFrames` takes and forwards it. The factor is
   `reindex`ed onto the market cap's own column subset and `fillna(1.0)`: a missing factor
   means "no adjustment", never "no market cap".

2. The 7 call sites, all threaded — `factors.py`, `step_cube_target.py`,
   `fundamental_features.py`, `intrinsic.py`, `institutional_features.py`,
   `superinvestor_features.py`, `insider_features.py` — plus the intermediate signatures they
   sit behind (`build_characteristics`, `_derived_fields`, `_intrinsic_fields`,
   `intrinsic_value_daily`, `build_fundamental_feature_panel`) and `_FIELDS` on all four
   transformers that read `PriceFrames`.

3. `momentum/features.py` (D3): `dollar_vol = split × volume × level_factor`, with
   `level_factor=None → 1.0` so the fingerprint harness is untouched. The module docstring's
   claim that "the split factor cancels between the two" is corrected: true for splits, FALSE
   for spinoffs, because nothing restates a share COUNT for a spinoff.

4. **The two Phase-0 questions, acted on as measured:**
   - `earnings_features` — **DISTORTED, fixed.** The price leg is multiplied by `S` before any
     yield is taken, so `fwd_eps_yield` and `forward_earnings_yield` land on the level basis.
   - `dividend_features.dividend_yield` — **legs cancel, deliberately NOT touched**, with the
     measurement recorded in the docstring so the next reader does not "fix" it.
   - ⚠ `dividend_features.py`'s inline `mcap = shares × close` **is** fixed — it is a market
     cap, not a yield, and it feeds the source-B fallback where the ex-date history misses a
     payer.

5. Tests — `tests/data_aggregate/test_level_basis.py` is now **13 tests**, the 9 factor cases
   plus 4 consumption-contract ones:
   - [x] `daily_market_cap(fund, close)` raises `TypeError` naming `level_factor`
   - [x] `S ≡ 1` is **bit-identical** to `level_factor=None` (`check_exact=True`)
   - [x] FDX 2020-12-17 → **$77.470bn**, Sharadar $77.470bn, from $62.425bn (−19.42%)
   - [x] alignment: a ticker with no factor is unchanged, a stray factor column is dropped
   - [x] the 3 existing `daily_market_cap` call sites in tests pass `level_factor=None`
   - [x] `test_price_part_roundtrip` gained a `level_factor` with a spinoff factor AND the
         non-null mask, so the mask itself round-trips

**Verification**:
- [x] `tests/data_aggregate/` → **8 failed, 214 passed**. Baseline was 9 failed; the 8 are the
      pre-existing `close=None` reaching `daily_market_cap` (5 in `test_fundamental_features`,
      1 in `test_latest_quarter_features`, both present in HEAD) and 2 pre-existing
      `test_part_registry` failures (`open`/`high`/`low` not in `ALL_FIELDS`; a
      `SimpleNamespace` missing `config_dir`). **No new failure.**
- [x] `TypeError` when `level_factor` is omitted — the contract is live
- [x] `ret` / `close_total` digests **byte-identical** before and after
      (`000ddd33daae0a38`, `13acfa1a2cd80947`)

---

### Phase 4: Validator ✅

**Goal**: `validate prices` measures the identity the cube actually builds. Read-only (D6).

**Changes — all done**:

1. `src/validate/prices.py`:
   - [x] `_level_factor_for()` computes `S(d)` **in memory** from `prices_splits` +
         `sharadar_actions` via the shared `level_basis.level_factor`. **No write, ever.**
         Deliberately NOT read from `cube_part_prices.level_factor`, for two reasons: the
         validator must run on a database whose cube is a build behind (the state it is most
         useful in), and reading the cube's own answer would make it a check that the cube
         agrees with itself instead of with the two SOURCES.
   - [x] `invariant_market_cap`: `close_split × S × sharesOutstanding / marketcap`
   - [x] `invariant_price_vintage`: `close_split × S / sharadar.price`
   - [x] `InvariantResult.raw_failed` / `.raw_share` — BOTH rates reported, so the wedge stays
         visible instead of being absorbed into a headline
   - [x] `MCAP_BLOCK_SHARE` stays **`None`**, with its comment rewritten: a gate is now a
         defensible decision rather than an impossible one, but it is a separate one
   - [x] The module docstring's "12.6% fails on a CORRECT table because Sharadar is the
         inconsistent side" paragraph **replaced** with the measured decomposition
         (leg_shares 100.00%, leg_vendor 99.82%, leg_price 87.59%) and the reusable lesson.
         `validate/cli.py`'s help text carried the same claim and is corrected too.

2. `tests/data_extract/prices/test_adjustment_basis.py`:
   - [x] `test_a_spinoff_ticker_fails_invariant_1_without_S_and_passes_with_it` — asserts BOTH
         directions (`raw_failed == 8`, `failed == 0`), because a test that only checked the
         corrected rate could not tell a fix from a widened tolerance
   - [x] The existing corruption fixture gained `level_factor = 1.0`

**Verification**:
- [x] `validate prices --skip-spike` → invariant 1 **98.30%** (target ≈98.21%), invariant 2
      **98.18%** (target ≈98.09%), each printed with `[without S(d): 87.44% / 87.33%]`
- [x] `tests/data_extract/prices/test_adjustment_basis.py` → **8 passed**
- [x] Invariant 3 unchanged — it never reads `S`

---

### Phase 5: Prove it, and write it down ✅

**Verification — every assertion met**:

| check | target | measured |
|---|---|---|
| invariant 1 | ≈98.21% | **98.30%** |
| invariant 2 | ≈98.09% | **98.18%** |
| newly-failing rows | ≤1 (OXY) | **3** (IVZ, OXY) — see below |
| FDX 2020-12-17 | $77.5bn | **$77.470bn**, Sharadar $77.470bn |
| control cohort `S` | exactly 1.0 | **`==` 1.0 on all 6**, asserted bit-exactly |
| `ret` / `close_total` digests | byte-identical | **identical** |

**Spinoff cohort vs Sharadar** — every sampled date now exact (`err_fixed` 0.00% for FDX, GE,
DD, T, HPQ, EXC, RTX at all four dates). DD was **−66.67% even after `S`** before Phase 1 and
is exact after it, which is the clearest single demonstration that the two phases had to be
ordered.

**Cross-sectional impact — what the MODEL sees.** Of 37,495 scored rows, **3,371 (8.99%)**
change size decile; **37.02%** of the 3,957 rows with `S ≠ 1`, and 5.68% of the rest (which
move only because their peers did). Larger than the plan's ~7.2% estimate. Ranked WITHIN each
`as_of` date, never pooled.

**Docs**:
- [x] `docs/data_schema.md` — the `level_factor` column, and `prices_splits`' entry corrected
      (it said "NOT a market-cap input"; it now IS one, as `level_factor`'s numerator)
- [x] `src/validate/README.md` — the corrected invariant-1 story, with the leg table
- [x] `reports/2026-09-01/price-shares-basis-fix__DATA.md` — the "87.4% is the ceiling /
      Sharadar is the inconsistent side" finding marked **SUPERSEDED**, separating what stood
      (the decomposition) from what was wrong (the conclusion drawn from it)
- [x] This plan

---

## OUTCOME

**Shipped.** Invariant 1 **87.44% → 98.30%**, invariant 2 **87.33% → 98.18%**; 5,516 rows
fixed against 3 broken; returns byte-identical; the control cohort bit-identical.

**Deviations from the plan, all deliberate and all measured:**

1. **The dividend leg needed no fix and the earnings leg did** — the opposite of the plan's
   ordering of suspicion. Both were settled by an A/B test against Sharadar's own level-basis
   yield on the `|S−1| > 10%` rows.
2. **A consumer the plan missed**: `dividend_features.py`'s inline market cap.
3. **Two tests the plan missed** asserted the reversed Phase-1 behaviour, in
   `test_sharadar_field_map.py`.
4. **The CNX review log** is scoped to the rows the veto used to drop, not to every odd-shaped
   ratio — the generic rule printed ~300 lines of legitimate 3:2 splits.
5. **"26 tickers"** is universe-wide; **6** are in this repo's roster.
6. **3 newly-failing rows, not ≤1.** IVZ and OXY. Traced: IVZ's uncorroborated Sharadar-only
   `0.5` on 2007-11-01 was a Bermuda re-domiciliation, not a reverse split, so `S` reads 2.0
   where it should read 1.0. **Pre-existing** — that same false positive already mis-de-adjusts
   IVZ's `sharesOutstandingPit` by 2×; `S` only makes it visible.

**One architectural exception to raise with the user**: `level_basis.py` imports
`split_events` from `data_extract`, crossing a subfolder boundary AGENTS.md forbids. Confined
to one documented line. The alternative — moving `split_events` to `src/utils/` — drags
`TranslationReport`, both registers and the corroboration rule with it.

**Also fixed to unblock the work**: `pyproject.toml` declared `[tool.ruff.lint]` twice, which
aborted **every** `pytest` run. Present in HEAD, unrelated to this plan.

**Not done, and the user's call:**

* **The cube and label rebuild.** `build-prices -F` has run (the `level_factor` column is
  live), but `build-fundamentals` / `build-momentum` / `build-target` / `build-extras` have
  not, so no downstream feature carries the corrected basis yet. Expect the mcap-derived
  families to move and `label.*` / `panel.betas` NOT to.
* **The model re-baseline** after that rebuild. 8.99% of rows changing size decile is enough
  to move a fit.
* **Two clusters this work surfaced but did not close**: IVZ's false-positive split event, and
  the largest un-named residuals **IP (102 rows, ratio 0.9339)** and **JCI (81, 1.3253)** —
  present before and after, the mirror image of the case `S` fixes (Sharadar restated, Yahoo
  did not).

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

- [x] invariant 1 ≥ 98% (**98.30%**), invariant 2 ≥ 98% (**98.18%**)
- [~] ≤1 newly-failing row → **3** (IVZ ×2, OXY), against 5,516 newly passing. Traced to a
      PRE-EXISTING Sharadar false positive on IVZ, not to this change.
- [x] Control cohort `S` exactly 1.0 on all 6; `ret` / `close_total` digests byte-identical
- [x] FDX 2020-12-17 `marketCap` = **$77.470bn** (Sharadar $77.470bn); was $62.425bn
- [x] `sharesOutstandingPit` carries the reverse split: HON 2×, DD 3×, MSI 7×, LDOS 4×, HLT 3×
- [x] `daily_market_cap` cannot be called without stating a basis (required keyword-only)
- [x] `level_factor` queryable in `cube_part_prices` (361,996 rows ≠ 1, 80 tickers)
- [x] No new test failure against the recorded baseline (`tests/data_aggregate/` 9 → 8)
- [~] The residual is the four named clusters — MNST 122, V 74, APA/SJM/HBAN/ORCL 79 are all
      present and unchanged, but **IP (102 rows) and JCI (81) are larger than two of them**
      and the plan never named them. Not new (both sit in the Phase-0 residual too) and not a
      regression, but the plan's enumeration was incomplete.

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
