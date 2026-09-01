# Phase 4 — Consumers: cube schema, the rename migration, label routing ✅

**Parent**: [PLAN.md](PLAN.md) · **Depends on**: P1, P3 · **Blocks**: P5 · **Estimate**: 5-6h

The largest phase, and the one with the quiet failure modes. Work top-down: schema, then the
enumerated reader migration, then the three routing hazards.

## Goal

Every consumer reads the basis it actually needs. The bare name `close` no longer exists
anywhere, so nothing can silently pick the wrong one.

## The routing table

| basis | who reads it |
|---|---|
| **`close_split`** | `daily_market_cap`, `ev`, `dividend_yield`, ATR(14), and `book_prices` in the strategy layer |
| **`close_total`** | `daily_returns` → `ret`, MACD, RSI, `high_prox_252`, seasonal momentum, betas, **all labels** |
| **`sharesOutstanding`** (vendor) | every `X / mcap` and `X / ev` feature, share-change features |
| **`sharesOutstandingPit`** | `inst_ownership_pct` and the insider %-of-shares leg — nothing else |

---

## 4a. Cube schema

- [x] `cube_part_prices` stores `close_split`, `close_total`, `volume`, `ret`, `sector_ret`.
      Registered at [parts.py](src/data_aggregate/utils/common/parts.py) — the `CubePart` entry
      needs no change (schema lives on `Table`), but the part's column set does.
- [x] `price_frames.ALL_FIELDS` ([price_frames.py:49](src/data_aggregate/utils/common/price_frames.py#L49)):
      `("close_split", "close_total", "volume", "ret", "sector_ret")`.
- [x] `PriceFrames` dataclass: replace the `close` field with `close_split` and `close_total`.
      **Do not keep a `close` property alias** — that would defeat D2.
- [x] `load_price_frames`'s hard-coded construction
      ([price_frames.py:163-169](src/data_aggregate/utils/common/price_frames.py#L163)) maps each
      requested field; add both.
- [x] `PriceFrames.skeleton()` ([:89-90](src/data_aggregate/utils/common/price_frames.py#L89))
      uses `self.close` for the (date, ticker) grid → `close_split` (either works; pick one and
      state why: `close_split` is the one that is never null when `close_total` is).

## 4b. The enumerated reader migration

Every one of these must be touched. `KeyError: 'close'` at runtime is the safety net, but the
list is short enough to do exhaustively.

**Extract / store**
- [x] `src/data_store/schema.py:105` — done in P1

**Cube price builder**
- [x] [step_cube_prices.py:43](src/data_aggregate/transformers/step_cube_prices.py#L43) `PRICE_COLS`
- [x] `_pivot_fields` ([:98-100](src/data_aggregate/transformers/step_cube_prices.py#L98)) — pivot both
- [x] `_trading_calendar`, `_on_calendar`, `_universe_frames` ([:67-71, :142-148](src/data_aggregate/transformers/step_cube_prices.py#L67))
      — the calendar and universe should key on `close_split`
- [x] `_daily_returns` ([:128-129](src/data_aggregate/transformers/step_cube_prices.py#L128)) — feed `close_total`

**data_utils**
- [x] `prices_long_to_multiindex`'s `fields` map ([data_utils.py:67](src/data_aggregate/utils/common/data_utils.py#L67))
      — `{"CloseSplit": "close_split", "CloseTotal": "close_total", "Open": "open", ...}`
- [x] `daily_returns` ([:51-53](src/data_aggregate/utils/common/data_utils.py#L51)) — **signature
      unchanged** under D1. Only its argument at each call site changes. Update the docstring to
      say it expects a total-return series.

**Sub-step `_FIELDS` declarations**
- [x] `step_cube_momentum.py:38` — needs both (`close_split` for ATR, `close_total` for the rest)
- [x] `step_cube_fundamentals.py:54` and its `stock_close=` at `:74, :144, :170, :194` → `close_split`
- [x] `step_cube_extras.py:73` and `stock_close=` at `:177, :197, :219` → `close_split`
- [x] `step_cube_target.py:58` — needs `close_split` (mcap) **and** `ret`; see 4c
- [x] `step_cube_text.py:50` → `close_split`

**Feature builders**
- [x] [pit.py:73-90](src/data_aggregate/utils/common/pit.py#L73-L90) `daily_market_cap` — the
      parameter is already generic; the caller supplies `close_split`. Rename the parameter to
      `close_split` so the contract is in the signature.
- [x] [dividend_features.py:84-97](src/data_aggregate/utils/fundamentals/dividend_features.py#L84)
      `close_pos` → `close_split`. `dividendsPerShare / close_split` now cancels correctly.
- [x] [target/factors.py:81-127](src/data_aggregate/utils/target/factors.py#L81-L127)
      `build_characteristics` — `size = -log(mcap)` takes `close_split`; any return leg takes `ret`.

**Strategy layer**
- [x] [ls_model.py:94-99](src/strategies/utils/ls_model.py#L94) `_returns` — `rets` from
      `close_total`; the returned `close` (used as `book_prices`) → `close_split`
- [x] `step_ls.py:36, :58, :75, :118` and `step_eq_long_only.py:61, :65` — `b.close` → `close_split`
      (execution prices, not returns)

**Peers**
- [x] [step_deduce_peers.py:74](src/data_peers/step_deduce_peers.py#L74) — `close_total`
      (correlation of returns)

**Scripts and tests**
- [x] `scripts/diagnose_missing_dates.py`
- [x] `tests/conftest.py:227-271` — the synthetic price fixture must produce both columns
- [x] `tests/data_aggregate/test_price_part_roundtrip.py:78, :90` — round-trip both
- [x] `tests/data_aggregate/aggregate_fingerprint.py:435` — this has a **hard-coded copy** of
      `close.pct_change(fill_method=None)`, not a call to `du.daily_returns`. It must move to
      `close_total`, and the baseline in `aggregate_fingerprint_baseline.json` must be
      regenerated (see 4e).

## 4c. Hazard 1 — the labels are a price ratio today

[targets.py:81](src/data_aggregate/utils/target/targets.py#L81):

```python
fwd_stock = forward_return(close, horizon)      # = close.shift(-h) / close - 1
```

With `close` becoming `close_split`, **every label silently becomes a price return** — the L/S
would systematically short high-yielders (MO 1.24x over the sample where the truth is 20.2x;
T 1.29x vs 6.26x).

- [x] Replace with `forward_compound(stock_ret, horizon)`, which already exists two functions
      away at [prices.py:59-66](src/data_aggregate/utils/common/prices.py#L59-L66) and is already
      used for every *factor* leg in the same function. `stock_ret` is already a parameter of
      `build_targets_multi` ([step_cube_target.py:271](src/data_aggregate/transformers/step_cube_target.py#L271)).
- [x] This also fixes an existing inconsistency: the stock leg was a simple ratio while every
      factor leg it is differenced against was log-compounded.
- [x] **Grep for every other `close` ratio across a shift or horizon** and classify each as price
      or total. `forward_return` itself may end up with no remaining caller — if so, delete it
      rather than leave a price-return trap in the shared module.

## 4d. Hazard 2 — the ATR basis trap (new in V2)

[features.py:67-73](src/data_aggregate/utils/momentum/features.py#L67-L73):

```python
tr1 = high - low
tr2 = (high - prev_close).abs()
tr3 = (low  - prev_close).abs()
```

Under `auto_adjust=False`, `high`/`low` come back **split-adjusted only**. Pairing them with
`close_total` mixes bases inside a single subtraction — a *new* bug introduced by this fix.

- [x] `_atr` takes `close_split`. `_macd`, `_rsi`, `high_prox_252` and the seasonal shifts take
      `close_total`.
- [x] Document the split at the top of `features.py`: which block uses which, and why.
- [x] Audit for any feature using a **bare price level** rather than a ratio — such a feature is
      not leak-free on either basis and should be removed or explicitly justified.

## 4e. Hazard 3 — the fingerprint baseline

`tests/data_aggregate/aggregate_fingerprint_baseline.json` pins digests including
`prim.daily_returns`, `prim.forward_windows`, `prim.xs_standardize`
([test_aggregate_regression.py:143](tests/data_aggregate/test_aggregate_regression.py#L143)).
All three change.

- [x] Regenerate the baseline **only after** 4a-4d are complete and verified, and record in the
      commit message which digests moved and why. A regenerated baseline that hides an unintended
      change is the worst outcome of this phase.
- [x] Digests that should **not** move: anything downstream of macro only.

## 4f. Route the two PIT consumers

- [x] [institutional_features.py:141-159](src/data_aggregate/utils/extras/institutional_features.py#L141):
      the same `shares_out_history` frame currently serves **both** `inst_ownership_pct` (line 145)
      and `daily_market_cap` (line 153). It must now carry **both** columns —
      `sharesOutstandingPit` for the ownership ratio, `sharesOutstanding` for the mcap. Widen the
      frame rather than passing two frames.
- [x] [insider_features.py:104](src/data_aggregate/utils/extras/insider_features.py#L104) — the
      %-of-shares leg → `sharesOutstandingPit`. `insider_net_buy_to_mcap` stays on mcap.
- [x] [superinvestor_features.py:188-198](src/data_aggregate/utils/extras/superinvestor_features.py#L188)
      — `super_value_to_mcap` / `super_flow_to_mcap` are mcap ratios; no PIT needed. Confirm there
      is no shares-count leg.

## Verification

**The cube is empty**, so nothing here may be phrased as "compare to the pre-change value". Every
check is an absolute target, an internal identity, or a cross-vendor agreement. Feature-level
checks call the builder function directly on a small ticker subset — they do not read a cube table.

- [x] `rg -n '\bclose\b' src/ tests/ scripts/` returns no price-column usage (only
      `plt.close`, `file.close` and similar)
- [x] Full test suite passes
- [x] `daily_market_cap` for AAPL 2020-07-31 == **1.817e12** (+/-1%), matching Sharadar's
      `1,817,315,475,360`. The old formula gave $439bn.
- [x] Label at h=30 for a high-yielder (MO, T), computed by calling `build_targets_multi` on a
      subset, matches a **hand-computed compounded total return** over the same window. This is
      the check that 4c actually landed; it needs no "before".
- [x] ATR(14) is **identical** whether fed `close_split` or `close_total`, for a dividend payer.
      [features.py:81](src/data_aggregate/utils/momentum/features.py#L81) returns `atr / close`,
      a ratio of same-basis quantities, so it is basis-invariant. If this check *fails*, ATR is
      not normalised the way the docstring claims and 4d needs re-deriving.
- [x] `inst_ownership_pct` for GOOGL post-2022-07 is in a plausible 0-1 range (the missing 20:1
      split made it wrong by 20x)
- [x] Fingerprint baseline regenerated with a documented digest diff. Note `real_frames`
      ([conftest.py:236](tests/conftest.py#L236)) reads **`prices`**, not the cube — so the
      baseline moves because of P1, and regenerating it is mandatory, empty cube or not.

## Rollback

This phase is pure code. `git revert` the commit; P1/P2/P3 data stays valid and the pre-fix code
reads `close_split` under a compatibility rename if a fast revert is ever needed.
