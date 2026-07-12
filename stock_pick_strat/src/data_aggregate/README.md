# qalpha — cross-sectional forward-residual signal pipeline

Builds the target and model we discussed: predict each stock's **cross-sectionally
ranked forward residual return** (market + sector stripped out) from price-based
alphas, using an XGBoost learning-to-rank model. The output is a daily per-stock
score to feed into a portfolio optimizer.

## Modules

| file | role |
|------|------|
| `data_utils.py`   | normalize a yfinance download into wide close/open matrices; daily returns |
| `sector_peers.py` | build each stock's peer basket from return correlation → per-stock "sector" return |
| `betas.py`        | rolling market & (orthogonalized) sector betas, with shrinkage |
| `targets.py`      | forward returns → residual `epsilon` → cross-sectional rank label (per horizon) |
| `features.py`     | price-only alphas (momentum, reversal, vol, trend, gap, peer-momentum), cross-sectionally ranked |
| `model.py`        | modeling panel, **purged/embargoed** walk-forward CV, XGBoost ranker, IC scoring |
| `run_pipeline.py` | end-to-end glue on real data |
| `smoke_test.py`   | synthetic-data verification of shapes, timing, and residualization |

## The target (recap)

```
eps_i(t) = r_i(t→t+h) − beta_m_i(t)·r_M(t→t+h) − beta_s_i(t)·r_S_orth(t→t+h)
label_i(t) = cross-sectional percentile rank of eps_i within day t
```

- **betas are point-in-time** (trailing window ending at t); they are only
  *applied* to forward returns. Never fit on the forward window — that leaks.
- **sector is orthogonalized against market** (via `gamma`) so the two betas
  decouple and the residual doesn't carry market exposure.
- **rank is scale-free**, so there is no "which std / which day" ambiguity, and
  it pairs with the ranking objective.

## Verified on synthetic data

`smoke_test.py` plants a known market + sector + AR-idiosyncratic structure:
- estimated market beta ≈ 1.03 (true ≈ 1.0)
- corr(epsilon, forward market) ≈ 0.00, corr(epsilon, forward idio) ≈ 0.89
  → market/sector correctly stripped
- purged-CV IC recovers the planted residual-momentum signal cleanly
  (high only because it's a noiseless toy; real IC of 0.03–0.05 is good)

## Important caveats before trusting a backtest

1. **Peer look-ahead.** `build_peer_dict` on full history uses the future to
   define peers. For an honest backtest use `build_peer_dict_rolling` recomputed
   on a schedule (monthly), so peers at t use only data ≤ t.
2. **Survivorship bias.** yfinance current-constituent lists exclude delisted
   names. Use point-in-time index membership or expect inflated returns.
3. **Embargo ≥ horizon.** Overlapping t→t+h labels are autocorrelated; the CV
   embargo must be ≥ your longest label horizon or IC is leaked-optimistic.
4. **Costs decide the horizon.** Re-evaluate each horizon net of realistic
   spread/slippage; the 1–5 day horizons often die net of cost.

## Next stage (not in this repo)

The `signal` is an expected-return proxy, **not** a portfolio. Feed it to an
optimizer: maximize `α'w − λ·w'Σw − cost(w − w_prev)` under β≈0, sector≈0,
style-factor≈0, position/ADV caps, trading a partial step toward the target
each day (Gârleanu–Pedersen aim-portfolio logic).

## Install

```
pip install yfinance xgboost pandas numpy scipy scikit-learn pyarrow
```
