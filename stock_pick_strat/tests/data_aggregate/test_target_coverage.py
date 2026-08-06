"""
Target COVERAGE: when is the label defined, and when must it legitimately be missing?

The rule a forward-looking label has to obey: at date t the target describes the move
from t to t+horizon, so it is defined for every t whose t+horizon has already happened,
and undefined ONLY for the last `horizon` rows. A leading gap is acceptable solely to the
extent the factor-neutral residual needs betas, i.e. `min_obs` observations.

The bug these tests pin: `estimate_betas_for_stock` used to `.dropna()` across ALL
regressors jointly, so the momentum style factor -- close.shift(21)/close.shift(252),
hence NaN for its first 252 trading days -- discarded the whole head of the sample for
the market and sector betas as well. The first beta landed at ~252 + 40 = ~291 trading
days, `compute_epsilon` propagated the NaN beta, and the live cube therefore had NO
target at all from 2011-07-18 to 2012-09-12 -- ~14 months and ~123k rows of training
data lost to a join, not to economics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.target.betas import estimate_all_betas, estimate_betas_for_stock
from src.data_aggregate.utils.target.factors import momentum_characteristic
from src.data_aggregate.utils.target.targets import build_targets_multi, forward_return

START = "2011-07-18"          # the live cube's first price date
N_DAYS = 900
MIN_OBS = 40                  # estimate_betas_for_stock default
SEED = 11


def _market() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    idx = pd.bdate_range(START, periods=N_DAYS)
    tickers = [f"T{i}" for i in range(40)]
    rng = np.random.default_rng(SEED)
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, (len(idx), len(tickers))), axis=0)),
        index=idx, columns=tickers)
    return close, close.pct_change(), close.pct_change().rolling(5).mean().bfill(), tickers


def _factor_panel(close: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    """Market (available from day 1) + the momentum style factor (252-day warm-up)."""
    return pd.DataFrame({"market": rets.mean(axis=1),
                         "momentum": momentum_characteristic(close).mean(axis=1)})


# --------------------------------------------------------------------------- #
# 1. the root cause                                                            #
# --------------------------------------------------------------------------- #
def test_slow_warming_factor_does_not_delay_the_other_betas():
    """A regressor with a long warm-up must not postpone the betas that are already
    estimable. Market + sector need `min_obs` rows; momentum needs 252. Before the fix
    the presence of momentum in the shared block pushed ALL of them to ~291."""
    close, rets, sector_ret, tickers = _market()
    y, sector = rets[tickers[0]], sector_ret[tickers[0]]
    market_only = pd.DataFrame({"market": rets.mean(axis=1)})
    with_slow = _factor_panel(close, rets)

    first_alone = estimate_betas_for_stock(y, market_only, sector)["beta_market"].dropna().index[0]
    first_mixed = estimate_betas_for_stock(y, with_slow, sector)["beta_market"].dropna().index[0]
    idx = close.index
    pos_alone, pos_mixed = idx.get_loc(first_alone), idx.get_loc(first_mixed)

    assert pos_mixed == pos_alone, (
        f"the momentum factor delayed the market beta: alone +{pos_alone} rows, "
        f"with momentum +{pos_mixed} rows")
    assert pos_mixed <= MIN_OBS + 5, f"market beta starts at +{pos_mixed}, expected ~{MIN_OBS}"

    # and momentum's own beta is simply 0 (not neutralized) while it is unusable,
    # rather than poisoning the row
    b = estimate_betas_for_stock(y, with_slow, sector)
    early = b.iloc[pos_mixed:pos_mixed + 5]
    assert early["beta_momentum"].abs().max() == 0.0, \
        "an unusable factor must get beta 0 (not neutralized), not NaN"

    print(f"\n[1] market beta first non-NaN: +{pos_alone} rows alone, +{pos_mixed} rows with a "
          f"252-day-warm-up factor alongside (was +291 before the fix)")
    print("    SANITY CHECK: the beta join no longer lets one slow factor blank the head "
          "of the sample; an unusable factor gets beta 0 = simply not neutralized there.")


# --------------------------------------------------------------------------- #
# 2. the user-visible contract                                                 #
# --------------------------------------------------------------------------- #
def test_target_is_defined_from_the_beta_warmup_and_missing_only_the_last_horizon():
    close, rets, sector_ret, tickers = _market()
    idx = close.index
    factor_panel = _factor_panel(close, rets)
    peers = {t: {p: 1.0 for p in tickers if p != t} for t in tickers}
    betas = estimate_all_betas(rets, factor_panel, sector_ret)
    horizons = (30, 60, 90)

    built = build_targets_multi(close, rets, peers, betas, factor_panel, macro_cols=[],
                               horizons=horizons, labels=("rank",),
                               sector_groups={"sector": {t: "S" for t in tickers}})

    rows = []
    for h in horizons:
        nn = built[h]["rank"].dropna(how="all")
        lead = idx.get_loc(nn.index[0])
        trail = len(idx) - 1 - idx.get_loc(nn.index[-1])
        # the tail must be EXACTLY the horizon: t+h has not happened yet, nothing else
        assert trail == h, f"h={h}: trailing gap {trail} rows, expected exactly {h}"
        # the head must be the beta warm-up, not the 252-day style-factor warm-up
        assert lead <= MIN_OBS + 5, f"h={h}: leading gap {lead} rows, expected ~{MIN_OBS}"
        rows.append((h, lead, trail))

    print(f"\n[2] {'h':>4} {'leading gap':>12} {'trailing gap':>13}")
    for h, lead, trail in rows:
        print(f"    {h:>4} {lead:>12} {trail:>13}")
    print(f"    SANITY CHECK: every horizon is defined from +{rows[0][1]} rows (the beta "
          "warm-up) and missing exactly its own horizon at the end -- so only the "
          "un-realised future is blank, which is what the label must do.")


def test_forward_return_is_the_move_from_t_to_t_plus_horizon():
    """The economic definition, pinned: the raw forward return at t is
    close[t+h]/close[t] - 1 -- 'today's stock vs the stock in h days'."""
    close, *_ = _market()
    h = 30
    fwd = forward_return(close, h)
    t = close.index[100]
    t_plus_h = close.index[130]
    tkr = close.columns[0]

    expected = close.loc[t_plus_h, tkr] / close.loc[t, tkr] - 1.0
    assert np.isclose(fwd.loc[t, tkr], expected), "forward_return is not close[t+h]/close[t]-1"
    # defined at t, and undefined only once t+h runs off the end
    assert fwd[tkr].iloc[: -h].notna().all(), "raw forward return has an interior gap"
    assert fwd[tkr].iloc[-h:].isna().all(), "raw forward return leaks past the last price"

    print(f"\n[3] h={h}: forward_return at {t.date()} = "
          f"{fwd.loc[t, tkr]:+.4%} = close[{t_plus_h.date()}]/close[{t.date()}]-1")
    print(f"    non-null for all but the final {h} rows")
    print("    SANITY CHECK: the raw forward return has NO leading gap at all -- any "
          "leading gap in the final target comes purely from the residualization's "
          "beta warm-up, which is why that warm-up must stay as short as it can be.")
