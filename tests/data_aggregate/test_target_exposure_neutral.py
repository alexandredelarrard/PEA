"""The target must not be predictable from a name's RISK EXPOSURES.

Subtracting `beta * factor` does not achieve that on its own, for two reasons no better beta
can fix: the hedge is a noisy forecast of the window it hedges, and a beta-hedged high-beta
name genuinely under-earns in a rising market. Measured on the live panel, a signal built from
nothing but a name's market beta earned a rank-IC of +0.073 (t +10.3) against the label -- free
IC a model would learn and the L/S optimizer would then neutralize back out.

These tests pin the fix and the two contracts it must not break:

  1. Exposure-neutral   -> a signal built ONLY from an exposure earns ~0 IC against the label,
                           and the per-day cross-sectional slope on it is ~0.
  2. Transformed, then  -> the projection runs on the RANK/ZSCORE label, not on epsilon,
     projected             because those transforms are non-linear.
  3. Scale preserved    -> `target_rank` stays a [0,1] percentile centred at ~0.5 and
                           `target_zscore` stays mean 0 / sd 1 / |z| <= 3. Downstream depends
                           on both (`model._graded_labels` multiplies the rank by 30).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.xs import xs_rank_pct
from src.data_aggregate.utils.target.targets import build_targets_multi

HORIZON = 10


def _panel_with_beta_tilt(n_dates: int = 400, n_tickers: int = 60, seed: int = 7):
    """A panel whose stocks load on the market with DISPERSED, KNOWN betas, on a market that
    drifts up -- the exact configuration that leaves a beta tilt in a hedged label."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_dates)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]

    market = pd.Series(rng.normal(0.0006, 0.010, n_dates), index=dates)   # upward drift
    true_beta = pd.Series(np.linspace(0.4, 1.8, n_tickers), index=tickers)
    idio = pd.DataFrame(rng.normal(0, 0.012, (n_dates, n_tickers)),
                        index=dates, columns=tickers)
    stock_ret = idio.add(pd.DataFrame(np.outer(market.to_numpy(), true_beta.to_numpy()),
                                      index=dates, columns=tickers))
    close = (1 + stock_ret).cumprod() * 100.0

    factor_panel = pd.DataFrame({"market": market})
    # the fitted loadings the label is hedged with AND projected against -- deliberately the
    # true betas here, so any surviving tilt cannot be blamed on estimation error
    betas = {t: pd.DataFrame({"beta_market": float(true_beta[t])}, index=dates)
             for t in tickers}
    sector = {t: ("EVEN" if i % 2 == 0 else "ODD") for i, t in enumerate(tickers)}
    return close, stock_ret, betas, factor_panel, {"industry_group": sector}, true_beta


def _daily_ic(signal: pd.Series, label: pd.DataFrame) -> float:
    """Mean per-day cross-sectional rank-IC of a STATIC per-ticker signal against the label."""
    ranked_signal = signal.rank(pct=True)
    ranked_label = xs_rank_pct(label)
    return float(ranked_label.corrwith(ranked_signal, axis=1).mean())


def _build(labels, **kwargs):
    close, stock_ret, betas, panel, groups, true_beta = _panel_with_beta_tilt()
    out = build_targets_multi(close, betas, panel, macro_cols=[], horizons=(HORIZON,),
                              labels=labels, min_names=20, neutralize_momentum=False,
                              sector_groups=groups, stock_ret=stock_ret, **kwargs)
    return out[HORIZON], true_beta


def test_label_is_not_predictable_from_the_exposure():
    built, true_beta = _build(("rank",))
    label = built["rank"]

    ic = _daily_ic(-true_beta, label)
    slope = label.corrwith(pd.Series(np.arange(len(true_beta)), index=true_beta.index),
                           axis=1).mean()

    assert abs(ic) < 0.02, f"-beta_market still earns free IC against the label: {ic:+.4f}"
    assert abs(slope) < 0.02, f"label still slopes on beta rank: {slope:+.4f}"

    print("\n=== SANITY CHECK: label not predictable from market beta ===")
    print(f"  mean daily rank-IC of the naive signal '-beta_market' = {ic:+.4f} (live panel "
          "measured +0.0727 before the projection)")
    print(f"  mean per-day cross-sectional corr(label, beta rank)   = {slope:+.4f}")
    print("  -> exposure alone can no longer predict the target. Validated.")


def test_projection_preserves_each_label_scale():
    built, _ = _build(("rank", "zscore", "epsilon"))
    rank, zscore = built["rank"], built["zscore"]
    populated = rank.notna().sum(axis=1) >= 20

    values = rank[populated].to_numpy()
    assert np.nanmin(values) >= 0.0 and np.nanmax(values) <= 1.0, "rank left [0,1]"
    assert abs(float(rank[populated].mean(axis=1).mean()) - 0.5) < 0.02, "rank not centred"
    assert zscore.abs().max().max() <= 3.0 + 1e-9, "zscore not winsorized to +-3"
    # mean 0 / sd 1 hold up to the +-3 WINSORIZATION, which is deliberate (XS_CLIP_LABEL) and
    # shifts both on any day where the clip binds -- hence a tolerance rather than 1e-9.
    assert zscore[populated].mean(axis=1).abs().max() < 0.02, "zscore not centred"
    assert abs(float(zscore[populated].std(axis=1).mean()) - 1.0) < 0.05, "zscore sd != 1"

    print("\n=== SANITY CHECK: the projection does not break the label contracts ===")
    print(f"  rank   in [{np.nanmin(values):.3f}, {np.nanmax(values):.3f}], "
          f"daily mean {float(rank[populated].mean(axis=1).mean()):.4f}")
    print(f"  zscore daily mean ~0, sd {float(zscore[populated].std(axis=1).mean()):.4f}, "
          f"|z|max {float(zscore.abs().max().max()):.3f}")
    print("  -> rank stays a percentile (model._graded_labels needs it), zscore stays "
          "standardized. Validated.")


def test_group_means_are_exactly_zero_per_day():
    built, _ = _build(("epsilon",))
    eps = built["epsilon"]
    _, _, _, _, groups, _ = _panel_with_beta_tilt()
    industry = groups["industry_group"]

    worst = 0.0
    for name in set(industry.values()):
        members = [t for t, g in industry.items() if g == name]
        worst = max(worst, float(eps[members].mean(axis=1).abs().max()))

    assert worst < 1e-9, f"industry group mean not zero: {worst:.2e}"
    print("\n=== SANITY CHECK: industry indicator block ===")
    print(f"  worst |per-day industry mean| of the projected epsilon = {worst:.2e}")
    print("  -> group membership cannot predict the target, exactly. Validated.")


def test_label_is_orthogonal_to_the_size_characteristic():
    """A LOADING does not span a CHARACTERISTIC, so the size characteristic gets its own
    regressor. On the live panel `beta_size` explained only R^2 0.26 of `-log(mcap)` and
    `-log_mcap` earned free rank-IC +0.0380 (t +7.4) at h=60; adding this took it to +0.0051.

    Uses `epsilon`, not `rank`: `_neutral_label` re-applies the transform AFTER projecting, and
    that transform is non-linear, so only the untransformed label is orthogonal to machine
    precision. Uses a COLUMN SUBSET because `pit.daily_market_cap` returns one -- a ticker with
    no filing history is absent, not NaN -- which is what makes the `reindex_like` in
    `_neutralizing_design` load-bearing rather than redundant. Uses `logspace` because a
    linearly-spaced $1bn-$400bn grid skews `log(mcap)` enough for XS_CLIP_CHARACTERISTIC to
    bind, which would break exact orthogonality for a legitimate reason.
    """
    close, *_ = _panel_with_beta_tilt()
    covered = close.columns[:50]                       # the other 10 have NO filing history
    mcap = close[covered].mul(np.logspace(9, 11.5, len(covered)), axis=1)

    built, _ = _build(("epsilon",), market_cap=mcap)
    corr = built["epsilon"].corrwith(np.log(mcap), axis=1)
    worst = float(corr.abs().max())

    assert worst < 1e-9, f"label still tilts on log market cap: {worst:.2e}"
    print("\n=== SANITY CHECK: label orthogonal to the size characteristic ===")
    print(f"  {len(covered)}/{len(close.columns)} names carry a market cap "
          "(the rest exercise the column-subset path that used to raise LinAlgError)")
    print(f"  worst |per-day xs corr(epsilon, log_mcap)| = {worst:.2e}")
    print("  -> size is removed exactly, not approximately. On the live panel this is the "
          "0.0380 -> 0.0051 free-IC drop at h=60. Validated.")


def test_vol_standardize_homogenises_label_magnitude():
    """P2: OFF by default, and its job is magnitude -- names' epsilon dispersion should stop
    being a volatility artefact once it is on."""
    plain, _ = _build(("epsilon",))
    scaled, _ = _build(("epsilon",), vol_standardize=True)

    _, stock_ret, _, _, _, _ = _panel_with_beta_tilt()
    vol = stock_ret.std()
    spread_plain = float(plain["epsilon"].std().corr(vol))
    spread_scaled = float(scaled["epsilon"].std().corr(vol))

    assert spread_plain > spread_scaled, "vol standardization did not reduce the vol coupling"
    print("\n=== SANITY CHECK: vol_standardize ===")
    print(f"  corr(sd(epsilon_i), vol_i): plain={spread_plain:+.3f} -> "
          f"standardized={spread_scaled:+.3f} (live panel measured +0.898 plain)")
    print("  -> the label's magnitude stops being a volatility artefact. Validated.")
