"""
features.py
-----------
Price-only alphas (you have close + open, no volume yet). EVERY feature is
point-in-time: computed from data up to and including t, never forward. The
label is the only forward object in the pipeline.

Features are then standardized CROSS-SECTIONALLY (ranked within each day) so the
model learns relative signals and days are comparable regardless of market vol.

Signal families included:
  * momentum   : 12-1 momentum (skip most recent month)
  * reversal   : short-term 5d / 21d reversal
  * volatility : trailing realized vol (low-vol tends to outperform, risk-adj)
  * trend      : close vs moving averages (50 / 200)
  * high_prox  : proximity to trailing 52-week high
  * gap        : average overnight gap (open vs prior close)
  * range      : average intraday range proxy (|close-open|/open)
  * peer_mom   : stock cum return minus its sector cum return (residual mom)
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def _safe(df):
    return df.replace([np.inf, -np.inf], np.nan)


def compute_raw_features(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    sector_returns: pd.DataFrame,
) -> dict:
    """
    Compute raw (un-standardized) feature frames. Returns dict:
        {feature_name: DataFrame[date x ticker]}
    """
    ret = close.pct_change(fill_method=None)
    feats = {}

    # 12-1 momentum: cumulative return from t-252 to t-21 (skip last month).
    feats["mom_12_1"] = _safe(close.shift(21) / close.shift(252) - 1.0)

    # Short-term reversal (negated: recent losers tend to bounce).
    feats["rev_5"] = _safe(-(close / close.shift(5) - 1.0))
    feats["rev_21"] = _safe(-(close / close.shift(21) - 1.0))

    # Trailing realized volatility (annualized-ish; scale irrelevant post-rank).
    feats["vol_21"] = _safe(ret.rolling(21).std())
    feats["vol_63"] = _safe(ret.rolling(63).std())

    # Trend: distance from moving averages.
    feats["ma_ratio_50"] = _safe(close / close.rolling(50).mean() - 1.0)
    feats["ma_ratio_200"] = _safe(close / close.rolling(200).mean() - 1.0)

    # Proximity to trailing 52-week high (near-high names show continuation).
    feats["high_prox_252"] = _safe(close / close.rolling(252).max())

    # Overnight gap: average of (open_t / close_{t-1} - 1) over 21d.
    gap = _safe(open_ / close.shift(1) - 1.0)
    feats["gap_21"] = gap.rolling(21).mean()

    # Intraday range proxy from close/open.
    rng = _safe((close - open_).abs() / open_)
    feats["range_21"] = rng.rolling(21).mean()

    # Peer-relative (residual) momentum: 63d stock cum ret minus sector cum ret.
    stock_cum = _safe(close / close.shift(63) - 1.0)
    sector_cum = _safe((1.0 + sector_returns).rolling(63).apply(np.prod, raw=True) - 1.0)
    feats["peer_mom_63"] = _safe(stock_cum - sector_cum)

    return feats


def cross_sectional_standardize(feat: pd.DataFrame, method: str = "rank") -> pd.DataFrame:
    """
    Standardize one feature within each day (across stocks).
      'rank'   -> percentile in [0,1] (robust to outliers)
      'zscore' -> demean/divide by cross-sectional std, clipped at +/-3
    """
    if method == "rank":
        return feat.rank(axis=1, pct=True, method="average")
    elif method == "zscore":
        mu = feat.mean(axis=1)
        sd = feat.std(axis=1)
        z = feat.sub(mu, axis=0).div(sd, axis=0)
        return z.clip(-3, 3)
    raise ValueError("method must be 'rank' or 'zscore'")


def build_feature_panel(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    sector_returns: pd.DataFrame,
    method: str = "rank",
) -> pd.DataFrame:
    """
    Build a long-format feature panel ready for modeling.

    Returns a tidy DataFrame with columns:
        ['date', 'ticker', <feature_1>, <feature_2>, ...]
    Each feature already cross-sectionally standardized within its date.
    """
    raw = compute_raw_features(close, open_, sector_returns)
    std = {name: cross_sectional_standardize(f, method) for name, f in raw.items()}

    long_frames = []
    for name, f in std.items():
        s = f.stack()
        s.index.set_names(["date", "ticker"], inplace=True)
        long_frames.append(s.rename(name))

    panel = pd.concat(long_frames, axis=1).reset_index()
    return panel
