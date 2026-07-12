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
  * technicals : MACD line + histogram, RSI(14), ATR(14) -- see below

Technical indicators (MACD / RSI / ATR) are computed on the price series and
then LAGGED ONE DAY (`.shift(1)`): the value on date t is built purely from
prices up to and including t-1, EXCLUDING t itself, so the indicator can never
peek at the close it is being lined up against. MACD and ATR are divided by the
close so they are comparable across stocks of different price levels before the
cross-sectional ranking.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def _safe(df):
    return df.replace([np.inf, -np.inf], np.nan)


def _rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """Wilder's RSI(n) per ticker. 100 when there are only gains in the window."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    rs = avg_gain / avg_loss.where(avg_loss > 0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # all-gain window -> avg_loss 0 -> RSI defined as 100
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    return _safe(rsi)


def _macd(close: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line and histogram, each normalized by close for cross-sectional
    comparability. Returns (macd_norm, hist_norm)."""
    ema_fast = close.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    hist = macd_line - signal_line
    denom = close.where(close > 0)
    return _safe(macd_line / denom), _safe(hist / denom)


def _atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """Wilder's ATR(n) as a fraction of close (ATR%). Uses the true range
    max(H-L, |H-Cprev|, |L-Cprev|)."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.DataFrame(
        np.maximum(np.maximum(tr1.to_numpy(), tr2.to_numpy()), tr3.to_numpy()),
        index=close.index, columns=close.columns,
    )
    atr = true_range.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    denom = close.where(close > 0)
    return _safe(atr / denom)


def compute_raw_features(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    sector_returns: pd.DataFrame,
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
) -> dict:
    """
    Compute raw (un-standardized) feature frames. Returns dict:
        {feature_name: DataFrame[date x ticker]}

    `high`/`low` are only needed for ATR(14); if absent, ATR is skipped and
    every other feature is still produced.
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

    # ---- Technical indicators, LAGGED one day (exclude t -> no leakage) ----
    macd_norm, macd_hist = _macd(close)
    feats["macd"] = macd_norm.shift(1)
    feats["macd_hist"] = macd_hist.shift(1)
    feats["rsi_14"] = _rsi(close, 14).shift(1)
    if high is not None and low is not None:
        feats["atr_14"] = _atr(high, low, close, 14).shift(1)

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
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a long-format feature panel ready for modeling.

    Returns a tidy DataFrame with columns:
        ['date', 'ticker', <feature_1>, <feature_2>, ...]
    Each feature already cross-sectionally standardized within its date.
    `high`/`low` enable the ATR(14) feature.
    """
    raw = compute_raw_features(close, open_, sector_returns, high=high, low=low)
    std = {name: cross_sectional_standardize(f, method) for name, f in raw.items()}

    long_frames = []
    for name, f in std.items():
        s = f.stack()
        s.index.set_names(["date", "ticker"], inplace=True)
        long_frames.append(s.rename(name))

    panel = pd.concat(long_frames, axis=1).reset_index()
    return panel
