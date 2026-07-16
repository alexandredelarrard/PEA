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
  * lottery    : MAX (extreme recent daily return) + return skewness (overpaid upside)
  * downside   : downside semi-deviation + idiosyncratic vol (low-vol anomaly)
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
    volume: pd.DataFrame | None = None,
    seasonal_horizons: list[int] | None = None,
    seasonal_years: int = 5,
) -> dict:
    """
    Compute raw (un-standardized) feature frames. Returns dict:
        {feature_name: DataFrame[date x ticker]}

    `high`/`low` are only needed for ATR(14); if absent, ATR is skipped and
    every other feature is still produced. `volume` enables the liquidity family
    (dollar volume, Amihud illiquidity, relative volume); if absent it is skipped.
    `seasonal_horizons` (e.g. the target horizons) enables the cross-sectional
    seasonality feature `seasonal_h<h>` per horizon (averaged over the last
    `seasonal_years` prior years); if absent it is skipped.
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

    # ---- Higher-moment / lottery / idiosyncratic-risk anomalies (price-only) ----
    # All point-in-time (trailing windows) and orthogonal-ish to the fundamentals.
    # MAX effect (Bali, Cakici, Whitelaw 2011): stocks with an extreme recent max
    # daily return underperform (lottery demand -> overpriced).
    feats["max_21"] = _safe(ret.rolling(21).max())
    # Return skewness (Boyer-Mitton-Vorkink): high positive skew underperforms
    # (investors overpay for lottery-like upside).
    feats["ret_skew_126"] = _safe(ret.rolling(126).skew())
    # Downside semi-deviation: std of only the negative daily returns (63d).
    neg = ret.where(ret < 0)
    feats["downside_vol_63"] = _safe(neg.rolling(63, min_periods=20).std())
    # Idiosyncratic volatility: vol of the market-relative return (stock minus the
    # equal-weight universe move) over 63d -> the low-idio-vol anomaly.
    mkt = ret.mean(axis=1)
    feats["idio_vol_63"] = _safe(ret.sub(mkt, axis=0).rolling(63).std())

    # ---- Liquidity / volume (point-in-time trailing windows; skipped w/o volume) ----
    if volume is not None:
        volume = volume.reindex_like(close)
        dollar_vol = close * volume                        # daily $ traded
        # Liquidity/size proxy: log average daily dollar volume (63d).
        feats["dollar_volume_63"] = _safe(
            np.log1p(dollar_vol.rolling(63, min_periods=20).mean()))
        # Amihud (2002) illiquidity = mean(|ret| / $volume). HIGHER = more illiquid
        # (illiquidity premium). Scale is irrelevant post cross-sectional ranking.
        amihud = _safe(ret.abs() / dollar_vol.where(dollar_vol > 0))
        feats["amihud_63"] = amihud.rolling(63, min_periods=20).mean()
        # Relative volume: recent 5d vs 63d average -> volume spike / attention.
        v5 = volume.rolling(5, min_periods=3).mean()
        v63 = volume.rolling(63, min_periods=20).mean()
        feats["rel_volume_5_63"] = _safe(v5 / v63.where(v63 > 0))

    # ---- Cross-sectional SEASONALITY at the forecast target t+h (Heston-Sadka) ----
    # A calendar dummy (month of t+h) is identical for every stock on a date -> it
    # has NO cross-sectional dispersion and cannot help a market-neutral ranker.
    # The cross-sectionally useful seasonal signal is the STOCK'S OWN average return
    # over the SAME calendar window in PRIOR years: seasonal_h(t) = mean over the
    # last few years y>=1 of the h-day forward return at t-252*y. Because only
    # y>=1 (>= a year back, fully realized) is used, it is strictly leak-free, and
    # it differs per stock (some names have real same-season repeatability).
    if seasonal_horizons:
        logr = np.log1p(ret.clip(lower=-0.999999))
        for h in sorted({int(x) for x in seasonal_horizons}):
            mp = max(1, int(round(h * 0.6)))
            fwd_h = np.expm1(logr[::-1].rolling(h, min_periods=mp).sum()[::-1].shift(-1))
            prior = np.stack([fwd_h.shift(252 * y).to_numpy() for y in range(1, seasonal_years + 1)])
            finite = np.isfinite(prior)
            cnt = finite.sum(axis=0)
            ssum = np.where(finite, prior, 0.0).sum(axis=0)
            seasonal = np.where(cnt > 0, ssum / np.maximum(cnt, 1), np.nan)
            feats[f"seasonal_h{h}"] = _safe(
                pd.DataFrame(seasonal, index=close.index, columns=close.columns))

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
    volume: pd.DataFrame | None = None,
    seasonal_horizons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Build a long-format feature panel ready for modeling.

    Returns a tidy DataFrame with columns:
        ['date', 'ticker', <feature_1>, <feature_2>, ...]
    Each feature already cross-sectionally standardized within its date.
    `high`/`low` enable the ATR(14) feature; `volume` enables the liquidity family;
    `seasonal_horizons` enables the per-horizon cross-sectional seasonality feature.
    """
    raw = compute_raw_features(close, open_, sector_returns, high=high, low=low,
                               volume=volume, seasonal_horizons=seasonal_horizons)
    std = {name: cross_sectional_standardize(f, method) for name, f in raw.items()}

    long_frames = []
    for name, f in std.items():
        s = f.stack()
        s.index.set_names(["date", "ticker"], inplace=True)
        long_frames.append(s.rename(name))

    panel = pd.concat(long_frames, axis=1).reset_index()
    return panel
