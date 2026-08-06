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

from src.data_aggregate.utils.common.frames import sanitize
from src.data_aggregate.utils.common.prices import forward_compound, momentum_characteristic, trailing_vol
from src.data_aggregate.utils.common.xs import xs_standardize


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
    return sanitize(rsi)


def _macd(close: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line and histogram, each normalized by close for cross-sectional
    comparability. Returns (macd_norm, hist_norm)."""
    ema_fast = close.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    hist = macd_line - signal_line
    denom = close.where(close > 0)
    return sanitize(macd_line / denom), sanitize(hist / denom)


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
    return sanitize(atr / denom)


def compute_raw_features(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    sector_returns: pd.DataFrame,
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
    seasonal_horizons: list[int] | None = None,
    seasonal_years: int = 5,
    *,
    returns: pd.DataFrame | None = None,
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

    `returns` lets the caller pass daily returns it ALREADY has instead of having them
    re-derived here -- `du.daily_returns` is literally `close.pct_change(fill_method=None)`,
    and the cube's price step persists that frame, so recomputing it was pure duplication.
    KEYWORD-ONLY so the existing positional call sites are untouched. On an incrementally
    trimmed window the passed frame is also strictly better: a recompute would return NaN on
    the window's first row where the full build had a value.
    """
    ret = close.pct_change(fill_method=None) if returns is None else returns.reindex_like(close)
    feats = {}

    # 12-1 momentum: cumulative return from t-252 to t-21 (skip last month).
    feats["mom_12_1"] = sanitize(momentum_characteristic(close))

    # Short-term reversal (negated: recent losers tend to bounce).
    feats["rev_5"] = sanitize(-(close / close.shift(5) - 1.0))
    feats["rev_21"] = sanitize(-(close / close.shift(21) - 1.0))

    # Trailing realized volatility (annualized-ish; scale irrelevant post-rank).
    feats["vol_21"] = sanitize(trailing_vol(ret, 21))
    feats["vol_63"] = sanitize(trailing_vol(ret, 63))

    # Trend: distance from moving averages.
    feats["ma_ratio_50"] = sanitize(close / close.rolling(50).mean() - 1.0)
    feats["ma_ratio_200"] = sanitize(close / close.rolling(200).mean() - 1.0)

    # Proximity to trailing 52-week high (near-high names show continuation).
    feats["high_prox_252"] = sanitize(close / close.rolling(252).max())

    # Overnight gap: average of (open_t / close_{t-1} - 1) over 21d.
    gap = sanitize(open_ / close.shift(1) - 1.0)
    feats["gap_21"] = gap.rolling(21).mean()

    # Intraday range proxy from close/open.
    rng = sanitize((close - open_).abs() / open_)
    feats["range_21"] = rng.rolling(21).mean()

    # Peer-relative (residual) momentum: 63d stock cum ret minus sector cum ret.
    stock_cum = sanitize(close / close.shift(63) - 1.0)
    sector_cum = sanitize((1.0 + sector_returns).rolling(63).apply(np.prod, raw=True) - 1.0)
    feats["peer_mom_63"] = sanitize(stock_cum - sector_cum)

    # ---- Higher-moment / lottery / idiosyncratic-risk anomalies (price-only) ----
    # All point-in-time (trailing windows) and orthogonal-ish to the fundamentals.
    # MAX effect (Bali, Cakici, Whitelaw 2011): stocks with an extreme recent max
    # daily return underperform (lottery demand -> overpriced).
    feats["max_21"] = sanitize(ret.rolling(21).max())
    # Return skewness (Boyer-Mitton-Vorkink): high positive skew underperforms
    # (investors overpay for lottery-like upside).
    feats["ret_skew_126"] = sanitize(ret.rolling(126).skew())
    # Downside semi-deviation: std of only the negative daily returns (63d).
    neg = ret.where(ret < 0)
    feats["downside_vol_63"] = sanitize(neg.rolling(63, min_periods=20).std())
    # Idiosyncratic volatility: vol of the market-relative return (stock minus the
    # equal-weight universe move) over 63d -> the low-idio-vol anomaly.
    mkt = ret.mean(axis=1)
    feats["idio_vol_63"] = sanitize(ret.sub(mkt, axis=0).rolling(63).std())

    # ---- Liquidity / volume (point-in-time trailing windows; skipped w/o volume) ----
    if volume is not None:
        volume = volume.reindex_like(close)
        dollar_vol = close * volume                        # daily $ traded
        # Liquidity/size proxy: log average daily dollar volume (63d).
        feats["dollar_volume_63"] = sanitize(
            np.log1p(dollar_vol.rolling(63, min_periods=20).mean()))
        # Amihud (2002) illiquidity = mean(|ret| / $volume). HIGHER = more illiquid
        # (illiquidity premium). Scale is irrelevant post cross-sectional ranking.
        amihud = sanitize(ret.abs() / dollar_vol.where(dollar_vol > 0))
        feats["amihud_63"] = amihud.rolling(63, min_periods=20).mean()
        # Relative volume: recent 5d vs 63d average -> volume spike / attention.
        v5 = volume.rolling(5, min_periods=3).mean()
        v63 = volume.rolling(63, min_periods=20).mean()
        feats["rel_volume_5_63"] = sanitize(v5 / v63.where(v63 > 0))

        # ---- Volume-flow dynamics ----
        # Signed-volume imbalance: up-day minus down-day volume as a fraction of
        # total volume (63d) -> net buying(+) / selling(-) pressure (order-flow proxy).
        signed = np.sign(ret) * volume
        num = signed.rolling(63, min_periods=20).sum()
        den = volume.rolling(63, min_periods=20).sum()
        feats["signed_vol_63"] = sanitize(num / den.where(den > 0))
        # Volume trend: recent (21d) vs long (252d) average volume (log) -> whether
        # trading activity is structurally rising or fading.
        v252 = volume.rolling(252, min_periods=60).mean()
        feats["volume_trend_63"] = sanitize(np.log(v63 / v252.where(v252 > 0)))
        # Volume dispersion (coefficient of variation, 63d) -> lumpy/event-driven
        # trading vs steady flow.
        feats["volume_cv_63"] = sanitize(
            volume.rolling(63, min_periods=20).std() / v63.where(v63 > 0))

    # ---- Cross-sectional SEASONALITY at the forecast target t+h (Heston-Sadka) ----
    # A calendar dummy (month of t+h) is identical for every stock on a date -> it
    # has NO cross-sectional dispersion and cannot help a market-neutral ranker.
    # The cross-sectionally useful seasonal signal is the STOCK'S OWN average return
    # over the SAME calendar window in PRIOR years: seasonal_h(t) = mean over the
    # last few years y>=1 of the h-day forward return at t-252*y. Because only
    # y>=1 (>= a year back, fully realized) is used, it is strictly leak-free, and
    # it differs per stock (some names have real same-season repeatability).
    if seasonal_horizons:
        for h in sorted({int(x) for x in seasonal_horizons}):
            # PARTIAL window on purpose (see prices.forward_compound): this averages the
            # last 5 years, so demanding a full h at the sample edge would drop the newest.
            fwd_h = forward_compound(ret, h, min_periods=max(1, int(round(h * 0.6))))
            prior = np.stack([fwd_h.shift(252 * y).to_numpy() for y in range(1, seasonal_years + 1)])
            finite = np.isfinite(prior)
            cnt = finite.sum(axis=0)
            ssum = np.where(finite, prior, 0.0).sum(axis=0)
            seasonal = np.where(cnt > 0, ssum / np.maximum(cnt, 1), np.nan)
            feats[f"seasonal_h{h}"] = sanitize(
                pd.DataFrame(seasonal, index=close.index, columns=close.columns))

    # ---- Tax-loss-selling / January-effect pressure (forced year-end flow) ----
    # Which names get dumped into year-end is a per-STOCK effect (driven by its
    # YTD loss), so it IS cross-sectional (a bare "it's December" dummy is flat
    # across names and useless). tax_loss_pressure = YTD loss magnitude, active
    # only in the tax-selling window (Oct-Dec, covering fund Oct-31 & individual
    # Dec-31 year-ends); it flags names under selling pressure now that tend to
    # rebound in January. Leak-free: YTD uses only past prices, the calendar of t
    # is known. The model learns the sign.
    #
    # OUTSIDE the window the value is NaN, NOT 0. A 0 off-season would feed the
    # per-day cross-sectional ranker a sea of equal zeros (9 of 12 months + every
    # YTD-winner in-season), collapsing the standardized feature to a near-constant
    # ~0.5 whose only variation is the rank's small-sample +1/(2N) bias -> as the
    # universe N grows over the years that bias shrinks, giving a SPURIOUS downward
    # trend linear in the year (a ranking artifact, not a stock signal). NaN makes
    # the feature simply ABSENT off-season, so the rank only ever orders the
    # in-window names (losers high vs non-losers) and the drift disappears.
    year_start = close.groupby(close.index.year).transform("first")
    ytd = sanitize(close / year_start.where(year_start > 0) - 1.0)
    tax_loss = (-ytd).clip(lower=0.0)                 # 0 for YTD winners, >0 for losers
    off_window = ~np.isin(close.index.month, [10, 11, 12])
    tax_loss.loc[off_window] = np.nan                 # NaN outside the Oct-Dec tax window (not 0)
    feats["tax_loss_pressure"] = sanitize(tax_loss)

    # ---- Technical indicators, LAGGED one day (exclude t -> no leakage) ----
    macd_norm, macd_hist = _macd(close)
    feats["macd"] = macd_norm.shift(1)
    feats["macd_hist"] = macd_hist.shift(1)
    feats["rsi_14"] = _rsi(close, 14).shift(1)
    if high is not None and low is not None:
        feats["atr_14"] = _atr(high, low, close, 14).shift(1)

    return feats


def build_feature_panel(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    sector_returns: pd.DataFrame,
    method: str = "rank",
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
    seasonal_horizons: list[int] | None = None,
    *,
    returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a long-format feature panel ready for modeling.

    Returns a tidy DataFrame with columns:
        ['date', 'ticker', <feature_1>, <feature_2>, ...]
    Each feature already cross-sectionally standardized within its date.
    `high`/`low` enable the ATR(14) feature; `volume` enables the liquidity family;
    `seasonal_horizons` enables the per-horizon cross-sectional seasonality feature.
    `returns` passes through daily returns the caller already holds (see
    `compute_raw_features`); keyword-only, so the eight-positional-arg call sites are
    unaffected.
    """
    raw = compute_raw_features(close, open_, sector_returns, high=high, low=low,
                               volume=volume, seasonal_horizons=seasonal_horizons,
                               returns=returns)
    std = {name: xs_standardize(f, method) for name, f in raw.items()}

    long_frames = []
    for name, f in std.items():
        s = f.stack()
        s.index.set_names(["date", "ticker"], inplace=True)
        long_frames.append(s.rename(name))

    panel = pd.concat(long_frames, axis=1).reset_index()
    return panel
