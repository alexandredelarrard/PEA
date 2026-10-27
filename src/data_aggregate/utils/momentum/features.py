"""
features.py
-----------
⚠ TWO PRICE BASES. `open`, `high` and `low` come back from yfinance SPLIT-ADJUSTED ONLY,
so they share a basis with `close_split`, NOT with `close_total`. Mixing them inside one
subtraction is a silent error -- it is the bug the two-column split could easily have
introduced while fixing market cap. Which block uses which:

  close_split  -- `atr_14`     : (high - prev_close), (low - prev_close)
                  `gap_21`     : open / prev_close - 1
                  `range_21`   : |close - open| / open
                  `dollar_volume_63` / `amihud_63` : price x volume x `level_factor`
                                 (the SPLIT factor cancels between price and volume; the
                                 SPINOFF one does NOT -- see below -- and the dividend
                                 factor would not cancel either)
  close_total  -- every RETURN: mom_12_1, rev_5/21, ma_ratio_50/200, high_prox_252,
                  peer_mom_63, the seasonal block, macd, rsi_14, and the `ret` the
                  vol / skew / idio family is built from

`atr_14` itself is basis-INVARIANT (it returns atr/close, a ratio of same-basis
quantities); the intermediate true range is not, which is exactly why it needs pinning.

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

⚠ SEAM MASKING. A registered `null_ret` entry leaves both price legs exactly as published, so
the level is on two different BASES either side of the bar and any window straddling it mixes
them -- see `level_basis`'s module docstring. `SEAM_WINDOWS` below masks each affected feature
over its own lookback. Which families are masked, and why the rest are not:

  masked      -- every LEVEL-RATIO feature: `mom_12_1`, `rev_5/21`, `ma_ratio_50/200`,
                 `high_prox_252` (the seam RAISES the rolling max), `peer_mom_63`, the lagged
                 `macd` / `macd_hist` / `rsi_14` / `atr_14`, and the `close_split`-based
                 `gap_21` / `range_21`.
  not masked  -- the `ret`-derived family (`vol_21/63`, `max_21`, `ret_skew_126`,
                 `downside_vol_63`, `idio_vol_63`): `apply_null_ret` already deleted the one
                 bad cell from `ret`, so these carry a single hole their `min_periods` absorbs
                 rather than a mixed basis.
               -- `seasonal_h*`: also `ret`-derived, through `forward_compound`, and its
                 deliberately partial `min_periods` tolerates the hole.
               -- the whole VOLUME family: `close_split x volume x level_factor` was measured
                 smooth through all six seams (`DHR` `dollar_volume_63` 0.837 -> 0.841 ->
                 0.848, `LDOS` 0.433 flat, `DD` 0.985 flat). The split factor cancels between
                 price and volume and `level_factor` carries the spinoff one.
The `close_split` family (`atr_14`, `gap_21`, `range_21`) is masked on the SAME dates even
though the seams are measured on `close_total`. Measured justification, and its cost: the seam
is present in both legs at `DD` / `LDOS` / `VTR` (the full step, identically) and partly at
`CNP`, and the stored ranks move accordingly -- `LDOS` `gap_21` jumps 0.007 -> 0.901 on its
seam date, `VTR` `gap_21` 0.591 -> 0.002 and `atr_14` 0.282 -> 0.978. `DHR` and `ETN` are the
exception: their `close_split` steps only ~3.5%, being back-adjusted by the `prices_splits`
ratio, so those two are over-masked by their features' own windows. That is the conservative
side of the trade and it is 21-43 bars on two tickers.
"""

from __future__ import annotations

import logging
from typing import Literal, cast

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.frames import sanitize
from src.data_aggregate.utils.common.panel import mask_to_availability
from src.data_aggregate.utils.common.xs import xs_standardize
from src.data_aggregate.utils.price.checks import mask_seam_windows
from src.data_aggregate.utils.price.frames import PriceFrames
from src.data_aggregate.utils.price.returns import forward_compound, momentum_characteristic, trailing_vol
from src.utils.price.levels import dollar_volume

logger = logging.getLogger(__name__)

#: A feature's cross-section is ranked only when its population that day is at least this
#: FRACTION of its OWN recent typical population. `xs_rank_pct` is `rank(axis=1, pct=True)`,
#: which ranks over whatever is non-null in the ROW -- so 45 names out of 491 silently become a
#: full-looking (0, 1] percentile whose minimum is 1/45 = 0.0222 instead of 1/491 = 0.0020.
#: The stored column carries no trace of how many names produced it, so a truncated extract run
#: publishes a decile that is not comparable to any other date's.
#: Measured over all 7,793 dates x 28 features of the live panel (216,137 non-empty cells):
#: this rule nulls 13 cells on exactly one date -- 2026-08-28, the truncated run -- and nothing
#: else.
#:
#: ⚠ The reference is the feature's OWN trailing median, NOT the day's best-populated feature.
#: The day-max variant cannot separate a STRUCTURAL population difference (a 1,260-day
#: seasonality feature is legitimately thinner than a 5-day reversal early in history) from a
#: SUDDEN collapse: measured, it also nulls 18 legitimate 1995 `downside_vol_63` cells. A
#: feature's own trailing median can: a warm-up ramp grows monotonically so the floor never
#: binds, while a truncated run drops the population to 9% of it.
MIN_XS_POPULATION_FRAC = 0.60

#: Trailing dates the reference median is taken over -- one month of sessions, long enough that
#: a single bad day cannot move it and short enough to track a genuinely growing universe.
XS_POPULATION_WINDOW = 21

#: How many EWMA spans back a seam is still treated as reaching. An EWMA has no finite window,
#: so the mask needs a decay cut-off rather than an exact one: at 3 spans the bad bar's weight
#: is `(1 - 1/14)^42 = 4.4%` for Wilder's RSI/ATR and `(1 - 2/27)^78 = 0.25%` for MACD's slow
#: leg. One consumer -- `SEAM_WINDOWS` immediately below -- so it lives here.
SEAM_EWMA_SPAN_MULTIPLE = 3

#: `(back, skip)` per masked feature: the STORED value at `t` reads its price input over index
#: positions `[t-back, t-skip]`, so it straddles a seam at `s` for `t` in `[s+skip, s+back-1]`.
#: The technical block's `skip=1` is its own `.shift(1)`, and its `back` is
#: `SEAM_EWMA_SPAN_MULTIPLE x span + 1` for the same reason.
#: Verified against the live panel at `DHR`/2016-07-05 -- see `mask_seam_windows`.
SEAM_WINDOWS: dict[str, tuple[int, int]] = {
    "mom_12_1": (252, 21),  # close.shift(21) / close.shift(252)
    "rev_5": (5, 0),
    "rev_21": (21, 0),
    "ma_ratio_50": (49, 0),  # rolling(50) covers t-49..t
    "ma_ratio_200": (199, 0),
    "high_prox_252": (251, 0),  # rolling(252) max -- the seam RAISES the max
    "peer_mom_63": (63, 0),
    "macd": (79, 1),  # 3 x the slow EWMA span (26), + the .shift(1)
    "macd_hist": (79, 1),
    "rsi_14": (43, 1),  # 3 x Wilder's n=14, + the .shift(1)
    # the `close_split` family. It shares no basis with `close_total`, but the seam is in BOTH
    # legs at DD / LDOS / VTR (the full step, identically) and partly at CNP (+6.06% against
    # close_total's +17.83%); only DHR and ETN are clean, being back-adjusted by the
    # `prices_splits` ratio. Masked on the close_total-measured dates anyway, so ~21-43 bars
    # are over-masked at those two -- the conservative side, and cheap.
    "atr_14": (43, 1),
    "gap_21": (21, 0),  # rolling(21) mean of open / close_split.shift(1) - 1
    "range_21": (21, 0),  # rolling(21) mean of |close_split - open| / open
}


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
        index=close.index,
        columns=close.columns,
    )
    atr = true_range.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    denom = close.where(close > 0)
    return sanitize(atr / denom)


def _variation_features(
    df_close: pd.DataFrame,
    df_open: pd.DataFrame,
    df_sector: pd.DataFrame,
    df_ret: pd.DataFrame,
    df_split: pd.DataFrame,
    seams: dict[str, list[pd.Timestamp]] | None,
) -> dict[str, pd.DataFrame]:
    """Compute matched-price momentum, reversal, trend and candle signals."""
    feats = {}
    # Total-return ratios and realized volatility.
    feats["mom_12_1"] = sanitize(momentum_characteristic(df_close, seams=seams))
    feats["rev_5"] = sanitize(-(df_close / df_close.shift(5) - 1.0))
    feats["rev_21"] = sanitize(-(df_close / df_close.shift(21) - 1.0))
    feats["vol_21"] = sanitize(trailing_vol(df_ret, 21))
    feats["vol_63"] = sanitize(trailing_vol(df_ret, 63))
    feats["ma_ratio_50"] = sanitize(df_close / df_close.rolling(50).mean() - 1.0)
    feats["ma_ratio_200"] = sanitize(df_close / df_close.rolling(200).mean() - 1.0)
    feats["high_prox_252"] = sanitize(df_close / df_close.rolling(252).max())
    # OHLC comparisons keep the split-adjusted quote basis.
    df_gap = sanitize(df_open / df_split.shift(1) - 1.0)
    feats["gap_21"] = df_gap.rolling(21).mean()
    df_range = sanitize((df_split - df_open).abs() / df_open)
    feats["range_21"] = df_range.rolling(21).mean()
    df_stock_cum = sanitize(df_close / df_close.shift(63) - 1.0)
    df_sector_cum = sanitize((1.0 + df_sector).rolling(63).apply(np.prod, raw=True) - 1.0)
    feats["peer_mom_63"] = sanitize(df_stock_cum - df_sector_cum)
    return feats


def _risk_features(df_ret: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute trailing extreme, skew, downside and idiosyncratic return risk."""
    df_negative = df_ret.where(df_ret < 0)
    market = df_ret.mean(axis=1)
    return {
        "max_21": sanitize(df_ret.rolling(21).max()),
        "ret_skew_126": sanitize(df_ret.rolling(126).skew()),
        "downside_vol_63": sanitize(df_negative.rolling(63, min_periods=5).std()),
        "idio_vol_63": sanitize(df_ret.sub(market, axis=0).rolling(63).std()),
    }


def _liquidity_features(
    prices: PriceFrames,
    df_ret: pd.DataFrame,
    df_split: pd.DataFrame,
    df_tape_support: pd.DataFrame | None,
) -> dict[str, pd.DataFrame]:
    """Value compatible tape in dollars and compute trailing volume signals."""
    if prices.volume is None:
        return {}
    assert prices.close_total is not None
    df_volume = prices.volume.reindex_like(prices.close_total)
    df_dollars = dollar_volume(df_split, df_volume, prices.level_factor, prices.close_pit, df_tape_support)
    df_dollar_log = cast(pd.DataFrame, np.log1p(df_dollars.rolling(63, min_periods=20).mean()))
    df_amihud = sanitize(df_ret.abs() / df_dollars.where(df_dollars > 0))
    df_amihud_mean = df_amihud.rolling(63, min_periods=20).mean()
    df_amihud_log = cast(pd.DataFrame, np.log(df_amihud_mean.where(df_amihud_mean > 0)))
    df_short = df_volume.rolling(5, min_periods=3).mean()
    df_medium = df_volume.rolling(63, min_periods=20).mean()
    df_signed = cast(pd.DataFrame, np.sign(df_ret) * df_volume)
    df_numerator = df_signed.rolling(63, min_periods=20).sum()
    df_denominator = df_volume.rolling(63, min_periods=20).sum()
    df_long = df_volume.rolling(252, min_periods=60).mean()
    df_trend = cast(pd.DataFrame, np.log(df_medium / df_long.where(df_long > 0)))
    return {
        "dollar_volume_63": sanitize(df_dollar_log),
        "amihud_63": sanitize(df_amihud_log),
        "rel_volume_5_63": sanitize(df_short / df_medium.where(df_medium > 0)),
        "signed_vol_63": sanitize(df_numerator / df_denominator.where(df_denominator > 0)),
        "volume_trend_63": sanitize(df_trend),
        "volume_cv_63": sanitize(df_volume.rolling(63, min_periods=20).std() / df_medium.where(df_medium > 0)),
    }


def _seasonal_features(
    df_ret: pd.DataFrame,
    df_close: pd.DataFrame,
    horizons: list[int] | None,
    years: int,
) -> dict[str, pd.DataFrame]:
    """Average each target window's realized return over prior calendar years."""
    feats = {}
    if not horizons:
        return feats
    for horizon in sorted({int(value) for value in horizons}):
        df_forward = cast(pd.DataFrame, forward_compound(df_ret, horizon, min_periods=max(1, int(round(horizon * 0.6)))))
        prior = np.stack([df_forward.shift(252 * year).to_numpy() for year in range(1, years + 1)])
        finite = np.isfinite(prior)
        count = finite.sum(axis=0)
        total = np.where(finite, prior, 0.0).sum(axis=0)
        seasonal = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        feats[f"seasonal_h{horizon}"] = sanitize(pd.DataFrame(seasonal, index=df_close.index, columns=df_close.columns))
    return feats


def _technical_features(prices: PriceFrames, df_split: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute one-session-lagged technical signals on their matched price bases."""
    assert prices.close_total is not None
    df_macd, df_histogram = _macd(prices.close_total)
    feats = {
        "macd": df_macd.shift(1),
        "macd_hist": df_histogram.shift(1),
        "rsi_14": _rsi(prices.close_total, 14).shift(1),
    }
    if prices.high is not None and prices.low is not None:
        feats["atr_14"] = _atr(prices.high, prices.low, df_split, 14).shift(1)
    return feats


def compute_raw_features(
    prices: PriceFrames,
    seasonal_horizons: list[int] | None = None,
    seasonal_years: int = 5,
    *,
    tape_support: pd.DataFrame | None = None,
    seams: dict[str, list[pd.Timestamp]] | None = None,
    split_seams: dict[str, list[pd.Timestamp]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute trailing raw signals, masking each price seam before any ranking."""
    prices.require("close_total", "open", "sector_ret")
    assert prices.close_total is not None and prices.open is not None and prices.sector_ret is not None
    df_close = prices.close_total
    df_ret = df_close.pct_change(fill_method=None) if prices.ret is None else prices.ret.reindex_like(df_close)
    df_split = df_close if prices.close_split is None else prices.close_split.reindex_like(df_close)
    # Preserve family order and optional-field behavior.
    feats = _variation_features(df_close, prices.open, prices.sector_ret, df_ret, df_split, seams)
    feats.update(_risk_features(df_ret))
    feats.update(_liquidity_features(prices, df_ret, df_split, tape_support))
    feats.update(_seasonal_features(df_ret, df_close, seasonal_horizons, seasonal_years))
    feats.update(_technical_features(prices, df_split))
    # Price-ratio windows must be masked before the cross-sectional rank.
    for name, (back, skip) in SEAM_WINDOWS.items():
        selected = (split_seams if split_seams is not None else seams) if name in {"atr_14", "gap_21", "range_21"} else seams
        if selected and name in feats:
            feats[name] = mask_seam_windows(feats[name], selected, back, skip)
    return feats


def _thin_cross_sections(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """(date x feature) booleans: True where that feature's cross-section is too thin to rank.

    Compares each feature's population on a date against its OWN trailing median population.
    See `MIN_XS_POPULATION_FRAC` for the measurement behind the threshold and for why the
    reference is the feature's own history rather than the day's best-populated feature.

    An all-empty date is left False: it has nothing to rank either way, and flagging it would
    fire on every date before a long-window feature's warm-up completes."""
    pop = pd.DataFrame({name: f.notna().sum(axis=1) for name, f in raw.items()})
    # shift(1): the reference is the population strictly BEFORE this date, so a collapse can
    # never dilute the very median it is being judged against.
    ref = pop.shift(1).rolling(XS_POPULATION_WINDOW, min_periods=5).median()
    return (pop > 0) & pop.lt(MIN_XS_POPULATION_FRAC * ref)


def _null_thin_cross_sections(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Null every too-thin (date, feature) cross-section instead of ranking it, and WARN.

    Mutates `raw` and returns it (the caller's dict comes straight from
    `compute_raw_features`, so there is nothing else holding a reference to it).

    ⚠ THE ORDERING IS LOAD-BEARING, for the same reason seam masking is (see
    `compute_raw_features`): this must run AFTER the seam mask and BEFORE `xs_standardize`. A
    value that reaches the rank has already contaminated every OTHER name's percentile, so a
    guard applied to the standardized output would be far too late.

    NULLING, not renormalising: there is no way to recover a comparable percentile from a
    fraction of the universe, and the 15 well-populated features on a thin date stay untouched
    and usable. The WARNING is half the fix -- D-01 was undetectable precisely because
    `rank(pct=True)` reports success on any population at all."""
    thin = _thin_cross_sections(raw)
    for name, f in raw.items():
        if not thin[name].any():
            continue
        dates = thin.index[thin[name]]
        logger.warning(
            "%s: cross-section too thin on %d date(s) -> NULLED rather than ranked (%s%s). A "
            "percentile drawn from a fraction of the universe is not comparable to one drawn "
            "from all of it.",
            name,
            len(dates),
            ", ".join(str(d.date()) for d in dates[:5]),
            ", ..." if len(dates) > 5 else "",
        )
        raw[name] = f.mask(thin[name], axis=0)
    return raw


def build_feature_panel(
    prices: PriceFrames,
    method: Literal["rank", "zscore"] = "rank",
    seasonal_horizons: list[int] | None = None,
    *,
    tape_support: pd.DataFrame | None = None,
    seams: dict[str, list[pd.Timestamp]] | None = None,
    split_seams: dict[str, list[pd.Timestamp]] | None = None,
) -> pd.DataFrame:
    """Standardize observed raw signals after seam and thin-population masks."""
    raw = compute_raw_features(prices, seasonal_horizons, tape_support=tape_support, seams=seams, split_seams=split_seams)
    assert prices.close_total is not None
    availability = (prices.close_split if prices.close_split is not None else prices.close_total).notna()
    raw = {name: mask_to_availability(frame, availability) for name, frame in raw.items()}
    raw = _null_thin_cross_sections(raw)
    standardized = {name: xs_standardize(frame, method) for name, frame in raw.items()}
    long_frames = []
    for name, df_feature in standardized.items():
        series = cast(pd.Series, df_feature.stack())
        series.index.set_names(["date", "ticker"], inplace=True)
        series.name = name
        long_frames.append(series)
    return pd.concat(long_frames, axis=1).dropna(how="all").reset_index()
