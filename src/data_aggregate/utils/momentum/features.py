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

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.frames import sanitize
from src.data_aggregate.utils.common.level_basis import mask_seam_windows
from src.data_aggregate.utils.common.prices import forward_compound, momentum_characteristic, trailing_vol
from src.data_aggregate.utils.common.xs import xs_standardize

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
    "mom_12_1":      (252, 21),   # close.shift(21) / close.shift(252)
    "rev_5":         (5, 0),
    "rev_21":        (21, 0),
    "ma_ratio_50":   (49, 0),     # rolling(50) covers t-49..t
    "ma_ratio_200":  (199, 0),
    "high_prox_252": (251, 0),    # rolling(252) max -- the seam RAISES the max
    "peer_mom_63":   (63, 0),
    "macd":          (79, 1),     # 3 x the slow EWMA span (26), + the .shift(1)
    "macd_hist":     (79, 1),
    "rsi_14":        (43, 1),     # 3 x Wilder's n=14, + the .shift(1)
    # the `close_split` family. It shares no basis with `close_total`, but the seam is in BOTH
    # legs at DD / LDOS / VTR (the full step, identically) and partly at CNP (+6.06% against
    # close_total's +17.83%); only DHR and ETN are clean, being back-adjusted by the
    # `prices_splits` ratio. Masked on the close_total-measured dates anyway, so ~21-43 bars
    # are over-masked at those two -- the conservative side, and cheap.
    "atr_14":        (43, 1),
    "gap_21":        (21, 0),     # rolling(21) mean of open / close_split.shift(1) - 1
    "range_21":      (21, 0),     # rolling(21) mean of |close_split - open| / open
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
        index=close.index, columns=close.columns,
    )
    atr = true_range.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    denom = close.where(close > 0)
    return sanitize(atr / denom)


def compute_raw_features(
    close_total: pd.DataFrame,
    open: pd.DataFrame,
    sector_returns: pd.DataFrame,
    close_split: pd.DataFrame | None = None,
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
    seasonal_horizons: list[int] | None = None,
    seasonal_years: int = 5,
    *,
    returns: pd.DataFrame | None = None,
    level_factor: pd.DataFrame | None = None,
    seams: dict[str, list[pd.Timestamp]] | None = None,
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

    ⚠ TWO PRICE BASES, and mixing them inside one subtraction is a silent bug. See the
    module docstring's basis table. `close_split` defaults to `close_total` only so a
    dividend-free synthetic fixture stays a one-liner; every real caller passes both.

    `returns` lets the caller pass daily returns it ALREADY has instead of having them
    re-derived here -- `du.daily_returns` is literally `close.pct_change(fill_method=None)`,
    and the cube's price step persists that frame, so recomputing it was pure duplication.
    KEYWORD-ONLY so the existing positional call sites are untouched. On an incrementally
    trimmed window the passed frame is also strictly better: a recompute would return NaN on
    the window's first row where the full build had a value.

    `seams` (from `level_basis.measure_seams`) masks every feature in `SEAM_WINDOWS` over its
    own lookback -- see the module docstring for which families are covered and which are
    deliberately not. `None` means no mask and reproduces the pre-mask frames bit-for-bit.
    """
    ret = (close_total.pct_change(fill_method=None) if returns is None
           else returns.reindex_like(close_total))
    # `open`, `high` and `low` come back SPLIT-ADJUSTED ONLY, so anything subtracting one of
    # them from a close must use `close_split`. Pairing them with `close_total` mixes bases
    # inside a single subtraction -- a NEW bug, introduced by the two-column fix rather than
    # fixed by it.
    split = close_total if close_split is None else close_split.reindex_like(close_total)
    feats = {}

    # 12-1 momentum: cumulative return from t-252 to t-21 (skip last month).
    # `seams` goes through the shared primitive, not the loop below, so this feature and the
    # target's neutralizer are masked by the SAME code rather than by two agreeing copies.
    feats["mom_12_1"] = sanitize(momentum_characteristic(close_total, seams=seams))

    # Short-term reversal (negated: recent losers tend to bounce).
    feats["rev_5"] = sanitize(-(close_total / close_total.shift(5) - 1.0))
    feats["rev_21"] = sanitize(-(close_total / close_total.shift(21) - 1.0))

    # Trailing realized volatility (annualized-ish; scale irrelevant post-rank).
    feats["vol_21"] = sanitize(trailing_vol(ret, 21))
    feats["vol_63"] = sanitize(trailing_vol(ret, 63))

    # Trend: distance from moving averages.
    feats["ma_ratio_50"] = sanitize(close_total / close_total.rolling(50).mean() - 1.0)
    feats["ma_ratio_200"] = sanitize(close_total / close_total.rolling(200).mean() - 1.0)

    # Proximity to trailing 52-week high (near-high names show continuation).
    feats["high_prox_252"] = sanitize(close_total / close_total.rolling(252).max())

    # Overnight gap: average of (open_t / close_{t-1} - 1) over 21d.
    gap = sanitize(open / split.shift(1) - 1.0)
    feats["gap_21"] = gap.rolling(21).mean()

    # Intraday range proxy from close/open.
    rng = sanitize((split - open).abs() / open)
    feats["range_21"] = rng.rolling(21).mean()

    # Peer-relative (residual) momentum: 63d stock cum ret minus sector cum ret.
    stock_cum = sanitize(close_total / close_total.shift(63) - 1.0)
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
    #
    # ⚠ min_periods=5, deliberately UNLIKE the 20 every other 63-day window here uses. Do not
    # "harmonise" it back. `neg` keeps only the DOWN days, so the period count is not a data-
    # availability measure -- it is a count of losing days. A name with fewer than 20 down days
    # in 63 sessions is a name that has been going UP, so at min_periods=20 the NaN IS the
    # outcome: measured 14,376 nulled cells across 410 tickers whose median trailing 63-day
    # return is +21.36%, against +3.87% where the feature is present -- a +17.48pp gap.
    # `baselines.py` then mean-imputes those to the median rank, which converts the pattern into
    # a feature VALUE and hands the model a fragment of its own label. A noisier standard
    # deviation estimated off 5 observations is the better side of that trade.
    neg = ret.where(ret < 0)
    feats["downside_vol_63"] = sanitize(neg.rolling(63, min_periods=5).std())
    # Idiosyncratic volatility: vol of the market-relative return (stock minus the
    # equal-weight universe move) over 63d -> the low-idio-vol anomaly.
    mkt = ret.mean(axis=1)
    feats["idio_vol_63"] = sanitize(ret.sub(mkt, axis=0).rolling(63).std())

    # ---- Liquidity / volume (point-in-time trailing windows; skipped w/o volume) ----
    if volume is not None:
        volume = volume.reindex_like(close_total)
        # `None` -> 1.0, so the synthetic fingerprint harness and every test that has no
        # factor to give are bit-identical to the pre-`S` result rather than skipped.
        lvl = (1.0 if level_factor is None
               else level_factor.reindex_like(close_total).fillna(1.0))
        # `split`, not `close_total`: dollar volume is a LEVEL x a share count, and both
        # carry the same SPLIT restatement so it cancels. On the total-return basis the
        # dollars traded would be depressed by every later dividend.

        # ⚠ Spinoff reduce price and backfilled, but vol does not
        # lvl is the factor to apply to volume to represent the spinoff shares added
        dollar_vol = split * volume * lvl                  # daily $ traded

        # Liquidity/size proxy: log average daily dollar volume (63d).
        feats["dollar_volume_63"] = sanitize(
            np.log1p(dollar_vol.rolling(63, min_periods=20).mean()))
        # Amihud (2002) illiquidity = mean(|ret| / $volume). HIGHER = more illiquid
        # (illiquidity premium).
        #
        # ⚠ np.log, NOT log1p. The raw statistic spans 4.1e-13 to 9.5e-5 -- eight orders of
        # magnitude -- and at that scale `log1p(x) == x` to four significant figures, so the
        # sibling's transform would be a literal no-op here. (`log1p` is right for
        # `dollar_volume_63`, whose values are ~1e8.) The transform is RANK-INVARIANT, so the
        # stored panel is bit-identical under the live `standardize_method: rank`; it exists so
        # a flip to `zscore` in configs/build_cube.yml does not crowd 76.8% of the
        # cross-section inside +/-0.25sd, which is what the un-logged statistic does.
        #
        # `.where(> 0)` before the log, not `sanitize` afterwards: |ret| is exactly 0 on a flat
        # day, so an all-flat 63-day window averages to 0 and `log(0)` is -inf. `sanitize` does
        # map +/-inf to NaN, but relying on that emits a NumPy divide warning every build.
        amihud = sanitize(ret.abs() / dollar_vol.where(dollar_vol > 0))
        amihud_mean = amihud.rolling(63, min_periods=20).mean()
        feats["amihud_63"] = sanitize(np.log(amihud_mean.where(amihud_mean > 0)))
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
                pd.DataFrame(seasonal, index=close_total.index, columns=close_total.columns))

    # ---- Technical indicators, LAGGED one day (exclude t -> no leakage) ----
    macd_norm, macd_hist = _macd(close_total)
    feats["macd"] = macd_norm.shift(1)
    feats["macd_hist"] = macd_hist.shift(1)
    feats["rsi_14"] = _rsi(close_total, 14).shift(1)
    if high is not None and low is not None:
        # `split`: `_atr` computes (high - prev_close) and (low - prev_close), so all three
        # legs must share one basis. It returns atr/close -- a ratio of same-basis
        # quantities -- so the FEATURE is basis-invariant; the intermediate is not.
        feats["atr_14"] = _atr(high, low, split, 14).shift(1)

    # ---- Seam masking, BEFORE the caller standardizes ----
    # ⚠ THE ORDERING IS LOAD-BEARING. `build_feature_panel` ranks cross-sectionally after this
    # returns; masking afterwards would leave the fabricated value in the cross-section, where
    # it takes rank 459/460 and shifts every OTHER name's rank by ~1/460. The defect is not
    # confined to the six seam tickers unless the mask lands first.
    # `mom_12_1` is in `SEAM_WINDOWS` for the spec table and was already masked above by
    # `momentum_characteristic` with the same window, so re-masking it here is exactly
    # idempotent.
    if seams:
        for name, (back, skip) in SEAM_WINDOWS.items():
            if name in feats:
                feats[name] = mask_seam_windows(feats[name], seams, back, skip)

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
            name, len(dates), ", ".join(str(d.date()) for d in dates[:5]),
            ", ..." if len(dates) > 5 else "")
        raw[name] = f.mask(thin[name], axis=0)
    return raw


def build_feature_panel(
    close_total: pd.DataFrame,
    open: pd.DataFrame,
    sector_returns: pd.DataFrame,
    method: str = "rank",
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
    seasonal_horizons: list[int] | None = None,
    *,
    returns: pd.DataFrame | None = None,
    close_split: pd.DataFrame | None = None,
    level_factor: pd.DataFrame | None = None,
    seams: dict[str, list[pd.Timestamp]] | None = None,
) -> pd.DataFrame:
    """
    Build a long-format feature panel ready for modeling.

    Returns a tidy DataFrame with columns:
        ['date', 'ticker', <feature_1>, <feature_2>, ...]
    Each feature already cross-sectionally standardized within its date.
    `close_split` is the split-adjusted-only series that `open`/`high`/`low` share a basis
    with -- see the module docstring. `high`/`low` enable the ATR(14) feature; `volume`
    enables the liquidity family;
    `seasonal_horizons` enables the per-horizon cross-sectional seasonality feature.
    `returns` passes through daily returns the caller already holds (see
    `compute_raw_features`); keyword-only, so the eight-positional-arg call sites are
    unaffected. `seams` masks the straddling lookback windows -- applied inside
    `compute_raw_features`, i.e. strictly BEFORE the `xs_standardize` below, which is what
    keeps a fabricated value out of the cross-section.

    The thin-cross-section guard sits in the same gap, and for the same reason -- see
    `_null_thin_cross_sections`.
    """
    raw = compute_raw_features(close_total, open, sector_returns, close_split=close_split,
                               high=high, low=low, volume=volume,
                               seasonal_horizons=seasonal_horizons, returns=returns,
                               level_factor=level_factor, seams=seams)
    raw = _null_thin_cross_sections(raw)
    std = {name: xs_standardize(f, method) for name, f in raw.items()}

    long_frames = []
    for name, f in std.items():
        s = f.stack()
        s.index.set_names(["date", "ticker"], inplace=True)
        long_frames.append(s.rename(name))

    panel = pd.concat(long_frames, axis=1).reset_index()
    return panel
