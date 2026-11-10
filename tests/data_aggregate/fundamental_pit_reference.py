"""Explicit legacy capital inputs for feature regression, separate from PIT correctness tests."""

from __future__ import annotations

import pandas as pd

from src.data_aggregate.utils.common.pit import _observations_to_daily, fundamentals_to_daily
from src.data_aggregate.utils.price.frames import PriceFrames
from src.utils.price.adjustments import ShareBasis


def known_no_share_events() -> ShareBasis:
    """Explicit event-free basis for known-truth feature fixtures."""
    df_events = pd.DataFrame({"ticker": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]"), "value": pd.Series(dtype=float)})
    return ShareBasis(df_events, df_events, df_events, {})


def legacy_capital_frames(
    history: pd.DataFrame | None, close: pd.DataFrame | None, level_factor: pd.DataFrame | None = None, max_age_days: int | None = 185
) -> PriceFrames | None:
    """Supply the former compatible price/count pair as explicit regression inputs.

    This fixture preserves old feature comparisons; source-backed price tests establish
    the actual historical quote, quantity units and issuer-cap coverage separately.
    """
    if close is None:
        return None
    df_counts = (
        pd.DataFrame(index=close.index)
        if history is None
        else fundamentals_to_daily(history, "sharesOutstanding", close.index, max_age_days=max_age_days)
    )
    df_counts = df_counts.reindex(columns=close.columns.intersection(df_counts.columns))
    df_price = close if level_factor is None else close * level_factor.reindex_like(close)
    df_cap = (df_price[df_counts.columns] * df_counts).where(lambda values: values > 0)
    return PriceFrames(
        trading_index=pd.DatetimeIndex(close.index),
        universe=tuple(map(str, close.columns)),
        peers={},
        close_split=close,
        level_factor=level_factor,
        close_pit=df_price,
        shares_outstanding_pit=df_counts,
        mcap_pit=df_cap,
    )


def legacy_fundamentals_to_daily(
    fundamentals: pd.DataFrame,
    field: str,
    trading_index: pd.DatetimeIndex,
    max_age_days: int | None = None,
    *,
    share_basis: ShareBasis | None = None,
) -> pd.DataFrame:
    """
    Turn a (ticker, as_of, <fields>) history into a daily wide frame for one
    field, forward-filled point-in-time: value on date d is the most recent
    as_of <= d. No look-ahead.
    """
    if field not in fundamentals.columns:
        return pd.DataFrame(index=trading_index)
    df = fundamentals[["ticker", "as_of", field]].copy()
    if share_basis is None or field != "sharesOutstandingPit":
        return _observations_to_daily(df, field, trading_index, max_age_days=max_age_days)
    configured = df["ticker"].isin(share_basis.policies)
    df.loc[configured, field] = df.loc[configured, field] * share_basis.factor(df.loc[configured, "ticker"], df.loc[configured, "as_of"], "actual")
    daily = _observations_to_daily(df, field, trading_index, max_age_days=max_age_days)
    for ticker in share_basis.policies.keys() & set(daily.columns):
        daily[ticker] /= share_basis.factor(pd.Series(ticker, index=trading_index), pd.Series(trading_index, index=trading_index), "actual")
    return daily


def legacy_daily_market_cap(
    fundamentals: pd.DataFrame,
    close_split: pd.DataFrame,
    *,
    level_factor: pd.DataFrame | None,
    max_age_days: int | None = None,
) -> pd.DataFrame:
    """Historical daily market cap = ffilled `sharesOutstanding` x `close_split` x `S(d)`.

    ⚠ BOTH PRICE LEGS MUST BE SPLIT-ADJUSTED, and the parameter is named for it. The vendor
    back-fills `sharesbas` to today's basis (`sharesbas(d) = real_shares(d) x F(d)`) and
    Yahoo restates `Close` to the same one (`close_split(d) = raw_price(d) / F(d)`), so the
    future-SPLIT factor CANCELS IDENTICALLY in the product.

    Handing it `close_total` instead reintroduces the defect this signature exists to
    prevent: nothing in a share count carries a dividend factor, so `D(d)` survives into the
    product -- median 0.618 in 2003, i.e. market cap 38% too low, and monotone in FUTURE
    dividends. Handing it a de-adjusted share count breaks the cancellation the other way.

    ⚠ `level_factor` IS REQUIRED AND KEYWORD-ONLY, deliberately, because the cancellation
    above holds for splits and FAILS for SPINOFFS. Yahoo back-adjusts `Close` across a
    spinoff and a spinoff does not change the share count, so that factor does not cancel and
    the product comes out LOW by exactly `S(d)`. Measured on FDX 2020-12-17:

        close_split 235.5036 x 265,070,592 shares  = $62.425bn
        x S = 1.241                                = $77.470bn
        Sharadar's own `marketcap`                 = $77.470bn

    GE reads 40% low before 2019 and DD 80% low before 2019 without it. There is no default:
    a caller that has not thought about the basis must get a `TypeError`, not a number that
    is quietly wrong for 80 of 491 tickers. Pass `None` ONLY to mean "S is 1.0 everywhere",
    which is true for a synthetic fixture and for ~89% of real cells but never for the
    universe as a whole.

    Requires a `sharesOutstanding` column (the VENDOR-basis one, not `sharesOutstandingPit`).
    """
    shares = fundamentals_to_daily(
        fundamentals,
        "sharesOutstanding",
        close_split.index,
        max_age_days=max_age_days,
    )
    if shares.empty:
        return pd.DataFrame(index=close_split.index)

    cols = [c for c in shares.columns if c in close_split.columns]
    if not cols:
        return pd.DataFrame(index=close_split.index)

    mcap = close_split[cols].mul(shares[cols])
    if level_factor is not None and not level_factor.empty:
        # Reindexed rather than assumed aligned: `level_factor` spans the whole universe
        # while `cols` is the intersection with the filing history. `fillna(1.0)` because a
        # missing factor means "no adjustment", never "no market cap" -- the alternative
        # silently NULLs a ticker's entire series.
        mcap = mcap.mul(level_factor.reindex(index=mcap.index, columns=cols).fillna(1.0))
    return mcap.where(mcap > 0)
