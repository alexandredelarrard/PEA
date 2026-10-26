"""Shared historical quote reconstruction and guarded level factors."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd

from src.utils.price.adjustments import ShareBasis, suffix_factor
from src.utils.price.repairs import BUGFIX_WEDGE_TOL, LEVEL_SNAP_TOL, observed_wedge

logger = logging.getLogger(__name__)


def level_factor(
    index: pd.DatetimeIndex,
    universe: Sequence[str],
    yf_splits: pd.DataFrame | None,
    genuine_splits: pd.DataFrame | ShareBasis,
) -> pd.DataFrame:
    """Return S=Y/(F*U), before hidden wedges, aligned to the requested date/ticker grid.

    A ShareBasis supplies separately verified price, vendor-count and quoted-unit views.
    The legacy DataFrame argument preserves the default event-product behavior.
    """
    if isinstance(genuine_splits, ShareBasis):
        basis = genuine_splits
        events = basis.vendor
        yf_splits = basis.price
    else:
        basis = None
        events = genuine_splits
    columns = sorted(set(map(str, universe)))
    idx = pd.DatetimeIndex(index)
    frame = pd.DataFrame(1.0, index=idx, columns=columns, dtype="float64")
    frame.index.name, frame.columns.name = "date", "ticker"

    stamps = idx.to_numpy(dtype="datetime64[ns]")
    numerator = suffix_factor(
        (yf_splits.rename(columns={"ratio": "value"}) if yf_splits is not None and "ratio" in yf_splits.columns else yf_splits), columns, stamps
    )
    denominator = suffix_factor(events, columns, stamps)

    for ticker in set(numerator) | set(denominator):
        up = numerator.get(ticker)
        down = denominator.get(ticker)
        if up is None:
            assert down is not None
            frame[ticker] = 1.0 / down
        elif down is None:
            frame[ticker] = up
        else:
            frame[ticker] = up / down

    if basis is not None:
        for ticker in basis.policies.keys() & set(columns):
            units = basis.factor(pd.Series(ticker, index=idx), pd.Series(idx, index=idx), "units")
            frame[ticker] = frame[ticker] / units

    # `copy=True`: pandas hands back a READ-ONLY view of a single-block frame, so writing the
    # snap into it raises rather than silently doing nothing.
    values = frame.to_numpy(dtype="float64", copy=True)
    values[np.abs(values - 1.0) < LEVEL_SNAP_TOL] = 1.0
    return pd.DataFrame(values, index=frame.index, columns=frame.columns)


def apply_level_bugfix(factor: pd.DataFrame, bugfix: dict, vendor_price: pd.DataFrame | None, close_split: pd.DataFrame) -> pd.DataFrame:
    """Fold the registered LEVEL wedges into `S`, re-measuring each one first.

    These are the cases `S` is structurally BLIND to: Yahoo adjusted the price and its splits
    feed never said so, so a factor derived from that feed reads exactly 1.0. IP is the
    clearest -- 107 of its 127 filing rows fail invariant 1 and `prices_splits` has no IP row
    at all.

    ⚠ NEVER TOUCHES A RETURN. It multiplies `S`, which reaches `cube_part_prices.level_factor`
    and every market cap built from it, and nothing else. That is the whole distinction from
    the two repairs above: a smooth back-adjustment leaves returns CORRECT and only the level
    short, so moving the returns would break what is currently right.

    THE VERIFICATION IS THE POINT. `vendor_price` (`fundamentals_sharadar.price`) is the only
    independent statement of what a stock actually traded at, so the observed wedge is
    `median(price / close_split)` over the segment's own rows. An entry whose wedge has moved
    outside `BUGFIX_WEDGE_TOL` is REFUSED rather than applied: a moved wedge means either
    Yahoo fixed it (the repair is no longer needed) or something else changed (the repair is
    no longer correct), and both call for a human rather than a silent multiply.

    ⚠ This makes invariant 2 partly SELF-REFERENTIAL for the registered tickers -- the factor
    is derived from the same vendor price the invariant scores against. The independent
    alternative is to derive the wedge from the spun-off child's own price via
    `sharadar_actions.contraticker`, which needs price history for securities outside the
    universe and is therefore its own task.
    """
    entries = bugfix.get("level_factor") or {}
    if not entries:
        return factor
    out = factor.copy()
    for ticker, spec in entries.items():
        if ticker not in out.columns:
            continue
        # ⚠ A SEGMENT IS A WINDOW, NOT A PREFIX, and `[lower, before)` is what makes both
        # halves of this loop correct. Measured on the live tables: IP reads 1.07078 before
        # its Veritiv spinoff and 1.05600 between Veritiv and Sylvamo, and HBAN steps
        # 1.61051 / 1.46410 / 1.33100 / 1.21000 / 1.10005 through its five stock dividends.
        # Every one of those is the WHOLE wedge for its own era, so segments must not
        # compound -- and an unbounded window would re-measure the era before it, which
        # skipped three of HBAN's five entries and applied IP's second on a 1.4% coincidence.
        lower = pd.Timestamp.min
        for segment in sorted(spec.get("segments", []), key=lambda s: s["before"]):
            before, claimed = pd.Timestamp(segment["before"]), float(segment["factor"])
            observed = observed_wedge(ticker, before, vendor_price, close_split, since=lower)
            window, lower = (out.index >= lower) & (out.index < before), before
            if observed is None:
                logger.info(
                    "price bugfix: %s < %s SKIPPED -- no vendor price in the segment, so the defect cannot be re-verified", ticker, before.date()
                )
                continue
            if abs(observed / claimed - 1.0) > BUGFIX_WEDGE_TOL:
                logger.info(
                    "price bugfix: %s < %s SKIPPED -- the defect is GONE or CHANGED: observed wedge %.5f, register says %.5f. Re-measure the entry.",
                    ticker,
                    before.date(),
                    observed,
                    claimed,
                )
                continue
            eligible = window & close_split[ticker].reindex(out.index).notna()
            out.loc[eligible, ticker] = out.loc[eligible, ticker] * claimed
            logger.info(
                "price bugfix: %s < %s level x%.5f APPLIED to %d bar(s) -- observed wedge %.5f (%s)",
                ticker,
                before.date(),
                claimed,
                int(eligible.sum()),
                observed,
                segment.get("event", "no event recorded"),
            )
    return out


def as_traded_close(
    close_split: pd.DataFrame | None,
    level_factor: pd.DataFrame | None,
    split_events: pd.DataFrame | ShareBasis,
) -> pd.DataFrame | None:
    """Restore quoted-security prices using vendor restatements and evidenced quote units."""
    if close_split is None or level_factor is None:
        return None
    df_close = close_split * level_factor.reindex_like(close_split)
    if isinstance(split_events, ShareBasis):
        basis = split_events
        events = basis.vendor
    else:
        basis = None
        events = split_events
    factors = suffix_factor(events, list(df_close.columns), df_close.index.to_numpy(dtype="datetime64[ns]"))
    for ticker, factor in factors.items():
        df_close[ticker] *= factor
    if basis is not None:
        for ticker in basis.policies.keys() & set(df_close.columns):
            df_close[ticker] *= basis.factor(pd.Series(ticker, index=df_close.index), pd.Series(df_close.index, index=df_close.index), "units")
    return df_close.where(np.isfinite(df_close) & df_close.gt(0))


def _period_close(close: pd.DataFrame, periods: pd.Series, max_age_days: int) -> pd.Series:
    """Last observed valuation-day quote, bounded by age and never from the future."""
    wanted = pd.DatetimeIndex(sorted(periods.dropna().unique()))
    df_quotes = close.sort_index().where(np.isfinite(close) & close.gt(0))
    union = df_quotes.index.union(wanted)
    df_asof = df_quotes.reindex(union).ffill().reindex(wanted)
    for ticker in df_quotes:
        observed = pd.Series(df_quotes.index, index=df_quotes.index).where(df_quotes[ticker].notna())
        last = observed.reindex(union).ffill().reindex(wanted)
        df_asof[ticker] = df_asof[ticker].where((pd.Series(wanted, index=wanted) - last).le(pd.Timedelta(days=max_age_days)))
    result = cast(pd.Series, df_asof.stack(future_stack=True))
    result.index = result.index.set_names(["period", "ticker"])
    return result


def rebase_security_quantities(
    basis: ShareBasis,
    quantities: pd.Series,
    tickers: pd.Series,
    source_dates: pd.Series,
    target_dates: pd.Series,
    ordinary_per_source: pd.Series,
) -> pd.Series:
    """Convert declared source quantities into target-date canonical security units.

    ordinary_per_source is 1 for ordinary counts, U(source) for canonical quotes,
    or an independently evidenced filed-class conversion. Unknown units stay null.
    """
    if not quantities.index.equals(ordinary_per_source.index):
        raise ValueError("Source quantity units must be index-aligned")
    units = pd.to_numeric(ordinary_per_source, errors="coerce")
    supported = np.isfinite(units) & units.gt(0)
    ordinary = pd.to_numeric(quantities, errors="coerce") * units.where(supported)
    rebased = basis.rebase(ordinary, tickers, source_dates, target_dates)
    result = rebased / basis.factor(tickers, target_dates, "units")
    return result.where(np.isfinite(result) & (ordinary.eq(0) | result.ne(0)))


def rebase_security_prices(
    basis: ShareBasis,
    prices: pd.Series,
    tickers: pd.Series,
    source_dates: pd.Series,
    target_dates: pd.Series,
    ordinary_per_source: pd.Series,
) -> pd.Series:
    """Apply the inverse quantity conversion to preserve a matched price/count value."""
    multiplier = rebase_security_quantities(basis, pd.Series(1.0, index=prices.index), tickers, source_dates, target_dates, ordinary_per_source)
    converted = pd.to_numeric(prices, errors="coerce") / multiplier
    return converted.where(np.isfinite(converted) & converted.gt(0))


def security_quantity_basis(basis: ShareBasis, index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    """Return GÃ—U on a canonical quote grid, keeping quantity and price wedges separate."""
    df_basis = pd.DataFrame(1.0 if basis.actual_available else np.nan, index=index, columns=columns, dtype="float64")
    if not basis.actual_available:
        return df_basis
    df_events = basis.actual
    invalid = df_events[["ticker", "date", "value"]].isna().any(axis=1)
    values = pd.to_numeric(df_events["value"], errors="coerce")
    if (invalid | ~np.isfinite(values) | values.le(0)).any():
        raise ValueError("Genuine quantity events require valid keys and positive finite factors")
    factors = suffix_factor(df_events, list(columns), index.to_numpy(dtype="datetime64[ns]"))
    for ticker in columns:
        units = basis.factor(pd.Series(ticker, index=index), pd.Series(index, index=index), "units")
        df_basis[ticker] = factors.get(str(ticker), np.ones(len(index))) * units
    return df_basis.where(np.isfinite(df_basis) & df_basis.gt(0))


def rebase_security_frame(basis: ShareBasis, df_quantities: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    """Express each dated canonical-security quantity in one target-date security unit."""
    df_result = df_quantities.copy()
    source = pd.Series(df_result.index, index=df_result.index)
    target = pd.Series(target_date, index=df_result.index)
    for ticker in df_result:
        names = pd.Series(str(ticker), index=df_result.index)
        units = basis.factor(names, source, "units")
        df_result[ticker] = rebase_security_quantities(basis, df_result[ticker], names, source, target, units)
    return df_result
