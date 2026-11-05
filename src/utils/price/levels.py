"""Shared historical quote reconstruction and guarded level factors."""

from __future__ import annotations

import logging
from collections.abc import Sequence

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


def _validate_quantity_events(df_events: pd.DataFrame) -> None:
    """Refuse malformed quantity evidence before suffix products can discard it."""
    if df_events.empty:
        return
    if not {"ticker", "date", "value"}.issubset(df_events):
        raise ValueError("Quantity events require ticker, date and value")
    values = pd.to_numeric(df_events["value"], errors="coerce")
    invalid = df_events[["ticker", "date", "value"]].isna().any(axis=1)
    invalid |= pd.to_datetime(df_events["date"], errors="coerce").isna() | df_events["ticker"].astype("string").str.strip().eq("")
    if (invalid | ~np.isfinite(values) | values.le(0)).any():
        raise ValueError("Quantity events require valid keys and positive finite factors")


def vendor_identity_support(basis: ShareBasis, tickers: pd.Series, source_dates: pd.Series) -> pd.Series:
    """Prove absence of future nonunit restatements; cancelling products are insufficient."""
    if not tickers.index.equals(source_dates.index):
        raise ValueError("Vendor ticker and source date must be index-aligned")
    _validate_quantity_events(basis.vendor)
    source = pd.to_datetime(source_dates, errors="coerce").dt.normalize()
    supported = source.notna() & tickers.notna() & tickers.astype("string").str.strip().ne("") & basis.actual_available
    if basis.vendor.empty:
        return supported
    df_events = basis.vendor.loc[pd.to_numeric(basis.vendor["value"]).ne(1)].copy()
    if df_events.empty:
        return supported
    df_events["date"] = pd.to_datetime(df_events["date"])
    last = tickers.map(df_events.groupby("ticker")["date"].max())
    return supported & (last.isna() | source.ge(last))


def vendor_observation_factor(basis: ShareBasis, tickers: pd.Series, source_dates: pd.Series, vintages: pd.Series) -> pd.Series:
    """Vendor restatements between a public observation and its evidenced representation vintage."""
    if any(not tickers.index.equals(values.index) for values in (source_dates, vintages)):
        raise ValueError("Vendor ticker, source date and vintage must be index-aligned")
    _validate_quantity_events(basis.vendor)
    source = pd.to_datetime(source_dates, errors="coerce").dt.normalize()
    vintage = pd.to_datetime(vintages, errors="coerce").dt.normalize()
    factor = basis.factor(tickers, source, "vendor") / basis.factor(tickers, vintage, "vendor")
    supported = source.notna() & vintage.notna() & vintage.ge(source) & tickers.notna() & tickers.astype("string").str.strip().ne("")
    return factor.where(supported & basis.actual_available & np.isfinite(factor) & factor.gt(0))


def security_quantity_basis(basis: ShareBasis, index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    """Return GÃ—U on a canonical quote grid, keeping quantity and price wedges separate."""
    df_basis = pd.DataFrame(1.0 if basis.actual_available else np.nan, index=index, columns=columns, dtype="float64")
    if not basis.actual_available:
        return df_basis
    df_events = basis.actual
    _validate_quantity_events(df_events)
    factors = suffix_factor(df_events, list(columns), index.to_numpy(dtype="datetime64[ns]"))
    for ticker in columns:
        units = basis.factor(pd.Series(ticker, index=index), pd.Series(index, index=index), "units")
        df_basis[ticker] = factors.get(str(ticker), np.ones(len(index))) * units
    return df_basis.where(np.isfinite(df_basis) & df_basis.gt(0))


def rebase_vendor_quantities(
    basis: ShareBasis, quantities: pd.Series, tickers: pd.Series, source_dates: pd.Series, target_dates: pd.Series
) -> pd.Series:
    """Undo the source's F restatement, then carry genuine quantities in target security units."""
    factor = basis.factor(tickers, source_dates, "vendor")
    original = pd.to_numeric(quantities, errors="coerce") / factor.where(np.isfinite(factor) & factor.gt(0))
    result = rebase_security_quantities(basis, original, tickers, source_dates, target_dates, pd.Series(1.0, index=quantities.index))
    return result if basis.actual_available else result.where(quantities.eq(0))


def split_cum_product(splits: pd.DataFrame | None, tickers: pd.Series, dates: pd.Series) -> np.ndarray:
    """`PROD(ratio)` over each ticker's events dated at or before each date, aligned to the inputs.

    1.0 before a ticker's first event and for a ticker with none; a NaT date resolves to 1.0.
    An event dated d is already in effect on d (effective-date convention).
    """
    out = np.ones(len(tickers), dtype="float64")
    if splits is None or splits.empty or not len(tickers) or not {"date", "ticker", "ratio"}.issubset(splits.columns):
        return out

    # Positive events only, both sides on ns midnight (mixed resolutions mis-place searchsorted).
    df_events = splits.dropna(subset=["date", "ticker", "ratio"]).copy()
    df_events["ratio"] = pd.to_numeric(df_events["ratio"], errors="coerce")
    df_events = df_events[df_events["ratio"] > 0]
    df_events["date"] = pd.to_datetime(df_events["date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
    df_events = df_events.sort_values(["ticker", "date"])
    df_events["cum"] = df_events.groupby("ticker")["ratio"].cumprod()
    want = pd.Series(np.asarray(tickers, dtype=object)).astype(str).reset_index(drop=True)
    when = pd.to_datetime(pd.Series(np.asarray(dates)), errors="coerce").astype("datetime64[ns]").fillna(pd.Timestamp("1900-01-01")).to_numpy()
    positions = want.groupby(want).indices

    # One groupby over events; prefix products looked up with `side="right"` (date <= d).
    for ticker, df_group in df_events.groupby("ticker", sort=False):
        rows = positions.get(str(ticker))
        if rows is None:
            continue
        cum = np.concatenate(([1.0], df_group["cum"].to_numpy()))
        out[rows] = cum[df_group["date"].to_numpy().searchsorted(when[rows], side="right")]
    return out


def security_basis(basis: ShareBasis, tickers: pd.Series, dates: pd.Series) -> pd.Series:
    """Forward quantity basis F(d): genuine events dated at or before d times the quote-unit change at or before d.

    A quoted-security quantity converts between dates by `q_t = q_s * F_t / F_s`; null where events are unavailable
    or units unknown. Later events never change F before their date.
    """
    products = split_cum_product(basis.actual.rename(columns={"value": "ratio"}), tickers, dates)
    start = pd.Series(pd.Timestamp("1900-01-01"), index=dates.index)
    units = basis.factor(tickers, start, "units") / basis.factor(tickers, dates, "units")
    result = pd.Series(products, index=dates.index) * units
    return result.where(np.isfinite(result) & result.gt(0) & basis.actual_available)


def security_basis_frame(basis: ShareBasis, index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    """`security_basis` on a date x ticker grid."""
    keys = pd.MultiIndex.from_product([index, columns])
    values = security_basis(basis, pd.Series(keys.get_level_values(1).astype(str)), pd.Series(keys.get_level_values(0)))
    return pd.DataFrame(values.to_numpy().reshape(len(index), len(columns)), index=index, columns=columns)


def convert_quantities(
    quantities: pd.DataFrame | pd.Series | np.ndarray, basis_from: pd.Series | np.ndarray, basis_to: pd.DataFrame | pd.Series | np.ndarray
) -> pd.DataFrame | pd.Series | np.ndarray:
    """Carry quoted-security quantities between two forward bases: `q * F_to / F_from` (null where either is null)."""
    return quantities * basis_to / basis_from


def lookup_asof(df_values: pd.DataFrame, tickers: pd.Series, dates: pd.Series) -> pd.Series:
    """Per (ticker, date), the value saved on the last grid date at or before the date; a null there stays null."""
    index = pd.DatetimeIndex(df_values.index)
    rows = index.searchsorted(pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]"), side="right") - 1
    cols = df_values.columns.get_indexer(tickers.astype(str))
    found = (rows >= 0) & (cols >= 0)
    values = df_values.to_numpy(dtype="float64")[np.maximum(rows, 0), np.maximum(cols, 0)]
    return pd.Series(np.where(found, values, np.nan), index=dates.index)


def compatible_tape_support(basis: ShareBasis, index: pd.DatetimeIndex, columns: pd.Index, terminal_date: pd.Timestamp) -> pd.DataFrame:
    """Prove compatibility of the retained adjusted-price/terminal-quoted-tape pair."""
    df_supported = pd.DataFrame(False, index=index, columns=columns)
    if not basis.actual_available:
        return df_supported
    _validate_quantity_events(basis.actual)
    _validate_quantity_events(basis.vendor)
    stamps = index.to_numpy(dtype="datetime64[ns]")
    actual = suffix_factor(basis.actual, list(columns), stamps)
    vendor = suffix_factor(basis.vendor, list(columns), stamps)
    names = pd.Series(columns.astype(str))
    terminal = pd.Series(terminal_date, index=names.index)
    target = basis.factor(names, terminal, "actual") * basis.factor(names, terminal, "units")
    for position, ticker in enumerate(columns):
        genuine = actual.get(str(ticker), np.ones(len(index)))
        restatement = vendor.get(str(ticker), np.ones(len(index)))
        units = basis.factor(pd.Series(str(ticker), index=index), pd.Series(index, index=index), "units")
        supported = np.isfinite(units) & units.gt(0) & np.isfinite(target.iloc[position]) & (target.iloc[position] > 0)
        supported &= np.isfinite(genuine) & (genuine > 0) & np.isfinite(restatement) & (restatement > 0)
        df_supported[ticker] = supported & np.isclose(restatement * target.iloc[position], genuine, rtol=1e-12, atol=0)
    return df_supported


def dollar_volume(
    df_close: pd.DataFrame,
    df_volume: pd.DataFrame,
    df_factor: pd.DataFrame | None,
    df_quote: pd.DataFrame | None,
    df_tape_support: pd.DataFrame | None,
) -> pd.DataFrame:
    """Value the compatible adjusted quote/tape pair with the published level factor."""
    if df_factor is None or df_quote is None or df_tape_support is None:
        return pd.DataFrame(np.nan, index=df_close.index, columns=df_close.columns, dtype="float64")
    df_tape = df_volume.reindex_like(df_close)
    df_level = df_factor.reindex_like(df_close)
    df_value = df_close * df_tape * df_level
    supported = df_close.gt(0) & df_tape.ge(0) & df_level.gt(0)
    supported &= np.isfinite(df_close) & np.isfinite(df_tape) & np.isfinite(df_level) & np.isfinite(df_value)
    df_quoted = df_quote.reindex_like(df_close)
    supported &= np.isfinite(df_quoted) & df_quoted.gt(0) & df_tape_support.reindex_like(df_close).eq(True)
    return df_value.where(supported)
