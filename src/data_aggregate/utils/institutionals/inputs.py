"""Input loading for the institutional cube part."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from src.context import Context
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.data_aggregate.utils.common.price_frames import PriceFrames, load_price_frames
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore

SHARES_OUT_COLUMNS = ("ticker", "as_of", "sharesOutstanding", "sharesOutstandingPit")


def _cik(value: object) -> str:
    digits = "".join(character for character in str(value) if character.isdigit())
    return digits.zfill(10) if digits else ""


def load_full_price_frames(
    store: DataStore,
    context: Context,
    fields: Sequence[str],
) -> PriceFrames:
    """Load full-calendar price inputs from the already-persisted peer dependency."""
    return load_price_frames(
        store,
        peers=load_peers_or_raise(context),
        fields=fields,
        since=None,
    )


def load_source(
    store: DataStore,
    log: logging.Logger,
    table: Table,
    universe: Sequence[str] | None = None,
) -> pd.DataFrame | None:
    """Load one source projected to its registry columns and scoped to the universe.

    The universe cut defines the cross-section rather than optimizing the read. Source
    tables can contain names outside the cube universe; excluding them at the read keeps
    every ``_xs`` leg on the same ticker denominator as the other cube parts.
    """
    where: dict[str, list[str]] | None = None
    if universe is not None and table.ticker_col:
        where = {table.ticker_col: sorted(set(map(str, universe)))}
        _report_off_universe(store, log, table, universe)

    frame = store.load(table, project=True, where=where, optional=True)
    if frame is None:
        log.warning("%s is absent or empty -> its features are skipped.", table.name)
    else:
        log.info("Loaded %s: %s rows x %s cols", table.name, len(frame), len(frame.columns))
    return frame


def _report_off_universe(
    store: DataStore,
    log: logging.Logger,
    table: Table,
    universe: Sequence[str],
) -> None:
    """Log source tickers excluded by the universe-scoped read."""
    column = table.ticker_col
    if not column:
        return
    present = {str(ticker) for ticker in store.distinct(table, column)}
    off_universe = sorted(present - set(map(str, universe)))
    if not off_universe:
        return
    log.info(
        "%s: %s of its %s ticker(s) are outside the %s-name universe (%s) -- not read, so the `_xs` cross-section is the cube's",
        table.name,
        len(off_universe),
        len(present),
        len(universe),
        ", ".join(off_universe[:15]) + (", ..." if len(off_universe) > 15 else ""),
    )


def load_shares_out(store: DataStore, log: logging.Logger) -> pd.DataFrame | None:
    """Load both share-count bases required by institutional market-cap features."""
    frame = store.load(
        Tables.fundamentals_history,
        columns=list(SHARES_OUT_COLUMNS),
        optional=True,
    )
    if frame is None:
        log.warning("No fundamentals history -> the market-cap-scaled ownership features are skipped.")
        return None
    return frame


def load_symbol_lineage(
    store: DataStore,
    log: logging.Logger,
    universe: Sequence[str],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, list[str]]:
    """Load current CIKs and every proven historical symbol for those issuers."""
    tickers = sorted(set(map(str, universe)))
    ticker_ciks = store.load(
        Tables.sp500_tickers,
        columns=("ticker", "cik"),
        where={"ticker": tickers},
        optional=True,
    )
    if ticker_ciks is None or ticker_ciks.empty:
        return None, ticker_ciks, tickers

    ciks = sorted({_cik(value) for value in ticker_ciks["cik"] if _cik(value)})
    if not ciks:
        return None, ticker_ciks, tickers
    symbol_tenure = store.load(
        Tables.symbol_tenure,
        columns=("symbol", "issuer_cik", "valid_from", "valid_to"),
        where={"issuer_cik": ciks},
        optional=True,
    )
    aliases = sorted(set(tickers) | (set(symbol_tenure["symbol"].astype(str)) if symbol_tenure is not None else set()))
    log.info("Symbol lineage: %s current tickers -> %s proven source symbols", len(tickers), len(aliases))
    return symbol_tenure, ticker_ciks, aliases
