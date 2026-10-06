"""Input loading for the institutional cube part."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from src.constants.constants import CANONICAL_ROLES, SECONDARY_CLASS
from src.context import Context
from src.data_aggregate.utils.common.level_basis import apply_volume_scale
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.data_aggregate.utils.common.price_frames import PriceFrames, load_price_frames
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore
from src.utils.string import pad_cik, yahoo_symbol

SHARES_OUT_COLUMNS = ("ticker", "as_of", "sharesOutstanding", "sharesOutstandingPit")
#: `symbol_tenure` sources behind the proven-tenure mask; cover-page `dei` rows stay out so features do not move.
TENURE_SOURCES = ("form345", "manual")
#: Columns a concurrently traded common class (`security_master` role `secondary_class`) needs for its volume.
CLASS_LINE_COLUMNS = ("canonical_company", "market_symbol", "conversion_ratio", "valid_from", "valid_to")
CLASS_BAR_COLUMNS = ("ticker", "date", "volume")


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
    where: dict[str, Any] | None = None,
) -> pd.DataFrame | None:
    """Load one source projected to its registry columns and scoped to the universe.

    The universe cut defines the cross-section rather than optimizing the read. Source
    tables can contain names outside the cube universe; excluding them at the read keeps
    every ``_xs`` leg on the same ticker denominator as the other cube parts. `where` adds row filters.
    """
    filters: dict[str, Any] = dict(where or {})
    if universe is not None and table.ticker_col:
        filters[table.ticker_col] = sorted(set(map(str, universe)))
        _report_off_universe(store, log, table, universe)

    frame = store.load(table, project=True, where=filters or None, optional=True)
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
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load the issuer lineage (`form345`/`manual` tenure on the roster CIKs) used to validate already-canonical source rows."""
    tickers = sorted(set(map(str, universe)))
    ticker_ciks = store.load(
        Tables.sp500_tickers,
        columns=("ticker", "cik"),
        where={"ticker": tickers},
        optional=True,
    )
    if ticker_ciks is None or ticker_ciks.empty:
        return None, ticker_ciks

    ciks = sorted({pad_cik(value) for value in ticker_ciks["cik"] if pad_cik(value)})
    if not ciks:
        return None, ticker_ciks
    symbol_tenure = store.load(
        Tables.symbol_tenure,
        columns=("symbol", "issuer_cik", "valid_from", "valid_to"),
        where={"issuer_cik": ciks, "source": list(TENURE_SOURCES)},
        optional=True,
    )
    aliases = len(set(symbol_tenure["symbol"].astype(str))) if symbol_tenure is not None else 0
    log.info("Symbol lineage: %s current tickers backed by %s proven historical symbols", len(tickers), aliases)
    return symbol_tenure, ticker_ciks


def load_secondary_classes(
    store: DataStore,
    log: logging.Logger,
    universe: Sequence[str],
    bugfix: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """The universe companies' `secondary_class` lines from `security_master` and those classes' `prices` volume bars.

    `prices` is read by exactly the classes' Yahoo symbols, which the universe filter of its other readers would drop.
    The price register's `volume_scale` entries (`bugfix`) repair a vendor volume unit defect on those bars.
    None when the master is absent or holds no class line, so the features keep the canonical tape.
    """
    lines = store.load(
        Tables.security_master,
        columns=list(CLASS_LINE_COLUMNS),
        where={"lineage_role": SECONDARY_CLASS, "canonical_company": sorted(set(map(str, universe)))},
        optional=True,
    )
    if lines is None or lines.empty:
        log.info("No secondary share-class lines in security_master -> ADV20 and RegSHO coverage use the canonical tape only.")
        return None
    symbols = sorted({yahoo_symbol(symbol) for symbol in lines["market_symbol"]})
    bars = store.load(Tables.prices, columns=list(CLASS_BAR_COLUMNS), where={"ticker": symbols}, optional=True)
    if bars is None:
        bars = pd.DataFrame(columns=list(CLASS_BAR_COLUMNS))
    if bugfix:
        bars = apply_volume_scale(bars, bugfix, log.info)
    log.info(
        "Secondary share classes: %s line(s), %s symbol(s) of %s companies, %s volume bar(s)",
        len(lines),
        len(symbols),
        lines["canonical_company"].nunique(),
        len(bars),
    )
    return lines, bars


def load_insider_transactions(store: DataStore, log: logging.Logger, universe: Sequence[str] | None = None) -> pd.DataFrame | None:
    """`insider_transactions` rows of the companies' own history: `lineage_role` canonical (predecessor or current).
    Acquired-constituent rows and rows without a stamp are not read."""
    return load_source(store, log, Tables.insider_transactions, universe, where={"lineage_role": list(CANONICAL_ROLES)})
