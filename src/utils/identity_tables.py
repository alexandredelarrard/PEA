"""The one reader of the published identity tables outside the identity build, and the 13F CUSIP tagging rule.

`read_identity` and `identity_columns` are the only way a fetcher, a cube part or a shared helper reads
`universe_cik_symbol`, `universe_cusip`, `symbol_tenure` or `cusip_ticker_map` (the static test
`test_identity_reads_only.py` enforces it); the identity build and the `Identity` resolver own their own reads.

`tag_cusips` is the two-layer 13F rule (user decision X2 + the superinvestor-book correction): a CUSIP of a
universe company is tagged by its dated `universe_cusip` line (canonical, predecessor or secondary class) with the
company's universe ticker, so every share class of a company lands on one ticker; any other CUSIP keeps its
market-wide `cusip_ticker_map` ticker, so a manager book keeps its holdings outside the universe.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.constants.constants import CANONICAL_ROLES, SECONDARY_CLASS
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore

#: The identity tables this reader serves.
IDENTITY_TABLES = (Tables.universe_cik_symbol, Tables.universe_cusip, Tables.symbol_tenure, Tables.cusip_ticker_map)
#: `universe_cusip` roles whose dated line tags a 13F CUSIP with its company (every share class of the company).
TAGGED_ROLES = (*CANONICAL_ROLES, SECONDARY_CLASS)
#: The `universe_cusip` columns `tag_cusips` reads.
LINE_COLUMNS = ["cusip", "canonical_company", "lineage_role", "market_symbol", "valid_from", "valid_to"]


def read_identity(store: DataStore, table: Table, **load: Any) -> pd.DataFrame | None:
    """`store.load(table, **load)` for one of the identity tables; raises for any other table."""
    if table not in IDENTITY_TABLES:
        raise ValueError(f"read_identity serves {[t.name for t in IDENTITY_TABLES]}, not {table.name}")
    return store.load(table, **load)


def identity_columns(store: DataStore, table: Table) -> list[str]:
    """The stored columns of one identity table (empty when absent)."""
    if table not in IDENTITY_TABLES:
        raise ValueError(f"identity_columns serves {[t.name for t in IDENTITY_TABLES]}, not {table.name}")
    return list(store.columns(table))


def load_cusip_lines(store: DataStore) -> pd.DataFrame:
    """The `universe_cusip` lines `tag_cusips` reads; an empty frame before the first identity build."""
    lines = read_identity(store, Tables.universe_cusip, columns=LINE_COLUMNS, optional=True)
    return lines if lines is not None else pd.DataFrame(columns=LINE_COLUMNS)


def load_cusip_map(store: DataStore) -> pd.DataFrame:
    """The stored market-wide `cusip_ticker_map` (`cusip`, `ticker`); an empty frame when absent."""
    cmap = read_identity(store, Tables.cusip_ticker_map, columns=["cusip", "ticker"], optional=True)
    return cmap if cmap is not None else pd.DataFrame(columns=["cusip", "ticker"])


def tag_cusips(holdings: pd.DataFrame, lines: pd.DataFrame, cmap: pd.DataFrame | None, date_col: str = "period") -> pd.Series:
    """The ticker of each `holdings` row (`cusip`, `date_col`), aligned on its index (string dtype); NA where nothing tags it.

    Layer 1: the one company whose tagged `universe_cusip` line (canonical, predecessor or secondary class) holds the
    date. Layer 2, every other row: the `cusip_ticker_map` ticker, unless it names a company of `universe_cusip` or a
    market symbol of the CUSIP's own tagged lines (a universe ticker, its classes and its former symbols, GOOG, FOX, EQR,
    are only reached through the dated lines, AC-009 and X2). An acquired constituent's or a warrant's
    CUSIP outside the tagged lines therefore keeps its own market ticker (AVB, OXY-WS), as a manager book needs.
    """
    # clean: one key per (cusip, day); the universe CUSIPs
    cusips = holdings["cusip"].astype("string").str.strip().str.upper().astype(object)
    days = pd.to_datetime(holdings[date_col], errors="coerce") if date_col in holdings else pd.Series(pd.NaT, index=holdings.index)
    keys = pd.DataFrame({"cusip": cusips.to_numpy(), "day": days.to_numpy()})
    pairs = keys.drop_duplicates(ignore_index=True)
    known = set(lines["cusip"].dropna().astype(str))
    company_names = set(lines["canonical_company"].dropna().astype(str))
    tagged = lines[lines["lineage_role"].isin(TAGGED_ROLES)].astype({"cusip": object})
    own_symbols = set(zip(tagged["cusip"].astype(str), tagged["market_symbol"].astype(str), strict=True))

    # compute: layer 1, the dated company line
    joined = pairs[pairs["cusip"].isin(known)].merge(tagged[["cusip", "canonical_company", "valid_from", "valid_to"]], on="cusip")
    start, end = pd.to_datetime(joined["valid_from"]), pd.to_datetime(joined["valid_to"])
    hit = joined[(start <= joined["day"]) & (end.isna() | (joined["day"] < end))]
    companies = hit.groupby(["cusip", "day"], as_index=False)["canonical_company"].agg(["nunique", "first"])
    companies["company"] = companies["first"].where(companies["nunique"].eq(1))

    # compute: layer 2, the market-wide map for every other CUSIP
    mapped = (
        cmap.dropna(subset=["cusip", "ticker"]).drop_duplicates("cusip", keep="last")
        if cmap is not None
        else pd.DataFrame(columns=["cusip", "ticker"])
    )
    market = pd.Series(mapped["ticker"].to_numpy(), index=mapped["cusip"].astype(str))

    # post-process: one ticker per pair, back on the holdings index
    pairs = pairs.merge(companies[["cusip", "day", "company"]], on=["cusip", "day"], how="left")
    fallback = pairs["cusip"].map(market)
    own = pd.Series([(c, t) in own_symbols for c, t in zip(pairs["cusip"], fallback, strict=True)], index=pairs.index)
    pairs["ticker"] = pairs["company"].where(pairs["company"].notna(), fallback.where(~fallback.isin(company_names) & ~own))
    out = keys.merge(pairs[["cusip", "day", "ticker"]], on=["cusip", "day"], how="left")["ticker"]
    return pd.Series(out.to_numpy(dtype=object), index=holdings.index).astype("str")


@dataclass(frozen=True)
class CusipTags:
    """The two inputs of `tag_cusips`, read once per run: the `universe_cusip` lines and the `cusip_ticker_map` rows."""

    lines: pd.DataFrame
    cmap: pd.DataFrame

    @classmethod
    def load(cls, store: DataStore) -> CusipTags:
        """Both tables from the store (empty frames when absent)."""
        return cls(load_cusip_lines(store), load_cusip_map(store))

    @classmethod
    def of_map(cls, cmap: pd.DataFrame | None) -> CusipTags:
        """A market-wide map alone, no company lines."""
        return cls(pd.DataFrame(columns=LINE_COLUMNS), cmap if cmap is not None else pd.DataFrame(columns=["cusip", "ticker"]))

    def tag(self, holdings: pd.DataFrame, universe: Collection[str] | None = None) -> pd.Series:
        """`tag_cusips` of `holdings` (dated by `period`); only the tickers of `universe` when given."""
        tickers = tag_cusips(holdings, self.lines, self.cmap)
        return tickers if universe is None else tickers.where(tickers.isin(set(universe)), None)


def tag_layers(holdings: pd.DataFrame, lines: pd.DataFrame) -> pd.Series:
    """`universe_cusip` (a CUSIP with a line) or `cusip_ticker_map`: the table a row's CUSIP is known to."""
    known = set(lines["cusip"].dropna().astype(str))
    cusips = holdings["cusip"].astype("string").str.strip().str.upper()
    return pd.Series(cusips.isin(known).map({True: Tables.universe_cusip.name, False: Tables.cusip_ticker_map.name}), index=holdings.index)


def universe_cusips(lines: pd.DataFrame, tickers: Collection[str]) -> frozenset[str]:
    """Every CUSIP a tagged `universe_cusip` line dates to one of `tickers` (any date)."""
    rows = lines[lines["lineage_role"].isin(TAGGED_ROLES) & lines["canonical_company"].isin(set(tickers))]
    return frozenset(rows["cusip"].dropna().astype(str))
