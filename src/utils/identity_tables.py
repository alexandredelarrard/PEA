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
#: The `universe_cusip` columns `tag_cusips` and `company_rows` read.
LINE_COLUMNS = ["cusip", "canonical_company", "lineage_role", "market_symbol", "conversion_ratio", "valid_from", "valid_to"]
#: 13F share legs counted in the canonical class (x the dated ratio); every other amount is summed as filed.
SHARE_LEGS = ("shares", "call_shares", "put_shares")
#: 13F amounts of one position row.
AMOUNT_LEGS = (*SHARE_LEGS, "value_usd", "call_value", "put_value", "debt_prn", "debt_value", "other_value")
#: The 13F company row's key: one row per manager x accession x company (U-67).
COMPANY_KEY = ["cik", "accession_number", "ticker"]


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


def company_rows(book: pd.DataFrame, lines: pd.DataFrame, cmap: pd.DataFrame | None) -> pd.DataFrame:
    """`book` (one 13F accession, one row per CUSIP) as stored in `sec13f_hr` (U-67): tagged by `tag_cusips`; the rows a
    dated company line tags make ONE row per company, on the company's canonical CUSIP of the period, its share legs in
    canonical-class shares (x the line's dated `conversion_ratio`), its dollar legs summed as filed (unknown if one is
    unknown). Every other row keeps its own CUSIP and market ticker."""
    # compute: each row's tag, its dated line (ratio) and its company's canonical CUSIP on the period
    tagged = book.assign(ticker=tag_cusips(book, lines, cmap))
    hits = company_lines(tagged, lines)
    if hits.empty:
        return tagged

    # compute: canonical-class shares, the target CUSIP first in each company group
    rows = tagged.loc[hits.index].assign(_target=hits["target"], _first=hits["target"].ne(tagged.loc[hits.index, "cusip"]))
    for leg in SHARE_LEGS:
        rows[leg] = pd.to_numeric(rows[leg], errors="coerce") * hits["ratio"]
    rows = rows.sort_values([*COMPANY_KEY, "_first", "cusip"], kind="stable")

    # merge: one row per company (metadata of its first row, amounts summed, unknown when one row is unknown)
    grouped = rows.groupby(COMPANY_KEY, sort=False)
    merged = grouped.first()
    amounts = [leg for leg in (*AMOUNT_LEGS, *(c for c in rows.columns if c.startswith("raw_"))) if leg in rows]
    merged[amounts] = grouped[amounts].sum(min_count=1).where(grouped[amounts].count().eq(grouped.size(), axis=0))
    merged["cusip"] = merged["_target"]
    merged = merged.reset_index().drop(columns=["_target", "_first"])
    return pd.concat([tagged.drop(index=hits.index), merged[tagged.columns]], ignore_index=True)


def company_lines(tagged: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Per row of `tagged` that a dated company line tags (its index): the line's `ratio` and `target`, the canonical
    CUSIP of the company on the period (canonical current before predecessor, then the latest start; the row's own CUSIP
    when the company has no canonical line that day)."""
    # clean: the rows and the tagged lines with their dates
    rows = pd.DataFrame({"row": tagged.index, "cusip": tagged["cusip"].astype(str).str.strip().str.upper(), "ticker": tagged["ticker"]})
    rows["day"] = pd.to_datetime(tagged["period"]).to_numpy()
    work = lines[lines["lineage_role"].isin(TAGGED_ROLES)].assign(
        start=pd.to_datetime(lines["valid_from"]), end=pd.to_datetime(lines["valid_to"]), cusip=lines["cusip"].astype(str)
    )

    # compute: the line holding each row's date for its own company, then the company's canonical CUSIP that day
    held = rows.merge(work, left_on=["cusip", "ticker"], right_on=["cusip", "canonical_company"])
    held = held[held["start"].isna() | (held["start"] <= held["day"])]
    held = held[held["end"].isna() | (held["day"] < held["end"])].drop_duplicates("row")
    canonical = work[work["lineage_role"].isin(CANONICAL_ROLES)].assign(rank=lambda w: w["lineage_role"].ne(CANONICAL_ROLES[1]))
    pairs = held[["row", "canonical_company", "day"]].merge(canonical[["canonical_company", "cusip", "rank", "start", "end"]], on="canonical_company")
    pairs = pairs[(pairs["start"].isna() | (pairs["start"] <= pairs["day"])) & (pairs["end"].isna() | (pairs["day"] < pairs["end"]))]
    target = pairs.sort_values(["rank", "start"], ascending=[True, False], na_position="last").drop_duplicates("row").set_index("row")["cusip"]

    # post-process: one (ratio, target) per tagged row
    out = held.set_index("row")[["conversion_ratio", "cusip"]].rename(columns={"conversion_ratio": "ratio"})
    out["ratio"] = pd.to_numeric(out["ratio"], errors="coerce").fillna(1.0)
    out["target"] = target.reindex(out.index).fillna(out["cusip"])
    return out[["ratio", "target"]]


@dataclass(frozen=True)
class CusipTags:
    """The two inputs of `tag_cusips`, read once per run: the `universe_cusip` lines and the `cusip_ticker_map` rows."""

    lines: pd.DataFrame
    cmap: pd.DataFrame

    @classmethod
    def load(cls, store: DataStore) -> CusipTags:
        """Both tables from the store (empty frames when absent)."""
        return cls(load_cusip_lines(store), load_cusip_map(store))

    def tag(self, holdings: pd.DataFrame, universe: Collection[str] | None = None) -> pd.Series:
        """`tag_cusips` of `holdings` (dated by `period`); only the tickers of `universe` when given."""
        tickers = tag_cusips(holdings, self.lines, self.cmap)
        return tickers if universe is None else tickers.where(tickers.isin(set(universe)), None)


def universe_cusips(lines: pd.DataFrame, tickers: Collection[str]) -> frozenset[str]:
    """Every CUSIP a tagged `universe_cusip` line dates to one of `tickers` (any date)."""
    rows = lines[lines["lineage_role"].isin(TAGGED_ROLES) & lines["canonical_company"].isin(set(tickers))]
    return frozenset(rows["cusip"].dropna().astype(str))
