"""
sync.py  (src/utils/tickers_universe/sync.py)
---------------------------------------------
`sync`: the universe writer of `universe_tickers`, one row per company ever in `universe.yml`.

* The yml decides `in_universe`: a ticker that leaves it keeps its row with `in_universe = false`; one that
  enters gets a row with CIK and name from the `sec_company_tickers` snapshot (Sharadar's `secfilings` CIK
  and name for a ticker the SEC no longer lists).
* `universe_added_on` = 2000-01-01 when the ticker already has prices (an established name), else `as_of`;
  it is stamped when a ticker enters or re-enters the yml.
* name, CIK and `universe_added_on` of a row in the yml are never rewritten.
* GICS follows the yml (`universe_classification`), `cusips` and the Sharadar category columns follow
  `sharadar_tickers`, every run; only rows that differ are written, so an unchanged night writes nothing.
* The identity columns belong to the identity step: the sync never writes them, except to create them
  empty when it builds the table.
* A yml ticker with no Sharadar row aborts the sync.
"""

from __future__ import annotations

import logging

import pandas as pd
from omegaconf import DictConfig

from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.string import pad_cik, yahoo_symbol
from src.utils.tickers_universe.universe import UniverseError, universe_classification, universe_config

logger = logging.getLogger(__name__)

ESTABLISHED_ADDED_ON = pd.Timestamp("2000-01-01")
CARRIED_COLUMNS = ["ticker", "name", "cik", "universe_added_on"]
#: written by the identity step; created empty by the sync's first build
IDENTITY_COLUMNS = ["company_id", "trading_symbol", "first_price_date", "ended_on", "status"]
#: sharadar_tickers column -> universe_tickers column
SHARADAR_COLUMNS = {
    "permaticker": "permaticker",
    "name": "sharadar_name",
    "exchange": "exchange",
    "category": "category",
    "sector": "sharadar_sector",
    "industry": "sharadar_industry",
    "sicsector": "sicsector",
    "sicindustry": "sicindustry",
    "famaindustry": "famaindustry",
    "scalemarketcap": "scalemarketcap",
    "scalerevenue": "scalerevenue",
    "location": "location",
    "isdelisted": "isdelisted",
}
SHARADAR_READ = ["ticker", "isdelisted", "lastupdated", "cusips", "secfilings", *[c for c in SHARADAR_COLUMNS if c != "isdelisted"]]


def sync(store: DataStore, config: DictConfig, as_of: pd.Timestamp) -> int:
    """Bring `universe_tickers` in line with `universe.yml`, `sec_company_tickers` and `sharadar_tickers`; returns the rows written."""
    # load the yml, the stored rows and the Sharadar rows of the yml tickers and their aliases
    tickers, aliases = universe_config(config)
    gics = universe_classification(config)
    stored = store.load(Tables.universe_tickers, optional=True)
    sharadar = _sharadar_rows(store, [*tickers, *(a for symbols in aliases.values() for a in symbols)])
    previous = stored if stored is not None else pd.DataFrame(columns=[*CARRIED_COLUMNS, "in_universe"])

    # check every yml ticker has a Sharadar row
    unmatched = sorted(set(tickers) - set(sharadar.index))
    if unmatched:
        raise UniverseError(f"no {Tables.sharadar_tickers} row for universe tickers {unmatched}: fix configs/universe.yml or refresh Sharadar")

    # split the tickers: entering (no row), re-entering (row out of the universe), leaving (row in it, not in the yml)
    in_yml = previous["ticker"].isin(tickers)
    active = previous["in_universe"].eq(True)
    entering = sorted(set(tickers) - set(previous["ticker"]))
    returning = sorted(previous.loc[in_yml & ~active, "ticker"])
    leaving = sorted(previous.loc[~in_yml & active, "ticker"])
    added_on = _added_on(store, [*entering, *returning], as_of)

    # carry the stored rows, re-stamp the re-entering ones, build the entering ones
    kept = previous.loc[in_yml, CARRIED_COLUMNS].copy()
    kept.loc[kept["ticker"].isin(returning), "universe_added_on"] = kept["ticker"].map(added_on)
    built = _new_rows(store, entering, sharadar, added_on) if entering else None
    carried = pd.concat([f for f in (kept, built) if f is not None and not f.empty], ignore_index=True)

    # merge the yml GICS and the Sharadar columns; a first build creates the identity columns empty
    table = (
        carried.merge(gics, left_on="ticker", right_index=True, how="left")
        .merge(_sharadar_columns(sharadar, tickers, aliases), on="ticker", how="left")
        .assign(in_universe=True)
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    if stored is None:
        table = table.assign(**dict.fromkeys(IDENTITY_COLUMNS))

    # save the changed rows, then flag the leaving ones out of the universe
    changed = _changed_rows(table, stored)
    written = store.save(Tables.universe_tickers, changed) if not changed.empty else 0
    written += store.update(Tables.universe_tickers, {"in_universe": False}, {"ticker": leaving}) if leaving else 0
    logger.info(
        "%s: %d tickers in the universe, %d entered, %d re-entered, %d left (kept, in_universe false), %d row(s) written "
        "(sharadar_tickers lastupdated %s)",
        Tables.universe_tickers,
        len(table),
        len(entering),
        len(returning),
        len(leaving),
        written,
        sharadar["lastupdated"].max(),
    )
    return written


def _sharadar_rows(store: DataStore, symbols: list[str]) -> pd.DataFrame:
    """One `sharadar_tickers` row per yfinance symbol (listed rows first, then the latest update), indexed by symbol."""
    spellings = sorted({s for symbol in symbols for s in (symbol, symbol.replace("-", "."))})
    rows = store.load(Tables.sharadar_tickers, columns=SHARADAR_READ, where={"ticker": spellings}, optional=True)
    if rows is None:
        return pd.DataFrame(columns=SHARADAR_READ).set_index("ticker")
    rows = rows.assign(ticker=rows["ticker"].map(yahoo_symbol), _listed=rows["isdelisted"].ne("N"))
    rows = rows.sort_values(["ticker", "_listed", "lastupdated"], ascending=[True, True, False], na_position="last")
    return rows.drop_duplicates("ticker").drop(columns="_listed").set_index("ticker")


def _sharadar_columns(sharadar: pd.DataFrame, tickers: list[str], aliases: dict[str, list[str]]) -> pd.DataFrame:
    """`cusips` (the union over the ticker and its alias rows) and the Sharadar category columns, per ticker."""
    out = sharadar.loc[tickers, list(SHARADAR_COLUMNS)].rename(columns=SHARADAR_COLUMNS).reset_index(names="ticker")
    out["permaticker"] = out["permaticker"].astype("int64")
    out["cusips"] = [_cusips(sharadar, [t, *aliases.get(t, [])]) for t in tickers]
    return out


def _cusips(sharadar: pd.DataFrame, symbols: list[str]) -> str | None:
    values = sharadar["cusips"].reindex(symbols).dropna()
    union = sorted({c.upper() for value in values for c in str(value).split()})
    return " ".join(union) or None


def _added_on(store: DataStore, tickers: list[str], as_of: pd.Timestamp) -> dict[str, pd.Timestamp]:
    """`universe_added_on` of tickers entering the yml: 2000-01-01 when already priced, else `as_of`."""
    priced = set(store.distinct(Tables.prices, "ticker", where={"ticker": tickers})) if tickers else set()
    return {t: ESTABLISHED_ADDED_ON if t in priced else as_of for t in tickers}


def _new_rows(store: DataStore, entering: list[str], sharadar: pd.DataFrame, added_on: dict[str, pd.Timestamp]) -> pd.DataFrame:
    """Carried columns for tickers with no stored row: the SEC snapshot first, Sharadar's name and CIK otherwise."""
    # load the SEC snapshot rows of the entering tickers
    sec = store.load(Tables.sec_company_tickers, columns=["ticker", "cik", "name"], where={"ticker": entering}, optional=True)
    rows = (sec if sec is not None else pd.DataFrame(columns=["ticker", "cik", "name"])).set_index("ticker").reindex(entering)

    # fall back on Sharadar's secfilings CIK and name for a ticker the SEC no longer lists
    unlisted = rows.index[rows["cik"].isna()].tolist()
    if unlisted:
        logger.warning("%s not in %s: Sharadar name and CIK", unlisted, Tables.sec_company_tickers)
        cik = sharadar.loc[unlisted, "secfilings"].str.extract(r"CIK=(\d+)", expand=False)
        rows.loc[unlisted, "cik"] = [pad_cik(c) if isinstance(c, str) else None for c in cik]
        rows.loc[unlisted, "name"] = sharadar.loc[unlisted, "name"]

    # check every entering ticker has a CIK, then stamp it
    no_cik = sorted(rows.index[rows["cik"].isna()])
    if no_cik:
        raise UniverseError(f"no CIK from {Tables.sec_company_tickers} or Sharadar for {no_cik}")
    rows["universe_added_on"] = [added_on[t] for t in entering]
    logger.info("%d ticker(s) enter %s", len(entering), Tables.universe_tickers)
    return rows.reset_index(names="ticker")[CARRIED_COLUMNS]


def _changed_rows(table: pd.DataFrame, stored: pd.DataFrame | None) -> pd.DataFrame:
    """Rows of `table` that are new or differ from the stored row in any column of `table`."""
    if stored is None:
        return table
    previous = stored.set_index("ticker").reindex(columns=table.columns.drop("ticker"))
    current = table.set_index("ticker")
    same = _comparable(current).eq(_comparable(previous.reindex(current.index))).all(axis=1)
    return table[~same.to_numpy()]


def _comparable(df: pd.DataFrame) -> pd.DataFrame:
    """Every value as text, dates as YYYY-MM-DD and nulls as '', so stored and rebuilt rows compare equal."""
    out = df.copy()
    out["universe_added_on"] = pd.to_datetime(out["universe_added_on"]).dt.strftime("%Y-%m-%d")
    out["permaticker"] = pd.to_numeric(out["permaticker"]).astype("Int64")
    out["in_universe"] = out["in_universe"].eq(True)
    return out.astype("string").fillna("")
