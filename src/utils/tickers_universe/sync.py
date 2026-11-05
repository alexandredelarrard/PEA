"""
sync.py  (src/utils/tickers_universe/sync.py)
---------------------------------------------
`sync`: the only writer of `tickers_universe`. Keeps one row per `universe.yml` ticker.

* A row is removed when its ticker leaves the yml and built when one enters: CIK and name from the
  `sec_company_tickers` snapshot (Sharadar's `secfilings` CIK and name for a ticker the SEC no longer
  lists); `universe_added_on` = 2000-01-01 when the ticker already has prices (an established name, so
  a first build does not mark everything new), else `as_of`.
* name, CIK and `universe_added_on` of an existing row are never rewritten.
* GICS follows the yml (`universe_classification`), `cusips` and the Sharadar category columns follow
  `sharadar_tickers`, every run; only rows that differ are written, so an unchanged night writes nothing.
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
#: sharadar_tickers column -> tickers_universe column
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
    """Bring `tickers_universe` in line with `universe.yml`, `sec_company_tickers` and `sharadar_tickers`; returns the rows written."""
    tickers, aliases = universe_config(config)
    gics = universe_classification(config)
    stored = store.load(Tables.tickers_universe, optional=True)
    sharadar = _sharadar_rows(store, [*tickers, *(a for symbols in aliases.values() for a in symbols)])
    unmatched = sorted(set(tickers) - set(sharadar.index))
    if unmatched:
        raise UniverseError(f"no {Tables.sharadar_tickers} row for universe tickers {unmatched}: fix configs/universe.yml or refresh Sharadar")

    kept = stored[stored["ticker"].isin(tickers)][CARRIED_COLUMNS] if stored is not None else pd.DataFrame(columns=CARRIED_COLUMNS)
    entering = sorted(set(tickers) - set(kept["ticker"]))
    built = _new_rows(store, entering, sharadar, as_of) if entering else None
    carried = pd.concat([f for f in (kept, built) if f is not None and not f.empty], ignore_index=True)
    table = (
        carried.merge(gics, left_on="ticker", right_index=True, how="left")
        .merge(_sharadar_columns(sharadar, tickers, aliases), on="ticker", how="left")
        .sort_values("ticker")
        .reset_index(drop=True)
    )

    leaving = sorted(set(stored["ticker"]) - set(tickers)) if stored is not None else []
    if leaving:
        store.delete(Tables.tickers_universe, {"ticker": leaving})
        logger.info("Removed %d ticker(s) from %s: %s", len(leaving), Tables.tickers_universe, leaving)
    changed = _changed_rows(table, stored)
    written = store.save(Tables.tickers_universe, changed) if not changed.empty else 0
    logger.info(
        "%s: %d tickers, %d entered, %d left, %d row(s) written (sharadar_tickers lastupdated %s)",
        Tables.tickers_universe,
        len(table),
        len(entering),
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


def _new_rows(store: DataStore, entering: list[str], sharadar: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Carried columns for tickers with no stored row: the SEC snapshot first, Sharadar's name and CIK otherwise."""
    priced = set(store.distinct(Tables.prices, "ticker", where={"ticker": entering}))
    sec = store.load(Tables.sec_company_tickers, columns=["ticker", "cik", "name"], where={"ticker": entering}, optional=True)
    rows = (sec if sec is not None else pd.DataFrame(columns=["ticker", "cik", "name"])).set_index("ticker").reindex(entering)

    unlisted = rows.index[rows["cik"].isna()].tolist()
    if unlisted:
        logger.warning("%s not in %s: Sharadar name and CIK", unlisted, Tables.sec_company_tickers)
        cik = sharadar.loc[unlisted, "secfilings"].str.extract(r"CIK=(\d+)", expand=False)
        rows.loc[unlisted, "cik"] = [pad_cik(c) if isinstance(c, str) else None for c in cik]
        rows.loc[unlisted, "name"] = sharadar.loc[unlisted, "name"]
    no_cik = sorted(rows.index[rows["cik"].isna()])
    if no_cik:
        raise UniverseError(f"no CIK from {Tables.sec_company_tickers} or Sharadar for {no_cik}")
    rows["universe_added_on"] = [ESTABLISHED_ADDED_ON if t in priced else as_of for t in entering]
    logger.info("%d ticker(s) enter %s (%d already priced)", len(entering), Tables.tickers_universe, len(priced))
    return rows.reset_index(names="ticker")[CARRIED_COLUMNS]


def _changed_rows(table: pd.DataFrame, stored: pd.DataFrame | None) -> pd.DataFrame:
    """Rows of `table` that are new or differ from the stored row in any column."""
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
    return out.astype("string").fillna("")
