"""
universe.py  (src/utils/tickers_universe/universe.py)
-----------------------------------------------------
`Universe`: the read-only view of `universe_tickers` every stage uses through `context.universe`.

It reads the table once, keeps the `in_universe` rows, checks their ticker set equals
`configs/universe.yml` and never writes; `sync.py` and the identity step are the writers. Values
are stored clean (upper-case yfinance tickers, 10-digit CIKs), so consumers do not re-normalise them.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping

import pandas as pd
from omegaconf import DictConfig

from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.string import yahoo_symbol
from src.utils.tickers_universe.vocabulary import GICS_GROUP_SECTOR, GICS_SUB_INDUSTRY_GROUP

GICS_COLUMNS = ("sector", "industry_group")
SYNC_COMMAND = "python -m src data_extract universe"
#: `{key: (first_price_date, ended_on)}`, None where open (`Universe.bounds`).
Bounds = Mapping[str, tuple[pd.Timestamp | None, pd.Timestamp | None]]


class UniverseError(RuntimeError):
    """`universe.yml` is invalid, `universe_tickers` is out of step with it, or a ticker is outside it."""


def universe_config(config: DictConfig) -> tuple[list[str], dict[str, list[str]]]:
    """The sorted `universe.tickers` keys and the `universe.aliases` map (owner -> alias symbols), validated."""
    if "universe" not in config:
        raise UniverseError("configs/universe.yml is missing (no `universe` key in the config)")
    entries = config.universe.get("tickers") or {}
    if not isinstance(entries, Mapping):
        raise UniverseError("configs/universe.yml `tickers` must map each ticker to its `sector` and `industry`")
    raw = [yahoo_symbol(t) for t in entries]
    if not raw:
        raise UniverseError("configs/universe.yml lists no ticker")
    duplicated = sorted({t for t in raw if raw.count(t) > 1})
    if duplicated:
        raise UniverseError(f"configs/universe.yml lists these tickers twice: {duplicated}")
    aliases = {yahoo_symbol(owner): [yahoo_symbol(a) for a in symbols] for owner, symbols in (config.universe.get("aliases") or {}).items()}
    orphans = sorted(set(aliases) - set(raw))
    if orphans:
        raise UniverseError(f"configs/universe.yml aliases keys not in its tickers: {orphans}")
    return sorted(raw), aliases


def universe_classification(config: DictConfig) -> pd.DataFrame:
    """`sector`, `industry_group`, `sub_industry` per universe ticker (indexed, sorted): the yml `sector` and
    `industry` (a GICS sub-industry), the group derived from the vocabulary. Raises listing every invalid entry."""
    rows, invalid = {}, []
    for ticker, entry in config.universe.tickers.items():
        sector, industry = (entry or {}).get("sector"), (entry or {}).get("industry")
        group = GICS_SUB_INDUSTRY_GROUP.get(industry)
        if group is None or GICS_GROUP_SECTOR[group] != sector:
            invalid.append(f"{ticker} ({sector!r}, {industry!r})")
        rows[yahoo_symbol(ticker)] = {"sector": sector, "industry_group": group, "sub_industry": industry}
    if invalid:
        raise UniverseError(f"configs/universe.yml: unknown GICS sub-industry or one outside its sector for {', '.join(invalid)}")
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


class Universe:
    """The universe tickers and their `universe_tickers` row (name, CIK, CUSIPs, GICS, Sharadar categories, identity)."""

    def __init__(self, store: DataStore, config: DictConfig) -> None:
        # load the yml and the table (the store and config kept for `reread`)
        self._store, self._config = store, config
        self._tickers, aliases = universe_config(config)
        frame = store.load(Tables.universe_tickers, optional=True)
        if frame is None:
            raise UniverseError(f"{Tables.universe_tickers} is missing or empty: run `{SYNC_COMMAND}`")

        # keep the in-universe rows and check they are the yml tickers
        frame = frame[frame["in_universe"].eq(True)]
        stale = sorted(set(frame["ticker"]) ^ set(self._tickers))
        if stale:
            raise UniverseError(f"{Tables.universe_tickers} differs from configs/universe.yml on {stale}: run `{SYNC_COMMAND}`")
        self._frame = frame.sort_values("ticker").reset_index(drop=True)
        self._alias_owner = {alias: owner for owner, symbols in aliases.items() for alias in symbols}

    def reread(self) -> Universe:
        """This universe read again from `universe_tickers`, e.g. after the identity publish changed its identity columns."""
        return Universe(self._store, self._config)

    @property
    def tickers(self) -> list[str]:
        return list(self._tickers)

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame.copy()

    @property
    def alias_symbols(self) -> frozenset[str]:
        return frozenset(self._alias_owner)

    @property
    def roster(self) -> pd.DataFrame:
        """`ticker, cik` of every universe ticker."""
        return self._frame[["ticker", "cik"]].copy()

    @property
    def cik_by_ticker(self) -> dict[str, str]:
        return dict(zip(self._frame["ticker"], self._frame["cik"], strict=True))

    @property
    def bounds(self) -> dict[str, tuple[pd.Timestamp | None, pd.Timestamp | None]]:
        """`(first_price_date, ended_on)` of every universe ticker; None where identity has no bound (open)."""
        first = pd.to_datetime(self._frame["first_price_date"])
        ended = pd.to_datetime(self._frame["ended_on"])
        pairs = [(None if pd.isna(f) else f, None if pd.isna(e) else e) for f, e in zip(first, ended, strict=True)]
        return dict(zip(self._frame["ticker"], pairs, strict=True))

    @property
    def trading_symbols(self) -> dict[str, str]:
        """The symbol each universe ticker trades under today (identity's `trading_symbol`; the key itself when unset)."""
        stored = self._frame["trading_symbol"].astype(object).fillna("").astype(str).str.strip()
        symbols = stored.where(stored.ne(""), self._frame["ticker"])
        return dict(zip(self._frame["ticker"], symbols, strict=True))

    @property
    def sector_map(self) -> dict[str, str]:
        return self._gics_map("sector")

    @property
    def industry_group_map(self) -> dict[str, str]:
        return self._gics_map("industry_group")

    def _gics_map(self, column: str) -> dict[str, str]:
        rows = self._frame.dropna(subset=[column])
        return dict(zip(rows["ticker"], rows[column], strict=True))

    def attach_gics(self, df: pd.DataFrame) -> pd.DataFrame:
        """A copy of `df` with `sector` and `industry_group` as group names, keyed on `ticker`."""
        out = df.copy()
        for column in GICS_COLUMNS:
            out[column] = out["ticker"].map(self._gics_map(column))
        return out

    def gics_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """A copy of `df` with `sector` and `industry_group` as int16 category codes (sorted names present in `df`; unknown -> -1)."""
        out = df.copy()
        for column in GICS_COLUMNS:
            cats = out["ticker"].astype(str).map(self._gics_map(column)).astype("category")
            out[column] = cats.cat.codes.astype("int16")
        return out

    def added_on_dates(self, tickers: Collection[str], as_of: pd.Timestamp) -> dict[str, pd.Timestamp]:
        """`universe_added_on` of each of `tickers` on or before `as_of`."""
        day = pd.Timestamp(as_of).normalize()
        added = pd.to_datetime(self._frame["universe_added_on"]).dt.normalize()
        wanted = set(tickers)
        return {t: d for t, d in zip(self._frame["ticker"], added, strict=True) if t in wanted and pd.notna(d) and d <= day}

    def new_since(self, overlap_days: int, as_of: pd.Timestamp) -> set[str]:
        """Tickers whose `universe_added_on` falls in the `overlap_days` ending on `as_of` (both ends included)."""
        day = pd.Timestamp(as_of).normalize()
        cutoff = day - pd.Timedelta(days=overlap_days - 1)
        return {t for t, d in self.added_on_dates(self._tickers, day).items() if d >= cutoff}

    def outside(self, scope: Collection[str]) -> frozenset[str]:
        """Universe tickers not in `scope`."""
        return frozenset(self._tickers) - frozenset(scope)

    def is_full(self, tickers: Collection[str]) -> bool:
        """True when `tickers` covers the whole universe."""
        return not self.outside(tickers)

    def subset(self, names: Iterable[str]) -> list[str]:
        """`names` in yfinance spelling and de-duplicated in order; raises listing every name outside the universe."""
        wanted = list(dict.fromkeys(t for t in (yahoo_symbol(n) for n in names) if t))
        known = set(self._tickers)
        unknown = [f"{t} (alias of {self._alias_owner[t]})" if t in self._alias_owner else t for t in wanted if t not in known]
        if unknown:
            raise UniverseError(f"not in configs/universe.yml: {', '.join(unknown)}")
        return wanted


def before_end(keys: pd.Series, dates: pd.Series, bounds: Bounds) -> pd.Series:
    """True where `dates` (aligned with `keys`) falls on or before the key's `ended_on`; a key without an end, or a null
    date, is kept. The document, proxy, call and surprise planners take the end only: their filings and calls follow the
    company's CIK and symbol lineage back before its current listing's `first_price_date` (CHTR before 2010, DELL, AVGO)."""
    names = keys.astype(object)
    edges = pd.DataFrame([(key, bounds.get(key, (None, None))[1]) for key in names.unique()], columns=["key", "ended"]).astype({"key": object})
    joined = pd.DataFrame({"key": names.to_numpy(), "day": pd.to_datetime(dates, errors="coerce").to_numpy()}).merge(edges, on="key", how="left")
    ended = pd.to_datetime(joined["ended"])
    keep = ended.isna() | joined["day"].isna() | (joined["day"] <= ended)
    return pd.Series(keep.to_numpy(), index=keys.index)
