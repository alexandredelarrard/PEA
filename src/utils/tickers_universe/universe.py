"""
universe.py  (src/utils/tickers_universe/universe.py)
-----------------------------------------------------
`Universe`: the read-only view of `tickers_universe` every stage uses through `context.universe`.

It reads the table once, checks its ticker set equals `configs/universe.yml` and never writes;
`sync.py` is the only writer. Values are stored clean (upper-case yfinance tickers, 10-digit
CIKs), so consumers do not re-normalise them.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable

import pandas as pd
from omegaconf import DictConfig

from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.string import normalise_ticker, yahoo_symbol

GICS_COLUMNS = ("sector", "industry_group")
SYNC_COMMAND = "python -m src data_extract universe"


class UniverseError(RuntimeError):
    """`universe.yml` is invalid, `tickers_universe` is out of step with it, or a ticker is outside it."""


def universe_config(config: DictConfig) -> tuple[list[str], dict[str, list[str]]]:
    """The sorted `universe.tickers` and the `universe.aliases` map (owner -> alias symbols), validated."""
    if "universe" not in config:
        raise UniverseError("configs/universe.yml is missing (no `universe` key in the config)")
    raw = [normalise_ticker(t) for t in config.universe.get("tickers") or []]
    if not raw:
        raise UniverseError("configs/universe.yml lists no ticker")
    duplicated = sorted({t for t in raw if raw.count(t) > 1})
    if duplicated:
        raise UniverseError(f"configs/universe.yml lists these tickers twice: {duplicated}")
    aliases = {normalise_ticker(owner): [yahoo_symbol(a) for a in symbols] for owner, symbols in (config.universe.get("aliases") or {}).items()}
    orphans = sorted(set(aliases) - set(raw))
    if orphans:
        raise UniverseError(f"configs/universe.yml aliases keys not in its tickers: {orphans}")
    return sorted(raw), aliases


class Universe:
    """The universe tickers and their `tickers_universe` row (name, CIK, CUSIPs, GICS, Sharadar categories)."""

    def __init__(self, store: DataStore, config: DictConfig) -> None:
        self._tickers, aliases = universe_config(config)
        frame = store.load(Tables.tickers_universe, optional=True)
        if frame is None:
            raise UniverseError(f"{Tables.tickers_universe} is missing or empty: run `{SYNC_COMMAND}`")
        stale = sorted(set(frame["ticker"]) ^ set(self._tickers))
        if stale:
            raise UniverseError(f"{Tables.tickers_universe} differs from configs/universe.yml on {stale}: run `{SYNC_COMMAND}`")
        self._frame = frame.sort_values("ticker").reset_index(drop=True)
        self._alias_owner = {alias: owner for owner, symbols in aliases.items() for alias in symbols}

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
    def cik_by_ticker(self) -> dict[str, str]:
        return dict(zip(self._frame["ticker"], self._frame["cik"], strict=True))

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
        """`df` with `sector` and `industry_group` as int16 category codes (sorted names present in `df`; unknown -> -1)."""
        for column in GICS_COLUMNS:
            cats = df["ticker"].astype(str).map(self._gics_map(column)).astype("category")
            df[column] = cats.cat.codes.astype("int16")
        return df

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
        """`names` normalised and de-duplicated in order; raises listing every name outside the universe."""
        wanted = list(dict.fromkeys(t for t in (normalise_ticker(n) for n in names) if t))
        known = set(self._tickers)
        unknown = [f"{t} (alias of {self._alias_owner[t]})" if t in self._alias_owner else t for t in wanted if t not in known]
        if unknown:
            raise UniverseError(f"not in configs/universe.yml: {', '.join(unknown)}")
        return wanted
