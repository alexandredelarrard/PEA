"""Seed a `tickers_universe` table and the matching `universe` config for tests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.string import pad_cik
from src.utils.tickers_universe.universe import Universe


def universe_config(tickers: Sequence[str] | Mapping[str, Mapping[str, str]], aliases: Mapping[str, Sequence[str]] | None = None) -> DictConfig:
    """The `universe` config of `tickers`: a ticker -> {sector, industry} mapping, or bare names (no classification)."""
    entries = {t: dict(v) for t, v in tickers.items()} if isinstance(tickers, Mapping) else {t: {} for t in tickers}
    return OmegaConf.create({"universe": {"tickers": entries, "aliases": {k: list(v) for k, v in (aliases or {}).items()}}})


def seed_universe(
    store: DataStore,
    tickers: Sequence[str],
    *,
    aliases: Mapping[str, Sequence[str]] | None = None,
    rows: Mapping[str, Mapping[str, object]] | None = None,
) -> Universe:
    """Write one `tickers_universe` row per ticker (CIK from its position unless `rows` gives one) and return its reader."""
    frame = pd.DataFrame(
        [
            {
                "ticker": t,
                "name": f"{t} Inc",
                "cik": f"{i + 1:010d}",
                "sector": None,
                "industry_group": None,
                "sub_industry": None,
                "universe_added_on": pd.Timestamp("2000-01-01"),
                "cusips": None,
                **(rows or {}).get(t, {}),
            }
            for i, t in enumerate(tickers)
        ]
    )
    store.save(Tables.tickers_universe, frame)
    return Universe(store, universe_config(tickers, aliases))


def universe_of(
    tickers: Sequence[str],
    *,
    aliases: Mapping[str, Sequence[str]] | None = None,
    rows: Mapping[str, Mapping[str, object]] | None = None,
) -> Universe:
    """A `Universe` over its own in-memory store, for a stub context that has no store to seed."""
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    return seed_universe(DataStore(engine), tickers, aliases=aliases, rows=rows)


def as_roster(roster: pd.DataFrame | Mapping[str, str] | Iterable[tuple[str, str]]) -> dict[str, str]:
    """`{ticker: 10-digit CIK}` from a fixture `(ticker, cik)` frame, mapping or pairs: the `cik_by_ticker` shape."""
    if isinstance(roster, pd.DataFrame):
        roster = zip(roster["ticker"], roster["cik"], strict=True)
    elif isinstance(roster, Mapping):
        roster = roster.items()
    return {str(ticker): pad_cik(cik) for ticker, cik in roster}


def universe_in(store: DataStore) -> Universe:
    """The `Universe` over the `tickers_universe` rows a test already seeded into `store`."""
    seeded = store.load(Tables.tickers_universe, columns=["ticker"])
    return Universe(store, universe_config(list(seeded["ticker"])))
