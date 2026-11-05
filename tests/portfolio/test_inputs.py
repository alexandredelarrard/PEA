"""Stored quote/return adoption through the actual portfolio input reader."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.constants.constants_price import MACRO_MARKET_SERIES
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.portfolio.utils.inputs import load_market
from tests.portfolio.test_simulation import settings
from tests.support.universe import seed_universe


@pytest.mark.parametrize("missing_quote", [False, True])
def test_market_uses_saved_pit_quote_and_repaired_total_return(
    sqlite_store: DataStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_quote: bool
) -> None:
    index = pd.bdate_range("2022-01-03", periods=4)
    df_raw = pd.DataFrame({"ticker": "AAA", "date": index, "close_split": 999.0, "close_total": 9999.0})
    quotes = [100.0, np.nan if missing_quote else 101.0, 25.5, 26.0]
    df_saved = pd.DataFrame({"ticker": "AAA", "date": index, "close_pit": quotes, "close_total": [25.0, 25.25, 25.5, 26.0], "level_factor": 1.0})
    universe = seed_universe(sqlite_store, ["AAA"], rows={"AAA": {"sector": "Tech"}})
    sqlite_store.save(Tables.prices, df_raw)
    sqlite_store.save(Tables.cube_part_prices, df_saved)
    sqlite_store.save(Tables.prices_macro, pd.DataFrame({"ticker": MACRO_MARKET_SERIES, "date": index, "close": 100.0}))
    sqlite_store.save(Tables.prices_splits, pd.DataFrame({"ticker": ["AAA"], "date": [index[2]], "ratio": [4.0]}))
    sqlite_store.save(
        Tables.sharadar_actions,
        pd.DataFrame({"ticker": ["AAA"], "date": [index[2]], "action": ["split"], "value": [4.0], "contraticker": [""]}),
    )
    config = OmegaConf.create({"data_extract": {"redundant_ticks": []}})
    context = SimpleNamespace(store=sqlite_store, config=config, config_dir=tmp_path, universe=universe, log=logging.getLogger(__name__))
    original = sqlite_store.load(Tables.prices)
    load = sqlite_store.load
    reads: list[object] = []

    def capture(table: object, **kwargs: object) -> pd.DataFrame | None:
        reads.append(table)
        return load(table, **kwargs)

    monkeypatch.setattr(sqlite_store, "load", capture)

    result = load_market(context, config, settings(end=index[-1]), {"identity": "known-truth"})

    assert result.quotes["AAA"].to_list() == pytest.approx(quotes, nan_ok=True)
    assert result.share_basis["AAA"].to_list() == [4.0, 4.0, 1.0, 1.0]
    expected_returns = pd.Series([25.0, 25.25, 25.5, 26.0], index=index).pct_change(fill_method=None)
    np.testing.assert_allclose(result.returns["AAA"], expected_returns, equal_nan=True)
    pd.testing.assert_frame_equal(load(Tables.prices), original)
    counts = Counter(reads)
    assert counts[Tables.sharadar_tickers] == counts[Tables.prices_splits] == counts[Tables.sharadar_actions] == 1
    assert not {Tables.prices, Tables.fundamentals_history, Tables.sharadar_fundamentals}.intersection(reads)
    assert result.metadata == {"identity": "known-truth"}
    print(
        "SANITY: portfolio reuses saved quotes/returns, reads quantity metadata/events once, preserves quote holes and never reloads price/share anchors."
    )


@pytest.mark.parametrize("evidence", ["category", "adr_event"])
def test_unknown_adr_quantity_units_cannot_become_an_identity(sqlite_store: DataStore, tmp_path: Path, evidence: str) -> None:
    index = pd.bdate_range("2024-01-02", periods=3)
    universe = seed_universe(sqlite_store, ["AAA"], rows={"AAA": {"sector": "Tech"}})
    sqlite_store.save(Tables.cube_part_prices, pd.DataFrame({"ticker": "AAA", "date": index, "close_pit": 50.0, "close_total": 50.0}))
    sqlite_store.save(Tables.prices_macro, pd.DataFrame({"ticker": MACRO_MARKET_SERIES, "date": index, "close": 100.0}))
    sqlite_store.save(Tables.prices_splits, pd.DataFrame({"ticker": ["_fixture_"], "date": [index[0]], "ratio": [1.0]}))
    sqlite_store.delete(Tables.prices_splits, where={"ticker": "_fixture_"})
    if evidence == "category":
        sqlite_store.save(
            Tables.sharadar_tickers,
            pd.DataFrame({"table": ["SEP"], "permaticker": [1], "ticker": ["AAA"], "category": ["ADR"], "secfilings": [""], "lastquarter": [pd.NaT]}),
        )
    else:
        sqlite_store.save(
            Tables.sharadar_actions,
            pd.DataFrame({"ticker": ["AAA"], "date": [index[1]], "action": ["adrratiosplit"], "value": [2.0], "contraticker": [""]}),
        )
    config = OmegaConf.create({"data_extract": {"redundant_ticks": []}})
    context = SimpleNamespace(store=sqlite_store, config=config, config_dir=tmp_path, universe=universe, log=logging.getLogger(__name__))

    result = load_market(context, config, settings(end=index[-1]), {})

    assert result.share_basis.AAA.isna().all()
    assert result.quotes.AAA.eq(50.0).all()
    assert result.returns.AAA.iloc[1:].eq(0.0).all()
    print(
        "SANITY: unverified ADR category/event evidence keeps holding units unknown; saved quote dollars and repaired total-return observations are unchanged."
    )
