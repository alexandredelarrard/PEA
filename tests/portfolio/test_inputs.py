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
    df_saved = pd.DataFrame(
        {"ticker": "AAA", "date": index, "close_pit": quotes, "close_total": [25.0, 25.25, 25.5, 26.0], "security_basis": [1.0, 1.0, 4.0, 4.0]}
    )
    universe = seed_universe(sqlite_store, ["AAA"], rows={"AAA": {"sector": "Tech"}})
    sqlite_store.save(Tables.prices, df_raw)
    sqlite_store.save(Tables.cube_part_prices, df_saved)
    sqlite_store.save(Tables.prices_macro, pd.DataFrame({"ticker": MACRO_MARKET_SERIES, "date": index, "close": 100.0}))
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
    assert result.security_basis["AAA"].to_list() == [1.0, 1.0, 4.0, 4.0]
    expected_returns = pd.Series([25.0, 25.25, 25.5, 26.0], index=index).pct_change(fill_method=None)
    np.testing.assert_allclose(result.returns["AAA"], expected_returns, equal_nan=True)
    pd.testing.assert_frame_equal(load(Tables.prices), original)
    counts = Counter(reads)
    assert counts[Tables.cube_part_prices] == 1
    assert not {Tables.prices, Tables.prices_splits, Tables.sharadar_actions, Tables.sharadar_tickers, Tables.fundamentals_history}.intersection(
        reads
    )
    assert result.metadata == {"identity": "known-truth"}
    print(
        "SANITY: portfolio reads saved quotes, returns and the forward basis F (1 -> 4 on the split) in one price-part read, preserves quote holes"
        " and never reads split or action evidence."
    )


def test_unsupported_saved_basis_keeps_holding_units_unknown(sqlite_store: DataStore, tmp_path: Path) -> None:
    index = pd.bdate_range("2024-01-02", periods=3)
    universe = seed_universe(sqlite_store, ["AAA"], rows={"AAA": {"sector": "Tech"}})
    df_saved = pd.DataFrame({"ticker": "AAA", "date": index, "close_pit": 50.0, "close_total": 50.0, "security_basis": np.nan})
    sqlite_store.save(Tables.cube_part_prices, df_saved)
    sqlite_store.save(Tables.prices_macro, pd.DataFrame({"ticker": MACRO_MARKET_SERIES, "date": index, "close": 100.0}))
    config = OmegaConf.create({"data_extract": {"redundant_ticks": []}})
    context = SimpleNamespace(store=sqlite_store, config=config, config_dir=tmp_path, universe=universe, log=logging.getLogger(__name__))

    result = load_market(context, config, settings(end=index[-1]), {})

    assert result.security_basis.AAA.isna().all()
    assert result.quotes.AAA.eq(50.0).all()
    assert result.returns.AAA.iloc[1:].eq(0.0).all()
    print("SANITY: a null saved security_basis (unsupported units) keeps holding units unknown; saved quotes and total returns are unchanged.")
