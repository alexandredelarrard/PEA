"""Stored quote/return adoption through the actual portfolio input reader."""

from __future__ import annotations

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


@pytest.mark.parametrize("missing_quote", [False, True])
def test_market_uses_saved_pit_quote_and_repaired_total_return(sqlite_store: DataStore, tmp_path: Path, missing_quote: bool) -> None:
    index = pd.bdate_range("2022-01-03", periods=4)
    df_raw = pd.DataFrame({"ticker": "AAA", "date": index, "close_split": 999.0, "close_total": 9999.0})
    quotes = [100.0, np.nan if missing_quote else 101.0, 25.5, 26.0]
    df_saved = pd.DataFrame({"ticker": "AAA", "date": index, "close_pit": quotes, "close_total": [25.0, 25.25, 25.5, 26.0], "level_factor": 1.0})
    sqlite_store.save(Tables.sp500_tickers, pd.DataFrame({"ticker": ["AAA"], "sector": ["Tech"]}))
    sqlite_store.save(Tables.prices, df_raw)
    sqlite_store.save(Tables.cube_part_prices, df_saved)
    sqlite_store.save(Tables.prices_macro, pd.DataFrame({"ticker": MACRO_MARKET_SERIES, "date": index, "close": 100.0}))
    sqlite_store.save(Tables.prices_splits, pd.DataFrame({"ticker": ["AAA"], "date": [index[2]], "ratio": [4.0]}))
    sqlite_store.save(
        Tables.sharadar_actions,
        pd.DataFrame({"ticker": ["AAA"], "date": [index[2]], "action": ["split"], "value": [4.0], "contraticker": [""]}),
    )
    config = OmegaConf.create({"data_extract": {"redundant_ticks": []}})
    context = SimpleNamespace(store=sqlite_store, config=config, config_dir=tmp_path)
    original = sqlite_store.load(Tables.prices)

    result = load_market(context, config, settings(end=index[-1]), {"identity": "known-truth"})

    assert result.quotes["AAA"].to_list() == pytest.approx(quotes, nan_ok=True)
    assert result.share_basis["AAA"].to_list() == [4.0, 4.0, 1.0, 1.0]
    expected_returns = pd.Series([25.0, 25.25, 25.5, 26.0], index=index).pct_change(fill_method=None)
    np.testing.assert_allclose(result.returns["AAA"], expected_returns, equal_nan=True)
    pd.testing.assert_frame_equal(sqlite_store.load(Tables.prices), original)
    assert result.metadata == {"identity": "known-truth"}
    print(
        "SANITY: portfolio reads saved quoted prices and repaired total returns; a quote hole stays unknown and the genuine split changes holdings only."
    )
