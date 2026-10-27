"""Liquidity arithmetic requires supported published price-level evidence."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.data_aggregate.transformers.step_cube_momentum import StepCubeMomentum
from src.data_aggregate.utils.momentum.features import compute_raw_features
from src.data_aggregate.utils.price.frames import PriceFrames
from src.data_store.schema import Tables
from src.data_store.store import DataStore


@pytest.mark.parametrize("missing_factor", [None, np.nan, 0.0, -1.0])
def test_missing_or_invalid_level_never_becomes_a_dollar_volume_identity(missing_factor: float | None) -> None:
    index = pd.bdate_range("2020-01-02", periods=100)
    df_close = pd.DataFrame({"AAA": 100 * np.cumprod(np.full(len(index), 1.001))}, index=index)
    df_volume = pd.DataFrame({"AAA": 1000.0}, index=index)
    df_factor = None if missing_factor is None else pd.DataFrame({"AAA": missing_factor}, index=index)

    fields = compute_raw_features(
        PriceFrames(
            trading_index=(df_close).index,
            universe=tuple((df_close).columns),
            peers={},
            close_total=df_close,
            open=df_close,
            sector_ret=df_close * 0,
            volume=df_volume,
            level_factor=df_factor,
            close_pit=df_close,
        ),
        tape_support=df_close.gt(0),
    )

    assert fields["dollar_volume_63"].isna().all().all()
    assert fields["amihud_63"].isna().all().all()
    assert fields["rel_volume_5_63"].iloc[-1, 0] == 1.0
    print("SANITY: unsupported published S cannot produce dollar liquidity; the independent tape-volume ratio remains supported.")


def test_supported_level_and_zero_tape_match_traded_dollars() -> None:
    index = pd.bdate_range("2020-01-02", periods=100)
    df_close = pd.DataFrame({"AAA": 25.0}, index=index)
    df_volume = pd.DataFrame({"AAA": [0.0] * 20 + [400.0] * 80}, index=index)
    df_factor = pd.DataFrame({"AAA": 2.0}, index=index)

    fields = compute_raw_features(
        PriceFrames(
            trading_index=(df_close).index,
            universe=tuple((df_close).columns),
            peers={},
            close_total=df_close,
            open=df_close,
            sector_ret=df_close * 0,
            volume=df_volume,
            level_factor=df_factor,
            close_pit=df_close,
        ),
        tape_support=df_close.gt(0),
    )

    expected = np.log1p((df_close * df_volume * df_factor).rolling(63, min_periods=20).mean())
    pd.testing.assert_frame_equal(fields["dollar_volume_63"], expected, check_exact=True)
    assert fields["dollar_volume_63"].iloc[19, 0] == 0.0
    assert fields["dollar_volume_63"].iloc[-1, 0] == pytest.approx(np.log1p(20_000))
    print("SANITY: split-adjusted 25×400×published2 gives20,000 traded dollars once; observed zero tape stays zero.")


def test_momentum_projects_published_quote_support() -> None:
    assert "close_pit" in StepCubeMomentum._FIELDS
    print("SANITY: momentum reads the published historical-quote support mask instead of trusting an unevidenced numericS identity.")


def test_numeric_level_identity_cannot_override_unknown_quote_support() -> None:
    index = pd.bdate_range("2020-01-02", periods=100)
    df_close = pd.DataFrame({"AAA": 100 * np.cumprod(np.full(len(index), 1.001))}, index=index)
    df_factor = pd.DataFrame({"AAA": 1.0}, index=index)
    df_unknown = pd.DataFrame({"AAA": np.nan}, index=index)
    fields = compute_raw_features(
        PriceFrames(
            trading_index=(df_close).index,
            universe=tuple((df_close).columns),
            peers={},
            close_total=df_close,
            open=df_close,
            sector_ret=df_close * 0,
            volume=df_close * 10,
            level_factor=df_factor,
            close_pit=df_unknown,
        ),
        tape_support=df_close.gt(0),
    )
    assert fields["dollar_volume_63"].isna().all().all()
    assert fields["amihud_63"].isna().all().all()
    assert fields["rel_volume_5_63"].iloc[-1, 0] > 0
    assert fields["rev_5"].iloc[-1, 0] < 0
    print(
        "SANITY: defaultS1 with unsupported historicalquote cannot invent USD tape; independent return and relativevolume features remain supported."
    )


def test_incompatible_tape_basis_keeps_dollar_liquidity_unknown() -> None:
    index = pd.bdate_range("2020-01-02", periods=100)
    df_close = pd.DataFrame({"AAA": 100 * np.cumprod(np.full(len(index), 1.001))}, index=index)
    df_factor = pd.DataFrame({"AAA": 0.25}, index=index)
    fields = compute_raw_features(
        PriceFrames(
            trading_index=(df_close).index,
            universe=tuple((df_close).columns),
            peers={},
            close_total=df_close,
            open=df_close,
            sector_ret=df_close * 0,
            volume=df_close * 10,
            level_factor=df_factor,
            close_pit=df_close,
        ),
        tape_support=df_close.lt(0),
    )
    assert fields["dollar_volume_63"].isna().all().all()
    assert fields["amihud_63"].isna().all().all()
    assert fields["rel_volume_5_63"].iloc[-1, 0] > 0
    assert fields["rev_5"].iloc[-1, 0] < 0
    print("SANITY: a valid quotedprice cannot certify incompatible Yahoo tape units; return and relativevolume features stay supported.")


def test_momentum_builder_gates_usd_on_saved_quote_support(sqlite_store: DataStore, tmp_path: Path) -> None:
    index = pd.bdate_range("2020-01-02", periods=100)
    df_close = pd.DataFrame({"AAA": 100 * np.cumprod(np.full(len(index), 1.001)), "BBB": 200.0}, index=index)
    df_quote = df_close.copy()
    df_quote["AAA"] = np.nan
    sqlite_store.save(Tables.prices_splits, pd.DataFrame({"ticker": ["AAA"], "date": [index[0]], "ratio": [4.0]}))
    sqlite_store.save(
        Tables.sharadar_actions,
        pd.DataFrame({"ticker": ["AAA"], "date": [index[0]], "action": ["split"], "value": [4.0], "contraticker": [""]}),
    )
    frames = PriceFrames(
        index,
        ("AAA", "BBB"),
        {},
        close_split=df_close,
        close_total=df_close,
        open=df_close,
        high=df_close * 1.01,
        low=df_close * 0.99,
        volume=df_close * 10,
        ret=df_close.pct_change(fill_method=None),
        sector_ret=df_close * 0,
        level_factor=df_close * 0 + 1,
        close_pit=df_quote,
    )
    step = object.__new__(StepCubeMomentum)
    step._context = SimpleNamespace(store=sqlite_store, config_dir=tmp_path)
    step._bugfix = {}
    step._cfg = OmegaConf.create({"features": {"standardize_method": "rank"}, "targets": {"horizons": []}})
    step._log = logging.getLogger("test.momentum_published_basis")

    df_panel = step._price_panel(frames).set_index(["date", "ticker"])

    assert df_panel.xs("AAA", level="ticker")["dollar_volume_63"].isna().all()
    assert df_panel.xs("BBB", level="ticker")["dollar_volume_63"].notna().any()
    assert df_panel.xs("AAA", level="ticker")["rev_5"].notna().any()
    print(
        "SANITY: the actual momentum builder keeps AAA returnfeatures while refusing its unverified quote/USD tape; supported BBB liquidity remains populated."
    )
