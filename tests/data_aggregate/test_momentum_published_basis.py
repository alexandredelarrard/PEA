"""Dollar liquidity is the saved as-traded `close_pit x volume_pit`, nothing else."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.data_aggregate.transformers.step_cube_momentum import StepCubeMomentum
from src.data_aggregate.utils.momentum.features import compute_raw_features
from src.data_aggregate.utils.price.frames import PriceFrames


def _frames(df_close: pd.DataFrame, df_volume: pd.DataFrame, df_quote: pd.DataFrame | None, df_tape: pd.DataFrame | None) -> PriceFrames:
    return PriceFrames(
        trading_index=df_close.index,
        universe=tuple(df_close.columns),
        peers={},
        close_split=df_close,
        close_total=df_close,
        open=df_close,
        high=df_close * 1.01,
        low=df_close * 0.99,
        ret=df_close.pct_change(fill_method=None),
        sector_ret=df_close * 0,
        volume=df_volume,
        close_pit=df_quote,
        volume_pit=df_tape,
    )


def test_dollar_volume_is_saved_as_traded_quote_times_as_traded_tape() -> None:
    index = pd.bdate_range("2020-01-02", periods=100)
    df_close = pd.DataFrame({"AAA": 25.0}, index=index)
    df_volume = pd.DataFrame({"AAA": 400.0}, index=index)
    # A 4:1 split on bar 50: before it the as-traded quote is 100 and the tape a quarter of the restated one.
    df_quote = pd.DataFrame({"AAA": [100.0] * 50 + [25.0] * 50}, index=index)
    df_tape = pd.DataFrame({"AAA": [0.0] * 20 + [100.0] * 30 + [400.0] * 50}, index=index)

    fields = compute_raw_features(_frames(df_close, df_volume, df_quote, df_tape))

    expected = np.log1p((df_quote * df_tape).rolling(63, min_periods=20).mean())
    pd.testing.assert_frame_equal(fields["dollar_volume_63"], expected, check_exact=True)
    assert fields["dollar_volume_63"].iloc[19, 0] == 0.0
    assert fields["dollar_volume_63"].iloc[-1, 0] == pytest.approx(np.log1p(10_000))
    assert fields["rel_volume_5_63"].iloc[-1, 0] == 1.0
    print(
        "SANITY: 100x100 before and 25x400 after a 4:1 split are both 10,000 traded dollars; observed zero tape stays zero; volume ratios use the tape."
    )


@pytest.mark.parametrize("missing", ["no_tape_column", "null_tape", "null_quote", "nonpositive_quote"])
def test_missing_as_traded_quote_or_tape_never_becomes_dollar_liquidity(missing: str) -> None:
    index = pd.bdate_range("2020-01-02", periods=100)
    df_close = pd.DataFrame({"AAA": 100 * np.cumprod(np.full(len(index), 1.001))}, index=index)
    df_volume = df_close * 10
    df_quote = {"null_quote": df_close * np.nan, "nonpositive_quote": df_close * 0}.get(missing, df_close)
    df_tape = {"no_tape_column": None, "null_tape": df_volume * np.nan}.get(missing, df_volume)

    fields = compute_raw_features(_frames(df_close, df_volume, df_quote, df_tape))

    assert fields["dollar_volume_63"].isna().all().all()
    assert fields["amihud_63"].isna().all().all()
    assert fields["rel_volume_5_63"].iloc[-1, 0] > 0
    assert fields["rev_5"].iloc[-1, 0] < 0
    print(f"SANITY: {missing} keeps dollar volume and Amihud unknown; return and relative-volume features stay supported.")


def test_momentum_builder_reads_only_saved_pit_fields() -> None:
    index = pd.bdate_range("2020-01-02", periods=100)
    df_close = pd.DataFrame({"AAA": 100 * np.cumprod(np.full(len(index), 1.001)), "BBB": 200.0}, index=index)
    df_quote = df_close.copy()
    df_quote["AAA"] = np.nan
    step = object.__new__(StepCubeMomentum)
    step._bugfix = {}
    step._cfg = OmegaConf.create({"features": {"standardize_method": "rank"}, "targets": {"horizons": []}})
    step._log = logging.getLogger("test.momentum_published_basis")
    step._context = SimpleNamespace()

    df_panel = step._price_panel(_frames(df_close, df_close * 10, df_quote, df_close * 10)).set_index(["date", "ticker"])

    assert {"close_pit", "volume_pit"} <= set(StepCubeMomentum._FIELDS) and "level_factor" not in StepCubeMomentum._FIELDS
    assert df_panel.xs("AAA", level="ticker")["dollar_volume_63"].isna().all()
    assert df_panel.xs("BBB", level="ticker")["dollar_volume_63"].notna().any()
    assert df_panel.xs("AAA", level="ticker")["rev_5"].notna().any()
    print(
        "SANITY: the momentum builder loads no basis evidence; AAA without a saved quote keeps return features and no dollar volume, BBB is populated."
    )
