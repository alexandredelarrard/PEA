"""Target characteristics consume supported saved cap without reconstructing it."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.data_aggregate.transformers import step_cube_target as target_step
from src.data_aggregate.transformers.step_cube_target import StepCubeTarget
from src.data_aggregate.utils.price.frames import PriceFrames
from src.data_aggregate.utils.target.factors import build_characteristics


@pytest.mark.parametrize("with_fundamentals", [False, True])
def test_characteristics_use_saved_cap_and_preserve_unsupported_cells(with_fundamentals: bool) -> None:
    index = pd.bdate_range("2020-01-02", periods=5)
    df_close = pd.DataFrame({"AAA": 100.0, "BBB": 200.0}, index=index)
    df_cap = pd.DataFrame({"AAA": [1000.0, 1000.0, np.nan, 1000.0, 1000.0], "BBB": 4000.0}, index=index)
    df_fund = pd.DataFrame({"ticker": ["AAA", "BBB"], "as_of": index[:2], "marketCap": 1e30, "sharesOutstanding": 1e20})

    fields = build_characteristics(df_close, df_close * 0, df_fund if with_fundamentals else None, mcap_pit=df_cap)

    pd.testing.assert_frame_equal(fields["size"], -np.log(df_cap), check_exact=True)
    assert pd.isna(fields["size"].at[index[2], "AAA"])
    print("SANITY: target size uses the distinct saved issuer cap, including its null, independently of a fundamentals/snapshot fallback.")


def test_characteristics_do_not_backcast_a_current_cap_snapshot() -> None:
    index = pd.bdate_range("2020-01-02", periods=5)
    df_close = pd.DataFrame({"AAA": np.arange(5) + 100.0}, index=index)
    df_fund = pd.DataFrame({"ticker": ["AAA"], "as_of": [index[0]], "marketCap": [1e12]})

    fields = build_characteristics(df_close, df_close.pct_change(fill_method=None), df_fund)

    assert "size" not in fields
    assert "value" not in fields
    print("SANITY: absent published cap leaves size/value unsupported; a current snapshot cannot fabricate historical capitalization.")


def test_target_neutralizer_passes_saved_cap_without_legacy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    index = pd.bdate_range("2020-01-02", periods=2)
    df_close = pd.DataFrame({"AAA": 100.0}, index=index)
    df_cap = pd.DataFrame({"AAA": [2000.0, np.nan]}, index=index)
    frames = PriceFrames(index, ("AAA",), {}, close_split=df_close, close_total=df_close, ret=df_close * 0, mcap_pit=df_cap)
    step = object.__new__(StepCubeTarget)
    step._cfg = OmegaConf.create({"targets": {"neutralize_log_mcap": True, "min_names": 1}})
    step._log = logging.getLogger("test.target_pit")
    observed: dict[str, Any] = {}

    def capture(**kwargs: Any) -> dict:
        observed.update(kwargs)
        return {}

    monkeypatch.setattr(target_step, "build_targets_multi", capture)
    step._build_targets(frames, ({}, pd.DataFrame(index=index), []), [1], ({}, None))

    pd.testing.assert_frame_equal(observed["market_cap"], df_cap, check_exact=True)
    assert observed["stock_ret"] is frames.ret
    print("SANITY: target neutralization receives saved cap/nulls and the original total-return tape; legacy price-times-count is unused.")
