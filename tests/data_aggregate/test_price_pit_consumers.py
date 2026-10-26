"""Known-truth boundaries between saved PIT fields and disclosure quantities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.institutionals.inputs import QuantityBasis, daily_share_quantities, short_quantity_history
from src.utils.price.adjustments import ShareBasis


def test_short_quantities_use_source_and_destination_security_units() -> None:
    days = pd.DatetimeIndex(["2020-01-02", "2020-01-03"])
    empty = pd.DataFrame(columns=["ticker", "date", "value"])
    events = ShareBasis(empty, empty, empty, {"TEST": {"quote_units": {"date": "2020-01-03", "before": 2.0, "after": 4.0}}})
    basis = QuantityBasis(events, days[-1])
    df_source = pd.DataFrame({"ticker": "TEST", "date": days, "short_volume": [10.0, 5.0], "total_volume": [20.0, 10.0]})
    result = short_quantity_history(df_source, basis)
    assert result is not None
    assert result["short_volume"].tolist() == [5.0, 5.0]
    assert result["total_volume"].tolist() == [10.0, 10.0]
    assert df_source["short_volume"].tolist() == [10.0, 5.0]
    print("SANITY: ADS source2-to-destination4 conversion halves both FINRA legs once and preserves the source.")


def test_saved_count_support_is_not_filled_from_a_cached_count() -> None:
    days = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06"])
    empty = pd.DataFrame(columns=["ticker", "date", "value"])
    basis = QuantityBasis(ShareBasis(empty, empty, empty, {}), days[-1])
    df_saved = pd.DataFrame({"TEST": [100.0, np.nan, 100.0]}, index=days)
    actual = daily_share_quantities(df_saved, days, basis)
    assert actual is not None
    pd.testing.assert_frame_equal(actual, df_saved)
    print("SANITY: a saved unsupported daily share count stays null between independently supported dates.")


def test_saved_daily_counts_and_finra_quantities_share_one_terminal_basis() -> None:
    days = pd.DatetimeIndex(["2020-01-02", "2020-01-03"])
    df_split = pd.DataFrame({"ticker": ["TEST"], "date": [days[-1]], "value": [4.0]})
    basis = QuantityBasis(ShareBasis(df_split, df_split, df_split, {}), days[-1])
    df_saved = pd.DataFrame({"TEST": [100.0, 400.0]}, index=days)
    df_short = pd.DataFrame({"ticker": "TEST", "date": days, "short_volume": [10.0, 40.0], "total_volume": [20.0, 80.0]})
    counts = daily_share_quantities(df_saved, days, basis)
    converted = short_quantity_history(df_short, basis)
    assert counts is not None and converted is not None
    assert counts["TEST"].tolist() == [400.0, 400.0]
    assert (converted["short_volume"].to_numpy() / counts["TEST"].to_numpy()).tolist() == pytest.approx([0.1, 0.1])
    print("SANITY: a no-trade4:1 split changes both quantity legs together; daily short turnover stays10%.")
