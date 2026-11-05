"""Known-truth boundaries between saved PIT fields and disclosure quantities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.common.quotes import terminal_basis_factor
from src.data_aggregate.utils.governance.clean import pit_shares_outstanding
from src.data_aggregate.utils.price import frames as price_frames
from src.utils.price.adjustments import ShareBasis, quantity_basis
from src.utils.price.levels import compatible_tape_support, lookup_asof, rebase_security_quantities, security_basis_frame


def _terminal(events: ShareBasis, days: pd.DatetimeIndex) -> pd.DataFrame:
    """The saved basis on the grid and the step's `F_T / F_d` factor from it."""
    return terminal_basis_factor(security_basis_frame(events, days, pd.Index(["TEST"])))


def test_short_quantities_use_source_and_destination_security_units() -> None:
    days = pd.DatetimeIndex(["2020-01-02", "2020-01-03"])
    empty = pd.DataFrame(columns=["ticker", "date", "value"])
    events = ShareBasis(empty, empty, empty, {"TEST": {"quote_units": {"date": "2020-01-03", "before": 2.0, "after": 4.0}}})
    df_source = pd.DataFrame({"ticker": "TEST", "date": days, "short_volume": [10.0, 5.0], "total_volume": [20.0, 10.0]})
    factor = lookup_asof(_terminal(events, days), df_source["ticker"], df_source["date"])
    assert (df_source["short_volume"] * factor).tolist() == [5.0, 5.0]
    assert (df_source["total_volume"] * factor).tolist() == [10.0, 10.0]
    assert df_source["short_volume"].tolist() == [10.0, 5.0]
    print("SANITY: ADS source2-to-destination4 conversion through the saved F halves both FINRA legs once and preserves the source.")


def test_saved_count_support_is_not_filled_from_a_cached_count() -> None:
    days = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06"])
    empty = pd.DataFrame(columns=["ticker", "date", "value"])
    df_saved = pd.DataFrame({"TEST": [100.0, np.nan, 100.0]}, index=days)
    actual = df_saved * _terminal(ShareBasis(empty, empty, empty, {}), days)
    pd.testing.assert_frame_equal(actual, df_saved, check_names=False)
    print("SANITY: a saved unsupported daily share count stays null between independently supported dates.")


def test_saved_daily_counts_and_finra_quantities_share_one_terminal_basis() -> None:
    days = pd.DatetimeIndex(["2020-01-02", "2020-01-03"])
    df_split = pd.DataFrame({"ticker": ["TEST"], "date": [days[-1]], "value": [4.0]})
    df_factor = _terminal(ShareBasis(df_split, df_split, df_split, {}), days)
    df_saved = pd.DataFrame({"TEST": [100.0, 400.0]}, index=days)
    df_short = pd.DataFrame({"ticker": "TEST", "date": days, "short_volume": [10.0, 40.0], "total_volume": [20.0, 80.0]})
    counts = df_saved * df_factor
    converted = df_short["short_volume"] * lookup_asof(df_factor, df_short["ticker"], df_short["date"])
    assert counts["TEST"].tolist() == [400.0, 400.0]
    assert (converted.to_numpy() / counts["TEST"].to_numpy()).tolist() == pytest.approx([0.1, 0.1])
    print("SANITY: a no-trade4:1 split in the saved F changes both quantity legs together; daily short turnover stays10%.")


@pytest.mark.parametrize("invalid_factor", [-2.0, 0.0, np.nan, np.inf])
def test_tape_support_rejects_malformed_genuine_or_vendor_evidence(invalid_factor: float) -> None:
    days = pd.bdate_range("2020-01-02", periods=3)
    df_events = pd.DataFrame({"ticker": ["TEST"], "date": [days[1]], "value": [invalid_factor]})
    basis = ShareBasis(df_events, df_events, df_events, {})
    with pytest.raises(ValueError, match="positive finite"):
        compatible_tape_support(basis, days, pd.Index(["TEST"]), days[-1])
    print("SANITY: identical invalid G/F factors cannot establish tape compatibility by being silently discarded.")


def test_unestablished_adr_units_never_become_a_trusted_identity() -> None:
    df_actions = pd.DataFrame({"ticker": ["TEST"], "date": pd.to_datetime(["2020-01-03"]), "action": ["adrratiosplit"], "value": [2.0]})
    basis = quantity_basis(df_actions, pd.DataFrame(columns=["ticker", "date", "ratio"]), {}, ["TEST"])
    assert basis.unknown_units == frozenset({"TEST"})
    names = pd.Series(["TEST"])
    source = pd.Series(pd.to_datetime(["2020-01-02"]))
    target = pd.Series(pd.to_datetime(["2020-01-06"]))
    result = rebase_security_quantities(basis, pd.Series([100.0]), names, source, target, pd.Series([1.0]))
    assert result.isna().all()
    assert basis.factor(names, target, "units").isna().all()
    print("SANITY: an observed ADR-ratio change without an initial unit convention stays unsupported across all quantity adapters.")


def test_proxy_count_is_the_saved_count_as_of_the_proxy_date() -> None:
    days = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    dates = pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-05", "2020-01-06", "2020-01-07", None])
    df_proxy = pd.DataFrame({"ticker": ["TEST"] * 5 + ["TEST"], "as_of": dates, "insider_shares": 10.0})
    # A 4:1 split on Monday 2020-01-06 is already in the saved count; Tuesday's count is unsupported.
    df_saved = pd.DataFrame({"TEST": [100.0, 100.0, 400.0, np.nan]}, index=days)
    frames = price_frames.PriceFrames(days, ("TEST",), {}, shares_outstanding_pit=df_saved)
    tally: dict[str, int] = {}
    actual = pit_shares_outstanding(df_proxy, frames, tally)
    assert actual is not None
    expected = pd.Series([np.nan, 100.0, 100.0, 400.0, np.nan, np.nan], index=df_proxy.index)
    pd.testing.assert_series_equal(actual, expected, check_names=False)
    assert pit_shares_outstanding(df_proxy, None, tally) is None and len(tally) == 1
    print("SANITY: a Sunday proxy reads Friday's saved count (not Monday's post-split x4), a saved null and a pre-history or missing date stay null.")
