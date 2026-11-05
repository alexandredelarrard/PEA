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


@pytest.mark.parametrize("later_publication", [False, True])
def test_weekend_count_uses_only_the_already_public_observation(later_publication: bool) -> None:
    days = pd.DatetimeIndex(["2020-01-03", "2020-01-06", "2020-01-07"])
    query = pd.DataFrame({"ticker": ["TEST"], "as_of": pd.to_datetime(["2020-01-05"])})
    df_public = pd.DataFrame({"ticker": "TEST", "as_of": pd.to_datetime(["2020-01-02", "2020-01-05"])})
    if later_publication:
        df_public = pd.concat([df_public, query.assign(as_of=pd.Timestamp("2020-01-06"))], ignore_index=True)
    df_events = pd.DataFrame({"ticker": ["TEST"], "date": [days[1]], "value": [4.0]})
    basis = ShareBasis(df_events, df_events, df_events, {})
    df_saved = pd.DataFrame({"TEST": [100.0, 600.0, 600.0]}, index=days)
    actual = price_frames.public_security_counts(query, df_saved, df_public, basis)
    assert pd.isna(actual.iloc[0]) if later_publication else actual.iloc[0] == pytest.approx(150.0)
    assert df_saved["TEST"].tolist() == [100.0, 600.0, 600.0]
    print(
        "SANITY: weekend-public count uses the same public observation and reverses only the evidenced Monday split; a later publication cannot leak backward."
    )


def test_exact_session_count_hole_and_missing_public_evidence_stay_unknown() -> None:
    days = pd.DatetimeIndex(["2020-01-03", "2020-01-06", "2020-01-07"])
    df_query = pd.DataFrame({"ticker": ["TEST"], "as_of": [days[1]]})
    df_public = df_query.assign(as_of=pd.Timestamp("2020-01-02"))
    empty = pd.DataFrame(columns=["ticker", "date", "value"])
    basis = ShareBasis(empty, empty, empty, {})
    df_saved = pd.DataFrame({"TEST": [100.0, np.nan, 100.0]}, index=days)
    assert price_frames.public_security_counts(df_query, df_saved, df_public, basis).isna().all()
    assert price_frames.public_security_counts(df_query, df_saved.fillna(100.0), df_public.iloc[:0], basis).isna().all()
    print("SANITY: an exact-session unsupported count is never filled; a valid saved number cannot establish its own public-observation lineage.")


def test_first_public_weekend_observation_does_not_require_an_older_supported_count() -> None:
    days = pd.DatetimeIndex(["2020-01-03", "2020-01-06"])
    df_query = pd.DataFrame({"ticker": ["TEST"], "as_of": pd.to_datetime(["2020-01-05"])})
    df_saved = pd.DataFrame({"TEST": [np.nan, 100.0]}, index=days)
    df_empty = pd.DataFrame(columns=["ticker", "date", "value"])
    basis = ShareBasis(df_empty, df_empty, df_empty, {})
    actual = price_frames.public_security_counts(df_query, df_saved, df_query, basis)
    assert actual.iloc[0] == 100.0
    print("SANITY: the first Sunday-public observation uses its Monday count in known units; no older observation is needed or fabricated.")


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
    assert basis.declared_units(names, target, pd.Series(["Common stock"])).isna().all()
    print("SANITY: an observed ADR-ratio change without an initial unit convention stays unsupported across all quantity adapters.")


@pytest.mark.parametrize("units", [1.0, 2.0, None])
def test_proxy_count_uses_saved_support_and_declared_unit_only(units: float | None) -> None:
    days = pd.bdate_range("2020-01-02", periods=3)
    df_proxy = pd.DataFrame({"ticker": "TEST", "as_of": days, "insider_shares": 10.0})
    df_public = pd.DataFrame({"ticker": ["TEST"], "as_of": [days[0]], "sharesOutstandingPit": [999999.0]})
    df_saved = pd.DataFrame({"TEST": [100.0, np.nan, 200.0]}, index=days)
    df_empty = pd.DataFrame(columns=["ticker", "date", "value"])
    policies = {} if units is None else {"TEST": {"quote_units": {"before": units, "after": units, "date": "2020-01-01"}}}
    basis = ShareBasis(df_empty, df_empty, df_empty, policies, unknown_units=frozenset({"TEST"}) if units is None else frozenset())
    frames = price_frames.PriceFrames(days, ("TEST",), {}, shares_outstanding_pit=df_saved)
    actual = pit_shares_outstanding(df_proxy, frames, df_public, basis)
    assert actual is not None
    expected = pd.Series([100.0, np.nan, 200.0] if units == 1.0 else np.nan, index=df_proxy.index)
    pd.testing.assert_series_equal(actual, expected, check_names=False)
    assert df_public["sharesOutstandingPit"].iloc[0] == 999999.0
    print("SANITY: proxy denominators use saved counts, preserve holes and refuse ambiguous ADS totals; poisoned cached counts are ignored.")
