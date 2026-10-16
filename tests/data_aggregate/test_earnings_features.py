"""Known-truth report-consensus arithmetic, observation clocks and share-basis guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.fundamentals.earnings_features import _derived_earnings_fields, reported_events
from src.utils.price_adjustments import ShareBasis


def _basis(events: list[tuple[str, str, float]] | None = None) -> ShareBasis:
    df_events = pd.DataFrame(events or [], columns=["ticker", "date", "value"])
    df_events["date"] = pd.to_datetime(df_events["date"])
    return ShareBasis(actual=df_events, vendor=df_events, price=df_events, policies={})


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["A"] * 6,
            "earnings_date": ["2024-02-01", "2024-05-01", "2024-08-01", "2024-11-01", "2025-02-01", "2025-05-01"],
            "eps_estimate": [0.9, 1.0, 1.1, 1.5, 2.0, 90.0],
            "eps_actual": [1.0, 1.1, 1.2, 1.4, 0.0, np.nan],
            "surprise_pct": [10.0, 10.0, 9.0, -7.0, -100.0, np.nan],
        }
    )


def _fields(df_history: pd.DataFrame, idx: pd.DatetimeIndex, basis: ShareBasis | None = None) -> dict[str, pd.DataFrame]:
    return _derived_earnings_fields(df_history, idx, pd.DataFrame({"A": 100.0}, index=idx), share_basis=_basis() if basis is None else basis)


def test_report_consensus_appears_only_next_actual_session() -> None:
    idx = pd.DatetimeIndex(["2024-10-31", "2024-11-01", "2024-11-05", "2024-11-06"])
    fields = _fields(_history(), idx)
    assert fields["reported_expected_eps_yield"].loc["2024-11-01", "A"] == pytest.approx(1.1 / 100.0)
    assert fields["reported_expected_eps_yield"].loc["2024-11-05", "A"] == pytest.approx(1.5 / 100.0)
    assert fields["eps_surprise_last"].loc["2024-11-01", "A"] == 9.0
    assert fields["eps_surprise_last"].loc["2024-11-05", "A"] == -7.0
    assert set(fields).isdisjoint({"fwd_eps_yield", "forward_earnings_yield", "eps_expectation_growth"})
    print("Sanity: report-day consensus and surprise remain unavailable until the next supplied trading session.")


def test_expected_ttm_excludes_current_actual_and_growth_uses_previous_actual() -> None:
    idx = pd.bdate_range("2024-10-30", "2024-11-08")
    fields = _fields(_history(), idx)
    assert fields["reported_expected_ttm_yield"].loc["2024-11-04", "A"] == pytest.approx((1.5 + 1.0 + 1.1 + 1.2) / 100.0)
    assert fields["reported_expected_eps_growth_qoq"].loc["2024-11-04", "A"] == pytest.approx(1.5 / 1.2 - 1)
    df_changed = _history()
    df_changed.loc[3, "eps_actual"] = 900.0
    changed = _fields(df_changed, idx)
    for name in ("reported_expected_ttm_yield", "reported_expected_eps_growth_qoq", "reported_expected_eps_yield"):
        pd.testing.assert_frame_equal(fields[name], changed[name])
    print("Sanity: expected trailing EPS is current report estimate plus three preceding actuals, with no doubled current quarter.")


def test_future_consensus_mutation_and_full_prefix_are_invariant() -> None:
    idx = pd.bdate_range("2024-01-25", "2025-07-01")
    df_history = _history()
    fields = _fields(df_history, idx)
    df_changed = df_history.copy()
    df_changed.loc[4:, ["eps_estimate", "eps_actual", "surprise_pct"]] = 9999.0
    changed = _fields(df_changed, idx)
    cutoff = pd.Timestamp("2025-02-03")
    prefix = _fields(df_history, idx[idx < cutoff])
    for name in fields:
        pd.testing.assert_frame_equal(fields[name].loc[:"2025-01-31"], changed[name].loc[:"2025-01-31"])
        pd.testing.assert_frame_equal(fields[name].loc[idx < cutoff], prefix[name])
    print("Sanity: future report estimates/outcomes cannot change an earlier prefix under the same source and basis snapshot.")


def test_zero_actual_is_a_report_and_scheduled_rows_do_not_replace_it() -> None:
    idx = pd.bdate_range("2025-01-30", "2025-05-06")
    fields = _fields(_history(), idx)
    assert fields["reported_expected_eps_yield"].loc["2025-02-03", "A"] == pytest.approx(0.02)
    assert fields["reported_expected_eps_yield"].loc["2025-05-02", "A"] == pytest.approx(0.02)
    events = reported_events(_history(), idx)
    assert len(events) == 5
    assert events.iloc[-1]["eps_actual"] == 0
    assert events.iloc[-1]["as_of"] == pd.Timestamp("2025-02-03")
    print("Sanity: zero EPS is a published report; an estimate-only scheduled event cannot advance the report state.")


def test_invalid_new_state_clears_old_values_and_report_state_expires() -> None:
    idx = pd.bdate_range("2024-10-30", "2025-08-08")
    df_history = _history()
    df_history.loc[4, ["eps_estimate", "surprise_pct"]] = np.nan
    fields = _fields(df_history, idx)
    assert pd.notna(fields["reported_expected_eps_yield"].loc["2025-01-31", "A"])
    for name in (
        "reported_expected_eps_yield",
        "reported_expected_ttm_yield",
        "reported_expected_eps_growth_qoq",
        "eps_surprise_last",
        "eps_surprise_4q_avg",
    ):
        assert pd.isna(fields[name].loc["2025-02-03", "A"])
    fresh = _fields(_history(), idx)
    assert pd.notna(fresh["reported_expected_eps_yield"].loc["2025-08-05", "A"])
    assert pd.isna(fresh["reported_expected_eps_yield"].loc["2025-08-06", "A"])
    assert pd.isna(fresh["eps_surprise_last"].loc["2025-08-06", "A"])
    print("Sanity: invalid report fields clear prior state atomically, and values expire after 185 calendar days from publication.")


@pytest.mark.parametrize("split_ratio", [4.0, 0.1])
def test_unknown_eps_units_across_true_or_reverse_splits_are_masked(split_ratio: float) -> None:
    idx = pd.bdate_range("2024-10-30", "2025-02-07")
    fields = _fields(_history(), idx, _basis([("A", "2025-01-15", split_ratio)]))
    assert pd.isna(fields["reported_expected_eps_yield"].loc["2024-11-04", "A"])
    assert fields["reported_expected_eps_yield"].loc["2025-02-03", "A"] == pytest.approx(0.02)
    assert pd.isna(fields["reported_expected_ttm_yield"].loc["2025-02-03", "A"])
    assert pd.isna(fields["reported_expected_eps_growth_qoq"].loc["2025-02-03", "A"])
    assert fields["eps_surprise_last"].loc["2024-11-04", "A"] == -7.0
    print(f"Sanity: unresolved EPS units before a genuine {split_ratio:g} split are masked; safe post-split quarterly yields resume.")


def test_spinoff_restores_price_level_without_rebasing_eps_and_requires_basis() -> None:
    idx = pd.bdate_range("2024-10-30", "2024-11-08")
    df_close = pd.DataFrame({"A": 50.0}, index=idx)
    df_level = pd.DataFrame({"A": 2.0}, index=idx)
    fields = _derived_earnings_fields(_history(), idx, df_close, df_level, share_basis=_basis())
    assert fields["reported_expected_eps_yield"].loc["2024-11-04", "A"] == pytest.approx(1.5 / 100.0)
    without_basis = _derived_earnings_fields(_history(), idx, df_close, df_level)
    assert without_basis["reported_expected_eps_yield"].isna().all().all()
    print("Sanity: a spinoff corrects the price level only; a level factor cannot establish the archive EPS share units.")


def test_missing_report_quarters_do_not_bridge_expected_comparisons() -> None:
    idx = pd.bdate_range("2024-10-30", "2024-11-08")
    df_history = _history().drop(index=2)
    fields = _fields(df_history, idx)
    assert pd.notna(fields["reported_expected_eps_yield"].loc["2024-11-04", "A"])
    assert pd.isna(fields["reported_expected_ttm_yield"].loc["2024-11-04", "A"])
    assert pd.isna(fields["reported_expected_eps_growth_qoq"].loc["2024-11-04", "A"])
    print("Sanity: missing report quarters retain the observed quarterly yield but block unsupported multi-quarter comparisons.")


def test_nonunitary_quote_convention_masks_price_ratios() -> None:
    idx = pd.bdate_range("2024-10-30", "2024-11-08")
    basis = _basis()
    basis.policies["A"] = {"quote_units": {"date": "2024-11-06", "before": 1.0, "after": 2.0}}
    fields = _fields(_history(), idx, basis)
    assert fields["reported_expected_eps_yield"].loc["2024-11-05", "A"] == pytest.approx(0.015)
    assert pd.isna(fields["reported_expected_eps_yield"].loc["2024-11-06", "A"])
    assert fields["eps_surprise_last"].loc["2024-11-06", "A"] == -7.0
    print("Sanity: a later nonunitary quoted-security conversion cannot silently change the EPS/price units.")


def test_negative_estimate_is_signed_and_nonpositive_prior_growth_is_unknown() -> None:
    idx = pd.bdate_range("2024-10-30", "2024-11-08")
    df_history = _history()
    df_history.loc[2, "eps_actual"] = -1.0
    df_history.loc[3, "eps_estimate"] = -0.5
    fields = _fields(df_history, idx)
    assert fields["reported_expected_eps_yield"].loc["2024-11-04", "A"] == pytest.approx(-0.005)
    assert pd.isna(fields["reported_expected_eps_growth_qoq"].loc["2024-11-04", "A"])
    print("Sanity: loss expectations retain their sign, while a nonpositive prior EPS cannot create a misleading growth rate.")


def test_no_observable_report_preserves_fixed_price_universe() -> None:
    idx = pd.bdate_range("2023-12-01", "2023-12-08")
    fields = _fields(_history(), idx)
    for values in fields.values():
        assert list(values.columns) == ["A"]
        assert values.isna().all().all()
    print("Sanity: a ticker with only future reports remains unknown throughout the fixed price universe.")
