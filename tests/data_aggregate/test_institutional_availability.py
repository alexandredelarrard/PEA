"""Institutionals source-availability contract and mask arithmetic."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_aggregate.utils.institutionals.availability import (
    InstitutionalAvailability,
    availability_date,
)
from src.data_store.schema import Tables
from src.utils.config import read_config


def _rules() -> InstitutionalAvailability:
    return InstitutionalAvailability.from_config(read_config("./configs").data_availability)


def test_config_inherits_table_dates_and_applies_field_and_derived_overrides() -> None:
    rules = _rules()

    assert rules.source_date(Tables.insider_transactions) == pd.Timestamp("2006-01-03")
    assert rules.source_date(Tables.insider_transactions, "filing_date") == pd.Timestamp("2006-01-03")
    assert rules.source_date(Tables.insider_transactions, "is_10b5_1") == pd.Timestamp("2023-04-01")
    assert rules.derived_date(
        "ic_insider_discretionary_sell_mcap_60d",
        [(Tables.insider_transactions, "is_10b5_1")],
    ) == pd.Timestamp("2023-04-01")

    print("\n=== SANITY CHECK: compact availability inheritance ===")
    print("  Insider fields inherit 2006-01-03; is_10b5_1 and its derived sale leg start 2023-04-01. Validated.")


def test_source_mask_is_inclusive_and_combines_per_cell_requirements() -> None:
    rules = _rules()
    idx = pd.date_range("2023-03-31", periods=3, freq="D")
    columns = pd.Index(["AAA", "BBB"], name="ticker")
    denominator = pd.DataFrame([[True, True], [True, False], [True, True]], index=idx, columns=columns)

    got = rules.source_mask(
        Tables.insider_transactions,
        idx,
        columns,
        field="is_10b5_1",
        requirements=(denominator,),
    )

    expected = pd.DataFrame([[False, False], [True, False], [True, True]], index=idx, columns=columns)
    pd.testing.assert_frame_equal(got, expected)
    print("\n=== SANITY CHECK: per-cell availability boundary ===")
    print("  2023-04-01 is inclusive, while a missing ticker denominator remains unavailable. Validated.")


def test_source_frontier_is_inclusive_and_cannot_extend_stale_data() -> None:
    idx = pd.date_range("2026-03-30", periods=4, freq="D")
    columns = pd.Index(["AAA", "BBB"], name="ticker")
    got = InstitutionalAvailability.through_mask(idx, columns, pd.Timestamp("2026-03-31"))
    assert got.loc[pd.Timestamp("2026-03-31")].all()
    assert not got.loc[pd.Timestamp("2026-04-01") :].to_numpy().any()
    print("\n=== SANITY CHECK: stale source frontier ===")
    print("  A source observed through 2026-03-31 is available on that date and unavailable after it. Validated.")


@pytest.mark.parametrize(
    "config, error",
    [
        ({"institutionals": {"not_a_table": {"__all__": "2020-01-01"}}}, KeyError),
        (
            {"institutionals": {"insider_transactions": {"__all__": "2006-01-03", "not_a_field": "2020-01-01"}}},
            KeyError,
        ),
        ({"institutionals": {"insider_transactions": {"__all__": "not-a-date"}}}, ValueError),
        (
            {"institutionals": {"insider_transactions": {"__all__": "2020-01-01", "is_10b5_1": "2019-01-01"}}},
            ValueError,
        ),
        ({"institutionals": {"derived_features": {"not_a_feature": "2020-01-01"}}}, KeyError),
    ],
)
def test_invalid_availability_declarations_fail_fast(config: dict, error: type[Exception]) -> None:
    with pytest.raises(error):
        InstitutionalAvailability.from_config(config)
    print(f"\n=== SANITY CHECK: invalid availability declaration ===\n  {error.__name__} raised before feature construction. Validated.")


def test_13f_period_floor_still_uses_publication_lag_and_trading_sessions() -> None:
    grid = pd.bdate_range("2013-08-12", "2013-08-23")
    period = pd.DatetimeIndex(["2013-06-30"])

    bare = availability_date(period, grid, settle_trading_days=0)
    settled = availability_date(period, grid, settle_trading_days=2)

    assert bare.iloc[0] == pd.Timestamp("2013-08-14")
    assert settled.iloc[0] == pd.Timestamp("2013-08-16")
    print("\n=== SANITY CHECK: 13F period versus tradable date ===")
    print("  2013-06-30 remains a period floor; its feature mask starts only after filing lag and session settlement. Validated.")
