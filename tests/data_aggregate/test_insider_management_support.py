"""Known-truth completeness of the active management account pool."""

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.institutionals.inputs import QuantityBasis
from src.data_aggregate.utils.institutionals.insider_features import _daily_positions
from src.data_aggregate.utils.institutionals.insider_positions import cumulative_positions, position_units
from src.data_aggregate.utils.institutionals.insider_quality import clean_transactions
from tests.data_aggregate.insider_pit_reference import _NO_SPLIT_FIXTURE, row


def _management(rows: list[dict[str, object]], idx: pd.DatetimeIndex) -> pd.Series:
    df_clean, diagnostics = clean_transactions(pd.DataFrame(rows))
    # Exercise the position contract with every already-prepared uncertainty record.
    df_trades = pd.concat(
        [df_clean, diagnostics.get("unpriced_events", pd.DataFrame()), diagnostics.get("unsupported_events", pd.DataFrame())],
        ignore_index=True,
    )
    df_trades = position_units(df_trades, QuantityBasis.from_context(_NO_SPLIT_FIXTURE, idx))
    df_flows = diagnostics.get("other_flows", pd.DataFrame())
    df_positions = _daily_positions(df_trades, df_flows)
    result = cumulative_positions(df_trades, df_positions, df_flows, idx, 60)
    return result.get("ic_insider_management_position_change_60d", pd.DataFrame(index=idx)).reindex(columns=["AAA"])["AAA"]


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({}, 950 / 10100),
        ({"shares_owned_after": np.nan}, None),
        ({"officer_title": "Unspecified officer", "role_unknown": True}, None),
        ({"n_reporting_owners": np.nan}, None),
        ({"n_reporting_owners": 0}, None),
        ({"n_reporting_owners": 1.5}, None),
        ({"owner_cik": "", "owner_ciks": ""}, None),
        ({"direct_indirect": ""}, None),
        ({"security_title": ""}, None),
        ({"price_per_share": np.nan, "value_usd": np.nan}, 950 / 10100),
        ({"direct_indirect": "I"}, -0.5),
        ({"n_reporting_owners": 2, "owner_ciks": "1,3"}, -0.5),
        ({"officer_title": "Former CEO", "role_unknown": False}, -0.5),
    ],
    ids=[
        "complete",
        "unsupported-ceo",
        "possible-ceo",
        "unknown-multiplicity",
        "zero-multiplicity",
        "noninteger-multiplicity",
        "unknown-person",
        "unknown-account",
        "unknown-class",
        "unpriced-shares",
        "indirect",
        "joint",
        "non-management",
    ],
)
def test_management_requires_all_eligible_and_possible_accounts(changes: dict[str, object], expected: float | None) -> None:
    ceo = row(accession_number="ceo", officer_title="CEO", is_officer=1, shares=1000, shares_owned_after=11000)
    ceo.update(changes)
    cfo = row(accession_number="cfo", owner_cik="2", officer_title="CFO", is_officer=1, transaction_code="S", shares=50, shares_owned_after=50)
    idx = pd.date_range("2024-01-10", "2024-03-11")
    values = _management([ceo, cfo], idx)
    if expected is None:
        assert values.loc[:"2024-03-09"].isna().all()
    else:
        assert np.allclose(values.loc[:"2024-03-09"], expected)
    assert values.loc["2024-03-10":].isna().all()
    print(f"SANITY: {changes=}; management={expected}; complete pool preserved, partial/possible accounts stay NULL and original age60 expires.")


def test_zero_start_account_contributes_to_positive_pooled_holdings() -> None:
    rows = [
        row(accession_number="opening", officer_title="CEO", is_officer=1, shares=100, shares_owned_after=100),
        row(accession_number="cfo", owner_cik="2", officer_title="CFO", is_officer=1, transaction_code="S", shares=50, shares_owned_after=50),
    ]
    values = _management(rows, pd.date_range("2024-01-10", periods=3))
    assert values.eq(0.5).all()
    print("SANITY: supported CEO opening100/0 contributes shares to CFO sale−50/100, giving pooled50/100 rather than a partial−0.5.")


def test_all_unsupported_accounts_remain_null() -> None:
    values = _management(
        [row(accession_number="unsupported", officer_title="CEO", is_officer=1, shares_owned_after=np.nan)],
        pd.date_range("2024-01-10", periods=3),
    )
    assert values.isna().all()
    print("SANITY: an eligible account without a supported holdings chain yields NULL throughout, never zero.")
