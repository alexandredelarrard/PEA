"""Known-truth completeness of the active account pool behind the purchase_pct_prior family."""

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.common.quotes import terminal_basis_factor
from src.data_aggregate.utils.institutionals.insider_features import _daily_positions
from src.data_aggregate.utils.institutionals.insider_positions import cumulative_positions, position_units
from src.data_aggregate.utils.institutionals.insider_quality import clean_transactions
from tests.data_aggregate.insider_pit_reference import _NO_SPLIT_FIXTURE, basis_frame, row, source_frame


def _pooled(rows: list[dict[str, object]], idx: pd.DatetimeIndex) -> pd.Series:
    df_clean, diagnostics = clean_transactions(source_frame(rows))
    # Exercise the position contract with every already-prepared uncertainty record.
    df_trades = pd.concat(
        [df_clean, diagnostics.get("unpriced_events", pd.DataFrame()), diagnostics.get("unsupported_events", pd.DataFrame())],
        ignore_index=True,
    )
    df_trades = position_units(df_trades, terminal_basis_factor(basis_frame(_NO_SPLIT_FIXTURE, df_trades["ticker"], idx.max())))
    df_flows = diagnostics.get("other_flows", pd.DataFrame())
    df_positions = _daily_positions(df_trades, df_flows)
    result = cumulative_positions(df_trades, df_positions, df_flows, idx, 60, {"ic_insider_purchase_pct_prior_60d": None})
    return result.get("ic_insider_purchase_pct_prior_60d", pd.DataFrame(index=idx)).reindex(columns=["AAA"])["AAA"]


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({}, 950 / 10100),
        ({"shares_owned_after": np.nan}, None),
        ({"officer_title": "Unspecified officer"}, 950 / 10100),
        ({"n_reporting_owners": np.nan}, -0.5),
        ({"n_reporting_owners": 0}, -0.5),
        ({"n_reporting_owners": 1.5}, -0.5),
        ({"owner_cik": "", "owner_ciks": ""}, -0.5),
        ({"direct_indirect": ""}, -0.5),
        ({"security_title": ""}, -0.5),
        ({"price_per_share": np.nan, "value_usd": np.nan}, 950 / 10100),
        ({"direct_indirect": "I"}, -0.5),
        ({"n_reporting_owners": 2, "owner_ciks": "1,3"}, -0.5),
        ({"officer_title": "Former CEO"}, 950 / 10100),
    ],
    ids=[
        "complete",
        "unsupported-ceo",
        "unresolved-title",
        "unknown-multiplicity",
        "zero-multiplicity",
        "noninteger-multiplicity",
        "unknown-person",
        "unknown-account",
        "unknown-class",
        "unpriced-shares",
        "indirect",
        "joint",
        "former-title",
    ],
)
def test_pool_requires_every_supported_direct_account_and_ignores_unidentified_rows(changes: dict[str, object], expected: float | None) -> None:
    ceo = row(accession_number="ceo", officer_title="CEO", is_officer=1, shares=1000, shares_owned_after=11000)
    ceo.update(changes)
    cfo = row(accession_number="cfo", owner_cik="2", officer_title="CFO", is_officer=1, transaction_code="S", shares=50, shares_owned_after=50)
    idx = pd.date_range("2024-01-10", "2024-03-11")
    values = _pooled([ceo, cfo], idx)
    if expected is None:
        assert values.loc[:"2024-03-09"].isna().all()
    else:
        assert np.allclose(values.loc[:"2024-03-09"], expected)
    assert values.loc["2024-03-10":].isna().all()
    print(
        f"SANITY: {changes=}; pooled ratio={expected}; an unsupported direct account keeps the pool NULL, rows that are not an identified direct account are not pooled, original age 60 expires."
    )


def test_zero_start_account_contributes_to_positive_pooled_holdings() -> None:
    rows = [
        row(accession_number="opening", officer_title="CEO", is_officer=1, shares=100, shares_owned_after=100),
        row(accession_number="cfo", owner_cik="2", officer_title="CFO", is_officer=1, transaction_code="S", shares=50, shares_owned_after=50),
    ]
    values = _pooled(rows, pd.date_range("2024-01-10", periods=3))
    assert values.eq(0.5).all()
    print("SANITY: supported CEO opening100/0 contributes shares to CFO sale−50/100, giving pooled50/100 rather than a partial−0.5.")


def test_all_unsupported_accounts_remain_null() -> None:
    values = _pooled(
        [row(accession_number="unsupported", officer_title="CEO", is_officer=1, shares_owned_after=np.nan)],
        pd.date_range("2024-01-10", periods=3),
    )
    assert values.isna().all()
    print("SANITY: an eligible account without a supported holdings chain yields NULL throughout, never zero.")
