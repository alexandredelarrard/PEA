"""Account-local non-P/S evidence and the frozen JPM common-share balance chain."""

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.institutionals.inputs import QuantityBasis
from src.data_aggregate.utils.institutionals.insider_features import _daily_positions
from src.data_aggregate.utils.institutionals.insider_positions import cumulative_positions, position_units
from src.data_aggregate.utils.institutionals.insider_quality import clean_transactions
from tests.data_aggregate.insider_pit_reference import _NO_SPLIT_FIXTURE, row, source_frame


def _account_result(df_raw: pd.DataFrame, idx: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    df_trades, diagnostics = clean_transactions(df_raw)
    df_trades = position_units(df_trades, QuantityBasis.from_context(_NO_SPLIT_FIXTURE, idx))
    df_flows = diagnostics.get("other_flows", pd.DataFrame())
    df_positions = _daily_positions(df_trades, df_flows)
    return df_trades, df_positions, cumulative_positions(df_trades, df_positions, df_flows, idx, 60)


@pytest.mark.parametrize("action", [None, "0001-01-01", "2024-02-01"], ids=["missing", "ancient", "post-filing"])
@pytest.mark.parametrize("code", ["G", "P", "S"])
@pytest.mark.parametrize("management", [True, False], ids=["management", "other-role"])
def test_unknown_action_cannot_improve_a_mixed_common_account(code: str, action: str | None, management: bool) -> None:
    rows = [
        row(accession_number="first", shares=10, shares_owned_after=110, officer_title="CEO", is_officer=1),
        row(
            accession_number="second",
            filing_date="2024-01-20",
            transaction_date="2024-01-18",
            shares=20,
            shares_owned_after=130,
            officer_title="CEO",
            is_officer=1,
        ),
        row(
            accession_number="unknown",
            filing_date="2024-01-22",
            transaction_date=action,
            transaction_code=code,
            shares=5,
            shares_owned_after=135 if code != "S" else 125,
            officer_title="CEO" if management else "Director",
            is_officer=int(management),
        ),
        row(
            accession_number="later",
            filing_date="2024-04-02",
            transaction_date="2024-04-01",
            shares=10,
            shares_owned_after=110,
            officer_title="CEO",
            is_officer=1,
        ),
    ]
    idx = pd.date_range("2024-01-20", "2024-04-03")
    _, _, result = _account_result(source_frame(rows), idx)
    values = result["ic_insider_management_position_change_60d"]["AAA"]
    assert values.loc["2024-01-21"] == pytest.approx(0.3)
    assert values.loc["2024-01-22":"2024-01-25"].isna().all()
    assert values.loc["2024-04-02"] == pytest.approx(0.1)
    print(f"SANITY: {code=} {action=} {management=}; unknown action cannot preserve a supported30/100 account; a post-filing chain recovers.")


@pytest.mark.parametrize("same_day", [False, True], ids=["cross-day", "same-day"])
@pytest.mark.parametrize(
    "changes,independent",
    [
        ({}, True),
        ({"security_type": "nonderiv", "security_title": "Common Stock"}, False),
        ({"transaction_code": "M", "security_type": "nonderiv", "security_title": "Class B Common Stock"}, False),
        ({"transaction_code": "M"}, False),
        ({"security_type": None}, False),
        ({"security_title": ""}, False),
        ({"direct_indirect": ""}, False),
        ({"owner_cik": "", "owner_ciks": ""}, False),
        ({"n_reporting_owners": np.nan}, False),
        ({"n_reporting_owners": 2, "owner_ciks": "1,2"}, False),
        ({"owner_cik": "0000000002", "n_reporting_owners": 2, "owner_ciks": "1,2"}, False),
        ({"owner_cik": "0000000002", "n_reporting_owners": 2, "owner_ciks": "2"}, False),
        ({"owner_cik": "0000000002", "n_reporting_owners": 2, "owner_ciks": ""}, False),
        ({"owner_cik": "0000000002", "n_reporting_owners": 1, "owner_ciks": "2"}, True),
        ({"document_type": ""}, False),
    ],
    ids=[
        "derivative-award",
        "common-award",
        "common-exercise",
        "derivative-exercise",
        "unknown-type",
        "unknown-class",
        "unknown-account",
        "unknown-owner",
        "unknown-multiplicity",
        "joint",
        "secondary-joint-owner",
        "incomplete-joint-list",
        "missing-joint-list",
        "complete-other-owner",
        "unknown-version",
    ],
)
def test_only_proved_independent_derivative_award_leaves_common_account_supported(changes: dict, independent: bool, same_day: bool) -> None:
    flow = row(
        accession_number="award",
        filing_date="2024-01-22",
        transaction_date="2024-01-18" if same_day else "2024-01-15",
        transaction_code="A",
        security_type="deriv",
        security_title="Restricted Stock Units",
    )
    flow.update(changes)
    rows = [
        row(accession_number="first", shares=10, shares_owned_after=110, officer_title="CEO", is_officer=1),
        row(
            accession_number="second",
            filing_date="2024-01-20",
            transaction_date="2024-01-18",
            shares=20,
            shares_owned_after=130,
            officer_title="CEO",
            is_officer=1,
        ),
        flow,
    ]
    idx = pd.date_range("2024-01-20", "2024-01-25")
    _, _, result = _account_result(source_frame(rows), idx)
    values = result["ic_insider_management_position_change_60d"]["AAA"]
    assert values.loc["2024-01-21"] == pytest.approx(0.3)
    if independent:
        assert values.loc["2024-01-22":].eq(0.3).all()
    else:
        assert values.loc["2024-01-22":].isna().all()
    print(f"SANITY: {same_day=}, {changes=}; only a proved independent derivative award preserves the 30/100 common account.")


@pytest.mark.parametrize("same_day", [False, True], ids=["cross-day", "same-day"])
@pytest.mark.parametrize("amendment_case", ["unambiguous", "ambiguous", "partial", "unrelated-ambiguous", "changed-membership"])
def test_blocker_account_metadata_amendment_changes_support_only_when_public(same_day: bool, amendment_case: str) -> None:
    flow_day = "2024-01-18" if same_day else "2024-01-15"
    rows = [
        row(accession_number="first", shares=10, shares_owned_after=110, officer_title="CEO", is_officer=1),
        row(
            accession_number="second",
            filing_date="2024-01-20",
            transaction_date="2024-01-18",
            shares=20,
            shares_owned_after=130,
            officer_title="CEO",
            is_officer=1,
        ),
        row(
            accession_number="award",
            filing_date="2024-01-22",
            transaction_date=flow_day,
            transaction_code="A",
            security_type="deriv",
            security_title="",
        ),
        row(
            accession_number="amendment",
            filing_date="2024-01-24",
            transaction_date=flow_day,
            document_type="4/A",
            original_submission_date="2024-01-22",
            transaction_code="A",
            security_type="deriv",
            security_title="Restricted Stock Units",
        ),
    ]
    if amendment_case == "changed-membership":
        rows[2].update(n_reporting_owners=2, owner_ciks="1,2")
    if amendment_case == "ambiguous":
        rows.append(
            row(
                accession_number="other",
                filing_date="2024-01-22",
                transaction_date=flow_day,
                transaction_code="A",
                security_type="deriv",
                security_title="Restricted Stock Units",
            )
        )
    if amendment_case == "partial":
        rows.append(
            row(
                accession_number="award",
                filing_date="2024-01-22",
                transaction_date=flow_day,
                transaction_code="A",
                security_type="deriv",
                security_title="",
                direct_indirect="I",
                row_sequence=2,
            )
        )
    if amendment_case == "unrelated-ambiguous":
        unrelated = dict(
            owner_cik="0000000002",
            owner_ciks="0000000002",
            transaction_date=flow_day,
            transaction_code="A",
            security_type="deriv",
            security_title="Restricted Stock Units",
        )
        rows.extend(
            [
                row(accession_number="other-first", filing_date="2024-01-22", **unrelated),
                row(accession_number="other-second", filing_date="2024-01-22", **unrelated),
                row(
                    accession_number="other-amendment",
                    filing_date="2024-01-24",
                    document_type="4/A",
                    original_submission_date="2024-01-22",
                    **unrelated,
                ),
            ]
        )
    idx = pd.date_range("2024-01-20", "2024-01-26")
    _, _, result = _account_result(source_frame(rows), idx)
    values = result["ic_insider_management_position_change_60d"]["AAA"]
    assert values.loc["2024-01-21"] == pytest.approx(0.3)
    assert values.loc["2024-01-22":"2024-01-23"].isna().all()
    if amendment_case in {"ambiguous", "partial", "changed-membership"}:
        assert values.loc["2024-01-24":].isna().all()
    else:
        assert values.loc["2024-01-24":].eq(0.3).all()
    print(f"SANITY: {amendment_case=}; only a matching unambiguous correction retires its blocker; partial/unresolved accounts remain unknown.")
