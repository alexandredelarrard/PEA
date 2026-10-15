"""Exact expiry and invalidation controls for annual governance measurements."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.common.pit import fundamentals_to_daily
from src.data_aggregate.utils.governance.panel import _ceo_pay_growth
from src.data_aggregate.utils.governance.pay_features import pay_fields
from src.data_aggregate.utils.governance.provisions_features import provision_fields
from src.data_aggregate.utils.governance.staleness import expire_stale
from src.data_aggregate.utils.governance.vote_dissent_features import vote_dissent_fields


@pytest.mark.parametrize("field", ["board_size", "sop_dissent", "ceo_pay_growth"])
def test_annual_measurement_expires_after_day_395(field: str) -> None:
    filed = pd.Timestamp("2022-03-01")
    idx = pd.date_range(filed, periods=397)
    hist = pd.DataFrame({"ticker": ["AAA"], "as_of": [filed], field: [0.2]})
    daily = expire_stale(fundamentals_to_daily(hist, field, idx), hist, field)
    assert daily.iloc[395, 0] == pytest.approx(0.2)
    assert pd.isna(daily.iloc[396, 0])
    print(f"SANITY: {field} survives exactly 395 calendar days and expires on day 396.")


def test_ceo_change_ends_both_pay_growth_series_immediately() -> None:
    hist = pd.DataFrame(
        {
            "ticker": ["AAA"] * 4,
            "as_of": pd.to_datetime(["2020-03-01", "2021-03-01", "2022-03-01", "2023-03-01"]),
            "ceo_total_comp": [10e6, 12e6, 20e6, 22e6],
            "ceo_name_proxy": ["Alice Smith", "Alice Smith", "Bob Jones", "Bob Jones"],
        }
    )
    idx = pd.date_range("2021-02-28", "2023-03-02")
    legacy = _ceo_pay_growth(hist, idx)
    frames, _ = pay_fields(hist, None, None, None, {}, idx)
    for frame in (legacy, frames["ceo_comp_growth_1y"]):
        assert pd.notna(frame.loc["2022-02-28", "AAA"])
        assert frame.loc["2022-03-01":"2023-02-28", "AAA"].isna().all()
        assert pd.notna(frame.loc["2023-03-01", "AAA"])
    print("SANITY: CEO turnover immediately ends old percentage/log pay growth; a valid new comparison reopens it.")


def test_rejected_independence_delta_ends_the_previous_delta() -> None:
    hist = pd.DataFrame(
        {
            "ticker": ["AAA"] * 4,
            "as_of": pd.to_datetime(["2020-03-01", "2021-03-01", "2022-03-01", "2023-03-01"]),
            "pct_independent_directors": [0.9, 0.8, 0.7, 0.8],
            "pct_independent_directors_imputed": [False, False, True, False],
        }
    )
    idx = pd.date_range("2021-02-28", "2023-03-02")
    frames, _ = provision_fields(hist, idx)
    delta = frames["board_independence_delta_1y"]
    assert pd.notna(delta.loc["2022-02-28", "AAA"])
    assert delta.loc["2022-03-01":, "AAA"].isna().all()
    assert frames["board_independence_drop_10pp"].loc["2022-03-01":, "AAA"].isna().all()
    print("SANITY: an interpolated annual pair ends the prior independence delta and its drop flag immediately.")


def test_invalid_new_vote_does_not_resurrect_an_older_vote() -> None:
    votes = pd.DataFrame(
        {
            "ticker": ["AAA"] * 3,
            "accession_number": ["a", "b", "c"],
            "proposal_seq": [1, 1, 1],
            "filing_date": pd.to_datetime(["2021-05-01", "2022-05-01", "2023-05-01"]),
            "proposal_type": ["say_on_pay"] * 3,
            "votes_for": [90.0, np.nan, 80.0],
            "votes_against": [10.0, 5.0, 20.0],
            "votes_abstain": [0.0, 0.0, 0.0],
        }
    )
    idx = pd.date_range("2022-04-30", "2023-05-02")
    frames, _ = vote_dissent_fields(votes, idx)
    dissent = frames["sop_dissent"]
    assert dissent.loc["2022-04-30", "AAA"] == pytest.approx(0.1)
    assert dissent.loc["2022-05-01":"2023-04-30", "AAA"].isna().all()
    assert dissent.loc["2023-05-01", "AAA"] == pytest.approx(0.2)
    print("SANITY: a new unreadable pay vote ends the old vote until a subsequent valid vote becomes public.")
