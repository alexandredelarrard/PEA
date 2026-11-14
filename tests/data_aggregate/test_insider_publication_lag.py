"""A Form 4 filed on day d is visible from d + 1 calendar day, rolled to the first session on or after it."""

from __future__ import annotations

import pandas as pd
import pytest

from src.constants.constants import SOURCE_AVAILABILITY_BUFFER_DAYS
from src.data_aggregate.utils.institutionals.insider_features import InsiderCoverage
from src.data_aggregate.utils.institutionals.insider_quality import publication_day
from src.data_store.schema import Tables
from tests.conftest import make_frames
from tests.data_aggregate.insider_pit_reference import build_prepared_insider_panel as build_insider_feature_panel
from tests.data_aggregate.test_insider_features import _frame, _peers, _txn

SESSIONS = pd.bdate_range("2015-06-01", "2015-07-31").drop(pd.Timestamp("2015-06-15"))  # Monday 06-15 is a holiday here


def test_the_lag_is_one_calendar_day() -> None:
    assert SOURCE_AVAILABILITY_BUFFER_DAYS[Tables.insider_transactions.name] == 1
    print("SANITY: the registered insider availability buffer is 1 calendar day.")


@pytest.mark.parametrize(
    ("filed", "visible"),
    [
        ("2015-06-04", "2015-06-05"),  # Thursday -> Friday
        ("2015-06-05", "2015-06-08"),  # Friday -> Monday
        ("2015-06-08", "2015-06-09"),  # Monday -> Tuesday
        ("2015-06-12", "2015-06-16"),  # Friday before the Monday holiday -> Tuesday
    ],
)
def test_publication_day_rolls_to_the_next_session(filed: str, visible: str) -> None:
    got = publication_day(pd.Series(pd.to_datetime([filed, None])), 1, SESSIONS)
    assert got.iloc[0] == pd.Timestamp(visible) and pd.isna(got.iloc[1])
    print(f"SANITY: filed {filed} -> visible {visible}; a missing date stays missing.")


@pytest.mark.parametrize(("filed", "first_visible"), [("2015-06-05", "2015-06-08"), ("2015-06-08", "2015-06-09")])
def test_a_trade_enters_every_window_from_its_lagged_session(filed: str, first_visible: str) -> None:
    close = pd.DataFrame(100.0, index=SESSIONS, columns=["AAA"])
    shares = pd.DataFrame({"ticker": ["AAA"], "as_of": [pd.Timestamp("2014-01-01")], "sharesOutstanding": [1_000_000.0]})
    rows = [_txn(accession_number="t", filing_date=filed, transaction_date="2015-06-03", shares=100.0, shares_owned_after=1_100.0)]
    frames = make_frames(SESSIONS, _peers(), close_split=close)
    coverage = InsiderCoverage.from_availability(None, SESSIONS.max())
    panel = build_insider_feature_panel(
        frames, _frame(rows), shares_out_history=shares, df_valuation_close=close, coverage=coverage, lag_days=1
    ).set_index("date")
    value = panel["f_ic_insider_buy_value_mcap_60d"]
    before = value[value.index < pd.Timestamp(first_visible)].dropna()
    after = value[value.index >= pd.Timestamp(first_visible)].iloc[:3]
    assert (before == 0).all() and (after > 0).all()
    buyers = panel["f_ic_insider_distinct_buyers_120d"]
    assert buyers.loc[first_visible] == 1 and (buyers[buyers.index < pd.Timestamp(first_visible)].dropna() == 0).all()
    print(f"SANITY: filed {filed}, buy value and buyer count are 0 up to the session before {first_visible} and positive from it.")
