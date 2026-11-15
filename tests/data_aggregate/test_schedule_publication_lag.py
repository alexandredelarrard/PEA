"""A Schedule 13D/13G filed on day d is visible from d + 1 calendar day, rolled to the first session on or after it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.constants.constants import SOURCE_AVAILABILITY_BUFFER_DAYS
from src.data_aggregate.utils.institutionals.ownership_features import SubjectIdentity, build_ownership_feature_panel
from src.data_store.schema import Tables
from tests.conftest import make_frames

SESSIONS = pd.bdate_range("2015-06-01", "2015-07-31").drop(pd.Timestamp("2015-06-15"))  # Monday 06-15 is a holiday here
TICKERS = ["AAA", "BBB", "CCC", "DDD"]
D13 = "f_ic_act_13d_count_126d_z"
G13 = "f_ic_bo_13g_count_126d_z"


def _peers() -> dict:
    return {t: {p: 1.0 for p in TICKERS if p != t} for t in TICKERS}


def _first_visible(df_schedule: pd.DataFrame, column: str, **kwargs) -> pd.Timestamp:
    frames = make_frames(SESSIONS, _peers())
    if column == D13:
        panel = build_ownership_feature_panel(frames, df_schedule, None, **kwargs)
    else:
        panel = build_ownership_feature_panel(frames, None, df_schedule, **kwargs)
    z = panel[panel["ticker"] == "AAA"].set_index("date")[column]
    return z[z > 0].index.min()


def test_the_lag_is_one_calendar_day() -> None:
    assert SOURCE_AVAILABILITY_BUFFER_DAYS[Tables.sec_13d.name] == 1 and SOURCE_AVAILABILITY_BUFFER_DAYS[Tables.sec_13g.name] == 1
    print("SANITY: the registered 13D and 13G availability buffers are 1 calendar day.")


@pytest.mark.parametrize(
    ("filed", "visible"),
    [
        ("2015-06-04", "2015-06-05"),  # Thursday -> Friday
        ("2015-06-05", "2015-06-08"),  # Friday -> Monday
        ("2015-06-12", "2015-06-16"),  # Friday before the Monday holiday -> Tuesday
    ],
)
@pytest.mark.parametrize("column", [D13, G13])
def test_a_filing_is_counted_from_its_publication_session(filed: str, visible: str, column: str) -> None:
    df_schedule = pd.DataFrame([{"ticker": "AAA", "filing_date": pd.Timestamp(filed), "accession_number": "a1"}])
    got = _first_visible(df_schedule, column)
    assert got == pd.Timestamp(visible)
    print(f"SANITY: {column} filed {filed} -> first counted {got:%Y-%m-%d}, expected {visible}.")


def test_a_marker_row_is_shifted_with_the_parsed_rows() -> None:
    rows = [
        {
            "ticker": "AAA",
            "filing_date": pd.Timestamp("2015-06-05"),
            "accession_number": "m1",
            "rp_seq": -1,
            "cusip": None,
            "reporting_person_cik": None,
        }
    ]
    identity = SubjectIdentity(ciks=pd.MultiIndex.from_arrays([[], []]), cusips=pd.MultiIndex.from_arrays([[], []]))
    got = _first_visible(pd.DataFrame(rows), G13, identity=identity)
    assert got == pd.Timestamp("2015-06-08")
    print("SANITY: a marker-only filing (rp_seq=-1) filed Friday 2015-06-05 is first counted Monday 2015-06-08.")


def test_a_filing_after_the_last_session_is_not_visible_and_the_frontier_moves_with_it() -> None:
    last = SESSIONS.max()
    df_schedule = pd.DataFrame(
        [
            {"ticker": "AAA", "filing_date": last, "accession_number": "z1"},
            {"ticker": "BBB", "filing_date": SESSIONS[3], "accession_number": "b1"},  # gives the cross-section a dispersion
        ]
    )
    frames = make_frames(SESSIONS, _peers())
    panel = build_ownership_feature_panel(frames, df_schedule, None, complete_through_13d=SESSIONS[20])
    z = panel[panel["ticker"] == "AAA"].set_index("date")[D13]
    assert not (z > 0).any()
    assert z.dropna().index.max() == SESSIONS[21], "frontier 2015-06-30 moved to its publication session"
    assert np.isnan(z.loc[SESSIONS[22]])
    print("SANITY: a last-session filing is never visible; a frontier of session k is observed through session k + 1.")
