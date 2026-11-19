"""Operational completeness survives a closed market without publishing future filings."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_aggregate.utils.institutionals.inputs import InsiderSource, lag_insider_source
from src.data_aggregate.utils.institutionals.insider_features import InsiderCoverage, _net_buy_mask
from tests.support.config import fixture_config


@pytest.mark.parametrize(
    ("first", "last", "next_session"),
    [("2026-11-12", "2026-11-13", "2026-11-16"), ("2026-11-25", "2026-11-27", "2026-11-30")],
    ids=["weekend", "thanksgiving"],
)
def test_closed_market_preserves_known_completeness_without_publishing_future_rows(first: str, last: str, next_session: str) -> None:
    # Known calendar truth: the first filing is eligible now; the terminal filing is not.
    calendar = pd.DatetimeIndex(pd.to_datetime([first, last]))
    df_rows = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "filing_date": pd.to_datetime([first, last]),
            "original_submission_date": pd.Series(pd.NaT, index=range(2), dtype="datetime64[ns]"),
        }
    )
    df_original = df_rows.copy(deep=True)
    source = InsiderSource(df_rows, pd.Timestamp(last), pd.Series({"AAPL": pd.Timestamp("2026-01-02")}))
    prepared = lag_insider_source(source, fixture_config(), calendar)
    assert prepared.transactions["_source_filing_date"].tolist() == [pd.Timestamp(first)]
    assert prepared.transactions["filing_date"].tolist() == [pd.Timestamp(last)]
    assert prepared.complete_through == pd.Timestamp(last)
    columns = pd.Index(["AAPL"], name="ticker")
    listed = pd.DataFrame(True, index=calendar, columns=columns)
    observed = _net_buy_mask(calendar, columns, listed, pd.Timestamp("2006-01-03"), InsiderCoverage(prepared.complete_through), 60)
    assert observed["AAPL"].all()

    # Once its actual publication session exists, the terminal filing becomes eligible.
    extended = pd.DatetimeIndex(calendar.append(pd.DatetimeIndex([pd.Timestamp(next_session)])))
    later = lag_insider_source(source, fixture_config(), extended)
    assert later.transactions["_source_filing_date"].tolist() == list(pd.to_datetime([first, last]))
    assert later.transactions["filing_date"].tolist() == list(pd.to_datetime([last, next_session]))
    assert later.complete_through == pd.Timestamp(next_session)
    pd.testing.assert_frame_equal(source.transactions, df_original)
    print(f"SANITY: {last} completeness remains observable; its unpublished filing stays absent until {next_session}, with raw evidence unchanged.")


@pytest.mark.parametrize("frontier", [None, pd.NaT], ids=["missing", "invalid"])
def test_unknown_frontier_never_certifies_quiet_coverage(frontier: pd.Timestamp | None) -> None:
    calendar = pd.DatetimeIndex(pd.to_datetime(["2026-11-12", "2026-11-13"]))
    source = InsiderSource(
        pd.DataFrame({"filing_date": [pd.Timestamp("2026-11-12")], "original_submission_date": [pd.NaT]}),
        frontier,
        pd.Series({"AAPL": pd.Timestamp("2026-01-02")}),
    )
    prepared = lag_insider_source(source, fixture_config(), calendar)
    assert prepared.complete_through is None
    columns = pd.Index(["AAPL"], name="ticker")
    listed = pd.DataFrame(True, index=calendar, columns=columns)
    assert not _net_buy_mask(calendar, columns, listed, pd.Timestamp("2006-01-03"), InsiderCoverage(prepared.complete_through), 60).any().any()
    print("SANITY: missing or invalid global completeness remains unknown and cannot create quiet zeros.")


def test_empty_calendar_has_no_completeness_or_eligible_filing() -> None:
    source = InsiderSource(
        pd.DataFrame({"filing_date": [pd.Timestamp("2026-11-13")], "original_submission_date": [pd.NaT]}),
        pd.Timestamp("2026-11-13"),
        pd.Series({"AAPL": pd.Timestamp("2026-01-02")}),
    )
    prepared = lag_insider_source(source, fixture_config(), pd.DatetimeIndex([]))
    assert prepared.complete_through is None and prepared.transactions.empty and prepared.first_filed.isna().all()
    print("SANITY: an empty retained calendar certifies neither publication nor operational completeness.")


def test_in_range_frontier_keeps_the_unobserved_tail_unknown() -> None:
    calendar = pd.DatetimeIndex(pd.to_datetime(["2026-11-12", "2026-11-13", "2026-11-16"]))
    source = InsiderSource(
        pd.DataFrame({"filing_date": [pd.Timestamp("2026-11-12")], "original_submission_date": [pd.NaT]}),
        pd.Timestamp("2026-11-12"),
        pd.Series({"AAPL": pd.Timestamp("2026-01-02")}),
    )
    prepared = lag_insider_source(source, fixture_config(), calendar)
    assert prepared.complete_through == pd.Timestamp("2026-11-13")
    columns = pd.Index(["AAPL"], name="ticker")
    listed = pd.DataFrame(True, index=calendar, columns=columns)
    mask = _net_buy_mask(calendar, columns, listed, pd.Timestamp("2006-01-03"), InsiderCoverage(prepared.complete_through), 60)
    assert mask["AAPL"].tolist() == [True, True, False]
    print("SANITY: a stale in-range frontier retains its original publication session; later quiet days remain unknown.")
