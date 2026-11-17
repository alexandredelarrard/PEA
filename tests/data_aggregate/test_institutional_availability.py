"""Institutionals source-availability contract and mask arithmetic."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest

from src.data_aggregate.utils.common.availability import InstitutionalAvailability
from src.data_aggregate.utils.institutionals.frontiers import schedule_complete_through
from src.data_aggregate.utils.institutionals.ownership_features import ScheduleSource, build_ownership_feature_panel
from src.data_extract.utils.edgar_extract.edgar_driver import FilingStamp, marker_row
from src.data_store.schema import Table, Tables
from src.utils.freshness import max_age_days
from src.utils.window import WindowRun
from tests.conftest import make_frames
from tests.support.config import fixture_config


def _rules() -> InstitutionalAvailability:
    return InstitutionalAvailability.from_config(fixture_config())


def test_config_inherits_table_dates_and_applies_field_and_derived_overrides() -> None:
    rules = _rules()

    assert rules.source_date(Tables.insider_transactions) == pd.Timestamp("2006-01-03")
    assert rules.source_date(Tables.insider_transactions, "filing_date") == pd.Timestamp("2006-01-03")
    assert rules.derived_date("ic_shortvol_high_x_weak_price", []) == pd.Timestamp("2019-02-14")
    assert "ic_inst_flow_to_mcap" not in rules.derived_features

    print("\n=== SANITY CHECK: compact availability inheritance ===")
    print("  Active derived features resolve their boundaries and retired features have no stale availability entry. Validated.")


def test_source_mask_is_inclusive_and_combines_per_cell_requirements() -> None:
    rules = InstitutionalAvailability(tables={Tables.insider_transactions.name: pd.Timestamp("2023-04-01")}, derived_features={})
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
    assert got.loc[pd.Timestamp("2026-03-31")].to_numpy().all()
    assert not got.loc[pd.Timestamp("2026-04-01") :].to_numpy().any()
    print("\n=== SANITY CHECK: stale source frontier ===")
    print("  A source observed through 2026-03-31 is available on that date and unavailable after it. Validated.")


def _schedule_rows(table: Table, rows: list[tuple[str, str, pd.Timestamp, str]]) -> pd.DataFrame:
    """`sec_13d`/`sec_13g` rows: (ticker, accession, filing date, reporting-person CIK)."""
    df_rows = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "cik": "0000000099",
                "accession_number": accession,
                "form": "SC 13D" if table is Tables.sec_13d else "SC 13G",
                "filing_date": filed,
                "rp_seq": 0,
                "cusip": f"{ticker}CUSIP",
                "reporting_person_cik": owner,
                "reporting_person_name": f"Holder {owner}",
            }
            for ticker, accession, filed, owner in rows
        ]
    )
    if table is Tables.sec_13d:
        df_rows["is_amendment"] = 0.0
    return df_rows


def _marker(table: Table, ticker: str, accession: str, filed: pd.Timestamp) -> pd.DataFrame:
    stamp = FilingStamp(accession, "SC 13G", "0000000001", filed, False, None, None)
    return marker_row(table, ticker, stamp)


def test_schedule_frontier_reaches_the_last_session_within_the_overlap(sqlite_store) -> None:
    """The 13D/13G zero frontier is read from the table, markers included; no manifest.

    A table whose latest filing lies within its 7-day overlap of the last price session is
    complete through that session; a staler table is complete only through its latest filing."""
    last = pd.Timestamp("2026-09-11")
    config = fixture_config()
    context = SimpleNamespace(store=sqlite_store, config=config, run=WindowRun("incremental", "fixed source frontier", last))
    tolerance = max_age_days(config, Tables.sec_13g)
    stale_day = last - pd.Timedelta(days=tolerance + 3)
    fresh_day = last - pd.Timedelta(days=tolerance)
    assert schedule_complete_through(context, Tables.sec_13g, last) is None
    sqlite_store.save(Tables.sec_13g, _schedule_rows(Tables.sec_13g, [("AAA", "g1", stale_day, "1")]))
    assert schedule_complete_through(context, Tables.sec_13g, last) == stale_day
    sqlite_store.save(Tables.sec_13g, _marker(Tables.sec_13g, "BBB", "m1", fresh_day))
    assert schedule_complete_through(context, Tables.sec_13g, last) == last
    assert schedule_complete_through(context, Tables.sec_13g, None) == fresh_day
    assert len(cast(pd.DataFrame, sqlite_store.load(Tables.sec_13g))) == 1
    print(
        f"SANITY: source completeness uses the configured{tolerance}-calendar-day freshness tolerance, including real empty-filing markers; it is separate from extraction overlap."
    )


def test_db_frontier_only_turns_nan_into_zero_inside_it(sqlite_store) -> None:
    """AC-007 (fixture): over the bounded tail, the DB frontier leaves every 13D/13G cell inside
    it identical or turns NaN into 0, compared with today's frontier (None)."""
    idx = pd.bdate_range("2026-03-02", periods=140)
    tickers = ["AAA", "BBB", "CCC"]
    rows_13d = [("AAA", "d1", idx[20], "1"), ("BBB", "d2", idx[90], "2"), ("BBB", "d3", idx[-4], "3")]
    rows_13g = [("AAA", "g1", idx[10], "1"), ("CCC", "g2", idx[70], "4"), ("AAA", "g3", idx[-2], "5")]
    sqlite_store.save(Tables.sec_13d, _schedule_rows(Tables.sec_13d, rows_13d))
    sqlite_store.save(Tables.sec_13g, _schedule_rows(Tables.sec_13g, rows_13g))
    frontier_13d = schedule_complete_through(
        SimpleNamespace(
            store=sqlite_store, config=fixture_config(), run=WindowRun("incremental", "fixed source frontier", pd.Timestamp("2026-11-16"))
        ),
        Tables.sec_13d,
        idx[-1],
    )
    frontier_13g = schedule_complete_through(
        SimpleNamespace(
            store=sqlite_store, config=fixture_config(), run=WindowRun("incremental", "fixed source frontier", pd.Timestamp("2026-11-16"))
        ),
        Tables.sec_13g,
        idx[-1],
    )
    assert frontier_13d is not None and frontier_13g is not None
    frames = make_frames(idx, {t: {p: 1.0 for p in tickers if p != t} for t in tickers}, close_split=pd.DataFrame(100.0, index=idx, columns=tickers))
    sec_13d, sec_13g = sqlite_store.load(Tables.sec_13d, project=True), sqlite_store.load(Tables.sec_13g, project=True)

    def build(complete_13d: pd.Timestamp | None, complete_13g: pd.Timestamp | None) -> pd.DataFrame:
        panel = build_ownership_feature_panel(
            frames,
            (ScheduleSource(Tables.sec_13d, sec_13d, complete_13d), ScheduleSource(Tables.sec_13g, sec_13g, complete_13g)),
            fixture_config(),
        )
        grid = pd.MultiIndex.from_product([idx[idx >= idx[-1] - pd.Timedelta(days=30)], tickers], names=["date", "ticker"])
        return panel.set_index(["date", "ticker"]).reindex(grid)

    old, new = build(None, None), build(frontier_13d, frontier_13g)
    columns = sorted(set(old.columns) | set(new.columns))
    old, new = old.reindex(columns=columns), new.reindex(columns=columns)
    frontier = pd.Series([frontier_13d if c.startswith("f_ic_act_") else frontier_13g for c in columns], index=columns)
    inside = pd.DataFrame({c: new.index.get_level_values("date") <= frontier[c] for c in columns}, index=new.index)
    same = old.eq(new) | (old.isna() & new.isna())
    nan_to_zero = old.isna() & new.eq(0.0)
    moved_inside = inside & ~same
    assert (~moved_inside | nan_to_zero).all().all(), new[moved_inside & ~nan_to_zero].stack()
    assert new[~inside].isna().all().all(), "past its frontier a cell is unknown, not zero"

    print("\n=== SANITY CHECK: DB frontier vs today's (None) on the bounded tail (AC-007) ===")
    print(
        f"  frontiers: 13D {frontier_13d.date()}, 13G {frontier_13g.date()}; tail {idx[idx >= idx[-1] - pd.Timedelta(days=30)][0].date()}..{idx[-1].date()}"
    )
    print(f"  inside the frontier: {int((inside & same).to_numpy().sum())} identical, {int((inside & nan_to_zero).to_numpy().sum())} NaN->0, 0 other")
    print(f"  past the frontier: {int((~inside).to_numpy().sum())} cell(s), all NaN")
    print("  CONCLUSION: inside the DB frontier a 13D/13G cell keeps its value or becomes a proven 0. Validated.")


@pytest.mark.parametrize("case", ["invalid-coverage", "unknown-derived"])
def test_invalid_availability_declarations_fail_fast(case: str) -> None:
    config = fixture_config()
    if case == "invalid-coverage":
        config.data.tables.insider_transactions.coverage_start = "not-a-date"
    else:
        config.data.derived_feature_starts["not_a_feature"] = "2020-01-01"
    with pytest.raises(ValueError if case == "invalid-coverage" else KeyError):
        InstitutionalAvailability.from_config(config)
    print("SANITY: malformed coverage and unknown derived features fail before feature construction.")


def test_unknown_source_fields_fail_fast() -> None:
    with pytest.raises(KeyError):
        _rules().source_date(Tables.insider_transactions, "not_a_field")
    print("SANITY: coverage dates come from data.yml while requested field identities are checked against the registry.")


def test_missing_coverage_clocks_never_become_today() -> None:
    dates = pd.DatetimeIndex(["2026-11-06", "2026-11-09"])
    names = pd.Index(["AAA"])
    assert not InstitutionalAvailability.date_mask(dates, names, pd.NaT).to_numpy().any()
    assert not InstitutionalAvailability.through_mask(dates, names, pd.NaT).to_numpy().any()
    print("SANITY: missing coverage dates remain unknown; pandas cannot turn a missing clock into today.")


def test_future_source_coverage_does_not_become_the_last_price_session() -> None:
    config = fixture_config()
    config.data.tables.sec_13g.coverage_start = "2027-01-01"
    dates = pd.bdate_range("2026-09-01", periods=10)
    rows = _schedule_rows(Tables.sec_13g, [("AAA", "g1", dates[0], "1")])
    frames = make_frames(dates, {}, close_split=pd.DataFrame({"AAA": 100.0, "BBB": 100.0}, index=dates))
    panel = build_ownership_feature_panel(
        frames, [ScheduleSource(Tables.sec_13g, rows, dates[-1])], config, availability=InstitutionalAvailability.from_config(config)
    )
    assert panel.drop(columns=["date", "ticker"]).isna().all().all()
    print(
        "SANITY: source-value eligibility stays unknown before future configured coverage, while completeness frontiers use their own bounded output extent."
    )
