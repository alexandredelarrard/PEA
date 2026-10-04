"""Institutionals source-availability contract and mask arithmetic."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_aggregate.utils.common.incremental import PART_REFRESH_TRADING_DAYS
from src.data_aggregate.utils.institutionals.availability import (
    InstitutionalAvailability,
    availability_date,
)
from src.data_aggregate.utils.institutionals.frontiers import schedule_complete_through
from src.data_aggregate.utils.institutionals.ownership_features import build_ownership_feature_panel
from src.data_extract.utils.common.edgar_driver import FilingStamp, marker_row
from src.data_store.schema import Table, Tables
from src.utils.config import read_config
from tests.conftest import make_frames


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
    assert "ic_inst_flow_to_mcap" not in rules.derived_features

    print("\n=== SANITY CHECK: compact availability inheritance ===")
    print("  Active derived features resolve their boundaries and retired features have no stale availability entry. Validated.")


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


def test_schedule_frontier_is_the_tables_latest_filing_date(sqlite_store) -> None:
    """The 13D/13G zero frontier is read from the table, markers included; no manifest."""
    assert schedule_complete_through(sqlite_store, Tables.sec_13g) is None

    sqlite_store.save(Tables.sec_13g, _schedule_rows(Tables.sec_13g, [("AAA", "g1", pd.Timestamp("2026-09-01"), "1")]))
    assert schedule_complete_through(sqlite_store, Tables.sec_13g) == pd.Timestamp("2026-09-01")

    sqlite_store.save(Tables.sec_13g, _marker(Tables.sec_13g, "BBB", "m1", pd.Timestamp("2026-09-03")))
    assert schedule_complete_through(sqlite_store, Tables.sec_13g) == pd.Timestamp("2026-09-03")
    assert len(sqlite_store.load(Tables.sec_13g)) == 1, "consumers still never see the marker"

    print("\n=== SANITY CHECK: Schedule absence frontier from the DB ===")
    print("  empty table -> None; real row 2026-09-01 -> 2026-09-01; a later marker (a read, empty filing) -> 2026-09-03")
    print("  CONCLUSION: the frontier is the table's own filing-date frontier, markers included, and no file is read. Validated.")


def test_db_frontier_only_turns_nan_into_zero_inside_it(sqlite_store) -> None:
    """AC-007 (fixture): over the bounded tail, the DB frontier leaves every 13D/13G cell inside
    it identical or turns NaN into 0, compared with today's frontier (None)."""
    idx = pd.bdate_range("2026-03-02", periods=140)
    tickers = ["AAA", "BBB", "CCC"]
    rows_13d = [("AAA", "d1", idx[20], "1"), ("BBB", "d2", idx[90], "2"), ("BBB", "d3", idx[-1], "3")]
    rows_13g = [("AAA", "g1", idx[10], "1"), ("CCC", "g2", idx[70], "4"), ("AAA", "g3", idx[-2], "5")]
    sqlite_store.save(Tables.sec_13d, _schedule_rows(Tables.sec_13d, rows_13d))
    sqlite_store.save(Tables.sec_13g, _schedule_rows(Tables.sec_13g, rows_13g))
    frontier_13d = schedule_complete_through(sqlite_store, Tables.sec_13d)
    frontier_13g = schedule_complete_through(sqlite_store, Tables.sec_13g)
    assert frontier_13d is not None and frontier_13g is not None
    frames = make_frames(idx, {t: {p: 1.0 for p in tickers if p != t} for t in tickers}, close_split=pd.DataFrame(100.0, index=idx, columns=tickers))
    sec_13d, sec_13g = sqlite_store.load(Tables.sec_13d, project=True), sqlite_store.load(Tables.sec_13g, project=True)

    def build(complete_13d: pd.Timestamp | None, complete_13g: pd.Timestamp | None) -> pd.DataFrame:
        panel = build_ownership_feature_panel(frames, sec_13d, sec_13g, complete_through_13d=complete_13d, complete_through_13g=complete_13g)
        grid = pd.MultiIndex.from_product([idx[-PART_REFRESH_TRADING_DAYS:], tickers], names=["date", "ticker"])
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
    print(f"  frontiers: 13D {frontier_13d.date()}, 13G {frontier_13g.date()}; tail {idx[-PART_REFRESH_TRADING_DAYS].date()}..{idx[-1].date()}")
    print(f"  inside the frontier: {int((inside & same).to_numpy().sum())} identical, {int((inside & nan_to_zero).to_numpy().sum())} NaN->0, 0 other")
    print(f"  past the frontier: {int((~inside).to_numpy().sum())} cell(s), all NaN")
    print("  CONCLUSION: inside the DB frontier a 13D/13G cell keeps its value or becomes a proven 0. Validated.")


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
