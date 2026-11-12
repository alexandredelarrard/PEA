"""The 13F parts sum a company's share classes back to one ticker before any feature builder.

Known truth (the dated `universe_cusip` lines as published: BRK-A = 1,500 BRK-B from 2010-01-21, GOOG = 1 GOOGL):
- an as-filed book BRK-A 2 + BRK-B 100 + GOOG 10 + GOOGL 5 (marker 4) gives BRK-B 3,100 and GOOGL 15;
- an additions amendment holding BRK-A 1 adds to the original's BRK-B company row: 3,100 then 4,600;
- a superinvestor single-version book as filed (marker 3) holds one BRK-B position of 3,100 shares;
- the same books stored merged (one row per company, the U-67 shape) give identical builder input;
- on real stored rows, the builders receive one row per (period, manager, ticker) whose shares are the book's company
  CUSIP shares x the dated ratio.
The stored `ticker` comes in two shapes: the market tag (`cusip_ticker_map`, as extraction stores it from P2) and the
company tag (rows stored before U-67). Neither is read: the cube tags every row itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import closing
from typing import Any, cast

import pandas as pd
import pytest

from src.data_aggregate.utils.sec_thirteen.holdings_clean import clean_public_batches
from src.data_aggregate.utils.sec_thirteen.inputs import iter_filing_vintages
from src.data_aggregate.utils.sec_thirteen.institutional_books import prepare_institutional_book_history
from src.data_aggregate.utils.sec_thirteen.superinvestor_books import prepare_superinvestor_book_history
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.identity_tables import LINE_COLUMNS, TAGGED_ROLES, CusipTags, universe_cusips
from tests.conftest import _store
from tests.data_aggregate.sec_thirteen_legacy_reference import iter_snapshot_books

BRK_A, BRK_B, GOOG, GOOGL, AVB = "084670108", "084670702", "02079K107", "02079K305", "053484101"
LINES = pd.DataFrame(
    [
        (BRK_B, "BRK-B", "canonical_current", "BRK-B", 1.0, "1998-12-31", None),
        (BRK_A, "BRK-B", "secondary_class", "BRK-A", 30.0, "1998-12-31", "2010-01-21"),
        (BRK_A, "BRK-B", "secondary_class", "BRK-A", 1500.0, "2010-01-21", None),
        (GOOGL, "GOOGL", "canonical_current", "GOOGL", 1.0, "2015-10-01", None),
        (GOOG, "GOOGL", "secondary_class", "GOOG", 1.0, "2015-10-01", None),
    ],
    columns=LINE_COLUMNS,
)
MARKET = {BRK_A: "BRK-A", BRK_B: "BRK-B", GOOG: "GOOG", GOOGL: "GOOGL", AVB: "AVB"}
COMPANY = {**MARKET, BRK_A: "BRK-B", GOOG: "GOOGL"}
TAGS = CusipTags(LINES, pd.DataFrame({"cusip": list(MARKET), "ticker": list(MARKET.values())}))
UNIVERSE = ("BRK-B", "GOOGL")
GRID = pd.bdate_range("2024-01-02", "2024-12-31")
CLOSE = pd.DataFrame({"BRK-B": 400.0, "GOOGL": 150.0}, index=GRID)
PERIOD = pd.Timestamp("2024-03-31")
FILED, MERGED = "0000000001", "0000000002"
#: (shares, value_usd) per CUSIP; every value at the canonical close (BRK-A 2 x 1,500 x 400 = 1.2m)
AS_FILED = {BRK_A: (2.0, 1_200_000.0), BRK_B: (100.0, 40_000.0), GOOG: (10.0, 1_500.0), GOOGL: (5.0, 750.0)}
AS_MERGED = {BRK_B: (3_100.0, 1_240_000.0), GOOGL: (15.0, 2_250.0)}


def _book(cik: str, filed: str, positions: Mapping[str, tuple[float, float]], tickers: Mapping[str, str], kind: str = "original") -> pd.DataFrame:
    """One stored `sec13f_hr` accession: its completion marker (`parsed_row_count` = stored rows) and positions."""
    empty = cast(tuple[str, str], Tables.sec13f_hr.empty_marker)[1]
    base = {
        "cik": cik,
        "period": PERIOD,
        "filing_date": pd.Timestamp(filed),
        "accession_number": f"{cik}-{filed}",
        "accepted_at": pd.Timestamp(filed, tz="UTC") + pd.Timedelta(hours=10),
        "amendment_type": kind,
        "amendment_number": int(kind != "original"),
        "report_type": "13F HOLDINGS REPORT",
        "restricted": False,
        "parsed_row_count": len(positions),
        "position_type": "common",
        "class_evidence": '["COM"]',
        **dict.fromkeys(("call_shares", "call_value", "put_shares", "put_value", "debt_prn", "debt_value", "other_value"), 0.0),
        **dict.fromkeys(("raw_call_value", "raw_put_value", "raw_debt_value", "raw_other_value"), 0.0),
    }
    marker = {**base, "cusip": empty, "ticker": empty, "parse_status": "complete", "shares": 0.0, "value_usd": 0.0}
    rows = [
        {**base, "cusip": cusip, "ticker": tickers[cusip], "parse_status": "position", "shares": shares, "value_usd": value, "raw_value_usd": value}
        for cusip, (shares, value) in positions.items()
    ]
    return pd.DataFrame([marker, *rows])


def _broad_snapshots(store: DataStore, tags: CusipTags, universe: tuple[str, ...]) -> list[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]]:
    """The broad part's own read, clean and resolve, replayed as the full public snapshot of each publication day."""
    packets = iter_filing_vintages(store, GRID.max(), universe=universe, cusips=universe_cusips(tags.lines, universe))
    assert packets is not None
    with closing(clean_public_batches(packets, tags, universe, CLOSE, None)) as batches:
        history = prepare_institutional_book_history(batches, GRID, settle_days=3)
    assert history is not None
    return list(iter_snapshot_books(*history))


def _manager(df_compact: pd.DataFrame, cik: str) -> pd.DataFrame:
    """One manager's compact rows, keyed (period, ticker)."""
    return df_compact.xs(cik, level="cik").sort_index()


@pytest.mark.parametrize("stored", ["market", "company"])
def test_an_as_filed_book_sums_each_company_onto_one_ticker(sqlite_store: Any, stored: str) -> None:
    tickers = MARKET if stored == "market" else COMPANY
    sqlite_store.save(Tables.sec13f_hr, pd.concat([_book(FILED, "2024-05-10", AS_FILED, tickers), _book(MERGED, "2024-05-10", AS_MERGED, COMPANY)]))
    _, df_compact, _ = _broad_snapshots(sqlite_store, TAGS, UNIVERSE)[-1]
    filed, merged = _manager(df_compact, FILED), _manager(df_compact, MERGED)
    assert filed.loc[(PERIOD, "BRK-B"), "shares"] == 3_100.0 and filed.loc[(PERIOD, "GOOGL"), "shares"] == 15.0
    assert filed.loc[(PERIOD, "BRK-B"), "value_usd"] == 1_240_000.0 and filed.loc[(PERIOD, "GOOGL"), "value_usd"] == 2_250.0
    assert filed["_cusips"].tolist() == [BRK_B, GOOGL]
    pd.testing.assert_frame_equal(filed, merged)
    print(
        f"SANITY ({stored} tickers stored): BRK-A 2 x 1,500 + BRK-B 100 = {filed.loc[(PERIOD, 'BRK-B'), 'shares']:,.0f} BRK-B shares and "
        f"GOOG 10 + GOOGL 5 = {filed.loc[(PERIOD, 'GOOGL'), 'shares']:,.0f} GOOGL, each on its canonical CUSIP; the merged book reads identically."
    )


@pytest.mark.parametrize("stored", ["market", "company"])
def test_an_additions_amendment_adds_its_class_a_shares_to_the_company(sqlite_store: Any, stored: str) -> None:
    tickers = MARKET if stored == "market" else COMPANY
    books = [
        _book(FILED, "2024-05-10", {BRK_A: AS_FILED[BRK_A], BRK_B: AS_FILED[BRK_B]}, tickers),
        _book(FILED, "2024-06-10", {BRK_A: (1.0, 600_000.0)}, tickers, "additions"),
        _book(MERGED, "2024-05-10", {BRK_B: AS_MERGED[BRK_B]}, COMPANY),
        _book(MERGED, "2024-06-10", {BRK_B: (1_500.0, 600_000.0)}, COMPANY, "additions"),
    ]
    sqlite_store.save(Tables.sec13f_hr, pd.concat(books, ignore_index=True))
    snapshots = _broad_snapshots(sqlite_store, TAGS, UNIVERSE)
    shares = [float(_manager(df_compact, FILED).loc[(PERIOD, "BRK-B"), "shares"]) for _, df_compact, _ in snapshots]
    assert list(dict.fromkeys(shares)) == [3_100.0, 4_600.0]
    for _, df_compact, df_states in snapshots:
        pd.testing.assert_frame_equal(_manager(df_compact, FILED), _manager(df_compact, MERGED))
        status = df_states.set_index("cik")[["book_known", "zero_eligible", "comparison_eligible", "resolution_status"]]
        pd.testing.assert_series_equal(status.loc[FILED], status.loc[MERGED], check_names=False)
    print(
        f"SANITY ({stored} tickers stored): the original's BRK-B company row ({shares[0]:,.0f}) is kept and the additions' "
        f"BRK-A 1 x 1,500 is added ({shares[-1]:,.0f}); the merged books resolve to the same rows and states on every day."
    )


@pytest.mark.parametrize("stored", ["market", "company"])
def test_a_superinvestor_book_as_filed_holds_one_position_per_company(sqlite_store: Any, stored: str) -> None:
    tickers = MARKET if stored == "market" else COMPANY
    filed = {BRK_A: AS_FILED[BRK_A], BRK_B: AS_FILED[BRK_B], AVB: (10.0, 2_000.0)}
    merged = {BRK_B: AS_MERGED[BRK_B], AVB: (10.0, 2_000.0)}
    sqlite_store.save(Tables.sec13f_hr, pd.concat([_book(FILED, "2024-05-10", filed, tickers), _book(MERGED, "2024-05-10", merged, COMPANY)]))
    packets = iter_filing_vintages(sqlite_store, GRID.max(), [FILED, MERGED])
    assert packets is not None
    with closing(clean_public_batches(packets, TAGS, None, CLOSE, None)) as batches:
        snapshots = prepare_superinvestor_book_history(batches, GRID, 90)
    assert snapshots
    archive = snapshots[-1].archive
    columns = ["cusip", "ticker", "position_type", "shares", "value_usd", "class_evidence", "value_basis_shares_source"]
    books = {cik: archive[(cik, PERIOD)].df_positions.reindex(columns=columns).sort_values("cusip").reset_index(drop=True) for cik in (FILED, MERGED)}
    assert books[FILED][["cusip", "ticker", "shares"]].values.tolist() == [[AVB, "AVB", 10.0], [BRK_B, "BRK-B", 3_100.0]]
    pd.testing.assert_frame_equal(books[FILED], books[MERGED])
    print(
        f"SANITY ({stored} tickers stored): a 3-row superinvestor book as filed (marker 3) resolves, then holds one BRK-B position of "
        "3,100 B shares on the Class B CUSIP and AVB as filed: the top-10 rank sees one company position, as in the merged book."
    )


def test_an_invalid_marker_never_reaches_the_cube_input(sqlite_store: Any) -> None:
    empty = cast(tuple[str, str], Tables.sec13f_hr.empty_marker)[1]
    invalid = {"cik": MERGED, "period": PERIOD, "filing_date": pd.Timestamp("2024-05-10"), "accession_number": "invalid-1"}
    marker = {**invalid, "cusip": empty, "ticker": empty, "parse_status": "invalid", "value_unit_evidence": "declared rows disagree"}
    sqlite_store.save(Tables.sec13f_hr, pd.concat([_book(FILED, "2024-05-10", AS_FILED, MARKET), pd.DataFrame([marker])], ignore_index=True))
    assert "invalid-1" in set(sqlite_store.load(Tables.sec13f_hr, markers=True).accession_number), "the marker is stored"

    reads = {
        "whole books": iter_filing_vintages(sqlite_store, GRID.max(), [FILED, MERGED]),
        "universe-scoped": iter_filing_vintages(sqlite_store, GRID.max(), universe=UNIVERSE, cusips=universe_cusips(TAGS.lines, UNIVERSE)),
    }
    for name, packets in reads.items():
        assert packets is not None
        with closing(packets):
            batches = list(packets)
        accessions = {a for df_filings, df_holdings in batches for a in (*df_filings["accession_number"], *df_holdings["accession_number"])}
        assert accessions == {f"{FILED}-2024-05-10"}, (name, accessions)
    print("SANITY: an `invalid` 13F marker is in sec13f_hr but neither the whole-book nor the universe-scoped cube read returns it.")


def test_real_stored_books_reach_the_builders_at_company_grain() -> None:
    """Real rows: the managers holding a multi-class universe company in the newest complete stored quarter (read-only)."""
    store = _store()
    lines = CusipTags.load(store)
    tagged = lines.lines.loc[lines.lines["lineage_role"].isin(TAGGED_ROLES)]
    multi = tagged.groupby("canonical_company")["cusip"].nunique().loc[lambda counts: counts.gt(1)].index
    if not store.exists(Tables.sec13f_hr) or multi.empty:
        pytest.skip("no published universe_cusip class lines or no sec13f_hr: run identity first")
    universe = tuple(sorted(multi))
    cusips = sorted(universe_cusips(tagged, universe))
    period = store.distinct(Tables.sec13f_hr, "period", where={"parse_status": "position", "cusip": cusips})
    newest = sorted(pd.to_datetime(pd.Series(period)))[-2]  # the newest quarter still being filed may not be public yet
    holders = store.load(Tables.sec13f_hr, columns=["cik", "accession_number"], where={"parse_status": "position", "cusip": cusips, "period": newest})
    assert holders is not None
    # Managers whose every stored book is one original: a partially copied amended book elsewhere cannot stop the read.
    scope = {"cik": sorted(set(holders["cik"])), "parse_status": "complete"}
    markers = store.load(Tables.sec13f_hr, columns=["cik", "period", "amendment_type"], where=scope, markers=True)
    assert markers is not None
    amended = markers["amendment_type"].ne("original") | markers.duplicated(["cik", "period"], keep=False)
    ciks = sorted(set(markers["cik"]) - set(markers.loc[amended, "cik"]))[:25]
    rows = store.load(Tables.sec13f_hr, columns=["cik", "cusip", "shares"], where={"cik": ciks, "period": newest, "cusip": cusips})
    assert rows is not None

    # The builders' input: the step's own read, clean and resolve on these managers.
    packets = iter_filing_vintages(store, pd.Timestamp.today(), ciks, universe=universe, cusips=cusips)
    assert packets is not None
    with closing(clean_public_batches(packets, lines, universe, None, None)) as batches:
        history = prepare_institutional_book_history(batches, pd.bdate_range(newest - pd.Timedelta(days=400), pd.Timestamp.today()), settle_days=3)
    assert history is not None
    df_compact, _ = history
    df_quarter = df_compact.xs(newest, level="period")
    assert not df_compact.index.duplicated().any() and list(df_compact.columns) == ["shares", "value_usd", "call_value", "put_value", "_cusips"]

    # Independent truth: each stored row x the ratio of its dated line, summed per manager x company.
    start, end = pd.to_datetime(tagged["valid_from"]), pd.to_datetime(tagged["valid_to"])
    live = tagged.loc[(start.isna() | (start <= newest)) & (end.isna() | (newest < end))]
    truth = rows.merge(live[["cusip", "canonical_company", "conversion_ratio"]], on="cusip")
    truth = (truth["shares"] * truth["conversion_ratio"]).groupby([truth["cik"], truth["canonical_company"]]).sum()
    actual = df_quarter["shares"].rename_axis(["cik", "canonical_company"]).reindex(truth.index)
    checked = actual.notna()
    assert checked.sum() >= 10
    pd.testing.assert_series_equal(actual[checked], truth[checked], check_names=False, check_index_type=False, rtol=1e-9)
    print(
        f"SANITY: {int(checked.sum())} real manager x company books of {newest.date()} ({len(ciks)} managers, {universe}) reach the "
        "builders once per (period, manager, ticker), shares = stored company-CUSIP shares x the dated ratio."
    )
