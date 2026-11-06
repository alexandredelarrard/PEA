"""U-67: `sec13f_hr` holds one row per manager x universe company x accession, in canonical-class shares.

Known-truth fixtures (the dated `universe_cusip` lines of BRK-B and GOOGL as published, hand-made books):
- a manager holding 1 BRK-A + 100 BRK-B in 2012 has one BRK-B row of 1,600 shares (1 x 1,500 + 100); in 2008, 130
  (1 x 30 + 100); the value is the sum as filed;
- GOOG + GOOGL in 2020 make one GOOGL row on Alphabet's Class A CUSIP;
- a holding outside the universe keeps its own CUSIP row and market ticker;
- the nightly walk saves the collapsed rows in `sec13f_hr` and the whole per-CUSIP book in `sec13f_manager_holdings`.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd

from src.data_extract.utils.institutionals import fetch_13f as f13
from src.data_store.schema import Tables
from src.utils.identity_tables import company_rows
from tests.data_extract.institutionals.test_13f_one_walk import ROSTER, _line

BRK_A, BRK_B = "084670108", "084670702"
GOOG, GOOGL, GOOG_OLD = "02079K107", "02079K305", "38259P508"
AVB = "053484101"
LINES = pd.DataFrame(
    [
        (BRK_B, "BRK-B", "canonical_current", "BRK-B", 1.0, "1998-12-31", None),
        (BRK_A, "BRK-B", "secondary_class", "BRK-A", 30.0, "1998-12-31", "2010-01-21"),
        (BRK_A, "BRK-B", "secondary_class", "BRK-A", 1500.0, "2010-01-21", None),
        (GOOG_OLD, "GOOGL", "canonical_predecessor", "GOOGL", 1.0, None, None),
        (GOOGL, "GOOGL", "canonical_current", "GOOGL", 1.0, "2015-10-01", None),
        (GOOG, "GOOGL", "secondary_class", "GOOG", 1.0, "2015-10-01", None),
        (AVB, "VMRK", "acquired_constituent", "AVB", 1.0, None, None),
    ],
    columns=["cusip", "canonical_company", "lineage_role", "market_symbol", "conversion_ratio", "valid_from", "valid_to"],
)
CMAP = pd.DataFrame({"cusip": [BRK_A, BRK_B, GOOG, GOOGL, AVB], "ticker": ["BRK-A", "BRK-B", "GOOG", "GOOGL", "AVB"]})


def _book(period: str, positions: list[tuple[str, float, float]]) -> pd.DataFrame:
    """One accession's per-CUSIP book: (cusip, shares, value)."""
    rows = pd.DataFrame(positions, columns=["cusip", "shares", "value_usd"])
    zero = {column: 0.0 for column in ("call_shares", "call_value", "put_shares", "put_value", "debt_prn", "debt_value", "other_value")}
    return rows.assign(
        cik=ROSTER,
        period=pd.Timestamp(period),
        filing_date=pd.Timestamp(period) + pd.Timedelta(days=40),
        accession_number=f"acc-{period}",
        issuer_name="ISSUER",
        title_of_class="COM",
        position_type="common",
        raw_value_usd=rows["value_usd"],
        raw_call_value=0.0,
        raw_put_value=0.0,
        raw_debt_value=0.0,
        raw_other_value=0.0,
        class_evidence='["COM"]',
        **zero,
    )


def test_two_berkshire_classes_make_one_row_in_class_b_shares_at_the_dated_ratio() -> None:
    for period, expected in (("2012-12-31", 1_600.0), ("2008-12-31", 130.0)):
        book = _book(period, [(BRK_A, 1.0, 120_000.0), (BRK_B, 100.0, 8_000.0), (AVB, 10.0, 2_000.0)])
        rows = company_rows(book, LINES, CMAP).set_index("ticker")
        assert len(rows) == 2
        assert rows.loc["BRK-B", "cusip"] == BRK_B and rows.loc["BRK-B", "shares"] == expected
        assert rows.loc["BRK-B", "value_usd"] == 128_000.0 and rows.loc["BRK-B", "raw_value_usd"] == 128_000.0
        assert rows.loc["AVB", "cusip"] == AVB and rows.loc["AVB", "shares"] == 10.0
        print(f"\n=== SANITY CHECK: Berkshire 1 A + 100 B at {period} ===")
        print(f"  one BRK-B row on {BRK_B}: {rows.loc['BRK-B', 'shares']:,.0f} B shares, value {rows.loc['BRK-B', 'value_usd']:,.0f} (sum as filed)")
        print("  AVB (outside the universe) keeps its own CUSIP row. Validated: dated ratio applied once, value conserved.")


def test_a_lone_class_a_holding_moves_onto_the_class_b_cusip() -> None:
    rows = company_rows(_book("2024-12-31", [(BRK_A, 2.0, 1_400_000.0)]), LINES, CMAP)
    assert rows[["ticker", "cusip", "shares", "value_usd"]].values.tolist() == [["BRK-B", BRK_B, 3_000.0, 1_400_000.0]]
    print("\n=== SANITY CHECK: 2 BRK-A alone in 2024 -> one BRK-B row of 3,000 B shares on the Class B CUSIP. Validated. ===")


def test_alphabet_classes_make_one_googl_row_on_the_current_class_a_cusip() -> None:
    book = _book("2020-12-31", [(GOOG, 10.0, 17_500.0), (GOOGL, 5.0, 8_750.0)])
    rows = company_rows(book, LINES, CMAP)
    assert rows[["ticker", "cusip", "shares", "value_usd"]].values.tolist() == [["GOOGL", GOOGL, 15.0, 26_250.0]]
    print("\n=== SANITY CHECK: GOOG 10 + GOOGL 5 at 2020-12-31 -> one GOOGL row of 15 shares on 02079K305. Validated. ===")


def test_a_book_without_a_second_class_is_unchanged() -> None:
    book = _book("2020-12-31", [(GOOGL, 5.0, 8_750.0), (AVB, 10.0, 2_000.0)])
    rows = company_rows(book, LINES, CMAP).sort_values("cusip").reset_index(drop=True)
    expected = book.assign(ticker=["GOOGL", "AVB"]).sort_values("cusip").reset_index(drop=True)
    pd.testing.assert_frame_equal(rows[expected.columns], expected, check_dtype=False)
    print("\n=== SANITY CHECK: a one-class book is stored row for row as before (ticker added only). Validated. ===")


def test_the_walk_saves_company_rows_in_sec13f_hr_and_the_whole_book_for_managers(filing_store, monkeypatch) -> None:
    monkeypatch.setattr(f13, "build_cusip_ticker_map", lambda context, cusips, **kwargs: CMAP)
    packet = f13.filing_vintage(
        pd.DataFrame([_line(BRK_A, "BERKSHIRE A", 1_400_000.0, 2), _line(BRK_B, "BERKSHIRE B", 47_000.0, 100), _line(AVB, "AVALONBAY", 2_000.0, 10)]),
        cik=ROSTER,
        accession_number="classes-source",
        period="2026-03-31",
        filing_date="2026-05-15",
        amendment_type="original",
        source="known_truth",
        declared_rows=3,
        declared_value=1_449_000.0,
    )
    ctx: Any = SimpleNamespace(store=filing_store)
    saver = f13._Saver(ctx, {"BRK-B"}, {ROSTER}, save_every=1)
    saver.walk.lines = LINES
    saver.save_packet(*packet)

    hr = filing_store.load(Tables.sec13f_hr, columns=["ticker", "cusip", "shares", "value_usd"]).set_index("ticker")
    book = filing_store.load(Tables.sec13f_manager_holdings, columns=["cusip", "shares"])
    assert sorted(hr.index) == ["AVB", "BRK-B"] and hr.loc["BRK-B", "shares"] == 3_100.0 and hr.loc["BRK-B", "cusip"] == BRK_B
    assert hr["value_usd"].sum() == 1_449_000.0 and len(book) == 3
    print("\n=== SANITY CHECK: the walk save of 2 BRK-A + 100 BRK-B + 10 AVB (2026) ===")
    print(f"  sec13f_hr: {len(hr)} rows, BRK-B {hr.loc['BRK-B', 'shares']:,.0f} B shares; value total {hr['value_usd'].sum():,.0f} = filed")
    print(f"  sec13f_manager_holdings: {len(book)} per-CUSIP rows (the whole book). Validated.")
