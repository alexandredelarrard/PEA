"""REQ-005: `sec13f_hr` stores each 13F book as filed, one row per manager x CUSIP x accession.

Known truth (hand-made books, BRK-A/BRK-B and GOOG/GOOGL as separate `cusip_ticker_map` entries):
- a walk save of 2 BRK-A + 100 BRK-B + 10 AVB stores 3 rows on their own CUSIPs with the filed shares, tickers from
  `cusip_ticker_map` only (BRK-A stays BRK-A), and a marker counting the 3 filed rows;
- GOOG 10 + GOOGL 5 stay two rows, GOOG and GOOGL; a CUSIP the map does not know is stored on the sentinel.
The company sum x ratio is the cube's job (`test_sec_thirteen_company_sum.py`).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd

from src.data_extract.utils.institutionals import fetch_13f as f13
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from tests.data_extract.institutionals.test_13f_one_walk import ROSTER, _line

BRK_A, BRK_B = "084670108", "084670702"
GOOG, GOOGL = "02079K107", "02079K305"
AVB, UNMAPPED = "053484101", "999999999"
CMAP = pd.DataFrame({"cusip": [BRK_A, BRK_B, GOOG, GOOGL, AVB], "ticker": ["BRK-A", "BRK-B", "GOOG", "GOOGL", "AVB"]})


def _packet(accession: str, lines: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    info = pd.DataFrame(lines)
    return f13.filing_vintage(
        info,
        cik=ROSTER,
        accession_number=accession,
        period="2026-03-31",
        filing_date="2026-05-15",
        amendment_type="original",
        source="known_truth",
        declared_rows=len(info),
        declared_value=float(info["VALUE"].sum()),
    )


def _stored(store: DataStore, accession: str) -> tuple[pd.DataFrame, int]:
    rows = store.load(Tables.sec13f_hr, columns=["ticker", "cusip", "shares", "value_usd"], where={"accession_number": accession})
    markers = store.load(Tables.sec13f_hr, columns=["accession_number", "parse_status", "parsed_row_count"], markers=True)
    marker = markers.loc[markers["accession_number"].eq(accession) & markers["parse_status"].eq("complete"), "parsed_row_count"]
    return rows.sort_values("cusip", ignore_index=True), int(marker.iloc[0])


def test_the_walk_stores_each_share_class_as_filed_with_its_map_ticker(filing_store: DataStore, monkeypatch: Any) -> None:
    monkeypatch.setattr(f13, "build_cusip_ticker_map", lambda context, cusips, **kwargs: CMAP)
    berkshire = _packet(
        "classes-brk",
        [_line(BRK_A, "BERKSHIRE A", 1_400_000.0, 2), _line(BRK_B, "BERKSHIRE B", 47_000.0, 100), _line(AVB, "AVALONBAY", 2_000.0, 10)],
    )
    alphabet = _packet(
        "classes-goog",
        [_line(GOOG, "ALPHABET C", 1_750.0, 10), _line(GOOGL, "ALPHABET A", 875.0, 5), _line(UNMAPPED, "NOBODY", 100.0, 1)],
    )
    saver = f13._Saver(SimpleNamespace(store=filing_store), save_every=1)
    saver.save_packet(*berkshire)
    saver.save_packet(*alphabet)

    brk, brk_marker = _stored(filing_store, "classes-brk")
    goog, goog_marker = _stored(filing_store, "classes-goog")
    sentinel = Tables.sec13f_hr.empty_marker[1]
    assert brk.values.tolist() == [
        ["AVB", AVB, 10.0, 2_000.0],
        ["BRK-A", BRK_A, 2.0, 1_400_000.0],
        ["BRK-B", BRK_B, 100.0, 47_000.0],
    ], brk
    assert brk_marker == 3, "the marker counts the filed rows"
    assert goog.values.tolist() == [["GOOG", GOOG, 10.0, 1_750.0], ["GOOGL", GOOGL, 5.0, 875.0], [sentinel, UNMAPPED, 1.0, 100.0]], goog
    assert goog_marker == 3
    print("\n=== SANITY CHECK: 13F books stored as filed (REQ-005) ===")
    print(f"  Berkshire book: {len(brk)} rows {brk[['ticker', 'shares']].values.tolist()}, marker {brk_marker} = filed lines")
    print(f"  Alphabet book: {len(goog)} rows {goog[['ticker', 'shares']].values.tolist()}, marker {goog_marker}")
    print("  Validated: no class merge, no ratio, tickers from cusip_ticker_map only (unmapped on the sentinel).")
