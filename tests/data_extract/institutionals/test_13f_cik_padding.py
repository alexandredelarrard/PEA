"""
test_13f_cik_padding.py (tests/data_extract/institutionals/test_13f_cik_padding.py)
------------------------------------------------------------------------------------
Every 13F writer stores the 10-digit padded CIK, whatever form reaches it: the nightly walk's batch
save (both tables), the manager catch-up's book save, and the data-set backfill. Real SQLite store.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd

from src.data_extract.utils.institutionals import fetch_13f as f13
from src.data_store.schema import Tables
from tests.data_extract.institutionals.test_13f_one_walk import CMAP, ROSTER, _line

_UNPADDED = ROSTER.lstrip("0")  # "1067983", the legacy form


def _book(cik: str) -> pd.DataFrame:
    book = f13._book_frame(ROSTER, "2026-05-15", "2026-03-31", pd.DataFrame([_line("037833100", "APPLE INC", 1_000.0, 10)]))
    return book.assign(cik=cik)


def _packet():
    return f13.filing_vintage(
        pd.DataFrame([_line("037833100", "APPLE INC", 1_000.0, 10)]),
        cik=_UNPADDED,
        accession_number="padding-source",
        period="2026-03-31",
        filing_date="2026-05-15",
        amendment_type="original",
        source="known_truth",
        declared_rows=1,
        declared_value=1000.0,
    )


def test_the_walk_save_pads_the_cik_in_both_tables(filing_store, monkeypatch):
    monkeypatch.setattr(f13, "build_cusip_ticker_map", lambda context, cusips, **kwargs: CMAP)
    ctx: Any = SimpleNamespace(store=filing_store)
    saver = f13._Saver(ctx, {"AAPL"}, {ROSTER}, save_every=1)
    saver.save_packet(*_packet())
    for table in [Tables.sec13f_hr, Tables.sec13f_manager_holdings]:
        assert filing_store.load(table, columns=["cik"])["cik"].tolist() == [ROSTER]
    print("SANITY: the actual saver stores padded source CIKs in both initialized destinations.")


def test_the_manager_filing_save_preserves_the_padded_cik(filing_store):
    f13.save_13f_filing(filing_store, *_packet())
    assert filing_store.load(Tables.sec13f_manager_holdings, columns=["cik"])["cik"].tolist() == [ROSTER]
    print("SANITY: manager persistence uses the same verified packet and padded source CIK.")


def test_the_hr_save_pads_the_cik(filing_store):
    ctx: Any = SimpleNamespace(store=filing_store)
    hr = _book(_UNPADDED).merge(CMAP, on="cusip")[f13._HR_COLS]

    f13.save_hr(ctx, hr.assign(accession_number="padding-direct"))

    assert list(filing_store.load(Tables.sec13f_hr, columns=["cik"])["cik"]) == [ROSTER]
    print("\n=== SANITY: 13F hr save CIK padding (walk and data-set backfill) ===")
    print(f"  save_hr stores {ROSTER!r} for an input cik {_UNPADDED!r}. Validated.")
