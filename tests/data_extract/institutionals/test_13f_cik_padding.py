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


def test_the_walk_batch_save_pads_the_cik_in_both_tables(sqlite_store, monkeypatch):
    monkeypatch.setattr(f13, "build_cusip_ticker_map", lambda context, cusips: CMAP)
    ctx: Any = SimpleNamespace(store=sqlite_store)

    f13._save_batch(ctx, _book(_UNPADDED), {"AAPL"}, {ROSTER}, f13._WalkState())

    hr = sqlite_store.load(Tables.sec13f_hr, columns=["cik"])
    book = sqlite_store.load(Tables.sec13f_manager_holdings, columns=["cik"])
    assert list(hr["cik"]) == [ROSTER] and list(book["cik"]) == [ROSTER]
    print("\n=== SANITY: 13F walk CIK padding ===")
    print(f"  a book carrying cik {_UNPADDED!r} is stored as {ROSTER!r} in sec13f_hr and sec13f_manager_holdings. Validated.")


def test_the_manager_book_save_pads_the_cik(sqlite_store):
    ctx: Any = SimpleNamespace(store=sqlite_store)

    f13._save_book(ctx, _book(_UNPADDED))

    assert list(sqlite_store.load(Tables.sec13f_manager_holdings, columns=["cik"])["cik"]) == [ROSTER]
    print("\n=== SANITY: 13F manager book CIK padding ===")
    print(f"  the catch-up's book save stores {ROSTER!r} for an input cik {_UNPADDED!r}. Validated.")


def test_the_hr_save_pads_the_cik(sqlite_store):
    ctx: Any = SimpleNamespace(store=sqlite_store)
    hr = _book(_UNPADDED).merge(CMAP, on="cusip")[f13._HR_COLS]

    f13.save_hr(ctx, hr)

    assert list(sqlite_store.load(Tables.sec13f_hr, columns=["cik"])["cik"]) == [ROSTER]
    print("\n=== SANITY: 13F hr save CIK padding (walk and data-set backfill) ===")
    print(f"  save_hr stores {ROSTER!r} for an input cik {_UNPADDED!r}. Validated.")
