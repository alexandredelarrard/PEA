"""The insider completeness frontier: read from `insider_transactions` alone (markers included),
the same rule as 13D/13G; no run manifest."""

from __future__ import annotations

from typing import cast

import pandas as pd

from src.data_aggregate.utils.institutionals.frontiers import schedule_complete_through
from src.data_extract.utils.common.edgar_driver import FilingStamp, marker_row
from src.data_store.schema import Tables

LAST_SESSION = pd.Timestamp("2026-10-02")


def _rows(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """(ticker, accession, filing date, source) -> one nonderiv insider row each."""
    return pd.DataFrame(
        {
            "accession_number": [r[1] for r in rows],
            "security_type": "nonderiv",
            "row_sequence": 1,
            "ticker": [r[0] for r in rows],
            "filing_date": pd.to_datetime([r[2] for r in rows]),
            "transaction_date": pd.to_datetime([r[2] for r in rows]),
            "source": [r[3] for r in rows],
        }
    )


def test_the_frontier_is_the_last_session_only_while_the_table_is_fresh(sqlite_store) -> None:
    assert schedule_complete_through(sqlite_store, Tables.insider_transactions, LAST_SESSION) is None

    sqlite_store.save(Tables.insider_transactions, _rows([("AAA", "z1", "2026-06-20", "zip"), ("BBB", "e1", "2026-09-22", "edgar")]))
    stale = schedule_complete_through(sqlite_store, Tables.insider_transactions, LAST_SESSION)
    assert stale == pd.Timestamp("2026-09-22"), "10 days behind the last session: only through the latest filing"

    stamp = FilingStamp("m1", "4", "0000000001", pd.Timestamp("2026-09-29"), False, None, None)
    sqlite_store.save(Tables.insider_transactions, marker_row(Tables.insider_transactions, "CCC", stamp))
    fresh = schedule_complete_through(sqlite_store, Tables.insider_transactions, LAST_SESSION)
    assert fresh == LAST_SESSION, "an owner-role marker 3 days back keeps the table fresh through the last session"
    assert len(cast(pd.DataFrame, sqlite_store.load(Tables.insider_transactions))) == 2, "consumers never see the marker"
    assert stale is not None and fresh is not None

    print("\n=== SANITY CHECK: insider absence frontier from the DB ===")
    print(f"  empty -> None; latest filing 2026-09-22 vs last session {LAST_SESSION.date()} -> {stale.date()}")
    print(f"  marker filed 2026-09-29 (inside the 7-day overlap) -> {fresh.date()}")
    print("  CONCLUSION: the insider table uses the 13D/13G DB rule; no manifest is read. Validated.")
