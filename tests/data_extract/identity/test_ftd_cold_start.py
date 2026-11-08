"""The identity FTD capture on a store whose FTD tables were dropped (the identity cold start)."""

from __future__ import annotations

from types import SimpleNamespace

from src.data_extract.utils.identity import cusip_history as sm
from src.data_extract.utils.zips import fetch_fails_to_deliver as ftd
from src.data_store.schema import Tables
from src.data_store.store import DataStore


def test_the_first_capture_creates_the_dropped_ftd_tables(sqlite_store: DataStore) -> None:
    """With neither FTD table present, an empty capture saves nothing and a captured line creates the raw table."""
    context = SimpleNamespace(store=sqlite_store)
    raw = ftd._parse_ftd_lines("SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n20240102|037833100|AAPL|10|APPLE INC|10\n").assign(
        period="202401a", security_id=None, ticker=None, lineage_role=None, security_class=None
    )
    assert not sqlite_store.exists(Tables.sec_fails_to_deliver) and not sqlite_store.exists(Tables.sec_fails_to_deliver_security)

    assert sm.save_ftd_lines(context, raw.iloc[:0], ["202401a"]) == 0
    assert sm.save_ftd_lines(context, raw, ["202401a"]) == 1
    assert sm.save_ftd_lines(context, raw, ["202401a"]) == 1

    assert sqlite_store.row_count(Tables.sec_fails_to_deliver_security) == 1
    print("\nSANITY: with no FTD table, an empty capture saves nothing, a captured line creates the table, and a rerun replaces it.")
