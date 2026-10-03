"""The live-test gate and its tripwire (`tests/live_guard.py`, wired in `tests/conftest.py`).

A plain `pytest` run must never write the live database or reach an external host: tests
that do are marked `@pytest.mark.live` and skip unless `PEA_LIVE_TESTS=1`. These checks pin
both halves on the default (not opted-in) run:

* every collected `live` item carries a skip marker, so its fixtures never set up;
* the tripwire refuses an external socket, a `curl_cffi` transfer and every `DataStore`
  write method on a non-SQLite engine, while loopback sockets stay open.

Deliberate trips are removed from `live_guard.TRIPS` so they do not fail the session.
"""

from __future__ import annotations

import socket
from collections.abc import Callable
from typing import Any

import live_guard
import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.data_store.store import DataStore

pytestmark = pytest.mark.skipif(live_guard.live_enabled(), reason=f"checks the default run; {live_guard.LIVE_FLAG}=1 is set")

#: Never connected: the guard must raise before SQLAlchemy opens a connection.
_PG_URL = "postgresql+psycopg2://guard:guard@127.0.0.1:1/guard"


def _expect_block(action: Callable[[], Any]) -> str:
    """Run `action`, require the tripwire to refuse it, and drop the deliberate trip."""
    before = len(live_guard.TRIPS)
    with pytest.raises(live_guard.LiveEffectBlockedError):
        action()
    trip = live_guard.TRIPS[before]
    del live_guard.TRIPS[before:]
    return trip


def test_every_live_item_is_skipped_before_setup(request: pytest.FixtureRequest) -> None:
    """A `live` item without a skip marker would set up its fixtures and run for real."""
    live = [item for item in request.session.items if item.get_closest_marker(live_guard.LIVE_MARKER)]
    unskipped = [item.nodeid for item in live if not any(mark.name == "skip" for mark in item.iter_markers())]

    print("\n=== SANITY CHECK: live items in this session are skipped ===")
    print(f"  collected items: {len(request.session.items)}; marked live: {len(live)}; live without a skip: {len(unskipped)}")
    assert not unskipped, f"live tests would run without {live_guard.LIVE_FLAG}=1: {unskipped}"
    print(f"  CONCLUSION: every live item skips before setup unless {live_guard.LIVE_FLAG}=1. Validated.")


def test_tripwire_blocks_network_and_live_db_writes() -> None:
    """The tripwire is installed and refuses each live path before it leaves the process."""
    from curl_cffi import requests as curl_requests

    blocked = [
        _expect_block(lambda: socket.create_connection(("192.0.2.1", 443), timeout=1)),
        _expect_block(lambda: curl_requests.get("https://192.0.2.1/", timeout=1)),
    ]
    store = DataStore(create_engine(_PG_URL))
    frame = pd.DataFrame({"ticker": ["ZZGUARD"]})
    writes: dict[str, Callable[[], Any]] = {
        "save": lambda: store.save("prices", frame),
        "bulk_seed": lambda: store.bulk_seed("prices", frame),
        "append_tail": lambda: store.append_tail("prices", frame, pd.Timestamp("2026-01-01")),
        "delete": lambda: store.delete("prices", {"ticker": "ZZGUARD"}),
        "replace": lambda: store.replace("prices", frame),
        "ensure_columns": lambda: store.ensure_columns("prices", frame),
        "drop": lambda: store.drop("prices"),
    }
    blocked += [_expect_block(action) for action in writes.values()]

    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=1):
            pass

    print("\n=== SANITY CHECK: tripwire refuses live effects ===")
    for trip in blocked:
        print(f"  blocked {trip}")
    print("  loopback socket: connected (local services stay reachable)")
    assert len(blocked) == 2 + len(writes)
    print(f"  CONCLUSION: network and all {len(writes)} DataStore write methods are refused on a non-SQLite engine. Validated.")
