"""Dataroma roster history seed: each snapshot is stamped at its capture date with its own source URL."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any, cast

from src.data_extract.utils.institutionals import fetch_superinvestors as si
from tests.fixtures.superinvestor_config import write_roster_config

HOME_HTTP = "http://www.dataroma.com:80/m/home.php"


def test_seed_stamps_capture_date(sqlite_store, tmp_path):
    """Each seeded snapshot is dated at its capture (never 1 January) and keeps its own source URL."""
    url_a = f"https://web.archive.org/web/20130328071542id_/{HOME_HTTP}"
    url_b = f"https://web.archive.org/web/20130627231000id_/{HOME_HTTP}"
    snapshots = [
        {"captured_at": "2013-03-28T07:15:42Z", "source_url": url_a, "managers": {"brk": "Warren Buffett - Berkshire Hathaway"}},
        {"captured_at": "2013-06-27T23:10:00Z", "source_url": url_b, "managers": {"brk": "Warren Buffett - Berkshire Hathaway"}},
    ]
    config_dir = write_roster_config(tmp_path, {"cik_overrides": {"brk": {"cik": "1067983"}}}, snapshots)

    def no_edgar(url):
        raise AssertionError(f"override must not hit EDGAR: {url}")

    ctx = cast(Any, SimpleNamespace(store=sqlite_store, config_dir=config_dir))
    si.rebuild_roster(ctx, get_fn=no_edgar, listing_fn=lambda cik: {date(2013, 3, 31)})  # activity gate: BRK filed 2013Q1
    stored = sqlite_store.load(si.Tables.superinvestor_roster)
    got = sorted((str(d)[:10], u) for d, u in zip(stored["snapshot_date"], stored["source_url"], strict=True))
    assert got == [("2013-03-28", url_a), ("2013-06-27", url_b)]
    print("\n=== SANITY: seed stamping ===")
    print(
        f"  {len(stored)} rows: snapshot_date = capture date (2013-03-28, 2013-06-27, not 1 Jan) and each row keeps its own Wayback URL. Validated."
    )
