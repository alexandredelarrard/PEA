"""
Dataroma roster HISTORY builder (src/data_extract/utils/institutionals/fetch_superinvestors.py).

Known-truth fixtures: a hand-written Wayback CDX and four tiny Dataroma home pages (a full roster,
a changed roster, a partial page and a hijacked-domain page). The builder keeps, per calendar
quarter, the newest capture that parses to a plausible roster; the seed stamps each snapshot at
its capture date with its own source URL.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from src.data_extract.utils.institutionals import fetch_superinvestors as si

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "dataroma"
HOME_HTTP = "http://www.dataroma.com:80/m/home.php"
HOME_HTTPS = "https://www.dataroma.com/m/home.php"

#: timestamp, status, original -- the `fl=timestamp,statuscode,original` CDX projection.
CDX = "\n".join(
    [
        f"20111220101010 200 {HOME_HTTP}",  # before `since`
        f"20120215105202 200 {HOME_HTTP}",
        f"20120307203705 200 {HOME_HTTP}",
        f"20120320000000 301 {HOME_HTTP}",  # redirect: not a capture of the page
        f"20120412093518 200 {HOME_HTTP}",
        f"20120412093518 200 {HOME_HTTPS}",  # same instant listed twice
        f"20120518011247 - {HOME_HTTP}",  # warc/revisit: no status of its own
        f"20121225141115 200 {HOME_HTTPS}",
        f"20121226000000 409 {HOME_HTTPS}",
        f"20130106214259 200 {HOME_HTTPS}",  # after `until`
        "",
    ]
)


def _capture(ts: str, original: str = HOME_HTTP) -> si.WaybackCapture:
    return si.WaybackCapture(timestamp=ts, original=original)


def _fixture_parse(pages: dict[str, str]):
    """`parse(capture) -> {code: name}` over the fixture page named for each timestamp."""

    def parse(capture: si.WaybackCapture) -> dict[str, str]:
        html = (FIXTURES / pages[capture.timestamp]).read_text(encoding="utf-8")
        return {e["code"]: e["name"] for e in si._parse_dataroma_roster(html)}

    return parse


def test_quarterly_captures_group_by_quarter_newest_first_and_drop_non_200():
    got = si.quarterly_capture_candidates(CDX, since=date(2012, 1, 1), until=date(2012, 12, 31))
    assert list(got) == ["2012Q1", "2012Q2", "2012Q4"]  # chronological; 2012Q3 had no capture
    assert [c.timestamp for c in got["2012Q1"]] == ["20120307203705", "20120215105202"]  # newest first, 301 dropped
    assert [c.timestamp for c in got["2012Q2"]] == ["20120412093518"]  # duplicate instant kept once, revisit dropped
    assert got["2012Q2"][0].original == HOME_HTTP  # the first listing of a duplicate instant wins
    assert [c.timestamp for c in got["2012Q4"]] == ["20121225141115"]  # 409 dropped
    assert got["2012Q4"][0].raw_url == f"https://web.archive.org/web/20121225141115id_/{HOME_HTTPS}"
    assert got["2012Q4"][0].captured_at == "2012-12-25T14:11:15Z"
    print("\n=== SANITY: Wayback captures per quarter ===")
    print(
        "  10 CDX lines -> 4 status-200 captures in 3 quarters, newest first; 301/409/revisit dropped, a "
        "duplicate instant kept once, raw `id_` URL keeps the CDX original. Validated."
    )


def test_quarterly_captures_respect_since_and_until():
    got = si.quarterly_capture_candidates(CDX, since=date(2012, 4, 1), until=date(2013, 1, 6))
    assert list(got) == ["2012Q2", "2012Q4", "2013Q1"]
    assert "2011Q4" not in si.quarterly_capture_candidates(CDX, until=date(2013, 12, 31))  # default floor is 2012
    assert si.ROSTER_HISTORY_START == date(2012, 1, 1)
    print("\n=== SANITY: capture window ===")
    print("  `since` drops 2011 and 2012Q1, `until` is inclusive of its day; the default floor is 2012-01-01. Validated.")


def test_build_history_rejects_blank_capture(caplog):
    """Per quarter, the newest VALID capture wins: a hijacked page (no manager) and a partial page
    (under 80% of the previous accepted roster) are rejected, the next older capture is tried, and
    a quarter with no valid capture is a logged gap."""
    candidates = {
        "2012Q1": [_capture("20120307203705")],
        "2012Q2": [_capture("20120620193720"), _capture("20120518011247"), _capture("20120412093518")],
        "2012Q3": [_capture("20120916084643")],
        "2012Q4": [_capture("20121225141115", HOME_HTTPS)],
    }
    pages = {
        "20120307203705": "roster_five.html",
        "20120620193720": "hijacked_domain.html",
        "20120518011247": "roster_partial.html",
        "20120412093518": "roster_five_changed.html",
        "20120916084643": "hijacked_domain.html",
        "20121225141115": "roster_five.html",
    }
    with caplog.at_level(logging.INFO, logger=si.logger.name):
        doc = si.history_from_captures(candidates, _fixture_parse(pages))

    assert list(doc) == ["_README", "snapshots"] and doc["_README"]
    snaps = doc["snapshots"]
    assert [s["captured_at"] for s in snaps] == ["2012-03-07T20:37:05Z", "2012-04-12T09:35:18Z", "2012-12-25T14:11:15Z"]
    assert [list(s) for s in snaps] == [["captured_at", "source_url", "managers"]] * 3
    assert snaps[1]["source_url"] == f"https://web.archive.org/web/20120412093518id_/{HOME_HTTP}"
    assert list(snaps[1]["managers"]) == ["brk", "GLRE", "psc", "BAUPOST", "AM"]  # page order kept
    assert snaps[0]["managers"]["brk"] == "Warren Buffett - Berkshire Hathaway -"
    log = caplog.text
    assert "20120620193720" in log and "20120518011247" in log  # both rejections named
    assert "2012Q3" in log and "gap" in log.lower()
    print("\n=== SANITY: quarterly history from captures ===")
    print(
        "  2012Q2: hijacked page (0 managers) and partial page (2 < 80% of 5) rejected, fell back to the "
        "older full capture; 2012Q3 (only a hijacked page) logged as a gap; 3 snapshots, each stamped "
        "at its capture instant with its raw Wayback URL. Validated."
    )


def test_build_history_first_quarter_needs_only_a_non_empty_roster():
    candidates = {"2012Q1": [_capture("20120307203705"), _capture("20120215105202")]}
    pages = {"20120307203705": "hijacked_domain.html", "20120215105202": "roster_partial.html"}
    doc = si.history_from_captures(candidates, _fixture_parse(pages))
    assert [len(s["managers"]) for s in doc["snapshots"]] == [2]
    print("\n=== SANITY: first quarter ===")
    print("  with no previous snapshot, any non-empty roster is accepted (2 managers); the empty page is not. Validated.")


def test_seed_stamps_capture_date(sqlite_store, tmp_path):
    """Each seeded snapshot is dated at its capture (never 1 January) and keeps its own source URL."""
    sub = tmp_path / "configs" / "superinvestors"
    sub.mkdir(parents=True)
    (sub / "overrides.json").write_text(json.dumps({"cik_overrides": {"brk": {"cik": "1067983"}}, "unresolvable": {}}), encoding="utf-8")
    url_a = f"https://web.archive.org/web/20130328071542id_/{HOME_HTTP}"
    url_b = f"https://web.archive.org/web/20130627231000id_/{HOME_HTTP}"
    history = {
        "_README": ["fixture"],
        "snapshots": [
            {"captured_at": "2013-03-28T07:15:42Z", "source_url": url_a, "managers": {"brk": "Warren Buffett - Berkshire Hathaway"}},
            {"captured_at": "2013-06-27T23:10:00Z", "source_url": url_b, "managers": {"brk": "Warren Buffett - Berkshire Hathaway"}},
        ],
    }
    (sub / "dataroma_roster_history.json").write_text(json.dumps(history), encoding="utf-8")

    def no_edgar(url):
        raise AssertionError(f"override must not hit EDGAR: {url}")

    ctx = cast(Any, SimpleNamespace(store=sqlite_store, config_dir=str(tmp_path / "configs")))
    si.seed_roster_history(ctx, get_fn=no_edgar)
    stored = sqlite_store.load(si.Tables.superinvestor_roster)
    got = sorted((str(d)[:10], u) for d, u in zip(stored["snapshot_date"], stored["source_url"], strict=True))
    assert got == [("2013-03-28", url_a), ("2013-06-27", url_b)]
    print("\n=== SANITY: seed stamping ===")
    print(
        f"  {len(stored)} rows: snapshot_date = capture date (2013-03-28, 2013-06-27, not 1 Jan) and each row keeps its own Wayback URL. Validated."
    )
