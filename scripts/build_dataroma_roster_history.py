"""
build_dataroma_roster_history.py  (scripts/)
--------------------------------------------------------------------------------------------
Regenerate the committed Dataroma roster history from the Wayback Machine: the full CDX of
dataroma.com/m/home.php, then per calendar quarter the newest capture that parses to a valid roster
(`quarterly_capture_candidates` + `history_from_captures` below).

Every Wayback response is cached under --cache-dir (`cdx.txt`, `<timestamp>.html`, and
`<timestamp>.unavailable` when Wayback serves another capture instead), so a rerun is offline and
byte-identical. Read-only GETs to web.archive.org only; no database access.

    "$PY" scripts/build_dataroma_roster_history.py --cache-dir <dir> --until 2026-09-07
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
from collections.abc import Callable, Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any, NamedTuple, cast

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_extract.utils.institutionals.fetch_superinvestors import (  # noqa: E402
    ROSTER_CAPTURE_MIN_RATIO,
    _http_get,
    _parse_dataroma_roster,
)
from src.data_store.schema import Resume, Tables  # noqa: E402
from src.utils.superinvestor_roster import roster_history_path  # noqa: E402

#: Wayback answers 429 to a browser User-Agent sent by a non-browser client; identify the script honestly.
WAYBACK_HEADERS = {"User-Agent": "stock-pick-strat-roster-history/1.0 (read-only research)"}
#: Seconds to wait after each Wayback request.
POLITENESS_DELAY_S = 1.5
#: Attempts per Wayback request before the run fails (a rerun resumes from the cache).
FETCH_ATTEMPTS = 3
# Every Wayback capture of the Dataroma home page, uncollapsed: one `timestamp status original` line each.
WAYBACK_CDX_URL = "https://web.archive.org/cdx/search/cdx?url=dataroma.com/m/home.php&fl=timestamp,statuscode,original"
# A capture's archived bytes as served (`id_`: no Wayback rewrite).
_WAYBACK_RAW_URL = "https://web.archive.org/web/{timestamp}id_/{original}"
#: The `_README` lines of the generated history document.
ROSTER_HISTORY_README = [
    "Dataroma superinvestor roster history: one snapshot per calendar quarter, the newest valid Wayback capture of "
    "dataroma.com/m/home.php in that quarter.",
    "captured_at is the capture instant (UTC); `data_extract superinvestors -F` dates the snapshot at its day. "
    "source_url is the raw (id_) capture; managers maps Dataroma code -> name in page order.",
    f"A capture is valid when it lists at least one manager and at least {ROSTER_CAPTURE_MIN_RATIO:.0%} of the previous "
    "snapshot's count; a quarter with no valid capture is absent (a logged gap).",
    "Generated file, do not hand-edit: scripts/build_dataroma_roster_history.py --cache-dir DIR --until YYYY-MM-DD.",
]

logger = logging.getLogger("build_dataroma_roster_history")


class WaybackCapture(NamedTuple):
    """One Wayback capture of the Dataroma home page: its UTC `YYYYMMDDhhmmss` timestamp and archived URL."""

    timestamp: str
    original: str

    @property
    def raw_url(self) -> str:
        """The capture's unmodified archived bytes."""
        return _WAYBACK_RAW_URL.format(timestamp=self.timestamp, original=self.original)

    @property
    def captured_at(self) -> str:
        """The capture instant as `YYYY-MM-DDTHH:MM:SSZ`."""
        return datetime.strptime(self.timestamp, "%Y%m%d%H%M%S").strftime("%Y-%m-%dT%H:%M:%SZ")


def quarterly_capture_candidates(cdx: str, since: date | None = None, until: date | None = None) -> dict[str, list[WaybackCapture]]:
    """`{"YYYYQn": [capture, newest first]}` in chronological quarter order, over the status-200 lines of a
    `timestamp status original` CDX captured on days in [since, until]; a repeated timestamp counts once."""
    since = since if since is not None else date.fromisoformat(cast(str, cast(Resume, Tables.superinvestor_roster.resume).source_start))
    by_quarter: dict[str, list[WaybackCapture]] = {}
    seen: set[str] = set()
    for line in filter(None, (raw.strip() for raw in cdx.splitlines())):
        timestamp, status, original = line.split()
        day = datetime.strptime(timestamp[:8], "%Y%m%d").date()
        if status != "200" or timestamp in seen or day < since or (until is not None and day > until):
            continue
        seen.add(timestamp)
        by_quarter.setdefault(str(pd.Period(day, "Q")), []).append(WaybackCapture(timestamp, original))
    return {quarter: sorted(caps, key=lambda c: c.timestamp, reverse=True) for quarter, caps in sorted(by_quarter.items())}


def _first_valid_capture(
    quarter: str, captures: list[WaybackCapture], parse: Callable[[WaybackCapture], dict[str, str]], floor: float
) -> tuple[WaybackCapture, dict[str, str]] | None:
    """The first capture whose roster is non-empty and holds at least `floor` managers; each rejection is logged."""
    for capture in captures:
        managers = parse(capture)
        if managers and len(managers) >= floor:
            return capture, managers
        logger.warning("Roster history %s: rejected capture %s (%d managers, floor %.1f)", quarter, capture.timestamp, len(managers), floor)
    return None


def history_from_captures(candidates: Mapping[str, list[WaybackCapture]], parse: Callable[[WaybackCapture], dict[str, str]]) -> dict[str, Any]:
    """The committed roster history document: per quarter (chronological), the newest capture whose
    `parse(capture) -> {code: name}` is valid (see `ROSTER_CAPTURE_MIN_RATIO`); a quarter without one is a logged gap."""
    snapshots: list[dict[str, Any]] = []
    for quarter in sorted(candidates):
        floor = ROSTER_CAPTURE_MIN_RATIO * len(snapshots[-1]["managers"]) if snapshots else 0.0
        accepted = _first_valid_capture(quarter, candidates[quarter], parse, floor)
        if accepted is None:
            logger.warning("Roster history %s: no valid capture among %d -- gap", quarter, len(candidates[quarter]))
            continue
        capture, managers = accepted
        snapshots.append({"captured_at": capture.captured_at, "source_url": capture.raw_url, "managers": managers})
    logger.info("Roster history: %d snapshots from %d quarters with captures", len(snapshots), len(candidates))
    return {"_README": list(ROSTER_HISTORY_README), "snapshots": snapshots}


def _fetch(url: str) -> requests.Response:
    """`_http_get` with a polite delay and a short backoff retry on transport or HTTP errors."""
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            return _http_get(url, headers=WAYBACK_HEADERS)
        except requests.RequestException as e:
            if attempt == FETCH_ATTEMPTS:
                raise
            logger.warning("Wayback GET failed (attempt %d/%d): %s -- %s", attempt, FETCH_ATTEMPTS, url, e)
            time.sleep(POLITENESS_DELAY_S * 4 * attempt)
        finally:
            time.sleep(POLITENESS_DELAY_S)
    raise AssertionError("unreachable")


def load_cdx(cache_dir: Path) -> str:
    """The full, uncollapsed CDX text, from `cache_dir/cdx.txt` or fetched once and cached there."""
    path = cache_dir / "cdx.txt"
    if not path.exists():
        path.write_bytes(_fetch(WAYBACK_CDX_URL).content)
        logger.info("CDX fetched and cached: %s", path)
    return path.read_bytes().decode("utf-8")


def _decode(raw: bytes) -> str:
    """Capture bytes -> text: gunzip a still-compressed body, then UTF-8 with a cp1252 fallback."""
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def capture_html(capture: WaybackCapture, cache_dir: Path) -> str | None:
    """The capture's raw HTML (cached by timestamp), or None when Wayback answers with a different capture."""
    page, unavailable = cache_dir / f"{capture.timestamp}.html", cache_dir / f"{capture.timestamp}.unavailable"
    if not page.exists() and not unavailable.exists():
        response = _fetch(capture.raw_url)
        if f"/web/{capture.timestamp}id_/" in response.url:
            page.write_bytes(response.content)
        else:
            unavailable.write_text(response.url + "\n", encoding="utf-8")
    if unavailable.exists():
        logger.warning("Capture %s unavailable: Wayback served %s", capture.timestamp, unavailable.read_text(encoding="utf-8").strip())
        return None
    return _decode(page.read_bytes())


def parse_capture(capture: WaybackCapture, cache_dir: Path) -> dict[str, str]:
    """`{code: name}` of one capture, empty when the capture is unavailable or lists no manager."""
    html = capture_html(capture, cache_dir)
    return {} if html is None else {e["code"]: e["name"] for e in _parse_dataroma_roster(html)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", required=True, type=Path, help="Wayback response cache (created if absent).")
    parser.add_argument("--out", type=Path, default=roster_history_path(ROOT / "configs"), help="History JSON to write.")
    parser.add_argument("--until", type=date.fromisoformat, default=None, help="Last capture day kept (YYYY-MM-DD).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    candidates = quarterly_capture_candidates(load_cdx(args.cache_dir), until=args.until)
    doc = history_from_captures(candidates, lambda capture: parse_capture(capture, args.cache_dir))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes((json.dumps(doc, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
    snaps = doc["snapshots"]
    logger.info("Wrote %s: %d snapshots, %s -> %s", args.out, len(snaps), snaps[0]["captured_at"], snaps[-1]["captured_at"])


if __name__ == "__main__":
    main()
