"""
build_dataroma_roster_history.py  (scripts/)
--------------------------------------------------------------------------------------------
Regenerate the committed Dataroma roster history from the Wayback Machine: the full CDX of
dataroma.com/m/home.php, then per calendar quarter the newest capture that parses to a valid roster
(`quarterly_capture_candidates` + `history_from_captures` in fetch_superinvestors.py).

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
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_extract.utils.institutionals.fetch_superinvestors import (  # noqa: E402
    WAYBACK_CDX_URL,
    WaybackCapture,
    _http_get,
    _parse_dataroma_roster,
    history_from_captures,
    quarterly_capture_candidates,
)
from src.utils.superinvestor_roster import roster_history_path  # noqa: E402

#: Wayback answers 429 to a browser User-Agent sent by a non-browser client; identify the script honestly.
WAYBACK_HEADERS = {"User-Agent": "stock-pick-strat-roster-history/1.0 (read-only research)"}
#: Seconds to wait after each Wayback request.
POLITENESS_DELAY_S = 1.5
#: Attempts per Wayback request before the run fails (a rerun resumes from the cache).
FETCH_ATTEMPTS = 3

logger = logging.getLogger("build_dataroma_roster_history")


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
