
from src.context import Context
from pathlib import Path
import json

from src.utils import polite_http as ph
from src.utils.crawler import Crawler

from src.constants.constants import (
    EARNINGS_CALL_CACHE_DIR,
    MOTLEY_FOOL_BASE_URL
    )
from src.data_extract.utils.common.bulk_cache import cache_dir

# --------------------------------------------------------------------------- #
# IO helpers                                                                    #
# --------------------------------------------------------------------------- #
_CRAWLER: Crawler | None = None

def _crawler() -> Crawler:
    global _CRAWLER
    if _CRAWLER is None:
        _CRAWLER = Crawler(retries=5, backoff=1.5, timeout=30, impersonate=True)
    return _CRAWLER


def _index_path(context: Context) -> Path:
    """The fool link index inside the transcript cache (data/call_transcripts/)."""
    return cache_dir(context, EARNINGS_CALL_CACHE_DIR) / "transcript_index.json"


def _load_index(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _get(url: str, timeout: int = 30, retries: int = 4, backoff: float = 3.0,
         log_missing: bool = True) -> str | None:
    """MF transcript GET -> HTML text on 200 (else None), via the shared IP-rotating `Crawler`
    (headless, no cookies/JS/images, rolling real-browser fingerprint, moving IPs over
    PEA_SCRAPE_PROXIES on a Cloudflare block, fast retry). `log_missing=False` silences the expected
    wrong-exchange 404 when probing quote pages. (timeout/retries/backoff are set on the shared
    crawler; kept in the signature for call-site compatibility.)"""
    return _crawler().get_text(url, log_missing=log_missing)


def _sleep_pace(base: float) -> None:
    """Paced inter-request sleep for fool.com (shared per-host slowdown + jitter)."""
    ph.sleep_pace(base, MOTLEY_FOOL_BASE_URL)


