"""
polite_http.py  (src/utils/polite_http.py)
------------------------------------------
Shared, source-agnostic anti-429 HTTP toolkit for extractors that scrape rate-limited public
endpoints (Motley Fool behind Cloudflare, Wikimedia, ...). We COOPERATE with rate limits, we
do NOT evade bans:

  * rotate a REAL browser IMPERSONATION profile per request (curl_cffi -> a coherent UA + TLS +
    header fingerprint; a python-requests JA3 gets blocked within a few calls). Plain-requests
    fallback when curl_cffi is unavailable / impersonation is off.
  * honour Retry-After; exponential backoff WITH JITTER on 429 / 403 / 5xx.
  * PER-HOST run-wide slowdown: a 429 from one host ratchets only THAT host's pace, so Google's
    throttling never slows Wikimedia and vice-versa.
  * optional bring-your-own proxy via env (PEA_SCRAPE_PROXY / HTTPS_PROXY). We deliberately do
    NOT ship or rotate anonymous / residential proxy pools (that is ban-evasion, not politeness).

Used by `fetch_earnings_calls` (Cloudflare) and `fetch_wiki_pageviews`; `fetch_google_trends`
keeps its bespoke cookie/token session client but shares `resolve_proxy`.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import random
import time
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

import requests
from curl_cffi import requests as cr

logger = logging.getLogger(__name__)

# curl_cffi impersonation targets (each = a coherent UA+TLS+header profile of a real browser)
IMPERSONATE_POOL = ("chrome124", "chrome123", "chrome120", "chrome131", "safari17_0", "edge101")
_PROXY_ENV = ("PEA_SCRAPE_PROXY", "HTTPS_PROXY", "https_proxy")
_UA_POOL = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
)
_PACE_CAP = 8.0
_PACE: dict[str, float] = {}          # host -> run-wide slowdown multiplier, ratcheted on 429


def resolve_proxy() -> dict | None:
    """`{'http':.., 'https':..}` from PEA_SCRAPE_PROXY / HTTPS_PROXY, else None (BYO proxy)."""
    for k in _PROXY_ENV:
        v = os.getenv(k)
        if v:
            return {"http": v, "https": v}
    return None


def random_headers(extra: dict | None = None) -> dict:
    """A realistic desktop-browser header set with a rotated UA (for the requests path)."""
    h = {"User-Agent": random.choice(_UA_POOL),
         "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
         "Accept-Language": "en-US,en;q=0.9", "Connection": "keep-alive"}
    if extra:
        h.update(extra)
    return h


def retry_after_seconds(resp) -> float | None:
    """Parse a Retry-After header (delta-seconds or HTTP-date) into seconds, if present."""
    try:
        ra = resp.headers.get("Retry-After")
    except Exception:
        return None
    if not ra:
        return None
    try:
        return float(ra)
    except (TypeError, ValueError):
        try:
            return max(0.0, (parsedate_to_datetime(ra)
                             - _dt.datetime.now(_dt.timezone.utc)).total_seconds())
        except Exception:
            return None


def _host(url_or_host: str) -> str:
    try:
        return urlsplit(url_or_host).netloc or url_or_host
    except Exception:
        return url_or_host


def note_throttle(url_or_host: str) -> None:
    """Ratchet up the per-host slowdown after a 429 (bounded at _PACE_CAP)."""
    h = _host(url_or_host)
    _PACE[h] = min(_PACE_CAP, _PACE.get(h, 1.0) * 1.6)


def pace_mult(url_or_host: str) -> float:
    return _PACE.get(_host(url_or_host), 1.0)


def sleep_pace(base: float, url_or_host: str = "") -> None:
    """Inter-request sleep = base * per-host-pace + jitter (non-robotic cadence that honours the
    post-429 slowdown). `base <= 0` disables throttling (tests)."""
    if base <= 0:
        return
    time.sleep(base * pace_mult(url_or_host) + random.uniform(0.1, 0.7))


def _raw_get(url, *, params=None, headers=None, timeout=30, impersonate=True):
    """ONE GET. curl_cffi with a ROTATED impersonation profile when `impersonate` (best for
    Cloudflare/JA3), else a plain requests GET (friendly REST APIs). Returns a response
    (.status_code/.text/.headers/.json) or None on a transport error. Isolated so tests can
    monkeypatch the transport."""
    proxies = resolve_proxy()
    if impersonate:
        try:
            prof = random.choice(IMPERSONATE_POOL)
            try:
                return cr.get(url, params=params, headers=headers, impersonate=prof,
                              timeout=timeout, proxies=proxies)
            except Exception:                       # unknown profile / TLS quirk -> generic chrome
                return cr.get(url, params=params, headers=headers, impersonate="chrome",
                              timeout=timeout, proxies=proxies)
        except Exception:
            pass                                    # curl_cffi transport error -> requests fallback
    try:
        return requests.get(url, params=params, headers=headers or random_headers(),
                            timeout=timeout, proxies=proxies)
    except Exception as exc:                            # noqa: BLE001
        # Log the CAUSE. Swallowing it silently left callers with only "GET failed
        # (transport)", which cannot distinguish an SSL/CA problem from DNS, a dead
        # proxy or a timeout -- the retry loop then burns its 4 attempts on it.
        logger.debug("transport error on %s: %s: %s", url, type(exc).__name__, exc)
        return None


def http_get(url, *, params=None, headers=None, timeout=30, retries=4, backoff=3.0,
             impersonate=True, log_missing=True):
    """Adaptive GET: rotated browser impersonation + retry with exponential backoff + jitter,
    honouring Retry-After and ratcheting a PER-HOST slowdown on each 429. Returns the response
    on HTTP 200; None on a terminal non-200 (logged unless `log_missing=False`) or transport
    failure. `impersonate=False` uses a plain requests GET (for friendly APIs that want their
    own descriptive User-Agent, e.g. Wikimedia)."""
    for attempt in range(retries + 1):
        r = _raw_get(url, params=params, headers=headers, timeout=timeout, impersonate=impersonate)
        if r is None:                                    # transport error (all paths failed)
            if attempt < retries:
                time.sleep(backoff * (2 ** attempt) + random.uniform(0.5, 2.0))
                continue
            logger.warning("GET failed (transport) %s", url)
            return None
        code = getattr(r, "status_code", 0)
        if code == 200:
            return r
        if code in (403, 429) or code >= 500:            # blocked / throttled / transient
            if attempt < retries:
                wait = max(retry_after_seconds(r) or 0.0,
                           backoff * (2 ** attempt)) + random.uniform(0.5, 2.5)
                if code == 429:
                    note_throttle(url)                   # slow THIS host for the rest of the run
                logger.warning("GET %s -> HTTP %d (rate-limited); wait %.1fs, host-pace x%.1f "
                               "(retry %d/%d)", url, code, wait, pace_mult(url), attempt + 1, retries)
                time.sleep(wait)
                continue
            logger.warning("GET %s -> HTTP %d after %d retries; giving up. Lower the rate or set "
                           "PEA_SCRAPE_PROXY.", url, code, retries)
        elif log_missing:                                # 4xx that won't fix on retry (404 etc.)
            logger.warning("GET %s -> HTTP %d", url, code)
        return None
    return None


def get_text(url, **kw) -> str | None:
    r = http_get(url, **kw)
    return r.text if r is not None else None


def get_json(url, **kw):
    r = http_get(url, **kw)
    if r is None:
        return None
    try:
        return r.json()
    except Exception:
        return None
