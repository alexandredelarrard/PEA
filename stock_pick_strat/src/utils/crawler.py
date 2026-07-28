"""
crawler.py  (src/utils/crawler.py)
----------------------------------
A clean, fast, low-footprint HTTP crawler for rate-limited public endpoints (Google Trends, Motley
Fool, Wikimedia, ...). Same ETHOS as `polite_http`: cooperate with rate limits and keep a minimal
fingerprint. For IP rotation it uses ONLY proxies YOU supply and are authorized to use
(`PEA_SCRAPE_PROXIES`); it does NOT fetch, scrape or bundle anonymous / residential proxy pools and
does NOT solve CAPTCHAs — that is ban-evasion, not crawling hygiene (and it's brittle).

Design (each request is independent, so a daily crawl stays fast):
  * HEADLESS / STATELESS — plain HTTP GETs, no browser: no JavaScript, no images, and NO cookie jar
    is carried between requests (nothing to fingerprint or expire).
  * ROLLING fingerprint — a fresh REAL-browser TLS impersonation (curl_cffi) + a rotated
    User-Agent / header set on EVERY request (a python-requests JA3 is blocked within a few calls).
  * MOVING IPs — on a detected block (403 / 429 / 503) it advances to the NEXT proxy in your
    configured list before retrying, so a flagged exit IP is dropped immediately. With no proxies
    configured it retries direct (you can't rotate IPs you don't have — supply your own).
  * FAST adaptive retry — short exponential backoff + jitter, honouring Retry-After, plus the
    shared per-host slowdown from `polite_http` (one host's throttle never slows another).

Configure the proxy pool (comma-separated, each an authorized proxy URL you own/rent):
    export PEA_SCRAPE_PROXIES="http://user:pass@host1:port,http://user:pass@host2:port"
Falls back to the single-proxy envs `polite_http` already honours (PEA_SCRAPE_PROXY / HTTPS_PROXY).
"""
from __future__ import annotations

import logging
import os
import random
import time
from urllib.parse import urlsplit, urlunsplit

import requests
from curl_cffi import requests as cr

from src.utils import polite_http as ph

logger = logging.getLogger(__name__)

# comma-separated list of YOUR authorized proxies (rotated on a detected block). No anonymous pools.
PROXY_POOL_ENV = "PEA_SCRAPE_PROXIES"
# HTTP status codes that mean "detected / throttled" -> rotate IP + retry
DEFAULT_ROTATE_ON = (403, 407, 429, 503)


def load_proxy_pool() -> list[str]:
    """The ordered list of authorized proxy URLs from PEA_SCRAPE_PROXIES (comma-separated); falls
    back to the single proxy in PEA_SCRAPE_PROXY / HTTPS_PROXY; else [] (direct connection)."""
    raw = os.getenv(PROXY_POOL_ENV)
    if raw:
        pool = [p.strip() for p in raw.split(",") if p.strip()]
        if pool:
            return pool
    single = ph.resolve_proxy()                       # {'http':.., 'https':..} or None
    return [single["https"]] if single else []


def _mask(proxy: str | None) -> str:
    """Proxy string with any user:pass credentials stripped, safe for logs."""
    if not proxy:
        return "direct"
    try:
        s = urlsplit(proxy)
        netloc = s.hostname or ""
        if s.port:
            netloc += f":{s.port}"
        return urlunsplit((s.scheme, netloc, "", "", "")) or "proxy"
    except Exception:                                 # noqa: BLE001
        return "proxy"


class Crawler:
    """Stateless rolling-fingerprint HTTP GET crawler with authorized-proxy rotation on block.

    Example:
        crawler = Crawler()                 # picks up PEA_SCRAPE_PROXIES if set
        html = crawler.get_text(url)        # None on a terminal failure
    """

    def __init__(self, *, retries: int = 5, backoff: float = 1.0, timeout: int = 25,
                 impersonate: bool = True, proxies: list[str] | None = None,
                 rotate_on: tuple[int, ...] = DEFAULT_ROTATE_ON, max_backoff: float = 20.0,
                 log_missing: bool = True) -> None:
        self._retries = int(retries)
        self._backoff = float(backoff)               # small base -> FAST retry
        self._max_backoff = float(max_backoff)
        self._timeout = int(timeout)
        self._impersonate = bool(impersonate)
        self._rotate_on = tuple(rotate_on)
        self._log_missing = bool(log_missing)
        self._proxies = list(proxies) if proxies is not None else load_proxy_pool()
        random.shuffle(self._proxies)                 # don't hammer the same first IP every process
        self._i = 0
        logger.info("Crawler ready: %d authorized proxy(ies) [%s], retries=%d",
                    len(self._proxies), ", ".join(_mask(p) for p in self._proxies) or "direct",
                    self._retries)

    # ------------------------------------------------------------------ #
    @property
    def n_proxies(self) -> int:
        return len(self._proxies)

    def _current_proxy(self) -> str | None:
        return self._proxies[self._i % len(self._proxies)] if self._proxies else None

    def _rotate(self) -> None:
        """Advance to the next authorized proxy (no-op when none / only one configured)."""
        if self._proxies:
            self._i = (self._i + 1) % len(self._proxies)

    def _raw_get(self, url: str, params: dict | None, headers: dict, proxy: str | None):
        """ONE stateless GET. curl_cffi with a ROTATED real-browser impersonation (best vs JA3
        fingerprinting), else plain requests. No Session -> no cookies persist. Returns a response
        (.status_code/.text/.headers/.json) or None on a transport error."""
        proxies = {"http": proxy, "https": proxy} if proxy else None
        if self._impersonate:
            try:
                prof = random.choice(ph.IMPERSONATE_POOL)
                try:
                    return cr.get(url, params=params, headers=headers, impersonate=prof,
                                  timeout=self._timeout, proxies=proxies, allow_redirects=True)
                except Exception:                     # unknown profile / TLS quirk -> generic chrome
                    return cr.get(url, params=params, headers=headers, impersonate="chrome",
                                  timeout=self._timeout, proxies=proxies, allow_redirects=True)
            except Exception:
                pass                                  # curl_cffi transport error -> requests fallback
        try:
            return requests.get(url, params=params, headers=headers, timeout=self._timeout,
                                proxies=proxies, allow_redirects=True)
        except Exception:
            return None

    def _wait(self, attempt: int, resp) -> float:
        """Fast backoff: max(Retry-After, base * 1.7**attempt) + jitter, capped."""
        ra = ph.retry_after_seconds(resp) if resp is not None else None
        base = min(self._max_backoff, self._backoff * (1.7 ** attempt))
        return min(self._max_backoff, max(ra or 0.0, base)) + random.uniform(0.1, 0.6)

    # ------------------------------------------------------------------ #
    def get(self, url: str, *, params: dict | None = None, headers: dict | None = None,
            log_missing: bool | None = None):
        """Fetch `url`, rotating IP + retrying on a detected block. Returns the response on HTTP 200,
        else None (a non-retryable 4xx like 404, or all retries exhausted). `headers` overrides the
        rolling browser headers (e.g. a friendly API's descriptive UA); `log_missing` overrides the
        instance default for this call (silence the expected 404 when probing)."""
        lm = self._log_missing if log_missing is None else log_missing
        for attempt in range(self._retries + 1):
            proxy = self._current_proxy()
            hdrs = headers or ph.random_headers()     # ROLLING headers per request
            ph.sleep_pace(0.0, url)                   # honour any accumulated per-host slowdown
            r = self._raw_get(url, params, hdrs, proxy)
            code = getattr(r, "status_code", 0) if r is not None else 0
            if code == 200:
                return r
            blocked = (r is None) or (code in self._rotate_on) or (code >= 500)
            if blocked and attempt < self._retries:
                if code == 429:
                    ph.note_throttle(url)             # slow THIS host for the rest of the run
                self._rotate()                        # MOVE IP before the retry
                wait = self._wait(attempt, r)
                logger.warning("crawl %s -> %s; rotate IP -> %s, wait %.1fs (retry %d/%d)",
                               url, code or "conn-fail", _mask(self._current_proxy()),
                               wait, attempt + 1, self._retries)
                time.sleep(wait)
                continue
            if not blocked and lm:                    # e.g. 404 -> won't fix on retry
                logger.warning("crawl %s -> HTTP %d", url, code)
            if blocked:
                logger.warning("crawl %s -> %s; giving up after %d retries "
                               "(configure PEA_SCRAPE_PROXIES with more authorized proxies)",
                               url, code or "conn-fail", self._retries)
            return None
        return None

    def get_text(self, url: str, *, log_missing: bool | None = None, **kw) -> str | None:
        r = self.get(url, log_missing=log_missing, **kw)
        return r.text if r is not None else None

    def get_json(self, url: str, *, log_missing: bool | None = None, **kw):
        r = self.get(url, log_missing=log_missing, **kw)
        if r is None:
            return None
        try:
            return r.json()
        except Exception:                             # noqa: BLE001
            return None
