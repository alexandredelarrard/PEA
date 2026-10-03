"""Stateless HTTP crawler for rate-limited public endpoints.

Each GET is independent: no cookie jar, a rotated real-browser TLS impersonation (curl_cffi) and header set per
request, short exponential backoff honouring Retry-After plus `polite_http`'s per-host slowdown. On a block
(403/407/429/503 or 5xx) it advances to the next proxy in `PEA_SCRAPE_PROXIES` (comma-separated, user-supplied
and authorized only; falls back to PEA_SCRAPE_PROXY / HTTPS_PROXY, else direct). It never sources proxy pools or solves CAPTCHAs.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests
from curl_cffi import requests as cr

from src.utils import polite_http as ph

logger = logging.getLogger(__name__)

# Comma-separated list of user-supplied authorized proxies, rotated on a detected block.
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
    single = ph.resolve_proxy()  # {'http':.., 'https':..} or None
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
    except Exception:  # noqa: BLE001
        return "proxy"


class Crawler:
    """Stateless rolling-fingerprint HTTP GET crawler with authorized-proxy rotation on block.

    Example:
        crawler = Crawler()                 # picks up PEA_SCRAPE_PROXIES if set
        html = crawler.get_text(url)        # None on a terminal failure
    """

    def __init__(
        self,
        *,
        retries: int = 5,
        backoff: float = 1.0,
        timeout: int = 25,
        impersonate: bool = True,
        proxies: list[str] | None = None,
        rotate_on: tuple[int, ...] = DEFAULT_ROTATE_ON,
        max_backoff: float = 20.0,
        log_missing: bool = True,
    ) -> None:
        self._retries = int(retries)
        self._backoff = float(backoff)  # small base -> FAST retry
        self._max_backoff = float(max_backoff)
        self._timeout = int(timeout)
        self._impersonate = bool(impersonate)
        self._rotate_on = tuple(rotate_on)
        self._log_missing = bool(log_missing)
        self._proxies = list(proxies) if proxies is not None else load_proxy_pool()
        random.shuffle(self._proxies)  # don't hammer the same first IP every process
        self._i = 0
        logger.info(
            "Crawler ready: %d authorized proxy(ies) [%s], retries=%d",
            len(self._proxies),
            ", ".join(_mask(p) for p in self._proxies) or "direct",
            self._retries,
        )

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
        """One stateless GET: curl_cffi with a random browser impersonation, else plain requests; None on a transport error."""
        proxies: Any = {"http": proxy, "https": proxy} if proxy else None
        if self._impersonate:
            try:
                prof = random.choice(ph.IMPERSONATE_POOL)
                try:
                    return cr.get(url, params=params, headers=headers, impersonate=prof, timeout=self._timeout, proxies=proxies, allow_redirects=True)
                except Exception:  # unknown profile / TLS quirk -> generic chrome
                    return cr.get(
                        url, params=params, headers=headers, impersonate="chrome", timeout=self._timeout, proxies=proxies, allow_redirects=True
                    )
            except Exception:
                pass  # curl_cffi transport error -> requests fallback
        try:
            return requests.get(url, params=params, headers=headers, timeout=self._timeout, proxies=proxies, allow_redirects=True)
        except Exception:
            return None

    def _wait(self, attempt: int, resp) -> float:
        """Fast backoff: max(Retry-After, base * 1.7**attempt) + jitter, capped."""
        ra = ph.retry_after_seconds(resp) if resp is not None else None
        base = min(self._max_backoff, self._backoff * (1.7**attempt))
        return min(self._max_backoff, max(ra or 0.0, base)) + random.uniform(0.1, 0.6)

    # ------------------------------------------------------------------ #
    def get(self, url: str, *, params: dict | None = None, headers: dict | None = None, log_missing: bool | None = None):
        """Fetch `url`, rotating proxy and retrying on a detected block; the response on HTTP 200, else None.

        `headers` overrides the rolling browser headers; `log_missing` overrides the instance default for this call
        (to silence an expected 404 when probing)."""
        lm = self._log_missing if log_missing is None else log_missing
        for attempt in range(self._retries + 1):
            proxy = self._current_proxy()
            hdrs = headers or ph.random_headers()  # ROLLING headers per request
            ph.sleep_pace(0.0, url)  # honour any accumulated per-host slowdown
            r = self._raw_get(url, params, hdrs, proxy)
            code = getattr(r, "status_code", 0) if r is not None else 0
            if code == 200:
                return r
            blocked = (r is None) or (code in self._rotate_on) or (code >= 500)
            if blocked and attempt < self._retries:
                if code == 429:
                    ph.note_throttle(url)  # slow THIS host for the rest of the run
                self._rotate()  # MOVE IP before the retry
                wait = self._wait(attempt, r)
                logger.warning(
                    "crawl %s -> %s; rotate IP -> %s, wait %.1fs (retry %d/%d)",
                    url,
                    code or "conn-fail",
                    _mask(self._current_proxy()),
                    wait,
                    attempt + 1,
                    self._retries,
                )
                time.sleep(wait)
                continue
            if not blocked and lm:  # e.g. 404 -> won't fix on retry
                logger.warning("crawl %s -> HTTP %d", url, code)
            if blocked:
                logger.warning(
                    "crawl %s -> %s; giving up after %d retries (configure PEA_SCRAPE_PROXIES with more authorized proxies)",
                    url,
                    code or "conn-fail",
                    self._retries,
                )
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
        except Exception:  # noqa: BLE001
            return None
