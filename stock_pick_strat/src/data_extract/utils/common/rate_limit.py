"""
rate_limit.py  (src/data_extract/utils/rate_limit.py)
-----------------------------------------------------
Shared retry-with-backoff helper for the free data sources (yfinance, Google
Trends, ...) that rate-limit with HTTP 429. Instead of silently dropping the
symbol (which leaves the same subset permanently missing across runs), we WAIT
and retry with exponential backoff.
"""
from __future__ import annotations

import time
import logging 

logger = logging.getLogger(__name__)

def is_rate_limited(exc: BaseException) -> bool:
    """True if an exception looks like a rate-limit / 429 / too-many-requests."""
    s = f"{type(exc).__name__} {exc}".lower()
    return ("429" in s or "too many requests" in s or "toomanyrequests" in s
            or "rate limit" in s or "ratelimit" in s)

def call_with_retries(fn, *, retries: int = 3, base_wait: float = 30.0,
                      label: str = "", retry_empty=None):
    """Call `fn()`; on a rate-limit error, WAIT (exponential backoff) and retry up
    to `retries` times before giving up. Non-rate-limit exceptions propagate
    immediately. If `retry_empty` is given (a predicate on the result), an "empty"
    result is also retried (empties are often a soft throttle).

    `retries=3` => up to 4 attempts total; waits base_wait, 2x, 4x, ...
    """
    attempt = 0
    while True:
        try:
            result = fn()
        except Exception as e:                     # noqa: BLE001
            if is_rate_limited(e) and attempt < retries:
                wait = base_wait * (2 ** attempt)
                logger.warning(f"[{label}] rate-limited (429); attempt {attempt + 1}/{retries}"
                        f" -> waiting {wait:.0f}s before retry")
                time.sleep(wait)
                attempt += 1
                continue
            raise
        if retry_empty is not None and attempt < retries and retry_empty(result):
            wait = base_wait * (2 ** attempt)
            logger.warning(f"[{label}] empty response; attempt {attempt + 1}/{retries}"
                    f" -> waiting {wait:.0f}s before retry")
            time.sleep(wait)
            attempt += 1
            continue
        return result
