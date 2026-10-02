"""
rate_limit.py  (src/data_extract/utils/common/rate_limit.py)
-----------------------------------------------------
Shared retry-with-backoff helper for free data sources that throttle with HTTP
429 or intermittently fail with retryable HTTP 5xx responses or transport
timeouts. Instead of silently dropping the symbol, wait and retry with
exponential backoff.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = frozenset({429, 502, 503, 504})
_RETRYABLE_CODE = re.compile(r"\b(429|50[234])\b")
_RETRYABLE_PHRASES = (
    "too many requests",
    "toomanyrequests",
    "rate limit",
    "ratelimit",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
    "readtimeout",
    "connecttimeout",
    "pooltimeout",
    "read operation timed out",
    "connection timed out",
)


def is_rate_limited(exc: BaseException) -> bool:
    """True for a throttle or transient upstream/transport failure worth retrying.

    An HTTP response's integer status code decides when present; otherwise the exception
    text is matched on whole status codes and known throttle/timeout phrases."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status in _RETRYABLE_STATUS
    text = f"{type(exc).__name__} {exc}".lower()
    return bool(_RETRYABLE_CODE.search(text)) or any(phrase in text for phrase in _RETRYABLE_PHRASES)


def _wait_before_retry(attempt: int, retries: int, base_wait: float, label: str, reason: str, on_retry: Callable[[], None] | None) -> None:
    """Sleep `base_wait * 2**attempt`, then run the optional `on_retry` hook; a hook failure is logged, not raised."""
    wait = base_wait * (2**attempt)
    rotation = " + rotating IP" if on_retry is not None else ""
    logger.warning(f"[{label}] {reason}; attempt {attempt + 1}/{retries} -> waiting {wait:.0f}s{rotation} before retry")
    time.sleep(wait)
    if on_retry is None:
        return
    try:
        on_retry()
    except Exception as e:  # noqa: BLE001
        logger.debug("on_retry hook failed (continuing): %s", e)


def call_with_retries[T](
    fn: Callable[[], T],
    *,
    retries: int = 3,
    base_wait: float = 30.0,
    label: str = "",
    retry_empty: Callable[[T], bool] | None = None,
    on_retry: Callable[[], None] | None = None,
) -> T:
    """Call `fn()`, retrying a retryable error (and, with `retry_empty`, an "empty" result) up to
    `retries` times with exponential waits `base_wait`, 2x, 4x, ... Non-retryable exceptions
    propagate immediately. `on_retry` runs before each retry, e.g. to move to the next proxy."""
    attempt = 0
    while True:
        try:
            result = fn()
        except Exception as e:  # noqa: BLE001
            if not (is_rate_limited(e) and attempt < retries):
                raise
            _wait_before_retry(attempt, retries, base_wait, label, "transient source error", on_retry)
            attempt += 1
            continue
        if retry_empty is None or attempt >= retries or not retry_empty(result):
            return result
        _wait_before_retry(attempt, retries, base_wait, label, "empty response", on_retry)
        attempt += 1
