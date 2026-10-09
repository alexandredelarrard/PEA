"""The one retry policy for every SEC request.

HTTP 429/502/503/504 are retried up to `attempts` calls in total, waiting the policy's growing
waits; a Retry-After (from `TooManyRequestsError.retry_after` or the response header) lengthens a
wait, capped. `sec_get` / `download` (requests) also retry connection errors and timeouts and are
spaced by the shared per-process limiter (<= ~9 req/s; run one EDGAR walk at a time). edgartools
calls are not retried on network errors, which edgartools already retries; such an error raises
`TransientReadError` at once. Every exhausted retry raises `TransientReadError`, chained to the
cause. edgartools' full-index `get_filings` (13F `.gz` listing) stays outside: edgartools retries it.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property, partial
from pathlib import Path
from typing import Any

import edgar
import httpx
import requests
from edgar.httprequests import is_unreachable
from edgar.sgml.sgml_parser import SECHTMLResponseError

from src.context import Context
from src.data_extract.utils.common.rate_limit import is_rate_limited

logger = logging.getLogger(__name__)

_MIN_INTERVAL = 0.11  # ~9 req/sec, safely under SEC's 10/sec limit
_DEFAULT_TIMEOUT = 30  # seconds; avoid a hung socket stalling a worker
_CHUNK = 1 << 20  # 1 MiB streaming chunks
_RETRYABLE_STATUS = frozenset({429, 502, 503, 504})
#: A request that never got a complete answer (requests, httpx, or a socket error edgartools let through).
_NETWORK_ERRORS = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
    httpx.TransportError,
    ConnectionError,
    TimeoutError,
)
_CAUSE_DEPTH = 8

_rate_lock = threading.Lock()
_next_slot = [0.0]  # monotonic time of the next allowed request start


class TransientReadError(RuntimeError):
    """An SEC read that still failed transiently (throttle, 5xx, network) after the retry policy."""


class ParseFailureError(RuntimeError):
    """A filing that was read but could not be parsed; deterministic, so retrying cannot help."""


class _StatusError(RuntimeError):
    """A retryable HTTP status seen on a response the caller streams itself."""

    def __init__(self, url: str, status_code: int, retry_after: Any) -> None:
        super().__init__(f"HTTP {status_code} from {url}")
        self.status_code = status_code
        self.retry_after = retry_after


class _EmptyHeaderError(RuntimeError):
    """edgartools' header fallback for an unreadable submission (an SEC error page); retried like a 503."""

    status_code = 503


@dataclass(frozen=True)
class RetryPolicy:
    """`attempts` calls in total; wait `waits[i]` before retry i+1 (last wait repeats), at least a
    Retry-After of up to `retry_after_cap` seconds. Defaults mirror `data_extract.sec_retry`."""

    attempts: int = 3
    waits: tuple[float, ...] = (5.0, 20.0)
    retry_after_cap: float = 120.0

    @classmethod
    def from_config(cls, config: Any) -> RetryPolicy:
        """The policy in `config.data_extract.sec_retry`; the defaults when there is no such section."""
        try:
            section = config.data_extract.sec_retry
        except (AttributeError, KeyError):
            return cls()
        if section is None:
            return cls()
        return cls(attempts=int(section.attempts), waits=tuple(float(w) for w in section.waits), retry_after_cap=float(section.retry_after_cap))

    def wait(self, retry_index: int, retry_after: float | None) -> float:
        """Seconds to wait before retry `retry_index` (0-based)."""
        base = self.waits[min(retry_index, len(self.waits) - 1)] if self.waits else 0.0
        return base if retry_after is None else max(base, min(retry_after, self.retry_after_cap))


@dataclass
class _State:
    """The process-wide policy, whether it was loaded from config, and the sleeper (tests inject one)."""

    policy: RetryPolicy = field(default_factory=RetryPolicy)
    configured: bool = False
    sleep: Callable[[float], None] = time.sleep


_STATE = _State()


class _Verdict(Enum):
    RETRY = "retry"
    FAIL = "fail"


def configure(context: Context) -> None:
    """Load the process-wide retry policy from `context.config` once; later calls are no-ops."""
    if _STATE.configured:
        return
    _STATE.policy = RetryPolicy.from_config(getattr(context, "config", None))
    _STATE.configured = True


def _causes(exc: BaseException) -> Iterator[BaseException]:
    """`exc` and its explicit `__cause__` chain (implicit `__context__` is not followed)."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(seen) < _CAUSE_DEPTH:
        seen.add(id(current))
        yield current
        current = current.__cause__


def _status_code(exc: BaseException) -> int | None:
    """The first HTTP status carried by `exc` or its causes (`status_code` or `response.status_code`)."""
    for err in _causes(exc):
        status = getattr(err, "status_code", None)
        if not isinstance(status, int):
            status = getattr(getattr(err, "response", None), "status_code", None)
        if isinstance(status, int):
            return status
    return None


def _seconds(value: Any) -> float | None:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


def _retry_after(exc: BaseException) -> float | None:
    """Retry-After seconds from `exc.retry_after` or a response header, searched along the causes."""
    for err in _causes(exc):
        value = getattr(err, "retry_after", None)
        if value is None:
            headers = getattr(getattr(err, "response", None), "headers", None)
            value = headers.get("Retry-After") if headers is not None else None
        if (seconds := _seconds(value)) is not None:
            return seconds
    return None


def _is_network(exc: BaseException) -> bool:
    """A request that never got an answer; an SSL failure is deterministic and does not count."""
    return any(
        is_unreachable(err) or (isinstance(err, _NETWORK_ERRORS) and not isinstance(err, requests.exceptions.SSLError)) for err in _causes(exc)
    )


def _verdict(exc: BaseException, *, network_retry: bool) -> _Verdict | None:
    """RETRY a throttle/5xx, an SEC HTML page served in place of a filing (its throttle page; edgartools
    raises it status-less), and a network error when `network_retry`; FAIL an edgartools network
    error at once, None (re-raise as is) for everything else, including this module's own errors."""
    if isinstance(exc, TransientReadError | ParseFailureError):
        return None
    if any(isinstance(err, SECHTMLResponseError) for err in _causes(exc)):
        return _Verdict.RETRY
    status = _status_code(exc)
    if status is not None:
        return _Verdict.RETRY if status in _RETRYABLE_STATUS else None
    if _is_network(exc):
        return _Verdict.RETRY if network_retry else _Verdict.FAIL
    return _Verdict.RETRY if is_rate_limited(exc) else None


def _with_retry[T](call: Callable[[], T], *, label: str, network_retry: bool) -> T:
    """`call()` under the process policy; raises `TransientReadError` once the policy is spent."""
    policy = _STATE.policy
    attempt = 1
    while True:
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 -- classified below; anything unclassified re-raises unchanged
            verdict = _verdict(exc, network_retry=network_retry)
            if verdict is None:
                raise
            if verdict is _Verdict.FAIL or attempt >= policy.attempts:
                raise TransientReadError(f"{label}: SEC read failed after {attempt} attempt(s): {type(exc).__name__}: {exc}") from exc
            wait = policy.wait(attempt - 1, _retry_after(exc))
            logger.info("%s: transient SEC failure (%s); attempt %d/%d -> retry in %.0fs", label, type(exc).__name__, attempt, policy.attempts, wait)
            _STATE.sleep(wait)
            attempt += 1


def sec_call[T](fn: Callable[..., T], *args: Any, label: str, **kwargs: Any) -> T:
    """`fn(*args, **kwargs)` (an edgartools call) under the retry policy; `label` keys the log."""
    return _with_retry(partial(fn, *args, **kwargs), label=label, network_retry=False)


def _reserve_slot() -> None:
    """Reserve the next evenly-spaced request slot; the wait happens outside the lock."""
    with _rate_lock:
        start = max(time.monotonic(), _next_slot[0])
        _next_slot[0] = start + _MIN_INTERVAL
    delay = start - time.monotonic()
    if delay > 0:
        time.sleep(delay)


def _raise_retryable(response: Any, url: str) -> None:
    """Raise `_StatusError` when the response carries a retryable status."""
    status = int(response.status_code)
    if status in _RETRYABLE_STATUS:
        headers = getattr(response, "headers", None) or {}
        raise _StatusError(url, status, headers.get("Retry-After"))


def _get_once(context: Context, url: str, kwargs: dict[str, Any]) -> requests.Response:
    _reserve_slot()
    response = context.sec_session.get(url, **kwargs)
    _raise_retryable(response, url)
    response.raise_for_status()
    return response


def sec_get(context: Context, url: str, **kwargs: Any) -> requests.Response:
    """Rate-limited, retried GET on `context.sec_session` (User-Agent pre-set); raises on HTTP error."""
    configure(context)
    kwargs.setdefault("timeout", _DEFAULT_TIMEOUT)
    return _with_retry(partial(_get_once, context, url, kwargs), label=url, network_retry=True)


def _stream_to(response: Any, path: Path) -> None:
    """Write the body to `path` through a `.part` file; a failure mid-stream deletes the `.part`."""
    tmp = path.with_suffix(".part")
    try:
        with open(tmp, "wb") as handle:
            handle.writelines(response.iter_content(chunk_size=_CHUNK))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(path)


def _download_once(context: Context, url: str, path: Path, timeout: int) -> int:
    _reserve_slot()
    response = context.sec_session.get(url, timeout=timeout, stream=True)
    try:
        _raise_retryable(response, url)
        status = int(response.status_code)
        if status == 200:
            _stream_to(response, path)
        return status
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


def download(context: Context, url: str, path: Path, *, timeout: int = 300) -> int:
    """Stream `url` to `path` under the retry policy; returns the HTTP status (200 = written).

    A non-retryable status (e.g. 404, an archive not published yet) returns without writing."""
    configure(context)
    return _with_retry(partial(_download_once, context, url, path, timeout), label=url, network_retry=True)


def _label(filing: Any, what: str) -> str:
    return f"{getattr(filing, 'accession_number', '?')} {what}"


def filing_obj(filing: Any) -> Any:
    """`filing.obj()`; a transient read raises `TransientReadError`, any other failure `ParseFailureError`."""
    try:
        return sec_call(filing.obj, label=_label(filing, "obj"))
    except TransientReadError:
        raise
    except Exception as exc:  # noqa: BLE001 -- the filer's document, parsed by edgartools
        raise ParseFailureError(f"{_label(filing, 'obj')}: {type(exc).__name__}: {exc}") from exc


def filing_text(filing: Any) -> str | None:
    """`filing.text()` under the retry policy."""
    return sec_call(filing.text, label=_label(filing, "text"))


def filing_html(filing: Any) -> str | None:
    """`filing.html()` under the retry policy."""
    return sec_call(filing.html, label=_label(filing, "html"))


def filing_xml(filing: Any) -> str | None:
    """`filing.xml()` under the retry policy."""
    return sec_call(filing.xml, label=_label(filing, "xml"))


def filing_xbrl(filing: Any) -> Any:
    """`filing.xbrl()` under the retry policy; None when the filing has no XBRL."""
    return sec_call(filing.xbrl, label=_label(filing, "xbrl"))


def filing_attachments(filing: Any) -> Any:
    """`filing.attachments` (None when the object has none) under the retry policy."""
    return sec_call(getattr, filing, "attachments", None, label=_label(filing, "attachments"))


def _is_empty_header(header: Any) -> bool:
    """edgartools' fallback header for a submission it could not read: no text and no party."""
    if header is None:
        return True
    parties = (getattr(header, name, None) for name in ("filers", "subject_companies", "reporting_owners", "issuer"))
    return getattr(header, "text", None) == "" and not any(parties)


def forget_sgml(filing: Any) -> None:
    """Drop edgartools' cached submission and header: the next read fetches them again, and a held Filing stops holding every document.

    The submission's documents, attachments and summary point back at it, so it is emptied first:
    dropping the reference alone leaves every document to the cyclic collector, which runs rarely.
    """
    if isinstance(getattr(type(filing), "header", None), cached_property):
        vars(filing).pop("header", None)
    if (sgml := getattr(filing, "_sgml", None)) is not None:
        _empty(sgml)
    if hasattr(filing, "_sgml"):
        filing._sgml = None


def _empty(obj: Any) -> None:
    """Delete every instance attribute of `obj`, `__slots__` included, breaking the cycles through it."""
    if hasattr(obj, "__dict__"):
        vars(obj).clear()
    for klass in type(obj).__mro__:
        for slot in getattr(klass, "__slots__", ()):
            if slot not in {"__dict__", "__weakref__"} and hasattr(obj, slot):
                delattr(obj, slot)


def _read_header(filing: Any) -> Any:
    header = filing.header
    if _is_empty_header(header):
        forget_sgml(filing)
        raise _EmptyHeaderError(f"{_label(filing, 'header')}: empty SGML header")
    return header


def filing_header(filing: Any) -> Any:
    """The SGML header; an empty one (edgartools' fallback for an SEC error page) is retried and,
    when it persists, raises `TransientReadError`."""
    return sec_call(_read_header, filing, label=_label(filing, "header"))


def company(cik_or_ticker: str | int) -> Any:
    """`edgar.Company(cik_or_ticker)` under the retry policy."""
    return sec_call(edgar.Company, cik_or_ticker, label=f"company {cik_or_ticker}")


def company_filings(entity: Any, forms: Sequence[str]) -> Any:
    """`entity.get_filings(form=forms)` (every submissions page) under the retry policy."""
    return sec_call(entity.get_filings, form=forms, label=f"filings {getattr(entity, 'cik', '?')}")
