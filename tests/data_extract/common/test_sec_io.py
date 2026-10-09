"""`sec_io`: the one retry policy for every SEC request (AC-011 checks 1-3).

Check 1 scans `src/data_extract` for SEC calls made outside `sec_io`. Check 2 serves 503, 503, 200
and expects one success after growing waits, Retry-After honoured. Check 3 serves 503 forever and
expects exactly 3 attempts, then `TransientReadError`. Offline: fake sessions and fake callables,
the sleeper and the rate limiter are injected, nothing waits for real.
"""

from __future__ import annotations

import gc
import io
import re
import tokenize
import weakref
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import edgar
import httpx
import pytest
import requests
from edgar.exceptions import TooManyRequestsError
from edgar.sgml.sgml_header import FilingHeader
from edgar.sgml.sgml_parser import SECHTMLResponseError
from omegaconf import OmegaConf

from src.data_extract.utils.common import edgar_fillings as ef
from src.data_extract.utils.common import sec_io
from src.data_extract.utils.common.sec_io import RetryPolicy, TransientReadError

_REPO = Path(__file__).resolve().parents[3]
_URL = "https://www.sec.gov/Archives/edgar/data/1/000000000000000001/doc.htm"

#: AC-011 check 1: SEC calls that must go through `sec_io`.
_SEC_CALL = re.compile(r"\.obj\(|\.xml\(|\.xbrl\(|\.html\(|\.header\b|\.attachments\b|\bCompany\(|get_filings\(|sec_session\.|retry=False")
#: The one sanctioned exception: 13F's full-index `.gz` listing, which edgartools retries itself.
_ALLOWED = {("fetch_13f.py", "get_filings(")}
_NOT_CODE = {tokenize.COMMENT, tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", tokenize.STRING)}


@pytest.fixture
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The default policy with a recording sleeper and no rate-limit spacing."""
    recorded: list[float] = []
    monkeypatch.setattr(sec_io, "_STATE", sec_io._State(policy=RetryPolicy(), configured=True, sleep=recorded.append))
    monkeypatch.setattr(sec_io, "_reserve_slot", lambda: None)
    return recorded


def _response(status: int, body: bytes = b"ok", retry_after: str | None = None) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = body
    response._content_consumed = True
    response.url = _URL
    if retry_after is not None:
        response.headers["Retry-After"] = retry_after
    return response


def _context(responses: Iterator[requests.Response], calls: list[str]) -> Any:
    def get(url: str, **kwargs: Any) -> requests.Response:
        calls.append(url)
        return next(responses)

    return SimpleNamespace(sec_session=SimpleNamespace(get=get), config=SimpleNamespace())


def _scripted(outcomes: list[Any]) -> tuple[Callable[[], Any], list[int]]:
    """A callable that raises or returns `outcomes` in order, and its call counter."""
    calls: list[int] = []

    def call() -> Any:
        calls.append(1)
        outcome = outcomes[min(len(calls), len(outcomes)) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return call, calls


def _http_status(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", _URL)
    return httpx.HTTPStatusError(f"{status}", request=request, response=httpx.Response(status, request=request))


def test_sec_get_retries_503_then_succeeds_with_growing_waits(waits: list[float]) -> None:
    calls: list[str] = []
    context = _context(iter([_response(503, retry_after="7"), _response(503), _response(200, b"page")]), calls)

    response = sec_io.sec_get(context, _URL)

    assert response.status_code == 200 and response.content == b"page"
    assert len(calls) == 3
    assert waits == [7.0, 20.0]  # max(5, Retry-After 7), then the second policy wait
    print("\n=== SANITY CHECK: AC-011 check 2 (503, 503, 200) ===")
    print(f"  3 GETs, one success; waits {waits}: Retry-After 7 s lengthened the first 5 s wait, the second grew to 20 s.")


def test_sec_get_gives_up_after_three_attempts(waits: list[float]) -> None:
    calls: list[str] = []
    context = _context(iter(_response(503) for _ in range(10)), calls)

    with pytest.raises(TransientReadError) as raised:
        sec_io.sec_get(context, _URL)

    assert len(calls) == 3
    assert waits == [5.0, 20.0]
    assert getattr(raised.value.__cause__, "status_code", None) == 503
    print("\n=== SANITY CHECK: AC-011 check 3 (503 forever) ===")
    print(f"  exactly {len(calls)} GETs, waits {waits}, then TransientReadError chained to the last 503.")


def test_sec_get_does_not_retry_a_404(waits: list[float]) -> None:
    calls: list[str] = []
    context = _context(iter([_response(404)]), calls)

    with pytest.raises(requests.HTTPError):
        sec_io.sec_get(context, _URL)

    assert len(calls) == 1 and waits == []
    print("\n=== SANITY CHECK: a 404 is deterministic ===")
    print("  one GET, no wait, the HTTPError surfaces unchanged.")


def test_edgartools_statuses_retry_and_retry_after_is_capped(waits: list[float]) -> None:
    call, calls = _scripted([_http_status(503), TooManyRequestsError(_URL, retry_after=30), "parsed"])
    assert sec_io.sec_call(call, label="obj") == "parsed"
    assert len(calls) == 3 and waits == [5.0, 30.0]

    waits.clear()
    call, calls = _scripted([TooManyRequestsError(_URL, retry_after=600), "parsed"])
    assert sec_io.sec_call(call, label="obj") == "parsed"
    assert waits == [120.0]
    print("\n=== SANITY CHECK: edgartools statuses ===")
    print("  httpx 503 and TooManyRequestsError are retried; Retry-After 30 s is honoured, 600 s is capped at 120 s.")


def test_edgartools_network_errors_are_not_retried(waits: list[float]) -> None:
    call, calls = _scripted([httpx.ConnectError("refused"), "parsed"])

    with pytest.raises(TransientReadError):
        sec_io.sec_call(call, label="obj")

    assert len(calls) == 1 and waits == []
    print("\n=== SANITY CHECK: no multiplied retry layers ===")
    print("  an edgartools network error (already retried by edgartools) raises TransientReadError after 1 call.")


def test_sec_html_page_in_place_of_a_filing_is_retried_with_waits(waits: list[float]) -> None:
    """The log's `SECHTMLResponseError ... failed after 1 attempt(s)`: edgartools raises it status-less."""
    html = SECHTMLResponseError("SEC returned HTML or XML content instead of expected SGML filing data.")
    call, calls = _scripted([html, html, "parsed"])
    assert sec_io.sec_call(call, label="period_of_report") == "parsed"
    assert len(calls) == 3 and waits == [5.0, 20.0]

    waits.clear()
    call, calls = _scripted([html])
    with pytest.raises(TransientReadError):
        sec_io.sec_call(call, label="period_of_report")
    assert len(calls) == 3 and waits == [5.0, 20.0]
    print("\n=== SANITY CHECK: SEC HTML page = throttle ===")
    print("  SECHTMLResponseError retries (5 s, 20 s) and succeeds on call 3; a persistent one raises TransientReadError after 3 calls, not 1.")


def test_deterministic_errors_pass_through(waits: list[float]) -> None:
    call, calls = _scripted([ValueError("bad xml"), "parsed"])
    with pytest.raises(ValueError, match="bad xml"):
        sec_io.sec_call(call, label="obj")

    obj_failure, _ = _scripted([KeyError("Item 9.99")])
    with pytest.raises(sec_io.ParseFailureError):
        sec_io.filing_obj(SimpleNamespace(accession_number="0001", obj=obj_failure))

    assert len(calls) == 1 and waits == []
    print("\n=== SANITY CHECK: deterministic failures ===")
    print("  a parse error is raised unchanged by sec_call and as ParseFailureError by filing_obj, never retried.")


def test_filing_header_retries_an_empty_header_then_raises(waits: list[float], monkeypatch: pytest.MonkeyPatch) -> None:
    """edgartools caches `header` and the submission; an error page yields a header with no text and no party."""
    empty = FilingHeader(text="", filing_metadata={"ACCESSION NUMBER": "0001104659-24-000001"})
    real = FilingHeader(text="<SEC-HEADER>", filing_metadata={}, filers=cast(Any, [SimpleNamespace(company_information=None)]))
    served: list[FilingHeader] = []

    def install(headers: list[FilingHeader]) -> None:
        queue = iter(headers)

        def sgml(self: Any) -> Any:
            if self._sgml is None:
                served.append(header := next(queue))
                self._sgml = SimpleNamespace(header=header)
            return self._sgml

        monkeypatch.setattr(edgar.Filing, "sgml", sgml)

    def filing() -> edgar.Filing:
        return edgar.Filing(cik=1, company="X", form="SC 13G", filing_date="2024-01-02", accession_no="0001104659-24-000001")

    install([empty, empty, real])
    assert sec_io.filing_header(filing()) is real
    assert served == [empty, empty, real] and waits == [5.0, 20.0]

    waits.clear()
    install([empty] * 5)
    with pytest.raises(TransientReadError):
        sec_io.filing_header(filing())
    assert waits == [5.0, 20.0]
    print("\n=== SANITY CHECK: 503 HTML header fixture ===")
    print("  edgartools' empty fallback header is re-fetched (cache dropped) and, when it persists, raises TransientReadError.")


class _Document:
    def __init__(self, submission: object) -> None:
        self.submission = submission  # edgartools' documents and attachments point back at their submission
        self.content = "x" * 1_000_000


class _Submission:
    __slots__ = ("header", "_documents_by_sequence", "__dict__", "__weakref__")

    def __init__(self) -> None:
        self.header = SimpleNamespace(text="<SEC-HEADER>")
        self._documents_by_sequence = [_Document(self)]
        self.attachments = [_Document(self)]


def test_forget_sgml_frees_the_submission_without_the_cyclic_collector() -> None:
    """A dropped submission sits in reference cycles with its documents; forgetting it must free it by refcount,
    because the cyclic collector rarely reaches it in a long walk."""
    filing = edgar.Filing(cik=1, company="X", form="10-K", filing_date="2024-01-02", accession_no="0001104659-24-000001")
    filing._sgml = _Submission()
    held = [weakref.ref(filing._sgml), weakref.ref(filing._sgml._documents_by_sequence[0]), weakref.ref(filing._sgml.attachments[0])]
    gc.disable()
    try:
        sec_io.forget_sgml(filing)
        alive = [ref() is not None for ref in held]
    finally:
        gc.enable()
    assert filing._sgml is None and alive == [False, False, False], alive
    print("\n=== SANITY CHECK: forget_sgml ===")
    print("  with the cyclic collector off, the submission, its documents and its attachments are freed once forgotten.")


def test_download_retries_a_broken_stream_and_never_leaves_a_part_file(waits: list[float], tmp_path: Path) -> None:
    class _Broken:
        status_code = 200
        headers: dict[str, str] = {}

        def iter_content(self, chunk_size: int) -> Iterator[bytes]:
            yield b"partial"
            raise requests.exceptions.ChunkedEncodingError("connection reset")

    good = _response(200, b"archive")
    calls: list[str] = []
    context = _context(iter([cast(Any, _Broken()), good]), calls)
    path = tmp_path / "2024q1.zip"

    assert sec_io.download(context, _URL, path) == 200
    assert path.read_bytes() == b"archive" and not path.with_suffix(".part").exists()
    assert len(calls) == 2 and waits == [5.0]

    missing = tmp_path / "2026q4.zip"
    assert sec_io.download(_context(iter([_response(404)]), calls), _URL, missing) == 404
    assert not missing.exists()
    print("\n=== SANITY CHECK: download ===")
    print("  a stream broken mid-way deletes its .part and is retried; a 404 returns 404 with nothing written.")


def test_list_filings_raises_when_an_older_page_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    recent = {"name": "X", "filings": {"recent": {"accessionNumber": []}, "files": [{"name": "CIK0000000001-submissions-001.json"}]}}

    def get(context: Any, url: str, **kwargs: Any) -> Any:
        if url.endswith("-001.json"):
            raise TransientReadError(f"{url}: SEC read failed after 3 attempt(s)")
        return SimpleNamespace(json=lambda: recent)

    monkeypatch.setattr(ef, "sec_get", get)

    with pytest.raises(TransientReadError):
        ef.list_filings(cast(Any, None), "1", ["DEF 14A"], years=40)
    print("\n=== SANITY CHECK: no silently short listing ===")
    print("  an older submissions page that cannot be read raises instead of being skipped.")


def test_policy_defaults_match_the_config() -> None:
    config = OmegaConf.load(_REPO / "configs" / "configs.yml")
    assert RetryPolicy.from_config(config) == RetryPolicy(attempts=3, waits=(5.0, 20.0), retry_after_cap=120.0)
    assert RetryPolicy.from_config(SimpleNamespace()) == RetryPolicy()
    print("\n=== SANITY CHECK: configs.yml data_extract.sec_retry ===")
    print("  3 attempts, waits 5 s then 20 s, Retry-After cap 120 s; a config without the section uses the same defaults.")


def _code_only(source: str) -> list[str]:
    """`source` lines with comments and string literals blanked out."""
    rows = [list(line) for line in source.splitlines()]
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in _NOT_CODE:
            continue
        (start_row, start_col), (end_row, end_col) = token.start, token.end
        for row in range(start_row, end_row + 1):
            line = rows[row - 1]
            for col in range(start_col if row == start_row else 0, end_col if row == end_row else len(line)):
                line[col] = " "
    return ["".join(line) for line in rows]


def test_no_sec_call_bypasses_sec_io() -> None:
    hits: list[str] = []
    for path in sorted((_REPO / "src" / "data_extract").rglob("*.py")):
        if path.name == "sec_io.py":
            continue
        for number, line in enumerate(_code_only(path.read_text(encoding="utf-8")), start=1):
            hits += [f"{path.name}:{number}: {m.group()}" for m in _SEC_CALL.finditer(line) if (path.name, m.group()) not in _ALLOWED]

    assert hits == []
    allowed = [f"{name}: {pattern}" for name, pattern in sorted(_ALLOWED)]
    print("\n=== SANITY CHECK: AC-011 check 1 (source scan) ===")
    print(f"  no SEC call outside sec_io in src/data_extract; allowlist {allowed} (edgartools retries the .gz itself).")
