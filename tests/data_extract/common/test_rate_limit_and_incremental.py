"""Rate-limit retry helper and the rate-limit matcher.

* call_with_retries: waits & retries on 429, succeeds on a later attempt, and
  re-raises non-rate-limit errors immediately.
* earnings _download_one: retries the rate-limited yfinance call (the fix for
  the ~fixed subset that was silently dropped every run).
"""

from __future__ import annotations

import types

import pandas as pd
import requests

from src.data_extract.utils.common.rate_limit import call_with_retries, is_rate_limited


def test_retry_waits_then_succeeds_and_reraises_other():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("HTTP 429 Too Many Requests")
        return "ok"

    out = call_with_retries(flaky, retries=3, base_wait=0.001, label="t")
    assert out == "ok" and calls["n"] == 3

    assert is_rate_limited(RuntimeError("429"))
    assert is_rate_limited(Exception("TooManyRequestsError"))  # throttle exception class name
    assert is_rate_limited(Exception("ReadTimeout: The read operation timed out"))
    assert not is_rate_limited(ValueError("bad symbol"))

    hits = {"n": 0}

    def boom():
        hits["n"] += 1
        raise ValueError("bad symbol")

    try:
        call_with_retries(boom, retries=3, base_wait=0.001)
        raised = False
    except ValueError:
        raised = True
    assert raised and hits["n"] == 1, "non-429 must not be retried"

    print("\n=== SANITY CHECK: rate-limit retry helper ===")
    print(
        "  429 -> waited & retried, succeeded on attempt 3; throttle-class and transport "
        "timeout errors detected; non-transient error re-raised immediately (1 call). "
        "Validated."
    )


def _http_error(status: int, url: str) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    response.reason = "Not Found" if status == 404 else "Service Unavailable"
    response.url = url
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        return exc
    raise AssertionError(f"status {status} did not raise")


def test_is_rate_limited_reads_the_status_code_not_digits_inside_identifiers():
    not_found = _http_error(404, "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000050302&type=4")
    unavailable = _http_error(503, "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193")
    timeout = requests.exceptions.ReadTimeout("HTTPSConnectionPool: Read timed out.")
    accession = ValueError("no XML in 0000950103-24-000001 primary document")

    assert not is_rate_limited(not_found), str(not_found)
    assert is_rate_limited(unavailable)
    assert is_rate_limited(timeout)
    assert not is_rate_limited(accession)

    print("\n=== SANITY CHECK: rate-limit matcher ===")
    print(f"  404 on CIK=0000050302 -> {is_rate_limited(not_found)}; 503 -> {is_rate_limited(unavailable)}; ")
    print(f"  ReadTimeout -> {is_rate_limited(timeout)}; accession 0000950103-24-000001 -> {is_rate_limited(accession)}.")
    print("  -> Only a real throttle/5xx/timeout is retried; a 404 whose URL contains '503' fails at once.")


def test_earnings_download_one_retries_rate_limit(monkeypatch):
    import src.data_extract.utils.fundamentals.fetch_earnings_surprises as fe

    state = {"n": 0}
    idx = pd.to_datetime(["2024-02-01"])
    good = pd.DataFrame({"EPS Estimate": [1.0], "Reported EPS": [1.1], "Surprise(%)": [10.0]}, index=idx)

    class _Tk:
        def __init__(self, t):
            pass

        def get_earnings_dates(self, limit):
            state["n"] += 1
            if state["n"] < 2:
                raise RuntimeError("YFRateLimitError: Too Many Requests 429")
            return good

    monkeypatch.setattr(fe.yf, "Ticker", _Tk)
    # fast backoff for the test
    monkeypatch.setattr(fe, "call_with_retries", lambda fn, **k: call_with_retries(fn, retries=3, base_wait=0.001))
    out = fe._download_one("AAPL", 8)
    assert out is not None and out["eps_actual"].iloc[0] == 1.1 and state["n"] == 2
    print("\n=== SANITY CHECK: earnings retries the throttled call ===")
    print(f"  get_earnings_dates 429'd once then succeeded (calls={state['n']}); ticker recovered instead of being dropped. Validated.")


if __name__ == "__main__":
    test_retry_waits_then_succeeds_and_reraises_other()
    mp = types.SimpleNamespace(setattr=lambda o, n, v: setattr(o, n, v))
    test_earnings_download_one_retries_rate_limit(mp)
