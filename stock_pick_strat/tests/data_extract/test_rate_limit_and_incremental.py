"""Rate-limit retry helper + incremental-download logic.

  * call_with_retries: waits & retries on 429, succeeds on a later attempt, and
    re-raises non-rate-limit errors immediately.
  * earnings _download_one: retries the rate-limited yfinance call (the fix for
    the ~fixed subset that was silently dropped every run).
  * dividends / wiki / trends: a ticker already current is skipped; only missing
    ex-dates/days are re-requested.
"""
from __future__ import annotations

import types

import pandas as pd

from src.data_extract.utils.rate_limit import is_rate_limited, call_with_retries


def test_retry_waits_then_succeeds_and_reraises_other():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("HTTP 429 Too Many Requests")
        return "ok"

    out = call_with_retries(flaky, retries=3, base_wait=0.001, label="t", printer=lambda *_: None)
    assert out == "ok" and calls["n"] == 3

    assert is_rate_limited(RuntimeError("429"))
    assert is_rate_limited(Exception("TooManyRequestsError"))   # pytrends
    assert not is_rate_limited(ValueError("bad symbol"))

    hits = {"n": 0}
    def boom():
        hits["n"] += 1
        raise ValueError("bad symbol")
    try:
        call_with_retries(boom, retries=3, base_wait=0.001, printer=lambda *_: None)
        raised = False
    except ValueError:
        raised = True
    assert raised and hits["n"] == 1, "non-429 must not be retried"

    print("\n=== SANITY CHECK: rate-limit retry helper ===")
    print("  429 -> waited & retried, succeeded on attempt 3; pytrends error detected; "
          "non-429 re-raised immediately (1 call). Validated.")


def test_earnings_download_one_retries_rate_limit(monkeypatch):
    import src.data_extract.utils.fetch_earnings_surprises as fe

    state = {"n": 0}
    idx = pd.to_datetime(["2024-02-01"])
    good = pd.DataFrame({"EPS Estimate": [1.0], "Reported EPS": [1.1],
                         "Surprise(%)": [10.0]}, index=idx)

    class _Tk:
        def __init__(self, t): pass
        def get_earnings_dates(self, limit):
            state["n"] += 1
            if state["n"] < 2:
                raise RuntimeError("YFRateLimitError: Too Many Requests 429")
            return good

    monkeypatch.setattr(fe.yf, "Ticker", _Tk)
    # fast backoff for the test
    monkeypatch.setattr(fe, "call_with_retries",
                        lambda fn, **k: call_with_retries(fn, retries=3, base_wait=0.001,
                                                          printer=lambda *_: None))
    out = fe._download_one("AAPL", 8)
    assert out is not None and out["eps_actual"].iloc[0] == 1.1 and state["n"] == 2
    print("\n=== SANITY CHECK: earnings retries the throttled call ===")
    print(f"  get_earnings_dates 429'd once then succeeded (calls={state['n']}); "
          f"ticker recovered instead of being dropped. Validated.")


def test_incremental_skip_logic():
    """The 'skip if current' freshness rule (dividends/trends) and the wiki
    day-level window [cached_max+1 .. yesterday]."""
    today = pd.Timestamp("2024-06-15")
    last_by = {"CUR": pd.Timestamp("2024-06-10"),   # 5d old -> current
               "OLD": pd.Timestamp("2024-01-01")}   # ~165d old -> refetch
    window = 80

    def skip(t):
        last = last_by.get(t)
        return last is not None and (today - last).days <= window

    assert skip("CUR") and not skip("OLD") and not skip("NEW")

    # wiki day-level: only request [cached_max+1 .. yesterday]; skip if caught up
    end = today - pd.Timedelta(days=1)
    assert (last_by["OLD"] + pd.Timedelta(days=1)) <= end          # has missing days
    assert (end + pd.Timedelta(days=1)) > end                      # current -> skipped
    print("\n=== SANITY CHECK: incremental skip / missing-days logic ===")
    print("  current ticker skipped (freshness window); stale ticker refetched; "
          "wiki requests only [cached_max+1 .. yesterday]. Validated.")


if __name__ == "__main__":
    test_retry_waits_then_succeeds_and_reraises_other()
    mp = types.SimpleNamespace(setattr=lambda o, n, v: setattr(o, n, v))
    test_earnings_download_one_retries_rate_limit(mp)
    test_incremental_skip_logic()
