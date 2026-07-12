"""Tests for the shared FMP client (src/data_extract/fmp_client.py):
key discovery, rate-limit detection, multi-key rotation, and the incremental
(pull-date based) fetch plan.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.data_extract.fmp_client import (
    collect_api_keys, is_rate_limited, plan_fetch, run_rotating_fetch,
    FMPRateLimitError,
)


class _FakeResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else []

    def json(self):
        return self._payload


def test_collect_api_keys(monkeypatch):
    import os
    for n in list(os.environ):
        if n.startswith("FMP_API_KEY"):
            monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv("FMP_API_KEY_alar", "k_alar")
    monkeypatch.setenv("FMP_API_KEY_valueartech", "k_va")
    monkeypatch.setenv("FMP_API_KEY_gardon", "k_gardon")
    monkeypatch.setenv("FMP_API_KEY_blank", "   ")   # empty -> ignored

    keys = collect_api_keys()
    names = [n for n, _ in keys]
    # NB: Windows upper-cases env var names -> compare case-insensitively.
    assert [n.upper() for n in names] == [
        "FMP_API_KEY_ALAR", "FMP_API_KEY_GARDON", "FMP_API_KEY_VALUEARTECH"]
    assert [k for _, k in keys] == ["k_alar", "k_gardon", "k_va"]   # blank dropped

    print("\n=== SANITY CHECK: FMP key discovery ===")
    print(f"  discovered (deterministic order, blank dropped): {names}. Validated.")


def test_is_rate_limited():
    assert is_rate_limited(_FakeResp(status_code=429))
    assert is_rate_limited(_FakeResp(status_code=403))
    assert is_rate_limited(_FakeResp(status_code=402))   # FMP daily-limit / plan limit
    assert is_rate_limited(_FakeResp(payload={"Error Message": "Limit Reach. Upgrade your plan"}))
    assert not is_rate_limited(_FakeResp(payload=[{"x": 1}]))
    assert not is_rate_limited(_FakeResp(status_code=200, payload=[]))

    print("\n=== SANITY CHECK: rate-limit detection ===")
    print("  402/403/429 + 'Limit Reach' -> True; normal 200 list -> False. Validated.")


def test_plan_fetch_uses_pull_date():
    today = pd.Timestamp.today().normalize()
    existing = pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        # filing dates old for both -> staleness must key off fetched_at, else an
        # annual filer would be re-pulled every run and waste quota.
        "as_of": [today - pd.Timedelta(days=300), today - pd.Timedelta(days=300)],
        "fetched_at": [today - pd.Timedelta(days=5),     # pulled recently -> skip
                       today - pd.Timedelta(days=60)],    # stale pull -> refetch
    })
    plan = plan_fetch(["AAA", "BBB", "CCC"], existing, refetch_window_days=30)
    assert "AAA" not in plan and "BBB" in plan and "CCC" in plan
    assert set(plan_fetch(["AAA", "BBB"], None, 30)) == {"AAA", "BBB"}   # cold start

    print("\n=== SANITY CHECK: incremental plan (pull-date based) ===")
    print(f"  AAA filed 300d ago but pulled 5d ago -> skipped; stale BBB + new CCC -> {sorted(plan)}.")
    print("  Validated.")


def test_rotation_settles_on_first_live_key_and_stops_rolling():
    """User's scenario: key #1 is full, keys #2-#4 are fine. Must roll off #1
    onto #2 and then STAY there -- #3 and #4 are never touched."""
    used = []

    def download_one(ticker, key):
        used.append(key)
        if key == "k1":
            raise FMPRateLimitError()          # only the first key is exhausted
        return pd.DataFrame([{"ticker": ticker, "as_of": pd.Timestamp("2024-01-01")}])

    tickers = ["AAA", "BBB", "CCC", "DDD"]
    frames, exhausted = run_rotating_fetch(
        tickers, ["k1", "k2", "k3", "k4"], logging.getLogger("test"),
        download_one, pause=0.0)

    assert not exhausted and len(frames) == len(tickers)
    # k1 is hit exactly once (first ticker), retired, then k2 serves everything.
    assert used.count("k1") == 1
    assert used.count("k2") == len(tickers)
    assert "k3" not in used and "k4" not in used

    print("\n=== SANITY CHECK: conservative rotation (key #1 full, #3/#4 fresh) ===")
    print(f"  key sequence: {used}")
    print("  -> k1 tried once & retired, settled on k2, k3/k4 NEVER touched. Validated.")


def test_premium_ticker_is_skipped_without_killing_keys_or_aborting():
    """A ticker that rate-limits on EVERY key (premium/unavailable) must be
    skipped without retiring the keys or aborting the whole run."""
    used = []

    def download_one(ticker, key):
        used.append((ticker, key))
        if ticker == "PREMIUM":
            raise FMPRateLimitError()          # 402 on every key for this ticker
        return pd.DataFrame([{"ticker": ticker, "as_of": pd.Timestamp("2024-01-01")}])

    tickers = ["AAA", "PREMIUM", "BBB"]
    frames, exhausted = run_rotating_fetch(
        tickers, ["k1", "k2"], logging.getLogger("test"), download_one, pause=0.0)

    got = {f["ticker"].iloc[0] for f in frames}
    assert not exhausted                    # one bad ticker must NOT abort the run
    assert got == {"AAA", "BBB"}            # PREMIUM skipped, the rest still fetched
    # After PREMIUM cascaded through both keys, BBB is still served by k1 (no key
    # was wrongly retired).
    assert ("BBB", "k1") in used

    print("\n=== SANITY CHECK: premium/unavailable ticker ===")
    print(f"  PREMIUM 402'd on both keys -> skipped; AAA/BBB fetched on k1; run not aborted.")
    print("  Validated.")


def test_rotation_stops_when_all_keys_truly_exhausted():
    """When every ticker fails on every key, conclude quota exhaustion after
    `max_consecutive_dead` consecutive full cascades and stop early."""
    def download_one(ticker, key):
        raise FMPRateLimitError()

    frames, exhausted = run_rotating_fetch(
        ["AAA", "BBB", "CCC", "DDD"], ["k1", "k2"], logging.getLogger("test"),
        download_one, pause=0.0, max_consecutive_dead=3)
    assert exhausted and frames == []

    print("\n=== SANITY CHECK: true quota exhaustion ===")
    print("  every ticker fails on every key -> stops early after 3 cascades. Validated.")
