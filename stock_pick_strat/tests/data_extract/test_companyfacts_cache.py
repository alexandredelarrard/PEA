"""`_fetch_companyfacts` re-uses the on-disk companyfacts cache while it is FRESH.

The cache (~2GB for the S&P 500) was written on every download but only readable when
`context.use_cache` was set — which no caller does — so it sat unread and every rebuild
re-downloaded all 500 payloads.

Why the window is SUB-DAILY (20h) rather than the 2 days that first seemed natural: the
extraction DAG runs at 01:00 daily, so any window >= 24h lets a run skip its refresh and
delays a new filing. Measured on the live filing calendar (782 business days of
`fundamentals_history.as_of`): filings land on 74% of business days, median 3/day, p90 32,
max 71, and they are strongly seasonal — Feb 20.9%, May 14.3%, Aug 14.1%, Nov 12.0%,
Oct 11.6%, i.e. 73% of all filings in five months, with 14% of business days carrying >= 20
filings. A 2-day window therefore costs a MEAN 1.0 business day of filing visibility,
concentrated on exactly the days that carry the most new information.
"""
from __future__ import annotations

import json
import os
import time
import types
from pathlib import Path

import pytest

import src.data_extract.utils.fundamentals.fetch_fundamentals as F

_CIK = "0000000001"
_PAYLOAD = {"facts": {"us-gaap": {"Assets": {"units": {"USD": []}}}}}


@pytest.fixture
def ctx(tmp_path):
    """Minimal stand-in for Context: just the cache dir + the use_cache flag."""
    return types.SimpleNamespace(paths={"SEC_BULK_CACHE_DIR": tmp_path}, use_cache=False)


@pytest.fixture
def spy(monkeypatch):
    """Count network attempts and make them fail, so cache use is unambiguous."""
    calls: list[str] = []

    def _blocked(url, **kw):
        calls.append(url)
        raise RuntimeError("network blocked by test")

    monkeypatch.setattr(F, "sec_get", _blocked)
    return calls


def _write_cache(ctx, age_hours: float) -> Path:
    p = Path(ctx.paths["SEC_BULK_CACHE_DIR"]) / f"companyfacts_CIK{_CIK}.json"
    p.write_text(json.dumps(_PAYLOAD), encoding="utf-8")
    when = time.time() - age_hours * 3600
    os.utime(p, (when, when))
    return p


def test_fresh_cache_is_reused_without_touching_the_network(ctx, spy):
    """The rebuild case: a payload downloaded earlier in the session costs nothing."""
    _write_cache(ctx, age_hours=1)
    assert F._fetch_companyfacts(ctx, _CIK) == _PAYLOAD
    assert spy == [], "hit the network despite a fresh cache"


def test_cache_older_than_the_window_triggers_a_refresh(ctx, spy):
    """The daily case: a ~24h-old cache (what the 01:00 run always sees) must refresh."""
    _write_cache(ctx, age_hours=24)
    F._fetch_companyfacts(ctx, _CIK)
    assert len(spy) == 1, "reused a cache older than the freshness window"


def test_caller_may_widen_the_window_for_a_deliberate_backfill(ctx, spy):
    _write_cache(ctx, age_hours=72)
    assert F._fetch_companyfacts(ctx, _CIK, max_age_hours=24 * 7) == _PAYLOAD
    assert spy == [], "ignored the caller's explicit window"


def test_use_cache_still_forces_the_cache_unconditionally(ctx, spy):
    """Offline / dev runs (`use_cache=True`) must never need the network."""
    ctx.use_cache = True
    _write_cache(ctx, age_hours=24 * 365)
    assert F._fetch_companyfacts(ctx, _CIK) == _PAYLOAD
    assert spy == []


def test_stale_cache_is_a_fallback_when_the_download_fails(ctx, spy):
    """A transient SEC outage used to return None and drop the ticker from the rebuild
    entirely, silently shrinking the universe for that run. A stale payload beats none."""
    _write_cache(ctx, age_hours=72)
    got = F._fetch_companyfacts(ctx, _CIK)
    assert len(spy) == 1, "should have tried to refresh first"
    assert got == _PAYLOAD, "stale cache not used as a fallback"


def test_absent_and_corrupt_caches_both_fall_through_to_download(ctx, spy):
    # absent
    assert F._fetch_companyfacts(ctx, _CIK) is None
    assert len(spy) == 1
    # corrupt but FRESH -> must not be trusted
    p = _write_cache(ctx, age_hours=0)
    p.write_text("{ this is not json", encoding="utf-8")
    os.utime(p, None)
    spy.clear()
    F._fetch_companyfacts(ctx, _CIK)
    assert len(spy) == 1, "trusted a corrupt cache"
