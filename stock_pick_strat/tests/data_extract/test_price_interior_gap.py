"""Regression: the incremental price updater must HEAL interior gaps, not only
append past each ticker's max_date.

Before the fix, a partial download saved once (e.g. SPY missing March/April)
advanced max_date past the hole, so no later run ever refetched it -- and because
the cube's trading calendar is defined by the market_ticker (SPY), a SPY interior
hole silently drops the whole universe for those dates.
"""
from __future__ import annotations

import pandas as pd

from src.data_extract.utils import fetch_prices as fp


def _existing_with_spy_gap():
    """Build a 'cached' price file that reaches back the full history window (so
    the backward-backfill path is not triggered): all names trade every business
    day from 2010 to 2025-05-30, EXCEPT SPY, which is missing all of March-April
    2025 (an interior hole) while still having data before and after."""
    cal = pd.bdate_range("2010-01-04", "2025-05-30")
    gap = pd.bdate_range("2025-03-01", "2025-04-30")
    stocks = [f"S{i:02d}" for i in range(8)]
    frames = []
    for tkr in stocks:
        frames.append(pd.DataFrame({"ticker": tkr, "date": cal, "close": 100.0}))
    spy_dates = cal.difference(gap)                    # SPY interior hole
    frames.append(pd.DataFrame({"ticker": "SPY", "date": spy_dates, "close": 400.0}))
    return pd.concat(frames, ignore_index=True)


def test_interior_spy_gap_is_scheduled_for_refetch():
    existing = _existing_with_spy_gap()

    cal = fp._trading_calendar(existing)
    spy_dates = existing.loc[existing["ticker"] == "SPY", "date"]
    gap_start = fp._interior_gap_start(spy_dates, cal)

    # first missing SPY trading day is the first business day of March 2025
    assert gap_start == pd.Timestamp("2025-03-03"), gap_start

    # a healthy ticker (no interior hole) reports no gap
    s00 = existing.loc[existing["ticker"] == "S00", "date"]
    assert fp._interior_gap_start(s00, cal) is None

    print("\n=== SANITY CHECK: interior-gap detection ===")
    print(f"  calendar has {len(cal)} reference trading days; SPY interior hole "
          f"detected starting {gap_start.date()}; healthy stock -> no gap. Validated.")


def test_download_plan_widens_to_cover_the_gap(monkeypatch=None):
    existing = _existing_with_spy_gap()
    # pin 'today' so the plan is deterministic (just after the file's last date)
    fixed_today = pd.Timestamp("2025-06-02")

    import types
    orig = pd.Timestamp.today
    pd.Timestamp.today = staticmethod(lambda: fixed_today)          # type: ignore
    try:
        plans = fp._tickers_needing_download(existing, ["SPY", "S00"], years_history=15)
    finally:
        pd.Timestamp.today = orig                                   # type: ignore

    # SPY's plan must start at the interior hole (heal it), not merely append tail
    assert plans["SPY"] is not None
    assert plans["SPY"][0] == pd.Timestamp("2025-03-03"), plans["SPY"]
    # a healthy, current ticker only needs a tail append (or nothing), never Mar
    assert plans["S00"] is None or plans["S00"][0] > pd.Timestamp("2025-05-30")

    print("\n=== SANITY CHECK: interior gap heals in the download plan ===")
    print(f"  SPY plan = {plans['SPY'][0].date()}..{plans['SPY'][1].date()} "
          f"(covers Mar-Apr hole + tail); S00 plan = {plans['S00']}. "
          f"Interior gaps are now refetched instead of orphaned. Validated.")


if __name__ == "__main__":
    test_interior_spy_gap_is_scheduled_for_refetch()
    test_download_plan_widens_to_cover_the_gap()
