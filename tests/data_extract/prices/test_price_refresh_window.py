"""
test_price_refresh_window.py  (tests/data_extract/prices/test_price_refresh_window.py)
------------------------------------------------------------------
The incremental price window has to be able to repair its own recent past.

`resume_since` answers "the oldest per-ticker MAX date", which is a correct frontier only if
every ticker's history is contiguous up to its max. Two live failures proved it is not:

  * an INTERIOR hole -- 45 of 491 tickers on 2026-08-28, with 491 on both neighbouring
    sessions. Every ticker's MAX stayed current, so no incremental run would ever revisit it.
  * a PARTIAL final bar -- written mid-session at 0.535x its own trailing median volume, and
    never revisited to pick up the settled close.

`_refresh_floor` fixes both by making every run re-pull the trailing week. These tests pin
the arithmetic and the two clamps that keep it from over-reaching.
"""
import pandas as pd

from src.data_extract.utils.prices.fetch_prices import (
    PRICE_REFRESH_TRADING_DAYS, _chunk_response_to_frames, _refresh_floor)

UNTIL = pd.Timestamp("2026-09-04")          # a Friday
WINDOW_START = pd.Timestamp("2011-09-04")   # 15 years back
FLOOR = UNTIL - pd.tseries.offsets.BDay(PRICE_REFRESH_TRADING_DAYS)


def test_floor_widens_a_current_resume_frontier():
    """The defect case: `resume_since` says the table is current, so nothing would be
    re-fetched and the hole would survive forever."""
    got = _refresh_floor(pd.Timestamp("2026-09-03"), UNTIL, WINDOW_START)
    print(f"  resume_since=2026-09-03 -> since={got.date()} (floor {FLOOR.date()})")
    assert got == FLOOR
    assert (UNTIL - got).days >= 7, "must reach back over a calendar week at least"


def test_floor_covers_the_live_hole():
    """2026-08-28 must fall inside the re-pulled window when run against 2026-09-04."""
    got = _refresh_floor(pd.Timestamp("2026-09-01"), UNTIL, WINDOW_START)
    print(f"  since={got.date()} <= 2026-08-28 ? {got <= pd.Timestamp('2026-08-28')}")
    assert got <= pd.Timestamp("2026-08-28")


def test_floor_never_narrows_an_older_frontier():
    """A genuinely stale table keeps its own wider window -- the floor only ever widens."""
    stale = pd.Timestamp("2026-01-15")
    got = _refresh_floor(stale, UNTIL, WINDOW_START)
    print(f"  resume_since=2026-01-15 -> since={got.date()}")
    assert got == stale


def test_floor_is_clamped_at_window_start():
    """A short `years_history` is still respected: the floor may look back from `until`, but
    never widen the configured history."""
    narrow = pd.Timestamp("2026-09-02")     # window_start only 2 days back
    got = _refresh_floor(pd.Timestamp("2026-09-03"), UNTIL, narrow)
    print(f"  window_start=2026-09-02 -> since={got.date()}")
    assert got == narrow
    assert got >= narrow


def test_floor_is_idempotent():
    """Re-flooring an already-floored `since` must not walk the window back a week per run."""
    once = _refresh_floor(pd.Timestamp("2026-09-03"), UNTIL, WINDOW_START)
    twice = _refresh_floor(once, UNTIL, WINDOW_START)
    assert once == twice == FLOOR


def test_unserved_ticker_is_warned_not_swallowed(caplog):
    """The one line that made the 45-of-491 day invisible at fetch time. yfinance declining
    to serve a ticker used to `continue` in silence, so a truncated response looked exactly
    like a complete one."""
    idx = pd.DatetimeIndex(["2026-09-03", "2026-09-04"], name="Date")
    cols = pd.MultiIndex.from_product([["AAPL"], ["Open", "High", "Low", "Close",
                                                  "Adj Close", "Volume"]])
    data = pd.DataFrame(1.0, index=idx, columns=cols)

    with caplog.at_level("WARNING"):
        frames = _chunk_response_to_frames(data, ["AAPL", "MSFT", "NVDA"])

    print(f"  frames={len(frames)}; warning={caplog.text.strip()[-90:]}")
    assert len(frames) == 1, "the served ticker is still returned"
    assert "MSFT" in caplog.text and "NVDA" in caplog.text
    assert "2 of 3" in caplog.text


def test_fully_served_chunk_logs_nothing(caplog):
    idx = pd.DatetimeIndex(["2026-09-04"], name="Date")
    cols = pd.MultiIndex.from_product([["AAPL", "MSFT"], ["Close", "Adj Close", "Volume"]])
    data = pd.DataFrame(1.0, index=idx, columns=cols)

    with caplog.at_level("WARNING"):
        frames = _chunk_response_to_frames(data, ["AAPL", "MSFT"])

    assert len(frames) == 2
    assert caplog.text == "", f"unexpected log on a complete response: {caplog.text}"
