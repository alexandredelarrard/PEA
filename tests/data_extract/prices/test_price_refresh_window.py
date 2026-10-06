"""
test_price_refresh_window.py  (tests/data_extract/prices/test_price_refresh_window.py)
------------------------------------------------------------------
A ticker yfinance declines to serve is warned about, never swallowed: its bars for the window are
MISSING, and the next run re-lists them because its frontier did not move. The re-pull window
itself (overlap, holes) is planned by `resume.series_windows` (see `test_resume_planner.py`).
"""

import pandas as pd

from src.data_extract.utils.prices.fetch_prices import _chunk_response_to_frames


def test_unserved_ticker_is_warned_not_swallowed(caplog):
    """The one line that made the 45-of-491 day invisible at fetch time. yfinance declining
    to serve a ticker used to `continue` in silence, so a truncated response looked exactly
    like a complete one."""
    idx = pd.DatetimeIndex(["2026-09-03", "2026-09-04"], name="Date")
    cols = pd.MultiIndex.from_product([["AAPL"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]])
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
