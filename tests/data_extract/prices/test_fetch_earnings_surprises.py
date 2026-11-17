"""Fixed-date source history and provider pagination for earnings surprises."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.prices import fetch_earnings_surprises as surprises
from src.data_store.schema import Tables
from src.utils.window import Window, WindowRun
from tests.data_extract.edgar_fixtures import fake_context


class _CappedYahooTicker:
    """Stands in for `yf.Ticker` over a fixed quarterly history ending 2026-07-30. Like yfinance 1.5,
    it refuses a limit above 100, serves `limit` rows from `offset` (newest first), returns None past
    the end, and caches by `limit` alone per instance (a reused instance ignores a new offset)."""

    calls: list[tuple[int, int]] = []
    history: pd.DatetimeIndex = pd.date_range(end="2026-07-30", periods=150, freq="91D", tz="America/New_York")

    def __init__(self, ticker: str) -> None:
        self._ticker = ticker
        self._cache: dict[int, pd.DataFrame | None] = {}

    def get_earnings_dates(self, limit: int = 12, offset: int = 0) -> pd.DataFrame | None:
        type(self).calls.append((limit, offset))
        if limit > 100:
            raise ValueError("Yahoo caps limit at 100")
        if limit in self._cache:
            return self._cache[limit]
        newest_first = type(self).history[::-1][offset : offset + limit]
        page = None
        if len(newest_first):
            page = pd.DataFrame({"EPS Estimate": 1.0, "Reported EPS": 1.1, "Surprise(%)": 10.0}, index=pd.Index(newest_first, name="Earnings Date"))
        self._cache[limit] = page
        return page


def test_cold_key_reads_configured_history_under_the_provider_cap(tmp_path, sqlite_store, monkeypatch: pytest.MonkeyPatch) -> None:
    context = fake_context(tmp_path, sqlite_store, ["HOG"])
    context.run = WindowRun("incremental", "cold earnings history", pd.Timestamp("2026-09-30"))
    _CappedYahooTicker.calls = []
    monkeypatch.setattr(surprises.yf, "Ticker", _CappedYahooTicker)
    monkeypatch.setattr(surprises.time, "sleep", lambda seconds: None)
    surprises.fetch_earnings_surprises(context, ["HOG"])
    saved = sqlite_store.load(Tables.earnings_surprises)
    assert _CappedYahooTicker.calls == [(100, 0)]
    assert pd.to_datetime(saved["earnings_date"]).min() >= pd.Timestamp("2002-10-01")
    assert pd.to_datetime(saved["earnings_date"]).max() <= context.run.as_of
    print("SANITY: a cold key requests provider-sized pages and saves only the configured source history.")


def test_full_history_pages_until_the_fixed_start_instead_of_a_quarter_count(monkeypatch: pytest.MonkeyPatch) -> None:
    _CappedYahooTicker.calls = []
    monkeypatch.setattr(surprises.yf, "Ticker", _CappedYahooTicker)
    window = Window(pd.Timestamp("1990-01-01"), pd.Timestamp("2026-09-30"))
    saved = surprises._download_one("HOG", window)
    wanted = _CappedYahooTicker.history.tz_localize(None).normalize()
    wanted = wanted[(wanted >= window.start) & (wanted <= window.end)]
    assert saved is not None and len(saved) == len(wanted)
    assert [offset for limit, offset in _CappedYahooTicker.calls] == [0, 100]
    assert pd.to_datetime(saved["earnings_date"]).min() == wanted.min()
    print("SANITY: full history crosses the100-row provider limit and stops at the configured date, without a year-derived count cap.")


def test_repeated_provider_page_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.date_range("2020-01-01", periods=100)
    page = pd.DataFrame({"ticker": "HOG", "earnings_date": dates, "eps_actual": 1.0})
    monkeypatch.setattr(surprises, "_download_page", lambda *args: page)
    with pytest.raises(ValueError, match="pagination did not advance"):
        surprises._download_one("HOG", Window(pd.Timestamp("1990-01-01"), pd.Timestamp("2026-09-30")))
    print("SANITY: a provider repeating its newest page cannot loop forever or certify complete history.")


def test_unavailable_source_does_not_change_existing_rows(tmp_path, sqlite_store, monkeypatch: pytest.MonkeyPatch) -> None:
    context = fake_context(tmp_path, sqlite_store, ["HOG"])
    existing = pd.DataFrame(
        {"ticker": "HOG", "earnings_date": [pd.Timestamp("2026-09-15")], "eps_estimate": 1.0, "eps_actual": 1.1, "surprise_pct": 10.0}
    )
    sqlite_store.save(Tables.earnings_surprises, existing)
    before = sqlite_store.load(Tables.earnings_surprises)
    monkeypatch.setattr(surprises, "_download_one", lambda *args: None)
    surprises.fetch_earnings_surprises(context, ["HOG"])
    pd.testing.assert_frame_equal(sqlite_store.load(Tables.earnings_surprises), before)
    print("SANITY: an unavailable candidate response preserves all prior source rows.")


def test_programming_errors_escape_the_provider_failure_boundary(tmp_path, sqlite_store, monkeypatch: pytest.MonkeyPatch) -> None:
    context = fake_context(tmp_path, sqlite_store, ["HOG"])

    def broken(*args: Any) -> None:
        raise TypeError("implementation error")

    monkeypatch.setattr(surprises, "_download_one", broken)
    with pytest.raises(TypeError, match="implementation error"):
        surprises.fetch_earnings_surprises(context, ["HOG"])
    assert not sqlite_store.exists(Tables.earnings_surprises)
    print("SANITY: programming errors fail visibly before source persistence.")
