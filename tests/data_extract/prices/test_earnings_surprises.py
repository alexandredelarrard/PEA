"""Known-truth parsing of Yahoo earnings dates and optional EPS fields."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.prices import fetch_earnings_surprises as surprises


def test_earnings_response_normalizes_columns_dates_and_missing_actual(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = pd.DataFrame(
        {"EPS Estimate": [1.5], "Surprise(%)": [3.0]}, index=pd.DatetimeIndex(["2026-11-10 16:00"], tz="America/New_York", name="Earnings Date")
    )

    class Ticker:
        def __init__(self, symbol: str) -> None:
            assert symbol == "AAPL"

        def get_earnings_dates(self, **kwargs: Any) -> pd.DataFrame:
            return raw

    monkeypatch.setattr(surprises.yf, "Ticker", Ticker)
    parsed = surprises._download_page("AAPL", 100, 0)
    assert parsed is not None and parsed.columns.tolist() == ["ticker", "earnings_date", "eps_estimate", "eps_actual", "surprise_pct"]
    assert parsed.iloc[0]["earnings_date"] == pd.Timestamp("2026-11-10")
    assert parsed.iloc[0]["eps_estimate"] == 1.5 and pd.isna(parsed.iloc[0]["eps_actual"])
    print("SANITY: dated source publications keep their market date; an unreported actual remains missing.")
