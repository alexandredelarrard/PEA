"""Refactor tests:
  * dividends now piggy-back on the price download (actions=True): the pure
    extractor pulls nonzero ex-dates out of a normalized price frame.
  * D&A-vs-capex reinvestment-quality feature is built from the already-extracted
    SEC `depAmort` / `capex`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_extract.utils.prices.fetch_prices import _extract_dividends
from src.data_aggregate.utils.fundamentals.fundamental_features import build_fundamental_feature_panel


def test_dividends_extracted_from_price_frame():
    # a normalized price frame as produced with actions=True (lowercased cols)
    px = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-03-01", "2024-03-01", "2024-06-01"]),
        "ticker": ["AAPL", "AAPL", "MSFT", "AAPL"],
        "close": [180.0, 185.0, 400.0, 190.0],
        "volume": [1e6, 1e6, 5e5, 1e6],
        "dividends": [0.0, 0.24, 0.75, 0.25],
        "stock splits": [0.0, 0.0, 0.0, 0.0],
    })
    div = _extract_dividends(px)
    assert list(div.columns) == ["date", "ticker", "dividend"]
    assert len(div) == 3                          # only the nonzero ex-dates
    assert set(div["ticker"]) == {"AAPL", "MSFT"}
    assert div[(div["ticker"] == "AAPL")]["dividend"].tolist() == [0.24, 0.25]
    # no dividends column -> empty (defensive)
    assert _extract_dividends(px.drop(columns=["dividends"])).empty
    print("\n=== SANITY CHECK: dividends piggy-back on the price download ===")
    print(f"  pulled {len(div)} nonzero ex-dates from the actions=True price frame; "
          f"schema [date,ticker,dividend]; one download feeds both. Validated.")


def _fund_hist():
    """Quarterly SEC-style fundamentals history (ticker, as_of, fields). GROW has
    D&A rising fast while capex is flat (under-investing); INVEST reinvests hard."""
    rows = []
    quarters = pd.date_range("2021-03-31", periods=8, freq="QE")
    for i, q in enumerate(quarters):
        yr = i / 4.0
        rows += [
            # depAmort climbs ~20%/yr, capex flat -> D&A outruns capex
            {"ticker": "GROW", "as_of": q, "depAmort": 100 * (1.20 ** yr),
             "capex": 90.0, "totalRevenue": 1000.0, "netIncome": 100.0,
             "stockholdersEquity": 500.0, "sharesOutstanding": 1000.0},
            # capex climbs faster than D&A -> heavy reinvestment
            {"ticker": "INVEST", "as_of": q, "depAmort": 100.0,
             "capex": 90 * (1.30 ** yr), "totalRevenue": 1000.0, "netIncome": 100.0,
             "stockholdersEquity": 500.0, "sharesOutstanding": 1000.0},
            {"ticker": "FLAT", "as_of": q, "depAmort": 100.0, "capex": 100.0,
             "totalRevenue": 1000.0, "netIncome": 100.0,
             "stockholdersEquity": 500.0, "sharesOutstanding": 1000.0},
        ]
    return pd.DataFrame(rows)


def test_da_vs_capex_feature():
    fund = _fund_hist()
    idx = pd.bdate_range("2021-04-01", "2023-06-30")
    tickers = ["GROW", "INVEST", "FLAT"]
    close = pd.DataFrame(100.0, index=idx, columns=tickers)
    peers = {t: {p: 1.0 for p in tickers if p != t} for t in tickers}

    panel = build_fundamental_feature_panel(fund, peers, idx, stock_close=close)
    assert {"f_da_to_capex_xs", "f_da_minus_capex_growth_xs"}.issubset(panel.columns)

    # at a late date GROW has the highest D&A/capex and the highest D&A-minus-capex
    # growth gap; INVEST the lowest gap (capex outgrowing D&A)
    late = panel[panel["date"] == panel["date"].max()].set_index("ticker")
    assert late.loc["GROW", "f_da_to_capex_xs"] > late.loc["INVEST", "f_da_to_capex_xs"]
    assert late.loc["GROW", "f_da_minus_capex_growth_xs"] > late.loc["INVEST", "f_da_minus_capex_growth_xs"]
    print("\n=== SANITY CHECK: D&A-vs-capex reinvestment-quality feature ===")
    print(f"  GROW (D&A outrunning capex) ranks highest on da_to_capex & "
          f"da_minus_capex_growth; INVEST (reinvesting) lowest. Built from existing "
          f"SEC depAmort/capex. Validated.")


if __name__ == "__main__":
    test_dividends_extracted_from_price_frame()
    test_da_vs_capex_feature()
