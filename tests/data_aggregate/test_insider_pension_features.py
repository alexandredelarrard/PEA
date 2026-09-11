"""Tests for the two SEC-bulk cube wirings:
  * pension_facts -> point-in-time net pension deficit (fundamental_features)
  * insider_transactions -> trailing-window insider-buying signal (insider_features)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.fundamentals.fundamental_features import _pension_deficit_daily, _NET_PENSION_TAGS
from src.data_aggregate.utils.institutionals.insider_features import build_insider_feature_panel

PRIMARY, VARIANT = _NET_PENSION_TAGS


def test_pension_deficit_daily_pit_latest_period_and_primary_preference():
    idx = pd.bdate_range("2024-01-01", "2024-12-31")
    pf = pd.DataFrame({
        "ticker":  ["VZ", "VZ", "GE", "GE", "MSFT"],
        "tag":     [PRIMARY, PRIMARY, PRIMARY, VARIANT, VARIANT],
        "ddate":   ["2023-12-31", "2022-12-31", "2023-12-31", "2023-12-31", "2023-12-31"],
        "qtrs":    [0, 0, 0, 0, 0],
        "value":   [13.2e9, 12.9e9, 7e9, 6e9, 5e8],
        "filed":   ["2024-02-15", "2024-02-15", "2024-02-20", "2024-02-20", "2024-07-30"],
    })
    out = _pension_deficit_daily(pf, idx)
    # point-in-time: nothing before the filing
    assert np.isnan(out.loc[pd.Timestamp("2024-02-01"), "VZ"])
    # after filing: the LATEST period-end (2023: 13.2B), not the prior-year comparative
    assert abs(out.loc[pd.Timestamp("2024-03-01"), "VZ"] - 13.2e9) < 1
    # GE reports BOTH tags same filing -> primary (7B) preferred over variant (6B)
    assert abs(out.loc[pd.Timestamp("2024-03-01"), "GE"] - 7e9) < 1
    # MSFT reports only the variant -> variant fills; still PIT (only after its July filing)
    assert np.isnan(out.loc[pd.Timestamp("2024-06-03"), "MSFT"])
    assert abs(out.loc[pd.Timestamp("2024-08-15"), "MSFT"] - 5e8) < 1

    print("\n=== SANITY: pension_facts -> PIT net deficit ===")
    print(f"  VZ 2024-03 = ${out.loc[pd.Timestamp('2024-03-01'),'VZ']:,.0f} (latest FY, PIT after filing); "
          f"GE = ${out.loc[pd.Timestamp('2024-03-01'),'GE']:,.0f} (primary tag preferred); "
          f"MSFT variant fills. Validated.")


def _insider_txns():
    """⚠ REWRITTEN 2026-09-10 for the Phase 2.3 feature set. The five-column version this
    replaced produced an EMPTY panel through the rebuilt builder, which reads the whole Form 4
    record; the two tests below would then have failed on a missing column rather than on the
    thing they exist to check. BULL's three purchases carry three DISTINCT owner CIKs so the
    breadth feature counts people, not filings."""
    n = 7
    return pd.DataFrame({
        "accession_number": [f"0001-24-{i:06d}" for i in range(n)],
        "ticker":           ["BULL", "BULL", "BULL", "BEAR", "BEAR", "MIX", "MIX"],
        "owner_cik":        ["0001", "0002", "0003", "0004", "0005", "0006", "0007"],
        "filing_date":      ["2024-03-01", "2024-03-05", "2024-03-10",
                             "2024-03-01", "2024-03-05", "2024-03-01", "2024-03-06"],
        "transaction_date": ["2024-02-28", "2024-03-01", "2024-03-06",
                             "2024-02-28", "2024-03-01", "2024-02-28", "2024-03-04"],
        "transaction_code": ["P", "P", "P", "S", "S", "P", "S"],
        "shares":           [2e4, 4e4, 2e4, 6e4, 4e4, 2e4, 2e4],
        "price_per_share":  [50.0] * n,
        "value_usd":        [1e6, 2e6, 1e6, 3e6, 2e6, 1e6, 1e6],
        "shares_owned_after": [1e5] * n,
        "security_type":    ["nonderiv"] * n,
        "security_title":   ["Common Stock"] * n,
        "direct_indirect":  ["D"] * n,
        "officer_title":    [""] * n,
        "is_director":      [1.0] * n,
        "is_officer":       [0.0] * n,
        "is_ten_pct_owner": [0.0] * n,
        "is_10b5_1":        [np.nan] * n,
        "acquired_disposed": ["A", "A", "A", "D", "D", "A", "D"],
    })


def test_insider_signal_ordering_and_leak_free():
    idx = pd.bdate_range("2024-01-01", "2024-12-31")
    peers = {"BULL": {"BEAR": 1.0, "MIX": 1.0},
             "BEAR": {"BULL": 1.0, "MIX": 1.0},
             "MIX":  {"BULL": 1.0, "BEAR": 1.0}}
    panel = build_insider_feature_panel(_insider_txns(), peers, idx)
    assert not panel.empty and "f_ic_insider_net_buy_ratio_180d" in panel.columns

    d = pd.Timestamp("2024-04-01")           # inside the 180d window, after the March filings
    row = panel[panel["date"] == d].set_index("ticker")
    nbr = row["f_ic_insider_net_buy_ratio_180d"]
    # net-buy ratio: BULL (all buys, +1) > MIX (balanced, 0) > BEAR (all sells, -1)
    assert nbr["BULL"] > nbr["MIX"] > nbr["BEAR"]
    # breadth: BULL had three DIFFERENT insiders buy -> the highest distinct-buyer count
    buyers = row["f_ic_insider_distinct_buyers_120d"]
    assert buyers["BULL"] == 3.0 and buyers["BULL"] == buyers.max()

    # leak-free: before any Form-4 filing the window is empty -> net-buy ratio undefined
    before = panel[panel["date"] == pd.Timestamp("2024-02-01")].set_index("ticker")
    assert before["f_ic_insider_net_buy_ratio_180d"].isna().all()

    print("\n=== SANITY: insider net-buy signal ===")
    print(f"  net_buy_ratio_180d BULL={nbr['BULL']:.2f} > MIX={nbr['MIX']:.2f} > "
          f"BEAR={nbr['BEAR']:.2f}; BULL shows {buyers['BULL']:.0f} distinct buyers "
          f"(3 owners, not 3 filings); pre-filing signal is NaN (leak-free). Validated.")


def test_insider_buy_value_is_scaled_by_market_cap():
    idx = pd.bdate_range("2024-01-01", "2024-12-31")
    peers = {"BULL": {"BEAR": 1.0, "MIX": 1.0}, "BEAR": {"BULL": 1.0, "MIX": 1.0},
             "MIX": {"BULL": 1.0, "BEAR": 1.0}}
    shares = pd.DataFrame({"ticker": ["BULL", "BEAR", "MIX"],
                           "as_of": ["2023-06-01"] * 3,
                           "sharesOutstanding": [1e8, 1e8, 1e8],
                           "sharesOutstandingPit": [1e8, 1e8, 1e8]})
    close = pd.DataFrame({t: 50.0 for t in ("BULL", "BEAR", "MIX")}, index=idx)
    panel = build_insider_feature_panel(_insider_txns(), peers, idx,
                                        shares_out_history=shares, stock_close=close)
    assert "f_ic_insider_buy_value_mcap_180d" in panel.columns
    d = pd.Timestamp("2024-04-01")
    row = panel[panel["date"] == d].set_index("ticker")
    got = row["f_ic_insider_buy_value_mcap_180d"]
    # BULL bought $4m on a $5bn cap = 0.08%; BEAR bought nothing, so its BUY leg is 0.
    assert got["BULL"] == pytest.approx(4e6 / 5e9) and got["BEAR"] == 0.0
    print("\n=== SANITY: insider buying / market cap ===")
    print(f"  BULL bought $4m against a $5bn cap -> {got['BULL']:.6f}, exactly 4e6/5e9; "
          f"BEAR bought nothing -> {got['BEAR']:.0f}, a zero rather than a NaN because the "
          f"window is populated and the answer is 'no buying'. Validated.")
