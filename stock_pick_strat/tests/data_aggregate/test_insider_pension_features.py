"""Tests for the two SEC-bulk cube wirings:
  * pension_facts -> point-in-time net pension deficit (fundamental_features)
  * insider_transactions -> trailing-window insider-buying signal (insider_features)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.fundamentals.fundamental_features import (
    _pension_deficit_daily, _NET_PENSION_TAGS)
from src.data_aggregate.utils.extras.insider_features import build_insider_feature_panel

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
    return pd.DataFrame({
        "ticker":           ["BULL", "BULL", "BULL", "BEAR", "BEAR", "MIX", "MIX"],
        "filing_date":      ["2024-03-01", "2024-03-05", "2024-03-10",
                             "2024-03-01", "2024-03-05", "2024-03-01", "2024-03-06"],
        "transaction_code": ["P", "P", "P", "S", "S", "P", "S"],
        "value_usd":        [1e6, 2e6, 1e6, 3e6, 2e6, 1e6, 1e6],
        "acquired_disposed": ["A", "A", "A", "D", "D", "A", "D"],
    })


def test_insider_signal_ordering_and_leak_free():
    idx = pd.bdate_range("2024-01-01", "2024-12-31")
    peers = {"BULL": {"BEAR": 1.0, "MIX": 1.0},
             "BEAR": {"BULL": 1.0, "MIX": 1.0},
             "MIX":  {"BULL": 1.0, "BEAR": 1.0}}
    panel = build_insider_feature_panel(_insider_txns(), peers, idx)
    assert not panel.empty and "f_insider_net_buy_ratio_xs" in panel.columns

    d = pd.Timestamp("2024-04-01")           # inside the 180d window, after the March filings
    row = panel[panel["date"] == d].set_index("ticker")
    nbr = row["f_insider_net_buy_ratio_xs"]
    # net-buy ratio: BULL (all buys) > MIX (balanced) > BEAR (all sells)
    assert nbr["BULL"] > nbr["MIX"] > nbr["BEAR"]
    # cluster buying: BULL made 3 purchases -> top buy-count percentile
    assert row["f_insider_buy_count_xs"]["BULL"] == row["f_insider_buy_count_xs"].max()

    # leak-free: before any Form-4 filing the window is empty -> net-buy ratio undefined
    before = panel[panel["date"] == pd.Timestamp("2024-02-01")].set_index("ticker")
    assert before["f_insider_net_buy_ratio_xs"].isna().all()

    print("\n=== SANITY: insider net-buy signal ===")
    print(f"  net-buy_ratio_xs BULL={nbr['BULL']:.2f} > MIX={nbr['MIX']:.2f} > BEAR={nbr['BEAR']:.2f}; "
          f"BULL tops cluster buy-count; pre-filing signal is NaN (leak-free). Validated.")


def test_insider_net_buy_to_mcap_uses_market_cap():
    idx = pd.bdate_range("2024-01-01", "2024-12-31")
    peers = {"BULL": {"BEAR": 1.0, "MIX": 1.0}, "BEAR": {"BULL": 1.0, "MIX": 1.0},
             "MIX": {"BULL": 1.0, "BEAR": 1.0}}
    shares = pd.DataFrame({"ticker": ["BULL", "BEAR", "MIX"],
                           "as_of": ["2023-06-01"] * 3,
                           "sharesOutstanding": [1e8, 1e8, 1e8]})
    close = pd.DataFrame({t: 50.0 for t in ("BULL", "BEAR", "MIX")}, index=idx)
    panel = build_insider_feature_panel(_insider_txns(), peers, idx,
                                        shares_out_history=shares, stock_close=close)
    assert "f_insider_net_buy_to_mcap_xs" in panel.columns
    d = pd.Timestamp("2024-04-01")
    row = panel[panel["date"] == d].set_index("ticker")
    # BULL bought net +$4m on a $5bn cap -> higher scaled conviction than net-selling BEAR
    assert row["f_insider_net_buy_to_mcap_xs"]["BULL"] > row["f_insider_net_buy_to_mcap_xs"]["BEAR"]
    print("\n=== SANITY: insider net-buy / market-cap ===")
    print("  net_buy_to_mcap present; BULL (net buyer) ranks above BEAR (net seller). Validated.")
