"""SEC Fails-to-Deliver: parse + semi-monthly period logic + the FTD feature
(fails/volume, publication-lagged so it's leak-free)."""
from __future__ import annotations

import pandas as pd

from src.data_extract.utils.prices import fetch_fails_to_deliver as ftd
from src.data_aggregate.utils.short_interest_features import (
    build_short_interest_feature_panel, _FTD_PUB_LAG)


def test_periods_semimonthly_bounded():
    ps = ftd._periods(1, today=pd.Timestamp("2024-03-10"))
    assert ps[:2] == ["202301a", "202301b"]                 # a=1-15, b=16-end
    assert ps[-2:] == ["202403a", "202403b"]                # up to the current month only
    assert "202312b" in ps
    assert all(int(p[:4]) >= ftd.SEC_FTD_FIRST_YEAR for p in ftd._periods(50, today=pd.Timestamp("2024-03-10")))
    print("\n=== SANITY: FTD semi-monthly periods ===")
    print(f"  years_history=1 @2024-03 -> {len(ps)} files 202301a..202403b (current month only). Validated.")


def test_parse_ftd_math_and_na_price():
    raw = ("SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
           "20240102|X|AAPL|1000|APPLE INC|180.50\n"
           "20240102|Y|MSFT|500|MICROSOFT|.\n"              # PRICE '.' = N/A
           "20240103|Z|AAPL|200|APPLE INC|181.00\n")
    df = ftd._parse_ftd(raw)
    a = df[(df["ticker"] == "AAPL") & (df["date"] == pd.Timestamp("2024-01-02"))].iloc[0]
    assert a["fails_quantity"] == 1000.0 and abs(a["fails_value"] - 180_500.0) < 1e-6
    m = df[df["ticker"] == "MSFT"].iloc[0]
    assert m["fails_quantity"] == 500.0 and pd.isna(m["fails_value"])   # '.' price -> value NaN
    assert set(df["ticker"]) == {"AAPL", "MSFT"} and len(df) == 3       # 2 AAPL dates + 1 MSFT
    print("\n=== SANITY: FTD parse ===")
    print("  AAPL 1000@180.5 -> fails_value $180.5k; MSFT price '.' -> fails_value NaN. Validated.")


def test_ftd_feature_ranks_high_fails_and_is_leak_free():
    idx = pd.bdate_range("2024-01-01", periods=120)
    days = idx[:30]
    fails = pd.concat([
        pd.DataFrame({"date": days, "ticker": "HI", "fails_quantity": 1e5}),
        pd.DataFrame({"date": days, "ticker": "MID", "fails_quantity": 1e4}),
        pd.DataFrame({"date": days, "ticker": "LO", "fails_quantity": 1e2}),
    ], ignore_index=True)
    volume = pd.DataFrame({t: 1e6 for t in ("HI", "MID", "LO")}, index=idx)
    peers = {"HI": {"MID": 1.0, "LO": 1.0}, "MID": {"HI": 1.0, "LO": 1.0}, "LO": {"HI": 1.0, "MID": 1.0}}

    panel = build_short_interest_feature_panel(None, peers, idx, fails_history=fails, volume=volume)
    assert "f_fails_to_deliver_ratio_xs" in panel.columns

    # after the publication lag, HI (0.1 fails/vol) ranks above LO (0.0001)
    d = idx[_FTD_PUB_LAG + 25]
    row = panel[panel["date"] == d].set_index("ticker")
    assert row["f_fails_to_deliver_ratio_xs"]["HI"] > row["f_fails_to_deliver_ratio_xs"]["LO"]

    # leak-free: before the publication lag the fails signal is not yet visible
    early = panel[panel["date"] == idx[5]]
    assert early.empty or early["f_fails_to_deliver_ratio_xs"].isna().all()

    print("\n=== SANITY: FTD feature (fails/volume, publication-lagged) ===")
    print(f"  HI fails/vol 0.10 ranks above LO 0.0001 after the {_FTD_PUB_LAG}d lag; "
          f"pre-lag signal absent (leak-free). Validated.")
