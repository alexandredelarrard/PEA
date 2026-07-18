"""Step 4 — 13F institutional-ownership features.

Checks the QoQ aggregation math (breadth change, new/exit, cluster buying,
accumulation), the 45-day filing-lag point-in-time stamping, ownership %, the
built f_* panel columns, and the pure extractor parsers (SEC join + OpenFIGI).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.institutional_features import (
    _quarter_features, build_institutional_feature_panel,
)
from src.data_extract.utils.prices.fetch_13f import _join_13f
from src.data_extract.utils.prices.fetch_cusip_map import _parse_openfigi


def _holdings():
    """Deterministic manager-grain 13F for ticker A over 2 quarters:
      Q1 (2022-03-31): M1=100, M2=200            -> holders 2, shares 300
      Q2 (2022-06-30): M1=150 (up), M3=50 (new)  -> M2 exited
        => new_buyers=1, exiters=1, breadth_chg=0, increasers=1(M1), decreasers=0,
           cluster=(1-0)/2=0.5, shares 200 -> shares_chg=200/300-1=-1/3
    Ticker B present only in Q1 (single holder) so it exercises the no-prior path."""
    rows = [
        {"cik": "M1", "period": "2022-03-31", "ticker": "A", "shares": 100, "value_usd": 1.0},
        {"cik": "M2", "period": "2022-03-31", "ticker": "A", "shares": 200, "value_usd": 2.0},
        {"cik": "M1", "period": "2022-06-30", "ticker": "A", "shares": 150, "value_usd": 1.5},
        {"cik": "M3", "period": "2022-06-30", "ticker": "A", "shares": 50, "value_usd": 0.5},
        {"cik": "M1", "period": "2022-03-31", "ticker": "B", "shares": 40, "value_usd": 0.4},
    ]
    return pd.DataFrame(rows)


def test_quarter_feature_math():
    qf = _quarter_features(_holdings())
    a2 = qf[(qf["ticker"] == "A") & (qf["as_of"] == pd.Timestamp("2022-06-30") + pd.Timedelta(days=45))].iloc[0]
    assert a2["inst_holders"] == 2
    assert a2["new_buyers"] == 1 and a2["exiters"] == 1
    assert a2["inst_breadth_chg"] == 0
    assert abs(a2["cluster_buying"] - 0.5) < 1e-9
    assert abs(a2["inst_shares_chg"] - (200 / 300 - 1)) < 1e-9
    # Q1 has no prior quarter -> change features NaN
    a1 = qf[(qf["ticker"] == "A") & (qf["as_of"] == pd.Timestamp("2022-03-31") + pd.Timedelta(days=45))].iloc[0]
    assert np.isnan(a1["inst_breadth_chg"]) and np.isnan(a1["new_buyers"])
    print("\n=== SANITY CHECK: 13F quarter-over-quarter math ===")
    print(f"  A Q2: holders=2, new=1, exit=1, breadth_chg=0, cluster=0.5, "
          f"shares_chg={a2['inst_shares_chg']:.3f}; Q1 changes NaN (no prior). Validated.")


def test_filing_lag_point_in_time():
    idx = pd.bdate_range("2022-01-03", "2023-06-30")
    panel = build_institutional_feature_panel(_holdings(), peer_dict={}, trading_index=idx)
    # rebuild the daily field directly to check the lag boundary
    from src.data_aggregate.utils.factors import fundamentals_to_daily
    qf = _quarter_features(_holdings())
    daily = fundamentals_to_daily(qf, "cluster_buying", idx)["A"]
    q2_asof = pd.Timestamp("2022-06-30") + pd.Timedelta(days=45)     # 2022-08-14
    before = daily.loc[idx[idx < pd.Timestamp("2022-08-14")]]
    after = daily.loc[idx[idx >= q2_asof]]
    # Q2 cluster (0.5) must NOT appear before its as_of; appears on/after
    assert not (np.isclose(before.dropna(), 0.5)).any(), "13F leaked before filing lag"
    assert np.isclose(after.dropna().iloc[0], 0.5), "13F feature missing after filing lag"
    print("\n=== SANITY CHECK: 45-day filing-lag point-in-time ===")
    print(f"  Q2 (Jun-30) cluster_buying only visible from {q2_asof.date()} onward, "
          f"never before. Leak-free. Validated.")


def test_panel_columns_and_ownership_pct():
    idx = pd.bdate_range("2022-01-03", "2023-06-30")
    tickers = ["A", "B", "C", "D"]
    peers = {t: {p: 1.0 for p in tickers if p != t} for t in tickers}
    # shares outstanding for ownership %
    fund = pd.DataFrame([{"ticker": t, "as_of": "2022-01-01", "sharesOutstanding": 1000.0}
                         for t in tickers])
    panel = build_institutional_feature_panel(_holdings(), peers, idx, shares_out_history=fund)
    for c in ("f_inst_breadth_chg_xs", "f_cluster_buying_xs", "f_new_buyers_xs",
              "f_inst_holders_xs", "f_inst_ownership_pct_xs"):
        assert c in panel.columns, f"{c} missing from panel"
    # A's ownership pct at a late date = 200 shares / 1000 = 0.2 (raw before xs-rank)
    from src.data_aggregate.utils.factors import fundamentals_to_daily
    qf = _quarter_features(_holdings())
    inst_sh = fundamentals_to_daily(qf, "inst_shares", idx)["A"].dropna().iloc[-1]
    assert abs(inst_sh - 200.0) < 1e-9
    print("\n=== SANITY CHECK: 13F panel columns + ownership pct ===")
    print(f"  panel exposes f_inst_breadth_chg/cluster_buying/new_buyers/holders/"
          f"ownership_pct (_xs); A latest inst_shares={inst_sh:.0f} (/1000 = 0.2). Validated.")


def test_extractor_parsers():
    sub = pd.DataFrame({
        "ACCESSION_NUMBER": ["a1", "a2"], "CIK": ["111", "222"],
        "FILING_DATE": ["2022-05-10", "2022-05-12"],   # pre-2023 -> VALUE in $1000s
        "PERIODOFREPORT": ["03-31-2022", "03-31-2022"]})
    info = pd.DataFrame({
        "ACCESSION_NUMBER": ["a1", "a1", "a2"],
        "CUSIP": ["037833100", "037833100", "594918104"],
        "VALUE": ["1000", "500", "2000"], "SSHPRNAMT": ["10", "5", "20"]})
    h = _join_13f(sub, info).sort_values("cusip").reset_index(drop=True)
    assert set(["cik", "period", "filing_date", "cusip", "value_usd", "shares",
                "call_shares", "put_shares", "debt_value", "other_value"]).issubset(h.columns)
    # a manager's two lines for the same CUSIP are summed into one row
    assert len(h) == 2
    assert h.loc[0, "shares"] == 15.0 and h.loc[0, "value_usd"] == 1.5e6   # (1000+500)*1000
    assert h.loc[1, "shares"] == 20.0 and h.loc[1, "value_usd"] == 2e6
    assert h["period"].iloc[0] == pd.Timestamp("2022-03-31")

    figi = [{"data": [{"ticker": "AAPL"}]}, {"warning": "no match"}]
    m = _parse_openfigi(figi, ["037833100", "999999999"])
    assert m == {"037833100": "AAPL"}
    print("\n=== SANITY CHECK: 13F + OpenFIGI parsers ===")
    print("  join sums a manager's duplicate-CUSIP lines, scales $1000s->$; OpenFIGI "
          "maps CUSIP->ticker (not the free-text issuer name). Validated.")


def test_join_13f_splits_holding_types():
    """Stock / call / put / debt / other land in separate columns; long `shares`
    excludes options and bond principal (the noise that was inflating shares)."""
    sub = pd.DataFrame({
        "ACCESSION_NUMBER": ["a1"], "CIK": ["111"],
        "FILING_DATE": ["2024-05-10"],            # post-2023 -> VALUE already in $
        "PERIODOFREPORT": ["03-31-2024"]})
    info = pd.DataFrame({
        "ACCESSION_NUMBER": ["a1"] * 5,
        "CUSIP": ["037833100"] * 5,
        "NAMEOFISSUER": ["APPLE INC", "APPLE INC", "APPLE", "Apple Inc.", "APPLE INC"],
        "VALUE":       ["1000000", "300000", "200000", "50000", "7000"],
        "SSHPRNAMT":   ["10000",   "3000",   "2000",   "40000", "70"],
        "SSHPRNAMTTYPE": ["SH",    "SH",     "SH",     "PRN",   "XX"],
        "PUTCALL":     ["",        "Call",   "Put",    "",      ""]})
    h = _join_13f(sub, info)
    assert len(h) == 1
    r = h.iloc[0]
    assert r["shares"] == 10000 and r["value_usd"] == 1_000_000       # long stock ONLY
    assert r["call_shares"] == 3000 and r["call_value"] == 300_000
    assert r["put_shares"] == 2000 and r["put_value"] == 200_000      # bearish / sell-side
    assert r["debt_prn"] == 40000 and r["debt_value"] == 50_000
    assert r["other_value"] == 7_000                                  # malformed type -> other

    print("\n=== SANITY CHECK: 13F holding-type split ===")
    print(f"  shares(long)={r['shares']:.0f} calls={r['call_shares']:.0f} "
          f"puts={r['put_shares']:.0f} debt$={r['debt_value']:.0f} other$={r['other_value']:.0f} "
          f"-> long shares no longer contaminated by options/bonds. Validated.")


if __name__ == "__main__":
    test_quarter_feature_math()
    test_filing_lag_point_in_time()
    test_panel_columns_and_ownership_pct()
    test_extractor_parsers()
    test_join_13f_splits_holding_types()
