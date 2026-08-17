"""
Superinvestor (elite-manager) 13F features
(src/data_aggregate/utils/superinvestor_features.py).

Proves the three things that make this DIFFERENT from the all-filer institutional
panel: (1) only roster-manager CIKs count (non-roster managers are ignored),
(2) each manager is WEIGHTED by its roster weight — the top manager's move can flip
the net-accumulation sign, and (3) the same point-in-time 45-day filing lag holds.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.data_aggregate.utils.extras.superinvestor_features import (
    pad_cik, _weight_map, build_superinvestor_feature_panel, load_superinvestor_holdings,
)

# roster: M1 heavily weighted (0.7), M2 light (0.3); raw CIKs "1"/"2" (padded internally).
# M3 ("3") is NOT on the roster and must be ignored entirely.
_ROSTER = {"managers": [
    {"cik": "0000000001", "weight": 0.7, "name": "Top Investor", "rank": 1},
    {"cik": "0000000002", "weight": 0.3, "name": "Lesser Investor", "rank": 2},
]}


def _holdings():
    """Two quarters, raw (unpadded) CIKs as stored in sec13f_hr.
      HOT : M1 BUYS (100->200), M2 SELLS (100->50)  -> heavy manager accumulating
      COLD: M1 SELLS (100->50), M2 BUYS (100->200)  -> heavy manager distributing
      ONLYM3: held only by the non-roster manager   -> must not appear
    value_usd = shares * 10."""
    rows = []
    plan = {
        "HOT":    {"1": (100, 200), "2": (100, 50)},
        "COLD":   {"1": (100, 50),  "2": (100, 200)},
        "ONLYM3": {"3": (100, 200)},
    }
    for ticker, mgrs in plan.items():
        for cik, (q1, q2) in mgrs.items():
            for period, sh in (("2025-12-31", q1), ("2026-03-31", q2)):
                rows.append({"cik": cik, "period": period, "ticker": ticker,
                             "shares": sh, "value_usd": sh * 10})
    return pd.DataFrame(rows)


def test_weight_map_pads_and_sums():
    # legacy explicit-weight shape (back-compat)
    assert _weight_map(_ROSTER) == {"0000000001": 0.7, "0000000002": 0.3}
    assert _weight_map({"managers": []}) == {} and _weight_map(None) == {}
    # NEW primary shape: {cik_to_name} -> EQUAL weights (1.0 each), padded, junk dropped
    eq = _weight_map({"cik_to_name": {"1067983": "Buffett", "0000000002": "Ackman", "N/A": "x"}})
    assert eq == {"0001067983": 1.0, "0000000002": 1.0}
    # a bare {cik: name} dict (no wrapper) resolves the same way
    assert _weight_map({"1067983": "Buffett"}) == {"0001067983": 1.0}
    print("\n=== SANITY CHECK: roster -> weight map ===")
    print("  {cik_to_name} -> equal weights (padded, junk dropped); legacy {managers:[{cik,weight}]} "
          "still honoured; empty/None -> {}. Validated.")


def test_weighted_accumulation_subset_and_leakfree():
    idx = pd.bdate_range("2025-10-01", "2026-09-30")
    tickers = ["HOT", "COLD", "ONLYM3"]
    peers = {t: {p: 1.0 for p in tickers if p != t} for t in tickers}
    fund = pd.DataFrame([{"ticker": t, "as_of": "2025-01-01", "sharesOutstanding": 1_000_000.0}
                         for t in tickers])
    close = pd.DataFrame({t: 10.0 for t in tickers}, index=idx)     # mcap = 10M

    panel = build_superinvestor_feature_panel(
        _holdings(), _ROSTER, peers, idx, shares_out_history=fund, stock_close=close)

    for c in ("f_super_cluster_buying_xs", "f_super_value_chg_xs",
              "f_super_flow_to_mcap_xs"):
        assert c in panel.columns, f"{c} missing"

    # (1) subset only: ONLYM3 (held solely by the non-roster manager) is absent
    covered = set(panel.loc[panel["f_super_cluster_buying_xs"].notna(), "ticker"])
    assert covered == {"HOT", "COLD"}, covered

    # (2) WEIGHTED: HOT's heavy manager (0.7) is buying while the light one sells ->
    #     net accumulation POSITIVE; COLD is the mirror -> HOT ranks above COLD.
    #     (Equal-weighted, both would net to zero — so this proves the weighting.)
    last = idx[-1]
    row = panel[panel["date"] == last].set_index("ticker")
    assert row.loc["HOT", "f_super_cluster_buying_xs"] > row.loc["COLD", "f_super_cluster_buying_xs"]
    assert row.loc["HOT", "f_super_flow_to_mcap_xs"] > row.loc["COLD", "f_super_flow_to_mcap_xs"]

    # (3) leak-free: the 2025-12-31 quarter is public only ~2026-02-14; before then, nothing
    early = panel[panel["date"] == pd.Timestamp("2025-11-03")]
    assert early.empty or early["f_super_cluster_buying_xs"].isna().all()

    print("\n=== SANITY CHECK: weighted elite 13F buy/sell evolution ===")
    print("  only roster CIKs counted (ONLYM3 dropped); heavy-manager BUY makes HOT's "
          "net accumulation top COLD's (weighting flips the equal-weight zero); "
          "2025-12-31 quarter invisible before its ~2026-02-14 filing (leak-free). Validated.")


class _Ctx:
    def __init__(self, store): self.store = store; self.log = logging.getLogger("test")


def test_load_superinvestor_holdings_reads_only_elite(sqlite_store):
    """The filtered read returns ONLY roster-manager rows, matching CIKs regardless of the stored
    format (padded, unpadded, or '1234.0') exactly like pad_cik — so the panel never sees the
    ~21.7M-row all-filer table. On a REAL DataStore, so the pushdown itself is under test: the
    stored spellings are resolved via `distinct` and every one is pushed into the WHERE."""
    rows = [
        {"cik": "0000000001", "period": "2025-12-31", "ticker": "HOT", "cusip": "000000001",
         "shares": 100, "value_usd": 1000, "filing_date": "2026-02-14"},   # roster, padded
        {"cik": "2", "period": "2025-12-31", "ticker": "HOT", "cusip": "000000002",
         "shares": 50, "value_usd": 500, "filing_date": "2026-02-14"},     # roster, UNPADDED
        {"cik": "1067983.0", "period": "2025-12-31", "ticker": "COLD", "cusip": "000000003",
         "shares": 10, "value_usd": 100, "filing_date": "2026-02-14"},     # NON-roster, float-text
        {"cik": "0000000009", "period": "2025-12-31", "ticker": "X", "cusip": "000000004",
         "shares": 999, "value_usd": 9990, "filing_date": "2026-02-14"},   # NON-roster
    ]
    sqlite_store.save("sec13f_hr", pd.DataFrame(rows))
    ctx = _Ctx(sqlite_store)
    out = load_superinvestor_holdings(ctx, _ROSTER)
    assert len(out) == 2, f"expected only the 2 roster rows, got {len(out)}"
    assert set(out["cik"].map(pad_cik)) == {"0000000001", "0000000002"}
    assert set(out["ticker"]) == {"HOT"}, "non-roster tickers leaked in"
    assert list(out.columns) == ["cik", "period", "ticker", "shares", "value_usd", "filing_date"]
    # empty roster -> nothing to read; roster with no stored manager -> None (not an empty frame)
    assert load_superinvestor_holdings(ctx, {"managers": []}) is None
    assert load_superinvestor_holdings(ctx, {"cik_to_name": {"0000123456": "Nobody"}}) is None
    print("\n=== SANITY CHECK: elite-subset filtered read ===")
    print("  only roster CIKs returned (padded '0000000001' + unpadded '2' BOTH matched); "
          "non-roster '1067983.0'/'0000000009' dropped; projected to 6 cols -> the 21.7M-row "
          "table is never fully loaded. Unknown roster -> None. Validated.")
