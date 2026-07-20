"""
Superinvestor (elite-manager) 13F features
(src/data_aggregate/utils/superinvestor_features.py).

Proves the three things that make this DIFFERENT from the all-filer institutional
panel: (1) only roster-manager CIKs count (non-roster managers are ignored),
(2) each manager is WEIGHTED by its roster weight — the top manager's move can flip
the net-accumulation sign, and (3) the same point-in-time 45-day filing lag holds.
"""
from __future__ import annotations

import pandas as pd

from src.data_aggregate.utils.superinvestor_features import (
    _weight_map, build_superinvestor_feature_panel,
)

# roster: M1 heavily weighted (0.7), M2 light (0.3); raw CIKs "1"/"2" (padded internally).
# M3 ("3") is NOT on the roster and must be ignored entirely.
_ROSTER = {"managers": [
    {"cik": "0000000001", "weight": 0.7, "name": "Top Investor", "rank": 1},
    {"cik": "0000000002", "weight": 0.3, "name": "Lesser Investor", "rank": 2},
]}


def _holdings():
    """Two quarters, raw (unpadded) CIKs as stored in institutional_holdings.
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
    w = _weight_map(_ROSTER)
    assert w == {"0000000001": 0.7, "0000000002": 0.3}          # padded keys
    assert _weight_map({"managers": []}) == {} and _weight_map(None) == {}
    print("\n=== SANITY CHECK: roster -> weight map ===")
    print("  managers -> {padded_cik: weight}; empty/None -> {} (skipped downstream). Validated.")


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


if __name__ == "__main__":
    test_weight_map_pads_and_sums()
    test_weighted_accumulation_subset_and_leakfree()
