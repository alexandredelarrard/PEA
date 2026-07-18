"""Forward earnings-yield feature (1 / forward P/E) from the accruing snapshot."""
from __future__ import annotations

import pandas as pd

from src.data_aggregate.utils.forward_features import build_forward_valuation_panel


def test_forward_earnings_yield_ranks_cheaper_higher():
    idx = pd.bdate_range("2026-01-05", periods=5)
    snap = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "as_of": [idx[0]] * 3,
        "forwardPE": [10.0, 20.0, 40.0],   # A cheapest -> highest fwd E/P (0.10)
        "marketCap": [1e9, 1e9, 1e9],
    })
    peers = {"A": {"B": 0.5, "C": 0.5}, "B": {"A": 0.5, "C": 0.5}, "C": {"A": 0.5, "B": 0.5}}

    panel = build_forward_valuation_panel(snap, peers, idx)

    assert not panel.empty
    assert {"f_forward_earnings_yield_vs_peers", "f_forward_earnings_yield_xs"} <= set(panel.columns)
    last = panel[panel["date"] == idx[-1]].set_index("ticker")
    # cheaper (A, PE 10, yield 0.10) ranks above expensive (C, PE 40, yield 0.025)
    assert last.loc["A", "f_forward_earnings_yield_xs"] > last.loc["C", "f_forward_earnings_yield_xs"]

    print("\n=== SANITY CHECK: forward earnings yield feature ===")
    print(f"  forwardPE A=10 B=20 C=40 -> fwd E/P 0.100/0.050/0.025")
    print(f"  xs percentile A={last.loc['A','f_forward_earnings_yield_xs']:.2f} > "
          f"C={last.loc['C','f_forward_earnings_yield_xs']:.2f} (cheaper ranks higher). Validated.")


def test_forward_valuation_empty_without_snapshot():
    idx = pd.bdate_range("2026-01-05", periods=3)
    assert build_forward_valuation_panel(None, {}, idx).empty
    assert build_forward_valuation_panel(pd.DataFrame(), {}, idx).empty
    print("\n=== SANITY CHECK: forward feature no-op ===")
    print("  empty/absent snapshot -> empty panel (feature skipped). Validated.")
