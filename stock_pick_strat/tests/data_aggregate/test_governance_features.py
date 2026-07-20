"""
Governance / executive-pay features from the DEF 14A LLM archive
(src/data_aggregate/utils/governance_features.py).

Synthetic proxy history (annual) + a quarterly fundamentals history, to prove:
  * CEO total-comp YoY growth and the pay-vs-revenue-growth MISALIGNMENT signal,
  * board/pay level fields flow through, and
  * the panel is point-in-time and empty when the archive is absent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.governance_features import (
    _governance_fields,
    build_governance_feature_panel,
)


def _def14a() -> pd.DataFrame:
    rows = []
    # (ticker, base CEO pay, CEO-since year, pct female directors) — the last two vary
    # cross-sectionally so ceo_tenure and pct_female_directors have real peer dispersion.
    specs = [("AAA", 10e6, 2010, 0.40), ("BBB", 12e6, 2018, 0.20),
             ("CCC", 8e6, 2005, 0.50), ("DDD", 15e6, 2022, 0.30)]
    for tkr, pay0, since_yr, fem in specs:
        for i, yr in enumerate((2023, 2024, 2025)):
            rows.append({
                "ticker": tkr, "as_of": pd.Timestamp(f"{yr}-04-01"),
                "ceo_total_comp": pay0 * (1.20 ** i),        # +20%/yr
                "ceo_pay_ratio": 200 + 50 * i,
                "ceo_equity_pay_pct": 0.70,
                "pct_independent_directors": 0.85,
                "pct_female_directors": fem,
                "board_size": 10,
                "avg_board_tenure": 7.0,
                "say_on_pay_support_pct": 0.92,
                "insider_ownership_pct": 0.01,
                "ceo_is_founder": 1.0 if tkr in ("AAA", "CCC") else 0.0,
                "ceo_since_year": since_yr,
            })
    return pd.DataFrame(rows)


def _fundamentals() -> pd.DataFrame:
    rows = []
    for tkr in ("AAA", "BBB", "CCC", "DDD"):
        for i, q in enumerate(pd.date_range("2022-12-31", periods=14, freq="QE")):
            rows.append({"ticker": tkr, "as_of": q, "totalRevenue": 1000 * (1.03 ** i)})  # ~12%/yr
    return pd.DataFrame(rows)


def test_governance_fields_pay_growth_and_misalignment():
    idx = pd.date_range("2023-01-02", "2026-07-01", freq="B")
    F = _governance_fields(_def14a(), idx, _fundamentals())

    assert "ceo_pay_growth" in F and "ceo_pay_vs_revenue_growth" in F
    for name in ("ceo_pay_ratio", "pct_independent_directors", "pct_female_directors",
                 "say_on_pay_support", "avg_board_tenure", "insider_ownership_pct",
                 "founder_ceo"):
        assert name in F, f"missing level field {name}"
    # founder-CEO flag surfaced from ceo_is_founder (AAA/CCC founder-led -> 1)
    assert F["founder_ceo"]["AAA"].dropna().iloc[-1] == 1.0
    assert F["founder_ceo"]["BBB"].dropna().iloc[-1] == 0.0

    # CEO tenure accrues by CALENDAR year (not a stale as_of snapshot): AAA CEO since
    # 2010 -> 15y on a 2025 date and 16y on a 2026 date; CCC (since 2005) outranks DDD (2022).
    assert "ceo_tenure" in F
    ten = F["ceo_tenure"]
    aaa_2025 = ten.loc[ten.index.year == 2025, "AAA"].dropna()
    aaa_2026 = ten.loc[ten.index.year == 2026, "AAA"].dropna()
    assert aaa_2025.iloc[-1] == pytest.approx(2025 - 2010)   # 15
    assert aaa_2026.iloc[-1] == pytest.approx(2026 - 2010)   # 16 -> grows with the calendar
    last = idx[-1]
    assert ten.loc[last, "CCC"] > ten.loc[last, "DDD"]       # 2005 vs 2022 start

    # CEO pay grew ~20%/yr; the latest observed pay_growth should be ~0.20
    pay_g = F["ceo_pay_growth"]["AAA"].dropna()
    assert pay_g.iloc[-1] == pytest.approx(0.20, abs=1e-6)
    # misalignment = pay growth (~20%) - revenue TTM growth (~12%) -> clearly positive
    mis = F["ceo_pay_vs_revenue_growth"]["AAA"].dropna()
    assert mis.iloc[-1] > 0.05

    print("\n=== SANITY CHECK: governance pay dynamics + CEO tenure ===")
    print(f"  ceo_pay_growth(last)={pay_g.iloc[-1]:.3f} (~0.20); "
          f"pay_vs_revenue_growth(last)={mis.iloc[-1]:.3f} (>0 = pay outpacing revenue).")
    print(f"  ceo_tenure AAA(since 2010): {aaa_2025.iloc[-1]:.0f}y in 2025 -> {aaa_2026.iloc[-1]:.0f}y in 2026 "
          f"(accrues by calendar year); CCC {ten.loc[last, 'CCC']:.0f}y > DDD {ten.loc[last, 'DDD']:.0f}y. Validated.")


def test_governance_panel_and_empty_guard():
    idx = pd.date_range("2023-01-02", "2026-07-01", freq="B")
    peers = {t: {p: 1.0 for p in ("AAA", "BBB", "CCC", "DDD") if p != t}
             for t in ("AAA", "BBB", "CCC", "DDD")}
    panel = build_governance_feature_panel(_def14a(), peers, idx, fundamentals_history=_fundamentals())
    cols = [c for c in panel.columns if c not in ("date", "ticker")]
    assert not panel.empty
    # both the peer-relative and the cross-sectional views of the headline signal exist
    assert "f_ceo_pay_vs_revenue_growth_vs_peers" in cols
    assert "f_ceo_pay_ratio_xs" in cols and "f_pct_independent_directors_xs" in cols
    # the two newly-wired governance features (both used peer-relative in modelling.yml)
    assert "f_ceo_tenure_vs_peers" in cols
    assert "f_pct_female_directors_vs_peers" in cols

    # no archive -> empty (optional-source semantics, never raises)
    empty = build_governance_feature_panel(None, peers, idx)
    assert list(empty.columns) == ["date", "ticker"] and empty.empty

    print("\n=== SANITY CHECK: governance panel ===")
    print(f"  built {len(cols)} governance features incl pay-vs-revenue misalignment, "
          f"f_ceo_tenure_vs_peers + f_pct_female_directors_vs_peers (both now in modelling.yml); "
          f"None archive -> empty panel (skipped, no crash). Validated.")
