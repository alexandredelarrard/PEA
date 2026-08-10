"""
Tests for src/utils/fundamentals_audit.py: the Tiingo -> Yahoo -> uncovered fallback
chain (`run_universe_audit`) and the ranked-findings consolidation
(`build_ranked_findings`).

Pure-synthetic, no network / no DB -- `tiingo_comparison.run_tiingo_audit` and
`yahoo_comparison.run_non_tiingo_audit` are monkeypatched directly (this module only
composes their outputs, it never re-tests their internals).
"""
from __future__ import annotations

import pandas as pd

from src.utils import fundamentals_audit as fa


def _empty_audit_result(value_col: str) -> dict[str, pd.DataFrame]:
    return {"comparison": pd.DataFrame(columns=["ticker", "field", "quarter", "our_value",
                                                value_col, "delta_pct", "kind", "bucket", "note"]),
           "ratio_outliers": pd.DataFrame(),
           "alignment": pd.DataFrame()}


def test_run_universe_audit_routes_uncovered_tickers_through_the_fallback_chain(monkeypatch):
    """AAA is covered by Tiingo, BBB only by Yahoo, CCC by neither -- must end up
    uncovered/logged, never dropped silently and never raising."""
    def fake_tiingo(context, tickers, **kw):
        result = _empty_audit_result("tiingo_value")
        result["comparison"] = pd.DataFrame([
            {"ticker": "AAA", "field": "totalRevenue", "quarter": "2024-12-31",
             "our_value": 100.0, "tiingo_value": 100.0, "delta_pct": 0.0,
             "kind": "flow", "bucket": "a", "note": ""},
        ])
        return result

    def fake_yahoo(context, tickers, **kw):
        assert set(tickers) == {"BBB", "CCC"}   # AAA must NOT be re-tried against Yahoo
        result = _empty_audit_result("yahoo_value")
        result["comparison"] = pd.DataFrame([
            {"ticker": "BBB", "field": "totalRevenue", "quarter": "2024-12-31",
             "our_value": 50.0, "yahoo_value": 50.0, "delta_pct": 0.0,
             "kind": "flow", "bucket": "a", "note": ""},
        ])
        return result

    monkeypatch.setattr(fa.tiingo_comparison, "run_tiingo_audit", fake_tiingo)
    monkeypatch.setattr(fa.yahoo_comparison, "run_non_tiingo_audit", fake_yahoo)

    out = fa.run_universe_audit(context=None, tickers=["AAA", "BBB", "CCC"], api_key="k")
    assert set(out["tiingo"]["comparison"]["ticker"]) == {"AAA"}
    assert set(out["yahoo"]["comparison"]["ticker"]) == {"BBB"}
    assert out["no_external_validation"]["ticker"].tolist() == ["CCC"]


def test_build_ranked_findings_ranks_cross_source_agreement_above_single_source_severity():
    """A cell flagged by BOTH Tiingo (bucket-a miss) and the internal outlier check
    (agreement_count=2) must outrank a single-source finding even if that other
    finding's own raw severity is numerically larger -- agreement is the primary key,
    not severity."""
    tiingo_cmp = pd.DataFrame([
        {"ticker": "ZZZ", "field": "netIncome", "quarter": "2024-12-31",
         "our_value": 100.0, "tiingo_value": 50.0, "delta_pct": 100.0,
         "kind": "flow", "bucket": "a", "note": ""},
        {"ticker": "YYY", "field": "netIncome", "quarter": "2024-12-31",
         "our_value": 100.0, "tiingo_value": 10.0, "delta_pct": 900.0,   # bigger raw severity
         "kind": "flow", "bucket": "a", "note": ""},
    ])
    internal_outliers = pd.DataFrame([
        {"ticker": "ZZZ", "field": "netIncome", "fiscal_year": 2024, "fiscal_period": "Q4",
         "level_z_score": 5.0, "is_yoy_outlier": False, "is_level_outlier": True},
    ])
    empty = pd.DataFrame()

    ranked = fa.build_ranked_findings(
        tiingo_comparison_df=tiingo_cmp, tiingo_ratio_outliers=empty,
        yahoo_comparison_df=empty, yahoo_ratio_outliers=empty,
        tag_breaks=empty, tag_misalignment=empty, internal_outliers=internal_outliers,
    )
    assert ranked.iloc[0][["ticker", "field"]].tolist() == ["ZZZ", "netIncome"]
    assert ranked.iloc[0]["agreement_count"] == 2
    yyy_rows = ranked[ranked["ticker"] == "YYY"]
    assert (yyy_rows["agreement_count"] == 1).all()
    assert ranked.iloc[0]["priority_score"] > yyy_rows["priority_score"].max()


def test_build_ranked_findings_handles_all_empty_sources():
    empty = pd.DataFrame()
    ranked = fa.build_ranked_findings(
        tiingo_comparison_df=empty, tiingo_ratio_outliers=empty,
        yahoo_comparison_df=empty, yahoo_ratio_outliers=empty,
        tag_breaks=empty, tag_misalignment=empty, internal_outliers=empty,
    )
    assert ranked.empty
    assert list(ranked.columns) == fa.FINDINGS_COLS

    print("\n=== SANITY CHECK: fundamentals_audit module ===")
    print("  fallback chain routes each ticker through Tiingo, then Yahoo for whatever")
    print("  Tiingo left uncovered, then logs anything neither source has (never drops a")
    print("  ticker silently); ranked findings put a cell flagged by 2 independent sources")
    print("  above a bigger single-source delta, and an all-empty input degrades to an")
    print("  empty, correctly-shaped frame instead of raising.")
    print("  Validated.")
