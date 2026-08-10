"""
Tests for src/utils/tiingo_comparison.py: Tiingo dataCode kind-dispatch
(flow/flow_abs/instant/latest_q), bucket classification, the ratio-outlier
check (reusing analyze_history.detect_level_outliers on the our/Tiingo ratio
series), and the alignment-summary tolerance scoring.

Pure-synthetic, no network / no DB -- fetch_tiingo_statements' HTTP call is
monkeypatched at the `get_json` seam (same convention as test_polite_http.py),
and `context.store.load` is a plain stub returning a synthetic
fundamentals_history-shaped frame.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.utils import tiingo_comparison as tc


def _tiingo_entry(date: str, quarter: int, **data_codes) -> dict:
    return {
        "date": date, "year": pd.Timestamp(date).year, "quarter": quarter,
        "statementData": {
            "incomeStatement": [{"dataCode": k, "value": v} for k, v in data_codes.items()
                                if k in ("revenue", "costRev", "sga", "opinc", "ebt",
                                         "netIncComStock", "taxExp", "intexp", "eps",
                                         "epsDil", "shareswa", "shareswaDil")],
            "balanceSheet": [{"dataCode": k, "value": v} for k, v in data_codes.items()
                             if k in ("cashAndEq", "investmentsCurrent", "assetsCurrent",
                                      "totalAssets", "ppeq", "liabilitiesCurrent",
                                      "totalLiabilities", "debtCurrent", "debtNonCurrent",
                                      "equity", "sharesBasic")],
            "cashFlow": [{"dataCode": k, "value": v} for k, v in data_codes.items()
                        if k in ("ncfo", "capex", "depamor", "sbcomp")],
        },
    }


def _fake_context(hist: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(store=SimpleNamespace(
        load=lambda name, columns=None, where=None, **kw: hist))


def test_fetch_tiingo_statements_filters_quarterly_and_caches(monkeypatch, tmp_path):
    calls = []
    raw = [
        _tiingo_entry("2024-12-31", 0, revenue=1000.0),   # annual -- must be dropped
        _tiingo_entry("2024-09-30", 3, revenue=250.0),
        _tiingo_entry("2024-12-31", 4, revenue=260.0),
    ]
    monkeypatch.setattr(tc, "get_json", lambda *a, **k: (calls.append(1), raw)[1])

    out = tc.fetch_tiingo_statements("ZZZ", api_key="k", cache_dir=tmp_path)
    assert len(out) == 2
    assert all(e["quarter"] in (1, 2, 3, 4) for e in out)
    assert (tmp_path / "ZZZ.json").exists()

    # second call must hit the cache, not get_json again
    out2 = tc.fetch_tiingo_statements("ZZZ", api_key="k", cache_dir=tmp_path)
    assert len(out2) == 2
    assert len(calls) == 1


def test_fetch_tiingo_statements_returns_none_on_failure(monkeypatch):
    monkeypatch.setattr(tc, "get_json", lambda *a, **k: None)
    assert tc.fetch_tiingo_statements("NOTCOVERED", api_key="k") is None


def test_build_comparison_frame_dispatches_by_kind(monkeypatch):
    """Four trailing quarters of Tiingo `revenue` (100,110,120,130) TTM-sum to 460 --
    matched exactly against our TTM `totalRevenue` for a `flow` field. `capex` is
    `flow_abs`: Tiingo's -50 (outflow-negative) sums to -200 over the trailing 4,
    compared as abs(200)=200 against our positive-convention 200. `totalAssets` is
    `instant`: single-quarter compare, no summing. `goodwill` has no Tiingo
    dataCode at all (kind=None) -- must come back with tiingo_value=None and a note,
    never silently dropped from the output."""
    entries = [
        _tiingo_entry("2024-03-31", 1, revenue=100.0, totalAssets=900.0, capex=-50.0),
        _tiingo_entry("2024-06-30", 2, revenue=110.0, totalAssets=950.0, capex=-50.0),
        _tiingo_entry("2024-09-30", 3, revenue=120.0, totalAssets=980.0, capex=-50.0),
        _tiingo_entry("2024-12-31", 4, revenue=130.0, totalAssets=1000.0, capex=-50.0),
    ]
    monkeypatch.setattr(tc, "fetch_tiingo_statements", lambda ticker, **k: entries)

    hist = pd.DataFrame([{
        "ticker": "ZZZ", "fiscal_end": "2024-12-31",
        "totalRevenue": 460.0, "totalAssets": 1000.0, "capex": 200.0, "goodwill": 5000.0,
    }])
    out = tc.build_comparison_frame(_fake_context(hist), ["ZZZ"], api_key="k")

    rev = out[out["field"] == "totalRevenue"].iloc[0]
    assert rev["tiingo_value"] == 460.0 and rev["delta_pct"] == 0.0

    assets = out[out["field"] == "totalAssets"].iloc[0]
    assert assets["tiingo_value"] == 1000.0 and assets["delta_pct"] == 0.0

    capex = out[out["field"] == "capex"].iloc[0]
    assert capex["tiingo_value"] == 200.0 and capex["delta_pct"] == 0.0   # sign-flipped, then matched

    gw = out[out["field"] == "goodwill"].iloc[0]
    assert pd.isna(gw["tiingo_value"])
    assert "no Tiingo equivalent" in gw["note"]
    assert gw["bucket"] == "c"


def test_build_comparison_frame_skips_a_ticker_tiingo_has_no_data_for(monkeypatch):
    monkeypatch.setattr(tc, "fetch_tiingo_statements", lambda ticker, **k: None)
    hist = pd.DataFrame([{"ticker": "NOTCOVERED", "fiscal_end": "2024-12-31", "totalRevenue": 100.0}])
    out = tc.build_comparison_frame(_fake_context(hist), ["NOTCOVERED"], api_key="k")
    assert out.empty


def test_classify_bucket():
    assert tc.classify_bucket("AXP", "totalRevenue", "flow") == "b"          # confirmed override
    assert tc.classify_bucket("MSFT", "totalRevenue", "flow") == "a"         # same field, no override
    assert tc.classify_bucket("AAPL", "goodwill", None) == "c"               # no Tiingo equivalent
    # whole-field bucket b: Tiingo's lease-inclusive debt/PP&E convention applies to
    # ANY ticker, not just the ones it happened to be confirmed on
    assert tc.classify_bucket("JPM", "longTermDebt", "instant") == "b"
    assert tc.classify_bucket("JPM", "ppeNet", "instant") == "b"


def _comparison_rows(ticker, field, quarters, ours, tiingos) -> list[dict]:
    return [{"ticker": ticker, "field": field, "quarter": q, "our_value": o, "tiingo_value": t}
           for q, o, t in zip(quarters, ours, tiingos)]


def test_ratio_outlier_check_flags_a_spike_but_stays_quiet_on_a_stable_structural_gap():
    quarters = pd.date_range("2023-03-31", periods=6, freq="QE").date
    # AXP-style: a STABLE ~1.08x structural gap, must never be flagged
    stable_tiingo = [100.0, 110.0, 105.0, 115.0, 108.0, 112.0]
    stable_ours = [v * 1.08 for v in stable_tiingo]
    stable_rows = _comparison_rows("AXP", "totalRevenue", quarters, stable_ours, stable_tiingo)

    # a NEW discrepancy: five quarters at ~1.00x, one quarter spiking to 5x
    spike_tiingo = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    spike_ours = [100.0, 101.0, 99.0, 500.0, 100.0, 102.0]
    spike_rows = _comparison_rows("ZZZ", "totalRevenue", quarters, spike_ours, spike_tiingo)

    frame = pd.DataFrame(stable_rows + spike_rows)
    out = tc.ratio_outlier_check(frame, threshold=3.5)

    assert out[out["ticker"] == "AXP"].empty, "a stable structural gap must not be flagged"
    flagged = out[out["ticker"] == "ZZZ"]
    assert len(flagged) == 1
    # quarters[3] = 2023-12-31, calendar Q4 -- the spiked point
    assert flagged.iloc[0]["fiscal_year"] == 2023 and flagged.iloc[0]["fiscal_period"] == "Q4"


def test_alignment_summary_scores_bucket_a_only_within_tolerance():
    frame = pd.DataFrame([
        # bucket "a", flow tolerance (2%): within
        {"ticker": "ZZZ", "field": "totalRevenue", "kind": "flow", "bucket": "a", "delta_pct": 1.0},
        # bucket "a", instant tolerance (1%): outside
        {"ticker": "ZZZ", "field": "totalAssets", "kind": "instant", "bucket": "a", "delta_pct": 3.0},
        # bucket "b": must be excluded from the denominator entirely regardless of its delta
        {"ticker": "AXP", "field": "totalRevenue", "kind": "flow", "bucket": "b", "delta_pct": 50.0},
    ])
    out = tc.alignment_summary(frame)
    rev = out[out["field"] == "totalRevenue"].iloc[0]
    assert rev["n"] == 1 and rev["pct_within_tolerance"] == 100.0
    assets = out[out["field"] == "totalAssets"].iloc[0]
    assert assets["n"] == 1 and assets["pct_within_tolerance"] == 0.0
    overall = out[out["field"] == "__all__"].iloc[0]
    assert overall["n"] == 2   # the bucket-b row never enters the denominator

    print("\n=== SANITY CHECK: Tiingo comparison module ===")
    print("  kind dispatch (flow/flow_abs/instant) matches exactly on a clean synthetic fixture;")
    print("  a field with no Tiingo dataCode (goodwill) reports tiingo_value=None with a note,")
    print("  never silently dropped; a stable ~1.08x structural gap (AXP-style) stays quiet")
    print("  under ratio_outlier_check while a genuine new spike is flagged; alignment_summary")
    print("  scores bucket-a rows only, excluding bucket-b overrides from the denominator.")
    print("  Validated.")
