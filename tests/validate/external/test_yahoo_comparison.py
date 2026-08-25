"""
Tests for src/validate/external/yahoo_comparison.py: yfinance dataframe kind-dispatch
(flow/flow_abs/instant/latest_q), bucket classification, the ratio-outlier check
(reusing outliers.detect_level_outliers on the our/yahoo ratio series), and the
alignment-summary tolerance scoring.

Pure-synthetic, no network / no DB -- `yf.Ticker` is monkeypatched at the module seam
(same convention as test_tiingo_comparison.py's `get_json` monkeypatch), and
`context.store.load` is a plain stub returning a synthetic fundamentals_history-shaped
frame.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.validate.external import yahoo_comparison as yc


def _quarterly_df(rows: dict[str, list[float]], dates: list[str]) -> pd.DataFrame:
    """yfinance shape: rows = statement line labels, columns = quarter-end dates."""
    return pd.DataFrame(rows, index=pd.to_datetime(dates)).T


def _fake_ticker(income: pd.DataFrame, balance: pd.DataFrame | None = None,
                 cashflow: pd.DataFrame | None = None):
    return SimpleNamespace(
        quarterly_income_stmt=income,
        quarterly_balance_sheet=balance if balance is not None else pd.DataFrame(),
        quarterly_cashflow=cashflow if cashflow is not None else pd.DataFrame(),
    )


def _fake_context(hist: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(store=SimpleNamespace(
        load=lambda name, columns=None, where=None, **kw: hist))


def test_fetch_yahoo_statements_caches_and_returns_none_on_failure(monkeypatch, tmp_path):
    dates = ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]
    income = _quarterly_df({"Total Revenue": [100.0, 110.0, 120.0, 130.0]}, dates)
    calls = []

    def fake_ticker_ctor(ticker):
        calls.append(ticker)
        return _fake_ticker(income)

    monkeypatch.setattr(yc.yf, "Ticker", fake_ticker_ctor)

    out = yc.fetch_yahoo_statements("ZZZ", cache_dir=tmp_path)
    assert out is not None and "Total Revenue" in out["income"].index
    assert (tmp_path / "ZZZ_income.parquet").exists()

    # second call must hit the cache, not construct yf.Ticker again
    out2 = yc.fetch_yahoo_statements("ZZZ", cache_dir=tmp_path)
    assert out2 is not None
    assert len(calls) == 1


def test_fetch_yahoo_statements_returns_none_on_empty_or_error(monkeypatch):
    monkeypatch.setattr(yc.yf, "Ticker", lambda t: _fake_ticker(pd.DataFrame()))
    assert yc.fetch_yahoo_statements("NOTCOVERED") is None

    def raises(ticker):
        raise ValueError("delisted")
    monkeypatch.setattr(yc.yf, "Ticker", raises)
    assert yc.fetch_yahoo_statements("BADTICKER") is None


def test_build_comparison_frame_dispatches_by_kind(monkeypatch):
    """Four trailing quarters of Yahoo `Total Revenue` (100,110,120,130) TTM-sum to 460
    -- matched exactly against our TTM `totalRevenue` for a `flow` field. `Capital
    Expenditure` is `flow_abs`: Yahoo's -50 (outflow-negative) sums to -200 over the
    trailing 4, compared as abs(200)=200 against our positive-convention 200. `Total
    Assets` is `instant`: single-quarter compare, no summing. `dividendsPerShare` has no
    Yahoo row at all (kind=None) -- must come back with yahoo_value=None and a note."""
    dates = ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]
    income = _quarterly_df({"Total Revenue": [100.0, 110.0, 120.0, 130.0]}, dates)
    balance = _quarterly_df({"Total Assets": [900.0, 950.0, 980.0, 1000.0]}, dates)
    cashflow = _quarterly_df({"Capital Expenditure": [-50.0, -50.0, -50.0, -50.0]}, dates)
    monkeypatch.setattr(yc, "fetch_yahoo_statements",
                        lambda ticker, **k: {"income": income, "balance": balance,
                                            "cashflow": cashflow})

    hist = pd.DataFrame([{
        "ticker": "ZZZ", "fiscal_end": "2024-12-31",
        "totalRevenue": 460.0, "totalAssets": 1000.0, "capex": 200.0,
        "dividendsPerShare": 0.5,
    }])
    out = yc.build_comparison_frame(_fake_context(hist), ["ZZZ"])

    rev = out[out["field"] == "totalRevenue"].iloc[0]
    assert rev["yahoo_value"] == 460.0 and rev["delta_pct"] == 0.0

    assets = out[out["field"] == "totalAssets"].iloc[0]
    assert assets["yahoo_value"] == 1000.0 and assets["delta_pct"] == 0.0

    capex = out[out["field"] == "capex"].iloc[0]
    assert capex["yahoo_value"] == 200.0 and capex["delta_pct"] == 0.0

    dps = out[out["field"] == "dividendsPerShare"].iloc[0]
    assert pd.isna(dps["yahoo_value"])
    assert "no Yahoo equivalent" in dps["note"]
    assert dps["bucket"] == "c"


def test_build_comparison_frame_skips_a_ticker_yahoo_has_no_data_for(monkeypatch):
    monkeypatch.setattr(yc, "fetch_yahoo_statements", lambda ticker, **k: None)
    hist = pd.DataFrame([{"ticker": "NOTCOVERED", "fiscal_end": "2024-12-31", "totalRevenue": 100.0}])
    out = yc.build_comparison_frame(_fake_context(hist), ["NOTCOVERED"])
    assert out.empty


def test_classify_bucket():
    assert yc.classify_bucket("JPM", "totalLiabilities", "instant") == "b"  # naming-implied gap
    assert yc.classify_bucket("MSFT", "totalRevenue", "flow") == "a"
    assert yc.classify_bucket("AAPL", "dividendsPerShare", None) == "c"


def _comparison_rows(ticker, field, quarters, ours, yahoos) -> list[dict]:
    return [{"ticker": ticker, "field": field, "quarter": q, "our_value": o, "yahoo_value": y}
           for q, o, y in zip(quarters, ours, yahoos)]


def test_ratio_outlier_check_flags_a_spike_but_stays_quiet_on_a_stable_structural_gap():
    quarters = pd.date_range("2023-03-31", periods=6, freq="QE").date
    stable_yahoo = [100.0, 110.0, 105.0, 115.0, 108.0, 112.0]
    stable_ours = [v * 1.08 for v in stable_yahoo]
    stable_rows = _comparison_rows("AXP", "totalRevenue", quarters, stable_ours, stable_yahoo)

    spike_yahoo = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    spike_ours = [100.0, 101.0, 99.0, 500.0, 100.0, 102.0]
    spike_rows = _comparison_rows("ZZZ", "totalRevenue", quarters, spike_ours, spike_yahoo)

    frame = pd.DataFrame(stable_rows + spike_rows)
    out = yc.ratio_outlier_check(frame, threshold=3.5)

    assert out[out["ticker"] == "AXP"].empty, "a stable structural gap must not be flagged"
    flagged = out[out["ticker"] == "ZZZ"]
    # TWO rows, and that is the post-decision-60 contract: the kernel scores the STEP, so a
    # one-quarter spike is anomalous going in AND coming back out. Before the log-change fix
    # this returned one row because it scored the LEVEL -- which is the same rule that flagged
    # the whole recent era of any growing company. One defect, its two boundaries.
    assert len(flagged) == 2, f"expected the spike and its reversion, got {len(flagged)}"
    periods = list(zip(flagged["fiscal_year"], flagged["fiscal_period"]))
    # quarters[3] = 2023-12-31 (the spike) and quarters[4] = 2024-03-31 (the reversion)
    assert periods == [(2023, "Q4"), (2024, "Q1")], periods
    print(f"\n  ratio spike 1.0x -> 5.0x -> 1.0x flagged at {periods}; "
          f"the stable 1.08x AXP gap: {len(out[out['ticker'] == 'AXP'])} findings")
    print("  SANITY: a STABLE structural difference is not a defect and a one-quarter jump "
          "is -- and the jump owns both of its edges.")


def test_alignment_summary_scores_bucket_a_only_within_tolerance():
    frame = pd.DataFrame([
        {"ticker": "ZZZ", "field": "totalRevenue", "kind": "flow", "bucket": "a", "delta_pct": 1.0},
        {"ticker": "ZZZ", "field": "totalAssets", "kind": "instant", "bucket": "a", "delta_pct": 3.0},
        {"ticker": "JPM", "field": "totalLiabilities", "kind": "instant", "bucket": "b", "delta_pct": 50.0},
    ])
    out = yc.alignment_summary(frame)
    rev = out[out["field"] == "totalRevenue"].iloc[0]
    assert rev["n"] == 1 and rev["pct_within_tolerance"] == 100.0
    assets = out[out["field"] == "totalAssets"].iloc[0]
    assert assets["n"] == 1 and assets["pct_within_tolerance"] == 0.0
    overall = out[out["field"] == "__all__"].iloc[0]
    assert overall["n"] == 2   # the bucket-b row never enters the denominator

    print("\n=== SANITY CHECK: Yahoo comparison module ===")
    print("  kind dispatch (flow/flow_abs/instant) matches exactly on a clean synthetic fixture;")
    print("  a field with no Yahoo row (dividendsPerShare) reports yahoo_value=None with a note,")
    print("  never silently dropped; a stable ~1.08x structural gap (AXP-style) stays quiet")
    print("  under ratio_outlier_check while a genuine new spike is flagged; alignment_summary")
    print("  scores bucket-a rows only, excluding bucket-b overrides from the denominator.")
    print("  Validated.")
