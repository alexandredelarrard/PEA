"""
Tests for the Google Trends extractor (curl_cffi engine + weekly-15y chunk/stitch).

test_weekly_windows_cover_15y_weekly_safe — windows span 15y, each <=4y (stay weekly), overlap
test_stitch_recovers_underlying_shape     — overlapping renormalized chunks -> one continuous 0-100 series
test_stitch_handles_gap_without_overlap   — non-overlapping chunks still concatenate (carry level)
test_scale_to_reference_aligns_on_overlap — appended window is levelled onto stored history
test_client_parses_timeseries_and_drops_partial — explore->widgetdata parsing, isPartial dropped
test_client_raises_on_429                 — HTTP 429 -> TrendsRateLimited (call_with_retries backs off)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data_extract.utils.behavioral import fetch_google_trends as gt
from src.data_extract.utils.common.rate_limit import is_rate_limited


# --------------------------------------------------------------------------- #
# Pure helpers                                                                 #
# --------------------------------------------------------------------------- #
def test_weekly_windows_cover_15y_weekly_safe():
    end = pd.Timestamp("2026-01-04")
    wins = gt._weekly_windows(15, end=end)
    span_years = (wins[-1][1] - wins[0][0]).days / 365.25
    assert span_years >= 14.9, f"windows span only {span_years:.1f}y"
    # every window must stay <= ~5y so Trends returns WEEKLY (not monthly) data
    assert all((e - s).days / 365.25 <= 4.05 for s, e in wins)
    # consecutive windows must overlap (so chunks can be chain-scaled)
    assert all(wins[i + 1][0] < wins[i][1] for i in range(len(wins) - 1))

    print("\n=== SANITY CHECK: weekly windows ===")
    print(f"  {len(wins)} windows, each <=4y, overlapping, spanning {span_years:.1f}y "
          f"({wins[0][0].date()}..{wins[-1][1].date()}). Validated.")


def _renorm_chunk(idx, true, a, b, scale_max):
    m = (idx >= a) & (idx < b)
    s = true[m].astype(float)
    s = s / s.max() * scale_max
    return pd.DataFrame({"date": idx[m], "search_interest": np.round(s, 2)})


def test_stitch_recovers_underlying_shape():
    idx = pd.date_range("2011-01-02", "2026-01-04", freq="W")
    true = 40 + 30 * np.sin(np.arange(len(idx)) / 25) + np.linspace(0, 25, len(idx))
    chunks = [_renorm_chunk(idx, true, "2011-01-01", "2016-06-01", 100),
              _renorm_chunk(idx, true, "2015-06-01", "2021-06-01", 80),   # different local scale
              _renorm_chunk(idx, true, "2020-06-01", "2026-06-01", 100)]
    st = gt._stitch_chunks([chunks[2], chunks[0], chunks[1]])             # unsorted input
    aligned = pd.Series(true, index=idx).reindex(st["date"].values)
    corr = np.corrcoef(aligned.values, st["search_interest"].values)[0, 1]

    assert corr > 0.99, f"stitch did not recover the shape (corr={corr:.3f})"
    assert st["search_interest"].min() >= 0 and abs(st["search_interest"].max() - 100) < 0.01
    steps = st["date"].diff().dt.days.dropna()
    assert (steps == 7).all(), "stitched series is not continuous weekly"

    print("\n=== SANITY CHECK: stitch overlapping weekly chunks ===")
    print(f"  3 chunks (local scales 100/80/100) -> {len(st)} continuous weekly pts, "
          f"corr vs true={corr:.4f}, range [0,100]. Validated.")


def test_stitch_handles_gap_without_overlap():
    a = pd.DataFrame({"date": pd.date_range("2011-01-02", periods=20, freq="W"),
                      "search_interest": np.linspace(10, 100, 20)})
    b = pd.DataFrame({"date": pd.date_range("2015-01-04", periods=20, freq="W"),
                      "search_interest": np.linspace(50, 100, 20)})
    st = gt._stitch_chunks([a, b])
    assert len(st) == 40 and st["search_interest"].max() == pytest.approx(100.0)
    print("\n=== SANITY CHECK: stitch with no overlap ===")
    print(f"  disjoint chunks concatenated to {len(st)} pts (level carried). Validated.")


def test_scale_to_reference_aligns_on_overlap():
    idx = pd.date_range("2020-01-05", periods=140, freq="W")
    true = 30 + np.linspace(0, 60, 140)
    ref = pd.DataFrame({"date": idx[:100],
                        "search_interest": np.round(true[:100] / true[:100].max() * 100, 2)})
    new = pd.DataFrame({"date": idx[80:140],
                        "search_interest": np.round(true[80:140] / true[80:140].max() * 100, 2)})
    scaled = gt._scale_to_reference(new, ref)
    ov = scaled[scaled["date"].isin(ref["date"])].set_index("date")["search_interest"]
    rv = ref.set_index("date")["search_interest"].reindex(ov.index)
    ratio = (ov / rv).mean()
    assert abs(ratio - 1.0) < 0.05, f"overlap not aligned (ratio={ratio:.3f})"
    print("\n=== SANITY CHECK: scale_to_reference ===")
    print(f"  appended window levelled onto stored history (overlap ratio {ratio:.3f}). Validated.")


def test_append_and_renormalize_full_trend():
    # A 3y weekly truth with a LATE spike well above the prior historical max.
    idx = pd.date_range("2021-01-03", periods=156, freq="W")
    true = 20 + np.linspace(0, 50, 156)
    true[150:] += np.array([10, 20, 35, 55, 80, 110])[: len(true[150:])]   # spike in the new tail
    # stored history = first 2y, normalised 0-100 within itself (peak at its own end)
    ref_true = true[:104]
    ref = pd.DataFrame({"date": idx[:104],
                        "search_interest": np.round(ref_true / ref_true.max() * 100, 2)})
    # a fresh OVERLAPPING window (weeks 80..156), independently normalised 0-100 within itself
    rec_true = true[80:]
    recent = pd.DataFrame({"date": idx[80:],
                           "search_interest": np.round(rec_true / rec_true.max() * 100, 2)})

    full = gt._append_and_renormalize(ref, recent)

    steps = full["date"].diff().dt.days.dropna()
    assert (steps == 7).all(), "reconciled series is not continuous weekly"
    assert len(full) == 156, "full span (history + appended weeks) not preserved"
    # ONE coherent 0-100 scale over the whole trend; the peak sits in the NEW (post-history) portion
    assert abs(full["search_interest"].max() - 100) < 0.01
    assert full.loc[full["search_interest"].idxmax(), "date"] > ref["date"].max()
    # history shape untouched: full/ref over the stored dates is a single constant factor
    ov = full[full["date"].isin(ref["date"])].set_index("date")["search_interest"]
    rv = ref.set_index("date")["search_interest"].reindex(ov.index)
    ratio = (ov / rv).replace([np.inf, -np.inf], np.nan).dropna()
    # a single rescale factor -> ratio is constant up to 2-decimal storage rounding
    assert ratio.std() / ratio.mean() < 1e-3, "history was not rescaled by a single factor (patchwork!)"
    # the whole series still tracks the underlying truth
    aligned = pd.Series(true, index=idx).reindex(full["date"].values)
    corr = np.corrcoef(aligned.values, full["search_interest"].values)[0, 1]
    assert corr > 0.99

    print("\n=== SANITY CHECK: full-trend reconciliation (append + renormalise) ===")
    print(f"  history {len(ref)}w + appended {len(full) - len(ref)}w -> {len(full)}w continuous weekly")
    print(f"  new spike renormalised the WHOLE series to one 0-100 scale (peak={full['search_interest'].max():.1f} "
          f"in the new tail), history rescaled by a single factor {float((ov/rv).mean()):.4f}, "
          f"corr vs truth={corr:.4f}. Validated.")


def test_append_noop_when_nothing_new():
    idx = pd.date_range("2022-01-02", periods=60, freq="W")
    ref = pd.DataFrame({"date": idx, "search_interest": np.linspace(10, 100, 60).round(2)})
    recent = ref.iloc[-20:].copy()                          # entirely within stored history
    out = gt._append_and_renormalize(ref, recent)
    pd.testing.assert_frame_equal(out.reset_index(drop=True), ref.reset_index(drop=True))
    print("\n=== SANITY CHECK: reconcile is a no-op with nothing new ===")
    print("  recent window inside stored history -> history returned unchanged (no rewrite). Validated.")


# --------------------------------------------------------------------------- #
# Client parsing (mocked curl_cffi session — no network)                       #
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, status, text):
        self.status_code = status
        self.text = text


def _make_fake_module(explore_resp, widget_resp):
    guard = ")]}',\n"

    class _FakeSession:
        def __init__(self, **kw):
            self.headers = {}

        def get(self, url, params=None):
            if url == gt.GOOGLE_TRENDS_HOME_URL:
                return _FakeResp(200, "ok")
            if url == gt.GOOGLE_TRENDS_EXPLORE_URL:
                return explore_resp
            if url == gt.GOOGLE_TRENDS_MULTILINE_URL:
                return widget_resp
            return _FakeResp(404, "")

    class _FakeReq:
        Session = _FakeSession

    return _FakeReq, guard


def test_client_parses_timeseries_and_drops_partial(monkeypatch):
    guard = ")]}',\n"
    explore = _FakeResp(200, guard + json.dumps(
        {"widgets": [{"id": "TIMESERIES", "token": "TOK", "request": {"x": 1}}]}))
    # 1609459200 = 2021-01-01, 1610064000 = 2021-01-08 (partial -> dropped)
    widget = _FakeResp(200, guard + json.dumps({"default": {"timelineData": [
        {"time": "1609459200", "value": [50]},
        {"time": "1610064000", "value": [60], "isPartial": True}]}}))
    fake_req, _ = _make_fake_module(explore, widget)
    monkeypatch.setattr(gt, "_cffi_requests", fake_req)

    client = gt._TrendsClient(verify=False)
    df = client.interest_over_time("Apple Inc", "today 5-y")

    assert list(df.columns) == ["date", "search_interest"]
    assert len(df) == 1                                    # partial bucket dropped
    assert df["date"].iloc[0] == pd.Timestamp("2021-01-01")
    assert df["search_interest"].iloc[0] == 50.0

    print("\n=== SANITY CHECK: Trends client parsing ===")
    print(f"  explore->widgetdata parsed to {len(df)} weekly pt (isPartial dropped); "
          f"{df['date'].iloc[0].date()}={df['search_interest'].iloc[0]:.0f}. Validated.")


def test_client_raises_on_429(monkeypatch):
    explore_429 = _FakeResp(429, "rate limited")
    fake_req, _ = _make_fake_module(explore_429, _FakeResp(200, ""))
    monkeypatch.setattr(gt, "_cffi_requests", fake_req)

    client = gt._TrendsClient(verify=False)
    with pytest.raises(gt.TrendsRateLimited) as ei:
        client.interest_over_time("Apple Inc", "today 5-y")
    assert is_rate_limited(ei.value), "429 not recognised as rate-limit by call_with_retries"

    print("\n=== SANITY CHECK: Trends 429 handling ===")
    print("  HTTP 429 -> TrendsRateLimited, recognised by call_with_retries (backs off). Validated.")
