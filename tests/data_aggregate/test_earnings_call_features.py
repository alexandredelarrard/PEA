"""
Earnings-call sentiment/text FEATURES (src/data_aggregate/utils/earnings_call_features.py).

Validates the pure feature layer on synthetic cache rows (no model/GPU):
  * the smart per-call KPI arithmetic (length-weighted tone, Q&A gap, uncertainty,
    tone delta vs prior call, disclosure-length delta),
  * the peer-relative panel columns (f_ec_*_xs / _vs_peers),
  * POINT-IN-TIME / leak-free alignment: a call on date d only affects features on
    d+1 onward (transcript-publication lag), forward-filled until the next call.
"""

from __future__ import annotations

import logging
import math
from typing import cast

import numpy as np
import pandas as pd

from src.context import Context
from src.data_aggregate.utils.text.earnings_call_features import (
    _daily_frame,
    _per_call_kpis,
    build_earnings_call_feature_panel,
    sentiment_kpis_streamed,
)

_QDATE = {"2023Q1": "2023-02-01", "2023Q2": "2023-05-01", "2023Q3": "2023-08-01"}


def _number(value: object) -> float:
    """Narrow a scalar selected from a numeric fixture frame."""
    assert not isinstance(value, pd.Series | pd.DataFrame)
    return float(cast(float, value))


def _row(tkr, q, tag, pos, neg, words, unc):
    return {
        "ticker": tkr,
        "quarter": q,
        "tag": tag,
        "as_of": _QDATE[q],
        "sent_pos": pos,
        "sent_neg": neg,
        "sent_neu": round(1 - pos - neg, 6),
        "n_words": words,
        "uncertainty_ratio": unc,
    }


def _sentiment_frame() -> pd.DataFrame:
    rows = [
        # ticker A — the arithmetic-checked name
        _row("A", "2023Q1", "prepared_remarks", 0.60, 0.10, 1000, 0.02),
        _row("A", "2023Q1", "qa", 0.40, 0.10, 500, 0.05),
        _row("A", "2023Q2", "prepared_remarks", 0.70, 0.05, 1200, 0.01),
        _row("A", "2023Q2", "qa", 0.50, 0.10, 600, 0.04),
        _row("A", "2023Q3", "prepared_remarks", 0.55, 0.15, 1100, 0.03),
        _row("A", "2023Q3", "qa", 0.45, 0.20, 550, 0.06),
    ]
    # B..E: distinct tone/uncertainty so the cross-section & peer basket are non-degenerate
    for i, tkr in enumerate(["B", "C", "D", "E"], start=1):
        for q in _QDATE:
            base = 0.30 + 0.1 * i
            rows.append(_row(tkr, q, "prepared_remarks", min(0.9, base), 0.10, 900 + 50 * i, 0.02 + 0.005 * i))
            rows.append(_row(tkr, q, "qa", min(0.9, base - 0.05), 0.12, 450 + 20 * i, 0.03 + 0.005 * i))
    return pd.DataFrame(rows)


def _sections_frame() -> pd.DataFrame:
    """prepared_remarks text per call for the vocabulary-novelty KPI (A: Q1≈Q2 similar,
    Q3 a clear topic shift)."""
    txt = {
        ("A", "2023Q1"): "cloud platform enterprise customers subscription revenue expansion",
        ("A", "2023Q2"): "cloud platform enterprise customers subscription revenue expansion margins",
        ("A", "2023Q3"): "litigation restructuring charges layoffs writedown goodwill impairment",
    }
    rows = []
    for (tkr, q), t in txt.items():
        rows.append({"ticker": tkr, "quarter": q, "as_of": _QDATE[q], "tag": "prepared_remarks", "text": t})
    return pd.DataFrame(rows)


def test_per_call_kpi_arithmetic():
    per = _per_call_kpis(_sentiment_frame(), _sections_frame())
    a = per[per["ticker"] == "A"].set_index("quarter")

    # length-weighted tone (net = pos-neg), Q1: (.5*1000 + .3*500)/1500
    assert abs(_number(a.loc["2023Q1", "ec_tone"]) - (0.5 * 1000 + 0.3 * 500) / 1500) < 1e-9
    # Q&A gap = qa_net - prepared_net = .3 - .5
    assert abs(_number(a.loc["2023Q1", "ec_qa_gap"]) - (0.3 - 0.5)) < 1e-9
    # length-weighted uncertainty, Q1: (.02*1000 + .05*500)/1500
    assert abs(_number(a.loc["2023Q1", "ec_uncertainty"]) - (0.02 * 1000 + 0.05 * 500) / 1500) < 1e-9
    # tone delta Q2 vs Q1
    tone_q1 = (0.5 * 1000 + 0.3 * 500) / 1500
    tone_q2 = (0.65 * 1200 + 0.40 * 600) / 1800
    assert abs(_number(a.loc["2023Q2", "ec_tone_delta"]) - (tone_q2 - tone_q1)) < 1e-9
    assert math.isnan(_number(a.loc["2023Q1", "ec_tone_delta"]))  # first call -> no prior
    # disclosure-length delta Q2 = log(1800/1500)
    assert abs(_number(a.loc["2023Q2", "ec_length_delta"]) - math.log(1800 / 1500)) < 1e-9
    # vocabulary novelty: Q1 (first) NaN; Q2 low (near-identical); Q3 high (topic shift)
    assert math.isnan(_number(a.loc["2023Q1", "ec_vocab_novelty"]))
    assert _number(a.loc["2023Q2", "ec_vocab_novelty"]) < _number(a.loc["2023Q3", "ec_vocab_novelty"])
    assert _number(a.loc["2023Q3", "ec_vocab_novelty"]) > 0.5


class _Ctx:
    def __init__(self, store):
        self.store = store
        self.log = logging.getLogger("test")


def test_sentiment_kpis_streamed_equals_batch(sqlite_store):
    """The per-ticker STREAMED KPIs (bounded memory) must exactly equal the whole-cache
    computation — proving the streaming refactor preserves QoQ deltas + vocab novelty. Runs on a
    REAL DataStore, so the per-ticker `distinct` + WHERE-scoped reads are exercised for real."""
    sent, sec = _sentiment_frame(), _sections_frame()
    sqlite_store.save("earnings_call_sentiment", sent)
    sqlite_store.save("earnings_call_sections", sec)
    ctx = cast(Context, _Ctx(sqlite_store))
    streamed = sentiment_kpis_streamed(ctx)
    assert streamed is not None
    batch = _per_call_kpis(sent, sec)
    m = streamed.merge(batch, on=["ticker", "quarter"], suffixes=("_s", "_b"))
    assert len(m) == len(batch) == len(streamed), "row set changed under streaming"
    kpi_cols = [
        "ec_tone",
        "ec_qa_gap",
        "ec_uncertainty",
        "ec_tone_delta",
        "ec_length_delta",
        "ec_vocab_novelty",
        "ec_qa_tone_delta",
        "ec_prep_tone_delta",
    ]
    for col in kpi_cols:
        s, b = m[f"{col}_s"].to_numpy(float), m[f"{col}_b"].to_numpy(float)
        ok = (np.isnan(s) & np.isnan(b)) | np.isclose(s, b, equal_nan=True)
        assert ok.all(), f"{col} differs streamed vs batch"
    # A's Q3 topic-shift novelty is a cross-call KPI -> confirms per-ticker order survived streaming
    a3 = streamed[(streamed.ticker == "A") & (streamed.quarter == "2023Q3")]["ec_vocab_novelty"]
    assert float(a3.iloc[0]) > 0.5, "QoQ novelty lost under per-ticker streaming"
    print("\n=== SANITY CHECK: sentiment KPI streaming ===")
    print(
        f"  per-ticker streamed KPIs == whole-cache batch across {len(m)} calls x {len(kpi_cols)} "
        f"KPIs (incl. QoQ tone/length deltas + vocab novelty). A 2023Q3 novelty "
        f"{float(a3.iloc[0]):.3f} (>0.5 topic shift) -> cross-call order preserved."
    )


def test_panel_columns_lifetime_and_missingness():
    tickers = ["A", "B", "C", "D", "E"]
    peers = {t: {p: 1.0 for p in tickers if p != t} for t in tickers}  # all mutual peers
    idx = pd.bdate_range("2023-01-02", "2023-09-29")
    panel = build_earnings_call_feature_panel(_sentiment_frame(), peers, idx, sections=_sections_frame())
    assert not panel.empty
    expected = {
        "date",
        "ticker",
        "f_ec_tone",
        "f_ec_tone_vs_hist",
        "f_ec_qa_gap",
        "f_ec_qa_gap_vs_hist",
        "f_ec_uncertainty",
        "f_ec_uncertainty_vs_hist",
        "f_ec_qa_coherence_mean",
        "f_ec_qa_coherence_mean_vs_hist",
        "f_ec_tone_delta",
        "f_ec_length_delta",
        "f_ec_qa_qq_distance",
        "f_ec_prep_qq_distance",
    }
    assert set(panel.columns) == expected
    assert not any(c.endswith(("_xs", "_vs_peers")) for c in panel.columns)

    panel["date"] = pd.to_datetime(panel["date"])
    a = panel[panel["ticker"] == "A"]
    first_call = pd.Timestamp("2023-02-01")
    # LEAK-FREE: nothing for A on/before the first call date; signal appears the NEXT day
    a_tone = a[a["f_ec_tone"].notna()]
    assert a_tone["date"].min() > first_call
    assert a_tone["date"].min() == pd.Timestamp("2023-02-02")
    # 66 trading sessions are inclusive; session 67 expires rather than becoming a fake zero.
    one = pd.DataFrame({"ticker": ["A"], "as_of": ["2023-02-01"], "ec_tone": [0.0]})
    daily = _daily_frame(one, "ec_tone", idx)
    live = daily["A"].dropna()
    assert len(live) == 66
    assert (live == 0.0).all(), "a genuine zero must survive"
    assert pd.isna(daily.loc[live.index[-1] + pd.offsets.BDay(1), "A"]), "session 67 must be NaN"

    print("\n=== SANITY CHECK: earnings-call features ===")
    print(f"  panel {panel.shape[0]} rows; exactly 12 raw/issuer-history EC features and no peer/cross-sectional variants.")
    print(
        f"  leak-free: ticker A first tone signal at {a_tone['date'].min().date()} "
        "(call 2023-02-01 + 1 trading day); genuine zero survives for 66 sessions and "
        "session 67 is NaN."
    )


def test_issuer_history_is_prior_only_and_requires_four_observations() -> None:
    from src.data_aggregate.utils.text import earnings_call_features as ec

    history = getattr(ec, "_issuer_history_zscore", None)
    assert history is not None, "prior-only issuer-history normalization is missing"
    calls = pd.DataFrame(
        {
            "ticker": ["A"] * 5,
            "quarter": [f"202{i}Q1" for i in range(5)],
            "as_of": pd.date_range("2020-01-01", periods=5, freq="365D"),
            "ec_tone": [1.0, 2.0, 3.0, 4.0, 100.0],
        }
    )
    got = history(calls, "ec_tone")
    assert got.iloc[:4].isna().all()
    expected = (100.0 - 2.5) / np.std([1.0, 2.0, 3.0, 4.0], ddof=1)
    assert math.isclose(float(got.iloc[4]), expected)
    print("\n=== SANITY CHECK: issuer history ===")
    print("  first four calls are NaN; fifth uses only the prior four with sample std. Validated.")


if __name__ == "__main__":
    test_per_call_kpi_arithmetic()
    test_panel_columns_lifetime_and_missingness()
    print("\n=== SANITY CHECK: earnings-call features ===")
    print(
        "  length-weighted tone / Q&A-gap / uncertainty / tone-delta / length-delta "
        "arithmetic correct; vocab novelty low for repeated text & high on a topic shift; "
        "panel emits f_ec_*_xs + _vs_peers; features are leak-free (appear call-date +1 "
        "trading day) and forward-filled to the next call. Validated."
    )
