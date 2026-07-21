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

import math

import numpy as np
import pandas as pd

from src.data_aggregate.utils.earnings_call_features import (
    _per_call_kpis,
    build_earnings_call_feature_panel,
)

_QDATE = {"2023Q1": "2023-02-01", "2023Q2": "2023-05-01", "2023Q3": "2023-08-01"}


def _row(tkr, q, tag, pos, neg, words, unc):
    return {"ticker": tkr, "quarter": q, "tag": tag, "as_of": _QDATE[q],
            "sent_pos": pos, "sent_neg": neg, "sent_neu": round(1 - pos - neg, 6),
            "n_words": words, "uncertainty_ratio": unc}


def _sentiment_frame() -> pd.DataFrame:
    rows = [
        # ticker A — the arithmetic-checked name
        _row("A", "2023Q1", "prepared_remarks", 0.60, 0.10, 1000, 0.02),
        _row("A", "2023Q1", "qa",               0.40, 0.10, 500,  0.05),
        _row("A", "2023Q2", "prepared_remarks", 0.70, 0.05, 1200, 0.01),
        _row("A", "2023Q2", "qa",               0.50, 0.10, 600,  0.04),
        _row("A", "2023Q3", "prepared_remarks", 0.55, 0.15, 1100, 0.03),
        _row("A", "2023Q3", "qa",               0.45, 0.20, 550,  0.06),
    ]
    # B..E: distinct tone/uncertainty so the cross-section & peer basket are non-degenerate
    for i, tkr in enumerate(["B", "C", "D", "E"], start=1):
        for q in _QDATE:
            base = 0.30 + 0.1 * i
            rows.append(_row(tkr, q, "prepared_remarks", min(0.9, base), 0.10, 900 + 50 * i, 0.02 + 0.005 * i))
            rows.append(_row(tkr, q, "qa",               min(0.9, base - 0.05), 0.12, 450 + 20 * i, 0.03 + 0.005 * i))
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
        rows.append({"ticker": tkr, "quarter": q, "as_of": _QDATE[q],
                     "tag": "prepared_remarks", "text": t})
    return pd.DataFrame(rows)


def test_per_call_kpi_arithmetic():
    per = _per_call_kpis(_sentiment_frame(), _sections_frame())
    a = per[per["ticker"] == "A"].set_index("quarter")

    # length-weighted tone (net = pos-neg), Q1: (.5*1000 + .3*500)/1500
    assert abs(a.loc["2023Q1", "ec_tone"] - (0.5 * 1000 + 0.3 * 500) / 1500) < 1e-9
    # Q&A gap = qa_net - prepared_net = .3 - .5
    assert abs(a.loc["2023Q1", "ec_qa_gap"] - (0.3 - 0.5)) < 1e-9
    # length-weighted uncertainty, Q1: (.02*1000 + .05*500)/1500
    assert abs(a.loc["2023Q1", "ec_uncertainty"] - (0.02 * 1000 + 0.05 * 500) / 1500) < 1e-9
    # tone delta Q2 vs Q1
    tone_q1 = (0.5 * 1000 + 0.3 * 500) / 1500
    tone_q2 = (0.65 * 1200 + 0.40 * 600) / 1800
    assert abs(a.loc["2023Q2", "ec_tone_delta"] - (tone_q2 - tone_q1)) < 1e-9
    assert math.isnan(a.loc["2023Q1", "ec_tone_delta"])          # first call -> no prior
    # disclosure-length delta Q2 = log(1800/1500)
    assert abs(a.loc["2023Q2", "ec_length_delta"] - math.log(1800 / 1500)) < 1e-9
    # vocabulary novelty: Q1 (first) NaN; Q2 low (near-identical); Q3 high (topic shift)
    assert math.isnan(a.loc["2023Q1", "ec_vocab_novelty"])
    assert a.loc["2023Q2", "ec_vocab_novelty"] < a.loc["2023Q3", "ec_vocab_novelty"]
    assert a.loc["2023Q3", "ec_vocab_novelty"] > 0.5


def test_panel_columns_and_leak_free():
    tickers = ["A", "B", "C", "D", "E"]
    peers = {t: {p: 1.0 for p in tickers if p != t} for t in tickers}   # all mutual peers
    idx = pd.bdate_range("2023-01-02", "2023-09-29")
    panel = build_earnings_call_feature_panel(_sentiment_frame(), peers, idx,
                                              sections=_sections_frame())
    assert not panel.empty
    for kpi in ["ec_tone", "ec_tone_delta", "ec_qa_gap", "ec_uncertainty",
                "ec_vocab_novelty", "ec_length_delta"]:
        assert f"f_{kpi}_xs" in panel.columns, f"missing f_{kpi}_xs"
        assert f"f_{kpi}_vs_peers" in panel.columns, f"missing f_{kpi}_vs_peers"

    panel["date"] = pd.to_datetime(panel["date"])
    a = panel[panel["ticker"] == "A"]
    first_call = pd.Timestamp("2023-02-01")
    # LEAK-FREE: nothing for A on/before the first call date; signal appears the NEXT day
    a_tone = a[a["f_ec_tone_xs"].notna()]
    assert a_tone["date"].min() > first_call
    assert a_tone["date"].min() == pd.Timestamp("2023-02-02")
    # PERSISTENCE: the Q1 signal is carried forward to a date between Q1 and Q2
    mid = pd.Timestamp("2023-03-15")
    assert mid in set(a_tone["date"])

    print("\n=== SANITY CHECK: earnings-call features ===")
    print(f"  panel {panel.shape[0]} rows; KPIs f_ec_{{tone,tone_delta,qa_gap,uncertainty,"
          "vocab_novelty,length_delta}}_{{xs,vs_peers}} present.")
    print(f"  leak-free: ticker A first tone signal at {a_tone['date'].min().date()} "
          "(call 2023-02-01 + 1 trading day), carried forward to 2023-03-15. "
          "Length-weighted tone / Q&A-gap / uncertainty / tone-delta / length-delta "
          "arithmetic + vocab-novelty direction validated in test_per_call_kpi_arithmetic.")


if __name__ == "__main__":
    test_per_call_kpi_arithmetic()
    test_panel_columns_and_leak_free()
    print("\n=== SANITY CHECK: earnings-call features ===")
    print("  length-weighted tone / Q&A-gap / uncertainty / tone-delta / length-delta "
          "arithmetic correct; vocab novelty low for repeated text & high on a topic shift; "
          "panel emits f_ec_*_xs + _vs_peers; features are leak-free (appear call-date +1 "
          "trading day) and forward-filled to the next call. Validated.")
