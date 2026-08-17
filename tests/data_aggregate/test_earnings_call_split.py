"""
Earnings-call feature split: SENTIMENT (FinBERT/LM) vs EMBEDDING (OpenAI Q&A-coherence + drift).

The two are independent cube parts / DAG tasks. This proves the KPI sets are a clean disjoint
partition, and that the embedding-only panel (`build_earnings_call_embedding_panel`) emits ONLY the
embedding KPIs (no tone/uncertainty), sourcing call dates from `sections` (no FinBERT pass needed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.text import earnings_call_features as ec


def test_kpi_sets_are_a_clean_partition():
    s, e = set(ec._SENTIMENT_KPI_COLS), set(ec._EMBEDDING_KPI_COLS)
    assert not (s & e), f"sentiment and embedding KPIs overlap: {s & e}"
    assert s | e == set(ec._KPI_COLS), "the two subsets must cover exactly _KPI_COLS"
    assert "ec_tone" in s and "ec_qa_coherence_mean" in e
    print("\n=== SANITY: KPI partition ===")
    print(f"  sentiment KPIs ({len(s)}): {sorted(s)}")
    print(f"  embedding KPIs ({len(e)}): {sorted(e)}")
    print("  disjoint + union == _KPI_COLS. Validated.")


def test_embedding_panel_emits_only_embedding_kpis(monkeypatch):
    # mock the OpenAI-embedding KPI extraction (per ticker/quarter)
    ekpi = pd.DataFrame({"ticker": ["AAA", "BBB"], "quarter": ["2025Q2", "2025Q2"],
                         "ec_qa_coherence_mean": [0.82, 0.61], "ec_n_qa": [6.0, 8.0],
                         "ec_qa_qq_sim": [0.7, 0.5]})
    monkeypatch.setattr(ec, "build_embedding_kpis", lambda emb: ekpi)

    sections = pd.DataFrame({"ticker": ["AAA", "BBB"], "quarter": ["2025Q2", "2025Q2"],
                             "as_of": ["2025-05-01", "2025-05-02"], "tag": ["qa", "qa"],
                             "text": ["x", "y"]})
    peers = {"AAA": ["BBB"], "BBB": ["AAA"]}
    idx = pd.bdate_range("2025-04-01", "2025-07-01")

    panel = ec.build_earnings_call_embedding_panel(embeddings=object(), peer_dict=peers,
                                                   trading_index=idx, sections=sections)
    assert not panel.empty, "embedding panel should build from the mocked KPIs"
    feat_cols = [c for c in panel.columns if c not in ("date", "ticker")]
    # every emitted feature must derive from an EMBEDDING KPI stem, and NONE from a sentiment one
    emb_stems = [c[len("ec_"):] for c in ec._EMBEDDING_KPI_COLS]
    sent_stems = ("tone", "uncertainty", "vocab", "length_delta", "qa_gap")
    assert all(any(stem in c for stem in emb_stems) for c in feat_cols), feat_cols
    assert not any(bad in c for c in feat_cols for bad in sent_stems), feat_cols

    print("\n=== SANITY: embedding-only panel ===")
    print(f"  {len(feat_cols)} feature cols, all from embedding KPIs (e.g. {sorted(feat_cols)[:4]})")
    print("  no tone/uncertainty/vocab columns; call dates taken from sections (no FinBERT). Validated.")


def test_embedding_panel_empty_safe():
    idx = pd.bdate_range("2025-01-01", "2025-03-01")
    # no embeddings / no sections -> empty frame, no crash
    assert ec.build_earnings_call_embedding_panel(None, {"A": []}, idx, None).empty
    print("\n=== SANITY: embedding panel empty-safe ===")
    print("  None embeddings/sections -> empty panel (no-op without an OpenAI key). Validated.")


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))
