"""
Earnings-call OpenAI-embedding layer (src/data_aggregate/utils/earnings_call_embeddings.py).
A STUB embedder (deterministic, no network/spend) drives the full path: Q&A pair splitting, the
cached `earning_calls_embedding` table (qa + prepared rows, n_qa, cosine(Q,A) mean/std), incremental
skip on re-run, and the quarter-to-quarter similarity + coherence KPIs.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.data_aggregate.utils.earnings_call_embeddings import (
    build_embedding_kpis,
    embed_earnings_calls,
    split_qa_pairs,
)

_KW = ("revenue", "margin", "growth", "guidance", "cash", "demand", "cost", "?")


def _vec(t: str):
    v = np.array([t.lower().count(w) for w in _KW], dtype="float64") + 0.1
    return v.tolist()


class _Emb:
    def __init__(self, parent): self.parent = parent
    def create(self, model, input):
        self.parent.n_calls += 1
        return SimpleNamespace(data=[SimpleNamespace(embedding=_vec(t)) for t in input])


class StubClient:
    def __init__(self): self.n_calls = 0; self.embeddings = _Emb(self)


class FakeStore:
    _PK = {"earning_calls_embedding": ["ticker", "quarter", "section"]}
    def __init__(self): self.t: dict[str, pd.DataFrame] = {}
    def load(self, table, columns=None): return self.t.get(table, pd.DataFrame()).copy()
    def save(self, table, df):
        both = pd.concat([self.t.get(table, pd.DataFrame()), df], ignore_index=True)
        pk = self._PK.get(table)
        if pk:
            both = both.drop_duplicates(subset=pk, keep="last")
        self.t[table] = both.reset_index(drop=True)


class FakeCtx:
    def __init__(self, store): self.store = store; self.log = logging.getLogger("test")


_QA = """Operator
Our first question comes from Jane of Big Bank.
Jane Doe -- Big Bank -- Analyst
Can you talk about revenue growth and the margin guidance for next year?
John Smith -- Chief Executive Officer
Sure. Revenue growth was strong and margins expanded on solid demand and cost control.
Operator
The next question comes from Mark.
Mark Roe -- Capital Markets -- Analyst
What are you seeing on cash generation and demand trends into the quarter?
Sue Lee -- Chief Financial Officer
Cash flow was healthy and demand stayed solid across our segments this quarter.
"""
_PREP = ("Thanks everyone. This quarter revenue grew nicely and margins improved as demand held up "
         "and we controlled cost. Our guidance reflects continued growth and strong cash generation.")


def _sections():
    rows = []
    for tkr in ("AAA", "BBB"):
        for q, aod in (("2024Q1", "2024-05-01"), ("2024Q2", "2024-08-01")):
            rows.append({"ticker": tkr, "quarter": q, "tag": "qa", "as_of": aod, "text": _QA})
            rows.append({"ticker": tkr, "quarter": q, "tag": "prepared_remarks", "as_of": aod,
                         "text": _PREP + (" We also launched a new AI platform." if q == "2024Q2" else "")})
    return pd.DataFrame(rows)


def test_qa_pair_split_embed_cache_and_kpis():
    pairs = split_qa_pairs(_QA)
    assert len(pairs) == 2, f"expected 2 Q&A exchanges, got {len(pairs)}"
    assert "revenue growth" in pairs[0][0].lower() and "margins expanded" in pairs[0][1].lower()

    store = FakeStore(); store.t["earnings_call_sections"] = _sections()
    ctx = FakeCtx(store); stub = StubClient()

    emb = embed_earnings_calls(ctx, client=stub)
    qa_rows = emb[emb["section"] == "qa"]
    prep_rows = emb[emb["section"] == "prepared_remarks"]
    assert len(qa_rows) == 4 and len(prep_rows) == 4, "one qa + one prepared row per 4 calls"
    assert (qa_rows["n_qa"] == 2).all(), "n_qa should be 2 for every call"
    assert qa_rows["qa_cos_mean"].between(-1, 1).all(), "cosine(Q,A) must be in [-1,1]"
    assert {"model", "run_at"}.issubset(emb.columns), "must stamp model + run timestamp"
    calls_after_first = stub.n_calls

    embed_earnings_calls(ctx, client=stub)                       # re-run: incremental
    assert stub.n_calls == calls_after_first, "re-run must make ZERO new embedding calls"

    kpi = build_embedding_kpis(emb).sort_values(["ticker", "quarter"]).reset_index(drop=True)
    assert (kpi["ec_n_qa"] == 2).all()
    q1 = kpi[kpi["quarter"] == "2024Q1"]; q2 = kpi[kpi["quarter"] == "2024Q2"]
    assert q1["ec_qa_qq_sim"].isna().all(), "first quarter has no prior -> QoQ sim NaN"
    assert q2["ec_prep_qq_sim"].notna().all() and q2["ec_prep_qq_sim"].between(-1, 1).all()

    print("\n=== SANITY CHECK: earnings-call embedding features ===")
    print(f"  Q&A split: {len(pairs)} exchanges; e.g. Q='{pairs[0][0][:40]}...' A='{pairs[0][1][:40]}...'")
    print(f"  cache: {len(qa_rows)} qa + {len(prep_rows)} prepared rows; n_qa=2; "
          f"coherence mean {qa_rows['qa_cos_mean'].mean():.3f} (cosine of question vs answer)")
    print(f"  incremental: re-run made 0 new OpenAI calls (still {stub.n_calls}).")
    print(f"  QoQ prepared-remarks similarity (2024Q2 vs Q1): "
          f"{q2['ec_prep_qq_sim'].mean():.3f}  (lower = narrative shift; the Q2 call added a new topic)")
    print("  CONCLUSION: Q&A coherence (cos(Q,A) mean/std) + n_qa + quarter-to-quarter embedding "
          "drift computed and cached with model+timestamp. Validated (stub embedder, no spend).")


if __name__ == "__main__":
    test_qa_pair_split_embed_cache_and_kpis()
