"""
Earnings-call OpenAI-embedding layer (src/data_aggregate/utils/earnings_call_embeddings.py).
A STUB embedder (deterministic, no network/spend) drives the full per-turn path: speaker-turn
splitting (question / answer / prepared with person + exchange pairing), the cached
`earning_calls_embedding` table (ONE ROW PER TURN with its own embedding + text + tag), incremental
skip on re-run, and the coherence + quarter-to-quarter drift KPIs DERIVED from the turns.
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
    split_turns,
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
    _PK = {"earning_calls_embedding": ["ticker", "quarter", "seq"]}
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


def test_per_turn_split_embed_cache_and_kpis():
    # ---- turn splitting: 2 questions + 2 answers, persons, exchange pairing -------------------
    turns = split_turns(_QA, "qa")
    tags = [t["tag"] for t in turns]
    assert tags == ["question", "answer", "question", "answer"], tags
    assert [t["person"] for t in turns] == ["Jane Doe", "John Smith", "Mark Roe", "Sue Lee"]
    assert turns[0]["exchange_idx"] == turns[1]["exchange_idx"] == 0     # Q0 pairs with A0
    assert turns[2]["exchange_idx"] == turns[3]["exchange_idx"] == 1
    prep = split_turns(_PREP, "prepared_remarks")
    assert len(prep) == 1 and prep[0]["tag"] == "prepared" and prep[0]["exchange_idx"] == -1
    assert len(split_qa_pairs(_QA)) == 2                                 # back-compat helper intact

    # ---- embed -> ONE ROW PER TURN, cached with all metadata ----------------------------------
    store = FakeStore(); store.t["earnings_call_sections"] = _sections()
    ctx = FakeCtx(store); stub = StubClient()
    emb = embed_earnings_calls(ctx, client=stub)

    qa_rows = emb[emb["section"] == "qa"]
    prep_rows = emb[emb["section"] == "prepared_remarks"]
    assert len(qa_rows) == 16, f"4 qa turns x 4 calls, got {len(qa_rows)}"      # 2 Q + 2 A per call
    assert len(prep_rows) == 4, f"1 prepared turn x 4 calls, got {len(prep_rows)}"
    assert set(emb["tag"]) == {"question", "answer", "prepared"}
    assert {"ticker", "quarter", "seq", "section", "tag", "exchange_idx", "person", "text",
            "as_of", "embedding", "model", "run_at"}.issubset(emb.columns)
    # a stored question turn and its answer turn share exchange_idx within a call
    one = qa_rows[(qa_rows["ticker"] == "AAA") & (qa_rows["quarter"] == "2024Q1")]
    q0 = one[(one["tag"] == "question") & (one["exchange_idx"] == 0)]
    a0 = one[(one["tag"] == "answer") & (one["exchange_idx"] == 0)]
    assert len(q0) == 1 and len(a0) == 1 and isinstance(q0.iloc[0]["embedding"], list)
    assert q0.iloc[0]["person"] == "Jane Doe" and "revenue growth" in q0.iloc[0]["text"].lower()
    calls_after_first = stub.n_calls

    embed_earnings_calls(ctx, client=stub)                              # re-run: incremental
    assert stub.n_calls == calls_after_first, "re-run must make ZERO new embedding calls"

    # ---- KPIs derived from the turns ----------------------------------------------------------
    kpi = build_embedding_kpis(emb).sort_values(["ticker", "quarter"]).reset_index(drop=True)
    assert (kpi["ec_n_qa"] == 2).all(), "2 exchanges per call"
    assert kpi["ec_qa_coherence_mean"].between(-1, 1).all()
    q1 = kpi[kpi["quarter"] == "2024Q1"]; q2 = kpi[kpi["quarter"] == "2024Q2"]
    assert q1["ec_qa_qq_sim"].isna().all(), "first quarter has no prior -> QoQ sim NaN"
    assert q2["ec_prep_qq_sim"].notna().all() and q2["ec_prep_qq_sim"].between(-1, 1).all()

    print("\n=== SANITY CHECK: per-turn earnings-call embeddings ===")
    print(f"  turns: qa split -> {tags} with persons Jane/John/Mark/Sue; Q0<->A0, Q1<->A1 paired.")
    print(f"  table: {len(qa_rows)} qa-turn rows + {len(prep_rows)} prepared-turn rows "
          f"(1 row/turn), each with its own embedding + text + person + tag + exchange_idx + "
          f"as_of + model/run_at.")
    print(f"  coherence (cos question vs its answer turns): mean over exchanges "
          f"{kpi['ec_qa_coherence_mean'].mean():.3f}; ec_n_qa=2.")
    print(f"  incremental: re-run made 0 new OpenAI calls (still {stub.n_calls}).")
    print(f"  QoQ prepared drift (2024Q2 vs Q1): {q2['ec_prep_qq_sim'].mean():.3f} "
          f"(lower = narrative shift; Q2 added a new AI-platform topic).")
    print("  CONCLUSION: each question & answer is now stored as its OWN row (ticker, quarter, "
          "seq, tag, person, text, as_of, run_at, embedding); cosine(Q,A) coherence + QoQ drift "
          "are DERIVED from those turns. Validated with a stub embedder (no spend).")


if __name__ == "__main__":
    test_per_turn_split_embed_cache_and_kpis()
