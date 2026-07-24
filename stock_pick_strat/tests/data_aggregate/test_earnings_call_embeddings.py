"""
Earnings-call OpenAI-embedding layer (src/data_aggregate/utils/earnings_call_embeddings.py).
A STUB embedder (deterministic, no network/spend) drives the full per-turn path on REAL Motley-Fool
transcript shapes: speaker-turn splitting (colon "Name:" headers with the analyst named in the
operator hand-off, AND legacy multi-line "Name / -- / Role" headers), NOISE CLEANING (operator /
IR-flow / pure-courtesy turns dropped, greeting/thanks/congrats preambles stripped so only the meaty
question / answer survives), the cached `earning_calls_embedding` table (ONE ROW PER TURN with its
own embedding + text + tag + person + exchange_idx + answer_idx), incremental skip on re-run, and the
coherence + quarter-to-quarter drift KPIs DERIVED from the turns.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.data_aggregate.utils.earnings_call_embeddings import (
    build_embedding_kpis,
    embed_earnings_calls,
    split_qa_exchanges,
    split_qa_pairs,
    split_turns,
)

_KW = ("revenue", "margin", "growth", "guidance", "cash", "demand", "cost", "backlog", "?")


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
    def replace(self, table, df, chunksize=200_000):
        self.t[table] = df.reset_index(drop=True); return len(df)


class FakeCtx:
    def __init__(self, store): self.store = store; self.log = logging.getLogger("test")


# Real 2024+ Motley-Fool shape: "Name:" headers, analyst named in the operator hand-off, and the
# usual noise -- analyst greeting/congrats openers, a pure-courtesy follow-up, an IR-host flow line,
# and a non-informative closing "question" -- all of which MUST be cleaned/dropped.
_QA = """Operator
The next question comes from the line of Jane Doe with Big Bank. Please proceed.
Jane Doe:
Thanks for taking my question, and congrats on a great quarter. Can you talk about revenue growth and the margin guidance for next year?
John Smith:
Thanks, Jane. Revenue growth was strong and margins expanded on solid demand and cost control.
Sarah Lee:
Let me add that cash generation supported continued growth and the buyback this quarter.
Jane Doe:
That's really helpful, I appreciate it. Thank you.
Jonathan Ng:
Thanks, Jane. Operator, next question, please.
Operator:
The next question comes from the line of Mark Roe with Capital Markets. Please proceed.
Mark Roe:
Good morning. What are you seeing on cash generation and demand trends into next quarter?
Sue Kim:
Demand stayed solid across our segments and cash flow was healthy this quarter on cost discipline.
Mark Roe:
Do you have any other questions for me?
"""
# Legacy 2024-2025 shape: multi-line "Name / -- / Role" headers + bare "Operator".
_QA_DASH = """Operator
[Operator instructions] Our first question comes from the line of Amy Ray with Tech Research. Please proceed.
Amy Ray
--
Analyst
Thank you for taking the question and congrats. How is the cloud backlog trending on demand this year?
Tim Fox
--
Chief Executive Officer
Thanks, Amy. Cloud backlog grew nicely on strong bookings, demand, and revenue growth.
"""
_PREP = ("Thanks, everyone. This quarter revenue grew nicely and margins improved as demand held up "
         "and we controlled cost. Our guidance reflects continued growth and strong cash generation.")


def _sections():
    rows = []
    for tkr in ("AAA", "BBB"):
        for q, aod in (("2024Q1", "2024-05-01"), ("2024Q2", "2024-08-01")):
            rows.append({"ticker": tkr, "quarter": q, "tag": "qa", "as_of": aod, "text": _QA})
            rows.append({"ticker": tkr, "quarter": q, "tag": "prepared_remarks", "as_of": aod,
                         "text": _PREP + (" We also launched a new AI platform." if q == "2024Q2" else "")})
    return pd.DataFrame(rows)


def test_per_turn_split_clean_embed_cache_and_kpis():
    # ---- colon format: cleaning drops noise, keeps mapped Q -> answers ------------------------
    turns = split_turns(_QA, "qa")
    tags = [t["tag"] for t in turns]
    assert tags == ["question", "answer", "answer", "question", "answer"], tags
    assert [t["answer_idx"] for t in turns] == [0, 1, 2, 0, 1], "0=question, 1..k=1st..last answer"
    assert [t["person"] for t in turns] == ["Jane Doe", "John Smith", "Sarah Lee", "Mark Roe", "Sue Kim"]
    assert turns[0]["exchange_idx"] == turns[1]["exchange_idx"] == turns[2]["exchange_idx"] == 0
    assert turns[3]["exchange_idx"] == turns[4]["exchange_idx"] == 1
    q0 = turns[0]["text"].lower()
    assert "revenue growth" in q0 and "thanks" not in q0 and "congrats" not in q0, \
        f"question preamble not stripped: {turns[0]['text']!r}"
    assert not turns[1]["text"].lower().startswith("thanks"), \
        f"answer lead-in 'Thanks, Jane.' not stripped: {turns[1]['text']!r}"
    # Jane's pure-courtesy follow-up, the IR-flow line, and Mark's "do you have questions" are gone
    assert all("appreciate it" not in t["text"].lower() for t in turns), "courtesy turn survived"
    assert all("next question" not in t["text"].lower() for t in turns), "IR-flow turn survived"
    assert all("do you have any other" not in t["text"].lower() for t in turns), "non-informative Q survived"

    # ---- multi-line dash format also parses + cleans ------------------------------------------
    td = split_turns(_QA_DASH, "qa")
    assert [t["tag"] for t in td] == ["question", "answer"] and td[0]["person"] == "Amy Ray"
    assert td[1]["person"] == "Tim Fox" and not td[1]["text"].lower().startswith("thanks")
    assert "backlog" in td[0]["text"].lower() and "congrats" not in td[0]["text"].lower()

    # ---- list of questions with mapped answers ------------------------------------------------
    ex = split_qa_exchanges(_QA)
    assert len(ex) == 2 and ex[0]["analyst"] == "Jane Doe" and ex[0]["managers"] == ["John Smith", "Sarah Lee"]
    assert len(split_qa_pairs(_QA)) == 2                                # back-compat helper intact

    # ---- embed -> ONE ROW PER TURN, cached with every requested column ------------------------
    store = FakeStore(); store.t["earnings_call_sections"] = _sections()
    ctx = FakeCtx(store); stub = StubClient()
    emb = embed_earnings_calls(ctx, client=stub)
    qa_rows, prep_rows = emb[emb["section"] == "qa"], emb[emb["section"] == "prepared_remarks"]
    assert len(qa_rows) == 20, f"5 qa turns x 4 calls, got {len(qa_rows)}"
    assert len(prep_rows) == 4, f"1 prepared turn x 4 calls, got {len(prep_rows)}"
    assert set(emb["tag"]) == {"question", "answer", "prepared"}
    assert {"ticker", "quarter", "seq", "section", "tag", "exchange_idx", "answer_idx", "person",
            "text", "as_of", "embedding", "model", "run_at"}.issubset(emb.columns)
    calls_after_first = stub.n_calls
    embed_earnings_calls(ctx, client=stub)                             # re-run: incremental
    assert stub.n_calls == calls_after_first, "re-run must make ZERO new embedding calls"

    # ---- KPIs derived from the turns ----------------------------------------------------------
    kpi = build_embedding_kpis(emb).sort_values(["ticker", "quarter"]).reset_index(drop=True)
    assert (kpi["ec_n_qa"] == 2).all(), "2 exchanges per call"
    assert (kpi["ec_n_answers"] == 3).all(), "3 answer turns per call (2 in ex0 + 1 in ex1)"
    assert (kpi["ec_qa_answer_ratio"] == 1.5).all(), "3 answers / 2 questions = 1.5"
    assert kpi["ec_qa_coherence_mean"].between(-1, 1).all()
    q1, q2 = kpi[kpi["quarter"] == "2024Q1"], kpi[kpi["quarter"] == "2024Q2"]
    assert q1["ec_qa_answer_ratio_qq"].isna().all(), "first quarter has no prior -> ratio QoQ NaN"
    assert (q2["ec_qa_answer_ratio_qq"] == 0.0).all(), "answer/question ratio unchanged QoQ -> delta 0"
    assert q1["ec_qa_qq_sim"].isna().all(), "first quarter has no prior"
    assert q2["ec_prep_qq_sim"].notna().all() and q2["ec_prep_qq_sim"].between(-1, 1).all()

    print("\n=== SANITY CHECK: per-turn earnings-call embeddings (cleaned) ===")
    print(f"  colon format -> tags {tags}, answer_idx {[t['answer_idx'] for t in turns]} "
          f"(0=Q, 1..k=1st..last answer); persons {[t['person'] for t in turns]}.")
    print(f"  cleaning: question preamble 'Thanks for taking my question, and congrats' STRIPPED; "
          f"answer lead-in 'Thanks, Jane.' STRIPPED; IR-flow + courtesy + 'do you have questions' DROPPED.")
    print(f"  multi-line dash format also parsed: Q={td[0]['person']} -> A={td[1]['person']}.")
    print(f"  table: {len(qa_rows)} qa-turn rows + {len(prep_rows)} prepared rows, each with its own "
          f"embedding + text + person + tag + exchange_idx + answer_idx + as_of + model/run_at.")
    print(f"  refined coherence (avg cos of Q vs EACH answer, then mean over exchanges) "
          f"{kpi['ec_qa_coherence_mean'].mean():.3f}; answer/question ratio {kpi['ec_qa_answer_ratio'].mean():.2f} "
          f"(QoQ delta {float(q2['ec_qa_answer_ratio_qq'].iloc[0]):.2f}); QoQ prepared drift "
          f"2024Q2 {q2['ec_prep_qq_sim'].mean():.3f} (new AI-platform topic added).")
    print(f"  incremental: re-run made 0 new OpenAI calls (still {stub.n_calls}). Validated with a stub (no spend).")


# HuggingFace-backbone shape: verbatim `content` with "Name:" colon headers, a long CEO remark that
# ENDS with "we'll open it up for questions", and a Q&A where an exec greets before the (substitute)
# analyst asks -- the cases that previously dropped the CEO / inverted Q&A.
_HF_PREP = ("Operator: Good afternoon and welcome to the call. [Operator Instructions] Sir, please go "
            "ahead. I would like to turn the call over to Mike. Mike Ceo: Thanks Ankur. "
            + "Revenue grew on strong demand and margins expanded across every region as we executed "
              "well this quarter. " * 5
            + "With that, we'll open it up for your questions.")
_HF_QA = ("Operator: Your first question comes from Doug of Cowen. Please go ahead. "
          "Mike Ceo: Hi Doug. "
          "Ryan Sub: Hi, this is Ryan on for Doug. Can you walk through the margin outlook and the "
          "cash flow trends you expect into next quarter? "
          "Mike Ceo: Sure. Margins improved and cash flow was strong on solid demand and cost control. "
          "Bob Cfo: And on cash generation, we expect continued strength across our segments next quarter.")


def test_hf_style_long_remarks_and_qa_classification():
    # a long prepared remark that MENTIONS opening for questions must NOT be dropped -> CEO is known
    prep = split_turns(_HF_PREP, "prepared_remarks")
    mgmt = {t["person"].strip().lower() for t in prep if t.get("person")}
    assert "mike ceo" in mgmt, f"long CEO remark wrongly dropped; mgmt={mgmt}"

    # with the CEO in mgmt, an exec greeting after the hand-off is NOT mistaken for the analyst
    ex = split_qa_exchanges(_HF_QA, mgmt_names=mgmt)
    assert len(ex) == 1, ex
    assert ex[0]["analyst"] == "Ryan Sub", f"analyst should be the (substitute) analyst, got {ex[0]['analyst']}"
    assert ex[0]["managers"] == ["Mike Ceo", "Bob Cfo"], ex[0]["managers"]
    q = ex[0]["question"].lower()
    assert "margin outlook" in q and "hi doug" not in q, f"question not cleaned: {ex[0]['question']!r}"

    # OLD (2007-2010) transcripts head turns with "Name - Firm/Role:" -> must still parse
    old = ("Operator: Your first question comes from Matthew Dodds - Citigroup.\n"
           "Matthew Dodds - Citigroup: Can you talk about the gross margin trend and the pricing outlook?\n"
           "William Weldon - Chairman: Margins improved on favorable mix and cost control this quarter.")
    eo = split_qa_exchanges(old)
    assert eo and eo[0]["analyst"] == "Matthew Dodds" and eo[0]["managers"] == ["William Weldon"], eo
    assert "gross margin" in eo[0]["question"].lower()

    print("\n=== SANITY CHECK: HuggingFace-shape content ===")
    print(f"  colon 'Name:' headers parse; long CEO remark ending 'we'll open it up for questions' "
          f"KEPT -> mgmt={sorted(mgmt)}.")
    print(f"  exec 'Hi Doug.' after the hand-off is NOT the analyst; ex0 analyst={ex[0]['analyst']!r} "
          f"-> managers {ex[0]['managers']}; question cleaned to the meaty ask. Validated.")


def test_force_reembed_drops_stale_turns():
    """A re-parse can yield FEWER turns than a prior run cached; a force re-embed must RECONCILE
    (drop the orphaned tail rows) so the table matches the current parse -- not leave stale answers
    that would inflate answer counts / pollute the KPIs."""
    store = FakeStore(); store.t["earnings_call_sections"] = _sections()
    ctx = FakeCtx(store); stub = StubClient()
    embed_earnings_calls(ctx, client=stub)                        # initial embed
    tbl = "earning_calls_embedding"
    n0 = len(store.t[tbl])
    stale = store.t[tbl].iloc[[0]].copy(); stale["seq"] = 999; stale["text"] = "stale orphan turn"
    store.t[tbl] = pd.concat([store.t[tbl], stale], ignore_index=True)   # simulate a prior longer parse
    assert (store.t[tbl]["seq"] == 999).any()
    embed_earnings_calls(ctx, client=stub, force=True)            # force re-embed -> reconcile
    assert not (store.t[tbl]["seq"] == 999).any(), "orphaned turn must be dropped on force re-embed"
    assert len(store.t[tbl]) == n0, "table matches the current parse exactly after reconcile"
    print("\n=== SANITY CHECK: force re-embed reconcile ===")
    print(f"  injected 1 orphaned turn (seq=999); force re-embed dropped it -> {len(store.t[tbl])} rows "
          f"== fresh parse {n0}. Stale turns cannot linger.")


if __name__ == "__main__":
    test_per_turn_split_clean_embed_cache_and_kpis()
    test_hf_style_long_remarks_and_qa_classification()
    test_force_reembed_drops_stale_turns()
