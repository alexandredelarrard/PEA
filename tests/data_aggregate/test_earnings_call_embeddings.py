"""
Earnings-call OpenAI-embedding layer (src/data_aggregate/utils/earnings_call_embeddings.py).
A STUB embedder (deterministic, no network/spend) drives the full per-turn path from stored
`earnings_call_sections` PARAGRAPH rows: the shared split (`src/utils/earnings_call_split.py`,
speakers from the source `speaker` field, no header parsing) supplies the turns, NOISE CLEANING
(operator / IR-flow / pure-courtesy turns dropped, greeting/thanks/congrats preambles stripped)
is visible in the stored text, the cached `earning_calls_embedding` table holds ONE ROW PER TURN
(embedding + text + tag + person + exchange_idx + answer_idx), re-runs are incremental, and the
coherence + quarter-to-quarter drift KPIs are DERIVED from the turns. On the 16 hand-labelled
real calls the stored rows are exactly `CallSplit.turns` and every analyst turn is a question.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd

from src.constants.constants import EARNINGS_CALL_EMBEDDING_MODEL, EARNINGS_CALL_TAG_QUESTION
from src.context import Context
from src.data_aggregate.utils.text.earnings_call_embeddings import (
    build_embedding_kpis,
    embed_earnings_calls,
    embedding_kpis_streamed,
)
from src.data_store.schema import name_of, resolve
from src.utils.earnings_call_split import split_call
from tests.fixtures.earnings_call_rows import FIXTURE_PATHS, PARAGRAPH_COLUMNS, fixture_paragraphs, load_fixture

_KW = ("revenue", "margin", "growth", "guidance", "cash", "demand", "cost", "backlog", "?")
_TURN_KEYS = ["section", "tag", "person", "text", "exchange_idx", "answer_idx"]


def _vec(t: str):
    v = np.array([t.lower().count(w) for w in _KW], dtype="float64") + 0.1
    return v.tolist()


class _Emb:
    def __init__(self, parent):
        self.parent = parent

    def create(self, model, input):
        self.parent.n_calls += 1
        return SimpleNamespace(data=[SimpleNamespace(embedding=_vec(t)) for t in input])


class StubClient:
    def __init__(self):
        self.n_calls = 0
        self.embeddings = _Emb(self)


class FakeStore:
    """In-memory stand-in mirroring the `DataStore` contract (`where=` equality/IN, raise-unless-
    `optional`, `distinct` with `where`, `delete`; `order_by` is accepted and ignored -- the split
    orders paragraphs itself).

    The one test store that cannot be the real SQLite `DataStore`: `earning_calls_embedding.embedding`
    is a `DOUBLE PRECISION[]` and SQLite's driver refuses to bind a Python list.
    """

    def __init__(self):
        self.t: dict[str, pd.DataFrame] = {}

    @staticmethod
    def _filter(df, where):
        for col, val in (where or {}).items():
            df = df[df[col].isin(val)] if isinstance(val, list | tuple | set) else df[df[col] == val]
        return df

    def load(self, table, columns=None, where=None, *, optional=False, **kw):
        df = self._filter(self.t.get(name_of(table), pd.DataFrame()), where)
        if df.empty:
            if optional:
                return None
            raise LookupError(f"{name_of(table)} is empty/missing and the read was not optional")
        return (df[list(columns)] if columns else df).copy().reset_index(drop=True)

    def distinct(self, table, column, *, where=None, **kw):
        df = self.t.get(name_of(table))
        if df is None or df.empty:
            return []
        return self._filter(df, where)[column].dropna().unique().tolist()

    def save(self, table, df, pk=None):
        name = name_of(table)
        both = pd.concat([self.t.get(name, pd.DataFrame()), df], ignore_index=True)
        pk = pk or list(resolve(table).pk)
        self.t[name] = both.drop_duplicates(subset=pk, keep="last").reset_index(drop=True)

    def delete(self, table, where):
        name = name_of(table)
        df = self.t.get(name)
        if df is None or df.empty:
            return 0
        drop = self._filter(df, where).index
        self.t[name] = df.drop(index=drop).reset_index(drop=True)
        return len(drop)


class FakeCtx:
    def __init__(self, store):
        self.store = store
        self.log = logging.getLogger("test")


# One call as source paragraphs (speaker, content), with the usual noise: analyst greeting/congrats
# openers, a pure-courtesy follow-up, an IR-host flow line, and a non-informative closing
# "question" -- all of which MUST be cleaned/dropped.
_PREP = (
    "Thanks, everyone. This quarter revenue grew nicely and margins improved as demand held up "
    "and we controlled cost. Our guidance reflects continued growth and strong cash generation."
)
_CALL = [
    ("Operator", "Good day and welcome to the earnings conference call. I will now turn the call over to management."),
    ("John Smith", _PREP),
    ("Operator", "The next question comes from the line of Jane Doe with Big Bank. Please proceed."),
    (
        "Jane Doe",
        "Thanks for taking my question, and congrats on a great quarter. Can you talk about revenue growth and the margin guidance for next year?",
    ),
    ("John Smith", "Thanks, Jane. Revenue growth was strong and margins expanded on solid demand and cost control."),
    ("Sarah Lee", "Let me add that cash generation supported continued growth and the buyback this quarter."),
    ("Jane Doe", "That's really helpful, I appreciate it. Thank you."),
    ("Jonathan Ng", "Thanks, Jane. Operator, next question, please."),
    ("Operator", "The next question comes from the line of Mark Roe with Capital Markets. Please proceed."),
    ("Mark Roe", "Good morning. What are you seeing on cash generation and demand trends into next quarter?"),
    ("Sue Kim", "Demand stayed solid across our segments and cash flow was healthy this quarter on cost discipline."),
    ("Mark Roe", "Do you have any other questions for me?"),
]


def _sections() -> pd.DataFrame:
    """`earnings_call_sections` paragraph rows: 2 tickers x 2 quarters of `_CALL`; 2024Q2 adds a topic."""
    rows = []
    for tkr in ("AAA", "BBB"):
        for q, aod in (("2024Q1", "2024-05-01"), ("2024Q2", "2024-08-01")):
            for number, (speaker, content) in enumerate(_CALL, start=1):
                if number == 2 and q == "2024Q2":
                    content = content + " We also launched a new AI platform."
                rows.append(
                    {"ticker": tkr, "quarter": q, "paragraph": number, "as_of": aod, "transcript_id": 1, "speaker": speaker, "content": content}
                )
    return pd.DataFrame(rows, columns=PARAGRAPH_COLUMNS)


def test_per_turn_split_clean_embed_cache_and_kpis():
    # ---- embed -> ONE ROW PER TURN, the turns of the shared split ------------------------------
    store = FakeStore()
    store.t["earnings_call_sections"] = _sections()
    ctx = cast(Context, FakeCtx(store))
    stub = StubClient()
    embed_earnings_calls(ctx, client=stub)
    emb = store.load("earning_calls_embedding")
    assert emb is not None
    one = emb[(emb["ticker"] == "AAA") & (emb["quarter"] == "2024Q1")].sort_values("seq")
    turns = one[_TURN_KEYS].to_dict("records")
    tags = [t["tag"] for t in turns]
    assert tags == ["prepared", "question", "answer", "answer", "question", "answer"], tags
    assert [t["answer_idx"] for t in turns] == [-1, 0, 1, 2, 0, 1], "0=question, 1..k=1st..last answer"
    assert [t["person"] for t in turns] == ["John Smith", "Jane Doe", "John Smith", "Sarah Lee", "Mark Roe", "Sue Kim"]
    assert [t["exchange_idx"] for t in turns] == [-1, 0, 0, 0, 1, 1]
    assert one["seq"].tolist() == list(range(len(turns)))
    q0 = turns[1]["text"].lower()
    assert "revenue growth" in q0 and "thanks" not in q0 and "congrats" not in q0, f"question preamble not stripped: {turns[1]['text']!r}"
    assert not turns[2]["text"].lower().startswith("thanks"), f"answer lead-in 'Thanks, Jane.' not stripped: {turns[2]['text']!r}"
    assert not turns[0]["text"].lower().startswith("thanks"), "prepared courtesy opener not stripped"
    # Jane's pure-courtesy follow-up, the IR-flow line, and Mark's "do you have questions" are gone
    assert all("appreciate it" not in t["text"].lower() for t in turns), "courtesy turn survived"
    assert all("next question" not in t["text"].lower() for t in turns), "IR-flow turn survived"
    assert all("do you have any other" not in t["text"].lower() for t in turns), "non-informative Q survived"

    qa_rows, prep_rows = emb[emb["section"] == "qa"], emb[emb["section"] == "prepared_remarks"]
    assert len(qa_rows) == 20, f"5 qa turns x 4 calls, got {len(qa_rows)}"
    assert len(prep_rows) == 4, f"1 prepared turn x 4 calls, got {len(prep_rows)}"
    assert set(emb["tag"]) == {"question", "answer", "prepared"}
    assert {
        "ticker",
        "quarter",
        "seq",
        "section",
        "tag",
        "exchange_idx",
        "answer_idx",
        "person",
        "text",
        "as_of",
        "embedding",
        "model",
        "run_at",
    }.issubset(emb.columns)
    assert set(emb["model"]) == {EARNINGS_CALL_EMBEDDING_MODEL}
    assert set(one["as_of"]) == {"2024-05-01"}, "the turn as_of is the paragraphs' call date"
    calls_after_first = stub.n_calls
    embed_earnings_calls(ctx, client=stub)  # re-run: incremental
    assert stub.n_calls == calls_after_first, "re-run must make ZERO new embedding calls"

    # ---- KPIs derived from the turns ----------------------------------------------------------
    kpi = build_embedding_kpis(emb)
    assert kpi is not None
    kpi = kpi.sort_values(["ticker", "quarter"]).reset_index(drop=True)
    assert set(kpi) == {
        "ticker",
        "quarter",
        "ec_qa_coherence_mean",
        "ec_qa_qq_distance",
        "ec_prep_qq_distance",
    }
    assert kpi["ec_qa_coherence_mean"].between(-1, 1).all()
    q1, q2 = kpi[kpi["quarter"] == "2024Q1"], kpi[kpi["quarter"] == "2024Q2"]
    assert q1["ec_qa_qq_distance"].isna().all(), "first quarter has no prior"
    assert q2["ec_prep_qq_distance"].notna().all() and q2["ec_prep_qq_distance"].between(0, 2).all()

    print("\n=== SANITY CHECK: per-turn earnings-call embeddings from paragraphs (cleaned) ===")
    print(f"  stored turns -> tags {tags}, answer_idx {[t['answer_idx'] for t in turns]}; persons {[t['person'] for t in turns]}.")
    print(
        "  cleaning: question preamble 'Thanks for taking my question, and congrats' STRIPPED; "
        "answer lead-in 'Thanks, Jane.' STRIPPED; IR-flow + courtesy + 'do you have questions' DROPPED."
    )
    print(f"  table: {len(qa_rows)} qa-turn rows + {len(prep_rows)} prepared rows, as_of = call date, model/run_at stamped.")
    print(
        f"  coherence {kpi['ec_qa_coherence_mean'].mean():.3f}; QoQ prepared drift "
        f"2024Q2 {q2['ec_prep_qq_distance'].mean():.3f} (new AI-platform topic added)."
    )
    print(f"  incremental: re-run made 0 new OpenAI calls (still {stub.n_calls}). Validated with a stub (no spend).")


def test_embedded_turns_are_the_split_turns_on_hand_labelled_calls():
    """AC-007 at the embedding layer: on the 16 real fixtures the stored rows are exactly
    `split_call(...).turns` (keys, values, order = seq), every hand-labelled analyst turn is a
    question and no management turn is."""
    store = FakeStore()
    store.t["earnings_call_sections"] = pd.concat([fixture_paragraphs(p.stem) for p in FIXTURE_PATHS], ignore_index=True)
    embed_earnings_calls(cast(Context, FakeCtx(store)), client=StubClient())
    emb = store.t["earning_calls_embedding"]
    analyst_turns = analyst_questions = mgmt_questions = n_turns = 0
    for path in FIXTURE_PATHS:
        fx = load_fixture(path.stem)
        quarter = f"{fx['fiscal_year']}Q{fx['fiscal_quarter']}"
        stored = emb[(emb["ticker"] == fx["symbol"]) & (emb["quarter"] == quarter)].sort_values("seq")
        expected = split_call(fx["paragraphs"]).turns
        assert stored[_TURN_KEYS].to_dict("records") == expected, path.stem
        assert stored["seq"].tolist() == list(range(len(expected)))
        analysts, mgmt = set(fx["truth"]["analysts"]), set(fx["truth"]["management"])
        is_q = stored["tag"].eq(EARNINGS_CALL_TAG_QUESTION)
        analyst_turns += int(stored["person"].isin(analysts).sum())
        analyst_questions += int((stored["person"].isin(analysts) & is_q).sum())
        mgmt_questions += int((stored["person"].isin(mgmt) & is_q).sum())
        n_turns += len(stored)
    assert analyst_turns > 0 and analyst_questions == analyst_turns
    assert mgmt_questions == 0
    print("\n=== SANITY CHECK: embedding turns = shared split turns ===")
    print(
        f"  {len(FIXTURE_PATHS)} real calls, {n_turns} stored turns identical to CallSplit.turns; "
        f"{analyst_questions}/{analyst_turns} analyst turns are questions, {mgmt_questions} management questions. Validated."
    )


def test_force_reembed_drops_stale_turns():
    """A re-parse can yield FEWER turns than a prior run cached; a force re-embed must RECONCILE
    (drop the orphaned tail rows) so the table matches the current parse -- not leave stale answers
    that would inflate answer counts / pollute the KPIs."""
    store = FakeStore()
    store.t["earnings_call_sections"] = _sections()
    ctx = cast(Context, FakeCtx(store))
    stub = StubClient()
    embed_earnings_calls(ctx, client=stub)  # initial embed
    tbl = "earning_calls_embedding"
    n0 = len(store.t[tbl])
    stale = store.t[tbl].iloc[[0]].copy()
    stale["seq"] = 999
    stale["text"] = "stale orphan turn"
    store.t[tbl] = pd.concat([store.t[tbl], stale], ignore_index=True)  # simulate a prior longer parse
    assert (store.t[tbl]["seq"] == 999).any()
    embed_earnings_calls(ctx, client=stub, force=True)  # force re-embed -> reconcile
    assert not (store.t[tbl]["seq"] == 999).any(), "orphaned turn must be dropped on force re-embed"
    assert len(store.t[tbl]) == n0, "table matches the current parse exactly after reconcile"
    print("\n=== SANITY CHECK: force re-embed reconcile ===")
    print(
        f"  injected 1 orphaned turn (seq=999); force re-embed dropped it -> {len(store.t[tbl])} rows == fresh parse {n0}. Stale turns cannot linger."
    )


def test_embedding_resume_requires_expected_model_and_reconciles_without_force() -> None:
    store = FakeStore()
    store.t["earnings_call_sections"] = _sections()
    ctx = cast(Context, FakeCtx(store))
    first = StubClient()
    embed_earnings_calls(ctx, client=first)
    table = store.t["earning_calls_embedding"]
    table["model"] = "legacy-or-other-model"
    stale = table.iloc[[0]].copy()
    stale["seq"] = 999
    store.t["earning_calls_embedding"] = pd.concat([table, stale], ignore_index=True)

    second = StubClient()
    embed_earnings_calls(ctx, client=second)

    refreshed = store.t["earning_calls_embedding"]
    assert second.n_calls == 4
    assert set(refreshed["model"]) == {EARNINGS_CALL_EMBEDDING_MODEL}
    assert not refreshed["seq"].eq(999).any()
    print("\n=== SANITY CHECK: embedding cache provenance ===")
    print("  wrong-model calls are not considered complete; normal resume re-embeds and removes orphaned legacy turns. Validated with a stub.")


def test_force_reembed_deletes_every_stale_turn_when_parse_becomes_empty() -> None:
    store = FakeStore()
    store.t["earnings_call_sections"] = _sections()
    ctx = cast(Context, FakeCtx(store))
    embed_earnings_calls(ctx, client=StubClient())
    assert len(store.t["earning_calls_embedding"]) > 0
    store.t["earnings_call_sections"]["content"] = "Thanks."
    embed_earnings_calls(ctx, client=StubClient(), force=True)
    assert store.t["earning_calls_embedding"].empty
    print("\n=== SANITY CHECK: empty force re-parse ===")
    print("  a call that now parses to zero turns deletes every previously cached turn. Validated.")


def test_embedding_kpis_require_consecutive_quarters_and_consistent_provenance() -> None:
    rows = []
    for quarter, value, model in (("2024Q1", [1.0, 0.0], "m1"), ("2024Q3", [0.0, 1.0], "m1")):
        for seq, tag in enumerate(("question", "answer")):
            rows.append(
                {
                    "ticker": "AAA",
                    "quarter": quarter,
                    "section": "qa",
                    "tag": tag,
                    "exchange_idx": 0,
                    "embedding": value,
                    "model": model,
                    "as_of": "2024-01-01",
                    "seq": seq,
                }
            )
    got = build_embedding_kpis(pd.DataFrame(rows))
    assert got is not None
    assert got["ec_qa_qq_distance"].isna().all(), "a missing Q2 must not bridge Q1 to Q3"

    mixed = pd.DataFrame(rows)
    mixed.loc[mixed.index[-1], "model"] = "m2"
    bad = build_embedding_kpis(mixed)
    assert bad is not None
    assert bad["ec_qa_coherence_mean"].isna().all(), "mixed embedding models are incomparable"
    missing_model = pd.DataFrame(rows).assign(model=None)
    missing = build_embedding_kpis(missing_model)
    assert missing is not None
    assert missing["ec_qa_coherence_mean"].isna().all(), "missing embedding provenance is incomparable"
    print("\n=== SANITY CHECK: embedding comparability ===")
    print("  missing quarters do not bridge QoQ distance; mixed model provenance yields NaN. Validated.")


def test_embedding_distance_continues_across_symbol_change_for_one_issuer() -> None:
    rows = []
    for ticker, quarter, as_of, shift in (
        ("OLD", "2023Q4", "2023-11-01", 0.0),
        ("NEW", "2024Q1", "2024-02-01", 0.2),
    ):
        for section, tag, vector in (
            ("prepared_remarks", "prepared_remarks", [1.0, shift + 0.1]),
            ("qa", "question", [1.0, shift + 0.2]),
            ("qa", "answer", [0.9, shift + 0.3]),
        ):
            rows.append(
                {
                    "issuer_id": "E1",
                    "ticker": ticker,
                    "quarter": quarter,
                    "as_of": as_of,
                    "section": section,
                    "tag": tag,
                    "exchange_idx": 0,
                    "embedding": vector,
                    "model": EARNINGS_CALL_EMBEDDING_MODEL,
                }
            )
    got = build_embedding_kpis(pd.DataFrame(rows))
    assert got is not None
    newest = got[got["ticker"].eq("NEW")].iloc[0]
    assert pd.notna(newest["ec_qa_qq_distance"])
    assert pd.notna(newest["ec_prep_qq_distance"])

    store = FakeStore()
    store.t["earning_calls_embedding"] = pd.DataFrame(rows).drop(columns="issuer_id")
    identity = pd.DataFrame({"ticker": ["OLD", "NEW"], "quarter": ["2023Q4", "2024Q1"], "issuer_id": ["E1", "E1"]})
    streamed = embedding_kpis_streamed(cast(Context, FakeCtx(store)), identity)
    assert streamed is not None
    streamed_new = streamed[streamed["ticker"].eq("NEW")].iloc[0]
    assert pd.notna(streamed_new["ec_qa_qq_distance"])
    assert pd.notna(streamed_new["ec_prep_qq_distance"])
    print("\n=== SANITY CHECK: issuer-level embedding continuity ===")
    print("  OLD 2023Q4 -> NEW 2024Q1 produces both consecutive-quarter distances for issuer E1. Validated.")


if __name__ == "__main__":
    test_per_turn_split_clean_embed_cache_and_kpis()
    test_embedded_turns_are_the_split_turns_on_hand_labelled_calls()
    test_force_reembed_drops_stale_turns()
