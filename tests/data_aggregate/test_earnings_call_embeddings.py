"""Embedding stage persistence and unchanged vector mathematics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.text.earnings_call_embeddings import build_embedding_kpis
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.text.earnings_call_split import split_call
from tests.data_aggregate.test_step_cube_text import configure_models, prepare_step, seeded_case
from tests.fixtures.earnings_call_rows import FIXTURE_PATHS, load_fixture


def test_embedded_turns_are_the_split_turns_on_hand_labelled_calls(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    context, config, store = seeded_case(sqlite_store, [path.stem for path in FIXTURE_PATHS])
    _, _, embedding = configure_models(monkeypatch)
    step, plan, prepared = prepare_step(context, config)
    step.run_embeddings(prepared, plan)
    fields = ["section", "tag", "person", "text", "exchange_idx", "answer_idx"]
    for path in FIXTURE_PATHS:
        fixture = load_fixture(path.stem)
        quarter = f"{fixture['fiscal_year']}Q{fixture['fiscal_quarter']}"
        rows = store.vector_rows[store.vector_rows["ticker"].eq(fixture["symbol"]) & store.vector_rows["quarter"].eq(quarter)].sort_values("seq")
        expected = split_call(fixture["paragraphs"]).turns
        assert rows[fields].to_dict("records") == expected
        assert rows["seq"].tolist() == list(range(len(expected)))
        analysts = rows["person"].isin(fixture["truth"]["analysts"])
        assert rows.loc[analysts, "tag"].eq("question").all()
        assert not rows.loc[rows["person"].isin(fixture["truth"]["management"]), "tag"].eq("question").any()
    assert len(store.vector_rows) == 466 and embedding.call_count == len(FIXTURE_PATHS)
    embedding.reset_mock()
    step, plan, prepared = prepare_step(context, config)
    step.run_embeddings(prepared, plan)
    assert not embedding.called
    print("SANITY: 466 stored real turns preserve canonical roles/order/cleaning; complete embeddings make zero new API calls.")


def test_stale_model_and_seq_tail_are_reconciled_without_force(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    context, config, store = seeded_case(sqlite_store, ["ADP_2026Q3", "AMZN_2025Q3"])
    _, _, embedding = configure_models(monkeypatch)
    step, plan, prepared = prepare_step(context, config)
    step.run_embeddings(prepared, plan)
    initial = len(store.vector_rows)
    store.save(Tables.earning_calls_embedding, store.vector_rows.iloc[[0]].assign(seq=999, model="old-model"))
    embedding.reset_mock()
    step, plan, prepared = prepare_step(context, config)
    step.run_embeddings(prepared, plan)
    assert embedding.call_count == 1 and len(store.vector_rows) == initial
    assert not store.vector_rows["seq"].eq(999).any()
    store.vector_rows["model"] = "text-embedding-3-small"
    embedding.reset_mock()
    step, plan, prepared = prepare_step(context, config)
    step.run_embeddings(prepared, plan)
    assert embedding.call_count == 2 and store.vector_rows["model"].eq(step._embedding_cache).all()
    print("SANITY: stale provenance/observable seq gaps select affected calls; normal writes remove tails without a force/test input.")


@pytest.mark.parametrize("kind", ["count", "zero", "nan", "dimension"])
def test_invalid_vectors_remain_retryable(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    context, config, store = seeded_case(sqlite_store, ["ADP_2026Q3"])
    _, _, embedding = configure_models(monkeypatch)

    def invalid(texts: list[str], **_kwargs: object) -> np.ndarray:
        if kind == "count":
            return np.ones((len(texts) - 1, 9))
        if kind == "zero":
            return np.zeros((len(texts), 9))
        if kind == "nan":
            return np.full((len(texts), 9), np.nan)
        return np.ones((len(texts), 0))

    embedding.side_effect = invalid
    step, plan, prepared = prepare_step(context, config)
    with pytest.raises(ValueError, match="Embedding output"):
        step.run_embeddings(prepared, plan)
    assert store.vector_rows.empty
    print(f"SANITY: {kind} vector output cannot create a falsely complete call.")


def test_embedding_kpis_require_consecutive_quarters_and_consistent_provenance() -> None:
    rows = []
    for quarter, value, model in (("2024Q1", [1.0, 0.0], "m1"), ("2024Q3", [0.0, 1.0], "m1")):
        for seq, tag in enumerate(("question", "answer")):
            rows.append(
                {
                    "ticker": "AAA",
                    "issuer_id": "E1",
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
    got = build_embedding_kpis([pd.DataFrame(rows)])
    assert got is not None
    assert got["ec_qa_qq_distance"].isna().all(), "a missing Q2 must not bridge Q1 to Q3"

    mixed = pd.DataFrame(rows)
    mixed.loc[mixed.index[-1], "model"] = "m2"
    bad = build_embedding_kpis([mixed])
    assert bad is not None
    assert bad["ec_qa_coherence_mean"].isna().all(), "mixed embedding models are incomparable"
    missing_model = pd.DataFrame(rows).assign(model=None)
    missing = build_embedding_kpis([missing_model])
    assert missing is not None
    assert missing["ec_qa_coherence_mean"].isna().all(), "missing embedding provenance is incomparable"
    print("\n=== SANITY CHECK: embedding comparability ===")
    print("  missing quarters do not bridge QoQ distance; mixed model provenance yields NaN. Validated.")


def test_question_only_distance_ignores_answer_changes() -> None:
    rows = []
    for quarter, date, answer in (("2024Q1", "2024-02-01", [0.0, 1.0]), ("2024Q2", "2024-05-01", [-1.0, 1.0])):
        for tag, vector in (("question", [1.0, 0.0]), ("answer", answer)):
            rows.append(
                dict(ticker="A", issuer_id="E1", quarter=quarter, as_of=date, section="qa", tag=tag, exchange_idx=0, embedding=vector, model="m")
            )
    actual = build_embedding_kpis([pd.DataFrame(rows)]).set_index("quarter")
    assert actual.loc["2024Q2", "ec_questions_only_qq_distance"] == 0.0
    assert actual.loc[["2024Q2"], "ec_qa_qq_distance"].gt(0).all()
    assert pd.isna(actual.loc["2024Q1", "ec_questions_only_qq_distance"])
    print("SANITY: unchanged questions have zero distance while answer changes move combined Q&A; first quarter remains null.")


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
                    "model": "fixture/embedding:speaker-clean-v2",
                }
            )
    got = build_embedding_kpis([pd.DataFrame(rows)])
    assert got is not None
    newest = got[got["ticker"].eq("NEW")].iloc[0]
    assert pd.notna(newest["ec_qa_qq_distance"])
    assert pd.notna(newest["ec_prep_qq_distance"])

    print("\n=== SANITY CHECK: issuer-level embedding continuity ===")
    print("  OLD 2023Q4 -> NEW 2024Q1 produces both consecutive-quarter distances for issuer E1. Validated.")


@pytest.mark.parametrize("chunksize", [1, 2, 5, 17])
def test_embedding_kpis_preserve_calls_and_issuer_comparisons_across_chunks(chunksize: int) -> None:
    rows = []
    for ticker, quarter, date, shift in (
        ("OLD", "2023Q4", "2023-11-01", 0.0),
        ("NEW", "2024Q1", "2024-02-01", 0.2),
        ("NEW", "2024Q2", "2024-05-01", 0.4),
    ):
        for seq, (section, tag, exchange, vector) in enumerate(
            [
                ("prepared_remarks", "prepared", -1, [1.0, shift + 0.1]),
                ("qa", "question", 0, [1.0, shift + 0.2]),
                ("qa", "answer", 0, [0.9, shift + 0.3]),
                ("qa", "question", 1, [0.8, shift + 0.4]),
                ("qa", "answer", 1, [0.7, shift + 0.5]),
            ]
        ):
            rows.append(
                dict(
                    ticker=ticker,
                    quarter=quarter,
                    as_of=date,
                    issuer_id="E1",
                    seq=seq,
                    section=section,
                    tag=tag,
                    exchange_idx=exchange,
                    embedding=vector,
                    model="m",
                )
            )
    frame = pd.DataFrame(rows).sort_values(["ticker", "quarter", "seq"]).reset_index(drop=True)
    expected = build_embedding_kpis(iter([frame])).sort_values(["ticker", "quarter"]).reset_index(drop=True)
    actual = build_embedding_kpis(frame.iloc[start : start + chunksize] for start in range(0, len(frame), chunksize))
    pd.testing.assert_frame_equal(expected, actual.sort_values(["ticker", "quarter"]).reset_index(drop=True), check_exact=True)
    newest = actual[actual["quarter"].eq("2024Q1")].iloc[0]
    assert newest[["ec_qa_qq_distance", "ec_prep_qq_distance", "ec_questions_only_qq_distance", "ec_qa_coherence_mean"]].notna().all()
    print(f"SANITY: {chunksize}-row chunks preserve complete exchanges and OLD -> NEW issuer comparisons exactly.")


@pytest.mark.parametrize("invalid", ["model", "dimension", "nan", "zero"])
def test_invalid_call_is_rejected_across_chunk_boundaries(invalid: str) -> None:
    rows = []
    for quarter, date in (("2024Q1", "2024-02-01"), ("2024Q2", "2024-05-01")):
        for seq, (tag, exchange) in enumerate((("question", 0), ("answer", 0), ("question", 1), ("answer", 1))):
            rows.append(
                dict(
                    ticker="A",
                    issuer_id="E1",
                    quarter=quarter,
                    as_of=date,
                    seq=seq,
                    section="qa",
                    tag=tag,
                    exchange_idx=exchange,
                    model="m",
                    embedding=[1.0, 1.0],
                )
            )
    if invalid == "model":
        rows[3]["model"] = "other"
    else:
        rows[3]["embedding"] = {"dimension": [1.0], "nan": [np.nan, 1.0], "zero": [0.0, 0.0]}[invalid]
    frame = pd.DataFrame(rows)
    actual = build_embedding_kpis(frame.iloc[start : start + 2] for start in range(0, len(frame), 2))
    actual = actual.set_index("quarter")
    assert pd.isna(actual.loc["2024Q1", "ec_qa_coherence_mean"])
    assert actual.loc["2024Q2", "ec_qa_coherence_mean"] == 1.0
    assert actual["ec_qa_qq_distance"].isna().all()
    print(f"SANITY: {invalid} in a later chunk invalidates the whole call and its comparison; the next call's coherence survives.")
