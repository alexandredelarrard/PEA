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
    keys = {(str(row["ticker"]), str(row["quarter"])) for row in rows}
    got = build_embedding_kpis(pd.DataFrame(rows), keys)
    assert got is not None
    assert got["ec_qa_qq_distance"].isna().all(), "a missing Q2 must not bridge Q1 to Q3"

    mixed = pd.DataFrame(rows)
    mixed.loc[mixed.index[-1], "model"] = "m2"
    bad = build_embedding_kpis(mixed, keys)
    assert bad is not None
    assert bad["ec_qa_coherence_mean"].isna().all(), "mixed embedding models are incomparable"
    missing_model = pd.DataFrame(rows).assign(model=None)
    missing = build_embedding_kpis(missing_model, keys)
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
                    "model": "fixture/embedding:speaker-clean-v2",
                }
            )
    keys = {(str(row["ticker"]), str(row["quarter"])) for row in rows}
    got = build_embedding_kpis(pd.DataFrame(rows), keys)
    assert got is not None
    newest = got[got["ticker"].eq("NEW")].iloc[0]
    assert pd.notna(newest["ec_qa_qq_distance"])
    assert pd.notna(newest["ec_prep_qq_distance"])

    print("\n=== SANITY CHECK: issuer-level embedding continuity ===")
    print("  OLD 2023Q4 -> NEW 2024Q1 produces both consecutive-quarter distances for issuer E1. Validated.")
