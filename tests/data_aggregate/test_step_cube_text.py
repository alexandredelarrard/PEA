"""Public text-step integration on real transcript fixtures and isolated storage."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from src.constants.constants import EARNINGS_CALL_FEATURES
from src.context import Context
from src.data_aggregate.transformers import step_cube_text as step_text
from src.data_aggregate.transformers.step_cube_text import StepCubeText
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from tests.fixtures.earnings_call_rows import FIXTURE_PATHS, fixture_paragraphs


class ReplayStore:
    """SQLite for normal tables, an array-preserving stand-in at the vector storage boundary."""

    def __init__(self, base: DataStore) -> None:
        self.base = base
        self.vector_rows = pd.DataFrame()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def _filtered(self, where: dict[str, Any] | None = None, since: Any = None, until: Any = None) -> pd.DataFrame:
        rows = self.vector_rows
        if rows.empty:
            return rows
        for column, value in (where or {}).items():
            rows = rows[rows[column].isin(value)] if isinstance(value, list | tuple | set) else rows[rows[column].eq(value)]
        if since is not None:
            rows = rows[pd.to_datetime(rows.as_of).ge(pd.Timestamp(since))]
        if until is not None:
            rows = rows[pd.to_datetime(rows.as_of).lt(pd.Timestamp(until).normalize() + pd.Timedelta(days=1))]
        return rows

    def load(
        self,
        table: Any,
        columns: Any = None,
        *,
        where: dict[str, Any] | None = None,
        since: Any = None,
        until: Any = None,
        order_by: Any = None,
        optional: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame | None:
        if table != Tables.earning_calls_embedding:
            return self.base.load(table, columns=columns, where=where, since=since, until=until, order_by=order_by, optional=optional, **kwargs)
        rows = self._filtered(where, since, until)
        if rows.empty:
            if optional:
                return None
            raise LookupError("empty vector fixture")
        if order_by is not None:
            rows = rows.sort_values(order_by, kind="stable")
        return (rows[list(columns)] if columns is not None else rows).copy().reset_index(drop=True)

    def save(self, table: Any, rows: pd.DataFrame, **kwargs: Any) -> int:
        if table != Tables.earning_calls_embedding:
            return self.base.save(table, rows, **kwargs)
        self.vector_rows = pd.concat([self.vector_rows, rows], ignore_index=True).drop_duplicates(["ticker", "quarter", "seq"], keep="last")
        return len(rows)

    def delete(self, table: Any, where: dict[str, Any]) -> int:
        if table != Tables.earning_calls_embedding:
            return self.base.delete(table, where)
        drop = self._filtered(where).index
        self.vector_rows = self.vector_rows.drop(index=drop)
        return len(drop)

    def distinct(self, table: Any, column: str, *, where: dict[str, Any] | None = None, **kwargs: Any) -> list[Any]:
        if table != Tables.earning_calls_embedding:
            return self.base.distinct(table, column, where=where, **kwargs)
        rows = self._filtered(where)
        return [] if rows.empty else rows[column].dropna().unique().tolist()

    def columns(self, table: Any) -> list[str]:
        return list(self.vector_rows) if table == Tables.earning_calls_embedding else self.base.columns(table)

    def exists(self, table: Any) -> bool:
        return not self.vector_rows.empty if table == Tables.earning_calls_embedding else self.base.exists(table)


def seeded_case(base: DataStore, stems: list[str]) -> tuple[Context, DictConfig, ReplayStore]:
    store = ReplayStore(base)
    source = pd.concat([fixture_paragraphs(stem) for stem in stems], ignore_index=True)
    store.save(Tables.earnings_call_sections, source)
    tickers = sorted(source.ticker.unique())
    calendar = pd.bdate_range("2020-01-01", "2027-03-01", name="date")
    prices = pd.MultiIndex.from_product([calendar, tickers], names=["date", "ticker"]).to_frame(index=False)
    store.replace(Tables.cube_part_prices, prices.assign(close_split=100.0))
    ciks = [str(i + 1).zfill(10) for i in range(len(tickers))]
    store.replace(Tables.sp500_tickers, pd.DataFrame({"ticker": tickers, "cik": ciks}))
    store.replace(
        Tables.entity_lineage,
        pd.DataFrame({"cik": ciks, "symbol": tickers, "entity_id": ciks, "role": "cik_window", "valid_from": "2000-01-01", "valid_to": None}),
    )
    store.replace(
        Tables.symbol_tenure,
        pd.DataFrame(
            {"symbol": tickers, "issuer_cik": ciks, "valid_from": "2000-01-01", "valid_to": None, "source": "manual", "evidence_period": "fixture"}
        ),
    )
    config = OmegaConf.create(
        {
            "build_cube": {
                "earnings_calls": {"availability_lag_days": 2, "sentiment_model": "yiyanghkust/finbert-tone", "refresh_calendar_days": 30}
            },
            "gpt": {"embedding": {"model": "text-embedding-3-small", "batch_size": 128, "max_chars": 28000}},
        }
    )
    context = cast(Context, SimpleNamespace(store=store, log=logging.getLogger("test.text"), config=config, config_dir=Path("configs")))
    return context, config, store


def keyword_vectors(texts: list[str], **_kwargs: Any) -> np.ndarray:
    words = ("growth", "revenue", "margin", "profit", "cost", "demand", "capital", "cash")
    return np.asarray([[1.0, *[text.lower().count(word) + 1.0 for word in words]] for text in texts])


def test_public_run_prepares_each_selected_call_once(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    context, config, store = seeded_case(sqlite_store, [path.stem for path in FIXTURE_PATHS])
    engine = MagicMock()
    engine.score_texts.side_effect = lambda texts: [{"pos": 0.6, "neg": 0.1, "neu": 0.3} for _ in texts]
    monkeypatch.setattr(step_text, "get_sentiment_engine", lambda model: engine)
    monkeypatch.setattr(step_text, "openai_api_key", lambda: "stub")
    monkeypatch.setattr(step_text, "embed_texts", keyword_vectors)
    with patch.object(store, "load", wraps=store.load) as reads:
        StepCubeText(context, config).run(full=True)
    payload = [
        call
        for call in reads.call_args_list
        if call.args[0] == Tables.earnings_call_sections and "content" in (call.kwargs.get("columns") or (call.args[1] if len(call.args) > 1 else []))
    ]
    assert len(payload) == 1, f"expected shared one-time paragraph load, got {len(payload)}"
    print("SANITY: actual public run loads source paragraphs once for both model stages.")


def configure_models(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock, MagicMock]:
    engine = MagicMock()
    engine.score_texts.side_effect = lambda texts: [{"pos": 0.6, "neg": 0.1, "neu": 0.3} for _ in texts]
    factory = MagicMock(return_value=engine)
    embedding = MagicMock(side_effect=keyword_vectors)
    monkeypatch.setattr(step_text, "get_sentiment_engine", factory)
    monkeypatch.setattr(step_text, "openai_api_key", lambda: "stub")
    monkeypatch.setattr(step_text, "embed_texts", embedding)
    return factory, engine, embedding


def prepare_step(context: Context, config: DictConfig) -> tuple[StepCubeText, Any, pd.DataFrame]:
    step = StepCubeText(context, config)
    calendar = pd.DatetimeIndex(pd.to_datetime(context.store.distinct(Tables.cube_part_prices, "date"))).sort_values()
    plan = step.plan_window(calendar, full=True)
    return step, plan, step.prepare_sections(plan)


def test_partial_calls_are_complete_and_warm_run_skips_models(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    context, config, store = seeded_case(sqlite_store, ["COIN_2026Q2", "PSKY_2025Q2"])
    factory, engine, embedding = configure_models(monkeypatch)
    step = StepCubeText(context, config)
    step.run(full=True)
    scores = store.load(Tables.earnings_call_sentiment)
    assert scores is not None and len(scores) == 4
    absent = scores[scores["n_words"].eq(0)]
    assert len(absent) == 2 and absent[["sent_pos", "sent_neg", "sent_neu", "uncertainty_ratio"]].isna().all().all()
    panel = store.load(Tables.cube_part_text)
    assert panel is not None and panel["f_ec_tone"].notna().any()
    assert panel["f_ec_qa_gap"].isna().all()
    assert engine.score_texts.call_count == 2 and embedding.call_count == 2
    factory.reset_mock()
    engine.reset_mock()
    embedding.reset_mock()
    with patch.object(store, "load", wraps=store.load) as reads, patch.object(step_text, "split_call", wraps=step_text.split_call) as splits:
        StepCubeText(context, config).run()
    assert not factory.called and not engine.score_texts.called and not embedding.called and not splits.called
    payload = [
        call for call in reads.call_args_list if call.args[0] == Tables.earnings_call_sections and "content" in (call.kwargs.get("columns") or [])
    ]
    assert not payload
    print("SANITY: real prepared-only/QA-only calls save genuine NULL absence; warm independent stages do zero model or paragraph work.")


@pytest.mark.parametrize("missing", ["sentiment", "embedding"])
def test_optional_stage_leaves_other_features_available(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    context, config, store = seeded_case(sqlite_store, ["ADP_2026Q3"])
    factory, _, embedding = configure_models(monkeypatch)
    if missing == "sentiment":
        factory.return_value = None
    else:
        monkeypatch.setattr(step_text, "openai_api_key", lambda: None)
    StepCubeText(context, config).run(full=True)
    panel = store.load(Tables.cube_part_text)
    assert panel is not None
    assert panel["f_ec_qa_coherence_mean" if missing == "sentiment" else "f_ec_tone"].notna().any()
    assert panel["f_ec_tone" if missing == "sentiment" else "f_ec_qa_coherence_mean"].isna().all()
    step = StepCubeText(context, config)
    plan = step.plan_window(pd.bdate_range("2020-01-01", "2027-03-01"), full=False)
    assert bool(plan.sentiment_work) == (missing == "sentiment")
    assert bool(plan.embedding_work) == (missing == "embedding")
    assert embedding.called == (missing == "sentiment")
    print(f"SANITY: unavailable {missing} remains retryable while the independently supported family is saved.")


@pytest.mark.parametrize("broken", ["gap", "mixed_model_tail"])
@pytest.mark.parametrize("full", [False, True])
def test_incomplete_embeddings_stay_pending_without_emitting_features(
    sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, broken: str, full: bool
) -> None:
    context, config, store = seeded_case(sqlite_store, ["ADP_2026Q3"])
    factory, _, embedding = configure_models(monkeypatch)
    StepCubeText(context, config).run(full=True)
    before = store.load(Tables.cube_part_text)
    assert before is not None and before["f_ec_qa_coherence_mean"].notna().all()
    if broken == "gap":
        answer = store.vector_rows[store.vector_rows["tag"].eq("answer")].iloc[0]
        store.delete(Tables.earning_calls_embedding, {"ticker": "ADP", "quarter": "2026Q3", "seq": int(answer.seq)})
    else:
        store.vector_rows.loc[store.vector_rows["seq"].eq(store.vector_rows["seq"].max()), "model"] = "stale-model:speaker-clean-v2"
    api_key = step_text.openai_api_key
    monkeypatch.setattr(step_text, "openai_api_key", lambda: None)
    factory.reset_mock()
    embedding.reset_mock()
    step = StepCubeText(context, config)
    assert ("ADP", "2026Q3") in step.plan_window(step_text.load_trading_calendar(store), full=full).embedding_work
    step.run(full=full)
    after = store.load(Tables.cube_part_text)
    assert after is not None
    cutoff = pd.to_datetime(before["date"]).max() - pd.Timedelta(days=30)
    rewritten = after if full else after[pd.to_datetime(after["date"]).ge(cutoff)]
    assert not rewritten.empty and rewritten["f_ec_tone"].notna().all()
    assert rewritten["f_ec_qa_coherence_mean"].isna().all()
    assert not factory.called and not embedding.called
    monkeypatch.setattr(step_text, "openai_api_key", api_key)
    StepCubeText(context, config).run(full=full)
    repaired = store.load(Tables.cube_part_text)
    assert repaired is not None and embedding.call_count == 1 and not factory.called
    pd.testing.assert_frame_equal(before, repaired, check_exact=True)
    print(
        f"SANITY: {broken} vectors remain pending with NULL rewritten embedding features; API recovery restores exact output without sentiment work (full={full})."
    )


def test_stage_order_and_configured_models(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    context, config, store = seeded_case(sqlite_store, ["ADP_2026Q3"])
    config.build_cube.earnings_calls.sentiment_model = "fixture/alternate-tone"
    config.gpt.embedding.model = "fixture/alternate-embedding"
    factory, _, embedding = configure_models(monkeypatch)
    step = StepCubeText(context, config)
    events = []
    for name in ("prepare_sections", "run_sentiments", "run_embeddings", "run_features"):
        original = getattr(step, name)

        def invoke(*args: Any, _name: str = name, _original: Any = original, **kwargs: Any) -> Any:
            events.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(step, name, invoke)
    original_load = store.load

    def read(table: Any, *args: Any, **kwargs: Any) -> Any:
        if table == Tables.cube_part_prices:
            assert events[-1] == "run_features"
        return original_load(table, *args, **kwargs)

    monkeypatch.setattr(store, "load", read)
    step.run(full=True)
    assert events == ["prepare_sections", "run_sentiments", "run_embeddings", "run_features"]
    factory.assert_called_once_with("fixture/alternate-tone")
    assert embedding.call_args.kwargs == {"model": "fixture/alternate-embedding", "batch_size": 128, "max_chars": 28000}
    scores = store.load(Tables.earnings_call_sentiment)
    assert scores is not None and scores["model"].eq("fixture/alternate-tone:speaker-clean-v2").all()
    assert store.vector_rows["model"].eq("fixture/alternate-embedding:speaker-clean-v2").all()
    print("SANITY: direct stage order, late price read and real model YAML values reach their factory/service and cache identities.")


def test_crash_after_model_save_reuses_outputs(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    context, config, store = seeded_case(sqlite_store, ["ADP_2026Q3"])
    factory, _, embedding = configure_models(monkeypatch)
    step = StepCubeText(context, config)

    def interrupted(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("before feature save")

    monkeypatch.setattr(step, "run_features", interrupted)
    with pytest.raises(RuntimeError, match="before feature save"):
        step.run(full=True)
    assert store.load(Tables.earnings_call_sentiment) is not None and not store.vector_rows.empty
    factory.reset_mock()
    embedding.reset_mock()
    StepCubeText(context, config).run()
    assert not factory.called and not embedding.called and store.load(Tables.cube_part_text) is not None
    print("SANITY: fresh-context retry after both independent model saves reuses outputs and builds the normal feature range.")


def test_old_correction_needs_explicit_full_features(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    context, config, store = seeded_case(sqlite_store, ["ADP_2026Q3"])
    factory, engine, embedding = configure_models(monkeypatch)
    StepCubeText(context, config).run(full=True)
    before = store.load(Tables.cube_part_text)
    assert before is not None
    cutoff = pd.to_datetime(before["date"]).max() - pd.Timedelta(days=30)
    store.delete(Tables.earnings_call_sentiment, {"ticker": "ADP"})
    engine.score_texts.side_effect = lambda texts: [{"pos": 0.8, "neg": 0.1, "neu": 0.1} for _ in texts]
    with patch.object(store, "load", wraps=store.load) as reads:
        StepCubeText(context, config).run()
    after = store.load(Tables.cube_part_text)
    assert after is not None

    def prefix(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[pd.to_datetime(frame["date"]).lt(cutoff)].reset_index(drop=True)

    pd.testing.assert_frame_equal(prefix(before), prefix(after), check_exact=True)
    assert after.loc[pd.to_datetime(after["date"]).ge(cutoff), "f_ec_tone"].eq(float(np.float32(0.7))).all()
    price_read = [call for call in reads.call_args_list if call.args[0] == Tables.cube_part_prices]
    assert len(price_read) == 1 and price_read[0].kwargs["since"] == cutoff
    vector_read = [
        call for call in reads.call_args_list if call.args[0] == Tables.earning_calls_embedding and "embedding" in (call.kwargs.get("columns") or [])
    ]
    assert len(vector_read) == 1 and vector_read[0].kwargs["date_col"] == "as_of"
    factory.reset_mock()
    embedding.reset_mock()
    StepCubeText(context, config).run(full=True)
    repaired = store.load(Tables.cube_part_text)
    assert repaired is not None and repaired["f_ec_tone"].eq(float(np.float32(0.7))).all()
    assert not factory.called and not embedding.called
    print("SANITY: exact 30-calendar-day tail leaves older rows untouched; explicit full repairs them without repeating paid inference.")


def test_invalidated_empty_call_clears_tail_then_full(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data_extract.utils.behavioral.utils_earnings_call_cache import invalidate_earnings_call_derivatives

    context, config, store = seeded_case(sqlite_store, ["ADP_2026Q3"])
    factory, _, embedding = configure_models(monkeypatch)
    StepCubeText(context, config).run(full=True)
    before = store.load(Tables.cube_part_text)
    assert before is not None
    cutoff = pd.to_datetime(before["date"]).max() - pd.Timedelta(days=30)
    rows = store.load(Tables.earnings_call_sections)
    assert rows is not None
    invalidate_earnings_call_derivatives(context, rows)
    store.replace(Tables.earnings_call_sections, rows.assign(content="Thank you."))
    factory.reset_mock()
    embedding.reset_mock()
    StepCubeText(context, config).run()
    tail = store.load(Tables.cube_part_text)
    assert tail is not None and pd.to_datetime(tail["date"]).lt(cutoff).all()
    StepCubeText(context, config).run(full=True)
    assert store.load(Tables.cube_part_text, optional=True) is None
    assert not factory.called and not embedding.called
    print("SANITY: unusable source clears stale ordinary-tail output, then explicit empty full rebuild clears all history.")


@pytest.mark.parametrize("frontier", ["2026-01-01", "2026-03-01", "2026-05-25", "2026-10-02"])
def test_output_overlap_is_calendar_days(sqlite_store: DataStore, frontier: str) -> None:
    context, config, store = seeded_case(sqlite_store, ["ADP_2026Q3"])
    rows = pd.DataFrame({"date": [frontier], "ticker": ["ADP"], **{f"f_{name}": [0.0] for name in EARNINGS_CALL_FEATURES}})
    store.replace(Tables.cube_part_text, rows)
    plan = StepCubeText(context, config).plan_window(pd.bdate_range("2020-01-01", "2027-03-01"), full=False)
    assert plan.window.refresh_from == pd.Timestamp(frontier) - pd.Timedelta(days=30)
    assert plan.window.since == plan.window.refresh_from
    print(f"SANITY: frontier {frontier} rewrites inclusively from exactly 30 calendar days earlier.")


def test_missed_runs_equal_full_without_repeating_models(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    context, config, store = seeded_case(sqlite_store, ["ADP_2026Q3"])
    factory, _, embedding = configure_models(monkeypatch)
    prices = store.load(Tables.cube_part_prices)
    source = store.load(Tables.earnings_call_sections)
    assert prices is not None and source is not None
    initial_edge = pd.Timestamp(source["as_of"].iloc[0]) + pd.offsets.BDay(8)
    store.replace(Tables.cube_part_prices, prices[pd.to_datetime(prices["date"]).le(initial_edge)])
    StepCubeText(context, config).run(full=True)
    store.replace(Tables.cube_part_prices, prices)
    factory.reset_mock()
    embedding.reset_mock()
    StepCubeText(context, config).run()
    incremental = store.load(Tables.cube_part_text)
    StepCubeText(context, config).run(full=True)
    full = store.load(Tables.cube_part_text)
    assert incremental is not None and full is not None
    pd.testing.assert_frame_equal(
        incremental.sort_values(["date", "ticker"]).reset_index(drop=True),
        full.sort_values(["date", "ticker"]).reset_index(drop=True),
        check_exact=True,
    )
    assert not factory.called and not embedding.called
    print("SANITY: after a long outage, stored-frontier overlap catches every missing session and equals full output without model work.")


def test_unmapped_answer_view_never_calls_embedding_api(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    context, config, store = seeded_case(sqlite_store, ["ADP_2026Q3"])
    _, engine, embedding = configure_models(monkeypatch)
    words = " ".join(["growth"] * 140)
    rows = pd.DataFrame(
        {
            "ticker": ["ADP"] * 3,
            "quarter": ["2026Q3"] * 3,
            "paragraph": [1, 2, 3],
            "as_of": ["2026-04-29"] * 3,
            "speaker": ["Operator", "Jane Analyst", "Chief Executive"],
            "content": ["Our first question comes from Jane Analyst.", "Can you explain growth please?", words],
        }
    )
    store.replace(Tables.earnings_call_sections, rows)
    step, plan, prepared = prepare_step(context, config)
    assert len(prepared) == 1 and prepared["turns"].iloc[0] == []
    step.run_sentiments(prepared, plan)
    step.run_embeddings(prepared, plan)
    assert engine.score_texts.call_count == 1 and not embedding.called
    assert store.vector_rows.empty
    print("SANITY: usable answer text without mapped exchanges is scored; no API call or fake vector is created.")
