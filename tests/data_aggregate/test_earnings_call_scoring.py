"""Independent sentiment stage completion and retry checks on real transcripts."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.transformers.step_cube_text import StepCubeText
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from tests.data_aggregate.test_step_cube_text import configure_models, prepare_step, seeded_case
from tests.fixtures.earnings_call_rows import FIXTURE_PATHS


def test_scoring_reads_each_table_once(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    context, config, store = seeded_case(sqlite_store, [path.stem for path in FIXTURE_PATHS])
    factory, engine, _ = configure_models(monkeypatch)
    with patch.object(store, "load", wraps=store.load) as reads:
        step, plan, prepared = prepare_step(context, config)
        step.run_sentiments(prepared, plan)
    source = [call for call in reads.call_args_list if call.args[0] == Tables.earnings_call_sections]
    assert len(source) == 2
    assert sum("content" in call.kwargs["columns"] for call in source) == 1
    scores = store.load(Tables.earnings_call_sentiment)
    assert scores is not None and len(scores) == 3 * len(FIXTURE_PATHS)
    factory.assert_called_once_with(config.build_cube.earnings_calls.sentiment_model)
    engine.reset_mock()
    factory.reset_mock()
    step, plan, prepared = prepare_step(context, config)
    step.run_sentiments(prepared, plan)
    assert not factory.called and not engine.score_texts.called
    print("SANITY: one compact source read plus one shared payload scores all eligible real calls; complete sentiment stage skips inference.")


def test_scoring_propagates_inference_failure(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    context, config, store = seeded_case(sqlite_store, ["ADP_2026Q3"])
    _, engine, _ = configure_models(monkeypatch)
    engine.score_texts.side_effect = RuntimeError("inference failed")
    step, plan, prepared = prepare_step(context, config)
    with pytest.raises(RuntimeError, match="inference failed"):
        step.run_sentiments(prepared, plan)
    assert store.load(Tables.earnings_call_sentiment, optional=True) is None
    print("SANITY: inference failure propagates without fabricating completion or saving a partial call.")


def test_scoring_resumes_saved_tickers_and_partial_calls(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    context, config, store = seeded_case(sqlite_store, ["ADP_2026Q3", "AMZN_2025Q3"])
    _, engine, _ = configure_models(monkeypatch)
    engine.score_texts.side_effect = [[{"pos": 0.6, "neg": 0.1, "neu": 0.3}] * 3, RuntimeError("interrupted")]
    step, plan, prepared = prepare_step(context, config)
    with pytest.raises(RuntimeError, match="interrupted"):
        step.run_sentiments(prepared, plan)
    scores = store.load(Tables.earnings_call_sentiment)
    assert scores is not None and len(scores) == 3 and scores["ticker"].eq("ADP").all()
    store.delete(Tables.earnings_call_sentiment, {"ticker": "ADP", "tag": "answers"})
    engine.score_texts.side_effect = lambda texts: [{"pos": 0.6, "neg": 0.1, "neu": 0.3} for _ in texts]
    engine.reset_mock()
    step, plan, prepared = prepare_step(context, config)
    step.run_sentiments(prepared, plan)
    assert [len(call.args[0]) for call in engine.score_texts.call_args_list] == [1, 3]
    scores = store.load(Tables.earnings_call_sentiment)
    assert scores is not None and len(scores) == 6
    print("SANITY: saved calls survive interruption; a valid partner tag is reused while only the missing section is scored.")


@pytest.mark.parametrize(
    "output", [[], [None, None, None], [{"pos": np.nan, "neg": 0.1, "neu": 0.3}] * 3, [{"pos": 0.9, "neg": 0.9, "neu": 0.3}] * 3]
)
def test_invalid_outputs_do_not_establish_completion(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, output: list) -> None:
    context, config, store = seeded_case(sqlite_store, ["ADP_2026Q3"])
    _, engine, _ = configure_models(monkeypatch)
    engine.score_texts.side_effect = lambda _texts: output
    step, plan, prepared = prepare_step(context, config)
    with pytest.raises(ValueError, match="Sentiment"):
        step.run_sentiments(prepared, plan)
    assert store.load(Tables.earnings_call_sentiment, optional=True) is None
    print("SANITY: invalid score count/probabilities remain absent and retryable, distinct from unavailable text.")


def test_legacy_answers_are_reused_and_only_questions_inferred(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    context, config, store = seeded_case(sqlite_store, ["ADP_2026Q3"])
    factory, engine, embedding = configure_models(monkeypatch)
    StepCubeText(context, config).run(full=True)
    original = store.load(Tables.earnings_call_sentiment)
    assert original is not None
    legacy = original[original["tag"].ne("questions")].replace({"tag": {"answers": "qa"}}).assign(uncertainty_ratio=0.123456)
    store.drop(Tables.earnings_call_sentiment)
    store.replace(Tables.earnings_call_sentiment, legacy)
    engine.reset_mock()
    embedding.reset_mock()
    StepCubeText(context, config).run(full=True)
    assert engine.score_texts.call_count == 1 and len(engine.score_texts.call_args.args[0]) == 1
    assert not embedding.called
    migrated = store.load(Tables.earnings_call_sentiment)
    assert migrated is not None and set(migrated["tag"]) == {"prepared_remarks", "questions", "answers"}
    partner = ["tag", "sent_pos", "sent_neg", "sent_neu", "n_words", "model"]
    pd.testing.assert_frame_equal(
        original[original["tag"].ne("questions")][partner].sort_values("tag").reset_index(drop=True),
        migrated[migrated["tag"].ne("questions")][partner].sort_values("tag").reset_index(drop=True),
        check_exact=True,
    )
    store.save(Tables.earnings_call_sentiment, legacy[legacy["tag"].eq("qa")].assign(sent_pos=0.0, sent_neg=1.0, sent_neu=0.0))
    factory.reset_mock()
    StepCubeText(context, config).run()
    assert not factory.called
    panel = store.load(Tables.cube_part_text)
    assert panel is not None and panel["f_ec_answers_tone"].eq(0.5).all()
    print(
        "SANITY: valid legacy management scores survive exactly; only questions inferred; canonical answers override leftover legacy rows without new models."
    )


def test_pending_questions_preserve_other_role_features(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    context, config, store = seeded_case(sqlite_store, ["ADP_2026Q3"])
    factory, _, embedding = configure_models(monkeypatch)
    StepCubeText(context, config).run(full=True)
    store.delete(Tables.earnings_call_sentiment, {"ticker": "ADP", "tag": "questions"})
    before = store.load(Tables.earnings_call_sentiment)
    factory.return_value = None
    embedding.reset_mock()
    StepCubeText(context, config).run(full=True)
    after = store.load(Tables.earnings_call_sentiment)
    assert before is not None and after is not None
    pd.testing.assert_frame_equal(before, after, check_exact=True)
    panel = store.load(Tables.cube_part_text)
    assert panel is not None and panel[["f_ec_prep_tone", "f_ec_answers_tone", "f_ec_qa_coherence_mean"]].notna().all().all()
    assert panel["f_ec_questions_tone"].isna().all() and not embedding.called
    plan = StepCubeText(context, config).plan_window(pd.bdate_range("2020-01-01", "2027-03-01"), full=False)
    assert plan.sentiment_work == {("ADP", "2026Q3")}
    print(
        "SANITY: missing question inference remains retryable; valid stored management/prepared scores and independent embeddings still produce their features."
    )


def test_stale_model_and_legacy_markers_do_not_complete_calls(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    context, config, store = seeded_case(sqlite_store, ["ADP_2026Q3"])
    configure_models(monkeypatch)
    step, plan, prepared = prepare_step(context, config)
    step.run_sentiments(prepared, plan)
    scores = store.load(Tables.earnings_call_sentiment)
    assert scores is not None
    store.save(Tables.earnings_call_sentiment, scores.assign(model=step._sentiment_cache + ":invalid-pending"))
    retry = StepCubeText(context, config).plan_window(pd.bdate_range("2020-01-01", "2027-03-01"), full=False)
    assert retry.sentiment_work == {("ADP", "2026Q3")} and not retry.eligible
    print("SANITY: legacy repair rows are excluded as scores and completion; normal key comparison retries without acknowledgement state.")
