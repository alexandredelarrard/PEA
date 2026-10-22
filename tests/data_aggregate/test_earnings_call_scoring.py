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
    assert scores is not None and len(scores) == 2 * len(FIXTURE_PATHS)
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
    engine.score_texts.side_effect = [[{"pos": 0.6, "neg": 0.1, "neu": 0.3}] * 2, RuntimeError("interrupted")]
    step, plan, prepared = prepare_step(context, config)
    with pytest.raises(RuntimeError, match="interrupted"):
        step.run_sentiments(prepared, plan)
    scores = store.load(Tables.earnings_call_sentiment)
    assert scores is not None and len(scores) == 2 and scores["ticker"].eq("ADP").all()
    store.delete(Tables.earnings_call_sentiment, {"ticker": "ADP", "tag": "qa"})
    engine.score_texts.side_effect = lambda texts: [{"pos": 0.6, "neg": 0.1, "neu": 0.3} for _ in texts]
    engine.reset_mock()
    step, plan, prepared = prepare_step(context, config)
    step.run_sentiments(prepared, plan)
    assert [len(call.args[0]) for call in engine.score_texts.call_args_list] == [1, 2]
    scores = store.load(Tables.earnings_call_sentiment)
    assert scores is not None and len(scores) == 4
    print("SANITY: saved calls survive interruption; a valid partner tag is reused while only the missing section is scored.")


@pytest.mark.parametrize("output", [[], [None, None], [{"pos": np.nan, "neg": 0.1, "neu": 0.3}] * 2, [{"pos": 0.9, "neg": 0.9, "neu": 0.3}] * 2])
def test_invalid_outputs_do_not_establish_completion(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, output: list) -> None:
    context, config, store = seeded_case(sqlite_store, ["ADP_2026Q3"])
    _, engine, _ = configure_models(monkeypatch)
    engine.score_texts.side_effect = lambda _texts: output
    step, plan, prepared = prepare_step(context, config)
    with pytest.raises(ValueError, match="Sentiment"):
        step.run_sentiments(prepared, plan)
    assert store.load(Tables.earnings_call_sentiment, optional=True) is None
    print("SANITY: invalid score count/probabilities remain absent and retryable, distinct from unavailable text.")


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
