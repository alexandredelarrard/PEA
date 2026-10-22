"""Real-transcript scoring reads each table once and preserves cache/resume behavior."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.context import Context
from src.data_aggregate.utils.text import earnings_call_features as ec
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.nlp_sentiment import EARNINGS_CALL_SENTIMENT_CACHE_MODEL
from tests.fixtures.earnings_call_rows import FIXTURE_PATHS, fixture_paragraphs


def test_scoring_reads_each_table_once(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    df_paragraphs = pd.concat([fixture_paragraphs(path.stem) for path in FIXTURE_PATHS], ignore_index=True)
    sqlite_store.save(Tables.earnings_call_sections, df_paragraphs.sample(frac=1, random_state=3))
    context = cast(Context, SimpleNamespace(store=sqlite_store, log=logging.getLogger(__name__)))
    engine = MagicMock(device="test")
    engine.score_texts.side_effect = lambda texts: [{"pos": 0.6, "neg": 0.1, "neu": 0.3} for _ in texts]
    factory = MagicMock(return_value=engine)
    monkeypatch.setattr(ec, "get_sentiment_engine", factory)
    with patch.object(sqlite_store, "load", wraps=sqlite_store.load) as reads:
        changed = ec.score_earnings_calls(context)
        assert [call.args[0] for call in reads.call_args_list] == [Tables.earnings_call_sections, Tables.earnings_call_sentiment]
    df_scored = sqlite_store.load(Tables.earnings_call_sentiment)
    assert df_scored is not None
    expected = {(ticker, quarter, tag) for ticker, quarter, _, quality in ec._valid_calls(df_paragraphs) for tag in quality.cleaned_sections}
    assert set(map(tuple, df_scored[["ticker", "quarter", "tag"]].to_numpy())) == expected
    assert changed == pd.to_datetime(df_scored["as_of"]).min()
    assert set(df_scored["model"]) == {EARNINGS_CALL_SENTIMENT_CACHE_MODEL}
    factory.assert_called_once_with()
    engine.reset_mock()
    factory.reset_mock()
    assert ec.score_earnings_calls(context) is None
    factory.assert_not_called()
    engine.score_texts.assert_not_called()
    print(
        f"SANITY: {len(df_paragraphs)} shuffled real paragraphs yield {len(df_scored)} sections with one read per table; invalid-only reruns skip the model."
    )


def test_scoring_propagates_inference_failure(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    sqlite_store.save(Tables.earnings_call_sections, fixture_paragraphs("ADP_2026Q3"))
    context = cast(Context, SimpleNamespace(store=sqlite_store, log=logging.getLogger(__name__)))
    engine = MagicMock(device="test")
    engine.score_texts.side_effect = RuntimeError("inference failed")
    monkeypatch.setattr(ec, "get_sentiment_engine", lambda: engine)
    with pytest.raises(RuntimeError, match="inference failed"):
        ec.score_earnings_calls(context)
    assert sqlite_store.load(Tables.earnings_call_sentiment, optional=True) is None
    print("SANITY: inference errors propagate without writing an incomplete call.")


def test_scoring_resumes_saved_tickers_and_partial_calls(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    df_paragraphs = pd.concat([fixture_paragraphs("ADP_2026Q3"), fixture_paragraphs("AMZN_2025Q3")], ignore_index=True)
    sqlite_store.save(Tables.earnings_call_sections, df_paragraphs)
    context = cast(Context, SimpleNamespace(store=sqlite_store, log=logging.getLogger(__name__)))
    engine = MagicMock(device="test")
    engine.score_texts.side_effect = [[{"pos": 0.6, "neg": 0.1, "neu": 0.3}] * 2, RuntimeError("interrupted")]
    factory = MagicMock(return_value=engine)
    monkeypatch.setattr(ec, "get_sentiment_engine", factory)
    with pytest.raises(RuntimeError, match="interrupted"):
        ec.score_earnings_calls(context)
    df_saved = sqlite_store.load(Tables.earnings_call_sentiment)
    assert df_saved is not None and len(df_saved) == 2 and set(df_saved["ticker"]) == {"ADP"}
    sqlite_store.delete(Tables.earnings_call_sentiment, {"ticker": "ADP", "tag": "qa"})
    engine.score_texts.side_effect = lambda texts: [{"pos": 0.6, "neg": 0.1, "neu": 0.3} for _ in texts]
    engine.reset_mock()
    ec.score_earnings_calls(context)
    assert [len(call.args[0]) for call in engine.score_texts.call_args_list] == [1, 2]
    df_final = sqlite_store.load(Tables.earnings_call_sentiment)
    assert df_final is not None and len(df_final) == 4
    factory.reset_mock()
    engine.reset_mock()
    assert ec.score_earnings_calls(context) is None
    factory.assert_not_called()
    engine.score_texts.assert_not_called()
    print("SANITY: interruption retains the completed ticker; retry scores only missing tags, and a complete rerun never initializes the engine.")
