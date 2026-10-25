"""Unchanged vintage statistics and public-prefix evidence are reused by the engine."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import pandas as pd
import pytest

from src.data_aggregate.utils.sec_thirteen import superinvestor_metrics as metrics
from src.data_aggregate.utils.sec_thirteen.superinvestor_features import compute_superinvestor_feature_history
from tests.data_aggregate.test_superinvestor_features import _book, _compute, _market, _snapshot


def test_each_unchanged_vintage_is_summarized_once_across_dates_and_cohorts(monkeypatch: pytest.MonkeyPatch) -> None:
    prior = _book("1", "2024-03-31", {"AAA": (100, 1000)})
    current = _book("1", "2024-06-30", {"AAA": (110, 1100)})
    first = _snapshot([prior], day="2024-05-16")
    second = _snapshot([prior, current])
    third = replace(second, day=pd.Timestamp("2024-08-16"))
    summarizer = Mock(wraps=metrics.summarize_book)
    monkeypatch.setattr(metrics, "summarize_book", summarizer)
    history = compute_superinvestor_feature_history(
        (first, second, third), {"": frozenset({"1"}), "topk": frozenset({"1"})}, ("AAA", "BBB"), _market()
    )
    assert summarizer.call_count == 2
    assert len(history.diagnostics) == 6
    print("SANITY: two resolved vintages are summarized exactly twice across three states, two masks and all lag/tenure consumers.")


def test_future_snapshot_does_not_mutate_previous_outputs() -> None:
    prior = _book("1", "2024-03-31", {"AAA": (100, 1000)})
    current = _book("1", "2024-06-30", {"AAA": (110, 1100)})
    snapshot = _snapshot([prior], day="2024-05-16")
    cohorts = {"": frozenset({"1"}), "topk": frozenset({"1"})}
    before = _compute(snapshot)
    after = compute_superinvestor_feature_history((snapshot, _snapshot([prior, current])), cohorts, ("AAA", "BBB"), _market())
    for name, values in before.levels.items():
        pd.testing.assert_frame_equal(values, after.levels[name].iloc[:1])
    print("SANITY: later public books neither rewrite previous sparse outputs nor change their values/nulls.")


def test_consuming_books_never_mutates_resolved_positions() -> None:
    book = _book("1", "2024-06-30", {"AAA": (100, 1000)})
    original = book.df_positions.copy(deep=True)
    _compute(_snapshot([book]))
    pd.testing.assert_frame_equal(book.df_positions, original)
    print("SANITY: level ranks, quantity flags and feature preparation leave the shared resolver DataFrame unchanged.")
