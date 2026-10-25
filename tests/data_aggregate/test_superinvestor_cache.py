"""Unchanged vintage statistics and public-prefix evidence are reused by the engine."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import pandas as pd
import pytest

from src.data_aggregate.utils.sec_thirteen import superinvestor_metrics as metrics
from src.data_aggregate.utils.sec_thirteen.superinvestor_features import compute_superinvestor_feature_history
from tests.data_aggregate.test_superinvestor_features import _book, _compute, _market, _snapshot, _value


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


@pytest.mark.parametrize("include_class", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_summary_retains_only_owned_comparison_evidence(include_class: bool, empty: bool) -> None:
    book = _book("1", "2024-06-30", {} if empty else {"BBB": (20, 200), "AAA": (10, 300), "OUT": (30, 300)})
    book.df_positions.index = pd.Index([] if empty else [8, 3, 5], name="position")
    book.df_positions["source_accession"] = "retained-on-public-book"
    if not include_class:
        book.df_positions.drop(columns="class_evidence", inplace=True)
    original = book.df_positions.copy(deep=True)
    summaries = metrics.BookSummaries()
    summary = summaries.get(book)

    order = [] if empty else [3, 5, 8]
    df_expected = original.loc[order, ["cusip", "ticker", "shares"]].astype({"shares": "float64"})
    df_expected["quantity_known"] = pd.Series(True, index=df_expected.index, dtype="bool")
    if include_class:
        df_expected["class_evidence"] = original.loc[order, "class_evidence"]
    pd.testing.assert_frame_equal(summary.df_positions, df_expected)
    pd.testing.assert_frame_equal(book.df_positions, original)
    assert summaries.get(book) is summary
    assert summary.total_value == (0 if empty else 800)
    if not empty:
        summary.df_positions.loc[3, "shares"] = -1
        pd.testing.assert_frame_equal(book.df_positions, original)
        book.df_positions.loc[5, "shares"] = -2
        assert summary.df_positions.loc[5, "shares"] == 30
    print(f"SANITY: empty={empty}, class={include_class}; cached comparison evidence preserves order/dtypes and owns only its required columns.")


@pytest.mark.parametrize("lag", [1, 4])
@pytest.mark.parametrize("class_mode", ["absent", "same", "changed"])
def test_optional_class_evidence_keeps_existing_comparison_support(lag: int, class_mode: str) -> None:
    prior = _book("1", "2024-03-31" if lag == 1 else "2023-06-30", {"AAA": (100, 1000)})
    current = _book("1", "2024-06-30", {"AAA": (110, 1100)})
    if class_mode == "absent":
        prior.df_positions.drop(columns="class_evidence", inplace=True)
        current.df_positions.drop(columns="class_evidence", inplace=True)
    elif class_mode == "changed":
        current.df_positions["class_evidence"] = '["preferred"]'
    history = _compute(_snapshot([prior, current]))
    for cohort in ("", "topk"):
        change = _value(history, "shares_chg" if lag == 1 else "shares_chg_yoy", cohort=cohort)
        assert pd.isna(change) if class_mode == "changed" else change == pytest.approx(0.1)
        assert _value(history, "holders", cohort=cohort) == 1
    print(f"SANITY: lag{lag}, class={class_mode}; optional class presence and changed-class abstention retain their existing policy in both cohorts.")
