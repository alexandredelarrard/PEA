"""Unchanged vintage statistics and public-prefix evidence are reused by the engine."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.sec_thirteen import superinvestor_features as feature_module
from src.data_aggregate.utils.sec_thirteen import superinvestor_metrics as metrics
from src.data_aggregate.utils.sec_thirteen.superinvestor_features import compute_superinvestor_feature_history
from tests.data_aggregate.test_superinvestor_features import _book, _compute, _long_duration_books, _market, _snapshot, _value


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


def test_unchanged_pairs_and_same_event_duration_are_prepared_once(monkeypatch: pytest.MonkeyPatch) -> None:
    prior = _book("1", "2024-03-31", {"AAA": (100, 1000)})
    current = _book("1", "2024-06-30", {"AAA": (110, 1100)})
    first = _snapshot([prior], day="2024-05-16")
    second = _snapshot([prior, current])
    repeated = replace(second, day=pd.Timestamp("2024-08-16"))
    joins = Mock(wraps=metrics._join_positions)
    durations = Mock(wraps=metrics._holding_duration)
    monkeypatch.setattr(metrics, "_join_positions", joins)
    monkeypatch.setattr(metrics, "_holding_duration", durations)
    history = compute_superinvestor_feature_history(
        (first, second, repeated), {"": frozenset({"1"}), "topk": frozenset({"1"})}, ("AAA", "BBB"), _market(split=True)
    )
    assert (joins.call_count, durations.call_count) == (1, 3)
    for values in history.levels.values():
        pd.testing.assert_frame_equal(values.iloc[1:2].reset_index(drop=True), values.iloc[2:].reset_index(drop=True))
    print("SANITY: one immutable pair join spans unchanged events; tenure is prepared once per event for both masks without repeated split mutation.")


@pytest.mark.parametrize("lag", [1, 4])
@pytest.mark.parametrize("endpoint", ["current", "prior"])
def test_pair_reuse_reselects_both_revised_endpoints(lag: int, endpoint: str, monkeypatch: pytest.MonkeyPatch) -> None:
    prior = _book("1", "2024-03-31" if lag == 1 else "2023-06-30", {"AAA": (100, 1000)})
    current = _book("1", "2024-06-30", {"AAA": (110, 1100)})
    first = _snapshot([prior, current])
    original = current if endpoint == "current" else prior
    df_revised = original.df_positions.copy(deep=True)
    df_revised[["shares", "value_usd"]] *= 2
    revised = replace(original, public_at=pd.Timestamp("2024-08-17"), df_positions=df_revised)
    changed = _snapshot([prior, revised] if endpoint == "current" else [revised, current], day="2024-08-17")
    joins = Mock(wraps=metrics._join_positions)
    monkeypatch.setattr(metrics, "_join_positions", joins)
    history = compute_superinvestor_feature_history(
        (first, replace(first, day=pd.Timestamp("2024-08-16")), changed), {"": frozenset({"1"}), "topk": frozenset({"1"})}, ("AAA", "BBB"), _market()
    )
    assert joins.call_count == 2
    metric = "shares_chg" if lag == 1 else "shares_chg_yoy"
    for cohort in ("", "topk"):
        assert _value(history, metric, cohort=cohort) == np.float32(1.2 if endpoint == "current" else -0.45)
    assert original.df_positions.shares.iloc[0] == (110 if endpoint == "current" else 100)
    print(f"SANITY: lag{lag} {endpoint} amendment creates one new pair and correct later values while the original endpoint remains unchanged.")


def test_pair_cache_is_scoped_to_active_managers_quarters_and_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    books = [
        _book(cik, period, {"AAA": (shares, shares * 10)}) for cik in ("1", "2") for period, shares in (("2024-03-31", 100), ("2024-06-30", 110))
    ]
    first = _snapshot(books)
    advanced = _snapshot([*books, _book("1", "2024-09-30", {"AAA": (120, 1200)})], day="2024-11-15")
    expired = replace(advanced, day=pd.Timestamp("2024-11-18"), level_books={})
    summaries = metrics.BookSummaries()
    sizes: list[int] = []
    original = summaries.prune_pairs

    def record() -> None:
        original()
        sizes.append(len(summaries._pairs))

    monkeypatch.setattr(summaries, "prune_pairs", record)
    monkeypatch.setattr(feature_module, "BookSummaries", lambda: summaries)
    joins = Mock(wraps=metrics._join_positions)
    monkeypatch.setattr(metrics, "_join_positions", joins)
    compute_superinvestor_feature_history(
        (first, replace(first, day=pd.Timestamp("2024-08-16")), advanced, expired),
        {"": frozenset({"1", "2"}), "topk": frozenset({"1"})},
        ("AAA",),
        _market(),
    )
    assert joins.call_count == 3 and sizes == [2, 2, 1, 0]
    assert not summaries._pairs
    print(
        "SANITY: identical positions in separate CIKs have separate pairs; quarter changes retain one eligible pair and expiry releases all comparisons."
    )


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


def test_duration_only_history_never_builds_full_summaries(monkeypatch: pytest.MonkeyPatch) -> None:
    books = _long_duration_books()
    snapshot = _snapshot(books)
    repeated = replace(snapshot, day=pd.Timestamp("2024-08-16"))
    summarizer = Mock(wraps=metrics.summarize_book)
    evidence = Mock(wraps=metrics._holding_evidence)
    monkeypatch.setattr(metrics, "summarize_book", summarizer)
    monkeypatch.setattr(metrics, "_holding_evidence", evidence)
    cohorts = {"": frozenset({"1"}), "topk": frozenset({"1"})}
    history = compute_superinvestor_feature_history((snapshot, repeated), cohorts, ("AAA", "OUT"), _market())
    expected = {books[-1], books[-2], books[-5]}
    assert {call.args[0] for call in summarizer.call_args_list} == expected
    assert summarizer.call_count == 3
    assert evidence.call_count == len(books)
    for values in history.levels.values():
        pd.testing.assert_frame_equal(values.iloc[:1].reset_index(drop=True), values.iloc[1:].reset_index(drop=True))
    assert _value(history, "quarters_held", ticker="OUT") == len(books)
    assert _value(history, "quarters_held", ticker="AAA") == 2
    print("SANITY: repeated dates and both masks build only3 full pair/level summaries; all other duration vintages avoid full summarization.")


def test_duration_limits_starting_tickers_before_walking_the_archive() -> None:
    books = _long_duration_books()
    snapshot = _snapshot(books)
    archive = Mock(wraps=snapshot.archive)
    snapshot = replace(snapshot, archive=archive)
    summaries = metrics.BookSummaries()
    for _ in range(2):
        archive.reset_mock()
        sample = metrics.prepare_level_sample(snapshot, ("AAA",), summaries)
        levels, coverage = metrics.level_metrics(sample, frozenset({"1"}), ("AAA",))
        assert levels["quarters_held"].AAA == 2
        assert levels["conviction_weight"].AAA == pytest.approx(1 / 11)
        assert coverage == {"observed": 1, "valued": 1}
        assert [call.args[0] for call in archive.get.call_args_list] == [(book.cik, book.period) for book in books[-2:-4:-1]]
    print("SANITY: a2-quarter output holding visits only the2 necessary prior books; a14-quarter off-universe chain cannot prolong the walk.")


@pytest.mark.parametrize("optional", ["all", "missing_source", "missing_unverified", "missing_both"])
def test_compact_holding_cache_reuses_upgrades_and_separates_revisions(optional: str, monkeypatch: pytest.MonkeyPatch) -> None:
    book = _book("1", "2024-06-30", {"BBB": (20, 200), "AAA": (10, 300), "OUT": (0, 0)})
    book.df_positions["shares"] = pd.Series([np.nan, "10", "0"], dtype=object)
    book.df_positions["value_usd"] = pd.Series(["200", "300", "0"], dtype=object)
    book.df_positions["source_accession"] = "full-provenance-stays-on-the-source"
    if optional in ("missing_source", "missing_both"):
        book.df_positions.drop(columns="value_basis_shares_source", inplace=True)
    if optional in ("missing_unverified", "missing_both"):
        book.df_positions.drop(columns="value_basis_inverse_unit_unverified", inplace=True)
    original = book.df_positions.copy(deep=True)
    evidence = Mock(wraps=metrics._holding_evidence)
    monkeypatch.setattr(metrics, "_holding_evidence", evidence)
    summaries = metrics.BookSummaries()

    compact = summaries.held(book)
    pd.testing.assert_index_equal(compact, pd.Index(["BBB", "AAA"], name="ticker"))
    assert summaries.held(book) is compact
    expected = ["ticker", "position_type", "shares", "value_usd"]
    expected.extend(name for name in ("value_basis_shares_source", "value_basis_inverse_unit_unverified") if name in original)
    assert list(evidence.call_args.args[0].columns) == expected
    assert evidence.call_count == 1
    assert not summaries._cache
    assert summaries._held_cache == {book: compact}

    summary = summaries.get(book)
    assert not summaries._held_cache
    assert summaries.get(book) is summary
    pd.testing.assert_index_equal(summaries.held(book).sort_values(), compact.sort_values())
    assert evidence.call_count == 2
    assert list(evidence.call_args.args[0].columns) == list(original.columns)

    revised = replace(book, public_at=book.public_at + pd.Timedelta(days=1), df_positions=original.copy(deep=True))
    revised.df_positions.loc[revised.df_positions.ticker.eq("AAA"), ["shares", "value_usd"]] = np.nan
    revised.df_positions.loc[revised.df_positions.ticker.eq("AAA"), "position_type"] = "call"
    revised_original = revised.df_positions.copy(deep=True)
    assert summaries.held(revised).tolist() == ["BBB"]
    assert evidence.call_count == 3
    assert revised in summaries._held_cache and book not in summaries._held_cache
    pd.testing.assert_index_equal(summaries.held(book).sort_values(), compact.sort_values())
    pd.testing.assert_frame_equal(book.df_positions, original)
    pd.testing.assert_frame_equal(revised.df_positions, revised_original)
    print(
        f"SANITY: {optional}; compact evidence projects only present raw fields, reuses one index, upgrades without duplication and isolates revisions."
    )
