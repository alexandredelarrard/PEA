"""Each revision is summarized once and each compared pair joined once; outputs never depend on that reuse."""

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

_BOTH = {"": frozenset({"1"}), "topk": frozenset({"1"})}


def test_each_unchanged_vintage_is_summarized_once_across_dates_and_cohorts(monkeypatch: pytest.MonkeyPatch) -> None:
    prior = _book("1", "2024-03-31", {"AAA": (100, 1000)})
    current = _book("1", "2024-06-30", {"AAA": (110, 1100)})
    first = _snapshot([prior], day="2024-05-16")
    second = _snapshot([prior, current])
    third = replace(second, day=pd.Timestamp("2024-08-16"))
    summarizer = Mock(wraps=metrics.summarize_books)
    monkeypatch.setattr(feature_module, "summarize_books", summarizer)
    history = compute_superinvestor_feature_history((first, second, third), _BOTH, ("AAA", "BBB"), _market())
    assert summarizer.call_count == 1
    assert summarizer.call_args.args[0] == [prior, current]
    assert len(history.diagnostics) == 6
    print("SANITY: two resolved vintages are summarized exactly once each, in one call, across three states, two masks and all consumers.")


def test_unchanged_pairs_and_same_event_duration_are_prepared_once(monkeypatch: pytest.MonkeyPatch) -> None:
    prior = _book("1", "2024-03-31", {"AAA": (100, 1000)})
    current = _book("1", "2024-06-30", {"AAA": (110, 1100)})
    first = _snapshot([prior], day="2024-05-16")
    second = _snapshot([prior, current])
    repeated = replace(second, day=pd.Timestamp("2024-08-16"))
    joins = Mock(wraps=metrics.join_pairs)
    durations = Mock(wraps=metrics.holding_duration)
    monkeypatch.setattr(feature_module, "join_pairs", joins)
    monkeypatch.setattr(feature_module, "holding_duration", durations)
    history = compute_superinvestor_feature_history((first, second, repeated), _BOTH, ("AAA", "BBB"), _market(split=True))
    assert joins.call_count == 1 and joins.call_args.args[0] == [(current, prior)]
    assert durations.call_count == 3
    for values in history.levels.values():
        pd.testing.assert_frame_equal(values.iloc[1:2].reset_index(drop=True), values.iloc[2:].reset_index(drop=True))
    print("SANITY: one pair join spans unchanged events; tenure is prepared once per event for both masks without repeated split mutation.")


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
    joins = Mock(wraps=metrics.join_pairs)
    monkeypatch.setattr(feature_module, "join_pairs", joins)
    history = compute_superinvestor_feature_history(
        (first, replace(first, day=pd.Timestamp("2024-08-16")), changed), _BOTH, ("AAA", "BBB"), _market()
    )
    assert joins.call_count == 1 and len(joins.call_args.args[0]) == 2
    metric = "shares_chg" if lag == 1 else "shares_chg_yoy"
    for cohort in ("", "topk"):
        assert _value(history, metric, cohort=cohort) == np.float32(1.2 if endpoint == "current" else -0.45)
    assert original.df_positions.shares.iloc[0] == (110 if endpoint == "current" else 100)
    print(f"SANITY: lag{lag} {endpoint} amendment adds exactly one new pair and correct later values while the original endpoint remains unchanged.")


def test_pairs_are_scoped_to_active_managers_quarters_and_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    books = [
        _book(cik, period, {"AAA": (shares, shares * 10)}) for cik in ("1", "2") for period, shares in (("2024-03-31", 100), ("2024-06-30", 110))
    ]
    first = _snapshot(books)
    advanced = _snapshot([*books, _book("1", "2024-09-30", {"AAA": (120, 1200)})], day="2024-11-15")
    expired = replace(advanced, day=pd.Timestamp("2024-11-18"), level_books={})
    joins = Mock(wraps=metrics.join_pairs)
    monkeypatch.setattr(feature_module, "join_pairs", joins)
    history = compute_superinvestor_feature_history(
        (first, replace(first, day=pd.Timestamp("2024-08-16")), advanced, expired),
        {"": frozenset({"1", "2"}), "topk": frozenset({"1"})},
        ("AAA",),
        _market(),
    )
    assert joins.call_count == 1 and len(joins.call_args.args[0]) == 3
    full = history.diagnostics.loc[history.diagnostics.cohort.eq("full")]
    assert full.quarter_pairs.tolist() == [2, 2, 1, 0]
    print("SANITY: identical positions in separate CIKs are separate pairs; quarter changes keep one eligible pair and expiry compares none.")


def test_future_snapshot_does_not_mutate_previous_outputs() -> None:
    prior = _book("1", "2024-03-31", {"AAA": (100, 1000)})
    current = _book("1", "2024-06-30", {"AAA": (110, 1100)})
    snapshot = _snapshot([prior], day="2024-05-16")
    before = _compute(snapshot)
    after = compute_superinvestor_feature_history((snapshot, _snapshot([prior, current])), _BOTH, ("AAA", "BBB"), _market())
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
    table = metrics.summarize_books([book], pd.Index(["AAA", "BBB"], dtype=object))

    rows = table.df_positions
    assert list(rows.columns) == ["_book", "_cusip", "ticker", "shares", "quantity_known", "class_evidence"]
    assert rows.ticker.tolist() == ([] if empty else ["AAA", "OUT", "BBB"])
    assert rows.shares.tolist() == ([] if empty else [10.0, 30.0, 20.0]) and rows.shares.dtype == "float64"
    assert rows.quantity_known.all() and rows.quantity_known.dtype == bool
    assert rows.class_evidence.notna().all() if include_class else rows.class_evidence.isna().all()
    assert table.has_class.tolist() == [include_class] and table.starts.tolist() == [0, len(rows)]
    assert table.total.tolist() == [0 if empty else 800]
    np.testing.assert_array_equal(table.held, [[0.0, 0.0]] if empty else [[1.0, 1.0]])
    pd.testing.assert_frame_equal(book.df_positions, original)
    if not empty:
        rows.loc[0, "shares"] = -1
        pd.testing.assert_frame_equal(book.df_positions, original)
        book.df_positions.loc[5, "shares"] = -2
        assert table.df_positions.loc[1, "shares"] == 30
    print(f"SANITY: empty={empty}, class={include_class}; the summary keeps rank order, dtypes and only its comparison columns, sharing no memory.")


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


def test_class_evidence_is_compared_only_when_both_books_carry_it() -> None:
    prior = _book("1", "2024-03-31", {"AAA": (100, 1000)})
    current = _book("1", "2024-06-30", {"AAA": (110, 1100)})
    prior.df_positions.drop(columns="class_evidence", inplace=True)
    for cohort in ("", "topk"):
        assert _value(_compute(_snapshot([prior, current])), "shares_chg", cohort=cohort) == pytest.approx(0.1)
    print("SANITY: a pair where only one book carries class evidence keeps its comparison, as when neither does.")


def test_duration_history_summarizes_every_revision_once(monkeypatch: pytest.MonkeyPatch) -> None:
    books = _long_duration_books()
    snapshot = _snapshot(books)
    repeated = replace(snapshot, day=pd.Timestamp("2024-08-16"))
    summarizer = Mock(wraps=metrics.summarize_books)
    monkeypatch.setattr(feature_module, "summarize_books", summarizer)
    history = compute_superinvestor_feature_history((snapshot, repeated), _BOTH, ("AAA", "OUT"), _market())
    assert summarizer.call_count == 1 and summarizer.call_args.args[0] == books
    for values in history.levels.values():
        pd.testing.assert_frame_equal(values.iloc[:1].reset_index(drop=True), values.iloc[1:].reset_index(drop=True))
    assert _value(history, "quarters_held", ticker="OUT") == len(books)
    assert _value(history, "quarters_held", ticker="AAA") == 2
    print("SANITY: repeated dates and both masks summarize each of the 14 archive vintages exactly once; tenure 14 and 2.")


def test_duration_limits_starting_tickers_before_walking_the_archive() -> None:
    books = _long_duration_books()
    snapshot = _snapshot(books)
    archive = Mock(wraps=snapshot.archive)
    universe = pd.Index(["AAA"], dtype=object)
    table = metrics.summarize_books(books, universe)
    for _ in range(2):
        archive.reset_mock()
        durations = {"1": metrics.holding_duration(books[-1], archive, table)}
        sample = metrics.prepare_level_sample(snapshot, table, durations, universe)
        levels, coverage = metrics.level_metrics(sample, frozenset({"1"}), universe)
        assert levels["quarters_held"].AAA == 2
        assert levels["conviction_weight"].AAA == pytest.approx(1 / 11)
        assert coverage == {"observed": 1, "valued": 1}
        assert [call.args[0] for call in archive.get.call_args_list] == [(book.cik, book.period) for book in books[-2:-4:-1]]
    print("SANITY: a 2-quarter output holding visits only the 2 necessary prior books; a 14-quarter off-universe chain cannot prolong the walk.")


@pytest.mark.parametrize("optional", ["all", "missing_source", "missing_unverified", "missing_both"])
def test_held_rows_follow_evidence_and_separate_revisions(optional: str) -> None:
    book = _book("1", "2024-06-30", {"BBB": (20, 200), "AAA": (10, 300), "OUT": (0, 0)})
    book.df_positions["shares"] = pd.Series([np.nan, "10", "0"], dtype=object)
    book.df_positions["value_usd"] = pd.Series(["200", "300", "0"], dtype=object)
    book.df_positions["source_accession"] = "full-provenance-stays-on-the-source"
    if optional in ("missing_source", "missing_both"):
        book.df_positions.drop(columns="value_basis_shares_source", inplace=True)
    if optional in ("missing_unverified", "missing_both"):
        book.df_positions.drop(columns="value_basis_inverse_unit_unverified", inplace=True)
    original = book.df_positions.copy(deep=True)
    revised = replace(book, public_at=book.public_at + pd.Timedelta(days=1), df_positions=original.copy(deep=True))
    revised.df_positions.loc[revised.df_positions.ticker.eq("AAA"), ["shares", "value_usd"]] = np.nan
    revised.df_positions.loc[revised.df_positions.ticker.eq("AAA"), "position_type"] = "call"
    revised_original = revised.df_positions.copy(deep=True)

    table = metrics.summarize_books([book, revised], pd.Index(["AAA", "BBB", "OUT"], dtype=object))
    # A proven zero needs filed shares; without the unverified flag the value counts as verified.
    expected_out = np.nan if optional in ("missing_source", "missing_both") else 0.0
    np.testing.assert_array_equal(table.held, [[1.0, 1.0, expected_out], [np.nan, 1.0, expected_out]])
    assert table.codes == {book: 0, revised: 1}
    pd.testing.assert_frame_equal(book.df_positions, original)
    pd.testing.assert_frame_equal(revised.df_positions, revised_original)
    print(
        f"SANITY: {optional}; held rows read only present raw fields, a typed unknown leg is unverified, and revisions stay separate and unmutated."
    )
