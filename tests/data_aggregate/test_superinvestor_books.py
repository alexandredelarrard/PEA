"""Known-truth publication, expiry and archival controls for science book state."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest

from src.data_aggregate.utils.sec_thirteen import institutional_vintages
from src.data_aggregate.utils.sec_thirteen.superinvestor_books import (
    PublicBookSnapshot,
    prepare_superinvestor_batches,
    prepare_superinvestor_book_history,
)
from src.utils.institutional_vintages import prepare_book_vintages


def _packet(
    cik: str, period: str, filed: str, positions: dict[str, float], kind: str = "original", scope: str = "complete"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    identity = {"cik": cik, "period": pd.Timestamp(period), "filing_date": pd.Timestamp(filed), "accession_number": cik + period + filed + kind}
    amounts = ("shares", "value_usd", "call_shares", "call_value", "put_shares", "put_value", "debt_prn", "debt_value", "other_value")
    rows = []
    for cusip, shares in positions.items():
        row = {**identity, "cusip": cusip, "ticker": "AAA", "position_type": "common", "class_evidence": '["common"]', **dict.fromkeys(amounts, 0.0)}
        row.update(shares=shares, value_usd=shares * 10, shares_provenance="filed", value_provenance="filed")
        row.update({"raw_" + column: row[column] for column in amounts if "value" in column})
        rows.append(row)
    columns = [*identity, "cusip", "ticker", "position_type", "class_evidence", *amounts, "shares_provenance", "value_provenance"]
    columns.extend("raw_" + column for column in amounts if "value" in column)
    manifest = {
        **identity,
        "amendment_type": kind,
        "amendment_number": 0,
        "report_type": "13F COMBINATION REPORT" if scope == "partial" else "13F HOLDINGS REPORT",
        "restricted": scope == "restricted",
        "parse_status": "failed" if scope == "failed" else "complete",
        "parsed_row_count": len(rows),
    }
    return pd.DataFrame([manifest]), pd.DataFrame(rows, columns=columns)


def _history(packets: list[tuple[pd.DataFrame, pd.DataFrame]], grid: pd.DatetimeIndex | None = None) -> tuple[PublicBookSnapshot, ...]:
    grid = pd.bdate_range("2023-01-02", "2026-03-31") if grid is None else grid
    df_filings = pd.concat([packet[0] for packet in packets], ignore_index=True)
    df_holdings = pd.concat([packet[1] for packet in packets], ignore_index=True)
    df_manifest = prepare_book_vintages(df_filings, df_holdings, grid, settle_days=0)
    history = prepare_superinvestor_book_history(((df_manifest, df_holdings) for _ in range(1)), grid, 90)
    assert history is not None
    return history


def _on(history: tuple[PublicBookSnapshot, ...], day: str) -> PublicBookSnapshot:
    return max((snapshot for snapshot in history if snapshot.day <= pd.Timestamp(day)), key=lambda snapshot: snapshot.day)


def test_guarded_manager_batches_keep_the_whole_table_revision_history() -> None:
    packets = [
        _packet("0000000001", "2024-03-31", "2024-05-15", {"111111111": 100}),
        _packet("0000000002", "2024-03-31", "2024-05-15", {"111111111": 200}),
        _packet("0000000002", "2024-06-30", "2024-08-14", {"111111111": 220}),
        _packet("0000000002", "2024-06-30", "2024-09-03", {"111111111": 250}, "restatement"),
        _packet("0000000003", "2024-06-30", "2024-08-14", {}),
    ]
    # A projected empty source retains the nonempty table's dtypes, rather than the fixture constructor's object fallback.
    packets[-1] = packets[-1][0], packets[-1][1].astype(packets[0][1].dtypes.to_dict())
    grid = pd.bdate_range("2024-01-02", "2024-12-31")
    expected = _history(packets, grid)
    batches = [
        tuple(pd.concat([packet[index] for packet in selected], ignore_index=True) for index in (0, 1)) for selected in (packets[:2], packets[2:])
    ]
    prepared = prepare_superinvestor_batches(((batch[0], batch[1]) for batch in batches), grid)
    actual = prepare_superinvestor_book_history(prepared, grid, 90)
    assert actual is not None and [state.day for state in actual] == [state.day for state in expected]
    for old, new in zip(expected, actual, strict=True):
        assert old.common_period == new.common_period and dict(old.latest_period_by_manager) == dict(new.latest_period_by_manager)
        assert set(old.level_books) == set(new.level_books) and set(old.archive) == set(new.archive)
        for key in old.archive:
            before, after = old.archive[key], new.archive[key]
            assert before.support == after.support and before.first_public_at == after.first_public_at
            pd.testing.assert_frame_equal(before.df_positions, after.df_positions)
    print(
        "SANITY: guarded many-book batches preserve exact positions, public/expiry dates, amendment history, complete-empty support and all archive states."
    )


def test_first_public_day90_is_included_and_day91_expires_without_a_filing() -> None:
    history = _history([_packet("0000000001", "2024-03-31", "2024-05-15", {"111111111": 100})])

    origin = _on(history, "2024-05-16").level_books["0000000001"].first_public_at
    assert origin == pd.Timestamp("2024-05-16")
    assert "0000000001" in _on(history, "2024-08-14").level_books
    expired = _on(history, "2024-08-15")
    assert expired.day == pd.Timestamp("2024-08-15")
    assert not expired.level_books
    assert expired.common_period == pd.Timestamp("2024-03-31")
    assert expired.archive[("0000000001", pd.Timestamp("2024-03-31"))].support.book_known
    print("SANITY: first-public day90 contributes; the day91 session expires it without fabricating a sale or deleting its archive.")


def test_expiry_uses_the_original_calendar_and_first_available_session() -> None:
    grid = pd.bdate_range("2024-01-01", "2024-12-31").difference(pd.DatetimeIndex(["2024-08-15", "2024-08-16"]))
    history = _history([_packet("0000000001", "2024-03-31", "2024-05-15", {"111111111": 100})], grid)

    assert "0000000001" in _on(history, "2024-08-14").level_books
    assert _on(history, "2024-08-19").day == pd.Timestamp("2024-08-19")
    assert not _on(history, "2024-08-19").level_books
    assert set(snapshot.day for snapshot in history).issubset(set(grid))
    print("SANITY: missing sessions and the weekend postpone the expiry event to the first supplied session, without densifying calendar gaps.")


def test_amendments_keep_the_original_age_and_cannot_resurrect_an_expired_book() -> None:
    history = _history(
        [
            _packet("0000000001", "2024-03-31", "2024-05-15", {"111111111": 100}),
            _packet("0000000001", "2024-03-31", "2024-07-01", {"111111111": 150}, "restatement"),
            _packet("0000000001", "2024-03-31", "2024-12-02", {"111111111": 200}, "restatement"),
        ]
    )

    revised = _on(history, "2024-07-02").level_books["0000000001"]
    assert revised.df_positions.shares.sum() == 150
    assert revised.first_public_at == pd.Timestamp("2024-05-16")
    assert not _on(history, "2024-08-15").level_books
    later = _on(history, "2024-12-03")
    assert not later.level_books
    assert later.archive[("0000000001", pd.Timestamp("2024-03-31"))].df_positions.shares.sum() == 200
    print("SANITY: amendments update their public archive prospectively but neither restart freshness nor revive an expired current contribution.")


@pytest.mark.parametrize("scope", ["restricted", "partial"])
def test_newer_unusable_books_are_barriers_and_old_backfills_do_not_replace_them(scope: str) -> None:
    history = _history(
        [
            _packet("0000000001", "2024-03-31", "2024-05-15", {"111111111": 100}),
            _packet("0000000001", "2024-06-30", "2024-08-13", {"111111111": 20}, scope=scope),
            _packet("0000000001", "2023-12-31", "2024-08-20", {"111111111": 999}),
        ]
    )

    current = _on(history, "2024-08-21")
    book = current.level_books["0000000001"]
    assert book.period == pd.Timestamp("2024-06-30")
    assert not book.support.zero_eligible and not book.support.comparison_eligible
    assert current.common_period == pd.Timestamp("2024-06-30")
    assert current.latest_period_by_manager["0000000001"] == book.period
    assert not _on(history, "2024-11-13").level_books
    assert _on(history, "2024-11-13").latest_period_by_manager["0000000001"] == pd.Timestamp("2024-06-30")
    print(f"SANITY: the latest {scope} quarter remains the support barrier; an older late original cannot replace or refresh it.")


@pytest.mark.parametrize("replacement_kind", ["original", "restatement"])
def test_orphan_additions_do_not_activate_q_and_first_resolved_replacement_starts_age(replacement_kind: str) -> None:
    history = _history(
        [
            _packet("0000000001", "2024-06-30", "2024-08-14", {"111111111": 100}),
            _packet("0000000001", "2024-09-30", "2024-11-14", {"222222222": 10}, "additions"),
            _packet("0000000001", "2024-09-30", "2024-11-20", {"111111111": 120}, replacement_kind),
        ]
    )

    orphan_state = _on(history, "2024-11-15")
    orphan = orphan_state.archive[("0000000001", pd.Timestamp("2024-09-30"))]
    assert not orphan.support.book_known and orphan.first_public_at is None
    assert orphan_state.common_period == pd.Timestamp("2024-06-30")
    replacement = _on(history, "2024-11-21")
    assert replacement.common_period == pd.Timestamp("2024-09-30")
    assert replacement.level_books["0000000001"].first_public_at == pd.Timestamp("2024-11-21")
    print("SANITY: an orphan addition has no level or common-quarter authority; the later resolved restatement establishes the one age origin.")


def test_carried_levels_and_common_q_are_separate_from_genuine_archival_pairs() -> None:
    history = _history(
        [
            _packet("0000000001", "2024-03-31", "2024-06-10", {"111111111": 100}),
            _packet("0000000002", "2024-03-31", "2024-05-15", {"111111111": 200}),
            _packet("0000000002", "2024-06-30", "2024-08-14", {"111111111": 220}),
        ]
    )

    current = _on(history, "2024-08-15")
    assert current.common_period == pd.Timestamp("2024-06-30")
    assert current.common_period is not None
    assert current.level_books["0000000001"].period == pd.Timestamp("2024-03-31")
    assert current.level_books["0000000002"].period == pd.Timestamp("2024-06-30")
    assert current.archive.get(("0000000001", current.common_period)) is None
    baseline = current.archive[("0000000002", pd.Timestamp("2024-03-31"))]
    assert baseline.first_public_at is not None
    assert (current.day - baseline.first_public_at).days == 91
    assert baseline.df_positions.shares.sum() == 200
    print("SANITY: a fresh old-quarter level is never promoted to common q; a genuine pair can still read its expired historical baseline.")


def test_complete_empty_missing_failed_and_orphan_are_distinct() -> None:
    history = _history(
        [
            _packet("0000000001", "2024-06-30", "2024-08-14", {}),
            _packet("0000000002", "2024-06-30", "2024-08-14", {}, "additions"),
            _packet("0000000003", "2024-06-30", "2024-08-14", {}, scope="failed"),
        ]
    )

    current = _on(history, "2024-08-15")
    empty = current.level_books["0000000001"]
    assert empty.df_positions.empty and empty.support.book_known and empty.support.zero_eligible
    assert empty.support.comparison_eligible
    assert "0000000002" not in current.level_books and "0000000003" not in current.level_books
    assert current.archive.get(("0000000004", pd.Timestamp("2024-06-30"))) is None
    assert not current.archive[("0000000002", pd.Timestamp("2024-06-30"))].support.book_known
    print("SANITY: only the complete empty original supports reported zeros; missing, failed and orphan books cannot manufacture exits.")


def test_archives_keep_exact_annual_baselines_and_report_quarter_gaps() -> None:
    history = _history(
        [
            _packet("0000000001", "2023-09-30", "2023-11-14", {"111111111": 10}),
            _packet("0000000001", "2023-12-31", "2024-02-14", {"111111111": 20}),
            _packet("0000000001", "2024-03-31", "2024-05-15", {"111111111": 30}),
            _packet("0000000001", "2024-09-30", "2024-11-14", {"111111111": 40}),
        ]
    )

    current = _on(history, "2024-11-15")
    assert current.common_period == pd.Timestamp("2024-09-30")
    assert current.archive[("0000000001", pd.Timestamp("2023-09-30"))].df_positions.shares.sum() == 10
    assert current.archive.get(("0000000001", pd.Timestamp("2024-06-30"))) is None
    assert current.archive[("0000000001", pd.Timestamp("2024-03-31"))].support.book_known
    print("SANITY: exact year-ago and intervening quarter archives survive daily expiry; an absent calendar quarter remains absent for tenure.")


def test_future_vintages_cannot_change_historical_snapshots_or_leak_archive_keys() -> None:
    original = _packet("0000000001", "2024-03-31", "2024-05-15", {"111111111": 100})
    future = _packet("0000000001", "2024-06-30", "2024-08-14", {"111111111": 150})
    prefix = _history([original])
    full = _history([original, future])
    cutoff = pd.Timestamp("2024-08-14")

    early_prefix = [snapshot for snapshot in prefix if snapshot.day <= cutoff]
    early_full = [snapshot for snapshot in full if snapshot.day <= cutoff]
    assert [snapshot.day for snapshot in early_prefix] == [snapshot.day for snapshot in early_full]
    for before, after in zip(early_prefix, early_full, strict=True):
        assert before.common_period == after.common_period
        assert dict(before.latest_period_by_manager) == dict(after.latest_period_by_manager)
        assert set(before.archive) == set(after.archive)
        assert len(after.archive) == len(before.archive)
        assert after.archive.get(("0000000001", pd.Timestamp("2024-06-30"))) is None
        pd.testing.assert_frame_equal(before.level_books["0000000001"].df_positions, after.level_books["0000000001"].df_positions)
    print("SANITY: adding future vintages changes neither historical events/current support nor the archive's visible keys or values.")


def test_unchanged_books_share_resolver_objects_and_resolve_only_changed_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    resolver = Mock(wraps=institutional_vintages.resolve_prepared_books)
    monkeypatch.setattr(institutional_vintages, "resolve_prepared_books", resolver)
    history = _history(
        [
            _packet("0000000001", "2024-03-31", "2024-05-15", {"111111111": 100}),
            _packet("0000000002", "2024-03-31", "2024-05-20", {"111111111": 200}),
            _packet("0000000002", "2024-03-31", "2024-06-03", {"111111111": 250}, "restatement"),
        ]
    )

    earlier = _on(history, "2024-05-16").level_books["0000000001"]
    later = _on(history, "2024-06-04").level_books["0000000001"]
    assert earlier is later
    assert earlier.df_positions is later.df_positions
    assert len(resolver.call_args_list) == 3
    assert [set(call.args[0].cik) for call in resolver.call_args_list] == [{"0000000001"}, {"0000000002"}, {"0000000002"}]
    assert earlier.df_positions.shares_provenance.eq("filed").all()
    assert earlier.df_positions.value_provenance.eq("filed").all()
    print("SANITY: three publications resolve only their changed manager-quarter; unchanged positions/provenance are shared across snapshots.")


def test_unlimited_cache_resolves_changes_without_scanning_retained_books(monkeypatch: pytest.MonkeyPatch) -> None:
    packets = [
        _packet("0000000002", "2024-03-31", "2024-05-15", {"111111111": 200}),
        _packet("0000000001", "2024-03-31", "2024-05-15", {"111111111": 100}),
        _packet("0000000001", "2024-03-31", "2024-06-03", {"111111111": 120}, "restatement"),
        _packet("0000000003", "2023-12-31", "2024-06-03", {"111111111": 30}),
        _packet("0000000001", "2024-06-30", "2024-08-14", {}),
    ]
    df_holdings = pd.concat([packet[1] for packet in packets], ignore_index=True)
    df_manifest = prepare_book_vintages(
        pd.concat([packet[0] for packet in packets], ignore_index=True), df_holdings, pd.bdate_range("2024-01-01", "2024-12-31"), settle_days=0
    )
    original_manifest, original_holdings = df_manifest.copy(deep=True), df_holdings.copy(deep=True)
    resolve = institutional_vintages.resolve_prepared_books
    getitem = pd.DataFrame.__getitem__
    retained_states: set[int] = set()

    def checked_resolve(manifests: pd.DataFrame, holdings: pd.DataFrame, as_of: object) -> tuple[pd.DataFrame, pd.DataFrame]:
        result = resolve(manifests, holdings, as_of)
        retained_states.add(id(result[1]))
        return result

    def checked_getitem(frame: pd.DataFrame, key: Any) -> Any:
        if id(frame) in retained_states and isinstance(key, str) and key == "book_known":
            raise AssertionError("unlimited archive scanned retained book support")
        return getitem(frame, key)

    resolver = Mock(side_effect=checked_resolve)
    monkeypatch.setattr(institutional_vintages, "resolve_prepared_books", resolver)
    monkeypatch.setattr(pd.DataFrame, "__getitem__", checked_getitem)
    updates = institutional_vintages._iter_public_book_cache(df_manifest, df_holdings, history_quarters=None)
    key2, key1, old_key, new_key = [
        (cik, pd.Timestamp(period))
        for cik, period in [("0000000002", "2024-03-31"), ("0000000001", "2024-03-31"), ("0000000003", "2023-12-31"), ("0000000001", "2024-06-30")]
    ]

    day, cache, first_changed = next(updates)
    assert day == pd.Timestamp("2024-05-16") and list(cache) == list(first_changed) == [key2, key1]
    assert first_changed is not cache
    unchanged, original = cache[key2], cache[key1]
    day, revised_cache, revised = next(updates)
    assert day == pd.Timestamp("2024-06-04") and revised_cache is cache
    assert list(cache) == [key2, key1, old_key] and list(revised) == [key1, old_key]
    assert cache[key2] is unchanged and cache[key1] is not original and first_changed[key1] is original
    assert original[0].shares.sum() == 100 and revised[key1][0].shares.sum() == 120
    day, final_cache, changed = next(updates)
    assert day == pd.Timestamp("2024-08-15") and final_cache is cache
    assert list(cache) == [key2, key1, old_key, new_key] and list(changed) == [new_key]
    assert changed[new_key][0].empty and changed[new_key][1].iloc[0]["book_known"]
    with pytest.raises(StopIteration):
        next(updates)
    assert resolver.call_count == 5
    pd.testing.assert_frame_equal(df_manifest, original_manifest)
    pd.testing.assert_frame_equal(df_holdings, original_holdings)
    print(
        "SANITY: unlimited history resolves every changed key in event order, shares its complete cache and never scans retained support or mutates source facts."
    )


def test_bounded_cache_uses_known_watermark_and_keeps_event_order(monkeypatch: pytest.MonkeyPatch) -> None:
    packets = [
        _packet("0000000002", "2024-03-31", "2024-05-15", {"111111111": 200}),
        _packet("0000000001", "2024-03-31", "2024-05-15", {"111111111": 100}),
        _packet("0000000001", "2024-06-30", "2024-08-14", {"111111111": 120}),
        _packet("0000000003", "2024-09-30", "2024-11-14", {"111111111": 30}, "additions"),
        _packet("0000000004", "2024-03-31", "2024-11-19", {"111111111": 40}),
        _packet("0000000001", "2024-09-30", "2024-11-21", {"111111111": 130}),
        _packet("0000000001", "2024-03-31", "2024-12-02", {"111111111": 150}, "restatement"),
    ]
    df_holdings = pd.concat([packet[1] for packet in packets], ignore_index=True)
    df_manifest = prepare_book_vintages(
        pd.concat([packet[0] for packet in packets], ignore_index=True), df_holdings, pd.bdate_range("2024-01-01", "2024-12-31"), settle_days=0
    )
    resolver = Mock(wraps=institutional_vintages.resolve_prepared_books)
    monkeypatch.setattr(institutional_vintages, "resolve_prepared_books", resolver)
    first, second, current, orphan, late, latest = [
        (cik, pd.Timestamp(period))
        for cik, period in [
            ("0000000002", "2024-03-31"),
            ("0000000001", "2024-03-31"),
            ("0000000001", "2024-06-30"),
            ("0000000003", "2024-09-30"),
            ("0000000004", "2024-03-31"),
            ("0000000001", "2024-09-30"),
        ]
    ]
    events = [
        (day, list(cache), list(changed))
        for day, cache, changed in institutional_vintages._iter_public_book_cache(df_manifest, df_holdings, history_quarters=1)
    ]
    assert events == [
        (pd.Timestamp("2024-05-16"), [first, second], [first, second]),
        (pd.Timestamp("2024-08-15"), [first, second, current], [current]),
        (pd.Timestamp("2024-11-15"), [first, second, current, orphan], [orphan]),
        (pd.Timestamp("2024-11-20"), [first, second, current, orphan, late], [late]),
        (pd.Timestamp("2024-11-22"), [current, orphan, latest], [latest]),
        (pd.Timestamp("2024-12-03"), [current, orphan, latest], []),
    ]
    assert resolver.call_count == 6
    print(
        "SANITY: orphan additions cannot advance bounded retention; the next known quarter prunes only older dependencies and stale revisions keep event/key order."
    )
