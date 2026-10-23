"""Known-truth controls for immutable 13F amendments and publication clocks."""

from __future__ import annotations

import importlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from click.testing import CliRunner
from sqlalchemy import create_engine, event
from sqlalchemy.engine import make_url

import src.data_aggregate.transformers.step_cube_sec_thirteen as step_module
import src.data_store.schema as schema
from src.context import get_config_context
from src.data_aggregate.transformers.step_cube_sec_thirteen import StepCubeSecThirteen
from src.data_aggregate.utils.common.availability import filing_public_dates, public_books
from src.data_aggregate.utils.common.price_frames import PriceFrames
from src.data_aggregate.utils.sec_thirteen.institutional_features import (
    build_institutional_feature_panel,
    vintage_changes,
    vintage_cohort_publications,
    vintage_cohort_snapshot,
    vintage_publications,
)
from src.data_aggregate.utils.sec_thirteen.institutional_vintages import iter_public_books
from src.data_extract.utils.institutionals import fetch_13f_managers as managers
from src.data_extract.utils.institutionals.fetch_13f import filing_vintage
from src.data_extract.utils.institutionals.fetch_13f_managers import pending_vintages
from src.data_extract.utils.institutionals.value_units import declared_units
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.institutional_vintages import reconcile_13f_source

GRID = pd.bdate_range("2024-01-01", "2025-12-31")


def completed(store: DataStore) -> pd.DataFrame:
    return store.load(Tables.sec13f_manager_holdings, where={"cusip": "_empty", "parse_status": "complete"}, markers=True)


def test_only_existing_tables_store_complete_versions(sqlite_store: DataStore) -> None:
    original = packet("same-day-original", "2024-08-14", {"111111111": 100, "222222222": 200})
    amended = packet("same-day-amendment", "2024-08-14", {"111111111": 90}, kind="restatement")
    for table in (Tables.sec13f_hr, Tables.sec13f_manager_holdings):
        for manifest, holdings in (original, amended):
            rows = holdings.assign(ticker="_empty") if table == Tables.sec13f_hr else holdings
            sqlite_store.save_13f_filing(manifest, rows, table=table)
        completed = sqlite_store.load(
            table,
            columns=["accession_number", "filing_date", "parsed_row_count"],
            where={"cusip": "_empty", "parse_status": "complete"},
            markers=True,
        )
        assert set(completed.accession_number) == {"same-day-original", "same-day-amendment"}
        assert len(sqlite_store.load(table)) == 3
        assert "accession_number" in table.pk
    assert not hasattr(Tables, "sec13f_filings")
    assert not hasattr(Tables, "sec13f_holdings_vintages")
    print("SANITY: only the two existing tables retain original and same-day amendment plus hidden completion markers.")


def test_equivalent_transport_replay_keeps_original_clock(sqlite_store: DataStore) -> None:
    manifest, rows = packet("transport", "2024-08-14", {"111111111": 100})
    sqlite_store.save_13f_filing(manifest, rows)
    replay = manifest.copy()
    replay.loc[0, "source"] = "edgar"
    replay.loc[0, "source_checksum"] = "different-byte-layout"
    replay.loc[0, "accepted_at"] = pd.Timestamp("2024-08-14 20:00:00")
    assert sqlite_store.save_13f_filing(replay, rows) == 0
    clock = sqlite_store.load(Tables.sec13f_manager_holdings, columns=["accepted_at"], where={"cusip": "_empty"}, markers=True)
    assert clock.accepted_at.isna().all()
    changed = rows.copy()
    changed.loc[0, "shares"] = 99
    with pytest.raises(ValueError, match="checksum"):
        sqlite_store.save_13f_filing(replay, changed)
    print("SANITY: equivalent ZIP/EDGAR content replays once, cannot retroactively improve a clock, and changed positions conflict.")


def packet(
    accession: str, filed: str, amounts: dict[str, float], *, kind: str = "original", period: str = "2024-06-30", restricted: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    info = pd.DataFrame(
        [
            {"CUSIP": key, "NAMEOFISSUER": key, "TITLEOFCLASS": "COM", "SSHPRNAMT": n, "SSHPRNAMTTYPE": "SH", "PUTCALL": "", "VALUE": n * 10}
            for key, n in amounts.items()
        ],
        columns=["CUSIP", "NAMEOFISSUER", "TITLEOFCLASS", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL", "VALUE"],
    )
    return filing_vintage(
        info,
        cik="1",
        accession_number=accession,
        period=period,
        filing_date=filed,
        amendment_type=kind,
        source="known_truth",
        declared_rows=len(info),
        declared_value=float(info.VALUE.sum()),
        restricted=restricted,
    )


def combine(*packets: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.concat([p[0] for p in packets], ignore_index=True), pd.concat([p[1] for p in packets], ignore_index=True)


@pytest.mark.parametrize(("received", "declared", "accepted"), [(17, 20, True), (16, 20, False), (22, 20, True), (23, 20, False), (20, 20, True)])
def test_source_row_coverage_boundary_and_idempotent_save(
    sqlite_store: DataStore, caplog: pytest.LogCaptureFixture, received: int, declared: int, accepted: bool
) -> None:
    info = pd.DataFrame(
        [
            {"CUSIP": f"{number:09d}", "TITLEOFCLASS": "COM", "SSHPRNAMT": 100, "SSHPRNAMTTYPE": "SH", "VALUE": 1000}
            for number in range(1, received + 1)
        ]
    )
    kwargs = dict(
        cik="1",
        accession_number=f"coverage-{received}-{declared}",
        period="2024-06-30",
        filing_date="2024-08-14",
        amendment_type="original",
        source="known-truth",
        declared_rows=declared,
        declared_value=declared * 1000,
    )
    with caplog.at_level(logging.WARNING):
        if not accepted:
            with pytest.raises(ValueError, match="accepted ratio=85%-110%"):
                filing_vintage(info, **kwargs)
        else:
            manifest, rows = filing_vintage(info, **kwargs)
            assert manifest.source_row_count.iloc[0] == received and manifest.declared_row_count.iloc[0] == declared
            for table in (Tables.sec13f_manager_holdings, Tables.sec13f_hr):
                positions = rows.assign(ticker="AAA") if table == Tables.sec13f_hr else rows
                assert sqlite_store.save_13f_filing(manifest, positions, table=table) == received
                assert sqlite_store.save_13f_filing(manifest, positions, table=table) == 0
            _, state = public_books(manifest, rows, "2024-08-21", GRID)
            assert state.iloc[0].book_known and state.iloc[0].comparison_eligible and state.iloc[0].zero_eligible
    if accepted and declared != received:
        assert any(f"accepted {received}/{declared}" in record.getMessage() for record in caplog.records)
    print(
        f"SANITY: received{received}/declared{declared}, accepted={accepted};85?110% bounds, both destinations and normal exit eligibility validated."
    )


@pytest.mark.parametrize(
    ("filed", "cover_factor", "usd_factor"),
    [
        ("2022-08-14", 0.001, 1000),
        ("2024-08-14", 1000, 1),
        ("2024-08-14", 0.001, 1000),
        ("2024-08-14", 10, 1),
        ("2024-08-14", 0.1, 1),
        ("2024-08-14", 100, 1),
        ("2024-08-14", 0.01, 1),
        ("2024-08-14", 10000, 1),
        ("2024-08-14", 0.0001, 1),
    ],
)
def test_cover_unit_mismatch_preserves_raw_rows_and_own_unit_basis(
    sqlite_store: DataStore, caplog: pytest.LogCaptureFixture, filed: str, cover_factor: float, usd_factor: float
) -> None:
    info = pd.DataFrame([{"CUSIP": "111111111", "TITLEOFCLASS": "COM", "SSHPRNAMT": 100, "SSHPRNAMTTYPE": "SH", "VALUE": 1000}])
    with caplog.at_level(logging.WARNING):
        manifest, rows = filing_vintage(
            info,
            cik="1",
            accession_number="cover-units",
            period="2022-06-30",
            filing_date=filed,
            amendment_type="original",
            source="known-truth",
            declared_rows=1,
            declared_value=1000 / cover_factor,
        )
    assert rows.raw_value_usd.iloc[0] == 1000 and rows.shares.iloc[0] == 100
    assert rows.value_usd.iloc[0] == 1000 * usd_factor
    assert manifest.declared_value_total.iloc[0] == 1000 / cover_factor and manifest.source_value_total.iloc[0] == 1000
    assert sqlite_store.save_13f_filing(manifest, rows) == 1
    assert sqlite_store.save_13f_filing(manifest, rows) == 0
    assert any("cover-to-row factor" in record.getMessage() for record in caplog.records)
    print(
        f"SANITY: cover-to-row{cover_factor:g} aligns the source summary without another multiplication of row dollars; raw values, shares and replay preserved."
    )


def test_unexplained_value_mismatch_is_kept_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    info = pd.DataFrame([{"CUSIP": "111111111", "SSHPRNAMT": 100, "SSHPRNAMTTYPE": "SH", "VALUE": 1250}])
    with caplog.at_level(logging.WARNING):
        manifest, rows = filing_vintage(
            info,
            cik="1",
            accession_number="bad-value",
            period="2024-06-30",
            filing_date="2024-08-14",
            amendment_type="original",
            source="known-truth",
            declared_rows=1,
            declared_value=1000,
        )
    assert manifest.iloc[0].declared_value_total == 1000 and manifest.iloc[0].source_value_total == 1250 and len(rows) == 1
    assert any("the cover is a reference only, filing kept" in record.getMessage() for record in caplog.records)
    print("SANITY: a 25% cover/VALUE contradiction is kept with a WARNING; the cover total is stored as filed, never as a unit.")


@pytest.mark.parametrize("factor", [0.000001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 1000000.0])
@pytest.mark.parametrize("ratio,accepted", [(0.85, True), (1.10, True), (0.84, False), (1.11, False)])
def test_source_value_ratio_boundaries(factor: float, ratio: float, accepted: bool) -> None:
    source_value = 1_000_000_000_000 * factor * ratio
    assert reconcile_13f_source(100, 100, source_value, 1_000_000_000_000) == (factor if accepted else None)
    print(f"SANITY: monetary ratio={ratio:g}, cover-to-row factor={factor:g}, aligned={accepted}; an unaligned cover total returns None (warn-only).")


def test_empty_source_requires_zero_values() -> None:
    assert reconcile_13f_source(0, 0, 0, 0) == 1.0
    for source_value, declared_value in ((1, 0), (1, 1)):
        with pytest.raises(ValueError, match="empty source"):
            reconcile_13f_source(0, 0, source_value, declared_value)
    print("SANITY: a completed empty book must have zero raw value; inconsistent emptiness cannot create exits.")


def test_zero_count_cover_keeps_its_positive_total_as_a_complete_empty_book() -> None:
    assert reconcile_13f_source(0, 0, 0.0, 1234.0) == 1.0
    with pytest.raises(ValueError, match="row count"):
        reconcile_13f_source(0, 2, 0.0, 1234.0)
    for declared_rows, accounted in ((None, True), (0, True)):
        with pytest.raises(ValueError, match="empty source"):
            reconcile_13f_source(0, declared_rows, 0.0, 1234.0, source_accounted=accounted)
    with pytest.raises(ValueError, match="row count"):
        reconcile_13f_source(3, 0, 1234.0, 1234.0)
    print(
        "SANITY: cover 0 entries / 1,234 over 0 rows -> factor 1.0 (an all-exits book keeps its filed total); "
        "count 2 over 0 rows, a count-less or source-accounted empty source, and 0 entries over 3 rows still fail."
    )


def test_filing_vintage_stores_a_zero_count_cover_total_as_filed() -> None:
    manifest, book = filing_vintage(
        pd.DataFrame(columns=["CUSIP", "VALUE", "SSHPRNAMT", "SSHPRNAMTTYPE"]),
        cik="1",
        accession_number="all-exits",
        period="2001-03-31",
        filing_date="2001-05-15",
        amendment_type="original",
        source="known-truth",
        declared_rows=0,
        declared_value=1234.0,
    )
    row = manifest.iloc[0]
    assert book.empty and row.source_row_count == 0 and row.declared_row_count == 0
    assert row.source_value_total == 0.0 and row.declared_value_total == 1234.0 and row.parse_status == "complete"
    print(f"SANITY: 0-entry cover worth 1,234 -> complete manifest, 0 rows, declared_value_total {row.declared_value_total:g} kept raw as filed.")


@pytest.mark.parametrize("ratio", [2.0, 20.0, 200.0, 2000.0])
def test_decimal_alignment_does_not_accept_arbitrary_multiples(ratio: float) -> None:
    assert reconcile_13f_source(1, 1, 1000 * ratio, 1000) is None
    print(f"SANITY: ratio{ratio:g} is not a power-of-ten alignment: no cover factor (None), the caller only warns.")


@pytest.mark.parametrize(("source_value", "declared_value"), [(0.0, 1e6), (1e6, 0.0), (1e300, 1e-300), (1e-300, 1e300)])
def test_decimal_alignment_rejects_zero_or_unrepresentable_factors(source_value: float, declared_value: float) -> None:
    assert reconcile_13f_source(1, 1, source_value, declared_value) is None
    print("SANITY: zero or unrepresentable decimal alignment yields no cover factor (None) and never crashes.")


def test_tolerated_count_discrepancy_participates_in_normal_exits() -> None:
    prior = packet("prior-full", "2024-08-14", {f"{number:09d}": 100 for number in range(1, 8)})
    info = pd.DataFrame(
        [{"CUSIP": f"{number:09d}", "TITLEOFCLASS": "COM", "SSHPRNAMT": 110, "SSHPRNAMTTYPE": "SH", "VALUE": 1100} for number in range(2, 8)]
    )
    current = filing_vintage(
        info,
        cik="1",
        accession_number="current-tolerated",
        period="2024-09-30",
        filing_date="2024-11-14",
        amendment_type="original",
        source="known-truth",
        declared_rows=7,
        declared_value=7700,
    )
    manifests, holdings = combine(prior, current)
    out = vintage_changes(manifests, holdings, "2024-11-21", GRID).set_index("cusip")
    missing = out.loc["000000001"]
    assert missing.reported_exit and missing.comparison_known and missing.shares_change == -100
    assert out.loc["000000002", "shares_change"] == 10
    control = vintage_changes(*prior, "2024-08-21", GRID)
    pd.testing.assert_frame_equal(vintage_changes(manifests, holdings, "2024-08-21", GRID), control)
    print(
        "SANITY: tolerated6/7 row count is trusted for exits; the disappeared position exits, supported sibling grows and earlier public states remain unchanged."
    )


@pytest.mark.parametrize("column,new_value", [("declared_row_count", 21), ("declared_value_total", 20)])
def test_replay_rejects_changed_cover_metadata(sqlite_store: DataStore, column: str, new_value: float) -> None:
    manifest, rows = packet("fixed-cover", "2024-08-14", {f"{number:09d}": 100 for number in range(1, 21)})
    assert sqlite_store.save_13f_filing(manifest, rows) == 20
    changed = manifest.copy()
    changed.loc[0, column] = new_value
    with pytest.raises(ValueError, match="checksum conflict"):
        sqlite_store.save_13f_filing(changed, rows)
    assert sqlite_store.save_13f_filing(manifest, rows) == 0
    print(
        f"SANITY: changing accepted{column} cannot silently replay an immutable accession; original compatible checksum and repeat save remain valid."
    )


def test_prospective_replacement_additions_and_future_invariance() -> None:
    original = packet("orig", "2024-08-14", {"111111111": 100, "222222222": 200})
    added = packet("add", "2024-10-17", {"333333333": 30, "111111111": 5}, kind="additions")
    restated = packet("rest", "2025-05-14", {"111111111": 100, "333333333": 30}, kind="restatement")
    m, h = combine(original, added, restated)
    old, _ = public_books(m, h, "2024-08-21", GRID)
    old_control, _ = public_books(*original, "2024-08-21", GRID)
    pd.testing.assert_frame_equal(old, old_control)
    assert dict(old.set_index("cusip").shares) == {"111111111": 100, "222222222": 200}
    after_add, _ = public_books(m, h, "2024-10-18", GRID)
    assert dict(after_add.set_index("cusip").shares) == {"111111111": 105, "222222222": 200, "333333333": 30}
    before, _ = public_books(m, h, "2025-05-14", GRID)
    after, _ = public_books(m, h, "2025-05-15", GRID)
    assert "222222222" in set(before.cusip) and "222222222" not in set(after.cusip)
    assert dict(original[1].set_index("cusip").shares) == {"111111111": 100, "222222222": 200}
    print("SANITY: original immutable; additions sum; restatement removes only from its public date; future prefix identical.")


def test_active_prior_correction_never_replaces_newer_levels() -> None:
    previous = packet("prev", "2024-08-14", {"111111111": 100})
    current = packet("curr", "2024-11-14", {"111111111": 120}, period="2024-09-30")
    revised = packet("rev", "2024-12-01", {"111111111": 90}, kind="restatement")
    m, h = combine(previous, current, revised)
    first = vintage_changes(m, h, "2024-11-21", GRID)
    later = vintage_changes(m, h, "2024-12-02", GRID)
    assert first.iloc[0].shares == later.iloc[0].shares == 120
    assert first.iloc[0].shares_change == 20
    assert later.iloc[0].shares_change == 30
    assert pd.Timestamp(later.iloc[0].period) == pd.Timestamp("2024-09-30")
    print("SANITY: prior amendment updates active comparison prospectively; current level remains 120.")


def test_empty_complete_book_and_unknown_addition_are_distinct() -> None:
    m, h = combine(packet("orig", "2024-08-14", {"111111111": 100}), packet("zero", "2024-11-14", {}, period="2024-09-30"))
    out = vintage_changes(m, h, "2024-11-21", GRID)
    assert out.iloc[0].shares == 0 and out.iloc[0].reported_exit
    assert out.iloc[0].shares_change == -100
    m2, h2 = packet("orphan", "2024-11-14", {"111111111": 20}, kind="additions")
    rows, states = public_books(m2, h2, "2024-11-21", GRID)
    assert rows.empty and not states.iloc[0].book_known
    print("SANITY: completed empty book supports a reported exit; additions without a base stay unknown.")


def test_restriction_blocks_exit_inference() -> None:
    m, h = combine(
        packet("orig", "2024-08-14", {"111111111": 100}), packet("restricted", "2024-11-14", {"222222222": 20}, period="2024-09-30", restricted=True)
    )
    out = vintage_changes(m, h, "2024-11-21", GRID)
    assert out.shares_change.isna().all() and not out.reported_exit.any()
    print("SANITY: confidentiality cannot create sells or exits.")


def test_atomic_save_replay_conflict_and_accession_resume(sqlite_store: DataStore) -> None:
    manifest, rows = packet("original", "2024-08-14", {"111111111": 100})
    assert sqlite_store.save_13f_filing(manifest, rows) == 1
    assert sqlite_store.save_13f_filing(manifest, rows) == 0
    changed, changed_rows = packet("original", "2024-08-14", {"111111111": 99})
    with pytest.raises(ValueError, match="checksum"):
        sqlite_store.save_13f_filing(changed, changed_rows)
    amendment, amended_rows = packet("amendment", "2025-05-14", {"111111111": 90}, kind="restatement")

    def crash(conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, executemany: bool) -> None:
        if (
            statement.startswith("INSERT")
            and Tables.sec13f_manager_holdings.name in statement
            and context.compiled_parameters[0].get("parse_status") == "complete"
        ):
            raise RuntimeError("injected crash after holdings")

    event.listen(sqlite_store.engine, "before_cursor_execute", crash)
    with pytest.raises(RuntimeError, match="injected"):
        sqlite_store.save_13f_filing(amendment, amended_rows)
    event.remove(sqlite_store.engine, "before_cursor_execute", crash)
    assert len(completed(sqlite_store)) == 1
    assert len(sqlite_store.load(Tables.sec13f_manager_holdings)) == 1
    context = SimpleNamespace(store=sqlite_store, log=logging.getLogger(__name__))
    assert pending_vintages(context, "0000000001", ["amendment", "original", "missing_original"]) == ["amendment", "missing_original"]
    print("SANITY: atomic rollback, identical replay, visible payload conflict, and accession gaps all enforced.")


def test_source_completion_and_amendment_identity_are_validated(sqlite_store: DataStore) -> None:
    manifest, rows = packet("valid", "2024-08-14", {"111111111": 100})
    bad = manifest.copy()
    bad.loc[0, "source_row_count"] = 0
    with pytest.raises(ValueError, match="source row count"):
        sqlite_store.save_13f_filing(bad, rows)
    with pytest.raises(ValueError, match="amendment"):
        packet("bad", "2024-08-14", {"111111111": 1}, kind="ambiguous")
    print("SANITY: unknown amendment semantics are rejected rather than interpreted as a full replacement.")


def test_present_unknown_quantity_is_not_an_exit() -> None:
    m, h = combine(packet("prev", "2024-08-14", {"111111111": 100}), packet("curr", "2024-11-14", {"111111111": 120}, period="2024-09-30"))
    h.loc[h.accession_number.eq("curr"), "shares"] = float("nan")
    out = vintage_changes(m, h, "2024-11-21", GRID)
    assert out.iloc[0].shares != out.iloc[0].shares
    assert not out.iloc[0].reported_exit and not out.iloc[0].comparison_known
    print("SANITY: a present unknown quantity stays unknown rather than zero or exit.")


def test_copied_book_addition_updates_positions_without_doubling() -> None:
    original = packet("orig", "2024-08-14", {"111111111": 100, "222222222": 200, "333333333": 300})
    amendment = packet("add", "2024-10-17", {"111111111": 100, "222222222": 205, "333333333": 300}, kind="additions")
    m, h = combine(original, amendment)
    before, _ = public_books(m, h, "2024-10-17", GRID)
    rows, states = public_books(m, h, "2024-10-18", GRID)
    assert dict(rows.set_index("cusip").shares) == {"111111111": 100, "222222222": 205, "333333333": 300}
    assert states.iloc[0].zero_eligible and states.iloc[0].comparison_eligible
    assert states.iloc[0].resolution_status == "inferred_position_patch"
    assert states.iloc[0].public_at == states.iloc[0].last_observed_at
    pd.testing.assert_frame_equal(before, public_books(*original, "2024-10-17", GRID)[0])
    assert m.iloc[-1].amendment_type == "additions"
    print("SANITY: copied majority updates all reported positions prospectively; no doubling or raw flag changes.")


def test_copied_partial_packet_preserves_omissions_and_present_comparisons() -> None:
    prior = packet(
        "prior", "2024-05-14", {"111111111": 90, "222222222": 190, "333333333": 290, "444444444": 390, "555555555": 490}, period="2024-03-31"
    )
    original = packet("orig", "2024-08-14", {"111111111": 100, "222222222": 200, "333333333": 300, "444444444": 400, "555555555": 500})
    patch = packet("patch", "2024-10-17", {"111111111": 100, "222222222": 200, "333333333": 300, "444444444": 405}, kind="additions")
    m, h = combine(prior, original, patch)
    rows, states = public_books(m, h, "2024-10-18", GRID)
    state = states.loc[states.period.eq(pd.Timestamp("2024-06-30"))].iloc[0]
    current = rows.loc[rows.period.eq(state.period)].set_index("cusip")
    assert current.loc["444444444", "shares"] == 405
    assert pd.isna(current.loc["555555555", "shares"])
    assert not state.zero_eligible and state.comparison_eligible
    result = vintage_changes(m, h, "2024-10-18", GRID).set_index("cusip")
    assert result.loc["444444444", "shares_change"] == 15
    assert result.loc["444444444", "comparison_known"]
    assert pd.isna(result.loc["555555555", "shares_change"])
    assert not result.loc["555555555", "reported_exit"]
    small = packet("small", "2024-10-17", {"111111111": 100, "666666666": 20, "777777777": 30}, kind="additions")
    supplied, status = public_books(*combine(original, small), "2024-10-18", GRID)
    assert supplied.set_index("cusip").loc["111111111", "shares"] == 200
    assert status.iloc[0].resolution_status == "resolved"
    print("SANITY: inferred patch recovers present corrections; omissions unknown, genuine small overlapping supplement remains additive.")


@pytest.mark.parametrize("quantity,value", [(float("inf"), 10), (1, float("inf")), (float("nan"), 10)])
def test_nonfinite_source_is_rejected(quantity: float, value: float) -> None:
    info = pd.DataFrame([{"CUSIP": "111111111", "SSHPRNAMT": quantity, "SSHPRNAMTTYPE": "SH", "VALUE": value}])
    with pytest.raises(ValueError, match="incomplete|negative|total"):
        filing_vintage(
            info,
            cik="1",
            accession_number="bad",
            period="2024-06-30",
            filing_date="2024-08-14",
            amendment_type="original",
            source="fixture",
            declared_rows=1,
            declared_value=value,
        )
    print("SANITY: nonfinite source quantities and values cannot complete a filing.")


def test_save_reconciles_actual_raw_legs(sqlite_store: DataStore) -> None:
    manifest, rows = packet("original", "2024-08-14", {"111111111": 100})
    rows.loc[0, "raw_value_usd"] = 0
    with pytest.raises(ValueError, match="raw value total"):
        sqlite_store.save_13f_filing(manifest, rows)
    print("SANITY: altered book rows cannot pass by copying valid manifest totals.")


def test_date_fallback_timezone_holiday_and_calendar_end() -> None:
    m, h = packet("orig", "2024-08-14", {"111111111": 100})
    m.loc[0, "accepted_at"] = pd.Timestamp("2024-08-22 01:30:00")
    grid = GRID[GRID != pd.Timestamp("2024-08-22")]
    assert filing_public_dates(m, grid).iloc[0] == pd.Timestamp("2024-08-23")
    assert pd.isna(filing_public_dates(m, GRID[GRID <= "2024-08-21"]).iloc[0])
    print("SANITY: UTC acceptance converted to SEC local day; next real session and calendar exhaustion enforced.")


@pytest.mark.live
def test_disposable_postgres_existing_key_migration(monkeypatch: pytest.MonkeyPatch) -> None:
    """An existing PostgreSQL key is altered without deleting a single source row."""
    url = os.environ.get("PEA_13F_TEST_DATABASE_URL")
    if not url:
        pytest.skip("requires dedicated disposable PostgreSQL URL")
    parsed = make_url(url)
    assert parsed.host == "127.0.0.1" and parsed.port == 55433 and parsed.database == "p3" and parsed.username == "p3"
    store = DataStore(create_engine(url))
    tables = (Tables.sec13f_hr, Tables.sec13f_manager_holdings)
    for table in tables:
        if store.exists(table):
            store.drop(table)
    original = packet("migration-original", "2024-08-14", {"111111111": 100})
    before: dict[str, pd.DataFrame] = {}
    with monkeypatch.context() as legacy:
        for table in tables:
            old = replace(table, pk=tuple(key for key in table.pk if key != "accession_number"))
            legacy.setitem(schema.BY_NAME, table.name, old)
            legacy.setattr(Tables, table.name, old)
        for table in tables:
            rows = original[1].drop(columns="accession_number")
            if table == tables[0]:
                rows = rows.assign(ticker="AAA")
            store.save(schema.resolve(table.name), rows)
            before[table.name] = store.load(schema.resolve(table.name))
    assert set(store.migrate_13f_tables()) == {table.name for table in tables}
    assert store.migrate_13f_tables() == []
    for table in tables:
        pd.testing.assert_frame_equal(before[table.name], store.load(table))
        assert set(store.load(table).accession_number) == {"_legacy"}
        rows = original[1].assign(ticker="AAA") if table == tables[0] else original[1]
        assert store.save_13f_filing(original[0], rows, table=table) == 1
        amended = packet("migration-amendment", "2025-05-14", {"111111111": 90}, kind="restatement")
        rows = amended[1].assign(ticker="AAA") if table == tables[0] else amended[1]
        assert store.save_13f_filing(amended[0], rows, table=table) == 1
        assert set(store.load(table).accession_number) == {"_legacy", "migration-original", "migration-amendment"}
        assert len(store.load(table, where={"parse_status": "complete"}, markers=True)) == 2
    for table in tables:
        store.drop(table)
    store.engine.dispose()
    print(
        "SANITY: PostgreSQL rekey retains exact legacy rows as unverified; original/amendment coexist in both existing tables; migration is idempotent."
    )


@pytest.mark.live
def test_disposable_postgres_atomicity() -> None:
    url = os.environ.get("PEA_13F_TEST_DATABASE_URL")
    if not url:
        pytest.skip("requires dedicated disposable P3 PostgreSQL URL")
    parsed = make_url(url)
    assert parsed.host == "127.0.0.1" and parsed.port == 55433 and parsed.database == "p3" and parsed.username == "p3"
    store = DataStore(create_engine(url))
    m, h = packet("pg-original", "2024-08-14", {"111111111": 100})
    assert store.save_13f_filing(m, h) == 1
    assert store.save_13f_filing(m, h) == 0
    m2, h2 = packet("pg-amend", "2025-05-14", {"111111111": 90}, kind="restatement")

    def crash(conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, executemany: bool) -> None:
        if (
            statement.startswith("INSERT")
            and Tables.sec13f_manager_holdings.name in statement
            and context.compiled_parameters[0].get("parse_status") == "complete"
        ):
            raise RuntimeError("injected PostgreSQL failure")

    event.listen(store.engine, "before_cursor_execute", crash)
    with pytest.raises(RuntimeError, match="injected PostgreSQL"):
        store.save_13f_filing(m2, h2)
    event.remove(store.engine, "before_cursor_execute", crash)
    assert len(completed(store)) == len(store.load(Tables.sec13f_manager_holdings)) == 1
    assert store.save_13f_filing(m2, h2) == 1
    empty, no_rows = packet("pg-empty", "2025-06-01", {}, kind="restatement")
    assert store.save_13f_filing(empty, no_rows) == 0
    assert len(completed(store)) == 3
    concurrent, concurrent_rows = packet("pg-concurrent", "2025-06-02", {"111111111": 80}, kind="restatement")
    with ThreadPoolExecutor(max_workers=2) as workers:
        outcomes = list(workers.map(lambda _: store.save_13f_filing(concurrent, concurrent_rows), range(2)))
    assert sorted(outcomes) == [0, 1] and len(completed(store)) == 4
    changed, changed_rows = packet("pg-original", "2024-08-14", {"111111111": 99})
    with pytest.raises(ValueError, match="checksum"):
        store.save_13f_filing(changed, changed_rows)
    store.engine.dispose()
    print("SANITY: real disposable PostgreSQL rollback, retry, empty completion, immutable replay/conflict passed; production untouched.")


def test_bounded_recovery_dry_run_cap_and_original_gap(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    context = SimpleNamespace(store=sqlite_store, log=logging.getLogger(__name__))
    old = SimpleNamespace(accession_number="orig", filed=pd.Timestamp("2024-08-14"), doc_url="fixture-original")
    new = SimpleNamespace(accession_number="amend", filed=pd.Timestamp("2025-05-14"), doc_url="fixture-amendment")
    monkeypatch.setattr(managers, "configure", lambda context: None)
    monkeypatch.setattr(managers, "_listed_filings", lambda cik, since: [(old, pd.Timestamp("2024-06-30")), (new, pd.Timestamp("2024-06-30"))])
    packets = {
        "orig": packet("orig", "2024-08-14", {"111111111": 100}),
        "amend": packet("amend", "2025-05-14", {"111111111": 90}, kind="restatement"),
    }
    monkeypatch.setattr(managers, "read_filing_vintage", lambda stamp: packets[stamp.accession_number])
    args = dict(
        ciks=["0000000001"],
        start_period="2024-06-30",
        end_period="2024-06-30",
        filed_through="2025-06-01",
        accessions=["orig", "amend"],
        max_filings=2,
    )
    preview = managers.recover_13f_vintages(context, **args)
    assert preview["pending_accessions"] == 2 and not sqlite_store.exists(Tables.sec13f_manager_holdings)
    with pytest.raises(ValueError, match="exceeding cap"):
        managers.recover_13f_vintages(context, **{**args, "max_filings": 1}, dry_run=False)
    sqlite_store.save_13f_filing(*packets["amend"])
    assert managers.recover_13f_vintages(context, **args, dry_run=False)["saved_rows"] == 1
    assert len(completed(sqlite_store)) == 2
    assert managers.recover_13f_vintages(context, **args, dry_run=False)["saved_rows"] == 0
    assert sqlite_store.exists(Tables.sec13f_hr)
    print("SANITY: dry run writes nothing; cap enforced before writes; stored amendment cannot hide original gap; replay is idempotent.")


def test_completed_empty_book_can_create_schema_first(sqlite_store: DataStore) -> None:
    assert sqlite_store.save_13f_filing(*packet("empty-first", "2024-08-14", {})) == 0
    assert sqlite_store.save_13f_filing(*packet("populated-second", "2024-11-14", {"111111111": 100}, period="2024-09-30")) == 1
    assert len(completed(sqlite_store)) == 2
    print("SANITY: empty-first source completion creates a stable schema that accepts later holdings.")


def test_vintage_recovery_cli_scope_and_legacy_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("src.data_extract.cli")
    context = SimpleNamespace(store=None)
    config = SimpleNamespace(data_extract=SimpleNamespace(years_history=9))
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(module, "get_config_context", lambda *args, **kwargs: (config, context))
    monkeypatch.setattr(module, "recover_13f_vintages", lambda ctx, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(module, "fetch_13f_managers", lambda ctx, **kwargs: calls.append(kwargs))
    runner = CliRunner()
    options = [
        "thirteen-f-managers",
        "--vintages",
        "--cik",
        "0000895421",
        "--accession",
        "0000895421-24-000408",
        "--period-start",
        "2024-06-30",
        "--period-end",
        "2024-06-30",
        "--filed-through",
        "2025-05-15",
        "--max-filings",
        "1",
        "--dry-run",
    ]
    result = runner.invoke(module.cli, options)
    assert result.exit_code == 0, result.output
    assert calls[-1] == dict(
        ciks=["0000895421"],
        start_period="2024-06-30",
        end_period="2024-06-30",
        filed_through="2025-05-15",
        accessions=["0000895421-24-000408"],
        max_filings=1,
        dry_run=True,
        archives=None,
    )
    assert runner.invoke(module.cli, ["thirteen-f-managers", "--vintages"]).exit_code == 2
    assert runner.invoke(module.cli, ["thirteen-f-managers", "--max-filings", "1"]).exit_code == 2
    assert runner.invoke(module.cli, ["thirteen-f-managers"]).exit_code == 0
    assert calls[-1] == {"years_history": 9}
    print("SANITY: exact scoped recovery CLI passes; missing scope is rejected; legacy invocation preserved.")


def test_recovered_real_amendments_and_active_prior_are_causal() -> None:
    """Read-only economic proof against actual SEC source vintages, never synthetic portfolios."""
    if os.getenv("PEA_LIVE_TESTS") != "1":
        pytest.skip("real-source proof requires explicit PEA_LIVE_TESTS=1")
    _, context = get_config_context(str(Path(__file__).resolve().parents[3] / "configs"), use_cache=False, save=False)
    if "parse_status" not in context.store.columns(Tables.sec13f_manager_holdings):
        pytest.skip("source versions have not been replayed into the existing holdings table")
    ciks = ["0000093751", "0000102909", "0000895421", "0001067983"]
    mc = [
        "accession_number",
        "cik",
        "period",
        "filing_date",
        "amendment_type",
        "amendment_number",
        "report_type",
        "accepted_at",
        "restricted",
        "parse_status",
        "parsed_row_count",
    ]
    hc = [
        "cik",
        "accession_number",
        "period",
        "filing_date",
        "cusip",
        "position_type",
        "class_evidence",
        "shares",
        "value_usd",
        "call_shares",
        "call_value",
        "put_shares",
        "put_value",
        "debt_prn",
        "debt_value",
        "other_value",
        "raw_value_usd",
        "raw_call_value",
        "raw_put_value",
        "raw_debt_value",
        "raw_other_value",
    ]
    m = context.store.load(
        Tables.sec13f_manager_holdings,
        columns=mc,
        where={"cik": ciks, "cusip": "_empty", "parse_status": "complete"},
        markers=True,
        since="2023-01-01",
        until="2026-06-30",
        date_col="period",
    )
    h = pd.concat(
        list(
            context.store.iter_load(
                Tables.sec13f_manager_holdings,
                columns=hc,
                where={"cik": ciks, "accession_number": m.accession_number.tolist()},
                since="2023-01-01",
                until="2026-06-30",
                date_col="period",
            )
        ),
        ignore_index=True,
    )
    calendar = context.store.load(Tables.prices, columns=["date"], where={"ticker": "AAPL"}, since="2022-01-01", until="2026-10-08")
    grid = pd.DatetimeIndex(calendar.date).unique().sort_values()
    cases = [
        ("0000895421-24-000408", "0000895421-25-000378", "2025-05-15", 7845, 7845),
        ("0000093751-26-000507", "0000093751-26-000583", "2026-09-23", 4177, 4261),
        ("0001104659-24-023150", "0001104659-24-032991", "2024-03-12", 4484, 4497),
        ("0000950123-24-002518", "0000950123-24-005664", "2024-05-16", 41, 42),
    ]
    original_snapshot = h.copy(deep=True)
    for original, amendment, day, before_count, after_count in cases:
        selection = m.loc[m.accession_number.isin([original, amendment])]
        rows = h.loc[h.accession_number.isin([original, amendment])]
        before_day = grid[grid < pd.Timestamp(day)][-1]
        before, _ = public_books(selection, rows, before_day, grid)
        control, _ = public_books(
            selection.loc[selection.accession_number.eq(original)], rows.loc[rows.accession_number.eq(original)], before_day, grid
        )
        pd.testing.assert_frame_equal(before, control)
        after, states = public_books(selection, rows, day, grid)
        assert len(before) == before_count and len(after) == after_count
        assert states.iloc[0].resolution_status == "resolved"
        if original == "0000895421-24-000408":
            assert before.set_index("cusip").loc["02215L209", "shares"] == 4134516
            assert after.set_index("cusip").loc["02215L209", "shares"] == 294270
        if original == "0001104659-24-023150":
            overlap = before.set_index("cusip").shares.to_frame("old").join(after.set_index("cusip").shares.rename("new"), how="inner")
            assert overlap.old.ne(overlap.new).sum() == 269
    ms = m.loc[m.cik.eq("0000895421")]
    hs = h.loc[h.cik.eq("0000895421")]
    a = vintage_changes(ms, hs, "2025-05-14", grid).set_index("cusip")
    b = vintage_changes(ms, hs, "2025-05-15", grid).set_index("cusip")
    pd.testing.assert_series_equal(a.shares, b.shares)
    assert a.loc["02215L209", "shares"] == b.loc["02215L209", "shares"] == 4116456
    assert a.loc["02215L209", "prior_shares"] == 4111624
    assert b.loc["02215L209", "prior_shares"] == 271378
    keep = ms.accession_number.ne("0000895421-25-000378")
    without_stale, hs_keep = ms.loc[keep], hs.loc[hs.accession_number.isin(ms.loc[keep].accession_number)]
    pd.testing.assert_frame_equal(b.reset_index(), vintage_changes(without_stale, hs_keep, "2025-05-15", grid))
    q4_amended = vintage_changes(ms, hs, "2025-05-16", grid).set_index("cusip")
    assert q4_amended.loc["02215L209", "shares"] == 276210
    assert q4_amended.loc["02215L209", "prior_shares"] == 271378
    latest = vintage_changes(ms, hs, "2025-05-20", grid).set_index("cusip")
    assert latest.loc["02215L209", "shares"] == 271741
    assert latest.loc["02215L209", "prior_shares"] == 276210
    assert not latest.loc["02215L209", "comparison_known"]  # Combination report cannot prove zeros.
    pd.testing.assert_frame_equal(h, original_snapshot)
    print(
        "SANITY: four real source cases preserve pre-amendment books; MS current levels unchanged, "
        "active prior corrected only on May15; stale Q2 cannot replace Q4/Q1; source snapshots untouched."
    )


def test_stale_amendment_is_audit_only_and_value_only_revision_is_emitted() -> None:
    previous = packet("q2", "2024-08-14", {"111111111": 100})
    current = packet("q3", "2024-11-14", {"111111111": 120}, period="2024-09-30")
    newest = packet("q4", "2025-02-14", {"111111111": 130}, period="2024-12-31")
    stale = packet("stale", "2025-03-15", {"111111111": 90}, kind="restatement")
    value_only = packet("value", "2025-03-20", {"111111111": 130}, kind="restatement", period="2024-12-31")
    value_only[1].loc[0, "value_usd"] *= 2
    other_previous = packet("bq2", "2024-08-14", {"111111111": 50})
    other_current = packet("bq4", "2025-02-14", {"111111111": 60}, period="2024-12-31")
    other_stale = packet("bstale", "2025-03-20", {"111111111": 40}, kind="restatement")
    other = [tuple(frame.assign(cik="0000000002") for frame in pair) for pair in (other_previous, other_current, other_stale)]
    m, h = combine(previous, current, newest, stale, value_only, *other)
    publications = vintage_publications(m, h, GRID, until="2025-03-31")
    assert set(publications.loc[publications.as_of.eq(pd.Timestamp("2025-03-21")), "cik"]) == {"0000000001"}
    assert pd.Timestamp("2025-03-17") not in set(publications.as_of)
    correction = publications.loc[publications.as_of.eq(pd.Timestamp("2025-03-21"))].iloc[0]
    assert correction.shares == 130 and correction.value_usd == value_only[1].value_usd.iloc[0]
    history, _ = public_books(m, h, "2025-03-17", GRID)
    assert history.loc[history.cik.eq("0000000001") & history.period.eq(pd.Timestamp("2024-06-30")), "shares"].iloc[0] == 90
    print("SANITY: stale correction remains in historical books but emits no current revision; value-only active correction emitted.")


def test_repeated_original_uses_report_quarter_and_latest_public_prior() -> None:
    original = packet("first-q2", "2024-08-14", {"111111111": 100, "222222222": 40})
    current = packet("first-q3", "2024-11-14", {"111111111": 120}, period="2024-09-30")
    # The source incorrectly calls this another original, but its report period is Q2.
    repeated = packet("second-q2", "2024-12-01", {"111111111": 90})
    m, h = combine(original, current, repeated)
    before = vintage_changes(m, h, "2024-11-21", GRID)
    control = vintage_changes(*combine(original, current), "2024-11-21", GRID)
    pd.testing.assert_frame_equal(before, control)
    after = vintage_changes(m, h, "2024-12-02", GRID)
    position = after.set_index("cusip").loc["111111111"]
    assert position.period == pd.Timestamp("2024-09-30") and position.shares == 120
    assert position.prior_period == pd.Timestamp("2024-06-30") and position.prior_shares == 90
    assert position.shares_change == 30 and position.prior_public_at == pd.Timestamp("2024-12-02")
    books, _ = public_books(m, h, "2024-12-02", GRID)
    prior = books.loc[books.period.eq(pd.Timestamp("2024-06-30"))]
    assert set(prior.cusip) == {"111111111"}
    assert set(original[1].cusip) == {"111111111", "222222222"}
    print("SANITY: repeated original updates its report quarter prospectively; newest quarter unchanged, baseline uses latest public prior version.")


def test_full_copy_addition_guard_survives_source_unit_transition() -> None:
    packets = []
    for accession, filed, kind, raw_value in (
        ("thousands", "2022-11-14", "original", 1),
        ("dollars", "2023-05-15", "additions", 1000),
    ):
        info = pd.DataFrame(
            [
                {
                    "CUSIP": cusip,
                    "NAMEOFISSUER": "Example",
                    "TITLEOFCLASS": "COM",
                    "SSHPRNAMT": 100,
                    "SSHPRNAMTTYPE": "SH",
                    "PUTCALL": "",
                    "VALUE": raw_value,
                }
                for cusip in ("111111111", "222222222", "333333333")
            ]
        )
        packets.append(
            filing_vintage(
                info,
                cik="1",
                accession_number=accession,
                period="2022-09-30",
                filing_date=filed,
                amendment_type=kind,
                source="known_truth",
                declared_rows=3,
                declared_value=raw_value * 3,
                units=declared_units(filed, schema_version="X0102" if kind == "original" else "X0202"),
            )
        )
    m, h = combine(*packets)
    assert h.raw_value_usd.tolist() == [1] * 3 + [1000] * 3 and h.value_usd.tolist() == [1000] * 6
    grid = pd.bdate_range("2022-01-01", "2024-01-31")
    before, _ = public_books(m, h, "2023-05-15", grid)
    after, state = public_books(m, h, "2023-05-16", grid)
    assert before.shares.iloc[0] == after.shares.iloc[0] == 100
    assert after.value_usd.iloc[0] == 1000 and state.resolution_status.iloc[0] == "inferred_position_patch"
    assert state.zero_eligible.iloc[0]
    unknown = h.copy()
    unknown.loc[unknown.accession_number.eq("dollars"), "raw_value_usd"] = 2000
    unknown["value_usd"] = pd.Series(pd.NA, index=unknown.index, dtype="Float64")
    supplied, status = public_books(m, unknown, "2023-05-16", grid)
    assert supplied.shares.iloc[0] == 200 and pd.isna(supplied.value_usd.iloc[0])
    assert status.resolution_status.iloc[0] == "resolved"
    print("SANITY: cross-unit copied book updates without doubling; unknown nullable dollars never establish equality.")


def test_inferred_prior_and_later_supplement_keep_provenance() -> None:
    original = packet("orig", "2024-08-14", {"111111111": 100, "222222222": 200, "333333333": 300})
    patch = packet("patch", "2024-10-17", {"111111111": 100, "222222222": 205, "333333333": 300}, kind="additions")
    supplement = packet("supplement", "2024-10-18", {"444444444": 40}, kind="additions")
    current = packet("current", "2024-11-14", {"111111111": 110}, period="2024-09-30")
    m, h = combine(original, patch, supplement, current)
    _, states = public_books(m, h, "2024-11-20", GRID)
    prior = states.loc[states.period.eq(pd.Timestamp("2024-06-30"))].iloc[0]
    assert prior.resolution_status == "inferred_position_patch"
    result = vintage_changes(m, h, "2024-11-20", GRID)
    assert result.resolution_status.eq("resolved").all()
    assert result.prior_resolution_status.eq("inferred_position_patch").all()
    clean = packet("clean", "2024-11-21", {"111111111": 100}, kind="restatement")
    result = vintage_changes(*combine(original, patch, supplement, current, clean), "2024-11-22", GRID)
    assert result.prior_resolution_status.eq("resolved").all()
    print("SANITY: inference survives genuine supplements and is visible in the prior baseline; only clean replacement clears it.")


def test_inferred_patch_does_not_turn_omitted_typed_leg_into_exit() -> None:
    prior = packet("prior", "2024-05-14", {"111111111": 90, "222222222": 190, "333333333": 290}, period="2024-03-31")
    original = packet("orig", "2024-08-14", {"111111111": 100, "222222222": 200, "333333333": 300})
    source = pd.DataFrame(
        [
            {"CUSIP": cusip, "TITLEOFCLASS": "COM", "SSHPRNAMT": quantity, "SSHPRNAMTTYPE": "SH", "PUTCALL": option, "VALUE": quantity * 10}
            for cusip, quantity, option in (("111111111", 100, ""), ("222222222", 200, ""), ("333333333", 20, "Call"))
        ]
    )
    patch = filing_vintage(
        source,
        cik="1",
        accession_number="patch",
        period="2024-06-30",
        filing_date="2024-10-17",
        amendment_type="additions",
        source="known_truth",
        declared_rows=3,
        declared_value=3200,
    )
    m, h = combine(prior, original, patch)
    result = vintage_changes(m, h, "2024-10-18", GRID).set_index("cusip")
    assert pd.isna(result.loc["333333333", "shares"]) and not result.loc["333333333", "reported_exit"]
    assert not result.loc["333333333", "comparison_known"]
    assert result.loc["111111111", "shares_change"] == 10
    assert patch[1].set_index("cusip").loc["333333333", "shares"] == 0
    repeated = filing_vintage(
        source,
        cik="1",
        accession_number="patch2",
        period="2024-06-30",
        filing_date="2024-10-18",
        amendment_type="additions",
        source="known_truth",
        declared_rows=3,
        declared_value=3200,
    )
    again = vintage_changes(*combine(prior, original, patch, repeated), "2024-10-21", GRID).set_index("cusip")
    assert pd.isna(again.loc["333333333", "shares"]) and not again.loc["333333333", "reported_exit"]
    one = packet("one", "2024-08-14", {"111111111": 100})
    extra = packet("extra", "2024-10-17", {"111111111": 100}, kind="additions")
    supplied, state = public_books(*combine(one, extra), "2024-10-18", GRID)
    assert supplied.shares.iloc[0] == 200 and state.resolution_status.iloc[0] == "resolved"
    print("SANITY: omitted common leg stays unknown despite reported CALL; one coincident equal-size addition cannot classify a book.")


def test_cohort_compares_books_not_arriving_holders() -> None:
    previous_a = packet("a-prev", "2024-08-14", {"111111111": 100})
    current_a = packet("a-cur", "2024-11-14", {"111111111": 110}, period="2024-09-30")
    previous_b = tuple(f.assign(cik="0000000002") for f in packet("b-prev", "2024-08-14", {"111111111": 900}))
    current_b = tuple(f.assign(cik="0000000002") for f in packet("b-cur", "2024-11-25", {"222222222": 50}, period="2024-09-30"))
    current_c = tuple(f.assign(cik="0000000003") for f in packet("c-cur", "2024-11-20", {"111111111": 500}, period="2024-09-30"))
    m, h = combine(previous_a, current_a, previous_b, current_b, current_c)
    keys = {"111111111": "A"}
    early = vintage_cohort_snapshot(m, h, "2024-11-15", GRID, security_tickers=keys, settle_days=0).iloc[0]
    assert early.shares == 110 and early.public_books == early.paired_books == 1
    assert early.shares_chg == pytest.approx(0.1) and early.exit_ratio == 0
    assert early.prior_share_coverage == 0.1 and early.public_prior_shares == 1000
    assert early.publication_age_days == 0
    new = vintage_cohort_snapshot(m, h, "2024-11-21", GRID, security_tickers=keys, settle_days=0).iloc[0]
    assert new.shares == 610 and new.public_books == 2 and new.paired_books == 1
    assert new.shares_chg == pytest.approx(0.1) and new.new_ratio == 0
    late = vintage_cohort_snapshot(m, h, "2024-11-26", GRID, security_tickers=keys, settle_days=0).iloc[0]
    assert late.shares == 610 and late.public_books == 3 and late.paired_books == 2
    assert late.matched_shares == 110 and late.matched_prior_shares == 1000
    assert late.shares_chg == pytest.approx(-0.89) and late.exit_ratio == 0.5
    assert late.breadth_chg == -0.5
    assert late.prior_share_coverage == 1 and late.public_prior_books == 2
    pd.testing.assert_frame_equal(
        vintage_cohort_snapshot(*combine(previous_a, current_a, previous_b), "2024-11-15", GRID, security_tickers=keys, settle_days=0),
        vintage_cohort_snapshot(m, h, "2024-11-15", GRID, security_tickers=keys, settle_days=0),
    )
    print(
        "SANITY: no waiting buffer; unfiled B cannot sell; C affects levels only; B's complete public absence then supports exit; cohort denominators agree."
    )


def test_cohort_empty_restricted_and_missing_books() -> None:
    previous = packet("prev", "2024-08-14", {"111111111": 100})
    empty = packet("empty", "2024-11-14", {}, period="2024-09-30")
    keys = {"111111111": "A"}
    out = vintage_cohort_snapshot(*combine(previous, empty), "2024-11-15", GRID, security_tickers=keys, settle_days=0).iloc[0]
    assert out.public_books == out.paired_books == out.matched_books == 1
    assert out.shares == 0 and out.shares_chg == -1 and out.exit_ratio == 1
    restricted = packet("restricted", "2024-11-14", {}, period="2024-09-30", restricted=True)
    out = vintage_cohort_snapshot(*combine(previous, restricted), "2024-11-15", GRID, security_tickers=keys, settle_days=0).iloc[0]
    assert out.public_books == 1 and out.paired_books == 0
    assert pd.isna(out.shares) and pd.isna(out.shares_chg) and pd.isna(out.exit_ratio)
    failed = empty[0].assign(parse_status="failed"), empty[1]
    assert vintage_cohort_snapshot(*combine(previous, failed), "2024-11-15", GRID, security_tickers=keys, period="2024-09-30", settle_days=0).empty
    orphan = packet("orphan", "2024-11-14", {"111111111": 20}, period="2024-09-30", kind="additions")
    assert vintage_cohort_snapshot(*orphan, "2024-11-15", GRID, security_tickers=keys, settle_days=0).empty
    print("SANITY: empty complete book stays in denominator; restricted, failed, orphan books cannot establish zero or exits.")


def test_cohort_prior_amendment_and_value_only_events_are_prospective() -> None:
    previous = packet("prev", "2024-08-14", {"111111111": 100})
    current = packet("cur", "2024-11-14", {"111111111": 120}, period="2024-09-30")
    prior_fix = packet("prior-fix", "2024-12-01", {"111111111": 90}, kind="restatement")
    value_fix = packet("value-fix", "2024-12-04", {"111111111": 120}, period="2024-09-30", kind="restatement")
    value_fix[1].loc[0, "value_usd"] = 2000
    stale = packet("stale", "2024-12-08", {"111111111": 50}, period="2024-03-31", kind="restatement")
    m, h = combine(previous, current, prior_fix, value_fix, stale)
    keys = {"111111111": "A"}
    first = vintage_cohort_snapshot(m, h, "2024-11-15", GRID, security_tickers=keys, settle_days=0).iloc[0]
    later = vintage_cohort_snapshot(m, h, "2024-12-02", GRID, security_tickers=keys, settle_days=0).iloc[0]
    assert first.shares == later.shares == 120
    assert first.shares_chg == pytest.approx(0.2) and later.shares_chg == pytest.approx(1 / 3)
    publications = vintage_cohort_publications(m, h, GRID, until="2024-12-10", security_tickers=keys, settle_days=0)
    assert pd.Timestamp("2024-12-02") in set(publications.as_of)
    assert pd.Timestamp("2024-12-05") in set(publications.as_of)
    assert pd.Timestamp("2024-12-09") not in set(publications.as_of)
    updated = publications.loc[publications.as_of.eq(pd.Timestamp("2024-12-05"))].iloc[0]
    assert updated.shares == 120 and updated.value_usd == 2000
    old_m, old_h = combine(previous, current)
    prefix = vintage_cohort_publications(old_m, old_h, GRID, until="2024-11-15", security_tickers=keys, settle_days=0)
    pd.testing.assert_frame_equal(prefix, publications.loc[publications.as_of.le(pd.Timestamp("2024-11-15"))].reset_index(drop=True))
    unknown = h.copy()
    unknown.loc[unknown.accession_number.eq("cur"), "value_usd"] = float("nan")
    out = vintage_cohort_snapshot(m, unknown, "2024-11-15", GRID, security_tickers=keys, settle_days=0).iloc[0]
    assert out.shares_chg == pytest.approx(0.2) and pd.isna(out.value_usd) and out.value_support == 0
    print("SANITY: baseline amendment, value correction, and unknown-dollar support remain prospective; stale quarter emits nothing.")


@pytest.mark.parametrize("factor", [10.0, 2.0, 1 / 3, 1.0])
def test_cohort_genuine_split_interval_and_future_invariance(factor: float) -> None:
    previous = packet("prev", "2024-08-14", {"111111111": 100})
    current = packet("cur", "2024-11-14", {"111111111": 100 * factor}, period="2024-09-30")
    m, h = combine(previous, current)
    keys = {"111111111": "A"}
    events = pd.DataFrame({"ticker": ["A", "A"], "date": pd.to_datetime(["2024-06-30", "2024-09-30"]), "ratio": [4.0, factor]})
    out = vintage_cohort_snapshot(m, h, "2024-11-15", GRID, security_tickers=keys, splits=events, settle_days=0)
    assert out.iloc[0].shares_chg == pytest.approx(0)
    assert out.iloc[0].matched_prior_shares == pytest.approx(100 * factor)
    future = pd.concat([events, pd.DataFrame({"ticker": ["A"], "date": [pd.Timestamp("2025-02-10")], "ratio": [100.0]})])
    pd.testing.assert_frame_equal(out, vintage_cohort_snapshot(m, h, "2024-11-15", GRID, security_tickers=keys, splits=future, settle_days=0))
    with pytest.raises(ValueError, match="stable CUSIP"):
        vintage_cohort_snapshot(m, h, "2024-11-15", GRID, security_tickers={"111111111": "A", "222222222": "A"}, settle_days=0)
    print(
        f"SANITY: genuine factor {factor:g}; interval excludes prior report date/includes current; future split cannot change history; unsupported CUSIP merger rejected."
    )


def test_cohort_identical_refiling_resets_observation_age() -> None:
    previous = packet("prev", "2024-08-14", {"111111111": 100})
    current = packet("cur", "2024-11-14", {"111111111": 120}, period="2024-09-30")
    identical = packet("refiled", "2024-11-26", {"111111111": 120}, period="2024-09-30")
    m, h = combine(previous, current, identical)
    out = vintage_cohort_publications(m, h, GRID, until="2024-11-29", security_tickers={"111111111": "A"}, settle_days=0)
    revised = out.loc[out.as_of.eq(pd.Timestamp("2024-11-27"))]
    assert len(revised) == 1, "identical re-filing changes the public observation clock"
    direct = vintage_cohort_snapshot(m, h, "2024-11-29", GRID, security_tickers={"111111111": "A"}, settle_days=0).iloc[0]
    carried_age = revised.iloc[0].publication_age_days + (pd.Timestamp("2024-11-29") - revised.iloc[0].as_of).days
    assert direct.publication_age_days == carried_age == 2
    assert revised.iloc[0].shares_chg == pytest.approx(0.2)
    assert revised.iloc[0].publication_reference_at == pd.Timestamp("2024-11-27")
    print("SANITY: identical current re-filing emits its reset clock; daily carried age equals direct snapshot; shares/flow unchanged.")


def test_ordinary_builders_use_versions_and_keep_prior_corrections_prospective(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    original = packet("old", "2024-08-14", {"111111111": 100, "222222222": 100})
    new = packet("new", "2024-11-14", {"111111111": 120, "222222222": 100}, period="2024-09-30")
    revised = packet("amend", "2024-12-01", {"111111111": 90, "222222222": 100}, kind="restatement")
    for m, h in (original, new, revised):
        sqlite_store.save_13f_filing(m, h)
        sqlite_store.save_13f_filing(m, h.assign(ticker="_empty"), table=Tables.sec13f_hr)
    sqlite_store.save(Tables.cusip_ticker_map, pd.DataFrame({"cusip": ["111111111", "222222222"], "ticker": ["AAA", "BBB"]}))
    step = object.__new__(StepCubeSecThirteen)
    step._store, step._log = sqlite_store, logging.getLogger(__name__)
    step._cfg, step._availability = {"institutionals": {"value_basis": {"min_securities": 1}, "f13_settle_trading_days": 0}}, None
    step._context = SimpleNamespace(store=sqlite_store, log=step._log, config_dir="configs")

    def arithmetic(frames: Any, holdings: pd.DataFrame, **kw: Any) -> pd.DataFrame:
        return build_institutional_feature_panel(frames, holdings, min_prior_holders=0, **kw)

    monkeypatch.setattr(step_module, "build_institutional_feature_panel", arithmetic)
    frames = PriceFrames(
        GRID,
        ("AAA", "BBB"),
        {},
        close_split=pd.DataFrame(10.0, index=GRID, columns=["AAA", "BBB"]),
        level_factor=pd.DataFrame(1.0, index=GRID, columns=["AAA", "BBB"]),
    )
    broad, elite = normal_panels(step, frames)
    before, after = (pd.Timestamp(d) for d in ("2024-11-21", "2024-12-02"))
    assert broad.loc[(before, "AAA"), "f_ic_inst_shares_chg"] == pytest.approx(0.2)
    assert broad.loc[(after, "AAA"), "f_ic_inst_shares_chg"] == pytest.approx(1 / 3)
    assert elite.loc[(before, "AAA"), "f_ic_super_shares_chg"] == pytest.approx(0.2)
    assert elite.loc[(after, "AAA"), "f_ic_super_shares_chg"] == pytest.approx(1 / 3)
    assert elite.loc[(before, "AAA"), "f_ic_super_conviction_weight"] == elite.loc[(after, "AAA"), "f_ic_super_conviction_weight"]
    assert set(completed(sqlite_store).accession_number) == {"old", "new", "amend"}
    print(
        "SANITY: both ordinary builders read accessions; prior amendment changes .2 to .3333 prospectively while current conviction stays unchanged."
    )


def ordinary_step(
    sqlite_store: DataStore,
    monkeypatch: pytest.MonkeyPatch,
    packets: tuple[tuple[pd.DataFrame, pd.DataFrame], ...],
) -> tuple[StepCubeSecThirteen, PriceFrames]:
    for manifest, holdings in packets:
        sqlite_store.save_13f_filing(manifest, holdings)
        sqlite_store.save_13f_filing(manifest, holdings.assign(ticker="_empty"), table=Tables.sec13f_hr)
    sqlite_store.save(Tables.cusip_ticker_map, pd.DataFrame({"cusip": ["111111111", "222222222", "333333333"], "ticker": ["AAA", "BBB", "CCC"]}))
    step = object.__new__(StepCubeSecThirteen)
    step._store, step._log = sqlite_store, logging.getLogger(__name__)
    step._cfg, step._availability = {"institutionals": {"value_basis": {"min_securities": 1}, "f13_settle_trading_days": 0}}, None
    step._context = SimpleNamespace(store=sqlite_store, log=step._log, config_dir="configs")

    def arithmetic(frames: Any, holdings: pd.DataFrame, **kw: Any) -> pd.DataFrame:
        return build_institutional_feature_panel(frames, holdings, min_prior_holders=0, **kw)

    monkeypatch.setattr(step_module, "build_institutional_feature_panel", arithmetic)
    frames = PriceFrames(
        GRID,
        ("AAA", "BBB", "CCC"),
        {},
        close_split=pd.DataFrame(10.0, index=GRID, columns=["AAA", "BBB", "CCC"]),
        level_factor=pd.DataFrame(1.0, index=GRID, columns=["AAA", "BBB", "CCC"]),
    )
    return step, frames


def normal_panels(step: StepCubeSecThirteen, frames: PriceFrames) -> tuple[pd.DataFrame, pd.DataFrame]:
    cusip_map = step._store.load(Tables.cusip_ticker_map)
    history = pd.DataFrame({"snapshot_date": [pd.Timestamp("2023-09-01")], "cik": ["0000000001"]})
    return (
        step._institutional_panel(frames, None, pd.DataFrame(), cusip_map, frames.close_split).set_index(["date", "ticker"]),
        step._superinvestor_panel(frames, pd.DataFrame(), cusip_map, {"0000000001"}, history, frames.close_split).set_index(["date", "ticker"]),
    )


def test_normal_builders_tolerated_count_discrepancy_keeps_exits(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    prior = packet("prior-full", "2024-08-14", {"111111111": 100, "222222222": 100, **{f"{number:09d}": 100 for number in range(3, 8)}})
    current = packet("current-six", "2024-11-14", {"222222222": 110, **{f"{number:09d}": 110 for number in range(3, 8)}}, period="2024-09-30")
    current[0]["declared_row_count"] = 7
    current[0]["declared_value_total"] = 7700.0
    step, frames = ordinary_step(sqlite_store, monkeypatch, (prior, current))
    broad, elite = normal_panels(step, frames)
    day = pd.Timestamp("2024-11-21")
    for panel, family in ((broad, "inst"), (elite, "super")):
        assert panel.loc[(day, "AAA"), f"f_ic_{family}_shares_chg"] == -1
        assert panel.loc[(day, "BBB"), f"f_ic_{family}_shares_chg"] == pytest.approx(0.1)
    assert elite.loc[(day, "AAA"), "f_ic_super_full_exits"] > 0
    print("SANITY: actual broad and elite builders trust the accepted6/7-row book; absent stock exits and the retained stock grows10%.")


@pytest.mark.parametrize("mode", ["off", "top_k"])
@pytest.mark.parametrize("unknown_value", [False, True])
def test_normal_elite_comparisons_keep_unknown_weights_and_zero_pools_safe(
    sqlite_store: DataStore,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    unknown_value: bool,
) -> None:
    sources = []
    for period, filed in (
        ("2023-09-30", "2023-11-14"),
        ("2023-12-31", "2024-02-14"),
        ("2024-03-31", "2024-05-14"),
        ("2024-06-30", "2024-08-14"),
        ("2024-09-30", "2024-11-14"),
    ):
        for cik in ("1", "2"):
            manifest, holdings = packet(f"{cik}-{period}", filed, {"111111111": 100, "222222222": 100}, period=period)
            manifest["cik"] = holdings["cik"] = cik.zfill(10)
            if unknown_value and cik == "2" and period == "2024-09-30":
                manifest["source_value_total"] = manifest["declared_value_total"] = 2000000.0
                holdings["value_usd"] = holdings["raw_value_usd"] = 1000000.0
            sources.append((manifest, holdings))
    step, _ = ordinary_step(sqlite_store, monkeypatch, tuple(sources))
    step._cfg["institutionals"]["value_basis"] = {}
    step._cfg["institutionals"]["superinvestor"] = {"selection": {"mode": mode, "k": 20, "min_quarters": 4, "min_positions": 0}}
    grid = pd.bdate_range("2023-09-01", "2024-12-02")
    close = pd.DataFrame(10.0, index=grid, columns=["AAA"])
    frames = PriceFrames(grid, ("AAA",), {}, close_split=close, level_factor=pd.DataFrame(1.0, index=grid, columns=["AAA"]))
    cusip_map = sqlite_store.load(Tables.cusip_ticker_map)
    history = pd.DataFrame({"snapshot_date": [pd.Timestamp("2023-09-01")] * 2, "cik": ["0000000001", "0000000002"]})
    elite = step._superinvestor_panel(frames, pd.DataFrame(), cusip_map, set(history.cik), history, close).set_index(["date", "ticker"])
    row = elite.loc[(pd.Timestamp("2024-11-15"), "AAA")]
    changes = row.reindex(["f_ic_super_shares_chg", "f_ic_super_conviction_chg", "f_ic_super_conviction_weight_yoy"])
    if mode == "top_k":
        assert changes.isna().all()  # The first selected quarter has no previous selected pool.
    else:
        assert changes.f_ic_super_shares_chg == 0
    for name in ("f_ic_super_conviction_chg", "f_ic_super_conviction_weight_yoy"):
        if mode == "top_k":
            continue
        if unknown_value:
            assert pd.isna(changes[name])
        else:
            assert changes[name] == 0
    print(
        f"SANITY: actual elite mode={mode}, unknown dollars={unknown_value}; unsupported weights/pools stay unknown; flat-selection shares stay unchanged."
    )


@pytest.mark.parametrize("restricted", [False, True])
def test_normal_removed_position_clears_carry_without_inventing_exit(
    sqlite_store: DataStore,
    monkeypatch: pytest.MonkeyPatch,
    restricted: bool,
) -> None:
    prior = packet("prior", "2024-08-14", {"222222222": 100})
    original = packet("original", "2024-11-14", {"111111111": 120, "222222222": 100}, period="2024-09-30")
    revised = packet("revised", "2024-12-01", {"222222222": 100}, kind="restatement", period="2024-09-30", restricted=restricted)
    step, frames = ordinary_step(sqlite_store, monkeypatch, (prior, original))
    before = normal_panels(step, frames)
    sqlite_store.save_13f_filing(*revised)
    sqlite_store.save_13f_filing(revised[0], revised[1].assign(ticker="_empty"), table=Tables.sec13f_hr)
    after = normal_panels(step, frames)
    for old, new, family in zip(before, after, ("inst", "super"), strict=True):
        cutoff = pd.Timestamp("2024-12-01")
        pd.testing.assert_frame_equal(old.loc[old.index.get_level_values("date") <= cutoff], new.loc[new.index.get_level_values("date") <= cutoff])
        assert old.loc[(pd.Timestamp("2024-11-21"), "AAA"), f"f_ic_{family}_holders"] > 0
        value = new.loc[(pd.Timestamp("2024-12-02"), "AAA"), f"f_ic_{family}_holders"]
        assert pd.isna(value) if restricted else value == 0
        if restricted:
            assert pd.isna(new.loc[(pd.Timestamp("2024-12-02"), "AAA"), f"f_ic_{family}_shares_chg"])
    print(
        f"SANITY: actual builders clear removed name; restricted={restricted}; old prefix identical; absent original-quarter position never carried."
    )


@pytest.mark.parametrize("restricted", [False, True])
def test_normal_empty_book_has_exit_only_with_complete_public_baseline(
    sqlite_store: DataStore,
    monkeypatch: pytest.MonkeyPatch,
    restricted: bool,
) -> None:
    prior = packet("prior", "2024-08-14", {"111111111": 100})
    current = packet("empty", "2024-11-14", {}, period="2024-09-30", restricted=restricted)
    step, frames = ordinary_step(sqlite_store, monkeypatch, (prior, current))
    broad, elite = normal_panels(step, frames)
    day = pd.Timestamp("2024-11-21")
    for panel, family in ((broad, "inst"), (elite, "super")):
        change = panel.reindex(columns=[f"f_ic_{family}_shares_chg"]).loc[(day, "AAA"), f"f_ic_{family}_shares_chg"]
        assert pd.isna(change) if restricted else change == -1
    event = elite.reindex(columns=["f_ic_super_full_exits"]).loc[(day, "AAA"), "f_ic_super_full_exits"]
    assert pd.isna(event) if restricted else event > 0
    print(
        f"SANITY: actual empty-book comparison, restricted={restricted}; true complete exit supported; confidentiality cannot create full-exit event."
    )


def test_normal_copied_amendment_unknown_leg_cannot_create_exit(
    sqlite_store: DataStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = packet("prior", "2024-05-14", {"111111111": 90, "222222222": 190, "333333333": 290}, period="2024-03-31")
    original = packet("original", "2024-08-14", {"111111111": 100, "222222222": 200, "333333333": 300})
    source = pd.DataFrame(
        [
            {"CUSIP": cusip, "TITLEOFCLASS": "COM", "SSHPRNAMT": shares, "SSHPRNAMTTYPE": "SH", "PUTCALL": option, "VALUE": shares * 10}
            for cusip, shares, option in (("111111111", 100, ""), ("222222222", 200, ""), ("333333333", 20, "Call"))
        ]
    )
    amendment = filing_vintage(
        source,
        cik="1",
        accession_number="patch",
        period="2024-06-30",
        filing_date="2024-10-17",
        amendment_type="additions",
        source="known_truth",
        declared_rows=3,
        declared_value=3200,
    )
    step, frames = ordinary_step(sqlite_store, monkeypatch, (prior, original, amendment))
    broad, elite = normal_panels(step, frames)
    day = pd.Timestamp("2024-10-18")
    for panel, family in ((broad, "inst"), (elite, "super")):
        assert pd.isna(panel.loc[(day, "CCC"), f"f_ic_{family}_shares_chg"])
        assert pd.isna(panel.loc[(day, "CCC"), f"f_ic_{family}_holders"])
    event = elite.reindex(columns=["f_ic_super_full_exits"]).loc[(day, "CCC"), "f_ic_super_full_exits"]
    assert pd.isna(event)
    assert broad.loc[(day, "AAA"), "f_ic_inst_shares_chg"] == pytest.approx(100 / 90 - 1)
    print(
        "SANITY: inferred common-leg omission remains unknown in both normal consumers and elite exit events; supported sibling quantities still compare."
    )


def test_normal_elite_expiry_does_not_wait_for_another_filing(
    sqlite_store: DataStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step, frames = ordinary_step(sqlite_store, monkeypatch, (packet("only", "2024-08-14", {"111111111": 100}),))
    _, elite = normal_panels(step, frames)
    assert elite.loc[(pd.Timestamp("2025-08-14"), "AAA"), "f_ic_super_holders"] == 1
    assert pd.isna(elite.loc[(pd.Timestamp("2025-08-18"), "AAA"), "f_ic_super_holders"])
    print("SANITY: existing four-quarter elite expiry clears the daily level even when no future source filing arrives.")


def test_same_day_versions_follow_acceptance_time_before_accession_order() -> None:
    original = packet("original", "2024-08-14", {"111111111": 100})
    early = packet("zz-earlier", "2024-10-17", {"111111111": 120}, kind="restatement")
    late = packet("aa-later", "2024-10-17", {"111111111": 140}, kind="restatement")
    early[0]["accepted_at"] = pd.Timestamp("2024-10-17T13:00:00Z")
    late[0]["accepted_at"] = pd.Timestamp("2024-10-17T14:00:00Z")
    manifests, holdings = combine(original, early, late)
    before, _ = public_books(manifests, holdings, "2024-10-17", GRID)
    after, _ = public_books(manifests, holdings, "2024-10-18", GRID)
    assert before.iloc[0].shares == 100
    assert after.iloc[0].shares == 140
    print("SANITY: same-day amendments follow verified acceptance time despite inverse accession order, without changing the prior public day.")


def test_normal_broad_retains_option_only_names(
    sqlite_store: DataStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = pd.DataFrame(
        [
            {
                "CUSIP": "111111111",
                "NAMEOFISSUER": "AAA",
                "TITLEOFCLASS": "COM",
                "SSHPRNAMT": 100,
                "SSHPRNAMTTYPE": "SH",
                "PUTCALL": "Call",
                "VALUE": 1000,
            }
        ]
    )
    option = filing_vintage(
        source,
        cik="1",
        accession_number="option-only",
        period="2024-06-30",
        filing_date="2024-08-14",
        amendment_type="original",
        source="known_truth",
        declared_rows=1,
        declared_value=1000,
    )
    step, frames = ordinary_step(sqlite_store, monkeypatch, (option,))
    cusip_map = sqlite_store.load(Tables.cusip_ticker_map)
    broad = step._institutional_panel(frames, None, pd.DataFrame(), cusip_map, frames.close_split).set_index(["date", "ticker"])
    row = broad.loc[(pd.Timestamp("2024-08-21"), "AAA")]
    assert row.f_ic_inst_net_options_ratio == 1
    assert row.f_ic_inst_holders == 0
    print("SANITY: actual broad builder retains an observable call-only name, with net option ratio1 and no common-stock holder.")


@pytest.mark.parametrize("clocked", ["original", "amendment"])
def test_same_day_missing_acceptance_clock_preserves_amendment_fallback(clocked: str) -> None:
    original = packet("zz-original", "2024-08-14", {"111111111": 100})
    amendment = packet("aa-amendment", "2024-08-14", {"111111111": 140}, kind="restatement")
    version = original if clocked == "original" else amendment
    version[0]["accepted_at"] = pd.Timestamp("2024-08-14T13:00:00Z")
    manifests, holdings = combine(original, amendment)
    resolved, _ = public_books(manifests, holdings, "2024-08-21", GRID)
    assert resolved.iloc[0].shares == 140
    print(f"SANITY: same-day incomplete acceptance clocks use amendment metadata; timestamp only on {clocked}; latest shares140.")


def test_expiry_sessions_are_unique_when_publications_share_a_future_session() -> None:
    packets = (
        packet("original", "2024-08-14", {"111111111": 100}),
        packet("amendment", "2024-08-15", {"111111111": 120}, kind="restatement"),
    )
    manifest, holdings = combine(*packets)
    days = pd.DatetimeIndex(
        [
            day
            for day, _, _ in iter_public_books(
                manifest,
                holdings,
                GRID,
                settle_days=0,
                history_quarters=None,
                expiry_quarters=4,
            )
        ]
    )
    assert days.is_unique
    assert list(days).count(pd.Timestamp("2025-08-18")) == 1
    print("SANITY: distinct publication dates that expire onto one trading session emit one snapshot.")


def test_normal_elite_handles_colliding_expiry_sessions(
    sqlite_store: DataStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packets = (
        packet("original", "2024-08-14", {"111111111": 100}),
        packet("amendment", "2024-08-15", {"111111111": 120}, kind="restatement"),
    )
    step, frames = ordinary_step(sqlite_store, monkeypatch, packets)
    _, elite = normal_panels(step, frames)
    assert elite.index.is_unique
    assert elite.loc[(pd.Timestamp("2025-08-15"), "AAA"), "f_ic_super_holders"] == 1
    assert pd.isna(elite.loc[(pd.Timestamp("2025-08-18"), "AAA"), "f_ic_super_holders"])
    print("SANITY: actual elite builder survives coincident expiry dates and clears stale holdings on time.")
