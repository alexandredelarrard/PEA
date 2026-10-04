"""Source-level freshness contract for the cube status gate."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd

from src.context import Context
from src.data_aggregate.utils.common.part_status import (
    _insider_source_status,
    cube_part_edge_report,
    part_status_report,
)
from src.data_aggregate.utils.common.parts import CUBE_PARTS, TERMINAL_TABLES
from src.data_extract.utils.common.run_manifest import record_run
from src.data_store.schema import Tables
from tests.data_extract.fake_context import extract_config

LOG = logging.getLogger(__name__)
SOURCE = "cube_part_institutionals:insider_transactions"


def _context(tmp_path: Path, store: Any) -> Any:
    return SimpleNamespace(
        store=store,
        paths={"DATA_STORE": tmp_path},
        config=extract_config(source_freshness={"insider_max_lag_days": 4}),
    )


def _edgar_run(context: Any, run_date: str, tickers: list[str], *, coverage_complete: bool = True) -> None:
    """The manifest entry the EDGAR insider run writes (`coverage_complete` only on a full run)."""
    record_run(context, Tables.insider_transactions, len(tickers), 1, run_date=run_date, coverage_complete=coverage_complete, tickers=tickers)


def _seed_prices(sqlite_store) -> None:
    sqlite_store.replace(
        Tables.cube_part_prices,
        pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-09-03"), pd.Timestamp("2026-09-04"), pd.Timestamp("2026-09-04")],
                "ticker": ["AAA", "AAA", "BBB"],
            }
        ),
    )


def test_a_complete_edgar_run_over_the_universe_makes_the_source_current(tmp_path, sqlite_store):
    _seed_prices(sqlite_store)
    context = _context(tmp_path, sqlite_store)
    _edgar_run(context, "2026-09-04", ["BBB", "AAA"])

    status = _insider_source_status(cast(Context, context), LOG, "2026-09-04", tolerance_days=4)

    assert status == {
        "complete_through": "2026-09-04",
        "part_max_date": "2026-09-04",
        "lag_days": 0,
        "tolerance_days": 4,
        "expected_tickers": 2,
        "ok": True,
    }
    print("SANITY: the EDGAR run covered both price tickers through 2026-09-04, so the insider source-to-part lag is 0 days and the gate passes.")


def test_a_partial_or_unproven_edgar_run_cannot_make_the_source_current(tmp_path, sqlite_store):
    _seed_prices(sqlite_store)
    context = _context(tmp_path, sqlite_store)
    seen: dict[str, object] = {}

    for label, tickers, complete in (("partial", ["AAA"], True), ("wrong member", ["AAA", "CCC"], True), ("no proof", ["AAA", "BBB"], False)):
        _edgar_run(context, "2026-09-04", tickers, coverage_complete=complete)
        status = _insider_source_status(cast(Context, context), LOG, "2026-09-04", tolerance_days=4)
        assert status["complete_through"] is None and status["lag_days"] is None and status["ok"] is False, label
        seen[label] = status["ok"]

    assert seen == {"partial": False, "wrong member": False, "no proof": False}
    print(
        "SANITY: a run over AAA only, a run over a different 2-ticker set, and an entry without the completeness proof all leave the frontier unknown and the gate red."
    )


def test_a_stale_edgar_run_fails_the_lag_gate(tmp_path, sqlite_store):
    _seed_prices(sqlite_store)
    context = _context(tmp_path, sqlite_store)
    _edgar_run(context, "2026-08-25", ["AAA", "BBB"])

    status = _insider_source_status(cast(Context, context), LOG, "2026-09-04", tolerance_days=4)

    assert status["complete_through"] == "2026-08-25" and status["lag_days"] == 10 and status["ok"] is False
    print("SANITY: a complete EDGAR run ten days older than the part edge exceeds the 4-day tolerance and fails the gate.")


class _ReportStore:
    def __init__(self, *, max_dates=None, missing=()):
        self.max_dates = max_dates or {}
        self.missing = set(missing)

    def exists(self, table) -> bool:
        return getattr(table, "name", table) not in self.missing

    def max_date(self, table):
        name = getattr(table, "name", table)
        return self.max_dates.get(name, pd.Timestamp("2026-09-04"))

    @staticmethod
    def row_count(table) -> int:
        return 1

    @staticmethod
    def distinct(table, column, where=None):
        return ["AAA", "BBB"]


def test_cube_part_edges_require_exact_price_alignment():
    one_day_behind = CUBE_PARTS[3].name
    one_day_ahead = CUBE_PARTS[4].name
    missing = CUBE_PARTS[5].name
    store = _ReportStore(
        max_dates={
            one_day_behind: pd.Timestamp("2026-09-03"),
            one_day_ahead: pd.Timestamp("2026-09-05"),
        },
        missing=(missing,),
    )

    edge: Any = cube_part_edge_report(cast(Any, store))

    assert edge["price_max_date"] == "2026-09-04"
    assert edge["parts"][one_day_behind] == {
        "status": "behind",
        "max_date": "2026-09-03",
        "delta_days": -1,
    }
    assert edge["parts"][one_day_ahead]["status"] == "ahead"
    assert edge["parts"][one_day_ahead]["delta_days"] == 1
    assert edge["parts"][missing]["status"] == "missing"
    assert set(edge["misaligned"]) == {one_day_behind, one_day_ahead, missing}
    print("\n=== SANITY CHECK: exact cube-part price-edge alignment ===")
    print(
        "  a one-day lag fails, a one-day lead fails, and an absent part fails; the shared "
        "four-day source-freshness tolerance is not applied to cube parts. Validated."
    )


def test_full_status_keeps_the_dag_contract_and_adds_source_detail(tmp_path):
    context = _context(tmp_path, _ReportStore())
    _edgar_run(context, "2026-09-04", ["AAA"])

    report = part_status_report(cast(Context, context))

    expected_parts = {part.name for part in CUBE_PARTS} | {table.name for table in TERMINAL_TABLES}
    assert {
        "ok",
        "behind",
        "parts",
        "as_of",
        "cube_max_date",
        "max_date",
        "price_max_date",
        "misaligned",
    }.issubset(report)
    assert set(report["parts"]) == expected_parts
    assert not report["ok"]
    assert SOURCE in report["behind"]
    insider = report["sources"]["cube_part_institutionals"]["insider_transactions"]
    assert insider["expected_tickers"] == 2 and insider["complete_through"] is None
    print(
        "SANITY: the DAG's ok/behind/parts/max_date contract is unchanged, and an EDGAR run over 1 of the "
        "2 universe tickers names the insider source in `behind` and fails the status."
    )


def test_status_fails_closed_for_missing_cube_and_part_edge_mismatch(tmp_path):
    lagging = Tables.cube_part_fundamentals.name
    context = _context(
        tmp_path,
        _ReportStore(max_dates={lagging: pd.Timestamp("2026-09-03")}, missing=(Tables.cube.name,)),
    )
    _edgar_run(context, "2026-09-04", ["AAA", "BBB"])

    report = part_status_report(cast(Context, context))

    assert not report["ok"]
    assert lagging in report["behind"]
    assert Tables.cube.name in report["behind"]
    assert SOURCE not in report["behind"], "a complete EDGAR run over the universe keeps the insider source current"
    assert set(report["misaligned"]) == {lagging, Tables.cube.name}
    print("\n=== SANITY CHECK: cube status fails closed ===")
    print("  the missing final cube and the one-day fundamentals lag are both named in `behind`; the current insider source is not. Validated.")
