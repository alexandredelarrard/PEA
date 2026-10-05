"""Source-level freshness contract for the cube status gate."""

from __future__ import annotations

import logging
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
from src.data_store.schema import Tables
from tests.data_extract.fake_context import extract_config

LOG = logging.getLogger(__name__)
SOURCE = "cube_part_institutionals:insider_transactions"


def _context(store: Any) -> Any:
    return SimpleNamespace(store=store, config=extract_config(source_freshness={"insider_max_lag_days": 4}))


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


def _seed_insider(sqlite_store, filing_date: str) -> None:
    """One stored insider row filed on `filing_date`."""
    sqlite_store.save(
        Tables.insider_transactions,
        pd.DataFrame(
            {
                "accession_number": ["a1"],
                "security_type": "nonderiv",
                "row_sequence": 1,
                "ticker": ["AAA"],
                "filing_date": [pd.Timestamp(filing_date)],
                "source": ["edgar"],
            }
        ),
    )


def test_a_filing_inside_the_overlap_makes_the_source_current(sqlite_store):
    _seed_prices(sqlite_store)
    _seed_insider(sqlite_store, "2026-09-01")

    status = _insider_source_status(cast(Context, _context(sqlite_store)), "2026-09-04", tolerance_days=4)

    assert status == {
        "complete_through": "2026-09-04",
        "part_max_date": "2026-09-04",
        "lag_days": 0,
        "tolerance_days": 4,
        "ok": True,
    }
    print(
        "SANITY: the latest insider filing (09-01) lies inside the 7-day overlap of the last price session, so the source is current through 09-04 and the gate passes."
    )


def test_an_empty_insider_table_cannot_make_the_source_current(sqlite_store):
    _seed_prices(sqlite_store)

    status = _insider_source_status(cast(Context, _context(sqlite_store)), "2026-09-04", tolerance_days=4)

    assert status["complete_through"] is None and status["lag_days"] is None and status["ok"] is False
    print("SANITY: with no stored insider row the frontier is unknown and the gate is red.")


def test_a_stale_insider_table_fails_the_lag_gate(sqlite_store):
    _seed_prices(sqlite_store)
    _seed_insider(sqlite_store, "2026-08-25")

    status = _insider_source_status(cast(Context, _context(sqlite_store)), "2026-09-04", tolerance_days=4)

    assert status["complete_through"] == "2026-08-25" and status["lag_days"] == 10 and status["ok"] is False
    print("SANITY: a latest filing ten days before the part edge is outside the overlap, so the frontier stays 08-25 and the 4-day tolerance fails.")


class _ReportStore:
    def __init__(self, *, max_dates=None, missing=()):
        self.max_dates = max_dates or {}
        self.missing = set(missing)

    def exists(self, table) -> bool:
        return getattr(table, "name", table) not in self.missing

    def max_date(self, table, column=None):
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


def test_full_status_keeps_the_dag_contract_and_adds_source_detail():
    context = _context(_ReportStore(max_dates={Tables.insider_transactions.name: pd.Timestamp("2026-08-20")}))

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
    assert insider["complete_through"] == "2026-08-20" and insider["lag_days"] == 15
    print(
        "SANITY: the DAG's ok/behind/parts/max_date contract is unchanged, and an insider table whose latest "
        "filing is 15 days behind the part edge names the insider source in `behind` and fails the status."
    )


def test_status_fails_closed_for_missing_cube_and_part_edge_mismatch():
    lagging = Tables.cube_part_fundamentals.name
    context = _context(_ReportStore(max_dates={lagging: pd.Timestamp("2026-09-03")}, missing=(Tables.cube.name,)))

    report = part_status_report(cast(Context, context))

    assert not report["ok"]
    assert lagging in report["behind"]
    assert Tables.cube.name in report["behind"]
    assert SOURCE not in report["behind"], "an insider filing on the last session keeps the source current"
    assert set(report["misaligned"]) == {lagging, Tables.cube.name}
    print("\n=== SANITY CHECK: cube status fails closed ===")
    print("  the missing final cube and the one-day fundamentals lag are both named in `behind`; the current insider source is not. Validated.")
