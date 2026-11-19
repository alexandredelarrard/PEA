"""Actual price support and exact cube-part alignment for the status gate."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from src.context import Context
from src.data_aggregate.utils.common.part_status import (
    cube_part_edge_report,
    part_status_report,
)
from src.data_aggregate.utils.common.parts import CUBE_PARTS, TERMINAL_TABLES
from src.data_store.schema import Tables
from src.utils.window import WindowRun
from tests.support.config import fixture_config
from tests.support.universe import universe_of

LOG = logging.getLogger(__name__)
SOURCE = "cube_part_institutionals:insider_transactions"


def _context(store: Any) -> Any:
    config = fixture_config()
    config.data_aggregate = {"assembly": {"optional_parts": [Tables.cube_part_text.name, Tables.cube_part_institutionals.name]}}
    return SimpleNamespace(
        store=store, config=config, log=LOG, universe=universe_of(["AAA"]), run=WindowRun("incremental", "part alignment", pd.Timestamp("2026-09-04"))
    )


class _ReportStore:
    def __init__(
        self,
        *,
        max_dates: dict[str, pd.Timestamp] | None = None,
        missing: tuple[str, ...] = (),
        empty: tuple[str, ...] = (),
        source_date: str | None = "2026-09-04",
        insider_date: str = "2026-09-04",
    ) -> None:
        self.max_dates = max_dates or {}
        self.missing = set(missing)
        self.empty = set(empty)
        self.source_date = source_date
        self.insider_date = insider_date

    @staticmethod
    def columns(table: Any) -> list[str]:
        return ["date", "ticker"]

    def load(self, table, **kwargs):
        if not self.exists(table) or getattr(table, "name", table) in self.empty:
            return None
        return pd.DataFrame({"ticker": ["AAA", "BBB"]})

    def max_date_by(self, table, key_col):
        if not self.exists(table) or self.source_date is None:
            return {}
        return {"AAA": pd.Timestamp(self.source_date), "EXCLUDED": pd.Timestamp("2030-01-01")}

    def exists(self, table) -> bool:
        return getattr(table, "name", table) not in self.missing

    def max_date(self, table, column=None):
        name = getattr(table, "name", table)
        if name in self.empty:
            return None
        if name == Tables.insider_transactions.name:
            return pd.Timestamp(self.insider_date)
        return self.max_dates.get(name, pd.Timestamp("2026-09-04"))

    def row_count(self, table) -> int:
        return 0 if getattr(table, "name", table) in self.empty else 1

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

    edge: Any = cube_part_edge_report(cast(Any, _context(store)))

    assert edge["price_max_date"] == "2026-09-04"
    assert {k: edge["parts"][one_day_behind][k] for k in ("status", "max_date", "delta_days")} == {
        "status": "behind",
        "max_date": "2026-09-03",
        "delta_days": -1,
    }
    assert edge["parts"][one_day_ahead]["status"] == "ahead"
    assert edge["parts"][one_day_ahead]["delta_days"] == 1
    assert edge["parts"][missing]["status"] == "skipped"
    assert set(edge["misaligned"]) == {one_day_behind, one_day_ahead}
    print("\n=== SANITY CHECK: exact cube-part price-edge alignment ===")
    print("  a one-day lag fails, a one-day lead fails, and an absent optional part skips. Validated.")


def test_full_status_keeps_the_dag_contract_without_source_age_gates():
    context = _context(_ReportStore(insider_date="2026-08-20"))

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
    assert report["ok"] and SOURCE not in report["behind"]
    assert Tables.cube_part_institutionals.name not in report["sources"]
    print(
        "SANITY: the DAG's ok/behind/parts/max_date contract is retained; a 15-day-old insider frontier does not impose an undeclared source-age gate."
    )


def test_status_fails_closed_for_missing_cube_and_part_edge_mismatch():
    lagging = Tables.cube_part_fundamentals.name
    context = _context(_ReportStore(max_dates={lagging: pd.Timestamp("2026-09-03")}, missing=(Tables.cube.name,)))

    report = part_status_report(cast(Context, context))

    assert not report["ok"]
    assert lagging in report["behind"]
    assert Tables.cube.name in report["behind"]
    assert SOURCE not in report["behind"], "an insider frontier on the reference date keeps the insider source current"
    assert set(report["misaligned"]) == {lagging, Tables.cube.name}
    print("\n=== SANITY CHECK: cube status fails closed ===")
    print("  the missing final cube and the one-day fundamentals lag are both named in `behind`; the current insider source is not. Validated.")


@pytest.mark.parametrize("empty", [False, True])
def test_optional_absence_skips_dependent_source_gate(empty: bool) -> None:
    optional = (Tables.cube_part_text.name, Tables.cube_part_institutionals.name)
    store = _ReportStore(empty=optional) if empty else _ReportStore(missing=optional)
    report = part_status_report(_context(store))
    assert report["ok"] and report["behind"] == []
    assert all(report["parts"][name]["edge_status"] == "skipped" for name in optional)
    assert Tables.cube_part_institutionals.name not in report["sources"]
    print("SANITY: absent/empty optional parts and their inapplicable insider check cannot make readiness red.")


def test_source_price_frontier_exposes_lagging_price_part() -> None:
    report = part_status_report(_context(_ReportStore(source_date="2026-09-08")))
    assert not report["ok"]
    assert report["reference_price_date"] == "2026-09-08"
    assert report["price_max_date"] == "2026-09-04"
    assert Tables.cube_part_prices.name in report["behind"]
    info = report["parts"][Tables.cube_part_prices.name]
    assert info["refresh_command"] == "build-prices" and info["expected_date"] == "2026-09-08"
    print("SANITY: source prices expose a lagging price part and name build-prices; excluded ticker cannot move reference.")


@pytest.mark.parametrize("source_date", [None, "2026-09-04"])
def test_stored_session_reference_and_missing_source(source_date: str | None) -> None:
    report = part_status_report(_context(_ReportStore(source_date=source_date)))
    assert report["ok"] == (source_date is not None)
    assert report["reference_price_date"] == source_date
    if source_date is None:
        assert Tables.prices.name in report["behind"]
        assert report["sources"][Tables.prices.name]["refresh_command"] == "price-history (universe if universe_tickers is absent)"
    print(
        "SANITY: stored preceding-session date is enough regardless of wall clock; unknown source reference fails closed and names the universe recovery."
    )
