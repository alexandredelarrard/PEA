"""Source-level freshness contract for the cube status gate."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from src.context import Context
from src.data_aggregate.utils.common.part_status import (
    _insider_source_status,
    cube_part_edge_report,
    part_status_report,
)
from src.data_aggregate.utils.common.parts import CUBE_PARTS, TERMINAL_TABLES
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from tests.data_extract.fake_context import extract_config

LOG = logging.getLogger(__name__)
SOURCE = "cube_part_institutionals:insider_transactions"


def _context(store: Any) -> Any:
    config = extract_config(source_freshness={"insider_max_lag_days": 4})
    config.data_extract = {"redundant_ticks": []}
    config.build_cube = {"assembly": {"optional_parts": [Tables.cube_part_text.name, Tables.cube_part_institutionals.name]}}
    return SimpleNamespace(store=store, config=config, log=LOG)


def _seed_prices(sqlite_store: DataStore) -> None:
    sqlite_store.save(Tables.sp500_tickers, pd.DataFrame({"ticker": ["AAA"]}))
    sqlite_store.replace(
        Tables.cube_part_prices,
        pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-09-03"), pd.Timestamp("2026-09-04"), pd.Timestamp("2026-09-04")],
                "ticker": ["AAA", "AAA", "BBB"],
            }
        ),
    )


def _seed_insider(sqlite_store: DataStore, filing_date: str, inventory_date: str | None) -> None:
    """One stored insider row with independently dated inventory proof."""
    sqlite_store.save(
        Tables.insider_transactions,
        pd.DataFrame(
            {
                "accession_number": ["a1"],
                "security_type": "nonderiv",
                "row_sequence": 1,
                "ticker": ["AAA"],
                "filing_date": [pd.Timestamp(filing_date)],
                "fetched_at": [pd.Timestamp("2026-09-05")],
                "inventory_checked_through": [pd.Timestamp(inventory_date) if inventory_date else pd.NaT],
                "source": ["edgar"],
            }
        ),
    )


def test_inventory_proof_through_the_part_edge_makes_the_source_current(sqlite_store: DataStore) -> None:
    _seed_prices(sqlite_store)
    _seed_insider(sqlite_store, "2026-09-01", "2026-09-04")

    status = _insider_source_status(cast(Context, _context(sqlite_store)), "2026-09-04", tolerance_days=4)

    assert status == {
        "complete_through": "2026-09-04",
        "part_max_date": "2026-09-04",
        "lag_days": 0,
        "tolerance_days": 4,
        "ok": True,
    }
    print("SANITY: explicit fetched inventory proof through the last price session makes the source current, independently of the filing date.")


def test_a_recent_filing_without_inventory_proof_cannot_make_the_source_current(sqlite_store: DataStore) -> None:
    _seed_prices(sqlite_store)
    _seed_insider(sqlite_store, "2026-09-04", None)

    status = _insider_source_status(cast(Context, _context(sqlite_store)), "2026-09-04", tolerance_days=4)

    assert status["complete_through"] is None and status["lag_days"] is None and status["ok"] is False
    print("SANITY: a filing on the last price session cannot certify an unchecked issuer inventory.")


def test_an_empty_insider_table_cannot_make_the_source_current(sqlite_store):
    _seed_prices(sqlite_store)

    status = _insider_source_status(cast(Context, _context(sqlite_store)), "2026-09-04", tolerance_days=4)

    assert status["complete_through"] is None and status["lag_days"] is None and status["ok"] is False
    print("SANITY: with no stored insider row the frontier is unknown and the gate is red.")


def test_a_stale_insider_table_fails_the_lag_gate(sqlite_store: DataStore) -> None:
    _seed_prices(sqlite_store)
    _seed_insider(sqlite_store, "2026-09-04", "2026-08-25")

    status = _insider_source_status(cast(Context, _context(sqlite_store)), "2026-09-04", tolerance_days=4)

    assert status["complete_through"] == "2026-08-25" and status["lag_days"] == 10 and status["ok"] is False
    print("SANITY: a recent filing cannot advance inventory proof ten days behind the part edge; the 4-day tolerance still fails.")


class _ReportStore:
    def __init__(
        self,
        *,
        max_dates: dict[str, pd.Timestamp] | None = None,
        missing: tuple[str, ...] = (),
        empty: tuple[str, ...] = (),
        source_date: str | None = "2026-09-04",
        inventory_date: str | None = "2026-09-04",
    ) -> None:
        self.max_dates = max_dates or {}
        self.missing = set(missing)
        self.empty = set(empty)
        self.source_date = source_date
        self.inventory_date = inventory_date

    def columns(self, table: Any) -> list[str]:
        if table == Tables.insider_transactions:
            return ["ticker", "inventory_checked_through", "fetched_at"]
        return ["date", "ticker"]

    def summarize(self, table: Any, group_by: str, aggregates: dict[str, tuple[str, str | None]], *, where: dict[str, list[str]]) -> pd.DataFrame:
        boundary = pd.Timestamp(self.inventory_date) if self.inventory_date else pd.NaT
        tickers = where["ticker"] if self.exists(table) and table.name not in self.empty else []
        return pd.DataFrame(
            {
                "ticker": tickers,
                "first": boundary,
                "last": boundary,
                "checked": int(pd.notna(boundary)),
                "fetched": 1,
                "rows": 1,
            }
        )

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
    print(
        "  a one-day lag fails, a one-day lead fails, and an absent optional part skips; the shared "
        "four-day source-freshness tolerance is not applied to cube parts. Validated."
    )


def test_full_status_keeps_the_dag_contract_and_adds_source_detail():
    context = _context(_ReportStore(inventory_date="2026-08-20"))

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
        "SANITY: the DAG's ok/behind/parts/max_date contract is unchanged, and inventory proof 15 days behind "
        "the part edge names the insider source in `behind` even with a current filing."
    )


def test_status_fails_closed_for_missing_cube_and_part_edge_mismatch():
    lagging = Tables.cube_part_fundamentals.name
    context = _context(_ReportStore(max_dates={lagging: pd.Timestamp("2026-09-03")}, missing=(Tables.cube.name,)))

    report = part_status_report(cast(Context, context))

    assert not report["ok"]
    assert lagging in report["behind"]
    assert Tables.cube.name in report["behind"]
    assert SOURCE not in report["behind"], "inventory proof through the last session keeps the insider source current"
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
    assert report["sources"][Tables.cube_part_institutionals.name]["insider_transactions"]["status"] == "skipped"
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
        assert report["sources"][Tables.prices.name]["refresh_command"] == "price-history (seed-universe if universe is absent)"
    print(
        "SANITY: stored preceding-session date is enough regardless of wall clock; unknown source reference fails closed and names seed-universe recovery."
    )
