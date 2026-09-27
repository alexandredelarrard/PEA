"""Source-level freshness contract for the cube status gate."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.data_aggregate.utils.common.part_status import (
    _insider_source_status,
    cube_part_edge_report,
    part_status_report,
)
from src.data_aggregate.utils.common.parts import CUBE_PARTS, TERMINAL_TABLES
from src.data_store.schema import Tables


def _seed_price_edge(sqlite_store) -> None:
    sqlite_store.replace(
        Tables.cube_part_prices,
        pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-09-04"), pd.Timestamp("2026-09-04")],
                "ticker": ["AAA", "BBB"],
            }
        ),
    )
    sqlite_store.replace(
        Tables.insider_transactions,
        pd.DataFrame(
            {
                "accession_number": ["A"],
                "security_type": ["nonderiv"],
                "transaction_sk": ["1"],
                "quarter": ["2026q2"],
                "transaction_date": [pd.Timestamp("2026-06-30")],
                "filing_date": [pd.Timestamp("2026-06-30")],
            }
        ),
    )


def test_all_ticker_live_coverage_makes_the_source_current(sqlite_store):
    _seed_price_edge(sqlite_store)
    sqlite_store.replace(
        Tables.insider_transactions_live_coverage,
        pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "complete_through": [pd.Timestamp("2026-09-04")] * 2,
                "updated_at": [pd.Timestamp("2026-09-04 23:00:00")] * 2,
            }
        ),
    )
    status = _insider_source_status(SimpleNamespace(store=sqlite_store), "2026-09-04", tolerance_days=4)
    assert status["ok"]
    assert status["live_complete_through"] == "2026-09-04"
    assert status["lag_days"] == 0
    print(
        "SANITY: both price-edge tickers were scanned through 2026-09-04, so the insider " "source-to-part lag is 0 days and the status gate passes."
    )


def test_partial_live_run_cannot_hide_the_stale_bulk_frontier(sqlite_store):
    _seed_price_edge(sqlite_store)
    sqlite_store.replace(
        Tables.insider_transactions_live_coverage,
        pd.DataFrame(
            {
                "ticker": ["AAA"],
                "complete_through": [pd.Timestamp("2026-09-04")],
                "updated_at": [pd.Timestamp("2026-09-04 23:00:00")],
            }
        ),
    )
    status = _insider_source_status(SimpleNamespace(store=sqlite_store), "2026-09-04", tolerance_days=4)
    assert not status["ok"]
    assert status["live_complete_through"] is None
    assert status["complete_through"] == "2026-06-30"
    assert status["lag_days"] == 66
    print(
        "SANITY: AAA succeeded but BBB did not; the live frontier stayed unavailable and "
        "the 66-day Q2-bulk lag failed instead of reporting a partial run as current."
    )


def test_unpromoted_bulk_overlap_cannot_make_status_green(sqlite_store):
    _seed_price_edge(sqlite_store)
    sqlite_store.replace(
        Tables.insider_transactions,
        pd.DataFrame(
            {
                "accession_number": ["Q3-A"],
                "security_type": ["nonderiv"],
                "transaction_sk": ["1"],
                "quarter": ["2026q3"],
                "transaction_date": [pd.Timestamp("2026-08-01")],
                "filing_date": [pd.Timestamp("2026-08-03")],
            }
        ),
    )
    sqlite_store.replace(
        Tables.insider_transactions_live,
        pd.DataFrame(
            {
                "accession_number": ["Q3-A"],
                "security_type": ["nonderiv"],
                "source_row_sequence": [1],
                "ticker": ["AAA"],
                "transaction_date": [pd.Timestamp("2026-08-01")],
                "filing_date": [pd.Timestamp("2026-08-03")],
            }
        ),
    )
    context = SimpleNamespace(
        store=sqlite_store,
        config=SimpleNamespace(source_freshness={"insider_bulk_authoritative_through": "2026Q2"}),
    )

    status = _insider_source_status(context, "2026-09-04", tolerance_days=4)

    assert not status["ok"]
    assert status["bulk_reported_quarter"] == "2026q3"
    assert status["bulk_complete_through"] == "2026-06-30"
    print(
        "SANITY: an overlapping but unpromoted Q3 ZIP leaves source status capped at Q2; "
        "a quarterly download cannot bypass the reconciliation gate."
    )


class _ReportStore:
    def __init__(self, *, max_dates=None, missing=(), complete_insider: bool = False):
        self.max_dates = max_dates or {}
        self.missing = set(missing)
        self.complete_insider = complete_insider

    def exists(self, table) -> bool:
        return getattr(table, "name", table) not in self.missing

    def max_date(self, table):
        name = getattr(table, "name", table)
        return self.max_dates.get(name, pd.Timestamp("2026-09-04"))

    @staticmethod
    def row_count(table) -> int:
        return 1

    @staticmethod
    def bounds(table, column):
        return "2010q1", "2026q2"

    @staticmethod
    def distinct(table, column, where=None):
        return ["AAA", "BBB"]

    def load(self, table, **kwargs):
        if table == Tables.insider_transactions:
            return pd.DataFrame(
                {
                    "accession_number": ["A"],
                    "filing_date": [pd.Timestamp("2026-06-30")],
                }
            )
        if table == Tables.insider_transactions_live_coverage:
            return pd.DataFrame(
                {
                    "ticker": ["AAA", "BBB"] if self.complete_insider else ["AAA"],
                    "complete_through": (
                        [pd.Timestamp("2026-09-04"), pd.Timestamp("2026-09-04")] if self.complete_insider else [pd.Timestamp("2026-09-04")]
                    ),
                }
            )
        return None


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
        complete_insider=True,
    )

    edge = cube_part_edge_report(store)

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
    context = SimpleNamespace(
        store=_ReportStore(),
        config=SimpleNamespace(
            source_freshness={
                "insider_max_lag_days": 4,
                "insider_bulk_authoritative_through": "2026Q2",
            }
        ),
    )
    report = part_status_report(context)
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
    assert "cube_part_institutionals:insider_transactions" in report["behind"]
    assert report["sources"]["cube_part_institutionals"]["insider_transactions"]["expected_tickers"] == 2
    print(
        "SANITY: the DAG's existing ok/behind/parts/max_date contract is unchanged, while "
        "the additive sources section exposes the partial insider scan and makes status fail."
    )


def test_status_fails_closed_for_missing_cube_and_part_edge_mismatch():
    lagging = Tables.cube_part_fundamentals.name
    context = SimpleNamespace(
        store=_ReportStore(
            max_dates={lagging: pd.Timestamp("2026-09-03")},
            missing=(Tables.cube.name,),
            complete_insider=True,
        ),
        config=SimpleNamespace(
            source_freshness={
                "insider_max_lag_days": 4,
                "insider_bulk_authoritative_through": "2026Q2",
            }
        ),
    )

    report = part_status_report(context)

    assert not report["ok"]
    assert lagging in report["behind"]
    assert Tables.cube.name in report["behind"]
    assert set(report["misaligned"]) == {lagging, Tables.cube.name}
    print("\n=== SANITY CHECK: cube status fails closed ===")
    print(
        "  the missing final cube and the one-day fundamentals lag are both named in the " "legacy behind list and the exact-edge detail. Validated."
    )
