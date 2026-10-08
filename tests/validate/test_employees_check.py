"""Employee headcount validator on known-truth frames: a real `DataStore` on SQLite and a hand-written cached
EDGAR index (Parquet), no network. AAA files every year; BBB moved registrant in 2015; CIK 99 is foreign.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data_store.schema import Tables
from src.validate.checks.employees import check_employees

AAA, PRED, SUCC, FOREIGN = "0000000001", "0000000020", "0000000002", "0000000099"
SINCE = pd.Timestamp("2010-01-01")
_INDEX = pa.schema([("cik", pa.int64()), ("company", pa.string()), ("form", pa.string()), ("filed", pa.date32()), ("accession", pa.string())])


def _lineage_row(ticker: str, cik: str, start: str = "1900-01-01", end: str | None = None) -> dict[str, Any]:
    return {
        "entity_id": f"E-{ticker}",
        "canonical_ticker": ticker,
        "cik": cik,
        "role": "cik_window",
        "symbol": "",
        "valid_from": pd.Timestamp(start),
        "valid_to": pd.Timestamp(end) if end else pd.NaT,
        "status": "curated",
        "sources": "register",
        "oracle": "register",
        "confidence": None,
        "n_observations": 1,
        "evidence": "fixture",
        "scope_changed_at": pd.Timestamp("2026-01-01"),
    }


def _write_index(directory: Path) -> None:
    rows = [
        (AAA, "10-K", "2018-02-15", "a1"),
        (AAA, "10-K", "2019-02-14", "a2"),
        (AAA, "10-K", "2020-02-13", "a3"),
        (AAA, "10-K/A", "2020-04-01", "a4"),
        (AAA, "10-K", "2021-02-12", "a5"),
        (AAA, "8-K", "2021-03-01", "a6"),  # not an annual form
        (AAA, "10-K", "2005-02-15", "a0"),  # before the window
        (PRED, "10-K", "2014-03-01", "b1"),
        (SUCC, "10-K", "2016-03-01", "b2"),
        (PRED, "10-K", "2016-03-05", "b3"),  # the predecessor after its window: not owned
        (FOREIGN, "10-K", "2019-02-14", "f1"),
    ]
    frame = pd.DataFrame(
        {
            "cik": [int(c) for c, *_ in rows],
            "company": "X",
            "form": [f for _, f, _, _ in rows],
            "filed": [pd.Timestamp(d).date() for _, _, d, _ in rows],
            "accession": [a for *_, a in rows],
        }
    )
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(frame, schema=_INDEX, preserve_index=False), directory / "2016Q1.parquet")


def _row(ticker: str, as_of: str, cik: str, accession: str, status: str, **values: Any) -> dict[str, Any]:
    components = {k: values.get(k) for k in ("employees_total", "employees_full_time", "employees_part_time")}
    quotes = values.get("quotes", {k.removeprefix("employees_"): f"{v:,} employees" for k, v in components.items() if v is not None})
    return {
        "ticker": ticker,
        "as_of": pd.Timestamp(as_of),
        "cik": cik,
        "accession_number": accession,
        "form": "10-K",
        **components,
        "basis": values.get("basis"),
        "status": status,
        "source_document": "primary" if status == "found" else None,
        "source_quote": json.dumps(quotes) if quotes else None,
        "measurement_period": None,
    }


def _component_rows() -> pd.DataFrame:
    rows = [
        _row("AAA", "2017-06-01", AAA, "s1", "not_disclosed"),  # stale: no owned filing that day
        _row("AAA", "2018-02-15", AAA, "a2", "found", employees_total=100, basis="total"),  # as_of is not a2's filing date
        _row("AAA", "2019-02-14", AAA, "a2", "found", employees_total=200, basis="total"),  # +100 % within one basis
        _row(
            "AAA",
            "2020-02-13",
            AAA,
            "a3",
            "found",
            employees_total=210,
            employees_full_time=150,
            employees_part_time=60,
            basis="full_part",
            quotes={"full_time": "150 full-time", "total": "210 employees"},  # no part-time quote
        ),
        _row("AAA", "2020-04-01", AAA, "a4", "found"),  # found with no component
        # AAA 2021-02-12 (a5) has no row: missing
        _row("BBB", "2014-03-01", PRED, "b1", "found", employees_total=50, basis="total"),
        _row("BBB", "2016-03-01", SUCC, "b2", "found", employees_total=52, basis="total"),
        _row("BBB", "2016-03-05", FOREIGN, "f9", "found", employees_total=900, basis="total"),  # foreign and stale
    ]
    frame = pd.DataFrame(rows)
    return frame.astype({c: "Int64" for c in ("employees_total", "employees_full_time", "employees_part_time")})


def _seed(store: Any, employees: pd.DataFrame) -> None:
    store.save(Tables.sp500_tickers, pd.DataFrame({"ticker": ["AAA", "BBB"], "cik": [AAA, SUCC]}))
    store.save(
        Tables.entity_lineage,
        pd.DataFrame([_lineage_row("AAA", AAA), _lineage_row("BBB", PRED, end="2015-07-01"), _lineage_row("BBB", SUCC, start="2015-07-01")]),
    )
    store.save(Tables.fundamentals_employees, employees)


def _context(store: Any) -> Any:
    return SimpleNamespace(store=store, log=logging.getLogger("test.employees_check"))


def test_component_table_findings_match_the_known_truth(sqlite_store, tmp_path):
    """AC-004..006 measures: one missing original date, two stale rows, one foreign row, provenance and status breaches."""
    _write_index(tmp_path / "index")
    _seed(sqlite_store, _component_rows())

    report = check_employees(_context(sqlite_store), since=SINCE, directory=tmp_path / "index")
    m = report.result.metrics

    assert m["shape"] == "component"
    assert (m["owned_dates"], m["owned_original_dates"]) == (7, 6)
    assert list(zip(report.missing["ticker"], report.missing["filed"].dt.date.astype(str), strict=True)) == [("AAA", "2021-02-12")]
    assert m["missing_original_dates"] == 1
    assert sorted(zip(report.stale["ticker"], report.stale["as_of"].dt.date.astype(str), strict=True)) == [
        ("AAA", "2017-06-01"),
        ("BBB", "2016-03-05"),
    ]
    assert m["foreign_rows"] == 1
    assert m["found_without_component"] == 1 and m["component_without_found"] == 0
    assert m["unknown_status"] == 0 and m["unknown_basis"] == 0
    assert m["provenance_incomplete_rows"] == 2
    assert m["provenance_problems"] == {"no quote": 1, "source_quote null": 1, "basis null": 1}
    assert m["as_of_not_filing_date"] == 1 and m["accession_not_in_index"] == 2
    assert m["status_mix"] == {"found": 7, "not_disclosed": 1}
    assert list(report.switches[["ticker", "from_basis", "to_basis"]].itertuples(index=False, name=None)) == [("AAA", "total", "full_part")]
    assert list(report.jumps[["ticker", "from_total", "to_total"]].itertuples(index=False, name=None)) == [("AAA", 100.0, 200.0)]
    assert report.result.status == "fail"
    per = report.per_ticker.set_index("ticker")
    assert per.loc["AAA", "missing_original_dates"] == 1 and per.loc["BBB", "foreign_rows"] == 1 and per.loc["BBB", "owned_dates"] == 2
    fields = {f.field for f in report.result.findings}
    assert fields >= {"coverage", "stale_rows", "foreign_rows", "found_without_component", "provenance", "as_of", "basis_switch", "scope_jump"}

    print("\n=== SANITY CHECK: employee validator, component shape ===")
    print(report.per_ticker.to_string(index=False))
    print(f"  metrics: {json.dumps({k: m[k] for k in ('missing_dates', 'stale_rows', 'foreign_rows', 'provenance_incomplete_rows')})}")
    print(
        "  OK: the 2021 10-K is the one missing date; the predecessor's post-window filing and the foreign CIK row are stale; "
        "the 8-K and pre-window rows are ignored; the switch and the +100 % jump are reported"
    )


def test_old_shape_table_degrades_to_coverage_and_jumps(sqlite_store, tmp_path):
    """The pre-component table (ticker, as_of, employees) is read as totals; the new-column checks are skipped, not crashed."""
    _write_index(tmp_path / "index")
    old = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA", "BBB"],
            "as_of": pd.to_datetime(["2018-02-15", "2019-02-14", "2017-06-01", "2014-03-01"]),
            "employees": [100.0, 400.0, None, 50.0],
        }
    )
    _seed(sqlite_store, old)

    report = check_employees(_context(sqlite_store), since=SINCE, directory=tmp_path / "index")
    m = report.result.metrics

    assert m["shape"] == "old"
    assert m["missing_dates"] == 4 and m["missing_original_dates"] == 3  # AAA 2020-02, 2020-04 (10-K/A), 2021; BBB 2016-03-01
    assert m["stale_rows"] == 1
    assert m["status_mix"] == {"found": 3, "null_status": 1}
    assert len(report.jumps) == 1 and report.switches.empty
    shape = [f for f in report.result.findings if f.field == "shape"]
    assert len(shape) == 1 and "provenance" in shape[0].evidence["skipped"]
    assert "foreign_rows" not in m

    print("\n=== SANITY CHECK: employee validator, old shape ===")
    print(report.per_ticker.to_string(index=False))
    print("  OK: coverage, stale rows and jumps measured on the old shape; provenance/identity/basis reported as skipped")


def test_a_missing_index_is_a_finding_not_a_crash_or_a_pass(sqlite_store, tmp_path):
    """No cached index: coverage cannot be measured, so the check fails on that finding and every row reads as stale."""
    _seed(sqlite_store, _component_rows())

    report = check_employees(_context(sqlite_store), since=SINCE, directory=tmp_path / "absent")

    assert report.result.metrics["owned_dates"] == 0 and report.result.metrics["stale_rows"] == 8
    assert any(f.field == "coverage" and f.score == 5 for f in report.result.findings) and report.result.status == "fail"

    print("\n=== SANITY CHECK: employee validator, no index ===")
    print("  OK: an absent index is reported as unmeasured coverage (score 5), never a pass")
