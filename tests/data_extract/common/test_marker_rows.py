"""AC-016 check 1 for the EDGAR document tables: an empty-filing marker built by `marker_row` is one
row whose key columns are filled (the declared sentinel included), whose data columns read back as
typed missing values (NaN / NaT / None) after a store round trip, which `load` hides, and which
carries only columns the table's DDL declares, so saving it never adds a column. The DDL is the table's
`sql/schema.sql` block, or for `sec_fundamentals_facts` (no block: `managed=False`) the one `store.ensure_table`
creates from the producer's typed frame."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from src.data_extract.utils.edgar_extract.edgar_driver import FilingStamp, marker_row
from src.data_extract.utils.fundamentals import fetch_fundamentals_sec as facts_mod
from src.data_extract.utils.fundamentals.fetch_fundamentals_sec import _row as _facts_row
from src.data_extract.utils.fundamentals.reason_codes import NOT_DISCLOSED
from src.data_extract.utils.fundamentals.xbrl_linkbase import UNRESOLVED, Resolution
from src.data_store import ddl
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore, ensure_table
from tests.canonical_schema import create_canonical
from tests.data_extract.edgar_fixtures import fake_filing
from tests.support.fundamentals_frames import sec_facts_frame

_SCHEMA_SQL = Path(__file__).resolve().parents[3] / "sql" / "schema.sql"
_FILED = "2026-09-10"

#: table -> (one real row with a float, a date and a text data column, the data columns checked)
_CASES: dict[str, tuple[Table, dict[str, object], tuple[str, str, str]]] = {
    "sec_fundamentals_facts": (
        Tables.sec_fundamentals_facts,
        {
            "field": "totalRevenue",
            "duration_type": "duration",
            "period_end": pd.Timestamp("2026-06-30"),
            "value": 1.0,
            "period_start": pd.Timestamp("2026-01-01"),
            "unit": "USD",
        },
        ("value", "period_start", "unit"),
    ),
    "sec_8k": (
        Tables.sec_8k,
        {"item": "8.01", "has_earnings": 1.0, "period_of_report": pd.Timestamp("2026-09-09"), "item_text": "body"},
        ("has_earnings", "period_of_report", "item_text"),
    ),
    "sec_filing_text": (
        Tables.filing_risk_text,
        {"section": "mda", "n_words": 3.0, "period_of_report": pd.Timestamp("2026-06-30"), "text": "body"},
        ("n_words", "period_of_report", "text"),
    ),
    "sec_employees_text": (
        Tables.sec_employees_text,
        {"source_document": "primary", "n_chars": 9.0, "period_of_report": pd.Timestamp("2025-12-31"), "windows": "[[0, 9]]"},
        ("n_chars", "period_of_report", "windows"),
    ),
    "sec_13d": (
        Tables.sec_13d,
        {"rp_seq": 0, "percent_of_class": 5.5, "date_of_event": pd.Timestamp("2026-09-01"), "cusip": "000000AA1"},
        ("percent_of_class", "date_of_event", "cusip"),
    ),
    "sec_13g": (
        Tables.sec_13g,
        {"rp_seq": 0, "percent_of_class": 5.5, "date_of_event": pd.Timestamp("2026-09-01"), "cusip": "000000AA1"},
        ("percent_of_class", "date_of_event", "cusip"),
    ),
    "insider_transactions": (
        Tables.insider_transactions,
        {
            "security_type": "nonderiv",
            "row_sequence": 1,
            "source": "edgar",
            "shares": 10.0,
            "transaction_date": pd.Timestamp("2026-09-01"),
            "owner_name": "X",
        },
        ("shares", "transaction_date", "owner_name"),
    ),
}


def _ddl(table: Table, monkeypatch: pytest.MonkeyPatch) -> str:
    """The table's `CREATE TABLE` text: its `sql/schema.sql` block, or the registry DDL from the producer's frame."""
    if not table.managed:
        assert table is Tables.sec_fundamentals_facts, table.name
        return ddl.table_ddl_from_frame(table, sec_facts_frame(monkeypatch))
    block = re.search(rf'CREATE TABLE IF NOT EXISTS "{table.name}" \(.*?\n\);', _SCHEMA_SQL.read_text(encoding="utf-8"), re.S)
    assert block is not None, table.name
    return block.group(0)


def _ddl_columns(table: Table, monkeypatch: pytest.MonkeyPatch) -> set[str]:
    return set(re.findall(r'^\s+"(\w+)"', _ddl(table, monkeypatch), re.M))


def _create(store: DataStore, table: Table, monkeypatch: pytest.MonkeyPatch) -> None:
    """The table as a fresh database gets it: `sql/schema.sql`, or `ensure_table` on the producer's typed frame."""
    if table.managed:
        create_canonical(store.engine, [table])
    else:
        ensure_table(store.engine, table.name, sec_facts_frame(monkeypatch))


def _real_row(table: Table, extra: dict[str, object]) -> pd.DataFrame:
    assert table.resume is not None and table.resume.frontier_col is not None
    base = {"ticker": "AAA", "accession_number": "0000000001-26-000001", table.resume.frontier_col: pd.Timestamp("2026-09-09")}
    return pd.DataFrame([base | extra])


@pytest.mark.parametrize("name", sorted(_CASES))
def test_a_marker_is_one_typed_null_row_with_ddl_columns_only(name: str, sqlite_store, monkeypatch: pytest.MonkeyPatch) -> None:
    table, extra, (float_col, date_col, text_col) = _CASES[name]
    _create(sqlite_store, table, monkeypatch)
    assert table.empty_marker is not None
    column, sentinel = table.empty_marker
    marker = marker_row(table, "AAA", FilingStamp.of(fake_filing("0000000001-26-000002", 1, _FILED, form="4"), "0000000001"))

    columns = _ddl_columns(table, monkeypatch)
    assert set(marker.columns) <= columns, set(marker.columns) - columns
    sqlite_store.save(table, _real_row(table, extra))
    sqlite_store.save(table, marker)

    shown = sqlite_store.load(table, markers=True)
    row = shown[shown[column] == sentinel].iloc[0]
    assert len(shown) == 2 and set(table.pk) <= set(marker.columns)
    assert pd.isna(row[float_col]) and isinstance(row[float_col], float)
    assert row[date_col] is None or row[date_col] is pd.NaT
    assert pd.isna(row[text_col]) and not isinstance(row[text_col], str)
    assert sqlite_store.load(table)["accession_number"].tolist() == ["0000000001-26-000001"]
    print(f"\n=== SANITY CHECK: {name} marker ===")
    print(
        f"  {column}={sentinel!r}; {float_col}={row[float_col]!r}, {date_col}={row[date_col]!r}, {text_col}={row[text_col]!r}; hidden from load; DDL columns only."
    )


def test_a_def14a_text_rejection_marker_is_hidden_beside_a_real_section(sqlite_store, monkeypatch: pytest.MonkeyPatch) -> None:
    table = Tables.sec_def14a_text
    create_canonical(sqlite_store.engine, [table])
    marker = marker_row(table, "AAA", FilingStamp.of(fake_filing("0000000001-26-000002", 1, _FILED, form="DEF 14A"), "0000000001"))
    real = _real_row(table, {"section": "audit", "primary_document": "", "payload": "fees", "n_chars": 4})

    assert set(marker.columns) <= _ddl_columns(table, monkeypatch), set(marker.columns) - _ddl_columns(table, monkeypatch)
    sqlite_store.save(table, real)
    sqlite_store.save(table, marker)

    shown = sqlite_store.load(table, markers=True).set_index("section")
    assert sorted(shown.index) == ["_empty", "audit"] and shown.loc["_empty", "form"] == "DEF 14A"
    assert pd.isna(shown.loc["_empty", "payload"]) and pd.isna(shown.loc["_empty", "n_chars"])
    assert shown.loc["audit", "primary_document"] == "" and sqlite_store.load(table)["section"].tolist() == ["audit"]
    print("\n=== SANITY CHECK: sec_def14a_text marker ===")
    print("  section='_empty' with NULL payload/n_chars beside a real 'audit' row; '' primary_document survives; load hides the marker. Validated.")


def test_a_value_less_facts_row_is_kept_without_label_columns(sqlite_store, monkeypatch: pytest.MonkeyPatch) -> None:
    """AC-004 / AM-1: a field the filing tags nowhere is still written as one value-less row (it marks the field as
    read), and the facts table carries no `dc_code`, `adjustment` or `is_amendment` column."""
    stamp = FilingStamp.of(fake_filing("0000000001-26-000003", 1, _FILED, form="10-Q"), "0000000001")
    resolution = Resolution(field="totalRevenue", method=UNRESOLVED, dc_code=NOT_DISCLOSED)
    row = _facts_row("AAA", stamp, pd.Timestamp("2026-06-30"), "industrial", "totalRevenue", resolution, None)
    table = Tables.sec_fundamentals_facts
    # no table yet: the producer's frame through `store.save` creates it, as on a cold night run
    monkeypatch.setattr(facts_mod, "filing_rows", lambda ticker, stamp, catalogue, gics: [row])
    sqlite_store.save(table, facts_mod.parse_fundamentals("AAA", "0000000001", stamp, None, catalogue=None, gics_by_ticker={})[table])  # type: ignore[arg-type]

    stored = sqlite_store.load(table).iloc[0]
    labels = {"dc_code", "adjustment", "is_amendment"}
    assert set(row) == set(table.retrieval_columns), set(row) ^ set(table.retrieval_columns)
    assert not labels & (set(table.retrieval_columns) | set(sqlite_store.columns(table)))
    assert pd.isna(stored["value"]) and stored["field"] == "totalRevenue" and stored["resolution_method"] == UNRESOLVED
    assert pd.Timestamp(stored["period_end"]) == pd.Timestamp("2026-06-30"), "a value-less row's PK falls back to the period of report"
    print("\n=== SANITY CHECK: value-less facts row ===")
    print(
        f"  stored one row: field={stored['field']}, value={stored['value']!r}, method={stored['resolution_method']}, "
        f"period_end={pd.Timestamp(stored['period_end']).date()}; no {sorted(labels)} column in the registry or the DDL."
    )


def test_marker_key_fill_for_the_composite_keys() -> None:
    stamp = FilingStamp.of(fake_filing("0000000001-26-000002", 1, _FILED, form="10-Q"), "0000000001")
    facts = marker_row(Tables.sec_fundamentals_facts, "AAA", stamp).iloc[0]
    insider = marker_row(Tables.insider_transactions, "AAA", stamp).iloc[0]
    employees = marker_row(Tables.sec_employees_text, "AAA", stamp).iloc[0]

    assert (facts["field"], facts["duration_type"], facts["period_end"]) == ("_empty", "_empty", pd.Timestamp(_FILED))
    assert (insider["security_type"], insider["row_sequence"], insider["source"]) == ("_empty", 0, "edgar") and "issuer_cik" not in insider.index
    assert employees["source_document"] == "_empty" and employees["cik"] == "0000000001"
    print("\n=== SANITY CHECK: marker key fill ===")
    print(
        "  fundamentals: field/duration_type '_empty', period_end = filing date; insider: ('_empty', 0, source 'edgar'), no issuer CIK; "
        "employees text: source_document '_empty'."
    )
