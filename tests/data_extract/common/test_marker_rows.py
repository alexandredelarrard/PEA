"""AC-016 check 1 for the EDGAR document tables: an empty-filing marker built by `marker_row` is one
row whose key columns are filled (the declared sentinel included), whose data columns read back as
typed missing values (NaN / NaT / None) after a store round trip, which `load` hides, and which
carries only columns the table's DDL in `sql/schema.sql` declares, so saving it never adds a column."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from src.data_extract.utils.common.edgar_driver import FilingStamp, marker_row
from src.data_store.schema import Table, Tables
from tests.data_extract.edgar_fixtures import fake_filing

_SCHEMA_SQL = Path(__file__).resolve().parents[3] / "sql" / "schema.sql"
_FILED = "2026-09-10"

#: table -> (one real row with a float, a date and a text data column, the data columns checked)
_CASES: dict[str, tuple[Table, dict[str, object], tuple[str, str, str]]] = {
    "fundamentals_facts": (
        Tables.fundamentals_facts,
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
    "sec_def14a": (
        Tables.def14a_edgar,
        {"form": "DEF 14A", "peo_total_comp": 1.0, "ecd_period_end": pd.Timestamp("2025-12-31"), "peo_name": "X"},
        ("peo_total_comp", "ecd_period_end", "peo_name"),
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


def _ddl_columns(name: str) -> set[str]:
    block = re.search(rf'CREATE TABLE IF NOT EXISTS "{name}" \((.*?)\n\);', _SCHEMA_SQL.read_text(encoding="utf-8"), re.S)
    assert block is not None, name
    return set(re.findall(r'^\s+"(\w+)"', block.group(1), re.M))


def _real_row(table: Table, extra: dict[str, object]) -> pd.DataFrame:
    assert table.resume is not None and table.resume.frontier_col is not None
    base = {"ticker": "AAA", "accession_number": "0000000001-26-000001", table.resume.frontier_col: pd.Timestamp("2026-09-09")}
    return pd.DataFrame([base | extra])


@pytest.mark.parametrize("name", sorted(_CASES))
def test_a_marker_is_one_typed_null_row_with_ddl_columns_only(name: str, sqlite_store) -> None:
    table, extra, (float_col, date_col, text_col) = _CASES[name]
    assert table.empty_marker is not None
    column, sentinel = table.empty_marker
    marker = marker_row(table, "AAA", FilingStamp.of(fake_filing("0000000001-26-000002", 1, _FILED, form="4"), "0000000001"))

    assert set(marker.columns) <= _ddl_columns(table.name), set(marker.columns) - _ddl_columns(table.name)
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


def test_marker_key_fill_for_the_composite_keys() -> None:
    stamp = FilingStamp.of(fake_filing("0000000001-26-000002", 1, _FILED, form="10-Q"), "0000000001")
    facts = marker_row(Tables.fundamentals_facts, "AAA", stamp).iloc[0]
    insider = marker_row(Tables.insider_transactions, "AAA", stamp).iloc[0]
    proxy = marker_row(Tables.def14a_edgar, "AAA", stamp).iloc[0]

    assert (facts["field"], facts["duration_type"], facts["period_end"]) == ("_empty", "_empty", pd.Timestamp(_FILED))
    assert (insider["security_type"], insider["row_sequence"], insider["source"]) == ("_empty", 0, "edgar") and "issuer_cik" not in insider.index
    assert proxy["form"] == "_empty" and proxy["cik"] == "0000000001"
    print("\n=== SANITY CHECK: marker key fill ===")
    print(
        "  fundamentals: field/duration_type '_empty', period_end = filing date; insider: ('_empty', 0, source 'edgar'), no issuer CIK; DEF 14A: form '_empty'."
    )
