"""Does `sql/schema.sql` declare exactly the columns the DEF 14A flatten writes?

A DDL that merely parses is not verified. Two failures hide behind a parsing DDL, and neither
shows up in any other test:

  * a column the flatten produces but the table lacks -> the insert fails at universe scale,
    after the LLM tokens have already been paid for;
  * a column the DDL declares but nothing writes -> permanently NULL, and a permanently-NULL
    column reads downstream as "this company does not disclose it" rather than "we never
    extracted it". That is exactly how the retired edgar table's `auditor_name` sat at 2.05%
    fill while the firm name is present in 98% of documents.

The parent row and all four child rows are built from a MAXIMAL extract -- every optional field
populated -- because the column set of a flatten is a function of the builder, not of the data:
a field that is None still produces its key. `def14a_llm`'s own column set is asserted the same
way, so removing a field from the Pydantic contract without touching the DDL fails here.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.data_extract.utils.schemas.def14a_schema import (
    BeneficialOwner,
    Def14AExtract,
    DirectorCompensation,
    DirectorInfo,
    ExecutiveCompensation,
    GovernanceProfile,
)
from src.data_extract.utils.structure.fetch_def14a_llm import _child_frames, _flatten

SCHEMA_SQL = Path(__file__).resolve().parents[3] / "sql" / "schema.sql"
#: Every table this module owns, parent first.
DEF14A_TABLES = ("def14a_llm", "def14a_executive_comp", "def14a_director_comp",
                 "def14a_ownership", "def14a_directors")


def _ddl_columns(table: str) -> list[str]:
    """Column names declared for `table` in sql/schema.sql, in declaration order."""
    sql = SCHEMA_SQL.read_text(encoding="utf-8")
    m = re.search(r'CREATE TABLE IF NOT EXISTS "%s" \((.*?)\n\);' % re.escape(table), sql, re.S)
    assert m, f"{table}: no CREATE TABLE block found in sql/schema.sql"
    return re.findall(r'^\s+"([a-z0-9_]+)"\s', m.group(1), re.M)


def _maximal_extract() -> Def14AExtract:
    """An extract with every array non-empty, so each builder emits its full key set."""
    return Def14AExtract(
        company_name="ACME Corp", fiscal_year=2024,
        directors=[DirectorInfo(name="Ada Byron", age=61, tenure_years=7.0,
                                is_independent=True, gender="female", gender_basis="stated",
                                other_public_company_boards=2)],
        compensation=[ExecutiveCompensation(name="Ada Byron", title="CEO", fiscal_year=2024,
                                            salary_usd=1_200_000.0,
                                            total_compensation_usd=18_000_000.0)],
        director_compensation=[DirectorCompensation(name="Grace Hopper", fiscal_year=2024,
                                                    fees_earned_usd=120_000.0,
                                                    total_compensation_usd=310_000.0)],
        ownership_holders=[BeneficialOwner(holder_name="The Vanguard Group",
                                           holder_type="5pct_holder", shares=41_000_000.0,
                                           percent_of_class=0.083)],
        governance=GovernanceProfile(board_size=11, auditor_name="Ernst & Young LLP",
                                     auditor_fees_usd=12_000_000.0),
    )


def _code_columns() -> dict[str, list[str]]:
    """The column set each table's builder actually produces."""
    extract = _maximal_extract()
    filing = pd.Series({"filing_date": pd.Timestamp("2025-04-01"), "cik": "0000001800",
                        "period_of_report": "2024-12-31", "accession_number": "a1"})
    out = {"def14a_llm": list(_flatten("ACME", filing, extract))}
    for name, rows in _child_frames("ACME", filing, extract).items():
        assert rows, f"{name}: the maximal extract produced no row -- builder bug, not a DDL gap"
        out[name] = list(rows[0])
    return out


def test_schema_sql_matches_the_flatten_column_for_column():
    code = _code_columns()
    assert set(code) | {"def14a_llm"} == set(DEF14A_TABLES), \
        "a child table was added or renamed without updating this test"

    report, bad = [], 0
    for table in DEF14A_TABLES:
        ddl, produced = _ddl_columns(table), code[table]
        missing = [c for c in produced if c not in ddl]
        extra = [c for c in ddl if c not in produced]
        bad += bool(missing or extra)
        report.append((table, len(ddl), len(produced), missing, extra))

    print("\n=== SANITY: sql/schema.sql vs the DEF 14A flatten ===")
    for table, n_ddl, n_code, missing, extra in report:
        print(f"  {table:<24} ddl {n_ddl:>3} / code {n_code:>3}  "
              f"{'OK' if not (missing or extra) else 'MISMATCH'}")
        if missing:
            print(f"    code writes but DDL lacks : {missing}")
        if extra:
            print(f"    DDL declares but unwritten: {extra}")
    print(f"  {len(DEF14A_TABLES)} tables, {bad} mismatch(es). A mismatch is an insert failure "
          f"at universe scale or a permanently-NULL column, not a style nit.")

    assert bad == 0, "sql/schema.sql and the flatten disagree (see the printed report)"


def test_primary_key_columns_are_never_null_in_a_built_row():
    """A PK column that can be NULL aborts the whole Postgres insert, not one row.

    `fiscal_year` is the live case: `Optional[int]` on the Pydantic model, yet part of
    `def14a_executive_comp`'s key -- so `_exec_comp_rows` skips a row that lacks one. 0 of 1,849
    rows replayed from the stored blobs were missing it, but that is evidence, not a guarantee,
    which is why the guard is structural and asserted here.
    """
    from src.data_extract.utils.structure.fetch_def14a_llm import _CHILD_SPEC

    filing = pd.Series({"filing_date": pd.Timestamp("2025-04-01"), "cik": "0000001800",
                        "period_of_report": "2024-12-31", "accession_number": "a1"})
    frames = _child_frames("ACME", filing, _maximal_extract())

    # a year-less NEO must be DROPPED, not written with a null key
    yearless = Def14AExtract(
        company_name="ACME Corp",
        compensation=[ExecutiveCompensation(name="Ada Byron", title="CEO", fiscal_year=None,
                                            salary_usd=1_200_000.0)])
    dropped = _child_frames("ACME", filing, yearless)["def14a_executive_comp"]

    print("\n=== SANITY: primary-key columns on the four child tables ===")
    for name, (_numeric, pk) in _CHILD_SPEC.items():
        rows = frames[name]
        nulls = {c: sum(r.get(c) is None for r in rows) for c in pk}
        print(f"  {name:<24} pk={pk} nulls={ {k: v for k, v in nulls.items() if v} or 'none'}")
        for c in pk:
            assert nulls[c] == 0, f"{name}.{c} is NULL in a built row -- PK columns are NOT NULL"
    print(f"  a NEO with no fiscal_year yields {len(dropped)} rows (dropped, never a null key).")
    assert dropped == []


def test_the_retired_technology_columns_are_gone_from_the_ddl():
    """They were an opinion, not an extraction (mean |delta| of 1.06 directors between
    consecutive filings of the same company, only 38.8% unchanged). `CREATE TABLE IF NOT EXISTS`
    cannot retire a column on a DB that already has one, so this only guards fresh bootstraps --
    an existing deployment needs the explicit ALTER recorded in PHASE-6."""
    ddl = _ddl_columns("def14a_llm")
    gone = [c for c in ("n_technology_directors", "pct_technology_directors",
                        "technology_committee") if c in ddl]
    print("\n=== SANITY: retired technology columns ===")
    print(f"  still declared in sql/schema.sql: {gone or 'none'}")
    print("  fresh bootstraps get 54 columns, none of them a board-technology opinion.")
    assert gone == []
