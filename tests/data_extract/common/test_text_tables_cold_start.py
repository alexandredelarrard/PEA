"""Cold start of the two EDGAR text tables and the five DEF 14A tables: each is created from its committed
`sql/schema.sql` block, whatever frame a walk saves first lands in that layout, and a first nightly run in DAG order on
a store without them (identity purge, both text walks, then the three LLM steps, which also run before any of them
exists) raises nothing and stores every row.

Offline: a real SQLite `DataStore`, a seeded local EDGAR index, filing stand-ins, stubbed SEC reads and recording
extractors (no provider, no LLM call).
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import src.data_extract.utils.fundamentals.fundamentals_employees as employees
from src.data_extract.transformers.step_extract_edgar import EdgarOptions, StepExtractEdgar
from src.data_extract.utils.edgar_extract import edgar_driver, edgar_index
from src.data_extract.utils.edgar_extract import fetch_def14a_text as def14a_text
from src.data_extract.utils.edgar_extract import fetch_employees_text as employees_text
from src.data_extract.utils.edgar_extract.edgar_driver import FilingStamp, marker_row, parse_filing_rows
from src.data_extract.utils.edgar_extract.employee_document import EmployeeText
from src.data_extract.utils.edgar_extract.fetch_def14a_text import DEF14A_TEXT_FETCH
from src.data_extract.utils.edgar_extract.fetch_employees_text import EMPLOYEES_TEXT_FETCH
from src.data_extract.utils.identity.identity_propagate import owned_scopes, purge_rows
from src.data_extract.utils.structure.def14a import fetch as def14a_llm
from src.data_extract.utils.structure.votes.fetch import fetch_8k_votes_llm
from src.data_store import ddl
from src.data_store.schema import Table, Tables
from src.data_store.store import reflect_columns
from src.gpt_extract.utils.schemas_gpt import LlmResult, LlmTask
from src.utils.filer_tables import PURGE_TABLES
from src.utils.string import run_date
from tests.canonical_schema import create_canonical
from tests.data_extract.common.scope_fixtures import SENTINEL, dated_identity
from tests.data_extract.common.test_edgar_text_walks import FOREIGN, ProxyWorld
from tests.data_extract.edgar_fixtures import FakeEmployeesLLM, TenK, employee_answer, fake_context, identity_for, patch_index_filings, seed_index

BLOCKS = ddl.existing_blocks((Path(__file__).resolve().parents[3] / "sql" / "schema.sql").read_text(encoding="utf-8"))
ROSTER = identity_for({"AAA": "320193"})
BODY = "As of December 31, 2021, we had approximately 41,000 employees."
SPAN = (BODY.index("41,000"), BODY.index("41,000") + len("41,000"))
VOTES_TEXT = "Item 5.07 Submission of Matters to a Vote of Security Holders. " + "Jane Example For 1,234,567 Against 12,345 Abstain 1,000. " * 8
#: The tables the LLM steps read or write that the night does not create: the text tables and the DEF 14A tables are
#: created by the night itself.
LIVE_TABLES = (Tables.fundamentals_employees, Tables.sec_8k, Tables.sec_8k_votes)
DEF14A_TABLES = tuple(section.table for section in def14a_llm.SECTIONS)


def _stamp(accession: str, period: object = "2023-12-31") -> FilingStamp:
    filing = SimpleNamespace(accession_number=accession, form="10-K", cik="320193", filing_date="2024-02-01", period_of_report=period)
    return FilingStamp.of(filing, "0000320193")


def _employee_shapes(monkeypatch: pytest.MonkeyPatch) -> dict[str, pd.DataFrame]:
    """The first frames the employees walk can save, built by its own row function (document read stubbed)."""
    monkeypatch.setattr(employees_text, "employee_text", lambda filing: EmployeeText(BODY, "primary", (SPAN,)))
    monkeypatch.setattr(employees_text, "forget_sgml", lambda filing: None)
    table = Tables.sec_employees_text
    data = parse_filing_rows("AAA", "0000320193", _stamp("a-1"), ROSTER, table=table, row_fn=employees_text._employee_rows)[table]
    no_period = parse_filing_rows("AAA", "0000320193", _stamp("a-3", None), ROSTER, table=table, row_fn=employees_text._employee_rows)[table]
    marker = marker_row(table, "AAA", _stamp("a-2"))
    return {
        "data": data,
        "data, no period": no_period,
        "marker": marker,
        "data then marker": pd.concat([data, marker], ignore_index=True),
        "marker then data": pd.concat([marker, data], ignore_index=True),
    }


def _def14a_shapes() -> dict[str, pd.DataFrame]:
    """The first frames the DEF 14A walk can save: section rows, the `_none` row, the subject-guard marker, mixed."""
    table = Tables.sec_def14a_text
    base = {"cik": "0000320193", "accession_number": "d-1", "form": "DEF 14A", "filing_date": pd.Timestamp("2024-01-05"), "primary_document": "p.htm"}
    rows = def14a_text._section_rows("AAA", base, {"audit": "Audit fees ...", "ownership": "Holders ..."})
    none_rows = def14a_text._section_rows("AAA", {**base, "accession_number": "d-2"}, {"audit": " "})
    stamp = _stamp("d-3")
    data = parse_filing_rows("AAA", "", stamp, ROSTER, table=table, row_fn=lambda t, s: rows)[table]
    none = parse_filing_rows("AAA", "", stamp, ROSTER, table=table, row_fn=lambda t, s: none_rows)[table]
    marker = marker_row(table, "AAA", stamp)
    return {
        "data": data,
        "_none": none,
        "marker": marker,
        "data then marker": pd.concat([data, marker], ignore_index=True),
        "marker then data": pd.concat([marker, data], ignore_index=True),
    }


SHAPES = [
    *((Tables.sec_employees_text, shape) for shape in ("data", "data, no period", "marker", "data then marker", "marker then data")),
    *((Tables.sec_def14a_text, shape) for shape in ("data", "_none", "marker", "data then marker", "marker then data")),
]


def _committed_layout(table: Table) -> list[tuple[str, str]]:
    """The `(column, type)` pairs of `table`'s `sql/schema.sql` block, in order."""
    return re.findall(r'^\s+"([^"]+)" ([A-Z]+)', BLOCKS[table.name], re.M)


@pytest.mark.parametrize(("table", "shape"), SHAPES, ids=[f"{t.name}-{s}" for t, s in SHAPES])
def test_every_first_frame_lands_in_the_committed_layout(table: Table, shape: str, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    df = (_employee_shapes(monkeypatch) if table is Tables.sec_employees_text else _def14a_shapes())[shape]

    sqlite_store.create_declared(table)
    sqlite_store.save(table, df)

    layout = reflect_columns(sqlite_store.engine, table.name)
    stored = sqlite_store.load(table, markers=True)
    assert layout == _committed_layout(table) and len(stored) == len(df)
    print(f"\n=== SANITY CHECK: {table.name} first save = {shape} ===")
    print(f"  created from sql/schema.sql ({len(layout)} columns, types EQUAL); the first save stores all {len(stored)} row(s), no column added.")


@pytest.mark.parametrize("table", [Tables.sec_employees_text, Tables.sec_def14a_text], ids=lambda t: t.name)
def test_marker_first_save_then_data_save_stores_every_row(table: Table, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    frames = _employee_shapes(monkeypatch) if table is Tables.sec_employees_text else _def14a_shapes()

    sqlite_store.create_declared(table)
    sqlite_store.save(table, frames["marker"])
    created = sqlite_store.columns(table.name)
    sqlite_store.save(table, frames["data"])

    stored = sqlite_store.load(table, markers=True)
    assert created == list(table.retrieval_columns)
    assert len(stored) == len(frames["marker"]) + len(frames["data"])
    print(f"\n=== SANITY CHECK: {table.name} marker-first cold save ===")
    print(
        f"  a marker-only first save into the declared table keeps all {len(created)} columns; the next data save stores, {len(stored)} rows in all."
    )


class SectionRecorder:
    """A DEF 14A section extractor with no provider: records each task and answers it with the section's empty answer."""

    threads = 1

    def __init__(self, section: Any) -> None:
        self.section = section
        self.tasks: list[LlmTask] = []

    def run(self, tasks: list[LlmTask]) -> list[LlmResult]:
        self.tasks += tasks
        empty = {"percent_unit": "percent"} if self.section.key in ("governance", "ownership") else {}
        return [LlmResult(seq=task.seq, task=task, parsed=self.section.schema.model_validate(empty)) for task in tasks]


class VotesRecorder:
    """An Item 5.07 extractor with no provider: records each task and returns no answer."""

    def __init__(self) -> None:
        self.tasks: list[LlmTask] = []

    def run_extraction(self, tasks: list[LlmTask], **_: Any) -> list[LlmResult]:
        self.tasks += tasks
        return []


@pytest.fixture
def night(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """AAA (two CIK windows: its only 10-K is blank and its only proxy names a foreign subject, so both of its first saves
    are markers) and BBB (one CIK: a 10-K with a headcount and a proxy with a board section). The store holds only the
    empty employees and 8-K tables (committed DDL); no text table and no DEF 14A table."""
    monkeypatch.setattr(edgar_driver, "_sleep", lambda seconds: None)
    monkeypatch.setattr(edgar_index, "refreshed_today", lambda context, years_history: True)
    create_canonical(sqlite_store.engine, LIVE_TABLES)
    identity = dated_identity(
        [
            ("AAA", "0009999991", "cik_window", SENTINEL, "2020-01-01"),
            ("AAA", "0000000001", "cik_window", "2020-01-01", None),
            ("BBB", "0000000002", "cik_window", SENTINEL, None),
        ],
        {"AAA": "0000000001", "BBB": "0000000002"},
    )
    monkeypatch.setattr(edgar_driver, "load_identity", lambda context: identity)
    monkeypatch.setattr(employees, "load_identity", lambda context: identity)
    budget = {"employees": 60_000, "def14a": 60_000, "votes": 60_000}
    ctx = fake_context(tmp_path, sqlite_store, ["AAA", "BBB"], ciks=["1", "2"], fundamentals_workers=1, llm_text_budget=budget)

    tenks = [TenK("0000000001-21-000001", "2021-03-01", "   ", cik=1), TenK("0000000002-22-000001", "2022-03-01", BODY, cik=2)]
    proxies = [(1, "AAA Inc", "DEF 14A", "2022-03-02", "0000000001-22-000002"), (2, "BBB Inc", "DEF 14A", "2022-03-02", "0000000002-22-000002")]
    seed_index(ctx, [(f.cik, "Fixture Co", f.form, f.filing_date, f.accession_number) for f in tenks] + proxies)
    patch_index_filings(monkeypatch, {f.accession_number: f for f in tenks})
    world = ProxyWorld(monkeypatch, {"0000000001": [(proxies[0][4], "a.htm")], "0000000002": [(proxies[1][4], "b.htm")]})
    world.subjects = {proxies[0][4]: frozenset({FOREIGN})}
    return SimpleNamespace(ctx=ctx, identity=identity, tenks=tenks, proxies=proxies)


def _llm_steps(ctx: Any) -> dict[str, int]:
    """`sec-employees-llm`, `sec-def14a-llm`, `sec-8k-votes` in DAG order; the tasks each extractor received."""
    tickers = ["AAA", "BBB"]
    employee_llm = FakeEmployeesLLM({"0000000002-22-000001": employee_answer(41_000, BODY)})
    sections = {section.key: SectionRecorder(section) for section in def14a_llm.SECTIONS}
    votes = VotesRecorder()
    employees.fetch_employees_llm(ctx, tickers, 15, [employee_llm])  # type: ignore[list-item]
    def14a_llm.fetch_def14a_llm(ctx, ctx.config, tickers, sections)  # type: ignore[arg-type]
    fetch_8k_votes_llm(ctx, tickers, votes)  # type: ignore[arg-type]
    return {"employees": len(employee_llm.tasks), "def14a": sum(len(r.tasks) for r in sections.values()), "votes": len(votes.tasks)}


def _committed_columns(table: Table) -> list[str]:
    """The column names of `table`'s `sql/schema.sql` block, in order."""
    return re.findall(r'^\s+"([^"]+)" ', BLOCKS[table.name], re.M)


def test_first_night_in_dag_order_creates_and_fills_both_text_tables(night: Any) -> None:
    ctx, store = night.ctx, night.ctx.store
    text_tables = (Tables.sec_employees_text, Tables.sec_def14a_text)
    created_tables = (*text_tables, *DEF14A_TABLES)

    # identity purge on the empty store, then the LLM steps before any text table exists
    missing_before = [table.name for table in created_tables if not store.exists(table)]
    purged = [r for spec in PURGE_TABLES for r in purge_rows(ctx, spec, ["AAA", "BBB"], owned_scopes(night.identity), night.identity.bounds, False)]
    before = _llm_steps(ctx)

    # the two text walks through the step (each creates its table), then the 8-K walk's 5.07 narrative and the LLM steps
    step = StepExtractEdgar(ctx, ctx.config, EdgarOptions(tickers=["AAA", "BBB"], years=15, as_of=run_date(None)))
    text_walks = ((EMPLOYEES_TEXT_FETCH, step.run_employees_text), (DEF14A_TEXT_FETCH, step.run_def14a_text))
    walks = {fetch.done.name: walk() for fetch, walk in text_walks}
    narrative = {"ticker": "BBB", "cik": "0000000002", "accession_number": "0000000002-22-000009", "item": "5.07", "form": "8-K"}
    store.save(
        Tables.sec_8k, pd.DataFrame([{**narrative, "filing_date": pd.Timestamp("2022-05-01"), "is_amendment": False, "item_text": VOTES_TEXT}])
    )
    after = _llm_steps(ctx)
    again = {fetch.done.name: walk().work.size for fetch, walk in text_walks}

    stored = {table.name: store.load(table, markers=True) for table in text_tables}
    layout = {table.name: store.exists(table) and store.columns(table.name) == _committed_columns(table) for table in created_tables}
    decided = store.load(Tables.fundamentals_employees, where={"ticker": "BBB"})
    answered = store.load(Tables.def14a_llm, columns=["ticker", "def14a_json"], where={"ticker": "BBB"})
    assert purged == [] and before == {"employees": 0, "def14a": 0, "votes": 0} and missing_before == [t.name for t in created_tables]
    assert all(layout.values()) and all(not summary.missing for summary in walks.values()) and again == dict.fromkeys(again, 0)
    assert answered["def14a_json"].notna().all() and len(answered) == 1
    assert sorted(stored["sec_employees_text"]["source_document"]) == ["_empty", "primary"]
    assert set(stored["sec_def14a_text"]["section"]) == {"_empty", "governance"}
    assert after == {"employees": 1, "def14a": 1, "votes": 1} and decided["employees_total"].tolist() == [41_000]
    print("\n=== SANITY CHECK: cold-start night in DAG order (AC-004) ===")
    print(f"  no text or DEF 14A table: identity purge {len(purged)} rows; LLM steps before the walks made {before} tasks;")
    print(
        f"  walks (each table created from its committed DDL): {', '.join(f'{n} {len(df)} rows' for n, df in stored.items())}, committed layout {layout};"
    )
    print(f"  LLM steps after the walks: {after} tasks; BBB decided {decided['employees_total'].tolist()}; BBB's governance answer archived;")
    print(f"  the 5 DEF 14A tables exist with their committed layout; a second walk lists {again}.")
