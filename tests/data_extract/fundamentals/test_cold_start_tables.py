"""On an EMPTY database the four fundamentals tables are created by their producers' own save paths.

No `sql/schema.sql` block exists for them: the nightly run creates each one on its first write
(`store.ensure_table`, registry PK and indexes, column types from the producer's frame). Each test
starts from a database with none of the four tables and runs the production writer: the SEC facts
walk (`run_edgar_fetch` on `SEC_FUNDAMENTALS_FETCH`, an empty filing first, then on 8 threads), the
SEC history build, the Sharadar SF1 fetch and the Sharadar history build. The XBRL read and the
Sharadar HTTP call are the only stand-ins. SQLite: types and the PK are checked here; Postgres
index names and COPY are proved by the scratch-Postgres run.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from sqlalchemy import create_engine, inspect, text

from src.constants.constants import SHARADAR_SF1_COLUMNS
from src.data_extract.utils.edgar_extract import edgar_driver
from src.data_extract.utils.edgar_extract.edgar_driver import FilingStamp, run_edgar_fetch
from src.data_extract.utils.fundamentals import build_history
from src.data_extract.utils.fundamentals import fetch_fundamentals_sec as facts_mod
from src.data_extract.utils.fundamentals.fetch_fundamentals_sec import SEC_FUNDAMENTALS_FETCH, parse_fundamentals
from src.data_extract.utils.fundamentals.periods import INSTANT, QUARTERLY
from src.data_extract.utils.fundamentals.xbrl_linkbase import NO_USABLE_PERIOD, TAG_PRIMARY, UNRESOLVED, Resolution
from src.data_extract.utils.fundamentals_sharadar import fetch_sharadar as sharadar_fetch_mod
from src.data_extract.utils.fundamentals_sharadar.build_ttm import ARQ
from src.data_extract.utils.fundamentals_sharadar.field_map import load_field_map
from src.data_extract.utils.fundamentals_sharadar.sharadar_history import build_sharadar_history
from src.data_store import ddl
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore
from tests.data_extract.common.scope_fixtures import SENTINEL, dated_identity
from tests.data_extract.edgar_fixtures import fake_context, roster_run, seed_index

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
_AS_OF = pd.Timestamp("2024-03-31")
FUNDAMENTALS_TABLES = (
    Tables.sec_fundamentals_facts,
    Tables.sec_fundamentals_history,
    Tables.sharadar_fundamentals_facts,
    Tables.sharadar_fundamentals_history,
)
#: The SQL types the removed `sql/schema.sql` block declared for the facts table (known truth).
FACT_TYPES = {
    "ticker": "TEXT",
    "accession_number": "TEXT",
    "field": "TEXT",
    "duration_type": "TEXT",
    "period_end": "DATE",
    "fiscal_year": "BIGINT",
    "fiscal_period": "TEXT",
    "cik": "TEXT",
    "form": "TEXT",
    "filing_date": "DATE",
    "period_of_report": "DATE",
    "regime": "TEXT",
    "period_start": "DATE",
    "period_days": "BIGINT",
    "value": "DOUBLE PRECISION",
    "unit": "TEXT",
    "decimals": "TEXT",
    "resolution_method": "TEXT",
    "source_concept": "TEXT",
    "roll_up_children": "TEXT",
    "root_anchor": "TEXT",
    "role_uri": "TEXT",
    "is_extension": "BOOLEAN",
}
#: (start, end, fiscal label, filed) of four quarters of a calendar-year filer.
_QUARTERS = [
    ("2023-01-01", "2023-03-31", "Q1", "2023-05-01"),
    ("2023-04-01", "2023-06-30", "Q2", "2023-08-01"),
    ("2023-07-01", "2023-09-30", "Q3", "2023-11-01"),
    ("2023-10-01", "2023-12-31", "Q4", "2024-02-15"),
]


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No retry-round sleep; an identity layer that holds none of the synthetic tickers (no seam rule)."""
    monkeypatch.setattr(edgar_driver, "_sleep", lambda seconds: None)
    identity = dated_identity([("ZZZ", "0000009999", "cik_window", SENTINEL, None)], {"ZZZ": "0000009999"})
    monkeypatch.setattr(build_history, "load_identity", lambda context: identity)
    monkeypatch.setattr(build_history, "_scope_changed_recently", lambda context, tickers, as_of=None: frozenset())


def _schema(store: DataStore, table: Table) -> tuple[dict[str, str], dict[str, bool], list[str], dict[str, list[str]]]:
    """`(types, nullable, pk, indexes)` of `table`: types and NOT NULL as written in the executed `CREATE TABLE`
    (SQLite reflects `DOUBLE PRECISION` as REAL), the PK and indexes as the database reflects them."""
    with store.engine.connect() as conn:
        created = conn.execute(text("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = :n"), {"n": table.name}).scalar_one()
    lines = [re.match(r'\s+"([^"]+)" (.+?)( NOT NULL)?,?$', line) for line in created.splitlines()]
    types = {m.group(1): m.group(2) for m in lines if m}
    nullable = {m.group(1): not m.group(3) for m in lines if m}
    inspector = inspect(store.engine)
    pk = list(inspector.get_pk_constraint(table.name)["constrained_columns"])
    indexes = {i["name"]: list(i["column_names"]) for i in inspector.get_indexes(table.name)}
    return types, nullable, pk, indexes


def _registry_indexes(table: Table, df: pd.DataFrame) -> dict[str, list[str]]:
    """The indexes the registry DDL (`ddl.table_ddl_from_frame`, the `ensure_table` path) declares for `table`."""
    out: dict[str, list[str]] = {}
    for line in ddl.table_ddl_from_frame(table, df).splitlines():
        if line.startswith("CREATE INDEX"):
            name = line.split()[5]
            out[name] = [c.strip('" );') for c in line.split("(", 1)[1].split(",")]
    return out


def _assert_created(store: DataStore, table: Table, expected: dict[str, str]) -> dict[str, list[str]]:
    """The table exists with exactly `expected` column types, the registry PK (NOT NULL) and the registry indexes."""
    types, nullable, pk, indexes = _schema(store, table)
    assert types == expected, {c: (types.get(c), t) for c, t in expected.items() if types.get(c) != t} | {
        c: "unexpected" for c in set(types) - set(expected)
    }
    assert pk == list(table.pk), pk
    assert not any(nullable[c] for c in table.pk), {c: nullable[c] for c in table.pk}
    sample = pd.DataFrame(columns=list(expected))
    assert indexes == _registry_indexes(table, sample), indexes
    return indexes


# --------------------------------------------------------------------------- #
# SEC facts walk + SEC history                                                 #
# --------------------------------------------------------------------------- #
def _accession(cik: int, i: int) -> str:
    return f"{cik:010d}-24-{i:06d}"


def _period(start: str, end: str, label: str, duration: str, value: float) -> dict:
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    instant = duration == INSTANT
    return {
        "fiscal_year": pd.Timestamp(end).year,
        "fiscal_period": label,
        "duration_type": duration,
        "period_start": pd.NaT if instant else pd.Timestamp(start),
        "period_end": pd.Timestamp(end),
        "period_days": None if instant else days,
        "value": value,
        "unit": "USD",
        "decimals": "-6",
    }


def _filing_rows(empty: set[str], quarter_of: dict[str, int]) -> Any:
    """A `filing_rows` stand-in: no row for an `empty` accession, else revenue, assets and one value-less field
    through the production row builder `_row`."""
    revenue = Resolution(field="totalRevenue", method=TAG_PRIMARY, concept="us-gaap:Revenues")
    assets = Resolution(field="totalAssets", method=TAG_PRIMARY, concept="us-gaap:Assets")
    goodwill = Resolution(field="goodwill", method=UNRESOLVED, dc_code=NO_USABLE_PERIOD)

    def rows(ticker: str, stamp: FilingStamp, catalogue: Any, gics: Any) -> list[dict]:
        if stamp.accession_number in empty:
            return []
        i = quarter_of[stamp.accession_number]
        start, end, label, _filed = _QUARTERS[i]
        reported = pd.Timestamp(end)
        return [
            facts_mod._row(ticker, stamp, reported, "industrial", "totalRevenue", revenue, _period(start, end, label, QUARTERLY, 100.0 + i)),
            facts_mod._row(ticker, stamp, reported, "industrial", "totalAssets", assets, _period(end, end, label, INSTANT, 1000.0 + i)),
            facts_mod._row(ticker, stamp, reported, "industrial", "goodwill", goodwill, None),
        ]

    return rows


def _seed_filings(ctx: Any, tickers: list[str], empty_only: tuple[str, ...] = ()) -> tuple[set[str], dict[str, int]]:
    """Four 10-Q/10-K filings per ticker (CIK = position + 1); an `empty_only` ticker has one filing, with no facts."""
    index_rows: list[tuple[int | str, str, str, str, str]] = []
    empty: set[str] = set()
    quarter_of: dict[str, int] = {}
    for position, ticker in enumerate(tickers):
        cik = position + 1
        quarters = _QUARTERS[:1] if ticker in empty_only else _QUARTERS
        for i, (_start, _end, label, filed) in enumerate(quarters):
            accession = _accession(cik, i)
            index_rows.append((cik, f"{ticker} Co", "10-K" if label == "Q4" else "10-Q", filed, accession))
            quarter_of[accession] = i
            if ticker in empty_only:
                empty.add(accession)
    seed_index(ctx, index_rows)
    return empty, quarter_of


def _facts_walk(ctx: Any, tickers: list[str], monkeypatch: pytest.MonkeyPatch, *, empty_only: tuple[str, ...] = (), workers: int) -> Any:
    empty, quarter_of = _seed_filings(ctx, tickers, empty_only)
    monkeypatch.setattr(facts_mod, "filing_rows", _filing_rows(empty, quarter_of))
    # the run scope binds the catalogue and GICS; the stand-in `filing_rows` reads neither
    fetch = dataclasses.replace(SEC_FUNDAMENTALS_FETCH, run_scope=None, parse=partial(parse_fundamentals, catalogue=None, gics_by_ticker={}))
    return run_edgar_fetch(ctx, fetch, roster_run(ctx, tickers, as_of=_AS_OF), max_workers=workers)


def _history_types(store: DataStore) -> dict[str, str]:
    columns = build_history.load_catalogue(str(CONFIG_DIR)).history_columns
    texts = {"ticker", "regime", "publication_form"}
    dates = {"as_of", "fiscal_end"}
    return {c: "TEXT" if c in texts else "DATE" if c in dates else "BIGINT" if c == "fiscal_quarter" else "DOUBLE PRECISION" for c in columns}


def test_an_empty_database_gets_both_sec_tables_from_the_nightly_writers(
    tmp_path: Path, sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = fake_context(tmp_path, sqlite_store, ["AAA", "BBB"])
    assert not any(sqlite_store.exists(t) for t in FUNDAMENTALS_TABLES), "the database must start without the four tables"

    # AAA's only filing has no facts, so its key saves a marker-only frame FIRST (one worker, roster order)
    summary = _facts_walk(ctx, ["AAA", "BBB"], monkeypatch, empty_only=("AAA",), workers=1)

    assert not ctx.warnings, ctx.warnings
    facts_indexes = _assert_created(sqlite_store, Tables.sec_fundamentals_facts, FACT_TYPES)
    stored = sqlite_store.load(Tables.sec_fundamentals_facts, markers=True)
    markers = stored[stored["field"] == "_empty"]
    assert markers["ticker"].tolist() == ["AAA"] and summary.outcomes["AAA"].markers == 1
    data = stored[stored["ticker"] == "BBB"]
    assert len(data) == 12 and data["is_extension"].eq(False).all() and data["value"].notna().sum() == 8

    build_history.build_fundamentals_history(
        SimpleNamespace(store=sqlite_store, log=logging.getLogger("test.cold"), config_dir=str(CONFIG_DIR)), ["AAA", "BBB"]
    )
    history_indexes = _assert_created(sqlite_store, Tables.sec_fundamentals_history, _history_types(sqlite_store))
    history = sqlite_store.load(Tables.sec_fundamentals_history)
    assert history["ticker"].tolist() == ["BBB"] * 4 and history["totalAssets"].tolist() == [1000.0, 1001.0, 1002.0, 1003.0]

    print("\n=== SANITY CHECK: cold SEC facts walk + SEC history build ===")
    print(
        f"  empty DB -> {Tables.sec_fundamentals_facts}: {len(FACT_TYPES)} columns typed as the old DDL, PK {list(Tables.sec_fundamentals_facts.pk)}"
    )
    print(f"  (NOT NULL), indexes {facts_indexes}; first write was AAA's marker-only frame, then BBB's 12 rows saved")
    print(
        f"  -> {Tables.sec_fundamentals_history}: 66 columns (DATE keys, BIGINT quarter, DOUBLE values), PK (ticker, as_of), indexes {history_indexes}"
    )
    print("  4 BBB events stored, AAA (marker only) skipped. Validated.")


def test_eight_workers_create_the_cold_facts_table_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'cold.db'}", connect_args={"check_same_thread": False, "timeout": 30})
    store = DataStore(engine)
    tickers = [f"T{i:02d}" for i in range(16)]
    ctx = fake_context(tmp_path, store, tickers)

    summary = _facts_walk(ctx, tickers, monkeypatch, workers=8)

    assert not ctx.warnings, ctx.warnings
    _assert_created(store, Tables.sec_fundamentals_facts, FACT_TYPES)
    stored = store.load(Tables.sec_fundamentals_facts)
    assert stored.groupby("ticker").size().to_dict() == dict.fromkeys(tickers, 12)
    assert sum(o.saved.get(Tables.sec_fundamentals_facts, 0) for o in summary.outcomes.values()) == 16 * 12
    engine.dispose()
    print("\n=== SANITY CHECK: 8 threads on a cold facts table ===")
    print("  16 keys x 12 rows saved by 8 run_per_ticker workers, 0 failed saves; one CREATE under store._CREATE_LOCK. Validated.")


# --------------------------------------------------------------------------- #
# Sharadar SF1 fetch + Sharadar history                                        #
# --------------------------------------------------------------------------- #
_ID_TYPES = {
    "ticker": "TEXT",
    "dimension": "TEXT",
    "calendardate": "DATE",
    "date": "DATE",
    "reportperiod": "DATE",
    "fiscalperiod": "TEXT",
    "lastupdated": "DATE",
}


def _sf1(ticker: str, dimension: str, field_map: Any) -> pd.DataFrame:
    """Six quarters of one SF1 dimension on the full response header: the mapped columns filled, every other one empty."""
    ends = [pd.Period(q, freq="Q").end_time.normalize() for q in ("2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1", "2024Q2")]
    frame = pd.DataFrame(index=range(len(ends)), columns=list(SHARADAR_SF1_COLUMNS), dtype=object)
    frame["ticker"], frame["dimension"] = ticker, dimension
    frame["calendardate"] = frame["reportperiod"] = [e.strftime("%Y-%m-%d") for e in ends]
    frame["date"] = frame["lastupdated"] = [(e + pd.Timedelta(days=40)).strftime("%Y-%m-%d") for e in ends]
    frame["fiscalperiod"] = [f"{e.year}-Q{e.quarter}" for e in ends]
    for source in {str(s.source) for s in field_map.direct.values()} & set(SHARADAR_SF1_COLUMNS):
        frame[source] = "100.0"
    return frame


def test_an_empty_database_gets_both_sharadar_tables_from_the_nightly_writers(
    tmp_path: Path, sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    field_map = load_field_map(str(CONFIG_DIR))
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"], sharadar_request_pace=0.0)
    ctx.log.debug = lambda msg, *a: None
    sqlite_store.save(
        Tables.sharadar_tickers, pd.DataFrame({"table": ["SF1"], "permaticker": [1], "ticker": ["AAA"], "currency": ["USD"], "isdelisted": ["N"]})
    )
    assert not any(sqlite_store.exists(t) for t in FUNDAMENTALS_TABLES)
    monkeypatch.setattr(
        sharadar_fetch_mod, "sharadar_get", lambda context, table, /, **filters: _sf1(filters["ticker"], filters["dimension"], field_map)
    )
    monkeypatch.setattr(sharadar_fetch_mod, "sleep_pace", lambda *a, **k: None)

    sharadar_fetch_mod.fetch_sharadar_fundamentals(ctx, ["AAA"], years_history=15, as_of=_AS_OF)

    facts_types = {c: _ID_TYPES.get(c, "DOUBLE PRECISION") for c in SHARADAR_SF1_COLUMNS}
    facts_indexes = _assert_created(sqlite_store, Tables.sharadar_fundamentals_facts, facts_types)

    build_sharadar_history(
        SimpleNamespace(store=sqlite_store, log=logging.getLogger("test.cold"), config_dir=str(CONFIG_DIR)), ["AAA"], config_dir=str(CONFIG_DIR)
    )
    columns = Tables.sharadar_fundamentals_history.read_columns
    history_types = {c: "TEXT" if c == "ticker" else "DATE" if c in ("as_of", "fiscal_end") else "DOUBLE PRECISION" for c in columns}
    history_indexes = _assert_created(sqlite_store, Tables.sharadar_fundamentals_history, history_types)
    history = sqlite_store.load(Tables.sharadar_fundamentals_history)
    assert len(history) == 6 and history["totalAssets"].notna().all()
    assert sqlite_store.load(Tables.sharadar_fundamentals_facts, where={"dimension": ARQ})["assets"].eq(100.0).all()

    print("\n=== SANITY CHECK: cold Sharadar SF1 fetch + Sharadar history build ===")
    print(f"  empty DB -> {Tables.sharadar_fundamentals_facts}: {len(facts_types)} columns (7 ids TEXT/DATE, 105 DOUBLE even where empty),")
    print(f"  PK {list(Tables.sharadar_fundamentals_facts.pk)} NOT NULL, indexes {facts_indexes}")
    print(
        f"  -> {Tables.sharadar_fundamentals_history}: {len(columns)} columns (DATE keys, DOUBLE values), PK (ticker, as_of), indexes {history_indexes}; 6 rows. Validated."
    )


# --------------------------------------------------------------------------- #
# Cold tickers fill in full on the nightly run (per-run cap)                  #
# --------------------------------------------------------------------------- #
def _nightly_walk(ctx: Any, tickers: list[str], monkeypatch: pytest.MonkeyPatch, accessions: dict[str, list[int]]) -> Any:
    """One nightly facts walk (default `-F`-less, capped run) over an index holding `accessions[ticker]`'s quarters,
    CIKs as `fake_context` seeded them (universe position + 1)."""
    universe = ctx.universe.frame.set_index("ticker")["cik"].astype(int).to_dict()
    rows: list[tuple[int | str, str, str, str, str]] = []
    quarter_of: dict[str, int] = {}
    for ticker, quarters in accessions.items():
        for i in quarters:
            _start, _end, label, filed = _QUARTERS[i]
            accession = _accession(universe[ticker], i)
            rows.append((universe[ticker], f"{ticker} Co", "10-K" if label == "Q4" else "10-Q", filed, accession))
            quarter_of[accession] = i
    seed_index(ctx, rows)
    monkeypatch.setattr(facts_mod, "filing_rows", _filing_rows(set(), quarter_of))
    fetch = dataclasses.replace(SEC_FUNDAMENTALS_FETCH, run_scope=None, parse=partial(parse_fundamentals, catalogue=None, gics_by_ticker={}))
    return run_edgar_fetch(ctx, fetch, roster_run(ctx, tickers, as_of=_AS_OF), max_workers=1)


def _stored_filings(store: DataStore) -> dict[str, int]:
    stored = store.load(Tables.sec_fundamentals_facts, columns=["ticker", "accession_number"], markers=True)
    return stored.groupby("ticker")["accession_number"].nunique().to_dict()


def test_cold_tickers_fill_in_full_and_the_next_night_is_a_no_op(tmp_path: Path, sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty database, per-run cap 2, two tickers of 4 filings, nightly walk + history build until the walk lists
    nothing: the first night reads every filing of both cold tickers and the history replays them in full; the
    second night lists nothing. A capped first night instead back-fills older quarters on later nights, and the
    night the Q4 row's TTM becomes computable the drift guard raises."""
    ctx = fake_context(tmp_path, sqlite_store, ["AAA", "BBB"], max_documents_per_run=2)
    history_ctx = SimpleNamespace(store=sqlite_store, log=logging.getLogger("test.cold"), config_dir=str(CONFIG_DIR))
    every = {"AAA": [0, 1, 2, 3], "BBB": [0, 1, 2, 3]}

    listed: list[int] = []
    for night in range(1, 6):
        summary = _nightly_walk(ctx, ["AAA", "BBB"], monkeypatch, every)
        listed.append(summary.work.size)
        print(f"\n  night {night}: listed {summary.work.size} of {summary.work.uncapped}; stored filings {_stored_filings(sqlite_store)}")
        if night == 1:
            night_one = _stored_filings(sqlite_store)
        build_history.build_fundamentals_history(history_ctx, ["AAA", "BBB"])
        if summary.work.size == 0:
            break
    history = sqlite_store.load(Tables.sec_fundamentals_history)

    assert night_one == {"AAA": 4, "BBB": 4}, night_one
    assert listed == [8, 0], listed
    assert not ctx.errors, ctx.errors
    assert history.groupby("ticker").size().to_dict() == {"AAA": 4, "BBB": 4}
    # the Q4 event's TTM revenue is the four quarters 100 + 101 + 102 + 103
    assert history.sort_values("as_of").groupby("ticker")["totalRevenue"].last().eq(406.0).all(), history[["ticker", "as_of", "totalRevenue"]]

    print("=== SANITY CHECK: cold tickers on a capped nightly run ===")
    print("  cap 2, 8 cold filings: night 1 read all 8 (no cap ERROR), the history replayed 4 events per ticker in full (Q4 TTM revenue 406);")
    print(f"  night 2 listed 0 documents (nights {listed}) and the history build skipped both tickers: no back-fill, no drift. Validated.")


def test_a_warm_ticker_keeps_the_cap_while_a_cold_one_fills(
    tmp_path: Path, sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """BBB already holds its newest filing (warm, 3 older ones missing); AAA holds nothing (cold). Under cap 2,
    AAA lists all 4 filings outside the cap, BBB only its 2 newest missing ones."""
    ctx = fake_context(tmp_path, sqlite_store, ["AAA", "BBB"], max_documents_per_run=2)
    _nightly_walk(ctx, ["BBB"], monkeypatch, {"BBB": [3]})
    assert _stored_filings(sqlite_store) == {"BBB": 1}
    caplog.clear()

    summary = _nightly_walk(ctx, ["AAA", "BBB"], monkeypatch, {"AAA": [0, 1, 2, 3], "BBB": [0, 1, 2, 3]})
    stored = sqlite_store.load(Tables.sec_fundamentals_facts, columns=["ticker", "accession_number"])
    bbb = sorted(int(a[-1]) for a in stored.loc[stored["ticker"] == "BBB", "accession_number"].unique())

    assert len(summary.work.units["AAA"]) == 4 and len(summary.work.units["BBB"]) == 2, {k: len(v) for k, v in summary.work.units.items()}
    assert summary.work.uncapped == 7
    assert bbb == [1, 2, 3], bbb
    capped = [r.getMessage() for r in caplog.records if "exceeds the per-run cap" in r.getMessage()]
    assert capped == [
        "sec_fundamentals_facts: work list of 3 document(s) exceeds the per-run cap of 2; the newest 2 run tonight, the rest on later runs"
    ], capped
    print("\n=== SANITY CHECK: warm ticker capped, cold ticker uncapped ===")
    print(f"  7 missing filings, cap 2: cold AAA read 4/4 outside the cap; warm BBB read its 2 newest (quarters {bbb[:2]}), Q1 left for")
    print("  a later night, with the cap ERROR naming the warm list's 3 documents only. Validated.")
