"""Publish stage (phase 4d): the only stage that saves the published tables (REQ-008).

A static scan proves no other identity stage writes them; a SQLite store whose `universe_tickers` is created from
`sql/schema.sql` (cik and universe_added_on NOT NULL, X19) proves the identity columns are written in place, the
sync-owned columns and a removed ticker's row (X17) are left untouched, and a second publish writes 0 rows (AC-014).
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from src.data_extract.transformers.step_extract_identity import StepExtractIdentity
from src.data_extract.utils.identity.cik_history import BACKLOG_COLUMNS, BLOCKED_COLUMNS
from src.data_extract.utils.identity.cik_history import TABLE_COLUMNS as LINEAGE_COLUMNS
from src.data_extract.utils.identity.cusip_history import TABLE_COLUMNS as CUSIP_COLUMNS
from src.data_extract.utils.identity.lifecycle import TABLE_COLUMNS as LIFE_COLUMNS
from src.data_extract.utils.identity.lifecycle import Lifecycle
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.identity_flags import FLAG_COLUMNS
from src.utils.tickers_universe.universe import Universe
from tests.canonical_schema import create_canonical
from tests.support.universe import universe_config

ROOT = Path(__file__).resolve().parents[3]
STEP = ROOT / "src" / "data_extract" / "transformers" / "step_extract_identity.py"
STAGES = ROOT / "src" / "data_extract" / "utils" / "identity"
PUBLISHED = {"universe_cik_symbol", "universe_cusip", "universe_tickers"}
WRITERS = {"save", "replace", "update", "delete", "insert", "bulk_seed", "append_tail", "save_identity", "save_unless_unchanged"}
SYNC_COLUMNS = ["ticker", "name", "cik", "sector", "universe_added_on", "permaticker", "in_universe", "cusips"]


def _published_writes(path: Path) -> set[tuple[str, str]]:
    """`(function, table)` for every writer call naming a published `Tables.<x>` in `path`."""
    out = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [(f"{c.name}.{f.name}", f) for c in tree.body if isinstance(c, ast.ClassDef) for f in c.body if isinstance(f, ast.FunctionDef)]
    functions += [(f.name, f) for f in tree.body if isinstance(f, ast.FunctionDef)]
    for name, function in functions:
        for call in (n for n in ast.walk(function) if isinstance(n, ast.Call)):
            writer = call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", "")
            tables = [a.attr for a in call.args[:2] if isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name) and a.value.id == "Tables"]
            out |= {(f"{path.stem}.{name}", table) for table in tables if writer in WRITERS and table in PUBLISHED}
    return out


def test_only_run_publish_saves_the_published_tables() -> None:
    """REQ-008: every write of a published table sits in `run_publish` or the publish helper it alone calls."""
    writes = set().union(*(_published_writes(path) for path in [STEP, *sorted(STAGES.glob("*.py"))]))
    expected = {
        ("step_extract_identity.StepExtractIdentity.run_publish", "universe_cik_symbol"),
        ("step_extract_identity.StepExtractIdentity.run_publish", "universe_cusip"),
        ("publish.update_identity_columns", "universe_tickers"),
    }
    callers = [
        f.name
        for c in ast.parse(STEP.read_text(encoding="utf-8")).body
        if isinstance(c, ast.ClassDef)
        for f in c.body
        if isinstance(f, ast.FunctionDef) and "update_identity_columns(" in ast.unparse(f)
    ]

    assert writes == expected, sorted(writes)
    assert callers == ["run_publish"], callers
    print("\n=== SANITY CHECK: only run_publish saves (REQ-008) ===")
    for function, table in sorted(writes):
        print(f"  {table:<20} written by {function}")
    print("  OK: the symbol, CUSIP and lifecycle stages compute only; update_identity_columns is called from run_publish alone")


def _lineage() -> pd.DataFrame:
    """AAA's roster window and its open symbol interval, typed as the symbol history builds them."""
    rows = [
        ("E0000000100", "AAA", "0000000100", "cik_window", "", "1900-01-01", "roster", "roster"),
        ("E0000000100", "AAA", "0000000100", "symbol", "AAA", "2006-01-03", "form345,roster", "roster"),
    ]
    frame = pd.DataFrame(rows, columns=["company_id", "canonical_ticker", "cik", "role", "symbol", "valid_from", "sources", "oracle"])
    return frame.assign(
        valid_from=pd.to_datetime(frame["valid_from"]),
        valid_to=pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]"),
        status="curated",
        confidence=float("nan"),
        n_observations=3,
        evidence="",
        scope_changed_at=pd.Timestamp("2026-11-01"),
    )[list(LINEAGE_COLUMNS)]


def _lines() -> pd.DataFrame:
    """AAA's one canonical CUSIP line."""
    row = dict.fromkeys(CUSIP_COLUMNS)
    row.update(
        security_id="C123456101", canonical_company="AAA", issuer_cik="0000000100", source="ftd", source_symbol="AAA", market_symbol="AAA",
        cusip="123456101", security_class="common", conversion_ratio=1.0, lineage_role="canonical_current", valid_from=pd.Timestamp("2009-07-01"),
        lineage_reason="", evidence="", n_observations=10, scope_changed_at=pd.Timestamp("2026-11-01"),
    )  # fmt: skip
    return pd.DataFrame([row])[list(CUSIP_COLUMNS)]


def _step(store: DataStore) -> Any:
    """A publish step over `store` with this run's builds preset (AAA only in the universe)."""
    log = logging.getLogger("test.identity.publish")
    context = SimpleNamespace(store=store, log=log, config_dir=str(ROOT / "configs"), universe=Universe(store, universe_config(["AAA"])))
    life = pd.DataFrame([("AAA", "E0000000100", "AAA", pd.Timestamp("2000-01-03"), pd.NaT, "active")], columns=list(LIFE_COLUMNS))
    step = cast(Any, object.__new__(StepExtractIdentity))
    step._context, step._config_dir, step._log = context, context.config_dir, log
    step._symbols = SimpleNamespace(
        rows=_lineage(), backlog=pd.DataFrame(columns=list(BACKLOG_COLUMNS)), blocked=pd.DataFrame(columns=list(BLOCKED_COLUMNS))
    )
    step._cik_history = SimpleNamespace(activity=pd.DataFrame(columns=["cik", "source", "first", "last"]))
    step._cusips = SimpleNamespace(rows=_lines(), items=pd.DataFrame(columns=list(FLAG_COLUMNS)))
    step._lifecycle = Lifecycle(rows=life, items=pd.DataFrame(columns=list(FLAG_COLUMNS)))
    return step


@pytest.fixture
def published_store(sqlite_store: DataStore) -> DataStore:
    """`universe_tickers` from `sql/schema.sql` holding AAA (in the universe, no identity yet) and GEHC (removed)."""
    tables = [Tables.universe_tickers, Tables.universe_cik_symbol, Tables.universe_cusip, Tables.symbol_tenure, Tables.insider_transactions]
    create_canonical(sqlite_store.engine, tables)
    rows = pd.DataFrame(
        [
            ("AAA", "AAA Corp", "0000000100", "Tech", pd.Timestamp("2000-01-01"), 1, True, "123456101", None, None, None, None, None),
            ("GEHC", "GE HealthCare", "0001932393", "Health", pd.Timestamp("2023-01-04"), 2, False, None, "E0001932393", "GEHC", pd.Timestamp("2023-01-04"), None, "active"),
        ],
        columns=[*SYNC_COLUMNS, "company_id", "trading_symbol", "first_price_date", "ended_on", "status"],
    )  # fmt: skip
    sqlite_store.save(Tables.universe_tickers, rows)
    return sqlite_store


def test_publish_writes_identity_columns_in_place_and_a_second_run_writes_nothing(published_store: DataStore) -> None:
    """X19 / AC-014: a NOT NULL `cik` table rejects an identity-only upsert; publish updates AAA's identity columns only,
    leaves GEHC's row and every sync column as stored, adds `company_id` to the CUSIP lines; the second run writes 0."""
    store = published_store
    before = store.load(Tables.universe_tickers).sort_values("ticker", ignore_index=True)
    with pytest.raises(Exception, match="NOT NULL"):
        store.save(Tables.universe_tickers, pd.DataFrame({"ticker": ["AAA"], "status": ["active"]}))

    first = _step(store).run_publish()
    second = _step(store).run_publish()

    after = store.load(Tables.universe_tickers).sort_values("ticker", ignore_index=True)
    lines = store.load(Tables.universe_cusip)
    aaa = after[after["ticker"].eq("AAA")].iloc[0]
    assert after[SYNC_COLUMNS].astype(str).equals(before[SYNC_COLUMNS].astype(str)), (before, after)
    assert after[after["ticker"].eq("GEHC")].astype(str).equals(before[before["ticker"].eq("GEHC")].astype(str))
    assert (aaa["company_id"], aaa["trading_symbol"], aaa["status"]) == ("E0000000100", "AAA", "active") and pd.isna(aaa["ended_on"]), aaa
    assert pd.Timestamp(aaa["first_price_date"]) == pd.Timestamp("2000-01-03"), aaa
    assert lines["company_id"].tolist() == ["E0000000100"] and len(store.load(Tables.universe_cik_symbol)) == 2
    assert first == 2 + 1 + 1 and second == 0, (first, second)
    print("\n=== SANITY CHECK: publish on a schema.sql universe_tickers (X19, AC-014) ===")
    print("  an identity-only upsert fails NOT NULL on cik: the publish uses store.update")
    print(f"  run 1 wrote {first} row(s): universe_cik_symbol 2, universe_cusip 1 (company_id E0000000100), AAA identity columns 1")
    print(f"  run 2 wrote {second} row(s); sync columns {SYNC_COLUMNS} unchanged; removed GEHC row unchanged")


def test_a_publish_recreates_dropped_identity_tables(published_store: DataStore) -> None:
    """X87 / U-87: with `universe_cik_symbol` and `universe_cusip` dropped, one publish recreates them inside its transaction."""
    store = published_store
    store.drop(Tables.universe_cik_symbol)
    store.drop(Tables.universe_cusip)
    assert not store.exists(Tables.universe_cik_symbol) and not store.exists(Tables.universe_cusip)

    written = _step(store).run_publish()

    assert written == 2 + 1 + 1, written
    assert len(store.load(Tables.universe_cik_symbol)) == 2 and store.load(Tables.universe_cusip)["company_id"].tolist() == ["E0000000100"]
    print("\n=== SANITY CHECK: a publish recreates the dropped identity tables (U-87) ===")
    print(f"  both tables dropped; one publish wrote {written} row(s): universe_cik_symbol 2, universe_cusip 1, AAA identity columns 1")
    print("  OK: no schema.sql step is needed before the first DAG run")
