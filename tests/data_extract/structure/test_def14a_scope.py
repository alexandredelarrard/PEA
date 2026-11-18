"""Proxy reconciliation uses shared configured dates and protects other source keys."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.structure.def14a import fetch, gender
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.window import WindowRun
from tests.canonical_schema import create_canonical
from tests.data_extract.edgar_fixtures import fake_context


def _world(tmp_path: Path, store: DataStore, monkeypatch: pytest.MonkeyPatch) -> Any:
    create_canonical(
        store.engine,
        [
            Tables.def14a_llm,
            Tables.def14a_directors,
            Tables.def14a_director_comp,
            Tables.def14a_executive_comp,
            Tables.def14a_ownership,
            Tables.sec_def14a_text,
        ],
    )
    context = fake_context(tmp_path, store, ["AAA"])
    context.run = WindowRun("incremental", "bounded proxy reconciliation", pd.Timestamp("2026-11-16"))
    for table in (Tables.def14a_llm, Tables.def14a_directors):
        item = context.config.data.tables[table.name]
        item.start_date, item.end_date = "2026-06-01", "2026-09-30"
    rows = [
        ("AAA", "inside", "Taylor Person", "2026-07-01", "male", "name"),
        ("AAA", "old", "Taylor Person", "2026-01-01", "male", "name"),
        ("AAA", "old", "Morgan Person", "2026-01-01", "female", "stated"),
        ("BBB", "evidence", "Taylor Person", "2026-06-01", "female", "stated"),
        ("BBB", "other", "Morgan Person", "2026-08-01", "male", "name"),
        ("AAA", "future", "Taylor Person", "2026-11-01", "male", "name"),
    ]
    directors = pd.DataFrame(rows, columns=gender._DIRECTOR_COLUMNS)
    directors["as_of"] = pd.to_datetime(directors.as_of)
    store.save(Tables.def14a_directors, directors)
    parents = gender.recompute_parent_gender(directors).merge(directors[["ticker", "accession_number", "as_of"]].drop_duplicates(), how="left")
    store.save(Tables.def14a_llm, parents)
    store.save(
        Tables.sec_def14a_text,
        pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "accession_number": "trigger",
                    "cik": "0000000001",
                    "filing_date": pd.Timestamp("2026-06-10"),
                    "form": "DEF 14A",
                    "section": "governance",
                    "payload": "Known source fixture",
                }
            ]
        ),
    )
    monkeypatch.setattr(fetch, "ticker_works", lambda *args: [])
    monkeypatch.setattr(fetch, "call_sections", lambda *args: 1)
    return context


def _snapshot(store: DataStore) -> dict[Any, pd.DataFrame]:
    return {table: store.load(table).sort_values(list(table.pk)).reset_index(drop=True) for table in (Tables.def14a_directors, Tables.def14a_llm)}


def test_gender_reconciliation_writes_only_authorized_dates_and_tickers(
    tmp_path: Path, sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _world(tmp_path, sqlite_store, monkeypatch)
    before = _snapshot(sqlite_store)
    fetch.fetch_def14a_llm(context, context.config, ["AAA"], {})
    after = _snapshot(sqlite_store)
    for table, original in before.items():
        protected = original.accession_number.ne("inside")
        pd.testing.assert_frame_equal(after[table].loc[protected].reset_index(drop=True), original.loc[protected].reset_index(drop=True))
    changed = after[Tables.def14a_directors].query("accession_number == 'inside'")
    assert changed.gender.tolist() == ["female"] and changed.gender_basis.tolist() == ["stated"]
    parent = after[Tables.def14a_llm].query("accession_number == 'inside'")
    assert parent.pct_female_directors.tolist() == [1.0] and parent.pct_gender_stated.tolist() == [1.0]
    print(
        "SANITY: cross-company evidence corrects the authorized proxy; older/future filings and the other ticker retain every stored cell and parent ratio."
    )


def test_misaligned_parent_repair_is_refused_before_any_write(tmp_path: Path, sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _world(tmp_path, sqlite_store, monkeypatch)
    context.config.data.tables.def14a_llm.end_date = "2026-06-30"
    before = _snapshot(sqlite_store)
    with pytest.raises(ValueError, match="parent.*Window"):
        fetch.fetch_def14a_llm(context, context.config, ["AAA"], {})
    after = _snapshot(sqlite_store)
    for table in before:
        pd.testing.assert_frame_equal(after[table], before[table])
    print("SANITY: a parent repair outside its configured Window is refused before changing either table.")


def test_parent_write_failure_rolls_back_the_authorized_director_update(
    tmp_path: Path, sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _world(tmp_path, sqlite_store, monkeypatch)
    before = _snapshot(sqlite_store)
    save = DataStore.save

    def fail_parent(self: DataStore, table: Any, frame: pd.DataFrame, *args: Any, **kwargs: Any) -> int:
        if table == Tables.def14a_llm:
            raise RuntimeError("parent write interrupted")
        return save(self, table, frame, *args, **kwargs)

    monkeypatch.setattr(DataStore, "save", fail_parent)
    with pytest.raises(RuntimeError, match="parent write interrupted"):
        fetch.fetch_def14a_llm(context, context.config, ["AAA"], {})
    after = _snapshot(sqlite_store)
    for table in before:
        pd.testing.assert_frame_equal(after[table], before[table])
    print("SANITY: interruption of the paired parent save rolls back the director update; both source tables retain their complete prior state.")
