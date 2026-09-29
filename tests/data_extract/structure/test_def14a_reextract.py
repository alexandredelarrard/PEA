"""Targeted DEF 14A remediation replaces only the named filing family."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from scripts import def14a_reextract as reextract
from src.data_extract.utils.structure.def14a.flatten import _CHILD_TABLES
from src.data_store.schema import Tables


def test_rejected_subject_backup_removes_parent_and_all_children(tmp_path, monkeypatch):
    rejected = "0001104659-26-016573"
    retained = "keep-me"
    tables = (Tables.def14a_llm, *_CHILD_TABLES.values())
    frames = {
        table.name: pd.DataFrame(
            [
                {"ticker": "PSKY", "accession_number": rejected, "marker": table.name},
                {"ticker": "PSKY", "accession_number": retained, "marker": table.name},
            ]
        )
        for table in tables
    }

    class Store:
        def load(self, table, *, where=None, optional=False):
            del optional
            frame = frames[table.name]
            wanted = set((where or {}).get("accession_number", []))
            return frame[frame["accession_number"].isin(wanted)].copy()

        def delete(self, table, where):
            frame = frames[table.name]
            wanted = set(where["accession_number"])
            dropped = frame["accession_number"].isin(wanted)
            frames[table.name] = frame[~dropped].reset_index(drop=True)
            return int(dropped.sum())

    monkeypatch.setattr(reextract, "BACKUP", tmp_path)
    context = SimpleNamespace(store=Store())

    first = reextract._remove_rejected_subjects(context, {rejected})
    second = reextract._remove_rejected_subjects(context, {rejected})

    assert first == len(tables) and second == 0
    assert len(list(tmp_path.glob("*.csv"))) == len(tables)
    for table in tables:
        assert frames[table.name]["accession_number"].tolist() == [retained]

    print("\n=== SANITY: wrong-subject DEF 14A family replacement ===")
    print(f"  backed up and removed one parent plus {len(tables) - 1} child families")
    print("  unrelated accession retained; second removal is an idempotent no-op")


def test_target_tasks_use_the_production_registrant_walk(monkeypatch):
    accession = "0000053669-15-000001"
    filing = pd.DataFrame([{"accession_number": accession, "cik": "0000053669", "filing_date": "2015-01-01"}])
    cutovers = {"JCI": object()}
    seen = {}

    def list_across(context, ticker, cik, company, years, since, registrations):
        seen.update(
            context=context,
            ticker=ticker,
            cik=cik,
            company=company,
            years=years,
            since=since,
            registrations=registrations,
        )
        return filing

    monkeypatch.setattr(reextract, "_list_across_registrants", list_across)
    monkeypatch.setattr(reextract, "_subject_is_accepted", lambda *args: True)
    monkeypatch.setattr(reextract, "_payload_for", lambda *args: "payload")
    context = SimpleNamespace(config=SimpleNamespace(data_extract=SimpleNamespace(years_history=31)))

    tasks, rejected, unresolved = reextract._tasks_for_ticker(
        context,
        "JCI",
        "0000833444",
        "Johnson Controls",
        {accession},
        frozenset({"0000053669", "0000833444"}),
        cutovers,
    )

    assert len(tasks) == 1 and tasks[0][1]["filing"]["cik"] == "0000053669"
    assert not rejected and not unresolved
    assert seen["registrations"] is cutovers and seen["since"] is None
    print("\n=== SANITY: targeted DEF 14A follows dated registrant history ===")
    print("  current JCI roster CIK -> predecessor-CIK accession found by production's segment walk")


def test_missing_or_unreadable_targets_are_unresolved(monkeypatch):
    listed = "0000053669-15-000001"
    absent = "0000053669-14-000001"
    filing = pd.DataFrame([{"accession_number": listed, "cik": "0000053669", "filing_date": "2015-01-01"}])
    monkeypatch.setattr(reextract, "_list_across_registrants", lambda *args: filing)
    monkeypatch.setattr(reextract, "_subject_is_accepted", lambda *args: True)
    monkeypatch.setattr(reextract, "_payload_for", lambda *args: None)
    context = SimpleNamespace(config=SimpleNamespace(data_extract=SimpleNamespace(years_history=31)))

    tasks, rejected, unresolved = reextract._tasks_for_ticker(
        context,
        "JCI",
        "0000833444",
        "Johnson Controls",
        {listed, absent},
        frozenset({"0000053669", "0000833444"}),
        {},
    )

    assert not tasks and not rejected
    assert unresolved == {listed, absent}
    print("\n=== SANITY: targeted DEF 14A cannot silently omit work ===")
    print("  absent listing + unreadable payload are both surfaced as unresolved failures")
