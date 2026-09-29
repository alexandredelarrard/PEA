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
