"""`scripts/def14a_legacy_purge.py` never selects an empty-filing marker: a marker owns no director
row and carries no new-schema column, so without the store's marker filter it would fall in the
delete set and its proxy would be sent to the LLM again."""

from __future__ import annotations

import pandas as pd

from scripts.def14a_legacy_purge import NEW_SCHEMA_COLUMNS, legacy_accessions
from src.data_extract.utils.common.empty_markers import marker_frame
from src.data_store.schema import Tables


def test_the_delete_set_never_holds_a_marker(sqlite_store) -> None:
    blank = dict.fromkeys(NEW_SCHEMA_COLUMNS, float("nan"))
    legacy = {**blank, "ticker": "AAA", "accession_number": "legacy-1", "as_of": "2020-04-01", "ceo_name_proxy": "Old Prompt", "def14a_json": "{}"}
    current = {
        **blank,
        "ticker": "AAA",
        "accession_number": "new-1",
        "as_of": "2025-04-01",
        "ceo_name_proxy": "New Prompt",
        "def14a_json": "{}",
        "sct_years": 3.0,
    }
    sqlite_store.save(Tables.def14a_llm, pd.DataFrame([legacy, current]))
    marker = marker_frame(Tables.def14a_llm, {"ticker": "AAA", "accession_number": "marker-1", "as_of": pd.Timestamp("2024-04-01")})
    marker["as_of"] = marker["as_of"].dt.strftime("%Y-%m-%d")
    sqlite_store.save(Tables.def14a_llm, marker)
    sqlite_store.save(Tables.def14a_directors, pd.DataFrame([{"ticker": "AAA", "accession_number": "new-1", "name": "Jane", "as_of": "2025-04-01"}]))

    target, _, evidence = legacy_accessions(sqlite_store)

    assert target == {"legacy-1"}
    assert evidence["parent_rows"] == 2
    print("\n=== SANITY CHECK: legacy purge and markers ===")
    print(f"  3 parent rows stored (1 legacy, 1 current, 1 marker) -> delete set {sorted(target)}; the marker is never read.")
