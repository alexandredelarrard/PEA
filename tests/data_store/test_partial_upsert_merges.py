"""Does `save` MERGE a partial frame, or blank the columns it omits?

This is a data-destruction test, not a style test. `upsert_dataframe` used to build its
`ON CONFLICT DO UPDATE SET` list from the TABLE's columns:

    update_cols = {c.name: ins.excluded[c.name] for c in tbl.c if c.name not in pk}

`excluded.<col>` for a column that is not in the INSERT resolves to that column's DEFAULT, so
every column the frame omitted was set to NULL. Measured cost when it fired: the DEF 14A
gender-consensus pass writes a 4-column patch (`ticker`, `accession_number`,
`pct_female_directors`, `pct_gender_stated`) over `def14a_llm`, and it nulled **392 of 409
rows** — including `def14a_json`, the only copy of the LLM answers those filings were
extracted from. 395 paid calls, unreplayable.

Narrow writes are the normal shape here, not an exotic one: `_finalise_gender` has two of
them and says so in its docstring ("both writes carry only the columns they change"). So the
merge is the contract, and this pins it on both dialects the repo runs on.

SQLite-backed and self-contained: no live DB, no network. SQLite takes the same
`on_conflict_do_update` branch as Postgres, so it exercises the fixed line.
"""
from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.data_store.store import DataStore

#: Two rows with every column populated -- the "already extracted" state.
_FULL = pd.DataFrame({
    "ticker": ["AAPL", "JPM"],
    "accession_number": ["a1", "a2"],
    "payload": ["{...aapl...}", "{...jpm...}"],
    "n_directors": [9.0, 11.0],
    "pct_female_directors": [0.33, 0.45],
})


@pytest.fixture()
def store() -> DataStore:
    engine = create_engine("sqlite://")
    _FULL.to_sql("extract", engine, index=False)
    with engine.begin() as conn:
        conn.exec_driver_sql(
            'CREATE UNIQUE INDEX ux ON "extract" ("ticker", "accession_number")')
    return DataStore(engine)


def test_a_partial_upsert_keeps_the_columns_it_does_not_carry(store):
    """The regression itself: patch two columns, keep the other two."""
    patch = pd.DataFrame({
        "ticker": ["AAPL"], "accession_number": ["a1"], "pct_female_directors": [0.40],
    })
    store.save("extract", patch, pk=["ticker", "accession_number"])

    row = store.load("extract", where={"ticker": "AAPL"}).iloc[0]
    print("\n=== SANITY CHECK: partial upsert merges ===")
    print(f"  after a 3-column patch: payload={row['payload']!r} "
          f"n_directors={row['n_directors']}")
    assert row["pct_female_directors"] == pytest.approx(0.40), "the patch did not apply"
    assert row["payload"] == "{...aapl...}", \
        "payload was blanked -- this is the def14a_json loss, 392 of 409 rows"
    assert row["n_directors"] == pytest.approx(9.0), "n_directors was blanked"


def test_the_untouched_row_is_untouched(store):
    """A patch keyed to one row must not reach the other."""
    patch = pd.DataFrame({
        "ticker": ["AAPL"], "accession_number": ["a1"], "pct_female_directors": [0.40],
    })
    store.save("extract", patch, pk=["ticker", "accession_number"])
    jpm = store.load("extract", where={"ticker": "JPM"}).iloc[0]
    assert jpm["payload"] == "{...jpm...}"
    assert jpm["pct_female_directors"] == pytest.approx(0.45)


def test_a_full_frame_still_overwrites_every_column(store):
    """The merge must not turn into "never overwrite": a full re-extraction has to replace
    the stored row, which is what the per-ticker DEF 14A save relies on."""
    fresh = pd.DataFrame({
        "ticker": ["AAPL"], "accession_number": ["a1"], "payload": ["{...v2...}"],
        "n_directors": [10.0], "pct_female_directors": [0.50],
    })
    store.save("extract", fresh, pk=["ticker", "accession_number"])
    row = store.load("extract", where={"ticker": "AAPL"}).iloc[0]
    assert row["payload"] == "{...v2...}"
    assert row["n_directors"] == pytest.approx(10.0)


def test_an_explicit_null_still_blanks_the_column(store):
    """Merging is about columns the frame OMITS. A column the frame carries as null is a
    deliberate blanking and must still land, or a corrected extract could never remove a
    value it no longer believes."""
    patch = pd.DataFrame({
        "ticker": ["AAPL"], "accession_number": ["a1"], "n_directors": [None],
    })
    store.save("extract", patch, pk=["ticker", "accession_number"])
    row = store.load("extract", where={"ticker": "AAPL"}).iloc[0]
    assert pd.isna(row["n_directors"]), "an explicit null must still blank the column"
    assert row["payload"] == "{...aapl...}", "omitted columns must still be kept"


def test_an_insert_of_a_new_key_is_unaffected(store):
    """No conflict, so no update list -- the new row lands with nulls where the frame is
    silent, which is correct: there was nothing to preserve."""
    new = pd.DataFrame({
        "ticker": ["MSFT"], "accession_number": ["a3"], "pct_female_directors": [0.38],
    })
    store.save("extract", new, pk=["ticker", "accession_number"])
    row = store.load("extract", where={"ticker": "MSFT"}).iloc[0]
    assert row["pct_female_directors"] == pytest.approx(0.38)
    assert pd.isna(row["payload"])
    assert len(store.load("extract")) == 3
