"""Never mark over data: an empty-filing marker whose key already holds a real row is dropped before
any save, for every table whose marker shares the real row's primary key (`def14a_llm`,
`sec_employees_text`). A `-F` re-fetch that now parses to nothing leaves the stored row untouched, while a
filing with no stored row still gets its marker."""

from __future__ import annotations

import pandas as pd

from src.data_extract.utils.edgar_extract.edgar_driver import EdgarFetch, EdgarScope, FilingStamp, marker_row, run_edgar_fetch
from src.data_extract.utils.edgar_extract.empty_markers import drop_markers_over_data, marker_shares_data_key
from src.data_store.schema import Table, Tables, marker_tables
from tests.data_extract.edgar_fixtures import fake_context, fake_filing, roster_run, seed_index

_AS_OF = pd.Timestamp("2026-09-30")
_KEPT = "0000000001-26-000001"
_NEW = "0000000001-26-000002"


class _NoReadStore:
    """A store double that fails the test on any read."""

    def load(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("a table whose marker has its own key must not be read")


def _real_text() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "accession_number": _KEPT,
                "filing_date": pd.Timestamp("2026-03-01"),
                "form": "10-K",
                "cik": "0000000001",
                "source_document": "primary",
                "n_chars": 12,
            }
        ]
    )


def test_the_shared_key_tables_are_def14a_llm_and_employees_text() -> None:
    shared = sorted(t.name for t in marker_tables() if marker_shares_data_key(t))
    assert shared == ["def14a_llm", "sec_employees_text"]
    assert not marker_shares_data_key(Tables.sec_def14a_text)
    print("\n=== SANITY CHECK: tables whose marker shares the real row's key ===")
    print(f"  {shared}; every other marker table (sec_def14a_text included) carries its sentinel inside the primary key.")


def test_an_employees_text_marker_over_a_stored_filing_is_dropped(sqlite_store) -> None:
    table = Tables.sec_employees_text
    stored = pd.DataFrame(
        [{"ticker": "AAA", "accession_number": _KEPT, "filing_date": pd.Timestamp("2026-03-01"), "form": "10-K", "source_document": "primary"}]
    )
    sqlite_store.save(table, stored)
    markers = pd.concat(
        [marker_row(table, "AAA", FilingStamp.of(fake_filing(accession, 1, "2026-03-01", form="10-K"), "0000000001")) for accession in (_KEPT, _NEW)],
        ignore_index=True,
    )

    kept = drop_markers_over_data(sqlite_store, table, markers)

    assert kept["accession_number"].tolist() == [_NEW] and kept["source_document"].tolist() == ["_empty"]
    print("\n=== SANITY CHECK: sec_employees_text marker over stored text ===")
    print(f"  2 markers built, the one over the stored filing dropped; kept {kept['accession_number'].tolist()} as source_document '_empty'.")


def test_a_marker_beside_its_real_row_in_the_same_frame_is_dropped(sqlite_store) -> None:
    marker = marker_row(Tables.sec_employees_text, "AAA", FilingStamp.of(fake_filing(_KEPT, 1, "2026-03-01", form="10-K"), "0000000001"))
    frame = pd.concat([_real_text(), marker], ignore_index=True)

    kept = drop_markers_over_data(sqlite_store, Tables.sec_employees_text, frame)

    assert kept["source_document"].tolist() == ["primary"]
    print("\n=== SANITY CHECK: marker and real row in one frame ===")
    print("  the real row survives; the marker with its key is dropped before the upsert can merge them.")


_SENTINEL = str(Tables.sec_employees_text.empty_marker[1]) if Tables.sec_employees_text.empty_marker else ""
_DATED = Table(
    "dated_marker_fixture", ("ticker", "as_of", "accession_number"), date_col="as_of", date_type_cols=("as_of",), empty_marker=("form", _SENTINEL)
)


class _StoredKeys:
    """A store double serving one stored real row's key (`as_of` as the DB's ISO day) and recording the read."""

    def __init__(self) -> None:
        self.reads: list[dict[str, object]] = []

    def load(self, table: Table, *, columns: list[str], where: dict[str, object], optional: bool) -> pd.DataFrame:
        self.reads.append({"columns": columns, "where": where})
        return pd.DataFrame([{"ticker": "AAA", "as_of": "2026-04-01", "accession_number": _KEPT}])


def test_a_marker_whose_dated_key_is_only_stored_is_dropped() -> None:
    store = _StoredKeys()
    markers = pd.DataFrame(
        [
            {"ticker": "AAA", "as_of": pd.Timestamp("2026-04-01 15:30"), "accession_number": _KEPT, "form": _SENTINEL},
            {"ticker": "AAA", "as_of": pd.Timestamp("2026-04-02"), "accession_number": _KEPT, "form": _SENTINEL},
            {"ticker": "BBB", "as_of": pd.Timestamp("2026-04-01"), "accession_number": _KEPT, "form": _SENTINEL},
            {"ticker": "AAA", "as_of": pd.Timestamp("2026-04-01"), "accession_number": _NEW, "form": _SENTINEL},
        ],
        index=[10, 11, 12, 13],
    )

    kept = drop_markers_over_data(store, _DATED, markers)  # type: ignore[arg-type]

    assert kept.index.tolist() == [11, 12, 13]
    assert kept.equals(markers.loc[[11, 12, 13]])
    assert store.reads == [{"columns": list(_DATED.pk), "where": {"ticker": ["AAA", "BBB"], "accession_number": sorted({_KEPT, _NEW})}}]
    print("\n=== SANITY CHECK: marker whose dated key exists only in the store ===")
    print("  4 markers, no real row in the frame; the one on the stored (AAA, 2026-04-01, accession) key is dropped")
    print("  (its time of day normalised to the ISO day); another day, another ticker and another accession are kept.")


def test_a_table_whose_sentinel_is_in_the_key_is_returned_unread() -> None:
    marker = marker_row(Tables.sec_8k, "AAA", FilingStamp.of(fake_filing(_KEPT, 1, "2026-04-01"), "0000000001"))

    out = drop_markers_over_data(_NoReadStore(), Tables.sec_8k, marker)  # type: ignore[arg-type]

    assert out is marker
    print("\n=== SANITY CHECK: sec_8k marker (item in the key) ===")
    print("  returned unchanged with no store read: its key can never equal a real row's key.")


def _parse_nothing(ticker: str, cik: str, stamp: FilingStamp, scope: EdgarScope) -> dict[Table, pd.DataFrame]:
    return {Tables.sec_employees_text: pd.DataFrame(columns=["ticker", "accession_number", "filing_date", "form"])}


def test_a_full_refetch_that_parses_nothing_leaves_the_stored_text_untouched(tmp_path, sqlite_store) -> None:
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"], ["1"])
    seed_index(ctx, [(1, "Fixture Co", "10-K", "2026-03-01", _KEPT), (1, "Fixture Co", "10-K", "2026-03-02", _NEW)])
    sqlite_store.save(Tables.sec_employees_text, _real_text())
    fetch = EdgarFetch(tables=(Tables.sec_employees_text,), parse=_parse_nothing)

    run_edgar_fetch(ctx, fetch, roster_run(ctx, ["AAA"], full=True, as_of=_AS_OF), max_workers=1)

    stored = sqlite_store.load(Tables.sec_employees_text, markers=True).set_index("accession_number")
    assert stored.loc[_KEPT, "source_document"] == "primary" and stored.loc[_KEPT, "n_chars"] == 12
    assert stored.loc[_NEW, "source_document"] == "_empty"
    print("\n=== SANITY CHECK: -F re-fetch that parses nothing ===")
    print(f"  {_KEPT}: source_document={stored.loc[_KEPT, 'source_document']!r}, n_chars={stored.loc[_KEPT, 'n_chars']} (untouched); {_NEW}: marker.")
