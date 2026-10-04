"""Never mark over data: an empty-filing marker whose key already holds a real row is dropped before
any save, for every table whose marker shares the real row's primary key (`sec_def14a`,
`def14a_llm`). A `-F` re-fetch that now parses to nothing leaves the stored row untouched, while a
filing with no stored row still gets its marker."""

from __future__ import annotations

import pandas as pd

from src.data_extract.utils.common.edgar_driver import EdgarFetch, EdgarScope, FilingStamp, marker_row, run_edgar_fetch
from src.data_extract.utils.common.empty_markers import drop_markers_over_data, marker_shares_data_key
from src.data_store.schema import Table, Tables, marker_tables
from tests.data_extract.edgar_fixtures import fake_context, fake_filing, seed_index

_AS_OF = pd.Timestamp("2026-09-30")
_KEPT = "0000000001-26-000001"
_NEW = "0000000001-26-000002"


class _NoReadStore:
    """A store double that fails the test on any read."""

    def load(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("a table whose marker has its own key must not be read")


def _real_proxy() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "accession_number": _KEPT,
                "filing_date": pd.Timestamp("2026-04-01"),
                "form": "DEF 14A",
                "cik": "0000000001",
                "peo_total_comp": 12.5,
            }
        ]
    )


def test_the_shared_key_tables_are_exactly_the_two_proxy_tables() -> None:
    shared = sorted(t.name for t in marker_tables() if marker_shares_data_key(t))
    assert shared == ["def14a_llm", "sec_def14a"]
    print("\n=== SANITY CHECK: tables whose marker shares the real row's key ===")
    print(f"  {shared}; every other marker table carries its sentinel inside the primary key.")


def test_a_marker_over_a_stored_row_is_dropped_and_a_new_one_kept(sqlite_store) -> None:
    sqlite_store.save(Tables.def14a_edgar, _real_proxy())
    markers = pd.concat(
        [
            marker_row(Tables.def14a_edgar, "AAA", FilingStamp.of(fake_filing(accession, 1, "2026-04-01", form="DEF 14A"), "0000000001"))
            for accession in (_KEPT, _NEW)
        ],
        ignore_index=True,
    )

    kept = drop_markers_over_data(sqlite_store, Tables.def14a_edgar, markers)

    assert kept["accession_number"].tolist() == [_NEW]
    print("\n=== SANITY CHECK: marker over a stored real row ===")
    print(f"  2 markers built, 1 over stored data dropped; kept {kept['accession_number'].tolist()}.")


def test_a_marker_beside_its_real_row_in_the_same_frame_is_dropped(sqlite_store) -> None:
    marker = marker_row(Tables.def14a_edgar, "AAA", FilingStamp.of(fake_filing(_KEPT, 1, "2026-04-01", form="DEF 14A"), "0000000001"))
    frame = pd.concat([_real_proxy(), marker], ignore_index=True)

    kept = drop_markers_over_data(sqlite_store, Tables.def14a_edgar, frame)

    assert kept["form"].tolist() == ["DEF 14A"]
    print("\n=== SANITY CHECK: marker and real row in one frame ===")
    print("  the real row survives; the marker with its key is dropped before the upsert can merge them.")


def test_a_table_whose_sentinel_is_in_the_key_is_returned_unread() -> None:
    marker = marker_row(Tables.sec_8k, "AAA", FilingStamp.of(fake_filing(_KEPT, 1, "2026-04-01"), "0000000001"))

    out = drop_markers_over_data(_NoReadStore(), Tables.sec_8k, marker)  # type: ignore[arg-type]

    assert out is marker
    print("\n=== SANITY CHECK: sec_8k marker (item in the key) ===")
    print("  returned unchanged with no store read: its key can never equal a real row's key.")


def _parse_nothing(ticker: str, cik: str, stamp: FilingStamp, scope: EdgarScope) -> dict[Table, pd.DataFrame]:
    return {Tables.def14a_edgar: pd.DataFrame(columns=["ticker", "accession_number", "filing_date", "form"])}


def test_a_full_refetch_that_parses_nothing_leaves_the_stored_proxy_untouched(tmp_path, sqlite_store) -> None:
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"], ["1"])
    seed_index(ctx, [(1, "Fixture Co", "DEF 14A", "2026-04-01", _KEPT), (1, "Fixture Co", "DEF 14A", "2026-04-02", _NEW)])
    sqlite_store.save(Tables.def14a_edgar, _real_proxy())
    fetch = EdgarFetch(desc="proxy test", tables=(Tables.def14a_edgar,), forms=("DEF 14A",), parse=_parse_nothing, identity_aware=False)

    run_edgar_fetch(ctx, ["AAA"], 15, fetch, full=True, as_of=_AS_OF, refresh_index=False, max_workers=1)

    stored = sqlite_store.load(Tables.def14a_edgar, markers=True).set_index("accession_number")
    assert stored.loc[_KEPT, "form"] == "DEF 14A" and stored.loc[_KEPT, "peo_total_comp"] == 12.5
    assert stored.loc[_NEW, "form"] == "_empty"
    print("\n=== SANITY CHECK: -F re-fetch losing the ECD facts ===")
    print(f"  {_KEPT}: form={stored.loc[_KEPT, 'form']!r}, peo_total_comp={stored.loc[_KEPT, 'peo_total_comp']} (untouched); {_NEW}: marker.")
