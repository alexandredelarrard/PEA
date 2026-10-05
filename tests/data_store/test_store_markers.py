"""Empty-filing marker rows: hidden from consumer reads, counted by resume reads, typed NULLs on read-back."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from src.data_store.schema import Tables

_REAL = "0000000001-24-000001"
_EMPTY_FILING = "0000000001-24-000002"


def _sec_13g_with_marker() -> pd.DataFrame:
    """Two reporting persons on one filing plus one marker row on a later, empty filing."""
    return pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "accession_number": [_REAL, _REAL, _EMPTY_FILING],
            "rp_seq": [0, 1, -1],
            "form": ["SC 13G", "SC 13G", "SC 13G"],
            "filing_date": pd.to_datetime(["2024-02-01", "2024-02-01", "2024-03-01"]),
            "date_of_event": pd.to_datetime(pd.Series(["2024-01-15", "2024-01-15", None])),
            "percent_of_class": [5.1, 2.0, np.nan],
            "cusip": ["000000AA1", "000000AA1", None],
        }
    )


def test_load_hides_marker_rows(sqlite_store) -> None:
    sqlite_store.save(Tables.sec_13g, _sec_13g_with_marker())

    df = sqlite_store.load(Tables.sec_13g)
    assert df is not None
    assert sorted(df["rp_seq"].tolist()) == [0, 1]
    assert _EMPTY_FILING not in set(df["accession_number"])

    print("\n=== SANITY CHECK: load hides markers ===")
    print(f"  stored 3 rows (1 marker) -> load returns {len(df)} rows, rp_seq {sorted(df['rp_seq'].tolist())}. Validated.")


def test_markers_true_and_iter_load(sqlite_store) -> None:
    sqlite_store.save(Tables.sec_13g, _sec_13g_with_marker())

    shown = sqlite_store.load(Tables.sec_13g, markers=True)
    assert shown is not None and len(shown) == 3
    chunks = list(sqlite_store.iter_load(Tables.sec_13g, columns=["accession_number", "rp_seq"], chunksize=1))
    hidden_stream = pd.concat(chunks, ignore_index=True)
    assert sorted(hidden_stream["rp_seq"].tolist()) == [0, 1]
    full_stream = pd.concat(sqlite_store.iter_load(Tables.sec_13g, columns=["rp_seq"], markers=True), ignore_index=True)
    assert sorted(full_stream["rp_seq"].tolist()) == [-1, 0, 1]
    # a projection that omits the marker column still filters on it
    projected = sqlite_store.load(Tables.sec_13g, columns=["accession_number"])
    assert projected is not None and set(projected["accession_number"]) == {_REAL}

    print("\n=== SANITY CHECK: marker opt-in ===")
    print(f"  load(markers=True) -> {len(shown)} rows; iter_load -> {len(hidden_stream)} hidden, {len(full_stream)} with markers=True.")
    print("  The filter applies even when the marker column is not projected. Validated.")


def test_marker_only_result_is_empty_for_consumers(sqlite_store) -> None:
    sqlite_store.save(Tables.sec_13g, _sec_13g_with_marker())

    assert sqlite_store.load(Tables.sec_13g, where={"accession_number": _EMPTY_FILING}, optional=True) is None
    only = sqlite_store.load(Tables.sec_13g, where={"accession_number": _EMPTY_FILING}, markers=True)
    assert only is not None and only["rp_seq"].tolist() == [-1]

    print("\n=== SANITY CHECK: an empty filing is not an observation ===")
    print("  a consumer read of the marker's filing alone returns no rows; markers=True returns the one marker. Validated.")


def test_resume_reads_count_markers(sqlite_store) -> None:
    sqlite_store.save(Tables.sec_13g, _sec_13g_with_marker())

    accessions = set(sqlite_store.distinct(Tables.sec_13g, "accession_number"))
    frontier = sqlite_store.max_date_by(Tables.sec_13g, "ticker")
    assert accessions == {_REAL, _EMPTY_FILING}
    assert frontier == {"AAA": pd.Timestamp("2024-03-01")}
    assert sqlite_store.row_count(Tables.sec_13g) == 3

    print("\n=== SANITY CHECK: resume sees markers ===")
    print(f"  distinct accessions {sorted(accessions)}; frontier {frontier['AAA'].date()} is the marker's filing date. Validated.")


def test_key_stats_first_last_n(sqlite_store) -> None:
    frame = pd.concat(
        [
            _sec_13g_with_marker(),
            pd.DataFrame(
                {
                    "ticker": ["BBB"],
                    "accession_number": ["0000000002-23-000001"],
                    "rp_seq": [0],
                    "filing_date": pd.to_datetime(["2023-06-30"]),
                }
            ),
        ],
        ignore_index=True,
    )
    sqlite_store.save(Tables.sec_13g, frame)

    stats = sqlite_store.key_stats(Tables.sec_13g, "ticker").set_index("key")
    assert list(stats.columns) == ["first", "last", "n"]
    assert stats.loc["AAA", "first"] == pd.Timestamp("2024-02-01")
    assert stats.loc["AAA", "last"] == pd.Timestamp("2024-03-01")
    assert stats.loc["AAA", "n"] == 3
    assert stats.loc["BBB", "first"] == stats.loc["BBB", "last"] == pd.Timestamp("2023-06-30")
    assert stats.loc["BBB", "n"] == 1
    assert pd.api.types.is_datetime64_any_dtype(stats["first"]) and pd.api.types.is_datetime64_any_dtype(stats["last"])

    scoped = sqlite_store.key_stats(Tables.sec_13g, "ticker", where={"ticker": "BBB"})
    assert scoped["key"].tolist() == ["BBB"]

    absent = sqlite_store.key_stats(Tables.prices, "ticker")
    assert absent.empty and list(absent.columns) == ["key", "first", "last", "n"]

    print("\n=== SANITY CHECK: key_stats ===")
    print(stats.to_string())
    print("  one GROUP BY: first/last as Timestamps, n counts the marker; absent table -> empty frame. Validated.")


def test_marker_typed_nulls_round_trip(sqlite_store) -> None:
    sqlite_store.save(Tables.sec_13g, _sec_13g_with_marker())
    df_13g = sqlite_store.load(Tables.sec_13g, markers=True)
    assert df_13g is not None
    marker = df_13g.loc[df_13g["rp_seq"] == -1].iloc[0]
    assert isinstance(marker["percent_of_class"], float) and np.isnan(marker["percent_of_class"])
    # pandas 3 reads TEXT as the `str` dtype, whose missing value is NaN; never a placeholder string
    assert pd.api.types.is_string_dtype(df_13g["cusip"]) and not isinstance(marker["cusip"], str) and pd.isna(marker["cusip"])
    assert marker["date_of_event"] is None
    assert isinstance(marker["filing_date"], dt.date)

    llm = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "accession_number": [_REAL, _EMPTY_FILING],
            "as_of": pd.to_datetime(["2024-04-01", "2025-04-01"]),
            "period": pd.to_datetime(pd.Series(["2023-12-31", None])),
            "avg_director_age": [61.5, np.nan],
            "def14a_json": ['{"directors": []}', "_empty"],
        }
    )
    sqlite_store.save(Tables.def14a_llm, llm)
    df_llm = sqlite_store.load(Tables.def14a_llm, markers=True)
    assert df_llm is not None
    llm_marker = df_llm.loc[df_llm["def14a_json"] == "_empty"].iloc[0]
    assert llm_marker["period"] is pd.NaT
    assert np.isnan(llm_marker["avg_director_age"])
    hidden = sqlite_store.load(Tables.def14a_llm)
    assert hidden is not None and hidden["accession_number"].tolist() == [_REAL]

    print("\n=== SANITY CHECK: typed NULLs on a marker row ===")
    print(
        f"  float -> {marker['percent_of_class']!r}, text -> {marker['cusip']!r}, DATE -> {marker['date_of_event']!r}, TIMESTAMP -> {llm_marker['period']!r}"
    )
    print("  No placeholder value ('' / 0 / 'nan') reaches a reader. Validated.")
