"""Stored call discovery uses one projected read, independent of ticker count."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from src.data_aggregate.utils.text.earnings_call_embeddings import call_keys
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from tests.fixtures.earnings_call_rows import FIXTURE_PATHS, fixture_paragraphs


def test_call_keys_reads_once_for_all_tickers(sqlite_store: DataStore) -> None:
    df_paragraphs = pd.concat([fixture_paragraphs(path.stem) for path in FIXTURE_PATHS], ignore_index=True)
    sqlite_store.save(Tables.earnings_call_sections, df_paragraphs)
    df_expected = df_paragraphs[["ticker", "quarter"]].drop_duplicates().sort_values(["ticker", "quarter"]).reset_index(drop=True)

    with patch.object(sqlite_store, "load", wraps=sqlite_store.load) as read:
        df_keys = call_keys(sqlite_store)

    read.assert_called_once_with(Tables.earnings_call_sections, ["ticker", "quarter"], where={"paragraph": 1}, optional=True)
    pd.testing.assert_frame_equal(df_keys.sort_values(["ticker", "quarter"]).reset_index(drop=True), df_expected)
    print(f"SANITY: {len(df_paragraphs)} real paragraph rows yield {len(df_keys)} unique calls in one read, without transcript text.")


@pytest.mark.parametrize("empty_table", [False, True], ids=["missing", "empty"])
def test_call_keys_handles_no_stored_calls(sqlite_store: DataStore, empty_table: bool) -> None:
    if empty_table:
        df_paragraphs = fixture_paragraphs(FIXTURE_PATHS[0].stem)
        sqlite_store.save(Tables.earnings_call_sections, df_paragraphs)
        sqlite_store.delete(Tables.earnings_call_sections, {"ticker": df_paragraphs["ticker"].unique().tolist()})

    with patch.object(sqlite_store, "load", wraps=sqlite_store.load) as read:
        df_keys = call_keys(sqlite_store)

    assert read.call_count == 1
    pd.testing.assert_frame_equal(df_keys, pd.DataFrame(columns=["ticker", "quarter"]))
    print(f"SANITY: {'empty' if empty_table else 'missing'} source returns an empty call-key frame after one optional read.")
