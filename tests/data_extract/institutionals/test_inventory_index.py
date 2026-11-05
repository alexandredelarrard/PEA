"""Exact public index parsing accounts for every source line; each refresh owns the bytes it validates."""

from __future__ import annotations

import gzip
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data_extract.utils.edgar_extract import edgar_index
from tests.data_extract.edgar_fixtures import MASTER_HEADER, fake_context


@pytest.mark.parametrize(
    "raw",
    [
        "1|Known issuer|4|2026-09-30|edgar/data/1/0000000001-26-000001.txt\n",
        MASTER_HEADER + "unaccounted malformed source line\n",
        MASTER_HEADER + "1|Known issuer|UNKNOWN_FORM|bad date|edgar/data/1/0000000001-26-000001.txt\n",
        MASTER_HEADER + "1|Known issuer|4|2026-10-01|edgar/data/1/0000000001-26-000001.txt\n",
    ],
)
def test_incomplete_master_index_is_rejected(raw: str) -> None:
    with pytest.raises(ValueError):
        edgar_index._parse_master_unit(raw.encode("latin-1"))
    print("SANITY: absent public boundary, unaccounted line, malformed nonselected form or post-boundary filing is rejected.")


def _legacy_write(df: pd.DataFrame, path: Path) -> None:
    """A quarter file written directly, with no cached original next to it."""
    pq.write_table(pa.Table.from_pandas(df, schema=edgar_index._SCHEMA, preserve_index=False), path)


def test_missing_original_is_retried_and_failed_rebuild_preserves_good_cache(
    tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = fake_context(tmp_path, sqlite_store, ["AAA"])
    directory = edgar_index.index_dir(context)
    quarter = edgar_index.Quarter(2026, 3)
    raw = gzip.compress((MASTER_HEADER + "1|Issuer|4|2026-09-30|edgar/data/1/0000000001-26-000001.txt\n").encode("latin-1"))
    cache = directory / "2026Q3.parquet"
    _legacy_write(edgar_index._parse_master_unit(raw)[0], cache)
    requested: list[str] = []

    def download(context: Any, url: str, path: Path) -> int:
        requested.append(url)
        path.write_bytes(raw if len(requested) == 1 else b"bad source")
        return 200

    monkeypatch.setattr(edgar_index.sec_io, "download", download)
    monkeypatch.setattr(edgar_index, "quarters_between", lambda first, last: [quarter])
    edgar_index.refresh(context, pd.Timestamp("2026-10-21"), 1)
    original = directory / "2026Q3.master.gz"
    assert len(requested) == 1 and original.read_bytes() == raw
    before = cache.read_bytes()
    edgar_index.refresh(context, pd.Timestamp("2026-10-22"), 1, rebuild=True)
    assert len(requested) == 2 and original.read_bytes() == raw and cache.read_bytes() == before
    print("SANITY: a cached-only quarter retries its actual original; a malformed refresh retains the good original and selected cache bytes.")


def test_each_index_writer_owns_its_temporary_source_bytes(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    context = fake_context(tmp_path, sqlite_store, ["AAA"])
    path = edgar_index.index_dir(context) / "2026Q3.parquet"
    raw = MASTER_HEADER + "1|Issuer|4|2026-09-30|edgar/data/1/0000000001-26-000001.txt\n"
    actual = edgar_index._parse_master_unit(raw.encode("latin-1"))[0]
    other = actual.iloc[:0]
    write = edgar_index.pq.write_table
    replace = edgar_index.os.replace
    entered = False
    competing = False

    def interrupted_replace(source: Any, target: Any) -> None:
        if competing:
            raise RuntimeError("controlled competing writer paused before publication")
        replace(source, target)

    def write_then_compete(table: Any, destination: Any, **kwargs: Any) -> None:
        nonlocal entered, competing
        write(table, destination, **kwargs)
        if not entered:
            entered, competing = True, True
            try:
                with pytest.raises(RuntimeError, match="controlled"):
                    edgar_index._write_atomic(other, path)
            finally:
                competing = False

    monkeypatch.setattr(edgar_index.pq, "write_table", write_then_compete)
    monkeypatch.setattr(edgar_index.os, "replace", interrupted_replace)
    edgar_index._write_atomic(actual, path)
    assert edgar_index.entries(context, {"1"}, ["4"])["accession"].tolist() == ["0000000001-26-000001"]
    assert not list(path.parent.glob("*.part"))
    print("SANITY: an interrupted concurrent index writer cannot replace another writer's staged payload.")


@pytest.mark.parametrize("compete", [False, True], ids=["coherent", "paused-bad-download"])
def test_each_index_download_owns_validated_original_bytes(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, compete: bool) -> None:
    first = fake_context(tmp_path, sqlite_store, ["AAA"])
    second = fake_context(tmp_path, sqlite_store, ["AAA"])
    quarter = edgar_index.Quarter(2026, 3)
    raw = gzip.compress((MASTER_HEADER + "1|Issuer|4|2026-09-30|edgar/data/1/0000000001-26-000001.txt\n").encode("latin-1"))
    directory = edgar_index.index_dir(first)
    original = directory / "2026Q3.master.gz"
    original.write_bytes(raw)
    wrote_bad, resume_bad = Event(), Event()
    writer = edgar_index._write_atomic
    future: Future[dict[str, int]] | None = None

    def download(context: Any, url: str, path: Path) -> int:
        path.write_bytes(raw if context is first else b"controlled invalid B payload")
        if context is second:
            wrote_bad.set()
            assert resume_bad.wait(5)
        return 200

    with ThreadPoolExecutor(max_workers=1) as executor:

        def publish_then_compete(frame: pd.DataFrame, path: Path) -> None:
            nonlocal future
            writer(frame, path)
            if compete:
                future = executor.submit(edgar_index.refresh, second, pd.Timestamp("2026-10-01"), 1, rebuild=True)
                assert wrote_bad.wait(5)

        monkeypatch.setattr(edgar_index, "quarters_between", lambda first, last: [quarter])
        monkeypatch.setattr(edgar_index.sec_io, "download", download)
        monkeypatch.setattr(edgar_index, "_write_atomic", publish_then_compete)
        try:
            assert edgar_index.refresh(first, pd.Timestamp("2026-10-01"), 1, rebuild=True) == {quarter.name: 1}
        finally:
            resume_bad.set()
        if future is not None:
            assert future.result(timeout=5) == {}
    assert original.read_bytes() == raw, "A's validated original cannot be replaced by B's unparsed download bytes"
    assert edgar_index.entries(first, {"1"}, ["4"])["accession"].tolist() == ["0000000001-26-000001"]
    assert edgar_index.entries(second, {"1"}, ["4"])["accession"].tolist() == ["0000000001-26-000001"], "B's failed download leaves A's file"
    assert not list(directory.glob("*.part"))
    print("SANITY: each actual download owns its temporary bytes; a competing failed download preserves the good original and selected cache.")
