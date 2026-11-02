"""Exact public index parsing must account for every source line before inventory proof."""

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
def test_incomplete_master_index_cannot_prove_inventory(raw: str) -> None:
    with pytest.raises(ValueError):
        edgar_index.parse_master(raw.encode("latin-1"))
    print("SANITY: absent public boundary, unaccounted line, malformed nonselected form or post-boundary filing refuses inventory proof.")


def _legacy_write(df: pd.DataFrame, path: Path) -> None:
    """A quarter file as written before footers carried the original's boundary."""
    pq.write_table(pa.Table.from_pandas(df, schema=edgar_index._SCHEMA, preserve_index=False), path)


def _certify(context: Any, as_of: str, source_start: str = "2026-07-01") -> pd.Timestamp | None:
    """Plan on the cache as it stands, then certify (`file_versions` is what a walk records when it plans)."""
    return edgar_index.inventory_complete_through(context, pd.Timestamp(as_of), source_start, edgar_index.file_versions(context))


def test_missing_original_is_retried_and_failed_rebuild_preserves_good_cache(
    tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = fake_context(tmp_path, sqlite_store, ["AAA"])
    directory = edgar_index.index_dir(context)
    quarter = edgar_index.Quarter(2026, 3)
    raw = gzip.compress((MASTER_HEADER + "1|Issuer|4|2026-09-30|edgar/data/1/0000000001-26-000001.txt\n").encode("latin-1"))
    cache = directory / "2026Q3.parquet"
    _legacy_write(edgar_index.parse_master(raw), cache)
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
    assert _certify(context, "2026-10-22") is None, "the failed refresh of 2026-10-22 validated nothing"
    print("SANITY: legacy cached-only quarter retries its actual original; malformed refresh retains good bytes and cannot advance inventory proof.")


def test_changed_cached_index_cannot_reuse_original_inventory_boundary(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    context = fake_context(tmp_path, sqlite_store, ["AAA"])
    quarter = edgar_index.Quarter(2026, 3)
    raw = gzip.compress(MASTER_HEADER.encode("latin-1"))
    monkeypatch.setattr(edgar_index, "quarters_between", lambda first, last: [quarter])

    def download(context: Any, url: str, path: Path) -> int:
        path.write_bytes(raw)
        return 200

    monkeypatch.setattr(edgar_index.sec_io, "download", download)
    edgar_index.refresh(context, pd.Timestamp("2026-10-01"), 1)
    planned = edgar_index.file_versions(context)
    assert edgar_index.inventory_complete_through(context, pd.Timestamp("2026-10-01"), "2026-07-01", planned) == pd.Timestamp("2026-09-30")
    cache = edgar_index.index_dir(context) / "2026Q3.parquet"
    edgar_index._write_atomic(pd.read_parquet(cache).iloc[:0], cache, pd.Timestamp("2026-09-30"), pd.Timestamp("2026-10-01"))
    rewritten = edgar_index.inventory_complete_through(context, pd.Timestamp("2026-10-01"), "2026-07-01", planned)
    cache.write_bytes(cache.read_bytes() + b"changed")
    corrupted = edgar_index.inventory_complete_through(context, pd.Timestamp("2026-10-01"), "2026-07-01", planned)
    assert rewritten is None and corrupted is None
    print("SANITY: a quarter file replaced (even with a valid footer) or altered between plan and certify refuses certification.")


def _served_master(boundary: str, rows: str = "") -> bytes:
    """A master index whose source declares `boundary` as its last data received."""
    return gzip.compress((MASTER_HEADER.replace("September 30, 2026", pd.Timestamp(boundary).strftime("%B %d, %Y")) + rows).encode("latin-1"))


def _in_process_through(context: Any, as_of: pd.Timestamp, source_start: str) -> pd.Timestamp | None:
    """The pre-footer rule, as reference: the boundary each cached original proves, checked quarter by quarter."""
    directory = edgar_index.index_dir(context)
    business_day = pd.offsets.CustomBusinessDay(calendar=edgar_index.USFederalHolidayCalendar())
    public_day = as_of.normalize() - business_day
    boundary = None
    for quarter in edgar_index.quarters_between(pd.Timestamp(source_start), public_day):
        boundary = edgar_index._parse_master_unit((directory / f"{quarter.name}.master.gz").read_bytes())[1]
        required = pd.date_range(quarter.start, min(quarter.end, public_day), freq=business_day)
        if edgar_index.Quarter.of(boundary) != quarter or required.empty or boundary < required[-1]:
            return None
    return min(boundary, public_day) if boundary is not None else None


@pytest.mark.parametrize("q3_downloaded", ["2026-10-01", "2026-09-16"], ids=["fully-refreshed", "short-closed-quarter"])
def test_footer_certification_equals_the_in_process_boundary(
    tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, q3_downloaded: str
) -> None:
    context = fake_context(tmp_path, sqlite_store, ["AAA"])
    quarters = [edgar_index.Quarter(2026, 3), edgar_index.Quarter(2026, 4)]
    monkeypatch.setattr(edgar_index, "quarters_between", lambda first, last: [q for q in quarters if q <= edgar_index.Quarter.of(last)])
    served_through: dict[str, str] = {}

    def download(context: Any, url: str, path: Path) -> int:
        path.write_bytes(_served_master(served_through["2026Q4" if "QTR4" in url else "2026Q3"]))
        return 200

    monkeypatch.setattr(edgar_index.sec_io, "download", download)
    served_through["2026Q3"] = str(min(pd.Timestamp(q3_downloaded) - pd.Timedelta(days=1), pd.Timestamp("2026-09-30")).date())
    served_through["2026Q4"] = "2026-10-01"
    edgar_index.refresh(context, pd.Timestamp(q3_downloaded), 1)
    served_through["2026Q4"] = "2026-10-19"
    as_of = pd.Timestamp("2026-10-20")
    edgar_index.refresh(context, as_of, 1)
    expected = _in_process_through(context, as_of, "2026-07-01")
    found = _certify(context, "2026-10-20")
    assert found == expected
    assert found == (pd.Timestamp("2026-10-19") if q3_downloaded == "2026-10-01" else None)
    print(f"SANITY: Q3 last downloaded {q3_downloaded}: footer certification {found} equals the in-process rule {expected}.")


def test_legacy_quarter_file_certifies_only_after_one_refresh(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    context = fake_context(tmp_path, sqlite_store, ["AAA"])
    directory = edgar_index.index_dir(context)
    quarter = edgar_index.Quarter(2026, 3)
    raw = _served_master("2026-09-30", "1|Issuer|4|2026-09-30|edgar/data/1/0000000001-26-000001.txt\n")
    (directory / "2026Q3.master.gz").write_bytes(raw)
    _legacy_write(edgar_index.parse_master(raw), directory / "2026Q3.parquet")
    monkeypatch.setattr(edgar_index, "quarters_between", lambda first, last: [quarter])
    monkeypatch.setattr(edgar_index.sec_io, "download", lambda context, url, path: pytest.fail("a cached closed quarter is not downloaded"))

    before = _certify(context, "2026-10-20")
    edgar_index.refresh(context, pd.Timestamp("2026-10-20"), 1)
    after = _certify(context, "2026-10-20")
    fresh = fake_context(tmp_path, sqlite_store, ["AAA"])
    assert before is None and after == pd.Timestamp("2026-09-30") and _certify(fresh, "2026-10-20") == after
    assert edgar_index.entries(context, {"1"}, ["4"])["accession"].tolist() == ["0000000001-26-000001"]
    print(
        "SANITY: a legacy file without footer stamps certifies nothing; one refresh re-derives its boundary from the cached original "
        "and a fresh process then certifies 2026-09-30."
    )


def test_each_index_writer_owns_its_temporary_source_bytes(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    context = fake_context(tmp_path, sqlite_store, ["AAA"])
    path = edgar_index.index_dir(context) / "2026Q3.parquet"
    raw = MASTER_HEADER + "1|Issuer|4|2026-09-30|edgar/data/1/0000000001-26-000001.txt\n"
    actual = edgar_index.parse_master(raw.encode("latin-1"))
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
                    edgar_index._write_atomic(other, path, pd.Timestamp("2026-09-30"), pd.Timestamp("2026-10-01"))
            finally:
                competing = False

    monkeypatch.setattr(edgar_index.pq, "write_table", write_then_compete)
    monkeypatch.setattr(edgar_index.os, "replace", interrupted_replace)
    edgar_index._write_atomic(actual, path, pd.Timestamp("2026-09-30"), pd.Timestamp("2026-10-01"))
    assert edgar_index.entries(context, {"1"}, ["4"])["accession"].tolist() == ["0000000001-26-000001"]
    assert not list(path.parent.glob("*.part"))
    print(
        "SANITY: an interrupted concurrent index writer cannot replace another writer's staged payload or bind its public boundary to different source bytes."
    )


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

        def publish_then_compete(frame: pd.DataFrame, path: Path, through: pd.Timestamp, refreshed: pd.Timestamp) -> None:
            nonlocal future
            writer(frame, path, through, refreshed)
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
    assert _certify(first, "2026-10-01") == pd.Timestamp("2026-09-30")
    assert _certify(second, "2026-10-01") == pd.Timestamp("2026-09-30"), "B's failed download leaves A's validated file and footer"
    assert not list(directory.glob("*.part"))
    print(
        "SANITY: each actual download owns its temporary bytes; a competing failed download preserves the good original, selected cache and proven A inventory."
    )
