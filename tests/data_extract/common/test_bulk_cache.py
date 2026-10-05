"""Shared SEC bulk helpers: one zip reader with a per-site corrupt policy, the cached-archive listing
and the period calendar used by statements, notes, insider and fails-to-deliver."""

from __future__ import annotations

import logging
import zipfile
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common.bulk_cache import (
    ZipRead,
    cached_periods,
    period_end,
    read_zip_tables,
    stored_period_clock,
)
from src.data_store.schema import Tables

LOG = logging.getLogger("test_bulk_cache")


class FakeStore:
    """`columns` + `distinct` over in-memory {table: frame}, recording every distinct query."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames
        self.queries: list[tuple[str, str]] = []

    def columns(self, table: Any) -> list[str]:
        frame = self.frames.get(str(table))
        return [] if frame is None else list(frame.columns)

    def distinct(self, table: Any, column: str, *, where: dict | None = None) -> list[Any]:
        self.queries.append((str(table), column))
        frame = self.frames[str(table)]
        for key, value in (where or {}).items():
            frame = frame[frame[key] == value]
        return list(frame[column].dropna().unique())


def _zip(path: Path, members: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in members.items():
            archive.writestr(name, text)
    return path


@pytest.mark.parametrize(("policy", "kept"), [("delete", False), ("skip", True)])
def test_corrupt_zip_follows_the_site_policy(tmp_path: Path, policy: str, kept: bool) -> None:
    path = tmp_path / "2026q1.zip"
    path.write_bytes(b"PK\x03\x04 truncated, not a zip")
    assert read_zip_tables(path, {"sub.txt": ZipRead()}, on_corrupt=policy, log=LOG) is None  # type: ignore[arg-type]
    assert path.exists() is kept
    print(f"\n=== SANITY CHECK: corrupt zip, policy={policy} ===")
    print(f"  returned None, file kept={kept}: delete sites re-download next run, skip sites keep the file. Validated.")


def test_members_case_insensitive_required_optional_and_upper(tmp_path: Path) -> None:
    path = _zip(tmp_path / "q.zip", {"Submission.tsv": "accession_number\tIssuerCik\tjunk\na1\t1\tx\n"})
    tables = read_zip_tables(
        path,
        {"SUBMISSION.TSV": ZipRead(usecols=frozenset({"ACCESSION_NUMBER", "ISSUERCIK"}), upper=True), "FOOTNOTES.TSV": ZipRead(required=False)},
        on_corrupt="skip",
        log=LOG,
    )
    assert tables is not None
    assert list(tables["SUBMISSION.TSV"].columns) == ["ACCESSION_NUMBER", "ISSUERCIK"]
    assert tables["SUBMISSION.TSV"]["ISSUERCIK"].tolist() == ["1"]  # dtype=str: no int coercion
    assert tables["FOOTNOTES.TSV"].empty and list(tables["FOOTNOTES.TSV"].columns) == []
    assert read_zip_tables(path, {"num.txt": ZipRead()}, on_corrupt="skip", log=LOG) == {}
    print("\n=== SANITY CHECK: member mapping ===")
    print("  mixed-case member found, columns upper-cased and projected, optional absent -> empty frame, required absent -> {}. Validated.")


def test_chunked_keep_concatenates_matches_and_returns_bare_frame_when_none_match(tmp_path: Path) -> None:
    rows = "".join(f"a{i}\t{'Pension' if i % 3 == 0 else 'Other'}\n" for i in range(10))
    path = _zip(tmp_path / "q.zip", {"num.txt": "adsh\ttag\n" + rows})
    spec = {"num.txt": ZipRead(keep=lambda chunk: chunk["tag"] == "Pension", chunksize=4)}
    tables = read_zip_tables(path, spec, on_corrupt="delete", log=LOG)
    assert tables is not None
    assert tables["num.txt"]["adsh"].tolist() == ["a0", "a3", "a6", "a9"]
    assert tables["num.txt"].index.tolist() == [0, 1, 2, 3]
    none = read_zip_tables(path, {"num.txt": ZipRead(keep=lambda chunk: chunk["tag"] == "Nope", chunksize=4)}, on_corrupt="delete", log=LOG)
    assert none is not None and none["num.txt"].shape == (0, 0)
    with pytest.raises(ValueError):
        ZipRead(keep=lambda chunk: chunk["tag"] == "Pension")
    print("\n=== SANITY CHECK: chunked row filter ===")
    print("  4 of 10 rows kept across 3 chunks with a fresh index; no match -> a bare empty frame; keep without chunksize rejected. Validated.")


def test_cached_periods_lists_non_empty_archives_by_their_tag(tmp_path: Path) -> None:
    for name, payload in {
        "cnsfails202401a.zip": b"x",
        "cnsfails202401b.zip": b"",
        "2026_08_notes.zip": b"x",
        "readme.txt": b"{}",
    }.items():
        (tmp_path / name).write_bytes(payload)
    assert cached_periods(tmp_path, prefix="cnsfails") == {"202401a"}  # the empty 202401b is not an archive
    assert cached_periods(tmp_path, suffix="_notes.zip") == {"2026_08"}
    print("\n=== SANITY CHECK: cached archive listing ===")
    print("  tags come from the archive names; an empty file and a non-zip file are not listed. Validated.")


def test_period_end_covers_quarterly_monthly_and_semi_monthly_tags() -> None:
    assert period_end("2026q1") == date(2026, 3, 31)
    assert period_end("2026_02") == date(2026, 2, 28)
    assert period_end("200907a") == date(2009, 7, 15)
    assert period_end("200907b") == date(2009, 7, 31)
    print("\n=== SANITY CHECK: archive period ends ===")
    print("  YYYYqN -> quarter end, YYYY_MM -> month end, FTD 'a' -> the 15th, 'b' -> month end. Validated.")


def test_stored_period_clock_is_shared_and_rejects_conflicts() -> None:
    store = FakeStore(
        {
            "notes_num": pd.DataFrame({"period": ["2026_08"], "available_at": [date(2026, 9, 14)]}),
            "notes_text": pd.DataFrame({"period": ["2026_08"], "available_at": [date(2026, 9, 15)]}),
        }
    )
    context = SimpleNamespace(store=store)
    assert stored_period_clock(context, (Tables.notes_num,), "2026_08") == date(2026, 9, 14)
    assert stored_period_clock(context, (Tables.notes_num,), "2026_09") is None
    with pytest.raises(ValueError, match="conflicting stored available_at values"):
        stored_period_clock(context, (Tables.notes_num, Tables.notes_text), "2026_08")
    print("\n=== SANITY CHECK: stored archive clock ===")
    print("  one stored clock is returned, none -> None, two tables disagreeing -> ValueError. Validated.")
