"""Shared SEC bulk helpers: one zip reader with a per-site corrupt policy, and the incremental
pending-period / processed-scope state used by statements, notes, insider and fails-to-deliver."""

from __future__ import annotations

import json
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
    mark_processed,
    pending_periods,
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


def test_pending_periods_skips_stored_only_when_converged(tmp_path: Path) -> None:
    store = FakeStore({"pension_facts": pd.DataFrame({"quarter": ["2026q1"]})})
    context = SimpleNamespace(store=store)
    periods = ["2026q1", "2026q2"]

    assert pending_periods(context, tmp_path, Tables.pension_facts, periods, ["AAPL"], column="quarter") == periods  # no sidecar yet
    mark_processed(tmp_path, Tables.pension_facts, ["MSFT", "AAPL"])
    sidecar = json.loads((tmp_path / "pension_facts_universe.json").read_text(encoding="utf-8"))
    assert sidecar["universe"] == ["AAPL", "MSFT"]

    assert pending_periods(context, tmp_path, Tables.pension_facts, periods, ["AAPL"], column="quarter") == ["2026q2"]
    assert pending_periods(context, tmp_path, Tables.pension_facts, periods, ["AAPL", "NVDA"], column="quarter") == periods
    queries = len(store.queries)
    assert pending_periods(context, tmp_path, Tables.pension_facts, periods, ["AAPL"], reparse=True, column="quarter") == periods
    assert len(store.queries) == queries  # reparse never reads the store
    print("\n=== SANITY CHECK: pending periods ===")
    print("  first run -> all; converged -> only the unstored 2026q2; new scope member or reparse -> all, reparse with no DB read. Validated.")


def test_pending_periods_unions_tables_and_keys_the_sidecar_on_the_first(tmp_path: Path) -> None:
    store = FakeStore(
        {
            "notes_num": pd.DataFrame({"period": ["2026_07"]}),
            "notes_text": pd.DataFrame({"period": ["2026_08"]}),
        }
    )
    context = SimpleNamespace(store=store)
    mark_processed(tmp_path, Tables.notes_num, ["AAPL"])
    tables = (Tables.notes_num, Tables.notes_text)
    assert pending_periods(context, tmp_path, tables, ["2026_07", "2026_08", "2026_09"], ["AAPL"]) == ["2026_09"]
    (tmp_path / "notes_num_universe.json").write_text("{not json", encoding="utf-8")
    assert pending_periods(context, tmp_path, tables, ["2026_07", "2026_08"], ["AAPL"]) == ["2026_07", "2026_08"]
    print("\n=== SANITY CHECK: multi-table pending periods ===")
    print("  stored periods union across notes_num + notes_text; an unreadable sidecar re-parses everything. Validated.")


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
