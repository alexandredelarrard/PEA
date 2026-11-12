"""AC-002: `StepExtractZips.run_download()` caches archives and writes no table (the SEC ticker list is `StepExtractUniverse`'s).

A real (SQLite) store whose write methods are counted; the SEC is replaced by stubs, so no network is used.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.data_extract.transformers.step_extract_zips import StepExtractZips
from src.data_extract.utils.common import bulk_cache
from src.data_extract.utils.zips import fetch_fails_to_deliver as ftd
from src.data_extract.utils.zips import fetch_financial_notes as notes
from src.data_extract.utils.zips import fetch_insider_transactions as insider
from src.data_extract.utils.zips import thirteen_f_datasets as f13ds
from src.data_extract.utils.zips import zip_archive
from tests.data_extract.zips.test_thirteen_f_datasets import write_13f_zip

WRITES = ("append_tail", "bulk_seed", "delete", "drop", "ensure_columns", "insert", "replace", "save", "update")


def _counted(store: Any, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    written: list[str] = []
    for name in WRITES:
        original = getattr(store, name)

        def record(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
            table = args[0] if args else kwargs.get("table")
            written.append(str(getattr(table, "name", table)))
            return _original(*args, **kwargs)

        monkeypatch.setattr(store, name, record)
    return written


def _offline(monkeypatch: pytest.MonkeyPatch, downloads: list[str]) -> None:
    def fake_zip(context: Any, path: Path, urls: Any, **kwargs: Any) -> Path:
        downloads.append(path.name)
        path.write_bytes(b"zip")
        return path

    def fake_13f_zip(context: Any, path: Path, urls: Any, **kwargs: Any) -> Path:
        downloads.append(path.name)
        write_13f_zip(path)
        return path

    for module in (insider, notes, ftd, zip_archive):
        monkeypatch.setattr(module, "ensure_zip", fake_zip)
    monkeypatch.setattr(f13ds, "ensure_zip", fake_13f_zip)
    monkeypatch.setattr(notes, "_scrape_available_periods", lambda context: ["2026q1"])
    monkeypatch.setattr(f13ds, "published_data_sets", lambda context: [f13ds.DataSet.parse("2026q1")])


def test_run_download_writes_no_table(sqlite_store, tmp_path, monkeypatch):
    downloads: list[str] = []
    _offline(monkeypatch, downloads)
    written = _counted(sqlite_store, monkeypatch)
    paths = SimpleNamespace(insider_transactions="insider", financial_notes="notes", fails_deliver="ftd", sec_13f_datasets="13f")
    config = SimpleNamespace(data_extract=SimpleNamespace(years_history=1), local=SimpleNamespace(paths=paths))
    context = SimpleNamespace(store=sqlite_store, config=config, config_dir="configs", paths={"DATA_STORE": tmp_path})

    StepExtractZips(context, config).run_download()  # type: ignore[arg-type]

    cached = {directory: len(list((tmp_path / directory).glob("*.zip"))) for directory in ("insider", "notes", "ftd", "13f")}
    assert written == [], written
    assert all(cached.values()), cached
    assert bulk_cache.is_cached(tmp_path / "notes" / "2026q1_notes.zip")

    print("\n=== SANITY CHECK: zips download writes (AC-002) ===")
    print(f"  {len(downloads)} archive(s) cached offline per source: {cached}")
    print("  OK: the download phase caches every source and writes no table")
