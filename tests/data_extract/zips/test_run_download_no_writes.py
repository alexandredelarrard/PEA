"""AC-002: `StepExtractZips.run_download()` caches archives and writes no table except `sec_company_tickers`.

A real (SQLite) store whose write methods are counted; the SEC is replaced by stubs, so no network is used.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.data_extract.transformers.step_extract_zips import StepExtractZips
from src.data_extract.utils.common import bulk_cache
from src.data_extract.utils.identity import sec_tickers
from src.data_extract.utils.zips import fetch_fails_to_deliver as ftd
from src.data_extract.utils.zips import fetch_financial_notes as notes
from src.data_extract.utils.zips import fetch_insider_transactions as insider
from src.data_extract.utils.zips import zip_archive

WRITES = ("append_tail", "bulk_seed", "delete", "drop", "ensure_columns", "insert", "replace", "save", "update")
PAYLOAD = {"fields": ["cik", "name", "ticker", "exchange"], "data": [[320193, "Apple Inc.", "AAPL", "Nasdaq"]]}


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

    for module in (insider, notes, ftd, zip_archive):
        monkeypatch.setattr(module, "ensure_zip", fake_zip)
    monkeypatch.setattr(notes, "_scrape_available_periods", lambda context: ["2026q1"])
    monkeypatch.setattr(sec_tickers, "sec_get", lambda context, url: SimpleNamespace(json=lambda: PAYLOAD))


def test_run_download_writes_only_the_sec_tickers_snapshot(sqlite_store, tmp_path, monkeypatch):
    downloads: list[str] = []
    _offline(monkeypatch, downloads)
    written = _counted(sqlite_store, monkeypatch)
    paths = SimpleNamespace(insider_transactions="insider", financial_notes="notes", fails_deliver="ftd")
    config = SimpleNamespace(data_extract=SimpleNamespace(years_history=1), local=SimpleNamespace(paths=paths))
    context = SimpleNamespace(store=sqlite_store, config=config, config_dir="configs", paths={"DATA_STORE": tmp_path})

    StepExtractZips(context, config).run_download()  # type: ignore[arg-type]

    cached = {directory: len(list((tmp_path / directory).glob("*.zip"))) for directory in ("insider", "notes", "ftd")}
    assert set(written) == {"sec_company_tickers"}, written
    assert all(cached.values()), cached
    assert bulk_cache.is_cached(tmp_path / "notes" / "2026q1_notes.zip")

    print("\n=== SANITY CHECK: zips download writes (AC-002) ===")
    print(f"  {len(downloads)} archive(s) cached offline per source: {cached}")
    print(f"  tables written: {sorted(set(written))} ({len(written)} write call(s))")
    print("  OK: the download phase caches every source and writes only the sec_company_tickers snapshot")
