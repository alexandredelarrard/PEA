"""`download_13f_datasets`: every published 13F data set ends up cached and readable, offline (the SEC is stubbed)."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.data_extract.utils.common import bulk_cache
from src.data_extract.utils.zips import thirteen_f_datasets as f13ds

_MEMBERS = ("SUBMISSION.tsv", "COVERPAGE.tsv", "SUMMARYPAGE.tsv", "INFOTABLE.tsv")


def write_13f_zip(path: Path) -> None:
    """A minimal ZIP holding the four 13F data-set members (header row only)."""
    with zipfile.ZipFile(path, "w") as archive:
        for name in _MEMBERS:
            archive.writestr(name, "ACCESSION_NUMBER\n")


def _context(tmp_path: Path) -> Any:
    config = SimpleNamespace(local=SimpleNamespace(paths=SimpleNamespace(sec_13f_datasets="sec_13f_datasets")))
    return SimpleNamespace(config=config, paths={"DATA_STORE": tmp_path})


def _landing(names: list[str]) -> Any:
    links = "".join(f'<a href="/files/structureddata/data/form-13f-data-sets/{n}_form13f.zip">{n}</a>' for n in names)
    return lambda *args, **kwargs: SimpleNamespace(text=f"<html>{links}</html>")


@pytest.mark.parametrize("name", ["2013q2", "01jun2026-31aug2026"])
def test_data_set_urls_keep_both_sec_locations(name: str) -> None:
    data_set = f13ds.DataSet.parse(name)
    assert data_set.urls == (
        f"https://www.sec.gov/files/structureddata/data/form-13f-data-sets/{name}_form13f.zip",
        f"https://www.sec.gov/files/datastandardsinnovation/data/form-13f-data-sets/{name}_form13f.zip",
    )
    print(f"SANITY: {name} tries structureddata first and datastandardsinnovation as fallback.")


def test_every_published_data_set_is_cached_and_readable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    context = _context(tmp_path)
    cache = tmp_path / "sec_13f_datasets"
    cache.mkdir()
    write_13f_zip(cache / "2013q2_form13f.zip")  # cached and valid: never downloaded again
    (cache / "2013q3_form13f.zip").write_bytes(b"truncated")  # cached but corrupt: replaced
    published = ["2013q2", "2013q3", "01jun2024-31aug2024", "01sep2024-30nov2024"]  # the last one is not served yet
    monkeypatch.setattr(f13ds, "sec_get", _landing(published))
    asked: list[str] = []

    def download(context: Any, url: str, path: Path, **kwargs: Any) -> int:
        asked.append(url)
        if "01sep2024-30nov2024" in url:
            return 404
        write_13f_zip(path)
        return 200

    monkeypatch.setattr(bulk_cache, "download", download)
    with caplog.at_level(logging.ERROR):
        cached = f13ds.download_13f_datasets(context)

    assert cached == 3
    assert [d.name for d in f13ds.cached_data_sets(cache)] == ["2013q2", "2013q3", "01jun2024-31aug2024"]
    assert not any("2013q2" in url for url in asked), "a valid cached ZIP is not downloaded again"
    assert sum("2013q3" in url for url in asked) == 1, "the corrupt ZIP is downloaded once more"
    assert sum("01sep2024-30nov2024" in url for url in asked) == 2, "both SEC locations are tried for a missing window"
    assert any("01sep2024-30nov2024" in r.getMessage() for r in caplog.records if r.levelno == logging.ERROR)
    print("\n=== SANITY CHECK: 13F data-set download ===")
    print(f"  published {len(published)}: 1 kept, 1 corrupt re-downloaded, 1 new downloaded, 1 not served -> {cached} cached;")
    print("  the window still missing is logged as an error. OK")


def test_an_unreadable_landing_page_downloads_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(*args: Any, **kwargs: Any) -> Any:
        raise f13ds.TransientReadError("landing page down")

    def no_download(*args: Any, **kwargs: Any) -> int:
        raise AssertionError("no download without a landing page")

    monkeypatch.setattr(f13ds, "sec_get", failing)
    monkeypatch.setattr(bulk_cache, "download", no_download)
    assert f13ds.download_13f_datasets(_context(tmp_path)) == 0
    print("SANITY: an unreadable landing page downloads nothing and returns 0; the next night retries.")
