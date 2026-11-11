"""Pieces the period-archive ingests share: the run scope and the one `ensure_zip -> clock -> skip` block."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.bulk_cache import archive_available_at, ensure_zip, is_cached
from src.data_extract.utils.common.resume import period_end


@dataclass(frozen=True)
class ZipScope:
    """One ingest's tickers (the `-t` subset or the universe), the run date and `full` re-parse."""

    tickers: list[str]
    run_date: pd.Timestamp
    full: bool


@dataclass(frozen=True)
class ArchiveSource:
    """Where one data set's period archive is cached and served, and when its clock is observed rather than estimated."""

    label: str
    file_template: str
    url_template: str
    observed_from: date
    timeout: int


def ensure_archive(context: Context, cache: Path, source: ArchiveSource, period: str) -> tuple[Path, date | None] | None:
    """The cached archive of `period` (downloaded once when absent) and its availability clock; None when no zip is served."""
    path = cache / source.file_template.format(period=period)
    downloaded = not is_cached(path)
    url = source.url_template.format(period=period, quarter=period)
    found = ensure_zip(context, path, url, label=f"{source.label} {period}", timeout=source.timeout)
    if found is None:
        return None
    return found, archive_available_at(period_end(period), found, observed_from=source.observed_from, downloaded=downloaded)
