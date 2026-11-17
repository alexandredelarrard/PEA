"""Cache, read and incremental-state helpers for the SEC bulk data sets.

Downloads go through `sec_io.download` (a `.part` file renamed on success); tab-separated zips are read through
`read_zip_tables`; `cached_periods` lists the archives on disk; `archive_available_at` /
`stored_period_clock` give each archive its availability date. Which periods a run parses is
decided by `resume.archive_worklist` from the stored rows alone.
"""

from __future__ import annotations

import logging
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from omegaconf import DictConfig

from src.constants.constants import MARKET_TIMEZONE
from src.context import Context
from src.data_extract.utils.common.sec_io import TransientReadError, download
from src.data_store.schema import Table, name_of
from src.utils.publication import archive_release_date, period_end
from src.utils.window import Window

__all__ = [
    "ZipRead",
    "archive_available_at",
    "cache_dir",
    "cached_periods",
    "ensure_zip",
    "is_cached",
    "quarter_periods",
    "read_zip_tables",
    "read_zip_text",
    "stored_period_clock",
]

DEFAULT_TIMEOUT = 300  # seconds

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ZipRead:
    """How to read one tab-separated zip member as str columns.

    `usecols` names the columns kept (compared upper-cased when `upper`, which also upper-cases the
    frame's columns); `keep` is a row filter applied to each `chunksize`-row chunk (the two go
    together); an absent member fails the read only when `required`.
    """

    usecols: frozenset[str] | None = None
    keep: Callable[[pd.DataFrame], pd.Series] | None = None
    chunksize: int | None = None
    required: bool = True
    upper: bool = False
    skip_bad_lines: bool = False

    def __post_init__(self) -> None:
        if (self.keep is None) != (self.chunksize is None):
            raise ValueError("ZipRead: keep and chunksize must be given together")


def cache_dir(context: Context, key: str) -> Path:
    """The (created) DATA_STORE sub-directory a bulk data set caches its archives in."""
    directory = context.paths["DATA_STORE"] / key
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def is_cached(path: Path) -> bool:
    """Recognize a complete ZIP container without decoding or decompressing its payload."""
    return path.is_file() and zipfile.is_zipfile(path)


def ensure_zip(
    context: Context,
    path: Path,
    urls: str | tuple[str, ...] | list[str],
    *,
    label: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> Path | None:
    """Local path to a cached archive, downloading it once if absent.

    Downloads through `sec_io.download` (retried, rate-limited, `.part` renamed only on success).
    `urls` may be several candidates tried in order. Returns None when no candidate serves the
    archive (normal for the newest period) or every candidate failed after the retry policy.
    Warn only when all candidates fail, retaining each failure's URL and reason.
    """
    if is_cached(path):
        return path
    candidates = [urls] if isinstance(urls, str) else [u for u in urls if u]
    failures: list[str] = []

    for url in candidates:
        try:
            status = download(context, url, path, timeout=timeout)
        except TransientReadError as exc:
            failures.append(f"{url}: failed after retries ({exc})")
            continue
        if status != 200:
            failures.append(f"{url}: HTTP {status}")
            continue
        if not is_cached(path):
            failures.append(f"{url}: invalid ZIP container")
            continue
        return path
    if failures:
        logger.warning("%s: ZIP unavailable at every download location (%s)", label, "; ".join(failures))
    return None


def _on_corrupt(path: Path, exc: Exception, policy: Literal["delete", "skip"]) -> None:
    """Log a corrupt archive; under "delete" remove it so the next run re-downloads it."""
    if policy == "skip":
        logger.warning("%s: corrupt zip (%s) -> SKIPPED", path.name, exc)
        return
    logger.warning("%s: corrupt zip (%s) -> deleting so it re-downloads next run", path.name, exc)
    path.unlink(missing_ok=True)


def _read_member(archive: zipfile.ZipFile, name: str, spec: ZipRead) -> pd.DataFrame:
    """One member as str columns, row-filtered chunk by chunk when the spec has `keep`."""
    wanted = spec.usecols
    usecols = None if wanted is None else (lambda c: (str(c).upper() if spec.upper else c) in wanted)
    with archive.open(name) as handle:
        read = pd.read_csv(
            handle,
            sep="\t",
            dtype=str,
            low_memory=False,
            usecols=usecols,
            chunksize=spec.chunksize,
            on_bad_lines="skip" if spec.skip_bad_lines else "error",
        )
        if spec.keep is None:
            frame = read
        else:
            kept: list[pd.DataFrame] = []
            empty = pd.DataFrame()
            for chunk in read:
                empty = chunk.iloc[:0]
                mask = spec.keep(chunk)
                if mask.any():
                    kept.append(chunk.loc[mask])
            frame = pd.concat(kept, ignore_index=True) if kept else empty
    if spec.upper:
        frame.columns = [str(c).upper() for c in frame.columns]
    return frame


def read_zip_tables(path: Path, specs: Mapping[str, ZipRead], *, on_corrupt: Literal["delete", "skip"]) -> dict[str, pd.DataFrame] | None:
    """Tab-separated members of one SEC bulk zip, keyed like `specs` (names match case-insensitively).

    Returns None for a corrupt archive (deleted under `on_corrupt="delete"`, kept under "skip"), `{}`
    when a required member is absent, and an empty frame for an absent optional member.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            names = {n.lower(): n for n in archive.namelist() if not n.endswith("/")}
            for member in specs:
                if member.lower() in names:
                    continue
                matches = [n for n in archive.namelist() if n.rsplit("/", 1)[-1].lower() == member.lower()]
                if len(matches) > 1:
                    logger.error("%s: ambiguous ZIP member %s", path.name, member)
                    return {}
                if matches:
                    names[member.lower()] = matches[0]
            if any(spec.required and member.lower() not in names for member, spec in specs.items()):
                return {}
            return {
                member: _read_member(archive, names[member.lower()], spec) if member.lower() in names else pd.DataFrame()
                for member, spec in specs.items()
            }
    except zipfile.BadZipFile as exc:
        _on_corrupt(path, exc, on_corrupt)
        return None


def read_zip_text(path: Path, *, encoding: str = "latin-1") -> str | None:
    """The first member of a zip as decoded text; a corrupt archive is deleted so it re-downloads."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if not names:
                return None
            with archive.open(names[0]) as handle:
                return handle.read().decode(encoding, errors="replace")
    except zipfile.BadZipFile as exc:
        _on_corrupt(path, exc, "delete")
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s: unreadable archive (%s)", path.name, exc)
        return None


def cached_periods(cache: Path, *, prefix: str = "", suffix: str = ".zip") -> set[str]:
    """Period tags of the non-empty archives cached in `cache`, named `{prefix}{period}{suffix}`."""
    return {path.name[len(prefix) : -len(suffix)] for path in cache.glob(f"{prefix}*{suffix}") if is_cached(path)}


def archive_available_at(period: str, path: Path, *, policy: DictConfig, downloaded: bool) -> date | None:
    """Configured release estimate for older archives; otherwise preserve observed download/file dates."""
    if period_end(period) < date.fromisoformat(str(policy.observed_from)):
        return archive_release_date(policy, period).date()
    if downloaded:
        return datetime.now(MARKET_TIMEZONE).date()
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=MARKET_TIMEZONE).date()
    except OSError:
        return None


def stored_period_clock(context: Context, tables: Sequence[Table], period: str) -> date | None:
    """The single `available_at` already stored for `period` across `tables`; raises when they disagree."""
    values: set[date] = set()
    for table in tables:
        if not {"period", "available_at"} <= set(context.store.columns(table)):
            continue
        raw = context.store.distinct(table, "available_at", where={"period": period})
        values.update(pd.to_datetime(pd.Series(raw, dtype=object), errors="coerce").dropna().dt.date)
    if len(values) > 1:
        raise ValueError(f"{'/'.join(name_of(t) for t in tables)} {period}: conflicting stored available_at values: {sorted(values)}")
    return next(iter(values), None)


def quarter_periods(window: Window) -> list[str]:
    """Closed quarter units whose source-period end is within explicit configured dates."""
    quarters = pd.period_range(window.start, window.end, freq="Q")
    return [f"{quarter.year}q{quarter.quarter}" for quarter in quarters if window.start <= quarter.end_time.normalize() <= window.end]
