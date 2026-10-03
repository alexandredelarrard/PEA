"""Cache, read and incremental-state helpers for the SEC bulk data sets.

Downloads stream to a `.part` file and rename on success; tab-separated zips are read through
`read_zip_tables`; `pending_periods` / `mark_processed` decide which cached periods a run re-parses;
`archive_available_at` / `stored_period_clock` give each archive its availability date.
"""

from __future__ import annotations

import json
import logging
import zipfile
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd

from src.constants.constants import MARKET_TIMEZONE
from src.context import Context
from src.data_extract.utils.common.incremental import stored_values
from src.data_store.schema import Table, name_of

__all__ = [
    "ZipRead",
    "archive_available_at",
    "cache_dir",
    "ensure_zip",
    "is_cached",
    "mark_processed",
    "pending_periods",
    "period_end",
    "quarter_periods",
    "read_zip_tables",
    "read_zip_text",
    "stored_period_clock",
]

_CHUNK = 1 << 20  # 1 MiB streaming chunks
_DEFAULT_TIMEOUT = 300  # seconds
_RELEASE_DAY = 12  # estimated release: this day of the month after the period end

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
    """True when `path` is a non-empty cached archive."""
    return path.exists() and path.stat().st_size > 0


def ensure_zip(
    context: Context,
    path: Path,
    urls: str | tuple[str, ...] | list[str],
    *,
    label: str,
    timeout: int = _DEFAULT_TIMEOUT,
    log: logging.Logger | None = None,
) -> Path | None:
    """Local path to a cached archive, downloading it once if absent.

    Streams to a `.part` file and renames only on success. `urls` may be several candidates tried
    in order. Returns None when no candidate serves the archive (normal for the newest period).
    """
    if is_cached(path):
        return path
    candidates = [urls] if isinstance(urls, str) else [u for u in urls if u]
    log = log or logger

    for url in candidates:
        try:
            response = context.sec_session.get(url, timeout=timeout, stream=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: download failed (%s): %s", label, url, exc)
            continue
        if response.status_code != 200:
            log.warning("%s: not available at %s (HTTP %s)", label, url, response.status_code)
            continue
        tmp = path.with_suffix(".part")
        with open(tmp, "wb") as fh:
            for chunk in response.iter_content(chunk_size=_CHUNK):
                fh.write(chunk)
        tmp.replace(path)
        return path
    return None


def _on_corrupt(path: Path, exc: Exception, policy: Literal["delete", "skip"], log: logging.Logger) -> None:
    """Log a corrupt archive; under "delete" remove it so the next run re-downloads it."""
    if policy == "skip":
        log.warning("%s: corrupt zip (%s) -> SKIPPED", path.name, exc)
        return
    log.warning("%s: corrupt zip (%s) -> deleting so it re-downloads next run", path.name, exc)
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
            kept = [chunk.loc[mask] for chunk in read if (mask := spec.keep(chunk)).any()]
            frame = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
    if spec.upper:
        frame.columns = [str(c).upper() for c in frame.columns]
    return frame


def read_zip_tables(
    path: Path, specs: Mapping[str, ZipRead], *, on_corrupt: Literal["delete", "skip"], log: logging.Logger
) -> dict[str, pd.DataFrame] | None:
    """Tab-separated members of one SEC bulk zip, keyed like `specs` (names match case-insensitively).

    Returns None for a corrupt archive (deleted under `on_corrupt="delete"`, kept under "skip"), `{}`
    when a required member is absent, and an empty frame for an absent optional member.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            names = {n.lower(): n for n in archive.namelist()}
            if any(spec.required and member.lower() not in names for member, spec in specs.items()):
                return {}
            return {
                member: _read_member(archive, names[member.lower()], spec) if member.lower() in names else pd.DataFrame()
                for member, spec in specs.items()
            }
    except zipfile.BadZipFile as exc:
        _on_corrupt(path, exc, on_corrupt, log)
        return None


def read_zip_text(path: Path, *, encoding: str = "latin-1", log: logging.Logger | None = None) -> str | None:
    """The first member of a zip as decoded text; a corrupt archive is deleted so it re-downloads."""
    log = log or logger
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if not names:
                return None
            with archive.open(names[0]) as handle:
                return handle.read().decode(encoding, errors="replace")
    except zipfile.BadZipFile as exc:
        _on_corrupt(path, exc, "delete", log)
        return None
    except Exception as exc:  # noqa: BLE001
        log.warning("%s: unreadable archive (%s)", path.name, exc)
        return None


def _sidecar(cache: Path, table: Table | str) -> Path:
    return cache / f"{name_of(table)}_universe.json"


def _processed_scope(cache: Path, table: Table | str) -> set[str]:
    """The scope a bulk table was last built against; empty when the sidecar is absent or unreadable."""
    path = _sidecar(cache, table)
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8")).get("universe", []))
    except Exception:  # noqa: BLE001
        return set()


def mark_processed(cache: Path, table: Table | str, scope: Collection[str]) -> None:
    """Record the scope a bulk table was built against, so a converged re-run skips stored periods."""
    payload = {"universe": sorted(scope), "saved": datetime.now(UTC).date().isoformat()}
    _sidecar(cache, table).write_text(json.dumps(payload), encoding="utf-8")


def pending_periods(
    context: Context,
    cache: Path,
    table: Table | str | Sequence[Table | str],
    periods: Sequence[str],
    scope: Collection[str],
    *,
    reparse: bool = False,
    column: str = "period",
) -> list[str]:
    """The `periods` a bulk fetcher must parse.

    All of them on `reparse` or when `scope` gained members since `mark_processed`; otherwise those
    with no row stored under `column`. Several tables union their stored periods and share the
    sidecar of the first.
    """
    tables = [table] if isinstance(table, Table | str) else list(table)
    added = set(scope) - _processed_scope(cache, tables[0])
    if reparse or added:
        reason = "reparse" if reparse else f"{len(added)} new scope member(s)"
        logger.info("%s: %s -> parsing all %d period(s), cached zips are not re-downloaded", name_of(tables[0]), reason, len(periods))
        return list(periods)
    stored = stored_values(context, tables, column)
    return [period for period in periods if period not in stored]


def period_end(tag: str) -> date:
    """Last calendar day covered by a `YYYYqN` or `YYYY_MM` archive tag."""
    month = int(tag[-2:]) if "_" in tag else int(tag[-1]) * 3
    return pd.Period(year=int(tag[:4]), month=month, freq="M").end_time.date()


def archive_available_at(end: date, path: Path, *, observed_from: date, downloaded: bool) -> date | None:
    """When an archive covering a period ending `end` became available.

    Before `observed_from` it is estimated as the 12th of the following month (a weekend rolls to
    Monday); afterwards it is the New York date of this run's download, else of the cached file's
    modification time (None when the file cannot be read).
    """
    if end < observed_from:
        release = (end.replace(day=1) + timedelta(days=32)).replace(day=_RELEASE_DAY)
        return release + timedelta(days=7 - release.weekday() if release.weekday() >= 5 else 0)
    if downloaded:
        return datetime.now(MARKET_TIMEZONE).date()
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=MARKET_TIMEZONE).date()
    except OSError:
        return None


def stored_period_clock(context: Context, tables: Sequence[Table], period: str, *, column: str = "period") -> date | None:
    """The single `available_at` already stored for `period` across `tables`; raises when they disagree."""
    values: set[date] = set()
    for table in tables:
        if not {column, "available_at"} <= set(context.store.columns(table)):
            continue
        raw = context.store.distinct(table, "available_at", where={column: period})
        values.update(pd.to_datetime(pd.Series(raw, dtype=object), errors="coerce").dropna().dt.date)
    if len(values) > 1:
        raise ValueError(f"{'/'.join(name_of(t) for t in tables)} {period}: conflicting stored available_at values: {sorted(values)}")
    return next(iter(values), None)


def quarter_periods(years_history: int, first_year: int, today: pd.Timestamp | None = None) -> list[str]:
    """`YYYYqN` tags covering the last `years_history` years, never before `first_year`."""
    now = (today or pd.Timestamp.today()).normalize()
    return [f"{year}q{q}" for year in range(now.year - years_history, now.year + 1) if year >= first_year for q in range(1, 5)]
