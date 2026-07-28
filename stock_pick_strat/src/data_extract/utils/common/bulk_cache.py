"""
bulk_cache.py  (src/data_extract/utils/common/bulk_cache.py)
------------------------------------------------------------
Shared cache-and-download scaffolding for the SEC BULK data sets, and for the earnings-
call transcript cache (behavioral), which turned out to need the identical `cache_dir`
shape even though it isn't a periodic SEC zip.

Six fetchers pull a large periodic zip (or, for earnings calls, a scraped-HTML tree) and
parse it incrementally -- 13F holdings, insider transactions (Forms 3/4/5), the
Financial Statement Data Sets, the Financial Statement & Notes Data Sets,
fails-to-deliver, and Motley Fool transcripts. Each had grown its OWN
`_cache_dir` / `_ensure_zip` / `_read_zip` / `_ingested_periods` / `_quarters`, all
structurally identical and differing only in the filename pattern, the URL, the timeout
and the log label. Six copies meant six places to fix a partial-download bug, five
hardcoded `User-Agent` strings (two still carrying the placeholder `contact@example.com`,
which SEC blocks), and -- the bug this consolidation surfaced -- callers that keep the
returned directory in a variable literally named `cache_dir`, shadowing the function of
the same name and passing IT (not the Path) into `load_processed_universe` /
`save_processed_universe`, which then raised `TypeError: unsupported operand type(s)
for /: 'function' and 'str'`. The convention going forward: this module owns the name
`cache_dir`; every caller stores its result in a variable named `cache`.

Everything here is deliberately IO-only and side-effect-explicit: the parsing of each
data set stays in its own fetcher, because that is the part that genuinely differs.
"""
from __future__ import annotations

import logging
import zipfile
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import requests
from sqlalchemy import inspect, text

from src.context import Context
from src.data_extract.utils.common.sec_utils import _sec_headers

__all__ = ["cache_dir", "ensure_zip", "read_zip_member", "read_zip_members",
           "read_zip_text", "ingested_periods", "quarter_periods"]

_CHUNK = 1 << 20                 # 1 MiB streaming chunks
_DEFAULT_TIMEOUT = 300           # seconds; the notes zips are ~380 MB


def cache_dir(context: Context, key: str) -> Path:
    """The (created) directory a bulk data set caches its archives in.

    `key` is either a registered `context.paths` key (e.g. SEC_13F_INSIDERS_DIR) or a
    plain sub-directory name under DATA_STORE. Each data set keeps its own directory so
    the multi-hundred-MB zips never mix with the companyfacts JSON cache."""
    path = context.paths.get(key)
    directory = Path(path) if path is not None else context.paths["DATA_STORE"] / key
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_zip(path: Path, urls: str | tuple[str, ...] | list[str], *, label: str,
               timeout: int = _DEFAULT_TIMEOUT,
               log: logging.Logger | None = None) -> Path | None:
    """Local path to a cached archive, downloading it once if absent.

    Streams to a `.part` file and renames only on success, so an interrupted download is
    never mistaken for a cache hit. `urls` may be several candidates tried in order (the
    fails-to-deliver archive moved between two SEC paths). Returns None when the period
    is simply not published yet -- that is normal for the newest quarter, not an error.
    """
    if path.exists() and path.stat().st_size > 0:
        return path
    candidates = [urls] if isinstance(urls, str) else [u for u in urls if u]
    log = log or logging.getLogger(__name__)

    for url in candidates:
        try:
            response = requests.get(url, headers=_sec_headers(), timeout=timeout,
                                    stream=True)
        except Exception as exc:                                    # noqa: BLE001
            log.warning("%s: download failed (%s): %s", label, url, exc)
            continue
        if response.status_code != 200:
            log.info("%s: not available at %s (HTTP %s)", label, url,
                     response.status_code)
            continue
        tmp = path.with_suffix(".part")
        with open(tmp, "wb") as fh:
            for chunk in response.iter_content(chunk_size=_CHUNK):
                fh.write(chunk)
        tmp.replace(path)
        return path
    return None


def _drop_corrupt(path: Path, exc: Exception, log: logging.Logger | None) -> None:
    """A truncated / corrupt archive is DELETED so the next run re-downloads it.

    Without this a bad cache entry is permanent: `ensure_zip` treats any non-empty file
    as a cache hit, so the period silently returns None for ever. Two of the six bulk
    fetchers had already grown this self-heal privately; all of them get it now."""
    (log or logging.getLogger(__name__)).warning(
        "%s: corrupt zip (%s) -> deleting so it re-downloads next run", path.name, exc)
    path.unlink(missing_ok=True)


def read_zip_member(path: Path, member: str, *, log: logging.Logger | None = None,
                    **read_csv_kwargs) -> pd.DataFrame | None:
    """One member of a zip read as a DataFrame (tab-separated by default -- every SEC
    bulk set ships .tsv). None when the archive or the member is unreadable, so a single
    corrupt period never aborts a multi-year ingest."""
    kwargs = {"sep": "\t", "dtype": str, "low_memory": False, **read_csv_kwargs}
    try:
        with zipfile.ZipFile(path) as archive:
            names = {n.lower(): n for n in archive.namelist()}
            actual = names.get(member.lower())
            if actual is None:
                return None
            with archive.open(actual) as handle:
                return pd.read_csv(handle, **kwargs)
    except zipfile.BadZipFile as exc:
        _drop_corrupt(path, exc, log)
        return None
    except Exception as exc:                                        # noqa: BLE001
        (log or logging.getLogger(__name__)).warning(
            "%s: unreadable member %s (%s)", path.name, member, exc)
        return None


def read_zip_members(path: Path, members: Sequence[str], *,
                     log: logging.Logger | None = None,
                     **read_csv_kwargs) -> dict[str, pd.DataFrame] | None:
    """Several members of ONE archive, keyed by the requested name -- opened once rather
    than once per member. All-or-nothing: None when the archive is unreadable OR any
    requested member is absent, because a caller that joins two tsvs (13F
    SUBMISSION + INFOTABLE) cannot do anything useful with just one of them."""
    kwargs = {"sep": "\t", "dtype": str, "low_memory": False, **read_csv_kwargs}
    try:
        with zipfile.ZipFile(path) as archive:
            names = {n.lower(): n for n in archive.namelist()}
            if any(m.lower() not in names for m in members):
                return None
            out: dict[str, pd.DataFrame] = {}
            for m in members:
                with archive.open(names[m.lower()]) as handle:
                    out[m] = pd.read_csv(handle, **kwargs)
            return out
    except zipfile.BadZipFile as exc:
        _drop_corrupt(path, exc, log)
        return None
    except Exception as exc:                                        # noqa: BLE001
        (log or logging.getLogger(__name__)).warning(
            "%s: unreadable members %s (%s)", path.name, list(members), exc)
        return None


def read_zip_text(path: Path, *, encoding: str = "latin-1",
                  log: logging.Logger | None = None) -> str | None:
    """The FIRST member of a zip as decoded text -- for the archives that hold one
    unnamed pipe/CSV file (fails-to-deliver). Undecodable bytes are replaced rather than
    raising: a single bad character must not drop a whole period."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if not names:
                return None
            with archive.open(names[0]) as handle:
                return handle.read().decode(encoding, errors="replace")
    except zipfile.BadZipFile as exc:
        _drop_corrupt(path, exc, log)
        return None
    except Exception as exc:                                        # noqa: BLE001
        (log or logging.getLogger(__name__)).warning(
            "%s: unreadable archive (%s)", path.name, exc)
        return None


def ingested_periods(context: Context, tables: str | Sequence[str],
                     column: str = "period") -> set[str]:
    """Distinct source-period tags already stored, so an incremental re-run skips them.

    Handles the three shapes the callers needed separately before: a single table
    (fails-to-deliver), a table whose `period` column may predate the feature and be
    absent (13F -- reflected first, so a stale table degrades to "nothing ingested"
    rather than raising), and a UNION across two sibling tables (the notes num/text
    pair). Empty set on the first run."""
    store = context.store
    names = [tables] if isinstance(tables, str) else list(tables)
    done: set[str] = set()
    for table in names:
        if not store.exists(table):
            continue
        if column not in {c["name"] for c in inspect(store.engine).get_columns(table)}:
            continue
        with store.engine.connect() as conn:
            got = pd.read_sql(text(f'SELECT DISTINCT "{column}" FROM "{table}"'), conn)
        done |= set(got[column].dropna().astype(str))
    return done


def quarter_periods(years_history: int, first_year: int,
                    today: pd.Timestamp | None = None) -> list[str]:
    """`['2015q1', '2015q2', ...]` covering the requested window, never starting before
    `first_year` (the year the data set itself begins)."""
    now = (today or pd.Timestamp.today()).normalize()
    return [f"{year}q{q}"
            for year in range(now.year - years_history, now.year + 1)
            if year >= first_year
            for q in range(1, 5)]
