"""Local copy of SEC's quarterly EDGAR filing index (`full-index/{y}/QTR{q}/master.gz`).

One Parquet file per quarter, `{y}Q{q}.parquet`, under `<DATA_STORE>/<paths.sec_edgar_index>`, with
every CIK's filings of `INDEX_FORMS`: `cik` (int64), `company`, `form`, `filed` (date), `accession`.
`refresh` downloads a missing closed quarter once, the current quarter on every call, and the
previous quarter again during the first `_PREVIOUS_QUARTER_DAYS` days of a quarter. Downloads go
through `sec_io`; a quarter file is replaced atomically. `entries` reads the cache with filters.
"""

from __future__ import annotations

import gzip
import logging
import os
import re
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol, cast

import edgar
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.context import Context
from src.data_extract.utils.common import sec_io
from src.data_extract.utils.common.bulk_cache import cache_dir
from src.data_store.schema import Resume, resume_tables
from src.utils.string import pad_cik_series

logger = logging.getLogger(__name__)

INDEX_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.gz"
#: The forms kept from the index: exactly the union of the registered extraction contracts.
INDEX_FORMS: frozenset[str] = frozenset(form for table in resume_tables() for form in cast(Resume, table.resume).forms)
#: Days into a quarter during which the previous quarter is downloaded again (late index lines).
_PREVIOUS_QUARTER_DAYS = 10
_ENCODING = "latin-1"
_SCHEMA = pa.schema([("cik", pa.int64()), ("company", pa.string()), ("form", pa.string()), ("filed", pa.date32()), ("accession", pa.string())])
_ROW_GROUP = 65_536


class _IndexRuntime(Protocol):
    _edgar_index_inventory: dict[str, pd.Timestamp]
    _edgar_index_versions: dict[str, tuple[int, int]]


@dataclass(frozen=True, order=True)
class Quarter:
    """One calendar quarter of the index."""

    year: int
    quarter: int

    @classmethod
    def of(cls, date: pd.Timestamp) -> Quarter:
        stamp = pd.Timestamp(date)
        return cls(stamp.year, (stamp.month - 1) // 3 + 1)

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(year=self.year, month=3 * self.quarter - 2, day=1)

    @property
    def end(self) -> pd.Timestamp:
        return (self.start + pd.offsets.QuarterEnd(0)).normalize()

    @property
    def name(self) -> str:
        return f"{self.year}Q{self.quarter}"

    def previous(self) -> Quarter:
        return Quarter(self.year, self.quarter - 1) if self.quarter > 1 else Quarter(self.year - 1, 4)

    def next(self) -> Quarter:
        return Quarter(self.year, self.quarter + 1) if self.quarter < 4 else Quarter(self.year + 1, 1)


def quarters_between(first: pd.Timestamp, last: pd.Timestamp) -> list[Quarter]:
    """Every quarter in the window, floored at the earliest registered EDGAR source start."""
    source_start = min(
        resume.source_start for table in resume_tables() if (resume := cast(Resume, table.resume)).forms and resume.source_start is not None
    )
    quarter = max(Quarter.of(first), Quarter.of(pd.Timestamp(source_start)))
    out: list[Quarter] = []
    while quarter <= Quarter.of(last):
        out.append(quarter)
        quarter = quarter.next()
    return out


def index_dir(context: Context) -> Path:
    """The (created) cache directory."""
    return cache_dir(context, str(context.config.local.paths.sec_edgar_index))


def _parse_master_unit(raw: bytes) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Account for all forms/lines and the source's own declared public-calendar boundary."""
    decoded = (gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw).decode(_ENCODING)
    lines = decoded.splitlines()
    header = "CIK|Company Name|Form Type|Date Filed|Filename"
    boundary = re.search(r"^Last Data Received:\s*(.+)$", decoded, re.MULTILINE)
    if header not in lines or boundary is None:
        raise ValueError("EDGAR master index lacks its source header/boundary")
    through = pd.Timestamp(boundary.group(1)).normalize()
    body = lines[lines.index(header) + 1 :]
    if not body or not body[0].strip() or set(body[0].strip()) != {"-"}:
        raise ValueError("EDGAR master index lacks its complete header separator")
    rows = [line.split("|") for line in body[1:] if line.strip()]
    if any(len(fields) != 5 or not fields[0].isdigit() or not fields[2] or not fields[4].endswith(".txt") for fields in rows):
        raise ValueError("EDGAR master index has an unaccounted source line")
    df = pd.DataFrame(rows, columns=["cik", "company", "form", "filed", "filename"])
    dates = pd.to_datetime(df["filed"], format="%Y-%m-%d", errors="raise")
    if dates.gt(through).any() or not (dates.dt.to_period("Q") == through.to_period("Q")).all():
        raise ValueError("EDGAR master index filing exceeds its declared quarter/boundary")
    df = df.loc[df["form"].isin(INDEX_FORMS)]
    result = pd.DataFrame(
        {
            "cik": df["cik"].astype("int64"),
            "company": df["company"].astype(object),
            "form": df["form"].astype(object),
            "filed": pd.to_datetime(df["filed"], format="%Y-%m-%d").dt.date,
            "accession": df["filename"].str.rsplit("/", n=1).str[-1].str.removesuffix(".txt").astype(object),
        }
    ).sort_values(["cik", "filed", "accession"], ignore_index=True)
    return result, through


def parse_master(raw: bytes) -> pd.DataFrame:
    """Typed selected forms, only after validating the complete source unit."""
    return _parse_master_unit(raw)[0]


def _write_atomic(df: pd.DataFrame, path: Path) -> tuple[int, int]:
    """Write `df` to `path` through a temporary file and `os.replace`, so a reader never sees a partial file."""
    with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".part", delete=False) as handle:
        tmp = Path(handle.name)
    try:
        pq.write_table(pa.Table.from_pandas(df, schema=_SCHEMA, preserve_index=False), tmp, row_group_size=_ROW_GROUP)
        version = (tmp.stat().st_mtime_ns, tmp.stat().st_size)
        os.replace(tmp, path)
        return version
    finally:
        tmp.unlink(missing_ok=True)


def _download_quarter(context: Context, quarter: Quarter, directory: Path) -> int | None:
    """Download, parse and cache one quarter; its row count, or None when SEC served no file."""
    gz_path = directory / f"{quarter.name}.master.gz"
    with NamedTemporaryFile(dir=directory, prefix=gz_path.name + ".", suffix=".part", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        status = sec_io.download(context, INDEX_URL.format(year=quarter.year, quarter=quarter.quarter), temporary)
        if status != 200:
            context.log.warning("edgar index: %s not served (HTTP %d)", quarter.name, status)
            return None
        df, through = _parse_master_unit(temporary.read_bytes())
        if Quarter.of(through) != quarter:
            raise ValueError("EDGAR master index source boundary does not match the requested quarter")
        version = _write_atomic(df, directory / f"{quarter.name}.parquet")
        os.replace(temporary, gz_path)
        _record_inventory(context, quarter, through, version)
        return len(df)
    finally:
        temporary.unlink(missing_ok=True)


def _record_inventory(context: Context, quarter: Quarter, through: pd.Timestamp, version: tuple[int, int]) -> None:
    """Bind a validated original's boundary to this run's derived cache version."""
    runtime = cast(_IndexRuntime, context)
    inventory = getattr(context, "_edgar_index_inventory", {})
    inventory[quarter.name] = through
    runtime._edgar_index_inventory = inventory
    versions = getattr(context, "_edgar_index_versions", {})
    versions[quarter.name] = version
    runtime._edgar_index_versions = versions


def _is_due(quarter: Quarter, current: Quarter, as_of: pd.Timestamp, path: Path, rebuild: bool) -> bool:
    """Refresh originals weekly for post-acceptance revisions, plus the current/early prior quarter."""
    if rebuild or not path.exists() or quarter == current:
        return True
    if quarter == current.previous() and (as_of - current.start).days < _PREVIOUS_QUARTER_DAYS:
        return True
    saturday = as_of - pd.Timedelta(days=(as_of.weekday() - 5) % 7)
    return pd.Timestamp.fromtimestamp(path.stat().st_mtime).normalize() < saturday.normalize()


def _validated_cached_quarter(context: Context, quarter: Quarter, directory: Path) -> None:
    """Rebuild the selected cache from the same original that establishes this run's boundary."""
    original = directory / f"{quarter.name}.master.gz"
    df, boundary = _parse_master_unit(original.read_bytes())
    if Quarter.of(boundary) != quarter:
        raise ValueError("Cached original boundary does not match its quarter")
    version = _write_atomic(df, directory / f"{quarter.name}.parquet")
    _record_inventory(context, quarter, boundary, version)


def refresh(context: Context, as_of: pd.Timestamp, years_history: int, *, rebuild: bool = False) -> dict[str, int]:
    """Bring the cache up to `as_of` over `years_history`; rows written per downloaded quarter.

    A transient SEC failure on one quarter is logged and leaves that quarter's previous file in place."""
    as_of = pd.Timestamp(as_of).normalize()
    directory = index_dir(context)
    current = Quarter.of(as_of)
    written: dict[str, int] = {}
    runtime = cast(_IndexRuntime, context)
    runtime._edgar_index_inventory = {}
    runtime._edgar_index_versions = {}
    for quarter in quarters_between(as_of - pd.DateOffset(years=years_history), as_of):
        original = directory / f"{quarter.name}.master.gz"
        try:
            if _is_due(quarter, current, as_of, original, rebuild):
                n_rows = _download_quarter(context, quarter, directory)
            else:
                _validated_cached_quarter(context, quarter, directory)
                n_rows = None
        except (sec_io.TransientReadError, ValueError, OSError, EOFError) as exc:
            context.log.warning("edgar index: %s not refreshed (%s)", quarter.name, exc)
            continue
        if n_rows is not None:
            written[quarter.name] = n_rows
    context.log.info("edgar index: %d quarter(s) downloaded through %s", len(written), as_of.date())
    return written


def inventory_complete_through(context: Context, as_of: pd.Timestamp, source_start: str) -> pd.Timestamp | None:
    """Every required original index must prove its source calendar; cache presence is insufficient."""
    inventory = getattr(context, "_edgar_index_inventory", {})
    versions = getattr(context, "_edgar_index_versions", {})
    directory = index_dir(context)
    business_day = pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())
    public_day = pd.Timestamp(as_of).normalize() - business_day
    quarters = quarters_between(pd.Timestamp(source_start), public_day)
    for quarter in quarters:
        boundary = inventory.get(quarter.name)
        required = pd.date_range(quarter.start, min(quarter.end, public_day), freq=business_day)
        if boundary is None or Quarter.of(boundary) != quarter or required.empty or boundary < required[-1]:
            return None
        path = directory / f"{quarter.name}.parquet"
        if not path.exists() or versions.get(quarter.name) != (path.stat().st_mtime_ns, path.stat().st_size):
            return None
    return min(pd.Timestamp(inventory[quarters[-1].name]), public_day) if quarters else None


def quarter_counts(context: Context) -> dict[str, int]:
    """Rows per cached quarter, read from the Parquet footers."""
    return {path.stem: pq.ParquetFile(path).metadata.num_rows for path in sorted(index_dir(context).glob("*.parquet"))}


def entries(context: Context, ciks: Collection[str], forms: Collection[str], since: pd.Timestamp | None = None) -> pd.DataFrame:
    """Cached index rows for `ciks` (padded or not) and `forms`, filed on or after `since`.

    Columns: `cik` (10-digit string), `company`, `form`, `filed` (Timestamp), `accession`; sorted by date."""
    directory = index_dir(context)
    first = Quarter.of(since) if since is not None else None
    files = [path for path in sorted(directory.glob("*.parquet")) if first is None or _file_quarter(path) >= first]
    columns = ["cik", "company", "form", "filed", "accession"]
    if not files or not ciks or not forms:
        return pd.DataFrame(columns=columns)
    expression = pc.field("cik").isin([int(c) for c in ciks]) & pc.field("form").isin(list(forms))
    if since is not None:
        expression &= pc.field("filed") >= pa.scalar(pd.Timestamp(since).date(), pa.date32())
    df = ds.dataset([str(path) for path in files], format="parquet", schema=_SCHEMA).to_table(filter=expression).to_pandas()
    df["cik"] = pad_cik_series(df["cik"])
    df["filed"] = pd.to_datetime(df["filed"])
    return df[columns].sort_values(["filed", "accession"], ignore_index=True)


def _file_quarter(path: Path) -> Quarter:
    year, quarter = path.stem.split("Q")
    return Quarter(int(year), int(quarter))


def index_filing(cik: object, company: object, form: object, filed: object, accession: object) -> edgar.Filing:
    """An edgartools `Filing` for one index row's cells; nothing is read until a property needs it."""
    filing_date = pd.Timestamp(str(filed)).strftime("%Y-%m-%d")
    return edgar.Filing(cik=int(str(cik)), company=str(company), form=str(form), filing_date=filing_date, accession_no=str(accession))
