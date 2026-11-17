"""Local copy of SEC's quarterly EDGAR filing index (`full-index/{y}/QTR{q}/master.gz`).

One Parquet file per quarter, `{y}Q{q}.parquet`, under `<DATA_STORE>/<paths.sec_edgar_index>`, with
every CIK's filings of `INDEX_FORMS`: `cik` (int64), `company`, `form`, `filed` (date), `accession`.
`refresh` downloads a missing closed quarter, a closed quarter again once a week (Saturday), the
current quarter on every call, and the previous quarter during its configured recheck period. `refreshed_today`
tells, from the cache files, whether today's refresh already covered a window. Downloads go
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
from typing import cast

import edgar
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.constants.constants import MARKET_TIMEZONE
from src.context import Context
from src.data_extract.utils.common import sec_io
from src.data_extract.utils.common.bulk_cache import cache_dir
from src.data_store.schema import Resume, resume_tables
from src.utils.string import pad_cik_series
from src.utils.window import Offset, Window, shift_date

logger = logging.getLogger(__name__)

INDEX_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.gz"
#: The forms kept from the index: exactly the union of the registered extraction contracts.
INDEX_FORMS: frozenset[str] = frozenset(form for table in resume_tables() for form in cast(Resume, table.resume).forms)
_ENCODING = "latin-1"
_SCHEMA = pa.schema([("cik", pa.int64()), ("company", pa.string()), ("form", pa.string()), ("filed", pa.date32()), ("accession", pa.string())])
_ROW_GROUP = 65_536


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
    """Every quarter intersecting the caller's already bounded filing window."""
    quarter = Quarter.of(first)
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


def _write_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write `df` to `path` through a temporary file and `os.replace`, so a reader never sees a partial file."""
    with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".part", delete=False) as handle:
        tmp = Path(handle.name)
    try:
        pq.write_table(pa.Table.from_pandas(df, schema=_SCHEMA, preserve_index=False), tmp, row_group_size=_ROW_GROUP)
        os.replace(tmp, path)
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
        _write_atomic(df, directory / f"{quarter.name}.parquet")
        os.replace(temporary, gz_path)
        return len(df)
    finally:
        temporary.unlink(missing_ok=True)


def _refresh_boundary(quarter: Quarter, context: Context) -> pd.Timestamp:
    """The last scheduled cache refresh, on the shared market-date anchor."""
    as_of = context.run.as_of
    current = Quarter.of(as_of)
    section = context.config.data_extract.edgar_index.previous_quarter_recheck
    grace = Offset(section.value, section.unit)
    if grace.unit != "calendar_days":
        raise ValueError("EDGAR index recheck must use calendar days")
    if quarter == current or (quarter == current.previous() and as_of < shift_date(current.start, grace, direction=1)):
        return as_of
    return as_of - pd.Timedelta(days=(as_of.weekday() - 5) % 7)


def _is_due(quarter: Quarter, context: Context, path: Path, rebuild: bool) -> bool:
    """Current and early prior quarters refresh on each requested run; closed originals refresh weekly."""
    if rebuild or not path.exists():
        return True
    if _refresh_boundary(quarter, context) == context.run.as_of:
        return True
    modified = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").tz_convert(MARKET_TIMEZONE).tz_localize(None).normalize()
    return modified < _refresh_boundary(quarter, context)


def refresh(context: Context, window: Window, *, rebuild: bool = False) -> dict[str, int]:
    """Refresh source quarters intersecting explicit filing bounds, retaining old bytes after failures."""
    if window.end > context.run.as_of:
        raise ValueError("EDGAR index Window exceeds the shared run date")
    directory = index_dir(context)
    written: dict[str, int] = {}
    for quarter in quarters_between(window.start, window.end):
        original = directory / f"{quarter.name}.master.gz"
        parsed = directory / f"{quarter.name}.parquet"
        if parsed.exists() and not _is_due(quarter, context, original, rebuild):
            continue
        try:
            n_rows = _download_quarter(context, quarter, directory)
        except (sec_io.TransientReadError, ValueError, OSError, EOFError) as exc:
            context.log.warning("edgar index: %s not refreshed (%s)", quarter.name, exc)
            continue
        if n_rows is not None:
            written[quarter.name] = n_rows
    context.log.info("edgar index: %d quarter(s) downloaded in %s", len(written), window)
    return written


def refreshed_today(context: Context, window: Window) -> bool:
    """Check every requested parsed/original pair against its scheduled refresh boundary."""
    directory = index_dir(context)
    for quarter in quarters_between(window.start, window.end):
        original = directory / f"{quarter.name}.master.gz"
        if not original.exists() or not (directory / f"{quarter.name}.parquet").exists():
            return False
        modified = pd.Timestamp(original.stat().st_mtime, unit="s", tz="UTC").tz_convert(MARKET_TIMEZONE).tz_localize(None).normalize()
        if modified < _refresh_boundary(quarter, context):
            return False
    return True


def quarter_counts(context: Context) -> dict[str, int]:
    """Rows per cached quarter, read from the Parquet footers."""
    return {path.stem: pq.ParquetFile(path).metadata.num_rows for path in sorted(index_dir(context).glob("*.parquet"))}


def entries(context: Context, ciks: Collection[str] | None, forms: Collection[str], window: Window) -> pd.DataFrame:
    """Read bounded index rows; None CIKs explicitly selects market-wide source metadata."""
    directory = index_dir(context)
    first, last = Quarter.of(window.start), Quarter.of(window.end)
    files = [path for path in sorted(directory.glob("*.parquet")) if first <= _file_quarter(path) <= last]
    columns = ["cik", "company", "form", "filed", "accession"]
    if not files or ciks is not None and not ciks or not forms:
        return pd.DataFrame(columns=columns)
    expression = pc.field("form").isin(list(forms))
    if ciks is not None:
        expression &= pc.field("cik").isin([int(c) for c in ciks])
    expression &= (pc.field("filed") >= pa.scalar(window.start.date(), pa.date32())) & (
        pc.field("filed") <= pa.scalar(window.end.date(), pa.date32())
    )
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
