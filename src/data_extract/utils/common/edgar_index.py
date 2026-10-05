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
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import edgar
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.context import Context
from src.data_extract.utils.common import sec_io
from src.data_extract.utils.common.bulk_cache import cache_dir
from src.data_extract.utils.common.registrant import FORM_POLICY
from src.utils.string import pad_cik_series

logger = logging.getLogger(__name__)

INDEX_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.gz"
#: The forms kept from the index: every form a document fetcher lists, plus the 13F family.
INDEX_FORMS: frozenset[str] = frozenset(FORM_POLICY) | {"13F-HR", "13F-HR/A"}
#: EDGAR's full index starts in 1993 Q1.
FIRST_INDEX_YEAR = 1993
#: Days into a quarter during which the previous quarter is downloaded again (late index lines).
_PREVIOUS_QUARTER_DAYS = 10
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
    """Every quarter from `first`'s to `last`'s, inclusive, never before `FIRST_INDEX_YEAR`."""
    quarter = max(Quarter.of(first), Quarter(FIRST_INDEX_YEAR, 1))
    out: list[Quarter] = []
    while quarter <= Quarter.of(last):
        out.append(quarter)
        quarter = quarter.next()
    return out


def index_dir(context: Context) -> Path:
    """The (created) cache directory."""
    return cache_dir(context, str(context.config.local.paths.sec_edgar_index))


def parse_master(raw: bytes) -> pd.DataFrame:
    """`master.idx` text (gzipped or plain) -> one row per line of an `INDEX_FORMS` form, typed like the cache."""
    text = (gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw).decode(_ENCODING)
    rows = [fields for line in text.splitlines() if len(fields := line.split("|")) == 5 and fields[0].isdigit() and fields[2] in INDEX_FORMS]
    df = pd.DataFrame(rows, columns=["cik", "company", "form", "filed", "filename"])
    return pd.DataFrame(
        {
            "cik": df["cik"].astype("int64"),
            "company": df["company"].astype(object),
            "form": df["form"].astype(object),
            "filed": pd.to_datetime(df["filed"], format="%Y-%m-%d").dt.date,
            "accession": df["filename"].str.rsplit("/", n=1).str[-1].str.removesuffix(".txt").astype(object),
        }
    ).sort_values(["cik", "filed", "accession"], ignore_index=True)


def _write_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write `df` to `path` through a temporary file and `os.replace`, so a reader never sees a partial file."""
    tmp = path.with_name(path.name + ".part")
    pq.write_table(pa.Table.from_pandas(df, schema=_SCHEMA, preserve_index=False), tmp, row_group_size=_ROW_GROUP)
    os.replace(tmp, path)


def _download_quarter(context: Context, quarter: Quarter, directory: Path) -> int | None:
    """Download, parse and cache one quarter; its row count, or None when SEC served no file."""
    gz_path = directory / f"{quarter.name}.master.gz"
    status = sec_io.download(context, INDEX_URL.format(year=quarter.year, quarter=quarter.quarter), gz_path)
    if status != 200:
        context.log.warning("edgar index: %s not served (HTTP %d)", quarter.name, status)
        return None
    try:
        df = parse_master(gz_path.read_bytes())
    finally:
        gz_path.unlink(missing_ok=True)
    _write_atomic(df, directory / f"{quarter.name}.parquet")
    return len(df)


def _is_due(quarter: Quarter, current: Quarter, as_of: pd.Timestamp, path: Path, rebuild: bool) -> bool:
    """Download when rebuilding, when missing, for the current quarter, or for the previous one early in a quarter."""
    if rebuild or not path.exists() or quarter == current:
        return True
    return quarter == current.previous() and (as_of - current.start).days < _PREVIOUS_QUARTER_DAYS


def refresh(context: Context, as_of: pd.Timestamp, years_history: int, *, rebuild: bool = False) -> dict[str, int]:
    """Bring the cache up to `as_of` over `years_history`; rows written per downloaded quarter.

    A transient SEC failure on one quarter is logged and leaves that quarter's previous file in place."""
    as_of = pd.Timestamp(as_of).normalize()
    directory = index_dir(context)
    current = Quarter.of(as_of)
    written: dict[str, int] = {}
    for quarter in quarters_between(as_of - pd.DateOffset(years=years_history), as_of):
        if not _is_due(quarter, current, as_of, directory / f"{quarter.name}.parquet", rebuild):
            continue
        try:
            n_rows = _download_quarter(context, quarter, directory)
        except sec_io.TransientReadError as exc:
            context.log.warning("edgar index: %s not refreshed (%s)", quarter.name, exc)
            continue
        if n_rows is not None:
            written[quarter.name] = n_rows
    context.log.info("edgar index: %d quarter(s) downloaded through %s", len(written), as_of.date())
    return written


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
