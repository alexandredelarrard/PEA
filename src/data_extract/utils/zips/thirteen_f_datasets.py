"""
thirteen_f_datasets.py (src/data_extract/utils/zips/thirteen_f_datasets.py)
-----------------------------------------------------------------------------
The SEC Form 13F data sets: one ZIP per filing window, quarterly names through 2023 ("2013q2") and
rolling windows after ("01jun2024-31aug2024"). The zips-download step caches every published window;
readers list the cache with `cached_data_sets` and never download.
"""

from __future__ import annotations

import logging
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from src.context import Context
from src.data_extract.utils.common.bulk_cache import cache_dir, cached_periods, ensure_zip
from src.data_extract.utils.common.sec_io import TransientReadError, sec_get

logger = logging.getLogger(__name__)

_LANDING_URL = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
_ZIP_URL = "https://www.sec.gov/files/{section}/data/form-13f-data-sets/{name}_form13f.zip"
_ZIP_NAME = re.compile(r"(\d{4}q[1-4]|\d{2}[a-z]{3}\d{4}-\d{2}[a-z]{3}\d{4})_form13f\.zip", re.IGNORECASE)
_SUFFIX = "_form13f.zip"
_REQUIRED_MEMBERS = frozenset({"submission.tsv", "coverpage.tsv", "summarypage.tsv", "infotable.tsv"})
_DOWNLOAD_TIMEOUT = 600  # seconds; each ZIP is 73-100 MB


@dataclass(frozen=True)
class DataSet:
    """One published 13F data set: its name and the filing-date window it covers."""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp

    @classmethod
    def parse(cls, name: str) -> DataSet:
        name = name.lower()
        if "-" in name:
            first, last = (pd.Timestamp(datetime.strptime(part, "%d%b%Y")) for part in name.split("-"))
            return cls(name, first, last)
        quarter = pd.Period(name.upper(), freq="Q")
        return cls(name, quarter.start_time.normalize(), quarter.end_time.normalize())

    @property
    def urls(self) -> tuple[str, ...]:
        return tuple(_ZIP_URL.format(section=section, name=self.name) for section in ("structureddata", "datastandardsinnovation"))

    @property
    def filename(self) -> str:
        return f"{self.name}{_SUFFIX}"


def thirteen_f_cache(context: Context) -> Path:
    """The (created) directory the 13F data-set ZIPs are cached in."""
    return cache_dir(context, str(context.config.local.paths.sec_13f_datasets))


def published_data_sets(context: Context) -> list[DataSet] | None:
    """Every data set the SEC landing page links, oldest first; None when the page cannot be read."""
    try:
        response = sec_get(context, _LANDING_URL, timeout=60)
    except (TransientReadError, requests.HTTPError) as exc:
        logger.warning("13F data sets: the landing page could not be read (%s)", exc)
        return None
    names = {name.lower() for name in _ZIP_NAME.findall(response.text)}
    return sorted((DataSet.parse(name) for name in names), key=lambda d: d.start)


def cached_data_sets(cache: Path) -> list[DataSet]:
    """The data sets whose ZIP is cached in `cache`, oldest first."""
    return sorted((DataSet.parse(name) for name in cached_periods(cache, suffix=_SUFFIX)), key=lambda d: d.start)


def _readable(path: Path) -> bool:
    """True when the cached ZIP opens and holds the four data-set tables (a truncated or foreign file does not)."""
    try:
        with zipfile.ZipFile(path) as archive:
            members = {name.rsplit("/", 1)[-1].lower() for name in archive.namelist()}
    except (zipfile.BadZipFile, OSError):
        return False
    return _REQUIRED_MEMBERS <= members


def download_13f_datasets(context: Context) -> int:
    """Cache every published 13F data set missing or unreadable on disk (an unreadable ZIP is deleted and downloaded
    again); returns the ZIPs cached and readable. A window still missing afterwards is logged as an error. Writes no
    table."""
    published = published_data_sets(context)
    if published is None:
        return 0
    cache = thirteen_f_cache(context)
    missing: list[str] = []
    for data_set in tqdm(published, desc="13F data sets (download)"):
        path = cache / data_set.filename
        if path.exists() and not _readable(path):
            logger.warning("13F data set %s: cached ZIP is unreadable; deleted and downloaded again", data_set.name)
            path.unlink()
        found = ensure_zip(context, path, data_set.urls, label=f"13F {data_set.name}", timeout=_DOWNLOAD_TIMEOUT)
        if found is None or not _readable(found):
            missing.append(data_set.name)
    if missing:
        logger.error("13F data sets: %d of %d published ZIP(s) not cached: %s", len(missing), len(published), ", ".join(missing))
    logger.info("13F data sets: %d of %d published ZIP(s) cached", len(published) - len(missing), len(published))
    return len(published) - len(missing)
