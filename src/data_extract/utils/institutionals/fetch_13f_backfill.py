"""
fetch_13f_backfill.py (src/data_extract/utils/institutionals/fetch_13f_backfill.py)
-----------------------------------------------------------------------------------
`sec13f_hr` history for tickers new to the universe, read from the SEC Form 13F data sets (one ZIP per
filing window from 2013 Q2), each downloaded once into the `sec_13f_datasets` cache.

A ticker is filled from every published data set whose window starts before its own earliest stored
`period`, newest first and saved one ZIP at a time, so that minimum is its resume point. Holdings map to
tickers through the stored `cusip_ticker_map` only. Each filing's value unit is decided by edgartools'
`_detect_value_in_thousands`, the rule behind the nightly rows, and rows are built by `fetch_13f`'s own
classifier. A ZIP whose roster-manager filings sit a unit factor away from their stored books is not
saved, and a stored row filed on or after the data-set row is kept.

The filings after the newest published data set and before a ticker joined the universe are in no data
set yet and were walked before the ticker was in it: one EDGAR walk (`fetch_13f` over that filing
window, the new tickers only) fills them, once per ticker.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import requests
from edgar.thirteenf.models import _detect_value_in_thousands
from tqdm import tqdm

from src.constants.constants import SEC_13F_FORMS
from src.context import Context
from src.data_extract.utils.common.bulk_cache import ZipRead, cache_dir, ensure_zip, read_zip_tables
from src.data_extract.utils.common.sec_io import TransientReadError, sec_get
from src.data_extract.utils.institutionals.fetch_13f import (
    _HR_COLS,
    _classify_holdings,
    _latest_per_key,
    _pick,
    _resolve_tickers,
    fetch_13f,
    save_hr,
)
from src.data_extract.utils.institutionals.fetch_cusip_map import build_cusip_ticker_map
from src.data_store.schema import Resume, Tables
from src.utils.string import pad_cik_series
from src.utils.superinvestor_roster import roster_cik_union
from src.utils.universe import added_on_dates, load_universe_tickers, new_tickers

logger = logging.getLogger(__name__)

_LANDING_URL = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
_ZIP_URL = "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/{name}_form13f.zip"
# Quarterly names through 2023 ("2013q2"), then rolling windows ("01jun2024-31aug2024").
_ZIP_NAME = re.compile(r"(\d{4}q[1-4]|\d{2}[a-z]{3}\d{4}-\d{2}[a-z]{3}\d{4})_form13f\.zip", re.IGNORECASE)
_DOWNLOAD_TIMEOUT = 600  # seconds; each ZIP is 73-100 MB
_CHUNK_ROWS = 500_000
# A filing whose data-set value total is this many times its stored book (either way) flipped unit.
_UNIT_FLIP = (500.0, 2000.0)
_UNIT_TOLERANCE = 0.005

_BOOK_COLS = [c for c in _HR_COLS if c != "ticker"]
_INFO_COLS = frozenset({"ACCESSION_NUMBER", "CUSIP", "VALUE", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL"})
# The edgartools infotable spelling of the data sets' SSHPRNAMTTYPE.
_AMOUNT_TYPES = {"SH": "Shares", "PRN": "Principal"}


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
    def url(self) -> str:
        return _ZIP_URL.format(name=self.name)

    @property
    def filename(self) -> str:
        return f"{self.name}_form13f.zip"


@dataclass(frozen=True)
class UnitCheck:
    """Roster-manager filings compared with their stored books: how many, how many within tolerance, and the unit flips."""

    compared: int
    within: int
    flips: list[str]


def published_data_sets(context: Context) -> list[DataSet] | None:
    """Every data set the SEC landing page links, oldest first; None when the page cannot be read."""
    try:
        response = sec_get(context, _LANDING_URL, timeout=60)
    except (TransientReadError, requests.HTTPError) as exc:
        logger.warning("13F backfill: the data-set landing page could not be read (%s); retried next run", exc)
        return None
    names = {name.lower() for name in _ZIP_NAME.findall(response.text)}
    return sorted((DataSet.parse(name) for name in names), key=lambda d: d.start)


def backfill_targets(context: Context, tickers: list[str] | None, as_of: pd.Timestamp, full: bool) -> list[str]:
    """The tickers to backfill: universe tickers added inside `sec13f_hr`'s overlap (narrowed to `tickers`
    when given); under `full`, `tickers` or else the whole universe, whatever their age."""
    universe = set(load_universe_tickers(context))
    asked = None if tickers is None else {str(t).strip().upper() for t in tickers}
    if full:
        return sorted(asked if asked is not None else universe)
    overlap = cast(Resume, Tables.sec13f_hr.resume).overlap_days
    new = new_tickers(context.store, overlap, as_of) & universe
    return sorted(new if asked is None else new & asked)


def resume_points(context: Context, tickers: list[str]) -> dict[str, pd.Timestamp | None]:
    """Each ticker's earliest stored `sec13f_hr.period` (None without a row), from one `key_stats` read."""
    df_stats = context.store.key_stats(Tables.sec13f_hr, "ticker", "period", where={"ticker": tickers})
    first = {str(key).upper(): pd.Timestamp(day) for key, day in zip(df_stats["key"], df_stats["first"], strict=True)}
    return {ticker: first.get(ticker) for ticker in tickers}


def backfill_work(data_sets: list[DataSet], points: dict[str, pd.Timestamp | None], full: bool) -> list[tuple[DataSet, list[str]]]:
    """`(data set, tickers)` newest first: a data set from the table's source start is read for the
    tickers whose resume point is after its window start (every ticker under `full`)."""
    source_start = pd.Timestamp(cast(str, cast(Resume, Tables.sec13f_hr.resume).source_start))
    work: list[tuple[DataSet, list[str]]] = []
    for data_set in sorted(data_sets, key=lambda d: d.start, reverse=True):
        due = sorted(t for t, first in points.items() if full or first is None or data_set.start < first)
        if due and data_set.start >= source_start:
            work.append((data_set, due))
    return work


def _holds(chunk: pd.DataFrame, cusips: frozenset[str]) -> pd.Series:
    return _pick(chunk, "CUSIP").astype("string").str.strip().str.upper().str.zfill(9).isin(cusips)


def _filed_in(chunk: pd.DataFrame, accessions: frozenset[str]) -> pd.Series:
    return _pick(chunk, "ACCESSION_NUMBER").astype("string").str.strip().isin(accessions)


def _submissions(df_sub: pd.DataFrame, df_cover: pd.DataFrame) -> pd.DataFrame:
    """One row per 13F-HR(/A) accession: padded `cik`, `filing_date`, `period`, the cover page's
    `report_period` (else `period`) and `is_amendment`."""
    form = _pick(df_sub, "SUBMISSIONTYPE").astype("string").str.strip().str.upper()
    df_out = pd.DataFrame(
        {
            "accession": _pick(df_sub, "ACCESSION_NUMBER").astype("string").str.strip(),
            "cik": pad_cik_series(_pick(df_sub, "CIK")),
            "filing_date": pd.to_datetime(_pick(df_sub, "FILING_DATE"), format="mixed", errors="coerce").dt.normalize(),
            "period": pd.to_datetime(_pick(df_sub, "PERIODOFREPORT"), format="mixed", errors="coerce").dt.normalize(),
            "is_amendment": form.str.endswith("/A").fillna(False).astype(bool),
        }
    )[form.isin(SEC_13F_FORMS).to_numpy()]
    report = pd.Series(dtype="datetime64[ns]")
    if not df_cover.empty:
        dates = pd.to_datetime(_pick(df_cover, "REPORTCALENDARORQUARTER"), format="mixed", errors="coerce").dt.normalize()
        report = pd.Series(dates.to_numpy(), index=_pick(df_cover, "ACCESSION_NUMBER").astype("string").str.strip()).dropna()
        report = report[~report.index.duplicated()]
    df_out["report_period"] = df_out["accession"].map(report).fillna(df_out["period"])
    return df_out.dropna(subset=["filing_date"]).reset_index(drop=True)


def _read_data_set(path: Path, cusips: frozenset[str], roster_ciks: set[str]) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """The data set's 13F-HR submissions and the info-table lines of every filing that holds one of
    `cusips` or comes from a roster CIK (two chunked passes over INFOTABLE); None for a corrupt or
    incomplete ZIP."""
    first = read_zip_tables(
        path,
        {
            "SUBMISSION.tsv": ZipRead(upper=True),
            "COVERPAGE.tsv": ZipRead(usecols=frozenset({"ACCESSION_NUMBER", "REPORTCALENDARORQUARTER"}), upper=True, required=False),
            "INFOTABLE.tsv": ZipRead(
                usecols=frozenset({"ACCESSION_NUMBER", "CUSIP"}), keep=partial(_holds, cusips=cusips), chunksize=_CHUNK_ROWS, upper=True
            ),
        },
        on_corrupt="delete",
        log=logger,
    )
    if not first:
        return None
    submissions = _submissions(first["SUBMISSION.tsv"], first["COVERPAGE.tsv"])
    holders = set(_pick(first["INFOTABLE.tsv"], "ACCESSION_NUMBER").astype("string").str.strip().dropna())
    wanted = frozenset((holders | set(submissions.loc[submissions["cik"].isin(roster_ciks), "accession"])) & set(submissions["accession"]))
    if not wanted:
        return submissions, pd.DataFrame(columns=sorted(_INFO_COLS))
    second = read_zip_tables(
        path,
        {"INFOTABLE.tsv": ZipRead(usecols=_INFO_COLS, keep=partial(_filed_in, accessions=wanted), chunksize=_CHUNK_ROWS, upper=True)},
        on_corrupt="delete",
        log=logger,
    )
    return None if not second else (submissions, second["INFOTABLE.tsv"])


def _value_in_thousands(df_info: pd.DataFrame, report_period: pd.Series) -> pd.Series:
    """Per accession, edgartools' unit decision on that filing's raw holdings (the data sets carry no
    schema version, so an ambiguous filing falls back on its report period)."""
    df_holdings = pd.DataFrame(
        {
            "accession": df_info["accession"],
            "Type": _pick(df_info, "SSHPRNAMTTYPE").astype("string").str.strip().str.upper().map(_AMOUNT_TYPES),
            "SharesPrnAmount": pd.to_numeric(_pick(df_info, "SSHPRNAMT"), errors="coerce"),
            "PutCall": _pick(df_info, "PUTCALL").astype("string").str.strip().fillna(""),
            "Value": pd.to_numeric(_pick(df_info, "VALUE"), errors="coerce"),
        }
    )
    decided: dict[str, bool] = {}
    for accession, df_filing in df_holdings.groupby("accession", sort=False):
        reported = report_period.get(accession)
        decided[str(accession)] = bool(
            _detect_value_in_thousands(df_filing, None, None if pd.isna(reported) else pd.Timestamp(reported).to_pydatetime())
        )
    return pd.Series(decided, dtype=bool)


def data_set_book(submissions: pd.DataFrame, df_info: pd.DataFrame) -> pd.DataFrame:
    """The data set's holdings as `fetch_13f` books: values scaled per filing, lines summed per
    (accession, cusip), the last filed kept per (cik, period, cusip)."""
    df_info = df_info.assign(accession=_pick(df_info, "ACCESSION_NUMBER").astype("string").str.strip())
    df_info = df_info[df_info["accession"].isin(submissions["accession"])]
    if df_info.empty:
        return pd.DataFrame(columns=_BOOK_COLS)
    thousands = _value_in_thousands(df_info, submissions.set_index("accession")["report_period"])
    scale = np.where(df_info["accession"].map(thousands).fillna(False).astype(bool), 1000.0, 1.0)
    typed = _classify_holdings(df_info.assign(VALUE=pd.to_numeric(df_info["VALUE"], errors="coerce") * scale))
    typed["accession"] = df_info["accession"].to_numpy()
    summed = typed.dropna(subset=["cusip"]).groupby(["accession", "cusip"], as_index=False).sum(numeric_only=True)
    meta = submissions[["accession", "cik", "period", "filing_date", "is_amendment"]]
    book = summed.merge(meta, on="accession", how="inner").dropna(subset=["period"])
    book = book.sort_values(["filing_date", "is_amendment", "accession"], kind="stable")
    return _latest_per_key(book)[_BOOK_COLS].reset_index(drop=True)


def unit_check(context: Context, book: pd.DataFrame, roster_ciks: set[str]) -> UnitCheck:
    """The data set's roster-manager filings against their stored books, per (cik, period, filing date)
    over the CUSIPs both hold: a value total a unit factor off is a flip."""
    keys = ["cik", "period", "filing_date"]
    df_zip = book.loc[book["cik"].isin(roster_ciks), [*keys, "cusip", "value_usd"]]
    df_stored = (
        None
        if df_zip.empty
        else context.store.load(
            Tables.sec13f_manager_holdings,
            columns=[*keys, "cusip", "value_usd"],
            where={"cik": sorted(set(df_zip["cik"]))},
            date_col="period",
            since=df_zip["period"].min(),
            until=df_zip["period"].max(),
            optional=True,
        )
    )
    if df_stored is None:
        return UnitCheck(0, 0, [])
    df_stored = df_stored.assign(**{col: pd.to_datetime(df_stored[col]).dt.normalize() for col in ("period", "filing_date")})
    df_merged = df_zip.merge(df_stored, on=[*keys, "cusip"], suffixes=("_zip", "_db"))
    df_totals = df_merged.groupby(keys)[["value_usd_zip", "value_usd_db"]].sum()
    ratio = (df_totals["value_usd_zip"] / df_totals["value_usd_db"])[df_totals["value_usd_db"] > 0]
    low, high = _UNIT_FLIP
    flipped = ratio.between(low, high) | ratio.between(1 / high, 1 / low)
    flips = [f"{cik} {period:%Y-%m-%d} filed {filed:%Y-%m-%d}" for cik, period, filed in ratio[flipped].index]
    return UnitCheck(len(ratio), int(((ratio - 1).abs() <= _UNIT_TOLERANCE).sum()), flips)


def _drop_superseded(context: Context, df_hr: pd.DataFrame) -> pd.DataFrame:
    """`df_hr` without the rows whose stored twin (same primary key) was filed on or after them."""
    pk = list(Tables.sec13f_hr.pk)
    df_stored = context.store.load(
        Tables.sec13f_hr,
        columns=[*pk, "filing_date"],
        where={"ticker": sorted(set(df_hr["ticker"]))},
        date_col="period",
        since=df_hr["period"].min(),
        until=df_hr["period"].max(),
        optional=True,
    )
    if df_stored is None:
        return df_hr
    df_stored = df_stored.assign(
        period=pd.to_datetime(df_stored["period"]).dt.normalize(), stored_filed=pd.to_datetime(df_stored["filing_date"]).dt.normalize()
    )
    df_merged = df_hr[[*pk, "filing_date"]].merge(df_stored[[*pk, "stored_filed"]], on=pk, how="left")
    keep = df_merged["stored_filed"].isna() | (df_merged["filing_date"] > df_merged["stored_filed"])
    return df_hr[keep.to_numpy()]


def _backfill_data_set(context: Context, path: Path, data_set: DataSet, tickers: list[str], cmap: pd.DataFrame, roster_ciks: set[str]) -> int:
    """Read one cached data set for `tickers` and save their rows; returns rows saved (0 when the ZIP is
    unreadable or fails the unit check)."""
    cusips = frozenset(cmap.loc[cmap["ticker"].isin(tickers), "cusip"])
    tables = _read_data_set(path, cusips, roster_ciks)
    if tables is None:
        logger.warning("13F backfill %s: unreadable data set; retried next run", data_set.name)
        return 0
    book = data_set_book(*tables)
    check = unit_check(context, book, roster_ciks)
    if check.flips:
        logger.error(
            "13F backfill %s: %d of %d roster filing(s) sit a unit factor away from their stored books (%s); the data set is not saved",
            data_set.name,
            len(check.flips),
            check.compared,
            ", ".join(check.flips[:5]),
        )
        return 0
    logger.info(
        "13F backfill %s: unit check, %d of %d roster filing(s) within %.1f%%", data_set.name, check.within, check.compared, 100 * _UNIT_TOLERANCE
    )
    df_hr = _resolve_tickers(book, cmap, set(tickers))
    df_hr = _drop_superseded(context, df_hr) if not df_hr.empty else df_hr
    saved = save_hr(context, df_hr)
    logger.info("13F backfill %s: saved %d row(s) for %s", data_set.name, saved, ", ".join(tickers))
    return saved


def gap_tickers(context: Context, tickers: list[str], since: pd.Timestamp, full: bool) -> dict[str, pd.Timestamp]:
    """`{ticker: added_on}` of the tickers whose gap `[since, added_on]` still needs the EDGAR walk.

    Only `[since, added_on - overlap)` counts: the join night's walk can write the rest. A ticker whose
    part is empty has no gap; one holding a `sec13f_hr` row filed inside it is done (every ticker with a
    gap is due under `full`)."""
    overlap = pd.Timedelta(days=cast(Resume, Tables.sec13f_hr.resume).overlap_days)
    joined = {t: d for t, d in added_on_dates(context.store, tickers).items() if d - overlap > since}
    if full or not joined:
        return joined
    df = context.store.load(
        Tables.sec13f_hr,
        columns=["ticker", "filing_date"],
        where={"ticker": sorted(joined)},
        date_col="filing_date",
        since=since,
        until=max(joined.values()) - overlap,
        optional=True,
    )
    done = (
        set() if df is None else {str(t) for t, f in zip(df["ticker"], df["filing_date"], strict=True) if pd.Timestamp(f) < joined[str(t)] - overlap}
    )
    return {t: d for t, d in joined.items() if t not in done}


def _walk_gap(context: Context, data_sets: list[DataSet], tickers: list[str], full: bool) -> None:
    """One EDGAR walk from the day after the newest data set ends to the latest join date, for the gap tickers only."""
    since = max(d.end for d in data_sets) + pd.Timedelta(days=1)
    due = gap_tickers(context, tickers, since, full)
    if not due:
        return
    until = max(due.values())
    logger.info("13F backfill: EDGAR walk %s:%s for %s (after the newest data set)", f"{since:%Y-%m-%d}", f"{until:%Y-%m-%d}", ", ".join(sorted(due)))
    years_history = int(context.config.data_extract.years_history)
    fetch_13f(context, tickers=sorted(due), years_history=years_history, filing_window=(f"{since:%Y-%m-%d}", f"{until:%Y-%m-%d}"))


def fetch_13f_backfill(context: Context, tickers: list[str] | None, as_of: pd.Timestamp, *, full: bool = False) -> int:
    """Fill `sec13f_hr` from the 13F data sets for the tickers `backfill_targets` selects, then walk
    EDGAR over their gap after the newest data set; returns rows saved from the data sets. With no
    ticker due it reads nothing but the universe and downloads nothing."""
    targets = backfill_targets(context, tickers, as_of, full)
    if not targets:
        logger.info("13F backfill: no new ticker; nothing to do")
        return 0
    cmap = build_cusip_ticker_map(context, [])
    unmapped = sorted(set(targets) - set(cmap["ticker"]))
    if unmapped:
        logger.warning("13F backfill: no CUSIP in %s maps to %s; skipped", Tables.cusip_ticker_map, ", ".join(unmapped))
    targets = [t for t in targets if t not in unmapped]
    data_sets = published_data_sets(context) if targets else None
    if not data_sets:
        return 0
    work = backfill_work(data_sets, resume_points(context, targets), full)
    roster_ciks = roster_cik_union(context)
    cache = cache_dir(context, str(context.config.local.paths.sec_13f_datasets))
    saved = 0
    for data_set, due in tqdm(work, desc="13F data sets"):
        path = ensure_zip(context, cache / data_set.filename, data_set.url, label=f"13F {data_set.name}", timeout=_DOWNLOAD_TIMEOUT, log=logger)
        if path is not None:
            saved += _backfill_data_set(context, path, data_set, due, cmap, roster_ciks)
    logger.info("13F backfill: saved %d row(s) for %d ticker(s) from %d data set(s)", saved, len(targets), len(work))
    _walk_gap(context, data_sets, targets, full)
    return saved
