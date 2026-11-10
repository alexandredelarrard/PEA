"""
fetch_13f_backfill.py (src/data_extract/utils/institutionals/fetch_13f_backfill.py)
-----------------------------------------------------------------------------------
`sec13f_hr` history for tickers new to the universe, read from the SEC Form 13F data sets (one ZIP per
filing window from 2013 Q2), each downloaded once into the `sec_13f_datasets` cache.

A ticker selects filing windows. A projected pass discovers relevant managers across those windows;
the second pass retains their full original/amended accession books, including omissions and empty
filings. Values preserve source raw legs and filing-date units; P1 repairs economic scaling later.
Historical replay never overwrites a different accession.

The filings after the newest data set on disk (listed and cached) and before a ticker joined the universe are in no data
set yet and were walked before the ticker was in it: one EDGAR walk (`fetch_13f` over that filing
window, the new tickers only) fills them. A completed walk saves one `sec13f_hr` marker per ticker,
dated the gap start, so each ticker's gap is walked once.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import cast

import pandas as pd
import requests
from tqdm import tqdm

from src.context import Context
from src.data_extract.utils.common.bulk_cache import ZipRead, cache_dir, ensure_zip, is_cached, read_zip_tables
from src.data_extract.utils.common.identity import load_identity
from src.data_extract.utils.common.resume import DONE_PER_KEY, stored_accessions
from src.data_extract.utils.common.sec_io import TransientReadError, sec_get
from src.data_extract.utils.edgar_extract.empty_markers import marker_frame
from src.data_extract.utils.institutionals.fetch_13f import (
    _pick,
    _Saver,
    _WalkState,
    fetch_13f,
    filing_vintage,
    save_hr,
)
from src.data_extract.utils.institutionals.fetch_cusip_map import build_cusip_ticker_map
from src.data_store.schema import Resume, Tables
from src.utils.identity_tables import universe_cusips
from src.utils.string import pad_cik_series
from src.utils.superinvestor_roster import roster_cik_union

logger = logging.getLogger(__name__)

_LANDING_URL = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
_ZIP_URL = "https://www.sec.gov/files/{section}/data/form-13f-data-sets/{name}_form13f.zip"
# Quarterly names through 2023 ("2013q2"), then rolling windows ("01jun2024-31aug2024").
_ZIP_NAME = re.compile(r"(\d{4}q[1-4]|\d{2}[a-z]{3}\d{4}-\d{2}[a-z]{3}\d{4})_form13f\.zip", re.IGNORECASE)
_DOWNLOAD_TIMEOUT = 600  # seconds; each ZIP is 73-100 MB
_CHUNK_ROWS = 500_000

_INFO_COLS = frozenset({"ACCESSION_NUMBER", "CUSIP", "VALUE", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL"})


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
        return self.urls[0]

    @property
    def urls(self) -> tuple[str, ...]:
        return tuple(_ZIP_URL.format(section=section, name=self.name) for section in ("structureddata", "datastandardsinnovation"))

    @property
    def filename(self) -> str:
        return f"{self.name}_form13f.zip"


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
    if full:
        return sorted(set(tickers)) if tickers is not None else context.universe.tickers
    new = context.universe.new_since(cast(Resume, Tables.sec13f_hr.resume).overlap_days, as_of)
    return sorted(new if tickers is None else new & set(tickers))


def backfill_work(data_sets: list[DataSet], tickers: list[str]) -> list[tuple[DataSet, list[str]]]:
    """Discover every eligible window; completed accessions, not an earliest quarter, skip decoding."""
    source_start = pd.Timestamp(cast(str, cast(Resume, Tables.sec13f_hr.resume).source_start))
    return [
        (data_set, sorted(tickers))
        for data_set in sorted(data_sets, key=lambda item: item.start, reverse=True)
        if tickers and data_set.start >= source_start
    ]


def _completed_accessions(context: Context, ciks: set[str]) -> frozenset[str]:
    """The accessions of `ciks` complete in `sec13f_hr`; only a completion marker certifies one."""
    done = stored_accessions(context, Tables.sec13f_hr, sorted(ciks), DONE_PER_KEY, where={"parse_status": "complete"})
    return frozenset().union(*done.values())


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
    )[form.isin(cast(Resume, Tables.sec13f_hr.resume).forms).to_numpy()]
    report = pd.Series(dtype="datetime64[ns]")
    if not df_cover.empty:
        dates = pd.to_datetime(_pick(df_cover, "REPORTCALENDARORQUARTER"), format="mixed", errors="coerce").dt.normalize()
        report = pd.Series(dates.to_numpy(), index=_pick(df_cover, "ACCESSION_NUMBER").astype("string").str.strip()).dropna()
        report = report[~report.index.duplicated()]
    df_out["report_period"] = df_out["accession"].map(report).fillna(df_out["period"])
    return df_out.dropna(subset=["filing_date"]).reset_index(drop=True)


def _archive_scope(path: Path, cusips: frozenset[str]) -> tuple[pd.DataFrame, set[str]] | None:
    """Find target holders from projected ZIP rows before freezing the recovery manager scope."""
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
    )
    if not first:
        return None
    submissions = _submissions(first["SUBMISSION.tsv"], first["COVERPAGE.tsv"])
    holders = set(_pick(first["INFOTABLE.tsv"], "ACCESSION_NUMBER").astype("string").str.strip().dropna())
    return submissions, set(submissions.loc[submissions.accession.isin(holders), "cik"])


def target_cusips(cmap: pd.DataFrame, lines: pd.DataFrame, tickers: list[str]) -> frozenset[str]:
    """The CUSIPs that tag `tickers`: their companies' `universe_cusip` lines (every share class, any date) and the
    `cusip_ticker_map` rows naming them."""
    return frozenset(cmap.loc[cmap.ticker.isin(tickers), "cusip"]) | universe_cusips(lines, tickers)


def _backfill_data_set(
    context: Context,
    path: Path,
    data_set: DataSet,
    tickers: list[str],
    cmap: pd.DataFrame,
    roster_ciks: set[str],
    submissions: pd.DataFrame | None = None,
    *,
    saver: _Saver | None = None,
    completed: frozenset[str] = frozenset(),
) -> int:
    """Preserve full source accessions for a scope fixed across all recovery archives."""
    saver = saver or _Saver(context, 1, _WalkState(looked_up=set(cmap.cusip), cmap=cmap))
    if submissions is None:
        scope = _archive_scope(path, target_cusips(cmap, saver.cusip_lines(), tickers))
        if scope is None:
            logger.warning("13F backfill %s: unreadable source; retried next run", data_set.name)
            return 0
        submissions, holders = scope
        roster_ciks = roster_ciks | holders
    selected = submissions.loc[submissions.cik.isin(roster_ciks)]
    selected = selected.loc[~selected.accession.isin(completed)]
    if selected.empty:
        logger.info("13F backfill %s: no pending accession; second source read skipped", data_set.name)
        return 0
    saved_before = saver.walk.hr_saved
    for manifest, holdings in cached_filing_vintages(
        path,
        selected.accession.tolist(),
        on_invalid=lambda accession, error: context.log.error("13F archive %s: accession %s NOT completed: %s", path.name, accession, error),
    ):
        saver.save_packet(manifest, holdings)
    saved = saver.walk.hr_saved - saved_before
    logger.info("13F backfill %s: preserved %d source position(s) with accession/date in %s", data_set.name, saved, Tables.sec13f_hr)
    return saved


def gap_tickers(context: Context, tickers: list[str], since: pd.Timestamp, as_of: pd.Timestamp, full: bool) -> dict[str, pd.Timestamp]:
    """`{ticker: added_on}` of the tickers joined by `as_of` whose gap `[since, added_on]` still needs the EDGAR walk.

    Only `[since, added_on - overlap)` counts: the join night's walk can write the rest, so a ticker
    whose part is empty has no gap. A ticker is done when a completed gap walk left its marker dated on
    or before `since` (every ticker with a gap is due under `full`)."""
    overlap = pd.Timedelta(days=cast(Resume, Tables.sec13f_hr.resume).overlap_days)
    joined = {t: d for t, d in context.universe.added_on_dates(tickers, as_of).items() if d - overlap > since}
    if full or not joined:
        return joined
    column, sentinel = cast(tuple[str, object], Tables.sec13f_hr.empty_marker)
    df = context.store.load(
        Tables.sec13f_hr,
        columns=["ticker"],
        where={"ticker": sorted(joined), column: sentinel, "cik": sentinel},
        until=since,
        markers=True,
        optional=True,
    )
    done = set() if df is None else {str(t) for t in df["ticker"]}
    return {t: d for t, d in joined.items() if t not in done}


def mark_gap_walked(context: Context, tickers: list[str], since: pd.Timestamp) -> int:
    """Save one `sec13f_hr` marker per ticker, `period` and `filing_date` both `since`: the record that a
    completed EDGAR walk covered the ticker's gap from `since`. Returns rows saved."""
    markers = [marker_frame(Tables.sec13f_hr, {"ticker": t, "period": since, "filing_date": since}) for t in tickers]
    return save_hr(context, pd.concat(markers, ignore_index=True)) if markers else 0


def _walk_gap(context: Context, data_sets: list[DataSet], cache: Path, tickers: list[str], as_of: pd.Timestamp, full: bool) -> None:
    """One EDGAR walk from the day after the newest cached data set ends to the latest join date, for
    the gap tickers only, then one marker per ticker. A listed data set whose ZIP is not cached (not
    published yet) is no coverage; with none cached the walk waits for the next run. A walk that raises
    or leaves a filing unread writes no marker, so the next run walks the gap again."""
    available = [d.end for d in data_sets if is_cached(cache / d.filename)]
    if not available:
        logger.warning("13F backfill: no data set is cached, so the gap start is unknown; the EDGAR gap walk is retried next run")
        return
    since = max(available) + pd.Timedelta(days=1)
    due = gap_tickers(context, tickers, since, as_of, full)
    if not due:
        return
    until = max(due.values())
    logger.info("13F backfill: EDGAR walk %s:%s for %s (after the newest data set)", f"{since:%Y-%m-%d}", f"{until:%Y-%m-%d}", ", ".join(sorted(due)))
    years_history = int(context.config.data_extract.years_history)
    unread = fetch_13f(
        context,
        tickers=sorted(due),
        years_history=years_history,
        filing_window=(f"{since:%Y-%m-%d}", f"{until:%Y-%m-%d}"),
    )
    if unread:
        logger.warning(
            "13F backfill: %d filing(s) of the gap walk stayed unread; no ticker is marked walked, so the gap is walked again next run", unread
        )
        return
    mark_gap_walked(context, sorted(due), since)


def fetch_13f_backfill(context: Context, tickers: list[str] | None, as_of: pd.Timestamp, *, full: bool = False) -> int:
    """Fill `sec13f_hr` from the 13F data sets for the tickers `backfill_targets` selects, then walk
    EDGAR over their gap after the newest data set. Roster managers' full books are saved too.
    Returns broad rows saved from the data sets. With no
    ticker due it reads nothing but the universe and downloads nothing."""
    targets = backfill_targets(context, tickers, as_of, full)
    if not targets:
        logger.info("13F backfill: no new ticker; nothing to do")
        return 0
    cmap = build_cusip_ticker_map(context, [], include_misses=True)
    lines = load_identity(context).cusip_lines
    unmapped = sorted(t for t in targets if not target_cusips(cmap, lines, [t]))
    if unmapped:
        logger.warning("13F backfill: no CUSIP in %s or %s maps to %s; skipped", Tables.universe_cusip, Tables.cusip_ticker_map, ", ".join(unmapped))
    targets = [t for t in targets if t not in unmapped]
    data_sets = published_data_sets(context) if targets else None
    if not data_sets:
        return 0
    work = backfill_work(data_sets, targets)
    roster_ciks = roster_cik_union(context)
    cache = cache_dir(context, str(context.config.local.paths.sec_13f_datasets))
    archived: list[tuple[DataSet, list[str], Path, pd.DataFrame]] = []
    cusips = target_cusips(cmap, lines, targets)
    relevant_ciks = roster_ciks | set(context.store.distinct(Tables.sec13f_hr, "cik", where={"cusip": sorted(cusips)}))
    for data_set, due in tqdm(work, desc="13F source scope"):
        path = ensure_zip(context, cache / data_set.filename, data_set.urls, label=f"13F {data_set.name}", timeout=_DOWNLOAD_TIMEOUT, log=logger)
        if path is None:
            continue
        scope = _archive_scope(path, cusips)
        if scope is None:
            logger.warning("13F backfill %s: unverifiable archive scope; retried next run", data_set.name)
            continue
        submissions, holders = scope
        relevant_ciks |= holders
        archived.append((data_set, due, path, submissions))
    completed = frozenset() if full else _completed_accessions(context, relevant_ciks)
    saved = 0
    saver = _Saver(context, 1, _WalkState(looked_up=set(cmap.cusip), cmap=cmap, lines=lines))
    for data_set, due, path, submissions in tqdm(archived, desc="13F filing versions"):
        saved += _backfill_data_set(context, path, data_set, due, cmap, relevant_ciks, submissions, saver=saver, completed=completed)
    logger.info("13F backfill: saved %d row(s) for %d ticker(s) from %d data set(s)", saved, len(targets), len(work))
    _walk_gap(context, data_sets, cache, targets, as_of, full)
    return saved


def cached_filing_vintages(
    path: Path, accessions: list[str], *, on_invalid: Callable[[str, ValueError], None] | None = None
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Read selected complete accessions before any latest-only filtering; archives stay read-only."""
    if not accessions:
        return []
    wanted = frozenset(accessions)
    tables = read_zip_tables(
        path,
        {
            "SUBMISSION.tsv": ZipRead(upper=True),
            "COVERPAGE.tsv": ZipRead(upper=True),
            "SUMMARYPAGE.tsv": ZipRead(upper=True),
            "INFOTABLE.tsv": ZipRead(
                upper=True,
                chunksize=_CHUNK_ROWS,
                keep=partial(_filed_in, accessions=wanted),
                usecols=_INFO_COLS | frozenset({"NAMEOFISSUER", "TITLEOFCLASS"}),
            ),
        },
        on_corrupt="skip",
    )
    if not tables:
        raise ValueError(f"Unverifiable cached 13F archive: {path.name}")
    submissions = _submissions(tables["SUBMISSION.tsv"], tables["COVERPAGE.tsv"]).set_index("accession")
    covers = tables["COVERPAGE.tsv"].set_index("ACCESSION_NUMBER")
    summaries = tables["SUMMARYPAGE.tsv"].set_index("ACCESSION_NUMBER")
    info_table = tables["INFOTABLE.tsv"]
    groups = info_table.groupby("ACCESSION_NUMBER", sort=False).indices
    packets: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for accession in accessions:
        try:
            if accession not in submissions.index or accession not in covers.index or accession not in summaries.index:
                raise ValueError(f"Incomplete SEC cover metadata: {accession}")
            submission, cover, summary = submissions.loc[accession], covers.loc[accession], summaries.loc[accession]
            info = info_table.iloc[groups.get(accession, [])]
            kind = str(cover["AMENDMENTTYPE"]) if submission.is_amendment else "original"
            restricted = str(summary.get("ISCONFIDENTIALOMITTED")).strip().lower() in {"true", "1", "y"}
            packets.append(
                filing_vintage(
                    info,
                    cik=str(submission.cik),
                    accession_number=accession,
                    period=submission.report_period,
                    filing_date=submission.filing_date,
                    amendment_type=kind,
                    source=path.name,
                    declared_rows=int(summary["TABLEENTRYTOTAL"]),
                    declared_value=float(summary["TABLEVALUETOTAL"]),
                    restricted=restricted,
                    report_type=str(cover["REPORTTYPE"]),
                    amendment_number=int(cover["AMENDMENTNO"]) if pd.notna(cover["AMENDMENTNO"]) else 0,
                )
            )
        except (ValueError, KeyError, TypeError) as error:
            if on_invalid is None:
                raise ValueError(f"Unverifiable SEC packet {accession}: {error}") from error
            on_invalid(accession, ValueError(str(error)))
    return packets
