"""
fetch_13f.py (src/data_extract/utils/institutionals/fetch_13f.py)
-----------------------------------------------------------------
One oldest-first walk over every 13F-HR by filing date (edgartools), feeding two tables from one parse: the
complete CUSIP book -> `sec13f_hr`, and roster managers also -> `sec13f_manager_holdings`.
Originals and amendments have separate accession keys. Tickers come from the
CUSIP map (OpenFIGI), never from issuer names.

The window runs from `max(sec13f_hr.filing_date)` minus the table's overlap to the run date. Transient read
failures get in-task retry rounds; a filing filed inside the overlap that still fails holds back every
filing after it (low watermark), so the next window starts at or before it. An older one is skipped with
an ERROR naming its accession. A scoped (`-t`) run never reads past the stored frontier.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, cast
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from edgar import Filing, Filings, get_filings
from edgar import obj as data_object
from edgar.httpclient import close_clients
from edgar.thirteenf.parsers.infotable_xml import parse_infotable_xml
from tqdm import tqdm

from src.constants.constants import SEC_13F_INFERRED_HOLDINGS_REPORT, SEC_13F_SOURCE_ACCOUNTED
from src.context import Context
from src.data_extract.utils.common.edgar_driver import FilingStamp, RetryRounds
from src.data_extract.utils.common.empty_markers import marker_mask
from src.data_extract.utils.common.resume import DONE_PER_KEY, DONE_TABLE, document_floor, stored_accessions
from src.data_extract.utils.common.sec_io import TransientReadError, configure, filing_obj, sec_call
from src.data_extract.utils.institutionals.fetch_cusip_map import build_cusip_ticker_map
from src.data_extract.utils.institutionals.legacy_13f_fallback import (
    _ENTRY_COUNT,
    _FOOTER_TOTAL,
    _VALUE_TOTAL,
    _genuine_holding_lines,
    _value_total_error,
    needs_legacy_fallback,
    parse_legacy_information_table,
)
from src.data_extract.utils.institutionals.value_units import ValueUnits, normalize_information_values, resolve_value_units
from src.data_store.schema import Resume, Table, Tables
from src.utils.institutional_vintages import literal_13f_cusip, reconcile_13f_source
from src.utils.string import pad_cik, pad_cik_series
from src.utils.superinvestor_roster import roster_cik_union
from src.utils.universe import load_universe_tickers

logger = logging.getLogger(__name__)
_CHECKED = r"(?:\[\s*[xX]\s*\]|\{\s*[xX]\s*\}|\(\s*[xX]\s*\))"
#: A checked legacy report-type box; its group is the report type as printed. Its presence with the cover total lets
#: the information-table text stand in for the full submission.
_CHECKED_REPORT = re.compile(_CHECKED + r"\s*(13\s*F\s+(?:HOLDINGS|COMBINATION|NOTICE)\s+REPORT)", re.I)
#: Unit words a legacy cover value-total line may state, each with its dollars-per-unit; "dollars" counts only alone.
_COVER_UNITS = (
    (re.compile(r"\bbillions?\b", re.I), 1e9),
    (re.compile(r"\bmillions?\b", re.I), 1e6),
    (re.compile(r"\bthou(?:sands?)?\b|\(\s*0{3}'?s\s*\)|\b1,?000'?s\b|x\s*\$?\s*1,?000\b|\(\s*\$\s*1,?000\s*\)", re.I), 1e3),
)
_COVER_DOLLARS = re.compile(r"\bdollars?\b", re.I)

#: The sleeper between retry rounds; tests replace it.
_sleep: Callable[[float], None] = time.sleep

# `quarter` is absent on purpose: it tagged the SOURCE bulk data set, not the period.
_HR_COLS = [
    "cik",
    "period",
    "filing_date",
    "ticker",
    "cusip",
    "shares",
    "value_usd",
    "call_shares",
    "call_value",
    "put_shares",
    "put_value",
    "debt_prn",
    "debt_value",
    "other_value",
]

_BOOK_COLS = [
    "cik",
    "period",
    "filing_date",
    "cusip",
    "issuer_name",
    "title_of_class",
    "position_type",
    "shares",
    "value_usd",
    "call_shares",
    "call_value",
    "put_shares",
    "put_value",
    "debt_prn",
    "debt_value",
    "other_value",
]

#: The columns `parse_legacy_information_table` emits; an empty-cover book carries them with no rows.
_LEGACY_INFO_COLS = ["CUSIP", "NAMEOFISSUER", "TITLEOFCLASS", "VALUE", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL"]

#: Legacy diagnostic reduction key; immutable source storage also keys by accession.
_BOOK_KEY = ["cik", "period", "cusip"]

# Cheap source sanity check; aggregation validates units/values against historical quotes.
_IMPLIED_PRICE_BAND = (1.0, 5000.0)

#: One position above this, or issuer text matching the info-table pattern, is a parse artefact.
_MAX_POSITION_USD = 1e12
_TABLE_TEXT_IN_NAME = r"\bSH\b.*\bSOLE\b|\d{1,3},\d{3} SH"

#: The five mutually exclusive holding classes, in tie-break order (`common` wins).
POSITION_TYPES = ("common", "call", "put", "debt", "other")

#: The per-type value columns, in `POSITION_TYPES` order; `_dominant_type` picks among them.
_VALUE_BY_TYPE = {"common": "value_usd", "call": "call_value", "put": "put_value", "debt": "debt_value", "other": "other_value"}


@dataclass(frozen=True)
class ReadFailure:
    """A failed `_read_filing`. `transient` is a throttle or an unreachable SEC (worth retrying);
    otherwise the failure is deterministic, e.g. an unparseable info table. `reason` names the error."""

    transient: bool
    reason: str


@dataclass
class _WalkState:
    """Running state of one 13F walk: the CUSIP map built so far and rows saved per table."""

    looked_up: set[str] = field(default_factory=set)
    cmap: pd.DataFrame | None = None
    hr_saved: int = 0
    hr_suspect: int = 0
    book_saved: int = 0
    book_suspect: int = 0


def _pick(df: pd.DataFrame, *candidates: str) -> pd.Series:
    """Return the first present column (case-insensitive) among candidates."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return df[lower[cand.lower()]]
    return pd.Series([pd.NA] * len(df), index=df.index)


def _holding_masks(infotable: pd.DataFrame) -> tuple[dict[str, pd.Series], pd.Series, pd.Series]:
    """`({position_type: mask}, amount, value)` for one info table: the single definition of a
    stock/option/debt/other line, shared by `_classify_holdings` and `position_type`."""
    putcall = _pick(infotable, "PUTCALL").astype("string").str.strip().str.upper().fillna("")
    amttype = _pick(infotable, "SSHPRNAMTTYPE", "Type").astype("string").str.strip().str.upper().fillna("")
    amt = pd.to_numeric(_pick(infotable, "SSHPRNAMT", "SharesPrnAmount"), errors="coerce").fillna(0.0)
    val = pd.to_numeric(_pick(infotable, "VALUE"), errors="coerce")
    if "raw_value" not in infotable:
        val = val.fillna(0.0)

    is_call = putcall == "CALL"
    is_put = putcall == "PUT"
    opt = is_call | is_put
    is_debt = (~opt) & amttype.isin(["PRN", "PRINCIPAL"])
    is_stock = (~opt) & amttype.isin(["SH", "SHARES", ""])
    is_other = ~(opt | is_debt | is_stock)

    return ({"common": is_stock, "call": is_call, "put": is_put, "debt": is_debt, "other": is_other}, amt, val)


def position_type(infotable: pd.DataFrame) -> pd.Series:
    """One `position_type` label per holding line, from the masks `_classify_holdings` buckets on."""
    masks, _, _ = _holding_masks(infotable)
    out = pd.Series("other", index=infotable.index, dtype="object")
    for name in reversed(POSITION_TYPES):  # `common` applied last, so it wins any overlap
        out = out.mask(masks[name], name)
    return out


def _classify_holdings(infotable: pd.DataFrame) -> pd.DataFrame:
    """One typed row per holding line: value/shares land in exactly one bucket keyed on put/call
    and amount type (SH/PRN or Shares/Principal); a blank type with no put/call is long stock."""
    return _typed_lines(infotable, *_holding_masks(infotable))


def _typed_lines(infotable: pd.DataFrame, masks: dict[str, pd.Series], amt: pd.Series, val: pd.Series) -> pd.DataFrame:
    """`_classify_holdings` from already computed `_holding_masks`."""
    is_stock, is_call = masks["common"], masks["call"]
    is_put, is_debt, is_other = masks["put"], masks["debt"], masks["other"]

    return pd.DataFrame(
        {
            "cusip": _pick(infotable, "CUSIP").map(literal_13f_cusip),
            "shares": amt.where(is_stock, 0.0),
            "value_usd": val.where(is_stock, 0.0),
            "call_shares": amt.where(is_call, 0.0),
            "call_value": val.where(is_call, 0.0),
            "put_shares": amt.where(is_put, 0.0),
            "put_value": val.where(is_put, 0.0),
            "debt_prn": amt.where(is_debt, 0.0),
            "debt_value": val.where(is_debt, 0.0),
            "other_value": val.where(is_other, 0.0),
        }
    )


def _dominant_type(grouped: pd.DataFrame) -> pd.Series:
    """`position_type` of a grouped row: the class with the most absolute value; ties go to `common`."""
    values = pd.DataFrame({name: grouped[col].fillna(0.0).abs() for name, col in _VALUE_BY_TYPE.items()})
    return values.idxmax(axis=1)


def _book_frame(cik: str, filing_date: Any, period: Any, infotable: pd.DataFrame) -> pd.DataFrame:
    """One filing's info table -> one `_BOOK_COLS` row per CUSIP, no universe filter. Pure.
    Numerics are summed over a CUSIP's lines; `issuer_name` / `title_of_class` take the first
    non-null; `position_type` is the dominant class. Rows without a period are dropped."""
    masks, amt, val = _holding_masks(infotable)
    typed = _typed_lines(infotable, masks, amt, val)
    numeric = list(typed.columns.drop("cusip"))
    typed["issuer_name"] = _pick(infotable, "NAMEOFISSUER", "Issuer").astype("string")
    typed["title_of_class"] = _pick(infotable, "TITLEOFCLASS", "Class").astype("string")
    aggregations = {**{c: "sum" for c in numeric}, "issuer_name": "first", "title_of_class": "first"}
    source = "raw_value" in infotable
    if source:
        # A source book keeps the first class in POSITION_TYPES order among a CUSIP's lines, and unknown
        # dollars cannot become zero through the CUSIP sum (including mixed legs).
        typed["_rank"] = np.select([masks[name].to_numpy(dtype=bool) for name in POSITION_TYPES], list(range(len(POSITION_TYPES))))
        aggregations["_rank"] = "min"
        for column in _VALUE_BY_TYPE.values():
            typed["_unknown_" + column] = typed[column].isna()
            aggregations["_unknown_" + column] = "any"
    typed = typed.dropna(subset=["cusip"])
    if typed.empty:
        return pd.DataFrame(columns=_BOOK_COLS)

    out = typed.groupby("cusip", as_index=False).agg(aggregations)
    if source:
        out["position_type"] = np.asarray(POSITION_TYPES, dtype=object)[out["_rank"].to_numpy()]
        for column in _VALUE_BY_TYPE.values():
            out[column] = out[column].mask(out["_unknown_" + column])
    else:
        out["position_type"] = _dominant_type(out)
    out["cik"] = pad_cik(cik)  # the stored form; the PK join depends on matching it
    out["period"] = pd.Timestamp(period)
    out["filing_date"] = pd.Timestamp(filing_date)
    return out.dropna(subset=["period"])[_BOOK_COLS]


def _garbage_rows(book: pd.DataFrame) -> pd.Series:
    """Rows no information table holds: a valued common line with zero shares, a position above
    `_MAX_POSITION_USD`, a negative amount, or table text merged into `issuer_name`."""
    zero_shares = (book["position_type"] == "common") & (book["shares"] == 0) & (book["value_usd"] > 0)
    table_text = book["issuer_name"].fillna("").str.contains(_TABLE_TEXT_IN_NAME, regex=True)
    return zero_shares | (book["value_usd"] > _MAX_POSITION_USD) | (book["shares"] < 0) | (book["value_usd"] < 0) | table_text


def _warn_blank_cusip_lines(accession_number: str, info: pd.DataFrame) -> None:
    """Name the holding lines the legacy parser set aside for a blank CUSIP; the log is their record."""
    if lines := info.attrs.get("blank_cusip_lines"):
        logger.warning(
            "13F %s: excluded %s holding line(s) without a source CUSIP, VALUE %s in source cells; the other %s row(s) reconcile with the cover",
            accession_number,
            lines,
            info.attrs.get("blank_cusip_value"),
            len(info),
        )


def _legacy_book(stamp: FilingStamp, raw: str) -> pd.DataFrame:
    """The book from the source-checked text parse. Raises `ValueError` when the text cannot be
    verified against its SEC cover totals or still yields a garbage row: never a partial book."""
    try:
        info = parse_legacy_information_table(raw)
        book = _book_frame(stamp.cik, stamp.filed, stamp.period_of_report, info)
    except ValueError as e:
        raise ValueError(f"legacy text table unverifiable against its SEC cover totals: {e}") from e
    _warn_blank_cusip_lines(stamp.accession_number, info)
    n_garbage = int(_garbage_rows(book).sum())
    if n_garbage:
        raise ValueError(f"legacy text table unverifiable: {n_garbage} source row(s) carry zero shares, a negative or >$1tn value, or table text")
    logger.info(f"13F {stamp.accession_number}: EdgarTools' text parse disagreed with the source; {len(book)} verified legacy row(s) used")
    return book


def _source_information_table(report: Any, stamp: FilingStamp, xml: str | None, raw: str | None) -> tuple[pd.DataFrame, ValueUnits]:
    """Raw parser output and source-unit evidence, for the versioned-book path only."""
    schema = sec_call(getattr, report, "_schema_version", None, label=f"{stamp.accession_number} schema_version")
    units = resolve_value_units(stamp.filed, schema_version=schema, information_table_text=raw)
    if xml:
        root = ElementTree.fromstring(xml)  # Strict parse: library recovery must not certify a partial book.
        source_rows = [element for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "infoTable"]
        table = parse_infotable_xml(xml)
        if len(table) != len(source_rows):
            raise ValueError("13F raw XML row count differs from parsed information table")
    elif raw:
        table = parse_legacy_information_table(raw, value_factor=1.0, verify_values=False)
        _warn_blank_cusip_lines(stamp.accession_number, table)
    else:
        return pd.DataFrame(), units
    if table.empty:
        return table, units
    normalized = normalize_information_values(table, units)
    if raw and units.factor is not None:
        value_column = next(column for column in normalized if column.lower() == "value")
        error = _value_total_error(raw, normalized[value_column].tolist(), default_factor=units.factor)
        if error:
            units = ValueUnits(units.expected_factor, units.stated_factor, None, "conflicting_cover_total")
            normalized = normalize_information_values(table, units)
    return normalized, units


class _SgmlXmlFiling:
    """An EdgarTools `Filing` whose `xml()` reads the SGML submission only.

    `Filing.xml()` falls back to the filing homepage when the submission has no primary XML, which is
    every pre-2013 13F. EdgarTools builds that URL as `<cik>/<accession>-index.html`, which EDGAR answers
    403, so the fallback costs one request and returns nothing. Every other attribute is the filing's own.
    """

    def __init__(self, filing: Filing) -> None:
        self._filing = filing

    def __getattr__(self, name: str) -> Any:
        return getattr(self._filing, name)

    def xml(self) -> str | None:
        return self._filing.sgml().xml()

    def obj(self) -> Any:
        """The filing's data object; a submission without primary XML is built on this wrapper."""
        return self._filing.obj() if self.xml() else data_object(self)


def _report(filing: Any) -> Any:
    """`filing_obj` without the pre-XML homepage request (see `_SgmlXmlFiling`)."""
    return filing_obj(_SgmlXmlFiling(filing) if isinstance(filing, Filing) else filing)


def _read_filing(stamp: FilingStamp, *, source_values: bool = False) -> pd.DataFrame | ReadFailure:
    """Fetch and parse one 13F-HR into its book: empty for an empty info table, a `ReadFailure`
    (not logged; the caller decides) when the read failed, so one bad filing never aborts a batch.
    The failure is transient exactly when `sec_io` gave up on a throttle, 5xx or network error.
    An XML-era book is EdgarTools' parse as is. A pre-XML text book that EdgarTools failed on,
    disagrees with the source text, or holds a garbage row is re-read by the source-checked legacy
    parser; an unverifiable one is a deterministic failure (skipped, never stored, never held back).
    The information-table documents are downloaded under `sec_io` before EdgarTools parses them, so
    a parse error is never read as a throttle."""
    try:
        report = _report(stamp.filing)
        xml = sec_call(getattr, report, "infotable_xml", None, label=f"{stamp.accession_number} infotable_xml")
        raw = None if xml else sec_call(getattr, report, "infotable_txt", None, label=f"{stamp.accession_number} infotable_txt")
        if source_values:
            infotable, units = _source_information_table(report, stamp, xml, raw)
            book = pd.DataFrame(columns=_BOOK_COLS) if infotable.empty else _book_frame(stamp.cik, stamp.filed, stamp.period_of_report, infotable)
            book.attrs["value_units"] = units
            return book
        try:
            infotable, edgar_error = report.infotable, None
        except Exception as e:  # noqa: BLE001 -- malformed legacy text makes EdgarTools raise
            infotable, edgar_error = None, e
        if edgar_error is not None and not raw:
            raise edgar_error
        book = pd.DataFrame() if infotable is None or infotable.empty else _book_frame(stamp.cik, stamp.filed, stamp.period_of_report, infotable)
        if raw and (edgar_error is not None or needs_legacy_fallback(raw, infotable) or _garbage_rows(book).any()):
            try:
                return _legacy_book(stamp, raw)
            except ValueError as e:
                return ReadFailure(transient=False, reason=f"ValueError: {e}")
        return book
    except Exception as e:  # noqa: BLE001
        return ReadFailure(transient=isinstance(e, TransientReadError), reason=f"{type(e).__name__}: {e}")


def _resolve_tickers(book: pd.DataFrame, cmap: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    """The `sec13f_hr` slice of a book: CUSIPs the map resolves to a universe ticker, `_HR_COLS`."""
    out = book[[c for c in _HR_COLS if c != "ticker"]].merge(cmap, on="cusip", how="inner")
    return out[out["ticker"].isin(universe)][_HR_COLS]


def _suspect_prices(df: pd.DataFrame) -> int:
    """Source common positions with extreme implied prices, before aggregate P1 repair."""
    if df.empty:
        return 0
    implied = (df["value_usd"] / df["shares"].where(df["shares"] > 0)).dropna()
    return int((~implied.between(*_IMPLIED_PRICE_BAND)).sum())


def _latest_per_key(df_book: pd.DataFrame) -> pd.DataFrame:
    """One row per `_BOOK_KEY`, the last filed winning: a stable sort on `filing_date`, so rows
    passed in (filed, amendment last, accession) order keep a same-day amendment after its
    original. Also keeps a PK from repeating in one upsert, which Postgres rejects."""
    return df_book.sort_values("filing_date", kind="stable").drop_duplicates(subset=_BOOK_KEY, keep="last")


def _padded(df: pd.DataFrame) -> pd.DataFrame:
    """`df` with its `cik` in the stored 10-digit form, whatever form it arrived in; a `sec13f_hr`
    marker row keeps its sentinel."""
    return df.assign(cik=pad_cik_series(df["cik"]).where(~marker_mask(Tables.sec13f_hr, df), df["cik"]))


def save_hr(context: Context, hr: pd.DataFrame) -> int:
    """Upsert `sec13f_hr` rows with padded CIKs; the one writer of that table. Returns rows saved."""
    return context.store.save(Tables.sec13f_hr, _padded(hr)) if not hr.empty else 0


def _save_book(context: Context, book: pd.DataFrame) -> tuple[int, int]:
    """Upsert manager-book rows with padded CIKs. Returns (rows saved, suspect-price rows)."""
    if book.empty:
        return 0, 0
    return context.store.save(Tables.sec13f_manager_holdings, _padded(book[_BOOK_COLS])), _suspect_prices(book)


def _ticker_map(context: Context, book: pd.DataFrame, walk: _WalkState) -> pd.DataFrame:
    """Reuse the full per-walk cache, including misses; failed lookups stay unknown and retryable."""
    new_cusips = set(book["cusip"].dropna()) - walk.looked_up
    if new_cusips or walk.cmap is None:
        walk.cmap = build_cusip_ticker_map(context, sorted(new_cusips), cached=walk.cmap, include_misses=True)
        walk.looked_up.update(walk.cmap["cusip"])
    return walk.cmap


## for test
def _save_batch(context: Context, book: pd.DataFrame, universe: set[str], roster_ciks: set[str], walk: _WalkState) -> None:
    """Upsert one batch of books: the universe slice to `sec13f_hr`, roster managers' rows to
    `sec13f_manager_holdings`; the last filed wins per (cik, period, cusip). Counts accumulate on
    `walk`."""
    book = _latest_per_key(_padded(book))
    hr = _resolve_tickers(book, _ticker_map(context, book, walk), universe)
    if not hr.empty:
        walk.hr_suspect += _suspect_prices(hr)
        walk.hr_saved += save_hr(context, hr)
    saved, suspect = _save_book(context, book[book["cik"].isin(roster_ciks)])
    walk.book_saved += saved
    walk.book_suspect += suspect


def _resolve_window(
    context: Context, years_history: int, filing_window: tuple[str, str] | None, as_of: pd.Timestamp, scoped: bool
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp | None]:
    """`(since, until, hold_from)` filing dates. A `filing_window` is read as given and never holds the
    walk back. Otherwise `max(filing_date)` minus the overlap (the history floor on an empty table) to
    `as_of`, capped at the stored frontier on a scoped run; a filing filed on or after `hold_from` that
    still fails holds back every filing after it."""
    if filing_window is not None:
        since, until = (cast(pd.Timestamp, pd.Timestamp(d).normalize()) for d in filing_window)
        if since > until:
            raise ValueError(f"filing_window is inverted: {since:%Y-%m-%d} > {until:%Y-%m-%d}")
        logger.warning(f"13F BACKFILL of filing window {since:%Y-%m-%d}:{until:%Y-%m-%d} -- the watermark is neither read nor advanced by this run")
        return since, until, None

    overlap = pd.Timedelta(days=cast(Resume, Tables.sec13f_hr.resume).overlap_days)
    hold_from = as_of - overlap
    legacy_frontier = context.store.max_date(Tables.sec13f_hr, "filing_date")
    if legacy_frontier is not None:
        context.log.warning("13F: filing versions start at the legacy frontier; older source versions need historical recovery.")
    else:
        since = document_floor(Tables.sec13f_hr, as_of, years_history)
        logger.warning(f"{Tables.sec13f_hr} has no stored filing_date -- full history from {since:%Y-%m-%d}")
        return since, as_of, hold_from
    if scoped:
        logger.info(f"13F: scoped run reads up to the stored frontier {legacy_frontier:%Y-%m-%d} only, so the next window is unchanged")
    return legacy_frontier - overlap, min(as_of, legacy_frontier) if scoped else as_of, hold_from


def _listing(forms: list[str], since: pd.Timestamp, until: pd.Timestamp) -> Filings | None:
    """The 13F listing filed in `since:until`, one quarterly index at a time on a fresh HTTP client:
    edgartools' file cache never closes a 304 revalidation, so each cached index leaks a pooled
    connection and a long history exhausts the pool (httpx.PoolTimeout)."""
    parts = []
    for quarter in pd.period_range(since, until, freq="Q"):
        close_clients()
        start, end = max(since, quarter.start_time), min(until, quarter.end_time.normalize())
        part = get_filings(form=forms, filing_date=f"{start:%Y-%m-%d}:{end:%Y-%m-%d}")
        if part:
            parts.append(part.data)
    return Filings(pa.concat_tables(parts)) if parts else None


def _oldest_first(listing: Filings) -> Filings:
    """The listing ordered by (filing_date, amendment last, accession): a same-day 13F-HR/A
    follows its original whatever the filer agents' accession prefixes."""
    data = listing.data
    is_amendment = pc.ends_with(pc.utf8_upper(data.column("form")), "/A")
    keys = [("filing_date", "ascending"), ("_is_amendment", "ascending"), ("accession_number", "ascending")]
    return Filings(data.take(pc.sort_indices(data.append_column("_is_amendment", is_amendment), sort_keys=keys)))


@dataclass
class _Read:
    """One listed filing in walk order and its latest read: a book, or a `ReadFailure`."""

    stamp: FilingStamp
    result: tuple[pd.DataFrame, pd.DataFrame] | ReadFailure

    @property
    def transient(self) -> bool:
        return isinstance(self.result, ReadFailure) and self.result.transient


@dataclass
class _Saver:
    """Save each accession atomically and accumulate walk diagnostics."""

    context: Context
    universe: set[str]
    roster_ciks: set[str]
    save_every: int
    walk: _WalkState = field(default_factory=_WalkState)

    def add(self, read: _Read) -> None:
        """Save a verified filing; report an unreadable source without certifying completion."""
        if isinstance(read.result, ReadFailure):
            logger.error(f"13F {read.stamp.accession_number} (filed {read.stamp.filed:%Y-%m-%d}) is unreadable ({read.result.reason}); skipped")
        else:
            manifest, book = read.result
            self.save_packet(manifest, book)

    def save_packet(self, manifest: pd.DataFrame, book: pd.DataFrame) -> None:
        """Preserve the full source book in each existing destination, without latest-only collapse."""
        cmap = _ticker_map(self.context, book, self.walk)
        broad = book.merge(cmap[["cusip", "ticker"]].drop_duplicates("cusip"), on="cusip", how="left")
        broad["ticker"] = broad["ticker"].fillna(Tables.sec13f_hr.empty_marker[1])
        self.walk.hr_saved += self.context.store.save_13f_filing(manifest, broad, table=Tables.sec13f_hr)
        self.walk.hr_suspect += _suspect_prices(book)
        if str(manifest.iloc[0].cik) in self.roster_ciks:
            self.walk.book_saved += self.context.store.save_13f_filing(manifest, book)
            self.walk.book_suspect += _suspect_prices(book)


def _done_in(context: Context, table: Table, joint_ciks: set[str]) -> Callable[[FilingStamp], bool]:
    """Whether a listed filing is complete in `table` (markers count): one table-wide read, plus one
    per-CIK read for the filers of an accession listed under several CIKs, stored once per filer."""
    where = {"parse_status": "complete"}
    everywhere = stored_accessions(context, table, [""], DONE_TABLE, where=where)[""]
    per_cik = stored_accessions(context, table, sorted(joint_ciks), DONE_PER_KEY, where=where) if joint_ciks else {}
    return lambda stamp: stamp.accession_number in per_cik.get(stamp.cik, everywhere)


def _first_pass(filings: Filings | list[Any], saver: _Saver) -> list[_Read]:
    """Read every filing in order, saving books until the first transient failure; from that
    filing on, the reads are returned unsaved for the retry rounds. A filing is skipped when it is
    complete in `sec13f_hr` and, for a roster CIK, in the manager table."""
    held: list[_Read] = []
    filers: dict[str, set[str]] = {}
    for filing in filings:
        stamp = FilingStamp.of(filing, "")
        filers.setdefault(stamp.accession_number, set()).add(stamp.cik)
    joint = {cik for ciks in filers.values() if len(ciks) > 1 for cik in ciks}
    roster = saver.roster_ciks & set().union(*filers.values())
    hr_done = _done_in(saver.context, Tables.sec13f_hr, joint)
    manager_done = _done_in(saver.context, Tables.sec13f_manager_holdings, joint & roster) if roster else hr_done
    for filing in tqdm(filings, total=len(filings), desc="13F-HR"):
        stamp = FilingStamp.of(filing, "")
        if hr_done(stamp) and (stamp.cik not in roster or manager_done(stamp)):
            continue
        read = _Read(stamp, read_filing_vintage(stamp))
        if held or read.transient:
            held.append(read)
        else:
            saver.add(read)
    return held


def _retry_rounds(context: Context, held: list[_Read]) -> None:
    """Re-read the transiently failed filings in up to `data_extract.retry_rounds.rounds` rounds, in place."""
    policy = RetryRounds.from_config(getattr(context, "config", None))
    for round_no in range(1, policy.rounds + 1):
        failed = [read for read in held if read.transient]
        if not failed:
            return
        wait = policy.waits[min(round_no - 1, len(policy.waits) - 1)]
        logger.info(f"13F: retry round {round_no}/{policy.rounds} for {len(failed)} filing(s) after {wait:.0f}s")
        _sleep(wait)
        for read in failed:
            read.result = read_filing_vintage(read.stamp)


def _save_held(held: list[_Read], hold_from: pd.Timestamp | None, saver: _Saver) -> int:
    """Save the held reads in order, stopping at the first one still failing that was filed on or after
    `hold_from`; an older failure is skipped with an ERROR. Returns the number of filings held back."""
    for i, read in enumerate(held):
        if not read.transient:
            saver.add(read)
            continue
        reason, filed = cast(ReadFailure, read.result).reason, read.stamp.filed.normalize()
        if hold_from is not None and filed >= hold_from:
            logger.warning(
                f"13F: {read.stamp.accession_number} (filed {filed:%Y-%m-%d}) still fails ({reason}); it and the "
                f"{len(held) - i - 1} filing(s) after it are held back, so the next run starts at or before it"
            )
            return len(held) - i
        logger.error(f"13F: {read.stamp.accession_number} (filed {filed:%Y-%m-%d}) still fails ({reason}) and is older than the overlap; skipped")
    return 0


def _log_walk(walk: _WalkState, total: int) -> None:
    """Suspect-price warnings and the saved-row summary for both tables."""
    for table, saved, suspect in (
        (Tables.sec13f_hr, walk.hr_saved, walk.hr_suspect),
        (Tables.sec13f_manager_holdings, walk.book_saved, walk.book_suspect),
    ):
        if suspect:
            logger.warning(
                f"13F: {suspect}/{saved} rows saved to {table} imply a share price outside "
                f"{_IMPLIED_PRICE_BAND} -- check edgartools' per-filing $thousands "
                f"detection before trusting value_usd"
            )
        logger.info(f"13F: saved {saved} row(s) from {total} filing(s) to {table}")


def fetch_13f(
    context: Context,
    tickers: list[str],
    years_history: int,
    save_every: int = 1000,
    filing_window: tuple[str, str] | None = None,
    as_of: pd.Timestamp | None = None,
    *,
    all_tickers: list[str] | None = None,
) -> int:
    """Save verified original/amendment accessions separately, oldest first in the resume window.

    Latest-only compatibility tables are still batched. Completed accession packets are not re-read.
    Returns transiently unread filings; explicit filing windows leave the regular frontier unchanged.
    """
    context.ensure_edgar_identity()
    configure(context)
    run_date = cast(pd.Timestamp, pd.Timestamp(as_of if as_of is not None else pd.Timestamp.today()).normalize())
    universe = set(tickers)
    scoped = bool(set(load_universe_tickers(context) if all_tickers is None else all_tickers) - universe)
    since, until, hold_from = _resolve_window(context, years_history, filing_window, run_date, scoped)

    # load CIKs of super investors
    roster_ciks = roster_cik_union(context)
    if not roster_ciks:
        logger.warning(f"13F: superinvestor_roster holds no CIK -- writing {Tables.sec13f_hr} only, no manager books")

    listing = _listing(list(cast(Resume, Tables.sec13f_hr.resume).forms), since, until)
    filings = _oldest_first(listing) if listing else []
    total = len(filings)
    logger.info(f"13F: {total} filing(s) to read in {since:%Y-%m-%d}:{until:%Y-%m-%d}")
    if not total:
        return 0

    saver = _Saver(context, universe, roster_ciks, save_every)
    held = _first_pass(filings, saver)
    _retry_rounds(context, held)
    n_held = _save_held(held, hold_from, saver)
    _log_walk(saver.walk, total - n_held)
    return sum(1 for read in held if read.transient)


def filing_vintage(
    information_table: pd.DataFrame,
    *,
    cik: str,
    accession_number: str,
    period: object,
    filing_date: object,
    amendment_type: str,
    source: str,
    declared_rows: int | None = None,
    declared_value: float | None = None,
    restricted: bool = False,
    report_type: str = "13F HOLDINGS REPORT",
    amendment_number: int = 0,
    units: ValueUnits | None = None,
    accepted_at: object = None,
    source_checksum: str | None = None,
    source_accounted: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare one complete accession; raw typed dollars never lose their source basis."""
    kind = amendment_type.strip().lower().replace("new holdings", "additions")
    if kind not in {"original", "restatement", "additions"}:
        raise ValueError(f"Unknown 13F amendment type: {amendment_type!r}")
    if not accession_number or pd.isna(pd.Timestamp(period)) or pd.isna(pd.Timestamp(filing_date)):
        raise ValueError("13F accession, report period and filing date are required")
    units = units or resolve_value_units(filing_date)
    declared_rows = None if pd.isna(declared_rows) else int(declared_rows)
    declared_value = None if pd.isna(declared_value) else float(declared_value)
    if source_accounted:
        units = replace(units, evidence=SEC_13F_SOURCE_ACCOUNTED + units.evidence)
    raw = information_table.copy()
    raw["VALUE"] = pd.to_numeric(_pick(raw, "VALUE"), errors="coerce")
    amounts = pd.to_numeric(_pick(raw, "SSHPRNAMT", "SharesPrnAmount"), errors="coerce")
    if not amounts.map(math.isfinite).all() or amounts.lt(0).any() or not raw["VALUE"].map(math.isfinite).all() or raw["VALUE"].lt(0).any():
        raise ValueError("13F source quantities/values are incomplete or negative")
    total = float(raw["VALUE"].sum())
    cover_factor = reconcile_13f_source(len(raw), declared_rows, total, declared_value, source_accounted=source_accounted)
    if declared_rows is None or declared_value is None:
        logger.warning(
            "13F %s: source accounted for %s position rows; cover count=%s, VALUE total=%s; missing declarations retained as unknown, market checks remain active",
            accession_number,
            len(raw),
            declared_rows,
            declared_value,
        )
    if declared_rows is not None and len(raw) != declared_rows:
        logger.warning(
            "13F %s: accepted %s/%s source position rows (%.2f%%); within source count tolerance; book participates normally in exits",
            accession_number,
            len(raw),
            declared_rows,
            100 * len(raw) / declared_rows,
        )
    if cover_factor != 1.0:
        logger.warning(
            "13F %s: source VALUE sum=%s, cover total=%s, cover-to-row factor=%s; raw values retained; row USD units=%s, aggregate market check remains active",
            accession_number,
            total,
            declared_value,
            cover_factor,
            units.factor,
        )
    raw["raw_value"] = raw["VALUE"]
    book = _book_frame(cik, filing_date, period, raw)
    cusips = _pick(raw, "CUSIP").map(literal_13f_cusip)
    if cusips.isna().any():
        raise ValueError("13F source CUSIP identity is missing")
    for column in _VALUE_BY_TYPE.values():
        book["raw_" + column] = book[column]
        book[column] = book[column] * units.factor if units.factor is not None else book[column].where(book[column].eq(0))
    classes: dict[str, set[str]] = {}
    titles = _pick(raw, "TITLEOFCLASS", "Class").fillna("").astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    for cusip, title in zip(cusips, titles, strict=True):
        classes.setdefault(cusip, set()).add(title)
    book["class_evidence"] = book["cusip"].map({cusip: json.dumps(sorted(seen)) for cusip, seen in classes.items()})
    book["accession_number"] = accession_number
    checksum = source_checksum
    if not checksum:
        # Include cover semantics in the canonical checksum: changed amendment meaning is a conflict.
        canonical = {
            "cik": pad_cik(cik),
            "accession": accession_number,
            "period": str(pd.Timestamp(period).date()),
            "filed": str(pd.Timestamp(filing_date).date()),
            "kind": kind,
            "restricted": restricted,
            "report_type": report_type,
            "amendment_number": amendment_number,
            "declared_rows": declared_rows,
            "declared_value": declared_value,
            "unit_factor": units.factor,
            "unit_evidence": units.evidence,
            "rows": raw.fillna("").astype(str).sort_values(list(raw.columns), kind="stable").to_dict("records"),
        }
        checksum = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
    accepted = pd.to_datetime(accepted_at, utc=True) if accepted_at is not None else pd.NaT
    manifest = pd.DataFrame(
        [
            {
                "accession_number": accession_number,
                "cik": pad_cik(cik),
                "period": pd.Timestamp(period),
                "filing_date": pd.Timestamp(filing_date),
                "form": "13F-HR" if kind == "original" else "13F-HR/A",
                "amendment_type": kind,
                "amendment_number": int(amendment_number),
                "report_type": report_type,
                "accepted_at": accepted,
                "clock_source": "SEC acceptance" if pd.notna(accepted) else "SEC filing date",
                "clock_timezone": "UTC" if pd.notna(accepted) else "America/New_York",
                "clock_precision": "timestamp" if pd.notna(accepted) else "date",
                "restricted": bool(restricted),
                "source": source,
                "source_checksum": checksum,
                "expected_unit_factor": units.expected_factor,
                "stated_unit_factor": units.stated_factor,
                "value_unit_factor": units.factor,
                "value_unit_evidence": units.evidence,
                "source_row_count": len(raw),
                "declared_row_count": declared_rows,
                "source_value_total": total,
                "declared_value_total": declared_value,
                "parsed_row_count": len(book),
                "parse_status": "complete",
                "completed_at": pd.Timestamp.now(tz="UTC").tz_localize(None),
            }
        ]
    )
    # Stable dtypes also when an empty or unknown-unit book is the first saved accession.
    manifest["declared_row_count"] = pd.to_numeric(manifest["declared_row_count"], errors="coerce").astype("Int64")
    for column in ("expected_unit_factor", "stated_unit_factor", "value_unit_factor", "declared_value_total"):
        manifest[column] = pd.to_numeric(manifest[column], errors="coerce").astype(float)
    manifest["accepted_at"] = pd.to_datetime(manifest["accepted_at"], utc=True).dt.tz_localize(None)
    for column in (*_VALUE_BY_TYPE.values(), *("raw_" + c for c in _VALUE_BY_TYPE.values()), "shares", "call_shares", "put_shares", "debt_prn"):
        book[column] = pd.to_numeric(book[column], errors="coerce").astype(float)
    for column in ("cik", "accession_number", "cusip", "issuer_name", "title_of_class", "position_type", "class_evidence"):
        book[column] = book[column].astype("string")
    return manifest, book


def read_filing_vintage(stamp: FilingStamp) -> tuple[pd.DataFrame, pd.DataFrame] | ReadFailure:
    """Strict source parse for accession recovery; never certify an unverified book."""
    try:
        report = _report(stamp.filing)
        xml = sec_call(getattr, report, "infotable_xml", None, label=f"{stamp.accession_number} infotable_xml")
        if not xml and getattr(report, "primary_form_information", None) is not None and getattr(report, "total_holdings", None) == 0:
            return _empty_xml_vintage(stamp, sec_call(stamp.filing.xml, label=f"{stamp.accession_number} cover XML"))
        if not xml:
            raw = sec_call(getattr, report, "infotable_txt", None, label=f"{stamp.accession_number} infotable_txt")
            # The information-table text alone serves when it carries the cover total and a checked report type.
            if not raw or not (_VALUE_TOTAL.search(raw) or _FOOTER_TOTAL.search(raw)) or not _CHECKED_REPORT.search(raw):
                raw = sec_call(stamp.filing.full_text_submission, label=f"{stamp.accession_number} full legacy submission")
            if not raw:
                raise ValueError("13F legacy information table is missing")
            return text_filing_vintage(raw, stamp)
        primary = sec_call(stamp.filing.xml, label=f"{stamp.accession_number} cover XML")
        return xml_filing_vintage(
            primary,
            xml,
            cik=stamp.cik,
            accession_number=stamp.accession_number,
            period=stamp.period_of_report,
            filing_date=stamp.filed,
            source=stamp.doc_url,
            accepted_at=getattr(stamp.filing, "acceptance_datetime", None),
            expected_form=stamp.form,
        )
    except Exception as exc:  # noqa: BLE001 -- one malformed source is not a complete filing
        return ReadFailure(isinstance(exc, TransientReadError), f"{type(exc).__name__}: {exc}")


def _stated_cover_unit(raw: str, summary: re.Match[str]) -> float | None:
    """Dollars per unit stated on the cover value-total line (caption to end of line); NaN when it states
    two different units, None when it states none."""
    end = raw.find("\n", summary.end())
    line = raw[summary.start() : end if end >= 0 else len(raw)]
    stated = {factor for pattern, factor in _COVER_UNITS if pattern.search(line)}
    if not stated and _COVER_DOLLARS.search(line):
        stated = {1.0}
    return None if not stated else stated.pop() if len(stated) == 1 else math.nan


def _cover_ratio_units(raw: str, units: ValueUnits, cell_total: float) -> ValueUnits:
    """Cell units from the cover total over the VALUE-cell sum. A unit stated on the cover line wins: the
    cover in dollars over the cell sum must be 1, 1000 or 1e6, else dollars are withheld. Unlabelled, a ratio
    of 1000 or 1e6 is a dollar cover over $1000s or $millions cells and 0.001 a $1000s cover over dollar
    cells; any other ratio, including 1, keeps `units`. A disagreeing caption withholds dollars."""
    summary = _VALUE_TOTAL.search(raw) or _FOOTER_TOTAL.search(raw)
    total = float(summary.group(1).replace(",", "")) if summary else 0.0
    if total <= 0 or cell_total <= 0:
        return units
    if units.factor is None and units.stated_factor is None:
        return units
    stated = _stated_cover_unit(raw, summary) if summary.re is _VALUE_TOTAL else None
    if stated is not None:
        factor = None if math.isnan(stated) else {0: 1.0, 3: 1000.0, 6: 1e6}.get(round(math.log10(total * stated / cell_total)))
        if factor is None:
            return ValueUnits(units.expected_factor, units.stated_factor, None, "conflicting_cover_total")
    else:
        factor = {-3: 1.0, 3: 1000.0, 6: 1e6}.get(round(math.log10(total / cell_total)))
    if factor is None or factor == units.factor:
        return units
    if units.stated_factor is not None and units.stated_factor != factor:
        return ValueUnits(units.expected_factor, units.stated_factor, None, "conflicting_cover_total")
    return ValueUnits(units.expected_factor, units.stated_factor, factor, "cover_total_ratio")


def text_filing_vintage(raw: str, stamp: FilingStamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reuse the source-reconciled pre-XML parser; unsupported cover semantics remain explicit.
    A cover declaring no entries (or a zero total and no count) is a complete empty book that keeps
    its filed total, not parsed, unless the text holds a holding (a valid CUSIP or a non-zero
    amount): that contradiction fails."""
    units = resolve_value_units(stamp.filed, information_table_text=raw)
    count = _ENTRY_COUNT.search(raw)
    total = _VALUE_TOTAL.search(raw) or _FOOTER_TOTAL.search(raw)
    declared_rows = int(count.group(1).replace(",", "")) if count else None
    declared_value = float(total.group(1).replace(",", "")) if total else None
    empty = declared_rows == 0 or (declared_rows is None and declared_value == 0)
    if empty:
        if holding_lines := _genuine_holding_lines(raw):
            raise ValueError(f"13F cover declares an empty book but the text holds {holding_lines} holding line(s)")
        info = pd.DataFrame(columns=_LEGACY_INFO_COLS)
        declared_rows, declared_value = 0, declared_value or 0.0
    else:
        info = parse_legacy_information_table(raw, value_factor=1.0, verify_values=False)
        _warn_blank_cusip_lines(stamp.accession_number, info)
        units = _cover_ratio_units(raw, units, float(info["VALUE"].sum()))
    kind = "original"
    if stamp.is_amendment:
        restated = re.search(_CHECKED + r"\s*(?:is\s+a\s+)?restatement", raw, re.I)
        added = re.search(_CHECKED + r"\s*(?:adds?\s+)?new\s+holdings", raw, re.I)
        if bool(restated) == bool(added):
            raise ValueError("13F legacy amendment type is unverifiable")
        kind = "restatement" if restated else "additions"
    reports = _CHECKED_REPORT.findall(raw)
    if len(set(reports)) > 1:
        raise ValueError("Conflicting checked legacy report types")
    report_type = re.sub(r"13\s*F", "13F", re.sub(r"\s+", " ", reports[0])).upper() if reports else SEC_13F_INFERRED_HOLDINGS_REPORT
    if not reports and not empty:
        logger.warning("13F %s: intact HR information table with no checked report type; inferred holdings-report scope", stamp.accession_number)
    return filing_vintage(
        info,
        cik=stamp.cik,
        accession_number=stamp.accession_number,
        period=stamp.period_of_report,
        filing_date=stamp.filed,
        amendment_type=kind,
        source=stamp.doc_url or "SEC legacy text",
        declared_rows=declared_rows,
        declared_value=declared_value,
        report_type=report_type,
        restricted=bool(re.search(r"confidential\s+information\s+has\s+been\s+omitted", raw, re.I)),
        accepted_at=getattr(stamp.filing, "acceptance_datetime", None),
        units=units,
        source_checksum=hashlib.sha256(raw.encode()).hexdigest(),
        source_accounted=bool(info.attrs.get("source_accounted")),
    )


def _xml_cover(primary_xml: str, *, period: object, expected_form: str | None) -> tuple[dict[str, Any], str | None]:
    """The validated cover XML as `filing_vintage` cover keywords, and its `schemaVersion`."""
    root = ElementTree.fromstring(primary_xml)
    fields = {element.tag.rsplit("}", 1)[-1]: (element.text or "").strip() for element in root.iter()}
    # Original SEC XML may omit the false checkbox; only its cover form can establish that default.
    default = "false" if fields.get("submissionType") == "13F-HR" else ""
    flag = fields.get("isAmendment", default).lower()
    if flag not in {"true", "false", "1", "0"}:
        raise ValueError("13F cover amendment flag is missing or invalid")
    is_amendment = flag in {"true", "1"}
    cover_form = fields.get("submissionType")
    if cover_form is not None and (cover_form not in cast(Resume, Tables.sec13f_hr.resume).forms or is_amendment != cover_form.endswith("/A")):
        raise ValueError("13F cover amendment flag disagrees with cover form")
    if expected_form is not None and is_amendment != expected_form.endswith("/A"):
        raise ValueError("13F cover amendment flag disagrees with filing form")
    reported = pd.to_datetime(fields.get("reportCalendarOrQuarter", ""), errors="coerce")
    if pd.isna(reported) or reported.normalize() != pd.Timestamp(period).normalize():
        raise ValueError("13F cover report period disagrees with accession report period")
    cover = {
        "amendment_type": fields.get("amendmentType", "") if is_amendment else "original",
        "declared_rows": int(fields["tableEntryTotal"]),
        "declared_value": float(fields["tableValueTotal"]),
        "restricted": fields.get("isConfidentialOmitted", "false").strip().lower() in {"true", "1", "y"},
        "report_type": fields.get("reportType", ""),
        "amendment_number": int(fields.get("amendmentNo", "0")),
    }
    return cover, fields.get("schemaVersion")


def _xml_checksum(primary_xml: str, information_xml: str) -> str:
    """The source checksum of an XML accession: both original documents, byte for byte."""
    return hashlib.sha256((primary_xml + "\n" + information_xml).encode()).hexdigest()


def xml_filing_vintage(
    primary_xml: str,
    information_xml: str,
    *,
    cik: str,
    accession_number: str,
    period: object,
    filing_date: object,
    source: str,
    accepted_at: object = None,
    expected_form: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exact original documents, using the same completion and unit contract as bulk sources."""
    cover, schema = _xml_cover(primary_xml, period=period, expected_form=expected_form)
    info_root = ElementTree.fromstring(information_xml)
    source_count = sum(element.tag.rsplit("}", 1)[-1] == "infoTable" for element in info_root.iter())
    info = parse_infotable_xml(information_xml)
    if source_count != len(info):
        raise ValueError("13F raw XML row count differs from parsed information table")
    return filing_vintage(
        info,
        cik=cik,
        accession_number=accession_number,
        period=period,
        filing_date=filing_date,
        source=source,
        accepted_at=accepted_at,
        units=resolve_value_units(filing_date, schema_version=schema),
        source_checksum=_xml_checksum(primary_xml, information_xml),
        **cover,
    )


def _empty_xml_vintage(stamp: FilingStamp, primary_xml: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """An XML cover declaring no entries, filed without an information table: a complete empty book."""
    cover, schema = _xml_cover(primary_xml, period=stamp.period_of_report, expected_form=stamp.form)
    if cover["declared_rows"] != 0:
        raise ValueError("13F cover XML entry count disagrees with the parsed summary page")
    return _stamp_vintage(pd.DataFrame(), stamp, cover, resolve_value_units(stamp.filed, schema_version=schema), _xml_checksum(primary_xml, ""))


def _stamp_vintage(
    info: pd.DataFrame, stamp: FilingStamp, cover: dict[str, Any], units: ValueUnits, checksum: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`filing_vintage` for one listed XML-era filing, from its stamp and validated cover."""
    return filing_vintage(
        info,
        cik=stamp.cik,
        accession_number=stamp.accession_number,
        period=stamp.period_of_report,
        filing_date=stamp.filed,
        source=stamp.doc_url,
        accepted_at=getattr(stamp.filing, "acceptance_datetime", None),
        units=units,
        source_checksum=checksum,
        **cover,
    )
