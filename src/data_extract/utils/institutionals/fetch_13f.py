"""
fetch_13f.py (src/data_extract/utils/institutionals/fetch_13f.py)
-----------------------------------------------------------------
One oldest-first walk over every 13F-HR by filing date (edgartools), feeding two tables from one parse: the
universe ticker slice of each book -> `sec13f_hr`, and the complete CUSIP book of every roster manager ->
`sec13f_manager_holdings`. In-batch dedup keeps the last filed (amendment wins). Tickers come from the
CUSIP map (OpenFIGI), never from issuer names.

The window runs from `max(sec13f_hr.filing_date)` minus the table's overlap to the run date. Transient read
failures get in-task retry rounds; a filing filed inside the overlap that still fails holds back every
filing after it (low watermark), so the next window starts at or before it. An older one is skipped with
an ERROR naming its accession. A scoped (`-t`) run never reads past the stored frontier.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import pandas as pd
import pyarrow.compute as pc
from edgar import Filings, get_filings
from tqdm import tqdm

from src.constants.constants import SEC_13F_FORMS
from src.context import Context
from src.data_extract.utils.common.edgar_driver import FilingStamp, RetryRounds
from src.data_extract.utils.common.resume import document_floor
from src.data_extract.utils.common.sec_io import TransientReadError, configure, filing_obj
from src.data_extract.utils.institutionals.fetch_cusip_map import build_cusip_ticker_map, normalize_cusip
from src.data_store.schema import Resume, Tables
from src.utils.string import pad_cik
from src.utils.superinvestor_roster import roster_cik_union
from src.utils.universe import load_universe_tickers

logger = logging.getLogger(__name__)

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

#: `sec13f_manager_holdings`' PK; the last filing wins on it.
_BOOK_KEY = ["cik", "period", "cusip"]

# edgartools infers the per-filing $thousands/$ones unit; an implied price outside this band flags a wrong inference.
_IMPLIED_PRICE_BAND = (1.0, 5000.0)

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
    val = pd.to_numeric(_pick(infotable, "VALUE"), errors="coerce").fillna(0.0)

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
    masks, amt, val = _holding_masks(infotable)
    is_stock, is_call = masks["common"], masks["call"]
    is_put, is_debt, is_other = masks["put"], masks["debt"], masks["other"]

    return pd.DataFrame(
        {
            # canonical 9-char CUSIP so the map lookup and the ticker merge use ONE form
            "cusip": _pick(infotable, "CUSIP").map(normalize_cusip),
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
    typed = _classify_holdings(infotable)
    typed["issuer_name"] = _pick(infotable, "NAMEOFISSUER", "Issuer").astype("string")
    typed["title_of_class"] = _pick(infotable, "TITLEOFCLASS", "Class").astype("string")
    typed = typed.dropna(subset=["cusip"])
    if typed.empty:
        return pd.DataFrame(columns=_BOOK_COLS)

    numeric = [c for c in typed.columns if c not in ("cusip", "issuer_name", "title_of_class")]
    out = typed.groupby("cusip", as_index=False).agg({**{c: "sum" for c in numeric}, "issuer_name": "first", "title_of_class": "first"})
    out["position_type"] = _dominant_type(out)
    out["cik"] = pad_cik(cik)  # the stored form; the PK join depends on matching it
    out["period"] = pd.Timestamp(period)
    out["filing_date"] = pd.Timestamp(filing_date)
    return out.dropna(subset=["period"])[_BOOK_COLS]


def _read_filing(stamp: FilingStamp) -> pd.DataFrame | ReadFailure:
    """Fetch and parse one 13F-HR into its book: empty for an empty info table, a `ReadFailure`
    (not logged; the caller decides) when the read failed, so one bad filing never aborts a batch.
    The failure is transient exactly when `sec_io` gave up on a throttle, 5xx or network error."""
    try:
        infotable = filing_obj(stamp.filing).infotable
        if infotable is None or infotable.empty:
            return pd.DataFrame()
        return _book_frame(stamp.cik, stamp.filed, stamp.period_of_report, infotable)
    except Exception as e:  # noqa: BLE001
        return ReadFailure(transient=isinstance(e, TransientReadError), reason=f"{type(e).__name__}: {e}")


def _resolve_tickers(book: pd.DataFrame, cmap: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    """The `sec13f_hr` slice of a book: CUSIPs the map resolves to a universe ticker, `_HR_COLS`."""
    out = book[[c for c in _HR_COLS if c != "ticker"]].merge(cmap, on="cusip", how="inner")
    return out[out["ticker"].isin(universe)][_HR_COLS]


def _suspect_prices(df: pd.DataFrame) -> int:
    """Rows whose implied share price falls outside `_IMPLIED_PRICE_BAND` -- the detector for
    edgartools inferring the wrong $thousands-vs-$ones unit on a filing."""
    if df.empty:
        return 0
    implied = (df["value_usd"] / df["shares"].where(df["shares"] > 0)).dropna()
    return int((~implied.between(*_IMPLIED_PRICE_BAND)).sum())


def _latest_per_key(df_book: pd.DataFrame) -> pd.DataFrame:
    """One row per `_BOOK_KEY`, the last filed winning: a stable sort on `filing_date`, so rows
    passed in (filed, amendment last, accession) order keep a same-day amendment after its
    original. Also keeps a PK from repeating in one upsert, which Postgres rejects."""
    return df_book.sort_values("filing_date", kind="stable").drop_duplicates(subset=_BOOK_KEY, keep="last")


def _save_book(context: Context, book: pd.DataFrame) -> tuple[int, int]:
    """Upsert manager-book rows. Returns (rows saved, suspect-price rows)."""
    if book.empty:
        return 0, 0
    return context.store.save(Tables.sec13f_manager_holdings, book[_BOOK_COLS]), _suspect_prices(book)


def _ticker_map(context: Context, book: pd.DataFrame, walk: _WalkState) -> pd.DataFrame:
    """The CUSIP->ticker map, rebuilt only when the book carries a CUSIP not looked up yet
    (`build_cusip_ticker_map` re-reads the whole map table)."""
    new_cusips = set(book["cusip"]) - walk.looked_up
    if new_cusips or walk.cmap is None:
        walk.cmap = build_cusip_ticker_map(context, sorted(new_cusips))
        walk.looked_up |= new_cusips
    return walk.cmap


def _save_batch(context: Context, book: pd.DataFrame, universe: set[str], roster_ciks: set[str], walk: _WalkState) -> None:
    """Upsert one batch of books: the universe slice to `sec13f_hr`, roster managers' rows to
    `sec13f_manager_holdings`; the last filed wins per (cik, period, cusip). Counts accumulate on
    `walk`."""
    book = _latest_per_key(book)
    hr = _resolve_tickers(book, _ticker_map(context, book, walk), universe)
    if not hr.empty:
        walk.hr_suspect += _suspect_prices(hr)
        walk.hr_saved += context.store.save(Tables.sec13f_hr, hr)
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
    watermark = context.store.max_date(Tables.sec13f_hr, "filing_date")
    if watermark is None:
        since = document_floor(Tables.sec13f_hr, as_of, years_history)
        logger.warning(f"{Tables.sec13f_hr} has no stored filing_date -- full history from {since:%Y-%m-%d}")
        return since, as_of, hold_from
    watermark = watermark.normalize()
    if scoped:
        logger.info(f"13F: scoped run reads up to the stored frontier {watermark:%Y-%m-%d} only, so the next window is unchanged")
    return watermark - overlap, min(as_of, watermark) if scoped else as_of, hold_from


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
    result: pd.DataFrame | ReadFailure

    @property
    def transient(self) -> bool:
        return isinstance(self.result, ReadFailure) and self.result.transient


@dataclass
class _Saver:
    """Books waiting to be saved, upserted every `save_every` filings with rows."""

    context: Context
    universe: set[str]
    roster_ciks: set[str]
    save_every: int
    walk: _WalkState = field(default_factory=_WalkState)
    batch: list[pd.DataFrame] = field(default_factory=list)

    def add(self, read: _Read) -> None:
        """Queue a read's book (an unreadable filing is skipped with an ERROR); save a full batch."""
        if isinstance(read.result, ReadFailure):
            logger.error(f"13F {read.stamp.accession_number} (filed {read.stamp.filed:%Y-%m-%d}) is unreadable ({read.result.reason}); skipped")
        elif not read.result.empty:
            self.batch.append(read.result)
        if len(self.batch) >= self.save_every:
            self.flush()

    def flush(self) -> None:
        if self.batch:
            _save_batch(self.context, pd.concat(self.batch, ignore_index=True), self.universe, self.roster_ciks, self.walk)
            self.batch.clear()


def _first_pass(filings: Filings | list[Any], saver: _Saver) -> list[_Read]:
    """Read every filing in order, saving books until the first transient failure; from that
    filing on, the reads are returned unsaved for the retry rounds."""
    held: list[_Read] = []
    for filing in tqdm(filings, total=len(filings), desc="13F-HR"):
        stamp = FilingStamp.of(filing, "")
        read = _Read(stamp, _read_filing(stamp))
        if held or read.transient:
            held.append(read)
        else:
            saver.add(read)
    saver.flush()
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
            read.result = _read_filing(read.stamp)


def _save_held(held: list[_Read], hold_from: pd.Timestamp | None, saver: _Saver) -> int:
    """Save the held reads in order, stopping at the first one still failing that was filed on or after
    `hold_from`; an older failure is skipped with an ERROR. Returns the number of filings held back."""
    for i, read in enumerate(held):
        if not read.transient:
            saver.add(read)
            continue
        reason, filed = cast(ReadFailure, read.result).reason, read.stamp.filed.normalize()
        if hold_from is not None and filed >= hold_from:
            saver.flush()
            logger.warning(
                f"13F: {read.stamp.accession_number} (filed {filed:%Y-%m-%d}) still fails ({reason}); it and the "
                f"{len(held) - i - 1} filing(s) after it are held back, so the next run starts at or before it"
            )
            return len(held) - i
        logger.error(f"13F: {read.stamp.accession_number} (filed {filed:%Y-%m-%d}) still fails ({reason}) and is older than the overlap; skipped")
    saver.flush()
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
    save_every: int = 600,
    filing_window: tuple[str, str] | None = None,
    as_of: pd.Timestamp | None = None,
) -> None:
    """Ingest every 13F-HR filed in the resume window (or the `filing_window` backfill, watermark
    untouched), oldest first, so an amendment overwrites its original. Each batch upserts the `tickers`
    slice to `sec13f_hr` and roster CIKs' books to `sec13f_manager_holdings`, idempotent on their PKs.
    One EDGAR walk at a time."""
    context.ensure_edgar_identity()
    configure(context)
    run_date = cast(pd.Timestamp, pd.Timestamp(as_of if as_of is not None else pd.Timestamp.today()).normalize())
    universe = {str(t).upper() for t in tickers}
    scoped = bool(set(load_universe_tickers(context)) - universe)
    since, until, hold_from = _resolve_window(context, years_history, filing_window, run_date, scoped)
    roster_ciks = roster_cik_union(context)
    if not roster_ciks:
        logger.warning(f"13F: superinvestor_roster holds no CIK -- writing {Tables.sec13f_hr} only, no manager books")

    listing = get_filings(form=cast(Any, SEC_13F_FORMS), filing_date=f"{since:%Y-%m-%d}:{until:%Y-%m-%d}")
    filings = _oldest_first(listing) if listing else []
    total = len(filings)
    logger.info(f"13F: {total} filing(s) to read in {since:%Y-%m-%d}:{until:%Y-%m-%d}")
    if not total:
        return

    saver = _Saver(context, universe, roster_ciks, save_every)
    held = _first_pass(filings, saver)
    _retry_rounds(context, held)
    n_held = _save_held(held, hold_from, saver)
    _log_walk(saver.walk, total - n_held)
