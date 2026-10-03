"""
fetch_13f.py (src/data_extract/utils/institutionals/fetch_13f.py)
-----------------------------------------------------------------
One oldest-first walk over every 13F-HR by filing date (edgartools), feeding two tables from one parse: the
universe ticker slice of each book -> `sec13f_hr`, and the complete CUSIP book of every roster manager ->
`sec13f_manager_holdings`. In-batch dedup keeps the last filed (amendment wins). Tickers come from the
CUSIP map (OpenFIGI), never from issuer names.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, cast

import pandas as pd
from edgar import Filings, get_filings
from tqdm import tqdm

from src.constants.constants import SEC_13F_FORMS
from src.context import Context
from src.data_extract.utils.common.edgar_driver import FilingStamp
from src.data_extract.utils.common.run_manifest import record_run
from src.data_extract.utils.institutionals.fetch_cusip_map import build_cusip_ticker_map, normalize_cusip
from src.data_store.schema import Tables
from src.utils.string import pad_cik
from src.utils.superinvestor_roster import roster_cik_union

logger = logging.getLogger(__name__)

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


def _read_filing(stamp: FilingStamp) -> pd.DataFrame:
    """Fetch and parse one 13F-HR into its book. Empty on any failure, logged with the accession:
    one unparseable filing must not abort a batch."""
    try:
        infotable = stamp.filing.obj().infotable
        if infotable is None or infotable.empty:
            return pd.DataFrame()
        return _book_frame(stamp.cik, stamp.filed, stamp.period_of_report, infotable)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"13F {stamp.accession_number}: {type(e).__name__}: {e}")
        return pd.DataFrame()


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
    passed in (filed, accession) order keep a same-day amendment after its original. Also keeps a
    PK from repeating in one upsert, which Postgres rejects."""
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
    context: Context, years_history: int, lookback_days: int, filing_window: tuple[str, str] | None
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(since, until) filing dates: the backfill window when given (watermark untouched), else
    `max(filing_date) - lookback_days` to today, or `years_history` back on an empty table."""
    today = cast(pd.Timestamp, pd.Timestamp.today().normalize())
    if filing_window is not None:
        since, until = (cast(pd.Timestamp, pd.Timestamp(d).normalize()) for d in filing_window)
        if since > until:
            raise ValueError(f"filing_window is inverted: {since:%Y-%m-%d} > {until:%Y-%m-%d}")
        logger.warning(f"13F BACKFILL of filing window {since:%Y-%m-%d}:{until:%Y-%m-%d} -- the watermark is neither read nor advanced by this run")
        return since, until
    watermark = context.store.max_date(Tables.sec13f_hr, "filing_date")
    if watermark is None:
        since = cast(pd.Timestamp, today - pd.DateOffset(years=years_history))
        logger.warning(f"{Tables.sec13f_hr} has no stored filing_date -- full history from {since:%Y-%m-%d}")
        return since, today
    return watermark - pd.Timedelta(days=lookback_days), today


def _record(context: Context, tickers: list[str] | None, saved: int, filing_window: tuple[str, str] | None) -> None:
    """Log the run in the manifest: as an incremental, or as a backfill that leaves every
    watermark (`last_run_date`) untouched."""
    n_tickers = len(set(tickers or ()))
    if filing_window is None:
        record_run(context, Tables.sec13f_hr, n_tickers, saved)
    else:
        record_run(context, Tables.sec13f_hr, n_tickers, saved, backfill_window=filing_window)


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
    tickers: list[str] | None = None,
    years_history: int = 15,
    save_every: int = 600,
    lookback_days: int = 7,
    filing_window: tuple[str, str] | None = None,
) -> None:
    """Ingest every 13F-HR filed since `sec13f_hr`'s latest `filing_date` minus `lookback_days`,
    or the `filing_window` backfill (watermark untouched). Each batch upserts the universe slice to
    `sec13f_hr` and roster CIKs' books to `sec13f_manager_holdings`, idempotent on their PKs. One
    EDGAR walk at a time; oldest-first, so an amendment overwrites its original."""
    context.ensure_edgar_identity()
    since, until = _resolve_window(context, years_history, lookback_days, filing_window)
    roster_ciks = roster_cik_union(context)
    if not roster_ciks:
        logger.warning(f"13F: superinvestor_roster holds no CIK -- writing {Tables.sec13f_hr} only, no manager books")

    listing = get_filings(form=cast(Any, SEC_13F_FORMS), filing_date=f"{since:%Y-%m-%d}:{until:%Y-%m-%d}")
    filings = Filings(listing.data.sort_by([("filing_date", "ascending"), ("accession_number", "ascending")])) if listing else []
    total = len(filings)
    logger.info(f"13F: {total} filing(s) to read in {since:%Y-%m-%d}:{until:%Y-%m-%d}")
    if not total:
        _record(context, tickers, 0, filing_window)
        return

    universe = set(cast(list[str], tickers))
    walk, batch = _WalkState(), []
    for i, filing in enumerate(tqdm(filings, total=total, desc="13F-HR"), start=1):
        rows = _read_filing(FilingStamp.of(filing, ""))
        if not rows.empty:
            batch.append(rows)
        if batch and (len(batch) >= save_every or i == total):
            _save_batch(context, pd.concat(batch, ignore_index=True), universe, roster_ciks, walk)
            batch.clear()

    _log_walk(walk, total)
    _record(context, tickers, walk.hr_saved, filing_window)
