"""
fetch_13f_managers.py (src/data_extract/utils/institutionals/fetch_13f_managers.py)
------------------------------------------------------------------------------------
The COMPLETE 13F book of every manager that has ever been on the Dataroma roster, at CUSIP
grain and with no universe filter -> `sec13f_manager_holdings`.

WHY A SECOND 13F TABLE. `fetch_13f` inner-joins the CUSIP map and then keeps only S&P 500
tickers, so `sec13f_hr` holds a SLICE of each manager's book. A portfolio weight computed from
that slice is inflated by a manager-specific factor -- measured over 12 roster managers' 2026Q1
filings, median S&P 500 coverage is 47% of positions and 52% of value, spanning Atlantic
Investment at 8.3%/13.1% to AltaRock at 100%/100%. Comparing conviction across managers on those
numbers compares their index exposure, not their conviction. This table is the denominator.

WALK SHAPE. Per MANAGER CIK, not by filing date: ~106 CIKs x ~60 quarters is ~6.4k filings
against the all-filer walk's ~528k. The scope is `roster_cik_union` -- the union of every CIK on
any `superinvestor_roster` snapshot, not today's roster. Walking only current members would
rebuild the exact survivorship bias the point-in-time roster table exists to remove: a manager
who left the roster in 2019 held real positions in 2016, and dropping them makes every
backtested roster statistic a function of who survived to today.

CLASSIFICATION IS IMPORTED, NOT FORKED. `_classify_holdings` and `position_type` both come from
`fetch_13f` and both read `_holding_masks`, the single definition of what makes a line stock, an
option, debt or residual. Two copies would drift and the drift would be silent.

VALUE UNITS. 13F `VALUE` is $thousands or $ones depending on the schema the FILER used -- per
filing, not per date. edgartools infers the unit and returns dollars; when it infers wrong it
scales `value_usd` by 1000x with nothing else looking odd. `_IMPLIED_PRICE_BAND` is the detector
and the count is logged, never silently accepted.

AMENDMENTS. Last filed wins per (cik, period): filings are walked oldest-first so a later
amendment upserts over the original. Version history is not preserved (README out of scope).
"""

from __future__ import annotations

import logging
import threading

import pandas as pd
from edgar import Company

from src.constants.constants import SEC_13F_FORMS
from src.context import Context
from src.data_extract.utils.common.edgar_driver import period_of_report
from src.data_extract.utils.common.parallel_fetch import run_per_ticker
from src.data_extract.utils.common.run_manifest import record_run
from src.data_extract.utils.institutionals.fetch_13f import (
    _IMPLIED_PRICE_BAND, _classify_holdings, _pick, position_type)
from src.data_store.schema import Tables
from src.utils.string import pad_cik
from src.utils.superinvestor_roster import roster_cik_union, roster_map_as_of

logger = logging.getLogger(__name__)

_COLS = ["cik", "period", "filing_date", "cusip", "issuer_name", "title_of_class",
         "position_type", "shares", "value_usd", "call_shares", "call_value",
         "put_shares", "put_value", "debt_prn", "debt_value", "other_value"]

#: The per-type value columns, in `POSITION_TYPES` order. `position_type` on a grouped row is
#: whichever of these carries the most value -- see `_dominant_type`.
_VALUE_BY_TYPE = {"common": "value_usd", "call": "call_value", "put": "put_value",
                  "debt": "debt_value", "other": "other_value"}


class SuperinvestorRosterEmpty(RuntimeError):
    """`superinvestor_roster` holds no CIK. The roster IS this walk's entire input, so an empty
    one must stop the run rather than let it report success over zero managers -- the failure
    mode a warning would produce is a table that silently stops growing."""


def _dominant_type(grouped: pd.DataFrame) -> pd.Series:
    """`position_type` for an already-grouped row: the class holding the most value.

    A CUSIP can appear on several lines of one filing with different classes -- a manager
    holding both the stock and calls on it -- and the PK collapses them to one row whose
    per-class VALUE COLUMNS keep the split. Those columns stay authoritative; this label just
    says what the row principally is, so a conviction query can exclude option and debt rows
    without unpacking five columns. Ties go to `common`, which `POSITION_TYPES` orders first."""
    values = pd.DataFrame({name: grouped[col].fillna(0.0).abs()
                           for name, col in _VALUE_BY_TYPE.items()})
    # idxmax returns the FIRST column at the max, and _VALUE_BY_TYPE is built in POSITION_TYPES
    # order, so `common` already wins a tie -- including the all-zero row, which is a disclosed
    # position with no value rather than an 'other'.
    return values.idxmax(axis=1)


def _manager_holdings_frame(cik: str, filing_date, period, infotable: pd.DataFrame
                            ) -> pd.DataFrame:
    """One filing's info table -> one row per CUSIP, with NO universe filter. Pure.

    `issuer_name` / `title_of_class` are carried because they are the only human-readable handle
    on a CUSIP this pipeline never resolves to a ticker; they are taken as the first non-null of
    the group, since a manager's split sub-account lines for one security repeat them."""
    typed = _classify_holdings(infotable)
    typed["issuer_name"] = _pick(infotable, "NAMEOFISSUER", "Issuer").astype("string")
    typed["title_of_class"] = _pick(infotable, "TITLEOFCLASS", "Class").astype("string")
    typed = typed.dropna(subset=["cusip"])
    if typed.empty:
        return pd.DataFrame(columns=_COLS)

    numeric = [c for c in typed.columns if c not in ("cusip", "issuer_name", "title_of_class")]
    out = typed.groupby("cusip", as_index=False).agg(
        {**{c: "sum" for c in numeric},
         "issuer_name": "first", "title_of_class": "first"})
    out["position_type"] = _dominant_type(out)
    out["cik"] = pad_cik(cik)          # the stored form; the PK join depends on matching it
    out["period"] = pd.Timestamp(period)
    out["filing_date"] = pd.Timestamp(filing_date)
    return out.dropna(subset=["period"])[_COLS]


def _read_filing(filing) -> pd.DataFrame:
    """Fetch and parse one 13F-HR. Empty on any failure: one unparseable filing must not abort a
    manager's whole history."""
    try:
        infotable = filing.obj().infotable
        if infotable is None or infotable.empty:
            return pd.DataFrame()
        return _manager_holdings_frame(filing.cik, filing.filing_date,
                                       period_of_report(filing), infotable)
    except Exception as e:                                          # noqa: BLE001
        logger.warning(f"13F-manager {filing.accession_number}: {type(e).__name__}: {e}")
        return pd.DataFrame()


def _suspect_prices(df: pd.DataFrame) -> int:
    """Rows whose implied share price falls outside `_IMPLIED_PRICE_BAND` -- the detector for
    edgartools inferring the wrong $thousands-vs-$ones unit on a filing."""
    if df.empty:
        return 0
    implied = (df["value_usd"] / df["shares"].where(df["shares"] > 0)).dropna()
    return int((~implied.between(*_IMPLIED_PRICE_BAND)).sum())


def fetch_13f_managers(context: Context, years_history: int = 15) -> int:
    """Walk every roster CIK's 13F-HR history and upsert the complete books. Returns rows saved.

    `years_history` bounds the walk by PERIOD, not filing date: a manager back-filing a 2019
    period in 2025 belongs in 2019's book, and the period is what every consumer joins on."""
    context.ensure_edgar_identity()

    ciks = sorted(roster_cik_union(context))
    if not ciks:
        raise SuperinvestorRosterEmpty(
            "superinvestor_roster is empty -- run `data_extract superinvestors --seed` first. "
            "The roster is this walk's entire scope; there is nothing to fetch without it.")
    names = roster_map_as_of(context)            # latest snapshot; only used for log lines
    since = pd.Timestamp.today().normalize() - pd.DateOffset(years=years_history)
    logger.info("13F managers: %d roster CIK(s), periods from %s", len(ciks), since.date())

    # `ensure_table` is a check-then-create with no lock, so on a COLD table several workers can
    # each see it missing and race the CREATE; the losers raise and lose their manager's rows.
    # Serialize writes until the table is known to exist -- afterwards `save` is a plain
    # concurrent upsert. Same pattern as `edgar_driver.run_edgar_fetch`.
    create_lock = threading.Lock()
    created: set[str] = set()

    def _save(df: pd.DataFrame) -> int:
        if Tables.sec13f_manager_holdings.name in created:
            return context.store.save(Tables.sec13f_manager_holdings, df)
        with create_lock:
            n = context.store.save(Tables.sec13f_manager_holdings, df)
            created.add(Tables.sec13f_manager_holdings.name)
            return n

    def _worker(name: str, cik: str) -> tuple[str, int, int]:
        try:
            filings = Company(cik).get_filings(form=SEC_13F_FORMS) or []
        except Exception as e:                                      # noqa: BLE001 -- one manager
            context.log.warning("13F managers: %s (%s) listing failed (%s)", name, cik, e)
            return cik, 0, 0
        # oldest first, so an amendment filed later upserts OVER the original it restates
        dated = sorted(((pd.Timestamp(f.filing_date), f) for f in filings), key=lambda p: p[0])
        frames = []
        for _, filing in dated:
            try:
                period = pd.Timestamp(filing.period_of_report)
            except Exception:                                       # noqa: BLE001
                continue
            if pd.isna(period) or period < since:
                continue
            rows = _read_filing(filing)
            if not rows.empty:
                frames.append(rows)
        if not frames:
            return cik, 0, 0
        book = pd.concat(frames, ignore_index=True)
        # last filed wins per (cik, period, cusip) -- concat order is already oldest-first
        book = book.drop_duplicates(subset=["cik", "period", "cusip"], keep="last")
        suspect = _suspect_prices(book)
        try:
            saved = _save(book)
        except Exception as e:                                      # noqa: BLE001
            context.log.warning("13F managers: %s (%s) save failed (%s)", name, cik, e)
            return cik, 0, 0
        return cik, saved, suspect

    scope = pd.DataFrame({"ticker": [names.get(c, c) for c in ciks], "cik": ciks})
    results = run_per_ticker(scope, _worker, desc="13F manager books")
    saved = sum(n for _, n, _ in results)
    suspect = sum(s for _, _, s in results)

    # A manager that produced NOTHING is the failure this walk cannot otherwise report. The
    # per-entity handler catches its own exceptions and returns zero rows, so an SEC 429 --
    # which arrives as a logged warning and an empty listing, not an error -- leaves the run
    # reporting success while whole books are missing. Measured 2026-09-08: 18 of 106 roster
    # CIKs came back empty that way, every one of them with real history in `sec13f_hr`.
    # Cross-checking against `sec13f_hr` is what separates "never filed a 13F" (expected: the
    # roster carries advisers that never did) from "we were throttled" (must be re-run).
    empty = [cik for cik, n, _ in results if n == 0]
    if empty:
        known_filers = set(context.store.distinct(
            Tables.sec13f_hr, "cik", where={"cik": [c.lstrip("0") for c in empty] + empty}))
        known = {c for c in empty if c in known_filers or c.lstrip("0") in known_filers}
        context.log.warning(
            "13F managers: %d/%d roster CIK(s) produced NO rows; %d of them have history in "
            "%s and must be re-run (throttling or a transient listing failure, not an absent "
            "filer): %s", len(empty), len(ciks), len(known), Tables.sec13f_hr,
            ", ".join(sorted(known)) or "none")

    if suspect:
        logger.warning("13F managers: %d/%d saved rows imply a share price outside %s -- check "
                       "edgartools' per-filing $thousands detection before trusting value_usd",
                       suspect, saved, _IMPLIED_PRICE_BAND)
    logger.info("13F managers: saved %d row(s) across %d manager(s) -> %s",
                saved, len(ciks), Tables.sec13f_manager_holdings)
    record_run(context, Tables.sec13f_manager_holdings, len(ciks), saved)
    return saved
