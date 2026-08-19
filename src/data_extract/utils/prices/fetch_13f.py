"""
fetch_13f.py (src/data_extract/utils/prices/fetch_13f.py)
---------------------------------------------------------
Institutional holdings from SEC Form 13F-HR, discovered by FILING DATE through
edgartools. Replaces the quarterly bulk TSV data sets, which SEC only publishes
weeks after a quarter closes.

Grain: one row per (manager cik, period, ticker, cusip), the holding split into
long stock / calls / puts / debt principal / residual. Tickers are reconciled via
CUSIP (OpenFIGI), never via unstandardized issuer names.
"""

from __future__ import annotations

import logging

import pandas as pd
from edgar import get_filings
from tqdm import tqdm

from src.constants.constants import SEC_13F_FORMS
from src.context import Context
from src.data_extract.utils.common.run_manifest import record_run
from src.data_extract.utils.common.sec_utils import load_cik_mapping
from src.data_extract.utils.fundamentals.fetch_fundamentals_edgar import _configure_identity
from src.data_extract.utils.prices.fetch_cusip_map import build_cusip_ticker_map, normalize_cusip
from src.data_store.schema import Tables
from src.utils.string import pad_cik

logger = logging.getLogger(__name__)

# `quarter` is absent on purpose: it tagged the SOURCE bulk data set, not the period.
_COLS = ["cik", "period", "filing_date", "ticker", "cusip", "shares", "value_usd",
         "call_shares", "call_value", "put_shares", "put_value",
         "debt_prn", "debt_value", "other_value"]

# 13F `VALUE` is in $thousands or $ones depending on the schema the FILER used -- not on the
# period, so the old pre/post-2023 date rule was wrong. edgartools infers the unit per filing
# and hands `infotable` back in dollars; an implied price outside this band means it inferred
# wrong, the one failure mode that would silently scale value_usd by 1000x.
_IMPLIED_PRICE_BAND = (1.0, 5000.0)


def _pick(df: pd.DataFrame, *candidates: str) -> pd.Series:
    """Return the first present column (case-insensitive) among candidates."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return df[lower[cand.lower()]]
    return pd.Series([pd.NA] * len(df), index=df.index)


def _classify_holdings(infotable: pd.DataFrame) -> pd.DataFrame:
    """One typed row per holding line: value/shares land in exactly one bucket keyed on
    put/call and the amount type. A blank type with no put/call is long stock (the
    overwhelmingly common case, and what older data sets omit). The amount type is spelled
    SH/PRN in the bulk TSVs and Shares/Principal by edgartools; both are accepted."""
    putcall = _pick(infotable, "PUTCALL").astype("string").str.strip().str.upper().fillna("")
    amttype = (_pick(infotable, "SSHPRNAMTTYPE", "Type")
               .astype("string").str.strip().str.upper().fillna(""))
    amt = pd.to_numeric(_pick(infotable, "SSHPRNAMT", "SharesPrnAmount"),
                        errors="coerce").fillna(0.0)
    val = pd.to_numeric(_pick(infotable, "VALUE"), errors="coerce").fillna(0.0)

    is_call = putcall == "CALL"
    is_put = putcall == "PUT"
    opt = is_call | is_put
    is_debt = (~opt) & amttype.isin(["PRN", "PRINCIPAL"])
    is_stock = (~opt) & amttype.isin(["SH", "SHARES", ""])
    is_other = ~(opt | is_debt | is_stock)

    return pd.DataFrame({
        # canonical 9-char CUSIP so the map lookup and the ticker merge use ONE form
        "cusip": _pick(infotable, "CUSIP").map(normalize_cusip),
        "shares": amt.where(is_stock, 0.0),      "value_usd": val.where(is_stock, 0.0),
        "call_shares": amt.where(is_call, 0.0),  "call_value": val.where(is_call, 0.0),
        "put_shares": amt.where(is_put, 0.0),    "put_value": val.where(is_put, 0.0),
        "debt_prn": amt.where(is_debt, 0.0),     "debt_value": val.where(is_debt, 0.0),
        "other_value": val.where(is_other, 0.0),
    })


def _holdings_frame(cik, filing_date, period, infotable: pd.DataFrame) -> pd.DataFrame:
    """One filing's info table -> one row per security. Pure."""
    typed = _classify_holdings(infotable).dropna(subset=["cusip"])
    # a manager's several lines for the same security (split sub-accounts) become one row
    out = typed.groupby("cusip", as_index=False).sum(numeric_only=True)
    out["cik"] = pad_cik(cik)      # the stored form; the PK join depends on matching it
    out["period"] = pd.Timestamp(period)
    out["filing_date"] = pd.Timestamp(filing_date)
    return out.dropna(subset=["period"])


def _read_filing(filing) -> pd.DataFrame:
    """Fetch and parse one 13F-HR. Empty on any failure: one unparseable filing must not
    abort a batch of thousands. `lookback_days` is what gets it retried on a later run."""
    try:
        infotable = filing.obj().infotable
        if infotable is None or infotable.empty:
            return pd.DataFrame()
        return _holdings_frame(filing.cik, filing.filing_date,
                               filing.period_of_report, infotable)
    except Exception as e:                                        # noqa: BLE001
        logger.warning(f"13F {filing.accession_number}: {type(e).__name__}: {e}")
        return pd.DataFrame()


def _resolve_tickers(holdings: pd.DataFrame, cmap: pd.DataFrame,
                     universe: set[str]) -> pd.DataFrame:
    out = holdings.merge(cmap, on="cusip", how="inner")
    return out[out["ticker"].isin(universe)]


def fetch_13f(context: Context, tickers: list[str] | None = None, years_history: int = 15,
              save_every: int = 600, lookback_days: int = 7) -> None:
    """Ingest every 13F-HR filed since `sec13f_hr`'s latest `filing_date`, minus
    `lookback_days`.

    The watermark is the FILING date, not the reported period: managers back-file old periods
    (seen: a 2019-06-30 period filed in 2025), which period-keyed dedup would skip for ever.
    `lookback_days` re-reads the tail of the window because there is no accession dedup and
    `_read_filing` swallows per-filing failures -- without it, a filing that failed
    transiently would sit behind the advanced watermark and never be retried. A full rescan
    is the wrong self-heal here: 15y is ~528k filings, ~16h."""
    _configure_identity()
    today = pd.Timestamp.today().normalize()

    watermark = context.store.max_date(Tables.sec13f_hr, "filing_date")
    if watermark is None:
        since = today - pd.DateOffset(years=years_history)
        logger.warning(f"{Tables.sec13f_hr} has no stored filing_date -- "
                       f"full history from {since:%Y-%m-%d}")
    else:
        since = watermark - pd.Timedelta(days=lookback_days)

    filings = get_filings(form=SEC_13F_FORMS,
                          filing_date=f"{since:%Y-%m-%d}:{today:%Y-%m-%d}") or []
    total = len(filings)
    logger.info(f"13F: {total} filing(s) to read since {since:%Y-%m-%d}")
    universe = set(tickers) if tickers is not None else set(load_cik_mapping(context)["ticker"])
    if not total:
        record_run(context, Tables.sec13f_hr, len(universe), 0)
        return

    saved, suspect, batch, looked_up, cmap = 0, 0, [], set(), None
    for i, filing in enumerate(tqdm(filings, total=total, desc="13F-HR"), start=1):
        rows = _read_filing(filing)
        if not rows.empty:
            batch.append(rows)
        if batch and (len(batch) >= save_every or i == total):
            holdings = pd.concat(batch, ignore_index=True)
            batch.clear()
            new_cusips = set(holdings["cusip"]) - looked_up
            if new_cusips or cmap is None:
                # build_cusip_ticker_map re-reads the WHOLE map table, so only call it for a
                # genuinely new cusip -- once per batch would reload it ~44x a quarter
                cmap = build_cusip_ticker_map(context, sorted(new_cusips))
                looked_up |= new_cusips
            out = _resolve_tickers(holdings, cmap, universe)
            if not out.empty:
                implied = (out["value_usd"] / out["shares"].where(out["shares"] > 0)).dropna()
                suspect += int((~implied.between(*_IMPLIED_PRICE_BAND)).sum())
                saved += context.store.save(Tables.sec13f_hr, out[_COLS])

    if suspect:
        logger.warning(f"13F: {suspect}/{saved} saved rows imply a share price outside "
                       f"{_IMPLIED_PRICE_BAND} -- check edgartools' per-filing $thousands "
                       f"detection before trusting value_usd")
    logger.info(f"13F: saved {saved} row(s) from {total} filing(s) to {Tables.sec13f_hr}")
    record_run(context, Tables.sec13f_hr, len(universe), saved)
