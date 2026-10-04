"""SEC current tickers with exchanges (`company_tickers_exchange.json`) -> `sec_company_tickers`, one snapshot per run.

A current map with no dates: the security master reads it for sibling share classes and `exchange` only.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.run_manifest import record_run
from src.data_extract.utils.common.sec_utils import sec_get
from src.data_store.schema import Tables
from src.utils.string import normalise_ticker, pad_cik_series

logger = logging.getLogger(__name__)

SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
COLUMNS = ("cik", "ticker", "name", "exchange", "fetched_at")


def parse_company_tickers_exchange(payload: Mapping[str, Any], *, fetched_at: pd.Timestamp) -> pd.DataFrame:
    """The `{"fields": [...], "data": [[...]]}` payload as one row per (cik, ticker), CIK padded, sorted."""
    frame = pd.DataFrame(payload.get("data") or [], columns=list(payload.get("fields") or ["cik", "name", "ticker", "exchange"]))
    out = pd.DataFrame(
        {
            "cik": pad_cik_series(frame["cik"]),
            "ticker": frame["ticker"].astype(str).map(normalise_ticker),
            "name": frame["name"].astype(str),
            "exchange": frame["exchange"].astype(object).where(frame["exchange"].notna(), None),
        }
    )
    out = out[out["ticker"].ne("")].drop_duplicates(["cik", "ticker"]).sort_values(["cik", "ticker"], ignore_index=True)
    out["fetched_at"] = fetched_at
    return out[list(COLUMNS)]


def download_sec_company_tickers(context: Context) -> int:
    """One GET of the SEC current-tickers file; replaces the snapshot table. Returns the rows written."""
    payload = sec_get(context, SEC_COMPANY_TICKERS_URL).json()
    frame = parse_company_tickers_exchange(payload, fetched_at=pd.Timestamp.now().floor("s"))
    written = context.store.replace(Tables.sec_company_tickers, frame)
    record_run(context, Tables.sec_company_tickers, 0, written, is_full_rescan=True)
    logger.info("sec_company_tickers: %d (cik, ticker) row(s) over %d CIK(s)", written, frame["cik"].nunique())
    return written
