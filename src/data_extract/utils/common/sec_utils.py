"""Shared SEC EDGAR helpers: rate-limited GET, the universe CIK mapping and the CIK -> ticker map.

SEC's fair-access policy requires a descriptive User-Agent and <= 10 requests/second
(https://www.sec.gov/os/webmaster-faq#developers). The limiter is thread-safe but per process:
request starts are spaced by `_MIN_INTERVAL` across threads while transfers overlap, so run one
EDGAR walk at a time.
"""

import threading
import time

import pandas as pd
import requests

from src.context import Context
from src.data_extract.utils.common.registrant import load_registrants
from src.data_store.schema import Tables
from src.utils.string import pad_cik_series

_MIN_INTERVAL = 0.11  # ~9 req/sec, safely under SEC's 10/sec limit
_DEFAULT_TIMEOUT = 30  # seconds; avoid a hung socket stalling a worker
_rate_lock = threading.Lock()
_next_slot = [0.0]  # monotonic time of the next allowed request start


def _reserve_slot() -> None:
    """Reserve the next evenly-spaced request slot; the wait happens outside the lock."""
    with _rate_lock:
        start = max(time.monotonic(), _next_slot[0])
        _next_slot[0] = start + _MIN_INTERVAL
    delay = start - time.monotonic()
    if delay > 0:
        time.sleep(delay)


def sec_get(context: Context, url: str, **kwargs) -> requests.Response:
    """Rate-limited, thread-safe GET on `context.sec_session` (User-Agent pre-set); raises on HTTP error."""
    kwargs.setdefault("timeout", _DEFAULT_TIMEOUT)
    _reserve_slot()
    resp = context.sec_session.get(url, **kwargs)
    resp.raise_for_status()
    return resp


#: The `sp500_tickers` projection every SEC fetcher resolves its universe through; test fixtures build from it.
CIK_MAPPING_COLS: tuple[str, ...] = ("ticker", "cik", "name", "sector", "industry_group", "sub_industry")


def load_cik_mapping(context: Context, tickers: list[str] | None = None) -> pd.DataFrame:
    """Ticker -> 10-digit CIK (+ name / GICS) from `sp500_tickers`, filtered server-side to `tickers` when given."""
    df = context.store.load(Tables.sp500_tickers, columns=list(CIK_MAPPING_COLS), where={"ticker": list(tickers)} if tickers is not None else None)
    assert df is not None

    df["cik"] = pad_cik_series(df["cik"])
    return df


def cik_to_ticker(cikmap: pd.DataFrame, *, config_dir: str | None = None) -> dict[str, str]:
    """CIK -> upper-case ticker, including every register predecessor CIK of a ticker in `cikmap`.

    For CIK-keyed bulk data sets. A consolidating (SPLIT) table must additionally drop rows filed
    outside the matched segment (`registrant.drop_rows_outside_segment`); UNION tables need no
    date filter. The insider path resolves through `identity.entity_ticker` instead.
    """
    if cikmap.empty or "ticker" not in cikmap.columns:
        return {}
    out = {str(c): str(t).upper() for c, t in zip(cikmap["cik"], cikmap["ticker"], strict=False)}
    universe = set(out.values())
    for ticker, entry in load_registrants(config_dir).items():
        if ticker.upper() not in universe:
            continue  # only tickers this run walks
        for cik in entry.all_ciks():
            out.setdefault(cik, ticker.upper())
    return out
