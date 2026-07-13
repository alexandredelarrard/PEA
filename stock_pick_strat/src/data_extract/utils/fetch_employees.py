"""
Historical employee counts from Financial Modeling Prep (FMP).

Unlike the yfinance `fullTimeEmployees` snapshot (current value only), FMP's
`historical-employee-count` endpoint returns the full chronological series a
company has reported in its 10-K / 10-Q filings -- typically back to ~1994 --
each row tagged with the period, the SEC form type and, crucially, the FILING
DATE (when the number became public). That filing date is our point-in-time
`as_of`, so the resulting features are genuinely historical and leak-free.

    symbol | periodOfReport | filingDate  | formType | employeeCount
    AAPL   | 2023-09-30     | 2023-11-03  | 10-K     | 161000
    ...

Auth, key rotation and incremental behaviour are all handled by the shared
`fmp_client` (multi-key `FMP_API_KEY*`, roll on rate-limit, `fetched_at`-based
incremental). One call returns the whole ticker history.

Run:
    python -m src.data_extract.fetch_employees
"""
from __future__ import annotations

import pandas as pd

from src.context import Context
from src.data_extract.utils.fmp_client import fetch_incremental

_DATA_COLUMNS = ["ticker", "as_of", "period", "employees", "form_type"]


def normalize_employees(records: list[dict], ticker: str) -> pd.DataFrame:
    """FMP JSON rows -> tidy (ticker, as_of, period, employees, form_type).

    `as_of` is the public filing date (fallbacks: acceptance time, then period).
    Rows without a usable date or a positive employee count are dropped.
    """
    if not records:
        return pd.DataFrame(columns=_DATA_COLUMNS)

    rows = []
    for r in records:
        emp = r.get("employeeCount", r.get("employee_count"))
        filed = (r.get("filingDate") or r.get("acceptanceTime")
                 or r.get("periodOfReport"))
        if emp is None or filed is None:
            continue
        rows.append({
            "ticker": ticker,
            "as_of": filed,
            "period": r.get("periodOfReport"),
            "employees": emp,
            "form_type": r.get("formType"),
        })
    if not rows:
        return pd.DataFrame(columns=_DATA_COLUMNS)

    df = pd.DataFrame(rows)
    df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce").dt.tz_localize(None).dt.normalize()
    df["period"] = pd.to_datetime(df["period"], errors="coerce").dt.tz_localize(None).dt.normalize()
    df["employees"] = pd.to_numeric(df["employees"], errors="coerce")
    df = df.dropna(subset=["as_of", "employees"])
    df = df[df["employees"] > 0]
    df = (df.sort_values(["ticker", "as_of"])
            .drop_duplicates(subset=["ticker", "as_of"], keep="last"))
    return df[_DATA_COLUMNS].reset_index(drop=True)


def fetch_employees(context: Context, tickers: list[str], pause: float = 0.3,
                    refetch_window_days: int = 30, session=None) -> pd.DataFrame:
    """Build/refresh the incremental historical-employee-count archive (rotating
    across all FMP_API_KEY* keys) and save it to EMPLOYEES_HISTORY_PATH."""
    return fetch_incremental(
        context, tickers,
        endpoint="historical-employee-count",
        normalize=normalize_employees,
        dedup_keys=["ticker", "as_of"],
        path_key="EMPLOYEES_HISTORY_PATH",
        refetch_window_days=refetch_window_days,
        pause=pause, session=session,
        desc="historical employee counts",
    )
