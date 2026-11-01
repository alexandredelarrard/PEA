"""Shared SEC EDGAR helpers: the universe CIK mapping.

Every SEC request goes through `sec_io` (rate limit and retry policy).
"""

import pandas as pd

from src.context import Context
from src.data_store.schema import Tables
from src.utils.string import pad_cik_series

#: The `sp500_tickers` projection every SEC fetcher resolves its universe through; test fixtures build from it.
CIK_MAPPING_COLS: tuple[str, ...] = ("ticker", "cik", "name", "sector", "industry_group", "sub_industry")


def load_cik_mapping(context: Context, tickers: list[str] | None = None) -> pd.DataFrame:
    """Ticker -> 10-digit CIK (+ name / GICS) from `sp500_tickers`, filtered server-side to `tickers` when given."""
    df = context.store.load(Tables.sp500_tickers, columns=list(CIK_MAPPING_COLS), where={"ticker": list(tickers)} if tickers is not None else None)
    assert df is not None

    df["cik"] = pad_cik_series(df["cik"])
    return df
