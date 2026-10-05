"""Shared SEC EDGAR helpers: the universe CIK mapping and the CIK -> ticker map.

Every SEC request goes through `sec_io` (rate limit and retry policy).
"""

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.registrant import load_registrants
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
