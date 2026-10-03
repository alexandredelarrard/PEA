import io
import logging
from typing import cast

import pandas as pd
import requests

from src.constants.constants import BROWSER_HEADERS
from src.context import Context
from src.data_extract.utils.common.gics import industry_group
from src.data_store.schema import Tables

logger = logging.getLogger(__name__)


def _dedupe_share_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep ONE row per CIK across dual-class listings: the longest symbol, i.e. the voting line
    (GOOGL over GOOG). Rows without a CIK are kept as-is."""
    if "cik" not in df.columns:
        return df
    cik = cast(pd.Series, df["cik"])
    has_cik = cast(pd.DataFrame, df[cik.notna() & (cik.astype(str).str.strip() != "")].copy())
    no_cik = cast(pd.DataFrame, df[~df.index.isin(has_cik.index)])
    has_cik["_len"] = cast(pd.Series, has_cik["ticker"]).str.len()
    kept = (
        cast(
            pd.DataFrame,
            has_cik.sort_values(by=["cik", "_len", "ticker"], ascending=[True, False, True]),
        )
        .drop_duplicates(subset=["cik"], keep="first")
        .drop(columns="_len")
    )
    dropped = sorted(set(has_cik["ticker"]) - set(kept["ticker"]))
    if dropped:
        logger.info(f"Deduplicated {len(dropped)} redundant share-class tickers: {dropped}")
    return pd.concat([kept, no_cik], ignore_index=True).sort_values(by="ticker").reset_index(drop=True)


def get_sp500_tickers(context: Context) -> None:
    """Scrape current S&P 500 tickers + sector info from Wikipedia into `sp500_tickers`, adding the GICS
    industry group and deduplicating dual-class listings."""

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url, headers=BROWSER_HEADERS, timeout=30)
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text))

    df = cast(pd.DataFrame, tables[0])
    df = df.rename(
        columns={
            "Symbol": "ticker",
            "Security": "name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
            "CIK": "cik",
        }
    )
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)  # yfinance format, e.g. BRK.B -> BRK-B
    if "cik" in df.columns:
        df["cik"] = df["cik"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)

    # GICS industry group (24) from sub-industry, sector fallback -> sector neutrality
    df["industry_group"] = [industry_group(s, sec) for s, sec in zip(df["sub_industry"], df["sector"], strict=False)]
    df = _dedupe_share_classes(df)

    keep = [c for c in ["ticker", "name", "sector", "industry_group", "sub_industry", "cik"] if c in df.columns]
    context.store.save(Tables.sp500_tickers, cast(pd.DataFrame, df[keep]))
    logger.info(f"Saved {len(df)} tickers to DB table {Tables.sp500_tickers}")
