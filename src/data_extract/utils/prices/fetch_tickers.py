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


def _with_added_on(context: Context, df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """`df` plus `added_on`: the stored value for a stored ticker, `as_of` for a ticker not yet stored.

    A stored table without the column is returned unchanged, so the save never adds the column and
    no established ticker is stamped as new."""
    stored_cols = context.store.columns(Tables.sp500_tickers)
    if stored_cols and "added_on" not in stored_cols:
        logger.warning(f"{Tables.sp500_tickers} has no added_on column; new tickers are not stamped until it exists")
        return df
    df_stored = context.store.load(Tables.sp500_tickers, columns=["ticker", "added_on"], optional=True) if stored_cols else None
    if df_stored is None:
        df_stored = pd.DataFrame({"ticker": pd.Series(dtype=str), "added_on": pd.Series(dtype="datetime64[ns]")})
    df_stored = df_stored.assign(added_on=pd.to_datetime(df_stored["added_on"], errors="coerce"))
    df_tickers = df.merge(df_stored, on="ticker", how="left")
    is_new = ~df_tickers["ticker"].isin(df_stored["ticker"])
    df_tickers["added_on"] = df_tickers["added_on"].mask(is_new, as_of)
    if is_new.any():
        logger.info(f"{int(is_new.sum())} ticker(s) enter {Tables.sp500_tickers} on {as_of.date()}: {sorted(df_tickers.loc[is_new, 'ticker'])}")
    return df_tickers


def get_sp500_tickers(context: Context, as_of: pd.Timestamp | None = None) -> None:
    """Scrape current S&P 500 tickers + sector info from Wikipedia into `sp500_tickers`, adding the GICS
    industry group and deduplicating dual-class listings. A ticker not yet stored gets `added_on = as_of`
    (default today); a stored ticker keeps its stored `added_on`."""
    run_date = cast(pd.Timestamp, pd.Timestamp(as_of if as_of is not None else pd.Timestamp.today())).normalize()

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
    context.store.save(Tables.sp500_tickers, _with_added_on(context, cast(pd.DataFrame, df[keep]), run_date))
    logger.info(f"Saved {len(df)} tickers to DB table {Tables.sp500_tickers}")
