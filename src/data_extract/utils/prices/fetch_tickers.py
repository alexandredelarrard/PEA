import io

import pandas as pd
import requests
import logging 

from src.data_store.schema import Tables
from src.constants.constants import _HEADERS
from src.data_extract.utils.common.gics import industry_group
from src.context import Context

logger = logging.getLogger(__name__)

def _dedupe_share_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Drop redundant dual-class listings (e.g. GOOG vs GOOGL, FOX vs FOXA, NWS vs
    NWSA): both share one CIK. Keep ONE row per CIK — the LONGEST symbol, which is
    the voting/Class-A line (GOOGL, FOXA, NWSA) rather than the non-voting Class-C
    (GOOG, FOX, NWS). Rows without a CIK are kept as-is."""
    if "cik" not in df.columns:
        return df
    has_cik = df[df["cik"].notna() & (df["cik"].astype(str).str.strip() != "")].copy()
    no_cik = df[~df.index.isin(has_cik.index)]
    has_cik["_len"] = has_cik["ticker"].str.len()
    kept = (has_cik.sort_values(["cik", "_len", "ticker"], ascending=[True, False, True])
            .drop_duplicates("cik", keep="first").drop(columns="_len"))
    dropped = sorted(set(has_cik["ticker"]) - set(kept["ticker"]))
    if dropped:
        logger.info(f"Deduplicated {len(dropped)} redundant share-class tickers: {dropped}")
    return pd.concat([kept, no_cik], ignore_index=True).sort_values("ticker").reset_index(drop=True)


def get_sp500_tickers(context: Context) -> list[str]:
    """Scrape current S&P 500 tickers + sector info from Wikipedia. Adds the GICS
    industry group (24-level, for sector-neutral construction) and deduplicates
    dual-class share listings."""

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url, headers=_HEADERS, timeout=30)
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text))

    df = tables[0]
    df = df.rename(columns={
        "Symbol": "ticker",
        "Security": "name",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_industry",
        "CIK": "cik",
    })
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)  # yfinance format, e.g. BRK.B -> BRK-B
    if "cik" in df.columns:
        df["cik"] = df["cik"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    
    # GICS industry group (24) from sub-industry, sector fallback -> sector neutrality
    tick_redundant = context.config.data_extract.redundant_ticks
    df["industry_group"] = [industry_group(s, sec)
                            for s, sec in zip(df["sub_industry"], df["sector"])]
    df = _dedupe_share_classes(df)

    keep = [c for c in ["ticker", "name", "sector", "industry_group", "sub_industry", "cik"]
            if c in df.columns]
    df = df.loc[~df['ticker'].isin(tick_redundant)].reset_index(drop=True)
    context.store.save(Tables.sp500_tickers, df[keep])
    logger.info(f"Saved {len(df)} tickers to DB table {Tables.sp500_tickers}")
    
    return df["ticker"].tolist()
