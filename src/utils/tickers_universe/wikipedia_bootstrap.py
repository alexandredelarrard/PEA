"""
wikipedia_bootstrap.py  (src/utils/tickers_universe/wikipedia_bootstrap.py)
---------------------------------------------------------------------------
Name, CIK and GICS for a ticker entering `tickers_universe` with no stored row, read from
Wikipedia's S&P 500 table. Wikipedia gives the GICS sub-industry but not the industry group,
so the GICS hierarchy of `vocabulary.py` maps one to the other (case/punctuation-insensitive;
an unknown sub-industry falls back to the sector).

Delete this module when Sharadar categories replace GICS.
"""

from __future__ import annotations

import io
import re
from typing import cast

import pandas as pd
import requests

from src.constants.constants import BROWSER_HEADERS
from src.utils.string import pad_cik_series, yahoo_symbol
from src.utils.tickers_universe.vocabulary import GICS_SUB_INDUSTRIES

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKIPEDIA_COLUMNS = {"Symbol": "ticker", "Security": "name", "GICS Sector": "sector", "GICS Sub-Industry": "sub_industry", "CIK": "cik"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


_SUB_TO_GROUP = {_norm(sub): grp for grp, subs in GICS_SUB_INDUSTRIES.items() for sub in subs}


def industry_group(sub_industry: str, sector: str | None = None) -> str:
    """GICS industry group of a sub-industry; the sector (or the raw sub-industry) when it is not in the table."""
    grp = _SUB_TO_GROUP.get(_norm(sub_industry))
    if grp is not None:
        return grp
    return sector if sector else str(sub_industry)


def parse_sp500_table(html: str) -> pd.DataFrame:
    """The first table of the Wikipedia page as `ticker, name, cik, sector, sub_industry, industry_group`,
    tickers in yfinance spelling and CIKs padded to 10 digits."""
    df = cast(pd.DataFrame, pd.read_html(io.StringIO(html))[0]).rename(columns=WIKIPEDIA_COLUMNS)
    df["ticker"] = df["ticker"].map(yahoo_symbol)
    df["cik"] = pad_cik_series(df["cik"])
    df["industry_group"] = [industry_group(sub, sector) for sub, sector in zip(df["sub_industry"], df["sector"], strict=True)]
    return cast(pd.DataFrame, df[["ticker", "name", "cik", "sector", "sub_industry", "industry_group"]])


def wikipedia_sp500() -> pd.DataFrame:
    """Fetch and parse Wikipedia's current S&P 500 constituents table."""
    response = requests.get(WIKIPEDIA_SP500_URL, headers=BROWSER_HEADERS, timeout=30)
    response.raise_for_status()
    return parse_sp500_table(response.text)
