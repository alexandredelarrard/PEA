"""
Shared helpers for talking to SEC EDGAR (free, no API key, but requires a
descriptive User-Agent and respectful rate limiting — SEC's fair-access
policy asks for <=10 requests/second; we go much slower than that to be safe).

https://www.sec.gov/os/webmaster-faq#developers
"""
import os
import time

import pandas as pd
import requests

from src.context import Context

_MIN_INTERVAL = 0.15  # ~6-7 req/sec, under SEC's 10/sec limit
_last_request_time = [0.0]


def _sec_headers() -> dict[str, str]:
    user_agent = os.getenv("SEC_USER_AGENT", "").strip()
    if not user_agent:
        raise RuntimeError(
            "SEC_USER_AGENT is not set. SEC EDGAR blocks requests without a "
            "descriptive User-Agent (name + email). Add to your .env file, e.g.\n"
            '  SEC_USER_AGENT="Your Name your.email@example.com"\n'
            "See https://www.sec.gov/os/webmaster-faq#developers"
        )
    return {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    }


def sec_get(url: str, **kwargs) -> requests.Response:
    """Rate-limited GET with the required SEC User-Agent header."""
    elapsed = time.time() - _last_request_time[0]
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    resp = requests.get(url, headers=_sec_headers(), **kwargs)
    _last_request_time[0] = time.time()
    resp.raise_for_status()
    return resp


def build_cik_mapping(context: Context, sp500_tickers: list[str] | None = None) -> pd.DataFrame:
    """
    Fetch SEC's full ticker->CIK mapping and filter to our S&P 500 universe.
    Source: https://www.sec.gov/files/company_tickers.json (free, no key).
    """
    resp = sec_get("https://www.sec.gov/files/company_tickers.json")
    raw = resp.json()  # dict of {index: {cik_str, ticker, title}}
    df = pd.DataFrame(raw.values())
    df["cik_str"] = df["cik_str"].astype(str).str.zfill(10)
    df = df.rename(columns={"cik_str": "cik", "title": "company_name"})

    if sp500_tickers:
        df = df[df["ticker"].isin(sp500_tickers)]

    df.to_csv(context.paths["CIK_MAPPING_PATH"], index=False)
    print(f"Saved CIK mapping for {len(df)} tickers to {context.paths["CIK_MAPPING_PATH"]}")
    return df


def load_cik_mapping(context: Context) -> pd.DataFrame:
    if context.paths["CIK_MAPPING_PATH"].exists():
        return pd.read_csv(context.paths["CIK_MAPPING_PATH"], dtype={"cik": str})
    tickers = pd.read_csv(context.paths["TICKERS_PATH"])["ticker"].tolist()
    return build_cik_mapping(context, tickers)
