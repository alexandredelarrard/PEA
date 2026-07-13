"""
Fetch recent news headlines per ticker via yfinance.

IMPORTANT: this only gives a handful of RECENT headlines per ticker —
there is no free source for 10 years of historical news at this scale.
For real historical news research, look at the GDELT Project
(https://www.gdeltproject.org), which is free but needs BigQuery or raw
file processing — out of scope for this simple script, but worth building
out later if news history matters a lot to your strategy.

Run:
    python -m data.fetch_news
"""
import time
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from src.context import Context


def fetch_news(context: Context, tickers: list[str], pause: float = 0.3) -> pd.DataFrame:
    rows = []
    for tkr in tqdm(tickers, desc="Fetching news"):
        try:
            items = yf.Ticker(tkr).news or []
        except Exception as e:
            print(f"{tkr}: failed ({e})")
            continue

        for item in items:
            content = item.get("content", item)  # yfinance schema has shifted over versions
            rows.append({
                "ticker": tkr,
                "title": content.get("title"),
                "publisher": (content.get("provider") or {}).get("displayName")
                             if isinstance(content.get("provider"), dict) else content.get("publisher"),
                "link": (content.get("canonicalUrl") or {}).get("url") if isinstance(content.get("canonicalUrl"), dict) else content.get("link"),
                "published": content.get("pubDate") or content.get("providerPublishTime"),
            })
        time.sleep(pause)

    df = pd.DataFrame(rows)
    df.to_parquet(context.paths["NEWS_LATEST_PATH"], index=False)
    print(f"Saved {len(df)} news rows to {context.paths["NEWS_LATEST_PATH"]}")
    return df

