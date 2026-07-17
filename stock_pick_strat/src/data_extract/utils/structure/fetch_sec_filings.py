"""
SEC filings — 10 years of history, free via SEC EDGAR, no API key (just
the required User-Agent, see config.SEC_USER_AGENT / sec_utils.py).

IMPORTANT — read before running full-text download on the whole universe:
This script has TWO stages, deliberately separated because of the volume
involved:

  1. `build_filing_index()` — fast, cheap. For every S&P 500 ticker, pulls
     the list of filings (form type, date, accession number, document URL)
     from SEC's submissions API. This alone is genuinely useful (you can
     see filing cadence, 8-K frequency/timing around events, etc.) and is
     ~500 lightweight JSON requests.

  2. `download_filing_text()` — pulls the actual full text of a SPECIFIC
     filing, on demand. Downloading full text for EVERY 10-K/10-Q/8-K for
     500 companies over 10 years is thousands of documents and multiple
     GB — running that for the whole universe will take hours even at
     SEC's allowed rate limit, and most of it you'll never read. This
     script does NOT do that automatically. Use `download_filings_bulk()`
     with an explicit filter (e.g. only 10-Ks, only certain tickers, only
     last N filings) when you actually need full text for something like
     NLP/sentiment analysis.

Run (index only, recommended default):
    python -m data.fetch_sec_filings

Run with bulk text download for a filtered subset — edit `main()` below or
call `download_filings_bulk()` directly from a notebook/script.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data_extract.utils.common.sec_utils import sec_get, load_cik_mapping
from src.context import Context

FORM_TYPES = ["10-K", "10-Q", "8-K"]


def _today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _index_meta_path(context: Context) -> Path:
    return context.paths["SEC_FILINGS_INDEX_PATH"].with_name("sec_filings_index_meta.json")


def _index_is_up_to_date(context: Context, cik_mapping: pd.DataFrame) -> bool:
    """True when the filing index was already rebuilt today for the full universe.
    Filings are event-driven, so a once-per-day re-index is enough."""
    path = _index_meta_path(context)
    if not path.exists() or not context.paths["SEC_FILINGS_INDEX_PATH"].exists():
        return False
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        meta.get("last_built") == _today_iso()
        and meta.get("universe_size", 0) >= len(cik_mapping)
    )


def build_filing_index(context: Context, cik_mapping: pd.DataFrame) -> pd.DataFrame:
    
    years = context.config.data_extract.years_history
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=years)
    rows = []

    for _, r in tqdm(cik_mapping.iterrows(), total=len(cik_mapping), desc="Indexing SEC filings"):
        cik, ticker = r["cik"], r["ticker"]
        try:
            resp = sec_get(f"https://data.sec.gov/submissions/CIK{cik}.json")
            data = resp.json()
        except Exception as e:
            print(f"{ticker} ({cik}): failed ({e})")
            continue

        recent = data.get("filings", {}).get("recent", {})
        n = len(recent.get("accessionNumber", []))
        for i in range(n):
            form = recent["form"][i]
            if form not in FORM_TYPES:
                continue
            filing_date = pd.Timestamp(recent["filingDate"][i])
            if filing_date < cutoff:
                continue

            accession = recent["accessionNumber"][i].replace("-", "")
            primary_doc = recent["primaryDocument"][i]
            doc_url = (f"https://www.sec.gov/Archives/edgar/data/"
                       f"{int(cik)}/{accession}/{primary_doc}")

            rows.append({
                "ticker": ticker, "cik": cik, "company_name": r.get("company_name"),
                "form": form, "filing_date": filing_date,
                "accession_number": recent["accessionNumber"][i],
                "primary_document": primary_doc, "doc_url": doc_url,
            })

        # Note: `data.get("filings", {}).get("files", [])` lists OLDER
        # filings paginated in separate JSON files for companies with long
        # histories — for a full 10y+ archive on very active filers you may
        # need to fetch those too. Omitted here for simplicity; most S&P
        # 500 10-K/10-Q/8-K history fits in the "recent" page.

    return pd.DataFrame(rows)


def download_filing_text(context: Context, doc_url: str) -> Path:
    """Download the full text of a single filing document to disk. Returns the local path."""
    context.paths["SEC_FILINGS_TEXT_DIR"].mkdir(parents=True, exist_ok=True)
    filename = doc_url.split("/")[-1]
    out_path = context.paths["SEC_FILINGS_TEXT_DIR"] / filename
    if out_path.exists():
        return out_path

    resp = sec_get(doc_url)
    out_path.write_bytes(resp.content)
    return out_path


def download_filings_bulk(index_df: pd.DataFrame, form_filter: list[str] | None = None,
                           ticker_filter: list[str] | None = None,
                           max_filings: int | None = None, pause: float = 0.2) -> list[Path]:
    """
    Explicit, filtered bulk text download — you must opt into this and
    should filter first. Example:

        idx = pd.read_parquet(context.paths["SEC_FILINGS_INDEX_PATH"])
        download_filings_bulk(idx, form_filter=["10-K"], ticker_filter=["AAPL", "MSFT"])
    """

    df = index_df.copy()
    if form_filter:
        df = df[df["form"].isin(form_filter)]
    if ticker_filter:
        df = df[df["ticker"].isin(ticker_filter)]
    if max_filings:
        df = df.head(max_filings)

    paths = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading filing text"):
        try:
            path = download_filing_text(row["doc_url"])
            paths.append(path)
        except Exception as e:
            print(f"{row['ticker']} {row['form']} {row['filing_date']}: failed ({e})")
        time.sleep(pause)

    return paths


def fetch_sec_filings(context: Context):
    cik_mapping = load_cik_mapping(context)

    if _index_is_up_to_date(context, cik_mapping):
        print(f"SEC filing index already built today for {_today_iso()} — skipping "
              f"{context.paths['SEC_FILINGS_INDEX_PATH']}")
        return

    index_df = build_filing_index(context, cik_mapping)
    index_df.to_parquet(context.paths["SEC_FILINGS_INDEX_PATH"], index=False)
    _index_meta_path(context).write_text(
        json.dumps({
            "last_built": _today_iso(),
            "row_count": len(index_df),
            "ticker_count": int(index_df["ticker"].nunique()),
            "universe_size": len(cik_mapping),
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Saved index of {len(index_df):,} filings "
          f"({index_df['ticker'].nunique()} tickers) to {context.paths["SEC_FILINGS_INDEX_PATH"]}")

    print("Full text NOT downloaded by default — see download_filings_bulk() "
          "in this file to pull text for a filtered subset.")
