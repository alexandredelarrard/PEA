"""
fetch_roic_transcripts.py  (src/data_extract/utils/behavioral/fetch_roic_transcripts.py)
----------------------------------------------------------------------------------------
Roic AI earnings-call transcripts — the PRIMARY recent-gap source, sitting BETWEEN the HuggingFace
backbone (deep history, ~2005->2025Q1) and the Motley Fool crawl (last resort). Many recent quarters
simply do not exist on Motley Fool, but Roic's clean JSON API covers ~2 years of history on its FREE
tier — so we fill the post-2025 gap from Roic first and only fall back to fool for what Roic lacks.

Flow (per ticker, only the quarters STILL missing after HF + whatever's already stored):
  1. LIST  `…/earnings-calls/list/{ticker}`      -> the (year, quarter, date) Roic has (one request).
  2. keep the intersection of {missing quarters} and {Roic has}.
  3. TRANSCRIPT `…/earnings-calls/transcript/{ticker}?year=&quarter=` -> {..., content} per quarter,
     parsed into the SAME sections the HF/MF paths produce (via `split_prepared_qa`) and upserted to
     `earnings_call_sections`. Because Roic writes to the DB here, the later fool discovery sees those
     quarters as present and skips them.

Auth is the `apikey` QUERY param (ROIC_API_KEY / ROIC_AI_API_KEY). No key -> logs a warning and is a
no-op (the pipeline continues to the fool fallback). Free tier is 5 req/min, so requests are paced.
Incremental: `missing_quarters_by_ticker` already excludes anything stored, so a re-run only fetches
genuinely new quarters.
"""
from __future__ import annotations

import logging
import os

import pandas as pd
from tqdm import tqdm

from src.constants.constants import (
    EARNINGS_CALL_SECTIONS_TABLE,
    ROIC_EARNINGS_LIST_URL,
    ROIC_EARNINGS_TRANSCRIPT_URL,
    ROIC_REQUEST_PAUSE,
)
from src.context import Context
from src.utils import polite_http as ph
# reuse the ONE gap definition + the shared transcript-section splitter
from src.data_extract.utils.behavioral.utils_split_qa import (
    split_prepared_qa,
)
from src.data_extract.utils.behavioral.utils_missing_quarters import (
    missing_quarters_by_ticker,
    _parse_quarter,
)

logger = logging.getLogger(__name__)
_TABLE = EARNINGS_CALL_SECTIONS_TABLE


def _api_key() -> str | None:
    return os.getenv("ROIC_API_KEY", None)


def roic_list_quarters(ticker: str, apikey: str) -> dict[str, str]:
    """{quarter_label: call_date} Roic has for `ticker` (one LIST request). Empty on miss/error."""
    data = ph.get_json(ROIC_EARNINGS_LIST_URL.format(ticker=ticker),
                       params={"apikey": apikey}, impersonate=False)
    out: dict[str, str] = {}
    for row in (data or []):
        try:
            q = f"{int(row['year'])}Q{int(row['quarter'])}"
        except (TypeError, ValueError, KeyError):
            continue
        out[q] = str(row.get("date") or "")[:10] or None
    return out


def roic_transcript_sections(ticker: str, quarter: str, apikey: str) -> tuple[dict, str | None]:
    """Fetch ONE quarter's transcript and split into sections. Returns (sections, as_of_date).
    ({}, None) on miss / empty content."""
    pq = _parse_quarter(quarter)
    if pq is None:
        return {}, None
    year, q = pq
    data = ph.get_json(ROIC_EARNINGS_TRANSCRIPT_URL.format(ticker=ticker),
                       params={"apikey": apikey, "year": year, "quarter": q}, impersonate=False)
    if not data:
        return {}, None
    content = data.get("content") if isinstance(data, dict) else None
    if not content or len(content) < 200:
        return {}, None
    as_of = str(data.get("date") or "")[:10] or None
    return split_prepared_qa(content), as_of


def fetch_roic_transcripts(context: Context, tickers: list[str] | None = None,
                           since: str = "2025-01-01", pause: float = ROIC_REQUEST_PAUSE) -> int:
    """Fill each ticker's MISSING recent quarters from Roic AI and upsert to `earnings_call_sections`.
    Returns the number of section rows saved. No-op (returns 0) without a ROIC API key.

    Per ticker: LIST once (what Roic has), fetch only the missing quarters Roic actually covers,
    parse -> sections -> save immediately (so a throttle/interrupt loses no work and the later fool
    step sees them). `since` is the recent-gap floor for names the HF backbone doesn't cover."""
    apikey = _api_key()
    if not apikey:
        context.log.warning("Roic AI transcripts skipped: no API key (set ROIC_API_KEY_ENV in .env). "
                            "Earnings calls will fall back to Motley Fool only.", " / ")
        return 0

    missing = missing_quarters_by_ticker(context, tickers=tickers, since=since)
    if not missing:
        context.log.info("Roic AI: no missing recent quarters to fill (HF backbone + DB current).")
        return 0

    total_saved, tickers_touched, no_roic = 0, 0, []
    for ticker in tqdm(sorted(missing), desc="Roic AI transcripts"):
        need = set(missing[ticker])
        avail = roic_list_quarters(ticker, apikey)               # 1 request
        ph.sleep_pace(pause, ROIC_EARNINGS_LIST_URL)
        to_fetch = sorted(need & set(avail),
                          key=lambda q: (_parse_quarter(q) or (0, 0)))
        if not to_fetch:
            no_roic.append(ticker)
            continue

        rows: list[dict] = []
        url = ROIC_EARNINGS_TRANSCRIPT_URL.format(ticker=ticker)
        for q in to_fetch:
            sections, as_of = roic_transcript_sections(ticker, q, apikey)
            ph.sleep_pace(pause, ROIC_EARNINGS_TRANSCRIPT_URL)
            for tag, text in sections.items():
                if len(text) < 40:
                    continue
                rows.append({"ticker": ticker, "quarter": q, "tag": tag,
                             "as_of": as_of or avail.get(q), "url": url, "text": text})
        if rows:                                                 # persist per ticker (resume-safe)
            saved = context.store.save(_TABLE, pd.DataFrame(rows))
            total_saved += saved
            tickers_touched += 1
            logger.info("Roic AI %s: +%d sections across %d quarter(s) %s",
                        ticker, saved, len(to_fetch), to_fetch)

    context.log.info("Roic AI transcripts: +%d sections across %d tickers; %d ticker(s) had no Roic "
                     "coverage for their gap (-> fool fallback).", total_saved, tickers_touched, len(no_roic))
    return total_saved
