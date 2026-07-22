"""
fetch_hf_transcripts.py  (src/data_extract/utils/behavioral/fetch_hf_transcripts.py)
------------------------------------------------------------------------------------
DEEP-history backbone for earnings-call transcripts: the FREE, MIT-licensed HuggingFace
dataset `kurry/sp500_earnings_transcripts` — full verbatim S&P 500 earnings calls 2005-2025
(33k+ transcripts / 685 companies), with a `content` (full text) and `structured_content`
(speaker-segmented) field per call. Motley Fool's global reverse-chron index can't reach
15 years (it would take ~5000 pages and gets throttled), so this dataset is the history
backbone and the MF crawl only fills the recent gap past the dataset's ~2025 cut.

One-time ~1.8 GB parquet download (cached under the call_transcripts dir), read in batches,
parsed into the SAME `earnings_call_sections` rows the MF path produces (ticker, quarter,
as_of, tag, text, url) so downstream FinBERT / text-metric features are source-agnostic.
Incremental: skips (ticker, quarter) already present in the table (from any source).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from src.constants.constants import (
    EARNINGS_CALL_SECTIONS_TABLE,
    HF_TRANSCRIPTS_CACHE,
    HF_TRANSCRIPTS_DATASET,
    HF_TRANSCRIPTS_PARQUET_URL,
)
from src.context import Context
from src.data_extract.utils.behavioral.fetch_earnings_calls import _cache_dir, split_prepared_qa

logger = logging.getLogger(__name__)
_TABLE = EARNINGS_CALL_SECTIONS_TABLE
_ROLE_PREFIX = re.compile(r"^[A-Za-z]\s*-\s*")            # "A - Jane Doe" / "E - John Roe" role tags


# --------------------------------------------------------------------------- #
# One-time parquet download (streamed; curl_cffi fallback for the corporate proxy CA)
# --------------------------------------------------------------------------- #
def _stream_download(url: str, dest: Path) -> None:
    try:
        import requests
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with dest.open("wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        return
    except Exception as e:                              # noqa: BLE001
        logger.warning("HF parquet via requests failed (%s); retrying with curl_cffi "
                       "(unverified TLS — the corporate proxy rejects the HF CA)", e)
    from curl_cffi import requests as cr
    with cr.get(url, stream=True, timeout=120, impersonate="chrome", verify=False) as r:
        with dest.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def download_hf_parquet(context: Context, force: bool = False) -> Path:
    """Cache the dataset parquet under data/call_transcripts/. Skips if already present."""
    dest = _cache_dir(context) / HF_TRANSCRIPTS_CACHE
    if dest.exists() and not force and dest.stat().st_size > 1_000_000:
        logger.info("HF transcripts parquet already cached (%.0f MB) -> %s",
                    dest.stat().st_size / 1e6, dest)
        return dest
    tmp = dest.with_suffix(".part")
    logger.warning("Downloading HF transcripts parquet (~1.8 GB, one time) from %s ...",
                   HF_TRANSCRIPTS_PARQUET_URL)
    _stream_download(HF_TRANSCRIPTS_PARQUET_URL, tmp)
    tmp.replace(dest)
    logger.warning("Cached HF transcripts parquet (%.0f MB) -> %s", dest.stat().st_size / 1e6, dest)
    return dest


# --------------------------------------------------------------------------- #
# Row -> sections (PURE; unit-tested without any download)                      #
# --------------------------------------------------------------------------- #
def _clean_speaker(name: str) -> str:
    return _ROLE_PREFIX.sub("", str(name or "")).strip()


def _participants_text(structured) -> str:
    """Distinct non-operator speakers (management + analysts), role-prefix stripped,
    order-preserved — the reliable participants list `structured_content` gives us."""
    seen: list[str] = []
    for turn in (structured if isinstance(structured, (list, tuple)) else []):
        s = _clean_speaker(turn.get("speaker", "") if isinstance(turn, dict) else "")
        if s and s.lower() != "operator" and s not in seen:
            seen.append(s)
    return "\n".join(seen)


def _text_from_structured(structured) -> str:
    return "\n".join(f"{_clean_speaker(t.get('speaker',''))}: {t.get('text','')}".strip()
                     for t in structured if isinstance(t, dict))


def row_sections(content: str | None, structured) -> dict[str, str]:
    """One dataset row -> {full, prepared_remarks?, qa?, participants?}. Uses the verbatim
    `content` (falls back to re-joining `structured_content`), the shared `split_prepared_qa`
    for prepared/qa, and `structured_content` for the participants list."""
    text = content or ""
    if len(text) < 200 and structured is not None:
        text = _text_from_structured(structured)
    if len(text) < 200:
        return {}
    out = split_prepared_qa(text)
    participants = _participants_text(structured)
    if participants:
        out["participants"] = participants
    return out


# --------------------------------------------------------------------------- #
# Ingest: batched parquet read -> earnings_call_sections (incremental)          #
# --------------------------------------------------------------------------- #
def _existing_keys(context: Context) -> set[tuple[str, str]]:
    """(ticker, quarter) already in the table — so a re-run skips them (from ANY source)."""
    try:
        df = context.store.load(_TABLE, columns=["ticker", "quarter"])
        if df is None or df.empty:
            return set()
        return set(map(tuple, df[["ticker", "quarter"]].drop_duplicates().to_numpy()))
    except Exception:                                    # table not created yet
        return set()


def ingest_hf_transcripts(context: Context, tickers: list[str] | None = None,
                          batch_size: int = 400, flush_rows: int = 8000) -> int:
    """Download (once) + parse the HF backbone into `earnings_call_sections`. Reads the
    parquet in batches (bounded memory), keeps only universe tickers, skips (ticker,quarter)
    already ingested, and upserts full/prepared_remarks/qa/participants per call. Returns
    rows upserted. `tickers` restricts to a subset (None = the full universe)."""
    import pyarrow.parquet as pq

    path = download_hf_parquet(context)
    universe = set(context.store.load("sp500_tickers", columns=["ticker"])["ticker"])
    keep = (universe & set(tickers)) if tickers is not None else universe
    existing = _existing_keys(context)
    url = f"hf://{HF_TRANSCRIPTS_DATASET}"

    pf = pq.ParquetFile(path)
    cols = ["symbol", "quarter", "year", "date", "content", "structured_content"]
    total, seen_calls, buf = 0, 0, []
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        for r in batch.to_pylist():
            tkr = str(r.get("symbol") or "").upper()
            if not tkr or tkr not in keep:
                continue
            try:
                quarter = f"{int(r['year'])}Q{int(r['quarter'])}"
            except (TypeError, ValueError):
                continue
            if (tkr, quarter) in existing:
                continue
            secs = row_sections(r.get("content"), r.get("structured_content"))
            if not secs:
                continue
            seen_calls += 1
            existing.add((tkr, quarter))                 # de-dup within this run too
            as_of = str(r.get("date"))[:10] if r.get("date") else None
            for tag, text in secs.items():
                if len(text) < 40:
                    continue
                buf.append({"ticker": tkr, "quarter": quarter, "tag": tag,
                            "as_of": as_of, "url": url, "text": text})
        if len(buf) >= flush_rows:
            total += context.store.save(_TABLE, pd.DataFrame(buf))
            buf = []
    if buf:
        total += context.store.save(_TABLE, pd.DataFrame(buf))
    logger.warning("HF transcripts: ingested %d sections from %d new calls (%d tickers) -> '%s'",
                   total, seen_calls, len({k[0] for k in existing}), _TABLE)
    return total
