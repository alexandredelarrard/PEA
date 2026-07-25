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
from sqlalchemy import text

from src.constants.constants import (
    EARNINGS_CALL_SECTIONS_TABLE,
    HF_BACKBONE_EARLY_QUARTER,
    HF_BACKBONE_LATE_QUARTER,
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


def hf_latest_quarter_by_ticker(context: Context, tickers: list[str] | None = None,
                                batch_size: int = 4000) -> dict[str, tuple[int, int]]:
    """{ticker: (year, quarter)} for the LATEST call the HF backbone holds per ticker. Reads only
    the (symbol, year, quarter) columns of the cached parquet (cheap), so the Motley Fool gap
    fill knows, PER TICKER, the first quarter HF does NOT cover (everything after it must come
    from fool). Returns {} when the parquet is not cached yet (caller falls back to a date floor).
    `tickers` restricts to a subset (None = all)."""
    import pyarrow.parquet as pq

    dest = _cache_dir(context) / HF_TRANSCRIPTS_CACHE
    if not dest.exists() or dest.stat().st_size < 1_000_000:
        logger.info("HF parquet not cached yet -> no per-ticker HF horizon (fool gap uses the date floor)")
        return {}
    keep = {str(t).upper() for t in tickers} if tickers is not None else None
    latest: dict[str, tuple[int, int]] = {}
    pf = pq.ParquetFile(dest)
    for batch in pf.iter_batches(batch_size=batch_size, columns=["symbol", "year", "quarter"]):
        for r in batch.to_pylist():
            tkr = str(r.get("symbol") or "").upper()
            if not tkr or (keep is not None and tkr not in keep):
                continue
            try:
                yq = (int(r["year"]), int(r["quarter"]))
            except (TypeError, ValueError):
                continue
            cur = latest.get(tkr)
            if cur is None or yq > cur:
                latest[tkr] = yq
    logger.info("HF backbone latest-quarter horizon for %d tickers (e.g. %s)", len(latest),
                {k: f"{v[0]}Q{v[1]}" for k, v in list(latest.items())[:3]})
    return latest


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
def _hf_backbone_already_ingested(context: Context) -> tuple[bool, str | None, str | None]:
    """Is the HF backbone already in earnings_call_sections? True when the table's quarter coverage
    reaches back to HF_BACKBONE_EARLY_QUARTER and forward to HF_BACKBONE_LATE_QUARTER. Quarters are
    fixed-width 'YYYYQN', so a plain string MIN/MAX is chronological -> one cheap aggregate, no
    parquet load. Returns (present, min_quarter, max_quarter); False/None when the table is empty
    or unavailable (so a fresh DB still triggers the full ingest)."""
    try:
        with context.store.engine.connect() as c:
            row = c.execute(text(f'SELECT MIN(quarter), MAX(quarter) FROM "{_TABLE}"')).first()
    except Exception:                                    # table absent / DB unreachable
        return False, None, None
    if not row or row[0] is None:
        return False, None, None
    min_q, max_q = str(row[0]), str(row[1])
    present = (min_q <= HF_BACKBONE_EARLY_QUARTER and max_q >= HF_BACKBONE_LATE_QUARTER)
    return present, min_q, max_q


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
                          batch_size: int = 400, flush_rows: int = 8000,
                          force: bool = False) -> int:
    """Download (once) + parse the HF backbone into `earnings_call_sections`. Reads the
    parquet in batches (bounded memory), keeps only universe tickers, skips (ticker,quarter)
    already ingested, and upserts full/prepared_remarks/qa/participants per call. Returns
    rows upserted. `tickers` restricts to a subset (None = the full universe).

    Short-circuit: the HF backbone is a one-time historical load, so if the table ALREADY spans
    the backbone range (see `_hf_backbone_already_ingested`) we skip the whole 1.8GB parquet scan
    (which would otherwise re-read 33k calls only to find every one already present -> the "0 new
    calls" no-op that stalls the ingest step). Pass `force=True` to re-ingest regardless."""
    import pyarrow.parquet as pq

    if not force:
        present, min_q, max_q = _hf_backbone_already_ingested(context)
        if present:
            logger.warning("HF backbone already ingested — '%s' spans %s..%s (>= %s..%s); skipping "
                           "the 1.8GB parquet scan (pass force=True to re-ingest).", _TABLE,
                           min_q, max_q, HF_BACKBONE_EARLY_QUARTER, HF_BACKBONE_LATE_QUARTER)
            return 0

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
