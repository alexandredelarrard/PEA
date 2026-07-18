"""
fetch_def14a_llm.py  (src/data_extract/utils/fetch_def14a_llm.py)
-----------------------------------------------------------------
Extract structured governance data from SEC DEF 14A proxy statements using an
LLM with structured output (Def14AExtract schema).

For each ticker, this fetches all DEF 14A filings from EDGAR (using existing
edgar_fillings helpers), sends targeted sections to the OpenAI Responses API
(which is constrained to the Def14AExtract Pydantic schema), and upserts the
results into the `def14a_llm` Postgres table — scalar summary columns plus a
raw JSON field for full downstream use.

Incremental: already-processed accession numbers are skipped so successive
runs only process new filings.

Requires OPENAI_API_KEY (or OPEN_AI_API_KEY) in the .env file.
If the key is absent the function logs a warning and returns whatever exists.

Output columns (DB table `def14a_llm`):
    ticker, as_of, period, accession_number, company_name, fiscal_year_extract,
    n_directors, avg_director_age, avg_board_tenure, pct_independent_directors,
    n_officers,
    ceo_name_proxy, ceo_salary, ceo_bonus, ceo_stock_awards,
    ceo_option_awards, ceo_total_comp,
    def14a_json  (full Def14AExtract as JSON for downstream use)
"""
from __future__ import annotations

import logging
import re

import pandas as pd
from tqdm import tqdm

from src.context import Context
from src.data_extract.utils.structure.def14a_schema import Def14AExtract
from src.data_extract.utils.common.edgar_extract import html_to_text
from src.data_extract.utils.common.edgar_fillings import list_filings
from src.data_extract.utils.common.llm_extractor import LLMExtractor
from src.data_extract.utils.common.sec_utils import (
    load_cik_mapping, load_extract_meta, save_extract_meta, sec_get, today_iso,
)

_FORM = ["DEF 14A"]

# flattened output columns that must be stored numeric (float) in the DB
_NUMERIC_COLS = [
    "fiscal_year_extract", "n_directors", "avg_director_age", "avg_board_tenure",
    "pct_independent_directors", "n_officers", "ceo_salary", "ceo_bonus",
    "ceo_stock_awards", "ceo_option_awards", "ceo_total_comp",
]

logger = logging.getLogger(__name__)

_SECTION_CHARS = 30_000   # chars extracted per section
_CONTEXT_PRE = 600        # chars of context before the anchor match

# Content-based patterns that reliably mark the START of each section's actual data.
# These match real content (biographical text, table rows, share tables) rather than
# section headings, which appear in the TOC and in cross-references throughout the doc.

# Director bios: "Director since YEAR" or "Age:" label in a bio block
_DIRECTOR_CONTENT_RE = re.compile(
    r"\bDirector\s+since\s+(?:19|20)\d{2}\b"
    r"|\bAge[:\s]+\d{2}\b"
    r"|\bhas\s+served\s+as.*?(?:Chief Executive|President|Chairman|Chief Financial)",
    re.I,
)
# SCT rows: a recent fiscal year (2020+) followed by two or more large numbers (salary cols)
_COMPENSATION_CONTENT_RE = re.compile(
    r"\b20(?:2[0-9])\b\s+[\d,]{4,}\s+[\d,]{4,}",
)
# Ownership table: large share counts (>10k) immediately followed by a percentage
_OWNERSHIP_CONTENT_RE = re.compile(
    r"\b\d[\d,]{4,}\s+\d+(?:\.\d+)?%",
)

# Fallback text anchors (for filings that don't have rich content patterns)
_DIRECTOR_ANCHORS = (
    "nominees for election",
    "election of directors",
    "director nominees",
    "our board of directors",
    "board of directors",
)
_COMPENSATION_ANCHORS = (
    "summary compensation table",
    "executive compensation",
    "named executive officers",
)
_OWNERSHIP_ANCHORS = (
    "security ownership of certain",
    "security ownership of management",
    "beneficial ownership",
)


def _find_content_section(
    text: str,
    content_re: re.Pattern,
    fallback_anchors: tuple[str, ...],
    context_pre: int = _CONTEXT_PRE,
    last_occurrence: bool = False,
) -> int:
    """Find the start of a section using a content-based regex.

    Searches for `content_re` and returns a position `context_pre` chars before
    the first (or last, if `last_occurrence=True`) match so the LLM sees full
    context. Falls back to text-anchor scanning if no content match is found.
    """
    if last_occurrence:
        matches = list(content_re.finditer(text))
        if matches:
            return max(0, matches[-1].start() - context_pre)
    else:
        m = content_re.search(text)
        if m:
            return max(0, m.start() - context_pre)

    # Fallback: text anchors, skipping early TOC hits (first 5% of document)
    low = text.lower()
    min_pos = max(5000, int(len(text) * 0.05))
    for anchor in fallback_anchors:
        start = min_pos
        while True:
            p = low.find(anchor, start)
            if p == -1:
                break
            after = text[p + len(anchor): p + len(anchor) + 200]
            letter_ratio = sum(c.isalpha() for c in after[:100]) / max(len(after[:100]), 1)
            if letter_ratio > 0.20:
                return max(0, p - context_pre)
            start = p + 1
    return -1


def prepare_def14a_sections(text: str) -> str:
    """Return a focused subset of the DEF 14A covering the three key sections:
    director nominees, executive compensation, and security ownership.

    Uses content-based pattern matching (bio text, SCT dollar rows, share tables)
    to find the actual body sections rather than section headings that appear in
    the TOC. Each section is capped at _SECTION_CHARS characters. Falls back to
    the first 100k chars when no patterns match (old-format / very short filings).
    """
    parts: list[str] = []

    for label, content_re, anchors, use_last in (
        ("DIRECTOR NOMINEES",   _DIRECTOR_CONTENT_RE,    _DIRECTOR_ANCHORS,    False),
        ("EXECUTIVE COMPENSATION", _COMPENSATION_CONTENT_RE, _COMPENSATION_ANCHORS, False),
        ("SECURITY OWNERSHIP",  _OWNERSHIP_CONTENT_RE,   _OWNERSHIP_ANCHORS,   True),
    ):
        pos = _find_content_section(text, content_re, anchors, last_occurrence=use_last)
        if pos == -1:
            continue
        end = min(len(text), pos + _SECTION_CHARS)
        parts.append(f"\n\n=== {label} ===\n{text[pos:end]}")

    return "".join(parts) if parts else text[:100_000]


def _ceo_from_compensation(extract: Def14AExtract) -> "ExecutiveCompensation | None":  # noqa: F821
    ceo_kws = ("chief executive", "ceo", "president and chief")
    for c in extract.compensation:
        if any(kw in c.title.lower() for kw in ceo_kws):
            return c
    return extract.compensation[0] if extract.compensation else None


def _flatten(ticker: str, filing: pd.Series, extract: Def14AExtract) -> dict:
    dirs = extract.directors
    ages = [d.age for d in dirs if d.age is not None]
    tenures = [d.tenure_years for d in dirs if d.tenure_years is not None]
    independent = [d for d in dirs if d.is_independent]
    ceo = _ceo_from_compensation(extract)

    return {
        "ticker": ticker,
        "as_of": filing["filing_date"],
        "period": pd.to_datetime(filing.get("period_of_report"), errors="coerce"),
        "accession_number": filing["accession_number"],
        "company_name": extract.company_name,
        "fiscal_year_extract": extract.fiscal_year,
        # Directors
        "n_directors": len(dirs),
        "avg_director_age": round(sum(ages) / len(ages), 1) if ages else None,
        "avg_board_tenure": round(sum(tenures) / len(tenures), 1) if tenures else None,
        "pct_independent_directors": (
            round(len(independent) / len(dirs), 3) if dirs else None
        ),
        # Officers
        "n_officers": len(extract.executive_officers),
        # CEO compensation
        "ceo_name_proxy": ceo.name if ceo else None,
        "ceo_salary": ceo.salary_usd if ceo else None,
        "ceo_bonus": ceo.bonus_usd if ceo else None,
        "ceo_stock_awards": ceo.stock_awards_usd if ceo else None,
        "ceo_option_awards": ceo.option_awards_usd if ceo else None,
        "ceo_total_comp": ceo.total_compensation_usd if ceo else None,
        # Full JSON for downstream access
        "def14a_json": extract.model_dump_json(),
    }


def _process_filing(
    ticker: str, filing: pd.Series, extractor: LLMExtractor
) -> dict | None:
    try:
        raw_html = sec_get(filing["doc_url"]).text
        text = html_to_text(raw_html)
        focused = prepare_def14a_sections(text)
        extract = extractor.extract(Def14AExtract, focused)
        return _flatten(ticker, filing, extract)
    except Exception as e:
        logger.warning("%s %s: DEF 14A LLM extraction failed (%s)",
                       ticker, filing.get("filing_date", ""), e)
        return None


def _is_up_to_date(context: Context, n_universe: int) -> bool:
    path = context.paths["DEF14A_LLM_PATH"]
    meta = load_extract_meta(path)
    if (meta is None or meta.get("last_built") != today_iso()
            or not context.store.exists("def14a_llm")):
        return False
    return meta.get("universe_size", 0) >= n_universe


def fetch_def14a_llm(
    context: Context,
    tickers: list[str],
    model: str = "gpt-4o-mini",
    max_chars: int = 100_000,
) -> pd.DataFrame:
    """Build/refresh the DEF 14A LLM governance extract.

    Incremental: skips filings already in the DB table (by accession_number).
    Skips gracefully when OPENAI_API_KEY is absent.
    """
    years = context.config.data_extract.years_history
    path = context.paths["DEF14A_LLM_PATH"]

    cik_map = load_cik_mapping(context)
    cik_map = cik_map[cik_map["ticker"].isin(tickers)]

    if _is_up_to_date(context, len(cik_map)):
        existing = context.store.load("def14a_llm")
        context.log.info("DEF 14A LLM already up to date — skipping (%d rows)", len(existing))
        return existing

    existing = context.store.load("def14a_llm")
    existing = None if existing.empty else existing
    seen: set[str] = set()
    if existing is not None and not existing.empty and "accession_number" in existing.columns:
        seen = set(existing["accession_number"].dropna())

    try:
        extractor = LLMExtractor(model=model, max_chars=max_chars)
    except EnvironmentError as e:
        context.log.warning("DEF 14A LLM extraction skipped: %s", e)
        return existing if existing is not None else pd.DataFrame(columns=["ticker", "as_of"])

    new_rows: list[dict] = []
    for _, r in tqdm(cik_map.iterrows(), total=len(cik_map), desc="DEF 14A LLM"):
        ticker, cik, company = r["ticker"], r["cik"], r.get("company_name", "")
        try:
            filings = list_filings(cik, _FORM, years, company)
        except Exception as e:
            context.log.warning("%s: DEF 14A filing list failed (%s)", ticker, e)
            continue

        for _, f in filings.iterrows():
            if f["accession_number"] in seen:
                continue
            row = _process_filing(ticker, f, extractor)
            if row is not None:
                new_rows.append(row)
                seen.add(f["accession_number"])

    new_df = pd.DataFrame(new_rows)
    # coerce numeric columns to float so the DB table is created with numeric
    # types even when a batch's optional fields (e.g. option awards) are all null
    for c in _NUMERIC_COLS:
        if c in new_df.columns:
            new_df[c] = pd.to_numeric(new_df[c], errors="coerce")
    parts = [d for d in (existing, new_df) if d is not None and not d.empty]

    if not parts:
        save_extract_meta(path, None, 0, len(cik_map))
        return existing if existing is not None else pd.DataFrame(columns=["ticker", "as_of"])

    out = pd.concat(parts, ignore_index=True)
    out["as_of"] = pd.to_datetime(out["as_of"]).dt.normalize()
    out = out.drop_duplicates(subset=["ticker", "accession_number"], keep="last")
    out = out.sort_values(["ticker", "as_of"]).reset_index(drop=True)
    if not new_df.empty:
        context.store.save("def14a_llm", new_df)

    last_fd = out["as_of"].max()
    save_extract_meta(
        path,
        last_fd.date().isoformat() if pd.notna(last_fd) else None,
        out["ticker"].nunique(),
        len(cik_map),
    )
    context.log.info(
        "DEF 14A LLM: %d rows, %d tickers (avg %.1f filings/ticker)",
        len(out), out["ticker"].nunique(),
        len(out) / max(out["ticker"].nunique(), 1),
    )
    return out
