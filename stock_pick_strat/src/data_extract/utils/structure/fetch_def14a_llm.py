"""
fetch_def14a_llm.py  (src/data_extract/utils/fetch_def14a_llm.py)
-----------------------------------------------------------------
Extract structured governance data from SEC DEF 14A proxy statements using an
LLM with structured output (Def14AExtract schema).

Per ticker, it fetches that ticker's DEF 14A filings from EDGAR, sends targeted
sections to the OpenAI Responses API (constrained to the Def14AExtract Pydantic
schema, temperature=0, prompt caching on), then **immediately upserts that
ticker's rows into the `def14a_llm` Postgres table** before moving to the next
ticker — so an interrupted run never loses the (expensive) LLM calls already made.

Year-incremental: the per-ticker cutoff is the latest `as_of` already stored in
the DB for that ticker, so only filings AFTER it are sent to the LLM. Tickers
with no rows yet are fetched over the full `years_history` window; tickers with
shorter histories simply have fewer filings — each resumes from its own latest
stored filing, never re-running the LLM on a year already saved.

Requires OPENAI_API_KEY (or OPEN_AI_API_KEY) in the .env file.
If the key is absent the function logs a warning and returns whatever exists.

Output columns (DB table `def14a_llm`), scalar summaries + raw JSON:
    keys        ticker, as_of, period, accession_number, company_name, fiscal_year_extract
    board       n_directors, board_size, avg_director_age, avg_board_tenure,
                pct_independent_directors, pct_female_directors,
                avg_other_public_boards, n_financial_experts, n_officers
    ceo         ceo_name_proxy, ceo_age, ceo_since_year, ceo_is_founder,
                ceo_is_board_chair, ceo_salary, ceo_bonus, ceo_stock_awards,
                ceo_option_awards, ceo_non_equity_incentive, ceo_all_other_comp,
                ceo_total_comp, ceo_equity_pay_pct
    neos        n_neos, total_neo_comp
    ownership   insider_ownership_pct, ceo_ownership_pct, n_five_percent_holders
    governance  independent_chair, lead_independent_director, classified_board,
                dual_class_shares, poison_pill, majority_voting, proxy_access,
                say_on_pay_frequency_years, say_on_pay_support_pct, ceo_pay_ratio,
                median_employee_pay, shareholder_proposals_count,
                auditor_name, auditor_fees
    def14a_json (full Def14AExtract as JSON for downstream use)
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
# (governance booleans are surfaced as 1.0/0.0 flags so they are usable features)
_NUMERIC_COLS = [
    "fiscal_year_extract",
    # board / directors
    "n_directors", "board_size", "avg_director_age", "avg_board_tenure",
    "pct_independent_directors", "pct_female_directors", "avg_other_public_boards",
    "n_financial_experts", "n_officers",
    # CEO
    "ceo_age", "ceo_since_year", "ceo_is_founder", "ceo_is_board_chair",
    "ceo_salary", "ceo_bonus", "ceo_stock_awards", "ceo_option_awards",
    "ceo_non_equity_incentive", "ceo_all_other_comp", "ceo_total_comp", "ceo_equity_pay_pct",
    # NEO aggregate
    "n_neos", "total_neo_comp",
    # ownership
    "insider_ownership_pct", "ceo_ownership_pct", "n_five_percent_holders",
    # governance provisions
    "independent_chair", "lead_independent_director", "classified_board",
    "dual_class_shares", "poison_pill", "majority_voting", "proxy_access",
    "say_on_pay_frequency_years", "say_on_pay_support_pct", "ceo_pay_ratio",
    "median_employee_pay", "shareholder_proposals_count", "auditor_fees",
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
# prose sections (no reliable numeric content pattern -> anchor-only)
_GOVERNANCE_ANCHORS = (
    "corporate governance",
    "board leadership structure",
    "governance highlights",
    "board composition",
    "director independence",
)
_PAYAUDIT_ANCHORS = (
    "ceo pay ratio",
    "pay ratio",
    "ratio of the annual total compensation",
    "median employee",
    "advisory vote to approve",   # say-on-pay proposal
    "say-on-pay",
    "independent registered public accounting firm",
    "audit and non-audit fees",
)

_AUX_SECTION_CHARS = 16_000       # smaller budget for the prose aux sections

def _find_content_section(
    text: str,
    content_re: re.Pattern | None,
    fallback_anchors: tuple[str, ...],
    context_pre: int = _CONTEXT_PRE,
    last_occurrence: bool = False,
) -> int:
    """Find the start of a section using a content-based regex.

    Searches for `content_re` and returns a position `context_pre` chars before
    the first (or last, if `last_occurrence=True`) match so the LLM sees full
    context. When `content_re` is None (prose sections), or no content match is
    found, falls back to text-anchor scanning (skipping the TOC).
    """
    if content_re is not None:
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
    """Return a focused subset of the DEF 14A covering the key sections:
    director nominees, executive compensation, security ownership, corporate
    governance, and CEO-pay-ratio / say-on-pay / auditor.

    The first three use content-based pattern matching (bio text, SCT dollar
    rows, share tables) to find the actual body rather than TOC headings; the
    last two are prose, so they are anchor-located (skipping the TOC). Each
    section is capped (main = _SECTION_CHARS, aux = _AUX_SECTION_CHARS). Falls
    back to the head of the document when no patterns match.
    """
    parts: list[str] = []

    for label, content_re, anchors, use_last, chars in (
        ("DIRECTOR NOMINEES",      _DIRECTOR_CONTENT_RE,     _DIRECTOR_ANCHORS,     False, _SECTION_CHARS),
        ("EXECUTIVE COMPENSATION", _COMPENSATION_CONTENT_RE, _COMPENSATION_ANCHORS, False, _SECTION_CHARS),
        ("SECURITY OWNERSHIP",     _OWNERSHIP_CONTENT_RE,    _OWNERSHIP_ANCHORS,    True,  _SECTION_CHARS),
        ("CORPORATE GOVERNANCE",   None,                     _GOVERNANCE_ANCHORS,   False, _AUX_SECTION_CHARS),
        ("PAY RATIO & AUDITOR",    None,                     _PAYAUDIT_ANCHORS,     False, _AUX_SECTION_CHARS),
    ):
        pos = _find_content_section(text, content_re, anchors, last_occurrence=use_last)
        if pos == -1:
            continue
        end = min(len(text), pos + chars)
        parts.append(f"\n\n=== {label} ===\n{text[pos:end]}")

    return "".join(parts) if parts else text[:120_000]


def _ceo_from_compensation(extract: Def14AExtract) -> "ExecutiveCompensation | None":  # noqa: F821
    """The CEO's Summary-Compensation-Table row: match on the extracted CEO name
    first, else on a CEO-like title, else the first (usually highest-paid) NEO."""
    ceo_name = (extract.ceo_name or "").strip().lower()
    if ceo_name:
        for c in extract.compensation:
            if (c.name or "").strip().lower() == ceo_name:
                return c
    ceo_kws = ("chief executive", "ceo", "president and chief")
    for c in extract.compensation:
        if any(kw in (c.title or "").lower() for kw in ceo_kws):
            return c
    return extract.compensation[0] if extract.compensation else None


def _ceo_age(extract: Def14AExtract) -> int | None:
    """CEO age: the explicit top-level field, else looked up by CEO name in the
    directors / officers lists (the CEO is almost always a director nominee)."""
    if extract.ceo_age is not None:
        return extract.ceo_age
    name = (extract.ceo_name or "").strip().lower()
    if name:
        for person in (*extract.directors, *extract.executive_officers):
            if (person.name or "").strip().lower() == name and person.age is not None:
                return person.age
    return None


def _bnum(x: bool | None) -> float | None:
    """Bool -> 1.0/0.0 (numeric flag for the feature store); None stays None."""
    return None if x is None else float(bool(x))


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 3) if xs else None


def _flatten(ticker: str, filing: pd.Series, extract: Def14AExtract) -> dict:
    dirs = extract.directors
    ages = [d.age for d in dirs if d.age is not None]
    tenures = [d.tenure_years for d in dirs if d.tenure_years is not None]
    genders = [(d.gender or "").strip().lower() for d in dirs if d.gender]
    other_boards = [d.other_public_company_boards for d in dirs
                    if d.other_public_company_boards is not None]
    ceo = _ceo_from_compensation(extract)
    gov = extract.governance
    g = lambda a: getattr(gov, a, None) if gov is not None else None  # noqa: E731
    own = extract.share_ownership
    ceo_name = (extract.ceo_name or (ceo.name if ceo else None) or "")

    ceo_equity = None
    if ceo and ceo.total_compensation_usd:
        equity = (ceo.stock_awards_usd or 0) + (ceo.option_awards_usd or 0)
        ceo_equity = round(equity / ceo.total_compensation_usd, 3)

    insider_pct = sum(o.percent_owned for o in own
                      if (o.is_director or o.is_officer) and o.percent_owned is not None) or None
    ceo_pct = next((o.percent_owned for o in own
                    if (o.name or "").strip().lower() == ceo_name.strip().lower()), None)

    return {
        "ticker": ticker,
        "as_of": filing["filing_date"],
        "period": pd.to_datetime(filing.get("period_of_report"), errors="coerce"),
        "accession_number": filing["accession_number"],
        "company_name": extract.company_name,
        "fiscal_year_extract": extract.fiscal_year,
        # ---- Board / directors ----
        "n_directors": len(dirs),
        "board_size": g("board_size") or (len(dirs) or None),
        "avg_director_age": _mean(ages),
        "avg_board_tenure": _mean(tenures),
        "pct_independent_directors": (
            round(sum(bool(d.is_independent) for d in dirs) / len(dirs), 3) if dirs else None),
        "pct_female_directors": (
            round(sum(x.startswith("f") for x in genders) / len(genders), 3) if genders else None),
        "avg_other_public_boards": _mean(other_boards),
        "n_financial_experts": sum(bool(d.audit_committee_financial_expert) for d in dirs) or None,
        # ---- Officers ----
        "n_officers": len(extract.executive_officers),
        # ---- CEO ----
        "ceo_name_proxy": ceo_name or None,
        "ceo_age": _ceo_age(extract),
        "ceo_since_year": extract.ceo_since_year,
        "ceo_is_founder": _bnum(extract.ceo_is_founder),
        "ceo_is_board_chair": _bnum(extract.ceo_is_board_chair
                                    if extract.ceo_is_board_chair is not None else g("ceo_is_board_chair")),
        "ceo_salary": ceo.salary_usd if ceo else None,
        "ceo_bonus": ceo.bonus_usd if ceo else None,
        "ceo_stock_awards": ceo.stock_awards_usd if ceo else None,
        "ceo_option_awards": ceo.option_awards_usd if ceo else None,
        "ceo_non_equity_incentive": ceo.non_equity_incentive_usd if ceo else None,
        "ceo_all_other_comp": ceo.all_other_comp_usd if ceo else None,
        "ceo_total_comp": ceo.total_compensation_usd if ceo else None,
        "ceo_equity_pay_pct": ceo_equity,
        # ---- NEO aggregate ----
        "n_neos": len(extract.compensation) or None,
        "total_neo_comp": sum(c.total_compensation_usd for c in extract.compensation
                              if c.total_compensation_usd is not None) or None,
        # ---- Ownership / alignment ----
        "insider_ownership_pct": round(insider_pct, 4) if insider_pct else None,
        "ceo_ownership_pct": ceo_pct,
        "n_five_percent_holders": sum(bool(o.is_five_percent_owner) for o in own) or None,
        # ---- Governance provisions ----
        "independent_chair": _bnum(g("independent_chair")),
        "lead_independent_director": _bnum(g("lead_independent_director")),
        "classified_board": _bnum(g("classified_board")),
        "dual_class_shares": _bnum(g("dual_class_shares")),
        "poison_pill": _bnum(g("poison_pill")),
        "majority_voting": _bnum(g("majority_voting_for_directors")),
        "proxy_access": _bnum(g("proxy_access")),
        "say_on_pay_frequency_years": g("say_on_pay_frequency_years"),
        "say_on_pay_support_pct": g("say_on_pay_support_pct"),
        "ceo_pay_ratio": g("ceo_pay_ratio"),
        "median_employee_pay": g("median_employee_pay_usd"),
        "shareholder_proposals_count": g("shareholder_proposals_count"),
        "auditor_name": g("auditor_name"),
        "auditor_fees": g("auditor_fees_usd"),
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


def _last_asof_by_ticker(existing: pd.DataFrame | None) -> dict[str, pd.Timestamp]:
    """Max `as_of` (filing date) already stored per ticker -> the per-ticker
    incremental cutoff. Tickers absent here are fetched over the full window."""
    if existing is None or existing.empty or "as_of" not in existing.columns:
        return {}
    s = existing[["ticker", "as_of"]].copy()
    s["as_of"] = pd.to_datetime(s["as_of"], errors="coerce")
    s = s.dropna(subset=["as_of"])
    return s.groupby("ticker")["as_of"].max().to_dict()


def _seen_accessions(existing: pd.DataFrame | None) -> set[str]:
    if existing is None or existing.empty or "accession_number" not in existing.columns:
        return set()
    return set(existing["accession_number"].dropna())


def _save_ticker_rows(context: Context, rows: list[dict]) -> int:
    """Upsert one ticker's freshly-extracted rows into `def14a_llm` right away
    (LLM calls are expensive — persist per ticker so a crash loses nothing)."""
    df = pd.DataFrame(rows)
    for c in _NUMERIC_COLS:                     # keep DB columns numeric
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            
    df["as_of"] = pd.to_datetime(df["as_of"]).dt.normalize()
    df = df.drop_duplicates(subset=["ticker", "accession_number"], keep="last")
    return context.store.save("def14a_llm", df)


def fetch_def14a_llm(
    context: Context,
    tickers: list[str],
    model: str = "gpt-4o-mini",
    max_chars: int = 130_000,
    temperature: float = 0.0,
    cache: bool = True,
) -> pd.DataFrame:
    """Build/refresh the DEF 14A LLM governance extract, one ticker at a time.

    For each ticker only filings AFTER its latest stored `as_of` are sent to the
    LLM (year-incremental), and the ticker's rows are upserted to Postgres
    immediately. Skips gracefully when OPENAI_API_KEY is absent.
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
    last_asof = _last_asof_by_ticker(existing)     # per-ticker year cutoff (from SQL)
    seen = _seen_accessions(existing)

    try:
        extractor = LLMExtractor(model=model, max_chars=max_chars,
                                 temperature=temperature, cache=cache)
    except EnvironmentError as e:
        context.log.warning("DEF 14A LLM extraction skipped: %s", e)
        return existing if existing is not None else pd.DataFrame(columns=["ticker", "as_of"])

    total_new, tickers_touched = 0, 0
    for _, r in tqdm(cik_map.iterrows(), total=len(cik_map), desc="DEF 14A LLM"):
        ticker, cik, company = r["ticker"], r["cik"], r.get("company_name", "")
        # only filings after this ticker's latest already in SQL (full window if none)
        since = last_asof.get(ticker)
        try:
            filings = list_filings(cik, _FORM, years, company, since=since)
        except Exception as e:
            context.log.warning("%s: DEF 14A filing list failed (%s)", ticker, e)
            continue

        ticker_rows: list[dict] = []
        for _, f in filings.iterrows():
            if f["accession_number"] in seen:
                continue
            row = _process_filing(ticker, f, extractor)
            if row is not None:
                ticker_rows.append(row)
                seen.add(f["accession_number"])

        # persist THIS ticker before moving on (don't batch — LLM calls are costly)
        if ticker_rows:
            _save_ticker_rows(context, ticker_rows)
            total_new += len(ticker_rows)
            tickers_touched += 1

    save_extract_meta(path, today_iso(), len(cik_map), len(cik_map))
    out = context.store.load("def14a_llm")
    context.log.info(
        "DEF 14A LLM: +%d new rows across %d tickers; table now %d rows, %d tickers",
        total_new, tickers_touched, len(out),
        out["ticker"].nunique() if not out.empty else 0,
    )
    return out
