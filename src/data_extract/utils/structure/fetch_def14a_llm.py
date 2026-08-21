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

Per-filing incremental (gap-filling): each ticker's FULL `years_history` window of DEF 14A
filings is listed, and the LLM is (re-)run ONLY on filings whose `accession_number` is NOT already
in the `def14a_llm` table. So any MISSING year/filing — including a hole in the middle of the
history, not just after the latest — is filled, while every already-extracted filing is skipped
(no repeat LLM cost). Tickers with no rows yet get the whole window; already-complete tickers make
no LLM calls at all.

Requires OPENAI_API_KEY (or OPEN_AI_API_KEY) in the .env file.
If the key is absent the function logs a warning and returns whatever exists.

Output columns (DB table `def14a_llm`), scalar summaries + raw JSON:
    keys        ticker, as_of, period, accession_number, company_name, fiscal_year_extract
    board       n_directors, board_size, avg_director_age, avg_board_tenure,
                pct_independent_directors, pct_female_directors,
                avg_other_public_boards, n_technology_directors,
                pct_technology_directors, technology_committee
    ceo         ceo_name_proxy, ceo_age, ceo_since_year, ceo_is_founder,
                ceo_is_board_chair, ceo_salary, ceo_bonus, ceo_stock_awards,
                ceo_option_awards, ceo_non_equity_incentive, ceo_all_other_comp,
                ceo_total_comp, ceo_equity_pay_pct
    neos        n_neos, total_neo_comp
    ownership   insider_ownership_pct, ceo_ownership_pct, n_five_percent_holders
    governance  independent_chair, lead_independent_director, classified_board,
                dual_class_shares, poison_pill, majority_voting,
                say_on_pay_support_pct, ceo_pay_ratio, median_employee_pay,
                auditor_fees
    def14a_json (full Def14AExtract as JSON for downstream use)
"""
from __future__ import annotations

import logging
import re

import pandas as pd
from tqdm import tqdm

from src.constants.constants import DATE_FORMAT, DEF14A_FORMS
from src.context import Context
from src.data_extract.utils.structure.def14a_schema import Def14AExtract
from src.data_extract.utils.common.edgar_extract import html_to_text
from src.data_extract.utils.common.edgar_fillings import list_filings
from src.data_extract.utils.common.llm_extractor import LLMExtractor
from src.data_extract.utils.common.run_manifest import get_entry, manifest_window, record_run
from src.data_extract.utils.common.sec_utils import existing_filings, load_cik_mapping, sec_get
from src.data_store.schema import Tables

# DEF 14A = the shareholder proxy; DEF 14C = the equivalent INFORMATION STATEMENT that
# CONTROLLED companies file instead (no vote solicited because a controlling holder already has
# the votes, e.g. ERIE = Hirt trusts). Same governance / exec-comp content, so both are extracted.
# Centralized in constants.py as DEF14A_FORMS (form_registry.py's single source of truth).
_FORM = DEF14A_FORMS

# flattened output columns that must be stored numeric (float) in the DB
# (governance booleans are surfaced as 1.0/0.0 flags so they are usable features)
_NUMERIC_COLS = [
    "fiscal_year_extract",
    # board / directors
    "n_directors", "board_size", "avg_director_age", "avg_board_tenure",
    "pct_independent_directors", "pct_female_directors", "avg_other_public_boards",
    "n_technology_directors", "pct_technology_directors", "technology_committee",
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
    "dual_class_shares", "poison_pill", "majority_voting", "say_on_pay_support_pct",
    "ceo_pay_ratio", "median_employee_pay", "auditor_fees",
]

# Task-tailored extraction prompt (cached per model+schema -> stays cheap). Precise
# instructions on WHERE each field lives and how to normalise it materially lift the
# fill rate versus a generic "extract structured data" prompt.
_DEF14A_PROMPT = (
    "You extract structured governance & compensation data from a SEC DEF 14A proxy (or the "
    "equivalent DEF 14C information statement filed by controlled companies). The "
    "input already contains the relevant sections (director nominees, Summary Compensation "
    "Table, corporate governance, say-on-pay, pay ratio, auditor fees, beneficial ownership).\n"
    "- CEO pay: take the CEO's row in the SUMMARY COMPENSATION TABLE for the MOST RECENT "
    "fiscal year; salary/bonus/stock_awards/option_awards/non_equity_incentive/all_other/total "
    "are the dollar columns (a '-' or blank cell = 0). Do not confuse it with the DIRECTOR "
    "compensation table.\n"
    "- Board composition: read the governance/board 'highlights' summary for board_size, "
    "n_independent_directors and n_women_directors (e.g. '7 of our 8 directors are independent').\n"
    "- Provisions (classified_board, dual_class_shares, poison_pill, majority_voting_for_directors): "
    "companies disclose these when they exist, so return FALSE when the proxy does not indicate the "
    "provision is in place — do NOT leave them null.\n"
    "- Ownership: insider_ownership_pct = the 'all directors and executive officers AS A GROUP' "
    "percent; ceo_ownership_pct = the CEO's own row; both as decimals (a '*' or '<1%' -> null). "
    "n_five_percent_holders = count of owners holding >=5%.\n"
    "- auditor_fees_usd = the TOTAL of all fee categories paid to the auditor. "
    "say_on_pay_support_pct as a decimal (92% -> 0.92); ceo_pay_ratio as a number (533:1 -> 533).\n"
    "- Board technology maturity: n_technology_directors = how many directors have material "
    "technology / software / IT / cybersecurity / digital expertise, per the board SKILLS-AND-"
    "QUALIFICATIONS matrix or their bios; technology_committee = True if the board has a dedicated "
    "technology / cybersecurity / digital / innovation committee (else False).\n"
    "Only use values stated in the text; use null when genuinely absent (except the provisions above)."
)

logger = logging.getLogger(__name__)

# Per-section char budgets. LLM extraction is far cheaper AND more accurate on narrow,
# precise text, so each section is capped tightly to just its data (the Summary Comp
# Table and pay-ratio/auditor lines are compact; only the director bios run long). Total
# ~51k chars vs the old ~122k -> ~2.5x cheaper / faster with no loss of the target fields.
_CONTEXT_PRE = 500        # chars of context before the anchor match

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
# EXECUTIVE Summary Compensation Table. PRIMARY anchor: the SEC-mandated "Summary Compensation
# Table" title that is actually FOLLOWED (within ~1.8k chars) by a NEO data row — a fiscal year
# then two large dollar figures (salary + an award column). The data-row lookahead is what
# separates the real table from the many TOC / CD&A / pay-vs-performance references to the same
# title (which are not followed by SCT rows). This is far more format-robust than a header-cluster
# regex (some proxies' "Salary" column isn't adjacent to a marker, or a CD&A "target pay" table
# mimics the header). FALLBACK (`_COMPENSATION_CONTENT_RE`): the column-header cluster — "Salary"
# immediately followed by a COLUMN MARKER (($)/footnote digit/Bonus/Stock/Option/Awards) then a
# plain "Total" (negative-lookahead skips "Total Cash/Direct/Realized" realized-pay tables) and an
# "Awards" column — for the rare filing whose SCT title doesn't survive flattening.
_COMPENSATION_TITLE_RE = re.compile(
    r"Summary\s+Compensation\s+Table"
    r"(?=[\s\S]{0,1800}?\b20\d\d\b[\s\S]{0,25}?[\d,]{6,}[\s\S]{0,25}?[\d,]{5,})", re.I)
_COMPENSATION_CONTENT_RE = re.compile(
    r"\bSalary\b\s*(?:\(\$\)|\$|\d|Bonus|Stock|Option|Awards)"
    r"(?=[\s\S]{0,400}?\bTotal\b(?!\s*(?:Cash|Direct|Realized)))(?=[\s\S]{0,400}?Awards)", re.I)
# Ownership: the target line is "all directors and executive officers as a group"; else a
# beneficial-ownership row (a share count of 5+ digits then a percent). Requiring the
# "beneficially"/"as a group" context avoids false matches on $-amount narratives.
_OWNERSHIP_CONTENT_RE = re.compile(
    r"directors\s+and\s+(?:executive\s+)?officers\s+as\s+a\s+group"
    r"|beneficial(?:ly)?\s+own[\s\S]{0,400}?\b\d[\d,]{4,}\b\s+(?:\*|\d+(?:\.\d+)?\s*%)",
    re.I,
)

# ---- Densest-window row tokens (primary anchoring for the two tabular sections) ----
# Proxies scatter the director bios / beneficial-ownership table in different places and formats
# (a summary matrix, per-director blocks, footnotes), so a single "first match" anchor is fragile.
# Instead, anchor each of these sections on the region with the HIGHEST concentration of its row
# pattern — which is, by construction, the table itself (see `_densest_window`). Lone prose hits
# ("independent director since 2005", "communicate with the directors as a group") never cluster.
# The (?![\d.]|,\d) guard stops a 2-digit "age" from matching the start of a larger number —
# critical because a director-COMPENSATION fee row ("Smith, 45,000") would otherwise look like
# "Smith, 45" and pull the densest window onto the fees table instead of the bios. It rejects a
# following digit/decimal or a thousands comma ("45,000") but still allows a grammatical trailing
# comma ("Alice Johnson, 58, has served ...").
_DIRECTOR_ROW_RE = re.compile(
    r"\bAge\b[:*\s]{0,4}(?:4\d|5\d|6\d|7\d|8\d)(?![\d.]|,\d)"   # "Age 62" / "Age: 70" / "Age** 60"
    r"|\b[A-Z][a-zA-Z]+,\s+(?:4\d|5\d|6\d|7\d|8\d)(?![\d.]|,\d)"  # "Douglas, 62" rows (name, age)
    r"|\bDirector\s+Since\b",                                     # "Director Since" column / label
    re.I,
)
_OWNERSHIP_ROW_RE = re.compile(
    r"\b\d[\d,]{4,}\b\s*(?:\(\d+\)\s*)?(?:\*|\d{1,2}(?:\.\d+)?\s*%)"   # share count + percent/'*'
    r"|\b(?:BlackRock|Vanguard|State\s+Street|FMR\s+LLC|T\.?\s*Rowe\s+Price|"
    r"Capital\s+(?:Research|Group)|Wellington|Massachusetts\s+Financial|Dodge\s*&\s*Cox)\b"  # 5% holders
    r"|\bas\s+a\s+group\b",                                            # insider summary row
    re.I,
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
# pay ratio / median pay, say-on-pay support, and auditor fees live in DIFFERENT parts of
# the proxy, so each gets its OWN small slice (one bundled slice can't reach all three).
_PAYRATIO_ANCHORS = (
    "ceo pay ratio",
    "pay ratio",
    "ratio of the annual total compensation",
    "median employee",
)
_SAYONPAY_ANCHORS = (
    "say-on-pay",
    "say on pay",
    "advisory vote to approve",
    "advisory vote on executive comp",
    "% of the votes cast",
)
_AUDITOR_ANCHORS = (
    "audit fees",
    "fees billed",
    "fees paid to",
    "audit and non-audit fees",
    "independent registered public accounting firm",
)
# say-on-pay result: a percent QUALIFIED as a vote result ("NN% of votes cast", "NN% approval",
# "NN% support", "NN% in favor") that is preceded within ~200 chars by a say-on-pay / advisory-vote
# / executive-compensation phrase. The context requirement anchors the slice on the say-on-pay
# result specifically — without it, an unrelated vote result (e.g. a director election % earlier in
# the doc) is matched first; the old "NN% … votes" pattern also missed "received 91% approval".
# Auditor fees stay a compact $ context.
_SAYONPAY_CONTENT_RE = re.compile(
    r"(?:say[- ]on[- ]pay|advisory\s+vote|executive\s+compensation)[\s\S]{0,200}?"
    r"\d{2}(?:\.\d+)?\s*%\s*(?:of\s+(?:the\s+)?(?:votes?|shares?)\s*(?:cast|voted)"
    r"|in\s+favor|approval|support|were\s+voted)",
    re.I,
)
_AUDITOR_CONTENT_RE = re.compile(r"\baudit\s+fees\b[^\n]{0,20}?[\$\d]", re.I)
# CEO pay-ratio: the heading anchors ("pay ratio") also match the comp-philosophy narrative,
# so anchor on the SEC-mandated disclosure sentence instead — "median … employee" or "ratio of
# the … compensation" — which sits with both the ratio (NNN to 1) and the median-pay $ figure.
_PAYRATIO_CONTENT_RE = re.compile(
    r"median\s+(?:of\s+(?:all\s+)?(?:our\s+)?(?:global\s+)?)?(?:employee|associate|colleague|teammate)"
    r"|\bratio\s+of\s+the\s+(?:annual\s+)?(?:total\s+)?(?:annual\s+)?compensation",
    re.I,
)

def _densest_window(
    text: str,
    row_re: re.Pattern,
    chars: int,
    context_pre: int = _CONTEXT_PRE,
    min_rows: int = 3,
) -> int:
    """Return the start of the `chars`-wide window holding the MOST `row_re` matches
    (the table), `context_pre` chars before its first row — or -1 if fewer than
    `min_rows` matches exist (caller then falls back to `_find_content_section`).

    Robust to proxies that scatter the director / ownership data or format it as a
    matrix vs per-person blocks: the densest cluster of row tokens IS the table,
    whereas isolated prose mentions never accumulate.
    """
    starts = [m.start() for m in row_re.finditer(text)]
    if len(starts) < min_rows:
        return -1
    best_start, best_n = starts[0], 0
    for i, p in enumerate(starts):
        n = 0
        for q in starts[i:]:
            if q <= p + chars:
                n += 1
            else:
                break
        if n > best_n:
            best_n, best_start = n, p
    return max(0, best_start - context_pre)


def _find_content_section(
    text: str,
    content_re: "re.Pattern | tuple[re.Pattern, ...] | None",
    fallback_anchors: tuple[str, ...],
    context_pre: int = _CONTEXT_PRE,
    last_occurrence: bool = False,
) -> int:
    """Find the start of a section using content-based regex(es).

    `content_re` may be a single pattern or a TUPLE of patterns tried in priority
    order (the first that matches wins — used for compensation: prefer the SCT
    title+data-row anchor, else the header-cluster fallback). Returns a position
    `context_pre` chars before the first (or last, if `last_occurrence=True`) match
    so the LLM sees full context. When `content_re` is None (prose sections) or no
    content match is found, falls back to text-anchor scanning (skipping the TOC).
    """
    patterns = (() if content_re is None
                else content_re if isinstance(content_re, tuple) else (content_re,))
    for pattern in patterns:
        if last_occurrence:
            matches = list(pattern.finditer(text))
            if matches:
                return max(0, matches[-1].start() - context_pre)
        else:
            m = pattern.search(text)
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

    The two TABULAR sections (director nominees, security ownership) are located by
    a densest-window scan of their row tokens — the region with the most rows IS the
    table — which is robust to matrix vs per-person layouts and footnotes. The rest
    use a content-based regex matching their real body (the SCT column-header row, the
    pay-ratio / say-on-pay result sentence, the audit-fee line) rather than a TOC
    heading. Every section falls back to text anchors (skipping the TOC) when its
    primary locator misses; CORPORATE GOVERNANCE is prose, so it is anchor-only. Each
    section is capped by its own budget and the slices are concatenated. Falls back to
    the head of the document when no patterns match.
    """
    parts: list[str] = []

    for label, dense_re, content_re, anchors, use_last, chars in (
        ("DIRECTOR NOMINEES",      _DIRECTOR_ROW_RE,  _DIRECTOR_CONTENT_RE,                            _DIRECTOR_ANCHORS,     False, 20_000),
        ("EXECUTIVE COMPENSATION", None,              (_COMPENSATION_TITLE_RE, _COMPENSATION_CONTENT_RE), _COMPENSATION_ANCHORS, False,  7_000),
        ("SECURITY OWNERSHIP",     _OWNERSHIP_ROW_RE, _OWNERSHIP_CONTENT_RE,    _OWNERSHIP_ANCHORS,    False, 10_000),
        ("CORPORATE GOVERNANCE",   None,              None,                     _GOVERNANCE_ANCHORS,   False,  6_000),
        ("PAY RATIO & MEDIAN PAY", None,              _PAYRATIO_CONTENT_RE,     _PAYRATIO_ANCHORS,     False,  3_500),
        ("SAY ON PAY",             None,              _SAYONPAY_CONTENT_RE,     _SAYONPAY_ANCHORS,     False,  2_000),
        ("AUDITOR FEES",           None,              _AUDITOR_CONTENT_RE,      _AUDITOR_ANCHORS,      True,   2_500),
    ):
        pos = _densest_window(text, dense_re, chars) if dense_re is not None else -1
        if pos == -1:
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
    directors list (the CEO is almost always a director nominee)."""
    if extract.ceo_age is not None:
        return extract.ceo_age
    name = (extract.ceo_name or "").strip().lower()
    if name:
        for person in extract.directors:
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
    ceo_name = (extract.ceo_name or (ceo.name if ceo else None) or "")

    ceo_equity = None
    if ceo and ceo.total_compensation_usd:
        equity = (ceo.stock_awards_usd or 0) + (ceo.option_awards_usd or 0)
        ceo_equity = round(equity / ceo.total_compensation_usd, 3)

    board_size = g("board_size") or (len(dirs) or None)
    # board composition: prefer the DIRECT governance-highlights counts (robust to text
    # trimming), fall back to computing from the per-director list.
    n_indep = g("n_independent_directors")
    if n_indep is not None and board_size:
        pct_independent = round(n_indep / board_size, 3)
    elif dirs and any(d.is_independent is not None for d in dirs):
        pct_independent = round(sum(bool(d.is_independent) for d in dirs) / len(dirs), 3)
    else:
        pct_independent = None
    n_women = g("n_women_directors")
    if n_women is not None and board_size:
        pct_female = round(n_women / board_size, 3)
    elif genders:
        pct_female = round(sum(x.startswith("f") for x in genders) / len(genders), 3)
    else:
        pct_female = None

    return {
        "ticker": ticker,
        "as_of": filing["filing_date"],
        "period": pd.to_datetime(filing.get("period_of_report"), errors="coerce"),
        "accession_number": filing["accession_number"],
        "company_name": extract.company_name,
        "fiscal_year_extract": extract.fiscal_year,
        # ---- Board / directors ----
        "n_directors": len(dirs) or None,
        "board_size": board_size,
        "avg_director_age": _mean(ages),
        "avg_board_tenure": _mean(tenures),
        "pct_independent_directors": pct_independent,
        "pct_female_directors": pct_female,
        "avg_other_public_boards": _mean(other_boards),
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
        # ---- Ownership / alignment (direct from the beneficial-ownership summary) ----
        "insider_ownership_pct": g("insider_ownership_pct"),
        "ceo_ownership_pct": g("ceo_ownership_pct"),
        "n_five_percent_holders": g("n_five_percent_holders"),
        # ---- Governance provisions ----
        "independent_chair": _bnum(g("independent_chair")),
        "lead_independent_director": _bnum(g("lead_independent_director")),
        "classified_board": _bnum(g("classified_board")),
        "dual_class_shares": _bnum(g("dual_class_shares")),
        "poison_pill": _bnum(g("poison_pill")),
        "majority_voting": _bnum(g("majority_voting_for_directors")),
        # ---- board technology / software maturity (AI-adoption governance signal) ----
        "n_technology_directors": g("n_technology_directors"),
        "pct_technology_directors": (round(g("n_technology_directors") / board_size, 3)
                                     if g("n_technology_directors") is not None and board_size
                                     else None),
        "technology_committee": _bnum(g("technology_committee")),
        "say_on_pay_support_pct": g("say_on_pay_support_pct"),
        "ceo_pay_ratio": g("ceo_pay_ratio"),
        "median_employee_pay": g("median_employee_pay_usd"),
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
        extract = extractor.extract(Def14AExtract, focused, instructions=_DEF14A_PROMPT)
        return _flatten(ticker, filing, extract)
    except Exception as e:
        logger.warning("%s %s: DEF 14A LLM extraction failed (%s)",
                       ticker, filing.get("filing_date", ""), e)
        return None


def _is_up_to_date(context: Context, requested_tickers: list[str]) -> bool:
    """Up to date only when EVERY requested ticker already has rows in the DB AND
    the shared extraction manifest (`run_manifest.py`) was refreshed today. The old
    check compared a DATE + a stored COUNT (`universe_size`), so a same-day rerun
    skipped tickers that were never actually extracted -- the '~15 tickers then it
    stops' bug. Checking per-ticker coverage (tickers x date) makes a rerun pick up
    the still-missing names; the per-ticker loop then skips already-done filings
    via `seen` (no re-LLM)."""
    if not context.store.exists(Tables.def14a_llm):
        return False
    entry = get_entry(context, Tables.def14a_llm)
    if entry is None or entry.get("last_run_date") != pd.Timestamp.today().strftime(DATE_FORMAT):
        return False
    have = set(context.store.load(Tables.def14a_llm, columns=["ticker"])["ticker"].dropna())
    return set(requested_tickers).issubset(have)


def _strip_nul(df: pd.DataFrame) -> pd.DataFrame:
    """Remove NUL (`\\x00`) characters from every string cell. Postgres TEXT columns
    cannot store NUL (psycopg2 raises 'a string literal cannot contain NUL characters'),
    and DEF 14A filings are HTML / PDF-derived, so the extracted strings (company_name,
    ceo_name_proxy, the def14a_json dump, ...) occasionally carry stray NULs. Pure.

    Dtype-agnostic (pandas 2 `object` AND pandas 3 `str` string columns): only the
    columns that actually hold a NUL-bearing string are rewritten, so numeric / datetime
    columns keep their dtype."""
    for c in df.columns:
        col = df[c]
        if col.map(lambda v: isinstance(v, str) and "\x00" in v).any():
            df[c] = col.map(lambda v: v.replace("\x00", "") if isinstance(v, str) else v)
    return df


def _save_ticker_rows(context: Context, rows: list[dict]) -> int:
    """Upsert one ticker's freshly-extracted rows into `def14a_llm` right away
    (LLM calls are expensive — persist per ticker so a crash loses nothing)."""
    df = pd.DataFrame(rows)
    for c in _NUMERIC_COLS:                     # keep DB columns numeric
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = _strip_nul(df)                         # Postgres TEXT rejects NUL (\x00)
    df["as_of"] = pd.to_datetime(df["as_of"]).dt.normalize()
    df = df.drop_duplicates(subset=["ticker", "accession_number"], keep="last")
    return context.store.save(Tables.def14a_llm, df)


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
    de = context.config.data_extract

    cik_map = load_cik_mapping(context, tickers)

    if _is_up_to_date(context, cik_map["ticker"].tolist()):
        existing = context.store.load(Tables.def14a_llm)
        context.log.info("DEF 14A LLM already up to date — every requested ticker present "
                         "(%d rows) — skipping", len(existing))
        return existing

    # accessions already extracted -> never re-LLM (accession-only dedup, same convention as
    # fetch_8k_edgar.py / fetch_13d_edgar.py / fetch_def14a_edgar.py's `existing_filings`)
    # a mutable copy: this fetcher is serial and adds each accession as it extracts it,
    # so a ticker filing twice in one run is not sent to the LLM twice
    seen = set(existing_filings(context, Tables.def14a_llm))

    # Manifest-driven listing window (see run_manifest.py): a routine run only lists
    # filings from the last run's date onward; a ticker-count change or the
    # `manifest_full_rescan_days` self-heal window falls back to the FULL `years`
    # window (gap-filling, same self-heal rationale as the other 4 EDGAR fetchers).
    # `list_filings`'s own `since` cutoff is STRICTLY AFTER the date passed, so we
    # step back one day to keep the last run's date itself inclusive.
    rescan_days = int(getattr(de, "manifest_full_rescan_days", 30))
    manifest_since, is_full_rescan = manifest_window(
        context, Tables.def14a_llm, len(cik_map),
        fallback_since=pd.Timestamp.today() - pd.DateOffset(years=years),
        full_rescan_days=rescan_days)
    list_since = None if is_full_rescan else (manifest_since - pd.Timedelta(days=1))

    try:
        extractor = LLMExtractor(model=model, max_chars=max_chars,
                                 temperature=temperature, cache=cache)
    except EnvironmentError as e:
        context.log.warning("DEF 14A LLM extraction skipped: %s", e)
        existing = context.store.load(Tables.def14a_llm, optional=True)
        return existing if existing is not None else pd.DataFrame(columns=["ticker", "as_of"])

    total_new, tickers_touched, total_skipped = 0, 0, 0
    for _, r in tqdm(cik_map.iterrows(), total=len(cik_map), desc="DEF 14A LLM"):
        ticker, cik, company = r["ticker"], r["cik"], r.get("company_name", "")
        # `list_since=None` (full-rescan runs) lists the FULL years_history window so a MISSING
        # filing anywhere in the history is discovered; otherwise only filings from the manifest's
        # last run date onward are listed. The accession skip below then sends ONLY the
        # not-yet-stored filings to the LLM (gap-filling, per ticker / per date).
        try:
            filings = list_filings(cik, _FORM, years, company, since=list_since)
        except Exception as e:
            context.log.warning("%s: DEF 14A filing list failed (%s)", ticker, e)
            continue

        ticker_rows: list[dict] = []
        skipped = 0
        for _, f in filings.iterrows():
            if f["accession_number"] in seen:      # already in the table -> skip this filing
                skipped += 1
                continue
            row = _process_filing(ticker, f, extractor)
            if row is not None:
                ticker_rows.append(row)
                seen.add(f["accession_number"])
        total_skipped += skipped

        # persist THIS ticker before moving on (don't batch — LLM calls are costly)
        if ticker_rows:
            _save_ticker_rows(context, ticker_rows)
            total_new += len(ticker_rows)
            tickers_touched += 1
            context.log.info("%s: +%d new DEF 14A filing(s) sent to the LLM (%d already in table)",
                             ticker, len(ticker_rows), skipped)

    record_run(context, Tables.def14a_llm, len(cik_map), total_new, is_full_rescan=is_full_rescan)
    out = context.store.load(Tables.def14a_llm)
    context.log.info(
        "DEF 14A LLM: +%d new rows across %d tickers (%d filings already present, skipped); "
        "table now %d rows, %d tickers", total_new, tickers_touched, total_skipped, len(out),
        out["ticker"].nunique() if not out.empty else 0,
    )
    return out
