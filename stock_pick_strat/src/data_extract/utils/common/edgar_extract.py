"""
edgar_extract.py  (src/data_extract/utils/edgar_extract.py)
-----------------------------------------------------------
Pure parsing helpers for pulling WORKFORCE data straight out of SEC EDGAR
documents. No API key, no third-party parser dependency (stdlib re / html only),
so these are safe to unit-test and drop into the pipeline.

What lives where on EDGAR (why we parse what we parse):
  * employee count      -> 10-K body text ("we had approximately N employees").
                           There is NO clean XBRL/GAAP concept for headcount, so
                           text extraction is the genuinely-historical route.

(The old executive-officer / insider regex parsers were retired: governance and
executive-pay data now come from the DEF 14A LLM extractor, fetch_def14a_llm.)

Everything is dated by the FILING date upstream (point-in-time / leak-free).
"""

from __future__ import annotations

import html
import re


# --------------------------------------------------------------------------- #
# HTML -> plain text                                                           #
# --------------------------------------------------------------------------- #
def html_to_text(raw: str) -> str:
    """Strip an EDGAR HTML/TXT filing to readable plain text."""
    if not raw:
        return ""
    raw = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>", "\n", raw)
    raw = re.sub(r"(?i)</(p|div|tr|td|th|table|li|h[1-6])>", " ", raw)
    raw = re.sub(r"<[^>]+>", " ", raw)            # remaining tags
    text = html.unescape(raw)
    text = text.replace("\xa0", " ")              # non-breaking space
    text = text.replace("​", "").replace("﻿", "")   # zero-width spaces (proxy tables)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def _flat(text: str) -> str:
    """Whitespace-flattened single-line view (newlines -> spaces)."""
    return re.sub(r"\s+", " ", text or "").strip()


# --------------------------------------------------------------------------- #
# Employee count                                                               #
# --------------------------------------------------------------------------- #
# The count and the noun are NOT reliably adjacent, so two complementary patterns
# are scanned (forward preferred, backward as fallback), each handling a real
# phrasing the strict "<number> employees" form missed:
#   * FORWARD  "1,541,000 full-time and part-time employees" (AMZN) — qualifiers
#     may sit between the number and the noun.
#   * BACKWARD "employees ... was approximately 205,000"      (MCD) — number after
#     the noun, and "62.3 ... thousand" (XOM) — a thousand/million multiplier.
# nouns filers use for their workforce, incl. company-specific ones (teammates=DVA,
# workers=QCOM, staff members=AMGN and many biotechs, crew members=airlines/cruise).
_EMP_NOUN = (r"employees|persons|associates|colleagues|team\s+members|teammates|workers|"
             r"staff\s+members|crew\s+members|workforce|personnel|people")
_NUM = r"\d{1,3}(?:,\d{3})+|\d{4,}|\d{1,3}(?:\.\d+)?"    # 161,000 | 12300 | 62.3
_QUAL = r"(?:(?:full|part)[-\s]?time|regular|salaried|hourly|temporary|permanent|equivalent|active|and|&|,)\s*"
# comparative "N1 and N2 [qual]* employees, respectively" (two fiscal years) -> the
# CURRENT year is the FIRST number, so capture N1 (the adjacent-to-noun form would
# otherwise return the prior year, e.g. KO "65,900 and 69,700 employees").
_CMP_RE = re.compile(
    rf"({_NUM})\s*(thousand|million)?\s+and\s+(?:{_NUM})\s*(?:thousand|million)?\s*(?:{_QUAL}){{0,3}}(?:{_EMP_NOUN})\b", re.I)
_FWD_RE = re.compile(rf"({_NUM})\s*(thousand|million)?\s*(?:{_QUAL}){{0,5}}(?:{_EMP_NOUN})\b", re.I)
# "(average/total) number of [qual] employees [was/:] N" — high precision via the
# "number of" prefix; also catches table rows where the number directly follows the
# noun with no connector (e.g. NSC "Average number of employees 30,456 29,482 ...").
_NUMOF_RE = re.compile(
    rf"number of\s+(?:{_QUAL}){{0,3}}(?:{_EMP_NOUN})\b\s*"
    rf"(?:was|were|is|are|:|of|approximately|about|totaled)?\s*({_NUM})\s*(thousand|million)?", re.I)
_BWD_RE = re.compile(
    rf"(?:{_EMP_NOUN})\b[^.]{{0,55}}?\b(?:was|were|of|is|are|totaled|numbered|approximately|about|at|:)\s+"
    rf"(?:approximately\s+|about\s+)?({_NUM})\s*(thousand|million)?", re.I)
_BAD_PRE = ("stock", "purchase plan", "benefit", "pension", "401(k)",
            "retirement", "savings plan", "stockholders", "shareholders",
            "restricted stock", "option", "per share", "shares", "payroll")
# SUBSET contexts (union / pension / segment counts) — a real total headcount is not
# one of these, so penalize when they surround the number (e.g. KO "400 employees ...
# covered by collective bargaining", CAT "18,000 active participants").
_SUBSET = ("union", "collective bargaining", "represented by", "covered by", "participants",
           "bargaining", "unionized", "subject to collective", "party to")


def _emp_value(raw: str, unit: str | None, tail: str) -> int:
    """Numeric headcount, applying a thousand/million unit — either captured right
    after the number, or (list form '62.3, 63.0, ... thousand') appearing shortly
    after a small decimal."""
    val = float(raw.replace(",", ""))
    if unit:
        val *= 1_000 if unit.lower() == "thousand" else 1_000_000
    elif val < 1_000 and "thousand" in tail:
        val *= 1_000
    elif val < 1_000 and "million" in tail:
        val *= 1_000_000
    return int(round(val))


def _emp_ctx_score(pre: str) -> int:
    s = 0
    if any(k in pre for k in ("approximately", "about", "roughly", "total of", "totaled")):
        s += 2
    if "as of" in pre:
        s += 2
    if any(k in pre for k in ("we had", "we employ", "employed", "workforce", "we have", "had a total")):
        s += 2
    if any(k in pre for k in ("full-time", "full time", "part-time")):
        s += 1
    if any(k in pre for k in ("worldwide", "globally", "global")):
        s += 1
    return s


def extract_employee_count(text: str) -> int | None:
    """
    Best-effort headcount from 10-K text. Scans three patterns — a two-year
    COMPARATIVE ('N1 and N2 employees', taking the current-year N1), a high-precision
    FORWARD ('N [qualifiers] noun'), and a BACKWARD fallback ('noun ... was/of N') —
    each supporting a thousand/million multiplier. Every candidate is scored by the
    context around the number ('as of', 'approximately', 'we had/employed', 'full-time',
    'worldwide'), penalizing benefit/stock-plan and subset (union/pension/segment)
    contexts and bare year-like values; the earlier patterns get a base bonus so the
    reliable phrasing wins. Returns the highest-scoring plausible count.
    """
    if not text:
        return None
    t = _flat(text)
    tl = t.lower()
    best, best_score = None, -1e9
    for regex, base in ((_CMP_RE, 3), (_NUMOF_RE, 2), (_FWD_RE, 2), (_BWD_RE, 0)):
        for m in regex.finditer(t):
            unit = m.group(2)
            ns, ne = m.start(1), m.end(1)
            val = _emp_value(m.group(1), unit, tl[ne:ne + 40])
            if val < 100 or val > 5_000_000:            # sanity band for S&P 500
                continue
            pre = tl[max(0, ns - 60):ns]
            post = tl[ne:ne + 45]
            score = base + _emp_ctx_score(pre)
            if any(b in pre for b in _BAD_PRE):
                score -= 6
            if any(sub in pre or sub in post for sub in _SUBSET):
                score -= 6                              # union / pension / segment subset
            if not unit and "thousand" not in post and 1990 <= val <= 2035:
                score -= 3                              # likely a calendar year, not a count
            if score > best_score:
                best, best_score = val, score
    return best
