"""
edgar_extract.py  (src/data_extract/utils/edgar_extract.py)
-----------------------------------------------------------
Pure parsing helpers for pulling management / workforce / insider data straight
out of SEC EDGAR documents. No API key, no third-party parser dependency
(stdlib re / html / xml.etree only), so these are safe to unit-test and drop
into the pipeline.

What lives where on EDGAR (why we parse what we parse):
  * employee count      -> 10-K body text ("we had approximately N employees").
                           There is NO clean XBRL/GAAP concept for headcount, so
                           text extraction is the genuinely-historical route.
  * executive officers  -> 10-K Part I "Information about our Executive Officers"
                           (name / age / title) -> CEO age, founder flags.
  * insider buying      -> Form 3/4/5 XML (structured; the reliable one).

Everything is dated by the FILING date upstream (point-in-time / leak-free).
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET


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


# --------------------------------------------------------------------------- #
# Executive officers (name / age / title) from the 10-K                        #
# --------------------------------------------------------------------------- #
_SECTION_ANCHORS = (
    "information about our executive officers",
    "executive officers of the registrant",
    "executive officers of the company",
    "our executive officers",
    "information about our executive",
)
# name = 2-4 capitalized tokens (allowing middle initials like "D."), then age 30-99.
# Greedy on purpose -- we then TRIM leading title/role words that got glued on.
# The name->age separator allows a comma ("Reed Hastings, 63, ...") so prose-style
# officer blurbs are parsed, not just whitespace-delimited "Name Age" tables.
_OFF_RE = re.compile(
    r"([A-Z][A-Za-z.'\-]+(?:\s+(?:[A-Z]\.?|[A-Z][A-Za-z.'\-]+)){1,3})[,\s]+([3-9]\d)\b")

# Founder detection. Real 10-Ks say "founded" / "co-founded" in a bio sentence far
# more often than the noun "Founder" in the position line, so match the whole
# family; `founded`/`founding` alone are also matched but the passive company-history
# form ("the Company was founded in 18xx") is stripped first so it doesn't count a
# non-founder CEO as a founder (see _is_founder).
_FOUNDER_RE = re.compile(
    r"\b(?:co[-\s]?found(?:ed|er|ers|ing)|founders?|found(?:ed|ing))\b", re.I)
_COMPANY_FOUNDED_RE = re.compile(r"\b(?:was|were|been)\s+found(?:ed|ing)\b", re.I)


def _is_founder(bio: str) -> bool:
    """True when the officer's blurb says THEY founded/co-founded the firm.
    A passive 'the Company was founded in 18xx' is removed first, so it only
    counts when some active founder mention (co-founded / founder / founded)
    remains after that."""
    if not bio or not _FOUNDER_RE.search(bio):
        return False
    return bool(_FOUNDER_RE.search(_COMPANY_FOUNDED_RE.sub(" ", bio)))

# Role / header words that must never start a person's name. In run-together
# text ("...Chief Executive Officer Luca Maestri 60...") the greedy name capture
# swallows the previous person's title; stripping these leading tokens recovers
# the real name and, crucially, lets the previous record keep its full title.
_TITLE_STOP = {
    "chief", "executive", "officer", "officers", "senior", "vice", "president",
    "financial", "operating", "general", "counsel", "secretary", "treasurer",
    "principal", "accounting", "technology", "information", "marketing",
    "revenue", "legal", "administrative", "human", "resources", "corporate",
    "and", "of", "the", "our", "name", "age", "position", "title", "mr", "ms",
    "dr", "years", "director", "group",
}


def _slice_officer_section(text: str) -> str | None:
    low = text.lower()
    for a in _SECTION_ANCHORS:
        i = low.find(a)
        if i != -1:
            return text[i:i + 4000]               # section is short; cap scope
    return None


def _clean_name(greedy: str) -> str:
    toks = greedy.split()
    while len(toks) > 2 and toks[0].lower().strip(".,") in _TITLE_STOP:
        toks.pop(0)
    if len(toks) > 3:
        toks = toks[-3:]                          # first [+ initial] + last
    return " ".join(toks)


def _people_from_scope(scope: str) -> list[dict]:
    """Name/age/title/bio records from a text scope (10-K officer block or DEF 14A
    director/officer section). Shared by both parsers so their behaviour matches."""
    scope = _flat(scope)
    # Pass 1: age anchors -> (clean name, char span). Title end is the NEXT
    # record's clean-name start, so titles keep their full text.
    recs = []
    for m in _OFF_RE.finditer(scope):
        greedy, age = m.group(1), int(m.group(2))
        name = _clean_name(greedy)
        if len(name.split()) < 2 or len(name) < 5:
            continue
        name_start = m.start(1) + greedy.rfind(name)
        recs.append({"name": name, "age": age,
                     "name_start": name_start, "age_end": m.end(2)})

    officers = []
    for i, r in enumerate(recs):
        end = recs[i + 1]["name_start"] if i + 1 < len(recs) else min(len(scope), r["age_end"] + 400)
        blurb = re.sub(r"\s+", " ", scope[r["age_end"]:end]).strip(" ,.;-")
        if not blurb:
            continue
        # `title` = the position line (first ~80 chars, used for display); `bio` =
        # the fuller blurb, where founder status usually lives ("... co-founded the
        # Company in ...") and which the CEO-role search below also scans.
        officers.append({"name": r["name"], "age": r["age"],
                         "title": blurb[:80], "bio": blurb[:400]})
    return officers


def _derive_officer_stats(officers: list[dict]) -> dict:
    """Derive CEO age / founder flags / officer stats from a people list."""
    out = {
        "officers": [], "ceo_name": None, "ceo_age": None,
        "founder_present": 0, "founder_ceo": 0,
        "n_officers": 0, "avg_officer_age": None,
    }
    if not officers:
        return out
    ages = [o["age"] for o in officers]
    out["officers"] = officers
    out["n_officers"] = len(officers)
    out["avg_officer_age"] = round(sum(ages) / len(ages), 1)
    out["founder_present"] = int(any(_is_founder(o["bio"]) for o in officers))

    # CEO = first officer whose blurb states the CEO role. Scan the blurb (not just
    # the 80-char title) so a bio that leads with other text isn't missed; cap the
    # window so a "former Chief Executive Officer of X" further down a bio doesn't
    # hijack the match.
    ceo = next((o for o in officers
                if "chief executive" in o["bio"][:220].lower()
                or re.search(r"\bceo\b", o["bio"][:220], re.I)), None)
    if ceo:
        out["ceo_name"] = ceo["name"]
        out["ceo_age"] = ceo["age"]
        out["founder_ceo"] = int(_is_founder(ceo["bio"]))
    return out


def extract_executive_officers(text: str) -> dict:
    """
    Parse the '(Information about our) Executive Officers' block into a list of
    {name, age, title}, plus derived CEO age / founder flags / officer stats.
    Falls back to scanning the whole doc if the section header isn't found.
    """
    if not text:
        return _derive_officer_stats([])
    scope = _slice_officer_section(text) or text
    return _derive_officer_stats(_people_from_scope(scope))


# --------------------------------------------------------------------------- #
# DEF 14A proxy fallback (directors + officers, when the 10-K has no ages)     #
# --------------------------------------------------------------------------- #
# Many firms incorporate executive-officer/age info by reference into the DEF 14A
# proxy rather than the 10-K, so ~half of 10-Ks yield no age. The proxy's director
# nominee / executive-officer tables carry Name + Age + bio in the same shape the
# 10-K parser already handles -- we just point it at the proxy's sections. The CEO
# is (almost always) a director nominee, so its age is recoverable here.
_DEF14A_ANCHORS = (
    "information about our executive officers",
    "executive officers of the",
    "our executive officers",
    "nominees for election",
    "nominees for director",
    "election of directors",
    "directors and nominees",
    "our board of directors",
    "board of directors",
)


def _slice_def14a_section(text: str) -> str | None:
    """Slice from the first director/officer anchor. Proxy tables run long, so the
    scope is wider than the 10-K's."""
    low = text.lower()
    best = min((low.find(a) for a in _DEF14A_ANCHORS if low.find(a) != -1),
               default=-1)
    return text[best:best + 20000] if best != -1 else None


def extract_management_from_def14a(text: str) -> dict:
    """Fallback management parse from a DEF 14A proxy: same {name, age, title, bio}
    extraction + CEO age / founder / officer stats, scoped to the proxy's director
    and executive-officer sections. Use only to FILL what the 10-K did not yield."""
    if not text:
        return _derive_officer_stats([])
    scope = _slice_def14a_section(text) or text
    return _derive_officer_stats(_people_from_scope(scope))


# --------------------------------------------------------------------------- #
# Form 4 insider transactions (XML)                                            #
# --------------------------------------------------------------------------- #
def _txt(node, path):
    el = node.find(path)
    return el.text.strip() if el is not None and el.text else None


def parse_form4(xml_bytes: str | bytes) -> dict:
    """
    Parse a Form 3/4/5 ownership XML into
        {symbol, period, owner, transactions:[{date, code, shares, ad, price,
                                               derivative}]}
    code: P=open-market buy, S=open-market sale, A=grant/award, M=option
    exercise, F=tax withholding, G=gift, etc.  ad: 'A' acquired / 'D' disposed.
    """
    out = {"symbol": None, "period": None, "owner": None, "transactions": []}
    if not xml_bytes:
        return out
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return out

    out["symbol"] = _txt(root, ".//issuer/issuerTradingSymbol") or _txt(root, ".//issuerTradingSymbol")
    out["period"] = _txt(root, ".//periodOfReport")
    out["owner"] = _txt(root, ".//reportingOwner/reportingOwnerId/rptOwnerName") \
        or _txt(root, ".//rptOwnerName")

    for tag, is_deriv in (("nonDerivativeTransaction", False),
                          ("derivativeTransaction", True)):
        for tr in root.iter(tag):
            date = _txt(tr, "transactionDate/value")
            code = _txt(tr, "transactionCoding/transactionCode")
            shares = _txt(tr, "transactionAmounts/transactionShares/value")
            ad = _txt(tr, "transactionAmounts/transactionAcquiredDisposedCode/value")
            price = _txt(tr, "transactionAmounts/transactionPricePerShare/value")
            if shares is None:
                continue
            try:
                shares_f = float(shares)
            except ValueError:
                continue
            out["transactions"].append({
                "date": date, "code": code, "shares": shares_f,
                "ad": ad, "price": float(price) if price else None,
                "derivative": is_deriv,
            })
    return out


def signed_open_market_shares(transactions: list[dict]) -> float:
    """
    Net OPEN-MARKET shares from one filing: + for code P (buy), - for code S
    (sale). Grants (A), option exercises (M), tax withholding (F) and gifts (G)
    are excluded -- they aren't discretionary conviction signals.
    """
    net = 0.0
    for tr in transactions:
        if tr["code"] == "P":
            net += tr["shares"]
        elif tr["code"] == "S":
            net -= tr["shares"]
    return net


def rolling_net_insider(records: list[dict], window_days: int = 182) -> list[dict]:
    """
    From per-filing insider records [{date: 'YYYY-MM-DD', net: signed_shares}],
    compute the trailing `window_days` net open-market shares AS OF each distinct
    filing date (point-in-time: only transactions on/before that date). Returns
    [{as_of, net_insider_shares}] sorted by date. Downstream you can normalize by
    shares outstanding to get a size-comparable fraction.
    """
    import datetime as _dt

    rows = []
    for r in records:
        if not r.get("date"):
            continue
        try:
            d = _dt.date.fromisoformat(r["date"][:10])
        except ValueError:
            continue
        rows.append((d, float(r.get("net", 0.0))))
    if not rows:
        return []
    rows.sort()
    dates = sorted({d for d, _ in rows})
    out = []
    for d in dates:
        lo = d - _dt.timedelta(days=window_days)
        net = sum(v for dd, v in rows if lo < dd <= d)
        out.append({"as_of": d.isoformat(), "net_insider_shares": net})
    return out