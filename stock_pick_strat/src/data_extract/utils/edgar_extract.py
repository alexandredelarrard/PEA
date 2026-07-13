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
# number = comma-grouped (161,000) or a bare run of 3+ digits (12300)
_NUM = r"(\d{1,3}(?:,\d{3})+|\d{3,})"
_EMP_NOUNS = r"(?:employees|persons|people|team\s+members|associates|colleagues|full[-\s]?time\s+employees)"
_EMP_RE = re.compile(
    rf"{_NUM}\s+(full[-\s]?time\s+)?(equivalent\s+)?{_EMP_NOUNS}", re.I)
_BAD_PRE = ("stock", "purchase plan", "benefit", "pension", "401(k)",
            "retirement", "savings plan", "stockholders", "shareholders")


def extract_employee_count(text: str) -> int | None:
    """
    Best-effort headcount from 10-K text. Scans every '<number> [full-time]
    employees' occurrence and scores it by surrounding context ('as of',
    'approximately', 'we had/employed', 'full-time', 'worldwide'), penalizing
    benefit-plan contexts. Returns the highest-scoring plausible count.
    """
    if not text:
        return None
    t = _flat(text)
    best, best_score = None, -10
    for m in _EMP_RE.finditer(t):
        num = int(m.group(1).replace(",", ""))
        if num < 100 or num > 5_000_000:           # sanity band for S&P 500
            continue
        pre = t[max(0, m.start() - 80):m.start()].lower()
        post = t[m.end():m.end() + 30].lower()
        score = 0
        if m.group(2):
            score += 3                             # "full-time"
        if any(k in pre for k in ("approximately", "about", "roughly", "total of", "totaled")):
            score += 2
        if "as of" in pre:
            score += 2
        if any(k in pre for k in ("we had", "we employed", "we employ", "we have", "employed", "workforce")):
            score += 2
        if any(k in post for k in ("worldwide", "globally", "in total")):
            score += 1
        if any(b in pre for b in _BAD_PRE):
            score -= 5
        if score > best_score:
            best, best_score = num, score
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
_OFF_RE = re.compile(
    r"([A-Z][A-Za-z.'\-]+(?:\s+(?:[A-Z]\.?|[A-Z][A-Za-z.'\-]+)){1,3})\s+([3-9]\d)\b")

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


def extract_executive_officers(text: str) -> dict:
    """
    Parse the '(Information about our) Executive Officers' block into a list of
    {name, age, title}, plus derived CEO age / founder flags / officer stats.
    Falls back to scanning the whole doc if the section header isn't found.
    """
    out = {
        "officers": [], "ceo_name": None, "ceo_age": None,
        "founder_present": 0, "founder_ceo": 0,
        "n_officers": 0, "avg_officer_age": None,
    }
    if not text:
        return out
    scope = _slice_officer_section(text) or text
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
        end = recs[i + 1]["name_start"] if i + 1 < len(recs) else min(len(scope), r["age_end"] + 100)
        title = re.sub(r"\s+", " ", scope[r["age_end"]:end]).strip(" ,.;-")[:80]
        if not title:
            continue
        officers.append({"name": r["name"], "age": r["age"], "title": title})

    if not officers:
        return out

    ages = [o["age"] for o in officers]
    out["officers"] = officers
    out["n_officers"] = len(officers)
    out["avg_officer_age"] = round(sum(ages) / len(ages), 1)
    out["founder_present"] = int(any("founder" in o["title"].lower() for o in officers))

    ceo = next((o for o in officers
                if "chief executive" in o["title"].lower()
                or re.search(r"\bceo\b", o["title"], re.I)), None)
    if ceo:
        out["ceo_name"] = ceo["name"]
        out["ceo_age"] = ceo["age"]
        out["founder_ceo"] = int("founder" in ceo["title"].lower())
    return out


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