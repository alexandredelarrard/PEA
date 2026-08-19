"""
fetch_superinvestors.py  (src/data_extract/utils/prices/fetch_superinvestors.py)
--------------------------------------------------------------------------------
Build the "superinvestors" roster JSON — a `{cik: investor_name}` map of the curated
elite managers — read by the feature layer to recompute the 13F buy/sell-evolution
features for JUST these managers (`superinvestor_features.build_superinvestor_feature_panel`),
on top of the all-filer institutional features.

TWO internet sources, combined:
  * Dataroma (dataroma.com) — the curated ROSTER of proven long-term investors
    (names only; it exposes no CIK, no returns).
  * SEC EDGAR company search — the AUTHORITATIVE fund-name -> 13F-manager CIK lookup.

Output JSON (`data/superinvestors/superinvestors.json`)
"""
from __future__ import annotations

import json
import logging
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from urllib3.exceptions import InsecureRequestWarning

from src.constants.constants import SEC_EDGAR_COMPANY_SEARCH_URL, _HEADERS
from src.context import Context
from src.data_extract.utils.common.sec_utils import sec_get
from src.utils.string import pad_cik

logger = logging.getLogger(__name__)

# manager-name tokens that carry no matching signal (legal / entity boilerplate)
_STOP_TOKENS = {
    "LP", "LLP", "LLC", "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "LTD",
    "LIMITED", "CAPITAL", "MANAGEMENT", "MGMT", "MGT", "PARTNERS", "PARTNER", "GROUP",
    "ADVISORS", "ADVISERS", "ADVISORY", "ASSET", "ASSETS", "FUND", "FUNDS", "HOLDINGS",
    "HOLDING", "INVESTMENT", "INVESTMENTS", "INTERNATIONAL", "GLOBAL", "AND", "THE",
    "COMPANY", "MASTER", "SECURITIES", "TRUST", "FINANCIAL", "RESEARCH", "SERVICES",
}

DATAROMA_HOME_URL = "https://www.dataroma.com/m/home.php"
SUPERINVESTOR_CIK_OVERRIDES: dict[str, str] = {
    "BRK": "0001067983",   # Berkshire Hathaway  (Warren Buffett)
    "HA" : "0000827280",
    "VAN" : "0000858172",
    "RC" : "0001570775", 
    "DAC": "0000200217",
    "PI": "0001549574",
    "MPF": "0000932223",
    "DAV": "0000200305",
    "T" : "0001002778",
    "OA" : "0000885665"
}

# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)                                                    #
# --------------------------------------------------------------------------- #
def _name_tokens(name: str) -> frozenset[str]:
    """Significant upper-case tokens of a manager/fund name (boilerplate dropped)."""
    words = re.sub(r"[^A-Za-z0-9 ]", " ", str(name).upper()).split()
    return frozenset(w for w in words if w not in _STOP_TOKENS and len(w) > 1)


def _fund_part(dataroma_name: str) -> str:
    """Dataroma lists 'Person - Fund'; the SEC filer is the FUND, so search on the
    part after the last dash (fall back to the whole string when there is no dash)."""
    parts = re.split(r"\s[-–—]\s", str(dataroma_name))
    return parts[-1].strip() if len(parts) > 1 else str(dataroma_name).strip()


def _parse_dataroma_roster(html: str) -> list[dict]:
    """Dataroma home page -> [{code, name}] for every `holdings.php?m=CODE` link.
    Deduplicated by code, order preserved. Robust to the surrounding markup."""
    soup = BeautifulSoup(html or "", "html.parser")
    out, seen = [], set()
    for a in soup.find_all("a", href=True):
        m = re.search(r"holdings\.php\?m=([A-Za-z0-9_.\-]+)", a["href"])
        if not m:
            continue
        code = m.group(1)
        name = re.sub(r"\s+", " ", a.get_text(" ", strip=True)).strip()
        # Dataroma appends "Updated <D Mon YYYY>" to each link text; strip it so the
        # date does not leak into the fund name / matching tokens.
        name = re.sub(r"\s+Updated\b.*$", "", name, flags=re.IGNORECASE).strip()
        if code and code not in seen and name:
            seen.add(code)
            out.append({"code": code, "name": name})
    return out


def _parse_edgar_matches(atom_text: str) -> list[tuple[str, str]]:
    """(padded-cik, conformed-name) for each `<company-info>` block in an EDGAR
    company-search atom feed. Tags are LOWER-case (`<cik>`, `<conformed-name>`); the
    conformed-name is empty on multi-match blocks that omit it."""
    out: list[tuple[str, str]] = []
    for block in re.split(r"<company-info", atom_text or "")[1:]:
        cik_m = re.search(r"<cik>(\d+)", block)
        if not cik_m:
            continue
        name_m = re.search(r"<conformed-name>([^<]*)", block)
        out.append((pad_cik(cik_m.group(1)), name_m.group(1).strip() if name_m else ""))
    return out


def _pick_best_match(pairs: list[tuple[str, str]], query: str) -> tuple[str, str] | None:
    """Pick the CIK whose filer name best token-matches `query`; a single match is
    trusted outright, and ties / the no-name multi-match case fall back to EDGAR's
    first (most-relevant) block."""
    if not pairs:
        return None
    if len(pairs) == 1:
        return pairs[0]
    qt = _name_tokens(query)
    idx = max(range(len(pairs)),
              key=lambda i: (len(qt & _name_tokens(pairs[i][1])), -i))
    return pairs[idx]


def _edgar_cik_for_name(fund_name: str, get_fn=sec_get) -> tuple[str | None, str | None]:
    """Resolve a fund name to its 13F-manager CIK via SEC EDGAR company search.
    Returns (cik, filer_name) or (None, None). `get_fn(url) -> response` is injected
    so tests can stub the network."""
    q = _fund_part(fund_name)
    try:
        text = get_fn(SEC_EDGAR_COMPANY_SEARCH_URL.format(company=quote(q))).text
    except Exception as e:                                     # noqa: BLE001
        logger.debug("EDGAR lookup failed for %r: %s", q, e)
        return None, None
    best = _pick_best_match(_parse_edgar_matches(text), q)
    return best if best else (None, None)


# --------------------------------------------------------------------------- #
# IO: Dataroma fetch (its cert chain is incomplete -> verified-then-relaxed)     #
# --------------------------------------------------------------------------- #
def _http_get(url: str) -> requests.Response:
    """GET with SSL verification, falling back to an UNVERIFIED retry on SSLError.
    Dataroma serves an incomplete certificate chain (missing intermediate) that
    OpenSSL cannot verify; the data is public and read-only, so an unverified fetch
    is acceptable here and is logged so the relaxation is never silent."""
    try:
        r = requests.get(url, headers=_HEADERS, timeout=60)
    except requests.exceptions.SSLError:
        logger.warning("Dataroma SSL chain incomplete -> retrying unverified (%s)", url)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InsecureRequestWarning)
            r = requests.get(url, headers=_HEADERS, timeout=60, verify=False)
    r.raise_for_status()
    return r


# --------------------------------------------------------------------------- #
# Entry points                                                                  #
# --------------------------------------------------------------------------- #
def _json_path(context: Context, out_path: str | Path | None) -> Path:
    return Path(out_path) if out_path else (context.paths["DATA_STORE"] / SUPERINVESTORS_JSON)


def build_superinvestors_json(
    context: Context,
    out_path: str | Path | None = None,
    get_fn=sec_get,
) -> dict:
    """Scrape Dataroma's curated superinvestors, resolve each fund name to its 13F
    CIK via SEC EDGAR, and persist a `{cik: investor_name}` roster JSON (also
    returned). CIKs come straight from EDGAR, so this does NOT depend on a local
    13F cache. Fund names EDGAR can't resolve are logged; force a specific CIK via
    SUPERINVESTOR_CIK_OVERRIDES (keyed by the Dataroma manager code)."""

    roster = _parse_dataroma_roster(_http_get(DATAROMA_HOME_URL).text)
    logger.info("Dataroma: parsed %d superinvestors", len(roster))

    cik_to_name: dict[str, str] = {}
    unresolved: list[str] = []
    for r in roster:
        code, name = r["code"], r["name"]
        if code in SUPERINVESTOR_CIK_OVERRIDES:
            cik_to_name[pad_cik(SUPERINVESTOR_CIK_OVERRIDES[code])] = name
            continue
        cik, _filer = _edgar_cik_for_name(name, get_fn=get_fn)
        if cik:
            cik_to_name[cik] = name
        else:
            unresolved.append(f'"{code}"  ({name})')       # show the code to override with

    if unresolved:
        logger.warning("Superinvestors: %d/%d resolved to a CIK; unresolved (add the "
                       "Dataroma code -> CIK to SUPERINVESTOR_CIK_OVERRIDES): %s",
                       len(cik_to_name), len(roster), ", ".join(unresolved[:15]))

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_roster": DATAROMA_HOME_URL,
        "source_cik": "SEC EDGAR company search (13F-HR)",
        "n_roster": len(roster),
        "n_resolved": len(cik_to_name),
        "cik_to_name": cik_to_name,
    }
    path = _json_path(context, out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("Saved superinvestors roster: %d CIKs (of %d roster) -> %s",
                   len(cik_to_name), len(roster), path)
    return out


def load_superinvestors(context: Context, out_path: str | Path | None = None) -> dict | None:
    """Read the persisted superinvestors roster JSON; None if it has not been built."""
    path = _json_path(context, out_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Superinvestors roster at %s is unreadable", path)
        return None
