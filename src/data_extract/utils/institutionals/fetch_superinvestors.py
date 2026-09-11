"""
fetch_superinvestors.py (src/data_extract/utils/institutionals/fetch_superinvestors.py)
--------------------------------------------------------------------------------
WRITE side of `superinvestor_roster` -- Dataroma's curated roster of proven long-term
managers, stored ONE ROW PER (snapshot_date, dataroma_code) so roster membership is a
fact over time. The read side is `src/utils/superinvestor_roster.py`.

TWO internet sources, combined:
  * Dataroma (dataroma.com) — the curated ROSTER of proven long-term investors
    (names only; it exposes no CIK, no returns).
  * SEC EDGAR company search — the AUTHORITATIVE fund-name -> 13F-manager CIK lookup.

Two entry points:
  * `seed_roster_history`   — one-off: the 13 web.archive.org captures committed at
    `data/superinvestors/dataroma_roster_history.json` (2013 -> 2026, 879 manager-rows,
    104 distinct codes). Committed rather than re-scraped because Wayback rate-limits
    hard and the walk is slow and flaky.
  * `upsert_roster_snapshot` — every run: scrape today's roster, write today's snapshot.

This REPLACES the old `{cik: investor_name}` JSON, which carried a single `generated_at`
and so could not say who was on the roster at a past date. Applying today's 81 names to
2013 drops the 23 managers Dataroma has since dropped -- six with real 13F history, two
of them (Arlington Value, Wintergreen) the concentrated managers a concentration selector
ranks highest. That bias runs in the same direction as the selection rule.
"""
from __future__ import annotations

import json
import logging
import re
import warnings
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib3.exceptions import InsecureRequestWarning

from src.constants.constants import SEC_EDGAR_COMPANY_SEARCH_URL, _HEADERS
from src.context import Context
from src.data_extract.utils.common.sec_utils import sec_get
from src.data_store.schema import Tables
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
# The seed captures. Wayback resolves `/web/<year>/<url>` to that year's nearest capture.
_WAYBACK_URL = "https://web.archive.org/web/{year}/" + DATAROMA_HOME_URL
_ROSTER_HISTORY_FILE = Path("superinvestors") / "dataroma_roster_history.json"

# Resolution provenance, stored per row so a hand-mapped CIK is never mistaken for one
# EDGAR returned.
RESOLUTION_EDGAR = "edgar"
RESOLUTION_OVERRIDE = "override"
RESOLUTION_UNRESOLVED = "unresolved"

# Dataroma code -> 13F-manager CIK, for the names EDGAR company search gets wrong or
# cannot see. Most failures are mutual-fund SHARE CLASSES whose 13F filer is the ADVISER,
# not the fund, so `_fund_part` hands EDGAR a fund name that never filed a 13F-HR:
# re-querying on the adviser name resolves them. Naive auto-resolution gets only 12 of the
# 23 managers Dataroma has dropped since 2013 (52%), and misses 16 more that are on the
# roster today.
#
# ⚠ AMBIGUITY IS SETTLED BY 13F ROW COUNT IN `sec13f_hr`, never by name alone. `lmvtx` had
# 2 candidates (14,379 rows vs 809), `oakvx` 3 (1,233 vs 1 vs 1), `FPACX`/`FPPTX` 3 (2,271
# vs 0 vs 0) and `pzfvx` 2 (2,789 vs 0). A name match that filed nothing is not the manager
# -- the retired JSON picked First Pacific Advisors INC (0 rows) over the LLC that actually
# files, so FPA contributed nothing to the elite features.
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
    "OA" : "0000885665",
    # --- managers dropped from the roster since 2013 (verified against sec13f_hr) --- #
    "HRSVX": "0000937394",  # Heartland Advisors                1,600 rows / 51q
    "TVAFX": "0001145020",  # Thornburg Investment Mgmt         2,820 rows / 51q
    "YAFFX": "0000905567",  # Yacktman Asset Management         1,826 rows / 51q
    "cfimx": "0001036325",  # Davis Selected Advisers           3,149 rows / 51q
    "lmvtx": "0001348883",  # ClearBridge / Legg Mason Capital 14,379 rows / 51q
    "oakvx": "0001085256",  # RS Investment Management          1,233 rows / 13q (ends 2016-06)
    "DJCO" : "0000783412",  # Daily Journal Corp                  147 rows / 49q
    "t2"   : "0001327388",  # T2 Partners Management, LP
    "FEVAX": "0001325447",  # First Eagle -- on today's roster too: a DEDUP, not a new manager
    # --- share classes whose ADVISER is the filer (all 13 snapshots, still on the roster) --- #
    "ARFFX": "0000936753",  # Ariel Focus Fund        -> Ariel Investments LLC        2,285 rows
    "CAAPX": "0000936753",  # Ariel Appreciation Fund -> Ariel Investments LLC   (same adviser)
    "FPACX": "0001377581",  # FPA Crescent Fund       -> First Pacific Advisors LLC   2,271 rows
    "FPPTX": "0001377581",  # FPA Queens Road         -> First Pacific Advisors LLC (same adviser)
    "LLPFX": "0000807985",  # Longleaf Partners       -> Southeastern Asset Mgmt        363 rows
    "MPGFX": "0001070134",  # Mairs & Power Growth    -> Mairs & Power Inc            5,912 rows
    "MVALX": "0001483859",  # Meridian Contrarian     -> ArrowMark Colorado Holdings  3,159 rows
    "TWEBX": "0000732905",  # Tweedy Browne Value     -> Tweedy, Browne Co LLC        1,211 rows
    "WVALX": "0000883965",  # Weitz Large Cap Equity  -> Weitz Investment Mgmt        1,283 rows
    "hcmax": "0001314620",  # Hillman Value Fund      -> Hillman Capital Management     953 rows
    "oaklx": "0000813917",  # Oakmark Select          -> Harris Associates L P        3,692 rows
    "pzfvx": "0001027796",  # Hancock Classic Value   -> Pzena Investment Mgmt        2,789 rows
    # --- operating companies / advisers EDGAR only matches on a shorter name --- #
    "CAS"  : "0001697591",  # CAS Investment Partners, LLC                              46 rows
    "FFH"  : "0000915191",  # Fairfax Financial Holdings Ltd/CAN                        512 rows
    "MAVFX": "0001016287",  # Matrix Asset Advisors Inc/NY                            2,728 rows
    "SA"   : "0001115373",  # Semper Augustus Investments Group LLC                     881 rows
    "oa"   : "0000885665",  # Leon Cooperman - Omega Advisors. The LOWER-case twin of "OA":
                            # Dataroma now serves the code as `oa` and the name as the bare
                            # person ("Leon Cooperman"), whose `_fund_part` is not a filer.
}

# The managers that are genuinely unresolvable, each with the reason. A row is still
# written (cik NULL, resolution='unresolved') so they stay visible and auditable: dropping
# them would shrink the eligible pool silently, which is the survivorship bug this table
# exists to remove, reintroduced through the back door. Any code that fails to resolve and
# is NOT listed here raises -- an unresolved manager must never pass quietly.
#
# Both entries are unresolvable for the SAME measured reason, and it is not a lookup failure:
# `SEC_EDGAR_COMPANY_SEARCH_URL` filters on `type=13F-HR`, so a company that never filed one
# returns an empty feed (HTTP 200, zero `<company-info>` blocks). Neither entity has a 13F
# filer identity at all, so no CIK would let them contribute to a 13F feature.
SUPERINVESTOR_UNRESOLVABLE: dict[str, str] = {
    "CMAFX": "Century Management / CM Advisers -- empty 13F-HR feed under 'Century "
             "Management Advisers', 'Century Management' and 'CM Advisers': never filed a "
             "13F-HR. On the roster 2013-2017.",
    "LUK": "Leucadia National, which became Jefferies Financial Group -- empty 13F-HR feed "
           "under both names: never filed a 13F-HR. On the roster 2013-2023.",
}


class SuperinvestorResolutionError(RuntimeError):
    """A roster manager resolved to no CIK and is not a recorded exception."""


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


def _edgar_cik_for_name(fund_name: str, get_fn) -> tuple[str | None, str | None]:
    """Resolve a fund name to its 13F-manager CIK via SEC EDGAR company search.
    Returns (cik, filer_name) or (None, None). `get_fn(url) -> response` is injected
    so tests can stub the network (production always passes a `context`-bound `sec_get`;
    see `upsert_roster_snapshot`).

    The search URL filters `type=13F-HR`, so an empty feed means "this name never filed a
    13F", not "no such company" -- which is why an unresolved manager is worth recording
    rather than chasing. A TRANSPORT failure is indistinguishable here from that empty feed,
    so it is logged at WARNING: `sec_get` does not retry, and the SEC returns 503s under
    load, which would otherwise be written into the table as a permanent NULL cik. The
    resolution gate is what stops that reaching the table."""
    q = _fund_part(fund_name)
    try:
        text = get_fn(SEC_EDGAR_COMPANY_SEARCH_URL.format(company=quote(q))).text
    except Exception as e:                                     # noqa: BLE001
        logger.warning("EDGAR lookup FAILED (not an empty result) for %r: %s", q, e)
        return None, None
    best = _pick_best_match(_parse_edgar_matches(text), q)
    return best if best else (None, None)


def snapshot_rows(roster: list[dict], snapshot_date, source_url: str,
                  resolver) -> list[dict]:
    """One `superinvestor_roster` row per roster entry, PURE given `resolver`.

    `resolver(code, name) -> (cik | None, resolution)` is the only impure part, so the row
    shape, the CIK padding and the unresolved handling are all testable without a network."""
    rows = []
    for entry in roster:
        code, name = entry["code"], entry["name"]
        cik, resolution = resolver(code, name)
        rows.append({
            "snapshot_date": snapshot_date,
            "dataroma_code": code,
            "manager_name": name,
            "cik": pad_cik(cik) or None,
            "resolution": resolution,
            "source_url": source_url,
        })
    return rows


def assert_fully_resolved(rows: list[dict]) -> list[str]:
    """RAISE unless every unresolved code is a recorded exception; return those codes.

    The gate D22 asks for: 100% resolution, or every exception named with its reason. An
    unresolved manager silently falls out of the eligible pool, so this fails loudly."""
    unresolved = sorted({r["dataroma_code"] for r in rows
                         if r["resolution"] == RESOLUTION_UNRESOLVED})
    unexpected = [c for c in unresolved if c not in SUPERINVESTOR_UNRESOLVABLE]
    if unexpected:
        names = {r["dataroma_code"]: r["manager_name"] for r in rows}
        raise SuperinvestorResolutionError(
            f"{len(unexpected)} roster manager(s) resolved to no CIK and are not recorded "
            "exceptions: "
            + ", ".join(f'"{c}" ({names[c]})' for c in unexpected)
            + ". Add the code -> CIK to SUPERINVESTOR_CIK_OVERRIDES, or record it in "
              "SUPERINVESTOR_UNRESOLVABLE with the reason it cannot be resolved.")
    return unresolved


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
# Resolution                                                                    #
# --------------------------------------------------------------------------- #
def _make_resolver(get_fn, name_history: dict[str, list[str]] | None = None,
                   known: dict[str, tuple[str, str]] | None = None):
    """`(code, name) -> (cik | None, resolution)`, memoised PER CODE.

    Memoised because the seed replays 879 manager-rows over 104 distinct codes: resolving
    per row would be an eight-fold EDGAR bill for the same answers, and the code -- not
    the name -- is the manager's identity across snapshots.

    Three sources, in precedence order, so a resolution never silently regresses:
      1. `SUPERINVESTOR_CIK_OVERRIDES` -- a hand mapping always wins, which is what lets an
         operator CORRECT a CIK the table already holds.
      2. `known` -- `{code: (cik, resolution)}` already stored for that code. Resolution is
         STICKY: Dataroma rewrites its display names constantly (52 of 104 codes were
         renamed at least once; `SEQUX` carries seven), and a rename must not turn a manager
         we have already identified back into an unresolved one.
      3. EDGAR, on the name in hand and then on the earlier names in `name_history`
         (NEWEST FIRST)."""
    cache: dict[str, tuple[str | None, str]] = {}

    def resolve(code: str, name: str) -> tuple[str | None, str]:
        if code in cache:
            return cache[code]
        if code in SUPERINVESTOR_CIK_OVERRIDES:
            out = (pad_cik(SUPERINVESTOR_CIK_OVERRIDES[code]), RESOLUTION_OVERRIDE)
        elif known and code in known:
            out = known[code]
        else:
            out = (None, RESOLUTION_UNRESOLVED)
            candidates = [name] + [n for n in (name_history or {}).get(code, [])
                                   if n != name]
            for candidate in candidates:
                cik, _filer = _edgar_cik_for_name(candidate, get_fn=get_fn)
                if cik:
                    out = (cik, RESOLUTION_EDGAR)
                    break
        cache[code] = out
        return out

    return resolve


def _stored_resolutions(context: Context) -> tuple[dict[str, tuple[str, str]],
                                                   dict[str, list[str]]]:
    """What `superinvestor_roster` already knows: `{code: (cik, resolution)}` for the codes
    that resolved, and `{code: [names, newest first]}`. Empty on a cold table."""
    df = context.store.load(Tables.superinvestor_roster,
                            columns=["snapshot_date", "dataroma_code", "manager_name",
                                     "cik", "resolution"], optional=True)
    if df is None or df.empty:
        return {}, {}
    df = df.sort_values("snapshot_date", ascending=False)
    known: dict[str, tuple[str, str]] = {}
    names: dict[str, list[str]] = {}
    for code, name, cik, resolution in df[
            ["dataroma_code", "manager_name", "cik", "resolution"]].itertuples(index=False):
        if (padded := pad_cik(cik)) and code not in known:
            known[code] = (padded, str(resolution))
        seen = names.setdefault(code, [])
        if (name := str(name)) not in seen:
            seen.append(name)
    return known, names


def _write(context: Context, rows: list[dict]) -> pd.DataFrame:
    """Upsert the rows and report the resolution split. Returns the written frame."""
    df = pd.DataFrame(rows)
    unresolved = assert_fully_resolved(rows)
    if unresolved:
        logger.warning(
            "Superinvestor roster: %d recorded-unresolvable manager(s) kept with a NULL "
            "cik -- %s", len(unresolved),
            "; ".join(f"{c}: {SUPERINVESTOR_UNRESOLVABLE[c]}" for c in unresolved))
    context.store.save(Tables.superinvestor_roster, df)
    logger.info("superinvestor_roster: wrote %d rows across %d snapshot(s); resolution %s",
                len(df), df["snapshot_date"].nunique(),
                df["resolution"].value_counts().to_dict())
    return df


# --------------------------------------------------------------------------- #
# Entry points                                                                  #
# --------------------------------------------------------------------------- #
def seed_roster_history(context: Context, get_fn=None) -> pd.DataFrame:
    """One-off: load the committed Wayback captures and write one row per
    (snapshot_date, dataroma_code).

    The capture file is keyed by YEAR, and the exact Wayback timestamps were not preserved,
    so each snapshot is dated **1 January of its year**. That is an approximation and the
    direction of its error is known: a manager Dataroma added mid-year reads as present from
    that January. It is the dating the downstream pool assumes (the 2016 roster of 64 is the
    eligible pool at 2016-06-30); a December dating would instead delay every addition by up
    to a year, which on a survivorship fix is the more damaging error."""
    get_fn = get_fn or (lambda url: sec_get(context, url))
    path = context.paths["DATA_STORE"] / _ROSTER_HISTORY_FILE
    history: dict[str, dict[str, str]] = json.loads(path.read_text(encoding="utf-8"))

    # newest name first, so a renamed code resolves on the name EDGAR is likeliest to know
    name_history: dict[str, list[str]] = {}
    for year in sorted(history, reverse=True):
        for code, name in history[year].items():
            names = name_history.setdefault(code, [])
            if name not in names:
                names.append(name)
    logger.info("Roster history: %d snapshots, %d manager-rows, %d distinct codes",
                len(history), sum(len(v) for v in history.values()), len(name_history))

    known, _ = _stored_resolutions(context)
    resolver = _make_resolver(get_fn, name_history, known)
    rows: list[dict] = []
    for year in sorted(history):
        roster = [{"code": c, "name": n} for c, n in history[year].items()]
        rows += snapshot_rows(roster, date(int(year), 1, 1),
                              _WAYBACK_URL.format(year=year), resolver)
    return _write(context, rows)


def upsert_roster_snapshot(context: Context, get_fn=None) -> pd.DataFrame:
    """Scrape Dataroma's roster today and write TODAY's snapshot into
    `superinvestor_roster`. Idempotent: re-running the same day upserts the same PK.

    CIKs come straight from EDGAR (or a code the table has already resolved -- see
    `_make_resolver`), so this does NOT depend on a local 13F cache. `get_fn` defaults to
    `sec_get` bound to `context` (which owns the SEC session / rate limiter); tests inject
    their own single-arg stub instead."""
    get_fn = get_fn or (lambda url: sec_get(context, url))
    roster = _parse_dataroma_roster(_http_get(DATAROMA_HOME_URL).text)
    logger.info("Dataroma: parsed %d superinvestors", len(roster))
    known, past_names = _stored_resolutions(context)
    rows = snapshot_rows(roster, datetime.now(timezone.utc).date(), DATAROMA_HOME_URL,
                         _make_resolver(get_fn, past_names, known))
    return _write(context, rows)
