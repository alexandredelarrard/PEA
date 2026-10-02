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
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from functools import cache
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib3.exceptions import InsecureRequestWarning

from src.constants.constants import BROWSER_HEADERS, SEC_EDGAR_COMPANY_SEARCH_URL
from src.context import Context
from src.data_extract.utils.common.config_paths import resolve_config_dir
from src.data_extract.utils.common.sec_utils import sec_get
from src.data_store.schema import Tables
from src.utils.string import pad_cik

logger = logging.getLogger(__name__)

# manager-name tokens that carry no matching signal (legal / entity boilerplate)
_STOP_TOKENS = {
    "LP",
    "LLP",
    "LLC",
    "INC",
    "INCORPORATED",
    "CORP",
    "CORPORATION",
    "CO",
    "LTD",
    "LIMITED",
    "CAPITAL",
    "MANAGEMENT",
    "MGMT",
    "MGT",
    "PARTNERS",
    "PARTNER",
    "GROUP",
    "ADVISORS",
    "ADVISERS",
    "ADVISORY",
    "ASSET",
    "ASSETS",
    "FUND",
    "FUNDS",
    "HOLDINGS",
    "HOLDING",
    "INVESTMENT",
    "INVESTMENTS",
    "INTERNATIONAL",
    "GLOBAL",
    "AND",
    "THE",
    "COMPANY",
    "MASTER",
    "SECURITIES",
    "TRUST",
    "FINANCIAL",
    "RESEARCH",
    "SERVICES",
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

# Hand resolutions (code -> CIK overrides and the recorded-unresolvable codes with their
# reasons) live in configs/sec/superinvestor_overrides.json.
OVERRIDES_CONFIG_SUBDIR = "sec"
OVERRIDES_CONFIG_FILENAME = "superinvestor_overrides.json"


@dataclass(frozen=True)
class SuperinvestorOverrides:
    """`cik_by_code` wins over any stored or EDGAR resolution; `unresolvable` names the codes
    allowed to stay NULL, each with its reason."""

    cik_by_code: dict[str, str]
    unresolvable: dict[str, str]


class SuperinvestorResolutionError(RuntimeError):
    """A roster manager resolved to no CIK and is not a recorded exception."""


def load_superinvestor_overrides(config_dir: str | None = None) -> SuperinvestorOverrides:
    """The superinvestor hand resolutions, cached per config DIRECTORY rather than per
    spelling of it -- see `resolve_config_dir`."""
    return _overrides_at(resolve_config_dir(config_dir))


@cache
def _overrides_at(config_dir: str) -> SuperinvestorOverrides:
    """`load_superinvestor_overrides`, keyed on a resolved absolute path. Raises when a CIK is
    blank or a code is both overridden and recorded unresolvable."""
    path = Path(config_dir) / OVERRIDES_CONFIG_SUBDIR / OVERRIDES_CONFIG_FILENAME
    blob = json.loads(path.read_text(encoding="utf-8"))
    cik_by_code = {code: pad_cik(entry["cik"]) for code, entry in blob["cik_overrides"].items()}
    unresolvable = {code: str(reason) for code, reason in blob["unresolvable"].items()}
    blank = sorted(code for code, cik in cik_by_code.items() if not cik)
    both = sorted(set(cik_by_code) & set(unresolvable))
    if blank or both:
        raise ValueError(f"{path}: blank CIK for {blank}; both overridden and unresolvable: {both}")
    return SuperinvestorOverrides(cik_by_code=cik_by_code, unresolvable=unresolvable)


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
        m = re.search(r"holdings\.php\?m=([A-Za-z0-9_.\-]+)", str(a["href"]))
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
    idx = max(range(len(pairs)), key=lambda i: (len(qt & _name_tokens(pairs[i][1])), -i))
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
    except Exception as e:  # noqa: BLE001
        logger.warning("EDGAR lookup FAILED (not an empty result) for %r: %s", q, e)
        return None, None
    best = _pick_best_match(_parse_edgar_matches(text), q)
    return best if best else (None, None)


def snapshot_rows(roster: list[dict], snapshot_date, source_url: str, resolver) -> list[dict]:
    """One `superinvestor_roster` row per roster entry, PURE given `resolver`.

    `resolver(code, name) -> (cik | None, resolution)` is the only impure part, so the row
    shape, the CIK padding and the unresolved handling are all testable without a network."""
    rows = []
    for entry in roster:
        code, name = entry["code"], entry["name"]
        cik, resolution = resolver(code, name)
        rows.append(
            {
                "snapshot_date": snapshot_date,
                "dataroma_code": code,
                "manager_name": name,
                "cik": pad_cik(cik) or None,
                "resolution": resolution,
                "source_url": source_url,
            }
        )
    return rows


def assert_fully_resolved(rows: list[dict], unresolvable: Mapping[str, str]) -> list[str]:
    """RAISE unless every unresolved code is in `unresolvable`; return those codes.

    The gate D22 asks for: 100% resolution, or every exception named with its reason. An
    unresolved manager silently falls out of the eligible pool, so this fails loudly."""
    unresolved = sorted({r["dataroma_code"] for r in rows if r["resolution"] == RESOLUTION_UNRESOLVED})
    unexpected = [c for c in unresolved if c not in unresolvable]
    if unexpected:
        names = {r["dataroma_code"]: r["manager_name"] for r in rows}
        raise SuperinvestorResolutionError(
            f"{len(unexpected)} roster manager(s) resolved to no CIK and are not recorded "
            "exceptions: "
            + ", ".join(f'"{c}" ({names[c]})' for c in unexpected)
            + f". Add the code -> CIK under `cik_overrides` in {OVERRIDES_CONFIG_SUBDIR}/"
            f"{OVERRIDES_CONFIG_FILENAME}, or record it under `unresolvable` with the reason "
            "it cannot be resolved."
        )
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
        r = requests.get(url, headers=BROWSER_HEADERS, timeout=60)
    except requests.exceptions.SSLError:
        logger.warning("Dataroma SSL chain incomplete -> retrying unverified (%s)", url)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InsecureRequestWarning)
            r = requests.get(url, headers=BROWSER_HEADERS, timeout=60, verify=False)
    r.raise_for_status()
    return r


# --------------------------------------------------------------------------- #
# Resolution                                                                    #
# --------------------------------------------------------------------------- #
def _edgar_resolution(candidates: list[str], get_fn) -> tuple[str | None, str]:
    """The first candidate name EDGAR resolves, as `(cik, RESOLUTION_EDGAR)`; unresolved otherwise."""
    for candidate in candidates:
        cik, _filer = _edgar_cik_for_name(candidate, get_fn=get_fn)
        if cik:
            return cik, RESOLUTION_EDGAR
    return None, RESOLUTION_UNRESOLVED


def _make_resolver(
    get_fn,
    cik_overrides: Mapping[str, str],
    name_history: dict[str, list[str]] | None = None,
    known: dict[str, tuple[str, str]] | None = None,
):
    """`(code, name) -> (cik | None, resolution)`, memoised PER CODE.

    Memoised because the seed replays 879 manager-rows over 104 distinct codes: resolving
    per row would be an eight-fold EDGAR bill for the same answers, and the code -- not
    the name -- is the manager's identity across snapshots.

    Three sources, in precedence order, so a resolution never silently regresses:
      1. `cik_overrides` -- a hand mapping always wins, which is what lets an
         operator CORRECT a CIK the table already holds.
      2. `known` -- `{code: (cik, resolution)}` already stored for that code. Resolution is
         STICKY: Dataroma rewrites its display names constantly (52 of 104 codes were
         renamed at least once; `SEQUX` carries seven), and a rename must not turn a manager
         we have already identified back into an unresolved one.
      3. EDGAR, on the name in hand and then on the earlier names in `name_history`
         (NEWEST FIRST)."""
    resolved_by_code: dict[str, tuple[str | None, str]] = {}

    def resolve(code: str, name: str) -> tuple[str | None, str]:
        if code in resolved_by_code:
            return resolved_by_code[code]
        if code in cik_overrides:
            out = (cik_overrides[code], RESOLUTION_OVERRIDE)
        elif known and code in known:
            out = known[code]
        else:
            earlier_names = [n for n in (name_history or {}).get(code, []) if n != name]
            out = _edgar_resolution([name, *earlier_names], get_fn)
        resolved_by_code[code] = out
        return out

    return resolve


def _stored_resolutions(context: Context) -> tuple[dict[str, tuple[str, str]], dict[str, list[str]]]:
    """What `superinvestor_roster` already knows: `{code: (cik, resolution)}` for the codes
    that resolved, and `{code: [names, newest first]}`. Empty on a cold table."""
    df = context.store.load(
        Tables.superinvestor_roster, columns=["snapshot_date", "dataroma_code", "manager_name", "cik", "resolution"], optional=True
    )
    if df is None or df.empty:
        return {}, {}
    df = df.sort_values("snapshot_date", ascending=False)
    known: dict[str, tuple[str, str]] = {}
    names: dict[str, list[str]] = {}
    for code, name, cik, resolution in df[["dataroma_code", "manager_name", "cik", "resolution"]].itertuples(index=False):
        if (padded := pad_cik(cik)) and code not in known:
            known[code] = (padded, str(resolution))
        seen = names.setdefault(code, [])
        if (name := str(name)) not in seen:
            seen.append(name)
    return known, names


def _write(context: Context, rows: list[dict], unresolvable: Mapping[str, str]) -> pd.DataFrame:
    """Upsert the rows and report the resolution split. Returns the written frame."""
    df = pd.DataFrame(rows)
    unresolved = assert_fully_resolved(rows, unresolvable)
    if unresolved:
        logger.warning(
            "Superinvestor roster: %d recorded-unresolvable manager(s) kept with a NULL cik -- %s",
            len(unresolved),
            "; ".join(f"{c}: {unresolvable[c]}" for c in unresolved),
        )
    context.store.save(Tables.superinvestor_roster, df)
    logger.info(
        "superinvestor_roster: wrote %d rows across %d snapshot(s); resolution %s",
        len(df),
        df["snapshot_date"].nunique(),
        df["resolution"].value_counts().to_dict(),
    )
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
    logger.info(
        "Roster history: %d snapshots, %d manager-rows, %d distinct codes", len(history), sum(len(v) for v in history.values()), len(name_history)
    )

    known, _ = _stored_resolutions(context)
    overrides = load_superinvestor_overrides(str(context.config_dir))
    resolver = _make_resolver(get_fn, overrides.cik_by_code, name_history, known)
    rows: list[dict] = []
    for year in sorted(history):
        roster = [{"code": c, "name": n} for c, n in history[year].items()]
        rows += snapshot_rows(roster, date(int(year), 1, 1), _WAYBACK_URL.format(year=year), resolver)
    return _write(context, rows, overrides.unresolvable)


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
    overrides = load_superinvestor_overrides(str(context.config_dir))
    resolver = _make_resolver(get_fn, overrides.cik_by_code, past_names, known)
    rows = snapshot_rows(roster, datetime.now(UTC).date(), DATAROMA_HOME_URL, resolver)
    return _write(context, rows, overrides.unresolvable)
