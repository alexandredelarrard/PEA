"""
fetch_superinvestors.py (src/data_extract/utils/institutionals/fetch_superinvestors.py)
--------------------------------------------------------------------------------
WRITE side of `superinvestor_roster`: Dataroma's manager roster (names only), one row per
(snapshot_date, dataroma_code) so membership is point-in-time, with CIKs resolved via SEC EDGAR
company search. Hand resolutions live in configs/sec/superinvestor_overrides.json. Entry points:
`seed_roster_history` (committed Wayback captures) and `upsert_roster_snapshot` (today's roster).
The read side is `src/utils/superinvestor_roster.py`.
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from functools import cache, partial
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

# Resolution provenance, stored per row.
RESOLUTION_EDGAR = "edgar"
RESOLUTION_OVERRIDE = "override"
RESOLUTION_UNRESOLVED = "unresolved"

# Hand resolutions (CIK overrides and recorded-unresolvable codes) live in configs/sec/superinvestor_overrides.json.
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
    """The superinvestor hand resolutions, cached per resolved config directory."""
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
    """The fund part of a 'Person - Fund' name (after the last dash), else the whole string."""
    parts = re.split(r"\s[-–—]\s", str(dataroma_name))
    return parts[-1].strip() if len(parts) > 1 else str(dataroma_name).strip()


def _parse_dataroma_roster(html: str) -> list[dict]:
    """Dataroma home page -> [{code, name}] per `holdings.php?m=CODE` link, deduplicated by code in order."""
    soup = BeautifulSoup(html or "", "html.parser")
    out, seen = [], set()
    for a in soup.find_all("a", href=True):
        m = re.search(r"holdings\.php\?m=([A-Za-z0-9_.\-]+)", str(a["href"]))
        if not m:
            continue
        code = m.group(1)
        name = re.sub(r"\s+", " ", a.get_text(" ", strip=True)).strip()
        # Strip the trailing "Updated <date>" Dataroma appends to link text.
        name = re.sub(r"\s+Updated\b.*$", "", name, flags=re.IGNORECASE).strip()
        if code and code not in seen and name:
            seen.add(code)
            out.append({"code": code, "name": name})
    return out


def _parse_edgar_matches(atom_text: str) -> list[tuple[str, str]]:
    """(padded-cik, conformed-name) per `<company-info>` block of an EDGAR company-search feed;
    the name is '' when the block omits it."""
    out: list[tuple[str, str]] = []
    for block in re.split(r"<company-info", atom_text or "")[1:]:
        cik_m = re.search(r"<cik>(\d+)", block)
        if not cik_m:
            continue
        name_m = re.search(r"<conformed-name>([^<]*)", block)
        out.append((pad_cik(cik_m.group(1)), name_m.group(1).strip() if name_m else ""))
    return out


def _pick_best_match(pairs: list[tuple[str, str]], query: str) -> tuple[str, str] | None:
    """The pair whose filer name best token-matches `query`; a single match is trusted, ties go to EDGAR's first block."""
    if not pairs:
        return None
    if len(pairs) == 1:
        return pairs[0]
    qt = _name_tokens(query)
    idx = max(range(len(pairs)), key=lambda i: (len(qt & _name_tokens(pairs[i][1])), -i))
    return pairs[idx]


def _edgar_cik_for_name(fund_name: str, get_fn) -> tuple[str | None, str | None]:
    """Resolve a fund name to its 13F-HR filer CIK via EDGAR company search: (cik, filer_name) or
    (None, None). A transport failure also returns (None, None) but logs a WARNING; the
    resolution gate keeps it out of the table. `get_fn(url) -> response` is injected."""
    q = _fund_part(fund_name)
    try:
        text = get_fn(SEC_EDGAR_COMPANY_SEARCH_URL.format(company=quote(q))).text
    except Exception as e:  # noqa: BLE001
        logger.warning("EDGAR lookup FAILED (not an empty result) for %r: %s", q, e)
        return None, None
    best = _pick_best_match(_parse_edgar_matches(text), q)
    return best if best else (None, None)


def snapshot_rows(roster: list[dict], snapshot_date, source_url: str, resolver) -> list[dict]:
    """One `superinvestor_roster` row per roster entry; pure given `resolver(code, name) -> (cik | None, resolution)`."""
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
    """Raise `SuperinvestorResolutionError` unless every unresolved code is in `unresolvable`;
    return the unresolved codes."""
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
    """GET with SSL verification, retrying unverified (logged) on SSLError; the data is public and read-only."""
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
    """`(code, name) -> (cik | None, resolution)`, memoised per code (the code is the identity).
    Precedence: `cik_overrides`, then the stored `known` resolution (sticky across renames), then
    EDGAR on the current name and earlier `name_history` names, newest first."""
    resolved_by_code: dict[str, tuple[str | None, str]] = {}
    return partial(
        _resolve_code,
        resolved_by_code=resolved_by_code,
        get_fn=get_fn,
        cik_overrides=cik_overrides,
        name_history=name_history,
        known=known,
    )


def _resolve_code(
    code: str,
    name: str,
    *,
    resolved_by_code: dict[str, tuple[str | None, str]],
    get_fn,
    cik_overrides: Mapping[str, str],
    name_history: dict[str, list[str]] | None,
    known: dict[str, tuple[str, str]] | None,
) -> tuple[str | None, str]:
    """`(cik | None, resolution)` for one manager code, memoised in `resolved_by_code` (precedence as `_make_resolver`)."""
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
    """One-off: write the committed Wayback captures, one row per (snapshot_date, dataroma_code).
    The capture file is keyed by year, so each snapshot is dated 1 January of its year."""
    get_fn = get_fn or (lambda url: sec_get(context, url))
    path = context.paths["DATA_STORE"] / _ROSTER_HISTORY_FILE
    history: dict[str, dict[str, str]] = json.loads(path.read_text(encoding="utf-8"))

    # Newest name first, so a renamed code resolves on its most recent name.
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
    """Scrape Dataroma's roster and upsert today's snapshot (idempotent per day). CIKs resolve via
    `_make_resolver`; `get_fn` defaults to the context-bound, rate-limited `sec_get`."""
    get_fn = get_fn or (lambda url: sec_get(context, url))
    roster = _parse_dataroma_roster(_http_get(DATAROMA_HOME_URL).text)
    logger.info("Dataroma: parsed %d superinvestors", len(roster))
    known, past_names = _stored_resolutions(context)
    overrides = load_superinvestor_overrides(str(context.config_dir))
    resolver = _make_resolver(get_fn, overrides.cik_by_code, past_names, known)
    rows = snapshot_rows(roster, datetime.now(UTC).date(), DATAROMA_HOME_URL, resolver)
    return _write(context, rows, overrides.unresolvable)
