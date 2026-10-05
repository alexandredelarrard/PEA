"""
fetch_superinvestors.py (src/data_extract/utils/institutionals/fetch_superinvestors.py)
--------------------------------------------------------------------------------
WRITE side of `superinvestor_roster`: Dataroma's manager roster (names only), one row per
(snapshot_date, dataroma_code) so membership is point-in-time, with CIKs resolved via SEC EDGAR
company search. Hand resolutions and the committed history live in configs/superinvestors/ and are
loaded by `src/utils/superinvestor_roster.py`, which is also the read side; a chained manager's row
stores the filer CIK valid at `snapshot_quarter(snapshot_date)`. Entry points:
`rebuild_roster` (the committed Wayback history plus every stored live snapshot, all resolved fresh, upserted,
then the stale keys deleted) and `upsert_roster_snapshot` (today's roster, written only when its code -> CIK mapping
differs from the latest stored snapshot).
Every write passes two gates: `assert_fully_resolved` (no unexpected NULL CIK) and `assert_active`
(each CIK filed a 13F-HR near its snapshot, from local tables, else the EDGAR listing).
The committed history is built by `quarterly_capture_candidates` + `history_from_captures`, driven by
`scripts/build_dataroma_roster_history.py`.
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, date, datetime
from functools import partial
from typing import Any, NamedTuple, cast
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup
from edgar import Company
from urllib3.exceptions import InsecureRequestWarning

from src.constants.constants import BROWSER_HEADERS, SEC_13F_FORMS, SEC_EDGAR_COMPANY_SEARCH_URL
from src.context import Context
from src.data_extract.utils.common.sec_utils import sec_get
from src.data_store.schema import Tables
from src.utils.string import pad_cik, pad_cik_series
from src.utils.superinvestor_roster import (
    OVERRIDES_CONFIG_FILENAME,
    SUPERINVESTORS_CONFIG_SUBDIR,
    InactiveRange,
    PeriodRange,
    SuperinvestorOverrides,
    config_dir_of,
    load_superinvestor_overrides,
    roster_history_path,
)

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
# Every Wayback capture of the Dataroma home page, uncollapsed: one `timestamp status original` line each.
WAYBACK_CDX_URL = "https://web.archive.org/cdx/search/cdx?url=dataroma.com/m/home.php&fl=timestamp,statuscode,original"
# A capture's archived bytes as served (`id_`: no Wayback rewrite).
_WAYBACK_RAW_URL = "https://web.archive.org/web/{timestamp}id_/{original}"
# First day of the committed roster history.
ROSTER_HISTORY_START = date(2012, 1, 1)
# A capture is valid only if it lists at least this share of the previous accepted snapshot's managers.
ROSTER_CAPTURE_MIN_RATIO = 0.8
ROSTER_HISTORY_README = [
    "Dataroma superinvestor roster history: one snapshot per calendar quarter, the newest valid Wayback capture of "
    "dataroma.com/m/home.php in that quarter.",
    "captured_at is the capture instant (UTC); `data_extract superinvestors --seed` dates the snapshot at its day. "
    "source_url is the raw (id_) capture; managers maps Dataroma code -> name in page order.",
    f"A capture is valid when it lists at least one manager and at least {ROSTER_CAPTURE_MIN_RATIO:.0%} of the previous "
    "snapshot's count; a quarter with no valid capture is absent (a logged gap).",
    "Generated file, do not hand-edit: scripts/build_dataroma_roster_history.py --cache-dir DIR --until YYYY-MM-DD.",
]

# Resolution provenance, stored per row.
RESOLUTION_EDGAR = "edgar"
RESOLUTION_OVERRIDE = "override"
RESOLUTION_UNRESOLVED = "unresolved"

# Activity gate window around Q(snapshot), in quarters.
GATE_QUARTERS_BEFORE = 4
GATE_QUARTERS_AFTER = 2
# First quarter with broad XML 13F coverage in the local tables; a pre-XML window extends to it.
SEC13F_XML_ERA_START = date(2013, 6, 30)

#: (padded cik, period) pairs of 13F activity.
Evidence = set[tuple[str, date]]
#: CIK -> report periods of its EDGAR-listed 13F-HR filings.
ListingFn = Callable[[str], set[date]]


class SuperinvestorResolutionError(RuntimeError):
    """A roster write is refused: an empty or partial scrape, or a manager with no CIK or no 13F activity near its
    snapshot that is not a recorded exception."""


class EdgarListingError(RuntimeError):
    """The EDGAR 13F-HR listing of a CIK could not be read, so its activity is unknown (not evidence of a wrong CIK)."""


class WaybackCapture(NamedTuple):
    """One Wayback capture of the Dataroma home page: its UTC `YYYYMMDDhhmmss` timestamp and archived URL."""

    timestamp: str
    original: str

    @property
    def raw_url(self) -> str:
        """The capture's unmodified archived bytes."""
        return _WAYBACK_RAW_URL.format(timestamp=self.timestamp, original=self.original)

    @property
    def captured_at(self) -> str:
        """The capture instant as `YYYY-MM-DDTHH:MM:SSZ`."""
        return datetime.strptime(self.timestamp, "%Y%m%d%H%M%S").strftime("%Y-%m-%dT%H:%M:%SZ")


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


def snapshot_quarter(snapshot_date: date | pd.Timestamp | str) -> date:
    """Q(snapshot): the last quarter end strictly before the date, i.e. the book most recently filed at it."""
    return (pd.Timestamp(snapshot_date).normalize() - pd.offsets.QuarterEnd()).date()


def snapshot_rows(roster: list[dict], snapshot_date, source_url: str, resolver, *, overrides: SuperinvestorOverrides) -> list[dict]:
    """One `superinvestor_roster` row per roster entry; pure given `resolver(code, name) -> (cik | None, resolution)`.
    A resolved CIK is stored as the member of its manager's chain valid at `snapshot_quarter(snapshot_date)`."""
    quarter = snapshot_quarter(snapshot_date)
    rows = []
    for entry in roster:
        code, name = entry["code"], entry["name"]
        cik, resolution = resolver(code, name)
        rows.append(
            {
                "snapshot_date": snapshot_date,
                "dataroma_code": code,
                "manager_name": name,
                "cik": overrides.member_at(cik, quarter) if pad_cik(cik) else None,
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
            + f". Add the code -> CIK under `cik_overrides` in {SUPERINVESTORS_CONFIG_SUBDIR}/"
            f"{OVERRIDES_CONFIG_FILENAME}, or record it under `unresolvable` with the reason "
            "it cannot be resolved."
        )
    return unresolved


def activity_window(snapshot_date: date | pd.Timestamp | str) -> PeriodRange:
    """The gate window [Q - GATE_QUARTERS_BEFORE, Q + GATE_QUARTERS_AFTER] around Q = `snapshot_quarter`. The local
    tables are sparse before `SEC13F_XML_ERA_START`, so a window ending earlier extends to that quarter."""
    q = pd.Timestamp(snapshot_quarter(snapshot_date))
    start = (q - pd.offsets.QuarterEnd(GATE_QUARTERS_BEFORE)).date()
    end = (q + pd.offsets.QuarterEnd(GATE_QUARTERS_AFTER)).date()
    return PeriodRange(start, max(end, SEC13F_XML_ERA_START))


def _nearest(periods: Iterable[date], target: date) -> str:
    """The period closest to `target` as ISO text, 'none' when there is none."""
    best = min(periods, key=lambda p: abs((p - target).days), default=None)
    return best.isoformat() if best else "none"


def assert_active(rows: list[dict], evidence: Evidence, inactive: Mapping[str, InactiveRange], listing_fn: ListingFn) -> list[str]:
    """Raise `SuperinvestorResolutionError` unless every row with a CIK shows 13F activity in its `activity_window`,
    from `evidence` or, on a local miss only, from `listing_fn(cik)` (called once per CIK). A row whose code has an
    `inactive` range covering Q(snapshot) is skipped; a `listing_fn` failure propagates. Returns the skipped codes."""
    by_cik: dict[str, set[date]] = {}
    for cik, period in evidence:
        by_cik.setdefault(cik, set()).add(period)
    listed: dict[str, set[date]] = {}
    skipped: set[str] = set()
    failures: dict[tuple[str, str], list[date]] = {}
    for row in rows:
        if not (cik := pad_cik(row["cik"])):
            continue
        code, snapshot = row["dataroma_code"], pd.Timestamp(row["snapshot_date"]).date()
        if code in inactive and inactive[code].window.covers(snapshot_quarter(snapshot)):
            skipped.add(code)
            continue
        window = activity_window(snapshot)
        if any(window.covers(p) for p in by_cik.get(cik, ())):
            continue
        if cik not in listed:
            listed[cik] = listing_fn(cik)
        if not any(window.covers(p) for p in listed[cik]):
            failures.setdefault((code, cik), []).append(snapshot)
    if failures:
        lines = []
        for (code, cik), snapshots in sorted(failures.items()):
            first, last = min(snapshots), max(snapshots)
            known = by_cik.get(cik, set()) | listed.get(cik, set())
            lines.append(
                f'"{code}" CIK {cik}: {len(snapshots)} snapshot(s) {first}..{last}, nearest known 13F period '
                f"{_nearest(known, snapshot_quarter(first))} (window at {first}: {activity_window(first).start}..{activity_window(first).end})"
            )
        raise SuperinvestorResolutionError(
            f"{len(failures)} roster code/CIK pair(s) have no 13F-HR activity near their snapshot and are not recorded "
            "inactive: " + "; ".join(lines) + f". Correct the CIK under `cik_overrides` in {SUPERINVESTORS_CONFIG_SUBDIR}/"
            f"{OVERRIDES_CONFIG_FILENAME} with its 13F evidence, or record the code under `inactive` with a dated range and reason."
        )
    return sorted(skipped)


def quarterly_capture_candidates(cdx: str, since: date = ROSTER_HISTORY_START, until: date | None = None) -> dict[str, list[WaybackCapture]]:
    """`{"YYYYQn": [capture, newest first]}` in chronological quarter order, over the status-200 lines of a
    `timestamp status original` CDX captured on days in [since, until]; a repeated timestamp counts once."""
    by_quarter: dict[str, list[WaybackCapture]] = {}
    seen: set[str] = set()
    for line in filter(None, (raw.strip() for raw in cdx.splitlines())):
        timestamp, status, original = line.split()
        day = datetime.strptime(timestamp[:8], "%Y%m%d").date()
        if status != "200" or timestamp in seen or day < since or (until is not None and day > until):
            continue
        seen.add(timestamp)
        by_quarter.setdefault(str(pd.Period(day, "Q")), []).append(WaybackCapture(timestamp, original))
    return {quarter: sorted(caps, key=lambda c: c.timestamp, reverse=True) for quarter, caps in sorted(by_quarter.items())}


def _first_valid_capture(
    quarter: str, captures: list[WaybackCapture], parse: Callable[[WaybackCapture], dict[str, str]], floor: float
) -> tuple[WaybackCapture, dict[str, str]] | None:
    """The first capture whose roster is non-empty and holds at least `floor` managers; each rejection is logged."""
    for capture in captures:
        managers = parse(capture)
        if managers and len(managers) >= floor:
            return capture, managers
        logger.warning("Roster history %s: rejected capture %s (%d managers, floor %.1f)", quarter, capture.timestamp, len(managers), floor)
    return None


def history_from_captures(candidates: Mapping[str, list[WaybackCapture]], parse: Callable[[WaybackCapture], dict[str, str]]) -> dict[str, Any]:
    """The committed roster history document: per quarter (chronological), the newest capture whose
    `parse(capture) -> {code: name}` is valid (see `ROSTER_CAPTURE_MIN_RATIO`); a quarter without one is a logged gap."""
    snapshots: list[dict[str, Any]] = []
    for quarter in sorted(candidates):
        floor = ROSTER_CAPTURE_MIN_RATIO * len(snapshots[-1]["managers"]) if snapshots else 0.0
        accepted = _first_valid_capture(quarter, candidates[quarter], parse, floor)
        if accepted is None:
            logger.warning("Roster history %s: no valid capture among %d -- gap", quarter, len(candidates[quarter]))
            continue
        capture, managers = accepted
        snapshots.append({"captured_at": capture.captured_at, "source_url": capture.raw_url, "managers": managers})
    logger.info("Roster history: %d snapshots from %d quarters with captures", len(snapshots), len(candidates))
    return {"_README": list(ROSTER_HISTORY_README), "snapshots": snapshots}


# --------------------------------------------------------------------------- #
# IO: Dataroma fetch (its cert chain is incomplete -> verified-then-relaxed)     #
# --------------------------------------------------------------------------- #
def _http_get(url: str, headers: Mapping[str, str] | None = None) -> requests.Response:
    """GET (default `BROWSER_HEADERS`) with SSL verification, retrying unverified (logged) on SSLError;
    the data is public and read-only."""
    headers = dict(headers or BROWSER_HEADERS)
    try:
        r = requests.get(url, headers=headers, timeout=60)
    except requests.exceptions.SSLError:
        logger.warning("SSL verification failed -> retrying unverified (%s)", url)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InsecureRequestWarning)
            r = requests.get(url, headers=headers, timeout=60, verify=False)
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


def _names_newest_first(pairs: Iterable[tuple[str, str]]) -> dict[str, list[str]]:
    """`{code: [distinct names in input order]}` over newest-first `(code, name)` pairs."""
    names: dict[str, list[str]] = {}
    for code, name in pairs:
        seen = names.setdefault(code, [])
        if name not in seen:
            seen.append(name)
    return names


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
    for code, cik, resolution in df[["dataroma_code", "cik", "resolution"]].itertuples(index=False):
        if (padded := pad_cik(cik)) and code not in known:
            known[code] = (padded, str(resolution))
    return known, _names_newest_first(zip(df["dataroma_code"], df["manager_name"].astype(str), strict=True))


def activity_evidence(context: Context, ciks: Iterable[str]) -> Evidence:
    """(padded cik, period) 13F activity of `ciks` from `sec13f_hr` and `sec13f_manager_holdings`: projected,
    CIK-scoped reads that match both padded and unpadded stored CIKs."""
    padded = sorted({p for c in ciks if (p := pad_cik(c))})
    if not padded:
        return set()
    forms = padded + [c.lstrip("0") for c in padded]
    out: Evidence = set()
    for table in (Tables.sec13f_hr, Tables.sec13f_manager_holdings):
        for chunk in context.store.iter_load(table, columns=["cik", "period"], where={"cik": forms}):
            pairs = chunk.drop_duplicates()
            periods = pd.to_datetime(pairs["period"], errors="coerce")
            out |= {(c, p.date()) for c, p in zip(pad_cik_series(pairs["cik"]), periods, strict=True) if c and pd.notna(p)}
    return out


def edgar_13f_report_dates(context: Context, cik: str) -> set[date]:
    """Report periods of every 13F-HR(/A) EDGAR lists for one CIK (the listing's `reportDate`); empty when it lists
    none. A failed listing raises `EdgarListingError` naming the CIK."""
    context.ensure_edgar_identity()
    try:
        listing = Company(int(cik)).get_filings(form=SEC_13F_FORMS)
        frame = listing.to_pandas() if listing else pd.DataFrame(columns=["reportDate"])
    except Exception as e:
        raise EdgarListingError(
            f"EDGAR 13F-HR listing failed for CIK {cik} ({type(e).__name__}: {e}); its 13F activity is unknown, so nothing "
            "was written. This is a listing/network failure, not evidence of a wrong CIK: rerun once EDGAR responds."
        ) from e
    periods = pd.to_datetime(frame["reportDate"], errors="coerce").dropna()
    return {p.date() for p in periods}


def _gate(context: Context, rows: list[dict], overrides: SuperinvestorOverrides, listing_fn: ListingFn | None = None) -> None:
    """Run both write gates on `rows` (resolution, then 13F activity) and log the recorded exceptions they let
    through. `listing_fn` defaults to `edgar_13f_report_dates`, asked only on a local miss."""
    unresolvable = overrides.unresolvable
    unresolved = assert_fully_resolved(rows, unresolvable)
    if unresolved:
        logger.warning(
            "Superinvestor roster: %d recorded-unresolvable manager(s) kept with a NULL cik -- %s",
            len(unresolved),
            "; ".join(f"{c}: {unresolvable[c]}" for c in unresolved),
        )
    evidence = activity_evidence(context, {r["cik"] for r in rows if r["cik"]})
    skipped = assert_active(rows, evidence, overrides.inactive, listing_fn or partial(edgar_13f_report_dates, context))
    if skipped:
        logger.warning(
            "Superinvestor roster: %d recorded-inactive manager(s) written without 13F activity -- %s",
            len(skipped),
            "; ".join(f"{c}: {overrides.inactive[c].reason}" for c in skipped),
        )


def _write(context: Context, rows: list[dict], overrides: SuperinvestorOverrides, listing_fn: ListingFn | None = None) -> pd.DataFrame:
    """Gate the rows (`_gate`), upsert them sorted by primary key and log the resolution split. Returns the written frame."""
    _gate(context, rows, overrides, listing_fn)
    df = pd.DataFrame(rows).sort_values(["snapshot_date", "dataroma_code"], ignore_index=True)
    context.store.save(Tables.superinvestor_roster, df)
    logger.info(
        "superinvestor_roster: wrote %d rows across %d snapshot(s); resolution %s",
        len(df),
        df["snapshot_date"].nunique(),
        df["resolution"].value_counts().to_dict(),
    )
    return df


def _live_snapshots(context: Context) -> list[tuple[date, list[dict]]]:
    """Every stored live snapshot (`source_url == DATAROMA_HOME_URL`) as `(snapshot_date, [{code, name}])`, oldest
    first, codes sorted; empty on a cold table."""
    df = context.store.load(
        Tables.superinvestor_roster,
        columns=["snapshot_date", "dataroma_code", "manager_name"],
        where={"source_url": DATAROMA_HOME_URL},
        optional=True,
    )
    if df is None or df.empty:
        return []
    df = df.assign(snapshot_date=pd.to_datetime(df["snapshot_date"]).dt.date).sort_values(["snapshot_date", "dataroma_code"])
    return [
        (cast(date, day), [{"code": c, "name": n} for c, n in grp[["dataroma_code", "manager_name"]].itertuples(index=False)])
        for day, grp in df.groupby("snapshot_date", sort=True)
    ]


def rebuild_rows(context: Context, get_fn, overrides: SuperinvestorOverrides) -> list[dict]:
    """The rows a full rebuild writes: every committed history snapshot (dated at its capture day, its capture URL
    as `source_url`) plus every stored live snapshot, none collapsed, all resolved fresh (overrides, then EDGAR on
    the code's names newest first; stored resolutions are ignored). Reads only, writes nothing."""
    history = json.loads(roster_history_path(config_dir_of(context)).read_text(encoding="utf-8"))
    snapshots: list[tuple[date, str, list[dict]]] = [
        (datetime.fromisoformat(s["captured_at"]).date(), s["source_url"], [{"code": c, "name": n} for c, n in s["managers"].items()])
        for s in sorted(history["snapshots"], key=lambda s: s["captured_at"])
    ]
    live = _live_snapshots(context)
    snapshots += [(day, DATAROMA_HOME_URL, roster) for day, roster in live]

    # Newest name first, so a renamed code resolves on its most recent name.
    name_history = _names_newest_first(
        (e["code"], e["name"]) for _day, _url, roster in sorted(snapshots, key=lambda s: s[0], reverse=True) for e in roster
    )
    logger.info(
        "Roster rebuild: %d history + %d live snapshots, %d manager-rows, %d distinct codes",
        len(history["snapshots"]),
        len(live),
        sum(len(roster) for _day, _url, roster in snapshots),
        len(name_history),
    )
    resolver = _make_resolver(get_fn, overrides.cik_by_code, name_history)
    rows: list[dict] = []
    for day, url, roster in snapshots:
        rows += snapshot_rows(roster, day, url, resolver, overrides=overrides)
    return rows


def _assert_unique_pk(rows: list[dict]) -> None:
    """Raise `ValueError` when two rows share a (snapshot_date, dataroma_code) primary key."""
    keys = Counter((str(r["snapshot_date"]), r["dataroma_code"]) for r in rows)
    duplicate = sorted(k for k, n in keys.items() if n > 1)
    if duplicate:
        raise ValueError(f"superinvestor_roster rebuild: {len(duplicate)} duplicate (snapshot_date, dataroma_code) key(s), e.g. {duplicate[:5]}")


# --------------------------------------------------------------------------- #
# Entry points                                                                  #
# --------------------------------------------------------------------------- #
def _delete_stale_keys(context: Context, df: pd.DataFrame) -> int:
    """Delete the stored (snapshot_date, dataroma_code) keys absent from `df`, one targeted delete per snapshot date.
    Returns the rows deleted."""
    table = Tables.superinvestor_roster
    stored = cast(pd.DataFrame, context.store.load(table, columns=["snapshot_date", "dataroma_code"]))
    keep = set(zip(pd.to_datetime(df["snapshot_date"]).dt.date, df["dataroma_code"], strict=True))
    stored_days = pd.to_datetime(stored["snapshot_date"]).dt.date
    stale = stored.assign(snapshot_date=stored_days)[[k not in keep for k in zip(stored_days, stored["dataroma_code"], strict=True)]]
    return sum(
        context.store.delete(table, where={"snapshot_date": day, "dataroma_code": sorted(codes)})
        for day, codes in stale.groupby("snapshot_date", sort=True)["dataroma_code"]
    )


def rebuild_roster(context: Context, get_fn=None, listing_fn: ListingFn | None = None) -> pd.DataFrame:
    """Rebuild `superinvestor_roster` from scratch: `rebuild_rows`, primary-key uniqueness, then `_write` (both gates,
    upsert of the rebuilt frame) and only then the delete of the stored keys it lacks. The live snapshots are read
    from this same table, so it is never emptied: a crash leaves the old table or a superset, and a rerun converges.
    A resolution, primary-key or gate failure raises before the table is touched. Returns the written frame."""
    get_fn = get_fn or (lambda url: sec_get(context, url))
    overrides = load_superinvestor_overrides(config_dir_of(context))
    rows = rebuild_rows(context, get_fn, overrides)
    _assert_unique_pk(rows)
    df = _write(context, rows, overrides, listing_fn)
    deleted = _delete_stale_keys(context, df)
    logger.info("superinvestor_roster: deleted %d stale row(s) absent from the rebuild", deleted)
    return df


def _code_to_cik(codes: Iterable[object], ciks: Iterable[object]) -> dict[str, str | None]:
    """`{dataroma_code: padded cik or None}`, the identity a live snapshot is compared on."""
    return {str(code): pad_cik(cik) or None for code, cik in zip(codes, ciks, strict=True)}


def _latest_mapping(context: Context) -> dict[str, str | None] | None:
    """The code -> CIK mapping of the latest stored snapshot; None on a cold table."""
    latest = context.store.max_date(Tables.superinvestor_roster, "snapshot_date")
    if latest is None:
        return None
    df = context.store.load(Tables.superinvestor_roster, columns=["dataroma_code", "cik"], where={"snapshot_date": latest.date()}, optional=True)
    return None if df is None else _code_to_cik(df["dataroma_code"], df["cik"])


def _assert_complete_scrape(n_parsed: int, latest: Mapping[str, str | None] | None) -> None:
    """Raise `SuperinvestorResolutionError` when the scrape lists no manager or, with a stored snapshot, fewer than
    `ROSTER_CAPTURE_MIN_RATIO` of its managers."""
    n_stored = len(latest or {})
    if n_parsed == 0 or n_parsed < ROSTER_CAPTURE_MIN_RATIO * n_stored:
        raise SuperinvestorResolutionError(
            f"Dataroma roster scrape of {DATAROMA_HOME_URL} parsed {n_parsed} managers against {n_stored} in the latest stored "
            f"snapshot; a scrape must list at least one manager and {ROSTER_CAPTURE_MIN_RATIO:.0%} of that count. The page "
            "or its parse is broken, so nothing was written."
        )


def upsert_roster_snapshot(context: Context, get_fn=None, listing_fn: ListingFn | None = None) -> pd.DataFrame:
    """Scrape Dataroma's roster and write it as today's snapshot, gated as `_write`, only when its code -> CIK
    mapping differs from the latest stored snapshot; otherwise write nothing and return an empty frame. An empty or
    partial scrape (`_assert_complete_scrape`) raises before any resolution. CIKs resolve via `_make_resolver`;
    `get_fn` defaults to the context-bound, rate-limited `sec_get`."""
    get_fn = get_fn or (lambda url: sec_get(context, url))
    roster = _parse_dataroma_roster(_http_get(DATAROMA_HOME_URL).text)
    logger.info("Dataroma: parsed %d superinvestors", len(roster))
    latest = _latest_mapping(context)
    _assert_complete_scrape(len(roster), latest)
    known, past_names = _stored_resolutions(context)
    overrides = load_superinvestor_overrides(config_dir_of(context))
    resolver = _make_resolver(get_fn, overrides.cik_by_code, past_names, known)
    rows = snapshot_rows(roster, datetime.now(UTC).date(), DATAROMA_HOME_URL, resolver, overrides=overrides)
    if _code_to_cik((r["dataroma_code"] for r in rows), (r["cik"] for r in rows)) == latest:
        logger.info("superinvestor_roster: %d managers unchanged since the latest snapshot, no snapshot written", len(rows))
        return pd.DataFrame(columns=list(rows[0]))
    return _write(context, rows, overrides, listing_fn)
