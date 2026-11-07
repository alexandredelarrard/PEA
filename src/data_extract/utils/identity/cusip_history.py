"""CUSIP history of `universe_cusip`: which security a market-tape line is, its issuer CIK, class and role, dated from the
first evidence of its CUSIP.

FTD capture: the cached FTD lines of the universe companies' lineage symbols and CUSIP-6 prefixes (their Sharadar CUSIPs,
the curated CUSIPs and the prefixes the lineage votes for on the FTD lines), recomputed every run; nothing else is kept.
Lines: one per (CUSIP, FTD symbol). The issuer of each CUSIP-9 comes from lineage symbol-interval votes (inherited across a
CUSIP-6 within the voters' era); a line runs over the FINRA days of its symbol and back to its CUSIP's first 13F period
when the company's Sharadar CUSIPs or its dated names confirm it (a name never confirms a foreign CUSIP-6). The class needs
positive evidence; the role follows the issuer CIK's windows: one canonical line per company and date.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
from tqdm import tqdm

from src.constants.constants import CANONICAL_CURRENT, CANONICAL_PREDECESSOR, CANONICAL_ROLES, SECONDARY_CLASS
from src.context import Context
from src.data_extract.utils.common.resume import recently_changed
from src.data_extract.utils.common.security_lines import (
    ACQUIRED_CONSTITUENT,
    EXCLUDED,
    SOURCE_FTD,
    SecurityManual,
    tape_intervals,
    trade_dates,
)
from src.data_extract.utils.identity.cik_history import ROLE_SYMBOL, ROLE_WINDOW, SENTINEL_START, entity_or_singleton
from src.data_extract.utils.zips import fetch_fails_to_deliver as ftd_zip
from src.data_store.schema import Tables
from src.utils.identity_flags import FLAG_COLUMNS
from src.utils.predecessor_series import SECURITY_MANUAL_FILE
from src.utils.string import normalise_ticker, pad_cik_series, squash

logger = logging.getLogger(__name__)

#: The stored FTD line columns the lines derive from, and those the capture votes with.
OBSERVATION_COLUMNS = ("date", "trade_date", "cusip", "source_symbol", "description", "price", "period")
VOTE_COLUMNS = ("trade_date", "cusip", "source_symbol")
#: The stored lineage columns the FTD capture scope reads.
FTD_LINEAGE_COLUMNS = ("company_id", "canonical_ticker", "cik", "role", "symbol", "valid_from", "valid_to", "status", "sources", "scope_changed_at")
#: The Sharadar ticker columns read for a company's CUSIPs and names; the SEC profile columns read for its dated names.
SHARADAR_CUSIP_COLUMNS = ("permaticker", "name", "cusips")
PROFILE_NAME_COLUMNS = ("cik", "name", "former_names")
#: The `symbol_tenure` columns the co-registrant classification reads.
EVIDENCE_COLUMNS = ("issuer_cik", "source", "valid_from", "valid_to")
#: The 13F holdings grain the back-extension reads (one count per CUSIP, period and issuer spelling).
HOLDING_GROUPS = ("cusip", "period", "issuer_name")

NON_COMMON_KINDS = ("preferred", "debt", "warrant", "unit", "right")
#: A security first (last) seen this close after (before) a window bound was created (ended) at that seam.
SEAM_TOLERANCE = pd.Timedelta(days=7)
#: A line is bridged to the next CUSIP of its symbol only across a gap this short (exchange reuse rule).
BRIDGE_MAX_GAP = pd.Timedelta(days=90)
#: Votes of one CIK covering less than this share of a CUSIP's life, ending this long before it, are seam lag.
WEAK_VOTE_SHARE = 0.1
WEAK_VOTE_GAP = pd.Timedelta(days=90)
#: A line seen in this many of the latest FTD periods is still current (open end).
CURRENT_PERIODS = 2
#: A class line runs on over FINRA days of its symbol only across gaps this short (a listed class trades daily).
FINRA_MAX_GAP = pd.Timedelta(days=183)
#: Canonical candidate overlapping better lines over at least this share of its life is a sibling class.
SIBLING_OVERLAP = 0.5
#: Ratio step check: rolling-median window (observations), the step that is flagged, the match window.
RATIO_WINDOW = 9
RATIO_STEP = 1.5
RATIO_MATCH_DAYS = pd.Timedelta(days=30)
#: 13F evidence (X3): periods from the first EDGAR 13F year up to the run date, with at least this many holder rows; a
#: name confirms periods from a year before it was adopted to two years after it was dropped (filers rename late).
FIRST_13F_PERIOD = pd.Timestamp("1998-01-01")
MIN_13F_HOLDERS = 2
NAME_LEAD = pd.Timedelta(days=365)
NAME_LAG = pd.Timedelta(days=730)
#: Words a holder's issuer spelling and a company name share without naming the company.
_NAME_STOP = frozenset(
    "INC INCORPORATED CORP CORPORATION CO COMPANY COS LTD LIMITED PLC LP L P LLC HLDGS HLDG HOLDINGS HOLDING GROUP GRP THE NEW DEL DE "
    "COM CL CLASS A B C SHS SH SHARES BEN INT TR TRUST NV N V SA AG SE ORD CAP STK COMMON STOCK PAR USD REIT ADR SPONSORED SPON "
    "MD PA NY NJ OH VA IND N".split()
)

TABLE_COLUMNS = (
    "security_id",
    "canonical_company",
    "issuer_cik",
    "source",
    "source_symbol",
    "market_symbol",
    "exchange",
    "cusip",
    "security_class",
    "conversion_ratio",
    "lineage_role",
    "valid_from",
    "valid_to",
    "lineage_reason",
    "source_accession",
    "evidence",
    "n_observations",
    "scope_changed_at",
)
FLAG_DETAIL_COLUMNS = (
    "kind",
    "canonical_company",
    "issuer_cik",
    "cusip",
    "source_symbol",
    "first",
    "last",
    "n_observations",
    "description",
    "detail",
)
REFUSED_COLUMNS = ("cusip", "company", "issuer_name", "first", "n_rows")
#: Flag kinds: (action, suggested action) for the WARNING block.
FLAG_KINDS: dict[str, tuple[bool, str]] = {
    "security_issuer_conflict": (True, "lineage symbol intervals of two companies vote for one CUSIP: add an issuer override"),
    "security_ratio_step": (True, "the class price ratio steps with no configured conversion-ratio change: check for a split"),
    "security_straddle": (False, "one CUSIP traded across a CIK seam: issuer dated by the company's lineage windows"),
    "security_weak_vote": (False, "a lineage symbol interval voted only at the start of a longer-lived CUSIP (seam lag or reuse): vote ignored"),
    "security_symbol_conflict": (False, "one FTD symbol under two CUSIPs on a date: symbol lookups stay unresolved, CUSIP rows are exact"),
    "security_unclassified": (False, "no positive class evidence: excluded; add a class override if it is a common class"),
}
#: Priority of the out-of-window part of a window CIK's common line (taken only where no other line is canonical).
FALLBACK_PRIORITY = 10
_PRIORITY_REASON = {
    -1: "manual_boundary",
    0: "ticker_symbol",
    1: "tape_symbol",
    2: "class_a_line",
    3: "tape_symbol",
    FALLBACK_PRIORITY: "company_line",
}
_SIGNATURE = ("security_id", "source", "source_symbol", "valid_from", "valid_to", "lineage_role", "security_class", "conversion_ratio", "issuer_cik")

_FAR = pd.Timestamp("2262-01-01")
_NEAR = pd.Timestamp("1700-01-01")
_DAY = pd.Timedelta(days=1)
_PLACEHOLDER = re.compile(r"REGWAY|(XXXX|ZZZZ|PAIROFF)$|^[A-Z]?\d")
_WARRANT = re.compile(r"\bWTS?\b|WARRANT|\bWRT", re.I)
_RIGHT = re.compile(r"\bRTS?\b|\bRIGHTS?\b|CONTINGENT|\bCONT\s+V|\bCONTIN$|\bCVR\b", re.I)
_UNIT = re.compile(r"\bUNITS?\b|TANGIBLE EQUIT|PURCHASE CONTRAC|\bCORP(?:ORATE)?\s+U(?:N(?:I(?:TS?)?)?)?$", re.I)
_COMMON_UNIT = re.compile(r"\bCOM(MON)?\s+UNITS?\b", re.I)
_DEBT = re.compile(
    r"\bNOTES?\b|\bNTS?\b|\bDEB\b|DEBENTURE|\bBONDS?\b|\bJR\s+SUB|\bSUB\s+N|\bSENIOR\s+NOT|\bMTG\b|\bETN\b|\bMITTS\b|\bBUFFER\b|\bLKD\b", re.I
)
_PREFERRED = re.compile(
    r"\bPFD\b|\bPREF|\bPRF\b|\bDEP(OSITARY|OS)?\s*SH|DPSTRY|REPSTG|\bDS\s+R|\bCUM(ULATIV)?|%|\bPERP|MANDATORY\s+CONV|\bCONV\s+PF", re.I
)
_PREF_SYMBOL = re.compile(r"^[A-Z]{1,5}PR[A-Z]{0,2}(CL)?$")
_CLASS_SYMBOL = re.compile(r"^[A-Z]+[-./]([A-C])$")
#: An SEC current-tickers spelling that names a common class (`BRK-A`, `MKC-V`); `-P` and unmarked extras are preferreds.
_LISTED_CLASS = re.compile(r"^[A-Z]+-[A-OQ-Z]$")
#: A FINRA class spelling (`BRK/B`).
_FINRA_CLASS = re.compile(r"[A-Z]+/[A-Z]")
_WARRANT_SYMBOL = re.compile(r"^[A-Z]{1,5}WS[A-Z]?$")
_LETTER = re.compile(
    r"\bCL(?:ASS)?[\s\-.']*([A-C])\b|\bSER(?:IES)?\s+([A-C])\s+COM|\(HLDG CO\)\s*([A-C])\b|\bSPL\s+([A-C])\b|\bCOM(?:MON)?\s+([A-C])$",
    re.I,
)

#: (start, end) half-open; `_FAR` stands for an open end.
Span = tuple[pd.Timestamp, pd.Timestamp]
#: A manual role boundary: (start, end, role, reason, source, issuer CIK or "").
Bound = tuple[pd.Timestamp, pd.Timestamp, str, str, str, str]


# The build's records
@dataclass(frozen=True)
class FtdScope:
    """What the FTD capture keeps: the squashed lineage symbols, the CUSIP-6 prefixes, the tape intervals that vote, and
    whether a lineage set it (`ready`)."""

    symbols: frozenset[str] = frozenset()
    prefixes: frozenset[str] = frozenset()
    tape: pd.DataFrame = field(default_factory=pd.DataFrame)
    ready: bool = False


@dataclass(frozen=True)
class FtdCapture:
    """Where and over which periods the FTD capture reads: the cache, the history in years, `full` and the run date."""

    cache: Path
    years_history: int
    full: bool
    as_of: pd.Timestamp


@dataclass(frozen=True)
class LineageView:
    """The lineage facts the lines read: each CIK's company, each company's ticker, CIK windows and symbols, the tape."""

    entity_by_cik: Mapping[str, str]
    ticker_by_entity: Mapping[str, str]
    roster_cik_by_ticker: Mapping[str, str]
    windows: Mapping[str, tuple[Span, ...]]
    tape: pd.DataFrame
    symbols_by_entity: Mapping[str, frozenset[str]]

    def entity(self, cik: str) -> str:
        """The company of a padded CIK; a CIK with no row is its own company."""
        return self.entity_by_cik.get(cik, f"E{cik}")

    def ticker(self, cik: str) -> str | None:
        """The universe ticker of a CIK's company, if any."""
        return self.ticker_by_entity.get(self.entity(cik))

    def window_owners(self, entity: str) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
        """`(cik, start, end)` of every window of a company's CIKs, oldest first."""
        owners = [(cik, a, b) for cik, spans in self.windows.items() if self.entity(cik) == entity for a, b in spans]
        return sorted(owners, key=lambda o: (o[1], o[0]))


@dataclass(frozen=True)
class SecListing:
    """The SEC current-tickers snapshot: `(cik, key) -> spelling`, `-> exchange`, and the keys spelled as a class."""

    symbols: Mapping[tuple[str, str], str]
    exchange: Mapping[tuple[str, str], str]
    classes: frozenset[tuple[str, str]]


@dataclass(frozen=True)
class Finra:
    """FINRA RegSHO evidence: the class symbols (`BRK/B` -> `BRKB`) and the sorted trading days of each unmarked key."""

    classes: frozenset[str]
    days: Mapping[str, Any]


@dataclass(frozen=True)
class ThirteenF:
    """The 13F back-extension: the first confirmed period of each line CUSIP with how, and the refused name matches (X4)."""

    starts: Mapping[str, tuple[pd.Timestamp, str]]
    refused: pd.DataFrame


@dataclass
class CusipFlags:
    """The per-line flags of the build, in the order they are found."""

    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(self, kind: str, line: tuple[str | None, str | None, str, str], life: tuple[object, object, int], description: str, detail: str) -> None:
        """One flag: `line` = (company, issuer CIK, CUSIP, symbols), `life` = (first, last, observations)."""
        self.rows.append(dict(zip(FLAG_DETAIL_COLUMNS, (kind, *line, *life, description, detail), strict=True)))


@dataclass(frozen=True)
class Issuer:
    """A CUSIP's issuer: the CIK owning its latest date, how it was found, and the dated `(cik, start, end)` spans."""

    primary: str
    how: str
    spans: tuple[tuple[str, pd.Timestamp, pd.Timestamp], ...]


@dataclass(frozen=True)
class _Segment:
    start: pd.Timestamp
    end: pd.Timestamp
    role: str
    reason: str
    issuer: str
    source: str | None = None


@dataclass(frozen=True)
class _Candidate:
    """A span a CUSIP may hold as its company's canonical line: traded spans before 13F-only (`early`) ones, then lower
    priority wins, later start first within one."""

    priority: int
    start: pd.Timestamp
    cusip: str
    issuer: str
    spans: tuple[Span, ...]
    source: str | None = None
    early: bool = False


@dataclass(frozen=True)
class CusipFact:
    """Per-CUSIP facts the roles read: `start`..`end` its life (13F included), `order_start` its first traded day."""

    cusip: str
    issuer: Issuer
    company: str
    start: pd.Timestamp
    end: pd.Timestamp
    order_start: pd.Timestamp
    kind: str | None
    letter: str | None
    priority: int | None
    class_evidence: str | None
    manual: tuple[Bound, ...]


@dataclass(frozen=True)
class CusipRoles:
    """The facts per CUSIP and the role segments of each CUSIP's life."""

    facts: Mapping[str, CusipFact]
    segments: dict[str, list[_Segment]]


@dataclass(frozen=True)
class _FactContext:
    """What a CUSIP's facts read besides its lines: lineage view, squashed symbols, class spellings, manual config, listings."""

    view: LineageView
    company_keys: Mapping[str, str]
    own_keys: Mapping[str, frozenset[str]]
    class_keys: Mapping[str, frozenset[tuple[str, str]]]
    manual_class: Mapping[str, str]
    bounds: Mapping[str, list[Bound]]
    listed: frozenset[tuple[str, str]]
    finra: frozenset[str]


@dataclass(frozen=True)
class RowContext:
    """What the table rows read besides the lines and roles: company keys, spellings, ratios and the SEC listing."""

    view: LineageView
    company_keys: Mapping[str, str]
    spelled: Mapping[tuple[str, str], str]
    ratios: Mapping[str, list[tuple[pd.Timestamp, pd.Timestamp, float, str]]]
    listing: SecListing


@dataclass(frozen=True)
class CusipHistory:
    """The `universe_cusip` rows, the per-line flags, the WARNING-block items and the 13F back-extension."""

    rows: pd.DataFrame
    flags: pd.DataFrame
    items: pd.DataFrame
    thirteenf: ThirteenF


# Small flat helpers
def is_placeholder(symbol: str) -> bool:
    """FTD transition placeholders at a CUSIP change, distribution or delisting (`XOMZZZZ`, `GOOGLXXXX`, `3106PS`, `C117REGWAY`)."""
    return bool(_PLACEHOLDER.search(str(symbol)))


def non_common_kind(description: str, symbol: str = "") -> str | None:
    """`warrant`, `right`, `unit`, `debt` or `preferred` from word-bounded description or symbol evidence; None otherwise."""
    text, sym = str(description or ""), str(symbol or "").strip().upper()
    if _WARRANT.search(text) or _WARRANT_SYMBOL.search(sym):
        return "warrant"
    if _RIGHT.search(text):
        return "right"
    if _UNIT.search(text) and not _COMMON_UNIT.search(text):
        return "unit"
    if _DEBT.search(text):
        return "debt"
    if _PREFERRED.search(text) or _PREF_SYMBOL.search(sym):
        return "preferred"
    return None


def class_letter(description: str) -> str | None:
    """The share-class letter a description names (`CL A`, `CLASS C`, `SER B COM`, `(HLDG CO)A`, `SPL A`), or None."""
    match = _LETTER.search(str(description or ""))
    return next((group.upper() for group in match.groups() if group), None) if match else None


def _records(frame: pd.DataFrame) -> list[Any]:
    """Rows as attribute records (`row.cusip`), loosely typed like the frame's cells."""
    return [SimpleNamespace(**record) for record in cast(list[dict[str, Any]], frame.to_dict("records"))]


def _bound(value: object) -> pd.Timestamp | None:
    """A date as a Timestamp; null and the open-start sentinel read as None."""
    if value is None or (not isinstance(value, str) and pd.isna(cast(Any, value))):
        return None
    stamp = pd.Timestamp(cast(Any, value))
    return None if stamp <= SENTINEL_START else stamp


def _intersect(spans: Iterable[Span], start: pd.Timestamp, end: pd.Timestamp) -> list[Span]:
    """The parts of `spans` inside `[start, end)`."""
    return [(max(a, start), min(b, end)) for a, b in spans if min(b, end) > max(a, start)]


def _subtract(spans: Iterable[Span], cut: Iterable[Span]) -> list[Span]:
    """The parts of `spans` outside every span of `cut`."""
    out = list(spans)
    for c_from, c_to in cut:
        out = [piece for a, b in out for piece in ((a, min(b, c_from)), (max(a, c_to), b)) if piece[1] > piece[0]]
    return out


def _length(spans: Iterable[Span], horizon: pd.Timestamp) -> float:
    """Days covered by `spans` up to `horizon` (an open end counts to the last observation, not to `_FAR`)."""
    return float(sum((min(b, horizon) - a).days for a, b in spans if min(b, horizon) > a))


# The FTD capture
def ftd_scope(lineage: pd.DataFrame | None, cusips: Mapping[str, frozenset[str]], stored: pd.DataFrame | None, manual: SecurityManual) -> FtdScope:
    """The capture scope: the lineage's non-noise symbols, and the CUSIP-6 of the universe companies' Sharadar CUSIPs, of
    the curated CUSIPs and of the captured FTD lines the lineage tape votes for (recomputed each run); unset without a lineage."""
    # nothing is in scope before the first lineage build
    if lineage is None:
        return FtdScope()

    # the lineage symbols and the curated and Sharadar prefixes
    rows = lineage[lineage["role"].eq(ROLE_SYMBOL) & lineage["status"].ne("noise")]
    symbols = frozenset(key for key in rows["symbol"].map(squash) if key)
    curated = set(manual.boundaries["cusip"].dropna()) | set(manual.ratios["cusip"].dropna()) | {c for group in cusips.values() for c in group}

    # the prefixes the lineage votes for on the captured lines
    tape = tape_intervals(lineage)
    keyed = pd.DataFrame(columns=["cusip", "key", "trade_date"])
    if stored is not None and not stored.empty:
        keyed = pd.DataFrame(
            {
                "cusip": stored["cusip"].astype(str).str.upper(),
                "key": stored["source_symbol"].map(squash),
                "trade_date": pd.to_datetime(stored["trade_date"]),
            }
        )
    voted = set(_votes(keyed.drop_duplicates(ignore_index=True), tape)["cusip"])
    return FtdScope(symbols=symbols, prefixes=frozenset(str(c)[:6] for c in curated | voted if c), tape=tape, ready=True)


def changed_ftd_scope(lineage: pd.DataFrame, cusips: Mapping[str, frozenset[str]], as_of: pd.Timestamp) -> FtdScope:
    """The symbols and Sharadar CUSIP-6 of the companies whose lineage changed recently: every cached period is re-read for them."""
    # the companies whose latest scope stamp falls in the re-check window
    stamps = pd.to_datetime(lineage["scope_changed_at"], errors="coerce").groupby(lineage["company_id"].astype(str)).max()
    entities = set(recently_changed({str(k): v for k, v in stamps.items()}, as_of))
    changed = lineage[lineage["company_id"].astype(str).isin(entities)]

    # their non-noise symbols and their tickers' Sharadar prefixes
    rows = changed[changed["role"].eq(ROLE_SYMBOL) & changed["status"].ne("noise")]
    symbols = frozenset(key for key in rows["symbol"].map(squash) if key)
    tickers = set(changed["canonical_ticker"].dropna().astype(str))
    return FtdScope(symbols=symbols, prefixes=frozenset(c[:6] for ticker in tickers for c in cusips.get(ticker, ())))


def capture_ftd_lines(context: Context, plan: FtdCapture, scope: FtdScope, changed: FtdScope) -> tuple[pd.DataFrame, list[str]]:
    """The in-scope lines of the pending cached periods, then of every cached period for a CUSIP-6 newly voted for or a
    changed scope; returns the unique lines and the periods they reconcile (none before the first lineage build)."""
    # nothing to capture before the first lineage build
    if not scope.ready:
        logger.info("FTD capture: no universe_cik_symbol yet (cold start); nothing captured until the lineage is built")
        return pd.DataFrame(columns=list(ftd_zip.SECURITY_COLUMNS)), []

    # the pending periods, read with the scope
    pending = ftd_zip.pending_ftd_periods(context, plan.cache, plan.years_history, full=plan.full, as_of=plan.as_of)
    cached = ftd_zip.cached_ftd_periods(plan.cache)
    first = {
        p: f
        for p in tqdm([p for p in pending if p in cached], desc="FTD capture")
        if (f := ftd_zip.read_ftd_period(context, plan.cache, p, download=False)) is not None
    }
    first = {p: f[f["source_symbol"].map(squash).isin(scope.symbols) | f["cusip"].str[:6].isin(scope.prefixes)] for p, f in first.items()}
    kept = pd.concat(first.values(), ignore_index=True) if first else pd.DataFrame(columns=list(ftd_zip.SECURITY_COLUMNS[:9]))

    # a CUSIP-6 the lineage votes for in these lines, or a changed scope, re-reads every cached period
    keyed = pd.DataFrame(
        {"cusip": kept["cusip"].astype(str).str.upper(), "key": kept["source_symbol"].map(squash), "trade_date": pd.to_datetime(kept["trade_date"])}
    )
    added = frozenset(str(c)[:6] for c in _votes(keyed.drop_duplicates(ignore_index=True), scope.tape)["cusip"]) - scope.prefixes
    rescan = added | changed.prefixes
    again: dict[str, pd.DataFrame] = {}
    if rescan or changed.symbols:
        logger.info(
            "FTD capture: re-reading every cached period for %d changed symbol(s) and %d CUSIP-6 prefix(es) (%s)",
            len(changed.symbols),
            len(rescan),
            ", ".join(sorted(rescan)),
        )
        frames = {
            p: f
            for p in tqdm(sorted(cached), desc="FTD capture")
            if (f := ftd_zip.read_ftd_period(context, plan.cache, p, download=False)) is not None
        }
        again = {
            p: f[f["source_symbol"].map(squash).isin(scope.symbols) | f["cusip"].str[:6].isin(scope.prefixes | rescan)] for p, f in frames.items()
        }

    # the unique lines, stamp columns empty until the master stamps them
    kept = ftd_zip.unique_ftd_keys(pd.concat([kept, *again.values()], ignore_index=True), "FTD capture")
    rows = kept.assign(security_id=None, ticker=None, lineage_role=None, security_class=None)[list(ftd_zip.SECURITY_COLUMNS)]
    logger.info(
        "FTD capture: %d period(s) read, %d in-scope line(s) (%d symbol(s), %d CUSIP-6 prefix(es) in scope)",
        len(pending),
        len(rows),
        len(scope.symbols),
        len(scope.prefixes | added),
    )
    return rows, sorted(set(first) | set(again))


def save_ftd_lines(context: Context, rows: pd.DataFrame, periods: Sequence[str]) -> int:
    """New raw units and retained, explicitly unverified derived rows commit as one source pair per period."""
    # the derived rows the periods already hold
    columns = context.store.columns(Tables.sec_fails_to_deliver)
    previous = context.store.load(Tables.sec_fails_to_deliver, columns=columns, where={"period": list(periods)}, optional=True) if periods else None

    # each period: its raw lines replace the stored ones, its derived rows stay but lose their checked quiet zeros
    observed = previous if previous is not None else ftd_zip.empty_fails_grain()
    if "source_checked_at" not in observed:
        observed = observed.assign(source_checked_at=pd.Series(pd.NaT, index=observed.index, dtype="datetime64[ns]"))
    written = 0
    for period in periods:
        payload = rows.loc[rows["period"].eq(period)]
        expected = observed.loc[observed["period"].eq(period)]
        retained = expected.assign(source_checked_at=pd.Series(pd.NaT, index=expected.index, dtype="datetime64[ns]"))
        quiet = expected["source_checked_at"].notna() & expected["fails_quantity"].eq(0)
        retained.loc[quiet, ["fails_quantity", "fails_value"]] = float("nan")
        counts = ftd_zip.replace_fails_source(
            context.store,
            {Tables.sec_fails_to_deliver_security: payload, Tables.sec_fails_to_deliver: retained},
            {Tables.sec_fails_to_deliver_security: {"period": period}, Tables.sec_fails_to_deliver: {"period": period}},
            expected_source={Tables.sec_fails_to_deliver: expected},
        )
        written += counts[Tables.sec_fails_to_deliver_security]
    return written


# The observations, the lineage view and the listings
def prepare_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """The FTD lines typed: settlement and trade dates, upper-case CUSIP, normalised symbol and its key, price, period."""
    obs = observations.copy()
    obs["date"] = pd.to_datetime(obs["date"])
    obs["trade_date"] = pd.to_datetime(obs["trade_date"]) if "trade_date" in obs.columns else trade_dates(obs["date"])
    obs["cusip"] = obs["cusip"].astype(str).str.strip().str.upper()
    obs["source_symbol"] = obs["source_symbol"].fillna("").map(normalise_ticker)
    obs["description"] = obs["description"].fillna("").astype(str).str.strip()
    obs["key"] = obs["source_symbol"].map(squash)
    obs["price"] = pd.to_numeric(obs["price"], errors="coerce") if "price" in obs.columns else float("nan")
    obs["period"] = obs["period"].astype(str) if "period" in obs.columns else ""
    return obs.sort_values(["cusip", "source_symbol", "trade_date", "date"], kind="mergesort", ignore_index=True)


def lineage_view(lineage: pd.DataFrame, roster_cik: Mapping[str, str], tape: pd.DataFrame) -> LineageView:
    """Each CIK's company, each roster ticker's company, the CIK windows (a roster CIK without one is open), each company's symbols."""
    # each CIK's company and each roster ticker's
    rows = lineage.assign(cik=pad_cik_series(lineage["cik"]))
    entity_by_cik = dict(zip(rows["cik"], rows["company_id"].astype(str), strict=False))
    ticker_by_entity = {entity_or_singleton(entity_by_cik, cik): ticker for ticker, cik in sorted(roster_cik.items())}

    # the dated windows of each CIK and the symbols of each company
    windows: dict[str, list[Span]] = {}
    for cik, start, end in rows.loc[rows["role"].eq(ROLE_WINDOW), ["cik", "valid_from", "valid_to"]].to_numpy().tolist():
        windows.setdefault(cik, []).append((_bound(start) or _NEAR, _bound(end) or _FAR))
    for cik in roster_cik.values():
        windows.setdefault(cik, [(_NEAR, _FAR)])
    symbols = rows[rows["role"].eq(ROLE_SYMBOL)].groupby("company_id")["symbol"].agg(lambda s: frozenset(map(str, s))).to_dict()
    return LineageView(
        entity_by_cik=entity_by_cik,
        ticker_by_entity=ticker_by_entity,
        roster_cik_by_ticker=roster_cik,
        windows={cik: tuple(sorted(spans)) for cik, spans in windows.items()},
        tape=tape,
        symbols_by_entity={str(k): v for k, v in symbols.items()},
    )


def sec_listing(sec_tickers: pd.DataFrame | None) -> SecListing:
    """`(cik, squashed ticker) -> market spelling` and `-> exchange` from the SEC current-tickers snapshot, and its class spellings."""
    if sec_tickers is None or sec_tickers.empty:
        return SecListing({}, {}, frozenset())
    ciks = pad_cik_series(sec_tickers["cik"])
    tickers = sec_tickers["ticker"].astype(str)
    exchanges = sec_tickers["exchange"] if "exchange" in sec_tickers.columns else pd.Series(None, index=sec_tickers.index)
    symbols = {(cik, squash(t)): normalise_ticker(t) for cik, t in zip(ciks, tickers, strict=True)}
    exchange = {(cik, squash(t)): ex for cik, t, ex in zip(ciks, tickers, exchanges, strict=True) if isinstance(ex, str)}
    return SecListing(symbols, exchange, frozenset(key for key, spelled in symbols.items() if _LISTED_CLASS.match(spelled)))


def finra_evidence(presence: pd.DataFrame | None) -> Finra:
    """The FINRA class symbols and the sorted RegSHO days of each unmarked key (a lower-case marker such as `BACpB` is not
    a common line's key); the stored days are one night old (X15)."""
    if presence is None or presence.empty:
        return Finra(frozenset(), {})
    symbols = presence["source_symbol"].astype(str)
    unmarked = symbols.eq(symbols.str.upper())
    classes = frozenset(squash(s) for s in symbols[unmarked].unique() if _FINRA_CLASS.fullmatch(s))
    frame = pd.DataFrame({"key": symbols.map(squash), "date": pd.to_datetime(presence["date"])})[unmarked.to_numpy()]
    return Finra(classes, {str(key): group.drop_duplicates().sort_values().to_numpy() for key, group in frame.groupby("key")["date"]})


def sharadar_cusips(universe: pd.DataFrame, sharadar: pd.DataFrame | None) -> dict[str, frozenset[str]]:
    """`{universe ticker: the CUSIPs every Sharadar row of its permaticker lists}`."""
    if sharadar is None or sharadar.empty or "permaticker" not in universe.columns:
        return {}
    rows = universe[["ticker", "permaticker"]].dropna().merge(sharadar[["permaticker", "cusips"]].dropna(), on="permaticker", how="inner")
    listed = rows.assign(cusip=rows["cusips"].astype(str).str.upper().str.split()).explode("cusip").dropna(subset=["cusip"])
    return {str(ticker): frozenset(group) for ticker, group in listed.groupby("ticker")["cusip"]}


# Lines and issuers
def ftd_lines(obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One line per (cusip, source_symbol): life in trade dates, observations, descriptions, latest period; and the life
    of each CUSIP over its real (not placeholder) lines."""
    # one row per (cusip, symbol)
    lines = obs.groupby(["cusip", "source_symbol"], as_index=False, sort=True).agg(
        key=("key", "first"),
        first=("trade_date", "min"),
        last=("trade_date", "max"),
        n_obs=("trade_date", "size"),
        last_period=("period", "max"),
        descriptions=("description", lambda s: tuple(sorted(set(s)))),
    )
    lines["placeholder"] = lines["source_symbol"].map(is_placeholder)
    lines["description"] = [max(descs, key=len) if descs else "" for descs in lines["descriptions"]]

    # each CUSIP's life over its real lines
    spans = (
        lines[~lines["placeholder"]]
        .groupby("cusip", as_index=False)
        .agg(
            first=("first", "min"),
            last=("last", "max"),
            n_obs=("n_obs", "sum"),
            description=("description", "first"),
            symbols=("source_symbol", lambda s: " ".join(sorted(s))),
        )
    )
    return lines, spans


def _votes(keyed: pd.DataFrame, tape: pd.DataFrame) -> pd.DataFrame:
    """`[cusip, cik, votes, first, last]`: the `(cusip, key, trade_date)` triples inside a tape interval of their key, per CIK."""
    merged = keyed.merge(tape, on="key", how="inner")
    merged = merged[(merged["trade_date"] >= merged["valid_from"]) & (merged["trade_date"] < merged["valid_to"])]
    if merged.empty:
        return pd.DataFrame(columns=["cusip", "cik", "votes", "first", "last"])
    return (
        merged.groupby(["cusip", "cik"], as_index=False)
        .agg(votes=("trade_date", "size"), first=("trade_date", "min"), last=("trade_date", "max"))
        .sort_values(["cusip", "cik"], ignore_index=True)
    )


def _strong_votes(votes: pd.DataFrame, spans: pd.DataFrame, view: LineageView, flags: CusipFlags) -> pd.DataFrame:
    """The votes that are not seam lag (they cover enough of the CUSIP's life, or reach near its end); the others flagged."""
    life = spans.set_index("cusip")
    kept = []
    for key, group in votes.groupby("cusip", sort=True):
        cusip = str(key)
        if cusip not in life.index:
            kept.append(group)
            continue
        span = cast(Any, life.loc[cusip])
        days = max((span["last"] - span["first"]).days, 1)
        share = (group["last"] - group["first"]).dt.days.clip(lower=1) / days
        keep = ~((days > BRIDGE_MAX_GAP.days) & (share < WEAK_VOTE_SHARE) & (span["last"] - group["last"] > WEAK_VOTE_GAP))
        for row in group[~keep].itertuples(index=False):
            detail = f"CIK {row.cik} n={row.votes} {row.first.date()}..{row.last.date()} of a life {span['first'].date()}..{span['last'].date()}"
            line = (view.ticker(row.cik), row.cik, cusip, span["symbols"])
            flags.add("security_weak_vote", line, (span["first"], span["last"], int(span["n_obs"])), span["description"], detail)
        kept.append(group[keep])
    return pd.concat(kept, ignore_index=True) if kept else votes.iloc[0:0]


def _resolve(cands: list[str], first: pd.Timestamp, last: pd.Timestamp, view: LineageView, how: str) -> Issuer | None:
    """One CIK; of several CIKs of one company, the one whose window holds the CUSIP's life, else the window owner per
    date. None when the votes name two companies, or several CIKs of one company none of which has a window."""
    if len(cands) == 1:
        return Issuer(cands[0], how, ((cands[0], _NEAR, _FAR),))
    entities = {view.entity(c) for c in cands}
    if len(entities) > 1:
        return None
    holding = [c for c in cands if any(first >= a - SEAM_TOLERANCE and last < b + SEAM_TOLERANCE for a, b in view.windows.get(c, ()))]
    if len(holding) == 1:
        return Issuer(holding[0], how, ((holding[0], _NEAR, _FAR),))
    owners = view.window_owners(entities.pop())
    if not owners:
        return None
    spans = tuple((cik, _NEAR if i == 0 else a, _FAR if i == len(owners) - 1 else owners[i + 1][1]) for i, (cik, a, _) in enumerate(owners))
    return Issuer(next(cik for cik, a, b in spans if a <= last < b), how, spans)


def cusip_issuers(spans: pd.DataFrame, obs: pd.DataFrame, view: LineageView, manual: SecurityManual, flags: CusipFlags) -> dict[str, Issuer]:
    """`{cusip: Issuer}` from manual rows, strong direct votes, else CUSIP-6 voters whose era overlaps the CUSIP's life."""
    # the strong votes of the tape intervals, per CUSIP and per CUSIP-6 era
    votes = _votes(obs[["cusip", "key", "trade_date"]].drop_duplicates(), view.tape)
    strong = _strong_votes(votes, spans, view, flags)
    voted, direct = set(votes["cusip"]), {cusip: group for cusip, group in strong.groupby("cusip")}
    by_prefix = (
        strong.assign(prefix=strong["cusip"].str[:6]).groupby(["prefix", "cik"], as_index=False).agg(first=("first", "min"), last=("last", "max"))
    )
    eras = {prefix: list(zip(g["cik"], g["first"], g["last"], strict=True)) for prefix, g in by_prefix.groupby("prefix")}
    manual_issuer = manual.boundaries.dropna(subset=["issuer_cik"]).groupby("cusip")["issuer_cik"].agg(lambda s: sorted(set(s))).to_dict()

    # one issuer per CUSIP: a manual boundary, its strong direct voters, else the CUSIP-6 voters of its era
    out: dict[str, Issuer] = {}
    for span in _records(spans):
        cusip, first, last = span.cusip, span.first, span.last
        if cusip in manual_issuer and len(manual_issuer[cusip]) == 1:
            out[cusip] = Issuer(manual_issuer[cusip][0], "manual market boundary", ((manual_issuer[cusip][0], _NEAR, _FAR),))
            continue
        if cusip in voted and cusip not in direct:
            continue
        group = direct.get(cusip)
        cands = sorted(set(group["cik"])) if group is not None else sorted({cik for cik, a, b in eras.get(cusip[:6], []) if a <= last and b >= first})
        how = (
            "votes "
            + ", ".join(
                f"{c} n={n} {a.date()}..{b.date()}" for c, n, a, b in zip(group["cik"], group["votes"], group["first"], group["last"], strict=True)
            )
            if group is not None
            else f"CUSIP-6 {cusip[:6]} voters " + ", ".join(cands)
        )
        issuer = _resolve(cands, first, last, view, how) if cands else None
        companies = ",".join(sorted({view.ticker(c) or view.entity(c) for c in cands}))
        line, life = (companies, ",".join(cands), cusip, span.symbols), (first, last, span.n_obs)
        if cands and issuer is None:
            flags.add("security_issuer_conflict", line, life, span.description, how)
        if issuer is not None and len(issuer.spans) > 1:
            dated = "; ".join(
                f"{c} {max(a, first).date()}..{'open' if b >= _FAR else b.date()}" for c, a, b in issuer.spans if b > first and a <= last
            )
            flags.add("security_straddle", line, life, span.description, f"issuer by window: {dated}")
        if issuer is not None:
            out[cusip] = issuer
    return out


def _line_ends(
    lines: pd.DataFrame, current_periods: frozenset[str], listed: frozenset[tuple[str, str]], issuer: Mapping[str, str]
) -> list[pd.Timestamp]:
    """Exclusive end of each line: open (`_FAR`) when current, bridged to the next CUSIP of its symbol, else the day after
    its last row. Being current keeps open only the line of the symbol's latest CUSIP."""
    starts = lines.groupby("key")[["first", "cusip"]].apply(lambda g: sorted(zip(g["first"], g["cusip"], strict=True))).to_dict()
    ends: list[pd.Timestamp] = []
    for cusip, key, first, last, period in lines[["cusip", "key", "first", "last", "last_period"]].to_numpy().tolist():
        nxt = next((start for start, other in starts.get(key, []) if other != cusip and start > last), None)
        superseded = any(other != cusip and start > first for start, other in starts.get(key, []))
        if not superseded and (period in current_periods or (issuer.get(cusip), key) in listed):
            ends.append(_FAR)
            continue
        ends.append(nxt if nxt is not None and nxt - last <= BRIDGE_MAX_GAP else last + _DAY)
    return ends


def line_ends(lines: pd.DataFrame, obs: pd.DataFrame, listing: SecListing, issuers: Mapping[str, Issuer]) -> pd.DataFrame:
    """Each line's exclusive end: open while seen in the latest FTD periods or listed by the SEC, else bridged or the day after."""
    periods = sorted(set(obs["period"]))
    issuer = {cusip: found.primary for cusip, found in issuers.items()}
    return lines.assign(end=_line_ends(lines, frozenset(periods[-CURRENT_PERIODS:]), frozenset(listing.symbols), issuer))


def _symbol_starts(lines: pd.DataFrame, issuers: Mapping[str, Issuer], view: LineageView) -> list[pd.Timestamp | None]:
    """Start of the issuer company's tape interval of each line's symbol that holds the line's first fail; None without one."""
    intervals: dict[tuple[str, str], list[Span]] = {}
    for key, cik, start, end in view.tape[["key", "cik", "valid_from", "valid_to"]].to_numpy().tolist():
        intervals.setdefault((view.entity(cik), key), []).append((_NEAR if pd.isna(start) else start, end))
    out: list[pd.Timestamp | None] = []
    for cusip, key, first in lines[["cusip", "key", "first"]].to_numpy().tolist():
        issuer = issuers.get(cusip)
        spans = intervals.get((view.entity(issuer.primary), key), []) if issuer else []
        out.append(next((start for start, end in spans if start <= first < end), None))
    return out


def _chain(days: Any, anchor: pd.Timestamp, stop: pd.Timestamp, forward: bool) -> pd.Timestamp:
    """The farthest FINRA day reached from `anchor` without a gap over `FINRA_MAX_GAP` or crossing `stop`."""
    reached = anchor
    picked = days[days > anchor.to_datetime64()] if forward else days[days < anchor.to_datetime64()][::-1]
    for raw in picked:
        day = pd.Timestamp(raw)
        if (day >= stop if forward else day < stop) or abs(day - reached) > FINRA_MAX_GAP:
            break
        reached = day
    return reached


def _forward_stop(line: Any, others: pd.DataFrame, manual_ends: Mapping[str, list[pd.Timestamp]]) -> pd.Timestamp:
    """Where a class line's run on stops: another CUSIP of its symbol trading later (at its current start), or a manual end."""
    later = others[others["last"] > line.last]
    stops = [max(start, line.last + _DAY) for start in later["start"]]
    stops += [end for end in manual_ends.get(line.cusip, []) if end > line.last]
    return min(stops, default=_FAR)


def _back_floor(
    line: Any, earlier: pd.DataFrame, manual_starts: Mapping[str, list[pd.Timestamp]], classed: bool, symbol_from: pd.Timestamp | None
) -> pd.Timestamp:
    """Where a line's run back stops: an earlier CUSIP of its symbol or an earlier symbol of its CUSIP (its last fail, or the
    end of its own run on), a manual start, and for a non-class line the start of its issuer's tape interval of the symbol."""
    ran_on = earlier["end"].where(earlier["finra_to"].notna(), earlier["last"] + _DAY)
    floors = [min(end, line.first) for end in ran_on]
    floors += [start for start in manual_starts.get(line.cusip, []) if start <= line.first]
    floors += [] if classed else [symbol_from]
    return max(floors, default=_NEAR)


def extend_lines(lines: pd.DataFrame, finra: Finra, manual: SecurityManual, issuers: Mapping[str, Issuer], view: LineageView) -> pd.DataFrame:
    """A class line runs over the FINRA days of its symbol around its fails (on, and back); another common line only back,
    no earlier than its issuer's tape interval of the symbol. A run back ends an earlier CUSIP's bridge at its new start;
    a line ending inside a manual boundary with an explicit end runs to it."""
    # per line: its symbol interval start, whether it is a common line and a class line by positive evidence
    starts = _symbol_starts(lines, issuers, view)
    kinds = [any(non_common_kind(d, s) for d in descs) for descs, s in zip(lines["descriptions"], lines["source_symbol"], strict=True)]
    common = [not placeholder and not kind for placeholder, kind in zip(lines["placeholder"], kinds, strict=True)]
    lettered = [any(class_letter(d) for d in descs) or key in finra.classes for descs, key in zip(lines["descriptions"], lines["key"], strict=True)]
    classed = [is_common and is_lettered for is_common, is_lettered in zip(common, lettered, strict=True)]
    manual_ends = manual.boundaries.dropna(subset=["valid_to"]).groupby("cusip")["valid_to"].agg(list).to_dict()
    manual_starts = manual.boundaries.dropna(subset=["valid_from"]).groupby("cusip")["valid_from"].agg(list).to_dict()

    # each line over the FINRA days of its symbol
    out = lines.assign(start=lines["first"], finra_from=pd.NaT, finra_to=pd.NaT)
    by_key = {key: list(index) for key, index in out.groupby("key").groups.items()}
    for i, line, symbol_from, is_common, is_class in zip(out.index, _records(out), starts, common, classed, strict=True):
        days = finra.days.get(line.key)
        if days is None or not (is_class or (symbol_from is not None and is_common)):
            continue
        others = out.loc[[j for j in by_key.get(line.key, []) if out.at[j, "cusip"] != line.cusip], ["start", "first", "last", "end", "finra_to"]]
        last = _chain(days, line.last, _forward_stop(line, others, manual_ends), forward=True) if is_class and line.end < _FAR else line.last
        if last > line.last and last + _DAY > line.end:
            out.at[i, "end"], out.at[i, "finra_to"] = last + _DAY, last
        earlier = others[others["first"] < line.first]
        # X91: nor back over the days its own CUSIP still traded under an earlier symbol (FB before META, to its last FINRA day)
        renamed = out.loc[out["cusip"].eq(line.cusip) & out["key"].ne(line.key) & out["first"].lt(line.first)]
        reach = [
            _chain(finra.days[k], last, _FAR, forward=True) if k in finra.days else last
            for k, last in zip(renamed["key"], renamed["last"], strict=True)
        ]
        renamed = renamed.assign(end=[day + _DAY for day in reach], finra_to=reach)[earlier.columns]
        first = _chain(days, line.first, _back_floor(line, pd.concat([earlier, renamed]), manual_starts, is_class, symbol_from), forward=False)
        if first < line.first:
            out.at[i, "start"], out.at[i, "finra_from"] = first, first
            out.loc[earlier.index[(earlier["end"] > first) & (earlier["end"] < _FAR) & earlier["finra_to"].isna()], "end"] = first

    # a line ending inside a manual boundary that names an explicit end runs to that end (the market date)
    for r in _records(manual.boundaries.dropna(subset=["valid_to"])):
        mask = out["cusip"].eq(r.cusip) & ~out["placeholder"] & out["last"].lt(r.valid_to) & out["end"].lt(r.valid_to)
        out.loc[mask & (r.valid_to - out["last"] <= BRIDGE_MAX_GAP), "end"] = r.valid_to
    return out


# The 13F back-extension (X3, X4)
def _tokens(name: object) -> tuple[str, ...]:
    """The significant words of a company or issuer name (legal, class and place words and numbers dropped)."""
    text = re.sub(r"[^A-Z0-9&]+", " ", str(name or "").upper())
    return tuple(t for t in text.split() if t not in _NAME_STOP and not re.fullmatch(r"\d+(\.\d+)?", t))


def _compatible(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """One name up to 13F abbreviation: same first word (or the first two swapped, `DISNEY WALT`), the second word a prefix
    or same-initial subsequence of the other (`INTL` ~ `INTERNATIONAL`); a one-word name needs 5 letters."""
    if not a or not b or a == b:
        return bool(a) and a == b
    if len(a) >= 2 and len(b) >= 2 and a[0] == b[1] and a[1] == b[0]:
        return True
    if a[0] != b[0] or len(a[0]) < 2:
        return False
    if len(a) < 2 or len(b) < 2:
        return len(a[0]) >= 5
    short, long_ = sorted((a[1], b[1]), key=len)
    letters = iter(long_)
    return long_.startswith(short) or (short[0] == long_[0] and len(short) >= 3 and all(ch in letters for ch in short))


def company_names(view: LineageView, universe: pd.DataFrame, sharadar: pd.DataFrame | None, profile: pd.DataFrame | None) -> pd.DataFrame:
    """`[company, tokens, valid_from, valid_to]`: each universe company's names, undated (universe, Sharadar) or dated by
    the SEC profile of each of its CIKs (the current name from the latest former name's end)."""
    # the undated names of each universe ticker: its own, its Sharadar spelling, every Sharadar row of its permaticker
    rows = [
        (t, n, _NEAR, _FAR)
        for column in ("name", "sharadar_name")
        if column in universe.columns
        for t, n in zip(universe["ticker"], universe[column], strict=True)
    ]
    if sharadar is not None and "permaticker" in universe.columns:
        named = universe[["ticker", "permaticker"]].dropna().merge(sharadar[["permaticker", "name"]].dropna(), on="permaticker", how="inner")
        rows += [(t, n, _NEAR, _FAR) for t, n in zip(named["ticker"], named["name"], strict=True)]

    # the dated SEC names of each CIK of a universe company
    for cik, name, former in [] if profile is None else profile[list(PROFILE_NAME_COLUMNS)].to_numpy().tolist():
        ticker = view.ticker(str(cik))
        dated = [
            (f.get("name"), pd.Timestamp(f["from"]) if f.get("from") else _NEAR, pd.Timestamp(f["to"]) if f.get("to") else _FAR)
            for f in (json.loads(former) if isinstance(former, str) and former else [])
        ]
        rows += [(ticker, n, a, b) for n, a, b in dated if ticker] + (
            [(ticker, name, max((b for *_, b in dated), default=_NEAR), _FAR)] if ticker else []
        )

    # each name's significant words
    frame = pd.DataFrame(rows, columns=["company", "name", "valid_from", "valid_to"]).dropna(subset=["company", "name"])
    frame["tokens"] = frame["name"].map(_tokens)
    return frame[frame["tokens"].map(bool)].drop_duplicates(["company", "tokens", "valid_from", "valid_to"], ignore_index=True)


def thirteenf_starts(
    holdings: pd.DataFrame | None, owners: Mapping[str, str], names: pd.DataFrame, listed: Mapping[str, frozenset[str]], as_of: pd.Timestamp
) -> ThirteenF:
    """Per CUSIP of a company's line, its first 13F period (from 1998, `MIN_13F_HOLDERS` holder rows) that the company's
    Sharadar CUSIPs or a name of the company valid then confirms (X3). A name never confirms a CUSIP whose CUSIP-6 the
    company does not own (X4): those matches are returned as refused."""
    # the 13F rows of a plausible period, each with the company owning its CUSIP's line
    if holdings is None or holdings.empty:
        return ThirteenF({}, pd.DataFrame(columns=list(REFUSED_COLUMNS)))
    rows = holdings.assign(
        period=pd.to_datetime(holdings["period"], errors="coerce"),
        cusip=holdings["cusip"].astype(str).str.upper(),
        issuer_name=holdings["issuer_name"].fillna("").astype(str),
    )
    holders = rows.groupby(["cusip", "period"])["n_rows"].transform("sum")
    rows = rows[rows["period"].between(FIRST_13F_PERIOD, as_of) & holders.ge(MIN_13F_HOLDERS)]
    rows = rows.assign(owner=rows["cusip"].map(owners))

    # each holder spelling with every company name it is compatible with (looked up by either of its first two words)
    index: dict[str, list[tuple[str, tuple[str, ...], pd.Timestamp, pd.Timestamp]]] = {}
    for company, tokens, start, end in names[["company", "tokens", "valid_from", "valid_to"]].to_numpy().tolist():
        for word in dict.fromkeys(tokens[:2]):
            index.setdefault(word, []).append((company, tokens, start, end))
    spellings = {spelling: _tokens(spelling) for spelling in rows["issuer_name"].unique()}
    candidates = {
        spelling: dict.fromkeys(entry for word in dict.fromkeys(words[:2]) for entry in index.get(word, [])) for spelling, words in spellings.items()
    }
    pairs = pd.DataFrame(
        [
            (spelling, company, start, end)
            for spelling, found in candidates.items()
            for company, tokens, start, end in found
            if _compatible(spellings[spelling], tokens)
        ],
        columns=["issuer_name", "company", "valid_from", "valid_to"],
    )

    # confirmed: the owner's Sharadar CUSIP list, or one of the owner's names valid at the period
    named = rows.merge(pairs, on="issuer_name", how="inner")
    named = named[(named["period"] >= named["valid_from"] - NAME_LEAD) & (named["period"] <= named["valid_to"] + NAME_LAG)]
    by_name = named[named["company"].eq(named["owner"])]
    by_name = by_name.assign(how="13F name '" + by_name["issuer_name"] + "'")
    on_list = pd.Series([c in listed.get(str(o), ()) for c, o in zip(rows["cusip"], rows["owner"], strict=True)], index=rows.index, dtype=bool)
    by_list = rows[on_list].assign(how="Sharadar cusips")
    confirmed = pd.concat([by_list, by_name], ignore_index=True).sort_values(["cusip", "period", "how"], kind="mergesort").drop_duplicates("cusip")

    # refused (X4): a name matching a company that owns no line of this CUSIP-6
    prefixes = {(ticker, cusip[:6]) for cusip, ticker in owners.items()}
    unowned = pd.Series([(t, c[:6]) not in prefixes for t, c in zip(named["company"], named["cusip"], strict=True)], index=named.index, dtype=bool)
    foreign = named[named["company"].ne(named["owner"]) & unowned]
    refused = foreign.groupby(["cusip", "company"], as_index=False).agg(
        issuer_name=("issuer_name", "first"), first=("period", "min"), n_rows=("n_rows", "sum")
    )
    return ThirteenF(
        {c: (p, h) for c, p, h in zip(confirmed["cusip"], confirmed["period"], confirmed["how"], strict=True)}, refused[list(REFUSED_COLUMNS)]
    )


def line_owners(issuers: Mapping[str, Issuer], view: LineageView) -> dict[str, str]:
    """`{cusip: universe ticker}` of every CUSIP whose issuer is a universe company."""
    owners = {cusip: view.ticker(issuer.primary) for cusip, issuer in issuers.items()}
    return {cusip: ticker for cusip, ticker in owners.items() if ticker}


def back_extend(lines: pd.DataFrame, starts: Mapping[str, tuple[pd.Timestamp, str]]) -> pd.DataFrame:
    """The earliest real line of each CUSIP starts at the CUSIP's first confirmed 13F period, never before an earlier CUSIP
    of its symbol ends; `ftd_start` keeps the traded start (the canonical order), `thirteenf` the evidence."""
    # the earliest real line of each CUSIP with 13F evidence before it
    out = lines.assign(ftd_start=lines["start"], thirteenf=None)
    first = out[~out["placeholder"] & out["cusip"].isin(set(starts))].sort_values(["cusip", "start"], kind="mergesort").groupby("cusip").head(1)

    # each starts at its evidence, floored at the end of an earlier CUSIP of its symbol
    for i, line in zip(first.index, _records(first), strict=True):
        earlier = out[out["key"].eq(line.key) & out["cusip"].ne(line.cusip) & out["start"].lt(line.start)]
        begin = max(starts[line.cusip][0], max((min(end, line.start) for end in earlier["end"]), default=_NEAR))
        if begin < line.start:
            out.at[i, "start"], out.at[i, "thirteenf"] = begin, f"13F from {begin.date()} ({starts[line.cusip][1]})"
    return out


# Facts and roles
def _cusip_fact(cusip: str, group: pd.DataFrame, issuer: Issuer, company: str, ctx: _FactContext) -> CusipFact:
    """One attributed CUSIP's facts: class letter, canonical priority, non-common kind and class evidence from its lines,
    its company's symbols and tape intervals, and the manual config."""
    # the class letter: of the descriptions, else of the company's class spellings of these keys
    entity, keys = ctx.view.entity(issuer.primary), set(group["key"])
    letters = {letter for found in group["letters"] for letter in found}
    spelled = {letter for letter, key in ctx.class_keys.get(entity, frozenset()) if key in keys}
    symbol_letter = len(letters) == 0 and len(spelled) == 1
    letters = letters or spelled
    letter = next(iter(letters)) if len(letters) == 1 else None

    # each common line's canonical priority, each other line's kind (its symbol read only when it is not the company's)
    tape = ctx.view.tape[ctx.view.tape["cik"].isin([c for c in ctx.view.windows if ctx.view.entity(c) == entity])]
    own = ctx.own_keys.get(entity, frozenset())
    priorities: list[int] = []
    kinds: list[tuple[int, str]] = []
    for line in group.itertuples(index=False):
        on_tape = not tape[(tape["key"] == line.key) & (tape["valid_from"] <= line.last) & (tape["valid_to"] > line.first)].empty
        kind = (line.kind_plain if (on_tape or line.key in own) else line.kind_symbol) or line.kind_any
        if line.key == ctx.company_keys.get(company) or not kind:
            priorities += (
                [0] if line.key == ctx.company_keys.get(company) else [1 if letter in (None, "A") else 3] if on_tape else [2] if letter == "A" else []
            )
            continue
        kinds.append((line.n_obs, kind))
    priority = min(priorities) if priorities else None
    kind = (
        ctx.manual_class[cusip] if ctx.manual_class.get(cusip) in NON_COMMON_KINDS else None if priority == 0 else (max(kinds)[1] if kinds else None)
    )

    # why it is a common class: manual config, a lineage class symbol, its description, the SEC listing or FINRA symbology
    evidence = (
        "manual_class" if cusip in ctx.manual_class
        else "lineage_class_symbol" if letter and symbol_letter
        else "class_description" if letter
        else "sec_tickers_listing" if any((c, k) in ctx.listed for c, _, _ in issuer.spans for k in group["key"])
        else "finra_symbology" if any(k in ctx.finra for k in group["key"])
        else None
    )  # fmt: skip
    manual = tuple((a, b, role, reason, source, cik or issuer.primary) for a, b, role, reason, source, cik in ctx.bounds.get(cusip, []))
    return CusipFact(
        cusip, issuer, company, group["start"].min(), max(group["end"]), group["ftd_start"].min(), kind, letter, priority, evidence, manual
    )


def cusip_facts(
    lines: pd.DataFrame, issuers: Mapping[str, Issuer], view: LineageView, manual: SecurityManual, listing: SecListing, finra: Finra
) -> list[CusipFact]:
    """Kind, class letter, canonical priority and class evidence of each attributed CUSIP (placeholder lines left out)."""
    # per real line: its class letters and its non-common kinds (with and without its symbol, of any description)
    real = lines[~lines["placeholder"]]
    kinds = {
        "kind_symbol": [non_common_kind(d, s) for d, s in zip(real["description"], real["source_symbol"], strict=True)],
        "kind_plain": [non_common_kind(d) for d in real["description"]],
        "kind_any": [next((k for k in (non_common_kind(d) for d in descs) if k), None) for descs in real["descriptions"]],
    }
    letters = [frozenset(found for d in descs if (found := class_letter(d))) for descs in real["descriptions"]]
    real = real.assign(letters=letters, **{column: pd.Series(values, index=real.index, dtype=object) for column, values in kinds.items()})

    # per company: its squashed symbols and class spellings; the manual classes and boundaries per CUSIP
    bounds: dict[str, list[Bound]] = {}
    for r in _records(manual.boundaries):
        bounds.setdefault(r.cusip, []).append(
            (_bound(r.valid_from) or _NEAR, _bound(r.valid_to) or _FAR, r.role, r.reason or "manual_boundary", str(r.source), r.issuer_cik or "")
        )
    ctx = _FactContext(
        view=view,
        company_keys={ticker: squash(ticker) for ticker in view.ticker_by_entity.values()},
        own_keys={entity: frozenset(squash(s) for s in symbols) for entity, symbols in view.symbols_by_entity.items()},
        class_keys={
            entity: frozenset((m.group(1), squash(s)) for s in symbols if (m := _CLASS_SYMBOL.match(str(s))))
            for entity, symbols in view.symbols_by_entity.items()
        },
        manual_class=dict(zip(manual.classes["cusip"], manual.classes["security_class"], strict=True)),
        bounds=bounds,
        listed=listing.classes,
        finra=finra.classes,
    )

    # one fact per CUSIP whose issuer is a universe company
    out: list[CusipFact] = []
    for key, group in real.groupby("cusip", sort=True):
        issuer = issuers.get(str(key))
        company = None if issuer is None else view.ticker(issuer.primary)
        if issuer is not None and company is not None:
            out.append(_cusip_fact(str(key), group, issuer, company, ctx))
    return out


def _window_spans(windows: tuple[Span, ...], start: pd.Timestamp, end: pd.Timestamp) -> list[Span]:
    """In-window parts of `[start, end)`; a bound within the seam tolerance of the life's edge moves to that edge."""
    out: list[Span] = []
    for lo, hi in windows:
        lo = (start if lo - start <= SEAM_TOLERANCE else (end if end - lo <= SEAM_TOLERANCE else lo)) if start < lo < end else lo
        hi = (end if end - hi <= SEAM_TOLERANCE else (start if hi - start <= SEAM_TOLERANCE else hi)) if start < hi < end else hi
        out += [(max(start, lo), min(end, hi))] if min(end, hi) > max(start, lo) else []
    return out


def _span_roles(
    fact: CusipFact, cik: str, has_window: bool, inside: list[Span], outside: list[Span], early: bool
) -> tuple[list[_Segment], list[_Candidate]]:
    """One issuer span's roles: acquired outside the CIK's windows (all of it for an event-only CIK); inside, a canonical
    candidate by priority (`early` for the 13F-only life), else excluded by kind, a secondary class with class evidence,
    or excluded as unclassified."""
    if not has_window:
        return [_Segment(lo, hi, ACQUIRED_CONSTITUENT, "event_only_cik", cik) for lo, hi in outside], []
    if fact.priority is not None and not fact.kind:
        cands = [_Candidate(fact.priority, fact.order_start, fact.cusip, cik, tuple(inside), early=early)] if inside else []
        return [], cands + ([_Candidate(FALLBACK_PRIORITY, fact.order_start, fact.cusip, cik, tuple(outside), early=early)] if outside else [])
    role, why = (EXCLUDED, fact.kind) if fact.kind else (SECONDARY_CLASS, fact.class_evidence) if fact.class_evidence else (EXCLUDED, "unclassified")
    acquired = [_Segment(lo, hi, ACQUIRED_CONSTITUENT, "outside_window", cik) for lo, hi in outside]
    return [_Segment(lo, hi, role, cast(str, why), cik) for lo, hi in inside] + acquired, []


def role_segments(
    facts: list[CusipFact], view: LineageView, co_registrants: frozenset[str]
) -> tuple[dict[str, list[_Segment]], dict[str, list[_Candidate]]]:
    """Each CUSIP's non-canonical segments (manual roles, acquired outside its issuers' windows, excluded by kind, secondary
    classes) and each company's canonical candidates, over its traded life and, apart, its 13F-only life before it (X3);
    a co-registrant's CUSIPs get none."""
    # the traded life of each CUSIP, then the 13F-only life before it (the seam rules read each life's own edges)
    kept = [fact for fact in facts if fact.issuer.primary not in co_registrants]
    lives = [(fact, (fact.order_start, fact.end), False) for fact in kept]
    lives += [(fact, (fact.start, fact.order_start), True) for fact in kept if fact.start < fact.order_start]

    # each life: its manual roles, then each issuer span's roles inside and outside the CIK's windows
    segments: dict[str, list[_Segment]] = {fact.cusip: [] for fact in kept}
    candidates: dict[str, list[_Candidate]] = {}
    for fact, bounds, early in lives:
        manual = [(lo, hi, role, reason, source, cik) for a, b, role, reason, source, cik in fact.manual for lo, hi in _intersect([bounds], a, b)]
        segments[fact.cusip] += [
            _Segment(lo, hi, role, reason, cik, source) for lo, hi, role, reason, source, cik in manual if role not in CANONICAL_ROLES
        ]
        cands = [
            _Candidate(-1, fact.order_start, fact.cusip, cik, ((lo, hi),), source, early)
            for lo, hi, role, _, source, cik in manual
            if role in CANONICAL_ROLES
        ]
        free = _subtract([bounds], [(a, b) for a, b, *_ in fact.manual])
        for cik, a, b in fact.issuer.spans:
            part, windows = _intersect(free, a, b), view.windows.get(cik)
            inside = [piece for lo, hi in part for piece in _window_spans(windows, lo, hi)] if windows is not None else []
            more, found = _span_roles(fact, cik, windows is not None, inside, _subtract(part, inside), early)
            segments[fact.cusip] += more
            cands += found
        candidates.setdefault(fact.company, []).extend(cands)
    return segments, candidates


def canonical_segments(
    candidates: Mapping[str, list[_Candidate]],
    facts: Mapping[str, CusipFact],
    view: LineageView,
    segments: dict[str, list[_Segment]],
    horizon: pd.Timestamp,
) -> None:
    """Per company, traded spans before 13F-only ones, best priority first and later start first within a priority: each
    candidate takes the dates no better line holds. One overlapping better lines over most of its life is a sibling
    class; a fallback part not taken stays acquired."""
    for company, cands in candidates.items():
        covered: list[Span] = []
        roster_cik = view.roster_cik_by_ticker.get(company)
        for cand in sorted(cands, key=lambda c: (c.early, c.priority, -c.start.value, c.cusip, c.issuer)):
            evidence = facts[cand.cusip].class_evidence
            free, whole = _subtract(cand.spans, covered), _length(cand.spans, horizon)
            sibling = 0 <= cand.priority < FALLBACK_PRIORITY and whole > 0 and 1 - _length(free, horizon) / whole >= SIBLING_OVERLAP
            taken = [] if sibling else free
            role = CANONICAL_CURRENT if cand.issuer == roster_cik else CANONICAL_PREDECESSOR
            fallback, secondary = cand.priority == FALLBACK_PRIORITY, sibling and evidence
            leftover = (
                (ACQUIRED_CONSTITUENT, "outside_window") if fallback else (SECONDARY_CLASS, evidence) if secondary else (EXCLUDED, "superseded")
            )
            segments[cand.cusip] += [_Segment(a, b, role, _PRIORITY_REASON[cand.priority], cand.issuer, cand.source) for a, b in taken]
            segments[cand.cusip] += [_Segment(a, b, leftover[0], cast(str, leftover[1]), cand.issuer) for a, b in _subtract(cand.spans, taken)]
            covered += taken


# The table rows
def row_context(view: LineageView, manual: SecurityManual, listing: SecListing) -> RowContext:
    """The squashed company keys, each company's spelling of a key, the manual conversion ratios per CUSIP, the listing."""
    spelled: dict[tuple[str, str], str] = {}
    for entity, symbols in view.symbols_by_entity.items():
        for symbol in sorted(symbols, reverse=True):
            spelled[(entity, squash(symbol))] = symbol
    ratios: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, float, str]]] = {}
    for r in _records(manual.ratios):
        ratios.setdefault(r.cusip, []).append((_bound(r.valid_from) or _NEAR, _bound(r.valid_to) or _FAR, float(r.ratio), str(r.source)))
    companies = {"", *view.ticker_by_entity.values()}
    return RowContext(view=view, company_keys={c: squash(c) for c in companies}, spelled=spelled, ratios=ratios, listing=listing)


def _line_pieces(line: Any, segments: list[_Segment], issuer: str) -> list[_Segment]:
    """The line clipped to its CUSIP's role segments (a placeholder is excluded over its whole life)."""
    if line.placeholder:
        return [_Segment(line.start, line.end, EXCLUDED, "transition_placeholder", issuer)]
    return [
        _Segment(max(line.start, s.start), min(line.end, s.end), s.role, s.reason, s.issuer, s.source)
        for s in segments
        if min(line.end, s.end) > max(line.start, s.start)
    ]


def _ratio_pieces(start: pd.Timestamp, end: pd.Timestamp, ratios: Sequence[tuple[pd.Timestamp, pd.Timestamp, float, str]]) -> list[tuple[Any, ...]]:
    """`[start, end)` split at the manual conversion-ratio bounds: `(a, b, ratio, source)`, ratio 1 without one."""
    pieces: list[tuple[Any, ...]] = [
        (max(lo, start), min(hi, end), ratio, source) for lo, hi, ratio, source in ratios if min(hi, end) > max(lo, start)
    ]
    rest = [(start, end)]
    for a, b, *_ in pieces:
        rest = [piece for x, y in rest for piece in ((x, min(y, a)), (max(x, b), y)) if piece[1] > piece[0]]
    return sorted(pieces + [(a, b, 1.0, None) for a, b in rest])


def _class_of(fact: CusipFact | None, role: str, reason: str) -> str:
    """The stored class of a piece: its kind, `class_X`, `common`, or `unclassified`."""
    if fact is None or (role == EXCLUDED and reason == "unclassified" and not fact.kind):
        return "unclassified"
    if fact.kind:
        return fact.kind
    if fact.letter:
        return f"class_{fact.letter}"
    return "common" if (fact.priority is not None or role in CANONICAL_ROLES or fact.class_evidence) else "unclassified"


def _market_symbol(line: Any, company: str, issuer: str, ctx: RowContext) -> str:
    """The company ticker for its own key, else the SEC spelling, else the company's own spelling of the key, else the FTD symbol."""
    if line.key == ctx.company_keys.get(company):
        return company
    if (issuer, line.key) in ctx.listing.symbols:
        return ctx.listing.symbols[(issuer, line.key)]
    return ctx.spelled.get((ctx.view.entity(issuer), line.key), line.source_symbol)


def _merged_pieces(line: Any, company: str, pieces: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
    """Adjacent pieces `(segment, a, b, ratio, source, class, market)` with the same role, reason, ratio, class and issuer merged."""
    merged: list[dict[str, Any]] = []
    for seg, a, b, ratio, ratio_source, cls, market in pieces:
        key = (seg.role, seg.reason, ratio, cls, seg.issuer)
        if merged and merged[-1]["valid_to"] == a and merged[-1]["_key"] == key:
            merged[-1]["valid_to"] = b
            continue
        merged.append(
            {"_key": key, "security_id": f"C{line.cusip}", "canonical_company": company, "issuer_cik": seg.issuer, "source": SOURCE_FTD,
             "source_symbol": line.source_symbol, "market_symbol": market, "cusip": line.cusip, "security_class": cls, "conversion_ratio": ratio,
             "lineage_role": seg.role, "valid_from": a, "valid_to": b, "lineage_reason": seg.reason, "source_accession": seg.source or ratio_source}
        )  # fmt: skip
    return merged


def _finish_rows(merged: list[dict[str, Any]], line: Any, issuer: Issuer, days: Any, exchange: Mapping[tuple[str, str], str]) -> list[dict[str, Any]]:
    """The merged pieces as table rows: observation counts, open end as None, listing exchange of an open row, evidence text."""
    for row in merged:
        row.pop("_key")
        start, end = row["valid_from"].to_datetime64(), row["valid_to"].to_datetime64()
        row["n_observations"] = int(((days >= start) & (days < end)).sum()) if days is not None else 0
        row["valid_to"] = None if row["valid_to"] >= _FAR else row["valid_to"]
        row["exchange"] = exchange.get((row["issuer_cik"], line.key)) if row["valid_to"] is None else None
        traded = (f"; FINRA from {line.finra_from.date()}" if pd.notna(line.finra_from) else "") + (
            f"; FINRA to {line.finra_to.date()}" if pd.notna(line.finra_to) else ""
        )
        traded += f"; {line.thirteenf}" if isinstance(line.thirteenf, str) else ""
        row["evidence"] = (
            f"issuer {issuer.primary} by {issuer.how}; FTD {line.first.date()}..{line.last.date()} n={line.n_obs}{traded}; {line.description}"
        )
    return merged


def cusip_rows(lines: pd.DataFrame, obs: pd.DataFrame, roles: CusipRoles, issuers: Mapping[str, Issuer], ctx: RowContext) -> list[dict[str, Any]]:
    """Each line clipped to its CUSIP's role segments, split at conversion-ratio dates, adjacent equal pieces merged."""
    # the FTD trade dates of each line
    dates = {key: group["trade_date"].to_numpy() for key, group in obs.groupby(["cusip", "source_symbol"])}

    # each line of an attributed CUSIP, piece by piece
    out: list[dict[str, Any]] = []
    for line in _records(lines[lines["cusip"].isin(set(roles.segments))]):
        issuer = issuers[line.cusip]
        company, fact = ctx.view.ticker(issuer.primary) or "", roles.facts.get(line.cusip)
        segments = sorted(_line_pieces(line, roles.segments[line.cusip], issuer.primary), key=lambda s: s.start)
        split = [(seg, *piece) for seg in segments for piece in _ratio_pieces(seg.start, seg.end, ctx.ratios.get(line.cusip, []))]
        pieces = [(*piece, _class_of(fact, piece[0].role, piece[0].reason), _market_symbol(line, company, piece[0].issuer, ctx)) for piece in split]
        out += _finish_rows(_merged_pieces(line, company, pieces), line, issuer, dates.get((line.cusip, line.source_symbol)), ctx.listing.exchange)
    return out


def _signatures(frame: pd.DataFrame) -> dict[str, tuple[tuple[str, ...], ...]]:
    """`{company: sorted signature rows}` of the compared columns, dates as text."""
    text = frame[list(_SIGNATURE)].copy()
    for column in ("valid_from", "valid_to"):
        text[column] = pd.to_datetime(text[column]).dt.strftime("%Y-%m-%d").fillna("")
    text = text.astype(str)
    return {str(company): tuple(sorted(map(tuple, group.to_numpy()))) for company, group in text.groupby(frame["canonical_company"].astype(str))}


def _closed_at_end(frame: pd.DataFrame, ended_on: Mapping[str, pd.Timestamp]) -> pd.Series:
    """`valid_to` with every open line of an ended company closed the day after its end, never before the line's start."""
    ends = frame["canonical_company"].map(lambda company: ended_on.get(str(company)))
    close = frame["valid_to"].isna() & ends.notna()
    starts = frame.loc[close, "valid_from"] + pd.Timedelta(days=1)
    closed = [max(pd.Timestamp(end) + pd.Timedelta(days=1), start) for end, start in zip(ends[close], starts, strict=True)]
    return frame["valid_to"].where(~close, pd.Series(closed, index=ends[close].index, dtype="datetime64[ns]"))


def finalise_rows(
    rows: list[dict[str, Any]], existing: pd.DataFrame | None, built_at: pd.Timestamp, ended_on: Mapping[str, pd.Timestamp]
) -> pd.DataFrame:
    """The typed table, sorted, an ended company's open lines closed the day after its `ended_on` (REQ-006, before the
    stamps); each company keeps its stored `scope_changed_at` while its rows are unchanged."""
    # the typed rows
    frame = pd.DataFrame(rows, columns=[c for c in TABLE_COLUMNS if c != "scope_changed_at"])
    frame["valid_from"] = pd.to_datetime(frame["valid_from"])
    frame["valid_to"] = pd.to_datetime(frame["valid_to"])
    frame["conversion_ratio"] = frame["conversion_ratio"].astype("float64")
    frame["n_observations"] = frame["n_observations"].astype("int64")
    for column in ("exchange", "source_accession", "canonical_company"):
        frame[column] = frame[column].astype(object).where(frame[column].notna(), None)
    frame["valid_to"] = _closed_at_end(frame, ended_on)

    # the stored stamp of each company whose rows are unchanged; `built_at` for every other
    companies = frame["canonical_company"].astype(str)
    stamps = pd.Series(built_at, index=frame.index, dtype="datetime64[ns]")
    if existing is not None and not existing.empty and "scope_changed_at" in existing.columns:
        before, after = _signatures(existing), _signatures(frame)
        stored = pd.to_datetime(existing["scope_changed_at"]).groupby(existing["canonical_company"].astype(str)).max().to_dict()
        kept = {c: stored[c] for c in set(companies) if before.get(c) == after.get(c) and pd.notna(stored.get(c))}
        stamps = pd.Series([kept.get(c, built_at) for c in companies], index=frame.index, dtype="datetime64[ns]")
    frame["scope_changed_at"] = stamps
    return frame[list(TABLE_COLUMNS)].sort_values(
        ["canonical_company", "security_id", "source", "source_symbol", "valid_from"], kind="mergesort", ignore_index=True
    )


# Checks and flags
def _symbol_conflicts(kept: pd.DataFrame, rows: pd.DataFrame, flags: CusipFlags) -> None:
    """A (settlement date, FTD symbol) under two CUSIPs, among the master's real lines."""
    clash = kept[kept.groupby(["date", "key"])["cusip"].transform("nunique") > 1]
    counts = clash.groupby(["date", "key"])["cusip"].agg(lambda s: tuple(sorted(set(s))))
    company = rows.drop_duplicates("cusip").set_index("cusip")["canonical_company"].to_dict()
    for (day, key), cusips in cast(Any, counts).items():
        symbol = kept.loc[kept["key"].eq(key), "source_symbol"].iloc[0]
        flags.add(
            "security_symbol_conflict",
            (company.get(cusips[0]), None, cusips[0], symbol),
            (day, day, len(cusips)),
            "",
            f"{symbol} on {day.date()} under " + ", ".join(cusips),
        )


def cusip_flags(flags: CusipFlags, roles: CusipRoles, lines: pd.DataFrame, obs: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """The flags of the build: issuer votes (found before), unclassified CUSIPs and FTD symbols under two CUSIPs."""
    # a CUSIP left unclassified somewhere
    for fact in roles.facts.values():
        if not any(s.reason == "unclassified" for s in roles.segments.get(fact.cusip, [])):
            continue
        part = lines[lines["cusip"].eq(fact.cusip) & ~lines["placeholder"]]
        life = (part["first"].min(), part["last"].max(), int(part["n_obs"].sum()))
        flags.add(
            "security_unclassified",
            (fact.company, fact.issuer.primary, fact.cusip, " ".join(part["source_symbol"])),
            life,
            part["description"].iloc[0],
            "no class evidence",
        )

    # one FTD symbol under two CUSIPs on a date
    real = obs[obs["cusip"].isin(set(rows["cusip"])) & ~obs["source_symbol"].map(is_placeholder)]
    _symbol_conflicts(real, rows, flags)
    return pd.DataFrame(flags.rows, columns=list(FLAG_DETAIL_COLUMNS))


def _ratio_median(series: pd.Series, canonical: pd.Series) -> pd.Series:
    """The rolling-median price ratio of a secondary class to its company's canonical line, on their common days."""
    both = pd.concat([series.rename("s"), canonical.rename("c")], axis=1, join="inner").sort_index()
    if len(both) < RATIO_WINDOW:
        return pd.Series(dtype="float64")
    return (both["s"] / both["c"]).rolling(RATIO_WINDOW, min_periods=RATIO_WINDOW).median().dropna()


def _step_factor(median: pd.Series, change: pd.Series, start: pd.Timestamp, stop: pd.Timestamp) -> float:
    """The ratio step across a cluster: the median after it over the median before it (the one-day change at the edges)."""
    before, after = median[median.index < start], median[median.index >= stop]
    return float(after.iloc[0] / before.iloc[-1]) if len(before) and len(after) else float(change.loc[start])


def _clusters(days: list[pd.Timestamp]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Days grouped into runs no more than `RATIO_MATCH_DAYS` apart: `(first, last)` of each run."""
    out: list[list[pd.Timestamp]] = []
    for day in days:
        if out and day - out[-1][1] <= RATIO_MATCH_DAYS:
            out[-1][1] = day
        else:
            out.append([day, day])
    return [(a, b) for a, b in out]


def _configured(cusip: str, start: pd.Timestamp, stop: pd.Timestamp, factor: float, ratios: pd.DataFrame) -> bool:
    """A configured conversion ratio of the CUSIP changes near the step by about the same factor."""
    rows = ratios[ratios["cusip"].eq(cusip)].sort_values("valid_from", na_position="first")
    for prev, nxt in zip(rows.itertuples(index=False), rows.iloc[1:].itertuples(index=False), strict=False):
        near = pd.notna(nxt.valid_from) and start - RATIO_MATCH_DAYS <= nxt.valid_from <= stop + RATIO_MATCH_DAYS
        if near and 1 / RATIO_STEP < factor / (nxt.ratio / prev.ratio) < RATIO_STEP:
            return True
    return False


def ratio_step_flags(obs: pd.DataFrame, rows: pd.DataFrame, ratios: pd.DataFrame) -> pd.DataFrame:
    """Steps of more than `RATIO_STEP` in the rolling-median price ratio of a secondary class to the canonical line; a
    step is accepted when a configured conversion ratio of that class changes within `RATIO_MATCH_DAYS` by about the
    same factor. The level is never checked."""
    # the priced FTD days of the canonical and secondary-class lines, each inside its row
    priced = obs[obs["price"].gt(0)][["cusip", "source_symbol", "trade_date", "price"]]
    spans = rows[rows["lineage_role"].isin((*CANONICAL_ROLES, SECONDARY_CLASS))][
        ["cusip", "source_symbol", "canonical_company", "lineage_role", "valid_from", "valid_to"]
    ]
    tagged = priced.merge(spans, on=["cusip", "source_symbol"])
    tagged = tagged[(tagged["trade_date"] >= tagged["valid_from"]) & (tagged["trade_date"] < tagged["valid_to"].fillna(_FAR))]
    canonical = tagged[tagged["lineage_role"].isin(CANONICAL_ROLES)].groupby(["canonical_company", "trade_date"])["price"].median()
    secondary = tagged[tagged["lineage_role"].eq(SECONDARY_CLASS)].groupby(["canonical_company", "cusip", "trade_date"])["price"].median()

    # each secondary class's steps: clustered days, the factor across each cluster
    steps: list[tuple[str, str, pd.Timestamp, pd.Timestamp, float, int]] = []
    for (company, cusip), series in cast(Any, secondary.groupby(level=[0, 1])):
        reference = canonical.loc[company] if company in canonical.index.get_level_values(0) else pd.Series(dtype="float64")
        median = _ratio_median(series.droplevel([0, 1]), reference)
        change = median / median.shift(1)
        n = len(pd.concat([series.droplevel([0, 1]), reference], axis=1, join="inner"))
        clusters = _clusters(list(change[(change > RATIO_STEP) | (change < 1 / RATIO_STEP)].index))
        steps += [(company, cusip, start, stop, _step_factor(median, change, start, stop), n) for start, stop in clusters]

    # the steps no configured ratio change explains
    flags = CusipFlags()
    for company, cusip, start, stop, factor, n in steps:
        if not _configured(cusip, start, stop, factor, ratios):
            flags.add(
                "security_ratio_step",
                (company, None, cusip, ""),
                (start, stop, n),
                "",
                f"price-ratio median x{factor:.3g} with no configured ratio change",
            )
    return pd.DataFrame(flags.rows, columns=list(FLAG_DETAIL_COLUMNS))


def log_cusip_history(rows: pd.DataFrame, detail: pd.DataFrame, thirteenf: ThirteenF, co_registrants: frozenset[str]) -> None:
    """One INFO summary: rows, CUSIPs, companies, roles, the 13F back-extension, co-registrant CIKs and flag counts."""
    extended = rows[rows["evidence"].str.contains("; 13F from ", regex=False)]
    logger.info(
        "universe_cusip: %d row(s) over %d CUSIP(s) and %d company(ies); roles %s; %d CUSIP(s) run back to their first confirmed 13F period "
        "(%d refused name match(es) on a foreign CUSIP-6); %d co-registrant CIK(s) not stored; flags %s",
        len(rows),
        rows["security_id"].nunique(),
        rows["canonical_company"].nunique(),
        rows.groupby("lineage_role")["security_id"].nunique().to_dict(),
        extended["cusip"].nunique(),
        thirteenf.refused["cusip"].nunique(),
        len(co_registrants),
        detail["kind"].value_counts().sort_index().to_dict(),
    )


def flag_items(flags: pd.DataFrame, ratio_steps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The flags sorted by kind, company, CUSIP and date, and one WARNING-block item (`FLAG_COLUMNS`) per kind."""
    # every flag, sorted
    detail = pd.concat([flags, ratio_steps], ignore_index=True)
    detail = detail.sort_values(["kind", "canonical_company", "cusip", "first"], kind="mergesort", na_position="last", ignore_index=True)

    # one item per kind: count, companies and the first examples
    items = []
    for kind, (action, suggested) in FLAG_KINDS.items():
        part = detail[detail["kind"].eq(kind)]
        if part.empty:
            continue
        examples = "; ".join(f"{r.cusip} {r.source_symbol} ({r.canonical_company}) {r.detail}" for r in part.head(12).itertuples(index=False))
        items.append(
            {"kind": kind, "action": action, "ticker": ",".join(sorted({str(t) for t in part["canonical_company"].dropna()})),
             "ciks": ",".join(sorted({str(c) for c in part["issuer_cik"].dropna()})), "evidence": f"{len(part)} line(s): {examples}" + (" ..." if len(part) > 12 else ""),
             "suggested_action": suggested, "config_file": f"configs/{SECURITY_MANUAL_FILE.as_posix()}"}
        )  # fmt: skip
    return detail, pd.DataFrame(items, columns=list(FLAG_COLUMNS))
