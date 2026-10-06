"""Security master: which security a market-tape line is, its issuer CIK, class and lineage role, dated.

`derive_security_master` is pure: FTD lines, `entity_lineage`, the roster, the `sec_company_tickers` snapshot
and `configs/sec/security_master_manual.json` in; table rows and flags out. The issuer of each CUSIP-9 comes
from lineage symbol-interval votes (inherited across a CUSIP-6 within the voters' era), the class needs
positive evidence, and the role follows the issuer CIK's lineage window: one canonical line per company
and date. `build_security_master` reads the inputs through the store and replaces the table unless unchanged.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd

from src.constants.constants import CANONICAL_CURRENT, CANONICAL_PREDECESSOR, CANONICAL_ROLES, SECONDARY_CLASS
from src.context import Context
from src.data_extract.utils.common.config_paths import resolve_config_dir
from src.data_extract.utils.common.entity_lineage import (
    ROLE_SYMBOL,
    ROLE_WINDOW,
    ROSTER_COLUMNS,
    SENTINEL_START,
    entity_by_cik_map,
    entity_or_singleton,
    roster_cik_map,
)
from src.data_extract.utils.common.incremental import matches_stored
from src.data_extract.utils.common.symbol_tenure import DEI_SOURCE
from src.data_store.schema import Tables
from src.utils.cutover_continuity import ShareExchange
from src.utils.identity_flags import FLAG_COLUMNS, cik_activity, identity_flags, log_identity_flags
from src.utils.string import normalise_ticker, pad_cik, pad_cik_series

MANUAL_CONFIG_FILENAME = "security_master_manual.json"
#: The stored FTD line columns the master derives from.
_OBSERVATION_COLUMNS = ("date", "trade_date", "cusip", "source_symbol", "description", "price", "period")

SOURCE_FTD = "ftd"
ACQUIRED_CONSTITUENT = "acquired_constituent"
EXCLUDED = "excluded"
ROLES = (*CANONICAL_ROLES, SECONDARY_CLASS, ACQUIRED_CONSTITUENT, EXCLUDED)
NON_COMMON_KINDS = ("preferred", "debt", "warrant", "unit", "right")

#: First trade date of each settlement cycle (T+2 from 2017-09-05, T+1 from 2024-05-28).
T2_FROM = pd.Timestamp("2017-09-05")
T1_FROM = pd.Timestamp("2024-05-28")
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
#: Flag kinds: (action, suggested action) for the WARNING block.
FLAG_KINDS: dict[str, tuple[bool, str]] = {
    "security_issuer_conflict": (True, "lineage symbol intervals of two companies vote for one CUSIP: add an issuer override"),
    "security_ratio_step": (True, "the class price ratio steps with no configured conversion-ratio change: check for a split"),
    "security_straddle": (False, "one CUSIP traded across a CIK seam: issuer dated by the company's lineage windows"),
    "security_weak_vote": (False, "a lineage symbol interval voted only at the start of a longer-lived CUSIP (seam lag or reuse): vote ignored"),
    "security_symbol_conflict": (False, "one FTD symbol under two CUSIPs on a date: symbol lookups stay unresolved, CUSIP rows are exact"),
    "security_unclassified": (False, "no positive class evidence: excluded; add a class override if it is a common class"),
}

_FAR = pd.Timestamp("2262-01-01")
_NEAR = pd.Timestamp("1700-01-01")
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
_WARRANT_SYMBOL = re.compile(r"^[A-Z]{1,5}WS[A-Z]?$")
_LETTER = re.compile(
    r"\bCL(?:ASS)?[\s\-.']*([A-C])\b|\bSER(?:IES)?\s+([A-C])\s+COM|\(HLDG CO\)\s*([A-C])\b|\bSPL\s+([A-C])\b|\bCOM(?:MON)?\s+([A-C])$",
    re.I,
)

#: (start, end) half-open; `_FAR` stands for an open end.
Span = tuple[pd.Timestamp, pd.Timestamp]


class SecurityManualError(ValueError):
    """A `security_master_manual.json` entry without a URL or accession, or with an unknown role."""


# --------------------------------------------------------------------------- small pure helpers


def squash(symbol: object) -> str:
    """The comparison key of a symbol: upper case without class separators (`BRK-B`, `BRK/B`, `BRK.B` -> `BRKB`)."""
    if symbol is None or (not isinstance(symbol, str) and pd.isna(cast(Any, symbol))):
        return ""
    return re.sub(r"[\s\-./]", "", str(symbol)).upper()


def trade_dates(settlement: pd.Series) -> pd.Series:
    """Trade dates of FTD settlement dates by the cycle in force (T+3, T+2 from 2017-09-05, T+1 from 2024-05-28)."""
    days = pd.to_datetime(settlement)
    bday = pd.offsets.BDay
    t1, t2, t3 = days - bday(1), days - bday(2), days - bday(3)
    out = t3.where(t2 < T2_FROM, t2).where(t1 < T1_FROM, t1)
    return out.where(days.notna())


def is_placeholder(symbol: str) -> bool:
    """FTD transition placeholders at a CUSIP change, distribution or delisting (`XOMZZZZ`, `GOOGLXXXX`, `3106PS`, `C117REGWAY`)."""
    return bool(_PLACEHOLDER.search(str(symbol)))


def non_common_kind(description: str, symbol: str = "") -> str | None:
    """`warrant`, `right`, `unit`, `debt` or `preferred` from word-bounded description or symbol evidence; None otherwise."""
    text, sym = str(description or ""), str(symbol or "").upper()
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
    return [(max(a, start), min(b, end)) for a, b in spans if min(b, end) > max(a, start)]


def _subtract(spans: Iterable[Span], cut: Iterable[Span]) -> list[Span]:
    out = list(spans)
    for c_from, c_to in cut:
        out = [piece for a, b in out for piece in ((a, min(b, c_from)), (max(a, c_to), b)) if piece[1] > piece[0]]
    return out


def _length(spans: Iterable[Span], horizon: pd.Timestamp) -> float:
    """Days covered by `spans` up to `horizon` (an open end counts to the last observation, not to `_FAR`)."""
    return float(sum((min(b, horizon) - a).days for a, b in spans if min(b, horizon) > a))


# --------------------------------------------------------------------------- manual config


@dataclass(frozen=True)
class SecurityManual:
    """Parsed `security_master_manual.json`: dated conversion ratios, CUSIP market boundaries, class overrides, merger
    metadata, the declared co-registrant CIKs and the merger exchange ratios."""

    ratios: pd.DataFrame
    boundaries: pd.DataFrame
    classes: pd.DataFrame
    mergers: tuple[dict[str, Any], ...] = ()
    co_registrants: tuple[str, ...] = ()
    exchanges: tuple[ShareExchange, ...] = ()

    @classmethod
    def empty(cls) -> SecurityManual:
        return parse_security_manual({})


_RATIO_COLUMNS = ("ticker", "cusip", "ratio", "valid_from", "valid_to", "source")
_BOUNDARY_COLUMNS = ("ticker", "cusip", "issuer_cik", "role", "valid_from", "valid_to", "reason", "source")
_CLASS_COLUMNS = ("cusip", "security_class", "source")
_EXCHANGE_COLUMNS = ("ticker", "predecessor_cik", "seam_date", "ratio", "source")


def _entries(blob: Mapping[str, Any], key: str, columns: tuple[str, ...]) -> pd.DataFrame:
    rows = list(blob.get(key) or [])
    for row in rows:
        if not str(row.get("source") or "").strip():
            raise SecurityManualError(f"security_master_manual.json: {key} entry {row} has no source (URL or accession)")
    frame = pd.DataFrame([{column: row.get(column) for column in columns} for row in rows], columns=list(columns))
    for column in ("valid_from", "valid_to"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column])
    if "cusip" in frame.columns:
        frame["cusip"] = frame["cusip"].astype("string").str.upper().astype(object)
    return frame


def parse_security_manual(blob: Mapping[str, Any]) -> SecurityManual:
    """Validate and parse the manual config; every entry must cite a URL or an accession."""
    boundaries = _entries(blob, "market_boundaries", _BOUNDARY_COLUMNS)
    unknown = set(boundaries["role"].dropna()) - set(ROLES)
    if unknown:
        raise SecurityManualError(f"security_master_manual.json: unknown role(s) {sorted(unknown)}")
    boundaries["issuer_cik"] = pad_cik_series(boundaries["issuer_cik"]) if not boundaries.empty else boundaries["issuer_cik"]
    ratios = _entries(blob, "conversion_ratios", _RATIO_COLUMNS)
    ratios["ratio"] = pd.to_numeric(ratios["ratio"]).astype("float64")
    mergers = tuple(blob.get("merger_metadata") or [])
    for entry in mergers:
        if not str(entry.get("source") or "").strip():
            raise SecurityManualError(f"security_master_manual.json: merger entry {entry} has no source (URL or accession)")
    co_registrants = _entries(blob, "co_registrants", ("cik", "ticker", "source"))
    return SecurityManual(
        ratios=ratios,
        boundaries=boundaries,
        classes=_entries(blob, "class_overrides", _CLASS_COLUMNS),
        mergers=mergers,
        co_registrants=tuple(sorted(set(pad_cik_series(co_registrants["cik"])))) if not co_registrants.empty else (),
        exchanges=_exchanges(_entries(blob, "exchange_ratios", _EXCHANGE_COLUMNS)),
    )


def _exchanges(rows: pd.DataFrame) -> tuple[ShareExchange, ...]:
    """The `exchange_ratios` entries; a ratio that is not a positive number is refused."""
    out = []
    for row in rows.itertuples(index=False):
        ratio = pd.to_numeric(row.ratio, errors="coerce")
        if pd.isna(ratio) or float(ratio) <= 0:
            raise SecurityManualError(f"security_master_manual.json: exchange ratio {row} is not a positive number")
        out.append(ShareExchange(normalise_ticker(str(row.ticker)), pad_cik(row.predecessor_cik), pd.Timestamp(str(row.seam_date)), float(ratio)))
    return tuple(out)


@dataclass(frozen=True)
class MergerBoundary:
    """One merger's boundary day for the insider lineage: who acquired whom on `seam_date`, and whether it closed after the market close."""

    ticker: str
    seam_date: pd.Timestamp
    after_close: bool | None
    legal_acquirer_cik: str
    accounting_predecessor_cik: str
    acquired_symbol: str


def merger_boundaries(manual: SecurityManual) -> tuple[MergerBoundary, ...]:
    """The manual `merger_metadata` entries as `MergerBoundary` rows; a null `closing_time` leaves `after_close` unknown."""
    out = []
    for entry in manual.mergers:
        closing = str(entry.get("closing_time") or "").strip().lower()
        out.append(
            MergerBoundary(
                ticker=normalise_ticker(str(entry["ticker"])),
                seam_date=pd.Timestamp(entry["seam_date"]),
                after_close=("after" in closing) if closing else None,
                legal_acquirer_cik=pad_cik(entry["legal_acquirer_cik"]),
                accounting_predecessor_cik=pad_cik(entry["accounting_predecessor_cik"]),
                acquired_symbol=str(entry["acquired_symbol"]).strip().upper(),
            )
        )
    return tuple(out)


def load_security_manual(config_dir: str | None = None) -> SecurityManual:
    """`configs/sec/security_master_manual.json`, parsed; an absent file is an empty config."""
    path = Path(resolve_config_dir(config_dir)) / "sec" / MANUAL_CONFIG_FILENAME
    if not path.exists():
        return SecurityManual.empty()
    return parse_security_manual(json.loads(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- lineage view


@dataclass(frozen=True)
class _Lineage:
    """The lineage facts the master reads: entities, windows, roster CIKs and tape symbol intervals."""

    entity_by_cik: Mapping[str, str]
    ticker_by_entity: Mapping[str, str]
    roster_cik_by_ticker: Mapping[str, str]
    windows: Mapping[str, tuple[Span, ...]]
    tape: pd.DataFrame
    symbols_by_entity: Mapping[str, frozenset[str]] = field(default_factory=dict)

    def entity(self, cik: str) -> str:
        return entity_or_singleton(self.entity_by_cik, cik)

    def ticker(self, cik: str) -> str | None:
        return self.ticker_by_entity.get(self.entity(cik))

    def window_owners(self, entity: str) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
        """`(cik, start, end)` of every window of an entity's CIKs, oldest first."""
        owners = [(cik, a, b) for cik, spans in self.windows.items() if self.entity(cik) == entity for a, b in spans]
        return sorted(owners, key=lambda o: (o[1], o[0]))


def _tape_intervals(lineage: pd.DataFrame) -> pd.DataFrame:
    """`[key, cik, valid_from, valid_to]` of the symbol rows a tape may read: not noise or conflict, not `dei` alone."""
    symbols = lineage[lineage["role"].eq(ROLE_SYMBOL)].copy()
    symbols["key"] = symbols["symbol"].map(squash)
    symbols["cik"] = pad_cik_series(symbols["cik"])
    sources = symbols["sources"].fillna("").astype(str).str.lower()
    tape_ok = ~symbols["status"].isin(("noise", "conflict")) & sources.ne(DEI_SOURCE) & symbols["key"].ne("")
    tape = symbols.loc[tape_ok, ["key", "cik", "valid_from", "valid_to"]].copy()
    tape["valid_from"] = pd.to_datetime(tape["valid_from"])
    tape["valid_to"] = pd.to_datetime(tape["valid_to"]).fillna(_FAR)
    return tape.sort_values(["key", "cik", "valid_from"], ignore_index=True)


def _lineage_view(lineage: pd.DataFrame, roster: pd.DataFrame) -> _Lineage:
    rows = lineage.copy()
    rows["cik"] = pad_cik_series(rows["cik"])
    entity_by_cik = entity_by_cik_map(rows)
    roster_cik = roster_cik_map(roster)
    ticker_by_entity = {entity_or_singleton(entity_by_cik, cik): ticker for ticker, cik in sorted(roster_cik.items())}
    windows: dict[str, list[Span]] = {}
    for cik, start, end in rows.loc[rows["role"].eq(ROLE_WINDOW), ["cik", "valid_from", "valid_to"]].to_numpy().tolist():
        windows.setdefault(cik, []).append((_bound(start) or _NEAR, _bound(end) or _FAR))
    for cik in roster_cik.values():
        windows.setdefault(cik, [(_NEAR, _FAR)])
    symbols = rows[rows["role"].eq(ROLE_SYMBOL)]
    symbols_by_entity = symbols.groupby("entity_id")["symbol"].agg(lambda s: frozenset(map(str, s))).to_dict()
    return _Lineage(
        entity_by_cik=entity_by_cik,
        ticker_by_entity=ticker_by_entity,
        roster_cik_by_ticker=roster_cik,
        windows={cik: tuple(sorted(spans)) for cik, spans in windows.items()},
        tape=_tape_intervals(rows),
        symbols_by_entity={str(k): v for k, v in symbols_by_entity.items()},
    )


def lineage_scope_symbols(lineage: pd.DataFrame) -> frozenset[str]:
    """Squashed symbols of every non-noise lineage symbol row: the symbol half of the FTD ingest scope."""
    rows = lineage[lineage["role"].eq(ROLE_SYMBOL) & lineage["status"].ne("noise")]
    return frozenset(key for key in rows["symbol"].map(squash) if key)


def cusip_votes(observations: pd.DataFrame, lineage: pd.DataFrame) -> pd.DataFrame:
    """`[cusip, cik, votes, first, last]`: FTD (cusip, symbol, trade date) triples inside a tape lineage interval, per CIK."""
    return _votes(_keyed(observations), _tape_intervals(lineage))


def _keyed(observations: pd.DataFrame) -> pd.DataFrame:
    """Distinct `(cusip, key, trade_date)` of the observations."""
    obs = observations
    trade = obs["trade_date"] if "trade_date" in obs.columns else trade_dates(obs["date"])
    frame = pd.DataFrame(
        {"cusip": obs["cusip"].astype(str).str.upper(), "key": obs["source_symbol"].map(squash), "trade_date": pd.to_datetime(trade)}
    )
    return frame.drop_duplicates(ignore_index=True)


def _votes(keyed: pd.DataFrame, tape: pd.DataFrame) -> pd.DataFrame:
    merged = keyed.merge(tape, on="key", how="inner")
    merged = merged[(merged["trade_date"] >= merged["valid_from"]) & (merged["trade_date"] < merged["valid_to"])]
    if merged.empty:
        return pd.DataFrame(columns=["cusip", "cik", "votes", "first", "last"])
    return (
        merged.groupby(["cusip", "cik"], as_index=False)
        .agg(votes=("trade_date", "size"), first=("trade_date", "min"), last=("trade_date", "max"))
        .sort_values(["cusip", "cik"], ignore_index=True)
    )


# --------------------------------------------------------------------------- lines


def _prepare(observations: pd.DataFrame) -> pd.DataFrame:
    obs = observations.copy()
    obs["date"] = pd.to_datetime(obs["date"])
    obs["trade_date"] = pd.to_datetime(obs["trade_date"]) if "trade_date" in obs.columns else trade_dates(obs["date"])
    obs["cusip"] = obs["cusip"].astype(str).str.strip().str.upper()
    obs["source_symbol"] = obs["source_symbol"].fillna("").astype(str).str.strip().str.upper()
    obs["description"] = obs["description"].fillna("").astype(str).str.strip()
    obs["key"] = obs["source_symbol"].map(squash)
    obs["price"] = pd.to_numeric(obs["price"], errors="coerce") if "price" in obs.columns else float("nan")
    obs["period"] = obs["period"].astype(str) if "period" in obs.columns else ""
    return obs.sort_values(["cusip", "source_symbol", "trade_date", "date"], kind="mergesort", ignore_index=True)


def _lines(obs: pd.DataFrame) -> pd.DataFrame:
    """One row per (cusip, source_symbol): life in trade dates, observations, descriptions, latest period."""
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
    return lines


def _line_ends(
    lines: pd.DataFrame, current_periods: frozenset[str], listed: frozenset[tuple[str, str]], issuer: Mapping[str, str]
) -> list[pd.Timestamp]:
    """Exclusive end of each line: open (`_FAR`) when current, bridged to the next CUSIP of its symbol, else the day after its last row.

    Being current (recent fails or a current SEC listing, which names a symbol, not a CUSIP) keeps open only the line
    of the symbol's latest CUSIP: a superseded CUSIP that still fails, or shares the listed symbol, ends normally.
    """
    starts = lines.groupby("key")[["first", "cusip"]].apply(lambda g: sorted(zip(g["first"], g["cusip"], strict=True))).to_dict()
    ends: list[pd.Timestamp] = []
    for cusip, key, first, last, period in lines[["cusip", "key", "first", "last", "last_period"]].to_numpy().tolist():
        nxt = next((start for start, other in starts.get(key, []) if other != cusip and start > last), None)
        superseded = any(other != cusip and start > first for start, other in starts.get(key, []))
        if not superseded and (period in current_periods or (issuer.get(cusip), key) in listed):
            ends.append(_FAR)
            continue
        ends.append(nxt if nxt is not None and nxt - last <= BRIDGE_MAX_GAP else last + pd.Timedelta(days=1))
    return ends


def _finra_days(presence: pd.DataFrame | None) -> dict[str, Any]:
    """Sorted FINRA trading days per FTD key; spellings with a lower-case marker (`BACpB`) are not a common line's key."""
    if presence is None or presence.empty:
        return {}
    symbols = presence["source_symbol"].astype(str)
    frame = pd.DataFrame({"key": symbols.map(squash), "date": pd.to_datetime(presence["date"])})[symbols.eq(symbols.str.upper()).to_numpy()]
    return {str(key): group.drop_duplicates().sort_values().to_numpy() for key, group in frame.groupby("key")["date"]}


def _is_common_line(descriptions: tuple[str, ...], symbol: str) -> bool:
    """Not a transition placeholder, and no non-common kind in any description or in the symbol."""
    return not is_placeholder(symbol) and not any(non_common_kind(d, symbol) for d in descriptions)


def _is_class_line(descriptions: tuple[str, ...], symbol: str, finra_classes: frozenset[str]) -> bool:
    """A common class by positive evidence: a class letter in a description or FINRA `X/Y` symbology, no non-common kind."""
    if not _is_common_line(descriptions, symbol):
        return False
    return any(class_letter(d) for d in descriptions) or squash(symbol) in finra_classes


def _symbol_starts(lines: pd.DataFrame, issuers: Mapping[str, _Issuer], view: _Lineage) -> list[pd.Timestamp | None]:
    """Start of the issuer entity's tape interval of each line's symbol that holds the line's first fail; None without one."""
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


def _extend_by_trading(
    lines: pd.DataFrame,
    finra_days: Mapping[str, Any],
    finra_classes: frozenset[str],
    manual: SecurityManual,
    symbol_starts: list[pd.Timestamp | None],
) -> pd.DataFrame:
    """A class line runs over the FINRA days of its symbol around its fails: on past its last fail, back before its first.
    Another common line only runs back, and no earlier than its issuer's tape interval of the symbol (`symbol_starts`).

    A run stops at a gap longer than `FINRA_MAX_GAP`, where another CUSIP of the symbol trades (going on, at its current
    start; going back, after its last fail or the end of its own run on), and at a manual bound. A line run back ends the
    bridge of an earlier CUSIP of its symbol at its new start, so two lines of one symbol never overlap.
    """
    out = lines.copy()
    out["start"], out["finra_from"], out["finra_to"] = out["first"], pd.NaT, pd.NaT
    if not finra_days:
        return out
    by_key = {key: list(index) for key, index in out.groupby("key").groups.items()}
    manual_ends = manual.boundaries.dropna(subset=["valid_to"]).groupby("cusip")["valid_to"].agg(list).to_dict()
    manual_starts = manual.boundaries.dropna(subset=["valid_from"]).groupby("cusip")["valid_from"].agg(list).to_dict()
    for i, line, symbol_from in zip(out.index, _records(out), symbol_starts, strict=True):
        days = finra_days.get(line.key)
        classed = _is_class_line(line.descriptions, line.source_symbol, finra_classes)
        if days is None or not (classed or (symbol_from is not None and _is_common_line(line.descriptions, line.source_symbol))):
            continue
        others = out.loc[[j for j in by_key.get(line.key, []) if out.at[j, "cusip"] != line.cusip], ["start", "first", "last", "end", "finra_to"]]
        if classed and line.end < _FAR:
            later = others[others["last"] > line.last]
            stops = [max(start, line.last + pd.Timedelta(days=1)) for start in later["start"]]
            stops += [end for end in manual_ends.get(line.cusip, []) if end > line.last]
            last = _chain(days, line.last, min(stops, default=_FAR), forward=True)
            if last > line.last and last + pd.Timedelta(days=1) > line.end:
                out.at[i, "end"], out.at[i, "finra_to"] = last + pd.Timedelta(days=1), last
        earlier = others[others["first"] < line.first]
        ran_on = earlier["end"].where(earlier["finra_to"].notna(), earlier["last"] + pd.Timedelta(days=1))
        floors = [min(end, line.first) for end in ran_on]
        floors += [start for start in manual_starts.get(line.cusip, []) if start <= line.first]
        floors += [] if classed else [symbol_from]
        first = _chain(days, line.first, max(floors, default=_NEAR), forward=False)
        if first < line.first:
            out.at[i, "start"], out.at[i, "finra_from"] = first, first
            bridged = earlier.index[(earlier["end"] > first) & (earlier["end"] < _FAR) & earlier["finra_to"].isna()]
            out.loc[bridged, "end"] = first
    return out


def _extend_to_manual_end(lines: pd.DataFrame, manual: SecurityManual) -> pd.DataFrame:
    """A line ending inside a manual boundary that names an explicit end runs to that end (the market date)."""
    out = lines.copy()
    for r in _records(manual.boundaries.dropna(subset=["valid_to"])):
        mask = out["cusip"].eq(r.cusip) & ~out["placeholder"] & out["last"].lt(r.valid_to) & out["end"].lt(r.valid_to)
        mask &= r.valid_to - out["last"] <= BRIDGE_MAX_GAP
        out.loc[mask, "end"] = r.valid_to
    return out


# --------------------------------------------------------------------------- issuer per CUSIP-9


@dataclass
class _Flags:
    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        kind: str,
        company: str | None,
        cik: str | None,
        cusip: str,
        symbol: str,
        first: object,
        last: object,
        n: int,
        description: str,
        detail: str,
    ) -> None:
        self.rows.append(dict(zip(FLAG_DETAIL_COLUMNS, (kind, company, cik, cusip, symbol, first, last, n, description, detail), strict=True)))


@dataclass(frozen=True)
class _Issuer:
    """A CUSIP's issuer: the CIK owning its latest date, how it was found, and the dated `(cik, start, end)` spans."""

    primary: str
    how: str
    spans: tuple[tuple[str, pd.Timestamp, pd.Timestamp], ...]


def _contains(windows: tuple[Span, ...] | None, first: pd.Timestamp, last: pd.Timestamp) -> bool:
    if windows is None:
        return False
    return any(first >= start - SEAM_TOLERANCE and last < end + SEAM_TOLERANCE for start, end in windows)


def _strong(group: pd.DataFrame, first: pd.Timestamp, last: pd.Timestamp) -> pd.Series:
    """Votes that are not seam lag: they cover enough of the CUSIP's life, or reach near its end."""
    life = max((last - first).days, 1)
    share = (group["last"] - group["first"]).dt.days.clip(lower=1) / life
    return ~((life > BRIDGE_MAX_GAP.days) & (share < WEAK_VOTE_SHARE) & (last - group["last"] > WEAK_VOTE_GAP))


def _issuers(spans: pd.DataFrame, votes: pd.DataFrame, view: _Lineage, manual: SecurityManual, flags: _Flags) -> dict[str, _Issuer]:
    """`{cusip: _Issuer}` from manual rows, strong direct votes, else CUSIP-6 voters whose era overlaps the CUSIP's life."""
    life = spans.set_index("cusip")
    strong_rows = []
    for key, group in votes.groupby("cusip", sort=True):
        cusip = str(key)
        keep = (
            _strong(group, cast(pd.Timestamp, life.at[cusip, "first"]), cast(pd.Timestamp, life.at[cusip, "last"]))
            if cusip in life.index
            else pd.Series(True, index=group.index)
        )
        for row in _records(group[~keep]):
            span = cast(Any, life.loc[cusip])
            detail = f"CIK {row.cik} n={row.votes} {row.first.date()}..{row.last.date()} of a life {span['first'].date()}..{span['last'].date()}"
            flags.add(
                "security_weak_vote",
                view.ticker(row.cik),
                row.cik,
                cusip,
                span["symbols"],
                span["first"],
                span["last"],
                int(span["n_obs"]),
                span["description"],
                detail,
            )
        strong_rows.append(group[keep])
    strong = pd.concat(strong_rows, ignore_index=True) if strong_rows else votes.iloc[0:0]
    voted = set(votes["cusip"])
    direct = {cusip: group for cusip, group in strong.groupby("cusip")}
    by_prefix = (
        strong.assign(prefix=strong["cusip"].str[:6]).groupby(["prefix", "cik"], as_index=False).agg(first=("first", "min"), last=("last", "max"))
    )
    eras = {prefix: list(zip(g["cik"], g["first"], g["last"], strict=True)) for prefix, g in by_prefix.groupby("prefix")}
    manual_issuer = manual.boundaries.dropna(subset=["issuer_cik"]).groupby("cusip")["issuer_cik"].agg(lambda s: sorted(set(s))).to_dict()
    out: dict[str, _Issuer] = {}
    for span in _records(spans):
        cusip, first, last = span.cusip, span.first, span.last
        if cusip in manual_issuer and len(manual_issuer[cusip]) == 1:
            cik = manual_issuer[cusip][0]
            out[cusip] = _Issuer(cik, "manual market boundary", ((cik, _NEAR, _FAR),))
            continue
        if cusip in direct:
            group = direct[cusip]
            cands = sorted(set(group["cik"]))
            how = "votes " + ", ".join(
                f"{c} n={n} {a.date()}..{b.date()}" for c, n, a, b in zip(group["cik"], group["votes"], group["first"], group["last"], strict=True)
            )
        elif cusip in voted:
            continue
        else:
            cands = sorted({cik for cik, a, b in eras.get(cusip[:6], []) if a <= last and b >= first})
            how = f"CUSIP-6 {cusip[:6]} voters " + ", ".join(cands)
        if not cands:
            continue
        issuer = _resolve(cands, first, last, view, how)
        companies = ",".join(sorted({view.ticker(c) or view.entity(c) for c in cands}))
        if issuer is None:
            flags.add("security_issuer_conflict", companies, ",".join(cands), cusip, span.symbols, first, last, span.n_obs, span.description, how)
            continue
        if len(issuer.spans) > 1:
            dated = "; ".join(
                f"{c} {max(a, first).date()}..{'open' if b >= _FAR else b.date()}" for c, a, b in issuer.spans if b > first and a <= last
            )
            flags.add(
                "security_straddle",
                companies,
                ",".join(cands),
                cusip,
                span.symbols,
                first,
                last,
                span.n_obs,
                span.description,
                f"issuer by window: {dated}",
            )
        out[cusip] = issuer
    return out


def _resolve(cands: list[str], first: pd.Timestamp, last: pd.Timestamp, view: _Lineage, how: str) -> _Issuer | None:
    """One CIK; of several CIKs of one company, the one whose window holds the CUSIP's life, else the window owner per date.

    None when the votes name two companies, or several CIKs of one company none of which has a window.
    """
    if len(cands) == 1:
        return _Issuer(cands[0], how, ((cands[0], _NEAR, _FAR),))
    entities = {view.entity(c) for c in cands}
    if len(entities) > 1:
        return None
    holding = [c for c in cands if _contains(view.windows.get(c), first, last)]
    if len(holding) == 1:
        return _Issuer(holding[0], how, ((holding[0], _NEAR, _FAR),))
    owners = view.window_owners(entities.pop())
    if not owners:
        return None
    spans = tuple((cik, _NEAR if i == 0 else a, _FAR if i == len(owners) - 1 else owners[i + 1][1]) for i, (cik, a, _) in enumerate(owners))
    primary = next(cik for cik, a, b in spans if a <= last < b)
    return _Issuer(primary, how, spans)


# --------------------------------------------------------------------------- roles


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
    """A span a CUSIP may hold as its company's canonical line: lower priority wins, later start first within one."""

    priority: int
    start: pd.Timestamp
    cusip: str
    issuer: str
    spans: tuple[Span, ...]
    source: str | None = None


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


@dataclass
class _Cusip:
    """Per-CUSIP facts the role assignment reads."""

    cusip: str
    issuer: _Issuer
    company: str
    start: pd.Timestamp
    end: pd.Timestamp
    kind: str | None
    letter: str | None
    priority: int | None
    class_evidence: str | None
    manual: list[tuple[pd.Timestamp, pd.Timestamp, str, str, str, str]]


def _window_spans(windows: tuple[Span, ...], start: pd.Timestamp, end: pd.Timestamp) -> list[Span]:
    """In-window parts of `[start, end)`; a bound within the seam tolerance of the life's edge moves to that edge."""
    out: list[Span] = []
    for lo, hi in windows:
        if start < lo < end:
            lo = start if lo - start <= SEAM_TOLERANCE else (end if end - lo <= SEAM_TOLERANCE else lo)
        if start < hi < end:
            hi = end if end - hi <= SEAM_TOLERANCE else (start if hi - start <= SEAM_TOLERANCE else hi)
        out += _intersect([(start, end)], lo, hi)
    return out


def _cusip_facts(
    lines: pd.DataFrame,
    issuers: Mapping[str, _Issuer],
    view: _Lineage,
    manual: SecurityManual,
    listed: frozenset[tuple[str, str]],
    finra: frozenset[str],
) -> list[_Cusip]:
    """Kind, class letter, canonical priority and class evidence of each attributed CUSIP (placeholder lines left out)."""
    manual_class = dict(zip(manual.classes["cusip"], manual.classes["security_class"], strict=True))
    out: list[_Cusip] = []
    for key, group in lines[~lines["placeholder"]].groupby("cusip", sort=True):
        cusip = str(key)
        issuer = issuers.get(cusip)
        company = None if issuer is None else view.ticker(issuer.primary)
        if issuer is None or company is None:
            continue
        out.append(_cusip_fact(cusip, group, issuer, company, view, manual, manual_class, listed, finra))
    return out


def _cusip_fact(
    cusip: str,
    group: pd.DataFrame,
    issuer: _Issuer,
    company: str,
    view: _Lineage,
    manual: SecurityManual,
    manual_class: Mapping[str, str],
    listed: frozenset[tuple[str, str]],
    finra: frozenset[str],
) -> _Cusip:
    """One attributed CUSIP's facts from its lines, its entity's symbols and tape intervals, and the manual config."""
    entity = view.entity(issuer.primary)
    symbols = view.symbols_by_entity.get(entity, frozenset())
    own = {squash(s) for s in symbols}
    tape = view.tape[view.tape["cik"].isin([c for c in view.windows if view.entity(c) == entity])]
    letters = {class_letter(d) for descs in group["descriptions"] for d in descs} - {None}
    spelled = {m.group(1) for s in symbols if (m := _CLASS_SYMBOL.match(str(s))) and squash(s) in set(group["key"])}
    symbol_letter = len(letters) == 0 and len(spelled) == 1
    letters = letters or spelled
    letter = next(iter(letters)) if len(letters) == 1 else None
    priorities, kinds = _line_priorities(group, company, tape, own, letter)
    priority = min(priorities) if priorities else None
    kind = None if priority == 0 else (max(kinds)[1] if kinds else None)
    if manual_class.get(cusip) in NON_COMMON_KINDS:
        kind = manual_class[cusip]
    evidence = _class_evidence(cusip, group, issuer, letter, symbol_letter, manual_class, listed, finra)
    return _Cusip(
        cusip, issuer, company, group["start"].min(), max(group["end"]), kind, letter, priority, evidence, _manual_bounds(cusip, issuer, manual)
    )


def _line_priorities(
    group: pd.DataFrame, company: str, tape: pd.DataFrame, own: set[str], letter: str | None
) -> tuple[list[int], list[tuple[int, str]]]:
    """Canonical priorities of a CUSIP's common lines, and `(observations, kind)` of its non-common ones."""
    priorities: list[int] = []
    kinds: list[tuple[int, str]] = []
    for line in _records(group):
        if line.key == squash(company):
            priorities.append(0)
            continue
        on_tape = not tape[(tape["key"] == line.key) & (tape["valid_from"] <= line.last) & (tape["valid_to"] > line.first)].empty
        kind = non_common_kind(line.description, "" if (on_tape or line.key in own) else line.source_symbol)
        kind = kind or next((k for k in (non_common_kind(d) for d in line.descriptions) if k), None)
        if kind:
            kinds.append((line.n_obs, kind))
            continue
        if on_tape:
            priorities.append(1 if letter in (None, "A") else 3)
        elif letter == "A":
            priorities.append(2)
    return priorities, kinds


def _class_evidence(
    cusip: str,
    group: pd.DataFrame,
    issuer: _Issuer,
    letter: str | None,
    symbol_letter: bool,
    manual_class: Mapping[str, str],
    listed: frozenset[tuple[str, str]],
    finra: frozenset[str],
) -> str | None:
    """Why a CUSIP is a common class: manual config, a lineage class symbol, its description, the SEC listing or FINRA symbology."""
    return (
        "manual_class"
        if cusip in manual_class
        else "lineage_class_symbol"
        if letter and symbol_letter
        else "class_description"
        if letter
        else "sec_tickers_listing"
        if any((c, k) in listed for c, _, _ in issuer.spans for k in group["key"])
        else "finra_symbology"
        if any(k in finra for k in group["key"])
        else None
    )


def _manual_bounds(cusip: str, issuer: _Issuer, manual: SecurityManual) -> list[tuple[pd.Timestamp, pd.Timestamp, str, str, str, str]]:
    """The CUSIP's manual role boundaries as `(start, end, role, reason, source, issuer CIK)`; open bounds at the far dates."""
    rows = manual.boundaries[manual.boundaries["cusip"].eq(cusip)]
    return [
        (
            _bound(r.valid_from) or _NEAR,
            _bound(r.valid_to) or _FAR,
            r.role,
            r.reason or "manual_boundary",
            str(r.source),
            r.issuer_cik or issuer.primary,
        )
        for r in _records(rows)
    ]


def _assign(facts: list[_Cusip], view: _Lineage, co_registrants: frozenset[str], horizon: pd.Timestamp) -> dict[str, list[_Segment]]:
    """Role segments per CUSIP: acquired outside its issuers' windows, excluded by kind, one canonical line per company per date."""
    segments: dict[str, list[_Segment]] = {}
    candidates: dict[str, list[_Candidate]] = {}
    for fact in facts:
        if fact.issuer.primary in co_registrants:
            continue
        segments[fact.cusip] = _fact_segments(fact, view, candidates)
    by_cusip = {fact.cusip: fact for fact in facts}
    for company, cands in candidates.items():
        _canonical(company, cands, by_cusip, view, segments, horizon)
    return segments


def _fact_segments(fact: _Cusip, view: _Lineage, candidates: dict[str, list[_Candidate]]) -> list[_Segment]:
    """One CUSIP's non-canonical segments over its life; its canonical candidates are appended to `candidates`."""
    life = [(fact.start, fact.end)]
    out = _manual_segments(fact, life, candidates)
    free = _subtract(life, [(a, b) for a, b, *_ in fact.manual])
    for cik, a, b in fact.issuer.spans:
        out += _span_segments(fact, cik, _intersect(free, a, b), view, candidates)
    return out


def _manual_segments(fact: _Cusip, life: list[Span], candidates: dict[str, list[_Candidate]]) -> list[_Segment]:
    """Segments of the CUSIP's manual boundaries; a manual canonical role becomes a top-priority candidate."""
    out: list[_Segment] = []
    for a, b, role, reason, source, cik in fact.manual:
        for lo, hi in _intersect(life, a, b):
            if role in CANONICAL_ROLES:
                candidates.setdefault(fact.company, []).append(_Candidate(-1, fact.start, fact.cusip, cik, ((lo, hi),), source))
            else:
                out.append(_Segment(lo, hi, role, reason, cik, source))
    return out


def _span_segments(fact: _Cusip, cik: str, part: list[Span], view: _Lineage, candidates: dict[str, list[_Candidate]]) -> list[_Segment]:
    """Segments of one issuer span's free part: acquired outside the CIK's windows; inside, excluded by kind, a canonical
    candidate by priority, else a secondary class with class evidence or excluded as unclassified."""
    windows = view.windows.get(cik)
    if windows is None:
        return [_Segment(lo, hi, ACQUIRED_CONSTITUENT, "event_only_cik", cik) for lo, hi in part]
    inside = [p for lo, hi in part for p in _window_spans(windows, lo, hi)]
    outside = _subtract(part, inside)
    if fact.priority is not None and not fact.kind:
        if inside:
            candidates.setdefault(fact.company, []).append(_Candidate(fact.priority, fact.start, fact.cusip, cik, tuple(inside)))
        if outside:
            candidates.setdefault(fact.company, []).append(_Candidate(FALLBACK_PRIORITY, fact.start, fact.cusip, cik, tuple(outside)))
        return []
    if fact.kind:
        role, why = EXCLUDED, fact.kind
    else:
        role, why = (SECONDARY_CLASS, fact.class_evidence) if fact.class_evidence else (EXCLUDED, "unclassified")
    return [_Segment(lo, hi, role, why, cik) for lo, hi in inside] + [
        _Segment(lo, hi, ACQUIRED_CONSTITUENT, "outside_window", cik) for lo, hi in outside
    ]


def _canonical(
    company: str, cands: list[_Candidate], facts: Mapping[str, _Cusip], view: _Lineage, segments: dict[str, list[_Segment]], horizon: pd.Timestamp
) -> None:
    """Best priority first, later start first within a priority: each takes the dates no better line holds.

    A candidate overlapping better lines over most of its life is a sibling class; a fallback part not taken stays acquired.
    """
    covered: list[Span] = []
    roster_cik = view.roster_cik_by_ticker.get(company)
    for cand in sorted(cands, key=lambda c: (c.priority, -c.start.value, c.cusip, c.issuer)):
        fact = facts[cand.cusip]
        free = _subtract(cand.spans, covered)
        whole = _length(cand.spans, horizon)
        sibling = 0 <= cand.priority < FALLBACK_PRIORITY and whole > 0 and 1 - _length(free, horizon) / whole >= SIBLING_OVERLAP
        taken = [] if sibling else free
        rest = _subtract(cand.spans, taken)
        role = CANONICAL_CURRENT if cand.issuer == roster_cik else CANONICAL_PREDECESSOR
        segments[cand.cusip] += [_Segment(a, b, role, _PRIORITY_REASON[cand.priority], cand.issuer, cand.source) for a, b in taken]
        if cand.priority == FALLBACK_PRIORITY:
            leftover = (ACQUIRED_CONSTITUENT, "outside_window")
        elif sibling and fact.class_evidence:
            leftover = (SECONDARY_CLASS, fact.class_evidence)
        else:
            leftover = (EXCLUDED, "superseded")
        segments[cand.cusip] += [_Segment(a, b, *leftover, cand.issuer) for a, b in rest]
        covered += taken


# --------------------------------------------------------------------------- rows


@dataclass(frozen=True)
class MasterBuild:
    """The derived table rows, the per-line flags, and the WARNING-block items."""

    rows: pd.DataFrame
    flags: pd.DataFrame
    items: pd.DataFrame


def _class_of(fact: _Cusip | None, role: str, reason: str) -> str:
    if fact is None:
        return "unclassified"
    if fact.kind:
        return fact.kind
    if role == EXCLUDED and reason == "unclassified":
        return "unclassified"
    if fact.letter:
        return f"class_{fact.letter}"
    return "common" if (fact.priority is not None or role in CANONICAL_ROLES or fact.class_evidence) else "unclassified"


def _ratio_pieces(
    cusip: str, start: pd.Timestamp, end: pd.Timestamp, ratios: pd.DataFrame
) -> list[tuple[pd.Timestamp, pd.Timestamp, float, str | None]]:
    rows = ratios[ratios["cusip"].eq(cusip)]
    pieces: list[tuple[pd.Timestamp, pd.Timestamp, float, str | None]] = []
    rest: list[Span] = [(start, end)]
    for r in _records(rows):
        for a, b in _intersect([(start, end)], _bound(r.valid_from) or _NEAR, _bound(r.valid_to) or _FAR):
            pieces.append((a, b, float(r.ratio), str(r.source)))
            rest = _subtract(rest, [(a, b)])
    pieces += [(a, b, 1.0, None) for a, b in rest]
    return sorted(pieces)


def _market_symbol(symbol: str, company: str, issuer: str, view: _Lineage, sec_symbols: Mapping[tuple[str, str], str]) -> str:
    key = squash(symbol)
    if key == squash(company):
        return company
    if (issuer, key) in sec_symbols:
        return sec_symbols[(issuer, key)]
    spelled = sorted(s for s in view.symbols_by_entity.get(view.entity(issuer), frozenset()) if squash(s) == key)
    return spelled[0] if spelled else symbol


@dataclass(frozen=True)
class _RowContext:
    view: _Lineage
    manual: SecurityManual
    sec_symbols: Mapping[tuple[str, str], str]
    sec_exchange: Mapping[tuple[str, str], str]


def _line_pieces(line: Any, segments: list[_Segment], placeholder_issuer: str) -> list[_Segment]:
    if line.placeholder:
        return [_Segment(line.start, line.end, EXCLUDED, "transition_placeholder", placeholder_issuer)]
    return [_Segment(a, b, s.role, s.reason, s.issuer, s.source) for s in segments for a, b in _intersect([(line.start, line.end)], s.start, s.end)]


def _line_rows(
    lines: pd.DataFrame,
    obs: pd.DataFrame,
    segments: Mapping[str, list[_Segment]],
    facts: Mapping[str, _Cusip],
    issuers: Mapping[str, _Issuer],
    ctx: _RowContext,
) -> list[dict[str, Any]]:
    """Each line clipped to its CUSIP's role segments, split at conversion-ratio dates, adjacent equal pieces merged."""
    dates = {key: group["trade_date"].to_numpy() for key, group in obs.groupby(["cusip", "source_symbol"])}
    out: list[dict[str, Any]] = []
    for line in _records(lines):
        if line.cusip not in segments:
            continue
        issuer = issuers[line.cusip]
        merged = _merged_pieces(line, segments[line.cusip], facts.get(line.cusip), issuer, ctx)
        out += _finish_rows(merged, line, issuer, dates.get((line.cusip, line.source_symbol)), ctx)
    return out


def _merged_pieces(line: Any, segments: list[_Segment], fact: _Cusip | None, issuer: _Issuer, ctx: _RowContext) -> list[dict[str, Any]]:
    """The line's role pieces split at conversion-ratio dates, adjacent pieces with the same role, reason, ratio, class and issuer merged."""
    company = ctx.view.ticker(issuer.primary) or ""
    merged: list[dict[str, Any]] = []
    for seg in sorted(_line_pieces(line, segments, issuer.primary), key=lambda s: s.start):
        for a, b, ratio, ratio_source in _ratio_pieces(line.cusip, seg.start, seg.end, ctx.manual.ratios):
            cls = _class_of(fact, seg.role, seg.reason)
            key = (seg.role, seg.reason, ratio, cls, seg.issuer)
            if merged and merged[-1]["valid_to"] == a and merged[-1]["_key"] == key:
                merged[-1]["valid_to"] = b
                continue
            merged.append(
                {
                    "_key": key,
                    "security_id": f"C{line.cusip}",
                    "canonical_company": company,
                    "issuer_cik": seg.issuer,
                    "source": SOURCE_FTD,
                    "source_symbol": line.source_symbol,
                    "market_symbol": _market_symbol(line.source_symbol, company, seg.issuer, ctx.view, ctx.sec_symbols),
                    "cusip": line.cusip,
                    "security_class": cls,
                    "conversion_ratio": ratio,
                    "lineage_role": seg.role,
                    "valid_from": a,
                    "valid_to": b,
                    "lineage_reason": seg.reason,
                    "source_accession": seg.source or ratio_source,
                }
            )
    return merged


def _finish_rows(merged: list[dict[str, Any]], line: Any, issuer: _Issuer, days: Any, ctx: _RowContext) -> list[dict[str, Any]]:
    """The merged pieces as table rows: observation counts, open end as None, listing exchange of an open row, evidence text."""
    for row in merged:
        row.pop("_key")
        start, end = row["valid_from"].to_datetime64(), row["valid_to"].to_datetime64()
        row["n_observations"] = int(((days >= start) & (days < end)).sum()) if days is not None else 0
        row["valid_to"] = None if row["valid_to"] >= _FAR else row["valid_to"]
        row["exchange"] = ctx.sec_exchange.get((row["issuer_cik"], squash(line.source_symbol))) if row["valid_to"] is None else None
        traded = (f"; FINRA from {line.finra_from.date()}" if pd.notna(line.finra_from) else "") + (
            f"; FINRA to {line.finra_to.date()}" if pd.notna(line.finra_to) else ""
        )
        row["evidence"] = (
            f"issuer {issuer.primary} by {issuer.how}; FTD {line.first.date()}..{line.last.date()} n={line.n_obs}{traded}; {line.description}"
        )
    return merged


def _finalise(rows: list[dict[str, Any]], existing: pd.DataFrame | None, built_at: pd.Timestamp) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=[c for c in TABLE_COLUMNS if c != "scope_changed_at"])
    frame["valid_from"] = pd.to_datetime(frame["valid_from"])
    frame["valid_to"] = pd.to_datetime(frame["valid_to"])
    frame["conversion_ratio"] = frame["conversion_ratio"].astype("float64")
    frame["n_observations"] = frame["n_observations"].astype("int64")
    for column in ("exchange", "source_accession", "canonical_company"):
        frame[column] = frame[column].astype(object).where(frame[column].notna(), None)
    frame["scope_changed_at"] = _scope_changed_at(frame, existing, built_at)
    return frame[list(TABLE_COLUMNS)].sort_values(
        ["canonical_company", "security_id", "source", "source_symbol", "valid_from"], kind="mergesort", ignore_index=True
    )


_SIGNATURE = ("security_id", "source", "source_symbol", "valid_from", "valid_to", "lineage_role", "security_class", "conversion_ratio", "issuer_cik")


def _signatures(frame: pd.DataFrame) -> dict[str, tuple[tuple[str, ...], ...]]:
    text = frame[list(_SIGNATURE)].copy()
    for column in ("valid_from", "valid_to"):
        text[column] = pd.to_datetime(text[column]).dt.strftime("%Y-%m-%d").fillna("")
    text = text.astype(str)
    return {str(company): tuple(sorted(map(tuple, group.to_numpy()))) for company, group in text.groupby(frame["canonical_company"].astype(str))}


def _scope_changed_at(frame: pd.DataFrame, existing: pd.DataFrame | None, built_at: pd.Timestamp) -> pd.Series:
    """The stored stamp of each canonical company whose rows are unchanged; `built_at` for every other."""
    companies = frame["canonical_company"].astype(str)
    if existing is None or existing.empty or "scope_changed_at" not in existing.columns:
        return pd.Series(built_at, index=frame.index, dtype="datetime64[ns]")
    before, after = _signatures(existing), _signatures(frame)
    stored = pd.to_datetime(existing["scope_changed_at"]).groupby(existing["canonical_company"].astype(str)).max().to_dict()
    kept = {c: stored[c] for c in set(companies) if before.get(c) == after.get(c) and pd.notna(stored.get(c))}
    return pd.Series([kept.get(c, built_at) for c in companies], index=frame.index, dtype="datetime64[ns]")


# --------------------------------------------------------------------------- checks


def _symbol_conflicts(obs: pd.DataFrame, rows: pd.DataFrame, flags: _Flags) -> None:
    """A (settlement date, FTD symbol) under two CUSIPs, among the master's securities."""
    kept = obs[obs["cusip"].isin(set(rows["cusip"])) & ~obs["source_symbol"].map(is_placeholder)]
    distinct = kept.groupby(["date", "key"])["cusip"].transform("nunique")
    clash = kept[distinct > 1]
    counts = clash.groupby(["date", "key"])["cusip"].agg(lambda s: tuple(sorted(set(s))))
    company = rows.drop_duplicates("cusip").set_index("cusip")["canonical_company"].to_dict()
    for (day, key), cusips in cast(Any, counts).items():
        symbol = kept.loc[kept["key"].eq(key), "source_symbol"].iloc[0]
        flags.add(
            "security_symbol_conflict",
            company.get(cusips[0]),
            None,
            cusips[0],
            symbol,
            day,
            day,
            len(cusips),
            "",
            f"{symbol} on {day.date()} under " + ", ".join(cusips),
        )


def ratio_step_flags(obs: pd.DataFrame, rows: pd.DataFrame, ratios: pd.DataFrame) -> pd.DataFrame:
    """Steps of more than `RATIO_STEP` in the rolling-median price ratio of a secondary class to the canonical line.

    A step is accepted when a configured conversion ratio of that class changes within `RATIO_MATCH_DAYS` by
    about the same factor. The level is never checked.
    """
    flags = _Flags()
    priced = obs[obs["price"].gt(0)][["cusip", "source_symbol", "trade_date", "price"]]
    spans = rows[rows["lineage_role"].isin((*CANONICAL_ROLES, SECONDARY_CLASS))][
        ["cusip", "source_symbol", "canonical_company", "lineage_role", "valid_from", "valid_to"]
    ]
    tagged = priced.merge(spans, on=["cusip", "source_symbol"])
    tagged = tagged[(tagged["trade_date"] >= tagged["valid_from"]) & (tagged["trade_date"] < tagged["valid_to"].fillna(_FAR))]
    canonical = tagged[tagged["lineage_role"].isin(CANONICAL_ROLES)].groupby(["canonical_company", "trade_date"])["price"].median()
    secondary = tagged[tagged["lineage_role"].eq(SECONDARY_CLASS)].groupby(["canonical_company", "cusip", "trade_date"])["price"].median()
    for (company, cusip), series in cast(Any, secondary.groupby(level=[0, 1])):
        if company not in canonical.index.get_level_values(0):
            continue
        both = pd.concat([series.droplevel([0, 1]).rename("s"), canonical.loc[company].rename("c")], axis=1, join="inner").sort_index()
        if len(both) < RATIO_WINDOW:
            continue
        median = (both["s"] / both["c"]).rolling(RATIO_WINDOW, min_periods=RATIO_WINDOW).median().dropna()
        change = median / median.shift(1)
        steps = change[(change > RATIO_STEP) | (change < 1 / RATIO_STEP)]
        for start, stop in _clusters(list(steps.index)):
            before, after = median[median.index < start], median[median.index >= stop]
            factor = float(after.iloc[0] / before.iloc[-1]) if len(before) and len(after) else float(change.loc[start])
            if not _configured(cusip, start, stop, factor, ratios):
                flags.add(
                    "security_ratio_step",
                    company,
                    None,
                    cusip,
                    "",
                    start,
                    stop,
                    len(both),
                    "",
                    f"price-ratio median x{factor:.3g} with no configured ratio change",
                )
    return pd.DataFrame(flags.rows, columns=list(FLAG_DETAIL_COLUMNS))


def _clusters(days: list[pd.Timestamp]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    out: list[list[pd.Timestamp]] = []
    for day in days:
        if out and day - out[-1][1] <= RATIO_MATCH_DAYS:
            out[-1][1] = day
        else:
            out.append([day, day])
    return [(a, b) for a, b in out]


def _configured(cusip: str, start: pd.Timestamp, stop: pd.Timestamp, factor: float, ratios: pd.DataFrame) -> bool:
    rows = ratios[ratios["cusip"].eq(cusip)].sort_values("valid_from", na_position="first")
    for prev, nxt in zip(_records(rows), _records(rows.iloc[1:]), strict=False):
        when = nxt.valid_from
        if (
            pd.notna(when)
            and start - RATIO_MATCH_DAYS <= when <= stop + RATIO_MATCH_DAYS
            and 1 / RATIO_STEP < factor / (nxt.ratio / prev.ratio) < RATIO_STEP
        ):
            return True
    return False


def flag_items(flags: pd.DataFrame) -> pd.DataFrame:
    """One WARNING-block item (`FLAG_COLUMNS`) per flag kind: count, companies and the first examples."""
    items = []
    for kind, (action, suggested) in FLAG_KINDS.items():
        part = flags[flags["kind"].eq(kind)]
        if part.empty:
            continue
        examples = "; ".join(f"{r.cusip} {r.source_symbol} ({r.canonical_company}) {r.detail}" for r in _records(part.head(12)))
        items.append(
            {
                "kind": kind,
                "action": action,
                "ticker": ",".join(sorted({str(t) for t in part["canonical_company"].dropna()})),
                "ciks": ",".join(sorted({str(c) for c in part["issuer_cik"].dropna()})),
                "evidence": f"{len(part)} line(s): {examples}" + (" ..." if len(part) > 12 else ""),
                "suggested_action": suggested,
                "config_file": f"configs/sec/{MANUAL_CONFIG_FILENAME}",
            }
        )
    return pd.DataFrame(items, columns=list(FLAG_COLUMNS))


# --------------------------------------------------------------------------- entry points


def _sec_maps(sec_tickers: pd.DataFrame | None) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    """`(cik, squashed ticker) -> market spelling` and `-> exchange` from the SEC current-tickers snapshot."""
    if sec_tickers is None or sec_tickers.empty:
        return {}, {}
    ciks = pad_cik_series(sec_tickers["cik"])
    tickers = sec_tickers["ticker"].astype(str)
    exchanges = sec_tickers["exchange"] if "exchange" in sec_tickers.columns else pd.Series(None, index=sec_tickers.index)
    symbols = {(cik, squash(t)): normalise_ticker(t) for cik, t in zip(ciks, tickers, strict=True)}
    exchange = {(cik, squash(t)): ex for cik, t, ex in zip(ciks, tickers, exchanges, strict=True) if isinstance(ex, str)}
    return symbols, exchange


def derive_security_master(
    observations: pd.DataFrame,
    lineage: pd.DataFrame,
    roster: pd.DataFrame,
    manual: SecurityManual,
    *,
    built_at: pd.Timestamp,
    sec_tickers: pd.DataFrame | None = None,
    finra_symbols: Collection[str] = (),
    finra_presence: pd.DataFrame | None = None,
    co_registrant_ciks: Collection[str] = (),
    existing: pd.DataFrame | None = None,
) -> MasterBuild:
    """Pure derivation of `security_master`; deterministic for fixed inputs and `built_at`, no DB, no network.

    `finra_presence` (`source_symbol`, `date`) are stored RegSHO days: trading evidence a line runs over.
    """
    obs = _prepare(observations)
    view = _lineage_view(lineage, roster)
    flags = _Flags()
    sec_symbols, sec_exchange = _sec_maps(sec_tickers)
    finra = frozenset(squash(s) for s in finra_symbols if "/" in str(s))
    if finra_presence is not None and not finra_presence.empty:
        finra |= frozenset(squash(s) for s in finra_presence["source_symbol"].astype(str) if re.fullmatch(r"[A-Z]+/[A-Z]", s))

    lines = _lines(obs)
    real = lines[~lines["placeholder"]]
    spans = real.groupby("cusip", as_index=False).agg(
        first=("first", "min"),
        last=("last", "max"),
        n_obs=("n_obs", "sum"),
        description=("description", "first"),
        symbols=("source_symbol", lambda s: " ".join(sorted(s))),
    )
    issuers = _issuers(spans, _votes(obs[["cusip", "key", "trade_date"]].drop_duplicates(), view.tape), view, manual, flags)
    periods = sorted(set(obs["period"]))
    lines["end"] = _line_ends(lines, frozenset(periods[-CURRENT_PERIODS:]), frozenset(sec_symbols), {c: v.primary for c, v in issuers.items()})
    lines = _extend_by_trading(lines, _finra_days(finra_presence), finra, manual, _symbol_starts(lines, issuers, view))
    lines = _extend_to_manual_end(lines, manual)
    listed_classes = frozenset(key for key, spelled in sec_symbols.items() if _LISTED_CLASS.match(spelled))
    facts = _cusip_facts(lines, issuers, view, manual, listed_classes, finra)
    co_registrants = frozenset(pad_cik(c) for c in co_registrant_ciks)
    horizon = obs["trade_date"].max() + pd.Timedelta(days=1) if not obs.empty else _FAR
    segments = _assign(facts, view, co_registrants, horizon)
    by_cusip = {fact.cusip: fact for fact in facts}
    context = _RowContext(view, manual, sec_symbols, sec_exchange)
    rows = _finalise(_line_rows(lines, obs, segments, by_cusip, issuers, context), existing, built_at)

    for fact in facts:
        if fact.issuer.primary in co_registrants or not any(s.reason == "unclassified" for s in segments.get(fact.cusip, [])):
            continue
        part = lines[lines["cusip"].eq(fact.cusip) & ~lines["placeholder"]]
        flags.add(
            "security_unclassified",
            fact.company,
            fact.issuer.primary,
            fact.cusip,
            " ".join(part["source_symbol"]),
            part["first"].min(),
            part["last"].max(),
            int(part["n_obs"].sum()),
            part["description"].iloc[0],
            "no class evidence",
        )
    _symbol_conflicts(obs, rows, flags)
    detail = pd.concat([pd.DataFrame(flags.rows, columns=list(FLAG_DETAIL_COLUMNS)), ratio_step_flags(obs, rows, manual.ratios)], ignore_index=True)
    detail = detail.sort_values(["kind", "canonical_company", "cusip", "first"], kind="mergesort", na_position="last", ignore_index=True)
    return MasterBuild(rows=rows, flags=detail, items=flag_items(detail))


def co_registrant_ciks(lineage: pd.DataFrame, evidence: pd.DataFrame | None, declared: Collection[str] = ()) -> frozenset[str]:
    """The extra CIKs the identity flags classify as co-registrants, except a former listing of the company, plus the
    `declared` ones (curated subsidiaries the flags never judge).

    A former listing held the company's ticker on its own, not alongside the home CIK (old GM as `GM` until 2009):
    it is a predecessor or an acquired constituent, not a subsidiary filing under its parent's symbol.
    """
    listed = frozenset(pad_cik(cik) for cik in declared)
    if evidence is None or evidence.empty or not {"role", "sources"} <= set(lineage.columns):  # membership rows only: nothing to judge
        return listed
    flags = identity_flags(lineage, cik_activity(evidence))
    tape = _tape_intervals(lineage)
    out = set()
    for value in flags.loc[flags["kind"].eq("co_registrant"), "ciks"]:
        other, home = (str(value).split(",") + [""])[:2]
        if not _former_listing(tape, other, home):
            out.add(other)
    return frozenset(out) | listed


def _former_listing(tape: pd.DataFrame, other: str, home: str) -> bool:
    """`other` held a symbol `home` also holds, and their intervals overlap by at most the seam margin."""
    mine, theirs = tape[tape["cik"].eq(other)], tape[tape["cik"].eq(home)]
    shared = set(mine["key"]) & set(theirs["key"])
    if not shared:
        return False
    for key in shared:
        a, b = mine[mine["key"].eq(key)], theirs[theirs["key"].eq(key)]
        overlap = max(
            (
                min(x_to, y_to) - max(x_from, y_from)
                for x_from, x_to in zip(a["valid_from"], a["valid_to"], strict=True)
                for y_from, y_to in zip(b["valid_from"], b["valid_to"], strict=True)
            ),
            default=pd.Timedelta(0),
        )
        if overlap > pd.Timedelta(days=31):
            return False
    return True


def _finra_presence(context: Context, spellings: list[str]) -> pd.DataFrame | None:
    """Stored RegSHO days of the unmarked FINRA spellings: the trading evidence a line runs over."""
    wanted = sorted(s for s in spellings if s == s.upper())
    if not wanted:
        return None
    return context.store.load(Tables.sec_short_volume_security, columns=["source_symbol", "date"], where={"source_symbol": wanted}, optional=True)


def build_security_master(
    context: Context, lineage: pd.DataFrame, config_dir: str | None = None, *, built_at: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Derive `security_master` from the stored FTD lines and the lineage just built; replace the table unless unchanged."""
    observations = context.store.load(Tables.sec_fails_to_deliver_security, columns=list(_OBSERVATION_COLUMNS), optional=True)
    if observations is None:
        context.log.warning("security_master: no stored FTD lines (run ftd-download first); table left unchanged")
        return pd.DataFrame(columns=list(TABLE_COLUMNS))
    roster = context.store.load(Tables.sp500_tickers, columns=list(ROSTER_COLUMNS))
    assert roster is not None
    sec_tickers = context.store.load(Tables.sec_company_tickers, project=True, optional=True)
    existing = context.store.load(Tables.security_master, project=True, optional=True)
    evidence = context.store.load(
        Tables.symbol_tenure, columns=["issuer_cik", "source", "valid_from", "valid_to"], where={"source": ["form345", DEI_SOURCE]}, optional=True
    )
    manual = load_security_manual(str(config_dir or context.config_dir))
    co_registrants = co_registrant_ciks(lineage, evidence, manual.co_registrants)
    finra = (
        [str(s) for s in context.store.distinct(Tables.sec_short_volume_security, "source_symbol")]
        if context.store.exists(Tables.sec_short_volume_security)
        else []
    )
    build = derive_security_master(
        observations,
        lineage,
        roster,
        manual,
        built_at=built_at or pd.Timestamp.now().floor("s"),
        sec_tickers=sec_tickers,
        finra_symbols=[symbol for symbol in finra if re.fullmatch(r"[A-Z]+/[A-Z]", symbol)],
        finra_presence=_finra_presence(context, finra),
        co_registrant_ciks=co_registrants,
        existing=existing,
    )
    rows = build.rows
    context.log.info(
        f"security_master: {len(rows)} row(s) over {rows['security_id'].nunique()} CUSIP(s) and {rows['canonical_company'].nunique()} company(ies); "
        f"roles {rows.groupby('lineage_role')['security_id'].nunique().to_dict()}; {len(co_registrants)} co-registrant CIK(s) not stored; "
        f"flags {build.flags['kind'].value_counts().sort_index().to_dict()}"
    )
    log_identity_flags(context.log, build.items)
    unchanged = matches_stored(existing, rows, Tables.security_master)
    written = 0 if unchanged else context.store.replace(Tables.security_master, rows)
    context.log.info("security_master: unchanged; replace skipped" if unchanged else f"security_master: wrote {written} row(s)")
    return rows
