"""Which company is this row about: the `Identity` accessor over `entity_lineage` and the roster.

`entity_lineage` holds, per entity, its CIK windows (consolidating filings), its event-only CIKs and
its dated symbol intervals. `filing_scope`, `ticker_for_cik` and `ticker_for_symbol` answer from those
rows; `tickers_for_ciks` and `symbol_rows_to_tickers` apply them to whole frames for the bulk data sets
and the symbol tapes. `security_on` answers from `security_master` (security grain: CUSIP, class, role). `owns(ticker, cik) == (entity_of(cik) == universe_entity(ticker))`. Invariant
violations raise at load, never per row. An unknown CIK is its own singleton entity `E{cik}`.
"""

from __future__ import annotations

import logging
import weakref
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from types import SimpleNamespace
from typing import Any, Literal, cast

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.entity_lineage import (
    FORM345_SOURCE,
    MANUAL_SOURCE,
    ROLE_EVENT,
    ROLE_SYMBOL,
    ROLE_WINDOW,
    ROSTER_COLUMNS,
    SENTINEL_START,
    IdentityError,
    TwoUniverseTickersOneEntityError,
    check_one_entity_per_cik,
    entity_by_cik_map,
    entity_or_singleton,
    roster_cik_map,
)
from src.data_extract.utils.common.security_master import squash
from src.data_extract.utils.common.symbol_tenure import DEI_SOURCE, normalise_market_symbol
from src.data_store.schema import Tables
from src.utils.string import normalise_ticker, pad_cik, pad_cik_series

logger = logging.getLogger(__name__)

#: Days a consolidating window is widened on each side of a seam between two windows of one entity.
SEAM_MARGIN_DAYS = 31
#: The `symbol_tenure` sources the resolver reads; `dei` evidence reaches it only through `entity_lineage`.
TENURE_SOURCES = (FORM345_SOURCE, MANUAL_SOURCE)
#: `event`: any CIK of the entity; `consolidating`: the CIK whose margin-widened window holds the filing date.
CikPolicy = Literal["event", "consolidating"]


class UnknownUniverseTickerError(IdentityError):
    """`universe_entity(T)` for a ticker absent from `sp500_tickers`, or holding no CIK."""


class AmbiguousSymbolTenureError(IdentityError):
    """A symbol resolves to more than one entity, at `as_of` or over all of history; never a silent pick."""


#: One axis-B tenure: (entity_id, valid_from, valid_to or None when open, n_filings).
TenureRow = tuple[str, pd.Timestamp, pd.Timestamp | None, int]
#: `resolution_verdict` of a symbol row: stored under `ticker`, or why not.
SymbolRowVerdict = Literal["resolved", "unresolved", "outside_universe", "redundant_share_class"]


#: Per-context cache (weak keys) so one database's identity never leaks into another context.
_CACHE: weakref.WeakKeyDictionary[Context, Identity] = weakref.WeakKeyDictionary()


def _as_timestamp(value) -> pd.Timestamp | None:
    """`None` for a null, a `Timestamp` for anything else (Postgres DATE returns `datetime.date`)."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, date | datetime | pd.Timestamp) or not pd.isna(value):
        return pd.Timestamp(value)
    return None


def _covers(start: pd.Timestamp | None, end: pd.Timestamp | None, day: pd.Timestamp) -> bool:
    """Half-open `start <= day < end`; a None bound is open."""
    return (start is None or start <= day) and (end is None or day < end)


@dataclass(frozen=True)
class CikWindow:
    """One CIK's consolidating window: declared `[valid_from, valid_to)` and the seam-widened `[listed_from, listed_to)`."""

    cik: str
    valid_from: pd.Timestamp | None
    valid_to: pd.Timestamp | None
    listed_from: pd.Timestamp | None
    listed_to: pd.Timestamp | None

    def owns(self, day: pd.Timestamp) -> bool:
        """Whether `day` falls inside the declared window."""
        return _covers(self.valid_from, self.valid_to, day)

    def admits(self, day: pd.Timestamp) -> bool:
        """Whether `day` falls inside the seam-widened window."""
        return _covers(self.listed_from, self.listed_to, day)


@dataclass(frozen=True)
class SymbolInterval:
    """One stored `symbol` row: the entity holding the symbol over `[valid_from, valid_to)` and its status."""

    entity: str
    valid_from: pd.Timestamp | None
    valid_to: pd.Timestamp | None
    status: str
    sources: frozenset[str] = frozenset()

    def covers(self, day: pd.Timestamp) -> bool:
        """Whether `day` falls inside the interval."""
        return _covers(self.valid_from, self.valid_to, day)

    @property
    def tape_symbol(self) -> bool:
        """False for an interval evidenced by cover-page `dei` alone: every listed security line of a filer
        (preferreds, notes, other share classes) carries one, so a symbol tape never maps it."""
        return self.sources != frozenset({DEI_SOURCE})


@dataclass(frozen=True)
class SecurityHit:
    """One `security_master` row's answer: which security, whose canonical company, in which role and class, at what ratio."""

    security_id: str
    canonical_company: str | None
    lineage_role: str
    security_class: str
    conversion_ratio: float


#: One dated master interval: (valid_from, valid_to or None when open, the hit).
SecurityInterval = tuple[pd.Timestamp, pd.Timestamp | None, SecurityHit]


@dataclass(frozen=True)
class FilingScope:
    """One universe ticker's filing scope: the only input of an EDGAR listing.

    `event_ciks` lists every CIK of the entity (event forms); `windows` the consolidating CIK windows,
    widened at seams; `scope_changed_at` the lineage timestamp of the last scope change.
    """

    ticker: str
    entity: str
    roster_cik: str
    event_ciks: tuple[str, ...]
    windows: tuple[CikWindow, ...]
    scope_changed_at: pd.Timestamp | None = None

    @classmethod
    def roster_only(cls, ticker: str, cik: str) -> FilingScope:
        """The scope of a ticker known only by its roster CIK: one open window, no other CIK."""
        key = pad_cik(cik)
        windows = (CikWindow(key, None, None, None, None),) if key else ()
        return cls(ticker=normalise_ticker(ticker), entity=f"E{key}", roster_cik=key, event_ciks=(key,) if key else (), windows=windows)


@dataclass(frozen=True)
class Identity:
    """The lineage accessor and the tenure resolver, validated once per run at construction and then read-only."""

    #: axis A: CIK -> entity_id, for the stored rows only. Absence means singleton.
    entity_by_cik: Mapping[str, str]
    #: universe ticker -> its roster CIK (10-digit).
    roster_cik: Mapping[str, str]
    #: entity_id -> the one universe ticker on it.
    ticker_by_entity: Mapping[str, str]
    #: axis B: symbol -> tuple of (entity_id, valid_from, valid_to, n_filings).
    tenure_by_symbol: Mapping[str, tuple[tuple[str, pd.Timestamp, pd.Timestamp | None, int], ...]]
    #: Manual subset of axis B; an active manual row takes precedence over derived evidence.
    manual_tenure_by_symbol: Mapping[str, tuple[tuple[str, pd.Timestamp, pd.Timestamp | None, int], ...]]
    #: Separately traded share classes deliberately absent from the modelling universe.
    redundant_symbols: frozenset[str]
    #: entity_id -> every CIK on its `cik_window` / `cik_event` rows, plus a universe ticker's roster CIK.
    event_ciks_by_entity: Mapping[str, frozenset[str]] = field(default_factory=dict)
    #: entity_id -> its seam-widened consolidating windows, oldest first.
    windows_by_entity: Mapping[str, tuple[CikWindow, ...]] = field(default_factory=dict)
    #: normalised symbol -> its stored `symbol` intervals.
    symbol_intervals: Mapping[str, tuple[SymbolInterval, ...]] = field(default_factory=dict)
    #: entity_id -> the latest `scope_changed_at` on its CIK rows.
    scope_changed_at_by_entity: Mapping[str, pd.Timestamp] = field(default_factory=dict)
    #: entity_id -> the latest `scope_changed_at` on its symbol rows (the symbol tapes' change stamp).
    symbols_changed_at_by_entity: Mapping[str, pd.Timestamp] = field(default_factory=dict)
    #: CUSIP-9 -> its dated `security_master` intervals.
    securities_by_cusip: Mapping[str, tuple[SecurityInterval, ...]] = field(default_factory=dict)
    #: (source, squashed source symbol) -> its dated `security_master` intervals.
    securities_by_symbol: Mapping[tuple[str, str], tuple[SecurityInterval, ...]] = field(default_factory=dict)

    # entity_lineage

    def entity_of(self, cik) -> str:
        """The entity a CIK belongs to. A CIK with no stored row IS its own entity."""
        return entity_or_singleton(self.entity_by_cik, pad_cik(cik))

    def universe_entity(self, ticker: str) -> str:
        """The entity of a universe ticker via its roster CIK; raises `UnknownUniverseTickerError` rather than returning None."""
        key = normalise_ticker(ticker)
        if key not in self.roster_cik:
            raise UnknownUniverseTickerError(
                f"identity: {key!r} is not a universe ticker in sp500_tickers (or its roster "
                f"row carries no CIK). {len(self.roster_cik)} ticker(s) are resolvable."
            )
        return self.entity_of(self.roster_cik[key])

    def filing_scope(self, ticker: str) -> FilingScope:
        """The ticker's event CIKs, seam-widened consolidating windows and scope timestamp."""
        key = normalise_ticker(ticker)
        entity = self.universe_entity(key)
        roster_cik = self.roster_cik[key]
        return FilingScope(
            ticker=key,
            entity=entity,
            roster_cik=roster_cik,
            event_ciks=tuple(sorted(self.event_ciks_by_entity.get(entity, frozenset({roster_cik})))),
            windows=self.windows_by_entity.get(entity, ()),
            scope_changed_at=self.scope_changed_at_by_entity.get(entity),
        )

    def symbols_changed_at(self, ticker: str) -> pd.Timestamp | None:
        """When the ticker's entity's symbol rows last changed; None when it has none."""
        return self.symbols_changed_at_by_entity.get(self.universe_entity(ticker))

    def ticker_for_cik(self, cik, filed=None, policy: CikPolicy = "event") -> str | None:
        """The universe ticker a filing by `cik` belongs to, or None.

        `event`: any CIK of a universe entity. `consolidating`: only when one of the CIK's
        seam-widened windows holds `filed`; no date means no answer.
        """
        key = pad_cik(cik)
        entity = self.entity_of(key)
        ticker = self.ticker_by_entity.get(entity)
        if ticker is None or key not in self.event_ciks_by_entity.get(entity, frozenset()):
            return None
        if policy == "event":
            return ticker
        stamp = _as_timestamp(filed)
        if stamp is None:
            return None
        return ticker if any(window.cik == key and window.admits(stamp) for window in self.windows_by_entity.get(entity, ())) else None

    def ticker_for_symbol(self, symbol: str, on, *, tape: bool = False) -> str | None:
        """The universe ticker holding `symbol` on date `on`, or None.

        `noise` intervals are ignored; a `conflict` interval on that date, or two entities, leaves it unresolved.
        `tape` (FTD, RegSHO) also ignores intervals evidenced by `dei` alone.
        """
        stamp = _as_timestamp(on)
        if stamp is None:
            return None
        rows = self.symbol_intervals.get(normalise_market_symbol(symbol), ())
        hits = [row for row in rows if row.status != "noise" and (row.tape_symbol or not tape) and row.covers(stamp)]
        if not hits or any(row.status == "conflict" for row in hits):
            return None
        entities = {row.entity for row in hits}
        return self.ticker_by_entity.get(entities.pop()) if len(entities) == 1 else None

    def universe_symbols(self, universe: Collection[str]) -> frozenset[str]:
        """The universe tickers plus every symbol a non-`noise` tape interval dates to one of their entities."""
        requested = frozenset(normalise_ticker(ticker) for ticker in universe)
        entities = {entity for entity, ticker in self.ticker_by_entity.items() if ticker in requested}
        dated = {
            symbol
            for symbol, rows in self.symbol_intervals.items()
            if any(row.status != "noise" and row.tape_symbol and row.entity in entities for row in rows)
        }
        return requested | dated

    def security_on(self, *, cusip: str | None = None, symbol: str | None = None, source: str, day: object) -> SecurityHit | None:
        """The security a tape line is on trade date `day`, by CUSIP when given, else by `(source, symbol)`.

        None when nothing covers the day, or when two securities do (a symbol `conflict`).
        """
        stamp = _as_timestamp(day)
        if stamp is None:
            return None
        rows = self.securities_by_cusip.get(str(cusip).strip().upper(), ()) if cusip else self.securities_by_symbol.get((source, squash(symbol)), ())
        hits = [hit for start, end, hit in rows if _covers(start, end, stamp)]
        if len({hit.security_id for hit in hits}) != 1:
            return None
        return hits[0]

    def owns(self, ticker: str, cik, on_date=None) -> bool:
        """Is this CIK's filing about this universe ticker's company?

        `on_date` is accepted but not read: union-policy forms (`registrant.FORM_POLICY`) keep a
        predecessor's filing past a registrant boundary. Callers pass `filing_date`, not `transaction_date`.
        """
        return self.entity_of(cik) == self.universe_entity(ticker)

    # symbol_tenure

    def entity_for(self, symbol: str, as_of=None) -> str | None:
        """The entity holding `symbol` at `as_of`; None when nobody did.

        Half-open `valid_from <= d < valid_to` (null end = open), like `registrant.Segment.covers`.
        An active manual tenure wins. Ambiguity is judged at entity grain, not CIK grain, and raises
        `AmbiguousSymbolTenureError`; without `as_of` it raises when more than one entity ever held it.
        """
        key = normalise_market_symbol(symbol)
        rows = self.tenure_by_symbol.get(key)
        if not rows:
            return None
        if as_of is None:
            entities = {entity for entity, _, _, _ in rows}
            if len(entities) > 1:
                raise AmbiguousSymbolTenureError(
                    f"identity: symbol {symbol!r} was held by {len(entities)} entities "
                    f"({', '.join(sorted(entities))}) and no date was given. Pass the filing "
                    "date; there is no defensible default and 'today' would silently pick "
                    "the incumbent for a row filed decades ago."
                )
            return next(iter(entities))

        stamp = pd.Timestamp(as_of)
        manual_rows = self.manual_tenure_by_symbol.get(key, ())
        manual_hits = {entity for entity, start, end, _ in manual_rows if start <= stamp and (end is None or stamp < end)}
        if len(manual_hits) > 1:
            raise AmbiguousSymbolTenureError(
                f"identity: symbol {symbol!r} has conflicting active MANUAL tenures at "
                f"{stamp.date()} ({', '.join(sorted(manual_hits))}). Fix the evidenced config; "
                "manual precedence cannot break a manual tie."
            )
        if manual_hits:
            return next(iter(manual_hits))
        derived_rows = tuple(row for row in rows if row not in manual_rows)
        hits = {entity for entity, start, end, _ in derived_rows if start <= stamp and (end is None or stamp < end)}
        if len(hits) > 1:
            raise AmbiguousSymbolTenureError(
                f"identity: symbol {symbol!r} resolves to {len(hits)} entities at "
                f"{stamp.date()} ({', '.join(sorted(hits))}). Two entities filing under one "
                "symbol on one date is a data condition worth reading, not a tie to break."
            )
        return next(iter(hits), None)


def tickers_for_ciks(identity: Identity, ciks: pd.Series, filed: pd.Series, policy: CikPolicy) -> pd.Series:
    """`ticker_for_cik` over aligned CIK and filing-date series, asked once per distinct pair; None where unresolved."""
    keys = pd.DataFrame({"cik": pad_cik_series(ciks).to_numpy(), "filed": pd.to_datetime(pd.Series(filed).to_numpy(), errors="coerce")})
    if policy == "event":
        keys["filed"] = pd.NaT  # an event filing belongs to the entity whatever its date
    pairs = keys.drop_duplicates(ignore_index=True)
    pairs["ticker"] = [identity.ticker_for_cik(cik, day, policy) for cik, day in zip(pairs["cik"], pairs["filed"], strict=True)]
    tickers = keys.merge(pairs, on=["cik", "filed"], how="left")["ticker"]
    return pd.Series(tickers.astype(object).where(tickers.notna(), None).to_numpy(), index=ciks.index, dtype=object)


def _symbol_row_verdict(identity: Identity, symbol: str, day: pd.Timestamp, requested: frozenset[str]) -> tuple[str | None, SymbolRowVerdict]:
    """The tape ticker of one (symbol, date) pair and its verdict against the requested universe."""
    ticker = identity.ticker_for_symbol(symbol, day, tape=True)
    if ticker is None:
        return ticker, "unresolved"
    if ticker not in requested:
        return ticker, "outside_universe"
    if symbol in identity.redundant_symbols and symbol not in requested and identity.ticker_for_symbol(ticker, day, tape=True) == ticker:
        return ticker, "redundant_share_class"
    return ticker, "resolved"


def symbol_rows_to_tickers(
    identity: Identity,
    frame: pd.DataFrame,
    universe: Collection[str],
    *,
    symbol_col: str = "source_symbol",
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(accepted, unresolved)` rows with `ticker` and `resolution_verdict`, one tape `ticker_for_symbol` call per distinct pair.

    A redundant share class (`redundant_symbols`, outside `universe`) is set aside while its entity's
    retained class trades, so two classes' rows are never added together.
    """
    requested = frozenset(normalise_ticker(ticker) for ticker in universe)
    work = frame.copy()
    work[symbol_col] = work[symbol_col].astype("string").str.strip().str.upper().str.replace(".", "-", regex=False).str.replace("/", "-", regex=False)
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    pairs = work[[symbol_col, date_col]].drop_duplicates(ignore_index=True)
    tickers: list[str | None] = []
    verdicts: list[SymbolRowVerdict] = []
    for symbol, day in pairs.itertuples(index=False, name=None):
        ticker, verdict = _symbol_row_verdict(identity, symbol, day, requested)
        tickers.append(ticker if verdict == "resolved" else None)
        verdicts.append(verdict)
    pairs["ticker"] = pd.Series(tickers, index=pairs.index, dtype=object)
    pairs["resolution_verdict"] = pd.Series(verdicts, index=pairs.index, dtype=object)
    resolved = work.merge(pairs, on=[symbol_col, date_col], how="left", validate="many_to_one")
    accepted = resolved["resolution_verdict"].eq("resolved")
    return resolved[accepted].copy(), resolved[~accepted].copy()


def log_symbol_resolutions(
    context: Context,
    source_name: str,
    accepted: pd.DataFrame,
    unresolved: pd.DataFrame,
    *,
    universe: frozenset[str],
    date_col: str = "date",
    symbol_col: str = "source_symbol",
) -> None:
    """Log compact verdict counts and every source-to-canonical relabel by name."""
    combined = pd.concat([accepted, unresolved], ignore_index=True)
    counts = combined["resolution_verdict"].value_counts().sort_index().to_dict()
    context.log.info(f"{source_name}: symbol-resolution verdicts {counts}")

    relabelled = accepted[accepted[symbol_col] != accepted["ticker"]]
    if not relabelled.empty:
        summary = (
            relabelled.groupby([symbol_col, "ticker"], as_index=False)
            .agg(rows=(date_col, "size"), first=(date_col, "min"), last=(date_col, "max"))
            .sort_values([symbol_col, "ticker"], kind="mergesort")
        )
        for row in summary.itertuples(index=False):
            context.log.info(
                f"{source_name}: {getattr(row, symbol_col)} -> {row.ticker}: "
                f"{row.rows} row(s), {pd.Timestamp(cast(Any, row.first)).date()}.."
                f"{pd.Timestamp(cast(Any, row.last)).date()}"
            )

    current_reuse = accepted[accepted[symbol_col].isin(universe) & (accepted[symbol_col] != accepted["ticker"])]
    if not current_reuse.empty:
        names = sorted(current_reuse[symbol_col].dropna().astype(str).unique())
        context.log.warning(f"{source_name}: current-looking symbol(s) resolved to another universe entity: {', '.join(names)}")

    if not unresolved.empty:
        by_verdict = unresolved.groupby("resolution_verdict")[symbol_col].agg(lambda values: ", ".join(sorted(set(map(str, values)))))
        for verdict, names in by_verdict.items():
            context.log.warning(f"{source_name}: {verdict}: {names}")


def build_identity(
    lineage: pd.DataFrame,
    tenure: pd.DataFrame,
    roster: pd.DataFrame,
    redundant_symbols: frozenset[str] | None = None,
    master: pd.DataFrame | None = None,
) -> Identity:
    """Validate the tables and return the frozen resolver; pure, no DB or config reads.

    `TwoUniverseTickersOneEntityError` is asserted before the reverse map is usable. D19 is the
    lineage build's check, not repeated here.
    """
    _require_tables(lineage, tenure, roster)
    check_one_entity_per_cik(lineage)
    entity_by_cik = entity_by_cik_map(lineage)
    roster_cik = roster_cik_map(roster)
    ticker_by_entity = _ticker_by_entity(roster_cik, entity_by_cik, lineage)
    tenure_by_symbol, manual_tenure_by_symbol = _tenure_maps(_tenure_evidence(tenure), entity_by_cik)
    event_ciks, windows = _cik_scopes(lineage, roster_cik, entity_by_cik)
    identity = Identity(
        entity_by_cik=entity_by_cik,
        roster_cik=roster_cik,
        ticker_by_entity=ticker_by_entity,
        tenure_by_symbol=tenure_by_symbol,
        manual_tenure_by_symbol=manual_tenure_by_symbol,
        redundant_symbols=frozenset(normalise_market_symbol(symbol) for symbol in (redundant_symbols or frozenset())),
        event_ciks_by_entity=event_ciks,
        windows_by_entity=windows,
        symbol_intervals=_symbol_intervals(lineage),
        scope_changed_at_by_entity=_scope_changed_at(lineage, symbols=False),
        symbols_changed_at_by_entity=_scope_changed_at(lineage, symbols=True),
        **_security_maps(master),
    )
    _log_identity(identity)
    return identity


def _require_tables(lineage: pd.DataFrame, tenure: pd.DataFrame, roster: pd.DataFrame) -> None:
    """Raise when any of the three mandatory identity inputs is missing or empty."""
    if lineage is None or lineage.empty:
        raise IdentityError(
            "identity: `entity_lineage` is empty. That is a LOUD failure and not an empty "
            "map: every universe ticker would fall back to a singleton entity, `owns()` "
            "would reject every predecessor row in the panel, and nothing would raise. Run "
            "`identity-tables` first. (`load_registrants` returning {} is a different case "
            "-- it is an OPTIONAL curated layer; these two tables are mandatory.)"
        )
    if tenure is None or tenure.empty:
        raise IdentityError(
            "identity: `symbol_tenure` is empty. The D19 cross-check would pass vacuously "
            "and Phase 6's symbol resolution would answer None for every symbol. Run "
            "`identity-tables` first."
        )
    if roster is None or roster.empty:
        raise IdentityError("identity: `sp500_tickers` is empty; there is no universe to resolve rows against.")


def _ticker_by_entity(roster_cik: Mapping[str, str], entity_by_cik: Mapping[str, str], lineage: pd.DataFrame) -> dict[str, str]:
    """`{entity_id: its one universe ticker}`; raises when an entity holds two universe tickers."""
    by_entity: dict[str, list[str]] = {}
    for ticker, cik in sorted(roster_cik.items()):
        by_entity.setdefault(entity_or_singleton(entity_by_cik, cik), []).append(ticker)
    collisions = {e: t for e, t in by_entity.items() if len(t) > 1}
    if not collisions:
        return {entity: tickers[0] for entity, tickers in by_entity.items()}
    entities = lineage["entity_id"].astype(str)
    detail = []
    for entity, tickers in sorted(collisions.items()):
        joined = lineage[entities == entity]
        rows = "; ".join(
            f"{pad_cik(r.cik)} via {getattr(r, 'oracle', None) or getattr(r, 'source', '')}" for r in joined.drop_duplicates("cik").itertuples()
        )
        detail.append(f"{entity} holds " + ", ".join(f"{t} (roster CIK {roster_cik[t]})" for t in tickers) + f" -- joined by: {rows}")
    raise TwoUniverseTickersOneEntityError(
        "identity: " + " | ".join(detail) + ". The reverse map is a dict, so one of these "
        "tickers would overwrite the other and EVERY ROW OF THE LOSER would be relabelled "
        "-- the only failure in this design that corrupts rather than drops. A spin-off "
        "into two index members (DowDuPont -> DD/DOW/CTVA) is TWO entities that share a "
        "past: split them with a curated row, never a wider merge."
    )


def _tenure_maps(tenure: pd.DataFrame, entity_by_cik: Mapping[str, str]) -> tuple[dict[str, tuple[TenureRow, ...]], dict[str, tuple[TenureRow, ...]]]:
    """(tenure rows by market symbol, the manual subset); a tenure with no `valid_from` cannot answer a dated test and is skipped."""
    tenure_by_symbol: dict[str, list[TenureRow]] = {}
    manual_tenure_by_symbol: dict[str, list[TenureRow]] = {}
    sources = tenure["source"].astype(str) if "source" in tenure.columns else pd.Series("form345", index=tenure.index)
    starts = pd.to_datetime(tenure["valid_from"])
    ends = pd.to_datetime(tenure["valid_to"])
    for symbol, cik, start, end, n, source in zip(
        tenure["symbol"].astype(str),
        pad_cik_series(tenure["issuer_cik"]),
        starts,
        ends,
        tenure["n_filings"],
        sources,
        strict=False,
    ):
        if start is pd.NaT:
            continue
        row = (entity_or_singleton(entity_by_cik, cik), start, None if end is pd.NaT else end, int(n))
        normalized_symbol = normalise_market_symbol(symbol)
        tenure_by_symbol.setdefault(normalized_symbol, []).append(row)
        if source.strip().lower() == "manual":
            manual_tenure_by_symbol.setdefault(normalized_symbol, []).append(row)
    return (
        {symbol: tuple(rows) for symbol, rows in tenure_by_symbol.items()},
        {symbol: tuple(rows) for symbol, rows in manual_tenure_by_symbol.items()},
    )


def _tenure_evidence(tenure: pd.DataFrame) -> pd.DataFrame:
    """The tenure rows the resolver reads: every source except `dei`, which reaches identity through `entity_lineage`."""
    if "source" not in tenure.columns:
        return tenure
    return tenure[tenure["source"].astype(str).str.strip().str.lower().ne(DEI_SOURCE)]


def _bound(value) -> pd.Timestamp | None:
    """A stored window start as a Timestamp; the open-start sentinel (or a null) reads as None."""
    stamp = _as_timestamp(value)
    return None if stamp is None or stamp <= SENTINEL_START else stamp


def _column(rows: pd.DataFrame, name: str) -> pd.Series:
    """`rows[name]`, or nulls when a projection lacks the column (an open end, an empty status)."""
    return rows[name] if name in rows.columns else pd.Series(None, index=rows.index, dtype=object)


def _roles(lineage: pd.DataFrame) -> pd.Series:
    """Each row's role; a frame without `role` holds membership rows only, read as event CIKs."""
    return lineage["role"].astype(str) if "role" in lineage.columns else pd.Series(ROLE_EVENT, index=lineage.index)


def _cik_scopes(
    lineage: pd.DataFrame, roster_cik: Mapping[str, str], entity_by_cik: Mapping[str, str]
) -> tuple[dict[str, frozenset[str]], dict[str, tuple[CikWindow, ...]]]:
    """(entity -> event CIKs, entity -> seam-widened windows); an entity with no `cik_window` row reads its roster CIK as one open window."""
    roles = _roles(lineage)
    rows = lineage[roles.isin((ROLE_WINDOW, ROLE_EVENT))]
    roles = roles[rows.index]
    ciks = pad_cik_series(rows["cik"])
    entities = rows["entity_id"].astype(str)
    event: dict[str, set[str]] = {}
    declared: dict[str, list[tuple[str, pd.Timestamp | None, pd.Timestamp | None]]] = {}
    for cik, entity in zip(ciks, entities, strict=True):
        event.setdefault(entity, set()).add(cik)
    is_window = roles.eq(ROLE_WINDOW)
    if is_window.any():
        windows = rows[is_window]
        for cik, entity, start, end in zip(ciks[is_window], entities[is_window], windows["valid_from"], _column(windows, "valid_to"), strict=True):
            declared.setdefault(entity, []).append((cik, _bound(start), _as_timestamp(end)))
    for cik in roster_cik.values():
        entity = entity_or_singleton(entity_by_cik, cik)
        event.setdefault(entity, set()).add(cik)
        declared.setdefault(entity, [(cik, None, None)])
    return {entity: frozenset(values) for entity, values in event.items()}, {entity: _widen_seams(values) for entity, values in declared.items()}


def _widen_seams(windows: list[tuple[str, pd.Timestamp | None, pd.Timestamp | None]]) -> tuple[CikWindow, ...]:
    """Windows oldest first; a bound within `SEAM_MARGIN_DAYS` of another window's opposite bound is widened by that margin."""
    margin = pd.Timedelta(days=SEAM_MARGIN_DAYS)
    ordered = sorted(windows, key=lambda window: (window[1] or SENTINEL_START, window[0]))
    out: list[CikWindow] = []
    for i, (cik, start, end) in enumerate(ordered):
        others = ordered[:i] + ordered[i + 1 :]
        seam_before = start is not None and any(other_end is not None and abs(other_end - start) <= margin for _, _, other_end in others)
        seam_after = end is not None and any(other_start is not None and abs(other_start - end) <= margin for _, other_start, _ in others)
        out.append(
            CikWindow(
                cik=cik,
                valid_from=start,
                valid_to=end,
                listed_from=start - margin if start is not None and seam_before else start,
                listed_to=end + margin if end is not None and seam_after else end,
            )
        )
    return tuple(out)


def _symbol_intervals(lineage: pd.DataFrame) -> dict[str, tuple[SymbolInterval, ...]]:
    """`{normalised symbol: its stored symbol intervals}`; empty for a frame without `role`."""
    if "role" not in lineage.columns:
        return {}
    rows = lineage[lineage["role"].astype(str).eq(ROLE_SYMBOL)]
    out: dict[str, list[SymbolInterval]] = {}
    for symbol, entity, start, end, status, sources in zip(
        rows["symbol"].astype(str),
        rows["entity_id"].astype(str),
        rows["valid_from"],
        _column(rows, "valid_to"),
        _column(rows, "status").fillna("").astype(str),
        _column(rows, "sources").fillna("").astype(str),
        strict=True,
    ):
        evidence = frozenset(part.strip().lower() for part in sources.split(",") if part.strip())
        out.setdefault(normalise_market_symbol(symbol), []).append(SymbolInterval(entity, _bound(start), _as_timestamp(end), status, evidence))
    return {symbol: tuple(values) for symbol, values in out.items()}


def _security_maps(master: pd.DataFrame | None) -> dict[str, Any]:
    """`securities_by_cusip` and `securities_by_symbol` from `security_master` rows (newest interval first)."""
    if master is None or master.empty:
        return {}
    by_cusip: dict[str, list[SecurityInterval]] = {}
    by_symbol: dict[tuple[str, str], list[SecurityInterval]] = {}
    ordered = master.sort_values(["valid_from", "security_id", "source_symbol"], ascending=[False, True, True], kind="mergesort")
    for record in cast(list[Any], ordered.to_dict("records")):
        row = SimpleNamespace(**record)
        company = row.canonical_company if isinstance(row.canonical_company, str) else None
        hit = SecurityHit(str(row.security_id), company, str(row.lineage_role), str(row.security_class), float(row.conversion_ratio))
        interval = (pd.Timestamp(row.valid_from), _as_timestamp(row.valid_to), hit)
        if isinstance(row.cusip, str) and row.cusip:
            by_cusip.setdefault(row.cusip, []).append(interval)
        by_symbol.setdefault((str(row.source), squash(row.source_symbol)), []).append(interval)
    return {
        "securities_by_cusip": {key: tuple(values) for key, values in by_cusip.items()},
        "securities_by_symbol": {key: tuple(values) for key, values in by_symbol.items()},
    }


def _scope_changed_at(lineage: pd.DataFrame, *, symbols: bool) -> dict[str, pd.Timestamp]:
    """`{entity_id: latest scope_changed_at}` over its symbol rows (`symbols`) or its CIK rows."""
    if "scope_changed_at" not in lineage.columns:
        return {}
    rows = lineage[_roles(lineage).eq(ROLE_SYMBOL) == symbols]
    stamps = pd.to_datetime(rows["scope_changed_at"], errors="coerce")
    latest = stamps.groupby(rows["entity_id"].astype(str)).max().dropna()
    return {str(entity): pd.Timestamp(value) for entity, value in latest.items()}


def _log_identity(identity: Identity) -> None:
    """One line of map sizes for the resolver just built."""
    logger.info(
        "identity: %d lineage CIK(s) over %d entity(ies); %d universe ticker(s); %d windowed entity(ies); "
        "%d symbol(s) with lineage intervals; %d symbol(s) with tenure; %d manual symbol(s); %d redundant symbol(s); %d master CUSIP(s)",
        len(identity.entity_by_cik),
        len(set(identity.entity_by_cik.values())),
        len(identity.roster_cik),
        len(identity.windows_by_entity),
        len(identity.symbol_intervals),
        len(identity.tenure_by_symbol),
        len(identity.manual_tenure_by_symbol),
        len(identity.redundant_symbols),
        len(identity.securities_by_cusip),
    )


def load_identity(context: Context, refresh: bool = False) -> Identity:
    """The resolver for this run, built once per context and cached on it (`refresh` rebuilds)."""
    cached = None if refresh else _CACHE.get(context)
    if cached is not None:
        return cached
    lineage = context.store.load(Tables.entity_lineage, project=True)
    tenure = context.store.load(Tables.symbol_tenure, project=True, where={"source": list(TENURE_SOURCES)})
    roster = context.store.load(Tables.sp500_tickers, columns=list(ROSTER_COLUMNS))
    master = context.store.load(Tables.security_master, project=True, optional=True)
    assert lineage is not None and tenure is not None and roster is not None
    identity = build_identity(
        lineage=lineage,
        tenure=tenure,
        roster=roster,
        redundant_symbols=frozenset(context.config.data_extract.redundant_ticks),
        master=master,
    )
    _CACHE[context] = identity
    return identity
