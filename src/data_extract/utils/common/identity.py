"""Which company is this row about: the `Identity` resolver over two separate axes.

Axis A, `entity_lineage`: which CIKs are the same company (the curated register is folded in there,
never parsed here). Axis B, `symbol_tenure`: who held symbol X on date d. The CIK predicate is
`owns(ticker, cik) == (entity_of(cik) == universe_entity(ticker))` and does not read tenure; tenure
serves the D19 cross-check and CIK-less symbol/date sources. Invariant violations raise at load,
never per row (an unresolvable row is quarantined by the caller). An unknown CIK is its own
singleton entity `E{cik}`.
"""

from __future__ import annotations

import logging
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal, cast

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.entity_lineage import (
    ROSTER_COLUMNS,
    TwoUniverseTickersOneEntityError,
    entity_by_cik_map,
    entity_or_singleton,
    load_d19_allowlist,
    roster_cik_map,
)
from src.data_store.schema import Tables
from src.utils.string import normalise_ticker, pad_cik, pad_cik_series

logger = logging.getLogger(__name__)


class IdentityError(ValueError):
    """Base of every identity failure, so one `except` covers the whole layer."""


class UnknownUniverseTickerError(IdentityError):
    """`universe_entity(T)` for a ticker absent from `sp500_tickers`, or holding no CIK."""


class CikInTwoEntitiesError(IdentityError):
    """One CIK carries two `entity_id`s; guards the builder, since the table's PK is `cik`."""


class UniverseEntityDisagreementError(IdentityError):
    """D19: the roster CIK and `symbol_tenure` name different entities for one ticker, with no D19 allow-list entry."""


class AmbiguousSymbolTenureError(IdentityError):
    """A symbol resolves to more than one entity, at `as_of` or over all of history; never a silent pick."""


SymbolVerdict = Literal[
    "exact_dated_tenure",
    "roster_tenure_proxy",
    "mapped_current_ticker",
    "redundant_share_class",
    "entity_not_in_universe",
    "unknown_symbol",
    "unknown_gap",
    "ambiguous",
]
SymbolMatchKind = Literal[
    "exact_dated_tenure",
    "roster_tenure_proxy",
]
#: One axis-B tenure: (entity_id, valid_from, valid_to or None when open, n_filings).
TenureRow = tuple[str, pd.Timestamp, pd.Timestamp | None, int]


@dataclass(frozen=True)
class SymbolResolution:
    """Point-in-time source-symbol verdict and optional canonical universe ticker."""

    source_symbol: str
    as_of: pd.Timestamp | None
    verdict: SymbolVerdict
    match_kind: SymbolMatchKind | None = None
    entity_id: str | None = None
    ticker: str | None = None

    @property
    def accepted(self) -> bool:
        """Whether the row may be stored under `ticker`."""
        return self.ticker is not None

    @property
    def relabelled(self) -> bool:
        """Whether the source symbol differs from the stored canonical ticker."""
        return self.ticker is not None and self.source_symbol != self.ticker


#: Per-context cache (weak keys) so one database's identity never leaks into another context.
_CACHE: weakref.WeakKeyDictionary[Context, Identity] = weakref.WeakKeyDictionary()


def normalise_market_symbol(value: object) -> str:
    """Use the roster's hyphen spelling for market share-class separators."""
    return str(value).strip().upper().replace(".", "-").replace("/", "-")


def _as_timestamp(value) -> pd.Timestamp | None:
    """`None` for a null, a `Timestamp` for anything else (Postgres DATE returns `datetime.date`)."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, date | datetime | pd.Timestamp) or not pd.isna(value):
        return pd.Timestamp(value)
    return None


@dataclass(frozen=True)
class FilingScope:
    """One universe ticker's identity-discovered filing scope.

    `ciks` is every CIK on the ticker's entity (stored lineage, roster CIK and tenure issuers);
    `symbols` the ticker plus every tenure symbol on the entity; `aliases` the other symbols
    filed under the roster CIK itself. All three are sorted.
    """

    ticker: str
    entity: str
    roster_cik: str
    ciks: tuple[str, ...]
    symbols: tuple[str, ...]
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class Identity:
    """Both identity axes, validated once per run at construction and then read-only."""

    #: axis A: CIK -> entity_id, for the stored rows only. Absence means singleton.
    entity_by_cik: Mapping[str, str]
    #: universe ticker -> its roster CIK (10-digit).
    roster_cik: Mapping[str, str]
    #: entity_id -> the one universe ticker on it (read by `entity_ticker`).
    ticker_by_entity: Mapping[str, str]
    #: axis B: symbol -> tuple of (entity_id, valid_from, valid_to, n_filings).
    tenure_by_symbol: Mapping[str, tuple[tuple[str, pd.Timestamp, pd.Timestamp | None, int], ...]]
    #: Manual subset of axis B; an active manual row takes precedence over derived evidence.
    manual_tenure_by_symbol: Mapping[str, tuple[tuple[str, pd.Timestamp, pd.Timestamp | None, int], ...]]
    #: D19-cleared roster symbol -> dated rows borrowed from filing symbols on its entity.
    roster_proxy_by_symbol: Mapping[str, tuple[tuple[str, pd.Timestamp, pd.Timestamp | None, int], ...]]
    #: Separately traded share classes deliberately absent from the modelling universe.
    redundant_symbols: frozenset[str]
    #: Raw symbol -> observed issuer CIKs, retained for filing-scope discovery.
    ciks_by_symbol: Mapping[str, frozenset[str]] = field(default_factory=dict)
    #: entity_id -> its stored CIKs (the inverse of `entity_by_cik`).
    ciks_by_entity: Mapping[str, frozenset[str]] = field(default_factory=dict)
    #: entity_id -> sorted (normalised symbol, padded CIK) pairs from `ciks_by_symbol`.
    scope_pairs_by_entity: Mapping[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)

    # axis A

    def entity_of(self, cik) -> str:
        """The entity a CIK belongs to. A CIK with no stored row IS its own entity."""
        return entity_or_singleton(self.entity_by_cik, pad_cik(cik))

    def ciks_for(self, entity_id: str) -> frozenset[str]:
        """Every stored CIK on an entity; empty for a singleton entity, whose CIK is its id minus `"E"`."""
        return self.ciks_by_entity.get(entity_id, frozenset())

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
        """The ticker's identity-discovered CIKs, symbols and same-CIK aliases."""
        key = normalise_ticker(ticker)
        entity = self.universe_entity(key)
        roster_cik = self.roster_cik[key]
        pairs = self.scope_pairs_by_entity.get(entity, ())
        return FilingScope(
            ticker=key,
            entity=entity,
            roster_cik=roster_cik,
            ciks=tuple(sorted(self.ciks_for(entity) | {roster_cik} | {cik for _, cik in pairs})),
            symbols=tuple(sorted({key} | {symbol for symbol, _ in pairs})),
            aliases=tuple(sorted({symbol for symbol, cik in pairs if cik == roster_cik and symbol and symbol != key})),
        )

    def entity_ticker(self, cik) -> str | None:
        """Today's universe ticker for a CIK's entity, or None when its entity holds none (CIK-first resolution)."""
        return self.ticker_by_entity.get(self.entity_of(cik))

    def owns(self, ticker: str, cik, on_date=None) -> bool:
        """Is this CIK's filing about this universe ticker's company?

        `on_date` is accepted but not read: union-policy forms (`registrant.FORM_POLICY`) keep a
        predecessor's filing past a registrant boundary. Callers pass `filing_date`, not `transaction_date`.
        """
        return self.entity_of(cik) == self.universe_entity(ticker)

    # axis B

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

    def dominant_entity(self, symbol: str) -> str | None:
        """The entity that owns a symbol today, by weight of filings rather than recency (used by D19).

        Prefers open manual tenures, then open tenures, then any, taking the most filings, so a one-filing typo tenure never wins.
        """
        key = normalise_market_symbol(symbol)
        rows = self.tenure_by_symbol.get(key)
        if not rows:
            return None
        manual_rows = self.manual_tenure_by_symbol.get(key, ())
        manual_open = [row for row in manual_rows if row[2] is None]
        if manual_open:
            return max(manual_open, key=lambda row: row[3])[0]
        openrows = [r for r in rows if r[2] is None]
        if openrows:
            return max(openrows, key=lambda row: row[3])[0]
        return max(manual_rows or rows, key=lambda row: row[3])[0]

    def candidate_symbols(self, universe: frozenset[str]) -> frozenset[str]:
        """Current symbols plus historical symbols owned by the requested universe."""
        requested = frozenset(normalise_ticker(ticker) for ticker in universe)
        historical = {
            symbol
            for symbol, rows in self.tenure_by_symbol.items()
            if any(self.ticker_by_entity.get(entity) in requested for entity, _, _, _ in rows)
        }
        return requested | historical

    def resolve_symbol_ticker(
        self,
        symbol: str,
        as_of: object,
        universe: frozenset[str],
    ) -> SymbolResolution:
        """Resolve one historical symbol/date to the caller's canonical universe ticker."""
        source_symbol = normalise_market_symbol(symbol)
        stamp = _as_timestamp(as_of)
        requested = frozenset(normalise_ticker(ticker) for ticker in universe)
        rows = self.tenure_by_symbol.get(source_symbol)
        is_roster_proxy = rows is None and source_symbol in self.roster_proxy_by_symbol
        if is_roster_proxy:
            rows = self.roster_proxy_by_symbol[source_symbol]
        if not rows:
            return SymbolResolution(source_symbol, stamp, "unknown_symbol")

        match_kind: SymbolMatchKind
        entity_id: str | None
        if is_roster_proxy:
            if stamp is None:
                return SymbolResolution(source_symbol, stamp, "unknown_gap")
            dated_hits = {entity for entity, start, end, _ in rows if start <= stamp and (end is None or stamp < end)}
            if len(dated_hits) > 1:
                return SymbolResolution(source_symbol, stamp, "ambiguous")
            if not dated_hits:
                return SymbolResolution(source_symbol, stamp, "unknown_gap")
            entity_id = next(iter(dated_hits))
            match_kind = "roster_tenure_proxy"
        else:
            if stamp is None:
                return SymbolResolution(source_symbol, stamp, "unknown_gap")
            try:
                entity_id = self.entity_for(source_symbol, stamp)
            except AmbiguousSymbolTenureError:
                return SymbolResolution(source_symbol, stamp, "ambiguous")
            if entity_id is None:
                # A last Form 4 is not a delisting: extend only the latest closed roster-entity interval, never a manual end.
                latest_start = max(start for _, start, _, _ in rows)
                latest = [row for row in rows if row[1] == latest_start]
                roster_entity = self.entity_of(self.roster_cik[source_symbol]) if source_symbol in self.roster_cik else None
                if (
                    roster_entity is None
                    or any(row[0] != roster_entity or row[2] is None or stamp < row[2] for row in latest)
                    or any(row in self.manual_tenure_by_symbol.get(source_symbol, ()) for row in latest)
                ):
                    return SymbolResolution(source_symbol, stamp, "unknown_gap")
                entity_id = roster_entity
                match_kind = "roster_tenure_proxy"
            else:
                match_kind = "exact_dated_tenure"

        ticker = self.ticker_by_entity.get(entity_id)
        if ticker is None or ticker not in requested:
            return SymbolResolution(
                source_symbol,
                stamp,
                "entity_not_in_universe",
                match_kind=match_kind,
                entity_id=entity_id,
            )

        # A redundant class is rejected only while the retained class is concurrently active (else it is a predecessor spelling).
        target_rows = self.tenure_by_symbol.get(ticker) or self.roster_proxy_by_symbol.get(ticker, ())
        target_is_concurrent = stamp is not None and any(
            target_entity == entity_id and start <= stamp and (end is None or stamp < end) for target_entity, start, end, _ in target_rows
        )
        if source_symbol in self.redundant_symbols and source_symbol not in requested and target_is_concurrent:
            return SymbolResolution(
                source_symbol,
                stamp,
                "redundant_share_class",
                match_kind=match_kind,
                entity_id=entity_id,
            )

        verdict: SymbolVerdict = "mapped_current_ticker" if ticker != source_symbol else match_kind
        return SymbolResolution(
            source_symbol,
            stamp,
            verdict,
            match_kind=match_kind,
            entity_id=entity_id,
            ticker=ticker,
        )


def resolve_symbol_rows(
    identity: Identity,
    frame: pd.DataFrame,
    universe: frozenset[str],
    *,
    symbol_col: str = "source_symbol",
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve unique symbol/date pairs once and merge their verdicts onto source rows."""
    if frame.empty:
        empty = frame.copy()
        for column in ("ticker", "resolution_verdict", "resolution_match", "resolution_entity"):
            empty[column] = pd.Series(dtype="object")
        return empty, empty.copy()

    work = frame.copy()
    work[symbol_col] = work[symbol_col].astype("string").str.strip().str.upper().str.replace(".", "-", regex=False).str.replace("/", "-", regex=False)
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    pairs = work[[symbol_col, date_col]].drop_duplicates(ignore_index=True)
    records = []
    for source_symbol, day in pairs.itertuples(index=False, name=None):
        resolution = identity.resolve_symbol_ticker(source_symbol, day, universe)
        records.append(
            {
                symbol_col: source_symbol,
                date_col: day,
                "ticker": resolution.ticker,
                "resolution_verdict": resolution.verdict,
                "resolution_match": resolution.match_kind,
                "resolution_entity": resolution.entity_id,
            }
        )
    verdicts = pd.DataFrame.from_records(records)
    resolved = work.merge(verdicts, on=[symbol_col, date_col], how="left", validate="many_to_one")
    accepted = resolved[resolved["ticker"].notna()].copy()
    unresolved = resolved[resolved["ticker"].isna()].copy()
    return accepted, unresolved


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
    d19_allowlist: Mapping[str, str] | None = None,
    redundant_symbols: frozenset[str] | None = None,
) -> Identity:
    """Validate the tables and return the frozen resolver; pure, no DB or config reads.

    `TwoUniverseTickersOneEntityError` is asserted before the reverse map is usable, then D19 is checked.
    """
    _require_tables(lineage, tenure, roster)
    _check_one_entity_per_cik(lineage)
    entity_by_cik = entity_by_cik_map(lineage)
    roster_cik = roster_cik_map(roster)
    ticker_by_entity = _ticker_by_entity(roster_cik, entity_by_cik, lineage)
    tenure_by_symbol, manual_tenure_by_symbol, ciks_by_symbol = _tenure_maps(tenure, entity_by_cik)
    allowlist = d19_allowlist or {}
    ciks_by_entity, scope_pairs_by_entity = _scope_maps(entity_by_cik, ciks_by_symbol)
    identity = Identity(
        entity_by_cik=entity_by_cik,
        roster_cik=roster_cik,
        ticker_by_entity=ticker_by_entity,
        tenure_by_symbol=tenure_by_symbol,
        manual_tenure_by_symbol=manual_tenure_by_symbol,
        roster_proxy_by_symbol=_roster_proxies(allowlist, roster_cik, entity_by_cik, tenure_by_symbol),
        redundant_symbols=frozenset(normalise_market_symbol(symbol) for symbol in (redundant_symbols or frozenset())),
        ciks_by_symbol=ciks_by_symbol,
        ciks_by_entity=ciks_by_entity,
        scope_pairs_by_entity=scope_pairs_by_entity,
    )
    _check_d19(identity, allowlist)
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


def _check_one_entity_per_cik(lineage: pd.DataFrame) -> None:
    """Raise when one CIK carries two entity_ids, which would make `entity_of` order-dependent."""
    per_cik = pd.DataFrame({"cik": pad_cik_series(lineage["cik"]), "entity_id": lineage["entity_id"].astype(str)}).drop_duplicates()
    clashes = per_cik[per_cik.duplicated("cik", keep=False)]
    if not clashes.empty:
        raise CikInTwoEntitiesError(
            f"identity: {clashes['cik'].nunique()} CIK(s) carry two entity_ids in "
            f"entity_lineage -- {clashes.sort_values('cik').to_dict('records')}. The table's "
            "primary key is `cik`, so this cannot come from the database; it is a builder "
            "bug, and it would make `entity_of` order-dependent."
        )


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
        rows = "; ".join(f"{pad_cik(r.cik)} via {r.source}" for r in joined.itertuples())
        detail.append(f"{entity} holds " + ", ".join(f"{t} (roster CIK {roster_cik[t]})" for t in tickers) + f" -- joined by: {rows}")
    raise TwoUniverseTickersOneEntityError(
        "identity: " + " | ".join(detail) + ". The reverse map is a dict, so one of these "
        "tickers would overwrite the other and EVERY ROW OF THE LOSER would be relabelled "
        "-- the only failure in this design that corrupts rather than drops. A spin-off "
        "into two index members (DowDuPont -> DD/DOW/CTVA) is TWO entities that share a "
        "past: split them with a curated row, never a wider merge."
    )


def _tenure_maps(
    tenure: pd.DataFrame, entity_by_cik: Mapping[str, str]
) -> tuple[dict[str, tuple[TenureRow, ...]], dict[str, tuple[TenureRow, ...]], dict[str, frozenset[str]]]:
    """(tenure rows by market symbol, the manual subset, issuer CIKs by raw symbol).

    A tenure with no `valid_from` cannot answer a dated test, so it feeds only the CIK map.
    """
    tenure_by_symbol: dict[str, list[TenureRow]] = {}
    manual_tenure_by_symbol: dict[str, list[TenureRow]] = {}
    ciks_by_symbol: dict[str, set[str]] = {}
    sources = tenure["source"].astype(str) if "source" in tenure.columns else pd.Series("form345", index=tenure.index)
    starts = pd.to_datetime(tenure["valid_from"])
    ends = pd.to_datetime(tenure["valid_to"])
    for has_symbol, symbol, cik, start, end, n, source in zip(
        tenure["symbol"].notna(),
        tenure["symbol"].astype(str),
        pad_cik_series(tenure["issuer_cik"]),
        starts,
        ends,
        tenure["n_filings"],
        sources,
        strict=False,
    ):
        if has_symbol:
            ciks_by_symbol.setdefault(symbol, set()).add(cik)
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
        {symbol: frozenset(symbol_ciks) for symbol, symbol_ciks in ciks_by_symbol.items()},
    )


def _roster_proxies(
    allowlist: Mapping[str, str],
    roster_cik: Mapping[str, str],
    entity_by_cik: Mapping[str, str],
    tenure_by_symbol: Mapping[str, tuple[TenureRow, ...]],
) -> dict[str, tuple[TenureRow, ...]]:
    """D19-cleared roster spellings absent from Form 345 -> the dated tenures of every filing symbol on the roster entity.

    Every entity seen under those symbols is kept, so the date settles the boundary.
    """
    roster_proxy_by_symbol: dict[str, tuple[TenureRow, ...]] = {}
    for ticker in sorted(set(allowlist) & set(roster_cik)):
        if ticker in tenure_by_symbol:
            continue
        roster_entity = entity_or_singleton(entity_by_cik, roster_cik[ticker])
        proxy_symbols = {symbol for symbol, rows in tenure_by_symbol.items() if any(entity == roster_entity for entity, _, _, _ in rows)}
        proxy_rows = tuple(row for symbol in sorted(proxy_symbols) for row in tenure_by_symbol[symbol])
        if proxy_rows:
            roster_proxy_by_symbol[ticker] = proxy_rows
    return roster_proxy_by_symbol


def _scope_maps(
    entity_by_cik: Mapping[str, str], ciks_by_symbol: Mapping[str, frozenset[str]]
) -> tuple[dict[str, frozenset[str]], dict[str, tuple[tuple[str, str], ...]]]:
    """(entity -> its stored CIKs, entity -> sorted (normalised symbol, CIK) filing-scope pairs)."""
    ciks_by_entity: dict[str, set[str]] = {}
    for cik, entity in entity_by_cik.items():
        ciks_by_entity.setdefault(entity, set()).add(cik)
    scope_pairs: dict[str, set[tuple[str, str]]] = {}
    for symbol, symbol_ciks in ciks_by_symbol.items():
        for cik in symbol_ciks:
            scope_pairs.setdefault(entity_or_singleton(entity_by_cik, cik), set()).add((normalise_ticker(symbol), cik))
    return (
        {entity: frozenset(entity_ciks) for entity, entity_ciks in ciks_by_entity.items()},
        {entity: tuple(sorted(pairs)) for entity, pairs in scope_pairs.items()},
    )


def _log_identity(identity: Identity) -> None:
    """One line of map sizes for the resolver just built."""
    logger.info(
        "identity: %d lineage CIK(s) over %d entity(ies); %d universe ticker(s); "
        "%d symbol(s) with tenure; %d manual symbol(s); %d D19 roster proxy symbol(s); "
        "%d redundant symbol(s)",
        len(identity.entity_by_cik),
        len(set(identity.entity_by_cik.values())),
        len(identity.roster_cik),
        len(identity.tenure_by_symbol),
        len(identity.manual_tenure_by_symbol),
        len(identity.roster_proxy_by_symbol),
        len(identity.redundant_symbols),
    )


def _check_d19(identity: Identity, allowlist: Mapping[str, str]) -> None:
    """D19: the roster CIK must name the same entity `symbol_tenure` does, unless the allow-list explains why not."""
    disagree = []
    for ticker, cik in sorted(identity.roster_cik.items()):
        from_tenure = identity.dominant_entity(ticker)
        if from_tenure is None:
            disagree.append((ticker, cik, "NO TENURE"))
        elif from_tenure != identity.entity_of(cik):
            disagree.append((ticker, cik, from_tenure))
    unexplained = [d for d in disagree if d[0] not in allowlist]
    if unexplained:
        listed = "; ".join(f"{t}: roster {c} -> {identity.entity_of(c)} but tenure -> {e}" for t, c, e in unexplained)
        raise UniverseEntityDisagreementError(
            f"identity: {len(unexplained)} universe ticker(s) whose roster CIK and whose "
            f"filings name different entities -- {listed}. Each is the XOM class of defect "
            "(a Wikipedia-sourced CIK pointing at a shell) until a written reading says "
            "otherwise. Fix the roster CIK, or add the ticker to `_d19_allowlist` in "
            "entity_lineage_manual.json WITH the evidence."
        )
    if disagree:
        logger.info(
            "identity: D19 cross-check -- %d/%d tickers agree, %d allow-listed with evidence",
            len(identity.roster_cik) - len(disagree),
            len(identity.roster_cik),
            len(disagree),
        )


def load_identity(context: Context, config_dir: str | None = None, refresh: bool = False) -> Identity:
    """The resolver for this run, built once per context and cached on it (`refresh` rebuilds)."""
    cached = None if refresh else _CACHE.get(context)
    if cached is not None:
        return cached
    lineage = context.store.load(Tables.entity_lineage, project=True)
    tenure = context.store.load(Tables.symbol_tenure, project=True)
    roster = context.store.load(Tables.sp500_tickers, columns=list(ROSTER_COLUMNS))
    assert lineage is not None and tenure is not None and roster is not None
    identity = build_identity(
        lineage=lineage,
        tenure=tenure,
        roster=roster,
        d19_allowlist=load_d19_allowlist(config_dir or str(context.config_dir)),
        redundant_symbols=frozenset(context.config.data_extract.redundant_ticks),
    )
    _CACHE[context] = identity
    return identity
