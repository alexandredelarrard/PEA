"""
identity.py (src/data_extract/utils/common/identity.py)
--------------------------------------------------------------------------------------------
WHICH COMPANY IS THIS ROW ABOUT? Two axes, kept separate on purpose.

  A -- entity continuity : which CIKs are the same economic company?   -> `entity_lineage`
  B -- symbol tenure     : who held symbol X on date d?                -> `symbol_tenure`

`registrant.py` is the CURATED layer above axis A and always wins. This module reads it
through `entity_lineage`, which folds the register in as its highest-priority oracle, and it
NEVER parses `registrant_cutover.json` itself -- one reader, one set of validations.

THE PREDICATE, IN FULL:

    owns(ticker, cik, on_date) -> entity_of(cik) == universe_entity(ticker)

`symbol_tenure` is DELIBERATELY NOT IN THAT HOT PATH, and the `IR` case is why. CIK
`0001466258` filed under `IR` for eleven years and is `TT`'s registrant today. A single-axis
"does this CIK own this symbol" oracle scores it "emphatically the same people" and is RIGHT
about continuity -- it is simply not `IR`'s entity. Composing the axes settles it with no
`IR`-specific rule anywhere in the code or the configs. Tenure earns its place elsewhere: it
is the discovery substrate axis A is built from, it is the D19 cross-check below, and it is
the only resolver `sec_fails_to_deliver` / `sec_short_interest` can use at all, because those
carry a symbol and a date and NO CIK.

⚠ RESOLUTION IS CIK-FIRST, AND THAT ADMITS ROWS AS WELL AS REJECTING THEM. `entity_ticker`
resolves the row's CIK to an entity and then to today's universe ticker. That is the inverse
of the symbol-first line it replaces, so a predecessor which changed BOTH its CIK and its
trading symbol -- invisible to the symbol path, and absent from `cik_to_ticker` unless it is
one of the register's hand entries -- now resolves instead of being silently dropped. Report
`n_admitted` next to `n_quarantined`: the change is not purely subtractive.

WHAT RAISES, AND WHEN. D9 says raise on unresolved and on ambiguous; D9a places those raises
at LOAD, never per row. A per-row raise would abort a 4.3M-filing parse on one unknown CIK,
so a row that cannot be resolved is quarantined with a reason and the run continues. The five
load-time raises are each an invariant whose violation would corrupt or silently empty the
panel -- above all `TwoUniverseTickersOneEntity`, which is asserted before any other map is
built because it is the only failure in this design that RELABELS rows rather than dropping
them.

⚠ `entity_of` ON AN UNKNOWN CIK IS A VERDICT, NOT A GAP. It returns `f"E{cik}"`, a singleton
entity, because `entity_lineage` stores only non-singleton groups and the roster (see
`schema.py`). A CIK with no row is its own company, which is exactly the answer `owns()`
needs from it, so silence here is an answer and not a missing lookup.
"""

from __future__ import annotations

import logging
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.entity_lineage import (
    TwoUniverseTickersOneEntityError,
    load_d19_allowlist,
)
from src.data_store.schema import Tables

logger = logging.getLogger(__name__)


class IdentityError(ValueError):
    """Base of every identity failure, so one `except` covers the whole layer."""


class UnknownUniverseTickerError(IdentityError):
    """`universe_entity(T)` for a ticker absent from `sp500_tickers`, or holding no CIK."""


class CikInTwoEntitiesError(IdentityError):
    """One CIK carries two `entity_id`s.

    Structurally impossible given `entity_lineage`'s primary key on `cik`, so this guards the
    BUILDER rather than the table -- the belt-and-braces twin of
    `registrant._check_ciks_unique_across_entries`, which exists for the same reason.
    """


class UniverseEntityDisagreementError(IdentityError):
    """D19: the roster CIK and `symbol_tenure` name different entities for one ticker.

    The roster CIK is Wikipedia-sourced and HAS been wrong -- `XOM` carried a shell CIK that
    returned 0 proxies until 2026-09. This is the free check that would have caught it, so a
    disagreement raises unless a written adjudication in the D19 allow-list explains it.
    """


class AmbiguousSymbolTenureError(IdentityError):
    """A symbol resolves to more than one ENTITY, at `as_of` or over all of history.

    Zipline's `lookup_symbol` contract: never a silent "today", never last-writer-wins. The
    grain matters -- see `Identity.entity_for`.
    """


SymbolVerdict = Literal[
    "exact_dated_tenure",
    "unique_entity_fallback",
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
    "unique_entity_fallback",
    "roster_tenure_proxy",
]


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


#: `load_identity` caches per CONTEXT, not per module. A module-level cache keyed on nothing
#: would leak one test's database into the next, and `load_registrants`' `@cache` is safe only
#: because it is keyed on a config directory. Weak keys so a finished context is collectable.
_CACHE: weakref.WeakKeyDictionary[Context, Identity] = weakref.WeakKeyDictionary()


def normalise_cik(value) -> str:
    """The 10-digit zero-padded spelling `sp500_tickers` and `entity_lineage` both use.

    The bulk zips write `320193`, the roster writes `320193.0` after a float round-trip and
    the register writes `0000320193`. Three spellings of one CIK would be three entities, so
    every entry point normalises before it looks anything up.
    """
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(10) if text.isdigit() else text


def _as_timestamp(value) -> pd.Timestamp | None:
    """`None` for a null, a `Timestamp` for anything else.

    ⚠ Postgres `DATE` columns come back as `datetime.date`, NOT `Timestamp`, and
    `date < Timestamp` raises rather than comparing. Every tenure comparison goes through
    here so the round-trip trap cannot reach the interval test.
    """
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, date | datetime | pd.Timestamp) or not pd.isna(value):
        return pd.Timestamp(value)
    return None


@dataclass(frozen=True)
class Identity:
    """Both identity axes, resolved once per run and then read-only.

    Frozen because the maps are invariants that were VALIDATED at construction: a caller that
    could add a CIK could re-introduce the two-tickers-one-entity collapse the load-time
    raise exists to prevent.
    """

    #: axis A: CIK -> entity_id, for the stored rows only. Absence means singleton.
    entity_by_cik: Mapping[str, str]
    #: universe ticker -> its roster CIK (10-digit).
    roster_cik: Mapping[str, str]
    #: entity_id -> the ONE universe ticker on it. The reverse map `entity_ticker` reads.
    ticker_by_entity: Mapping[str, str]
    #: axis B: symbol -> tuple of (entity_id, valid_from, valid_to, n_filings).
    tenure_by_symbol: Mapping[str, tuple[tuple[str, pd.Timestamp, pd.Timestamp | None, int], ...]]
    #: D19-cleared roster symbol -> dated rows borrowed from filing symbols on its entity.
    roster_proxy_by_symbol: Mapping[str, tuple[tuple[str, pd.Timestamp, pd.Timestamp | None, int], ...]]
    #: Separately traded share classes deliberately absent from the modelling universe.
    redundant_symbols: frozenset[str]

    # ------------------------------------------------------------------ axis A #

    def entity_of(self, cik) -> str:
        """The entity a CIK belongs to. A CIK with no stored row IS its own entity."""
        key = normalise_cik(cik)
        return self.entity_by_cik.get(key, f"E{key}")

    def ciks_for(self, entity_id: str) -> frozenset[str]:
        """Every STORED CIK on an entity; empty for an entity that owns no stored row.

        Empty is not "unknown": a singleton entity has no row by design, and its one CIK is
        recoverable from the id itself (`entity_id_for` is `"E" + min(cik)`).
        """
        return frozenset(c for c, e in self.entity_by_cik.items() if e == entity_id)

    def universe_entity(self, ticker: str) -> str:
        """The entity of a universe ticker, via its roster CIK (D19).

        Raises rather than returning None: every call site here is resolving a row that is
        ABOUT to be kept or quarantined, and a None would make "not in the universe" and
        "roster row is broken" indistinguishable -- the exact confusion `cik_to_ticker` has.
        """
        key = str(ticker).strip().upper()
        if key not in self.roster_cik:
            raise UnknownUniverseTickerError(
                f"identity: {key!r} is not a universe ticker in sp500_tickers (or its roster "
                f"row carries no CIK). {len(self.roster_cik)} ticker(s) are resolvable."
            )
        return self.entity_of(self.roster_cik[key])

    def entity_ticker(self, cik) -> str | None:
        """Today's universe ticker for a CIK, or None when its entity holds none.

        THE IMPLEMENTATION CORE. `owns()` is the contract, but resolution inverts it: take the
        row's own CIK, find its entity, and keep the row iff that entity is a universe
        ticker's. Driven by a derived table rather than by hand entries, which is the whole
        point of axis A -- a future universe ticker is resolved by this same line.
        """
        return self.ticker_by_entity.get(self.entity_of(cik))

    def owns(self, ticker: str, cik, on_date=None) -> bool:
        """Is this CIK's filing about this universe ticker's company?

        ⚠ `on_date` IS ACCEPTED AND DELIBERATELY NOT READ. Forms 3/4/5 are UNION events
        (`registrant.FORM_POLICY`): a predecessor's Form 4 filed after a registrant boundary
        is still a real transaction in this issuer's security, so a date filter here would be
        the named XOM-`SCHEDULE 13G` regression. It is in the signature because it is recorded
        on every quarantine row (a verdict without the date it was taken against is not
        auditable), because Phase 6's CIK-less tables must resolve through `entity_for(symbol,
        date)`, and because a future dated universe drops into `universe_entity` without one
        caller changing.

        The date to pass is `filing_date` (D18) -- the field `symbol_tenure` is derived from,
        so any later boundary comparison aligns by construction. NOT `transaction_date`: Form
        5 is annual and lags by up to a year, and the field is null or repaired on ~0.02% of
        rows.
        """
        return self.entity_of(cik) == self.universe_entity(ticker)

    # ------------------------------------------------------------------ axis B #

    def entity_for(self, symbol: str, as_of=None) -> str | None:
        """The entity holding `symbol` at `as_of`; None when nobody did.

        `valid_from <= d < valid_to`, half-open, identical to `registrant.Segment.covers`, so
        adjacent tenures are disjoint by construction. A null `valid_to` means "no end
        OBSERVED", never "forever" -- but for a membership test at a date those read the same,
        which is why the distinction lives in the table's docstring and not in this branch.

        ⚠ AMBIGUITY IS MEASURED AT ENTITY GRAIN, NOT CIK GRAIN. 993 adjacent tenure pairs
        overlap in time and most of those overlaps are two CIKs of ONE entity filing under
        both registrants through a reorganisation -- a fact about paperwork, not about
        identity. Collapsing to entities first is what keeps this raise rare enough to mean
        something when it fires.
        """
        rows = self.tenure_by_symbol.get(str(symbol).strip().upper())
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
        hits = {entity for entity, start, end, _ in rows if start <= stamp and (end is None or stamp < end)}
        if len(hits) > 1:
            raise AmbiguousSymbolTenureError(
                f"identity: symbol {symbol!r} resolves to {len(hits)} entities at "
                f"{stamp.date()} ({', '.join(sorted(hits))}). Two entities filing under one "
                "symbol on one date is a data condition worth reading, not a tie to break."
            )
        return next(iter(hits), None)

    def dominant_entity(self, symbol: str) -> str | None:
        """The entity that OWNS a symbol today, by weight of filings rather than by recency.

        Separate from `entity_for` on purpose, and the D19 check uses this one. The strict
        resolver above is right for a dated lookup and useless for "whose symbol is this
        now", because a filer's single mistyped `ISSUERTRADINGSYMBOL` opens a one-filing
        tenure that is both CURRENT and WRONG -- `COO` and `SPG` each carry one such typo
        against a thousand real filings, and a latest-observation rule hands them the symbol.
        So: prefer still-open tenures, then take the heaviest.
        """
        rows = self.tenure_by_symbol.get(str(symbol).strip().upper())
        if not rows:
            return None
        openrows = [r for r in rows if r[2] is None]
        return max(openrows or list(rows), key=lambda r: r[3])[0]

    def candidate_symbols(self, universe: frozenset[str]) -> frozenset[str]:
        """Current symbols plus historical symbols owned by the requested universe."""
        requested = frozenset(str(ticker).strip().upper() for ticker in universe)
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
        source_symbol = str(symbol).strip().upper().replace(".", "-")
        stamp = _as_timestamp(as_of)
        requested = frozenset(str(ticker).strip().upper() for ticker in universe)
        rows = self.tenure_by_symbol.get(source_symbol)
        is_roster_proxy = rows is None and source_symbol in self.roster_proxy_by_symbol
        if is_roster_proxy:
            rows = self.roster_proxy_by_symbol[source_symbol]
        if not rows:
            return SymbolResolution(source_symbol, stamp, "unknown_symbol")

        entities = {entity for entity, _, _, _ in rows}
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
        elif len(entities) == 1:
            entity_id = next(iter(entities))
            dated_hits = {entity for entity, start, end, _ in rows if stamp is not None and start <= stamp and (end is None or stamp < end)}
            match_kind = "exact_dated_tenure" if dated_hits else "unique_entity_fallback"
        else:
            if stamp is None:
                return SymbolResolution(source_symbol, stamp, "unknown_gap")
            try:
                entity_id = self.entity_for(source_symbol, stamp)
            except AmbiguousSymbolTenureError:
                return SymbolResolution(source_symbol, stamp, "ambiguous")
            if entity_id is None:
                return SymbolResolution(source_symbol, stamp, "unknown_gap")
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

        # A redundant spelling is a separately traded sibling only while the retained class is
        # independently active for the same entity. Before that boundary it can be the retained
        # security's predecessor spelling (GOOG before GOOGL), which must remain admissible.
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
    work[symbol_col] = work[symbol_col].astype("string").str.strip().str.upper().str.replace(".", "-", regex=False)
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
                f"{row.rows} row(s), {pd.Timestamp(row.first).date()}.."
                f"{pd.Timestamp(row.last).date()}"
            )

    current_reuse = accepted[accepted[symbol_col].isin(universe) & (accepted[symbol_col] != accepted["ticker"])]
    if not current_reuse.empty:
        names = sorted(current_reuse[symbol_col].dropna().astype(str).unique())
        context.log.warning(f"{source_name}: current-looking symbol(s) resolved to another universe " f"entity: {', '.join(names)}")

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
    today=None,
) -> Identity:
    """Validate both tables and return the frozen resolver. Pure -- no DB, no config reads.

    The order of the checks is the order of their blast radius. `TwoUniverseTickersOneEntityError`
    is asserted BEFORE the reverse map is usable, because that is the one failure that
    relabels a company's rows onto another company rather than dropping them.
    """
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
        raise IdentityError("identity: `sp500_tickers` is empty; there is no universe to " "resolve rows against.")

    # --- axis A ------------------------------------------------------------- #
    ciks = lineage["cik"].map(normalise_cik)
    entities = lineage["entity_id"].astype(str)
    per_cik = pd.DataFrame({"cik": ciks, "entity_id": entities}).drop_duplicates()
    clashes = per_cik[per_cik.duplicated("cik", keep=False)]
    if not clashes.empty:
        raise CikInTwoEntitiesError(
            f"identity: {clashes['cik'].nunique()} CIK(s) carry two entity_ids in "
            f"entity_lineage -- {clashes.sort_values('cik').to_dict('records')}. The table's "
            "primary key is `cik`, so this cannot come from the database; it is a builder "
            "bug, and it would make `entity_of` order-dependent."
        )
    entity_by_cik = dict(zip(per_cik["cik"], per_cik["entity_id"], strict=False))

    # --- the universe ------------------------------------------------------- #
    roster_cik: dict[str, str] = {}
    for ticker, cik in zip(roster["ticker"].astype(str), roster["cik"], strict=False):
        if pd.isna(cik) or not str(cik).strip():
            continue  # `universe_entity` raises for it, naming the ticker
        roster_cik[ticker.strip().upper()] = normalise_cik(cik)

    # --- ⚠ the one raise asserted before any map is trusted ------------------ #
    by_entity: dict[str, list[str]] = {}
    for ticker, cik in sorted(roster_cik.items()):
        by_entity.setdefault(entity_by_cik.get(cik, f"E{cik}"), []).append(ticker)
    collisions = {e: t for e, t in by_entity.items() if len(t) > 1}
    if collisions:
        detail = []
        for entity, tickers in sorted(collisions.items()):
            joined = lineage[entities == entity]
            rows = "; ".join(f"{normalise_cik(r.cik)} via {r.source}" for r in joined.itertuples())
            detail.append(f"{entity} holds " + ", ".join(f"{t} (roster CIK {roster_cik[t]})" for t in tickers) + f" -- joined by: {rows}")
        raise TwoUniverseTickersOneEntityError(
            "identity: " + " | ".join(detail) + ". The reverse map is a dict, so one of these "
            "tickers would overwrite the other and EVERY ROW OF THE LOSER would be relabelled "
            "-- the only failure in this design that corrupts rather than drops. A spin-off "
            "into two index members (DowDuPont -> DD/DOW/CTVA) is TWO entities that share a "
            "past: split them with a curated row, never a wider merge."
        )
    ticker_by_entity = {entity: tickers[0] for entity, tickers in by_entity.items()}

    # --- axis B -------------------------------------------------------------- #
    tenure_by_symbol: dict[str, list[tuple[str, pd.Timestamp, pd.Timestamp | None, int]]] = {}
    for symbol, cik, start, end, n in zip(
        tenure["symbol"].astype(str), tenure["issuer_cik"], tenure["valid_from"], tenure["valid_to"], tenure["n_filings"], strict=False
    ):
        stamp = _as_timestamp(start)
        if stamp is None:
            continue  # a tenure with no start cannot answer a dated test
        key = normalise_cik(cik)
        tenure_by_symbol.setdefault(symbol.strip().upper(), []).append((entity_by_cik.get(key, f"E{key}"), stamp, _as_timestamp(end), int(n)))

    # D19 already records the exceptional cases where the roster spelling is absent from, or
    # disagrees with, Form 345. For an absent spelling only, borrow the DATED tenure of every
    # filing symbol observed on the roster entity. Include every entity ever seen under those
    # proxy symbols: FOXA may borrow FOX, but FOX belonged to old 21st Century Fox before the
    # current Fox Corp. The date must settle that boundary; a roster CIK must never rewrite it.
    allowlist = d19_allowlist or {}
    roster_proxy_by_symbol: dict[str, tuple[tuple[str, pd.Timestamp, pd.Timestamp | None, int], ...]] = {}
    for ticker in sorted(set(allowlist) & set(roster_cik)):
        if ticker in tenure_by_symbol:
            continue
        roster_entity = entity_by_cik.get(roster_cik[ticker], f"E{roster_cik[ticker]}")
        proxy_symbols = {symbol for symbol, rows in tenure_by_symbol.items() if any(entity == roster_entity for entity, _, _, _ in rows)}
        proxy_rows = tuple(row for symbol in sorted(proxy_symbols) for row in tenure_by_symbol[symbol])
        if proxy_rows:
            roster_proxy_by_symbol[ticker] = proxy_rows

    identity = Identity(
        entity_by_cik=entity_by_cik,
        roster_cik=roster_cik,
        ticker_by_entity=ticker_by_entity,
        tenure_by_symbol={s: tuple(v) for s, v in tenure_by_symbol.items()},
        roster_proxy_by_symbol=roster_proxy_by_symbol,
        redundant_symbols=frozenset(str(symbol).strip().upper().replace(".", "-") for symbol in (redundant_symbols or frozenset())),
    )

    _check_d19(identity, allowlist, today)
    logger.info(
        "identity: %d lineage CIK(s) over %d entity(ies); %d universe ticker(s); "
        "%d symbol(s) with tenure; %d D19 roster proxy symbol(s); %d redundant symbol(s)",
        len(entity_by_cik),
        len(set(entity_by_cik.values())),
        len(roster_cik),
        len(tenure_by_symbol),
        len(roster_proxy_by_symbol),
        len(identity.redundant_symbols),
    )
    return identity


def _check_d19(identity: Identity, allowlist: Mapping[str, str], today=None) -> None:
    """D19: the roster CIK must name the same entity `symbol_tenure` does.

    Two independent sources for one fact, so a disagreement is information. The allow-list is
    a config with per-ticker PROSE, not a bare set -- an unexplained entry is how a check like
    this rots into a no-op, and the 10 live entries each say in writing why the roster CIK and
    the filings differ.
    """
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
            "identity: D19 cross-check -- %d/%d tickers agree, %d allow-listed " "with evidence",
            len(identity.roster_cik) - len(disagree),
            len(identity.roster_cik),
            len(disagree),
        )


def load_identity(context: Context, config_dir: str | None = None, refresh: bool = False) -> Identity:
    """The resolver for this run, read once per context and then cached on it.

    Cached on the CONTEXT rather than at module level: the tables live in a database, and a
    module-level cache would hand one test's database to the next. `load_registrants` gets to
    use `@cache` only because its key is a config directory on disk.
    """
    cached = None if refresh else _CACHE.get(context)
    if cached is not None:
        return cached
    identity = build_identity(
        lineage=context.store.load(Tables.entity_lineage, project=True),
        tenure=context.store.load(Tables.symbol_tenure, project=True),
        roster=context.store.load(Tables.sp500_tickers),
        d19_allowlist=load_d19_allowlist(config_dir or str(context.config_dir)),
        redundant_symbols=frozenset(context.config.data_extract.redundant_ticks),
    )
    _CACHE[context] = identity
    return identity
