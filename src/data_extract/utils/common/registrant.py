"""
registrant.py (src/data_extract/utils/common/registrant.py)
--------------------------------------------------------------------------------------------
THE SINGLE AUTHORITY ON WHICH CIKs A TICKER'S FILINGS CAN COME FROM.

A ticker's price history follows the ECONOMIC ENTITY; its filings follow the LEGAL
REGISTRANT. When a company reorganises under a new holding company or re-registers in another
jurisdiction, the registrant's CIK changes and the price history does not. `Company(ticker)`
resolves exactly one CIK -- the one EDGAR's ticker table points at *today* -- so everything
the predecessor filed becomes invisible **with no error, no exception and no gap signal**. The
ticker simply arrives with 22 filings where its peers have 62.

WHY `common/` AND NOT `fundamentals/`. This was diagnosed once, for fundamentals, and the
register was filed under `configs/fundamentals/` accordingly. That home is the reason the
other two tiers were wired late and inconsistently: five tier-A fetchers reached across into
`fundamentals` for it, and the three bulk-dataset fetchers (`insider_transactions`,
`notes_*`, `pension_facts`) never learned it existed at all. It is a cross-cutting concern and
it lives at the lowest layer.

WHY SEGMENTS AND NOT TWO CIKs. Chains are real. `PSKY` is CBS -> Viacom -> ViacomCBS ->
Paramount Global -> Paramount Skydance. A two-CIK schema gives it one hop and leaves the rest
truncated, so an entry is an ORDERED, CONTIGUOUS list of segments and a boundary is the seam
between two of them.

⚠ THIS FILE'S CONFIG IS A RISK ZONE. The failure mode is QUIET DATA LOSS: a `valid_to` set a
year early drops the predecessor's last four filings and admits nothing in their place, and
nothing downstream can tell that from a company that genuinely filed nothing. A wrong entry
cannot raise on its own, so the loader raises on every malformed one it *can* detect, and
every segment must carry its evidence.

The one check that CANNOT live here is "the boundary falls inside the predecessor's own
filing window" -- that needs EDGAR, and a config loader must never make a network call on a
nightly path. `tests/data_extract/common/test_registrant_live.py` asserts it instead.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cache
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import edgar
import pandas as pd

from src.data_extract.utils.common.config_paths import resolve_config_dir
from src.data_extract.utils.common.sec_atom import (
    SEC_INSIDER_OWNER_ATOM_PAGE_SIZE,
    atom_filing,
    atom_page_url,
    fetch_atom_entries,
    parse_atom_entry,
)
from src.utils.string import pad_cik, pad_cik_series

if TYPE_CHECKING:  # identity -> entity_lineage -> registrant: annotation-only import breaks the cycle
    from src.data_extract.utils.common.identity import FilingScope, Identity

logger = logging.getLogger(__name__)

#: `_company_or_none` kinds; each is also the wording of its "could not be resolved" warning.
_CIK_KIND = "register CIK"
_ALIAS_KIND = "historical alias"

#: Above this many filings retained AFTER cheap SGML subject-CIK filtering, discovery is
#: considered incomplete. The broad owner-inclusive candidate book is deliberately uncapped:
#: BLK can appear as filer on tens of thousands of unrelated schedules while having only a small
#: issuer-side history of its own.
#:
#: ⚠ The limit is a hard incomplete outcome, never a successful empty result. Full filing
#: objects are created only for retained issuer matches (plus headers whose subject is genuinely
#: unavailable), so the former memory failure is avoided without losing late issuer filings.
SCHEDULE_SUBJECT_CAP = 2_000
# SEC's legacy company-browse Atom endpoint returned a stable HTTP 503 at offset 5,100 for
# BLK's reporting-person book on 2026-09-24. Stop before that deep-pagination boundary and
# bisect the requested date range; each child is independently exhausted, so no filing is
# inferred away and ordinary issuer CIKs still use one request.
SCHEDULE_ATOM_SAFE_OFFSET = 5_000

#: `configs/sec/registrant_cutover.json`. Declared here rather than in `constants.py`, whose
#: rule is "a literal two or more non-test `src/` modules share": this module is the only
#: reader, and every other module reaches the register through `load_registrants`.
REGISTRANT_CONFIG_SUBDIR = "sec"
REGISTRANT_CONFIG_FILENAME = "registrant_cutover.json"

#: `reorganisation` = a new legal parent (Apache Corp -> APA Corp holding company);
#: `domestication` = the same business re-registered in another jurisdiction (Eaton's 2012
#: move to Ireland). Both change the CIK.
CUTOVER_KINDS: frozenset[str] = frozenset({"reorganisation", "domestication"})

#: The `kind` a reader will reach for and must NOT use, rejected by name with its reason
#: attached. A rename keeps the CIK -- CVS Caremark -> CVS Health, Facebook -> Meta -- so an
#: entry would walk one CIK twice and duplicate every filing. Naming it explicitly is what
#: stops the next person adding one; Sharadar records a name change whether or not the CIK
#: moved, so the shell-name evidence that motivates most entries also fits a pure rename.
RENAME_KIND = "rename"


@dataclass(frozen=True)
class Segment:
    """One registrant's tenure over a ticker: `[valid_from, valid_to)`.

    `valid_from is None` on the oldest segment and `valid_to is None` on the newest, so the
    chain is open at both ends and every date in history lands in exactly one segment.
    """

    cik: str
    valid_from: pd.Timestamp | None
    valid_to: pd.Timestamp | None
    evidence: str

    def covers(self, date) -> bool:
        """Strictly-before / on-or-after, the convention the dated split has always used, so
        two adjacent segments are disjoint by construction rather than by de-duplication."""
        stamp = pd.Timestamp(date)
        if self.valid_from is not None and stamp < self.valid_from:
            return False
        return not (self.valid_to is not None and stamp >= self.valid_to)


@dataclass(frozen=True)
class Registrant:
    """One ticker's registrant chain, oldest segment first."""

    ticker: str
    kind: str
    segments: tuple[Segment, ...]

    def segment_for(self, date) -> Segment:
        """The registrant that owned `date`. Total by construction -- the chain is contiguous
        and open at both ends -- so a miss is a loader bug, not a data condition."""
        for segment in self.segments:
            if segment.covers(date):
                return segment
        raise ValueError(f"registrant[{self.ticker}]: no segment covers {date} -- the chain is not contiguous, which the loader should have caught")

    def all_ciks(self) -> tuple[str, ...]:
        """Every CIK in the chain, oldest first. This is the UNION set for event forms."""
        return tuple(s.cik for s in self.segments)

    @property
    def boundaries(self) -> tuple[pd.Timestamp, ...]:
        """The seam dates, oldest first. `len(segments) - 1` of them."""
        return tuple(s.valid_from for s in self.segments[1:] if s.valid_from is not None)


def load_registrants(config_dir: str | None = None) -> dict[str, Registrant]:
    """The registrant register, keyed by ticker, cached per config DIRECTORY rather than per
    spelling of it -- see `resolve_config_dir`."""
    return _registrants_at(resolve_config_dir(config_dir))


@cache
def _registrants_at(config_dir: str) -> dict[str, Registrant]:
    """`load_registrants`, keyed on a resolved absolute path. `{}` when the file is absent.

    Validation is strict and happens at LOAD time, because a typo here silently deletes a
    decade of history rather than raising -- see the module docstring.
    """
    path = Path(config_dir) / REGISTRANT_CONFIG_SUBDIR / REGISTRANT_CONFIG_FILENAME
    if not path.exists():
        return {}
    blob = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, Registrant] = {}
    for ticker, entry in blob.items():
        if ticker.startswith("_"):
            continue
        out[ticker] = _parse_entry(ticker, entry)
    _check_ciks_unique_across_entries(out)
    return out


def _parse_entry(ticker: str, entry: dict[str, Any]) -> Registrant:
    """One validated `Registrant`, or a `ValueError` naming the ticker and the rule broken."""
    kind = str(entry.get("kind", ""))
    if kind == RENAME_KIND:
        raise ValueError(
            f"registrant[{ticker}]: kind='{RENAME_KIND}' is not a cutover. A rename keeps the "
            "CIK (CVS Caremark -> CVS Health, Facebook -> Meta), so an entry here would walk "
            "one CIK twice and duplicate every filing. Delete it."
        )
    if kind not in CUTOVER_KINDS:
        raise ValueError(f"registrant[{ticker}]: kind={kind!r} not in {sorted(CUTOVER_KINDS)}")

    raw = entry.get("segments")
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError(f"registrant[{ticker}]: `segments` must be a list of at least 2 entries -- one segment is not a boundary.")

    segments: list[Segment] = []
    for i, seg in enumerate(raw):
        if not str(seg.get("evidence", "")).strip():
            raise ValueError(
                f"registrant[{ticker}] segment {i}: empty `evidence`. An undocumented cutover "
                "is a guess that deletes history, which is exactly what this register replaces."
            )
        first, last = i == 0, i == len(raw) - 1
        if first and "valid_from" in seg:
            raise ValueError(f"registrant[{ticker}] segment 0: the oldest segment must omit `valid_from` -- the chain is open at the old end.")
        if last and "valid_to" in seg:
            raise ValueError(f"registrant[{ticker}] segment {i}: the newest segment must omit `valid_to` -- the chain is open at the new end.")
        if not first and "valid_from" not in seg:
            raise ValueError(f"registrant[{ticker}] segment {i}: missing `valid_from`.")
        if not last and "valid_to" not in seg:
            raise ValueError(f"registrant[{ticker}] segment {i}: missing `valid_to`.")
        segments.append(
            Segment(
                cik=pad_cik(seg["cik"]),
                valid_from=_stamp(ticker, i, seg.get("valid_from")),
                valid_to=_stamp(ticker, i, seg.get("valid_to")),
                evidence=str(seg["evidence"]),
            )
        )

    for i in range(len(segments) - 1):
        if segments[i].valid_to != segments[i + 1].valid_from:
            raise ValueError(
                f"registrant[{ticker}]: segment {i} ends {segments[i].valid_to} but segment "
                f"{i + 1} starts {segments[i + 1].valid_from}. Segments must be CONTIGUOUS -- "
                "a gap loses every filing inside it and an overlap double-counts them, and "
                "neither raises anywhere downstream."
            )

    ciks = [s.cik for s in segments]
    if len(set(ciks)) != len(ciks):
        raise ValueError(
            f"registrant[{ticker}]: repeated CIK in {ciks}. Two equal CIKs mean a RENAME, not "
            "a cutover, and would walk one CIK twice and duplicate every filing."
        )
    return Registrant(ticker=ticker, kind=kind, segments=tuple(segments))


def _check_ciks_unique_across_entries(registrants: dict[str, Registrant]) -> None:
    """No CIK may appear in two tickers' chains.

    Two tickers claiming one CIK is a register bug that produces a duplicate ticker in the
    CIK->ticker map tier C resolves through, which would route one company's bulk rows to two
    names. It cannot be caught inside a single entry, so it is checked across all of them.
    """
    owner: dict[str, str] = {}
    for ticker, reg in sorted(registrants.items()):
        for cik in reg.all_ciks():
            if cik in owner:
                raise ValueError(
                    f"registrant: CIK {cik} is claimed by both {owner[cik]} and {ticker}. One "
                    "CIK belongs to one ticker; two claims would route its bulk-dataset rows "
                    "to both names."
                )
            owner[cik] = ticker


class Combine(StrEnum):
    """How a form's filings combine across a registrant boundary."""

    UNION = "union"  # events: additive, because an event happened whoever indexed it
    SPLIT = "split"  # consolidating: one registrant owns each date, disjoint by construction


#: Every form this repo fetches, and how it combines across a registrant boundary.
#:
#: ⚠ FAIL CLOSED. A form absent here RAISES. Defaulting is exactly how this defect class stayed
#: invisible for a year -- `Company(ticker)` silently returned one registrant and nothing said
#: so -- and a new form family silently taking the wrong rule is the same failure again.
#:
#: THE TWO RULES LOOK CONTRADICTORY AND ARE BOTH RIGHT. One measurement settles it. Against
#: APA's 2021-03-01 boundary the predecessor Apache Corp (CIK 6769) filed:
#:
#:     4  (insider events)     4,291 filings  2003-06-17 .. 2024-03-20    4 after the boundary
#:     10-K / 10-Q (consol.)     140 filings  1994-03-21 .. 2024-11-07   15 after the boundary
#:     DEF 14A                    28 filings  1994-03-29 .. 2020-04-03    0 after the boundary
#:
#: A UNION of the event stream gains 4,287 Form 4s and risks 4 ambiguous ones. A UNION of the
#: consolidating stream would blend 15 subsidiary 10-K/10-Qs into the parent's accounts -- a
#: fuller-looking history that is quietly wrong, which is the dangerous direction. And a SPLIT
#: of the event stream is a regression on a named example: XOM's SCHEDULE 13G of 2026-08-07 is
#: filed under the PREDECESSOR, after the boundary, so a split would discard a filing already
#: in the database.
#:
#: ⚠ THE PROXY IS SPLIT, NOT UNION, AND THAT IS DELIBERATE. A proxy is a consolidating annual
#: disclosure of one registrant's board and pay; two registrants' proxies for the same year
#: would give the governance panel two boards. APA measured 0 predecessor proxies after its
#: boundary, so the split costs nothing observed and protects the case that would corrupt the
#: grain.
#:
#: ⚠ `FILING_TEXT_FORMS` IS 10-K/10-Q -- THE SAME FORMS, SPLIT. Item 1A/7 text is a
#: registrant's own narrative, so a subsidiary's MD&A must not land in the parent's series.
FORM_POLICY: dict[str, Combine] = {
    # events -- 8-K
    "8-K": Combine.UNION,
    "8-K/A": Combine.UNION,
    "8-K12B": Combine.UNION,
    # events -- beneficial ownership. Both spellings of each: EDGAR renamed the form type at
    # the 2024-12-17 structured-XML mandate and `get_filings(form=...)` matches EXACTLY.
    "SC 13D": Combine.UNION,
    "SC 13D/A": Combine.UNION,
    "SCHEDULE 13D": Combine.UNION,
    "SCHEDULE 13D/A": Combine.UNION,
    "SC 13G": Combine.UNION,
    "SC 13G/A": Combine.UNION,
    "SCHEDULE 13G": Combine.UNION,
    "SCHEDULE 13G/A": Combine.UNION,
    # events -- insider transactions
    "3": Combine.UNION,
    "4": Combine.UNION,
    "5": Combine.UNION,
    "3/A": Combine.UNION,
    "4/A": Combine.UNION,
    "5/A": Combine.UNION,
    # consolidating -- periodic reports and the narrative carved out of them
    "10-K": Combine.SPLIT,
    "10-K/A": Combine.SPLIT,
    "10-K405": Combine.SPLIT,
    "10-Q": Combine.SPLIT,
    "10-Q/A": Combine.SPLIT,
    # consolidating -- the proxy family
    "DEF 14A": Combine.SPLIT,
    "DEF 14C": Combine.SPLIT,
    "DEFC14A": Combine.SPLIT,
}


def combine_for(forms: Sequence[str]) -> Combine:
    """The one policy governing `forms`, or a `ValueError` naming what is wrong.

    A MIXED list raises rather than picking a winner: the caller is asking one question with
    two right answers, which is a bug at the call site and not something this function can
    resolve on its behalf.
    """
    unknown = [f for f in forms if f not in FORM_POLICY]
    if unknown:
        raise ValueError(
            f"no registrant-combination policy declared for form(s) {unknown}. Add them to "
            "`FORM_POLICY` as UNION (an event: additive across a boundary) or SPLIT (a "
            "consolidating disclosure: one registrant owns each date). This RAISES rather "
            "than defaulting because a silent default is how the registrant-cutover defect "
            "stayed invisible for a year."
        )
    policies = {FORM_POLICY[f] for f in forms}
    if len(policies) > 1:
        raise ValueError(
            f"forms {list(forms)} mix {sorted(p.value for p in policies)} policies. One call "
            "cannot both union and split; split the call at the site that knows which "
            "question it is asking."
        )
    if not policies:
        raise ValueError("no forms given, so no combination policy applies")
    return policies.pop()


class AmbiguousRegistrantScopeError(RuntimeError):
    """Identity found a CIK transition that has no complete dated registrant chain."""


def identity_scope_fingerprint(scope: FilingScope, entry: Registrant | None) -> str:
    """Stable digest of one ticker's discovered scope and its authoritative register segments."""
    segments = (
        []
        if entry is None
        else [
            {
                "cik": segment.cik,
                "valid_from": None if segment.valid_from is None else pd.Timestamp(segment.valid_from).date().isoformat(),
                "valid_to": None if segment.valid_to is None else pd.Timestamp(segment.valid_to).date().isoformat(),
            }
            for segment in entry.segments
        ]
    )
    payload = {
        "canonical_ticker": scope.ticker,
        "roster_cik": scope.roster_cik,
        "candidate_ciks": list(scope.ciks),
        "candidate_symbols": list(scope.symbols),
        "authoritative_segments": segments,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _split_scope_check(scope: FilingScope, entry: Registrant | None, policy: Combine) -> tuple[str, ...]:
    """The identity CIKs outside the roster CIK and the register chain, sorted.

    Raises for a SPLIT form when identity found more than one CIK and the register does not
    curate all of them: consolidating filings need an explicit dated chain.
    """
    discovered = set(scope.ciks)
    curated = set(entry.all_ciks()) if entry is not None else set()
    missing_from_chain = discovered - curated
    if policy is Combine.SPLIT and len(discovered) > 1 and missing_from_chain:
        raise AmbiguousRegistrantScopeError(
            f"{scope.ticker}: identity discovered registrant CIK(s) "
            f"{', '.join(sorted(discovered))}, including uncurated "
            f"{', '.join(sorted(missing_from_chain))}; add a complete explicit dated "
            "registrant chain before fetching consolidating filings"
        )
    return tuple(sorted(discovered - {scope.roster_cik} - curated))


@dataclass
class _FilingWindow:
    """The `since` / `done_accessions` filter shared by every walk; counts skipped stored accessions."""

    since: pd.Timestamp | None
    done_accessions: frozenset[str]
    stats: dict[str, int] | None
    skipped_existing: set[str] = field(default_factory=set)

    def filed(self, filing: Any) -> pd.Timestamp | None:
        """The filing date when `filing` is kept, else None."""
        if filing.accession_number in self.done_accessions:
            if self.stats is not None:
                self.skipped_existing.add(filing.accession_number)
                self.stats["skipped_existing"] = len(self.skipped_existing)
            return None
        filed = pd.Timestamp(filing.filing_date)
        return None if self.since is not None and filed < self.since else filed


def resolve_registrant_filings(
    ticker: str,
    forms: Sequence[str],
    *,
    since: pd.Timestamp | None,
    done_accessions: frozenset[str],
    registrants: dict[str, Registrant],
    identity: Identity | None = None,
    stats: dict[str, int] | None = None,
) -> list:
    """Every filing of `forms` for `ticker`, across its registrant chain, oldest first.

    Combination is per-form, from `FORM_POLICY`; a mixed `forms` list raises. With no register
    entry and no identity-discovered alias or CIK this is `Company(ticker).get_filings(...)`.
    Identity adds same-CIK historical symbols; a discovered CIK is additive for UNION forms and
    fails closed for SPLIT forms until the register supplies a dated chain.

    UNION walks `Company(ticker)` first, then every segment / identity CIK, deduping on
    accession (first writer wins). SPLIT gives each segment only the filings inside
    `[valid_from, valid_to)`; a duplicate accession there is logged as a register error.
    `since` and `done_accessions` are applied before the sort.
    """
    policy = combine_for(forms)
    entry = registrants.get(ticker)
    forms = list(forms)
    if stats is not None:
        stats.setdefault("skipped_existing", 0)
    aliases: tuple[str, ...] = ()
    identity_ciks: tuple[str, ...] = ()
    if identity is not None:
        scope = identity.filing_scope(ticker)
        aliases, identity_ciks = scope.aliases, _split_scope_check(scope, entry, policy)
    window = _FilingWindow(since=since, done_accessions=done_accessions, stats=stats)
    if entry is not None and policy is Combine.SPLIT:
        return _split_walk(ticker, entry, forms, window.filed)
    sources = _filing_sources(ticker, entry, policy, aliases, identity_ciks)
    filings, contributions = _union_walk(sources, forms, window.filed)
    _log_contributions(ticker, entry, forms, contributions)
    return filings


def _filing_sources(
    ticker: str,
    entry: Registrant | None,
    policy: Combine,
    aliases: tuple[str, ...],
    identity_ciks: tuple[str, ...],
) -> list[tuple[str, Any | None]]:
    """`(label, Company)` pairs in provenance order, the ticker-resolved registrant first.

    Without a register entry the ticker is labelled by its own symbol and followed by its aliases
    and (UNION only) identity CIKs; with one it is labelled "ticker" and followed by the chain
    CIKs and identity CIKs. Every Company is built before any listing is read.
    """
    if entry is None:
        additive_ciks = identity_ciks if policy is Combine.UNION else ()
        return (
            [(ticker, edgar.Company(ticker))]
            + [(alias, _company_or_none(alias, ticker, _ALIAS_KIND)) for alias in aliases]
            + [(cik, _company_or_none(cik, ticker, _CIK_KIND)) for cik in additive_ciks]
        )
    union_ciks = tuple(dict.fromkeys((*entry.all_ciks(), *identity_ciks)))
    return [("ticker", edgar.Company(ticker))] + [(cik, _company_or_none(cik, ticker, _CIK_KIND)) for cik in union_ciks]


def _union_walk(
    sources: list[tuple[str, Any | None]],
    forms: list[str],
    keep: Callable[[Any], pd.Timestamp | None],
) -> tuple[list, dict[str, int]]:
    """Kept filings across `sources` sorted by filing date, first writer per accession, and per-label counts."""
    by_accession: dict[str, tuple[pd.Timestamp, object]] = {}
    contributions: dict[str, int] = {}
    for label, company in sources:
        if company is None:
            continue
        for filing in company.get_filings(form=forms):
            if filing.accession_number in by_accession:
                continue
            filed = keep(filing)
            if filed is None:
                continue
            by_accession[filing.accession_number] = (filed, filing)
            contributions[label] = contributions.get(label, 0) + 1
    return [filing for _, filing in sorted(by_accession.values(), key=itemgetter(0))], contributions


def _split_walk(ticker: str, entry: Registrant, forms: list[str], keep: Callable[[Any], pd.Timestamp | None]) -> list:
    """Each segment's kept filings inside its own dates, sorted; a duplicate accession is warned and dropped."""
    dated: list[tuple[pd.Timestamp, object]] = []
    seen: dict[str, str] = {}
    for segment in entry.segments:
        company = _company_or_none(segment.cik, ticker, _CIK_KIND)
        for filing in [] if company is None else company.get_filings(form=forms):
            filed = keep(filing)
            if filed is None or not segment.covers(filed):
                continue
            if filing.accession_number in seen:
                logger.warning(
                    "%s: accession %s kept by BOTH segment %s and %s -- the dated split makes that impossible, so the register's boundary is wrong",
                    ticker,
                    filing.accession_number,
                    seen[filing.accession_number],
                    segment.cik,
                )
                continue
            seen[filing.accession_number] = segment.cik
            dated.append((filed, filing))
    return [filing for _, filing in sorted(dated, key=itemgetter(0))]


def _log_contributions(ticker: str, entry: Registrant | None, forms: list[str], contributions: dict[str, int]) -> None:
    """Log when a source other than the ticker-resolved registrant contributed a filing."""
    if entry is None:
        if any(label != ticker for label in contributions):
            logger.info(
                "%s: identity scope added filings (%s)",
                ticker,
                ", ".join(f"{n} from {label}" for label, n in contributions.items()),
            )
        return
    if any(label != "ticker" for label in contributions):
        logger.info(
            "%s: %s across the %s boundary (%s)",
            ticker,
            ", ".join(f"{n} from {k}" for k, n in contributions.items()),
            " -> ".join(entry.all_ciks()),
            ",".join(forms),
        )


def issuer_ciks(
    ticker: str,
    roster_cik: str,
    registrants: dict[str, Registrant],
    identity: Identity | None = None,
) -> frozenset[str]:
    """Every CIK that legitimately identifies THIS ticker as the SUBJECT of a schedule.

    The roster CIK, every register segment CIK and every identity-lineage CIK on the ticker's
    entity: the 13D/13G issuer guard must accept a pre-boundary schedule filed about a
    predecessor, which carries the predecessor's issuer CIK.
    """
    ciks = {pad_cik(roster_cik)} if roster_cik else set()
    entry = registrants.get(ticker)
    if entry is not None:
        ciks.update(entry.all_ciks())
    if identity is not None:
        ciks.update(identity.ciks_for(identity.universe_entity(ticker)))
    return frozenset(c for c in ciks if c)


class ScheduleDiscoveryIncompleteError(RuntimeError):
    """A subject-first schedule search could not prove its result complete."""


def header_subject_ciks(filing: object) -> frozenset[str]:
    """Read schedule subject CIKs from the SGML header, before ``filing.obj()``."""
    header = getattr(filing, "header", None)
    companies = getattr(header, "subject_companies", ()) or ()
    raw_ciks = (getattr(company, "cik", None) or getattr(getattr(company, "company_information", None), "cik", "") for company in companies)
    return frozenset(normalized for cik in raw_ciks if (normalized := pad_cik(cik)))


def filter_schedule_subject_filings(
    candidates: Sequence[object],
    subject_ciks: frozenset[str],
    *,
    cap: int = SCHEDULE_SUBJECT_CAP,
) -> tuple[list[object], dict[str, int]]:
    """Keep issuer-side schedules cheaply; unknown headers pass to the full issuer guard."""
    kept: list[object] = []
    stats = {"candidates": len(candidates), "subject_matches": 0, "unknown_headers": 0}
    for filing in candidates:
        try:
            subjects = header_subject_ciks(filing)
        except Exception:  # noqa: BLE001 -- the full object guard is the safe fallback
            subjects = frozenset()
        if subjects:
            if subjects.isdisjoint(subject_ciks):
                continue
            stats["subject_matches"] += 1
        else:
            stats["unknown_headers"] += 1
        kept.append(filing)
        if len(kept) > cap:
            raise ScheduleDiscoveryIncompleteError(
                f"subject-first schedule discovery retained more than {cap} filing(s) after "
                "header filtering; aborting this ticker as incomplete instead of publishing "
                "a partial result"
            )
    return kept, stats


@dataclass(frozen=True)
class _ScheduleQuery:
    """What one subject-first schedule search keeps from the Atom feed."""

    ticker: str
    target_forms: frozenset[str]
    done_accessions: frozenset[str]


@dataclass
class WindowResult:
    """One exhausted date window: candidate filings by accession, pages read, owner rows excluded."""

    candidates: dict[str, object] = field(default_factory=dict)
    pages: int = 0
    owner_rows_excluded: int = 0

    def absorb(self, other: WindowResult) -> None:
        """Add `other`'s pages and owner rows; its candidates overwrite same-accession entries."""
        self.candidates.update(other.candidates)
        self.pages += other.pages
        self.owner_rows_excluded += other.owner_rows_excluded


def _is_multi_year(start: pd.Timestamp, end: pd.Timestamp) -> bool:
    return int((end - start).days) + 1 > 366


def _window_children(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Inclusive child windows covering `[start, end]`: one-year chunks when the span exceeds 366 days, else two halves."""
    if not _is_multi_year(start, end):
        older_end = (start + (end - start) / 2).normalize()
        return [(start, older_end), (older_end + pd.Timedelta(days=1), end)]
    children: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + pd.DateOffset(years=1) - pd.Timedelta(days=1), end)
        children.append((cursor, chunk_end))
        cursor = chunk_end + pd.Timedelta(days=1)
    return children


def _collect_window(query: _ScheduleQuery, cik: str, family: str, window_start: pd.Timestamp, window_end: pd.Timestamp) -> WindowResult:
    """Exhaust one inclusive date window, splitting it into child windows before unsafe deep pagination.

    Pages read before a split count toward `pages` and `owner_rows_excluded`; their candidates
    are replaced by the children's.
    """
    result = WindowResult()
    start = 0
    while True:
        if start >= SCHEDULE_ATOM_SAFE_OFFSET:
            return _split_window(query, cik, family, window_start, window_end, start, result)
        url = atom_page_url(cik, family, window_start, window_end, start)
        try:
            entries = fetch_atom_entries(url, f"{query.ticker} {family} offset {start}", retry=True)
        except Exception as exc:  # noqa: BLE001 -- completeness is the contract
            raise ScheduleDiscoveryIncompleteError(
                f"{query.ticker}: schedule search failed for subject CIK {cik}, form {family}, offset {start}: {exc!r}"
            ) from exc
        result.pages += 1
        if not entries:
            break
        for raw in entries:
            entry = parse_atom_entry(raw)
            if (
                entry is None
                or entry.form not in query.target_forms
                or entry.accession is None
                or entry.accession in query.done_accessions
                or entry.accession in result.candidates
                or entry.filing_date > window_end
                or entry.filing_date < window_start
            ):
                continue
            # SEC includes a file number only when the queried CIK is the SUBJECT issuer;
            # reporting-person rows omit it. The SGML subject-CIK guard remains the final authority.
            if entry.file_number is None:
                result.owner_rows_excluded += 1
                continue
            result.candidates[entry.accession] = atom_filing(entry, cik=cik, company=query.ticker)
        if len(entries) < SEC_INSIDER_OWNER_ATOM_PAGE_SIZE:
            break
        start += SEC_INSIDER_OWNER_ATOM_PAGE_SIZE
    return result


def _split_window(
    query: _ScheduleQuery,
    cik: str,
    family: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    offset: int,
    read: WindowResult,
) -> WindowResult:
    """Replace a window that reached the safe offset by its exhausted children; a single day raises."""
    if window_start >= window_end:
        raise ScheduleDiscoveryIncompleteError(
            f"{query.ticker}: schedule search reached the safe pagination limit "
            f"inside the unsplittable date {window_start.date()} for subject CIK "
            f"{cik}, form {family}"
        )
    logger.info(
        "%s: schedule search reached offset %d for subject CIK %s, form %s; partitioning %s..%s into complete one-year windows"
        if _is_multi_year(window_start, window_end)
        else "%s: schedule search reached offset %d for subject CIK %s, form %s; bisecting %s..%s into complete date windows",
        query.ticker,
        offset,
        cik,
        family,
        window_start.date(),
        window_end.date(),
    )
    merged = WindowResult(pages=read.pages, owner_rows_excluded=read.owner_rows_excluded)
    for child_start, child_end in _window_children(window_start, window_end):
        merged.absorb(_collect_window(query, cik, family, child_start, child_end))
    return merged


def resolve_schedule_subject_filings(
    ticker: str,
    subject_ciks: frozenset[str],
    forms: Sequence[str],
    *,
    since: pd.Timestamp | None,
    done_accessions: frozenset[str],
    through: pd.Timestamp | None = None,
) -> list[object]:
    """Discover schedules by issuer CIK through SEC's owner-inclusive Atom search, then filter SGML headers.

    Every page must load and parse; otherwise this raises so the run manifest cannot advance.
    """
    end_date = pd.Timestamp(through or pd.Timestamp.today()).normalize()
    if since is None:
        raise ScheduleDiscoveryIncompleteError(
            f"{ticker}: subject-first schedule discovery requires a finite start date so deep result sets can be split without truncation"
        )
    start_date = pd.Timestamp(since).normalize()
    query = _ScheduleQuery(ticker=ticker, target_forms=frozenset(forms), done_accessions=done_accessions)
    total = WindowResult()
    for cik in sorted(subject_ciks):
        for family in sorted({form.removesuffix("/A") for form in forms}):
            total.absorb(_collect_window(query, cik, family, start_date, end_date))
    filtered, stats = filter_schedule_subject_filings(list(total.candidates.values()), subject_ciks)
    filtered.sort(key=lambda filing: pd.Timestamp(cast(Any, filing).filing_date))
    logger.info(
        "%s: subject-first schedules -- %d page(s), %d owner-side row(s) excluded from Atom metadata, "
        "%d candidate(s), %d subject match(es), %d unknown header(s), %d retained for full parsing",
        ticker,
        total.pages,
        total.owner_rows_excluded,
        stats["candidates"],
        stats["subject_matches"],
        stats["unknown_headers"],
        len(filtered),
    )
    return filtered


def drop_rows_outside_segment(df: pd.DataFrame, *, cik_col: str, ticker_col: str, filed_col: str, registrants: dict[str, Registrant]) -> pd.DataFrame:
    """Drop bulk-dataset rows whose `filed` date lies outside the segment that CIK owns.

    For CONSOLIDATING bulk tables: a predecessor CIK resolves to the ticker through
    `cik_to_ticker`, so a row is kept only when the segment covering its `filed` date is the
    segment whose CIK filed it. Tickers with no register entry are untouched. Vectorised.
    """
    if df.empty or not registrants:
        return df
    covered = df[ticker_col].isin(registrants)
    if not covered.any():
        return df

    filed = pd.to_datetime(df[filed_col], errors="coerce")
    cik = pad_cik_series(df[cik_col])
    owner = pd.Series(pd.NA, index=df.index, dtype="object")
    for ticker, reg in registrants.items():
        rows = covered & df[ticker_col].eq(ticker)
        if not rows.any():
            continue
        for segment in reg.segments:
            in_segment = (
                rows
                & filed.notna()
                & ((filed >= segment.valid_from) if segment.valid_from is not None else True)
                & ((filed < segment.valid_to) if segment.valid_to is not None else True)
            )
            owner[in_segment] = segment.cik

    keep = ~covered | (owner.notna() & (owner == cik))
    dropped = int((~keep).sum())
    if dropped:
        logger.info(
            "registrant split: dropped %d bulk row(s) filed outside their segment (%s)",
            dropped,
            ", ".join(sorted(set(df.loc[~keep, ticker_col].astype(str)))),
        )
    return df[keep]


def _company_or_none(key: str, ticker: str, kind: str) -> Any | None:
    """`Company` for a register/identity CIK or a historical alias; an unresolvable one is warned and skipped."""
    try:
        return edgar.Company(int(key) if kind == _CIK_KIND else key)
    except Exception:  # noqa: BLE001 -- a dead CIK or stale alias, not a bug
        logger.warning("%s: %s %s could not be resolved", ticker, kind, key)
        return None


def _stamp(ticker: str, i: int, value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError(f"registrant[{ticker}] segment {i}: unparseable date {value!r}")
    return stamp
