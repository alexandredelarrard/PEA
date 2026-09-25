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

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from xml.etree import ElementTree

import pandas as pd

from src.constants.constants import SEC_INSIDER_OWNER_ATOM_PAGE_SIZE, SEC_INSIDER_OWNER_ATOM_URL
from src.data_extract.utils.common.config_paths import resolve_config_dir
from src.data_extract.utils.common.rate_limit import call_with_retries
from src.utils.string import pad_cik

logger = logging.getLogger(__name__)

_ATOM_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}

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
        raise ValueError(
            f"registrant[{self.ticker}]: no segment covers {date} " "-- the chain is not contiguous, which the loader should have caught"
        )

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
        raise ValueError(f"registrant[{ticker}]: `segments` must be a list of at least 2 " "entries -- one segment is not a boundary.")

    segments: list[Segment] = []
    for i, seg in enumerate(raw):
        if not str(seg.get("evidence", "")).strip():
            raise ValueError(
                f"registrant[{ticker}] segment {i}: empty `evidence`. An undocumented cutover "
                "is a guess that deletes history, which is exactly what this register replaces."
            )
        first, last = i == 0, i == len(raw) - 1
        if first and "valid_from" in seg:
            raise ValueError(f"registrant[{ticker}] segment 0: the oldest segment must omit " "`valid_from` -- the chain is open at the old end.")
        if last and "valid_to" in seg:
            raise ValueError(f"registrant[{ticker}] segment {i}: the newest segment must omit " "`valid_to` -- the chain is open at the new end.")
        if not first and "valid_from" not in seg:
            raise ValueError(f"registrant[{ticker}] segment {i}: missing `valid_from`.")
        if not last and "valid_to" not in seg:
            raise ValueError(f"registrant[{ticker}] segment {i}: missing `valid_to`.")
        segments.append(
            Segment(
                cik=_normalise_cik(seg["cik"]),
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


def resolve_registrant_filings(
    ticker: str,
    forms: Sequence[str],
    *,
    since: pd.Timestamp | None,
    done_accessions: frozenset[str],
    registrants: dict[str, Registrant] | None = None,
) -> list:
    """Every filing of `forms` for `ticker`, across its registrant chain, oldest first.

    THE SINGLE PLACE THAT DECIDES WHICH CIKs A TICKER'S FILINGS COME FROM. Combination is
    per-form, from `FORM_POLICY`; a mixed `forms` list raises.

    ⚠ NO REGISTER ENTRY -> EXACTLY TODAY'S BEHAVIOUR. That path must stay byte-identical to
    `Company(ticker).get_filings(...)`, because ~449 of the 491 tickers take it and a change
    there would move every one of them while the register moved none.

    UNION walks `Company(ticker)` PLUS every segment CIK and dedups on accession, with the
    ticker-resolved registrant FIRST so first-writer-wins preserves the provenance of every
    filing the old implementation already returned. Register CIKs are ADDED, never
    substituted, so a stale or wrong entry can never LOSE a filing.

    SPLIT gives each segment only the filings inside `[valid_from, valid_to)`, disjoint by
    construction. A duplicate accession is therefore impossible; if one appears the REGISTER
    is wrong, so it is logged rather than silently deduped.

    `since` and `done_accessions` are applied BEFORE the sort, so a routine incremental run
    orders a handful of new filings rather than a ticker's full multi-decade history.
    """
    from edgar import Company

    policy = combine_for(forms)
    registrants = load_registrants() if registrants is None else registrants
    entry = registrants.get(ticker)
    forms = list(forms)

    def _keep(f) -> pd.Timestamp | None:
        if f.accession_number in done_accessions:
            return None
        filed = pd.Timestamp(f.filing_date)
        return None if since is not None and filed < since else filed

    if entry is None:
        dated = [(d, f) for f in Company(ticker).get_filings(form=forms) if (d := _keep(f)) is not None]
        dated.sort(key=lambda pair: pair[0])
        return [f for _, f in dated]

    if policy is Combine.SPLIT:
        dated: list[tuple[pd.Timestamp, object]] = []
        seen: dict[str, str] = {}
        for segment in entry.segments:
            for f in _filings(Company, segment.cik, forms):
                filed = _keep(f)
                if filed is None or not segment.covers(filed):
                    continue
                if f.accession_number in seen:
                    logger.warning(
                        "%s: accession %s kept by BOTH segment %s and %s -- the dated split "
                        "makes that impossible, so the register's boundary is wrong",
                        ticker,
                        f.accession_number,
                        seen[f.accession_number],
                        segment.cik,
                    )
                    continue
                seen[f.accession_number] = segment.cik
                dated.append((filed, f))
        dated.sort(key=lambda pair: pair[0])
        return [f for _, f in dated]

    # UNION. The ticker-resolved registrant goes first so first-writer-wins keeps the
    # provenance of everything the pre-register implementation already returned.
    by_accession: dict[str, tuple[pd.Timestamp, object]] = {}
    contributed: dict[str, int] = {}
    for label, company in [("ticker", Company(ticker))] + [(cik, _company_or_none(Company, cik, ticker)) for cik in entry.all_ciks()]:
        if company is None:
            continue
        listing = company.get_filings(form=forms)
        for f in listing:
            if f.accession_number in by_accession:
                continue
            filed = _keep(f)
            if filed is None:
                continue
            by_accession[f.accession_number] = (filed, f)
            contributed[label] = contributed.get(label, 0) + 1

    # Log whenever a SEGMENT contributed, not merely when provenance is split. `len() > 1`
    # was the wrong test: a ticker whose successor accessions are all already stored recovers
    # its predecessor's book and reports NOTHING, which is the same silence this whole class
    # hid behind. Contributions keyed "ticker" alone are the no-op case and stay quiet.
    if any(label != "ticker" for label in contributed):
        logger.info(
            "%s: %s across the %s boundary (%s)",
            ticker,
            ", ".join(f"{n} from {k}" for k, n in contributed.items()),
            " -> ".join(entry.all_ciks()),
            ",".join(forms),
        )
    dated = sorted(by_accession.values(), key=lambda pair: pair[0])
    return [f for _, f in dated]


def issuer_ciks(ticker: str, roster_cik: str, registrants: dict[str, Registrant] | None = None) -> frozenset[str]:
    """Every CIK that legitimately identifies THIS ticker as the SUBJECT of a schedule.

    ⚠ THE ISSUER/FILER GUARD ON 13D/13G WAS A SINGLE-CIK TEST, AND THE REGISTER BROKE IT.
    `build_ticker_13g_edgar` (and its 13D twin) keeps a filing only when the issuer CIK read off
    the schedule equals `pad_cik(cik)` -- the ROSTER's CIK. Before the register that pairing was
    accidentally consistent: one CIK was listed and the same one was compared. Widening the
    listing to every segment without widening the comparison means a genuine pre-boundary
    schedule -- filed ABOUT the predecessor, carrying the predecessor's issuer CIK -- is
    rejected as "the ticker is a FILER here".

    Measured 2026-09-10: `sec-13d` ran 16/16 ok and stored **+0** rows after resolving MDT 28,
    BLK 50, VTRS 16+6 and ICE 10 predecessor filings. That +0 read like "nothing to add"; it was
    the guard discarding every one. Across all 16 register tickers `sec_13g` and `sec_13d` hold
    **zero** rows on any predecessor CIK, and each ticker's window starts at its boundary --
    APA 2021-04-09 against a 2021-03-01 cutover, GOOGL 2016-01-28 against 2015-10-02.
    """
    ciks = {pad_cik(roster_cik)} if roster_cik else set()
    registrants = load_registrants() if registrants is None else registrants
    entry = registrants.get(ticker)
    if entry is not None:
        ciks.update(entry.all_ciks())
    return frozenset(c for c in ciks if c)


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
# Compatibility name for callers/tests written before subject-first discovery. The cap now
# applies only AFTER header-level subject filtering; broad candidate books are never skipped.
SCHEDULE_SEGMENT_CAP = SCHEDULE_SUBJECT_CAP


class ScheduleDiscoveryIncompleteError(RuntimeError):
    """A subject-first schedule search could not prove its result complete."""


def _atom_text(entry: ElementTree.Element, name: str) -> str | None:
    value = entry.findtext(f"atom:content/atom:{name}", namespaces=_ATOM_NAMESPACE)
    return value.strip() if value and value.strip() else None


def _header_subject_ciks(filing: object) -> frozenset[str]:
    """Read schedule subject CIKs from the SGML header, before ``filing.obj()``."""
    header = filing.header
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
            subjects = _header_subject_ciks(filing)
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


def resolve_schedule_subject_filings(
    ticker: str,
    subject_ciks: frozenset[str],
    forms: Sequence[str],
    *,
    since: pd.Timestamp | None,
    done_accessions: frozenset[str],
    through: pd.Timestamp | None = None,
) -> list[object]:
    """Discover schedules by issuer CIK, then filter SGML headers before full parsing.

    SEC's owner-inclusive Atom search reaches filings submitted under a reporting person's CIK.
    Every page must load and parse; otherwise this raises so the run manifest cannot advance.
    """
    from edgar import Filing
    from edgar.httprequests import download_text

    start_date = pd.Timestamp(since).normalize() if since is not None else None
    end_date = pd.Timestamp(through or pd.Timestamp.today()).normalize()
    target_forms = frozenset(forms)
    families = sorted({form.removesuffix("/A") for form in forms})
    candidates: dict[str, object] = {}
    pages = 0
    owner_rows_excluded = 0

    def _collect_window(
        cik: str,
        family: str,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
    ) -> tuple[dict[str, object], int]:
        """Exhaust one inclusive date window, bisecting before unsafe deep pagination."""
        nonlocal owner_rows_excluded
        start = 0
        window_candidates: dict[str, object] = {}
        window_pages = 0
        while True:
            if start >= SCHEDULE_ATOM_SAFE_OFFSET:
                if window_start >= window_end:
                    raise ScheduleDiscoveryIncompleteError(
                        f"{ticker}: schedule search reached the safe pagination limit "
                        f"inside the unsplittable date {window_start.date()} for subject CIK "
                        f"{cik}, form {family}"
                    )
                span_days = int((window_end - window_start).days) + 1
                if span_days > 366:
                    logger.info(
                        "%s: schedule search reached offset %d for subject CIK %s, form %s; " "partitioning %s..%s into complete one-year windows",
                        ticker,
                        start,
                        cik,
                        family,
                        window_start.date(),
                        window_end.date(),
                    )
                    partitioned: dict[str, object] = {}
                    partition_pages = window_pages
                    cursor = window_start
                    while cursor <= window_end:
                        chunk_end = min(
                            cursor + pd.DateOffset(years=1) - pd.Timedelta(days=1),
                            window_end,
                        )
                        found, found_pages = _collect_window(cik, family, cursor, chunk_end)
                        partitioned.update(found)
                        partition_pages += found_pages
                        cursor = chunk_end + pd.Timedelta(days=1)
                    return partitioned, partition_pages

                older_end = (window_start + (window_end - window_start) / 2).normalize()
                newer_start = older_end + pd.Timedelta(days=1)
                logger.info(
                    "%s: schedule search reached offset %d for subject CIK %s, form %s; " "bisecting %s..%s into complete date windows",
                    ticker,
                    start,
                    cik,
                    family,
                    window_start.date(),
                    window_end.date(),
                )
                older, older_pages = _collect_window(cik, family, window_start, older_end)
                newer, newer_pages = _collect_window(cik, family, newer_start, window_end)
                older.update(newer)
                return older, window_pages + older_pages + newer_pages

            url = SEC_INSIDER_OWNER_ATOM_URL.format(
                cik=cik,
                form=quote_plus(family),
                date_from=window_start.strftime("%Y%m%d"),
                date_to=window_end.strftime("%Y%m%d"),
                start=start,
                count=SEC_INSIDER_OWNER_ATOM_PAGE_SIZE,
            )
            try:
                payload = call_with_retries(
                    lambda url=url: download_text(url),
                    label=f"{ticker} {family} offset {start}",
                )
                root = ElementTree.fromstring(payload)
            except Exception as exc:  # noqa: BLE001 -- completeness is the contract
                raise ScheduleDiscoveryIncompleteError(
                    f"{ticker}: schedule search failed for subject CIK {cik}, form " f"{family}, offset {start}: {exc!r}"
                ) from exc
            window_pages += 1
            entries = root.findall("atom:entry", _ATOM_NAMESPACE)
            if not entries:
                break
            for entry in entries:
                form = _atom_text(entry, "filing-type")
                accession = _atom_text(entry, "accession-number")
                filing_date = pd.to_datetime(_atom_text(entry, "filing-date"), errors="coerce")
                file_number = _atom_text(entry, "file-number")
                if (
                    form not in target_forms
                    or accession is None
                    or accession in done_accessions
                    or accession in window_candidates
                    or pd.isna(filing_date)
                    or filing_date.normalize() > window_end
                    or filing_date.normalize() < window_start
                ):
                    continue
                # In an owner-inclusive Schedule result, SEC includes a file number only
                # when the queried CIK is the SUBJECT issuer. Reporting-person rows omit it.
                # This is the cheap role discriminator: BLK's 2023-02-14 feed contained 32
                # results but only wanted accession 0001306550-23-008694 carried 005-82091.
                # The SGML subject-CIK guard below remains the final authority.
                if file_number is None:
                    owner_rows_excluded += 1
                    continue
                window_candidates[accession] = Filing(
                    cik=int(cik),
                    company=ticker,
                    form=form,
                    filing_date=filing_date.strftime("%Y-%m-%d"),
                    accession_no=accession,
                )
            if len(entries) < SEC_INSIDER_OWNER_ATOM_PAGE_SIZE:
                break
            start += SEC_INSIDER_OWNER_ATOM_PAGE_SIZE
        return window_candidates, window_pages

    if start_date is None:
        raise ScheduleDiscoveryIncompleteError(
            f"{ticker}: subject-first schedule discovery requires a finite start date so " "deep result sets can be split without truncation"
        )
    for cik in sorted(subject_ciks):
        for family in families:
            found, window_pages = _collect_window(cik, family, start_date, end_date)
            pages += window_pages
            candidates.update(found)

    filtered, stats = filter_schedule_subject_filings(list(candidates.values()), subject_ciks)
    filtered.sort(key=lambda filing: pd.Timestamp(filing.filing_date))
    logger.info(
        "%s: subject-first schedules -- %d page(s), %d owner-side row(s) excluded from Atom "
        "metadata, %d candidate(s), %d subject match(es), %d unknown header(s), %d retained "
        "for full parsing",
        ticker,
        pages,
        owner_rows_excluded,
        stats["candidates"],
        stats["subject_matches"],
        stats["unknown_headers"],
        len(filtered),
    )
    return filtered


def drop_rows_outside_segment(
    df: pd.DataFrame, *, cik_col: str, ticker_col: str, filed_col: str, registrants: dict[str, Registrant] | None = None
) -> pd.DataFrame:
    """Drop bulk-dataset rows whose `filed` date lies outside the segment that CIK owns.

    THE SECOND HALF OF TIER C, AND WITHOUT IT THE FIRST HALF IS A REGRESSION. Adding every
    predecessor CIK to `cik_to_ticker` makes a predecessor's rows resolve to the ticker --
    which is right for an EVENT table and wrong for a CONSOLIDATING one. Apache Corp filed
    its own 10-K/10-Q as a subsidiary until 2024-11-07, so its post-2021 notes and pension
    facts would land under APA and blend a subsidiary's disclosures into the parent's, which
    is exactly what the dated split exists to prevent.

    So a row is kept only when the segment covering its `filed` date is the segment whose CIK
    filed it. Tickers with no register entry are untouched, which is ~449 of 491.

    Vectorised on purpose: `notes_num` and `notes_text` are read a period at a time and a
    per-row Python loop over 77 periods is minutes of pure overhead.
    """
    registrants = load_registrants() if registrants is None else registrants
    if df.empty or not registrants:
        return df
    covered = df[ticker_col].isin(registrants)
    if not covered.any():
        return df

    filed = pd.to_datetime(df[filed_col], errors="coerce")
    cik = df[cik_col].astype(str).str.zfill(10)
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
            "registrant split: dropped %d bulk row(s) filed outside their segment " "(%s)",
            dropped,
            ", ".join(sorted(set(df.loc[~keep, ticker_col].astype(str)))),
        )
    return df[keep]


def _company_or_none(company_cls, cik: str, ticker: str):
    """A dead or unresolvable CIK costs that registrant's filings, never the whole walk."""
    try:
        return company_cls(int(cik))
    except Exception:  # noqa: BLE001 -- a dead CIK, not a bug
        logger.warning("%s: register CIK %s could not be resolved", ticker, cik)
        return None


def _filings(company_cls, cik: str, forms: list[str]):
    company = _company_or_none(company_cls, cik, "")
    return [] if company is None else company.get_filings(form=forms)


def _normalise_cik(value: Any) -> str:
    """CIKs are 10-digit zero-padded everywhere in this repo (see the 13F loader), and a
    config written as an int or a bare string must join against that without a surprise."""
    return str(value).strip().zfill(10)


def _stamp(ticker: str, i: int, value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError(f"registrant[{ticker}] segment {i}: unparseable date {value!r}")
    return stamp
