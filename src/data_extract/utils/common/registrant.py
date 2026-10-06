"""The single authority on which CIKs a ticker's filings are listed from.

Filings follow the legal registrant, whose CIK changes on a reorganisation or domestication.
`registrant_cutover.json` declares each such ticker as an ordered, contiguous chain of evidenced
`[valid_from, valid_to)` segments, validated strictly at load; the lineage build turns it into dated
CIK windows. `FORM_POLICY` decides per form whether filings UNION across a ticker's event CIKs
(insider forms) or SPLIT by its CIK windows (everything else; `scope_policy` unions 8-K / 13D / 13G for
a ticker deferred to the traded-security realignment); `resolve_registrant_entries` applies
it to local EDGAR index rows and `resolve_registrant_filings` to `Company` listings, both by CIK
only, never by symbol.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cache
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.data_extract.utils.common import sec_io
from src.data_extract.utils.common.config_paths import resolve_config_dir
from src.utils.string import pad_cik

if TYPE_CHECKING:  # identity -> entity_lineage -> registrant: annotation-only import breaks the cycle
    from src.data_extract.utils.common.identity import CikWindow, FilingScope

logger = logging.getLogger(__name__)

#: `configs/sec/registrant_cutover.json`, read only through `load_registrants`.
REGISTRANT_CONFIG_SUBDIR = "sec"
REGISTRANT_CONFIG_FILENAME = "registrant_cutover.json"

#: `reorganisation` = a new legal parent; `domestication` = re-registered in another jurisdiction. Both change the CIK.
CUTOVER_KINDS: frozenset[str] = frozenset({"reorganisation", "domestication"})

#: Rejected by name: a rename keeps the CIK, so an entry would walk one CIK twice and duplicate every filing.
RENAME_KIND = "rename"


@dataclass(frozen=True)
class Segment:
    """One registrant's tenure over a ticker: `[valid_from, valid_to)`.

    The oldest segment has no `valid_from` and the newest no `valid_to`, so every date lands in exactly one segment.
    """

    cik: str
    valid_from: pd.Timestamp | None
    valid_to: pd.Timestamp | None
    evidence: str

    def covers(self, date) -> bool:
        """Half-open membership, so adjacent segments are disjoint by construction."""
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

    def all_ciks(self) -> tuple[str, ...]:
        """Every CIK in the chain, oldest first (the UNION set for event forms)."""
        return tuple(s.cik for s in self.segments)

    @property
    def boundaries(self) -> tuple[pd.Timestamp, ...]:
        """The seam dates, oldest first. `len(segments) - 1` of them."""
        return tuple(s.valid_from for s in self.segments[1:] if s.valid_from is not None)


def load_registrants(config_dir: str | None = None) -> dict[str, Registrant]:
    """The registrant register keyed by ticker, cached per resolved config directory; `{}` when absent."""
    return _registrants_at(resolve_config_dir(config_dir))


@cache
def _registrants_at(config_dir: str) -> dict[str, Registrant]:
    """`load_registrants` keyed on a resolved path; validation is strict and raises at load."""
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
    """No CIK may appear in two tickers' chains (it would route one company's bulk rows to two names)."""
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

    UNION = "union"  # insider forms: every CIK of the entity, each row stamped with its lineage role
    SPLIT = "split"  # one registrant owns each date: a CIK's filings count only inside its seam-widened window


#: Every form this repo fetches and how it combines across a registrant boundary; an absent form raises (fail closed).
#: Insider forms UNION (an acquired company's rows are kept with their role); 8-Ks, schedules, periodic reports
#: and the proxy family SPLIT, since a CIK speaks for the company only on the dates it is the company.
FORM_POLICY: dict[str, Combine] = {
    # events -- 8-K
    "8-K": Combine.SPLIT,
    "8-K/A": Combine.SPLIT,
    "8-K12B": Combine.SPLIT,
    # events -- beneficial ownership, by subject company; both spellings, since EDGAR renamed the form types and matching is exact.
    "SC 13D": Combine.SPLIT,
    "SC 13D/A": Combine.SPLIT,
    "SCHEDULE 13D": Combine.SPLIT,
    "SCHEDULE 13D/A": Combine.SPLIT,
    "SC 13G": Combine.SPLIT,
    "SC 13G/A": Combine.SPLIT,
    "SCHEDULE 13G": Combine.SPLIT,
    "SCHEDULE 13G/A": Combine.SPLIT,
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
    """The one policy governing `forms`; raises `ValueError` for an unknown form, a mixed list or no forms."""
    unknown = [f for f in forms if f not in FORM_POLICY]
    if unknown:
        raise ValueError(
            f"no registrant-combination policy declared for form(s) {unknown}. Add them to "
            "`FORM_POLICY` as UNION (every CIK of the entity, role stamped per row) or SPLIT "
            "(one registrant owns each date). This RAISES rather "
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


#: The event forms SPLIT by dated CIK window (P35); a scope with `undated_events` lists them UNION instead.
DATED_EVENT_FORMS = frozenset(
    {"8-K", "8-K/A", "8-K12B", "SC 13D", "SC 13D/A", "SCHEDULE 13D", "SCHEDULE 13D/A", "SC 13G", "SC 13G/A", "SCHEDULE 13G", "SCHEDULE 13G/A"}
)


def scope_policy(scope: FilingScope, forms: Sequence[str]) -> Combine:
    """`combine_for(forms)`, except that a ticker deferred to the traded-security realignment unions its 8-K / 13D / 13G."""
    policy = combine_for(forms)
    if policy is Combine.SPLIT and scope.undated_events and set(forms) <= DATED_EVENT_FORMS:
        return Combine.UNION
    return policy


@dataclass
class _FilingWindow:
    """The filter shared by every walk: the scope guard, `since` and `done_accessions`.

    A filing whose CIK is outside the listed CIKs is skipped and counted in `stats["foreign_skipped"]`,
    never raised; a stored accession is counted in `stats["skipped_existing"]`.
    """

    ticker: str
    since: pd.Timestamp | None
    done_accessions: frozenset[str]
    stats: dict[str, int] | None
    scope_ciks: frozenset[str]
    skipped_existing: set[str] = field(default_factory=set)
    foreign: set[str] = field(default_factory=set)
    foreign_ciks: set[str] = field(default_factory=set)

    def filed(self, filing: Any) -> pd.Timestamp | None:
        """The filing date when `filing` is kept, else None."""
        filer = getattr(filing, "cik", None)
        if filer is not None and (cik := pad_cik(filer)) not in self.scope_ciks:
            self.foreign.add(str(filing.accession_number))
            if self.stats is not None:
                self.stats["foreign_skipped"] = len(self.foreign)
            if cik not in self.foreign_ciks:
                self.foreign_ciks.add(cik)
                logger.warning("%s: skipped filing(s) from CIK %s, outside the filing scope (first: %s)", self.ticker, cik, filing.accession_number)
            return None
        if filing.accession_number in self.done_accessions:
            if self.stats is not None:
                self.skipped_existing.add(filing.accession_number)
                self.stats["skipped_existing"] = len(self.skipped_existing)
            return None
        filed = pd.Timestamp(filing.filing_date)
        return None if self.since is not None and filed < self.since else filed


def resolve_registrant_filings(
    scope: FilingScope,
    forms: Sequence[str],
    *,
    since: pd.Timestamp | None,
    done_accessions: frozenset[str],
    stats: dict[str, int] | None = None,
) -> list:
    """Every filing of `forms` in `scope`, oldest first, listed by CIK only.

    UNION (`FORM_POLICY`) lists every event CIK; SPLIT lists each CIK window and keeps its filings
    inside the seam-widened dates, so an event-only CIK contributes nothing. A filing from a CIK outside the listed CIKs is skipped and counted
    (`stats["foreign_skipped"]`). `since` and `done_accessions` filter before the sort.
    """
    policy = scope_policy(scope, forms)
    forms = list(forms)
    if stats is not None:
        stats.setdefault("skipped_existing", 0)
        stats.setdefault("foreign_skipped", 0)
    walks: list[tuple[str, CikWindow | None]]
    if policy is Combine.UNION:
        walks = [(cik, None) for cik in scope.event_ciks]
    else:
        walks = [(window.cik, window) for window in scope.windows]
    keep = _FilingWindow(scope.ticker, since, done_accessions, stats, frozenset(cik for cik, _ in walks))
    filings, contributions = _walk(scope.ticker, walks, forms, keep.filed, scope.windows)
    if len(contributions) > 1:
        logger.info(
            "%s: %s listed by %s across %s",
            scope.ticker,
            ",".join(forms),
            policy.value,
            ", ".join(f"{n} from {cik}" for cik, n in contributions.items()),
        )
    return filings


def _walk(
    ticker: str,
    walks: list[tuple[str, CikWindow | None]],
    forms: list[str],
    keep: Callable[[Any], pd.Timestamp | None],
    windows: Sequence[CikWindow] = (),
) -> tuple[list, dict[str, int]]:
    """Kept filings of every walk sorted by filing date, one per accession, and the count per CIK.

    A windowed walk keeps only filings its widened window admits. An accession two walks list (a joint
    filing, or a successor listing its predecessor's history) goes to the CIK whose stated window owns
    its date, else the first walk, so the stamp is the window owner's CIK.
    """
    by_accession: dict[str, tuple[pd.Timestamp, Any, str, bool]] = {}
    for cik, window in walks:
        company = _company_or_none(cik, ticker)
        for filing in [] if company is None else sec_io.company_filings(company, forms):
            prior = by_accession.get(filing.accession_number)
            if prior is not None and prior[3]:
                continue
            filed = keep(filing)
            if filed is None or (window is not None and not window.admits(filed)):
                continue
            owned = window.owns(filed) if window is not None else any(w.cik == cik and w.owns(filed) for w in windows)
            if prior is None or owned:
                by_accession[filing.accession_number] = (filed, filing, cik, owned)
    contributions: dict[str, int] = {}
    for _, _, cik, _ in by_accession.values():
        contributions[cik] = contributions.get(cik, 0) + 1
    return [filing for _, filing, _, _ in sorted(by_accession.values(), key=itemgetter(0))], contributions


def listing_ciks(scope: FilingScope) -> tuple[str, ...]:
    """Every CIK whose index rows may list the scope's filings, roster first (the candidate superset)."""
    ciks = [scope.roster_cik, *scope.event_ciks, *(window.cik for window in scope.windows)]
    return tuple(dict.fromkeys(c for c in ciks if c))


def resolve_registrant_entries(scope: FilingScope, df_entries: pd.DataFrame, forms: Sequence[str]) -> pd.DataFrame:
    """The scope's index rows of `forms` (`cik`, `company`, `form`, `filed`, `accession`), one per accession, oldest first.

    The `FORM_POLICY` rules of `resolve_registrant_filings`, applied to index rows: UNION keeps every
    event CIK's rows; SPLIT keeps each CIK window's rows inside its seam-widened dates. An accession
    two CIKs list goes to the CIK whose stated window owns its date, else the first CIK listed.
    """
    policy = scope_policy(scope, forms)
    df = df_entries[df_entries["form"].isin(list(forms))]
    walks: list[tuple[str, CikWindow | None]]
    if policy is Combine.UNION:
        walks = [(cik, None) for cik in scope.event_ciks]
    else:
        walks = [(window.cik, window) for window in scope.windows]
    filed = pd.to_datetime(df["filed"])
    parts: list[pd.DataFrame] = []
    for rank, (cik, window) in enumerate(walks):
        rows = df["cik"].eq(cik)
        if window is not None:
            rows &= filed.map(window.admits).astype(bool)
        owners = [window] if window is not None else [w for w in scope.windows if w.cik == cik]
        days: list[pd.Timestamp] = list(filed[rows])
        parts.append(df[rows].assign(_owned=[any(w.owns(day) for w in owners) for day in days], _rank=rank))
    if not parts:
        return df.iloc[:0].reset_index(drop=True)
    kept = pd.concat(parts, ignore_index=True).sort_values(["_owned", "_rank"], ascending=[False, True], kind="mergesort")
    kept = kept.drop_duplicates("accession", keep="first")
    contributions = kept["cik"].value_counts()
    if len(contributions) > 1:
        logger.info(
            "%s: %s index rows by %s across %s",
            scope.ticker,
            ",".join(forms),
            policy.value,
            ", ".join(f"{n} from {cik}" for cik, n in contributions.items()),
        )
    return kept.drop(columns=["_owned", "_rank"]).sort_values(["filed", "accession"], ignore_index=True)


def header_subject_ciks(filing: object) -> frozenset[str]:
    """Read schedule subject CIKs from the SGML header, before ``filing.obj()``.

    An unreadable header raises (`sec_io.TransientReadError` when SEC served an error page)."""
    header = sec_io.filing_header(filing)
    companies = getattr(header, "subject_companies", ()) or ()
    raw_ciks = (getattr(company, "cik", None) or getattr(getattr(company, "company_information", None), "cik", "") for company in companies)
    return frozenset(normalized for cik in raw_ciks if (normalized := pad_cik(cik)))


def _company_or_none(cik: str, ticker: str) -> Any | None:
    """`Company` for one scope CIK; an unresolvable CIK is warned and skipped, a transient SEC failure raises."""
    try:
        return sec_io.company(int(cik))
    except sec_io.TransientReadError:
        raise
    except Exception:  # noqa: BLE001 -- a dead CIK, not a bug
        logger.warning("%s: CIK %s could not be resolved", ticker, cik)
        return None


def _stamp(ticker: str, i: int, value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError(f"registrant[{ticker}] segment {i}: unparseable date {value!r}")
    return stamp
