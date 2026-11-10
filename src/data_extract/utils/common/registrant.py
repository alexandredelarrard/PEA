"""The single authority on which CIKs a ticker's filings are listed from.

Filings follow the legal registrant, whose CIK changes on a reorganisation or domestication.
`registrant_cutover.json` declares each such ticker as an ordered, contiguous chain of evidenced
`[valid_from, valid_to)` segments, validated strictly at load; a `fresh_start` entry is one CIK counted
only from its `valid_from`. The lineage build turns it into dated CIK windows. `FORM_POLICY` decides per
form whether filings UNION across a ticker's event CIKs (insider forms) or SPLIT by its CIK windows (everything
else); `resolve_registrant_entries` applies it to local EDGAR index rows, by CIK only, never by symbol. Config and
local rows only: this module imports no SEC client.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from src.data_extract.utils.common.config_paths import resolve_config_dir
from src.data_store.schema import Resume, Tables
from src.utils.string import pad_cik

if TYPE_CHECKING:  # identity -> universe_cik_symbol -> registrant: annotation-only import breaks the cycle
    from src.data_extract.utils.common.identity import CikWindow, FilingScope

logger = logging.getLogger(__name__)

#: `configs/sec/registrant_cutover.json`, read only through `load_registrants`.
REGISTRANT_CONFIG_SUBDIR = "sec"
REGISTRANT_CONFIG_FILENAME = "registrant_cutover.json"

#: `reorganisation` = a new legal parent; `domestication` = re-registered in another jurisdiction. Both change the CIK.
CUTOVER_KINDS: frozenset[str] = frozenset({"reorganisation", "domestication"})

#: The CIK is unchanged but the ticker's filings count only from `valid_from` (a same-CIK bankruptcy fresh start).
FRESH_START_KIND = "fresh_start"

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
    """One validated `Registrant`, or a `ValueError` naming the ticker and the rule broken.

    A cutover kind is a contiguous chain of at least two CIKs; `fresh_start` is exactly one segment with `valid_from`.
    """
    kind = str(entry.get("kind", ""))
    if kind == RENAME_KIND:
        raise ValueError(
            f"registrant[{ticker}]: kind='{RENAME_KIND}' is not a cutover. A rename keeps the "
            "CIK (CVS Caremark -> CVS Health, Facebook -> Meta), so an entry here would walk "
            "one CIK twice and duplicate every filing. Delete it."
        )
    if kind == FRESH_START_KIND:
        return _parse_fresh_start(ticker, entry)
    if kind not in CUTOVER_KINDS:
        raise ValueError(f"registrant[{ticker}]: kind={kind!r} not in {sorted(CUTOVER_KINDS | {FRESH_START_KIND})}")

    raw = entry.get("segments")
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError(f"registrant[{ticker}]: `segments` must be a list of at least 2 entries -- one segment is not a boundary.")

    segments: list[Segment] = []
    for i, seg in enumerate(raw):
        _require_evidence(ticker, i, seg)
        first, last = i == 0, i == len(raw) - 1
        if first and "valid_from" in seg:
            raise ValueError(f"registrant[{ticker}] segment 0: the oldest segment must omit `valid_from` -- the chain is open at the old end.")
        if last and "valid_to" in seg:
            raise ValueError(f"registrant[{ticker}] segment {i}: the newest segment must omit `valid_to` -- the chain is open at the new end.")
        if not first and "valid_from" not in seg:
            raise ValueError(f"registrant[{ticker}] segment {i}: missing `valid_from`.")
        if not last and "valid_to" not in seg:
            raise ValueError(f"registrant[{ticker}] segment {i}: missing `valid_to`.")
        segments.append(_segment(ticker, i, seg))

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


def _parse_fresh_start(ticker: str, entry: dict[str, Any]) -> Registrant:
    """A `fresh_start` entry: exactly one evidenced segment, closed at `valid_from` and open at the new end."""
    raw = entry.get("segments")
    if not isinstance(raw, list) or len(raw) != 1:
        raise ValueError(f"registrant[{ticker}]: kind='{FRESH_START_KIND}' takes exactly 1 segment -- the CIK does not change.")
    seg = raw[0]
    _require_evidence(ticker, 0, seg)
    if seg.get("valid_from") is None:
        raise ValueError(f"registrant[{ticker}] segment 0: kind='{FRESH_START_KIND}' requires `valid_from` -- the date its filings start to count.")
    if "valid_to" in seg:
        raise ValueError(f"registrant[{ticker}] segment 0: a fresh start must omit `valid_to` -- it is open at the new end.")
    return Registrant(ticker=ticker, kind=FRESH_START_KIND, segments=(_segment(ticker, 0, seg),))


def _require_evidence(ticker: str, i: int, seg: dict[str, Any]) -> None:
    """Raise unless segment `i` documents its evidence."""
    if not str(seg.get("evidence", "")).strip():
        raise ValueError(
            f"registrant[{ticker}] segment {i}: empty `evidence`. An undocumented entry "
            "is a guess that deletes history, which is exactly what this register replaces."
        )


def _segment(ticker: str, i: int, seg: dict[str, Any]) -> Segment:
    """One parsed segment: padded CIK, parsed dates, evidence."""
    return Segment(
        cik=pad_cik(seg["cik"]),
        valid_from=_stamp(ticker, i, seg.get("valid_from")),
        valid_to=_stamp(ticker, i, seg.get("valid_to")),
        evidence=str(seg["evidence"]),
    )


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
    form: policy
    for table, policy in (
        (Tables.sec_8k, Combine.SPLIT),
        (Tables.sec_13d, Combine.SPLIT),
        (Tables.sec_13g, Combine.SPLIT),
        (Tables.insider_transactions, Combine.UNION),
        (Tables.sec_fundamentals_facts, Combine.SPLIT),
        (Tables.filing_risk_text, Combine.SPLIT),
        (Tables.sec_employees_text, Combine.SPLIT),
        (Tables.sec_def14a_text, Combine.SPLIT),
    )
    for form in cast(Resume, table.resume).forms
}


def combine_for(forms: Sequence[str]) -> Combine:
    """The one policy governing `forms`; raises `ValueError` for an unknown form, a mixed list or no forms."""
    unknown = [f for f in forms if f not in FORM_POLICY]
    if unknown:
        raise ValueError(
            f"no registrant-combination policy declared for form(s) {unknown}. Add them to "
            "the owning table's Resume.forms and declare that table in `FORM_POLICY` as UNION "
            "(every CIK of the entity, role stamped per row) or SPLIT "
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


def listing_ciks(scope: FilingScope) -> tuple[str, ...]:
    """Every CIK whose index rows may list the scope's filings, roster first (the candidate superset)."""
    ciks = [scope.roster_cik, *scope.event_ciks, *(window.cik for window in scope.windows)]
    return tuple(dict.fromkeys(c for c in ciks if c))


def resolve_registrant_entries(scope: FilingScope, df_entries: pd.DataFrame, forms: Sequence[str]) -> pd.DataFrame:
    """The scope's index rows of `forms` (`cik`, `company`, `form`, `filed`, `accession`), one per accession, oldest first.

    The `FORM_POLICY` rules applied to index rows: UNION keeps every
    event CIK's rows; SPLIT keeps each CIK window's rows inside its seam-widened dates. An accession
    two CIKs list goes to the CIK whose stated window owns its date, else the first CIK listed.
    """
    policy = combine_for(forms)
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


def _stamp(ticker: str, i: int, value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError(f"registrant[{ticker}] segment {i}: unparseable date {value!r}")
    return stamp
