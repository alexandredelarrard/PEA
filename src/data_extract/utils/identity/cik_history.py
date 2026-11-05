"""CIK history: which CIKs are one universe company, and over which dates a CIK's consolidating filings are the company's.

Membership oracles, highest priority first: `register` (`registrant_cutover.json`), `manual` (`entity_lineage_manual.json`),
`sec_succession` (an 8-K12B on one candidate CIK and a Form 25 or 15 on another within `SWITCH_TOLERANCE_DAYS`, read from
`sec_cik_profile`), `symbol_handoff` (one symbol passing between two CIKs in both filed sources), `owner_overlap`
(shared Form 3/4/5 reporting owners) and `roster`. No merge joins two universe tickers.

Windows: the register's segments; else the SEC succession chain; else the automatic chain both filed sources agree on;
else the roster CIK alone, the other CIKs event-only. `company_id` is the ticker's anchored id, else `E` + the oldest CIK
of its group, and is never re-keyed. Rows are the `cik_window` / `cik_event` rows of `universe_cik_symbol`.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_extract.utils.common.config_paths import resolve_config_dir
from src.data_extract.utils.common.registrant import Registrant
from src.data_extract.utils.identity.symbol_tenure import DEI_SOURCE
from src.utils.identity_flags import RESOLUTIONS_KEY
from src.utils.string import normalise_ticker, pad_cik, pad_cik_series

logger = logging.getLogger(__name__)

#: `configs/sec/entity_lineage_manual.json`.
MANUAL_CONFIG_SUBDIR = "sec"
MANUAL_CONFIG_FILENAME = "entity_lineage_manual.json"
#: Key in `entity_lineage_manual.json` holding the D19 cross-check's exceptions; underscore-prefixed so the lineage loader skips it.
D19_ALLOWLIST_KEY = "_d19_allowlist"

#: Oracle `owner_overlap`: shared == 0 -> unrelated; jaccard >= 0.05 OR shared >= 5 -> same entity; between is a grey band.
OVERLAP_JACCARD_SAME = 0.05
OVERLAP_SHARED_SAME = 5

SOURCE_PRIORITY = ("register", "manual", "sec_succession", "symbol_handoff", "owner_overlap", "roster")

#: Stored value of an open start: `valid_from` is a PK column and cannot be NULL.
SENTINEL_START = pd.Timestamp("1900-01-01")

ROLE_WINDOW = "cik_window"
ROLE_EVENT = "cik_event"
ROLE_SYMBOL = "symbol"

FORM345_SOURCE = "form345"
MANUAL_SOURCE = "manual"
ROSTER_SOURCE = "roster"
REGISTER_SOURCE = "register"
SEC_PROFILE_SOURCE = "sec_profile"
#: The two filed evidence sources a symbol handoff and an automatic window need, both on both CIKs.
EVIDENCE_SOURCES = (FORM345_SOURCE, DEI_SOURCE)

#: A CIK switch: the predecessor's last and the successor's first filing within this many days, in each source.
SWITCH_TOLERANCE_DAYS = 31
#: Automatic CIK windows: on; a register entry always wins and every automatic chain is listed in `validate identity`.
AUTO_WINDOWS_ENABLED = True
#: The successor's form and the predecessor's forms that pair into an SEC succession (Form 25 / 15 are filed per security).
SUCCESSOR_FORMS = frozenset({"8-K12B"})
PREDECESSOR_END_FORMS = frozenset({"25", "25-NSE", "15-12B", "15-12G"})

#: Column order of `universe_cik_symbol`.
TABLE_COLUMNS = (
    "company_id",
    "canonical_ticker",
    "cik",
    "role",
    "symbol",
    "valid_from",
    "valid_to",
    "status",
    "sources",
    "oracle",
    "confidence",
    "n_observations",
    "evidence",
    "scope_changed_at",
)
BACKLOG_COLUMNS = ("kind", "canonical_ticker", "company_id", "cik", "symbol", "detail")
BLOCKED_COLUMNS = ("cik_a", "cik_b", "proposed_by", "confidence", "tickers_a", "tickers_b")

#: The evidence sources behind each membership oracle (owner sets are read off Forms 3/4/5).
_ORACLE_SOURCES = {
    "register": REGISTER_SOURCE,
    "manual": MANUAL_SOURCE,
    "sec_succession": SEC_PROFILE_SOURCE,
    "symbol_handoff": ",".join(sorted(EVIDENCE_SOURCES)),
    "owner_overlap": FORM345_SOURCE,
    "roster": ROSTER_SOURCE,
}
_ORACLE_STATUS = {"register": "curated", "manual": "curated", "sec_succession": "corroborated", "symbol_handoff": "corroborated"}
_ROSTER_VERDICT = ("roster", None, "roster CIK with no predecessor found by any oracle")

#: A CIK's consolidating window: (valid_from, valid_to or None, sources, status, evidence note).
Window = tuple[pd.Timestamp, pd.Timestamp | None, str, str, str]


class IdentityError(ValueError):
    """Base of every identity failure, so one `except` covers the whole layer."""


class CikInTwoEntitiesError(IdentityError):
    """One CIK carries two `company_id`s; asserted by the build, since the PK does not guarantee it."""


class UniverseEntityDisagreementError(IdentityError):
    """D19: the roster CIK and `symbol_tenure` name different entities for one ticker, with no D19 allow-list entry."""


class TwoUniverseTickersOneEntityError(ValueError):
    """A merge would put two universe tickers in one entity; the fix is a curated row, never a wider merge."""


class CompanyIdCollisionError(IdentityError):
    """A frozen `company_id` is also the minted id of another group."""


@dataclass
class Membership:
    """The union-find over candidate CIKs after the oracles, with what each oracle decided."""

    parent: dict[str, str]
    roster_of: dict[str, frozenset[str]]
    verdicts: dict[str, tuple[str, float | None, str]] = field(default_factory=dict)
    curated: set[str] = field(default_factory=set)
    blocked: list[tuple[str, str, str, float | None]] = field(default_factory=list)
    grey: list[tuple[str, str, str, int, float]] = field(default_factory=list)
    overlap_verdicts: Counter = field(default_factory=Counter)
    joins: Counter = field(default_factory=Counter)


@dataclass(frozen=True)
class Entities:
    """`company_id` of every member CIK, the CIKs of each company and the universe ticker each company holds."""

    entity_of: dict[str, str]
    groups: dict[str, set[str]]
    ticker_by_entity: dict[str, str]


@dataclass(frozen=True)
class Switch:
    """A predecessor -> successor CIK switch read from both sources, or the reason it is undecided."""

    boundary: pd.Timestamp | None
    window: tuple[pd.Timestamp, pd.Timestamp] | None
    detail: str


@dataclass(frozen=True)
class CikHistory:
    """The CIK rows of `universe_cik_symbol` and what the symbol history reads off them."""

    rows: pd.DataFrame
    entity_of: Mapping[str, str]
    ticker_by_entity: Mapping[str, str]
    window_end: Mapping[str, pd.Timestamp | None]
    oracle_of: Mapping[str, str]
    backlog: pd.DataFrame
    blocked: pd.DataFrame
    evidence: pd.DataFrame
    activity: pd.DataFrame
    ended_on: Mapping[str, pd.Timestamp] = field(default_factory=dict)


# Shared row helpers
def entity_or_singleton(entity_by_cik: Mapping[str, str], cik: str) -> str:
    """The stored entity of a padded CIK; a CIK with no stored row is its own entity `E{cik}`."""
    return entity_by_cik.get(cik, f"E{cik}")


def entity_by_cik_map(lineage: pd.DataFrame) -> dict[str, str]:
    """`{cik: company_id}` from `universe_cik_symbol` rows with padded CIKs (several rows per CIK share one entity)."""
    return dict(zip(lineage["cik"].astype(str), lineage["company_id"].astype(str), strict=False))


def check_one_entity_per_cik(lineage: pd.DataFrame) -> None:
    """Raise `CikInTwoEntitiesError` when one (padded) CIK carries two company ids, which would make `entity_of` order-dependent."""
    per_cik = pd.DataFrame({"cik": lineage["cik"].astype(str), "company_id": lineage["company_id"].astype(str)}).drop_duplicates()
    clashes = per_cik[per_cik.duplicated("cik", keep=False)]
    if not clashes.empty:
        raise CikInTwoEntitiesError(
            f"identity: {clashes['cik'].nunique()} CIK(s) carry two company ids in universe_cik_symbol -- "
            f"{clashes.sort_values(['cik', 'company_id']).to_dict('records')}. One CIK belongs to one company."
        )


# Curated configuration
def manual_lineage_blob(config_dir: str | None) -> dict:
    """`configs/sec/entity_lineage_manual.json` as raw JSON; `{}` when absent."""
    # read the file once; the lineage entries and the D19 allow-list are parsed from it
    path = Path(resolve_config_dir(config_dir)) / MANUAL_CONFIG_SUBDIR / MANUAL_CONFIG_FILENAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def d19_allowlist(blob: Mapping[str, Any]) -> dict[str, str]:
    """`{ticker: why its roster CIK may disagree with symbol_tenure}`; any unlisted disagreement raises."""
    allow = blob.get(D19_ALLOWLIST_KEY, {})
    return {ticker: str(why) for ticker, why in allow.items() if not ticker.startswith("_")}


def parse_manual_lineage(blob: Mapping[str, Any]) -> dict[str, dict]:
    """The manual entries `{key: {same_entity, own_entity, evidence}}`, CIKs padded; an entry deciding nothing raises (the
    `_`-prefixed keys and the `resolutions` list are not entries)."""
    # validate every entry: a relationship of >= 2 CIKs or an own-entity verdict, always with evidence
    out: dict[str, dict] = {}
    for key, entry in blob.items():
        if key.startswith("_") or key == RESOLUTIONS_KEY:
            continue
        same = [pad_cik(c) for c in entry.get("same_entity", [])]
        own = [pad_cik(c) for c in entry.get("own_entity", [])]
        if not same and not own:
            raise ValueError(f"entity_lineage_manual[{key}]: needs `same_entity` or `own_entity`; an entry that asserts nothing decides nothing.")
        if len(same) == 1:
            raise ValueError(f"entity_lineage_manual[{key}]: `same_entity` needs >= 2 CIKs -- one CIK is not a relationship. Use `own_entity`.")
        if not str(entry.get("evidence", "")).strip():
            raise ValueError(f"entity_lineage_manual[{key}]: empty `evidence`. An undocumented verdict is a guess that moves a decade of rows.")
        out[key] = {"same_entity": same, "own_entity": own, "evidence": str(entry["evidence"]).strip()}
    return out


# Evidence
def cik_evidence(tenure: pd.DataFrame, dei: pd.DataFrame | None) -> pd.DataFrame:
    """`form345`, `manual` and collapsed `dei` intervals as `[symbol, cik, source, first, end, n, evidence]`; `end` NaT = open."""
    # one frame over every source
    frame = pd.concat([tenure] if dei is None or dei.empty else [tenure, dei], ignore_index=True)
    sources = frame["source"].astype(str) if "source" in frame.columns else pd.Series(FORM345_SOURCE, index=frame.index)

    # normalise symbols, CIKs and bounds
    out = pd.DataFrame(
        {
            "symbol": frame["symbol"].map(normalise_ticker),
            "cik": pad_cik_series(frame["issuer_cik"]),
            "source": sources.str.strip().str.lower(),
            "first": pd.to_datetime(frame["valid_from"]),
            "end": pd.to_datetime(frame["valid_to"]),
            "n": pd.to_numeric(frame["n_filings"], errors="coerce").fillna(0).astype("int64"),
            "evidence": frame["evidence"].fillna("").astype(str) if "evidence" in frame.columns else "",
        }
    )

    # drop unusable rows and sort deterministically
    out = out[out["first"].notna() & out["cik"].ne("") & out["symbol"].ne("")]
    return out.sort_values(["symbol", "cik", "source", "first"], kind="mergesort", ignore_index=True)


def evidence_index(evidence: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    """The `form345` and `dei` evidence rows keyed by `(cik, source)`."""
    filed = evidence[evidence["source"].isin(EVIDENCE_SOURCES)]
    return {(str(cik), str(source)): rows for (cik, source), rows in filed.groupby(["cik", "source"], sort=False)}


def candidate_ciks(tenure: pd.DataFrame, roster_cik: Mapping[str, str], seeds: Mapping[str, str]) -> tuple[frozenset[str], dict[str, set[str]]]:
    """(all candidates, {ticker: CIKs seen under its symbol, its roster CIK and its Sharadar CIK})."""
    # every roster ticker starts from its roster CIK and its Sharadar `secfilings` CIK
    by_ticker: dict[str, set[str]] = {t: {roster_cik[t]} | ({seeds[t]} if seeds.get(t) else set()) for t in roster_cik}

    # add every issuer CIK that filed under a universe symbol
    seen = tenure[tenure["symbol"].astype(str).isin(set(roster_cik))]
    for symbol, cik in zip(seen["symbol"].astype(str), seen["issuer_cik"].astype(str), strict=False):
        by_ticker[str(symbol)].add(str(cik))
    return frozenset().union(*by_ticker.values()), by_ticker


def owner_sets(owner_pairs: pd.DataFrame, ciks: frozenset[str]) -> dict[str, set[str]]:
    """`{issuer_cik: {padded reporting-owner CIKs}}` for `ciks`, from the Form 3/4/5 (issuer_cik, owner_cik_raw) pairs."""
    # keep the pairs of the candidate issuers
    owners: dict[str, set[str]] = {cik: set() for cik in ciks}
    matched = owner_pairs[owner_pairs["issuer_cik"].isin(ciks)]
    owner_ciks = pad_cik_series(matched["owner_cik_raw"])

    # one owner set per issuer across every symbol it filed under
    for issuer, issuer_owner_ciks in owner_ciks.groupby(matched["issuer_cik"], sort=False):
        owners[str(issuer)].update(issuer_owner_ciks)
    logger.info(
        "universe_cik_symbol: owner sets for %d CIK(s) (%d distinct issuer-owner pair(s) matched); %d CIK(s) have no Form 345 owner at all",
        len(ciks),
        len(matched),
        sum(1 for v in owners.values() if not v),
    )
    return owners


def succession_pairs(profile: pd.DataFrame | None, by_ticker: Mapping[str, set[str]]) -> pd.DataFrame:
    """`[ticker, predecessor, successor, boundary, evidence]`: an 8-K12B on one candidate CIK and a Form 25 / 15 on another.

    The two filings must lie within `SWITCH_TOLERANCE_DAYS` of each other; the boundary is the 8-K12B date.
    """
    # nothing to pair without the profile
    columns = ["ticker", "predecessor", "successor", "boundary", "evidence"]
    if profile is None or profile.empty:
        return pd.DataFrame(columns=columns)

    # the dated succession and end forms of each candidate CIK
    successors = _profile_events(profile, "form_8k12b", SUCCESSOR_FORMS)
    ends = _profile_events(profile, "form_25", PREDECESSOR_END_FORMS)
    ends = pd.concat([ends, _profile_events(profile, "form_15", PREDECESSOR_END_FORMS)], ignore_index=True)
    members = pd.DataFrame([(ticker, cik) for ticker, ciks in by_ticker.items() for cik in sorted(ciks)], columns=["ticker", "cik"])

    # pair a successor and a predecessor of the same ticker filed within the tolerance
    succ = members.merge(successors, on="cik").rename(columns={"cik": "successor", "filed": "boundary", "form": "succ_form"})
    pred = members.merge(ends, on="cik").rename(columns={"cik": "predecessor", "filed": "end_filed", "form": "end_form"})
    pairs = succ.merge(pred, on="ticker")
    near = (pairs["end_filed"] - pairs["boundary"]).abs() <= pd.Timedelta(days=SWITCH_TOLERANCE_DAYS)
    pairs = pairs[near & pairs["predecessor"].ne(pairs["successor"])]

    # one pair per (ticker, predecessor, successor): the first succession filing and its nearest end form
    pairs = pairs.assign(gap=(pairs["end_filed"] - pairs["boundary"]).abs()).sort_values(["boundary", "gap"], kind="mergesort")
    pairs = pairs.drop_duplicates(["ticker", "predecessor", "successor"], keep="first")
    pairs["evidence"] = [
        f"{s} {sf} {b.date()} with {p} {ef} {e.date()} (sec_cik_profile)"
        for s, sf, b, p, ef, e in zip(
            pairs["successor"], pairs["succ_form"], pairs["boundary"], pairs["predecessor"], pairs["end_form"], pairs["end_filed"], strict=True
        )
    ]
    return pairs[columns].sort_values(["ticker", "boundary", "predecessor"], kind="mergesort", ignore_index=True)


def _profile_events(profile: pd.DataFrame, column: str, forms: frozenset[str]) -> pd.DataFrame:
    """`[cik, form, filed]` of the `forms` listed in one JSON column of `sec_cik_profile`."""
    rows = [
        (str(cik), str(event["form"]), event["filed"])
        for cik, text in zip(profile["cik"], profile[column], strict=True)
        if isinstance(text, str)
        for event in json.loads(text)
        if event.get("form") in forms and event.get("filed")
    ]
    events = pd.DataFrame(rows, columns=["cik", "form", "filed"])
    return events.assign(filed=pd.to_datetime(events["filed"]))


# Membership
def curated_membership(
    candidates: frozenset[str], roster_cik: Mapping[str, str], registrants: Mapping[str, Registrant], manual: Mapping[str, dict]
) -> Membership:
    """The union-find over the candidates after the two curated oracles: the register, then the manual entries."""
    # one singleton per candidate and roster CIK
    rosters = frozenset(roster_cik.values())
    parent = {cik: cik for cik in sorted(candidates | rosters)}
    membership = Membership(parent=parent, roster_of={cik: frozenset({cik}) for cik in rosters})

    # the register: every segment of a ticker's chain is one company
    for ticker, entry in sorted(registrants.items()):
        ciks = [segment.cik for segment in entry.segments]
        for other in ciks[1:]:
            _merge(membership, ciks[0], other, f"register[{ticker}]", None)
        for segment in entry.segments:
            membership.curated.add(segment.cik)
            _claim(membership, segment.cik, ("register", None, f"{ticker} {entry.kind}: {segment.evidence}"))

    # the manual entries: `same_entity` CIKs are one company, `own_entity` verdicts are recorded
    for key, entry in sorted(manual.items()):
        for cik in entry["same_entity"] + entry["own_entity"]:
            membership.parent.setdefault(cik, cik)
            membership.curated.add(cik)
            _claim(membership, cik, ("manual", None, f"{key}: {entry['evidence']}"))
        for other in entry["same_entity"][1:]:
            _merge(membership, entry["same_entity"][0], other, f"manual[{key}]", None)
    return membership


def join_successions(membership: Membership, successions: pd.DataFrame) -> None:
    """Oracle `sec_succession`: join each predecessor to its successor; the joining non-roster CIKs record the verdict."""
    # in boundary order, each pair not already one company
    rosters = frozenset().union(*membership.roster_of.values())
    for ticker, pred, succ, boundary, evidence in successions.sort_values("boundary", kind="mergesort").itertuples(index=False):
        if _root(membership.parent, pred) == _root(membership.parent, succ) or not _merge(membership, pred, succ, f"sec_succession[{ticker}]", None):
            continue

        # the joining CIKs that are not a roster CIK record the verdict
        membership.joins["sec_succession"] += 1
        why = f"{ticker} successor registrant from {pd.Timestamp(boundary).date()}: {evidence}"
        for cik in sorted({pred, succ} - rosters):
            _claim(membership, cik, ("sec_succession", None, why))


def join_symbol_handoffs(
    membership: Membership, evidence: pd.DataFrame, index: Mapping[tuple[str, str], pd.DataFrame], roster_cik: Mapping[str, str]
) -> None:
    """Oracle `symbol_handoff`: join a CIK that handed one symbol to (or took it from) a universe-company CIK, in both sources."""
    # the symbols a universe company typed in both sources, and every CIK holding them in both
    parent = membership.parent
    roots = {_root(parent, cik) for cik in roster_cik.values()}
    universe = {cik for cik in list(parent) if _root(parent, cik) in roots}
    filed = evidence[evidence["source"].isin(EVIDENCE_SOURCES)]
    both = filed.groupby(["symbol", "cik"])["source"].nunique()
    both = both[both == len(EVIDENCE_SOURCES)].reset_index()[["symbol", "cik"]]
    symbols = sorted(set(both.loc[both["cik"].isin(universe), "symbol"]))
    holders = both[both["symbol"].isin(symbols)].groupby("symbol")["cik"].agg(sorted).to_dict()
    pairs = [
        (symbol, pred, succ)
        for symbol in symbols
        for pred, succ in combinations(sorted(holders.get(symbol, []), key=lambda cik, s=symbol: (_first_of(index, cik, s), cik)), 2)
    ]

    # each handoff both sources date joins the newcomer to the universe company
    for symbol, pred, succ in pairs:
        in_universe = (_root(parent, pred) in roots, _root(parent, succ) in roots)
        if not any(in_universe) or _root(parent, pred) == _root(parent, succ):
            continue
        switch = cik_switch(index, pred, succ, frozenset({symbol}))
        if switch.boundary is None or not _merge(membership, pred, succ, f"symbol_handoff[{symbol}]", None):
            continue
        membership.joins["symbol_handoff"] += 1
        _claim(membership, succ if in_universe[0] else pred, ("symbol_handoff", None, f"{symbol} handed {pred} -> {succ}: {switch.detail}"))
        roots = {_root(parent, cik) for cik in roster_cik.values()}


def join_owner_overlaps(
    membership: Membership, by_ticker: Mapping[str, set[str]], roster_cik: Mapping[str, str], owners: Mapping[str, set[str]]
) -> None:
    """Oracle `owner_overlap`: score each non-roster candidate against its ticker's roster CIK and merge `same`; grey is kept apart."""
    # every (ticker, candidate, roster CIK) pair, curated and already-joined CIKs counted but not re-opened
    pairs = [(ticker, cik, roster_cik[ticker]) for ticker in sorted(by_ticker) for cik in sorted(by_ticker[ticker]) if cik != roster_cik[ticker]]
    for ticker, cik, home in pairs:
        shared, jaccard = score_overlap(owners.get(cik, set()), owners.get(home, set()))
        verdict = classify_overlap(shared, jaccard)
        if cik in membership.curated or _root(membership.parent, cik) == _root(membership.parent, home):
            membership.overlap_verdicts[f"{verdict} (pre-decided)"] += 1
            continue

        # a same verdict merges; a grey one is listed for curation
        membership.overlap_verdicts[verdict] += 1
        if verdict == "grey":
            membership.grey.append((ticker, cik, home, shared, jaccard))
        elif verdict == "same" and _merge(membership, cik, home, f"owner_overlap[{ticker}]", jaccard):
            _claim(membership, cik, ("owner_overlap", jaccard, f"{ticker}: {shared} reporting owner(s) shared with {home}, jaccard {jaccard:.3f}"))


def score_overlap(a: set[str], b: set[str]) -> tuple[int, float]:
    """(shared owners, jaccard). Empty on either side -> (0, 0.0), i.e. no evidence."""
    if not a or not b:
        return 0, 0.0
    shared = len(a & b)
    return shared, shared / len(a | b)


def classify_overlap(shared: int, jaccard: float) -> str:
    """`unrelated` | `same` | `grey` per the `OVERLAP_*` thresholds."""
    if shared == 0:
        return "unrelated"
    if jaccard >= OVERLAP_JACCARD_SAME or shared >= OVERLAP_SHARED_SAME:
        return "same"
    return "grey"


def _root(parent: dict[str, str], cik: str) -> str:
    """The root of `cik` in the union-find (a new CIK is its own root)."""
    parent.setdefault(cik, cik)
    root = cik
    while parent[root] != root:
        root = parent[root]
    return root


def _merge(membership: Membership, a: str, b: str, source: str, confidence: float | None) -> bool:
    """Join the groups of `a` and `b` under the older root; False (and recorded) when both groups hold a roster CIK."""
    # the two roots
    parent = membership.parent
    root_a, root_b = parent.setdefault(a, a), parent.setdefault(b, b)
    while parent[root_a] != root_a:
        root_a = parent[root_a]
    while parent[root_b] != root_b:
        root_b = parent[root_b]
    if root_a == root_b:
        return True

    # refuse a merge that would put two universe tickers in one company
    rosters = membership.roster_of.get(root_a, frozenset()) | membership.roster_of.get(root_b, frozenset())
    if len(rosters) > 1:
        membership.blocked.append((a, b, source, confidence))
        return False

    # the oldest CIK roots the group
    older, newer = sorted((root_a, root_b))
    parent[newer] = older
    membership.roster_of.pop(newer, None)
    if rosters:
        membership.roster_of[older] = rosters
    return True


def _claim(membership: Membership, cik: str, verdict: tuple[str, float | None, str]) -> None:
    """Record `verdict` (source, confidence, evidence) for `cik` unless a higher-priority oracle already spoke."""
    prior = membership.verdicts.get(cik)
    if prior is None or SOURCE_PRIORITY.index(verdict[0]) < SOURCE_PRIORITY.index(prior[0]):
        membership.verdicts[cik] = verdict


def _first_of(index: Mapping[tuple[str, str], pd.DataFrame], cik: str, symbol: str) -> pd.Timestamp:
    """The first observation of `symbol` under `cik` across both sources."""
    firsts = [rows.loc[rows["symbol"].eq(symbol), "first"].min() for source in EVIDENCE_SOURCES if (rows := index.get((cik, source))) is not None]
    return min(first for first in firsts if pd.notna(first))


# Switches between two CIKs
def cik_switch(index: Mapping[tuple[str, str], pd.DataFrame], pred: str, succ: str, symbols: frozenset[str] | None) -> Switch:
    """Both sources must place `pred`'s last and `succ`'s first filing within `SWITCH_TOLERANCE_DAYS`.

    `symbols` limits the bounds to those symbols (a handoff); None reads every symbol of each CIK (an automatic window).
    """
    # each source: a shared symbol, a closed predecessor and a successor starting at its end
    tolerance = pd.Timedelta(days=SWITCH_TOLERANCE_DAYS)
    lasts: list[pd.Timestamp] = []
    firsts: list[pd.Timestamp] = []
    notes: list[str] = []
    for source in EVIDENCE_SOURCES:
        p_rows, s_rows = index.get((pred, source)), index.get((succ, source))
        if p_rows is None or s_rows is None:
            return Switch(None, None, f"no {source} evidence on both CIKs")
        shared = set(p_rows["symbol"]) & set(s_rows["symbol"])
        if symbols is not None:
            shared &= symbols
            p_rows, s_rows = p_rows[p_rows["symbol"].isin(symbols)], s_rows[s_rows["symbol"].isin(symbols)]
        if not shared:
            return Switch(None, None, f"no shared {source} symbol")
        if p_rows["end"].isna().any():
            return Switch(None, None, f"{source}: predecessor still observed")
        p_first, p_last, s_first = p_rows["first"].min(), p_rows["end"].max() - pd.Timedelta(days=1), s_rows["first"].min()
        if not p_first < s_first:
            return Switch(None, None, f"{source}: successor observed from {s_first.date()}, before the predecessor's first {p_first.date()}")
        if abs(s_first - p_last) > tolerance:
            return Switch(None, None, f"{source}: predecessor last {p_last.date()} vs successor first {s_first.date()}")
        lasts.append(p_last)
        firsts.append(s_first)
        notes.append(f"{source} {pred} last {p_last.date()} -> {succ} first {s_first.date()} ({'/'.join(sorted(shared))})")

    # the two sources must agree on the date
    if max(firsts) - min(firsts) > tolerance:
        return Switch(None, None, "sources disagree: " + "; ".join(notes))
    boundary, last = min(firsts), max(lasts)
    return Switch(boundary, (min(last, boundary), max(last, boundary)), "; ".join(notes))


def automatic_chains(
    index: Mapping[tuple[str, str], pd.DataFrame], entities: Entities, roster_cik: Mapping[str, str]
) -> dict[str, tuple[list[tuple[str, pd.Timestamp, pd.Timestamp | None]] | None, str]]:
    """`{company_id: (dated chain oldest first, or None, and the note)}` for every multi-CIK universe company."""
    # order each company's observed CIKs by first filing, then read the switch between neighbours
    chains: dict[str, tuple[list[tuple[str, pd.Timestamp, pd.Timestamp | None]] | None, str]] = {}
    for entity, ticker in sorted(entities.ticker_by_entity.items()):
        ciks = entities.groups.get(entity, {roster_cik[ticker]})
        if len(ciks) < 2:
            continue
        order, note = _chain_order(index, ciks, roster_cik[ticker])
        switches = [cik_switch(index, pred, succ, None) for pred, succ in pairwise(order)] if order else []
        chains[entity] = _chain_windows(order, switches, note)
    return chains


def _chain_order(index: Mapping[tuple[str, str], pd.DataFrame], ciks: Iterable[str], newest: str) -> tuple[list[str], str]:
    """The CIKs with filed evidence ordered by first observation, or [] and the reason to abstain."""
    observed: dict[str, pd.Timestamp] = {}
    for cik in ciks:
        firsts = [rows["first"].min() for source in EVIDENCE_SOURCES if (rows := index.get((cik, source))) is not None]
        if firsts:
            observed[cik] = min(firsts)
    order = sorted(observed, key=lambda cik: (observed[cik], cik))
    if len(order) < 2:
        return [], "fewer than two CIKs with symbol evidence"
    return (order, "") if order[-1] == newest else ([], f"the newest CIK by evidence is {order[-1]}, not {newest}")


def _chain_windows(order: list[str], switches: list[Switch], note: str) -> tuple[list[tuple[str, pd.Timestamp, pd.Timestamp | None]] | None, str]:
    """The dated chain from consecutive switches, or None and the first undecided switch."""
    if not order:
        return None, note
    for (pred, succ), switch in zip(pairwise(order), switches, strict=True):
        if switch.boundary is None:
            return None, f"{pred}->{succ}: {switch.detail}"
    boundaries = [switch.boundary for switch in switches if switch.boundary is not None]
    starts, ends = [SENTINEL_START, *boundaries], [*boundaries, None]
    return list(zip(order, starts, ends, strict=True)), "; ".join(switch.detail for switch in switches)


# Companies
def company_ids(membership: Membership, anchors: Mapping[str, str], roster_cik: Mapping[str, str]) -> Entities:
    """Each group's `company_id`: its universe ticker's anchored id, else `E` + its oldest CIK; never re-keyed."""
    # the groups of the union-find
    groups: dict[str, set[str]] = {}
    for cik in list(membership.parent):
        groups.setdefault(_root(membership.parent, cik), set()).add(cik)
    ticker_of_root = {_root(membership.parent, cik): ticker for ticker, cik in roster_cik.items()}

    # the frozen id of an anchored ticker, the minted id of every other group
    ids = {root: anchors.get(ticker_of_root.get(root, ""), "E" + min(members)) for root, members in groups.items()}
    taken = Counter(ids.values())
    clashes = sorted(entity for entity, n in taken.items() if n > 1)
    if clashes:
        raise CompanyIdCollisionError(f"identity: company id(s) {clashes} name two groups; a frozen id collides with a minted one")

    # every member CIK, the members of each company and its universe ticker
    entity_of = {cik: ids[root] for root, members in groups.items() for cik in members}
    by_entity = {ids[root]: members for root, members in groups.items()}
    ticker_by_entity = {entity_of.get(cik, f"E{cik}"): ticker for ticker, cik in roster_cik.items()}
    return Entities(entity_of=entity_of, groups=by_entity, ticker_by_entity=ticker_by_entity)


def check_d19(tenure: pd.DataFrame, roster_cik: Mapping[str, str], entity_of: Mapping[str, str], allowlist: Mapping[str, str]) -> None:
    """D19: the dominant `form345`/`manual` holder of each roster ticker must be the roster CIK's company, unless allow-listed."""
    # the universe symbols' tenure rows
    frame = pd.DataFrame(
        {
            "symbol": tenure["symbol"].map(normalise_ticker),
            "cik": pad_cik_series(tenure["issuer_cik"]),
            "open": tenure["valid_to"].isna(),
            "n": pd.to_numeric(tenure["n_filings"], errors="coerce").fillna(0),
            "manual": tenure["source"].astype(str).eq(MANUAL_SOURCE) if "source" in tenure.columns else False,
        }
    )
    by_symbol = {symbol: group for symbol, group in frame[frame["symbol"].isin(set(roster_cik))].groupby("symbol", sort=False)}

    # each roster ticker: the dominant holder's company against the roster CIK's
    unexplained = []
    for ticker, cik in sorted(roster_cik.items()):
        dominant = _dominant_cik(by_symbol.get(ticker))
        home = entity_of.get(cik, f"E{cik}")
        if (dominant is None or entity_of.get(dominant, f"E{dominant}") != home) and ticker not in allowlist:
            unexplained.append(f"{ticker}: roster {cik} -> {home} but tenure -> {dominant or 'NO TENURE'}")
    if unexplained:
        raise UniverseEntityDisagreementError(
            f"identity: {len(unexplained)} universe ticker(s) whose roster CIK and whose filings name different entities -- "
            + "; ".join(unexplained)
            + ". Fix the roster CIK, or add the ticker to `_d19_allowlist` in entity_lineage_manual.json WITH the evidence."
        )


def _dominant_cik(rows: pd.DataFrame | None) -> str | None:
    """The holder with the most filings among the open manual rows, else the open rows, else the manual rows, else all."""
    if rows is None:
        return None
    manual_open, opened = rows[rows["manual"] & rows["open"]], rows[rows["open"]]
    pool = manual_open if not manual_open.empty else opened if not opened.empty else rows[rows["manual"]] if rows["manual"].any() else rows
    return str(pool.sort_values(["n", "cik"], kind="mergesort").iloc[-1]["cik"])


# Windows and rows
def cik_windows(
    entities: Entities,
    roster_cik: Mapping[str, str],
    registrants: Mapping[str, Registrant],
    chains: Mapping[str, tuple[list[tuple[str, pd.Timestamp, pd.Timestamp | None]] | None, str]],
    successions: pd.DataFrame,
    curated: set[str],
) -> tuple[dict[str, Window], list[dict[str, Any]]]:
    """`{cik: Window}` of every consolidating CIK, and one `multi_cik_no_window` backlog row per undecided company."""
    # the register entry each CIK belongs to
    register_by_cik = {segment.cik: ticker for ticker, entry in registrants.items() for segment in entry.segments}
    windows: dict[str, Window] = {}
    backlog: list[dict[str, Any]] = []
    for entity, ciks in sorted(entities.groups.items()):
        ticker = entities.ticker_by_entity.get(entity)
        entry = _register_entry(registrants, register_by_cik, ticker, ciks)

        # the register's segments, else (universe companies only) the SEC succession, the automatic chain or the roster CIK
        if entry is not None:
            windows.update(_register_windows(entry, ciks))
            continue
        if ticker is None:
            continue
        found, kind, note = _derived_windows(ciks, roster_cik[ticker], successions[successions["ticker"].eq(ticker)], chains.get(entity))
        windows.update(found)
        uncurated = sorted(cik for cik in ciks - {roster_cik[ticker]} if cik not in curated)
        if kind == ROSTER_SOURCE and uncurated:
            backlog.append(_multi_cik_backlog(ticker, entity, uncurated, roster_cik[ticker], chains.get(entity), note))
    return windows, backlog


def _register_entry(
    registrants: Mapping[str, Registrant], register_by_cik: Mapping[str, str], ticker: str | None, ciks: set[str]
) -> Registrant | None:
    """The register entry of the company's ticker, else the one entry its CIKs belong to."""
    entry = registrants.get(ticker) if ticker is not None else None
    if entry is not None:
        return entry
    tickers = {register_by_cik[cik] for cik in ciks if cik in register_by_cik}
    return registrants.get(sorted(tickers)[0]) if len(tickers) == 1 else None


def _register_windows(entry: Registrant, ciks: set[str]) -> dict[str, Window]:
    """One curated window per register segment whose CIK is in the company."""
    return {
        segment.cik: (segment.valid_from or SENTINEL_START, segment.valid_to, REGISTER_SOURCE, "curated", f"register window ({entry.ticker})")
        for segment in entry.segments
        if segment.cik in ciks
    }


def _derived_windows(
    ciks: set[str],
    home: str,
    successions: pd.DataFrame,
    chain: tuple[list[tuple[str, pd.Timestamp, pd.Timestamp | None]] | None, str] | None,
) -> tuple[dict[str, Window], str, str]:
    """The windows of an unregistered company: an SEC succession chain ending at `home`, else the automatic chain, else `home` alone."""
    # an SEC succession chain over the company's CIKs, oldest first, ending at the roster CIK
    links = successions[successions["predecessor"].isin(ciks) & successions["successor"].isin(ciks)].sort_values("boundary", kind="mergesort")
    order = [*links["predecessor"].iloc[:1], *links["successor"]]
    linked = len(order) > 1 and order[-1] == home and len(set(order)) == len(order) and list(links["predecessor"]) == order[:-1]
    if linked:
        starts, ends = [SENTINEL_START, *links["boundary"]], [*links["boundary"], None]
        notes = list(links["evidence"])
        found = {
            cik: (start, end, SEC_PROFILE_SOURCE, "corroborated", f"sec succession: {'; '.join(notes)}")
            for cik, start, end in zip(order, starts, ends, strict=True)
        }
        return found, SEC_PROFILE_SOURCE, ""

    # the automatic chain both filed sources agree on
    roster_only = {home: (SENTINEL_START, None, ROSTER_SOURCE, "single_source", "roster CIK window")}
    if len(ciks) == 1 or chain is None:
        return roster_only, ROSTER_SOURCE, ""
    dated, note = chain
    if dated is not None and AUTO_WINDOWS_ENABLED:
        sources = ",".join(sorted(EVIDENCE_SOURCES))
        return {cik: (start, end, sources, "corroborated", f"automatic window: {note}") for cik, start, end in dated}, "automatic", note
    return roster_only, ROSTER_SOURCE, note


def _multi_cik_backlog(
    ticker: str,
    entity: str,
    uncurated: list[str],
    home: str,
    chain: tuple[list[tuple[str, pd.Timestamp, pd.Timestamp | None]] | None, str] | None,
    note: str,
) -> dict[str, Any]:
    """The backlog row of a multi-CIK company left on its roster CIK alone."""
    dated = chain[0] if chain is not None else None
    suggestion = "; suggested chain " + ", ".join(f"{cik} from {start.date()}" for cik, start, _ in dated) if dated is not None else ""
    reason = "automatic windows disabled" if dated is not None else f"abstained: {note}"
    detail = f"{reason}; listing the roster CIK {home} only{suggestion}"
    return {
        "kind": "multi_cik_no_window",
        "canonical_ticker": ticker,
        "company_id": entity,
        "cik": ",".join(uncurated),
        "symbol": "",
        "detail": detail,
    }


def cik_rows(entities: Entities, membership: Membership, windows: Mapping[str, Window], evidence: pd.DataFrame) -> pd.DataFrame:
    """`cik_window` / `cik_event` rows of every universe company and every curated group (no `scope_changed_at`)."""
    # filed observations per CIK
    counts = evidence[evidence["source"].isin(EVIDENCE_SOURCES)].groupby("cik")["n"].sum().to_dict()

    # one row per member CIK: its window, else an event-only row under the oracle that joined it
    rows: list[dict[str, Any]] = []
    for entity, ciks in sorted(entities.groups.items()):
        ticker = entities.ticker_by_entity.get(entity)
        if ticker is None and not (ciks & membership.curated):
            continue
        rows += [
            _cik_row(entity, ticker, cik, membership.verdicts.get(cik, _ROSTER_VERDICT), windows.get(cik), int(counts.get(cik, 0)))
            for cik in sorted(ciks)
        ]
    return pd.DataFrame(rows, columns=list(TABLE_COLUMNS[:-1]))


def _cik_row(entity: str, ticker: str | None, cik: str, verdict: tuple[str, float | None, str], window: Window | None, n: int) -> dict[str, Any]:
    """One CIK row: dated by its window, or an open event-only row."""
    oracle, confidence, why = verdict
    start, end, sources, status, note = window if window is not None else (SENTINEL_START, None, _ORACLE_SOURCES[oracle], "", "")
    return {
        "company_id": entity,
        "canonical_ticker": ticker,
        "cik": cik,
        "role": ROLE_WINDOW if window is not None else ROLE_EVENT,
        "symbol": "",
        "valid_from": start,
        "valid_to": end,
        "status": status or _ORACLE_STATUS.get(oracle, "single_source"),
        "sources": sources,
        "oracle": oracle,
        "confidence": confidence,
        "n_observations": n,
        "evidence": "; ".join(part for part in (why, note) if part),
    }


def grey_backlog(membership: Membership, entities: Entities) -> list[dict[str, Any]]:
    """One backlog row per grey-band pair of a universe company, which stays out of it."""
    rows = []
    for ticker, cik, home, shared, jaccard in membership.grey:
        entity = entities.entity_of.get(home, f"E{home}")
        if entities.ticker_by_entity.get(entity) == ticker:
            detail = f"owner overlap with {home}: shared={shared}, jaccard={jaccard:.3f}; excluded until curated"
            rows.append({"kind": "grey_band", "canonical_ticker": ticker, "company_id": entity, "cik": cik, "symbol": ticker, "detail": detail})
    return rows


def blocked_frame(membership: Membership, roster_cik: Mapping[str, str]) -> pd.DataFrame:
    """The refused merges, each side named by the universe tickers its group holds."""
    tickers_of = {}
    for ticker, cik in roster_cik.items():
        tickers_of.setdefault(_root(membership.parent, cik), []).append(ticker)
    rows = [
        (
            a,
            b,
            source,
            conf,
            ",".join(sorted(tickers_of.get(_root(membership.parent, a), []))),
            ",".join(sorted(tickers_of.get(_root(membership.parent, b), []))),
        )
        for a, b, source, conf in membership.blocked
    ]
    return pd.DataFrame(rows, columns=list(BLOCKED_COLUMNS))


def sharadar_outside(sharadar: Mapping[str, str], entities: Entities, roster_cik: Mapping[str, str]) -> list[str]:
    """The universe tickers whose Sharadar `secfilings` CIK is not one of their company's CIKs (AC-004)."""
    return sorted(t for t, cik in sharadar.items() if t in roster_cik and entities.entity_of.get(cik) != entities.entity_of.get(roster_cik[t]))


def log_cik_history(history: CikHistory, membership: Membership, n_candidates: int, outside: list[str]) -> None:
    """One INFO summary; the refused merges and the CIK curation backlog in detail at DEBUG (PD-14: their items reach the
    publish review as `refused_merge`, `grey_band` and the extra-CIK kinds)."""
    # the summary, with the Sharadar CIKs outside their company and the backlog counts
    rows = history.rows
    logger.info(
        "universe_cik_symbol: %d candidate CIK(s) -> %d CIK row(s) over %d company(ies); roles %s; owner-overlap verdicts %s; joins %s; "
        "automatic windows %s; %d merge(s) BLOCKED as two-universe-tickers; %d Sharadar CIK(s) outside their company %s; "
        "CIK curation backlog %s (not applied)",
        n_candidates,
        len(rows),
        rows["company_id"].nunique(),
        _plain_counts(rows["role"]),
        dict(membership.overlap_verdicts),
        dict(membership.joins),
        "enabled" if AUTO_WINDOWS_ENABLED else "disabled",
        len(history.blocked),
        len(outside),
        outside,
        _plain_counts(history.backlog["kind"]),
    )

    # what the build refused or left undecided, in detail
    if not history.blocked.empty:
        logger.debug("universe_cik_symbol: refused merges:\n%s", history.blocked.to_string(index=False))
    if not history.backlog.empty:
        logger.debug("universe_cik_symbol: CIK curation backlog:\n%s", history.backlog.to_string(index=False))


def _plain_counts(values: pd.Series) -> dict[str, int]:
    """`{value: count}` sorted by value, plain integers."""
    return {str(value): int(count) for value, count in values.value_counts().sort_index().items()}


def cik_history_from_rows(rows: pd.DataFrame, evidence: pd.DataFrame, activity: pd.DataFrame) -> CikHistory:
    """The CIK history as stored in `universe_cik_symbol` (its CIK rows), for a symbol history run on its own."""
    # the stored CIK rows, typed
    cik = rows[rows["role"].ne(ROLE_SYMBOL)].copy()
    cik["cik"] = pad_cik_series(cik["cik"])
    cik["valid_from"] = pd.to_datetime(cik["valid_from"])
    cik["valid_to"] = pd.to_datetime(cik["valid_to"])

    # what the symbol history reads: companies, tickers, window ends and oracles
    entity_of = dict(zip(cik["cik"], cik["company_id"].astype(str), strict=True))
    tickers = cik[cik["canonical_ticker"].notna()]
    windows = cik[cik["role"].eq(ROLE_WINDOW)]
    return CikHistory(
        rows=cik[list(TABLE_COLUMNS[:-1])].reset_index(drop=True),
        entity_of=entity_of,
        ticker_by_entity=dict(zip(tickers["company_id"].astype(str), tickers["canonical_ticker"].astype(str), strict=True)),
        window_end={c: (None if pd.isna(end) else pd.Timestamp(end)) for c, end in zip(windows["cik"], windows["valid_to"], strict=True)},
        oracle_of=dict(zip(cik["cik"], cik["oracle"].astype(str), strict=True)),
        backlog=pd.DataFrame(columns=list(BACKLOG_COLUMNS)),
        blocked=pd.DataFrame(columns=list(BLOCKED_COLUMNS)),
        evidence=evidence,
        activity=activity,
    )
