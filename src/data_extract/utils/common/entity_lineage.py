"""Ticker identity: which CIKs and symbols belong to one economic company, and when (`entity_lineage`).

`symbol_tenure` holds the raw evidence; this table is the dated verdict. Membership oracles, highest
priority first: `register` (`registrant_cutover.json`), `manual` (`entity_lineage_manual.json`),
`symbol_handoff` (one symbol passing between two CIKs, seen on Forms 3/4/5 and on cover pages),
`owner_overlap` (shared Form 3/4/5 reporting owners) and `roster`. The union-find refuses any merge
that would put two universe tickers in one entity.

Rows by `role`: `cik_window` (consolidating filings of that CIK belong to the entity over
`[valid_from, valid_to)`), `cik_event` (event forms only) and `symbol` (a dated symbol interval with a
`status`). Invariants: an open start is stored as `SENTINEL_START`, an open end as NULL; a `symbol`
row is current when its `valid_to` is NULL; one CIK belongs to one entity. Grey-band scores,
conflicts, uncorroborated extra CIKs and older-CIK rekeys are excluded, logged and backlogged; they
never stop the build.
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

from src.context import Context
from src.data_extract.utils.common.config_paths import resolve_config_dir
from src.data_extract.utils.common.incremental import matches_stored
from src.data_extract.utils.common.registrant import Registrant, load_registrants
from src.data_extract.utils.common.symbol_tenure import DEI_SOURCE, collapse_dei_periods, load_manual_symbol_tenure
from src.data_store.schema import Tables
from src.utils.identity_flags import cik_activity, identity_flags, log_identity_flags
from src.utils.string import normalise_ticker, pad_cik, pad_cik_series

logger = logging.getLogger(__name__)

#: `configs/sec/entity_lineage_manual.json`.
MANUAL_CONFIG_SUBDIR = "sec"
MANUAL_CONFIG_FILENAME = "entity_lineage_manual.json"

#: Oracle 4: shared == 0 -> unrelated; jaccard >= 0.05 OR shared >= 5 -> same entity; anything
#: between is a grey band the build excludes and backlogs instead of deciding.
OVERLAP_JACCARD_SAME = 0.05
OVERLAP_SHARED_SAME = 5

SOURCE_PRIORITY = ("register", "manual", "symbol_handoff", "owner_overlap", "roster")

#: The `sp500_tickers` columns the identity layer reads (`roster_cik_map`).
ROSTER_COLUMNS = ("ticker", "cik")

#: Key in `entity_lineage_manual.json` holding the D19 cross-check's exceptions; underscore-prefixed so the lineage loader skips it.
D19_ALLOWLIST_KEY = "_d19_allowlist"

#: Stored value of an open start: `valid_from` is a PK column and cannot be NULL.
SENTINEL_START = pd.Timestamp("1900-01-01")

ROLE_WINDOW = "cik_window"
ROLE_EVENT = "cik_event"
ROLE_SYMBOL = "symbol"

FORM345_SOURCE = "form345"
MANUAL_SOURCE = "manual"
ROSTER_SOURCE = "roster"
REGISTER_SOURCE = "register"
#: The two filed evidence sources a symbol handoff and an automatic window need, both on both CIKs.
EVIDENCE_SOURCES = (FORM345_SOURCE, DEI_SOURCE)

#: Two entities holding one symbol closer than this (or overlapping) is a reuse conflict; exchanges allow reuse after 90 days.
REUSE_GAP_DAYS = 90
#: A single-source interval with at most this many filings, next to another entity's interval, is noise.
NOISE_MAX_OBSERVATIONS = 5
#: A `dei` interval whose last filing is this close to the latest `dei` filing is still current.
CURRENT_RECENCY_DAYS = 120
#: A CIK switch: the predecessor's last and the successor's first filing within this many days, in each source.
SWITCH_TOLERANCE_DAYS = 31
#: D3 automatic CIK windows. Enabled only while the register reproduction shows zero contradictions.
AUTO_WINDOWS_ENABLED = False

#: Column order of `entity_lineage`.
TABLE_COLUMNS = (
    "entity_id",
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
BACKLOG_COLUMNS = ("kind", "canonical_ticker", "entity_id", "cik", "symbol", "detail")
#: Older-CIK exclusion passes before the build gives up excluding (each pass removes at least one CIK).
_MAX_REKEY_PASSES = 10


class IdentityError(ValueError):
    """Base of every identity failure, so one `except` covers the whole layer."""


class CikInTwoEntitiesError(IdentityError):
    """One CIK carries two `entity_id`s; asserted by the build, since the PK no longer guarantees it."""


class UniverseEntityDisagreementError(IdentityError):
    """D19: the roster CIK and `symbol_tenure` name different entities for one ticker, with no D19 allow-list entry."""


class TwoUniverseTickersOneEntityError(ValueError):
    """A merge would put two universe tickers in one entity; the fix is a curated row, never a wider merge."""


class ManualTenureEntityError(ValueError):
    """A manual ticker interval names a CIK outside its canonical ticker's entity."""


@dataclass
class _Union:
    """Union-find over CIKs, refusing any merge that joins two roster CIKs or touches an excluded CIK."""

    roster_ciks: frozenset[str]
    excluded: frozenset[str] = frozenset()
    parent: dict[str, str] = field(default_factory=dict)
    blocked: list[tuple[str, str, str, float | None]] = field(default_factory=list)

    def find(self, cik: str) -> str:
        self.parent.setdefault(cik, cik)
        root = cik
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[cik] != root:  # path compression
            self.parent[cik], cik = root, self.parent[cik]
        return root

    def roster_members(self, cik: str) -> set[str]:
        root = self.find(cik)
        return {c for c in self.parent if c in self.roster_ciks and self.find(c) == root}

    def union(self, a: str, b: str, *, source: str, confidence: float | None = None) -> bool:
        """Merge `a` and `b`; returns False (and records a two-roster refusal) if the merge is refused."""
        if a in self.excluded or b in self.excluded:
            return False
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return True
        if len(self.roster_members(a) | self.roster_members(b)) > 1:
            self.blocked.append((a, b, source, confidence))
            return False
        # The oldest CIK roots the group, so `entity_id` is stable under merge order.
        older, newer = sorted((root_a, root_b))
        self.parent[newer] = older
        return True

    def groups(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for cik in self.parent:
            out.setdefault(self.find(cik), set()).add(cik)
        return out


def entity_id_for(ciks: set[str]) -> str:
    """`"E" + the oldest (numerically smallest) padded CIK in the group`.

    A natural key with no allocation state; a group that later gains an older CIK shifts its id,
    which `detect_older_cik_rekeys` guards.
    """
    return "E" + min(ciks)


def entity_or_singleton(entity_by_cik: Mapping[str, str], cik: str) -> str:
    """The stored entity of a padded CIK; a CIK with no stored row is its own entity `E{cik}`."""
    return entity_by_cik.get(cik, f"E{cik}")


def entity_by_cik_map(lineage: pd.DataFrame) -> dict[str, str]:
    """`{padded cik: entity_id}` from `entity_lineage` rows (several rows per CIK share one entity)."""
    return dict(zip(pad_cik_series(lineage["cik"]), lineage["entity_id"].astype(str), strict=False))


def roster_cik_map(roster: pd.DataFrame) -> dict[str, str]:
    """`{universe ticker: padded roster CIK}` from `sp500_tickers`, skipping a row with no CIK."""
    ciks = pad_cik_series(roster["cik"])
    has_cik = ciks.ne("")
    return dict(zip(roster.loc[has_cik, "ticker"].map(normalise_ticker), ciks[has_cik], strict=False))


def check_one_entity_per_cik(lineage: pd.DataFrame) -> None:
    """Raise `CikInTwoEntitiesError` when one CIK carries two entity_ids, which would make `entity_of` order-dependent."""
    per_cik = pd.DataFrame({"cik": pad_cik_series(lineage["cik"]), "entity_id": lineage["entity_id"].astype(str)}).drop_duplicates()
    clashes = per_cik[per_cik.duplicated("cik", keep=False)]
    if not clashes.empty:
        raise CikInTwoEntitiesError(
            f"identity: {clashes['cik'].nunique()} CIK(s) carry two entity_ids in "
            f"entity_lineage -- {clashes.sort_values(['cik', 'entity_id']).to_dict('records')}. One CIK "
            "belongs to one entity; two would make `entity_of` order-dependent."
        )


@dataclass
class _Provenance:
    """The highest-priority oracle verdict per CIK, and every CIK a curated layer spoke for."""

    verdicts: dict[str, tuple[str, float | None, str]] = field(default_factory=dict)
    curated: set[str] = field(default_factory=set)

    def claim(self, cik: str, source: str, confidence: float | None, evidence: str) -> None:
        """Record `source`'s verdict for `cik` unless a higher-priority oracle already spoke."""
        prior = self.verdicts.get(cik)
        if prior is None or SOURCE_PRIORITY.index(source) < SOURCE_PRIORITY.index(prior[0]):
            self.verdicts[cik] = (source, confidence, evidence)


def _manual_blob(config_dir: str | None) -> dict:
    """`configs/sec/entity_lineage_manual.json` as raw JSON; `{}` when absent."""
    path = Path(resolve_config_dir(config_dir)) / MANUAL_CONFIG_SUBDIR / MANUAL_CONFIG_FILENAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_d19_allowlist(config_dir: str | None = None) -> dict[str, str]:
    """`{ticker: why this ticker's roster CIK may disagree with symbol_tenure}` for the D19 assertion.

    An entry clears a disagreement; any unlisted disagreement must raise (a roster CIK pointing at the wrong company).
    """
    allow = _manual_blob(config_dir).get(D19_ALLOWLIST_KEY, {})
    return {t: str(why) for t, why in allow.items() if not t.startswith("_")}


def load_manual_lineage(config_dir: str | None = None) -> dict[str, dict]:
    """`configs/sec/entity_lineage_manual.json` -> `{key: entry}`; `{}` when absent.

    `"same_entity"` (>= 2 CIKs that are one company) and/or `"own_entity"` (CIKs that are their
    own entity, not the ticker they filed under); every entry needs non-empty `evidence`.
    """
    blob = _manual_blob(config_dir)
    out: dict[str, dict] = {}
    for key, entry in blob.items():
        if key.startswith("_"):
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


# Oracle 4 -- reporting-owner overlap
def derive_owner_sets(owner_pairs: pd.DataFrame, ciks: frozenset[str]) -> dict[str, set[str]]:
    """`{issuer_cik: {padded reporting owner CIKs}}` for `ciks`, from Form 345 (issuer_cik, owner_cik_raw) pairs.

    Keyed on the issuer CIK across every symbol it filed under, not on (symbol, cik).
    """
    owners: dict[str, set[str]] = {cik: set() for cik in ciks}
    df_matched = owner_pairs[owner_pairs["issuer_cik"].isin(ciks)]
    owner_ciks = pad_cik_series(df_matched["owner_cik_raw"])
    for issuer, issuer_owner_ciks in owner_ciks.groupby(df_matched["issuer_cik"], sort=False):
        owners[str(issuer)].update(issuer_owner_ciks)
    logger.info(
        "entity_lineage: owner sets for %d CIK(s) (%d distinct issuer-owner pair(s) matched); %d CIK(s) have no Form 345 owner at all",
        len(ciks),
        len(df_matched),
        sum(1 for v in owners.values() if not v),
    )
    return owners


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


def candidate_ciks(tenure: pd.DataFrame, roster: pd.DataFrame) -> tuple[frozenset[str], dict[str, set[str]], dict[str, str]]:
    """(all candidates, {ticker: CIKs seen under its symbol}, {ticker: roster CIK}).

    Candidates are every issuer CIK that filed under a current-universe symbol plus every roster
    CIK; any other CIK is a singleton by default.
    """
    roster_cik = roster_cik_map(roster)
    seen = tenure[tenure["symbol"].astype(str).isin(set(roster_cik))]
    by_ticker: dict[str, set[str]] = {t: {roster_cik[t]} for t in roster_cik}
    for symbol, cik in zip(seen["symbol"].astype(str), seen["issuer_cik"].astype(str), strict=False):
        by_ticker[str(symbol)].add(str(cik))
    return frozenset().union(*by_ticker.values()), by_ticker, roster_cik


def validate_manual_tenure_entities(manual: pd.DataFrame, lineage: pd.DataFrame, roster: pd.DataFrame) -> None:
    """Require each manual CIK to belong to its configured current ticker's entity."""
    entity_by_cik = entity_by_cik_map(lineage)
    roster_cik = roster_cik_map(roster)
    errors: list[str] = []
    for row in manual.itertuples(index=False):
        canonical_ticker = str(row.canonical_ticker)
        issuer_cik = str(row.issuer_cik)
        home_cik = roster_cik.get(canonical_ticker)
        if home_cik is None:
            errors.append(f"{canonical_ticker}: absent from the current roster")
            continue
        expected = entity_or_singleton(entity_by_cik, home_cik)
        actual = entity_or_singleton(entity_by_cik, issuer_cik)
        if actual != expected:
            errors.append(f"{canonical_ticker}/{row.symbol}/{issuer_cik}: manual entity {actual}, roster entity {expected}")
    if errors:
        raise ManualTenureEntityError("symbol_tenure_manual contains CIKs outside their canonical current entity: " + "; ".join(errors))


def detect_older_cik_rekeys(existing: pd.DataFrame, candidate: pd.DataFrame) -> list[dict[str, Any]]:
    """Return stable-group ID changes caused by a newly joined numerically older CIK."""
    old = existing[["cik", "entity_id"]].drop_duplicates().copy()
    new = candidate[["cik", "entity_id"]].drop_duplicates().copy()
    old["cik"] = pad_cik_series(old["cik"])
    new["cik"] = pad_cik_series(new["cik"])
    old_map = dict(zip(old["cik"], old["entity_id"].astype(str), strict=False))
    new_map = dict(zip(new["cik"], new["entity_id"].astype(str), strict=False))
    new_members = {str(entity): list(members) for entity, members in new.groupby("entity_id")["cik"].agg(lambda values: sorted(set(values))).items()}
    impacts: list[dict[str, Any]] = []
    for old_entity_value, group in old.groupby("entity_id", sort=True):
        old_entity = str(old_entity_value)
        members = sorted(set(group["cik"]))
        mapped = {new_map[cik] for cik in members if cik in new_map}
        if len(mapped) != 1:
            continue
        new_entity = str(next(iter(mapped)))
        if new_entity == old_entity or not (old_entity.startswith("E") and new_entity.startswith("E")):
            continue
        added = sorted(set(new_members.get(new_entity, [])) - set(old_map))
        if not added or new_entity[1:] >= str(old_entity)[1:]:
            continue
        impacts.append(
            {
                "old_entity_id": str(old_entity),
                "new_entity_id": new_entity,
                "existing_ciks": members,
                "new_older_ciks": [cik for cik in added if cik < min(members)],
                "candidate_ciks": new_members.get(new_entity, []),
            }
        )
    return [impact for impact in impacts if impact["new_older_ciks"]]


# Evidence: one frame of (symbol, cik, source) intervals
def _evidence(tenure: pd.DataFrame, dei: pd.DataFrame | None) -> pd.DataFrame:
    """`form345`, `manual` and collapsed `dei` intervals as `[symbol, cik, source, first, end, n, evidence]`; `end` NaT = open."""
    parts = [tenure]
    if dei is not None and not dei.empty:
        parts.append(dei)
    frame = pd.concat(parts, ignore_index=True)
    sources = frame["source"].astype(str) if "source" in frame.columns else pd.Series(FORM345_SOURCE, index=frame.index)
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
    out = out[out["first"].notna() & out["cik"].ne("") & out["symbol"].ne("")]
    return out.sort_values(["symbol", "cik", "source", "first"], kind="mergesort", ignore_index=True)


class _EvidenceIndex:
    """`form345` and `dei` evidence rows by (CIK, source), read lazily."""

    def __init__(self, evidence: pd.DataFrame) -> None:
        self.frame = evidence[evidence["source"].isin(EVIDENCE_SOURCES)].reset_index(drop=True)
        self._positions = self.frame.groupby(["cik", "source"], sort=False).indices

    def rows(self, cik: str, source: str) -> pd.DataFrame | None:
        positions = self._positions.get((cik, source))
        return None if positions is None else self.frame.iloc[positions]

    def first_observed(self, cik: str) -> pd.Timestamp | None:
        firsts = [rows["first"].min() for source in EVIDENCE_SOURCES if (rows := self.rows(cik, source)) is not None]
        return min(firsts) if firsts else None


@dataclass(frozen=True)
class _Switch:
    """A predecessor -> successor CIK switch read from both sources, or the reason it is undecided."""

    boundary: pd.Timestamp | None
    window: tuple[pd.Timestamp, pd.Timestamp] | None
    detail: str


def _switch(index: _EvidenceIndex, pred: str, succ: str, symbols: frozenset[str] | None) -> _Switch:
    """Both sources must place `pred`'s last and `succ`'s first filing within `SWITCH_TOLERANCE_DAYS`.

    `symbols` limits the bounds to those symbols (a handoff); None reads every symbol of each CIK, so the
    successor must be new at the switch (an automatic window). A shared symbol is required in each source.
    """
    tolerance = pd.Timedelta(days=SWITCH_TOLERANCE_DAYS)
    lasts: list[pd.Timestamp] = []
    firsts: list[pd.Timestamp] = []
    notes: list[str] = []
    for source in EVIDENCE_SOURCES:
        p_rows, s_rows = index.rows(pred, source), index.rows(succ, source)
        if p_rows is None or s_rows is None:
            return _Switch(None, None, f"no {source} evidence on both CIKs")
        shared = set(p_rows["symbol"]) & set(s_rows["symbol"])
        if symbols is not None:
            shared &= symbols
            p_rows, s_rows = p_rows[p_rows["symbol"].isin(symbols)], s_rows[s_rows["symbol"].isin(symbols)]
        if not shared:
            return _Switch(None, None, f"no shared {source} symbol")
        if p_rows["end"].isna().any():
            return _Switch(None, None, f"{source}: predecessor still observed")
        p_first, p_last, s_first = p_rows["first"].min(), p_rows["end"].max() - pd.Timedelta(days=1), s_rows["first"].min()
        if not p_first < s_first:
            return _Switch(None, None, f"{source}: successor observed from {s_first.date()}, before the predecessor's first {p_first.date()}")
        if abs(s_first - p_last) > tolerance:
            return _Switch(None, None, f"{source}: predecessor last {p_last.date()} vs successor first {s_first.date()}")
        lasts.append(p_last)
        firsts.append(s_first)
        notes.append(f"{source} {pred} last {p_last.date()} -> {succ} first {s_first.date()} ({'/'.join(sorted(shared))})")
    if max(firsts) - min(firsts) > tolerance:
        return _Switch(None, None, "sources disagree: " + "; ".join(notes))
    boundary, last = min(firsts), max(lasts)
    return _Switch(boundary, (min(last, boundary), max(last, boundary)), "; ".join(notes))


def _auto_chain(index: _EvidenceIndex, ciks: Iterable[str], newest: str) -> tuple[list[tuple[str, pd.Timestamp, pd.Timestamp | None]] | None, str]:
    """D3: dated windows for the CIKs with symbol evidence, oldest first, or None and the reason to abstain."""
    observed = {cik: first for cik in ciks if (first := index.first_observed(cik)) is not None}
    if len(observed) < 2:
        return None, "fewer than two CIKs with symbol evidence"
    order = sorted(observed, key=lambda cik: (observed[cik], cik))
    if order[-1] != newest:
        return None, f"the newest CIK by evidence is {order[-1]}, not {newest}"
    boundaries: list[pd.Timestamp] = []
    notes: list[str] = []
    for pred, succ in pairwise(order):
        switch = _switch(index, pred, succ, None)
        if switch.boundary is None:
            return None, f"{pred}->{succ}: {switch.detail}"
        boundaries.append(switch.boundary)
        notes.append(switch.detail)
    starts = [SENTINEL_START, *boundaries]
    ends: list[pd.Timestamp | None] = [*boundaries, None]
    return [(cik, start, end) for cik, start, end in zip(order, starts, ends, strict=True)], "; ".join(notes)


def reproduce_register(registrants: Mapping[str, Registrant], index: _EvidenceIndex) -> pd.DataFrame:
    """Replay every register seam through the automatic-window rule (AC-018).

    `reproduced`: both sources decide and the register boundary lies within [predecessor last, successor
    first] (one day either side); `abstained`: undecided; `contradicted`: decided elsewhere, or the
    entity-level chain orders or splits the CIKs differently from the register.
    """
    rows = []
    for ticker, entry in sorted(registrants.items()):
        chain, chain_note = _auto_chain(index, entry.all_ciks(), entry.segments[-1].cik)
        chain_order = [cik for cik, _, _ in chain] if chain is not None else None
        for pred, succ in pairwise(entry.segments):
            switch = _switch(index, pred.cik, succ.cik, None)
            boundary = succ.valid_from
            verdict = "abstained"
            if switch.window is not None and boundary is not None:
                lo, hi = switch.window
                inside = lo - pd.Timedelta(days=1) <= boundary <= hi + pd.Timedelta(days=1)
                verdict = "reproduced" if inside else "contradicted"
            if chain_order is not None and (
                pred.cik not in chain_order or succ.cik not in chain_order or chain_order.index(succ.cik) != chain_order.index(pred.cik) + 1
            ):
                verdict = "contradicted"
            rows.append(
                {
                    "ticker": ticker,
                    "predecessor": pred.cik,
                    "successor": succ.cik,
                    "register_boundary": boundary,
                    "auto_boundary": switch.boundary,
                    "window_lo": switch.window[0] if switch.window else pd.NaT,
                    "window_hi": switch.window[1] if switch.window else pd.NaT,
                    "verdict": verdict,
                    "detail": switch.detail,
                    "entity_chain": "decided" if chain is not None else "abstained",
                    "entity_detail": chain_note,
                }
            )
    return pd.DataFrame(rows)


# Membership
@dataclass
class _Membership:
    """The union-find after every oracle, with what each oracle saw."""

    union: _Union
    provenance: _Provenance
    grey: list[tuple[str, str, str, int, float]]
    verdicts: Counter
    handoffs: int

    def entity_of(self) -> dict[str, str]:
        return {cik: entity_id_for(members) for members in self.union.groups().values() for cik in members}


def _join(union: _Union, ciks: list[str], source: str) -> None:
    """Join every non-excluded CIK of a curated group to its first non-excluded member."""
    kept = [cik for cik in ciks if cik not in union.excluded]
    for other in kept[1:]:
        union.union(kept[0], other, source=source)


def _membership(
    candidates: frozenset[str],
    by_ticker: Mapping[str, set[str]],
    roster_cik: Mapping[str, str],
    registrants: Mapping[str, Registrant],
    manual: Mapping[str, dict],
    owners: Mapping[str, set[str]],
    index: _EvidenceIndex,
    excluded: frozenset[str],
) -> _Membership:
    """Run the oracles in priority order over the candidate CIKs, refusing merges with `excluded` CIKs."""
    union = _Union(roster_ciks=frozenset(roster_cik.values()), excluded=excluded)
    for cik in sorted(candidates):
        union.parent.setdefault(cik, cik)
    provenance = _Provenance()
    for ticker, entry in sorted(registrants.items()):
        for cik in entry.all_ciks():
            union.parent.setdefault(cik, cik)
        _join(union, list(entry.all_ciks()), f"register[{ticker}]")
        for segment in entry.segments:
            provenance.curated.add(segment.cik)
            provenance.claim(segment.cik, "register", None, f"{ticker} {entry.kind}: {segment.evidence}")
    for key, entry in sorted(manual.items()):
        for cik in entry["same_entity"] + entry["own_entity"]:
            union.parent.setdefault(cik, cik)
            provenance.curated.add(cik)
            provenance.claim(cik, "manual", None, f"{key}: {entry['evidence']}")
        _join(union, entry["same_entity"], f"manual[{key}]")
    handoffs = _apply_symbol_handoff(union, provenance, index, roster_cik)
    grey, verdicts = _apply_owner_overlap(union, provenance, by_ticker, roster_cik, owners)
    return _Membership(union, provenance, grey, verdicts, handoffs)


def _apply_symbol_handoff(union: _Union, provenance: _Provenance, index: _EvidenceIndex, roster_cik: Mapping[str, str]) -> int:
    """Oracle 3: join a CIK that handed one symbol to (or took it from) a universe-entity CIK, read from both sources.

    One pass over the symbols the universe entities typed in both sources; returns the joins made.
    """
    roots = {union.find(cik) for cik in roster_cik.values()}
    universe = {cik for cik in list(union.parent) if union.find(cik) in roots}
    frame = index.frame
    both = frame.groupby(["symbol", "cik"])["source"].nunique()
    both = both[both == len(EVIDENCE_SOURCES)].reset_index()[["symbol", "cik"]]
    symbols = sorted(set(both.loc[both["cik"].isin(universe), "symbol"]))
    holders_by_symbol = both[both["symbol"].isin(symbols)].groupby("symbol")["cik"].agg(sorted).to_dict()
    joins = 0
    for symbol in symbols:
        holders = sorted(holders_by_symbol.get(symbol, []), key=lambda cik: (_first_of(index, cik, symbol), cik))
        for pred, succ in combinations(holders, 2):
            in_universe = (union.find(pred) in roots, union.find(succ) in roots)
            if not any(in_universe) or union.find(pred) == union.find(succ):
                continue
            switch = _switch(index, pred, succ, frozenset({symbol}))
            if switch.boundary is None:
                continue
            newcomer = succ if in_universe[0] else pred
            if union.union(pred, succ, source=f"symbol_handoff[{symbol}]"):
                joins += 1
                provenance.claim(newcomer, "symbol_handoff", None, f"{symbol} handed {pred} -> {succ}: {switch.detail}")
                roots = {union.find(cik) for cik in roster_cik.values()}
    return joins


def _first_of(index: _EvidenceIndex, cik: str, symbol: str) -> pd.Timestamp:
    """The first observation of `symbol` under `cik` across both sources."""
    firsts = [rows.loc[rows["symbol"].eq(symbol), "first"].min() for source in EVIDENCE_SOURCES if (rows := index.rows(cik, source)) is not None]
    return min(first for first in firsts if pd.notna(first))


def _apply_owner_overlap(
    union: _Union,
    provenance: _Provenance,
    by_ticker: Mapping[str, set[str]],
    roster_cik: Mapping[str, str],
    owners: Mapping[str, set[str]],
) -> tuple[list[tuple[str, str, str, int, float]], Counter]:
    """Oracle 4: score each non-roster candidate against its ticker's roster CIK and merge `same`.

    Curated CIKs and CIKs already joined to the roster CIK are not re-opened. Returns grey-band pairs and verdict counts.
    """
    grey: list[tuple[str, str, str, int, float]] = []
    verdicts: Counter = Counter()
    pairs = [(ticker, cik, roster_cik[ticker]) for ticker in sorted(by_ticker) for cik in sorted(by_ticker[ticker]) if cik != roster_cik[ticker]]
    for ticker, cik, home in pairs:
        shared, jaccard = score_overlap(owners.get(cik, set()), owners.get(home, set()))
        verdict = classify_overlap(shared, jaccard)
        if cik in provenance.curated or union.find(cik) == union.find(home) or cik in union.excluded:
            verdicts[f"{verdict} (pre-decided)"] += 1
            continue
        verdicts[verdict] += 1
        if verdict == "grey":
            grey.append((ticker, cik, home, shared, jaccard))
        elif verdict == "same" and union.union(cik, home, source=f"owner_overlap[{ticker}]", confidence=jaccard):
            provenance.claim(cik, "owner_overlap", jaccard, f"{ticker}: {shared} reporting owner(s) shared with {home}, jaccard {jaccard:.3f}")
    return grey, verdicts


# The build
@dataclass(frozen=True)
class LineageBuild:
    """`rows` for `entity_lineage`, plus what the build refused, excluded or measured."""

    rows: pd.DataFrame
    blocked: pd.DataFrame
    backlog: pd.DataFrame
    holders: pd.DataFrame
    reproduction: pd.DataFrame


def derive_entity_lineage(
    tenure: pd.DataFrame,
    roster: pd.DataFrame,
    owner_pairs: pd.DataFrame,
    config_dir: str | None = None,
    *,
    dei: pd.DataFrame | None = None,
    existing: pd.DataFrame | None = None,
    approved_rekeys: frozenset[tuple[str, str]] = frozenset(),
    built_at: pd.Timestamp | None = None,
    auto_windows: bool | None = None,
) -> LineageBuild:
    """The dated `entity_lineage` rows from `form345`/`manual` tenure, collapsed `dei` rows, the roster and owner pairs.

    Pure: no DB, no network. `existing` (the stored table) drives the older-CIK rekey exclusion and
    `scope_changed_at`; `built_at` pins the change timestamp. Raises only on a D19 disagreement or a
    builder bug (`CikInTwoEntitiesError`).
    """
    stamp = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("s") if built_at is None else pd.Timestamp(built_at)
    use_auto = AUTO_WINDOWS_ENABLED if auto_windows is None else auto_windows
    registrants = load_registrants(config_dir)
    manual = load_manual_lineage(config_dir)
    evidence = _evidence(tenure, dei)
    index = _EvidenceIndex(evidence)
    candidates, by_ticker, roster_cik = candidate_ciks(tenure, roster)
    owners = derive_owner_sets(owner_pairs, candidates)
    backlog: list[dict[str, Any]] = []

    excluded: frozenset[str] = frozenset()
    membership = _membership(candidates, by_ticker, roster_cik, registrants, manual, owners, index, excluded)
    for _ in range(_MAX_REKEY_PASSES):
        newly = _unapproved_rekey_ciks(existing, membership.entity_of(), approved_rekeys, backlog)
        if not newly:
            break
        excluded |= newly
        membership = _membership(candidates, by_ticker, roster_cik, registrants, manual, owners, index, excluded)
    entity_of = membership.entity_of()
    ticker_by_entity = {entity_of.get(cik, f"E{cik}"): ticker for ticker, cik in roster_cik.items()}
    backlog += _grey_backlog(membership.grey, entity_of, ticker_by_entity)

    _check_d19(tenure, roster_cik, entity_of, load_d19_allowlist(config_dir))
    holders = _symbol_holders(evidence, entity_of, ticker_by_entity, roster_cik)
    cik_rows, windows = _cik_rows(membership, entity_of, ticker_by_entity, roster_cik, registrants, index, evidence, use_auto, backlog)
    symbol_rows = _close_on_window(holders[holders["canonical_ticker"].notna()], windows).assign(role=ROLE_SYMBOL, confidence=float("nan"))
    symbol_rows["oracle"] = [membership.provenance.verdicts.get(cik, ("roster",))[0] for cik in symbol_rows["cik"]]
    backlog += _conflict_backlog(symbol_rows)
    rows = _finalise(pd.concat([cik_rows, symbol_rows[list(TABLE_COLUMNS[:-1])]], ignore_index=True), existing, stamp)
    check_one_entity_per_cik(rows)
    out = LineageBuild(
        rows=rows,
        blocked=_blocked_frame(membership.union, roster_cik),
        backlog=pd.DataFrame(backlog, columns=list(BACKLOG_COLUMNS)),
        holders=holders,
        reproduction=reproduce_register(registrants, index),
    )
    _log_build(membership, out, len(candidates), excluded, use_auto)
    return out


def _unapproved_rekey_ciks(
    existing: pd.DataFrame | None, entity_of: Mapping[str, str], approved: frozenset[tuple[str, str]], backlog: list[dict[str, Any]]
) -> frozenset[str]:
    """The older CIKs whose join would rename a stored entity without approval; each is backlogged."""
    if existing is None or existing.empty or not {"cik", "entity_id"} <= set(existing.columns):
        return frozenset()
    candidate = pd.DataFrame({"cik": list(entity_of), "entity_id": list(entity_of.values())})
    newly: set[str] = set()
    for impact in detect_older_cik_rekeys(existing, candidate):
        pair = (str(impact["old_entity_id"]), str(impact["new_entity_id"]))
        if pair in approved:
            logger.warning("entity_lineage: applying explicitly approved older-CIK rekey %s -> %s", *pair)
            continue
        older = [str(cik) for cik in impact["new_older_ciks"]]
        newly.update(older)
        for cik in older:
            backlog.append(
                {
                    "kind": "rekey",
                    "canonical_ticker": None,
                    "entity_id": pair[0],
                    "cik": cik,
                    "symbol": "",
                    "detail": f"joining {cik} would rename {pair[0]} -> {pair[1]}; excluded (approve with --approve-rekey {pair[0]}:{pair[1]})",
                }
            )
    return frozenset(newly)


def _grey_backlog(
    grey: list[tuple[str, str, str, int, float]], entity_of: Mapping[str, str], ticker_by_entity: Mapping[str, str]
) -> list[dict[str, Any]]:
    """One backlog row per grey-band pair, which stays out of the entity."""
    return [
        {
            "kind": "grey_band",
            "canonical_ticker": ticker,
            "entity_id": entity_of.get(home, f"E{home}"),
            "cik": cik,
            "symbol": ticker,
            "detail": f"owner overlap with {home}: shared={shared}, jaccard={jaccard:.3f}; excluded until curated",
        }
        for ticker, cik, home, shared, jaccard in grey
        if ticker_by_entity.get(entity_of.get(home, f"E{home}")) == ticker
    ]


def _check_d19(tenure: pd.DataFrame, roster_cik: Mapping[str, str], entity_of: Mapping[str, str], allowlist: Mapping[str, str]) -> None:
    """D19: the dominant `form345`/`manual` holder of each roster ticker must be the roster CIK's entity, unless allow-listed."""
    frame = pd.DataFrame(
        {
            "symbol": tenure["symbol"].map(normalise_ticker),
            "cik": pad_cik_series(tenure["issuer_cik"]),
            "open": tenure["valid_to"].isna(),
            "n": pd.to_numeric(tenure["n_filings"], errors="coerce").fillna(0),
            "manual": tenure["source"].astype(str).eq(MANUAL_SOURCE) if "source" in tenure.columns else False,
        }
    )
    frame = frame[frame["symbol"].isin(set(roster_cik))]
    by_symbol = {symbol: group for symbol, group in frame.groupby("symbol", sort=False)}
    unexplained = []
    for ticker, cik in sorted(roster_cik.items()):
        rows = by_symbol.get(ticker)
        if rows is None:
            dominant = None
        else:
            manual_open = rows[rows["manual"] & rows["open"]]
            opened = rows[rows["open"]]
            pool = manual_open if not manual_open.empty else opened if not opened.empty else rows[rows["manual"]] if rows["manual"].any() else rows
            dominant = str(pool.sort_values(["n", "cik"], kind="mergesort").iloc[-1]["cik"])
        home = entity_of.get(cik, f"E{cik}")
        if (dominant is None or entity_of.get(dominant, f"E{dominant}") != home) and ticker not in allowlist:
            unexplained.append(f"{ticker}: roster {cik} -> {home} but tenure -> {dominant or 'NO TENURE'}")
    if unexplained:
        raise UniverseEntityDisagreementError(
            f"identity: {len(unexplained)} universe ticker(s) whose roster CIK and whose filings name different entities -- "
            + "; ".join(unexplained)
            + ". Fix the roster CIK, or add the ticker to `_d19_allowlist` in entity_lineage_manual.json WITH the evidence."
        )


# Symbol rows
def _symbol_holders(
    evidence: pd.DataFrame, entity_of: Mapping[str, str], ticker_by_entity: Mapping[str, str], roster_cik: Mapping[str, str]
) -> pd.DataFrame:
    """Every holder interval of the symbols the universe entities ever typed, classified (other entities included)."""
    universe_ciks = {cik for cik, entity in entity_of.items() if entity in ticker_by_entity}
    scope = set(evidence.loc[evidence["cik"].isin(universe_ciks), "symbol"]) | set(roster_cik)
    in_scope = evidence[evidence["symbol"].isin(scope)]
    dei_rows = evidence[evidence["source"].eq(DEI_SOURCE)]
    current_after = dei_rows["end"].max() - pd.Timedelta(days=CURRENT_RECENCY_DAYS + 1) if not dei_rows.empty else None
    rows = [_merge_holder(group, current_after) for _, group in in_scope.groupby(["symbol", "cik"], sort=True)]
    rows = [row for merged in rows for row in merged]
    rows += _roster_rows(rows, roster_cik)
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_symbol.setdefault(str(row["symbol"]), []).append(row)
    out = [row for symbol in sorted(by_symbol) for row in _classify_symbol(by_symbol[symbol], entity_of)]
    frame = pd.DataFrame(out).drop(columns=["entity", "curated", "roster"])
    frame["entity_id"] = [entity_of.get(cik, f"E{cik}") for cik in frame["cik"]]
    frame["canonical_ticker"] = frame["entity_id"].map(ticker_by_entity)
    frame["valid_from"] = pd.to_datetime(frame["valid_from"])
    frame["valid_to"] = pd.to_datetime(frame["valid_to"])
    return frame.sort_values(["symbol", "valid_from", "cik"], kind="mergesort", ignore_index=True)


def _merge_holder(group: pd.DataFrame, current_after: pd.Timestamp | None) -> list[dict[str, Any]]:
    """One (symbol, CIK)'s intervals: manual rows as they are, else `form345` and `dei` merged into one interval."""
    symbol, cik = str(group["symbol"].iloc[0]), str(group["cik"].iloc[0])
    manual = group[group["source"].eq(MANUAL_SOURCE)]
    if not manual.empty:
        return [
            {
                "symbol": symbol,
                "cik": cik,
                "valid_from": row.first,
                "valid_to": None if pd.isna(row.end) else row.end,
                "sources": frozenset({MANUAL_SOURCE}),
                "n_observations": int(group.loc[group["source"].ne(MANUAL_SOURCE), "n"].sum()),
                "evidence": f"manual: {row.evidence}",
                "curated": True,
                "roster": False,
            }
            for row in manual.sort_values("first").itertuples(index=False)
        ]
    derived = group[group["source"].isin(EVIDENCE_SOURCES)]
    if derived.empty:
        return []
    form345 = derived[derived["source"].eq(FORM345_SOURCE)]
    dei = derived[derived["source"].eq(DEI_SOURCE)]
    is_open = bool(form345["end"].isna().any()) or (current_after is not None and not dei.empty and dei["end"].max() > current_after)
    parts = [
        f"{source} {part['first'].min().date()}..{_last_text(part)} n={int(part['n'].sum())}"
        + (f" ({part['evidence'].iloc[-1]})" if part["evidence"].iloc[-1] else "")
        for source, part in ((FORM345_SOURCE, form345), (DEI_SOURCE, dei))
        if not part.empty
    ]
    return [
        {
            "symbol": symbol,
            "cik": cik,
            "valid_from": derived["first"].min(),
            "valid_to": None if is_open else derived["end"].max(),
            "sources": frozenset(derived["source"]),
            "n_observations": int(derived["n"].sum()),
            "evidence": "; ".join(parts),
            "curated": False,
            "roster": False,
        }
    ]


def _last_text(part: pd.DataFrame) -> str:
    """The last observed filing date of an evidence part, or `open`."""
    return "open" if part["end"].isna().any() else str((part["end"].max() - pd.Timedelta(days=1)).date())


def _roster_rows(rows: list[dict[str, Any]], roster_cik: Mapping[str, str]) -> list[dict[str, Any]]:
    """Mark each roster ticker's latest interval on its roster CIK as current; add one when no filing spells the ticker."""
    added: list[dict[str, Any]] = []
    for ticker, cik in sorted(roster_cik.items()):
        own = [row for row in rows if row["symbol"] == ticker and row["cik"] == cik]
        latest = max(own, key=lambda row: row["valid_from"], default=None)
        if latest is not None and not (latest["curated"] and latest["valid_to"] is not None):
            latest["roster"] = True
            latest["sources"] = frozenset(latest["sources"]) | {ROSTER_SOURCE}
            latest["valid_to"] = None
            continue
        starts = [row["valid_from"] for row in rows if row["cik"] == cik]
        start = latest["valid_to"] if latest is not None else min(starts, default=SENTINEL_START)
        added.append(
            {
                "symbol": ticker,
                "cik": cik,
                "valid_from": start,
                "valid_to": None,
                "sources": frozenset({ROSTER_SOURCE}),
                "n_observations": 0,
                "evidence": "roster: sp500_tickers names this ticker on this CIK today; no filing spells it this way",
                "curated": False,
                "roster": True,
            }
        )
    return added


def _gap_days(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    """Days from the end of one half-open interval to the start of the other; negative when they overlap (open end = +inf)."""
    a_after_b = (pd.Timestamp(a["valid_from"]) - pd.Timestamp(b["valid_to"])).days if b["valid_to"] is not None else float("-inf")
    b_after_a = (pd.Timestamp(b["valid_from"]) - pd.Timestamp(a["valid_to"])).days if a["valid_to"] is not None else float("-inf")
    return max(a_after_b, b_after_a)


def _is_weak(row: Mapping[str, Any]) -> bool:
    """A derived single-source interval with at most `NOISE_MAX_OBSERVATIONS` filings."""
    filed = frozenset(row["sources"]) & set(EVIDENCE_SOURCES)
    return not row["curated"] and not row["roster"] and len(filed) == 1 and int(row["n_observations"]) <= NOISE_MAX_OBSERVATIONS


def _classify_symbol(rows: list[dict[str, Any]], entity_of: Mapping[str, str]) -> list[dict[str, Any]]:
    """Status per interval of one symbol: noise, clipping by anchors of other entities, then reuse conflicts.

    Anchors are curated rows and each roster ticker's interval on its roster CIK; a curated row clips a roster
    one, and both clip derived rows of other entities. A weak row is noise when another entity held the symbol
    within `REUSE_GAP_DAYS`, or at any time with a stronger interval. Derived rows of two entities within
    `REUSE_GAP_DAYS` of each other are both `conflict`.
    """
    for row in rows:
        row["entity"] = entity_of.get(str(row["cik"]), f"E{row['cik']}")
        row["status"] = ""
    for row in rows:
        if _is_weak(row) and any(
            other["entity"] != row["entity"] and (not _is_weak(other) or _gap_days(row, other) < REUSE_GAP_DAYS) for other in rows
        ):
            row["status"] = "noise"
    by_start = sorted(rows, key=lambda row: pd.Timestamp(row["valid_from"]))
    curated = [row for row in by_start if row["curated"]]
    roster = [row for row in by_start if row["roster"] and not row["curated"]]
    _clip_by_anchors([row for row in rows if not row["curated"]], curated)
    _clip_by_anchors([row for row in rows if not row["curated"] and not row["roster"]], roster)
    live = [row for row in rows if not row["status"] and not row["curated"] and not row["roster"]]
    for row, other in combinations(live, 2):
        if other["entity"] != row["entity"] and _gap_days(row, other) < REUSE_GAP_DAYS:
            row["status"] = other["status"] = "conflict"
    for row in rows:
        if row["curated"]:
            row["status"] = "curated"
        elif not row["status"]:
            filed = frozenset(row["sources"]) & set(EVIDENCE_SOURCES)
            row["status"] = "corroborated" if len(filed) == len(EVIDENCE_SOURCES) else "single_source"
    return rows


def _clip_by_anchors(targets: list[dict[str, Any]], anchors: list[dict[str, Any]]) -> None:
    """Clip each unclassified target by every overlapping anchor of another entity, in anchor order."""
    for row in targets:
        for holder in anchors:
            if row["status"]:
                break
            if holder["entity"] != row["entity"] and _gap_days(row, holder) < 0:
                _clip(row, holder)


def _clip(row: dict[str, Any], holder: Mapping[str, Any]) -> None:
    """Cut `row` to the part outside an anchor interval of another entity; fully inside it, the row is noise."""
    start, end = pd.Timestamp(row["valid_from"]), row["valid_to"]
    h_start, h_end = pd.Timestamp(holder["valid_from"]), holder["valid_to"]
    if h_end is not None and (end is None or pd.Timestamp(end) > pd.Timestamp(h_end)):
        row["valid_from"] = max(start, pd.Timestamp(h_end))
    elif start < h_start:
        row["valid_to"] = h_start
    else:
        row["status"] = "noise"
        row["evidence"] = f"{row['evidence']}; superseded by the {'curated' if holder['curated'] else 'roster'} interval of {holder['cik']}"
        return
    row["evidence"] = f"{row['evidence']}; clipped by the {'curated' if holder['curated'] else 'roster'} interval of {holder['cik']}"


def _close_on_window(symbol_rows: pd.DataFrame, windows: Mapping[str, pd.Timestamp | None]) -> pd.DataFrame:
    """An open symbol row on a CIK whose consolidating window has closed ends at that window (never current)."""
    out = symbol_rows.copy()
    ends = out["cik"].map(lambda cik: windows.get(cik))
    roster_anchored = pd.Series([ROSTER_SOURCE in sources for sources in out["sources"]], index=out.index, dtype=bool)
    close = out["valid_to"].isna() & ends.notna() & ~roster_anchored
    out.loc[close, "valid_to"] = [
        max(pd.Timestamp(end), pd.Timestamp(start) + pd.Timedelta(days=1))
        for end, start in zip(ends[close], out.loc[close, "valid_from"], strict=True)
    ]
    return out


# CIK rows
def _cik_rows(
    membership: _Membership,
    entity_of: Mapping[str, str],
    ticker_by_entity: Mapping[str, str],
    roster_cik: Mapping[str, str],
    registrants: Mapping[str, Registrant],
    index: _EvidenceIndex,
    evidence: pd.DataFrame,
    use_auto: bool,
    backlog: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, pd.Timestamp | None]]:
    """`cik_window` / `cik_event` rows per entity, and each CIK's window end (None = open) for closing symbol rows."""
    groups: dict[str, set[str]] = {}
    for cik, entity in entity_of.items():
        groups.setdefault(entity, set()).add(cik)
    provenance = membership.provenance
    counts = evidence[evidence["source"].isin(EVIDENCE_SOURCES)].groupby("cik")["n"].sum().to_dict()
    register_by_cik = {segment.cik: (ticker, segment) for ticker, entry in registrants.items() for segment in entry.segments}
    rows: list[dict[str, Any]] = []
    window_end: dict[str, pd.Timestamp | None] = {}
    for entity, ciks in sorted(groups.items()):
        ticker = ticker_by_entity.get(entity)
        if ticker is None and not (ciks & provenance.curated):
            continue
        windows = _entity_windows(entity, ciks, ticker, roster_cik, registrants, register_by_cik, index, use_auto, provenance, backlog)
        for cik in sorted(ciks):
            oracle, confidence, why = provenance.verdicts.get(cik, ("roster", None, "roster CIK with no predecessor found by any oracle"))
            window = windows.get(cik)
            role = ROLE_WINDOW if window is not None else ROLE_EVENT
            start, end, window_sources, window_status, window_note = (
                window if window is not None else (SENTINEL_START, None, _ORACLE_SOURCES[oracle], "", "")
            )
            if window is not None:
                window_end[cik] = end
            rows.append(
                {
                    "entity_id": entity,
                    "canonical_ticker": ticker,
                    "cik": cik,
                    "role": role,
                    "symbol": "",
                    "valid_from": start,
                    "valid_to": end,
                    "status": window_status or _membership_status(oracle),
                    "sources": window_sources,
                    "oracle": oracle,
                    "confidence": confidence,
                    "n_observations": int(counts.get(cik, 0)),
                    "evidence": "; ".join(part for part in (why, window_note) if part),
                }
            )
    return pd.DataFrame(rows), window_end


#: A CIK's consolidating window: (valid_from, valid_to or None, sources, status, evidence note).
_Window = tuple[pd.Timestamp, pd.Timestamp | None, str, str, str]


#: The evidence sources behind each membership oracle (owner sets are read off Forms 3/4/5).
_ORACLE_SOURCES = {
    "register": REGISTER_SOURCE,
    "manual": MANUAL_SOURCE,
    "symbol_handoff": ",".join(sorted(EVIDENCE_SOURCES)),
    "owner_overlap": FORM345_SOURCE,
    "roster": ROSTER_SOURCE,
}


def _membership_status(oracle: str) -> str:
    """Status of a CIK row from the oracle that put the CIK in its entity."""
    return {"register": "curated", "manual": "curated", "symbol_handoff": "corroborated"}.get(oracle, "single_source")


def _entity_windows(
    entity: str,
    ciks: set[str],
    ticker: str | None,
    roster_cik: Mapping[str, str],
    registrants: Mapping[str, Registrant],
    register_by_cik: Mapping[str, tuple[str, object]],
    index: _EvidenceIndex,
    use_auto: bool,
    provenance: _Provenance,
    backlog: list[dict[str, Any]],
) -> dict[str, _Window]:
    """`{cik: (valid_from, valid_to, sources, status, note)}` for the CIKs whose consolidating filings belong to the entity.

    The register's segments; else D3 automatic windows; else the roster CIK alone (an extra uncurated CIK is backlogged).
    """
    entry = registrants.get(ticker) if ticker is not None else None
    if entry is None:
        register_tickers = {register_by_cik[cik][0] for cik in ciks if cik in register_by_cik}
        entry = registrants.get(sorted(register_tickers)[0]) if len(register_tickers) == 1 else None
    if entry is not None:
        return {
            segment.cik: (segment.valid_from or SENTINEL_START, segment.valid_to, REGISTER_SOURCE, "curated", f"register window ({entry.ticker})")
            for segment in entry.segments
            if segment.cik in ciks
        }
    if ticker is None:
        return {}
    home = roster_cik[ticker]
    roster_only: dict[str, _Window] = {home: (SENTINEL_START, None, ROSTER_SOURCE, "single_source", "roster CIK window")}
    if len(ciks) == 1:
        return roster_only
    chain, note = _auto_chain(index, ciks, home)
    if chain is not None and use_auto:
        sources = ",".join(sorted(EVIDENCE_SOURCES))
        return {cik: (start, end, sources, "corroborated", f"automatic window: {note}") for cik, start, end in chain}
    uncurated = sorted(cik for cik in ciks - {home} if cik not in provenance.curated)
    if uncurated:
        suggestion = "; suggested chain " + ", ".join(f"{cik} from {start.date()}" for cik, start, _ in chain) if chain is not None else ""
        backlog.append(
            {
                "kind": "multi_cik_no_window",
                "canonical_ticker": ticker,
                "entity_id": entity,
                "cik": ",".join(uncurated),
                "symbol": "",
                "detail": ("automatic windows disabled" if chain is not None else f"abstained: {note}")
                + f"; listing the roster CIK {home} only{suggestion}",
            }
        )
    return roster_only


def _conflict_backlog(symbol_rows: pd.DataFrame) -> list[dict[str, Any]]:
    """One backlog row per universe symbol interval left in `conflict`."""
    conflicts = symbol_rows[symbol_rows["status"].eq("conflict")]
    starts = pd.to_datetime(conflicts["valid_from"]).dt.strftime("%Y-%m-%d")
    ends = pd.to_datetime(conflicts["valid_to"]).dt.strftime("%Y-%m-%d").fillna("open")
    return [
        {
            "kind": "conflict",
            "canonical_ticker": ticker,
            "entity_id": entity,
            "cik": cik,
            "symbol": symbol,
            "detail": f"{symbol} held by another entity within {REUSE_GAP_DAYS} days of {start}..{end}",
        }
        for ticker, entity, cik, symbol, start, end in zip(
            conflicts["canonical_ticker"], conflicts["entity_id"], conflicts["cik"], conflicts["symbol"], starts, ends, strict=True
        )
    ]


def _finalise(rows: pd.DataFrame, existing: pd.DataFrame | None, stamp: pd.Timestamp) -> pd.DataFrame:
    """Typed, sorted table rows with `scope_changed_at` carried from `existing` for unchanged scopes."""
    out = rows.copy()
    out["symbol"] = out["symbol"].fillna("").astype(str)
    out["sources"] = [value if isinstance(value, str) else ",".join(sorted(value)) for value in out["sources"]]
    out["valid_from"] = pd.to_datetime(out["valid_from"])
    out["valid_to"] = pd.to_datetime(out["valid_to"])
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce").astype("float64")
    out["n_observations"] = out["n_observations"].astype("int64")
    out["canonical_ticker"] = out["canonical_ticker"].astype(object).where(out["canonical_ticker"].notna(), None)
    out["scope_changed_at"] = _scope_changed_at(out, existing, stamp)
    return out[list(TABLE_COLUMNS)].sort_values(["entity_id", "role", "cik", "symbol", "valid_from"], kind="mergesort", ignore_index=True)


def _scope_key(frame: pd.DataFrame) -> pd.Series:
    """Per-row scope key: the canonical ticker, else the entity id."""
    return frame["canonical_ticker"].where(frame["canonical_ticker"].notna(), frame["entity_id"]).astype(str)


def _stamp_group(frame: pd.DataFrame) -> pd.Series:
    """Per-row stamp group: `cik` for the CIK rows (EDGAR and bulk scope), `symbol` for the symbol rows (tapes)."""
    return frame["role"].astype(str).eq(ROLE_SYMBOL).map({True: ROLE_SYMBOL, False: "cik"})


def _scope_signatures(frame: pd.DataFrame) -> dict[tuple[str, str], tuple[tuple[str, ...], ...]]:
    """`{(scope key, stamp group): sorted signature}`: `(cik, role, valid_from, valid_to)` over the CIK rows and
    `(symbol, cik, valid_from, valid_to, status)` over the symbol rows a tape can read (not `dei` alone)."""
    bounds = [pd.to_datetime(frame[column]).dt.strftime("%Y-%m-%d").fillna("") for column in ("valid_from", "valid_to")]
    sources = frame["sources"].fillna("").astype(str) if "sources" in frame.columns else pd.Series("", index=frame.index)
    status = frame["status"].fillna("").astype(str) if "status" in frame.columns else pd.Series("", index=frame.index)
    symbol = frame["symbol"].fillna("").astype(str) if "symbol" in frame.columns else pd.Series("", index=frame.index)
    out: dict[tuple[str, str], list[tuple[str, ...]]] = {}
    rows = zip(
        _scope_key(frame), _stamp_group(frame), pad_cik_series(frame["cik"]), frame["role"].astype(str), *bounds, symbol, status, sources, strict=True
    )
    for key, group, cik, role, start, end, sym, stat, src in rows:
        if group == ROLE_SYMBOL:
            if src.strip().lower() == DEI_SOURCE:
                continue
            signature: tuple[str, ...] = (sym, cik, start, end, stat)
        else:
            signature = (cik, role, start, end)
        out.setdefault((key, group), []).append(signature)
    return {key: tuple(sorted(values)) for key, values in out.items()}


def _scope_changed_at(rows: pd.DataFrame, existing: pd.DataFrame | None, stamp: pd.Timestamp) -> pd.Series:
    """The stored timestamp of each unchanged (scope, stamp group); `stamp` for every other.

    CIK rows and symbol rows are stamped apart: a symbol-only change moves the tapes, not the EDGAR relist.
    """
    keys = list(zip(_scope_key(rows), _stamp_group(rows), strict=True))
    if existing is None or existing.empty or not {"role", "canonical_ticker", "scope_changed_at", "valid_to"} <= set(existing.columns):
        return pd.Series(stamp, index=rows.index, dtype="datetime64[ns]")
    before, after = _scope_signatures(existing), _scope_signatures(rows)
    stored_stamps = pd.to_datetime(existing["scope_changed_at"])
    stored = stored_stamps.groupby([_scope_key(existing), _stamp_group(existing)]).max().to_dict()
    kept = {key: stored[key] for key in set(keys) if before.get(key, ()) == after.get(key, ()) and pd.notna(stored.get(key))}
    return pd.Series([kept.get(key, stamp) for key in keys], index=rows.index, dtype="datetime64[ns]")


def _blocked_frame(union: _Union, roster_cik: Mapping[str, str]) -> pd.DataFrame:
    """The refused merges, each side named by the universe tickers it already holds."""
    return pd.DataFrame(
        [
            {
                "cik_a": a,
                "cik_b": b,
                "proposed_by": source,
                "confidence": conf,
                "tickers_a": ",".join(sorted(t for t, c in roster_cik.items() if c in union.roster_members(a))),
                "tickers_b": ",".join(sorted(t for t, c in roster_cik.items() if c in union.roster_members(b))),
            }
            for a, b, source, conf in union.blocked
        ],
        columns=["cik_a", "cik_b", "proposed_by", "confidence", "tickers_a", "tickers_b"],
    )


def _log_build(membership: _Membership, build: LineageBuild, n_candidates: int, excluded: frozenset[str], use_auto: bool) -> None:
    """One INFO summary, and one WARNING listing every excluded or undecided case (the curation backlog)."""
    rows = build.rows
    symbol_rows = rows[rows["role"].eq(ROLE_SYMBOL)]
    logger.info(
        "entity_lineage: %d candidate CIK(s) -> %d row(s) over %d entity(ies); roles %s; symbol statuses %s; "
        "owner-overlap verdicts %s; %d symbol handoff join(s); automatic windows %s; %d merge(s) BLOCKED as two-universe-tickers",
        n_candidates,
        len(rows),
        rows["entity_id"].nunique(),
        dict(rows["role"].value_counts().sort_index()),
        dict(symbol_rows["status"].value_counts().sort_index()),
        dict(membership.verdicts),
        membership.handoffs,
        "enabled" if use_auto else "disabled",
        len(build.blocked),
    )
    if excluded:
        logger.warning("entity_lineage: %d older CIK(s) excluded to keep stored entity ids: %s", len(excluded), ", ".join(sorted(excluded)))
    if not build.backlog.empty:
        logger.warning(
            "entity_lineage: curation backlog %d item(s) %s (excluded or undecided, not applied):\n%s",
            len(build.backlog),
            dict(build.backlog["kind"].value_counts().sort_index()),
            build.backlog.to_string(index=False),
        )
    else:
        logger.info("entity_lineage: curation backlog 0 item(s)")


def build_entity_lineage(
    context: Context,
    tenure: pd.DataFrame,
    owner_pairs: pd.DataFrame,
    config_dir: str | None = None,
    *,
    approved_rekeys: frozenset[tuple[str, str]] = frozenset(),
    built_at: pd.Timestamp | None = None,
    redundant_symbols: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """Derive `entity_lineage` from tenure, the stored `dei` evidence and owner pairs; replace the table unless unchanged.

    An older-CIK rekey is excluded and backlogged unless its ``(old_entity_id, new_entity_id)`` pair is in
    `approved_rekeys`. Logs the items needing a manual decision (`redundant_symbols`: the configured
    redundant share classes). Returns the derived rows.
    """
    roster = context.store.load(Tables.sp500_tickers, columns=list(ROSTER_COLUMNS))
    assert roster is not None
    existing = _stored_rows(context.store.load(Tables.entity_lineage, project=True, optional=True))
    dei_rows = context.store.load(Tables.symbol_tenure, project=True, where={"source": DEI_SOURCE}, optional=True)
    dei = collapse_dei_periods(dei_rows) if dei_rows is not None and not dei_rows.empty else None
    build = derive_entity_lineage(
        tenure, roster, owner_pairs, config_dir, dei=dei, existing=existing, approved_rekeys=approved_rekeys, built_at=built_at
    )
    out = build.rows
    manual_tenure = load_manual_symbol_tenure(config_dir or context.config_dir)
    validate_manual_tenure_entities(manual_tenure, out, roster)
    context.log.info(
        f"entity_lineage: {len(manual_tenure)} manual symbol interval(s) resolve to "
        f"their {manual_tenure['canonical_ticker'].nunique()} canonical entity(ies); "
        f"dei evidence {0 if dei is None else len(dei)} (symbol, CIK) interval(s); curation backlog {len(build.backlog)} item(s)"
    )
    evidence = tenure if dei is None else pd.concat([tenure, dei], ignore_index=True)
    log_identity_flags(context.log, identity_flags(out, cik_activity(evidence), redundant_symbols=redundant_symbols, backlog=build.backlog))
    if not build.blocked.empty:
        logger.warning(
            "entity_lineage: %d merge(s) refused because they would put two universe tickers in one entity:\n%s",
            len(build.blocked),
            build.blocked.to_string(index=False),
        )
    if existing is None or existing.empty:
        context.log.info(f"entity_lineage: cold build with {len(out)} row(s) over {out['entity_id'].nunique()} entity(ies)")
    else:
        _log_changed_assignments(context, existing, out, roster)
    unchanged = matches_stored(existing, out, Tables.entity_lineage)
    written = 0 if unchanged else context.store.replace(Tables.entity_lineage, out)
    if unchanged:
        logger.info("entity_lineage: unchanged (%d row(s)); replace skipped", len(out))
    else:
        logger.info("entity_lineage: wrote %d row(s)", written)
    return out


def _stored_rows(existing: pd.DataFrame | None) -> pd.DataFrame | None:
    """The stored table with its date and timestamp columns parsed (a store may return them as text or `date`)."""
    if existing is None:
        return None
    out = existing.copy()
    for column in ("valid_from", "valid_to", "scope_changed_at"):
        if column in out.columns:
            out[column] = pd.to_datetime(out[column])
    return out


def _log_changed_assignments(context: Context, existing: pd.DataFrame, out: pd.DataFrame, roster: pd.DataFrame) -> None:
    """Name every CIK whose entity changed and the current tickers those entities hold."""
    old_map = entity_by_cik_map(existing)
    new_map = entity_by_cik_map(out)
    changed = sorted(cik for cik in set(old_map) | set(new_map) if old_map.get(cik) != new_map.get(cik))
    entity_tickers = {entity_or_singleton(new_map, cik): ticker for ticker, cik in roster_cik_map(roster).items()}
    affected = sorted({entity_tickers[entity] for cik in changed for entity in (old_map.get(cik), new_map.get(cik)) if entity in entity_tickers})
    context.log.info(
        f"entity_lineage: {len(changed)} changed CIK assignment(s): "
        f"{', '.join(changed) if changed else 'none'}; affected current ticker(s): "
        f"{', '.join(affected) if affected else 'none'}"
    )
