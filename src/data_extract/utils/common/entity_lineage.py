"""
entity_lineage.py (src/data_extract/utils/common/entity_lineage.py)
--------------------------------------------------------------------------------------------
AXIS A OF TICKER IDENTITY: WHICH CIKs ARE THE SAME ECONOMIC COMPANY.

`symbol_tenure` answers "who held symbol X on date d". This answers "are these two CIKs the
same firm", and it is the one the `owns()` predicate reads. The two questions are genuinely
independent, and collapsing them is what the whole identity defect is made of: CIK
`0001466258` filed under `IR` for eleven years and is now `TT`'s registrant, so an
owner-overlap oracle scores it 0.400 against `IR` -- "emphatically the same people", and
CORRECT, because it IS one continuous entity. It is simply **`TT`'s** entity, not `IR`'s.
Only the entity comparison settles that, which is why this oracle is asked "is C the same
entity as R" and NEVER "does C own symbol T".

FOUR ORACLES, IN PRIORITY ORDER. Each may only be overruled by one above it:

  1. `register`       -- `configs/sec/registrant_cutover.json`, 16 hand-evidenced chains read
                         through `load_registrants()`. The curated top layer; always wins.
  2. `manual`         -- `configs/sec/entity_lineage_manual.json`, the hand adjudications of
                         cases the automatic oracle cannot decide, each with prose evidence.
  3. `owner_overlap`  -- do the SAME reporting owners appear on both CIKs' Forms 3/4/5? A
                         reorganisation hands the same directors and officers to the new
                         registrant; a symbol reassignment to an unrelated company does not.
  4. `roster`         -- a roster CIK that ended in no group is recorded as its own entity,
                         so the table states a verdict for all 500 rather than staying silent.

⚠ `sharadar_permaticker` IS DELIBERATELY ABSENT, AND THAT IS A MEASURED RESULT. Sharadar
mints a NEW permaticker for a delisted predecessor -- DuPont E I is `DD1`/199769 against
DuPont de Nemours' 199776, Chubb Corp `CB1`/199850 against Chubb Ltd's 197681 -- so across
the 621 candidate CIKs exactly ZERO permatickers are shared by two of them, and on the 8
known predecessor pairs it says DIFFERENT 4 times and is absent 4 times (SAME: 0). It cannot
seed this table. It is used as a ticker->CIK CROSS-CHECK instead, so no row here carries
Sharadar-licensed content.

⚠ THE UNION-FIND IS CONSTRAINED, AND THAT CONSTRAINT IS THE MOST IMPORTANT LINE IN THE FILE.
A merge that would put TWO universe tickers in one entity is REJECTED and listed for hand
adjudication, never applied. `DD`, `DOW` and `CTVA` all descend from DowDuPont and shared
directors through 2017-2019 -- exactly the signal `owner_overlap` keys on -- but a spin-off
into two index members is TWO entities that share a past. Merging them would make the reverse
map collapse one onto the other and RELABEL EVERY ROW OF THE LOSER, which is the only failure
in this design that corrupts data rather than dropping it.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.config_paths import resolve_config_dir
from src.data_extract.utils.common.incremental import matches_stored
from src.data_extract.utils.common.registrant import Registrant, load_registrants
from src.data_extract.utils.common.run_manifest import record_run
from src.data_extract.utils.common.symbol_tenure import load_manual_symbol_tenure
from src.data_store.schema import Tables
from src.utils.string import normalise_ticker, pad_cik, pad_cik_series

logger = logging.getLogger(__name__)

#: `configs/sec/entity_lineage_manual.json`. Sibling of `registrant_cutover.json` and read
#: only here, so it is declared here rather than in `constants.py`.
MANUAL_CONFIG_SUBDIR = "sec"
MANUAL_CONFIG_FILENAME = "entity_lineage_manual.json"

#: Oracle 3's rule, verbatim from the research that hand-labelled 63 groups and got 55 right:
#:   shared == 0                              -> UNRELATED, no lineage row
#:   jaccard >= 0.05 OR shared >= 5           -> SAME ENTITY
#:   anything between                         -> GREY BAND, which the builder REFUSES to
#:                                               decide; it must carry a curated row or the
#:                                               build raises.
#: Two thresholds and not one because the two failure shapes differ: a huge board dilutes the
#: jaccard on a genuine predecessor, and a tiny board inflates it on an unrelated pair.
OVERLAP_JACCARD_SAME = 0.05
OVERLAP_SHARED_SAME = 5

SOURCE_PRIORITY = ("register", "manual", "owner_overlap", "roster")

#: The key inside `entity_lineage_manual.json` holding the D19 cross-check's exception list.
#: Underscore-prefixed so the lineage loader skips it -- it is a different shape and a
#: different question, but it belongs in the same curated file because it is the same kind of
#: hand verdict about identity.
D19_ALLOWLIST_KEY = "_d19_allowlist"


class TwoUniverseTickersOneEntityError(ValueError):
    """A merge would put two INVESTABLE tickers in one entity.

    Raised rather than applied: `universe_ticker_by_entity` would collapse one ticker onto
    the other and relabel every row of the loser. A spin-off into two index members is two
    entities that share a past, so the correct fix is a curated row, never a wider merge.
    """


class UndecidedGreyBandError(ValueError):
    """An owner-overlap score landed between the thresholds and no curated row decides it.

    The oracle is explicitly not allowed to break a tie: the 7 grey-band groups measured in
    2026-09 include both a real predecessor (`COHR`) and a real reuse (`CEG`), and a rule
    that guessed either way would silently delete or silently import a decade of filings.
    """


class ManualTenureEntityError(ValueError):
    """A manual ticker interval names a CIK outside its canonical ticker's entity."""


class EntityRekeyError(RuntimeError):
    """A newly discovered older CIK would silently change an existing entity ID."""

    def __init__(self, impacts: list[dict[str, object]], manifest: Path) -> None:
        self.impacts = impacts
        self.manifest = manifest
        super().__init__(
            f"entity_lineage: {len(impacts)} older-CIK rekey(s) detected; no table was written. "
            f"Review {manifest} and approve a separately scoped migration."
        )


@dataclass
class _Union:
    """Union-find over CIKs, refusing any merge that would join two roster CIKs."""

    roster_ciks: frozenset[str]
    parent: dict[str, str] = field(default_factory=dict)
    blocked: list[tuple[str, str, str, float | None]] = field(default_factory=list)

    def add(self, cik: str) -> None:
        self.parent.setdefault(cik, cik)

    def find(self, cik: str) -> str:
        self.add(cik)
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
        """Merge `a` and `b`; returns False (and records the pair) if the merge is refused."""
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return True
        if len(self.roster_members(a) | self.roster_members(b)) > 1:
            self.blocked.append((a, b, source, confidence))
            return False
        # the OLDEST cik roots the group, so `entity_id` is stable under merge order
        older, newer = sorted((root_a, root_b))
        self.parent[newer] = older
        return True

    def groups(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for cik in self.parent:
            out.setdefault(self.find(cik), set()).add(cik)
        return out


def entity_id_for(ciks: set[str]) -> str:
    """`"E" + the oldest (numerically smallest) CIK in the group`, zero-padded.

    A natural key: no allocation state, so two independent derivations agree. The known
    failure -- a group that later gains an OLDER cik shifts its id -- is acceptable because
    nothing outside the identity layer joins on `entity_id`, and the quarantine table records
    both the resolved and the expected id so a shift reads as a diff, not a silent re-verdict.
    """
    return "E" + min(ciks)


def entity_or_singleton(entity_by_cik: Mapping[str, str], cik: str) -> str:
    """The stored entity of a padded CIK; a CIK with no stored row is its own entity `E{cik}`."""
    return entity_by_cik.get(cik, f"E{cik}")


def entity_by_cik_map(lineage: pd.DataFrame) -> dict[str, str]:
    """`{padded cik: entity_id}` from `entity_lineage` rows."""
    return dict(zip(pad_cik_series(lineage["cik"]), lineage["entity_id"].astype(str), strict=False))


def roster_cik_map(roster: pd.DataFrame) -> dict[str, str]:
    """`{universe ticker: padded roster CIK}` from `sp500_tickers`, skipping a row with no CIK."""
    ciks = pad_cik_series(roster["cik"])
    has_cik = ciks.ne("")
    return dict(zip(roster.loc[has_cik, "ticker"].map(normalise_ticker), ciks[has_cik], strict=False))


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
    """`{ticker: why this ticker's roster CIK may disagree with symbol_tenure}`.

    Consumed by the D19 assertion. An entry is a hand reading that CLEARS a disagreement;
    anything not listed must raise, because an unexplained disagreement is the XOM class of
    defect -- a roster CIK, sourced from Wikipedia, pointing at the wrong company.
    """
    allow = _manual_blob(config_dir).get(D19_ALLOWLIST_KEY, {})
    return {t: str(why) for t, why in allow.items() if not t.startswith("_")}


def load_manual_lineage(config_dir: str | None = None) -> dict[str, dict]:
    """`configs/sec/entity_lineage_manual.json` -> `{key: entry}`; `{}` when absent.

    Two entry shapes, because the grey band produces two kinds of verdict:
      * `"same_entity": [cik, ...]` -- these CIKs are one company;
      * `"own_entity": [cik, ...]`  -- this CIK is NOT the ticker it filed under, recorded so
        the verdict is stated rather than merely absent.
    Both require non-empty `evidence`, for the reason `registrant.py` gives: an undocumented
    identity decision is a guess that silently deletes or imports a decade of filings.
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


# --------------------------------------------------------------------------- #
# Oracle 3 -- reporting-owner overlap, from the same cached zips               #
# --------------------------------------------------------------------------- #
def derive_owner_sets(owner_pairs: pd.DataFrame, ciks: frozenset[str]) -> dict[str, set[str]]:
    """`{issuer_cik: {padded reporting owner CIKs}}` for `ciks` only, from a Form 345 cache scan's
    (issuer_cik, owner_cik_raw) pairs.

    Keyed on the ISSUER CIK across every symbol it ever filed under, not on (symbol, cik): the
    question is "is C the same company as R", and one symbol's filings would answer a narrower one.
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
    """`unrelated` | `same` | `grey` -- see `OVERLAP_JACCARD_SAME` for why two thresholds."""
    if shared == 0:
        return "unrelated"
    if jaccard >= OVERLAP_JACCARD_SAME or shared >= OVERLAP_SHARED_SAME:
        return "same"
    return "grey"


# --------------------------------------------------------------------------- #
# The build                                                                     #
# --------------------------------------------------------------------------- #
def candidate_ciks(tenure: pd.DataFrame, roster: pd.DataFrame) -> tuple[frozenset[str], dict[str, set[str]], dict[str, str]]:
    """(all candidates, {ticker: CIKs seen under its symbol}, {ticker: roster CIK}).

    Candidates are every issuer CIK that ever filed under a TODAY-universe symbol, union
    every roster CIK. Every other CIK in EDGAR needs no row: it is a singleton by default,
    which is exactly the verdict `owns()` needs from it.
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


def detect_older_cik_rekeys(existing: pd.DataFrame, candidate: pd.DataFrame) -> list[dict[str, object]]:
    """Return stable-group ID changes caused by a newly joined numerically older CIK."""
    old = existing[["cik", "entity_id"]].drop_duplicates().copy()
    new = candidate[["cik", "entity_id"]].drop_duplicates().copy()
    old["cik"] = pad_cik_series(old["cik"])
    new["cik"] = pad_cik_series(new["cik"])
    old_map = dict(zip(old["cik"], old["entity_id"].astype(str), strict=False))
    new_map = dict(zip(new["cik"], new["entity_id"].astype(str), strict=False))
    new_members = {str(entity): list(members) for entity, members in new.groupby("entity_id")["cik"].agg(lambda values: sorted(set(values))).items()}
    impacts: list[dict[str, object]] = []
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


def _write_rekey_manifest(config_dir: str | None, impacts: list[dict[str, object]]) -> Path:
    """Persist the stop-condition evidence without changing any identity table."""
    repo_root = Path(resolve_config_dir(config_dir)).resolve().parent
    path = repo_root / "reports" / "validate" / "identity-rekey-impact.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "blocked_before_write",
        "reason": "newly discovered older CIK would change a derived entity_id",
        "source_tables": ["symbol_tenure", "entity_lineage"],
        "dependent_tables": ["sec_insider_transactions_quarantine"],
        "impacts": impacts,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def derive_entity_lineage(
    tenure: pd.DataFrame, roster: pd.DataFrame, owner_pairs: pd.DataFrame, config_dir: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(`entity_lineage` rows, the blocked-merge report) from the tenure table, the roster and a
    Form 345 cache scan's (issuer_cik, owner_cik) pairs.

    Raises `UndecidedGreyBandError` when an automatic score lands between the thresholds and no
    curated row decides it. A blocked merge is RECORDED rather than raised, so one run reports
    the whole list instead of stopping at the first.
    """
    candidates, by_ticker, roster_cik = candidate_ciks(tenure, roster)
    roster_ciks = frozenset(roster_cik.values())
    union = _Union(roster_ciks=roster_ciks)
    for cik in candidates:
        union.add(cik)
    provenance = _Provenance()
    _apply_register(union, provenance, load_registrants(config_dir))
    _apply_manual(union, provenance, load_manual_lineage(config_dir))
    owners = derive_owner_sets(owner_pairs, candidates)
    grey, verdicts = _apply_owner_overlap(union, provenance, by_ticker, roster_cik, owners)
    _raise_grey_band(grey)
    out = _lineage_frame(union, provenance, roster_ciks)
    blocked = _blocked_frame(union, roster_cik)
    logger.info(
        "entity_lineage: %d candidate CIK(s) -> %d row(s) over %d entity(ies); "
        "sources %s; owner-overlap verdicts %s; %d merge(s) BLOCKED as two-universe-"
        "tickers",
        len(candidates),
        len(out),
        out["entity_id"].nunique(),
        dict(out["source"].value_counts()),
        dict(verdicts),
        len(blocked),
    )
    return out, blocked


def _apply_register(union: _Union, provenance: _Provenance, registrants: Mapping[str, Registrant]) -> None:
    """Oracle 1, highest priority: join every register chain and mark its CIKs curated."""
    for ticker, entry in sorted(registrants.items()):
        ciks = list(entry.all_ciks())
        for cik in ciks:
            union.add(cik)
        for other in ciks[1:]:
            union.union(ciks[0], other, source=f"register[{ticker}]")
        for segment in entry.segments:
            provenance.curated.add(segment.cik)
            provenance.claim(segment.cik, "register", None, f"{ticker} {entry.kind}: {segment.evidence}")


def _apply_manual(union: _Union, provenance: _Provenance, manual: Mapping[str, dict]) -> None:
    """Oracle 2: the curated adjudications; `same_entity` CIKs are joined, all are marked curated."""
    for key, entry in sorted(manual.items()):
        same, own = entry["same_entity"], entry["own_entity"]
        for cik in same + own:
            union.add(cik)
            provenance.curated.add(cik)
            provenance.claim(cik, "manual", None, f"{key}: {entry['evidence']}")
        for other in same[1:]:
            union.union(same[0], other, source=f"manual[{key}]")


def _apply_owner_overlap(
    union: _Union,
    provenance: _Provenance,
    by_ticker: Mapping[str, set[str]],
    roster_cik: Mapping[str, str],
    owners: Mapping[str, set[str]],
) -> tuple[list[tuple[str, str, str, int, float]], Counter]:
    """Oracle 3: score each non-roster candidate against its ticker's roster CIK and merge `same`.

    A CIK a curated layer spoke for (or already joined to the roster CIK) is not re-opened: the
    register and the manual file rank above this oracle. Returns the grey-band pairs and the
    verdict counts.
    """
    grey: list[tuple[str, str, str, int, float]] = []
    verdicts: Counter = Counter()
    pairs = [(ticker, cik, roster_cik[ticker]) for ticker in sorted(by_ticker) for cik in sorted(by_ticker[ticker]) if cik != roster_cik[ticker]]
    for ticker, cik, home in pairs:
        shared, jaccard = score_overlap(owners.get(cik, set()), owners.get(home, set()))
        verdict = classify_overlap(shared, jaccard)
        if cik in provenance.curated or union.find(cik) == union.find(home):
            verdicts[f"{verdict} (pre-decided)"] += 1
            continue
        verdicts[verdict] += 1
        if verdict == "grey":
            grey.append((ticker, cik, home, shared, jaccard))
        elif verdict == "same" and union.union(cik, home, source=f"owner_overlap[{ticker}]", confidence=jaccard):
            provenance.claim(cik, "owner_overlap", jaccard, f"{ticker}: {shared} reporting owner(s) shared with {home}, jaccard {jaccard:.3f}")
    return grey, verdicts


def _raise_grey_band(grey: list[tuple[str, str, str, int, float]]) -> None:
    """Refuse to break a grey-band tie: every such pair needs a curated row."""
    if not grey:
        return
    listed = "; ".join(f"{t}/{c} (shared={s}, jaccard={j:.3f})" for t, c, _, s, j in grey)
    raise UndecidedGreyBandError(
        f"{len(grey)} owner-overlap score(s) landed between shared>0 and "
        f"jaccard>={OVERLAP_JACCARD_SAME}/shared>={OVERLAP_SHARED_SAME}, and no curated "
        f"row decides them: {listed}. Add each to {MANUAL_CONFIG_FILENAME} (or to the "
        "register, when it is a real chain with prose evidence). The oracle is NOT "
        "allowed to break this tie -- the measured grey band holds both a genuine "
        "predecessor (COHR) and a genuine symbol reuse (CEG)."
    )


def _lineage_frame(union: _Union, provenance: _Provenance, roster_ciks: frozenset[str]) -> pd.DataFrame:
    """One row per CIK in a non-singleton group, per roster CIK and per curated CIK.

    Curated CIKs are stored even when singleton, so an `own_entity` verdict stays visible instead
    of reading like a CIK nobody looked at.
    """
    groups = union.groups()
    entity_of = {cik: entity_id_for(members) for members in groups.values() for cik in members}
    non_singleton = {cik for members in groups.values() if len(members) > 1 for cik in members}
    rows = []
    for cik in sorted(non_singleton | roster_ciks | provenance.curated):
        source, confidence, evidence = provenance.verdicts.get(cik, ("roster", None, "roster CIK with no predecessor found by any oracle"))
        rows.append({"cik": cik, "entity_id": entity_of[cik], "source": source, "confidence": confidence, "evidence": evidence})
    return pd.DataFrame(rows, columns=["cik", "entity_id", "source", "confidence", "evidence"])


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


def build_entity_lineage(
    context: Context,
    tenure: pd.DataFrame,
    owner_pairs: pd.DataFrame,
    config_dir: str | None = None,
    *,
    approved_rekeys: frozenset[tuple[str, str]] = frozenset(),
) -> pd.DataFrame:
    """Derive `entity_lineage` from the materialized `symbol_tenure` frame and a cache scan's
    owner pairs, and REPLACE the table unless it is unchanged; returns the derived frame.

    An older CIK changes the natural entity ID. Such a write stays fail-closed unless every
    observed ``(old_entity_id, new_entity_id)`` pair is acknowledged exactly for this call.
    """
    roster = context.store.load(Tables.sp500_tickers)
    assert roster is not None
    existing = context.store.load(Tables.entity_lineage, project=True, optional=True)
    out, blocked = derive_entity_lineage(tenure, roster, owner_pairs, config_dir)
    manual_tenure = load_manual_symbol_tenure(config_dir or context.config_dir)
    validate_manual_tenure_entities(manual_tenure, out, roster)
    context.log.info(
        f"entity_lineage: {len(manual_tenure)} manual symbol interval(s) resolve to "
        f"their {manual_tenure['canonical_ticker'].nunique()} canonical entity(ies)"
    )
    if not blocked.empty:
        logger.warning(
            "entity_lineage: %d merge(s) refused because they would put two universe tickers in one entity:\n%s",
            len(blocked),
            blocked.to_string(index=False),
        )
    if existing is None:
        context.log.info(f"entity_lineage: cold build with {len(out)} CIK assignment(s) over {out['entity_id'].nunique()} entity(ies)")
    else:
        _check_rekeys(context, existing, out, config_dir, approved_rekeys)
        _log_changed_assignments(context, existing, out, roster)
    unchanged = matches_stored(existing, out, Tables.entity_lineage)
    written = 0 if unchanged else context.store.replace(Tables.entity_lineage, out)
    record_run(context, Tables.entity_lineage, 0, written, is_full_rescan=True)
    if unchanged:
        logger.info("entity_lineage: unchanged (%d row(s)); replace skipped", len(out))
    else:
        logger.info("entity_lineage: wrote %d row(s)", written)
    return out


def _check_rekeys(
    context: Context, existing: pd.DataFrame, out: pd.DataFrame, config_dir: str | None, approved_rekeys: frozenset[tuple[str, str]]
) -> None:
    """Stop before any write when an older CIK would rename an entity without exact approval."""
    rekeys = detect_older_cik_rekeys(existing, out)
    actual_rekeys = frozenset((str(impact["old_entity_id"]), str(impact["new_entity_id"])) for impact in rekeys)
    if rekeys and actual_rekeys != approved_rekeys:
        manifest = _write_rekey_manifest(config_dir or str(context.config_dir), rekeys)
        context.log.error(f"entity_lineage: older-CIK rekey stop; wrote impact manifest to {manifest}")
        raise EntityRekeyError(rekeys, manifest)
    if actual_rekeys:
        approved = ", ".join(f"{old}->{new}" for old, new in sorted(actual_rekeys))
        context.log.warning(f"entity_lineage: applying explicitly approved older-CIK rekey(s): {approved}")


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
