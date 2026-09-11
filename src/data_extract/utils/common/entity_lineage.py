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
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.config_paths import resolve_config_dir
from src.data_extract.utils.common.registrant import load_registrants
from src.data_extract.utils.common.run_manifest import record_run
from src.data_extract.utils.common.symbol_tenure import SUBMISSION_MEMBER, _quarter_of
from src.data_store.schema import Tables

logger = logging.getLogger(__name__)

#: `configs/sec/entity_lineage_manual.json`. Sibling of `registrant_cutover.json` and read
#: only here, so it is declared here rather than in `constants.py`.
MANUAL_CONFIG_SUBDIR = "sec"
MANUAL_CONFIG_FILENAME = "entity_lineage_manual.json"

#: The Form 345 zip member naming each filing's reporting owners.
OWNER_MEMBER = "REPORTINGOWNER.TSV"

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


class TwoUniverseTickersOneEntity(ValueError):
    """A merge would put two INVESTABLE tickers in one entity.

    Raised rather than applied: `universe_ticker_by_entity` would collapse one ticker onto
    the other and relabel every row of the loser. A spin-off into two index members is two
    entities that share a past, so the correct fix is a curated row, never a wider merge.
    """


class UndecidedGreyBand(ValueError):
    """An owner-overlap score landed between the thresholds and no curated row decides it.

    The oracle is explicitly not allowed to break a tie: the 7 grey-band groups measured in
    2026-09 include both a real predecessor (`COHR`) and a real reuse (`CEG`), and a rule
    that guessed either way would silently delete or silently import a decade of filings.
    """


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
        while self.parent[cik] != root:            # path compression
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


def _manual_blob(config_dir: str | None) -> dict:
    """`configs/sec/entity_lineage_manual.json` as raw JSON; `{}` when absent."""
    path = (Path(resolve_config_dir(config_dir)) / MANUAL_CONFIG_SUBDIR
            / MANUAL_CONFIG_FILENAME)
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
        same = [str(c).strip().zfill(10) for c in entry.get("same_entity", [])]
        own = [str(c).strip().zfill(10) for c in entry.get("own_entity", [])]
        if not same and not own:
            raise ValueError(f"entity_lineage_manual[{key}]: needs `same_entity` or "
                             "`own_entity`; an entry that asserts nothing decides nothing.")
        if len(same) == 1:
            raise ValueError(f"entity_lineage_manual[{key}]: `same_entity` needs >= 2 CIKs "
                             "-- one CIK is not a relationship. Use `own_entity`.")
        if not str(entry.get("evidence", "")).strip():
            raise ValueError(f"entity_lineage_manual[{key}]: empty `evidence`. An "
                             "undocumented verdict is a guess that moves a decade of rows.")
        out[key] = {"same_entity": same, "own_entity": own,
                    "evidence": str(entry["evidence"]).strip()}
    return out


# --------------------------------------------------------------------------- #
# Oracle 3 -- reporting-owner overlap, from the same cached zips               #
# --------------------------------------------------------------------------- #
def derive_owner_sets(cache: Path, ciks: frozenset[str]) -> dict[str, set[str]]:
    """`{issuer_cik: {reporting owner CIKs}}` for `ciks` only, from the cached Form 345 zips.

    Keyed on the ISSUER CIK across every symbol it ever filed under, not on (symbol, cik):
    the question is "is C the same company as R", and restricting C's owners to the filings
    made under one symbol would answer a narrower question with less evidence.
    """
    owners: dict[str, set[str]] = {c: set() for c in ciks}
    zips = sorted(p for p in cache.glob("*.zip") if _quarter_of(p) is not None)
    read = 0
    for path in zips:
        try:
            archive = zipfile.ZipFile(path)
        except zipfile.BadZipFile:
            logger.warning("entity_lineage: %s is a corrupt zip -> SKIPPED", path.name)
            continue
        with archive:
            names = {n.upper(): n for n in archive.namelist()}
            if SUBMISSION_MEMBER not in names or OWNER_MEMBER not in names:
                continue
            with archive.open(names[SUBMISSION_MEMBER]) as handle:
                sub = pd.read_csv(handle, sep="\t", dtype=str, low_memory=False,
                                  usecols=lambda c: c.upper() in
                                  {"ACCESSION_NUMBER", "ISSUERCIK"})
            sub.columns = [c.upper() for c in sub.columns]
            sub["ISSUERCIK"] = sub["ISSUERCIK"].astype("string").str.strip().str.zfill(10)
            sub = sub[sub["ISSUERCIK"].isin(ciks)]
            if sub.empty:
                continue
            issuer_of = dict(zip(sub["ACCESSION_NUMBER"].astype(str), sub["ISSUERCIK"]))
            with archive.open(names[OWNER_MEMBER]) as handle:
                own = pd.read_csv(handle, sep="\t", dtype=str, low_memory=False,
                                  usecols=lambda c: c.upper() in
                                  {"ACCESSION_NUMBER", "RPTOWNERCIK"})
            own.columns = [c.upper() for c in own.columns]
            own["ISSUER"] = own["ACCESSION_NUMBER"].astype(str).map(issuer_of)
            own = own.dropna(subset=["ISSUER", "RPTOWNERCIK"])
            own["RPTOWNERCIK"] = own["RPTOWNERCIK"].astype("string").str.strip().str.zfill(10)
            for issuer, grp in own.groupby("ISSUER", sort=False):
                owners[issuer].update(grp["RPTOWNERCIK"])
            read += len(own)
    logger.info("entity_lineage: owner sets for %d CIK(s) from %d quarter(s) (%d owner "
                "rows matched); %d CIK(s) have no Form 345 owner at all",
                len(ciks), len(zips), read, sum(1 for v in owners.values() if not v))
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
def candidate_ciks(tenure: pd.DataFrame, roster: pd.DataFrame
                   ) -> tuple[frozenset[str], dict[str, set[str]], dict[str, str]]:
    """(all candidates, {ticker: CIKs seen under its symbol}, {ticker: roster CIK}).

    Candidates are every issuer CIK that ever filed under a TODAY-universe symbol, union
    every roster CIK. Every other CIK in EDGAR needs no row: it is a singleton by default,
    which is exactly the verdict `owns()` needs from it.
    """
    roster_cik = {str(t): str(c).strip().zfill(10)
                  for t, c in zip(roster["ticker"], roster["cik"]) if pd.notna(c)}
    universe = set(roster_cik)
    seen = tenure[tenure["symbol"].astype(str).isin(universe)]
    by_ticker: dict[str, set[str]] = {t: {roster_cik[t]} for t in roster_cik}
    for symbol, cik in zip(seen["symbol"].astype(str), seen["issuer_cik"].astype(str)):
        by_ticker[symbol].add(cik)
    return frozenset().union(*by_ticker.values()), by_ticker, roster_cik


def derive_entity_lineage(cache: Path, tenure: pd.DataFrame, roster: pd.DataFrame,
                          config_dir: str | None = None,
                          owners: dict[str, set[str]] | None = None
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(`entity_lineage` rows, the blocked-merge report).

    Raises `UndecidedGreyBand` when an automatic score lands between the thresholds and no
    curated row decides it. A blocked merge is RECORDED rather than raised, so one run reports
    the whole list instead of stopping at the first.

    `owners` is the reporting-owner map; it is derived from `cache` when not supplied. Passing
    it in is how a test reaches the grey band without a 45-second read of 81 real zips, and
    how a caller that already has the map avoids deriving it twice.
    """
    candidates, by_ticker, roster_cik = candidate_ciks(tenure, roster)
    roster_ciks = frozenset(roster_cik.values())
    union = _Union(roster_ciks=roster_ciks)
    for cik in candidates:
        union.add(cik)

    provenance: dict[str, tuple[str, float | None, str]] = {}

    def claim(cik: str, source: str, confidence: float | None, evidence: str) -> None:
        """Record the HIGHEST-priority oracle that spoke for `cik`."""
        prior = provenance.get(cik)
        if prior is None or SOURCE_PRIORITY.index(source) < SOURCE_PRIORITY.index(prior[0]):
            provenance[cik] = (source, confidence, evidence)

    # Every CIK a CURATED layer has spoken for. Oracle 3 must not re-open one: the register
    # and the manual file are hand-evidenced and rank above it, and without this the
    # `IR`/`TT` case re-enters the grey band through the back door -- `0001466258` is TT's
    # registrant, so scoring it against IR's roster CIK necessarily lands between the
    # thresholds and there is no IR-specific verdict that could ever resolve it.
    curated: set[str] = set()

    # --- oracle 1: the register, highest priority --------------------------- #
    registrants = load_registrants(config_dir)
    for ticker, entry in sorted(registrants.items()):
        ciks = list(entry.all_ciks())
        for cik in ciks:
            union.add(cik)
        for other in ciks[1:]:
            union.union(ciks[0], other, source=f"register[{ticker}]")
        for segment in entry.segments:
            curated.add(segment.cik)
            claim(segment.cik, "register", None,
                  f"{ticker} {entry.kind}: {segment.evidence}")

    # --- oracle 2: the curated adjudications -------------------------------- #
    manual = load_manual_lineage(config_dir)
    for key, entry in sorted(manual.items()):
        same, own = entry["same_entity"], entry["own_entity"]
        for cik in same + own:
            union.add(cik)
            curated.add(cik)
            claim(cik, "manual", None, f"{key}: {entry['evidence']}")
        for other in same[1:]:
            union.union(same[0], other, source=f"manual[{key}]")

    # --- oracle 3: reporting-owner overlap ---------------------------------- #
    owners = derive_owner_sets(cache, candidates) if owners is None else owners
    grey: list[tuple[str, str, str, int, float]] = []
    verdicts: Counter = Counter()
    for ticker in sorted(by_ticker):
        home = roster_cik[ticker]
        for cik in sorted(by_ticker[ticker]):
            if cik == home:
                continue
            shared, jaccard = score_overlap(owners.get(cik, set()), owners.get(home, set()))
            verdict = classify_overlap(shared, jaccard)
            # the curated layer has already spoken for this CIK; the oracle does not re-open it
            if cik in curated or union.find(cik) == union.find(home):
                verdicts[f"{verdict} (pre-decided)"] += 1
                continue
            verdicts[verdict] += 1
            if verdict == "grey":
                grey.append((ticker, cik, home, shared, jaccard))
            elif verdict == "same" and union.union(cik, home, source=f"owner_overlap[{ticker}]",
                                                   confidence=jaccard):
                claim(cik, "owner_overlap", jaccard,
                      f"{ticker}: {shared} reporting owner(s) shared with {home}, "
                      f"jaccard {jaccard:.3f}")
    if grey:
        listed = "; ".join(f"{t}/{c} (shared={s}, jaccard={j:.3f})"
                           for t, c, _, s, j in grey)
        raise UndecidedGreyBand(
            f"{len(grey)} owner-overlap score(s) landed between shared>0 and "
            f"jaccard>={OVERLAP_JACCARD_SAME}/shared>={OVERLAP_SHARED_SAME}, and no curated "
            f"row decides them: {listed}. Add each to {MANUAL_CONFIG_FILENAME} (or to the "
            "register, when it is a real chain with prose evidence). The oracle is NOT "
            "allowed to break this tie -- the measured grey band holds both a genuine "
            "predecessor (COHR) and a genuine symbol reuse (CEG).")

    # --- mint ids ----------------------------------------------------------- #
    groups = union.groups()
    entity_of = {cik: entity_id_for(members)
                 for members in groups.values() for cik in members}
    non_singleton = {cik for members in groups.values() if len(members) > 1 for cik in members}

    # `curated` is in the stored set on purpose. A hand verdict of "this CIK is NOT that
    # ticker" (`own_entity`) leaves a SINGLETON, which the default already produces -- so
    # without this the adjudication would vanish from the table and be indistinguishable from
    # a CIK nobody ever looked at. CEG's old Constellation Energy Group is exactly that case.
    rows = []
    for cik in sorted(non_singleton | roster_ciks | curated):
        source, confidence, evidence = provenance.get(
            cik, ("roster", None, "roster CIK with no predecessor found by any oracle"))
        rows.append({"cik": cik, "entity_id": entity_of[cik], "source": source,
                     "confidence": confidence, "evidence": evidence})
    out = pd.DataFrame(rows, columns=["cik", "entity_id", "source", "confidence", "evidence"])

    blocked = pd.DataFrame(
        [{"cik_a": a, "cik_b": b, "proposed_by": source, "confidence": conf,
          "tickers_a": ",".join(sorted(t for t, c in roster_cik.items()
                                       if c in union.roster_members(a))),
          "tickers_b": ",".join(sorted(t for t, c in roster_cik.items()
                                       if c in union.roster_members(b)))}
         for a, b, source, conf in union.blocked],
        columns=["cik_a", "cik_b", "proposed_by", "confidence", "tickers_a", "tickers_b"])

    logger.info("entity_lineage: %d candidate CIK(s) -> %d row(s) over %d entity(ies); "
                "sources %s; owner-overlap verdicts %s; %d merge(s) BLOCKED as two-universe-"
                "tickers", len(candidates), len(out), out["entity_id"].nunique(),
                dict(out["source"].value_counts()), dict(verdicts), len(blocked))
    return out, blocked


def build_entity_lineage(context: Context, cache: Path,
                         config_dir: str | None = None) -> pd.DataFrame:
    """Derive and REPLACE `entity_lineage`; returns the frame written."""
    tenure = context.store.load(Tables.symbol_tenure, project=True)
    roster = context.store.load(Tables.sp500_tickers)
    out, blocked = derive_entity_lineage(cache, tenure, roster, config_dir)
    if not blocked.empty:
        logger.warning("entity_lineage: %d merge(s) refused because they would put two "
                       "universe tickers in one entity:\n%s", len(blocked),
                       blocked.to_string(index=False))
    written = context.store.replace(Tables.entity_lineage, out)
    record_run(context, Tables.entity_lineage, 0, written, is_full_rescan=True)
    logger.info("entity_lineage: wrote %d row(s)", written)
    return out
