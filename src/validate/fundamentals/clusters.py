"""
clusters.py  (src/validate/fundamentals/)
--------------------------------------------------------------------------------------------
Turn a ledger of findings into a RANKED LIST OF FIXABLE ISSUES.

## 11,926 findings were never 11,926 bugs

That is the measurement this module exists to answer, and it is worth stating precisely.
On run 2, 739 of 1,893 `(ticker, field)` series carried 8,160 of the 10,362 queue findings.
MCD `capex` alone tripped NINE checks for 54 findings. Nothing in the system said which fix
closed the most rows, so the queue was ordered by severity and then ALPHABETICALLY -- which
is to say, not ordered at all.

A CLUSTER is one `(ticker, field)` defect. `check_name` is evidence INSIDE it, never part of
its key: nine checks agreeing is not nine jobs, it is one job with nine witnesses, and the
corroboration is the strongest prior an agent gets before opening a filing.

## The score is a POLICY, not a fact, and the report says so every time

    cluster_score = (sum over findings of w(severity) * w(tier)) * corroboration(n_checks)

Volume enters naturally rather than through a fudge factor, and a cluster mixing tiers needs
no special case. The weights are module constants, they are printed in every report, and they
are meant to be retuned once a human has read a list and disagreed. That has already happened
once, which is what the third term is.

### Why corroboration multiplies rather than adds

The first ranking was volume-only, and it put the wrong things on top. Measured on calibration
run 3: HCA `minorityInterest` led with 62 findings from **two** checks (score 244), while MCD
`capex` -- 55 findings from **ten independent checks** -- sat at 148 and never reached the
menu. That is backwards. One check firing 62 times is one opinion repeated; ten checks
agreeing is ten different arguments for the same conclusion, and it is the strongest prior an
agent gets before opening a filing.

So the corroboration term is a MULTIPLIER, not a bonus: it scales whatever the cluster is
worth rather than adding a constant that a large enough pile of single-check findings would
drown. At `CORROBORATION_BONUS = 0.25` a ten-check cluster is worth 3.25x a one-check cluster
of the same weight -- MCD `capex` moves 148 -> 481 and leads; HCA moves 244 -> 305.

## Families, and the DQC_0118 lesson made mechanical

Clusters roll up by FIELD. The point is not tidiness -- it is that breadth is diagnostic:

  * one ticker with a broken `capex`               -> fix the filer's resolution
  * forty tickers with a broken `incomeTaxExpense` -> fix the FIELD. The catalogue is wrong.

XBRL-US says it plainly of DQC_0118: *"inconsistencies reported to filers can be overwhelming
as many don't represent real errors."* A family spanning most of the roster is far more likely
to be our specification than a simultaneous failure by forty independent filers, and
`routing_hint` says so BEFORE an agent has spent an hour reading 10-Ks.

## SETTLEMENT: what it takes to claim a defect is closed

A cluster does not have to reach zero to be settled. MCD `capex` went from 55 findings to 4
and every one of the 4 is benign -- one weighted-zero `info`, two `peer_ratio` on a documented
blind spot, one `series_shape` coverage gap. A strict set difference calls that OPEN forever,
which is how a real fix ends up with no record anywhere.

So settlement has four conditions and each exists to block a specific way of faking it:

  1. no UNWAIVED queue-severity finding is left. `info` never needs a waiver -- nothing reads
     it as work, so it cannot be hiding any;
  2. every waiver is still WITHIN the size it was decided against. One that grew has expired,
     and a cluster resting on an expired judgement is REOPENED, not settled;
  3. a FIX ROW EXISTS at this scope. Without this, waiving every check one at a time
     manufactures a settlement with nobody having fixed anything -- the deleted suppression
     register, reassembled from parts;
  4. that fix row REDUCED THE QUEUE. A row that moved nothing stays on the record and proves
     nothing.

Fail 3 or 4 with 1 and 2 satisfied and the cluster reads WONTFIX: tolerated, not solved, and
it says so. That is why no new status word was added -- a fully-waived, unfixed cluster
already IS what `wontfix` means.

NOTHING HERE SUBTRACTS A ROW. A waived finding is still written, still counted, still fires,
and still appears in `fundamentals_check`. Waivers and fixes are read when a report is
RENDERED. That is the property that keeps a row-count drop usable as proof, and a test pins
it directly by counting ledger rows with and without any of this present.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.validate.fundamentals.finding import (
    CRITICAL, HIGH, INFO, MEDIUM, QUEUE_SEVERITIES, SEVERITY_ORDER)

#: What `check_name` means in a waiver row when it is empty: the waiver covers the WHOLE
#: cluster. Mirrors `ledger.CLUSTER_WIDE`; re-declared rather than imported because
#: `clusters.py` is pure frame logic and must not depend on the store-reading layer.
CLUSTER_WIDE = ""

# --------------------------------------------------------------------------- #
# the scoring policy (D3/D4) -- starting values, printed, meant to be retuned  #
# --------------------------------------------------------------------------- #

#: What a TIER is worth. Tier IS the field-importance proxy, deliberately: Tier 1 runs on
#: `fundamentals_history`, which is the table the cube actually reads, so a Tier-1 defect is
#: already in the model. Tier 3 is an internal-consistency signal about the as-filed facts.
#: Weighting fields individually was rejected -- nobody has measured a per-field importance,
#: and an unmeasured weight would quietly become the ranking.
TIER_WEIGHTS: dict[int, int] = {1: 4, 2: 2, 3: 1}

#: What a SEVERITY is worth. `info` is ZERO, not small: an `info` finding is declared,
#: quantified and expected, and nothing reads it as work -- so a cluster made entirely of
#: `info` scores 0 and cannot outrank a real one on volume alone. That is the property that
#: keeps `restatement_ledger` and the benign gap codes out of the rankings.
SEVERITY_WEIGHTS: dict[str, int] = {CRITICAL: 4, HIGH: 2, MEDIUM: 1, INFO: 0}

#: What each ADDITIONAL agreeing check is worth, as a fraction of the cluster's base score.
#: 0.25 -> a 10-check cluster scores 3.25x a 1-check cluster of equal weight.
#:
#: Set from a measurement rather than a preference: on calibration run 3 the volume-only score
#: ranked HCA `minorityInterest` (62 findings, 2 checks, 244) above MCD `capex` (55 findings,
#: 10 checks, 148). Anything above ~0.09 reverses that pair; 0.25 does it decisively and keeps
#: the multiplier interpretable -- "each extra witness is worth a quarter of the case".
#:
#: Note what it does NOT do: it cannot rescue an all-`info` cluster, because the base score is
#: already 0 and 0 x anything is 0. Ten checks agreeing that something is benign is still benign.
CORROBORATION_BONUS = 0.25

#: A field family is OUR problem rather than the filers' when it spans at least this many
#: tickers AND this share of the run's roster. Both, not either: 5 tickers out of 500 is a
#: coincidence, and 30% of a 3-ticker run is one ticker. Constants, printed in the report,
#: tunable -- they encode a prior, not a law.
FAMILY_BREADTH_MIN_TICKERS = 5
FAMILY_BREADTH_MIN_SHARE = 0.30

LIKELY_CHECK = "likely-check-or-catalogue"
LIKELY_FILER = "likely-filer"

#: A cluster's derived state. `open` and `settled` are DERIVED from the ledger and are never
#: stored; only `wontfix` is a human's assertion, and `reopened` is that assertion expiring.
OPEN, WONTFIX, REOPENED, SETTLED = "open", "wontfix", "reopened", "settled"


@dataclass(frozen=True, slots=True)
class Cluster:
    """One `(ticker, field)` defect, with every witness to it and what they add up to."""

    cluster_id: str
    ticker: str
    field: str
    findings: int
    score: float
    #: check name -> how many findings it contributed. THE corroboration signal: N independent
    #: checks agreeing is a far stronger prior than one check firing N times.
    checks: tuple[tuple[str, int], ...]
    tiers: tuple[tuple[int, int], ...]
    severities: tuple[tuple[str, int], ...]
    period_range: str
    edgar_url: str | None
    why: str
    status: str = OPEN
    note: str = ""
    findings_at_decision: int | None = None
    #: Queue-severity findings a human has explicitly tolerated, and the checks they sit on.
    #: Carried on the cluster so a renderer can say "SETTLED (3 findings waived across 2
    #: checks)" without recomputing the waiver algebra and risking a different answer.
    waived_findings: int = 0
    waived_checks: tuple[str, ...] = ()
    first_seen: str = ""
    last_seen: str = ""
    runs_open: int = 1
    routing_hint: str = LIKELY_FILER
    family_breadth: str = ""

    @property
    def checks_agreeing(self) -> int:
        """How many DISTINCT checks fired on this cluster."""
        return len(self.checks)

    @property
    def worst_severity(self) -> str:
        """The worst severity present. What a reader scans for before the score."""
        present = {s for s, _n in self.severities}
        return next((s for s in SEVERITY_ORDER if s in present), INFO)

    @property
    def is_work(self) -> bool:
        """Does this cluster contain anything an agent works? A score of 0 is all-`info`."""
        return self.score > 0

    @property
    def waiver_summary(self) -> str:
        """`3 finding(s) waived across 2 check(s)`, or `clean` when nothing was tolerated.

        Rendered beside every settled cluster so the BASIS of a settlement is never invisible
        -- a settlement whose waivers are not printed is a suppression with better manners.
        """
        if not self.waived_findings:
            return "clean"
        return (f"{self.waived_findings} finding(s) waived across "
                f"{len(self.waived_checks)} check(s)")

    def as_dict(self) -> dict[str, Any]:
        """The agent handoff contract, exactly. See `.claude/agents/fundamentals-triage.md`.

        Emitted as JSON so agent B PARSES its work rather than scraping prose out of markdown.
        Every field here is one B cannot start without; a missing one is a defect in the
        report, and B is instructed to say so rather than improvise.
        """
        return {
            "cluster_id": self.cluster_id, "ticker": self.ticker, "field": self.field,
            "score": round(self.score, 2), "findings": self.findings,
            "checks_agreeing": dict(self.checks),
            "severity_mix": dict(self.severities), "tier_mix": {str(t): n for t, n in self.tiers},
            "period_range": self.period_range,
            "routing_hint": self.routing_hint, "family_breadth": self.family_breadth,
            "edgar_url": self.edgar_url, "why": self.why,
            "status": self.status, "note": self.note,
            # D8's reopen trigger. B needs the DISTANCE, not just the label: a wontfix sitting
            # exactly on its threshold reopens on the next finding and is not really settled.
            "findings_at_decision": self.findings_at_decision,
            "waived_findings": self.waived_findings,
            "waived_checks": list(self.waived_checks),
            "first_seen": self.first_seen, "last_seen": self.last_seen,
            "runs_open": self.runs_open,
        }


@dataclass(frozen=True, slots=True)
class SettledCluster:
    """One cluster that CLOSED, and the evidence for saying so.

    Richer than the bare `cluster_id` this used to be, because "settled" now has a basis that
    a reader is entitled to see: how many findings remain, how many of those were tolerated
    rather than fixed, and which intervention earned the claim. A renderer that had to
    recompute those would be a second copy of the settlement rule.
    """

    cluster_id: str
    ticker: str = ""
    field: str = ""
    #: Findings still on the ledger for this cluster in the LATEST run. Zero is the clean
    #: case; non-zero means the residue was assessed and tolerated, never hidden.
    findings_after: int = 0
    waived_findings: int = 0
    waived_checks: tuple[str, ...] = ()
    #: The `FixRecord` that carried the settlement. Typed loosely on purpose: `clusters.py`
    #: stays free of the ledger layer, and the renderer only reads attributes off it.
    fix: Any = None

    @property
    def basis(self) -> str:
        """`clean` or `3 finding(s) waived across 2 check(s)` -- printed beside the id."""
        if not self.waived_findings:
            return "clean"
        return (f"{self.waived_findings} finding(s) waived across "
                f"{len(self.waived_checks)} check(s)")


@dataclass(frozen=True, slots=True)
class Family:
    """Every cluster on one FIELD. Breadth is the diagnosis; see the module docstring."""

    field: str
    clusters: tuple[Cluster, ...]
    total_score: float
    tickers: int
    roster_size: int

    @property
    def findings(self) -> int:
        return sum(c.findings for c in self.clusters)

    @property
    def share(self) -> float:
        """The fraction of the run's roster this family touches. 0.0 on an empty roster."""
        return self.tickers / self.roster_size if self.roster_size else 0.0

    @property
    def routing_hint(self) -> str:
        """`likely-check-or-catalogue` or `likely-filer` -- what to challenge FIRST.

        Not a verdict. It is the DQC_0118 prior made mechanical: forty filers do not fail
        simultaneously and independently on one field, so a wide family is our specification
        until proven otherwise, and an agent that opens a 10-K first has spent the hour before
        it read the catalogue entry.
        """
        return (LIKELY_CHECK
                if self.tickers >= FAMILY_BREADTH_MIN_TICKERS
                and self.share >= FAMILY_BREADTH_MIN_SHARE
                else LIKELY_FILER)

    @property
    def breadth(self) -> str:
        """`47 of 54 tickers` -- the evidence behind the hint, never the hint alone."""
        return f"{self.tickers} of {self.roster_size} tickers"


# --------------------------------------------------------------------------- #
# building                                                                     #
# --------------------------------------------------------------------------- #

def build_clusters(findings: pd.DataFrame, *,
                   waivers: dict[str, dict[str, dict[str, Any]]] | None = None,
                   history: pd.DataFrame | None = None,
                   roster_size: int = 0) -> list[Cluster]:
    """Every cluster in `findings`, ranked by score, worst first.

    `waivers` is the NESTED map `{cluster_id: {check_name: row}}` from `Ledger.status_map`,
    `''` meaning the whole cluster. It applies a human's `wontfix` (and reopens it if the
    waived population grew -- D8). `history` carries `first_seen` / `last_seen` /
    `runs_open` across comparable runs. Both optional: a first run has neither and must still
    produce a usable ranking.

    SETTLED is never assigned here. It needs the PREVIOUS comparable run and a fix row, and
    neither is knowable from one run's findings -- see `settled_clusters`.
    """
    if findings is None or findings.empty:
        return []
    waivers = waivers or {}
    seen = _history_map(history)
    families = _family_sizes(findings, roster_size)

    clusters: list[Cluster] = []
    for cluster_id, rows in findings.groupby("cluster_id", sort=False):
        field = str(rows["field"].iloc[0] or "")
        family = families.get(field)
        clusters.append(_one(str(cluster_id), rows, waivers.get(str(cluster_id)),
                             seen.get(str(cluster_id)),
                             hint=family[0] if family else LIKELY_FILER,
                             breadth=family[1] if family else ""))
    return sorted(clusters, key=lambda c: (-c.score, -c.findings, c.ticker, c.field))


def build_families(clusters: list[Cluster], roster_size: int = 0) -> list[Family]:
    """Clusters rolled up by FIELD, ranked by summed score."""
    by_field: dict[str, list[Cluster]] = {}
    for cluster in clusters:
        by_field.setdefault(cluster.field, []).append(cluster)
    families = [
        Family(field=field, clusters=tuple(members),
               total_score=sum(c.score for c in members),
               tickers=len({c.ticker for c in members}), roster_size=roster_size)
        for field, members in by_field.items()]
    return sorted(families, key=lambda f: (-f.total_score, -f.findings, f.field))


def settled_clusters(previous: pd.DataFrame, latest: pd.DataFrame, *,
                     waivers: dict[str, dict[str, dict[str, Any]]] | None = None,
                     fixes: dict[str, Any] | None = None) -> list[SettledCluster]:
    """Every cluster that CLOSED between two comparable runs, with the basis for saying so.

    Sound only because both frames come from runs of ONE scope -- `ledger.comparable_runs`
    enforces that. A cluster missing from a NARROWER run did not close; it was never looked
    at.

    The four conditions are in the module docstring. Two of them are worth restating at the
    call site because they are the ones somebody will later be tempted to relax:

      * `fixes` is REQUIRED for a settlement. It maps `cluster_id -> FixRecord` and the
        caller builds it from `Ledger.qualifying_fix`, which is where the scope and
        improvement tests live. Passing `{}` settles NOTHING -- deliberately. Waivers alone
        settling a cluster is the suppression list reassembled one check at a time;
      * only QUEUE severities are considered. An `info` residue needs no waiver, because
        nothing reads `info` as work and it therefore cannot be concealing any.

    Backward-compatible in the case that matters: a cluster that went to ZERO findings still
    needs a fix row, because "it vanished and nobody knows why" is not proof either. What
    changed is that a cluster can now settle while still carrying assessed, tolerated rows.
    """
    if previous is None or previous.empty:
        return []
    waivers, fixes = waivers or {}, fixes or {}
    before = set(previous["cluster_id"].dropna().astype(str))
    has_latest = latest is not None and not latest.empty

    settled: list[SettledCluster] = []
    for cluster_id in sorted(before):
        fix = fixes.get(cluster_id)
        if fix is None:                      # condition 3+4, tested by the caller's predicate
            continue
        rows = (latest[latest["cluster_id"].astype(str) == cluster_id]
                if has_latest else latest)
        rows = rows if rows is not None and not rows.empty else None
        if rows is None:                                    # closed outright, nothing to weigh
            settled.append(SettledCluster(cluster_id=cluster_id, fix=fix))
            continue
        entries = waivers.get(cluster_id, {})
        unwaived, waived, checks = _weigh_residue(rows, entries)
        if unwaived or _expired(rows, entries, len(rows)):   # conditions 1 and 2
            continue
        settled.append(SettledCluster(
            cluster_id=cluster_id, ticker=str(rows["ticker"].iloc[0]),
            field=str(rows["field"].iloc[0] or ""), findings_after=len(rows),
            waived_findings=waived, waived_checks=checks, fix=fix))
    return settled


def derive_status(cluster_id: str, findings: int,
                  waivers: dict[str, dict[str, Any]] | None,
                  check_counts: dict[str, int] | None = None,
                  unwaived_queue: int = 0) -> tuple[str, str, int | None]:
    """`(status, note, findings_at_decision)` for one cluster. D6 and D8.

    THREE VALUES, and no more (decision 2). `open` unless a human has written a `wontfix` --
    and that `wontfix` EXPIRES BY ITSELF the moment the population it covers grows past the
    size that was actually assessed. A judgement made about 3 findings is not a judgement
    about 30, and nobody should have to remember to revisit it.

    That auto-reopen is what replaces the deleted register's STALENESS REPORT, which listed
    every settled finding whose check had stopped firing so the register "decayed visibly
    instead of accumulating suppressions nobody can justify". The same job, done by the data
    rather than by a reader noticing a line in a report: a decision here cannot outlive the
    evidence it was made on, and the report's wontfix footer is never omitted.

    `waivers` is PER CHECK now (`''` = the whole cluster), and each entry expires against its
    OWN population: a cluster-wide entry against `findings`, a per-check entry against that
    check's count in `check_counts`. Measuring a `peer_ratio` waiver against the cluster total
    would reopen it every time an unrelated check fired once more, which is not what anybody
    decided.

    `unwaived_queue` is what keeps a PARTIAL waiver honest: with real work still uncovered the
    cluster is plainly OPEN, whatever has been tolerated beside it. No SETTLED is returned
    here -- that needs the previous run and a fix row, neither visible from one cluster.
    """
    if not waivers:
        return OPEN, "", None
    counts = check_counts or {}
    note, at_decision = _first_note(waivers, findings, counts)
    live = {check: entry for check, entry in waivers.items()
            if str(entry.get("status") or "") == WONTFIX}
    if not live:
        return OPEN, note, at_decision

    for check, entry in sorted(live.items()):
        size = findings if check == CLUSTER_WIDE else counts.get(check, 0)
        decided = entry.get("findings_at_decision")
        decided = int(decided) if pd.notna(decided) else None
        if decided is not None and size > decided:
            return REOPENED, str(entry.get("note") or ""), decided
    return (OPEN if unwaived_queue else WONTFIX), note, at_decision


# --------------------------------------------------------------------------- #
# internals                                                                    #
# --------------------------------------------------------------------------- #

def _one(cluster_id: str, rows: pd.DataFrame, waivers: dict[str, dict[str, Any]] | None,
         seen: dict[str, Any] | None, *, hint: str, breadth: str) -> Cluster:
    """One cluster from its findings. The score is computed here and nowhere else."""
    severities = Counter(str(s) for s in rows["severity"])
    tiers = Counter(int(t) for t in rows["tier"].dropna())
    checks = Counter(str(c) for c in rows["check_name"])
    base = float(sum(SEVERITY_WEIGHTS.get(str(row.severity), 0)
                     * TIER_WEIGHTS.get(int(row.tier) if pd.notna(row.tier) else 0, 0)
                     for row in rows.itertuples()))
    score = base * corroboration(len(checks))

    worst = _worst_row(rows)
    entries = waivers or {}
    unwaived, waived, waived_checks = _weigh_residue(rows, entries)
    state, note, at_decision = derive_status(cluster_id, len(rows), entries,
                                             dict(checks), unwaived)
    return Cluster(
        cluster_id=cluster_id,
        ticker=str(rows["ticker"].iloc[0]), field=str(rows["field"].iloc[0] or ""),
        findings=len(rows), score=score,
        checks=tuple(sorted(checks.items(), key=lambda kv: (-kv[1], kv[0]))),
        tiers=tuple(sorted(tiers.items())),
        severities=tuple((s, severities[s]) for s in SEVERITY_ORDER if severities.get(s)),
        period_range=_period_range(rows["period_key"]),
        edgar_url=_first_url(rows),
        why=_why(worst.get("detail") if worst is not None else None),
        status=state, note=note, findings_at_decision=at_decision,
        waived_findings=waived, waived_checks=waived_checks,
        first_seen=(seen or {}).get("first_seen", ""),
        last_seen=(seen or {}).get("last_seen", ""),
        runs_open=int((seen or {}).get("runs_open", 1) or 1),
        routing_hint=hint, family_breadth=breadth)


def _weigh_residue(rows: pd.DataFrame,
                   waivers: dict[str, dict[str, Any]]) -> tuple[int, int, tuple[str, ...]]:
    """`(unwaived_queue, waived_queue, waived_checks)` for one cluster's findings.

    QUEUE SEVERITIES ONLY. An `info` finding is declared, quantified and expected, nothing
    reads it as work, and its severity weight is 0 -- so it can neither block a settlement nor
    hide behind a waiver, and requiring one for it would turn the benign `catalogue_exclusion_
    cost` residue into paperwork. That is deliberate and is pinned by its own test.

    A `''` entry covers every check on the cluster; a named entry covers only its own. Only
    `wontfix` entries count: a row whose status says anything else is not a tolerance.
    """
    live = {check for check, entry in waivers.items()
            if str(entry.get("status") or "") == WONTFIX}
    if "severity" not in rows.columns:
        return 0, 0, ()
    queue = rows[rows["severity"].astype(str).isin(QUEUE_SEVERITIES)]
    if queue.empty:
        return 0, 0, ()
    if CLUSTER_WIDE in live:
        covered = queue
    else:
        covered = queue[queue["check_name"].astype(str).isin(live)]
    checks = tuple(sorted({str(c) for c in covered["check_name"]})) if len(covered) else ()
    return len(queue) - len(covered), len(covered), checks


def _expired(rows: pd.DataFrame, waivers: dict[str, dict[str, Any]], findings: int) -> bool:
    """Has ANY live waiver on this cluster grown past the size it was decided against?

    Per entry, against its OWN population -- a `peer_ratio` waiver against `peer_ratio`'s
    count, a cluster-wide one against the total. An expired judgement cannot carry a
    settlement, which is what stops a stale `wontfix` from quietly closing a growing defect.
    """
    counts = Counter(str(c) for c in rows["check_name"]) if len(rows) else Counter()
    for check, entry in waivers.items():
        if str(entry.get("status") or "") != WONTFIX:
            continue
        decided = entry.get("findings_at_decision")
        if decided is None or not pd.notna(decided):
            continue
        size = findings if check == CLUSTER_WIDE else counts.get(check, 0)
        if size > int(decided):
            return True
    return False


def _first_note(waivers: dict[str, dict[str, Any]], findings: int,
                counts: dict[str, int]) -> tuple[str, int | None]:
    """The note and `findings_at_decision` a single-valued render shows for a cluster.

    A cluster can now carry several waivers with several notes, and the footer has one column
    for it. The CLUSTER-WIDE entry wins when present -- it is the broadest statement anyone
    made -- otherwise the largest per-check population, which is the one a reader most needs
    to see. `waived_checks` carries the rest, so nothing is lost, only ordered.
    """
    if CLUSTER_WIDE in waivers:
        chosen = waivers[CLUSTER_WIDE]
    else:
        chosen = max(waivers.values(),
                     key=lambda e: counts.get(str(e.get("check_name") or ""), 0))
    decided = chosen.get("findings_at_decision")
    return str(chosen.get("note") or ""), (int(decided) if pd.notna(decided) else None)


def corroboration(n_checks: int) -> float:
    """What `n_checks` INDEPENDENT checks agreeing is worth, as a multiplier on the base score.

    Linear in the number of extra witnesses. A curve was rejected: nobody has measured how
    corroboration should decay, and an unmeasured curve would quietly become the ranking --
    the same argument that keeps per-field importance weights out of `TIER_WEIGHTS`.
    """
    return 1.0 + CORROBORATION_BONUS * max(0, n_checks - 1)


def _worst_row(rows: pd.DataFrame):
    """The highest-severity finding in the cluster -- whose `why` the packet quotes.

    The check's OWN stated mechanism, for the worst thing it saw. Quoting the first row
    instead would hand an agent a `medium` statistical candidate as the explanation of a
    `critical` structural break.
    """
    order = {severity: i for i, severity in enumerate(SEVERITY_ORDER)}
    ranked = rows.assign(_rank=rows["severity"].map(order)).sort_values("_rank")
    return ranked.iloc[0] if len(ranked) else None


def _family_sizes(findings: pd.DataFrame, roster_size: int) -> dict[str, tuple[str, str]]:
    """`{field: (routing_hint, breadth)}` -- computed ONCE, before the per-cluster loop.

    Per-cluster recomputation would rescan the whole frame per cluster: ~1,900 passes over
    ~12,000 rows. The same "a helper quietly rescans its input" trap `_by_series` documents
    in the Tier-3 checks.
    """
    grouped = findings.groupby("field")["ticker"].nunique()
    out: dict[str, tuple[str, str]] = {}
    for field, tickers in grouped.items():
        family = Family(field=str(field), clusters=(), total_score=0.0,
                        tickers=int(tickers), roster_size=roster_size)
        out[str(field)] = (family.routing_hint, family.breadth)
    return out


def _history_map(history: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if history is None or history.empty:
        return {}
    return {str(row["cluster_id"]): {
        "first_seen": _date(row.get("first_seen")),
        "last_seen": _date(row.get("last_seen")),
        "runs_open": row.get("runs_open", 1)} for _, row in history.iterrows()}


def _period_range(keys: pd.Series) -> str:
    """`2011-10-26..2026-07-31` over a cluster's periods -- which filings to pull.

    `period_key` is polymorphic by grain (a date, a `start..end` span, or `''` for a
    ticker-level check), so the range is taken over every date-shaped token found in any of
    them rather than over the strings themselves.
    """
    tokens: list[str] = []
    for key in keys.dropna().astype(str):
        tokens.extend(part for part in key.split("..") if len(part) == 10 and part[4] == "-")
    if not tokens:
        return "ticker-level"
    low, high = min(tokens), max(tokens)
    return low if low == high else f"{low}..{high}"


def _first_url(rows: pd.DataFrame) -> str | None:
    """An EDGAR url from anywhere in the cluster, or None.

    None is common and is not a defect in this function: 1,427 of run 2's findings are
    Tier-1-only, and Tier-1 checks read `fundamentals_history`, which carries no accession.
    That is Phase 7's trigger, and the agent is told to say so plainly rather than guess.
    """
    if "edgar_url" not in rows.columns:
        return None
    urls = rows["edgar_url"].dropna()
    return str(urls.iloc[0]) if len(urls) else None


def _why(detail) -> str:
    """The check's own stated mechanism, out of its JSON `detail`.

    Prose in the OUTPUT is useful; prose as the machine-readable payload is not. `detail` is
    JSON and `why` is one of its keys, so a reader gets the sentence and an agent gets the
    structure -- rather than an agent parsing English to decide what to fix.
    """
    if detail is None or (isinstance(detail, float) and pd.isna(detail)):
        return ""
    try:
        blob = json.loads(detail) if isinstance(detail, str) else detail
    except (TypeError, ValueError):
        return ""
    return str(blob.get("why") or blob.get("verdict") or "") if isinstance(blob, dict) else ""


def _date(value) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(pd.Timestamp(value).date())


__all__ = ["CLUSTER_WIDE", "CORROBORATION_BONUS", "Cluster", "Family",
           "FAMILY_BREADTH_MIN_SHARE", "FAMILY_BREADTH_MIN_TICKERS",
           "LIKELY_CHECK", "LIKELY_FILER", "OPEN", "REOPENED", "SETTLED", "SEVERITY_WEIGHTS",
           "SettledCluster", "TIER_WEIGHTS", "WONTFIX", "build_clusters", "build_families",
           "corroboration", "derive_status", "settled_clusters"]
