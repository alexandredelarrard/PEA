"""
report.py  (src/validate/fundamentals/)
--------------------------------------------------------------------------------------------
THE artifact: a check-health gate, a delta, and a ranked list of fixable issues.

## What this replaces, and why the old shape could not work

The previous report printed a fire-rate table and then dumped the queue -- truncated to 25
rows per severity, in the FILE as well as the terminal, while its own docstring claimed the
markdown was untruncated. Run 2's file therefore showed 57 of 10,898 queue findings, ordered
by severity and then alphabetically. Nothing in it said which fix closed the most rows.

## The order of the sections is the argument, and it is deliberate

  1. **HEADER** -- scope and `run_id`. A reader who does not know what was looked at cannot
     interpret anything below.
  2. **CHECK HEALTH, BEFORE THE RANKINGS.** A cluster list drawn from a mis-calibrated run
     reads as authoritative regardless of how wrong it is; this is the only thing that stops
     that. A check over its own ceiling has a THRESHOLD BUG until proven otherwise and buries
     real findings under itself; a check that ABSTAINED examined nothing, which is never a
     pass. If either is present, a banner says the rankings may be inflated.
  3. **THE DELTA** vs the previous COMPARABLE run. Omitted, with a note, when there is none --
     a first run must never render as a trend.
  4. **FIELD FAMILIES**, with breadth and a routing hint. Wide -> fix the spec; narrow -> fix
     the filer.
  5. **CLUSTERS**, ranked, the top five marked as agent B's menu.
  6. **PACKETS** for the top clusters: every check that fired, the mixes, the period range,
     the EDGAR url, and the worst member's own stated mechanism.
  7. **THE WONTFIX FOOTER**, never omitted (D8).

## Truncation, stated honestly this time

Both the FILE and the TERMINAL cap the ranked table (`FILE_CLUSTERS` / `TERMINAL_CLUSTERS`),
and both **state the full count** so the backlog is never hidden -- "the top 25 of 1,939" is
information; a silent 25 is a lie, which is exactly what the report this replaces did while
its docstring denied it.

Listing all 1,939 was the first attempt and was wrong for a simple reason: agent B works ONE
cluster from a menu of five, so rows 26 through 1,939 are read by nobody and cost every reader
the scroll. What a reader needs from the tail is its SIZE, and that is one number.

Families are listed in full -- there are ~50 of them and breadth is the routing signal.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field as dataclass_field
from typing import Any

import pandas as pd

from src.validate.fundamentals.clusters import (
    Cluster, CORROBORATION_BONUS, Family, FAMILY_BREADTH_MIN_SHARE,
    FAMILY_BREADTH_MIN_TICKERS, REOPENED, SEVERITY_WEIGHTS, TIER_WEIGHTS, WONTFIX,
    build_clusters, build_families, corroboration, settled_clusters)
from src.validate.fundamentals.finding import SEVERITY_ORDER

#: How many clusters get a FULL packet in the markdown file. Everything else is one table row.
FILE_PACKETS = 25

#: How many ranked clusters the FILE tables. Agent B works one of the top 5; the rest are
#: context. The total is always printed beside it, so the backlog is visible without being
#: enumerated.
FILE_CLUSTERS = 25

#: How many the TERMINAL tables. Harder cap: a terminal has an attention budget, a file has a
#: scrollbar.
TERMINAL_CLUSTERS = 10

#: How many clusters reach the JSON. Agent B picks from the menu, but a wider list lets it see
#: what it is choosing against without a second query.
JSON_CLUSTERS = 50

#: The menu. Agent B works exactly one of these, and the user picks which.
MENU_SIZE = 5


@dataclass
class ReportModel:
    """Everything the renderers read. Built once, from a run or from the ledger.

    The two constructors exist because the two paths have genuinely different information.
    A `--no-write` run has no `run_id` in any table, so it can have no delta and no history --
    and that is reported as an absence rather than papered over with an empty delta section.
    """

    run_id: str
    run_date: pd.Timestamp
    findings: pd.DataFrame
    health: pd.DataFrame
    clusters: list[Cluster]
    families: list[Family]
    roster: str = ""
    tickers: tuple[str, ...] = ()
    fields: tuple[str, ...] = ()
    tiers: tuple[int, ...] = ()
    previous_label: str = ""
    settled: list[str] = dataclass_field(default_factory=list)
    persisted: bool = True
    #: Set when there is no comparable prior run: WHY there is no delta, in words.
    no_delta_reason: str = ""

    # ------------------------------------------------------------------ construction ---
    @classmethod
    def from_run(cls, run, *, status: dict[str, dict[str, Any]] | None = None,
                 persisted: bool = False) -> "ReportModel":
        """From an in-memory run. No delta and no cluster history -- see the class docstring."""
        clusters = build_clusters(run.findings, status=status,
                                  roster_size=len(run.scope.tickers))
        return cls(
            run_id=run.run_id, run_date=run.run_date, findings=run.findings,
            health=run.check_runs(), clusters=clusters,
            families=build_families(clusters, roster_size=len(run.scope.tickers)),
            roster=run.scope.roster, tickers=run.scope.tickers, fields=run.scope.fields,
            tiers=run.scope.tiers, persisted=persisted,
            no_delta_reason=("this run was not written (--no-write), so it has no `run_id` in "
                             "the ledger to compare against"
                             if not persisted else
                             "no earlier run of this exact scope is on record"))

    @classmethod
    def from_ledger(cls, ledger, run_id: str) -> "ReportModel":
        """From the tables. The path `validate report` takes -- no re-run, no writes.

        Re-running to read a report is how a stale-row false green happens: the checks are
        re-executed against data that may have changed underneath, and the delta then measures
        two different things. This path measures what was recorded.
        """
        record = ledger.run(run_id)
        if record is None:
            known = (sorted(ledger.runs["run_id"].unique())[:10]
                     if not ledger.runs.empty else ["(none recorded)"])
            raise KeyError(f"run_id {run_id!r} is not in fundamentals_check_run. "
                           f"Known runs: {known}")
        findings = ledger.findings_for(run_id)
        previous = ledger.previous_comparable(run_id)
        clusters = build_clusters(
            findings, status=ledger.status_map(),
            history=ledger.cluster_history(record.scope_hash),
            roster_size=record.scope_tickers)
        settled = (settled_clusters(ledger.findings_for(previous.run_id), findings)
                   if previous else [])
        return cls(
            run_id=run_id, run_date=record.run_date, findings=findings,
            health=ledger.check_health(run_id), clusters=clusters,
            families=build_families(clusters, roster_size=record.scope_tickers),
            roster=record.scope_roster or "",
            tickers=tuple(json.loads(_ticker_list(ledger, run_id) or "[]")),
            fields=tuple(json.loads(record.scope_fields or "[]")),
            tiers=tuple(json.loads(record.scope_tiers or "[]")),
            previous_label=previous.label if previous else "",
            settled=settled, persisted=True,
            no_delta_reason=("" if previous else
                             "no earlier run of this exact scope is on record; two runs are "
                             "comparable only when their scope hash matches"))

    # ------------------------------------------------------------------------ views ---
    @property
    def menu(self) -> list[Cluster]:
        """The top clusters agent B chooses from. `wontfix` ones are not on the menu."""
        return [c for c in self.clusters
                if c.is_work and c.status != WONTFIX][:MENU_SIZE]

    @property
    def reopened(self) -> list[Cluster]:
        """Clusters whose `wontfix` expired because they grew (D8)."""
        return [c for c in self.clusters if c.status == REOPENED]

    @property
    def wontfix(self) -> list[Cluster]:
        return [c for c in self.clusters if c.status == WONTFIX]

    @property
    def abstained(self) -> pd.DataFrame:
        return _flagged(self.health, "abstained")

    @property
    def over_ceiling(self) -> pd.DataFrame:
        return _flagged(self.health, "over_ceiling")

    @property
    def unhealthy(self) -> bool:
        """Is any check abstaining or over its ceiling? Drives the banner."""
        return not self.abstained.empty or not self.over_ceiling.empty


# --------------------------------------------------------------------------- #
# markdown                                                                     #
# --------------------------------------------------------------------------- #

def render(model: ReportModel, *, packets: int = FILE_PACKETS,
           clusters: int | None = FILE_CLUSTERS) -> str:
    """The whole report as markdown. `clusters=N` caps the ranked TABLE; the TOTAL is always
    printed. `clusters=None` lists every one, which no caller wants and one might."""
    parts = [
        _header(model),
        _health(model),
        _delta(model),
        _families(model),
        _clusters(model, limit=clusters),
        _packets(model, limit=packets),
        _wontfix(model),
    ]
    return "\n\n".join(p for p in parts if p) + "\n"


def _header(model: ReportModel) -> str:
    counts = (model.findings["severity"].value_counts().to_dict()
              if not model.findings.empty else {})
    by_severity = "  ".join(f"{s}={counts.get(s, 0)}" for s in SEVERITY_ORDER)
    scope = (f"{len(model.tickers)} ticker(s)"
             f"{f' [{model.roster}]' if model.roster else ''} | "
             f"tiers {','.join(map(str, model.tiers)) or 'all'} | "
             f"fields {','.join(model.fields) or 'all'}")
    return (f"# fundamentals validation -- {model.run_date.date()}\n\n"
            f"**run_id `{model.run_id}`** | {scope}\n\n"
            f"{len(model.findings)} finding(s) | {len(model.clusters)} cluster(s) | "
            f"{len(model.families)} field family(ies)\n\n"
            f"severity: {by_severity}\n\n"
            + ("" if model.persisted else
               "> *`--no-write`: nothing was persisted, so this run has no ledger row and no "
               "delta.*\n\n")
            + "*Nothing here gates. The nightly build of `fundamentals_facts` / "
              "`fundamentals_history` runs to completion regardless.*")


def _health(model: ReportModel) -> str:
    """The gate. Renders ABOVE the rankings, always, and leads with a banner when it must."""
    if model.health.empty:
        return "## check health\n\n*No per-check record for this run.*"
    lines: list[str] = ["## check health -- read this before the rankings", ""]

    if model.unhealthy:
        problems = []
        if not model.over_ceiling.empty:
            names = ", ".join(f"`{r.check_name}`" for r in model.over_ceiling.itertuples())
            problems.append(f"{len(model.over_ceiling)} check(s) fired ABOVE their own "
                            f"declared ceiling ({names})")
        if not model.abstained.empty:
            names = ", ".join(f"`{r.check_name}`" for r in model.abstained.itertuples())
            problems.append(f"{len(model.abstained)} check(s) ABSTAINED -- they examined "
                            f"nothing, which is not a pass ({names})")
        lines += [
            "> **/!\\ THE RANKINGS BELOW MAY BE INFLATED.** " + "; ".join(problems) + ".",
            ">",
            "> A check over its ceiling has a THRESHOLD BUG until proven otherwise, and it "
            "buries every real finding under itself -- DQC_0118: *\"inconsistencies reported "
            "to filers can be overwhelming as many don't represent real errors.\"* Clusters "
            "carried by such a check are weak evidence and should be treated as a suspected "
            "check defect first. A check that abstained examined nothing, so whatever it "
            "tests went UNCHECKED on this roster.",
            "",
        ]
    else:
        lines += ["*Every check examined something and every check fired within its own "
                  "declared ceiling.*", ""]

    lines += ["| check | tier | substrate | examined | queue | info | rate | ceiling | verdict |",
              "|---|---|---|---|---|---|---|---|---|"]
    for row in model.health.sort_values(["tier", "check_name"]).itertuples():
        examined = int(row.examined or 0)
        queued = int(row.queued or 0)
        if bool(row.abstained):
            verdict, rate = "**ABSTAINED** -- nothing to examine, NOT a pass", "--"
        elif bool(row.over_ceiling):
            verdict = "**THRESHOLD BUG** -- above its own declared ceiling"
            rate = f"{queued / examined:.2%}" if examined else "--"
        else:
            verdict = "ok"
            rate = f"{queued / examined:.2%}" if examined else "--"
        lines.append(f"| `{row.check_name}` | {row.tier} | {row.substrate} | {examined:,} | "
                     f"{queued} | {int(row.info or 0)} | {rate} | "
                     f"{float(row.ceiling or 0):.1%} | {verdict} |")
    lines += ["", "*`rate` is QUEUE findings / examined. `info` findings are shown but "
                  "excluded from the rate -- nothing reads them as work, so they cannot bury "
                  "anything.*"]
    return "\n".join(lines)


def _delta(model: ReportModel) -> str:
    """What changed since the last COMPARABLE run, or an explicit statement that nothing can be
    said. The second case is not a formality: a first run presented as a trend is a lie."""
    if not model.previous_label:
        return ("## delta\n\n"
                f"*No delta: {model.no_delta_reason}.*\n\n"
                "*A run is only comparable to one of the SAME scope -- same tickers, same "
                "fields, same tiers. Differencing a 54-ticker baseline against a one-ticker "
                "re-validation would report ~11,800 findings \"closed\".*")
    lines = [f"## delta vs {model.previous_label}", ""]
    if model.settled:
        lines.append(f"- **{len(model.settled)} cluster(s) SETTLED** -- present in that run, "
                     f"absent from this one. That is the proof a fix worked:")
        lines += [f"    - `{cid}`" for cid in model.settled[:20]]
        if len(model.settled) > 20:
            lines.append(f"    - *... and {len(model.settled) - 20} more.*")
    else:
        lines.append("- no cluster closed since that run")
    if model.reopened:
        lines.append(f"- **{len(model.reopened)} cluster(s) REOPENED** -- a `wontfix` expired "
                     f"because the cluster GREW past the size that was assessed (D8):")
        lines += [f"    - `{c.cluster_id}` {c.ticker} {c.field}: "
                  f"{c.findings_at_decision} findings at the decision, {c.findings} now"
                  for c in model.reopened]
    new = [c for c in model.clusters if c.runs_open == 1 and c.is_work]
    if new:
        lines.append(f"- {len(new)} cluster(s) appear for the first time in this scope")
    return "\n".join(lines)


#: Above this share of families landing on ONE routing hint, the hint has stopped
#: discriminating and the report says so. Measured need: on the 54-ticker calibration roster
#: 48 of 50 families came back `likely-check-or-catalogue`, because a statistical check that
#: fires on ~2.4% of everything touches nearly every ticker on nearly every field. A hint that
#: says the same thing about everything is not evidence, and presenting it as though it were
#: is exactly the false confidence the check-health gate exists to prevent.
HINT_DEGENERATE_SHARE = 0.80


def _families(model: ReportModel) -> str:
    families = [f for f in model.families if f.total_score > 0]
    if not families:
        return ""
    lines = [f"## field families -- {len(families)} with work in them", "",
             f"*A family spanning >= {FAMILY_BREADTH_MIN_TICKERS} tickers AND "
             f">= {FAMILY_BREADTH_MIN_SHARE:.0%} of the roster is routed "
             f"`likely-check-or-catalogue`: forty filers do not fail independently and "
             f"simultaneously on one field. Both thresholds are constants in `clusters.py`.*",
             ""]
    note = _hint_degeneracy(families)
    if note:
        lines += [note, ""]
    lines += ["| field | score | findings | clusters | breadth | routing |",
              "|---|---|---|---|---|---|"]
    for family in families:
        lines.append(f"| `{family.field}` | {family.total_score:,.0f} | {family.findings} | "
                     f"{len(family.clusters)} | {family.breadth} | **{family.routing_hint}** |")
    return "\n".join(lines)


def _hint_degeneracy(families: list[Family]) -> str:
    """A warning when nearly every family routes the same way -- i.e. the hint says nothing.

    Reported rather than silently retuned. The thresholds encode a prior about how many filers
    can plausibly fail at once, and moving them to make a list look better is how a diagnostic
    turns into a decoration. What a reader needs is to know the signal is flat HERE.
    """
    counts: dict[str, int] = {}
    for family in families:
        counts[family.routing_hint] = counts.get(family.routing_hint, 0) + 1
    hint, n = max(counts.items(), key=lambda kv: kv[1])
    if n / len(families) < HINT_DEGENERATE_SHARE:
        return ""
    return (f"> **The routing hint is NOT discriminating on this roster:** {n} of "
            f"{len(families)} families are `{hint}`. Read it as noise here, not as evidence. "
            f"A hint that says the same thing about every family cannot tell an agent what to "
            f"challenge first -- most likely a broad statistical check is touching nearly "
            f"every ticker on nearly every field, which makes every family look wide. The "
            f"breadth column is still worth reading directly; the label is not.")


def _clusters(model: ReportModel, *, limit: int | None) -> str:
    work = [c for c in model.clusters if c.is_work]
    if not work:
        return ("## clusters\n\nNo cluster carries anything at `critical`, `high` or "
                "`medium`.\n\n*Read the check-health table above before concluding anything "
                "from that: a check that ABSTAINED found nothing because it looked at "
                "nothing.*")
    shown = work[:limit] if limit else work
    header = (f"## clusters -- top {len(shown)} of {len(work)} with work in them"
              if limit and len(work) > limit else
              f"## clusters -- {len(work)} with work in them, ranked")
    lines = [header, "",
             f"*`score = (sum over findings of w(severity) x w(tier)) x corroboration`, with "
             f"tier {TIER_WEIGHTS}, severity {SEVERITY_WEIGHTS}, and each additional agreeing "
             f"check worth +{CORROBORATION_BONUS:.0%} "
             f"(so {corroboration(10):.2f}x at ten checks). Those weights are a POLICY, not a "
             f"fact, and they are module constants in `clusters.py` meant to be retuned once "
             f"somebody has read a list and disagreed -- which is where the corroboration "
             f"term came from: volume alone ranked a 62-finding 2-check cluster above a "
             f"55-finding 10-check one.*", "",
             "| # | cluster_id | ticker | field | score | findings | checks | worst | routing |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, cluster in enumerate(shown, 1):
        menu = " **<-- B's menu**" if i <= MENU_SIZE else ""
        lines.append(
            f"| {i} | `{cluster.cluster_id}`{menu} | {cluster.ticker} | `{cluster.field}` | "
            f"{cluster.score:,.0f} | {cluster.findings} | {cluster.checks_agreeing} | "
            f"{cluster.worst_severity} | {cluster.routing_hint} |")
    if limit and len(work) > limit:
        lines.append(
            f"\n*{len(work) - limit:,} further cluster(s) carry work and are not listed. "
            f"Agent B works ONE of the top 5; the rest of this table is context, and the tail "
            f"is a backlog SIZE rather than a reading list. Query `fundamentals_check` by "
            f"`cluster_id` for any of them, or widen with `render(clusters=None)`.*")
    return "\n".join(lines)


def _packets(model: ReportModel, *, limit: int) -> str:
    work = [c for c in model.clusters if c.is_work][:limit]
    if not work:
        return ""
    lines = [f"## the packets -- top {len(work)}", "",
             "*Everything needed to start, without a second query. If a packet is not enough "
             "to begin on, that is a defect in the CHECK and worth reporting on its own.*", ""]
    for i, cluster in enumerate(work, 1):
        lines.append(_packet(i, cluster))
    return "\n".join(lines)


def _packet(rank: int, cluster: Cluster) -> str:
    """One cluster, as the self-contained investigation packet agent B consumes."""
    menu = "  **<-- B's menu**" if rank <= MENU_SIZE else ""
    checks = ", ".join(f"`{name}`x{n}" for name, n in cluster.checks)
    severities = ", ".join(f"{s}={n}" for s, n in cluster.severities)
    tiers = ", ".join(f"T{t}={n}" for t, n in cluster.tiers)
    lines = [
        f"### {rank}. {cluster.ticker} `{cluster.field}` -- `{cluster.cluster_id}`{menu}",
        "",
        f"- **score {cluster.score:,.0f}** from {cluster.findings} finding(s) across "
        f"{cluster.checks_agreeing} check(s)",
        f"- checks agreeing: {checks}",
        f"- severity: {severities} | tier: {tiers}",
        f"- periods: {cluster.period_range}",
        f"- routing: **{cluster.routing_hint}** ({cluster.family_breadth} on this field)",
    ]
    if cluster.checks_agreeing >= 3:
        lines.append(f"- *{cluster.checks_agreeing} INDEPENDENT checks agree here. That is a "
                     f"far stronger prior than one check firing {cluster.findings} times.*")
    elif cluster.checks_agreeing == 1:
        lines.append("- *carried by a SINGLE check. Weak evidence on its own -- if that check "
                     "is over its ceiling above, treat it as a suspected check defect first.*")
    if cluster.runs_open > 1:
        lines.append(f"- seen in {cluster.runs_open} comparable run(s), "
                     f"{cluster.first_seen} -> {cluster.last_seen}")
    if cluster.edgar_url:
        lines.append(f"- {cluster.edgar_url}")
    else:
        lines.append("- **no `edgar_url`** -- a Tier-1-only cluster reads "
                     "`fundamentals_history`, which carries no accession. Resolve it manually "
                     "and say so: this is Phase 7's trigger.")
    if cluster.why:
        lines.append(f"- _{cluster.why}_")
    return "\n".join(lines) + "\n"


def _wontfix(model: ReportModel) -> str:
    """NEVER omitted (D8). The register was deleted; this footer is half of what replaces it."""
    decided = model.wontfix + model.reopened
    lines = ["## `wontfix` clusters", ""]
    if not decided:
        return "\n".join(lines + [
            "*None on file.*", "",
            "*This section is never omitted. A `wontfix` that stops being listed is a "
            "suppression, which is precisely what the deleted JSON register was drifting "
            "toward.*"])
    lines += ["| cluster_id | ticker | field | now | at decision | runs | state | note |",
              "|---|---|---|---|---|---|---|---|"]
    for cluster in decided:
        state = "**REOPENED**" if cluster.status == REOPENED else "wontfix"
        near = ""
        if (cluster.status == WONTFIX and cluster.findings_at_decision
                and cluster.findings == cluster.findings_at_decision):
            near = " *(one more finding reopens it)*"
        lines.append(f"| `{cluster.cluster_id}` | {cluster.ticker} | `{cluster.field}` | "
                     f"{cluster.findings} | {cluster.findings_at_decision} | "
                     f"{cluster.runs_open} | {state}{near} | {cluster.note} |")
    lines += ["", "*A `wontfix` records how many findings it was decided against and REOPENS "
                  "by itself the moment the cluster grows past that. A judgement made about 3 "
                  "findings is not a judgement about 30.*"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# json -- the AGENT artifact (6.0's handoff contract)                          #
# --------------------------------------------------------------------------- #

def render_json(model: ReportModel, *, clusters: int = JSON_CLUSTERS) -> str:
    """The same report as structured data, so agent B parses rather than scrapes prose.

    Markdown stays the human artifact. An agent reading English to decide what to fix is the
    failure mode this whole rebuild exists to remove, so the fields agent B needs are emitted
    with names and types instead of being embedded in sentences.
    """
    work = [c for c in model.clusters if c.is_work]
    menu = {c.cluster_id for c in model.menu}
    payload = {
        "run_id": model.run_id,
        "run_date": str(model.run_date.date()),
        "persisted": model.persisted,
        "scope": {"roster": model.roster or None, "tickers": list(model.tickers),
                  "ticker_count": len(model.tickers), "fields": list(model.fields),
                  "tiers": list(model.tiers)},
        "totals": {"findings": int(len(model.findings)), "clusters": len(model.clusters),
                   "clusters_with_work": len(work), "families": len(model.families)},
        "check_health": {
            "healthy": not model.unhealthy,
            "abstained": [str(r.check_name) for r in model.abstained.itertuples()],
            "over_ceiling": [{"check": str(r.check_name),
                              "rate": (int(r.queued or 0) / int(r.examined))
                              if int(r.examined or 0) else None,
                              "ceiling": float(r.ceiling or 0)}
                             for r in model.over_ceiling.itertuples()],
            "warning": ("rankings may be inflated: a check over its ceiling has a threshold "
                        "bug until proven otherwise, and an abstained check examined nothing"
                        if model.unhealthy else None),
        },
        "delta": {
            "previous_run": model.previous_label or None,
            "comparable": bool(model.previous_label),
            "reason": model.no_delta_reason or None,
            "settled_clusters": model.settled,
            "reopened_clusters": [c.cluster_id for c in model.reopened],
        },
        "routing_hint_discriminating": not _hint_degeneracy(
            [f for f in model.families if f.total_score > 0]),
        "families": [{"field": f.field, "score": round(f.total_score, 2),
                      "findings": f.findings, "clusters": len(f.clusters),
                      "tickers": f.tickers, "roster_size": f.roster_size,
                      "breadth": f.breadth, "routing_hint": f.routing_hint}
                     for f in model.families if f.total_score > 0],
        "weights": {"tier": TIER_WEIGHTS, "severity": SEVERITY_WEIGHTS,
                    "corroboration_bonus_per_extra_check": CORROBORATION_BONUS,
                    "note": "a policy, not a fact. score = (sum of w(severity)*w(tier)) * "
                            "(1 + bonus*(checks-1)). Retunable; the corroboration term exists "
                            "because volume alone ranked a 2-check cluster above a 10-check one"},
        "menu": [c.cluster_id for c in model.menu],
        "clusters": [{**c.as_dict(), "menu": c.cluster_id in menu, "run_id": model.run_id}
                     for c in work[:clusters]],
        "wontfix": [c.as_dict() for c in model.wontfix + model.reopened],
    }
    return json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n"


# --------------------------------------------------------------------------- #
# internals                                                                    #
# --------------------------------------------------------------------------- #

def _flagged(health: pd.DataFrame, column: str) -> pd.DataFrame:
    """The rows where a stored boolean flag is set, or an empty frame OF THE SAME SHAPE.

    Same shape, not `pd.DataFrame()`: callers iterate `.itertuples()` over the result and a
    bare empty frame has no `check_name` attribute to reach for.
    """
    if health.empty or column not in health.columns:
        return health.iloc[0:0]
    return health[health[column].fillna(False).astype(bool)]


def _ticker_list(ledger, run_id: str) -> str:
    rows = ledger.runs[ledger.runs["run_id"] == run_id]
    return str(rows.iloc[0]["scope_ticker_list"]) if len(rows) else "[]"


__all__ = ["FILE_CLUSTERS", "FILE_PACKETS", "HINT_DEGENERATE_SHARE", "JSON_CLUSTERS", "MENU_SIZE", "ReportModel", "TERMINAL_CLUSTERS",
           "render", "render_json"]
