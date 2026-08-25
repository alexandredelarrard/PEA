"""
report.py  (src/validate/fundamentals/)
--------------------------------------------------------------------------------------------
The run's output, for a human and for an agent: a fire-rate table, a ranked queue, and the
register's own health.

## Everything here ENUMERATES `CHECK_REGISTRY`

AGENTS.md's rule, and the reason it matters here specifically: a hand-written report section
per check is how a check ends up running for months while nobody reads its output. If a check
is in the registry it is in the report, including when it found nothing -- and *especially*
when it ABSTAINED, which is not the same thing and must never render as a clean pass.

## The three sections, and why each exists

  1. **FIRE RATES.** Findings per unit examined, against each check's own declared ceiling.
     A check over its ceiling is labelled a THRESHOLD BUG in the output, not left for a reader
     to infer from a big number. DQC_0118: *"inconsistencies reported to filers can be
     overwhelming as many don't represent real errors."*
  2. **THE QUEUE.** Open findings at `critical` -> `high` -> `medium`, worst first. `info` is
     excluded by construction.
  3. **REGISTER HEALTH.** How many findings were subtracted, which settled entries have gone
     STALE, which `config_proposed` fixes are still awaiting approval, and which `fixed`
     outcomes no full-roster sweep has confirmed. Without this the register silently becomes a
     suppression list, which is the failure mode the whole lifecycle is designed against.
"""
from __future__ import annotations

import json

import pandas as pd

from src.validate.fundamentals.finding import SEVERITY_ORDER
from src.validate.fundamentals.validator import ValidationRun

#: How many queue rows the printed report shows before summarising. The MARKDOWN report is not
#: truncated -- a file has no attention budget, a terminal does.
PRINTED_QUEUE_ROWS = 25


def render(run: ValidationRun, *, printed_rows: int = PRINTED_QUEUE_ROWS) -> str:
    """The whole report as markdown. Written to `--report PATH` and printed to the log."""
    parts = [
        _header(run),
        _fire_rate_table(run),
        _queue_section(run, limit=printed_rows),
        _register_section(run),
    ]
    return "\n\n".join(p for p in parts if p) + "\n"


def _header(run: ValidationRun) -> str:
    counts = (run.findings["severity"].value_counts().to_dict()
              if not run.findings.empty else {})
    by_severity = "  ".join(f"{s}={counts.get(s, 0)}" for s in SEVERITY_ORDER)
    return (f"# fundamentals validation -- {run.run_date.date()}\n\n"
            f"{len(run.tickers)} ticker(s) | {len(run.findings)} open finding(s) | "
            f"{run.settled_total} settled and subtracted\n\n"
            f"severity: {by_severity}\n\n"
            f"*Nothing here gates. The nightly build of `fundamentals_facts` / "
            f"`fundamentals_history` runs to completion regardless.*")


def _fire_rate_table(run: ValidationRun) -> str:
    """One row per REGISTERED check, whether or not it fired. See the module docstring."""
    lines = ["## fire rates",
             "",
             "*`rate` is QUEUE findings / examined. `info` findings are shown but excluded "
             "from the rate — they never enter the work queue, so they cannot bury anything.*",
             "",
             "| check | tier | substrate | examined | queue | info | rate | ceiling | verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for outcome in sorted(run.outcomes, key=lambda o: (o.spec.tier, o.spec.name)):
        spec = outcome.spec
        if outcome.abstained:
            verdict = "**ABSTAINED** -- nothing to examine, NOT a pass"
            rate = "--"
        elif outcome.over_ceiling:
            verdict = "**THRESHOLD BUG** -- above its own declared ceiling"
            rate = f"{outcome.fire_rate:.2%}"
        else:
            verdict = "ok"
            rate = f"{outcome.fire_rate:.2%}"
        info = len(outcome.findings) - outcome.queued
        lines.append(f"| `{spec.name}` | {spec.tier} | {spec.substrate} | "
                     f"{outcome.examined:,} | {outcome.queued} | {info} | {rate} | "
                     f"{spec.expected_fire_rate_ceiling:.1%} | {verdict} |")

    over = run.over_ceiling
    if over:
        lines += ["", "> **Challenge the check before challenging the data.** "
                      f"{len(over)} check(s) fired above their own declared ceiling: "
                      + ", ".join(f"`{o.spec.name}` ({o.fire_rate:.1%})" for o in over)
                      + ". A check over its ceiling has a threshold bug until proven "
                        "otherwise, and it buries every real finding under itself."]
    return "\n".join(lines)


def _queue_section(run: ValidationRun, *, limit: int) -> str:
    queue = run.queue
    if queue.empty:
        return ("## the queue\n\n"
                "No open findings at `critical`, `high` or `medium`.\n\n"
                "*Read the fire-rate table before concluding anything from this: a check "
                "that ABSTAINED found nothing because it looked at nothing.*")
    lines = [f"## the queue -- {len(queue)} open finding(s), worst first", ""]
    for severity in SEVERITY_ORDER:
        rows = queue[queue["severity"] == severity]
        if rows.empty:
            continue
        lines.append(f"### {severity} ({len(rows)})")
        lines.append("")
        for _, row in rows.head(limit).iterrows():
            lines.append(_render_finding(row))
        if len(rows) > limit:
            lines.append(f"*... and {len(rows) - limit} more `{severity}` finding(s).*")
        lines.append("")
    return "\n".join(lines)


def _render_finding(row: pd.Series) -> str:
    """One finding as an investigation packet a reader can act on without a second query."""
    identity = (f"- **`{row['check_name']}`** {row['ticker']} "
                f"{row['field'] or ''} @ {row['period_key'] or 'ticker-level'} "
                f"`[{row['finding_id']}]`")
    claim = []
    if pd.notna(row.get("observed")):
        claim.append(f"observed={_number(row['observed'])}")
    if pd.notna(row.get("expected")):
        claim.append(f"expected={_number(row['expected'])}")
    if pd.notna(row.get("deviation")):
        claim.append(f"deviation={_number(row['deviation'])}")

    provenance = [f"{key}={row[key]}" for key in
                  ("source_concept", "resolution_method", "root_anchor")
                  if row.get(key)]
    lines = [identity]
    if claim:
        lines.append(f"    {' | '.join(claim)}")
    if provenance:
        lines.append(f"    {' | '.join(provenance)}")
    if row.get("edgar_url"):
        lines.append(f"    {row['edgar_url']}")
    why = _why(row.get("detail"))
    if why:
        lines.append(f"    _{why}_")
    return "\n".join(lines)


def _why(detail) -> str:
    """The `why` a check put in its `detail` payload, if any.

    Prose in the OUTPUT is fine and useful; prose as the machine-readable payload is not. The
    payload is JSON and `why` is one of its keys, so a reader gets the sentence and an agent
    gets the structure -- rather than an agent having to parse English to decide what to fix.
    """
    if not detail:
        return ""
    try:
        blob = json.loads(detail) if isinstance(detail, str) else detail
    except (TypeError, ValueError):
        return ""
    return str(blob.get("why") or blob.get("verdict") or "")


def _register_section(run: ValidationRun) -> str:
    register = run.register
    lines = ["## register health", "",
             f"- {len(register)} settled finding(s) on file; "
             f"{run.settled_total} subtracted from this run"]

    proposals = register.open_proposals()
    if proposals:
        lines.append(f"- **{len(proposals)} `config_proposed` fix(es) STILL OPEN** -- a "
                     "`configs/` diff is proposed and not approved, so the data is still "
                     "wrong and the finding is still work:")
        lines += [f"    - `{e.finding_id}` {e.check} {e.ticker} {e.field}" for e in proposals]

    unswept = register.unswept_fixes()
    if unswept:
        lines.append(f"- **{len(unswept)} `fixed` outcome(s) with `regression_swept: false`** "
                     "-- closed on the affected tickers only. A phase must not close while "
                     "this is non-empty: four defects were once *created by* a fix and were "
                     "visible only on a full re-sweep:")
        lines += [f"    - `{e.finding_id}` {e.check} {e.ticker} {e.field}" for e in unswept]

    if run.stale_entries:
        lines.append(f"- {len(run.stale_entries)} settled entr(ies) whose check did NOT fire "
                     "this run. Not an error -- a fixed defect *should* stop firing -- but "
                     "reported so the register decays visibly instead of accumulating "
                     "suppressions nobody can justify:")
        lines += [f"    - `{e.finding_id}` {e.check} {e.ticker} {e.field} ({e.outcome})"
                  for e in run.stale_entries[:20]]
        if len(run.stale_entries) > 20:
            lines.append(f"    - *... and {len(run.stale_entries) - 20} more.*")
    return "\n".join(lines)


def _number(value) -> str:
    """A payload number, readable. Financial values span cents to hundreds of billions, so a
    fixed format is wrong at one end or the other."""
    number = float(value)
    if number and (abs(number) >= 1e6 or abs(number) < 1e-2):
        return f"{number:,.4g}"
    return f"{number:,.4f}".rstrip("0").rstrip(".")
