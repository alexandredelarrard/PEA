"""
cli.py  (src/validate/cli.py)
--------------------------------------------------------------------------------------------
VALIDATION command-line interface -- part 2 of the three-part loop. Invoked as:

    python -m src validate fundamentals [-t AAPL] [--roster in_sample] [--field capex]
                                        [--tier 1,2,3] [--since 2026-01-01]
                                        [--report PATH] [--no-write]
    python -m src validate report [--run-id X] [--format md|json|both]
    python -m src validate status set <cluster_id> --note "..."
    python -m src validate status clear <cluster_id>
    python -m src validate checks

One command per DOMAIN, so a future prices or insider validator is a sibling rather than a
flag. Every form is read-only against every table but the three the validator owns, and
NOTHING here gates anything: the nightly extraction runs to completion whatever this reports.

## The forms, and what each is for

    --roster in_sample        the tuned stress set. A pass proves CONSISTENCY, not
                              generalisation -- every rule in the resolver was tuned on it
    --roster out_of_sample    zero overlap, never tuned. A finding here is a genuine
                              generalisation failure
    --roster random_cold      the only honest estimate of the error rate on an arbitrary
                              ticker; both designed rosters measure robustness to KNOWN-HARD
                              shapes instead
    --roster A --roster B     REPEATABLE. "both samples" is two flags, not a third roster
                              entry that would then have to be kept in sync with the two
    --field X                 the NEW-FIELD ACCEPTANCE SHEET (decision 44). A catalogue field
                              is born `status: probation`; promotion to `active` requires
                              this sheet clean, or its gaps recorded as `wontfix` clusters
                              with quantified evidence
    --tier 1                  the nightly full-table pass. Tiers 2-3 nightly run only on
                              tickers that received a filing (decision 53) -- a series can
                              only change where a filing landed
    --no-write                print and write a report, touch no table. The default for
                              exploring a threshold. NO DELTA is possible on this path: the
                              run has no ledger row to compare against, and the report says so

## `validate report` re-renders WITHOUT re-running

Because re-running to read a report is how a stale-row false green happens. After a fix and a
rebuild, the checks must be re-executed -- but to read what a recorded run FOUND, read the
tables. The two are different questions and the CLI keeps them apart.
"""
import json
from datetime import date
from pathlib import Path

import click
import pandas as pd

from src.constants.command_line_interface import (
    CONFIG_ARGS, CONFIG_KWARGS, TICKERS_ARGS, TICKERS_KWARGS)
from src.context import Context, get_config_context
from src.data_store.schema import Tables
from src.utils.cli_helper import SpecialHelpOrder
from src.utils.universe import load_universe_tickers
from src.validate.fundamentals import report as report_module
from src.validate.fundamentals.clusters import WONTFIX
from src.validate.fundamentals.ledger import Ledger
from src.validate.fundamentals.report import ReportModel
from src.validate.fundamentals.validator import FundamentalsValidator

#: Where the rosters live. Read here rather than re-declared: `fundamentals_rosters.json`
#: records WHY each ticker is on its list, which is the property a bare list of symbols loses
#: and the reason a roster is worth having at all.
ROSTERS_PATH = "fundamentals/fundamentals_rosters.json"

#: `--roster all` is the full universe, on demand. Named rather than implied so that a nightly
#: job cannot reach it by leaving a flag off.
ROSTER_ALL = "all"

#: Where reports go (D9). Dated directory, scope-named file, `.md` beside `.json`.
REPORT_ROOT = "reports/validate"


@click.group(cls=SpecialHelpOrder)
def cli() -> None:
    """VALIDATION — read-only audits. Writes findings; gates nothing."""


def _ctx(config_path: str) -> tuple[object, Context]:
    return get_config_context(config_path, use_cache=False, save=False)


def _roster_tickers(context: Context, config_path: str, rosters: tuple[str, ...],
                    tickers: str | None) -> tuple[list[str] | None, str]:
    """`(tickers, scope_label)`. `--tickers` wins, then `--roster`, then None (= everything).

    None rather than the full universe when neither is given: `store.load` with no `where`
    reads the whole table in one statement, which is what we want, whereas passing 500 symbols
    builds a 500-element IN clause for no benefit.

    `--roster` is REPEATABLE and the result is the UNION, de-duplicated. Two rosters that
    overlap must not load a ticker twice -- the scope hash is built from what was loaded, and
    a duplicate would make an otherwise identical run look like a different scope.
    """
    if tickers:
        parsed = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        return parsed, (parsed[0] if len(parsed) == 1 else f"{len(parsed)}_tickers")
    if not rosters:
        return None, "all_loaded"
    if ROSTER_ALL in rosters:
        return load_universe_tickers(context), ROSTER_ALL

    blob = json.loads((Path(config_path) / ROSTERS_PATH).read_text(encoding="utf-8"))
    collected: set[str] = set()
    for roster in rosters:
        names = blob.get(roster)
        if not names:
            available = sorted(k for k in blob if not k.startswith("_"))
            raise click.BadParameter(f"roster {roster!r} is not in {ROSTERS_PATH}; "
                                     f"available: {available + [ROSTER_ALL]}")
        collected.update(str(t).upper() for t in names)
    return sorted(collected), "_".join(sorted(rosters))


def _tiers(tier: str | None) -> list[int] | None:
    """`--tier 1,3` as `[1, 3]`. None means every tier."""
    if not tier:
        return None
    try:
        return sorted({int(t.strip()) for t in tier.split(",") if t.strip()})
    except ValueError as exc:
        raise click.BadParameter(f"--tier takes a comma-separated list of 1,2,3: {exc}")


def _default_report_path(scope_label: str) -> Path:
    """`reports/validate/YYYY-MM-DD/<scope>.md` (D9). Dated, so runs do not overwrite."""
    return Path(REPORT_ROOT) / date.today().isoformat() / f"{scope_label}.md"


def _write_report(context: Context, model: ReportModel, path: Path, fmt: str) -> None:
    """Write the markdown and/or the JSON, same directory and basename.

    Both by default. The markdown is the HUMAN artifact and the JSON is the AGENT artifact --
    agent B parses the second rather than scraping prose out of the first, which is the whole
    point of emitting two files instead of asking one format to serve both readers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt in ("md", "both"):
        path.write_text(report_module.render(model), encoding="utf-8")
        context.log.info("validate: markdown report -> %s", path)
    if fmt in ("json", "both"):
        json_path = path.with_suffix(".json")
        json_path.write_text(report_module.render_json(model), encoding="utf-8")
        context.log.info("validate: json report -> %s", json_path)


@cli.command(help="Validate fundamentals_history / fundamentals_facts. Read-only; gates nothing.",
             help_priority=1)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option("--roster", "rosters", multiple=True,
              help="A named roster from configs/fundamentals/fundamentals_rosters.json "
                   "(in_sample, out_of_sample, amendment_pair, random_cold), or 'all' for the "
                   "full universe. REPEATABLE -- pass it twice to cover both samples. "
                   "Ignored when --tickers is given.")
@click.option("--field", "fields", default=None,
              help="Narrow to one or more catalogue fields (comma-separated). This is the "
                   "NEW-FIELD ACCEPTANCE SHEET: a probation field is promoted to active only "
                   "when this is clean or its gaps are recorded with evidence.")
@click.option("--tier", default=None,
              help="Comma-separated tiers to run (default: all). Tier 1 is the nightly "
                   "full-table pass; tiers 2-3 nightly run only where a filing landed.")
@click.option("--check", "check_names", default=None,
              help="Comma-separated check names to run (default: every registered check). "
                   "An unknown name RAISES -- '0 findings' and a typo must never look alike.")
@click.option("--since", default=None,
              help="Only load facts filed on or after this date. Decision 53's nightly "
                   "shape for tiers 2-3: a series can only change where a filing landed.")
@click.option("--report", "report_path", default=None,
              help="Where to write the report. Defaults to "
                   "reports/validate/YYYY-MM-DD/<scope>.md, with the .json beside it.")
@click.option("--format", "fmt", type=click.Choice(["md", "json", "both"]), default="both",
              help="Which report file(s) to write. Markdown is the human artifact; JSON is "
                   "the agent artifact.")
@click.option("--no-write", is_flag=True, default=False,
              help="Do not append to fundamentals_check. Print and report only. No delta is "
                   "possible on this path and the report says so.")
def fundamentals(config_path: str, tickers: str | None, rosters: tuple[str, ...],
                 fields: str | None, tier: str | None, check_names: str | None,
                 since: str | None, report_path: str | None, fmt: str,
                 no_write: bool) -> None:
    _, context = _ctx(config_path)
    scope, scope_label = _roster_tickers(context, config_path, rosters, tickers)
    tiers = _tiers(tier)
    names = [c.strip() for c in check_names.split(",") if c.strip()] if check_names else None
    field_list = [f.strip() for f in fields.split(",") if f.strip()] if fields else None

    context.log.info("validate fundamentals: %s ticker(s), tiers=%s, checks=%s, fields=%s",
                     len(scope) if scope else "all", tiers or "all", names or "all",
                     field_list or "all")
    validator = FundamentalsValidator.from_context(
        context, tickers=scope, config_dir=config_path, tiers=tiers, since=since)
    run = validator.run(tiers=tiers, names=names, fields=field_list,
                        roster=" ".join(rosters))

    if no_write:
        context.log.info("validate fundamentals: --no-write, %d finding(s) NOT persisted",
                         len(run.findings))
        model = ReportModel.from_run(run, persisted=False)
    else:
        FundamentalsValidator.write(context, run)
        # Read BACK, so the report carries the delta and the cluster history the tables know
        # about and the in-memory run cannot. This is the same code path `validate report`
        # takes, which is what stops the two from ever disagreeing.
        ledger = Ledger.load(context, scope_hash=run.scope.scope_hash)
        model = ReportModel.from_ledger(ledger, run.run_id)

    context.log.info("\n%s", report_module.render(
        model, packets=report_module.MENU_SIZE, clusters=report_module.TERMINAL_CLUSTERS))
    _write_report(context, model, Path(report_path or _default_report_path(scope_label)), fmt)


@cli.command(help="Re-render a recorded run's report from the tables. No re-run, no writes.",
             help_priority=2)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option("--run-id", default=None,
              help="Which run to render (default: the most recent one on record).")
@click.option("--report", "report_path", default=None,
              help="Where to write it. Defaults to reports/validate/YYYY-MM-DD/<scope>.md.")
@click.option("--format", "fmt", type=click.Choice(["md", "json", "both"]), default="both")
def report(config_path: str, run_id: str | None, report_path: str | None, fmt: str) -> None:
    """The read-only path. Agent B needs this: re-running to read a report reads stale rows."""
    _, context = _ctx(config_path)
    ledger = Ledger.load(context)
    if ledger.runs.empty:
        raise click.ClickException(
            "fundamentals_check_run is empty -- no run has been recorded yet. Run "
            "`validate fundamentals` first; `--no-write` records nothing by design.")
    if not run_id:
        latest = ledger.runs.sort_values("run_date").iloc[-1]
        run_id = str(latest["run_id"])
        context.log.info("validate report: no --run-id given, using the most recent: %s",
                         run_id)

    model = ReportModel.from_ledger(ledger, run_id)
    context.log.info("\n%s", report_module.render(
        model, packets=report_module.MENU_SIZE, clusters=report_module.TERMINAL_CLUSTERS))
    label = model.roster.replace(" ", "_") or f"run_{run_id}"
    _write_report(context, model, Path(report_path or _default_report_path(label)), fmt)


@cli.group(help="Record a human decision about one cluster. The ONLY mutable validator state.")
def status() -> None:
    """`wontfix`, and nothing else.

    `open` and `settled` are DERIVED from the ledger and are deliberately not writable: a
    stored `settled` that says so while the check still fires is exactly the suppression list
    the deleted JSON register was drifting toward.
    """


@status.command("set", help="Mark a cluster wontfix. Requires a QUANTIFIED --note.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.argument("cluster_id")
@click.option("--note", default=None,
              help="The evidence. Must contain a NUMBER -- a quantified cost, not an "
                   "adjective. NEE's $5.2bn understatement is a defensible wontfix precisely "
                   "and only because the number is written down.")
def status_set(config_path: str, cluster_id: str, note: str | None) -> None:
    if not note or not note.strip():
        raise click.ClickException(
            "--note is required and may not be empty. A wontfix with no evidence is a "
            "suppression with a label on it.")
    if not any(ch.isdigit() for ch in note):
        raise click.ClickException(
            f"--note must carry a QUANTIFIED cost -- a number, not an adjective. Got: "
            f"{note!r}. The rule being enforced is 'somebody measured it', and any real "
            f"measurement carries a numeral.")

    _, context = _ctx(config_path)
    ledger = Ledger.load(context)
    rows = (ledger.findings[ledger.findings["cluster_id"] == cluster_id]
            if not ledger.findings.empty else pd.DataFrame())
    if rows.empty:
        raise click.ClickException(
            f"cluster {cluster_id!r} is not in fundamentals_check. A wontfix on a cluster "
            f"nobody has measured cannot record how big it was at the decision, so it could "
            f"never reopen.")
    latest = rows[rows["run_date"] == rows["run_date"].max()]

    frame = pd.DataFrame([{
        "cluster_id": cluster_id,
        "ticker": str(latest["ticker"].iloc[0]),
        "field": str(latest["field"].iloc[0] or ""),
        "status": WONTFIX,
        "note": note.strip(),
        # D8: the size the judgement was actually made against. One more finding reopens it.
        "findings_at_decision": int(len(latest)),
        "decided_at": date.today(),
    }])
    frame["findings_at_decision"] = frame["findings_at_decision"].astype("Int64")
    context.store.save(Tables.fundamentals_check_status, frame)
    context.log.info("validate status: %s (%s %s) -> wontfix at %d finding(s). It REOPENS "
                     "automatically if the cluster grows past that.",
                     cluster_id, latest["ticker"].iloc[0], latest["field"].iloc[0],
                     len(latest))


@status.command("clear", help="Remove a cluster's wontfix, putting it back in the queue.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.argument("cluster_id")
def status_clear(config_path: str, cluster_id: str) -> None:
    _, context = _ctx(config_path)
    removed = context.store.delete(Tables.fundamentals_check_status,
                                   {"cluster_id": cluster_id})
    context.log.info("validate status: cleared %d row(s) for cluster %s", removed, cluster_id)


@cli.command(help="Print CHECK_REGISTRY: every check, its tier, substrate, severity and ceiling.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
def checks(config_path: str) -> None:
    """The registry itself, so 'what does this tool actually test?' needs no source dive."""
    from src.validate.fundamentals.checks import CHECK_REGISTRY

    _, context = _ctx(config_path)
    rows = sorted(CHECK_REGISTRY.values(), key=lambda s: (s.tier, s.name))
    lines = [f"{len(rows)} registered check(s)", ""]
    for spec in rows:
        lines.append(f"  tier {spec.tier}  {spec.name:28s} {spec.substrate:8s} "
                     f"{spec.severity:9s} grain={spec.grain:7s} "
                     f"ceiling={spec.expected_fire_rate_ceiling:.1%}")
        lines.append(f"      {spec.doc}")
    context.log.info("\n%s", "\n".join(lines))
