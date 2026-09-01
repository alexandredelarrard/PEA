"""
cli.py  (src/validate/cli.py)
--------------------------------------------------------------------------------------------
VALIDATION command-line interface -- part 2 of the three-part loop. Invoked as:

    python -m src validate prices [-t AAPL] [--since 2020-01-01] [--report PATH] [--no-write]
    python -m src validate fundamentals [-t AAPL] [--roster in_sample] [--field capex]
                                        [--tier 1,2,3] [--since 2026-01-01]
                                        [--report PATH] [--no-write]
    python -m src validate report [--run-id X] [--format md|json|both]
    python -m src validate status set <cluster_id> [--check NAME] --note "..."
    python -m src validate status clear <cluster_id> [--check NAME]
    python -m src validate fix record <cluster_id> --layer L --root-cause "..." \
                                      --evidence '{...}' --commit SHA --test PATH [--waive ...]
    python -m src validate fix show <cluster_id>
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

## `fix record` writes the thing a row-count drop cannot say

A drop proves SOMETHING changed. It cannot say what was done, why, at which layer, against
which filings, or whether the remaining rows were assessed. Cluster `1c9a517eaa47` was fixed
on 2026-08-25 and its only record anywhere was a commit sha -- not in the status table (which
accepts `wontfix` and nothing else), not in the settled set (a strict difference a 55->4 drop
does not satisfy), not in the report (it fell off the top-50).

So `fix record` is ATOMIC and DERIVED: it writes the fix row and its per-check waivers in one
call, because recording a fix and tolerating its benign residue is one decision, and a fix row
whose cluster still reads OPEN is the half-recorded state this exists to remove. Only
`cluster_id` is required -- both run ids, the ticker, the field, the scope hash and all four
counts come off the ledger. The human supplies only what no machine can know.

It REFUSES what it cannot verify (unknown layer, unparseable evidence, missing evidence keys
for that layer, an unresolvable commit, a missing test file, incomparable runs, a waiver for a
check that is not firing) and it WARNS without refusing when the fix closed nothing -- that
row is legitimate and simply cannot settle the cluster. Permissive to record, strict to settle.

## `validate report` re-renders WITHOUT re-running

Because re-running to read a report is how a stale-row false green happens. After a fix and a
rebuild, the checks must be re-executed -- but to read what a recorded run FOUND, read the
tables. The two are different questions and the CLI keeps them apart.
"""
import json
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import click
import pandas as pd

from src.constants.command_line_interface import (
    CONFIG_ARGS, CONFIG_KWARGS, TICKERS_ARGS, TICKERS_KWARGS)
from src.constants.constants import FIX_EVIDENCE_KEYS, FIX_LAYERS
from src.context import Context, get_config_context
from src.data_store.schema import Tables
from src.utils.cli_helper import SpecialHelpOrder
from src.utils.universe import load_universe_tickers
from src.validate.fundamentals import report as report_module
from src.validate.fundamentals.clusters import WONTFIX
from src.validate.fundamentals.finding import QUEUE_SEVERITIES
from src.validate.fundamentals.ledger import CLUSTER_WIDE, Ledger
from src.validate.fundamentals.report import ReportModel
from src.validate.fundamentals.validator import FundamentalsValidator
from src.validate.prices import run_prices_validation

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


@cli.command(help="Validate fundamentals_history_sec / fundamentals_facts. Read-only; gates nothing.",
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


@cli.command(help="Validate the price / share-count ADJUSTMENT BASIS. Read-only.",
             help_priority=2)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option("--since", default=None,
              help="Only score filing rows on/after this date (the price scan is full-history).")
@click.option("--report", "report_path", default=None,
              help="Where to write the markdown report (default: reports/validate/<date>/).")
@click.option("--skip-spike", is_flag=True, default=False,
              help="Skip invariant 3, the only one that reads the full price history.")
@click.option("--no-write", is_flag=True, default=False, help="Print only; write no file.")
def prices(config_path: str, tickers: str | None, since: str | None,
           report_path: str | None, skip_spike: bool, no_write: bool) -> None:
    """The three adjustment-basis invariants. Reports; corrects nothing.

    Invariant 1 is the primary gate. Its failures are vendor DISAGREEMENTS -- Sharadar's own
    `marketcap` is internally inconsistent across spinoffs -- so they are recorded and
    settled, never auto-corrected. See src/validate/prices.py."""
    _, context = _ctx(config_path)
    names, _ = _roster_tickers(context, config_path, (), tickers)
    report = run_prices_validation(context, tickers=names, since=since,
                                   skip_spike=skip_spike)
    for result in report.invariants:
        context.log.info(result.summary())

    text = report.to_markdown()
    if no_write:
        click.echo(text)
        return
    path = Path(report_path) if report_path else _default_report_path("prices")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    context.log.info("Wrote %s", path)


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


def _require_quantified_note(note: str | None, subject: str) -> str:
    """The QUANTIFIED-NOTE rule, enforced in ONE place.

    `status set` and every `fix record --waive` are the same assertion -- "I looked at this,
    it is real, and it is not worth repairing" -- so they get the same test. A second copy of
    this rule is a copy that gets relaxed first.

    A number is a weak proxy for a measurement and it is deliberately weak: it cannot verify
    that the figure is right, only that somebody wrote one down. NEE's $5.2bn understatement
    is a defensible wontfix precisely and only because the number is on the record.
    """
    if not note or not note.strip():
        raise click.ClickException(
            f"{subject} needs a note and it may not be empty. A wontfix with no evidence is "
            f"a suppression with a label on it.")
    if not any(ch.isdigit() for ch in note):
        raise click.ClickException(
            f"{subject} must carry a QUANTIFIED cost -- a number, not an adjective. Got: "
            f"{note!r}. The rule being enforced is 'somebody measured it', and any real "
            f"measurement carries a numeral.")
    return note.strip()


def _latest_rows(ledger: Ledger, cluster_id: str, subject: str) -> pd.DataFrame:
    """The cluster's findings from the most recent run that saw it. Raises if never measured.

    A decision about a cluster nobody has measured cannot record how big it was at the time,
    so it could never reopen -- which makes it a permanent suppression rather than a
    self-expiring judgement.
    """
    rows = (ledger.findings[ledger.findings["cluster_id"] == cluster_id]
            if not ledger.findings.empty else pd.DataFrame())
    if rows.empty:
        raise click.ClickException(
            f"cluster {cluster_id!r} is not in fundamentals_check, so {subject} would rest on "
            f"nothing. Nothing has ever been measured about it.")
    return rows[rows["run_date"] == rows["run_date"].max()]


def _waiver_row(cluster_id: str, check: str, note: str, rows: pd.DataFrame) -> dict[str, Any]:
    """One `fundamentals_check_status` row, sized against the population it actually covers.

    A cluster-wide entry (`check == ''`) is decided against the whole cluster; a per-check one
    against that check alone. Measuring a `peer_ratio` waiver against the cluster total would
    reopen it whenever an unrelated check fired once more, which is not what anybody decided.
    """
    covered = rows if check == CLUSTER_WIDE else rows[rows["check_name"].astype(str) == check]
    return {
        "cluster_id": cluster_id, "check_name": check,
        "ticker": str(rows["ticker"].iloc[0]),
        "field": str(rows["field"].iloc[0] or ""),
        "status": WONTFIX, "note": note,
        # D8: the size the judgement was actually made against. One more finding reopens it.
        "findings_at_decision": int(len(covered)),
        "decided_at": date.today(),
    }


def _save_waivers(context: Context, rows: list[dict[str, Any]]) -> int:
    """Write waiver rows with `findings_at_decision` typed so a NULL cannot become 0.0."""
    frame = pd.DataFrame(rows)
    frame["findings_at_decision"] = frame["findings_at_decision"].astype("Int64")
    return context.store.save(Tables.fundamentals_check_status, frame)


@status.command("set", help="Mark a cluster (or one check on it) wontfix. QUANTIFIED --note.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.argument("cluster_id")
@click.option("--check", "check_name", default=CLUSTER_WIDE,
              help="Tolerate only THIS check's findings on the cluster. Default: the whole "
                   "cluster. Per-check is the narrowest tolerance that can be expressed, and "
                   "each expires on its OWN findings_at_decision.")
@click.option("--note", default=None,
              help="The evidence. Must contain a NUMBER -- a quantified cost, not an "
                   "adjective. NEE's $5.2bn understatement is a defensible wontfix precisely "
                   "and only because the number is written down.")
def status_set(config_path: str, cluster_id: str, check_name: str, note: str | None) -> None:
    note = _require_quantified_note(note, "--note")
    _, context = _ctx(config_path)
    ledger = Ledger.load(context)
    latest = _latest_rows(ledger, cluster_id, "a wontfix")
    check = (check_name or CLUSTER_WIDE).strip()
    if check and check not in set(latest["check_name"].astype(str)):
        raise click.ClickException(
            f"--check {check!r} has no finding on cluster {cluster_id!r} in its latest run "
            f"({sorted(set(latest['check_name'].astype(str)))}). Tolerating a check that is "
            f"not firing records a decision about nothing and can never expire.")

    row = _waiver_row(cluster_id, check, note, latest)
    _save_waivers(context, [row])
    context.log.info("validate status: %s %s (%s %s) -> wontfix at %d finding(s). It REOPENS "
                     "automatically if that population grows past it. A waiver alone NEVER "
                     "settles a cluster -- that needs a `validate fix record` row.",
                     cluster_id, f"[{check}]" if check else "[whole cluster]",
                     row["ticker"], row["field"], row["findings_at_decision"])


@status.command("clear", help="Remove a wontfix (or just one check's), putting it back.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.argument("cluster_id")
@click.option("--check", "check_name", default=None,
              help="Clear only this check's waiver. Default: every waiver on the cluster.")
def status_clear(config_path: str, cluster_id: str, check_name: str | None) -> None:
    _, context = _ctx(config_path)
    where: dict[str, Any] = {"cluster_id": cluster_id}
    if check_name is not None:
        where["check_name"] = check_name
    removed = context.store.delete(Tables.fundamentals_check_status, where)
    context.log.info("validate status: cleared %d waiver row(s) for cluster %s%s",
                     removed, cluster_id,
                     f" check {check_name!r}" if check_name is not None else "")


# --------------------------------------------------------------------------- #
# fix -- the record a row-count drop cannot make                               #
# --------------------------------------------------------------------------- #

def _resolve_runs(ledger: Ledger, cluster_id: str, after: str | None,
                  before: str | None) -> tuple[str, str, str]:
    """`(run_id_before, run_id_after, scope_hash)`, DERIVED unless overridden (decision 7).

    `after` defaults to the latest run that saw the cluster -- the run that PROVED the fix --
    and `before` to that run's previous COMPARABLE one. Asking a human to type two 12-hex
    ids they would have to look up is how a wrong id ends up on an evidence row.

    The scope hash equality is not a courtesy. Two runs of different scope are not a weaker
    comparison, they are not a comparison: a fix "proved" on a one-ticker re-validation
    against a 54-ticker baseline would claim ~11,800 findings closed.
    """
    rows = (ledger.findings[ledger.findings["cluster_id"] == cluster_id]
            if not ledger.findings.empty else pd.DataFrame())
    if after is None:
        if rows.empty:
            raise click.ClickException(
                f"cluster {cluster_id!r} is not in fundamentals_check, so there is no run to "
                f"record the fix against. Pass --after explicitly if you know the run id.")
        after = str(rows.loc[rows["run_date"].idxmax(), "run_id"])

    after_run = ledger.run(after)
    if after_run is None:
        raise click.ClickException(
            f"--after {after!r} is not in fundamentals_check_run. A fix is recorded against "
            f"the run that PROVED it, and an unrecorded run proved nothing.")
    if before is None:
        previous = ledger.previous_comparable(after)
        if previous is None:
            raise click.ClickException(
                f"run {after!r} has no earlier COMPARABLE run, so there is no before/after "
                f"to measure. Two runs are comparable only when their scope_hash matches; "
                f"re-run the validator at the same scope you fixed against.")
        before = previous.run_id

    before_run = ledger.run(before)
    if before_run is None:
        raise click.ClickException(f"--before {before!r} is not in fundamentals_check_run.")
    if before_run.scope_hash != after_run.scope_hash:
        raise click.ClickException(
            f"runs {before} and {after} have different scope hashes "
            f"({before_run.scope_hash} vs {after_run.scope_hash}), so differencing them is "
            f"not a comparison and cannot be proof. {before_run.label} vs {after_run.label}.")
    return before, after, after_run.scope_hash


def _counts(ledger: Ledger, cluster_id: str, run_id: str) -> tuple[int, int]:
    """`(findings, queued)` for one cluster in one run. Queue severities exclude `info`.

    Settlement is judged on the QUEUE count because nothing reads an `info` finding as work:
    a "fix" that closed only `info` rows closed no work, and counting them would let it claim
    an improvement it did not make.
    """
    rows = ledger.findings_for(run_id)
    rows = rows[rows["cluster_id"] == cluster_id] if not rows.empty else rows
    if rows.empty:
        return 0, 0
    queued = rows[rows["severity"].astype(str).isin(QUEUE_SEVERITIES)]
    return len(rows), len(queued)


def _validated_evidence(layer: str, evidence: str) -> str:
    """`evidence` parsed, shape-checked against its LAYER, and returned as canonical JSON.

    JSON, never prose -- the rule `fundamentals_check.detail` already follows, because an
    agent parsing English to decide what to trust is the failure mode this subsystem exists
    to remove. Prose belongs in `--root-cause`.

    The required keys differ by layer on purpose (decision 5). A `check` fix has no filing at
    fault, so demanding an accession would make it cite an irrelevant one; its evidence is
    the false-positive population it was measured against.
    """
    try:
        blob = json.loads(evidence)
    except (TypeError, ValueError) as exc:
        raise click.ClickException(
            f"--evidence must be parseable JSON, never prose: {exc}. Prose belongs in "
            f"--root-cause; this field is what a later reader QUERIES.")
    if not isinstance(blob, dict):
        raise click.ClickException(
            f"--evidence must be a JSON OBJECT, got {type(blob).__name__}. A bare list has "
            f"no key to say what it is a list of.")
    missing = sorted(FIX_EVIDENCE_KEYS[layer] - set(blob))
    if missing:
        raise click.ClickException(
            f"--evidence for layer {layer!r} is missing {missing}. Required: "
            f"{sorted(FIX_EVIDENCE_KEYS[layer])}. A `{layer}` fix that cannot name those has "
            f"not shown its work.")
    return json.dumps(blob, sort_keys=True)


def _verified_commit(commit: str) -> str:
    """The commit, confirmed to exist with `git rev-parse --verify`.

    An unresolvable sha is worse than none in a table whose entire purpose is evidence: it
    reads as a reproducible fix and is not one, and nobody discovers that until they try.
    """
    try:
        proc = subprocess.run(["git", "rev-parse", "--verify", f"{commit}^{{commit}}"],
                              capture_output=True, text=True, check=False)
    except OSError as exc:
        raise click.ClickException(f"could not run git to verify --commit {commit!r}: {exc}")
    if proc.returncode != 0:
        raise click.ClickException(
            f"--commit {commit!r} does not resolve to a commit in this repository "
            f"({proc.stderr.strip() or 'git rev-parse failed'}). An unreproducible fix record "
            f"is worse than a missing one.")
    return commit.strip()


def _verified_test(test_path: str) -> str:
    """The regression test, confirmed to exist on disk.

    Without one, the fix is an assertion. The test is what stops the same defect returning
    silently, and a path that does not exist cannot do that job.
    """
    if not Path(test_path).exists():
        raise click.ClickException(
            f"--test {test_path!r} does not exist. A fix with no regression test is an "
            f"assertion: nothing stops the same defect returning on the next refactor.")
    return test_path


def _parse_waivers(waive: tuple[str, ...], cluster_id: str,
                   rows: pd.DataFrame) -> list[dict[str, Any]]:
    """`--waive "check:quantified note"` -> waiver rows, each checked against a LIVE finding.

    Waiving a check that is not firing records a decision about nothing, and it can never
    expire -- exactly the permanent suppression that `findings_at_decision` exists to
    prevent. So the check must have a finding on this cluster in the run being recorded.
    """
    live = set(rows["check_name"].astype(str))
    out: list[dict[str, Any]] = []
    for raw in waive:
        check, _, note = raw.partition(":")
        check = check.strip()
        if not check or not note.strip():
            raise click.ClickException(
                f"--waive takes 'check_name:quantified note', got {raw!r}. The note is the "
                f"evidence and it is not optional.")
        if check not in live:
            raise click.ClickException(
                f"--waive {check!r}: that check has no finding on cluster {cluster_id!r} in "
                f"this run. Firing checks here: {sorted(live)}. Tolerating a check that is "
                f"not firing records a decision about nothing and can never expire.")
        out.append(_waiver_row(cluster_id, check,
                               _require_quantified_note(note, f"--waive {check!r}"), rows))
    return out


@cli.group(help="Record and read back an INTERVENTION on a cluster. Append-only; filters "
                "nothing.")
def fix() -> None:
    """What was done, why, at which layer, and what it measurably closed.

    A DIFFERENT KIND OF THING from `status`: a fix is an EVENT that happened and is never
    revised, a waiver is a STATE that persists and expires. Two fixes of one cluster are two
    rows, because the second did not un-happen the first.

    NOTHING HERE FILTERS A FINDING. A fix row records; it never subtracts, and every waived
    finding is still written, still counted and still fires. That is what keeps "fewer rows
    than the last comparable run" usable as proof.
    """


@fix.command("record", help="Record a fix and waive its benign residue, atomically.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.argument("cluster_id")
@click.option("--layer", required=True, type=click.Choice(sorted(FIX_LAYERS)),
              help="What the edit DID, not which file it lives in. check = the check was "
                   "wrong; catalogue = the field spec was wrong; extraction = any code that "
                   "PRODUCES a value; rows = the code was right and the stored data stale.")
@click.option("--root-cause", required=True,
              help="The prose explanation, in one sentence. This is the ONLY prose field -- "
                   "--evidence is JSON so a later reader can query it.")
@click.option("--evidence", required=True,
              help="JSON, never prose. Required keys vary by --layer: accessions for "
                   "extraction/rows/catalogue; examined+benign for check, which has no "
                   "filing at fault to cite.")
@click.option("--commit", required=True,
              help="The commit that carries the fix. Verified with `git rev-parse`.")
@click.option("--test", "test_path", required=True,
              help="The regression test that pins it. Verified to exist on disk.")
@click.option("--waive", multiple=True,
              help="'check_name:quantified note' for a benign residual finding. REPEATABLE. "
                   "Lands in the same transaction as the fix row -- recording a fix and "
                   "tolerating its residue is ONE decision.")
@click.option("--after", default=None,
              help="Override the run that PROVED the fix (default: the latest run that saw "
                   "the cluster).")
@click.option("--before", default=None,
              help="Override the baseline run (default: --after's previous comparable run).")
def fix_record(config_path: str, cluster_id: str, layer: str, root_cause: str, evidence: str,
               commit: str, test_path: str, waive: tuple[str, ...],
               after: str | None, before: str | None) -> None:
    _, context = _ctx(config_path)
    ledger = Ledger.load(context)

    # Everything is validated BEFORE anything is written. A refusal after a partial write is
    # the half-recorded state this command exists to remove.
    run_before, run_after, scope_hash = _resolve_runs(ledger, cluster_id, after, before)
    payload = _validated_evidence(layer, evidence)
    commit = _verified_commit(commit)
    test_path = _verified_test(test_path)

    latest = ledger.findings_for(run_after)
    latest = latest[latest["cluster_id"] == cluster_id] if not latest.empty else latest
    identity = latest if not latest.empty else _latest_rows(ledger, cluster_id, "a fix record")
    waivers = _parse_waivers(waive, cluster_id, latest) if len(waive) else []

    findings_before, queued_before = _counts(ledger, cluster_id, run_before)
    findings_after, queued_after = _counts(ledger, cluster_id, run_after)
    if findings_before == 0:
        raise click.ClickException(
            f"cluster {cluster_id!r} has no findings in the baseline run {run_before!r}, so "
            f"there is nothing this fix could have closed. Check --before.")

    frame = pd.DataFrame([{
        "cluster_id": cluster_id, "run_id_after": run_after, "run_id_before": run_before,
        "scope_hash": scope_hash,
        "ticker": str(identity["ticker"].iloc[0]),
        "field": str(identity["field"].iloc[0] or ""),
        "findings_before": findings_before, "findings_after": findings_after,
        "queued_before": queued_before, "queued_after": queued_after,
        "layer": layer, "root_cause": root_cause.strip(), "evidence": payload,
        "commit_sha": commit, "test_path": test_path, "decided_at": date.today(),
    }])
    for column in ("findings_before", "findings_after", "queued_before", "queued_after"):
        frame[column] = frame[column].astype("Int64")

    # Atomic (decision 6): the waivers first, the fix row last, and any failure undoes both.
    # Ordered that way so the failure mode is "nothing recorded" rather than "a fix row whose
    # cluster still reads OPEN". `prior` lets a rollback restore a waiver that already
    # existed instead of deleting somebody else's decision.
    prior = ledger.waivers_for(cluster_id)
    written = [row["check_name"] for row in waivers]
    try:
        if waivers:
            _save_waivers(context, waivers)
        context.store.save(Tables.fundamentals_check_fix, frame)
    except Exception:
        _rollback(context, cluster_id, run_after, written, prior)
        raise

    if queued_after >= queued_before:
        context.log.warning(
            "validate fix: %s closed NO queue findings (%d -> %d). The row is on the record "
            "-- correcting a wrong-but-plausible value where no check was firing is a real "
            "fix -- but it CANNOT settle the cluster. Settlement requires "
            "queued_after < queued_before.", cluster_id, queued_before, queued_after)
    context.log.info(
        "validate fix: recorded %s (%s %s) layer=%s | findings %d -> %d, queue %d -> %d | "
        "%s -> %s | commit %s | %d waiver(s): %s",
        cluster_id, frame["ticker"].iloc[0], frame["field"].iloc[0], layer,
        findings_before, findings_after, queued_before, queued_after,
        run_before, run_after, commit, len(waivers), written or "none")


def _rollback(context: Context, cluster_id: str, run_after: str,
              written: list[str], prior: dict[str, dict[str, Any]]) -> None:
    """Undo a half-completed `fix record`, restoring any waiver that existed beforehand.

    Restoring rather than just deleting: a `--waive` may have OVERWRITTEN somebody else's
    earlier decision on the same check, and a rollback that deletes it turns a failed write
    into silent data loss.
    """
    context.store.delete(Tables.fundamentals_check_fix,
                         {"cluster_id": cluster_id, "run_id_after": run_after})
    restore = [prior[check] for check in written if check in prior]
    drop = [check for check in written if check not in prior]
    if drop:
        context.store.delete(Tables.fundamentals_check_status,
                             {"cluster_id": cluster_id, "check_name": drop})
    if restore:
        _save_waivers(context, restore)
    context.log.error("validate fix: %s FAILED and was rolled back -- %d waiver(s) removed, "
                      "%d restored. Nothing was half-recorded.",
                      cluster_id, len(drop), len(restore))


@fix.command("show", help="Every recorded fix on a cluster, its waivers, and what is pending.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.argument("cluster_id")
def fix_show(config_path: str, cluster_id: str) -> None:
    """The read-back: reason / what was done / before / after / still pending, in one query."""
    _, context = _ctx(config_path)
    ledger = Ledger.load(context)
    history = ledger.fixes_for(cluster_id)
    waivers = ledger.waivers_for(cluster_id)
    rows = (ledger.findings[ledger.findings["cluster_id"] == cluster_id]
            if not ledger.findings.empty else pd.DataFrame())

    lines = [f"cluster {cluster_id}", ""]
    if rows.empty:
        lines.append("  never measured -- not in fundamentals_check")
    else:
        latest_run = str(rows.loc[rows["run_date"].idxmax(), "run_id"])
        latest = rows[rows["run_id"] == latest_run]
        queued = latest[latest["severity"].astype(str).isin(QUEUE_SEVERITIES)]
        waived_checks = ({str(c) for c in latest["check_name"]} if CLUSTER_WIDE in waivers
                         else set(waivers))
        pending = queued[~queued["check_name"].astype(str).isin(waived_checks)]
        lines += [f"  {latest['ticker'].iloc[0]} `{latest['field'].iloc[0]}` "
                  f"as of run {latest_run}",
                  f"  {len(latest)} finding(s), {len(queued)} at queue severity, "
                  f"{len(pending)} of those UNWAIVED",
                  ""]
        if len(pending):
            mix = pending["check_name"].astype(str).value_counts().to_dict()
            lines.append(f"  still pending: {mix}")
        else:
            lines.append("  nothing unwaived at queue severity remains")

    lines += ["", f"  {len(history)} recorded fix(es):"]
    if not history:
        lines.append("    none. A fix that was never recorded left only a commit sha -- "
                     "which is the gap `fix record` exists to close.")
    for record in history:
        stamp = record.decided_at.date() if record.decided_at is not None else "?"
        lines += [
            f"    {stamp}  layer={record.layer}  commit={record.commit_sha}",
            f"      why      : {record.root_cause}",
            f"      measured : findings {record.findings_before} -> {record.findings_after}, "
            f"queue {record.queued_before} -> {record.queued_after} "
            f"({record.run_id_before} -> {record.run_id_after})",
            f"      evidence : {json.dumps(record.evidence_json, sort_keys=True)}",
            f"      test     : {record.test_path}",
            f"      settles  : {'yes' if record.improved else 'NO -- it closed no queue finding'}",
        ]

    lines += ["", f"  {len(waivers)} waiver(s):"]
    for check, row in sorted(waivers.items()):
        label = check or "(whole cluster)"
        lines.append(f"    {label:<28} at {row.get('findings_at_decision')} finding(s): "
                     f"{row.get('note')}")
    if not waivers:
        lines.append("    none")
    lines += ["", "  A waiver alone NEVER settles a cluster: settlement also needs a fix row "
                  "at this scope that reduced the queue. Nothing above filters a finding -- "
                  "every row is still in fundamentals_check."]
    context.log.info("\n%s", "\n".join(lines))


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
