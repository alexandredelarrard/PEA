"""
cli.py  (src/validate/cli.py)
--------------------------------------------------------------------------------------------
VALIDATION command-line interface -- part 2 of the three-part loop. Invoked as:

    python -m src validate fundamentals [-t AAPL] [--roster in_sample] [--field capex]
                                        [--tier 1,2,3] [--since 2026-01-01]
                                        [--report PATH] [--no-write]

One command per DOMAIN, so a future prices or insider validator is a sibling rather than a
flag. Every form is read-only against every table but `fundamentals_check`, and NOTHING here
gates anything: the nightly extraction runs to completion whatever this reports.

## The forms, and what each is for

    --roster in_sample        the tuned stress set. A pass proves CONSISTENCY, not
                              generalisation -- every rule in the resolver was tuned on it
    --roster out_of_sample    zero overlap, never tuned. A finding here is a genuine
                              generalisation failure
    --roster random_cold      the only honest estimate of the error rate on an arbitrary
                              ticker; both designed rosters measure robustness to KNOWN-HARD
                              shapes instead
    --field X                 the NEW-FIELD ACCEPTANCE SHEET (decision 44). A catalogue field
                              is born `status: probation`; promotion to `active` requires this
                              sheet clean, or its gaps recorded in fundamentals_check.json
    --tier 1                  the nightly full-table pass. Tiers 2-3 nightly run only on
                              tickers that received a filing (decision 53) -- a series can
                              only change where a filing landed
    --no-write                print and write a report, touch no table. The default for
                              exploring a threshold
"""
import json
from pathlib import Path

import click

from src.constants.command_line_interface import (
    CONFIG_ARGS, CONFIG_KWARGS, TICKERS_ARGS, TICKERS_KWARGS)
from src.context import Context, get_config_context
from src.utils.cli_helper import SpecialHelpOrder
from src.utils.universe import load_universe_tickers
from src.validate.fundamentals import report as report_module
from src.validate.fundamentals.validator import FundamentalsValidator

#: Where the rosters live. Read here rather than re-declared: `fundamentals_rosters.json`
#: records WHY each ticker is on its list, which is the property a bare list of symbols loses
#: and the reason a roster is worth having at all.
ROSTERS_PATH = "fundamentals/fundamentals_rosters.json"

#: `--roster all` is the full universe, on demand. Named rather than implied so that a nightly
#: job cannot reach it by leaving a flag off.
ROSTER_ALL = "all"


@click.group(cls=SpecialHelpOrder)
def cli() -> None:
    """VALIDATION — read-only audits. Writes findings; gates nothing."""


def _ctx(config_path: str) -> tuple[object, Context]:
    return get_config_context(config_path, use_cache=False, save=False)


def _roster_tickers(context: Context, config_path: str, roster: str | None,
                    tickers: str | None) -> list[str] | None:
    """The ticker scope: `--tickers` wins, then `--roster`, then None (= everything loaded).

    None rather than the full universe when neither is given: `store.load` with no `where`
    reads the whole table in one statement, which is what we want, whereas passing 500 symbols
    builds a 500-element IN clause for no benefit.
    """
    if tickers:
        return [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not roster:
        return None
    if roster == ROSTER_ALL:
        return load_universe_tickers(context)
    blob = json.loads((Path(config_path) / ROSTERS_PATH).read_text(encoding="utf-8"))
    names = blob.get(roster)
    if not names:
        available = sorted(k for k in blob if not k.startswith("_"))
        raise click.BadParameter(f"roster {roster!r} is not in {ROSTERS_PATH}; "
                                 f"available: {available + [ROSTER_ALL]}")
    return [str(t).upper() for t in names]


def _tiers(tier: str | None) -> list[int] | None:
    """`--tier 1,3` as `[1, 3]`. None means every tier."""
    if not tier:
        return None
    try:
        return sorted({int(t.strip()) for t in tier.split(",") if t.strip()})
    except ValueError as exc:
        raise click.BadParameter(f"--tier takes a comma-separated list of 1,2,3: {exc}")


@cli.command(help="Validate fundamentals_history / fundamentals_facts. Read-only; gates nothing.",
             help_priority=1)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option("--roster", default=None,
              help="A named roster from configs/fundamentals/fundamentals_rosters.json "
                   "(in_sample, out_of_sample, amendment_pair, random_cold), or 'all' for the "
                   "full universe. Ignored when --tickers is given.")
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
              help="Write the markdown report here as well as printing it.")
@click.option("--no-write", is_flag=True, default=False,
              help="Do not append to fundamentals_check. Print and report only.")
def fundamentals(config_path: str, tickers: str | None, roster: str | None,
                 fields: str | None, tier: str | None, check_names: str | None,
                 since: str | None, report_path: str | None, no_write: bool) -> None:
    _, context = _ctx(config_path)
    scope = _roster_tickers(context, config_path, roster, tickers)
    tiers = _tiers(tier)
    names = [c.strip() for c in check_names.split(",") if c.strip()] if check_names else None
    field_list = [f.strip() for f in fields.split(",") if f.strip()] if fields else None

    context.log.info("validate fundamentals: %s ticker(s), tiers=%s, checks=%s, fields=%s",
                     len(scope) if scope else "all", tiers or "all", names or "all",
                     field_list or "all")
    validator = FundamentalsValidator.from_context(
        context, tickers=scope, config_dir=config_path, tiers=tiers, since=since)
    run = validator.run(tiers=tiers, names=names, fields=field_list)

    rendered = report_module.render(run)
    context.log.info("\n%s", rendered)
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(rendered, encoding="utf-8")
        context.log.info("validate fundamentals: report -> %s", report_path)

    if no_write:
        context.log.info("validate fundamentals: --no-write, %d finding(s) NOT persisted",
                         len(run.findings))
        return
    FundamentalsValidator.write(context, run)


@cli.command(help="Print CHECK_REGISTRY: every check, its tier, substrate, severity and ceiling.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
def checks(config_path: str) -> None:
    """The registry itself, so 'what does this tool actually test?' needs no source dive."""
    from src.validate.fundamentals.checks import CHECK_REGISTRY

    _, context = _ctx(config_path)
    rows = sorted(CHECK_REGISTRY.values(), key=lambda s: (s.tier, s.name))
    lines = [f"{len(rows)} registered check(s)", ""]
    for spec in rows:
        lines.append(f"  tier {spec.tier}  {spec.name:24s} {spec.substrate:8s} "
                     f"{spec.severity:9s} grain={spec.grain:7s} "
                     f"ceiling={spec.expected_fire_rate_ceiling:.1%}")
        lines.append(f"      {spec.doc}")
    context.log.info("\n%s", "\n".join(lines))
