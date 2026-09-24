"""
cli.py  (src/validate/cli.py)
--------------------------------------------------------------------------------------------
One command per check, plus the `pull` that materialises the snapshot they share:

    python -m src validate pull      -T cube_part_institutionals -o reports/validate/<slug>
    python -m src validate grain     -T cube_part_institutionals -o reports/validate/<slug>
    ... coverage profile redundancy leakage clip timeseries bounds catalogue

EXIT CODES ARE THE CONTRACT: 0 pass · 1 a finding stands · **3 the check ABSTAINED**. Three is
not two: a table that declares no clip convention has no on-clip share, and "0 legs over the
limit" reads identically whether the check looked and found none or never looked at all. A
zero from a check that abstained is not a pass.

This module owns every `print()` and every `sys.exit()` in the package -- the check functions
log through `logging.getLogger(__name__)` and return a `CheckResult`, so they compose from a
report's `_scripts/` without a process boundary.

⚠ The group MUST be named `cli`: `src/cli.py` does not import this module, it `eval`s the file
and pulls the `cli` symbol out of the resulting namespace, so a rename or a top-level import
error surfaces only as a bare "Command not found".
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path

import click

from src.constants.command_line_interface import CONFIG_ARGS, CONFIG_KWARGS, TICKERS_ARGS, TICKERS_KWARGS
from src.context import get_config_context
from src.data_store.schema import resolve
from src.utils.cli_helper import SpecialHelpOrder
from src.validate import checks
from src.validate.insider_reconciliation import (
    run_completed_quarter_reconciliation,
    write_reconciliation_report,
)
from src.validate.io import pull as pull_snapshot
from src.validate.io import write_result
from src.validate.result import EXIT, CheckResult
from src.validate.spec import UndeclaredTableError

TABLE_ARGS = ("-T", "--table")
TABLE_KWARGS = dict(required=True, help="Table to validate, as named in `Tables` (src/data_store/schema.py).")

OUT_ARGS = ("-o", "--out")
OUT_KWARGS = dict(
    required=True, help="Run directory -- _cache/, _out/ and plots/ are created inside it. " "Use reports/validate/<slug>; NEVER a path under src/."
)


def _shared(fn: Callable) -> Callable:
    """The options every check command takes. Declared once so `--out` cannot mean two
    things in two commands."""
    for decorator in (
        click.option(*CONFIG_ARGS, **CONFIG_KWARGS),
        click.option(*TICKERS_ARGS, **TICKERS_KWARGS),
        click.option(
            "--cache/--no-cache", "use_cache", default=True, show_default=True, help="Read the `pull` snapshot in <out>/_cache when one exists."
        ),
        click.option(*OUT_ARGS, **OUT_KWARGS),
        click.option(*TABLE_ARGS, **TABLE_KWARGS),
    ):
        fn = decorator(fn)
    return fn


@click.group(cls=SpecialHelpOrder)
def cli() -> None:
    """VALIDATE — table-agnostic data checks. 0 pass · 1 finding · 3 ABSTAINED."""


@cli.command(help="Stream one table into <out>/_cache/<table>.parquet, reused by every check.")
@click.option(*TABLE_ARGS, **TABLE_KWARGS)
@click.option(*OUT_ARGS, **OUT_KWARGS)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
def pull(table: str, out: str, config_path: str) -> None:
    _, context = get_config_context(config_path, use_cache=False, save=False)
    started = time.perf_counter()
    path = pull_snapshot(context, resolve(table), out)
    size_mb = path.stat().st_size / 1e6
    click.echo(f"pulled {table} -> {path} ({size_mb:,.1f} MB, {time.perf_counter() - started:,.1f}s)")


def _run(check: Callable[..., CheckResult], table: str, out: str, use_cache: bool, config_path: str, tickers: str | None, **kwargs) -> None:
    """Build the context, run one check, write its JSON, print its summary, exit on its status.

    An `UndeclaredTableError` from anywhere inside the check becomes ABSTAIN, never a pass:
    that exception exists precisely so a missing declaration cannot be read as a clean result.
    """
    config, context = get_config_context(config_path, use_cache=False, save=False)
    spec = resolve(table)
    started = time.perf_counter()
    names = [t.strip().upper() for t in tickers.split(",")] if tickers else None
    try:
        # `out` as well as `cache`: they are not the same thing. `--no-cache` says "do not
        # read the parquet snapshot", while `out` is where this run's other checks wrote
        # their JSON -- `redundancy` reads `profile`'s means from there and should still
        # find them on a run that deliberately bypassed the snapshot.
        result = check(context, spec, config=config, cache=out if use_cache else None, out=out, tickers=names, **kwargs)
    except UndeclaredTableError as exc:
        result = CheckResult.abstained(check.__name__.removeprefix("check_"), spec.name, str(exc))
    result.scope.setdefault("elapsed_s", round(time.perf_counter() - started, 1))
    path = write_result(out, result)
    click.echo(result.summary())
    click.echo(f"  -> {path}")
    sys.exit(EXIT[result.status])


def _command(name: str, check: Callable[..., CheckResult], help_text: str):
    """Register one check as one command. The nine differ only in the function they call, so
    hand-writing nine identical bodies would be nine places for the exit contract to drift."""

    @cli.command(name=name, help=help_text)
    @_shared
    def _cmd(table: str, out: str, use_cache: bool, tickers: str | None, config_path: str) -> None:
        _run(check, table, out, use_cache, config_path, tickers)

    return _cmd


grain = _command("grain", checks.check_grain, "Rows vs distinct declared-pk keys, panel edges, nulls in a key column.")
coverage = _command("coverage", checks.check_coverage, "Tickers present vs the 491-name universe, each against its own prices span.")
profile = _command("profile", checks.check_profile, "Per-column n/min/p01/p50/p99/max/sd/n_distinct, dead and constant legs.")
redundancy = _command("redundancy", checks.check_redundancy, "Exact pairwise-complete Pearson over every pair, plus exact-equality counts.")
leakage = _command("leakage", checks.check_leakage, "Horizon recession on the labels, and each feature against its source's publication clock.")
clip = _command("clip", checks.check_clip, "On-clip share per peer-z leg and tie mass per percentile leg.")
timeseries = _command("timeseries", checks.check_timeseries, "Per (ticker, leg): jumps, holes and frozen spells over the leg's own support.")
bounds = _command("bounds", checks.check_bounds, "Declared [lo, hi] per leg; abstains when the table declares none.")


@cli.command(help="Live columns vs a feature catalogue, asserted in BOTH directions.")
@_shared
@click.option(
    "--catalogue",
    "catalogue_path",
    default=None,
    type=click.Path(path_type=Path),
    help="JSON mapping field -> description, or a .py exposing a dict named CATALOGUE. " "Absent -> the check abstains.",
)
def catalogue(table: str, out: str, use_cache: bool, tickers: str | None, config_path: str, catalogue_path: Path | None) -> None:
    _run(checks.check_catalogue, table, out, use_cache, config_path, tickers, catalogue=catalogue_path)


@cli.command(
    name="insider-parity",
    help="Replay a completed Form 3/4/5 quarter through EDGAR and gate ZIP promotion.",
)
@click.option("--quarter", required=True, metavar="YYYYQn")
@click.option("--workers", default=8, show_default=True, type=click.IntRange(1, 16))
@click.option(
    "--refresh-replay",
    is_flag=True,
    help="Ignore the retained EDGAR replay cache and fetch the quarter again.",
)
@click.option(*OUT_ARGS, **OUT_KWARGS)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
def insider_parity(
    quarter: str,
    workers: int,
    refresh_replay: bool,
    out: str,
    config_path: str,
) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=False)
    result = run_completed_quarter_reconciliation(
        context,
        config,
        quarter,
        max_workers=workers,
        replay_cache=Path(out) / "_cache" / f"edgar_ownership_{quarter.upper()}.parquet",
        refresh_replay=refresh_replay,
    )
    json_path, markdown_path = write_reconciliation_report(out, result)
    click.echo(f"insider parity {result['quarter']}: " f"{'PASS' if result['passed'] else 'FAIL'} -> {json_path} ({markdown_path})")
    sys.exit(0 if result["passed"] else 1)
