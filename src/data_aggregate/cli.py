"""
cli.py  (src/data_aggregate/cli.py)
-----------------------------------
DATA-AGGREGATION CLI — the cube build as eight sequential, memory-bounded steps, one command
each, for the Airflow `data_aggregation` DAG:

    python -m src data_aggregate build-prices          # -> cube_part_prices
    python -m src data_aggregate build-target          # -> cube_part_targets + cube_part_betas
    python -m src data_aggregate build-fundamentals    # -> cube_part_fundamentals
    python -m src data_aggregate build-momentum        # -> cube_part_momentum
    python -m src data_aggregate build-text            # -> cube_part_text
    python -m src data_aggregate build-institutionals  # -> cube_part_institutionals
    python -m src data_aggregate build-governance      # -> cube_part_governance
    python -m src data_aggregate build-manager-profile # -> profile columns of sec_13f_manager_details (not a cube part)
    python -m src data_aggregate assemble-cube         # read the parts -> build + save `cube`
    python -m src data_aggregate cube-status           # JSON status of every part
    python -m src data_aggregate build-cube            # all eight in ONE process

`build-prices` normalizes the raw `prices` table ONCE (pivot, trading calendar, returns,
universe restriction, peer sector returns); every later step reads those part tables back
projected to the fields it needs, instead of re-loading and re-pivoting ~1.9M price rows.

Each command is incremental by default: it reads its part's latest date, recomputes only a
warm-up-padded trailing window, and appends. `--full` forces a rebuild.

The old `features -g <group>` command is gone: it existed only to drive a
`getattr(self, method_name_string)` dispatch over fourteen feature groups, which the sub-step
split replaces.
"""

import json
from datetime import datetime

import click

from src.constants.command_line_interface import (
    AS_OF_ARGS,
    AS_OF_KWARGS,
    CONFIG_ARGS,
    CONFIG_KWARGS,
    FULL_ARGS,
    FULL_KWARGS,
    TICKERS_ARGS,
    TICKERS_KWARGS,
)
from src.context import Context, get_config_context
from src.data_aggregate.step_build_cube import StepBuildCube
from src.data_aggregate.transformers.step_assemble_cube import StepAssembleCube
from src.data_aggregate.transformers.step_cube_fundamentals import StepCubeFundamentals
from src.data_aggregate.transformers.step_cube_governance import StepCubeGovernance
from src.data_aggregate.transformers.step_cube_institutionals import StepCubeInstitutionals
from src.data_aggregate.transformers.step_cube_momentum import StepCubeMomentum
from src.data_aggregate.transformers.step_cube_prices import StepCubePrices
from src.data_aggregate.transformers.step_cube_sec_thirteen import StepCubeSecThirteen
from src.data_aggregate.transformers.step_cube_target import StepCubeTarget
from src.data_aggregate.transformers.step_cube_text import StepCubeText
from src.data_aggregate.transformers.step_manager_profile import StepManagerProfile
from src.data_aggregate.utils.common.assembly import read_receipts, write_receipt
from src.utils.cli_helper import SpecialHelpOrder
from src.utils.step import Step


@click.group(cls=SpecialHelpOrder)
def cli() -> None:
    """DATA AGGREGATION — the cube build as eight sequential steps."""


def _step[S: Step](cls: type[S], config_path: str, full: bool, as_of: datetime | None) -> tuple[S, Context]:
    config, context = get_config_context(config_path, use_cache=False, save=False, full=full, as_of=as_of)
    return cls(context=context, config=config), context


@cli.command(
    help="Normalize `prices` -> cube_part_prices (pivot, trading "
    "calendar, returns, universe restriction, peer sector returns). Every other "
    "step reads these instead of re-loading the whole prices table. `-t X,Y -F` deletes and rebuilds only X and Y.",
    help_priority=1,
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
def build_prices(config_path: str, tickers: str | None, full: bool, as_of: datetime | None) -> None:
    if tickers and not full:
        raise click.UsageError("-t/--tickers rebuilds those tickers from scratch and needs -F/--full")
    step, context = _step(StepCubePrices, config_path, full, as_of)
    click.echo(write_receipt(context, step.run(tickers=tickers.split(",") if tickers else [])))


@cli.command(
    help="Factor panel + rolling betas + multi-horizon targets -> cube_part_targets / cube_part_betas. Refreshes the trailing maturing-label window.",
    help_priority=2,
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
def build_target(config_path: str, full: bool, as_of: datetime | None) -> None:
    step, context = _step(StepCubeTarget, config_path, full, as_of)
    click.echo(write_receipt(context, step.run()))


@cli.command(help="SEC-filing features (fundamental, sector KPI, earnings, workforce, dividend) -> cube_part_fundamentals.", help_priority=3)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
def build_fundamentals(config_path: str, full: bool, as_of: datetime | None) -> None:
    step, context = _step(StepCubeFundamentals, config_path, full, as_of)
    click.echo(write_receipt(context, step.run()))


@cli.command(
    help="Price-variation features (momentum, reversal, vol, trend, lottery, liquidity, seasonality, MACD/RSI/ATR) -> cube_part_momentum.",
    help_priority=4,
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
def build_momentum(config_path: str, full: bool, as_of: datetime | None) -> None:
    step, context = _step(StepCubeMomentum, config_path, full, as_of)
    click.echo(write_receipt(context, step.run()))


@cli.command(help="Earnings-call text features (FinBERT role sentiment + embedding coherence & drift) -> cube_part_text.", help_priority=5)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
def build_text(config_path: str, full: bool, as_of: datetime | None) -> None:
    step, context = _step(StepCubeText, config_path, full, as_of)
    click.echo(write_receipt(context, step.run()))


@cli.command(help="Insider, beneficial ownership and short-flow features -> cube_part_institutionals.", help_priority=6)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
def build_institutionals(config_path: str, full: bool, as_of: datetime | None) -> None:
    step, context = _step(StepCubeInstitutionals, config_path, full, as_of)
    click.echo(write_receipt(context, step.run()))


@cli.command(help="Broad institutional and elite-manager 13F features -> cube_part_sec_thirteen.", help_priority=7)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
def build_sec_thirteen(config_path: str, full: bool, as_of: datetime | None) -> None:
    step, context = _step(StepCubeSecThirteen, config_path, full, as_of)
    click.echo(write_receipt(context, step.run()))


@cli.command(
    help="DEF 14A + Item 5.07 governance alpha (shareholder dissent, executive pay, governance provisions) -> cube_part_governance.",
    help_priority=8,
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
def build_governance(config_path: str, full: bool, as_of: datetime | None) -> None:
    step, context = _step(StepCubeGovernance, config_path, full, as_of)
    click.echo(write_receipt(context, step.run()))


@cli.command(
    help="Read configured columns from required/populated optional parts onto price keys; save the cube including unlabelled prediction rows.",
    help_priority=9,
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
@click.option("--component-windows", envvar="CUBE_COMPONENT_WINDOWS", help="Successful upstream write receipts supplied by the DAG.")
def assemble_cube(config_path: str, full: bool, as_of: datetime | None, component_windows: str | None) -> None:
    step, context = _step(StepAssembleCube, config_path, full, as_of)
    written = read_receipts(context, component_windows) if component_windows is not None else None
    click.echo(write_receipt(context, step.run(written)))


@cli.command(
    help="Run all eight sub-steps in ONE process (what main.py does): prices -> "
    "target -> fundamentals -> momentum -> text -> institutionals -> sec-thirteen -> governance -> "
    "assemble.",
    help_priority=10,
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
def build_cube(config_path: str, full: bool, as_of: datetime | None) -> None:
    step, context = _step(StepBuildCube, config_path, full, as_of)
    click.echo(write_receipt(context, step.run()))


@cli.command(
    help="13F manager category and 4-quarter behavioural profile -> profile columns of sec_13f_manager_details (not a cube part). "
    "Incremental: profiles the quarters whose rows lack a profile or whose name changed; `--full` rebuilds every quarter.",
    help_priority=12,
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
def build_manager_profile(config_path: str, full: bool, as_of: datetime | None) -> None:
    step, _ = _step(StepManagerProfile, config_path, full, as_of)
    step.run()


@cli.command(
    help="Report the latest date + row count of every cube_part_* (+ cube / "
    "predictions) as a JSON last line (the DAG pushes it to XCom); exits "
    "non-zero if required or populated optional parts/cube differ from the latest stored universe price date.",
    help_priority=11,
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
def cube_status(config_path: str, as_of: datetime | None) -> None:
    step, _ = _step(StepBuildCube, config_path, False, as_of)
    report = step.cube_parts_status()
    click.echo(json.dumps(report, separators=(",", ":")))
    if not report["ok"]:
        raise SystemExit(2)
