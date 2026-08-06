"""
cli.py  (src/data_aggregate/cli.py)
-----------------------------------
DATA-AGGREGATION CLI — the cube build as seven sequential, memory-bounded steps, one command
each, for the Airflow `data_aggregation` DAG:

    python -m src data_aggregate build-prices          # -> cube_part_prices + cube_part_market
    python -m src data_aggregate build-target          # -> cube_part_targets + cube_part_betas
    python -m src data_aggregate build-fundamentals    # -> cube_part_fundamentals
    python -m src data_aggregate build-momentum        # -> cube_part_momentum
    python -m src data_aggregate build-text            # -> cube_part_text
    python -m src data_aggregate build-extras          # -> cube_part_extras
    python -m src data_aggregate assemble-cube         # read the parts -> build + save `cube`
    python -m src data_aggregate cube-status           # JSON status of every part
    python -m src data_aggregate build-cube            # all seven in ONE process

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

import click

from src.constants.command_line_interface import CONFIG_ARGS, CONFIG_KWARGS
from src.context import get_config_context
from src.data_aggregate.step_build_cube import StepBuildCube
from src.data_aggregate.transformers.step_assemble_cube import StepAssembleCube
from src.data_aggregate.transformers.step_cube_extras import StepCubeExtras
from src.data_aggregate.transformers.step_cube_fundamentals import StepCubeFundamentals
from src.data_aggregate.transformers.step_cube_momentum import StepCubeMomentum
from src.data_aggregate.transformers.step_cube_prices import StepCubePrices
from src.data_aggregate.transformers.step_cube_target import StepCubeTarget
from src.data_aggregate.transformers.step_cube_text import StepCubeText
from src.utils.cli_helper import SpecialHelpOrder

_FULL_HELP = "Force a full rebuild (ignore the stored part)."


@click.group(cls=SpecialHelpOrder)
def cli() -> None:
    """DATA AGGREGATION — the cube build as seven sequential steps."""


def _step(cls, config_path: str):
    config, context = get_config_context(config_path, use_cache=False, save=False)
    return cls(context=context, config=config)


@cli.command(help="Normalize `prices` -> cube_part_prices + cube_part_market (pivot, trading "
                  "calendar, returns, universe restriction, peer sector returns). Every other "
                  "step reads these instead of re-loading the whole prices table.",
             help_priority=1)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option("-F", "--full", is_flag=True, default=False, help=_FULL_HELP)
def build_prices(config_path: str, full: bool) -> None:
    _step(StepCubePrices, config_path).run(full=full)


@cli.command(help="Factor panel + rolling betas + multi-horizon targets -> cube_part_targets "
                  "/ cube_part_betas. Refreshes the trailing maturing-label window.",
             help_priority=2)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option("-F", "--full", is_flag=True, default=False, help=_FULL_HELP)
def build_target(config_path: str, full: bool) -> None:
    _step(StepCubeTarget, config_path).run(full=full)


@cli.command(help="SEC-filing features (fundamental, sector KPI, earnings, workforce, "
                  "dividend) -> cube_part_fundamentals.", help_priority=3)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option("-F", "--full", is_flag=True, default=False, help=_FULL_HELP)
def build_fundamentals(config_path: str, full: bool) -> None:
    _step(StepCubeFundamentals, config_path).run(full=full)


@cli.command(help="Price-variation features (momentum, reversal, vol, trend, lottery, "
                  "liquidity, seasonality, MACD/RSI/ATR) -> cube_part_momentum.",
             help_priority=4)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option("-F", "--full", is_flag=True, default=False, help=_FULL_HELP)
def build_momentum(config_path: str, full: bool) -> None:
    _step(StepCubeMomentum, config_path).run(full=full)


@cli.command(help="Earnings-call text features (FinBERT/LM sentiment + OpenAI-embedding "
                  "coherence & drift) -> cube_part_text.", help_priority=5)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option("-F", "--full", is_flag=True, default=False, help=_FULL_HELP)
def build_text(config_path: str, full: bool) -> None:
    _step(StepCubeText, config_path).run(full=full)


@cli.command(help="Governance, 13F institutional, elite 13F, insider, short interest and "
                  "attention features -> cube_part_extras.", help_priority=6)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option("-F", "--full", is_flag=True, default=False, help=_FULL_HELP)
def build_extras(config_path: str, full: bool) -> None:
    _step(StepCubeExtras, config_path).run(full=full)


@cli.command(help="Read all persisted parts -> features + composites + betas + peers + "
                  "targets -> save the `cube` table.", help_priority=7)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
def assemble_cube(config_path: str) -> None:
    _step(StepAssembleCube, config_path).run()


@cli.command(help="Run all seven sub-steps in ONE process (what main.py does): prices -> "
                  "target -> fundamentals -> momentum -> text -> extras -> assemble.",
             help_priority=8)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option("-F", "--full", is_flag=True, default=False, help=_FULL_HELP)
def build_cube(config_path: str, full: bool) -> None:
    _step(StepBuildCube, config_path).run(full=full)


@cli.command(help="Report the latest date + row count of every cube_part_* (+ cube / "
                  "predictions) as a JSON last line (the DAG pushes it to XCom); exits "
                  "non-zero if any part is behind.", help_priority=9)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
def cube_status(config_path: str) -> None:
    report = _step(StepBuildCube, config_path).cube_parts_status()
    click.echo(json.dumps(report, separators=(",", ":")))
    if not report["ok"]:
        raise SystemExit(2)
