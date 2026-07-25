"""
cli.py  (src/data_aggregate/cli.py)
-----------------------------------
DATA-AGGREGATION CLI — the cube build EXPLODED into small, memory-light, parallelisable steps for
the Airflow `data_aggregation` DAG:

    python -m src data_aggregate build-target                 # -> cube_part_targets / cube_part_betas
    python -m src data_aggregate features -g fundamental      # -> cube_part_fundamental
    python -m src data_aggregate features -g price            #    (one task per feature group)
    ...
    python -m src data_aggregate assemble-cube                # read all parts -> build + save `cube`

Each `features`/`build-target` step loads only prices + peers + its own source table(s) (never all
tables at once) and persists a compact part; `assemble-cube` reads the parts and writes the cube.
"""
import json

import click

from src.constants.command_line_interface import CONFIG_ARGS, CONFIG_KWARGS
from src.context import get_config_context
from src.utils.cli_helper import SpecialHelpOrder
from src.data_aggregate.step_build_cube import StepBuildCube


@click.group(cls=SpecialHelpOrder)
def cli() -> None:
    """DATA AGGREGATION — exploded cube build (target, per-group features, assemble)."""


def _step(config_path: str) -> StepBuildCube:
    config, context = get_config_context(config_path, use_cache=False, save=False)
    return StepBuildCube(context=context, config=config)


@cli.command(help="Build multi-horizon targets + betas -> persist cube_part_targets / cube_part_betas. "
                  "Incremental by default (refreshes the trailing maturing-target window); "
                  "--full forces a rebuild.",
             help_priority=1)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option("-F", "--full", is_flag=True, default=False, help="Force a full rebuild (ignore the stored part).")
def build_target(config_path: str, full: bool) -> None:
    _step(config_path).run_target(full=full)


@cli.command(help="Build ONE feature group -> persist cube_part_<group>. "
                  "Groups: price, fundamental, sector, earnings, governance, employee, dividend, "
                  "attention, institutional, superinvestor, insider, short_interest, earnings_call. "
                  "Incremental by default (appends only new dates); --full forces a rebuild.",
             help_priority=2)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option("-g", "--group", required=True, help="Feature group name (see help).")
@click.option("-F", "--full", is_flag=True, default=False, help="Force a full rebuild (ignore the stored part).")
def features(config_path: str, group: str, full: bool) -> None:
    _step(config_path).run_feature_group(group, full=full)


@cli.command(help="Read all persisted parts -> assemble features+composites+betas+peers+targets -> "
                  "save the `cube` table.", help_priority=3)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
def assemble_cube(config_path: str) -> None:
    _step(config_path).assemble_cube_from_parts()


@cli.command(help="Report the latest date + row count of every cube_part_* (+ cube / predictions) as "
                  "a JSON last line (the DAG pushes it to XCom); exits non-zero if any part is behind.",
             help_priority=4)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
def cube_status(config_path: str) -> None:
    report = _step(config_path).cube_parts_status()
    click.echo(json.dumps(report, separators=(",", ":")))
    if not report["ok"]:
        raise SystemExit(2)
