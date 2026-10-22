"""Run the backtest under the exact assessed modelling run identity."""

from typing import Any, cast

import click

from src.constants.command_line_interface import CONFIG_ARGS, CONFIG_KWARGS
from src.context import get_config_context
from src.portfolio import StepPortfolio
from src.portfolio.transformers.signals import select_cutoff
from src.utils.cli_helper import SpecialHelpOrder
from src.utils.modelling_runs import validate_run_id


@click.group(cls=SpecialHelpOrder)
def cli() -> None:
    """PORTFOLIO — long/short, S&P 500 and cash backtest."""


@cli.command(help="Simulate nine books and write <model run>/backtest_portfolio/report.html.", help_priority=1)
@click.option(*CONFIG_ARGS, **cast(dict[str, Any], CONFIG_KWARGS))
@click.option("--run-id", default=None, help="Exact assessed training; otherwise the latest successfully assessed run.")
def backtest(config_path: str, run_id: str | None) -> None:
    if run_id is not None:
        validate_run_id(run_id)
    config, context = get_config_context(config_path, use_cache=False, save=False)
    run, _ = select_cutoff(context, config, run_id)
    config.model.models_dir = str(run / "cutoff")
    context.paths = {**context.paths, "OUTPUT_DIR": run / "backtest_portfolio"}
    StepPortfolio(context=context, config=config).run()
