"""
cli.py  (src/portfolio/cli.py)
------------------------------
PORTFOLIO command-line interface — runs the unified backtest that blends the configured strategy
sleeves (the L/S equity sleeve reads back the trained model artifacts) into one portfolio. Invoked
by the Airflow `modelling` DAG after training, as:

    python -m src portfolio backtest [-c ./configs]

`StepPortfolio` reads `configs/portfolio.yml`, runs each sleeve OOS, risk-parity/ERC-blends them to
the global vol target, reports per-strategy vs global Sharpe, and saves the backtest pictures
(equity curve vs SP500, dynamic sleeve weights, per-sleeve analysis plots) + tables under
data/output/portfolio/.
"""
import click

from src.constants.command_line_interface import CONFIG_ARGS, CONFIG_KWARGS
from src.context import get_config_context
from src.utils.cli_helper import SpecialHelpOrder
from src.portfolio import StepPortfolio


@click.group(cls=SpecialHelpOrder)
def cli() -> None:
    """PORTFOLIO — unified sleeve-blend backtest + pictures."""


@cli.command(help="Run the portfolio backtest: blend the configured sleeves (portfolio.yml), "
                  "report per-strategy vs global Sharpe, and save the equity/weights/analysis "
                  "pictures + tables under data/output/portfolio/.",
             help_priority=1)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
def backtest(config_path: str) -> None:
    # save=True so the analysis pictures + tables are written to data/output/portfolio/
    config, context = get_config_context(config_path, use_cache=False, save=True)
    StepPortfolio(context=context, config=config).run()
