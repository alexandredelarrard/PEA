"""
cli.py  (src/portfolio/cli.py)
------------------------------
PORTFOLIO command-line interface — two commands over the same sleeve blend:

    python -m src portfolio backtest        [-c ./configs]   # weekly `modelling` DAG
    python -m src portfolio strategy-moves  [-c ./configs]   # daily `strat_prediction` DAG

`backtest` (StepPortfolio) reads `configs/portfolio.yml`, runs each sleeve OOS, risk-parity/ERC-
blends them to the global vol target, reports per-strategy vs global Sharpe, and saves the backtest
pictures (equity curve vs SP500, dynamic sleeve weights, per-sleeve analysis plots) + tables under
data/output/portfolio/.

`strategy-moves` (StepStrategyMoves) runs the SAME blend but reports it as a tradeable LEDGER: each
sleeve's book re-sized to its dynamic ERC allocation x leverage, turned into per-day share-accurate
moves, FIFO-matched into round trips, and upserted to the `strategy` table with each position's
entry price, exit price and realized P&L.
"""
import click

from src.constants.command_line_interface import CONFIG_ARGS, CONFIG_KWARGS
from src.context import get_config_context
from src.utils.cli_helper import SpecialHelpOrder
from src.portfolio import StepPortfolio
from src.portfolio.step_strategy_moves import StepStrategyMoves


@click.group(cls=SpecialHelpOrder)
def cli() -> None:
    """PORTFOLIO — unified sleeve-blend backtest + the daily trading ledger."""


@cli.command(help="Run the portfolio backtest: blend the configured sleeves (portfolio.yml), "
                  "report per-strategy vs global Sharpe, and save the equity/weights/analysis "
                  "pictures + tables under data/output/portfolio/.",
             help_priority=1)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
def backtest(config_path: str) -> None:
    # save=True so the analysis pictures + tables are written to data/output/portfolio/
    config, context = get_config_context(config_path, use_cache=False, save=True)
    StepPortfolio(context=context, config=config).run()


@cli.command("strategy-moves",
             help="Daily TRADING LEDGER -> the `strategy` table: every (day, sleeve, ticker) move "
                  "the portfolio would place, sized by the sleeve's dynamic ERC allocation x "
                  "leverage, FIFO-matched into round trips so each position carries its entry "
                  "price, exit price and realized P&L. Upserts, so past BUY rows gain their "
                  "price_sold/pnl on the day they close.",
             help_priority=2)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
def strategy_moves(config_path: str) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=True)
    StepStrategyMoves(context=context, config=config).run()
