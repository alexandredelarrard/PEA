"""
cli.py  (src/portfolio/cli.py)
------------------------------
PORTFOLIO command-line interface — two commands over the same sleeve blend:

    python -m src portfolio backtest --run-id <same-id> [-c ./configs]  # weekly `modelling` DAG
    python -m src portfolio strategy-moves [-c ./configs]              # daily `strat_prediction` DAG

`backtest` (StepPortfolio) reads `configs/portfolio.yml`, runs each sleeve OOS, risk-parity/ERC-
blends them to the global vol target, reports per-strategy vs global Sharpe, and saves the backtest
pictures (equity curve vs SP500, dynamic sleeve weights, per-sleeve analysis plots) + tables under
<configured modelling assessment root>/<run_id>/portfolio/
(default: data/output/modelling_backtest/<run_id>/portfolio/).

`strategy-moves` (StepStrategyMoves) runs the SAME blend but reports it as a tradeable LEDGER: each
sleeve's book re-sized to its dynamic ERC allocation x leverage, turned into per-day share-accurate
moves, FIFO-matched into round trips, and upserted to the `strategy` table with each position's
entry price, exit price and realized P&L.
"""

import json
from typing import Any, cast

import click

from src.constants.command_line_interface import CONFIG_ARGS, CONFIG_KWARGS
from src.context import get_config_context
from src.portfolio import StepPortfolio
from src.portfolio.step_strategy_moves import StepStrategyMoves
from src.utils.cli_helper import SpecialHelpOrder
from src.utils.modelling_runs import select_run


@click.group(cls=SpecialHelpOrder)
def cli() -> None:
    """PORTFOLIO — unified sleeve-blend backtest + the daily trading ledger."""


@cli.command(
    help="Run the portfolio backtest: blend the configured sleeves (portfolio.yml), "
    "report per-strategy vs global Sharpe, and save the equity/weights/analysis "
    "pictures + tables under <configured modelling assessment root>/<run_id>/portfolio/ "
    "(default: data/output/modelling_backtest/<run_id>/portfolio/).",
    help_priority=1,
)
@click.option(*CONFIG_ARGS, **cast(dict[str, Any], CONFIG_KWARGS))
@click.option("--run-id", default=None, help="Exact assessed training; defaults to the latest successfully assessed run.")
def backtest(config_path: str, run_id: str | None) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=True)
    run, training = select_run(context, config, run_id, assessed=True)
    path = run / "cutoff/metadata.json"
    cutoff = json.loads(path.read_text(encoding="utf-8"))
    if cutoff.get("run_id") != training["run_id"] or cutoff.get("full_history") is not False:
        raise ValueError("Portfolio backtest requires the named cutoff models")
    config.model.models_dir = str(run / "cutoff")
    context.paths = {**context.paths, "OUTPUT_DIR": run / "portfolio"}
    StepPortfolio(context=context, config=config).run()


@cli.command(
    "strategy-moves",
    help="Daily TRADING LEDGER -> the `strategy` table: every (day, sleeve, ticker) move "
    "the portfolio would place, sized by the sleeve's dynamic ERC allocation x "
    "leverage, FIFO-matched into round trips so each position carries its entry "
    "price, exit price and realized P&L. Upserts, so past BUY rows gain their "
    "price_sold/pnl on the day they close.",
    help_priority=2,
)
@click.option(*CONFIG_ARGS, **cast(dict[str, Any], CONFIG_KWARGS))
def strategy_moves(config_path: str) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=True)
    StepStrategyMoves(context=context, config=config).run()
