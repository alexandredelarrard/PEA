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

import hashlib
import json
import re
from pathlib import Path
from typing import Any, cast

import click

from src.constants.command_line_interface import CONFIG_ARGS, CONFIG_KWARGS
from src.context import get_config_context
from src.portfolio import StepPortfolio
from src.portfolio.step_strategy_moves import StepStrategyMoves
from src.utils.cli_helper import SpecialHelpOrder


@click.group(cls=SpecialHelpOrder)
def cli() -> None:
    """PORTFOLIO — unified sleeve-blend backtest + the daily trading ledger."""


@cli.command(
    help="Run the portfolio backtest: blend the configured sleeves (portfolio.yml), "
    "report per-strategy vs global Sharpe, and save the equity/weights/analysis "
    "pictures + tables under data/output/portfolio/.",
    help_priority=1,
)
@click.option(*CONFIG_ARGS, **cast(dict[str, Any], CONFIG_KWARGS))
@click.option("--run-id", required=True, help="Exact completed modelling assessment run; uses its cutoff models.")
def backtest(config_path: str, run_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", run_id) or re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", run_id.upper()):
        raise click.BadParameter("Use one safe dated run ID", param_hint="--run-id")
    config, context = get_config_context(config_path, use_cache=False, save=True)
    root = Path(str(config.model.assessment.output_dir))
    root = (root if root.is_absolute() else Path(context.paths["DATA_STORE"]) / root).resolve()
    run = (root / run_id).resolve()
    if run.parent != root:
        raise ValueError("Modelling run must stay inside its configured output folder")
    training = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    assessed = json.loads((run / "assessment/metadata.json").read_text(encoding="utf-8"))
    if (
        training.get("run_id") != run_id
        or assessed.get("run_id") != run_id
        or training.get("training_complete") is not True
        or assessed.get("complete") is not True
    ):
        raise ValueError("Portfolio requires this exact run's completed training and assessment")
    path = run / "cutoff/metadata.json"
    if training["cutoff_metadata"] != {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}:
        raise ValueError("Assessed cutoff metadata changed")
    cutoff = json.loads(path.read_text(encoding="utf-8"))
    if cutoff.get("run_id") != run_id or cutoff.get("full_history") is not False:
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
