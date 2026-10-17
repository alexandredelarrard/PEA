"""
cli.py  (src/modelling/cli.py)
------------------------------
MODELLING command-line interface — trains the cross-sectional long/short ensemble on the cube for
the configured training window. Invoked by the Airflow `modelling` DAG as:

    python -m src modelling train [-c ./configs] [--train-start 2011-01-01] [--train-end 2022-01-01]

`StepLongShort.run_train` reads the training window from `modellling.yml` (`train.start_date` /
`train.end_date`); the optional `--train-start` / `--train-end` flags OVERRIDE those config dates
for this run (leaving them out uses the config as-is). It trains one per-horizon ensemble of the
`model.ensemble` families, saves the pickled members + metadata.json (the strategies read them
back), the `predictions` / `cube_signal` tables, and the per-run diagnostics + backtest report.
"""

import os
from pathlib import Path
from typing import Any, cast

import click

from src import context as context_module
from src.constants.command_line_interface import CONFIG_ARGS, CONFIG_KWARGS
from src.context import get_config_context
from src.modelling.steps.step_assess_long_short import StepAssessLongShort
from src.modelling.steps.step_long_short import StepLongShort
from src.utils.cli_helper import SpecialHelpOrder


@click.group(cls=SpecialHelpOrder)
def cli() -> None:
    """MODELLING — train the per-horizon long/short ensemble on the cube."""


@cli.command(
    help="Train the per-horizon ensemble on the cube for the configured training window "
    "(modellling.yml train.start_date/end_date; override with --train-start/--train-end). "
    "Saves model artifacts + metadata.json + predictions + diagnostics.",
    help_priority=1,
)
@click.option(*CONFIG_ARGS, **cast(dict[str, Any], CONFIG_KWARGS))
@click.option("--train-start", default=None, help="Override train.start_date (YYYY-MM-DD).")
@click.option("--train-end", default=None, help="Override train.end_date (YYYY-MM-DD).")
def train(config_path: str, train_start: str | None, train_end: str | None) -> None:
    # save=True so the model artifacts, metadata.json and diagnostics pictures are written to disk
    config, context = get_config_context(config_path, use_cache=False, save=True)
    if train_start:
        config.train.start_date = train_start
    if train_end:
        config.train.end_date = train_end
    context.log.info("Training window: %s -> %s", config.train.start_date, config.train.end_date)
    StepLongShort(context=context, config=config).run_train()


@cli.command(
    help="PRODUCTION train on ALL history up to the latest cube date (no train_end cutoff, "
    "no OOS holdout). Runs after the backtest; the fitted model feeds `predict`.",
    help_priority=2,
)
@click.option(*CONFIG_ARGS, **cast(dict[str, Any], CONFIG_KWARGS))
def full_train(config_path: str) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=True)
    StepLongShort(context=context, config=config).run_train(full_history=True)


@cli.command(
    help="Predict the latest cube date(s) per horizon (pred_h<h>) + the blended signal -> "
    "`predictions_latest` (consumed by the allocation DAG). Loads the full-trained "
    "artifacts from disk; no retraining.",
    help_priority=3,
)
@click.option(*CONFIG_ARGS, **cast(dict[str, Any], CONFIG_KWARGS))
@click.option("--n-dates", default=1, show_default=True, type=int, help="How many of the most recent cube dates to predict.")
def predict(config_path: str, n_dates: int) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=True)
    StepLongShort(context=context, config=config).run_predict(n_dates=n_dates)


@cli.command(help="Isolated h60 feature selection, sealed evaluation and offline assessment report.", help_priority=4)
@click.option(*CONFIG_ARGS, **cast(dict[str, Any], CONFIG_KWARGS))
@click.option("--output", "-o", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--stage", type=click.Choice(["development", "evaluation", "report", "all"]), default="development", show_default=True)
@click.option("--resume", is_flag=True, help="Reuse completed candidates only after checking unchanged input and experiment identities.")
def assess(config_path: str, output: Path, stage: str, resume: bool) -> None:
    worktree = Path(context_module.__file__).resolve().parents[1]
    git_file = worktree / ".git"
    if git_file.is_file() and "ROOT_PATH" not in os.environ:
        raw = git_file.read_text(encoding="utf-8").removeprefix("gitdir:").strip()
        git_dir = (worktree / raw).resolve()
        os.environ["ROOT_PATH"] = str(git_dir.parents[2])
    config, context = get_config_context(config_path, use_cache=False, save=False)
    config.model.assessment.output_dir = str(output.resolve())
    config.model.assessment.stage = stage
    config.model.assessment.resume = resume
    result = StepAssessLongShort(context=context, config=config).run()
    context.log.info("Assessment %s complete: %s", stage, result)
