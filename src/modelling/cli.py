"""Train dated original folds/cutoff models, assess their saved evidence, then refit production."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import click

from src import context as context_module
from src.constants.command_line_interface import CONFIG_ARGS, CONFIG_KWARGS
from src.context import get_config_context
from src.modelling.steps.step_assess_long_short import StepAssessLongShort
from src.modelling.steps.step_long_short import StepLongShort
from src.modelling.utils.artifacts import validate_run_id
from src.utils.cli_helper import SpecialHelpOrder


def _set_data_root() -> None:
    """A linked checkout uses the original repository's data unless explicitly overridden."""
    worktree = Path(context_module.__file__).resolve().parents[1]
    git_file = worktree / ".git"
    if git_file.is_file() and "ROOT_PATH" not in os.environ:
        raw = git_file.read_text(encoding="utf-8").removeprefix("gitdir:").strip()
        os.environ["ROOT_PATH"] = str((worktree / raw).resolve().parents[2])


def _run_identity(ctx: click.Context, param: click.Parameter, value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return validate_run_id(value)
    except ValueError as exc:
        raise click.BadParameter(str(exc), ctx=ctx, param=param) from exc


@click.group(cls=SpecialHelpOrder)
def cli() -> None:
    """MODELLING — train, assess saved models, refit production and predict."""


@cli.command(help="Save original purged folds and fresh cutoff models in a dated run; uses configured pools/parameters.", help_priority=1)
@click.option(*CONFIG_ARGS, **cast(dict[str, Any], CONFIG_KWARGS))
@click.option("--train-start", default=None, help="Override train.start_date (YYYY-MM-DD).")
@click.option("--train-end", default=None, help="Override train.end_date (YYYY-MM-DD).")
@click.option("--run-id", default=None, callback=_run_identity, help="Common dated run ID; generated when absent.")
@click.option("--resume", is_flag=True, help="Reuse only verified matching original folds and cutoff models.")
def train(config_path: str, train_start: str | None, train_end: str | None, run_id: str | None, resume: bool) -> None:
    _set_data_root()
    config, context = get_config_context(config_path, use_cache=False, save=True)
    config.model.assessment.run_id = run_id or config.model.assessment.get("run_id")
    config.model.assessment.resume = resume
    if train_start:
        config.train.start_date = train_start
    if train_end:
        config.train.end_date = train_end
    context.log.info("Training window: %s -> %s", config.train.start_date, config.train.end_date)
    StepLongShort(context=context, config=config).run_train()


@cli.command(
    help="Assess all configured horizons using only the named run's saved original-fold/cutoff models; never fits or searches.", help_priority=2
)
@click.option(*CONFIG_ARGS, **cast(dict[str, Any], CONFIG_KWARGS))
@click.option("--run-id", default=None, callback=_run_identity, help="Exact completed training run, or configured model.assessment.run_id.")
@click.option("--resume", is_flag=True, help="Reuse completed analysis only after verifying models, prepared inputs and diagnostic identities.")
@click.option("--report-only", is_flag=True, help="Rerender completed saved analysis without database/model loading or fitting.")
def assess(config_path: str, run_id: str | None, resume: bool, report_only: bool) -> None:
    _set_data_root()
    config, context = get_config_context(config_path, use_cache=False, save=False)
    config.model.assessment.run_id = run_id or config.model.assessment.get("run_id")
    config.model.assessment.resume = resume
    config.model.assessment.report_only = report_only
    result = StepAssessLongShort(context=context, config=config).run()
    context.log.info("Saved-model assessment complete: %s", result)


@cli.command(
    help="Refit all latest mature labels using the named cutoff run's saved fold rounds; publish a complete production generation.", help_priority=3
)
@click.option(*CONFIG_ARGS, **cast(dict[str, Any], CONFIG_KWARGS))
@click.option("--run-id", default=None, callback=_run_identity, help="Completed cutoff run whose fold-selected rounds must be reused.")
@click.option("--resume", is_flag=True, help="Verify and reuse a completed publication without refitting.")
def full_train(config_path: str, run_id: str | None, resume: bool) -> None:
    _set_data_root()
    config, context = get_config_context(config_path, use_cache=False, save=True)
    config.model.assessment.run_id = run_id or config.model.assessment.get("run_id")
    config.model.assessment.resume = resume
    StepLongShort(context=context, config=config).run_train(full_history=True)


@cli.command(help="Predict latest feature-only cube dates using the complete production generation; no training.", help_priority=4)
@click.option(*CONFIG_ARGS, **cast(dict[str, Any], CONFIG_KWARGS))
@click.option("--n-dates", default=1, show_default=True, type=int, help="How many of the most recent cube dates to predict.")
def predict(config_path: str, n_dates: int) -> None:
    _set_data_root()
    config, context = get_config_context(config_path, use_cache=False, save=True)
    StepLongShort(context=context, config=config).run_predict(n_dates=n_dates)
