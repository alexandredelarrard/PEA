"""Known-truth resolved scopes for tests of public writer and builder interfaces."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import pandas as pd

from src.utils.window import Window, WindowPlan, WindowRun


def window_plan(
    requested: Window,
    *,
    scope: Sequence[str],
    read_start: pd.Timestamp | None = None,
    write_start: pd.Timestamp | None = None,
    frontier: pd.Timestamp | None = None,
    mode: Literal["incremental", "initial", "full", "recovery"] = "incremental",
) -> WindowPlan:
    """Supply hand-chosen bounds without reproducing the production overlap resolver."""
    return WindowPlan(
        requested,
        Window(requested.start if read_start is None else read_start, requested.end),
        Window(requested.start if write_start is None else write_start, requested.end),
        WindowRun(mode, "known-truth test scope", requested.end),
        frontier,
        (("ticker", tuple(scope)),),
    )


def run_cube(step: Any, *, full: bool = False, tickers: tuple[str, ...] = ()) -> dict:
    """Drive the production root-run contract on an isolated fixture context."""
    context = step._context
    original = context.run
    context.run = WindowRun("full" if full else "incremental", "known-truth public cube run", original.instant)
    try:
        return step.run(tickers=tickers) if tickers else step.run()
    finally:
        context.run = original


def run_method(step: Any, method: Callable[[], Any], *, full: bool = False) -> Any:
    """Drive one public sub-run of a step that has several (`run_profile`, `run_clustering`) under the root-run contract."""
    context = step._context
    original = context.run
    context.run = WindowRun("full" if full else "incremental", "known-truth public sub-run", original.instant)
    try:
        return method()
    finally:
        context.run = original


def plan_cube(step: Any, calendar: pd.DatetimeIndex, *, full: bool) -> Any:
    """Plan text cache work using the same fixture root-run contract."""
    context = step._context
    original = context.run
    context.run = WindowRun("full" if full else "incremental", "known-truth public cube plan", original.instant)
    try:
        return step.plan_window(calendar)
    finally:
        context.run = original
