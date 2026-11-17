"""Known-truth resolved scopes for tests of public writer and builder interfaces."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

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
