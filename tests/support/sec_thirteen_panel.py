"""The unsaved SEC13F panel, taken from the production `run()` at its writer boundary."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from src.data_aggregate.transformers import step_cube_sec_thirteen as step_module
from src.data_aggregate.transformers.step_cube_sec_thirteen import StepCubeSecThirteen
from src.utils.window import WindowPlan
from tests.support.windows import run_cube


def unsaved_panel(step: StepCubeSecThirteen, window: WindowPlan | None = None) -> pd.DataFrame:
    """The panel `run()` hands its first save for `window` (default: the whole history) over the saved schema; nothing is written."""
    with patch.object(step_module, "write_part", return_value=0) as write:
        if window is None:
            run_cube(step, full=True)
        else:
            with patch.object(step_module, "plan_window", return_value=window):
                step.run()
    return write.call_args.args[2]
