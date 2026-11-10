"""The unsaved SEC13F panel, taken from the production `run()` at its writer boundary."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from src.data_aggregate.transformers import step_cube_sec_thirteen as step_module
from src.data_aggregate.transformers.step_cube_sec_thirteen import StepCubeSecThirteen
from src.data_aggregate.utils.common.incremental import PartWindow


def unsaved_panel(step: StepCubeSecThirteen, window: PartWindow | None = None) -> pd.DataFrame:
    """The panel `run()` hands its first save for `window` (default: the whole history) over the saved schema; nothing is written."""
    planned = PartWindow(None, None) if window is None else window
    with patch.object(step_module, "plan_window", return_value=planned), patch.object(step_module, "write_part", return_value=0) as write:
        step.run()
    return write.call_args.args[2]
