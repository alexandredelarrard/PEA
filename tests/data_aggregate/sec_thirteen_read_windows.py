"""Declared reader scopes shared by the known-truth whole-book tests."""

from __future__ import annotations

import pandas as pd

from src.data_store.schema import Table, Tables, resolve
from src.utils.data_policy import input_read_window
from src.utils.window import Window
from tests.support.config import fixture_config


def filing_window(end: object) -> Window:
    """An explicit test prefix retaining the original end-of-source observation scope."""
    return Window(pd.Timestamp(fixture_config().data.tables.sec13f_hr.start_date), pd.Timestamp(end))


def read_windows(write: Window) -> dict[Table, Window | None]:
    """Compile the real dependency map for the test's actual output interval."""
    config = fixture_config()
    return {
        resolve(str(name)): input_read_window(config, Tables.cube_part_sec_thirteen, resolve(str(name)), write, write.end)
        for name in config.data.tables.cube_part_sec_thirteen.warmup
    }
