"""Narrow pandas' scalar-column selection for the type checker."""

from collections.abc import Hashable
from typing import cast

import pandas as pd


def frame_column(frame: pd.DataFrame, name: Hashable) -> pd.Series:
    """Return the Series selected by a scalar column label."""
    return cast(pd.Series, frame[name])
