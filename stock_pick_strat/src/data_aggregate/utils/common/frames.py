"""
frames.py  (src/data_aggregate/utils/common/frames.py)
-----------------------------------------------------
Shape-level arithmetic: the safe division and inf-sanitizing every panel builder needs.

Four near-identical helpers had grown up in four modules -- `panel._ratio` (frames),
`sector_features._safe_div` (Series), `dividend_features._clean_ratio` (frames, no
masking) and `features._safe` (inf-strip only) -- plus ~40 inline
`.replace([np.inf, -np.inf], np.nan)` calls. They are collected here, but deliberately
NOT collapsed into one function:

  * `ratio` works on FRAMES and intersects their columns; `safe_div` works on SERIES and
    tolerates a `None` denominator (a `capital.*` helper with none of its inputs present).
    A shape-polymorphic merge would have to drop either the column intersection or the
    None-tolerance.
  * `safe_div` does NOT sanitize inf, and must not start doing so: `sector_features` calls
    it ~200 times and relies on the current behaviour.

`safe_div`'s third parameter therefore keeps its exact name (`den_positive`) and stays
positional, because `sector_features` passes it positionally throughout.
"""
from __future__ import annotations

from typing import TypeVar

import numpy as np
import pandas as pd

Shaped = TypeVar("Shaped", pd.Series, pd.DataFrame)


def sanitize(x: Shaped) -> Shaped:
    """+/-inf -> NaN. Was `features._safe`, and the tail of ~40 inline `.replace` calls."""
    return x.replace([np.inf, -np.inf], np.nan)


def ratio(num: pd.DataFrame, den: pd.DataFrame, positive_den: bool = False) -> pd.DataFrame:
    """Column-aligned num/den on the common tickers, inf -> NaN. When
    `positive_den` the denominator is masked to strictly-positive values.

    Returns an EMPTY frame when either side is empty or the column intersection is --
    callers test `.empty` to decide whether the feature exists at all."""
    if num.empty or den.empty:
        return pd.DataFrame()
    cols = num.columns.intersection(den.columns)
    if len(cols) == 0:
        return pd.DataFrame()
    d = den[cols]
    d = d.where(d > 0) if positive_den else d.where(d != 0)
    out = num[cols] / d
    return out.replace([np.inf, -np.inf], np.nan)


def safe_div(num: pd.Series, den: pd.Series | None, den_positive: bool = False) -> pd.Series:
    """Elementwise num/den, NaN where den is 0/NaN (or <=0 if den_positive). A `None`
    denominator (a `capital.*` helper with none of its inputs present) yields all-NaN
    rather than raising.

    Note it does NOT strip inf -- see the module docstring."""
    if den is None:
        return pd.Series(np.nan, index=num.index)
    den = den.where(den > 0) if den_positive else den.replace(0, np.nan)
    return num / den
