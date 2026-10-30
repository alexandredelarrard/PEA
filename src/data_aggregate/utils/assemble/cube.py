"""
Assemble the modelling cube: one row per (date, ticker) with betas, peers, targets, and
features. Targets are WIDE -- one column per (label, horizon), named by `target_column`
below -- so a (date, ticker) appears exactly once whatever the horizon grid is.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

CUBE_NON_FEATURE_COLS = frozenset({"date", "ticker", "beta_m", "beta_s", "gamma", "peers"})

# The ONE definition of the wide-target column contract, for writers and readers alike:
# `target_<label>_h<horizon>`. Never re-derive the format string at a call site -- a reader
# that spells it itself silently stops matching the moment a label name grows an underscore.
TARGET_COL_RE = re.compile(r"^target_(?P<label>[a-z_]+)_h(?P<horizon>\d+)$")


def is_meta_column(col: str) -> bool:
    """True for a key, a beta/peers column, or ANY target column -- i.e. everything a model
    must NOT be fed as a feature.

    Targets are matched by PATTERN, not by a listed name: `target_<label>_h<horizon>` is
    generated from `build_cube.targets`, so an enumerated set silently leaks a new horizon's
    or label's column into the feature list the day it is configured.
    """
    return col in CUBE_NON_FEATURE_COLS or TARGET_COL_RE.match(col) is not None


def target_column(label: str, horizon: int) -> str:
    return f"target_{label}_h{int(horizon)}"


def horizons_in(columns: Iterable[str], label: str) -> list[int]:
    """Sorted horizons this label has a column for. Schema-only: no data scan."""
    out = {int(m["horizon"]) for c in columns if (m := TARGET_COL_RE.match(c)) and m["label"] == label}
    return sorted(out)


def labels_to_wide(
    labels: dict,
    valid_keys: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """{horizon: {label: DataFrame(date x ticker)}} -> one row per (date, ticker), one column
    per (label, horizon).

    An IMMATURE label must not cost the row: a label at date d needs prices through
    d + horizon, so on the newest ~max_horizon trading days `target_*_h90` is still NaN while
    `target_*_h30` already has a value. Those dates are the ones `predict_latest` needs, so
    the row survives with the immature columns NaN.

    Each label is stacked on its own and the DataFrame constructor aligns them on the UNION of
    their keys, which makes the result independent of whether `stack` keeps NaN cells (pandas
    3 does; `dropna=True` did not, and passing `dropna` at all now raises): a (date, ticker)
    a label drops is one that label has nothing for, and a cell EVERY label drops is exactly
    the all-NaN row the explicit dropna below removes anyway.
    """
    cols: dict[str, pd.Series] = {}
    for horizon, per in labels.items():
        if not isinstance(per, dict):
            raise TypeError("labels must be {horizon: {label: DataFrame}} -- rebuild with build_targets_multi")
        for label, df in per.items():
            s = df.stack()
            s.index = s.index.set_names(["date", "ticker"])
            cols[target_column(label, horizon)] = s
    out = pd.DataFrame(cols)
    if valid_keys is None:
        out = out.dropna(axis=0, how="all")  # direct callers retain the old contract
    else:
        keys = valid_keys[["date", "ticker"]].copy()
        if keys.duplicated(["date", "ticker"]).any():
            raise ValueError("valid target keys must be unique by (date, ticker)")
        key_index = pd.MultiIndex.from_frame(keys, names=["date", "ticker"])
        out = out.reindex(key_index)  # keep all valid immature rows, no others
    return out.reset_index()
