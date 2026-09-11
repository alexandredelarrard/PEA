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
    out = {int(m["horizon"]) for c in columns
           if (m := TARGET_COL_RE.match(c)) and m["label"] == label}
    return sorted(out)


def _betas_to_long(betas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for ticker, bdf in betas.items():
        tmp = bdf.reset_index()
        if "index" in tmp.columns:
            tmp = tmp.rename(columns={"index": "date"})
        elif tmp.columns[0] != "date":
            tmp = tmp.rename(columns={tmp.columns[0]: "date"})
        tmp["ticker"] = ticker
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True)


def labels_to_wide(labels: dict) -> pd.DataFrame:
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
            raise TypeError("labels must be {horizon: {label: DataFrame}} -- rebuild with "
                            "build_targets_multi")
        for label, df in per.items():
            s = df.stack()
            s.index = s.index.set_names(["date", "ticker"])
            cols[target_column(label, horizon)] = s
    out = pd.DataFrame(cols)
    out = out.dropna(axis=0, how="all")          # nothing known yet -> store nothing
    return out.reset_index()


def panel_from_cube(
    cube: pd.DataFrame,
    horizon: int,
    label_name: str = "y",
    feature_cols: list[str] | None = None,
    target_type: str = "rank",
) -> pd.DataFrame:
    """Extract a modelling panel for one horizon from the saved cube.

    The cube is one row per (date, ticker) with the targets WIDE, so `horizon` picks a
    COLUMN (`target_<target_type>_h<horizon>`) rather than a row subset -- every row is a
    candidate for every horizon. `target_type` selects which label family that column comes
    from: "rank" -> `target_rank_h<h>`, "zscore" -> `target_zscore_h<h>`, etc.

    When `feature_cols` is given, only those columns are kept as model inputs.
    Rows are dropped ONLY when the label is missing -- NOT when individual
    features are NaN. LightGBM handles missing values natively; requiring all
    selected features to be present (dropna on the full feature list) collapses
    a ~500-name universe to ~88 names because fundamentals have uneven coverage.
    Dropping on the label is also what keeps the training rows identical to the
    long-cube era: an immature label is NaN here exactly where it produced no row there.
    """
    target_col = target_column(target_type, horizon)
    if target_col not in cube.columns:
        avail = sorted(c for c in cube.columns if c.startswith("target"))
        raise KeyError(
            f"Target column '{target_col}' not in cube; rebuild the cube with '{target_type}' "
            f"in build_cube.targets.labels and {horizon} in build_cube.targets.horizons "
            f"(available: {avail}).")
    panel = cube.rename(columns={target_col: label_name})

    if feature_cols is None:
        feature_cols = [c for c in panel.columns
                        if not is_meta_column(c) and c != label_name]
    else:
        feature_cols = [c for c in feature_cols if c in panel.columns]
    keep = ["date", "ticker", label_name] + feature_cols
    panel = panel[[c for c in keep if c in panel.columns]]
    panel = panel.dropna(subset=[label_name])
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def feature_columns_from_cube(panel: pd.DataFrame, label_name: str = "y") -> list[str]:
    return [c for c in panel.columns if not is_meta_column(c) and c != label_name]
