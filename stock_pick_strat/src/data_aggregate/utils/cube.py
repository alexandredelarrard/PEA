"""
Assemble the modelling cube: one row per (date, ticker, target_horizon) with
betas, peers, targets, and features.
"""
from __future__ import annotations

import json

import pandas as pd

CUBE_META_COLS = frozenset({
    "date", "ticker", "target_horizon", "target",
    "beta_m", "beta_s", "gamma", "peers",
})


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


def _labels_to_long(labels: dict[int, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for horizon, label_df in labels.items():
        lab = label_df.stack()
        lab.index.set_names(["date", "ticker"], inplace=True)
        lab = lab.rename("target").reset_index()
        lab["target_horizon"] = horizon
        frames.append(lab)
    return pd.concat(frames, ignore_index=True)


def build_cube_dataframe(
    feature_panel: pd.DataFrame,
    labels: dict[int, pd.DataFrame],
    betas: dict[str, pd.DataFrame],
    peers: dict[str, dict],
) -> pd.DataFrame:
    """
    Merge features, rolling betas, static peer baskets, and multi-horizon
    targets into a single long dataframe keyed by (date, ticker, target_horizon).
    """
    betas_long = _betas_to_long(betas)
    targets_long = _labels_to_long(labels)

    base = feature_panel.merge(betas_long, on=["date", "ticker"], how="left")
    base["peers"] = base["ticker"].map(
        lambda t: json.dumps(peers.get(t, {}), ensure_ascii=False),
    )

    cube = targets_long.merge(base, on=["date", "ticker"], how="inner")
    cube["date"] = pd.to_datetime(cube["date"]).dt.normalize()
    cube = cube.sort_values(["date", "ticker", "target_horizon"]).reset_index(drop=True)
    return cube


def panel_from_cube(
    cube: pd.DataFrame,
    horizon: int,
    label_name: str = "y",
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Extract a modelling panel for one horizon from the saved cube.

    When `feature_cols` is given, only those columns are kept as model inputs.
    Rows are dropped ONLY when the label is missing -- NOT when individual
    features are NaN. LightGBM handles missing values natively; requiring all
    selected features to be present (dropna on the full feature list) collapses
    a ~500-name universe to ~88 names because fundamentals have uneven coverage.
    """
    panel = cube[cube["target_horizon"] == horizon].copy()
    panel = panel.rename(columns={"target": label_name})
    if feature_cols is None:
        feature_cols = [c for c in panel.columns if c not in CUBE_META_COLS]
    else:
        feature_cols = [c for c in feature_cols if c in panel.columns]
    keep = ["date", "ticker", label_name] + feature_cols
    panel = panel[[c for c in keep if c in panel.columns]]
    panel = panel.dropna(subset=[label_name])
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def feature_columns_from_cube(panel: pd.DataFrame, label_name: str = "y") -> list[str]:
    exclude = CUBE_META_COLS | {label_name}
    return [c for c in panel.columns if c not in exclude]
