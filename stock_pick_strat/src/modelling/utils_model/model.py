"""
model.py
--------
Learn to predict the cross-sectional rank label from price-based features,
using LightGBM's learning-to-rank objective with each DAY as a query group.

Includes:
  * make_panel        : merge features + label into one modeling table
  * purged_wf_splits  : purged + embargoed walk-forward CV
  * train_ranker      : fit LightGBM ranker grouped by date
  * predict           : score new dates
  * daily_ic          : information coefficient (daily Spearman)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from scipy.stats import spearmanr

EARLY_STOPPING_ROUNDS = 40

# --------------------------------------------------------------------------- #
# 1. Assemble the modeling panel                                              #
# --------------------------------------------------------------------------- #
def make_panel(feature_panel: pd.DataFrame, label_df: pd.DataFrame,
               label_name: str = "y") -> pd.DataFrame:
    lab = label_df.stack()
    lab.index.set_names(["date", "ticker"], inplace=True)
    lab = lab.rename(label_name).reset_index()

    panel = feature_panel.merge(lab, on=["date", "ticker"], how="inner")
    feature_cols = [c for c in feature_panel.columns if c not in ("date", "ticker")]
    panel = panel.dropna(subset=feature_cols + [label_name])
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    return panel


def feature_columns(panel: pd.DataFrame, label_name: str = "y") -> list:
    return [c for c in panel.columns if c not in ("date", "ticker", label_name)]


# --------------------------------------------------------------------------- #
# 2. Purged + embargoed walk-forward CV                                       #
# --------------------------------------------------------------------------- #
def purged_wf_splits(dates: pd.Series, n_splits: int = 5, embargo: int = 20):
    unique_days = np.sort(pd.unique(dates))
    n = len(unique_days)
    fold = n // (n_splits + 1)
    if fold <= embargo:
        raise ValueError("Not enough dates for the requested n_splits/embargo.")

    for k in range(1, n_splits + 1):
        train_end = fold * k
        test_start = train_end + embargo
        test_end = min(test_start + fold, n)
        if test_start >= n:
            break
        train_days = unique_days[:train_end]
        test_days = unique_days[test_start:test_end]
        yield train_days, test_days


# --------------------------------------------------------------------------- #
# 3. Train / predict with LightGBM learning-to-rank                           #
# --------------------------------------------------------------------------- #
def _group_sizes(panel: pd.DataFrame) -> list[int]:
    return panel.groupby("date", sort=False).size().tolist()


def _graded_labels(panel: pd.DataFrame, label_name: str) -> np.ndarray:
    # LightGBM lambdarank expects integer relevance in [0, n_levels); 31 levels -> 0..30.
    return np.clip((panel[label_name].to_numpy() * 30).round().astype(int), 0, 30)


def _build_datasets(
    panel: pd.DataFrame,
    feats: list,
    label_name: str,
) -> tuple:

    x = panel[feats].to_numpy(dtype="float32")
    y = _graded_labels(panel, label_name)
    groups = _group_sizes(panel)
    return lgb.Dataset(x, label=y, group=groups, feature_name=feats)


def train_ranker(
    panel: pd.DataFrame,
    feats: list,
    label_name: str = "y",
    params: dict | None = None,
    num_boost_round: int = 400,
    valid_panel: pd.DataFrame | None = None,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
):
    """Fit a LightGBM lambdarank model. Labels are bucketed into graded relevance
    levels. When valid_panel is provided, early stopping is applied."""
    default = dict(
        objective="lambdarank",
        metric="ndcg",
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        lambda_l1=0.0,
        lambda_l2=5.0,
        verbosity=-1,
    )
    if params:
        default.update(params)

    train_set = _build_datasets(panel, feats, label_name)
    valid_sets = []
    callbacks = []

    if valid_panel is not None and not valid_panel.empty:
        valid_set = _build_datasets(valid_panel, feats, label_name)
        valid_sets = [valid_set]
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))

    booster = lgb.train(
        default,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets or None,
        callbacks=callbacks or None,
    )
    booster.feature_names = feats
    return booster


def predict(booster, panel: pd.DataFrame, feats: list) -> pd.Series:
    """Return a prediction Series indexed like `panel` (higher = long side)."""
    preds = booster.predict(panel[feats].to_numpy(dtype="float32"))
    return pd.Series(preds, index=panel.index, name="score")


def feature_importance(booster, feats: list) -> dict[str, float]:
    """LightGBM gain importance keyed by feature name."""
    gains = booster.feature_importance(importance_type="gain")
    return dict(zip(feats, gains))


def temporal_valid_split(
    panel: pd.DataFrame,
    train_frac: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last portion of dates for validation (early stopping)."""
    dates = np.sort(panel["date"].unique())
    cut = max(1, int(len(dates) * train_frac))
    if cut >= len(dates):
        cut = len(dates) - 1
    train_dates = dates[:cut]
    valid_dates = dates[cut:]
    train = panel[panel["date"].isin(train_dates)]
    valid = panel[panel["date"].isin(valid_dates)]
    return train, valid


# --------------------------------------------------------------------------- #
# 4. Evaluation: Information Coefficient                                       #
# --------------------------------------------------------------------------- #
def daily_ic(panel: pd.DataFrame, preds: pd.Series, label_name: str = "y") -> dict:
    df = panel[["date", label_name]].copy()
    df["pred"] = preds.to_numpy()
    ics = []
    for _, g in df.groupby("date", sort=True):
        if g["pred"].nunique() > 2 and g[label_name].nunique() > 2:
            ic, _ = spearmanr(g["pred"], g[label_name])
            if np.isfinite(ic):
                ics.append(ic)
    ics = np.array(ics)
    return {
        "mean_ic": float(ics.mean()) if len(ics) else np.nan,
        "ic_std": float(ics.std()) if len(ics) else np.nan,
        "ic_ir": float(ics.mean() / ics.std() * np.sqrt(252)) if ics.std() > 0 else np.nan,
        "n_days": int(len(ics)),
    }


def cross_validate(
    panel: pd.DataFrame,
    feats: list,
    label_name: str = "y",
    n_splits: int = 5,
    embargo: int = 20,
    **train_kw,
):
    results = []
    for train_days, test_days in purged_wf_splits(panel["date"], n_splits, embargo):
        train = panel[panel["date"].isin(train_days)]
        test = panel[panel["date"].isin(test_days)]
        if train.empty or test.empty:
            continue
        sub_train, sub_valid = temporal_valid_split(train)
        booster = train_ranker(
            sub_train, feats, label_name, valid_panel=sub_valid, **train_kw,
        )
        preds = predict(booster, test, feats)
        results.append(daily_ic(test, preds, label_name))
    return results
