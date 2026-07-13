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

import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from scipy.stats import spearmanr

EARLY_STOPPING_ROUNDS = 40
CALENDAR_DAYS_PER_YEAR = 365.25
DEFAULT_SEED = 42  # any fixed value works; overridden by the pipeline's global seed


def time_decay_weights(
    dates: pd.Series,
    half_life_years: float,
    reference: pd.Timestamp | None = None,
) -> np.ndarray:
    """Exponential sample weights by age: w = 0.5 ** (age / half_life).

    ``reference`` defaults to the latest date in ``dates`` (weight = 1.0 there).
    Age is in calendar days; a 2-year half-life gives weight 0.5 for rows exactly
    two years before ``reference``.
    """
    if half_life_years <= 0:
        raise ValueError("half_life_years must be positive")
    dt = pd.to_datetime(dates).dt.normalize()
    ref = pd.Timestamp(reference).normalize() if reference is not None else dt.max()
    half_life_days = half_life_years * CALENDAR_DAYS_PER_YEAR
    age = (ref - dt).dt.days.to_numpy(dtype=np.float64)
    age = np.clip(age, 0.0, None)
    return np.power(0.5, age / half_life_days).astype(np.float32)

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
    params: dict,
    panel: pd.DataFrame,
    feats: list,
    label_name: str,
    weights: np.ndarray | None = None,
) -> lgb.Dataset:
    x = panel[feats].to_numpy(dtype="float32")
    kw: dict = {"feature_name": feats}
    if weights is not None:
        kw["weight"] = weights
    if params["objective"] == "lambdarank":
        y = _graded_labels(panel, label_name)
        groups = _group_sizes(panel)
        return lgb.Dataset(x, label=y, group=groups, **kw)
    y = panel[label_name].to_numpy(dtype="float32")
    return lgb.Dataset(x, label=y, **kw)


def train_ranker(
    panel: pd.DataFrame,
    feats: list,
    label_name: str = "y",
    params: dict | None = None,
    num_boost_round: int = 400,
    valid_panel: pd.DataFrame | None = None,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    half_life_years: float | None = None,
):
    """Fit a LightGBM model. When ``half_life_years`` is set, training rows are
    weighted with exponential time decay (most recent = 1.0); validation is
    unweighted so early stopping reflects recent out-of-sample fit."""
    default = dict(
        objective="regression", #"lambdarank",
        metric="rmse", #"ndcg",
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        lambda_l1=0.0,
        lambda_l2=5.0,
        verbosity=-1,
        n_jobs=-2,
        # --- reproducibility: identical results on every rerun ---
        # multithreaded LightGBM sums gradients in a nondeterministic order;
        # deterministic + force_row_wise make it bit-for-bit reproducible even
        # across thread counts, and `seed` fixes every internal RNG (bagging /
        # feature_fraction / data sampling). Without these a rerun drifts and
        # early stopping flips between num_boost_round and ~1 round.
        seed=DEFAULT_SEED,
        deterministic=True,
        force_row_wise=True,
    )
    if params:
        default.update(params)

    train_w = (time_decay_weights(panel["date"], half_life_years)
               if half_life_years is not None else None)
    train_set = _build_datasets(default, panel, feats, label_name, train_w)
    valid_sets = []
    callbacks = []

    if valid_panel is not None and not valid_panel.empty:
        valid_set = _build_datasets(default, valid_panel, feats, label_name)
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
    train_frac: float = 0.9,
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
def daily_ic(panel: pd.DataFrame, preds: pd.Series, label_name: str = "y",
             horizon: int = 1, trading_days_per_year: int = 252) -> dict:
    """Daily cross-sectional IC (Spearman) and its annualized information ratio.

    The IC is measured EVERY trading day, but the label is an `horizon`-day
    forward return, so consecutive daily ICs overlap and are ~horizon-day
    autocorrelated. Annualizing the IR with sqrt(252) therefore assumes 252
    INDEPENDENT observations per year and overstates it by ~sqrt(horizon) (which
    is why the 60-day horizon showed the largest IR). We instead annualize by the
    number of independent horizon-length windows per year, 252/horizon, i.e.
        ic_ir = mean(IC)/std(IC) * sqrt(252 / horizon).
    With horizon=1 this reduces to the classic sqrt(252) daily IR.
    """
    df = panel[["date", label_name]].copy()
    df["pred"] = preds.to_numpy()
    ics = []
    for _, g in df.groupby("date", sort=True):
        if g["pred"].nunique() > 2 and g[label_name].nunique() > 2:
            ic, _ = spearmanr(g["pred"], g[label_name])
            if np.isfinite(ic):
                ics.append(ic)
    ics = np.array(ics)
    periods_per_year = trading_days_per_year / max(1, int(horizon))
    return {
        "mean_ic": float(ics.mean()) if len(ics) else np.nan,
        "ic_std": float(ics.std()) if len(ics) else np.nan,
        "ic_ir": (float(ics.mean() / ics.std() * np.sqrt(periods_per_year))
                  if ics.std() > 0 else np.nan),
        "n_days": int(len(ics)),
    }


def cross_validate(
    panel: pd.DataFrame,
    feats: list,
    label_name: str = "y",
    n_splits: int = 5,
    embargo: int = 20,
    horizon: int = 1,
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
        results.append(daily_ic(test, preds, label_name, horizon=horizon))
    return results


# --------------------------------------------------------------------------- #
# 5. Persist / reload trained rankers (one pickle per horizon)                #
# --------------------------------------------------------------------------- #
def model_pickle_path(models_dir: Path, horizon: int) -> Path:
    return Path(models_dir) / f"ranker_h{int(horizon)}.pkl"


def save_models(models_dir: Path, models: dict[int, lgb.Booster], meta: dict) -> None:
    """Pickle one self-contained file per horizon for ``pickle.load`` + predict."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    shared = {
        **meta,
        "horizons": sorted(int(h) for h in models),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    for h, booster in models.items():
        payload = {**shared, "horizon": int(h), "model": booster}
        with model_pickle_path(models_dir, h).open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: Path) -> dict:
    """Load a single horizon pickle (``model``, ``feature_cols``, ``horizon``, ...)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model pickle not found: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def load_models(models_dir: Path) -> tuple[dict[int, lgb.Booster], dict]:
    """Load every ``ranker_h*.pkl`` in ``models_dir``."""
    models_dir = Path(models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    paths = sorted(models_dir.glob("ranker_h*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No ranker_h*.pkl files in {models_dir}")

    models: dict[int, lgb.Booster] = {}
    meta: dict | None = None
    for path in paths:
        bundle = load_model(path)
        h = int(bundle["horizon"])
        models[h] = bundle["model"]
        if meta is None:
            meta = {k: v for k, v in bundle.items() if k != "model"}
    meta = meta or {}
    meta["horizons"] = sorted(models)
    return models, meta
