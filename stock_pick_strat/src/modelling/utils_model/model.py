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

from omegaconf import ListConfig, OmegaConf
from scipy.stats import spearmanr

EARLY_STOPPING_ROUNDS = 40
VALID_MONOTONE_DIRECTIONS = frozenset({-1, 0, 1})
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


def parse_monotone_feature_map(features_cfg: dict | ListConfig | None) -> dict[str, int]:
    """Parse ``inputs.monotonic.features`` from modellling.yml.

    Accepts either a mapping ``{feature: direction}`` or a list of single-key
    mappings ``[{feature: direction}, ...]``. Directions must be -1 (decreasing),
    0 (free), or +1 (increasing).
    """
    if features_cfg is None:
        return {}
    out: dict[str, int] = {}
    if OmegaConf.is_dict(features_cfg):
        for name, direction in features_cfg.items():
            d = int(direction)
            if d not in VALID_MONOTONE_DIRECTIONS:
                raise ValueError(f"Invalid monotone direction {d} for {name}")
            out[str(name)] = d
        return out

    for item in features_cfg:
        if not OmegaConf.is_dict(item):
            raise ValueError(
                "inputs.monotonic.features must be a mapping or a list of "
                "single-key mappings like `- f_sales_yield_xs: 1`"
            )
        if len(item) != 1:
            raise ValueError(
                "Each monotone list entry must contain exactly one feature "
                f"and direction, got {dict(item)}"
            )
        name, direction = next(iter(item.items()))
        d = int(direction)
        if d not in VALID_MONOTONE_DIRECTIONS:
            raise ValueError(f"Invalid monotone direction {d} for {name}")
        out[str(name)] = d
    return out


def build_monotone_constraints(
    feats: list[str],
    feature_map: dict[str, int],
) -> list[int]:
    """LightGBM ``monotone_constraints`` vector aligned to ``feats`` order."""
    return [int(feature_map.get(f, 0)) for f in feats]


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
    categorical_features: list[str] | None = None,
) -> lgb.Dataset:
    # With categoricals, pass a DataFrame (int category codes kept as int, so
    # LightGBM makes native categorical splits); else the fast all-float numpy path.
    if categorical_features:
        cat = set(categorical_features)
        x = panel[feats].copy()
        num = [f for f in feats if f not in cat]
        if num:
            x[num] = x[num].astype("float32")
        for c in categorical_features:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(-1).astype("int32")
        kw: dict = {"feature_name": feats, "categorical_feature": list(categorical_features)}
    else:
        x = panel[feats].to_numpy(dtype="float32")
        kw = {"feature_name": feats}
    if weights is not None:
        kw["weight"] = weights
    if params["objective"] == "lambdarank":
        y = _graded_labels(panel, label_name)
        groups = _group_sizes(panel)
        return lgb.Dataset(x, label=y, group=groups, **kw)
    y = panel[label_name].to_numpy(dtype="float32")
    return lgb.Dataset(x, label=y, **kw)


def _ic_eval_factory(val_dates: np.ndarray, val_label: np.ndarray, min_names: int = 5):
    """Build a LightGBM custom eval that returns the mean DAILY cross-sectional
    IC (Spearman) on the validation set. Early-stopping on this — instead of
    RMSE — aligns the stopping rule with what the strategy actually cares about
    (cross-sectional ranking), so a near-noise target does not stop after a
    couple of RMSE-flat rounds. Label ranks are precomputed once; each round only
    ranks the predictions per day."""
    dates = np.asarray(val_dates)
    y = np.asarray(val_label, dtype=float)
    if len(dates) == 0:
        groups, y_ranks = [], []
    else:
        _, inv = np.unique(dates, return_inverse=True)
        groups = [np.where(inv == g)[0] for g in range(int(inv.max()) + 1)]
        y_ranks = [pd.Series(y[idx]).rank().to_numpy() for idx in groups]

    def _feval(preds, _data):
        preds = np.asarray(preds, dtype=float)
        ics = []
        for idx, yr in zip(groups, y_ranks):
            if len(idx) > min_names:
                pr = pd.Series(preds[idx]).rank().to_numpy()
                if pr.std() > 0 and yr.std() > 0:
                    ic = float(np.corrcoef(pr, yr)[0, 1])
                    if np.isfinite(ic):
                        ics.append(ic)
        return "daily_ic", (float(np.mean(ics)) if ics else 0.0), True  # higher = better

    return _feval


def train_ranker(
    panel: pd.DataFrame,
    feats: list,
    label_name: str = "y",
    params: dict | None = None,
    num_boost_round: int = 400,
    valid_panel: pd.DataFrame | None = None,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    half_life_years: float | None = None,
    eval_metric: str = "rmse",
    categorical_features: list[str] | None = None,
):
    """Fit a LightGBM model. When ``half_life_years`` is set, training rows are
    weighted with exponential time decay (most recent = 1.0); validation is
    unweighted so early stopping reflects recent out-of-sample fit.

    ``eval_metric``: "rmse" (built-in) or "ic" -> early-stop on the mean daily
    cross-sectional IC of the validation fold (the ranking metric we optimize).
    """
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
    train_set = _build_datasets(default, panel, feats, label_name, train_w,
                                categorical_features=categorical_features)
    valid_sets = []
    callbacks = []
    feval = None

    if valid_panel is not None and not valid_panel.empty:
        valid_set = _build_datasets(default, valid_panel, feats, label_name,
                                    categorical_features=categorical_features)
        valid_sets = [valid_set]
        if eval_metric == "ic":
            # disable the built-in metric so early stopping keys on the custom IC
            default["metric"] = "None"
            feval = _ic_eval_factory(valid_panel["date"].to_numpy(),
                                     valid_panel[label_name].to_numpy())
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))

    booster = lgb.train(
        default,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets or None,
        feval=feval,
        callbacks=callbacks or None,
    )
    booster.feature_names = feats
    return booster


def predict(booster, panel: pd.DataFrame, feats: list) -> pd.Series:
    """Return a prediction Series indexed like `panel` (higher = long side).

    Passes a DataFrame (not a float32 array) so integer category codes survive:
    LightGBM then applies the SAME categorical bins it learned at train time, and
    the linear member's own predict() casts its numeric features to float. Result
    is identical to the old numpy path for all-numeric models."""
    preds = booster.predict(panel[feats])
    return pd.Series(preds, index=panel.index, name="score")


def per_day_zscore(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Cross-sectionally z-score ``values`` within each day.

    Days with < 2 names, or zero / undefined dispersion (e.g. a member whose
    coefficients were all shrunk to zero under strong L1 -> a CONSTANT prediction),
    return NaN -- computed WITHOUT tripping numpy's 'Degrees of freedom <= 0' /
    'invalid value encountered in divide' RuntimeWarnings: the < 2 guard skips the
    ddof=1 std on a single element, and the finite/positive guard skips the divide.
    """
    df = pd.DataFrame({"date": np.asarray(dates), "v": np.asarray(values, dtype=float)})

    def _z(s: pd.Series) -> pd.Series:
        if len(s) < 2:
            return pd.Series(np.nan, index=s.index)
        sd = s.std()
        if not np.isfinite(sd) or sd <= 0.0:
            return pd.Series(np.nan, index=s.index)
        return (s - s.mean()) / sd

    return df.groupby("date")["v"].transform(_z).to_numpy()


def ensemble_predict(models: dict, panel: pd.DataFrame, feats: list):
    """Average the CROSS-SECTIONALLY-STANDARDIZED (per day) predictions of several
    models into one ensemble score per row.

    Each model's raw output is z-scored within each day BEFORE averaging, so a
    GBDT and a linear model (very different output scales) are put on a common
    ranking scale first -- that is what makes model-averaging help. With one model
    this reduces to its per-day z-score. `models` is {name: fitted_model}; each
    model just needs a `.predict(X)` (LightGBM booster or a LinearModel).
    """
    if not models:
        raise ValueError("ensemble_predict received no models")

    dates = panel["date"].to_numpy()
    members: dict[str, pd.Series] = {}
    zs = []
    for name, m in models.items():
        # each member scores on ITS OWN feature list (the LightGBM member also has
        # the categorical columns; the linear member is numeric-only) -> falls back
        # to the shared `feats` for models without a stored feature_names.
        mfeats = list(getattr(m, "feature_names", None) or feats)
        raw = predict(m, panel, mfeats).to_numpy()
        z = per_day_zscore(raw, dates)          # NaN on <2-name days / a constant member
        zs.append(z)
        members[str(name)] = pd.Series(z, index=panel.index, name=str(name))
    # nan-mean across members WITHOUT np.nanmean, so an all-NaN row (a day no member
    # could standardize -- e.g. a single-name day) yields NaN rather than a
    # "Mean of empty slice" RuntimeWarning.
    stack = np.column_stack(zs)
    cnt = np.isfinite(stack).sum(axis=1)
    avg = np.where(cnt > 0, np.nansum(stack, axis=1) / np.where(cnt > 0, cnt, 1), np.nan)
    blended = pd.Series(avg, index=panel.index, name="score")
    return blended, members


def _pairwise_corr(M: np.ndarray) -> np.ndarray:
    """Correlation matrix of the columns of M, computed pairwise-complete (ignores
    rows where either column is NaN) so horizons with different coverage still get
    a valid off-diagonal. Degenerate pairs (constant / too few obs) -> 0."""
    n = M.shape[1]
    C = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = M[:, i], M[:, j]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() > 2 and a[m].std() > 0 and b[m].std() > 0:
                c = float(np.corrcoef(a[m], b[m])[0, 1])
                C[i, j] = C[j, i] = c if np.isfinite(c) else 0.0
    return C


def optimal_forecast_weights(signals: dict[int, np.ndarray],
                             ir: dict[int, float],
                             shrink: float = 0.5) -> dict[int, float]:
    """Optimal combination of correlated per-horizon forecasts (Grinold-Kahn):

        w  ∝  Σ⁻¹ · IR

    IR = each horizon's information ratio (risk-adjusted skill); Σ = correlation of
    the horizon signals. Highly-correlated horizons (30/60/90 move together) then
    SHARE weight instead of triple-counting their common component, and a horizon
    with a weak *standalone* IR can still earn weight if it DIVERSIFIES the others.

    Robustness (fixes "a horizon drops out"):
      * Σ is shrunk toward the identity by `shrink` for a stable inverse.
      * an unestimable IR (NaN) is replaced by the MEAN of the finite IRs -- a
        neutral prior -- so a horizon whose CV IR could not be measured is never
        silently zeroed (only a genuinely non-positive IR loses weight).
      * negative optimal weights are floored at 0 and the result renormalized;
        all-invalid or singular cases fall back to equal weights.
    """
    hs = list(signals)
    n = len(hs)
    if n == 0:
        return {}
    if n == 1:
        return {hs[0]: 1.0}

    ir_vec = np.array([ir.get(h, np.nan) for h in hs], dtype=float)
    finite = np.isfinite(ir_vec)
    if not finite.any():
        return {h: 1.0 / n for h in hs}
    ir_vec[~finite] = ir_vec[finite].mean()          # neutral prior for NaN IR
    mu = np.clip(ir_vec, 0.0, None)
    if mu.sum() <= 0:
        return {h: 1.0 / n for h in hs}

    M = np.column_stack([np.asarray(signals[h], float) for h in hs])
    C = _pairwise_corr(M)
    C = (1.0 - shrink) * C + shrink * np.eye(n)
    try:
        w = np.linalg.solve(C, mu)
    except np.linalg.LinAlgError:
        w = mu.copy()
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        w = mu.copy()
    w = w / w.sum()
    return {h: float(w[i]) for i, h in enumerate(hs)}


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
    ics = np.asarray(ics, dtype=float)
    n = len(ics)
    periods_per_year = trading_days_per_year / max(1, int(horizon))
    # Guard EVERY mean/std behind n>0: np.std([]) emits "Mean of empty slice" +
    # "Degrees of freedom <= 0" + "invalid value encountered in divide". `ics` is
    # empty whenever a member's predictions are constant (no day has >2 unique
    # preds), e.g. an all-zero-coefficient elasticnet under strong L1.
    mean_ic = float(ics.mean()) if n else np.nan
    std_ic = float(ics.std()) if n else np.nan
    ic_ir = (mean_ic / std_ic * np.sqrt(periods_per_year)) if (n and std_ic > 0) else np.nan
    return {"mean_ic": mean_ic, "ic_std": std_ic, "ic_ir": ic_ir, "n_days": n}


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
