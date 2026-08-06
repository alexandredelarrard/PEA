import gc
import numpy as np
import json
import pickle
import pandas as pd
from datetime import datetime, timedelta
from omegaconf import DictConfig
from sqlalchemy import text
import lightgbm as lgb

from src.utils.step import Step
from src.context import Context
from src.constants.constants import (
    PREDICTION_MODEL_BLENDED,
    PREDICTION_MODEL_ENSEMBLE,
    PREDICTIONS_LATEST_TABLE,
)
from src.data_aggregate.utils.assemble.cube import panel_from_cube, CUBE_META_COLS
from src.modelling.long_short.utils import model as ml
from src.modelling.long_short.utils import baselines
from src.modelling.long_short.utils import diagnostics


class StepModelling(Step):
    """
    Stronger model than the single-horizon ranker:

      1. Trains ONE ranker PER HORIZON (5/10/20/60) on the full feature set,
         including the peer-relative fundamental features merged into the cube.
      2. Purged walk-forward CV per horizon -> per-horizon IC, and an IC-based
         weight for each horizon (horizons that predict better and more stably
         count more; cost-heavy short horizons naturally get down-weighted if
         their net IC is weak).
      3. Blends per-horizon predictions into ONE signal by IR-weighting, after
         standardizing each horizon's scores cross-sectionally per day so they
         are on a comparable scale before averaging.
      4. Logs feature importance so you can see whether the fundamentals-vs-peers
         features are actually contributing.
      5. Final signal is cross-sectionally ranked per day (relative, tradeable).
    """

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self._cfg = config.build_cube

    def run(self, full_history: bool = False):
        """Train the per-horizon ensemble. `full_history=True` = the PRODUCTION train on ALL data up
        to the latest cube date (no train_end cutoff, no OOS holdout) — used after the backtest to
        fit the model that generates the live predictions for the allocation.

        MEMORY-LIGHT: the cube is LONG by target_horizon and models are trained PER HORIZON, so we
        never hold every horizon at once. `_setup` resolves columns/horizons with NO data load, then
        `_process_horizons` reads ONLY one horizon's rows+cols at a time (float32), cross-validates,
        trains, saves, scores it for the blend, and FREES it before the next -> peak memory ~ a
        single horizon instead of the whole labelled cube (x its column-duplication across horizons)."""
        self._full_history = full_history
        self._setup()                        # columns + horizons; NO cube data loaded
        self._process_horizons()             # per-horizon: load -> CV -> train -> save-score -> free
        self.blend_and_generate_signal()     # blend the small per-horizon score frames
        self.log_feature_importance()
        self.save_models()
        self.save_outputs()
        self._save_run_diagnostics_kpis()    # run-level kpis (needs the blend weights)

    # ------------------------------------------------------------------ #
    def _setup(self):
        """Resolve the target column, horizons and per-member feature-column sets from the cube's
        SCHEMA only (information_schema + a cheap DISTINCT) — no feature data is read here."""
        self.primary_horizon = self._cfg.targets.primary_horizon
        self.label_column = self._config.model.label_column
        self.target_type = self._config.model.get("target_type", "rank")
        # ensemble: train EVERY listed model family and average their per-day standardized
        # predictions. `model.ensemble` (list) wins; else fall back to the single `model.type`.
        ens = self._config.model.get("ensemble", None)
        self.model_types = list(ens) if ens else [self._config.model.get("type", "lightgbm")]

        cube_cols = self._cube_columns()
        self._target_col = self._target_column(cube_cols)
        self._load_cols, dropped = self._select_load_columns(cube_cols, self._target_col)
        self.horizons = self._distinct_horizons(self._target_col)
        if not self.horizons:
            raise FileNotFoundError(
                f"No cube rows with a non-null target '{self._target_col}'. Run StepBuildCube "
                f"first (and confirm build_cube.targets.labels includes '{self.target_type}').")

        # per-member column sets = configured allow-list INTERSECTED with what the cube actually has
        # (from the schema; no data scan). Each member/horizon keeps its own (possibly leaner) set.
        avail = {c for c in cube_cols if c not in CUBE_META_COLS and c != self.label_column}
        self.linear_cols = [c for c in self._linear_columns() if c in avail]
        self.lgbm_cols = [c for c in self._lgbm_columns() if c in avail]
        self.rf_cols = [c for c in self._rf_columns() if c in avail]
        self.categorical_cols = [c for c in self._lgbm_categoricals() if c in avail]
        self.linear_cols_by_h = {h: [c for c in self._linear_columns(h) if c in avail] for h in self.horizons}
        self.lgbm_cols_by_h = {h: [c for c in self._lgbm_columns(h) if c in avail] for h in self.horizons}
        self.rf_cols_by_h = {h: [c for c in self._rf_columns(h) if c in avail] for h in self.horizons}
        allcols = set(self.linear_cols) | set(self.lgbm_cols) | set(self.rf_cols)
        for h in self.horizons:
            allcols |= set(self.linear_cols_by_h[h]) | set(self.lgbm_cols_by_h[h]) | set(self.rf_cols_by_h[h])
        self.feature_cols = sorted(allcols)
        if not self.linear_cols and not self.lgbm_cols:
            raise ValueError("No configured features present in the cube; check "
                             "linear_modelling.yml / lgbm_modelling.yml `columns`.")
        self._panel_cols = list(dict.fromkeys(self.feature_cols + self.categorical_cols))
        self._log.info("Setup: target=%s horizons=%s target_type=%s models=%s | features -> "
                       "linear:%d lgbm:%d (+%d cats) union:%d", self._target_col, self.horizons,
                       self.target_type, self.model_types, len(self.linear_cols), len(self.lgbm_cols),
                       len(self.categorical_cols), len(self.feature_cols))
        if dropped:
            self._log.info("modelling.yml columns absent from the cube (not loaded, %d): %s",
                           len(dropped), dropped)

    def _distinct_horizons(self, target_col: str) -> list:
        """Sorted horizons that HAVE a non-null label — a cheap DISTINCT (no feature scan)."""
        q = text(f'SELECT DISTINCT target_horizon FROM cube WHERE "{target_col}" IS NOT NULL '
                 "ORDER BY target_horizon")
        with self._context.store.engine.connect() as c:
            return pd.read_sql(q, c)["target_horizon"].tolist()

    @staticmethod
    def _downcast_f32(df: pd.DataFrame) -> pd.DataFrame:
        """float64 feature columns -> float32 (halve the loaded panel; labels/ranks need no float64)."""
        f64 = df.select_dtypes(include=["float64"]).columns
        if len(f64):
            df[f64] = df[f64].astype("float32")
        return df

    def _load_horizon_panel(self, horizon) -> pd.DataFrame | None:
        """Read ONLY this horizon's labelled rows (SQL WHERE target_horizon = h), projected to the
        model's columns and float32, then shape into a modelling panel + apply the train window.
        None if the horizon has no rows after filtering."""
        projected = ", ".join(f'"{c}"' for c in self._load_cols)
        q = text(f'SELECT {projected} FROM cube WHERE "{self._target_col}" IS NOT NULL '
                 "AND target_horizon = :h")
        with self._context.store.engine.connect() as c:
            raw = pd.read_sql(q, c, params={"h": int(horizon)}, parse_dates=["date"])
        if raw.empty:
            return None
        raw = self._downcast_f32(raw)
        panel = panel_from_cube(raw, horizon=horizon, label_name=self.label_column,
                                feature_cols=self._panel_cols, target_type=self.target_type)
        raw = None
        tr = self._config.get("train", None)
        if tr:
            start_date = tr.start_date
            end_date = None if getattr(self, "_full_history", False) else tr.end_date
            if start_date:
                panel = panel.loc[panel["date"] >= start_date]
            if end_date:
                panel = panel.loc[panel["date"] <= end_date]
        return panel if not panel.empty else None

    # ------------------------------------------------------------------ #
    def _cube_columns(self) -> set[str]:
        """Column names actually present in the `cube` table (no data scan)."""
        q = text("SELECT column_name FROM information_schema.columns "
                 "WHERE table_name = 'cube'")
        with self._context.store.engine.connect() as c:
            return set(pd.read_sql(q, c)["column_name"])

    def _target_column(self, cube_cols: set[str]) -> str:
        """Stored target column for the configured `target_type` (falls back to a
        legacy single `target` column), matching panel_from_cube's own resolution."""
        col = f"target_{self.target_type}"
        if col in cube_cols:
            return col
        if "target" in cube_cols:
            return "target"
        raise KeyError(
            f"Target column '{col}' not in cube; rebuild the cube with "
            f"'{self.target_type}' in build_cube.targets.labels.")

    # ---- per-model config accessors: linear_modelling.yml (`linear:`) and
    #      lgbm_modelling.yml (`lgbm:`), each with its OWN hyperparams + `columns`.
    #      Fall back to the legacy single-file layout (model.linear / model.lightgbm /
    #      inputs.*) so an old modellling.yml still works. ----
    def _lin_cfg(self):
        return self._config.get("linear") or (self._config.model.get("linear") or {})

    def _lgb_cfg(self):
        return self._config.get("lgbm") or (self._config.model.get("lightgbm") or {})

    @staticmethod
    def _by_horizon(cfg, horizon) -> list[str] | None:
        """Horizon-specific column override for a member config: returns
        `cfg.columns_by_horizon[<horizon>]` when the member declares a list for this
        horizon, else None so the caller falls back to the default `columns`. Keys are
        compared as ints, so a YAML `30:` matches a numpy/int horizon regardless of how
        OmegaConf typed the key."""
        by_h = cfg.get("columns_by_horizon") if cfg else None
        if not by_h or horizon is None:
            return None
        try:
            h = int(horizon)
        except (TypeError, ValueError):
            return None
        for k in by_h.keys():
            try:
                if int(k) == h and by_h.get(k):
                    return list(by_h.get(k))
            except (TypeError, ValueError):
                continue
        return None

    def _linear_columns(self, horizon=None) -> list[str]:
        c = self._config.get("linear")
        sp = self._by_horizon(c, horizon)
        if sp:
            return sp
        if c and c.get("columns"):
            return list(c.columns)
        inp = self._config.get("inputs")
        return list(inp.columns) if (inp and inp.get("columns")) else []

    def _lgbm_columns(self, horizon=None) -> list[str]:
        c = self._config.get("lgbm")
        sp = self._by_horizon(c, horizon)
        if sp:
            return sp
        if c and c.get("columns"):
            return list(c.columns)
        inp = self._config.get("inputs")
        return list(inp.columns) if (inp and inp.get("columns")) else []

    def _lgbm_categoricals(self) -> list[str]:
        c = self._config.get("lgbm")
        if c and c.get("categoricals") is not None:
            return list(c.categoricals)
        inp = self._config.get("inputs")
        return list(inp.categoricals) if (inp and inp.get("categoricals")) else []

    def _lgbm_monotonic(self):
        c = self._config.get("lgbm")
        if c and c.get("monotonic") is not None:
            return c.get("monotonic")
        inp = self._config.get("inputs")
        return inp.get("monotonic") if inp else None

    def _rf_cfg(self):
        return self._config.get("random_forest") or {}

    def _rf_columns(self, horizon=None) -> list[str]:
        c = self._config.get("random_forest")
        sp = self._by_horizon(c, horizon)
        if sp:
            return sp
        if c and c.get("columns"):
            return list(c.columns)
        return self._lgbm_columns(horizon)   # default: RF reuses the LightGBM feature set

    def _union_all_columns(self) -> list[str]:
        """Every column any member might use at ANY horizon: each member's default
        `columns` PLUS all of its `columns_by_horizon` overrides (+ the legacy inputs
        fallback). Drives the cube projection so a horizon-specific feature is still
        loaded from the cube even when it is absent from the default `columns`."""
        out: list[str] = []
        for key in ("linear", "lgbm", "random_forest"):
            c = self._config.get(key)
            if not c:
                continue
            if c.get("columns"):
                out += list(c.columns)
            by_h = c.get("columns_by_horizon")
            if by_h:
                for v in by_h.values():
                    if v:
                        out += list(v)
        inp = self._config.get("inputs")
        if inp and inp.get("columns"):
            out += list(inp.columns)
        return list(dict.fromkeys(out))

    def _select_load_columns(self, cube_cols: set[str], target_col: str
                             ) -> tuple[list[str], list[str]]:
        """Columns to pull: the index + target, plus the modelling.yml numeric
        allow-list and categoricals that exist in the cube. Returns (load_cols,
        dropped) where `dropped` are configured names absent from the cube."""
        wanted = self._union_all_columns()   # default + every columns_by_horizon override
        cats = self._lgbm_categoricals()
        requested = wanted + cats
        meta = ["date", "ticker", "target_horizon", target_col]
        feats = [c for c in requested if c in cube_cols and c not in meta]
        dropped = [c for c in requested if c not in cube_cols]
        load_cols = list(dict.fromkeys(meta + feats))          # de-dup, keep order
        return load_cols, dropped

    def _load_cube_where_labelled(self, load_cols: list[str], target_col: str
                                  ) -> pd.DataFrame:
        """SELECT the projected columns for rows whose target is non-null."""
        projected = ", ".join(f'"{c}"' for c in load_cols)
        q = text(f'SELECT {projected} FROM cube WHERE "{target_col}" IS NOT NULL')
        with self._context.store.engine.connect() as c:
            return pd.read_sql(q, c, parse_dates=["date"])

    def _select_categoricals(self, available: list[str]) -> list[str]:
        """Categorical columns (`inputs.categoricals` in modellling.yml, e.g. sector /
        industry_group). Fed to the LightGBM member as native categoricals; the
        linear member ignores them. Missing-from-cube names are skipped."""
        cats = self._lgbm_categoricals()
        avail = set(available)
        present = [c for c in cats if c in avail]
        missing = [c for c in cats if c not in avail]
        if missing:
            self._log.warning("Configured categoricals not in cube (skipped): %s", missing)
        if present:
            self._log.info("Categorical features (LightGBM-only): %s", present)
        return present

    def _lgb_feats(self, horizon=None) -> list[str]:
        """Feature list for the LightGBM member at `horizon` = its horizon-resolved
        numeric columns (columns_by_horizon override, else the default `columns`) + categoricals."""
        by_h = getattr(self, "lgbm_cols_by_h", None)
        base = by_h.get(horizon) if (by_h is not None and horizon is not None) else None
        cols = base if base is not None else getattr(self, "lgbm_cols", self.feature_cols)
        return list(cols) + list(getattr(self, "categorical_cols", []))

    def _rf_feats(self, horizon=None) -> list[str]:
        """Feature list for the Random Forest member at `horizon` = its horizon-resolved
        columns (columns_by_horizon override, else default `columns`, else LightGBM's) + categoricals."""
        by_h = getattr(self, "rf_cols_by_h", None)
        base = by_h.get(horizon) if (by_h is not None and horizon is not None) else None
        cols = base if base is not None else getattr(self, "rf_cols", getattr(self, "lgbm_cols", self.feature_cols))
        return list(cols) + list(getattr(self, "categorical_cols", []))

    def _monotone_constraints(self, feats: list[str] | None = None) -> list[int] | None:
        """Build LightGBM monotone_constraints from inputs.monotonic in config, aligned
        to `feats` (defaults to the LightGBM member's feature order). Passing the member's
        own feature list keeps the constraint vector aligned when members use different
        (per-horizon) column sets."""
        mono = self._lgbm_monotonic()
        if not mono or not mono.get("enabled", False):
            return None

        feature_map = ml.parse_monotone_feature_map(mono.get("features"))
        if not feature_map:
            self._log.warning("inputs.monotonic.enabled but no features configured")
            return None

        # align to the given feature order (numeric + categoricals); categoricals
        # are unconstrained (0). Only the tree members consume these.
        lgb_feats = feats if feats is not None else self._lgb_feats()
        constraints = ml.build_monotone_constraints(lgb_feats, feature_map)
        # A constrained feature absent from the trained set: if it IS in the allow-list
        # it's simply not in the cube yet (already reported once by load_cube) -> stay
        # quiet. Only WARN for a feature constrained but not even listed in
        # inputs.columns -- a genuine config typo.
        allow = set(self._lgbm_columns())
        absent = [f for f in feature_map if f not in lgb_feats]
        typos = [f for f in absent if f not in allow]
        if typos:
            self._log.warning("Monotone constraint on feature(s) not in inputs.columns "
                              "(typo?): %s", typos)
        elif absent:
            self._log.info("%d monotone constraint(s) inactive (feature not in the current "
                           "cube; applies after a rebuild)", len(absent))
        return constraints

    def _train_kwargs(self, horizon=None) -> dict:
        c = self._lgb_cfg()
        kw = {
            "params": {
                "learning_rate": c.learning_rate, "max_depth": c.max_depth,
                "num_leaves": c.get("num_leaves", 31),
                # subsample (=bagging_fraction) only takes effect when bagging_freq>0
                "subsample": c.subsample, "bagging_freq": c.get("bagging_freq", 0),
                "colsample_bytree": c.colsample_bytree,
                "min_child_samples": c.min_child_samples,
                "lambda_l1": c.lambda_l1, "lambda_l2": c.lambda_l2,
                # deterministic training keyed on the pipeline's global seed so a
                # rerun reproduces results bit-for-bit (see model.train_ranker)
                "seed": int(self._config.get("seed", ml.DEFAULT_SEED)),
                "deterministic": True,
                "force_row_wise": True,
            },
            "num_boost_round": c.num_boost_round,
            "early_stopping_rounds": int(c.get("early_stopping_rounds", ml.EARLY_STOPPING_ROUNDS)),
            # early-stop metric: lgbm-config `eval_metric` wins, else legacy model.eval_metric
            "eval_metric": c.get("eval_metric") or self._config.model.get("eval_metric", "rmse"),
        }
        monotone = self._monotone_constraints(self._lgb_feats(horizon))   # LightGBM member, at this horizon
        if monotone is not None:
            kw["params"]["monotone_constraints"] = monotone
        wd = self._config.model.get("weight_decay")
        if wd and wd.get("enabled", False):
            kw["half_life_years"] = float(wd.half_life_years)
        return kw

    def _half_life(self) -> float | None:
        wd = self._config.model.get("weight_decay")
        return float(wd.half_life_years) if (wd and wd.get("enabled", False)) else None

    def _fit_one(self, kind: str, train: pd.DataFrame, valid: pd.DataFrame | None,
                 horizon=None):
        """Fit ONE model family at `horizon` (each member resolves its own
        horizon-specific column set; see build_panels). LightGBM uses the purged
        validation fold for IC early stopping; the linear baselines ignore it."""
        if kind in ("ridge", "elasticnet"):
            lc = self._lin_cfg()
            by_h = getattr(self, "linear_cols_by_h", None)
            lin = by_h.get(horizon) if (by_h is not None and horizon is not None) else None
            lin = lin if lin is not None else self.linear_cols
            # linear member: its OWN numeric columns (categoricals are LightGBM-native)
            return baselines.train_linear(
                train, lin, self.label_column, kind=kind,
                alpha=float(lc.get("alpha", 1e-3)), l1_ratio=float(lc.get("l1_ratio", 0.5)),
                max_iter=int(lc.get("max_iter", 1000)), tol=float(lc.get("tol", 1e-6)),
                half_life_years=self._half_life())
        if kind == "random_forest":
            return self._fit_rf(train, horizon)
        # LightGBM member: numeric + categorical, with native categorical splits
        return ml.train_ranker(train, self._lgb_feats(horizon), self.label_column,
                               valid_panel=valid,
                               categorical_features=self.categorical_cols or None,
                               **self._train_kwargs(horizon))

    def _fit_rf(self, train: pd.DataFrame, horizon=None):
        """Random Forest = LightGBM in `boosting='rf'` mode: bagged INDEPENDENT trees, a FIXED
        count, NO early stopping (more trees only lower variance, never overfit). Reuses the
        LightGBM monotone map + categoricals; NaNs handled natively."""
        rf = self._rf_cfg()
        params = {
            "objective": "regression", "metric": "rmse", "boosting": rf.get("boosting", "rf"),
            "bagging_fraction": float(rf.get("bagging_fraction", 0.7)),
            "bagging_freq": int(rf.get("bagging_freq", 1)),
            "feature_fraction": float(rf.get("feature_fraction", 0.6)),
            "max_depth": int(rf.get("max_depth", 8)), "num_leaves": int(rf.get("num_leaves", 127)),
            "min_child_samples": int(rf.get("min_child_samples", 50)),
            "lambda_l1": float(rf.get("lambda_l1", 0.0)), "lambda_l2": float(rf.get("lambda_l2", 1.0)),
            "verbosity": -1, "seed": int(self._config.get("seed", ml.DEFAULT_SEED)),
            "deterministic": True, "force_row_wise": True,
        }
        feats = self._rf_feats(horizon)
        mono = self._monotone_constraints(feats)   # aligned to THIS member's (horizon) feature order
        if mono is not None and len(mono) == len(feats):
            params["monotone_constraints"] = mono
        return ml.train_ranker(train, feats, self.label_column, valid_panel=None,
                               params=params, num_boost_round=int(rf.get("num_boost_round", 500)),
                               categorical_features=self.categorical_cols or None,
                               half_life_years=self._half_life())

    def _fit_models(self, train: pd.DataFrame, valid: pd.DataFrame | None,
                    horizon=None) -> dict:
        """Fit every configured family at `horizon` -> {kind: model}. Averaged at
        predict time. Each member resolves its own horizon-specific column set."""
        return {kind: self._fit_one(kind, train, valid, horizon) for kind in self.model_types}

    def _process_horizons(self):
        """The per-horizon pipeline: for each horizon load ONLY its rows+cols, cross-validate,
        train the final ensemble, save it, score it for the blend, run its diagnostics, then FREE
        the panel before the next horizon. Peak memory is one horizon, not the whole cube."""
        self.models, self.cv_results, self.horizon_ic = {}, {}, {}
        self.member_ic = {}          # {h: {member_name: {mean_ic, ic_ir}}}  per-model CV IC
        self.oos_predictions = {}    # {h: concatenated OOS ENSEMBLE preds}  -> IC-over-time diag
        self._score_frames = []      # small per-horizon blend inputs (date,ticker,z_h,member cols)
        self._diag_summaries = {}    # {h: diagnostics summary} -> the run-level kpis.{json,csv}
        train_ends = []
        run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if self._context.save else None
        self._run_stamp = run_stamp
        if self._half_life():
            self._log.info("Time-decay sample weights enabled (half_life=%.1f years)",
                           self._half_life())

        for h in self.horizons:
            panel = self._load_horizon_panel(h)
            if panel is None or panel.empty:
                self._log.warning("horizon %s: no labelled rows after the train window -> skipped", h)
                continue
            self._log.info("horizon %s: %s rows, %s tickers, %s days", h, len(panel),
                           panel["ticker"].nunique(), panel["date"].nunique())
            train_ends.append(panel["date"].max())
            self._cv_one_horizon(h, panel)
            self.models[h] = self._train_final_one(h, panel)
            self._score_frames.append(self._score_one_horizon(h, self.models[h], panel))
            if run_stamp:
                self._horizon_diagnostics(h, panel, run_stamp)
            panel = None
            gc.collect()                        # hand this horizon's panel back before the next load

        if not self.models:
            raise RuntimeError("No horizon produced a model (empty cube / train window?).")
        self._train_end_effective = max(train_ends, default=None)
        self._log.info("Trained %s horizons x %s (%s)", len(self.models),
                       len(self.model_types), self.model_types)

    def _cv_one_horizon(self, h, panel: pd.DataFrame):
        """Purged walk-forward CV for ONE horizon; records IC (ensemble + per-member) + OOS preds."""
        cfg = self._config.model
        embargo = (cfg.cv.embargo or h)          # embargo must be >= horizon
        fold_results = []
        member_folds: dict[str, list] = {}       # member_name -> [per-fold daily_ic dicts]
        oos_frames = []
        for train_days, test_days in ml.purged_wf_splits(panel["date"], cfg.cv.n_splits, embargo):
            train = panel[panel["date"].isin(train_days)]
            test = panel[panel["date"].isin(test_days)]
            if train.empty or test.empty:
                continue
            sub_tr, sub_val = ml.temporal_valid_split(train)
            models = self._fit_models(sub_tr, sub_val, h)
            preds, members = ml.ensemble_predict(models, test, self.feature_cols)
            # annualize the IC IR by the horizon (overlapping labels), else it is inflated
            # ~sqrt(horizon) and long horizons look artificially strong
            fold_results.append(ml.daily_ic(test, preds, self.label_column, horizon=h))
            for name, mpred in members.items():
                member_folds.setdefault(name, []).append(
                    ml.daily_ic(test, mpred, self.label_column, horizon=h))
            oos_frames.append(pd.DataFrame({
                "date": test["date"].to_numpy(), "ticker": test["ticker"].to_numpy(),
                "pred": preds.to_numpy(), self.label_column: test[self.label_column].to_numpy(),
            }))

        self.cv_results[h] = fold_results
        if oos_frames:
            self.oos_predictions[h] = pd.concat(oos_frames, ignore_index=True)
        mean_ic = np.nanmean([r["mean_ic"] for r in fold_results]) if fold_results else np.nan
        mean_ir = np.nanmean([r["ic_ir"] for r in fold_results]) if fold_results else np.nan
        self.horizon_ic[h] = {"mean_ic": mean_ic, "ic_ir": mean_ir}
        self._log.info("horizon %s: [ENSEMBLE] CV mean_IC=%+.4f  IC_IR=%+.2f", h, mean_ic, mean_ir)
        self.member_ic[h] = {}
        for name, folds in member_folds.items():
            m_ic = np.nanmean([r["mean_ic"] for r in folds]) if folds else np.nan
            m_ir = np.nanmean([r["ic_ir"] for r in folds]) if folds else np.nan
            self.member_ic[h][name] = {"mean_ic": m_ic, "ic_ir": m_ir}
            self._log.info("horizon %s:   [%-10s] CV mean_IC=%+.4f  IC_IR=%+.2f",
                           h, name, m_ic, m_ir)

    def _train_final_one(self, h, panel: pd.DataFrame) -> dict:
        """Fit the full ensemble ({kind: model}) for one horizon on its whole panel."""
        sub_tr, sub_val = ml.temporal_valid_split(panel, train_frac=0.9)
        return self._fit_models(sub_tr, sub_val, h)

    def _score_one_horizon(self, h, models: dict, panel: pd.DataFrame) -> pd.DataFrame:
        """Ensemble-score one horizon's panel -> a SMALL frame (date, ticker, z_<h>, member preds)
        the blend consumes, so the wide panel can be freed immediately after."""
        scores, members = ml.ensemble_predict(models, panel, self.feature_cols)
        df = panel[["date", "ticker"]].copy()
        df["score"] = scores.to_numpy()
        # cross-sectional z per day so horizons are comparable (NaN-safe on <2-name days)
        df["z"] = ml.per_day_zscore(df["score"].to_numpy(), df["date"].to_numpy())
        member_cols = []
        for name, mz in members.items():
            col = f"pred_{name}_h{h}"
            df[col] = mz.to_numpy()
            member_cols.append(col)
        return df[["date", "ticker", "z"] + member_cols].rename(columns={"z": f"z_{h}"})

    def _booster_members(self, h) -> dict:
        """This horizon's tree members as `{member_name: booster}`, resolved by TYPE.

        Never by member NAME: this hook used to hardcode `.get("lightgbm")`, so it silently
        returned nothing the moment `model.ensemble` was rewritten to `[elasticnet, lgbm,
        random_forest]` -- no PDP, no SHAP, no KPIs, and no error either. `isinstance` is the
        honest test (random_forest is also an `lgb.Booster`, via `boosting='rf'`), so renaming
        or adding a tree member can never switch the diagnostics off again."""
        return {name: m for name, m in (self.models.get(h) or {}).items()
                if isinstance(m, lgb.Booster)}

    def _member_feature_cols(self, h) -> dict:
        """`{member_name: feature list}` for this horizon's boosters, taken from the booster
        ITSELF (`feature_name()`) rather than re-deriving it from config -- that is the exact
        column order it was trained on, so SHAP / PDP can never be mis-aligned."""
        return {name: list(b.feature_name()) for name, b in self._booster_members(h).items()}

    def _horizon_kpis(self, h) -> dict:
        """CV KPIs the diagnostics writer cannot know: ensemble + per-member IC / IC_IR."""
        ens = self.horizon_ic.get(h, {})

        full_train_end = datetime.today() - timedelta(days=30 + 30)
        train_end = (full_train_end.strftime("%Y-%m-%d") if getattr(self, "_full_history", False)
                          else self._config.train.end_date)
        self._context.log.info(f"Train from {self._config.train.start_date} to {train_end} ")

        return {
            "cv_mean_ic": ens.get("mean_ic"),
            "cv_ic_ir": ens.get("ic_ir"),
            "target_type": self.target_type,
            "label_column": self.label_column,
            "train_start": self._config.train.start_date,
            "train_end": train_end,
            "full_history": bool(getattr(self, "_full_history", False)),
            "members": {name: {"cv_mean_ic": m.get("mean_ic"), "cv_ic_ir": m.get("ic_ir")}
                        for name, m in (self.member_ic.get(h) or {}).items()},
        }

    def _horizon_diagnostics(self, h, panel: pd.DataFrame, run_stamp: str):
        """Per-horizon SHAP values / PDP / IC-over-time / KPIs for EVERY booster member,
        written under one shared run-stamp folder while the panel is still resident.
        Best-effort; never fails the run."""
        diag = self._config.model.get("diagnostics", {}) or {}
        if not diag.get("enabled", True):
            return
        boosters = self._booster_members(h)
        if not boosters:
            # LOUD, not silent: an ensemble of linear members only has no PDP/SHAP to give,
            # and the previous silence is exactly why the missing artifacts went unnoticed.
            self._log.warning("horizon %s diagnostics: no tree member in the ensemble %s -> "
                              "no SHAP/PDP possible (add 'lgbm' or 'random_forest' to "
                              "model.ensemble)", h, self.model_types)
            return
        run_dir = self._context.paths["OUTPUT_DIR"] / "diagnostics" / run_stamp
        try:
            summary = diagnostics.save_horizon_diagnostics(
                horizon=h, booster=None, panel=panel,
                feature_cols=self._lgb_feats(h), out_dir=run_dir / f"h{h}",
                oos_predictions=self.oos_predictions.get(h),
                boosters=boosters, feature_cols_by_member=self._member_feature_cols(h),
                kpis=self._horizon_kpis(h),
                label_name=self.label_column, top_n=int(diag.get("top_n_features", 15)),
                shap_sample=int(diag.get("shap_sample", 2000)),
                pdp_grid=int(diag.get("pdp_grid", 30)), logger=self._log,
            )
            self._diag_summaries[int(h)] = summary
            self._log.info("horizon %s diagnostics -> %s | %s | OOS IC %+.4f over %d days", h,
                           run_dir / f"h{h}",
                           ", ".join(f"{n}: {m['n_pdp']} PDPs, shap={m['shap_available']}"
                                     f"({m['shap_rows']} rows)"
                                     for n, m in summary["members"].items()),
                           summary["ic_mean"], summary["ic_days"])
        except Exception as e:                       # noqa: BLE001 - diagnostics are best-effort
            self._log.warning("horizon %s diagnostics failed: %s", h, e)

    def _save_run_diagnostics_kpis(self):
        """Run-level `kpis.json` + flat `kpis.csv` across horizons -- written after the blend so
        each horizon's IR weight is known. Best-effort; never fails the run."""
        if not getattr(self, "_diag_summaries", None) or not self._run_stamp:
            return
        weights = getattr(self, "horizon_weights", {}) or {}
        for h, summary in self._diag_summaries.items():
            summary["blend_weight"] = float(weights.get(h, float("nan")))
        run_dir = self._context.paths["OUTPUT_DIR"] / "diagnostics" / self._run_stamp
        try:
            diagnostics.save_run_kpis(
                run_dir,
                {"run_stamp": self._run_stamp, "model_types": list(self.model_types),
                 "target_type": self.target_type,
                 "horizons": self._diag_summaries},
                logger=self._log)
        except Exception as e:                       # noqa: BLE001
            self._log.warning("run-level diagnostics KPIs failed: %s", e)

    def _horizon_weights(self) -> dict:
        """IR-weight horizons; floor negatives at 0, fall back to equal if all <=0."""
        irs = {h: max(0.0, self.horizon_ic[h]["ic_ir"]) for h in self.models
               if np.isfinite(self.horizon_ic[h]["ic_ir"])}
        total = sum(irs.values())
        if total <= 0:
            return {h: 1.0 / len(self.models) for h in self.models}
        return {h: irs.get(h, 0.0) / total for h in self.models}

    def blend_and_generate_signal(self):
        """IR-weight the per-horizon z-scores (already computed in _score_one_horizon), blend, rank.
        Consumes the SMALL per-horizon score frames (never re-touches the wide panels, which are
        already freed)."""
        weights = self._horizon_weights()
        self.horizon_weights = weights
        self._log.info("Blend weights: %s", {h: round(w, 3) for h, w in weights.items()})

        blended = None
        for df in self._score_frames:
            blended = df if blended is None else blended.merge(df, on=["date", "ticker"], how="outer")

        zcols = [f"z_{h}" for h in self.models]
        w = np.array([weights[h] for h in self.models])
        z = blended[zcols].to_numpy()
        # weighted nanmean across available horizons
        mask = ~np.isnan(z)
        wmat = np.where(mask, w, 0.0)
        wsum = wmat.sum(axis=1)
        blended["combined"] = np.where(wsum > 0,
                                       np.nansum(np.where(mask, z * w, 0.0), axis=1) / np.where(wsum > 0, wsum, 1),
                                       np.nan)
        # final per-day cross-sectional rank -> tradeable relative signal
        blended["signal"] = blended.groupby("date")["combined"].rank(pct=True)
        self.predictions = blended

        last_date = blended["date"].max()
        latest = blended[blended["date"] == last_date].sort_values("signal", ascending=False)
        self.signal = latest.set_index("ticker")["signal"]
        self.signal_date = last_date
        self._log.info("Blended signal for %s (%s names)",
                       pd.Timestamp(last_date).date(), self.signal.notna().sum())

    def log_feature_importance(self):
        """Aggregate gain importance across horizon models -> which features matter."""
        try:
            imp = {}
            for h, models in self.models.items():
                for model in models.values():
                    # LightGBM gain vs |coef| for the linear baselines; normalize
                    # each member to sum 1 first so the two scales are comparable.
                    gains = (ml.feature_importance(model, list(model.feature_names))
                             if isinstance(model, lgb.Booster)
                             else baselines.linear_importance(model))
                    s = pd.Series(gains, dtype=float)
                    tot = s.sum()
                    if tot > 0:
                        s = s / tot
                    for f, g in s.items():
                        imp[f] = imp.get(f, 0.0) + float(g)
            imp_s = pd.Series(imp).sort_values(ascending=False)
            imp_s = imp_s / imp_s.sum()
            self.feature_importance = imp_s
            top = imp_s.head(15)
            self._log.info("Top features by gain:\n%s", top.round(4).to_string())
            fund_share = imp_s[[f for f in imp_s.index if f.startswith("f_")]].sum()
            self._log.info("Peer-relative fundamentals share of importance: %.1f%%",
                           100 * fund_share)
        except Exception as e:  # feature_importance helper may not exist in ml
            self._log.warning("Feature importance unavailable: %s", e)
            self.feature_importance = None

    def save_outputs(self):
        out = self._cfg.output
        # full rebuild each run -> replace the predictions table
        self._context.store.replace("predictions", self.predictions)
        self._log.info("Saved predictions to DB table 'predictions'")

        if not self._context.save:
            return

        if out.save_cv_results:
            rows = []
            for h, folds in self.cv_results.items():
                for i, r in enumerate(folds):
                    rows.append({"horizon": h, "fold": i, **r})
            # CV results are a diagnostics artifact -> keep as a file
            pd.DataFrame(rows).to_parquet(self._context.paths["CUBE_CV_RESULTS_PATH"], index=False)

        if out.save_signal:
            sig = self.signal.rename("signal").reset_index()
            sig.insert(0, "date", self.signal_date)
            self._context.store.replace("cube_signal", sig)
            self._log.info("Saved blended signal to DB table 'cube_signal'")

    def save_models(self):
        """Persist each horizon's booster + metadata for the backtest step."""
        models_dir = self._context.paths["MODELS_DIR"]
        for h, models in self.models.items():
            for kind, model in models.items():
                # booster kinds (lightgbm AND random_forest via boosting='rf') -> .txt;
                # linear baselines -> .pkl. member_model_path is the shared naming rule the
                # backtest + app read back, so every chosen member round-trips.
                path = ml.member_model_path(models_dir, h, kind)
                if isinstance(model, lgb.Booster):
                    model.save_model(str(path))
                else:                               # linear baseline -> pickle
                    with path.open("wb") as f:
                        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        meta = {
            "horizons": [int(h) for h in self.models],
            "feature_cols": list(self.feature_cols),        # union (backtest panel + fallback)
            "linear_cols": list(getattr(self, "linear_cols", [])),
            "lgbm_cols": list(getattr(self, "lgbm_cols", [])),
            "rf_cols": list(getattr(self, "rf_cols", [])),
            # per-horizon resolved feature sets (columns_by_horizon override, else default)
            "linear_cols_by_h": {int(h): v for h, v in getattr(self, "linear_cols_by_h", {}).items()},
            "lgbm_cols_by_h": {int(h): v for h, v in getattr(self, "lgbm_cols_by_h", {}).items()},
            "rf_cols_by_h": {int(h): v for h, v in getattr(self, "rf_cols_by_h", {}).items()},
            "categorical_cols": list(getattr(self, "categorical_cols", [])),
            "label_column": self.label_column,
            "target_type": self.target_type,
            "model_types": list(self.model_types),
            "train_start": self._config.train.start_date,
            # full-history run records the ACTUAL latest trained date; normal run keeps the config cutoff
            "train_end": (pd.Timestamp(self._train_end_effective).strftime("%Y-%m-%d")
                          if getattr(self, "_full_history", False)
                          and getattr(self, "_train_end_effective", None) is not None
                          else self._config.train.end_date),
            "full_history": bool(getattr(self, "_full_history", False)),
            # blend weights for the backtest (IC_IR per horizon, floored at 0)
            "train_ic_ir": {int(h): (float(self.horizon_ic[h]["ic_ir"])
                                     if np.isfinite(self.horizon_ic[h]["ic_ir"]) else 0.0)
                            for h in self.models},
        }
        (models_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        self._log.info("Saved %d models + metadata.json to %s", len(self.models), models_dir)

    # ================================================================== #
    # Standalone PRODUCTION prediction (own DAG step after full-train)
    # ================================================================== #
    def _load_saved_ensemble(self) -> tuple[dict, dict]:
        """Load metadata.json + each horizon's member models from MODELS_DIR (disk), so prediction
        runs as its OWN process/DAG step without retraining. Returns (meta, {horizon: {kind: model}})."""
        models_dir = self._context.paths["MODELS_DIR"]
        meta_path = models_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No metadata.json in {models_dir}; run full-train first.")
        meta = json.loads(meta_path.read_text())
        model_types = list(meta.get("model_types") or [meta.get("model_type", "lightgbm")])
        models: dict[int, dict] = {}
        for h in (int(x) for x in meta["horizons"]):
            members: dict = {}
            for kind in model_types:
                p = ml.member_model_path(models_dir, h, kind)
                if not p.exists():
                    continue
                if kind in ml.BOOSTER_MEMBER_KINDS:
                    b = lgb.Booster(model_file=str(p)); b.feature_names = b.feature_name()
                    members[kind] = b
                else:
                    with p.open("rb") as f:
                        members[kind] = pickle.load(f)
            if members:
                models[h] = members
        if not models:
            raise FileNotFoundError(f"No saved model files in {models_dir}.")
        return meta, models

    def _latest_cube_dates(self, n_dates: int) -> list[pd.Timestamp]:
        with self._context.store.engine.connect() as c:
            rows = c.execute(text("SELECT DISTINCT date FROM cube ORDER BY date DESC LIMIT :n"),
                             {"n": int(n_dates)}).fetchall()
        return sorted(pd.Timestamp(r[0]).normalize() for r in rows)

    @staticmethod
    def predicts_for(as_of, horizon) -> pd.Timestamp:
        """The date a horizon-`h` prediction is ABOUT: `as_of` + h business days.

        The cube target is a forward return over h ROWS of the daily price panel, i.e. h TRADING
        days -- so a business-day offset is the right arithmetic, not a calendar one. It still
        ignores market holidays, so treat this as the target date +/- a few sessions rather than
        a settlement date."""
        return pd.Timestamp(as_of) + pd.tseries.offsets.BDay(int(round(float(horizon))))

    def predict_latest(self, n_dates: int = 1) -> pd.DataFrame:
        """PRODUCTION prediction. Loads the (full-trained) ensemble artifacts, scores the LATEST
        `n_dates` cube date(s) for EVERY horizon, and persists `predictions_latest` in LONG form:

            predicted_at | date | ticker | horizon | model | predicts_for | pred | rank

        one row per (as-of date, ticker, horizon, model), where `model` is each ensemble member,
        `ensemble` for that horizon's member average, and `blended` for the IR-weighted blend
        ACROSS horizons. Long rather than wide because `predicts_for` is per horizon: the h30 and
        h90 predictions made today are about different future dates, which one column cannot hold.
        `predicted_at` is when this run produced the row -- distinct from `date`, the as-of date of
        the FEATURES it was computed from.

        Crucially it does NOT use panel_from_cube (which drops null-target rows) — the latest date's
        forward target has not matured, so we build the feature panel DIRECTLY from the cube so the
        newest date is actually predictable."""
        meta, models = self._load_saved_ensemble()
        feat_cols = list(meta["feature_cols"])
        cat_cols = list(meta.get("categorical_cols", []))
        train_ic = {int(k): float(v) for k, v in meta.get("train_ic_ir", {}).items()}
        predicted_at = pd.Timestamp.now().floor("s")

        dates = self._latest_cube_dates(n_dates)
        if not dates:
            raise RuntimeError("cube is empty -> nothing to predict.")
        start = min(dates)

        cube_cols = self._cube_columns()
        want = [c for c in dict.fromkeys(feat_cols + cat_cols) if c in cube_cols]
        load_cols = list(dict.fromkeys(["date", "ticker", "target_horizon"] + want))
        hlist = ", ".join(str(int(h)) for h in models)
        q = text(f'SELECT {", ".join(chr(34)+c+chr(34) for c in load_cols)} FROM cube '
                 f"WHERE target_horizon IN ({hlist}) AND date >= :d")
        with self._context.store.engine.connect() as c:
            cube = pd.read_sql(q, c, params={"d": str(start.date())}, parse_dates=["date"])
        if cube.empty:
            raise RuntimeError(f"No cube rows on/after {start.date()} for horizons {list(models)}.")

        long_rows: list[pd.DataFrame] = []
        ens_wide = None                       # per-horizon ensemble z, for the cross-horizon blend
        for h, members in models.items():
            sub = cube[cube["target_horizon"] == h]
            if sub.empty:
                continue
            present = [c for c in (feat_cols + cat_cols) if c in sub.columns]
            missing = [c for c in (feat_cols + cat_cols) if c not in sub.columns]
            panel = sub[["date", "ticker"] + present].copy()
            if missing:                                      # add any absent model feature as NaN (one concat)
                panel = pd.concat(
                    [panel, pd.DataFrame(np.nan, index=panel.index, columns=missing)], axis=1)
            scores, member_preds = ml.ensemble_predict(members, panel, feat_cols)
            keys = panel[["date", "ticker"]]
            # every MEMBER, then the horizon's ENSEMBLE — all per-day z-scored so they are
            # comparable across horizons and members
            per_model = {**{name: p.to_numpy() for name, p in member_preds.items()},
                         PREDICTION_MODEL_ENSEMBLE: scores.to_numpy()}
            for name, raw in per_model.items():
                long_rows.append(self._prediction_rows(keys, raw, h, name, predicted_at))
            ez = keys.copy()
            ez[f"z{h}"] = ml.per_day_zscore(scores.to_numpy(), keys["date"].to_numpy())
            ens_wide = ez if ens_wide is None else ens_wide.merge(ez, on=["date", "ticker"], how="outer")
        if ens_wide is None:
            raise RuntimeError("No horizon produced a prediction for the latest cube date(s).")

        # cross-horizon blend: IR-weighted mean of the per-horizon ensembles (from metadata)
        hs = [int(h) for h in models if f"z{h}" in ens_wide.columns]
        irs = {h: max(0.0, train_ic.get(h, 0.0)) for h in hs}
        tot = sum(irs.values())
        w = {h: (irs[h] / tot if tot > 0 else 1.0 / len(hs)) for h in hs}
        z = ens_wide[[f"z{h}" for h in hs]].to_numpy()
        mask = ~np.isnan(z)
        wv = np.array([w[h] for h in hs])
        wsum = np.where(mask, wv, 0.0).sum(axis=1)
        blend = np.where(wsum > 0,
                         np.nansum(np.where(mask, z * wv, 0.0), axis=1) / np.where(wsum > 0, wsum, 1),
                         np.nan)
        # the blend has no single horizon, so it is stamped with the IR-WEIGHTED AVERAGE horizon
        # (rounded) -- the average distance into the future the signal is actually about.
        blend_h = int(round(sum(w[h] * h for h in hs))) if hs else 0
        long_rows.append(self._prediction_rows(ens_wide[["date", "ticker"]], blend, blend_h,
                                               PREDICTION_MODEL_BLENDED, predicted_at))

        out = pd.concat(long_rows, ignore_index=True)
        out = out.sort_values(["date", "model", "horizon", "rank"],
                             ascending=[True, True, True, False]).reset_index(drop=True)
        self._context.store.replace(PREDICTIONS_LATEST_TABLE, out)

        last = out[(out["date"] == out["date"].max())
                   & (out["model"] == PREDICTION_MODEL_BLENDED)]
        self._log.info("predict_latest: %d row(s) -> '%s' | as-of %s | horizons %s x models %s | "
                       "blend weights %s", len(out), PREDICTIONS_LATEST_TABLE,
                       [str(d.date()) for d in dates], sorted(out["horizon"].unique()),
                       sorted(out["model"].unique()), {h: round(w[h], 3) for h in hs})
        self._log.info("blended (h~%d) on %s: %d names, predicts_for %s, pred range [%.3f, %.3f]",
                       blend_h, out["date"].max().date(), len(last),
                       last["predicts_for"].max().date() if not last.empty else None,
                       float(last["pred"].min()), float(last["pred"].max()))
        return out

    def _prediction_rows(self, keys: pd.DataFrame, raw: np.ndarray, horizon, model: str,
                         predicted_at: pd.Timestamp) -> pd.DataFrame:
        """One (horizon, model) slice as long rows: per-day z-scored `pred` + its cross-sectional
        `rank`, stamped with when it was predicted and which date it is about."""
        df = keys.copy()
        df["horizon"] = int(horizon)
        df["model"] = str(model)
        df["predicted_at"] = predicted_at
        df["predicts_for"] = df["date"].map(lambda d: self.predicts_for(d, horizon))
        df["pred"] = ml.per_day_zscore(np.asarray(raw, dtype="float64"), df["date"].to_numpy())
        df["rank"] = df.groupby("date")["pred"].rank(pct=True)
        return df[["predicted_at", "date", "ticker", "horizon", "model", "predicts_for",
                   "pred", "rank"]]
