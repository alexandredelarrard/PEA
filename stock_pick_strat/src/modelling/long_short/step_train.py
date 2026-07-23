import numpy as np
import json
import pickle
import pandas as pd
from datetime import datetime
from omegaconf import DictConfig
from sqlalchemy import text
import lightgbm as lgb

from src.utils.step import Step
from src.context import Context
from src.data_aggregate.utils.cube import panel_from_cube, feature_columns_from_cube
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

    def run(self):
        self.load_cube()
        self.build_panels()
        self.cross_validate_all_horizons()
        self.train_final_models()
        self.save_models()
        self.blend_and_generate_signal()
        self.log_feature_importance()
        self.save_diagnostics()
        self.save_outputs()

    # ------------------------------------------------------------------ #
    def load_cube(self):
        self.primary_horizon = self._cfg.targets.primary_horizon
        self.label_column = self._config.model.label_column
        self.target_type = self._config.model.get("target_type", "rank")
        # ensemble: train EVERY listed model family and average their per-day
        # standardized predictions. `model.ensemble` (list) wins; else fall back to
        # the single `model.type`.
        ens = self._config.model.get("ensemble", None)
        self.model_types = list(ens) if ens else [self._config.model.get("type", "lightgbm")]

        # Load ONLY what the model needs: the index/target columns plus the
        # modelling.yml allow-list + categoricals that actually exist in the cube,
        # and ONLY rows where the target is available. The cube's NULL-label rows
        # (the beta-warmup head + the forward-return tail) are dropped downstream by
        # panel_from_cube anyway, so pushing the filter into SQL yields the SAME
        # panels (CV / training / blending / saved predictions unchanged) while
        # avoiding a full 159-column x millions-of-rows load.
        cube_cols = self._cube_columns()
        target_col = self._target_column(cube_cols)
        load_cols, dropped = self._select_load_columns(cube_cols, target_col)
        self.cube = self._load_cube_where_labelled(load_cols, target_col)
        if self.cube.empty:
            raise FileNotFoundError(
                f"No cube rows with a non-null target '{target_col}'. Run StepBuildCube "
                "first (and confirm build_cube.targets.labels includes "
                f"'{self.target_type}').")

        self.horizons = sorted(self.cube["target_horizon"].unique())
        self._log.info("Loaded cube: %s labelled rows x %s cols (target=%s), horizons=%s, "
                       "target_type=%s, models=%s", len(self.cube), len(load_cols),
                       target_col, self.horizons, self.target_type, self.model_types)
        if dropped:
            self._log.info("modelling.yml columns absent from the cube (not loaded, "
                           "%d): %s", len(dropped), dropped)

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

    def build_panels(self):
        """One modeling panel per horizon, restricted to the configured features."""

        start_date = None
        if self._config.get("train", None):
            start_date = self._config.get("train").start_date
            end_date = self._config.get("train").end_date
       
        available = feature_columns_from_cube(self.cube, self.label_column)
        avail = set(available)
        # each family trains on its OWN column list (linear_modelling.yml / lgbm_modelling.yml)
        self.linear_cols = [c for c in self._linear_columns() if c in avail]   # horizon-agnostic default
        self.lgbm_cols = [c for c in self._lgbm_columns() if c in avail]
        self.rf_cols = [c for c in self._rf_columns() if c in avail]
        self.categorical_cols = [c for c in self._lgbm_categoricals() if c in avail]  # tree members
        # per-horizon resolved feature lists: a member uses its columns_by_horizon[h]
        # override when present, else its default `columns` (see _by_horizon). Members
        # with no override resolve to the same default list for every horizon.
        self.linear_cols_by_h = {h: [c for c in self._linear_columns(h) if c in avail] for h in self.horizons}
        self.lgbm_cols_by_h = {h: [c for c in self._lgbm_columns(h) if c in avail] for h in self.horizons}
        self.rf_cols_by_h = {h: [c for c in self._rf_columns(h) if c in avail] for h in self.horizons}
        # union across members AND horizons drives the cube projection + is the
        # ensemble_predict fallback; each fitted model still stores + predicts on its
        # OWN feature_names, so per-horizon members keep their own (leaner) column set.
        allcols = set(self.linear_cols) | set(self.lgbm_cols) | set(self.rf_cols)
        for h in self.horizons:
            allcols |= set(self.linear_cols_by_h[h]) | set(self.lgbm_cols_by_h[h]) | set(self.rf_cols_by_h[h])
        self.feature_cols = sorted(allcols)
        for name, want, got in (("linear", self._linear_columns(), self.linear_cols),
                                ("lgbm", self._lgbm_columns(), self.lgbm_cols)):
            miss = [c for c in want if c not in avail]
            if miss:
                self._log.warning("%s: %d configured feature(s) absent from cube (skipped): %s",
                                  name, len(miss), miss)
        if not self.linear_cols and not self.lgbm_cols:
            raise ValueError("No configured features present in the cube; check "
                             "linear_modelling.yml / lgbm_modelling.yml `columns`.")
        self._log.info("Feature sets -> linear:%d  lgbm:%d  (+%d categoricals)  union:%d",
                       len(self.linear_cols), len(self.lgbm_cols),
                       len(self.categorical_cols), len(self.feature_cols))
        panel_cols = list(dict.fromkeys(self.feature_cols + self.categorical_cols))

        self.panels = {}
        for h in self.horizons:
            panel = panel_from_cube(self.cube, horizon=h, label_name=self.label_column,
                                    feature_cols=panel_cols,
                                    target_type=self.target_type)

            if start_date:
                panel = panel.loc[panel["date"] >= start_date]

            if end_date:
                panel = panel.loc[panel["date"] <= end_date]

            if not panel.empty:
                self.panels[h] = panel
        self._log.info("Built %s horizon panels, %s features",
                       len(self.panels), len(self.feature_cols))

        for h, panel in self.panels.items():
            self._log.info("  h=%s: %s rows, %s tickers, %s days",
                           h, len(panel), panel["ticker"].nunique(), panel["date"].nunique())

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

    def cross_validate_all_horizons(self):
        """Purged walk-forward CV per horizon; collect IC to weight the blend."""
        
        cfg = self._config.model
        self.cv_results = {}
        self.horizon_ic = {}
        self.member_ic = {}         # {h: {member_name: {mean_ic, ic_ir}}}  per-model CV IC
        self.oos_predictions = {}   # concatenated out-of-sample ENSEMBLE preds -> IC-over-time
        
        if self._half_life():
            self._log.info("Time-decay sample weights enabled (half_life=%.1f years)",
                           self._half_life())

        for h, panel in self.panels.items():
            embargo = (cfg.cv.embargo or h)      # embargo must be >= horizon
            fold_results = []
            member_folds: dict[str, list] = {}   # member_name -> [per-fold daily_ic dicts]
            oos_frames = []
            for train_days, test_days in ml.purged_wf_splits(
                panel["date"], cfg.cv.n_splits, embargo
            ):
                train = panel[panel["date"].isin(train_days)]
                test = panel[panel["date"].isin(test_days)]
                if train.empty or test.empty:
                    continue
                sub_tr, sub_val = ml.temporal_valid_split(train)
                models = self._fit_models(sub_tr, sub_val, h)
                # ENSEMBLE preds + each member's per-day-standardized preds
                preds, members = ml.ensemble_predict(
                    models, test, self.feature_cols)
                # annualize the IC IR by the horizon (overlapping labels), else it
                # is inflated ~sqrt(horizon) and long horizons look artificially strong
                fold_results.append(ml.daily_ic(test, preds, self.label_column, horizon=h))
                # per-member IC on the SAME test fold -> compare ensemble vs each model
                for name, mpred in members.items():
                    member_folds.setdefault(name, []).append(
                        ml.daily_ic(test, mpred, self.label_column, horizon=h))
                oos_frames.append(pd.DataFrame({
                    "date": test["date"].to_numpy(),
                    "ticker": test["ticker"].to_numpy(),
                    "pred": preds.to_numpy(),
                    self.label_column: test[self.label_column].to_numpy(),
                }))

            self.cv_results[h] = fold_results
            if oos_frames:
                self.oos_predictions[h] = pd.concat(oos_frames, ignore_index=True)

            mean_ic = np.nanmean([r["mean_ic"] for r in fold_results]) if fold_results else np.nan
            mean_ir = np.nanmean([r["ic_ir"] for r in fold_results]) if fold_results else np.nan
            self.horizon_ic[h] = {"mean_ic": mean_ic, "ic_ir": mean_ir}
            self._log.info("horizon %s: [ENSEMBLE] CV mean_IC=%+.4f  IC_IR=%+.2f",
                           h, mean_ic, mean_ir)

            # per-model IC / IC_IR so it is clear which member carries the ensemble
            self.member_ic[h] = {}
            for name, folds in member_folds.items():
                m_ic = np.nanmean([r["mean_ic"] for r in folds]) if folds else np.nan
                m_ir = np.nanmean([r["ic_ir"] for r in folds]) if folds else np.nan
                self.member_ic[h][name] = {"mean_ic": m_ic, "ic_ir": m_ir}
                self._log.info("horizon %s:   [%-10s] CV mean_IC=%+.4f  IC_IR=%+.2f",
                               h, name, m_ic, m_ir)

    def save_diagnostics(self):
        """Per-run, per-horizon diagnosis folder under
        <OUTPUT_DIR>/diagnostics/<run_stamp>/h<H>/: top-N individual partial-
        dependence plots, SHAP importance (SHAP only), an Excel importance table,
        and the out-of-sample IC-over-time curve (CV folds concatenated). See
        utils_model/diagnostics.py. Optional deps (shap / xlsx engine) degrade
        gracefully so this never fails the pipeline."""
        if not self._context.save:
            return
        # SHAP / gain / PDP are tree-specific -> run them on the LightGBM MEMBER of
        # each horizon's ensemble; the IC-over-time curve uses the ENSEMBLE OOS preds.
        tree_models = {h: m["lightgbm"] for h, m in self.models.items() if "lightgbm" in m}
        if not tree_models:
            self._log.info("No LightGBM member in the ensemble -> tree diagnostics skipped")
            return
        diag = self._config.model.get("diagnostics", {}) or {}
        if not diag.get("enabled", True):
            return
        top_n = int(diag.get("top_n_features", 15))
        shap_sample = int(diag.get("shap_sample", 2000))
        pdp_grid = int(diag.get("pdp_grid", 30))

        run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = self._context.paths["OUTPUT_DIR"] / "diagnostics" / run_stamp
        try:
            diagnostics.save_run_diagnostics(
                run_dir, tree_models, self.panels, self._lgb_feats(),
                getattr(self, "oos_predictions", {}),
                label_name=self.label_column, top_n=top_n,
                shap_sample=shap_sample, pdp_grid=pdp_grid, logger=self._log,
            )
            self._log.info("Saved per-horizon model diagnostics to %s", run_dir)
        except Exception as e:
            self._log.warning("Diagnostics generation failed: %s", e)

    def _horizon_weights(self) -> dict:
        """IR-weight horizons; floor negatives at 0, fall back to equal if all <=0."""
        irs = {h: max(0.0, self.horizon_ic[h]["ic_ir"]) for h in self.panels
               if np.isfinite(self.horizon_ic[h]["ic_ir"])}
        total = sum(irs.values())
        if total <= 0:
            return {h: 1.0 / len(self.panels) for h in self.panels}
        return {h: irs.get(h, 0.0) / total for h in self.panels}

    def train_final_models(self):
        """Fit the full ensemble ({kind: model}) per horizon on the full panel."""
        self.models = {}
        for h, panel in self.panels.items():
            sub_tr, sub_val = ml.temporal_valid_split(panel, train_frac=0.9)
            self.models[h] = self._fit_models(sub_tr, sub_val, h)
        self._log.info("Trained final models for %s horizons x %s (%s)",
                       len(self.models), len(self.model_types), self.model_types)

    def blend_and_generate_signal(self):
        """Standardize each horizon's scores per day, IR-weight, blend, rank."""
        weights = self._horizon_weights()
        self.horizon_weights = weights
        self._log.info("Blend weights: %s", {h: round(w, 3) for h, w in weights.items()})

        blended = None
        for h, models in self.models.items():
            panel = self.panels[h]
            # ensemble = per-day-standardized average of the trained families;
            # also keep each member's standardized prediction for transparency
            scores, members = ml.ensemble_predict(
                models, panel, self.feature_cols)
            df = panel[["date", "ticker"]].copy()
            df["score"] = scores.to_numpy()
            # cross-sectional z-score per day so horizons are comparable (warning-safe:
            # NaN on <2-name days / a constant member, no DOF/empty-slice RuntimeWarnings)
            df["z"] = ml.per_day_zscore(df["score"].to_numpy(), df["date"].to_numpy())
            # per-model predictions (already per-day standardized in ensemble_predict)
            member_cols = []
            for name, mz in members.items():
                col = f"pred_{name}_h{h}"
                df[col] = mz.to_numpy()
                member_cols.append(col)
            df = df[["date", "ticker", "z"] + member_cols].rename(columns={"z": f"z_{h}"})
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
            "train_end": self._config.train.end_date,
            # blend weights for the backtest (IC_IR per horizon, floored at 0)
            "train_ic_ir": {int(h): (float(self.horizon_ic[h]["ic_ir"])
                                     if np.isfinite(self.horizon_ic[h]["ic_ir"]) else 0.0)
                            for h in self.models},
        }
        (models_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        self._log.info("Saved %d models + metadata.json to %s", len(self.models), models_dir)
