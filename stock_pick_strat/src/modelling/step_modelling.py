import numpy as np
import json
import pandas as pd
from omegaconf import DictConfig

from src.utils.step import Step
from src.context import Context
from src.data_aggregate.utils.cube import panel_from_cube, feature_columns_from_cube
from src.modelling.utils_model import model as ml
from src.modelling.utils_model import shap_analysis


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
        self.save_shap_explanations()
        self.train_final_models()
        self.save_models()
        self.blend_and_generate_signal()
        self.log_feature_importance()
        self.save_outputs()

    # ------------------------------------------------------------------ #
    def load_cube(self):
        cube_path = self._context.paths["CUBE_PATH"]
        if not cube_path.exists():
            raise FileNotFoundError(f"Cube not found at {cube_path}. Run StepBuildCube first.")
        self.cube = pd.read_parquet(cube_path)
        self.horizons = sorted(self.cube["target_horizon"].unique())
        self.primary_horizon = self._cfg.targets.primary_horizon
        self.label_column = self._config.model.label_column
        self._log.info("Loaded cube (%s rows), horizons=%s", len(self.cube), self.horizons)

    def _select_features(self, available: list[str]) -> list[str]:
        """Restrict training to the config allow-list (`inputs.columns` in
        modellling.yml). Commented-out YAML entries are simply not parsed, so
        only the uncommented names are used. Falls back to ALL cube features
        when no list is configured."""
        cfg_inputs = self._config.get("inputs", None)
        wanted = list(cfg_inputs.columns) if (cfg_inputs and cfg_inputs.get("columns")) else None
        if not wanted:
            self._log.warning("inputs.columns not configured -> training on ALL %d "
                              "cube features", len(available))
            return available

        avail = set(available)
        selected = [c for c in wanted if c in avail]
        missing = [c for c in wanted if c not in avail]
        excluded = [c for c in available if c not in set(wanted)]
        if missing:
            self._log.warning("Configured features not in cube (skipped): %s", missing)
        if excluded:
            self._log.info("Cube features excluded by allow-list (%d): %s",
                           len(excluded), excluded)
        if not selected:
            raise ValueError("No configured features are present in the cube; "
                             "check inputs.columns in modellling.yml")
        self._log.info("Training on %d configured features", len(selected))
        return selected

    def build_panels(self):
        """One modeling panel per horizon, restricted to the configured features."""

        start_date = None
        if self._config.get("train", None):
            start_date = self._config.get("train").start_date
            end_date = self._config.get("train").end_date
       
        available = feature_columns_from_cube(self.cube, self.label_column)
        self.feature_cols = self._select_features(available)

        self.panels = {}
        for h in self.horizons:
            panel = panel_from_cube(self.cube, horizon=h, label_name=self.label_column,
                                    feature_cols=self.feature_cols)

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

    def _train_kwargs(self) -> dict:
        c = self._config.model.lightgbm
        return {
            "params": {
                "learning_rate": c.learning_rate, "max_depth": c.max_depth,
                "subsample": c.subsample, "colsample_bytree": c.colsample_bytree,
                "min_child_samples": c.min_child_samples,
                "lambda_l1": c.lambda_l1, "lambda_l2": c.lambda_l2,
            },
            "num_boost_round": c.num_boost_round,
            "early_stopping_rounds": ml.EARLY_STOPPING_ROUNDS,
        }

    def cross_validate_all_horizons(self):
        """Purged walk-forward CV per horizon; collect IC to weight the blend."""
        cfg = self._config.model
        self.cv_results = {}
        self.horizon_ic = {}
        self.last_cv_folds = {}
        train_kw = self._train_kwargs()

        for h, panel in self.panels.items():
            embargo = (cfg.cv.embargo or h)      # embargo must be >= horizon
            fold_results = []
            last_fold = None
            for train_days, test_days in ml.purged_wf_splits(
                panel["date"], cfg.cv.n_splits, embargo
            ):
                train = panel[panel["date"].isin(train_days)]
                test = panel[panel["date"].isin(test_days)]
                if train.empty or test.empty:
                    continue
                sub_tr, sub_val = ml.temporal_valid_split(train)
                booster = ml.train_ranker(sub_tr, self.feature_cols,
                                          self.label_column, valid_panel=sub_val, **train_kw)
                preds = ml.predict(booster, test, self.feature_cols)
                fold_results.append(ml.daily_ic(test, preds, self.label_column))
                last_fold = {"model": booster, "test_panel": test}

            self.cv_results[h] = fold_results
            if last_fold is not None:
                self.last_cv_folds[h] = last_fold

            mean_ic = np.nanmean([r["mean_ic"] for r in fold_results]) if fold_results else np.nan
            mean_ir = np.nanmean([r["ic_ir"] for r in fold_results]) if fold_results else np.nan
            self.horizon_ic[h] = {"mean_ic": mean_ic, "ic_ir": mean_ir}
            self._log.info("horizon %s: CV mean_IC=%+.4f  IC_IR=%+.2f", h, mean_ic, mean_ir)

    def save_shap_explanations(self):
        """SHAP + dependence plots for the last CV fold of the primary horizon."""
        if not getattr(self._cfg.output, "save_shap", False) or not self._context.save:
            return

        fold = self.last_cv_folds.get(self.primary_horizon)
        if fold is None:
            self._log.warning("No CV fold available for SHAP analysis")
            return

        out_dir = self._context.paths["SHAP_OUTPUT_DIR"]
        try:
            imp = shap_analysis.save_shap_analysis(
                fold["model"],
                fold["test_panel"],
                self.feature_cols,
                out_dir,
                horizon=self.primary_horizon,
            )
            self._log.info(
                "Saved SHAP analysis to %s (top feature: %s)",
                out_dir, imp.index[0],
            )
        except Exception as e:
            self._log.warning("SHAP analysis failed: %s", e)

    def _horizon_weights(self) -> dict:
        """IR-weight horizons; floor negatives at 0, fall back to equal if all <=0."""
        irs = {h: max(0.0, self.horizon_ic[h]["ic_ir"]) for h in self.panels
               if np.isfinite(self.horizon_ic[h]["ic_ir"])}
        total = sum(irs.values())
        if total <= 0:
            return {h: 1.0 / len(self.panels) for h in self.panels}
        return {h: irs.get(h, 0.0) / total for h in self.panels}

    def train_final_models(self):
        """Fit one model per horizon on the full panel."""
        train_kw = self._train_kwargs()
        self.models = {}
        for h, panel in self.panels.items():
            sub_tr, sub_val = ml.temporal_valid_split(panel)
            self.models[h] = ml.train_ranker(sub_tr, self.feature_cols,
                                             self.label_column, valid_panel=sub_val, **train_kw)
        self._log.info("Trained %s per-horizon final models", len(self.models))

    def blend_and_generate_signal(self):
        """Standardize each horizon's scores per day, IR-weight, blend, rank."""
        weights = self._horizon_weights()
        self.horizon_weights = weights
        self._log.info("Blend weights: %s", {h: round(w, 3) for h, w in weights.items()})

        blended = None
        for h, model in self.models.items():
            panel = self.panels[h]
            scores = ml.predict(model, panel, self.feature_cols).to_numpy()
            df = panel[["date", "ticker"]].copy()
            df["score"] = scores
            # cross-sectional z-score per day so horizons are comparable
            df["z"] = df.groupby("date")["score"].transform(
                lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else np.nan)
            )
            df = df[["date", "ticker", "z"]].rename(columns={"z": f"z_{h}"})
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
            for h, model in self.models.items():
                gains = ml.feature_importance(model, self.feature_cols)  # {feat: gain}
                for f, g in gains.items():
                    imp[f] = imp.get(f, 0.0) + g
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
        predictions_path = self._context.paths["PREDICTIONS_PATH"]
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        self.predictions.to_parquet(predictions_path, index=False)
        self._log.info("Saved predictions to %s", predictions_path)

        if not self._context.save:
            return

        if out.save_cv_results:
            rows = []
            for h, folds in self.cv_results.items():
                for i, r in enumerate(folds):
                    rows.append({"horizon": h, "fold": i, **r})
            pd.DataFrame(rows).to_parquet(self._context.paths["CUBE_CV_RESULTS_PATH"], index=False)

        if out.save_signal:
            sig = self.signal.rename("signal").reset_index()
            sig.insert(0, "date", self.signal_date)
            sig.to_parquet(self._context.paths["CUBE_SIGNAL_PATH"], index=False)
            self._log.info("Saved blended signal to %s", self._context.paths["CUBE_SIGNAL_PATH"])

    def save_models(self):
        """Persist each horizon's booster + metadata for the backtest step."""
        models_dir = self._context.paths["MODELS_DIR"]
        for h, model in self.models.items():
            model.save_model(str(models_dir / f"model_h{h}.txt"))

        meta = {
            "horizons": [int(h) for h in self.models],
            "feature_cols": list(self.feature_cols),
            "label_column": self.label_column,
            "train_start": self._config.train.start_date,
            "train_end": self._config.train.end_date,
            # blend weights for the backtest (IC_IR per horizon, floored at 0)
            "train_ic_ir": {int(h): (float(self.horizon_ic[h]["ic_ir"])
                                     if np.isfinite(self.horizon_ic[h]["ic_ir"]) else 0.0)
                            for h in self.models},
        }
        (models_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        self._log.info("Saved %d models + metadata.json to %s", len(self.models), models_dir)
