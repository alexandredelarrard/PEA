"""
Ensemble member persistence round-trip — including random_forest.

StepModelling.save_models writes each member with `ml.member_model_path` (booster kinds
lightgbm/random_forest -> .txt via Booster.save_model; linear baselines -> .pkl), and
StepBacktest.load_models reads them back the SAME way (restoring the Booster's
`.feature_names`). Regression guard: a `random_forest` member (LightGBM boosting='rf') must
be SAVED and RELOADED as a booster and participate in the ensemble — previously the
lightgbm-only `.txt` check skipped it, so it silently vanished from the backtest/app.
"""
from __future__ import annotations

import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.modelling.utils_model import baselines, model as ml


def _panel(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f1, f2 = rng.standard_normal(n), rng.standard_normal(n)
    y = 0.5 * f1 - 0.3 * f2 + 0.05 * rng.standard_normal(n)
    dates = pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 120, n), unit="D")
    return pd.DataFrame({"date": dates, "ticker": np.arange(n) % 200, "y": y,
                         "f1": f1, "f2": f2,
                         "sector": rng.integers(0, 5, n), "industry_group": rng.integers(0, 8, n)})


def _rf(panel, feats, cats):
    return ml.train_ranker(
        panel, feats, "y", valid_panel=None, categorical_features=cats, num_boost_round=40,
        params={"objective": "regression", "metric": "rmse", "boosting": "rf",
                "bagging_fraction": 0.7, "bagging_freq": 1, "feature_fraction": 0.7})


def test_random_forest_member_saves_and_reloads_into_the_ensemble(tmp_path):
    panel = _panel()
    num, cats = ["f1", "f2"], ["sector", "industry_group"]
    feats = num + cats
    members = {
        "elasticnet": baselines.train_elasticnet(panel, num, "y", alpha=1e-3, l1_ratio=0.3),
        "lightgbm": ml.train_ranker(panel, feats, "y", valid_panel=None, categorical_features=cats,
                                    num_boost_round=40, params={"objective": "regression", "metric": "rmse"}),
        "random_forest": _rf(panel, feats, cats),
    }

    # shared naming rule: booster kinds -> .txt, linear -> .pkl
    assert ml.member_model_path(tmp_path, 60, "random_forest").suffix == ".txt"
    assert ml.member_model_path(tmp_path, 60, "lightgbm").suffix == ".txt"
    assert ml.member_model_path(tmp_path, 60, "elasticnet").suffix == ".pkl"

    # SAVE exactly like StepModelling.save_models
    for kind, m in members.items():
        p = ml.member_model_path(tmp_path, 60, kind)
        if isinstance(m, lgb.Booster):
            m.save_model(str(p))
        else:
            with p.open("wb") as f:
                pickle.dump(m, f, protocol=pickle.HIGHEST_PROTOCOL)

    # RELOAD exactly like StepBacktest.load_models
    reloaded = {}
    for kind in members:
        p = ml.member_model_path(tmp_path, 60, kind)
        assert p.exists(), f"{kind} was not saved to {p.name}"
        if kind in ml.BOOSTER_MEMBER_KINDS:
            b = lgb.Booster(model_file=str(p))
            b.feature_names = b.feature_name()
            reloaded[kind] = b
        else:
            with p.open("rb") as f:
                reloaded[kind] = pickle.load(f)

    # random_forest came back as a booster whose feature_names include the categoricals
    assert isinstance(reloaded["random_forest"], lgb.Booster)
    assert list(reloaded["random_forest"].feature_names) == feats

    # booster predictions are identical pre/post reload
    for kind in ("lightgbm", "random_forest"):
        assert np.allclose(members[kind].predict(panel[feats]), reloaded[kind].predict(panel[feats]))

    # the FULL 3-member ensemble scores without a feature-count mismatch, and every member
    # contributes a real (non-constant) per-day signal
    scores, mem = ml.ensemble_predict(reloaded, panel, num)
    assert set(mem) == {"elasticnet", "lightgbm", "random_forest"}
    assert np.isfinite(scores.to_numpy()).mean() > 0.9
    disp = {k: float(pd.Series(v.to_numpy(), index=panel.index).groupby(panel["date"]).std().mean())
            for k, v in mem.items()}
    assert all(d > 1e-6 for d in disp.values()), f"a member is degenerate: {disp}"

    print("\n=== SANITY CHECK: ensemble member persistence (incl. random_forest) ===")
    print(f"  files: {[ml.member_model_path(tmp_path,60,k).name for k in members]}")
    print(f"  random_forest reloaded as Booster, feature_names={list(reloaded['random_forest'].feature_names)}")
    print(f"  per-day dispersion by member: {{k: round(v,3) for k,v in disp.items()}}"
          .replace("{k: round(v,3) for k,v in disp.items()}", str({k: round(v, 3) for k, v in disp.items()})))
    print("  CONCLUSION: a random_forest member is saved (.txt) AND reloaded as a booster with its "
          "feature_names, and the 3-way ensemble scores cleanly -> reused by backtest + app. Validated.")


if __name__ == "__main__":
    import tempfile, pathlib
    test_random_forest_member_saves_and_reloads_into_the_ensemble(pathlib.Path(tempfile.mkdtemp()))
