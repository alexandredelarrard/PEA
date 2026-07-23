"""Sector / industry categoricals go to the LightGBM member as NATIVE categorical
splits (int codes), while the linear ensemble member stays numeric-only. The
ensemble scores each member on its OWN feature_names, so the two feature sets
coexist (src/modelling/utils_model/model.py + baselines.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.modelling.long_short.utils import model as ml
from src.modelling.long_short.utils import baselines


def _panel(seed: int = 0) -> tuple[pd.DataFrame, list[str], list[str]]:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2020-01-01", periods=80)
    rows = []
    for d in days:
        for t in range(30):
            sector = t % 5
            num1 = rng.normal()
            # label depends on num1 AND the sector (a non-linear categorical shift)
            y = num1 + (2.0 if sector == 3 else -0.5) + rng.normal(scale=0.1)
            rows.append({"date": d, "ticker": f"T{t}", "num1": num1,
                         "num2": rng.normal(), "sector": np.int16(sector), "y": y})
    df = pd.DataFrame(rows)
    # store the model target on a 0..1 scale like the cube's rank target
    df["y"] = df.groupby("date")["y"].rank(pct=True)
    return df, ["num1", "num2"], ["sector"]


def test_categorical_is_lgbm_only_and_ensemble_scores():
    df, numeric, cats = _panel()
    lgb_model = ml.train_ranker(df, numeric + cats, "y", categorical_features=cats,
                                num_boost_round=40)
    lin = baselines.train_linear(df, numeric, "y", kind="elasticnet")

    # the tree member carries the categorical; the linear member does NOT
    assert "sector" in lgb_model.feature_names
    assert list(lin.feature_names) == numeric and "sector" not in lin.feature_names

    # ensemble scores each member on its own feature_names -> different sets coexist
    blended, members = ml.ensemble_predict({"lightgbm": lgb_model, "elasticnet": lin},
                                           df, numeric)
    assert set(members) == {"lightgbm", "elasticnet"}
    assert np.isfinite(blended.to_numpy()).any()

    # the tree actually USES sector (it drives the label) -> non-trivial gain importance
    gains = ml.feature_importance(lgb_model, list(lgb_model.feature_names))
    assert gains.get("sector", 0.0) > 0.0

    # numeric-only path unchanged: a booster without categoricals still trains/predicts
    plain = ml.train_ranker(df, numeric, "y", num_boost_round=20)
    assert "sector" not in plain.feature_names
    p = ml.predict(plain, df, numeric)
    assert np.isfinite(p.to_numpy()).all()

    print("\n=== SANITY CHECK: sector categorical -> LightGBM-only ===")
    print(f"  LightGBM feats={lgb_model.feature_names} (sector native categorical, "
          f"gain={gains['sector']:.0f}); linear feats={lin.feature_names} (numeric only); "
          f"ensemble blends both. Numeric-only booster still works. Validated.")
