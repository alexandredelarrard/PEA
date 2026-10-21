"""Native contribution reconstruction and observed-domain PDP mathematics."""

from __future__ import annotations

from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pytest

from src.modelling.transformers.assessment_diagnostics import partial_dependence, shap_row_values, shap_score_scale


@pytest.mark.parametrize(
    "boosting,objective,n_trees", [("rf", "regression", 3), ("rf", "regression", 11), ("gbdt", "regression", 7), ("rf", "binary", 5)]
)
def test_shap_contributions_reconcile_in_model_score_units(boosting: str, objective: str, n_trees: int) -> None:
    x = np.random.default_rng(12).normal(size=(120, 3))
    y = (x[:, 0] > 0).astype(float) if objective == "binary" else 1.25 + 0.7 * x[:, 0] - 0.4 * x[:, 1]
    trained = lgb.train(
        {
            "boosting_type": boosting,
            "objective": objective,
            "verbosity": -1,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "feature_fraction": 0.8,
            "num_leaves": 7,
            "min_data_in_leaf": 3,
            "num_threads": 1,
            "seed": 7,
        },
        lgb.Dataset(x, label=y, feature_name=["f0", "f1", "f2"]),
        num_boost_round=n_trees,
    )
    booster = lgb.Booster(model_str=trained.model_to_string())
    assert booster.num_trees() == n_trees
    predictions = np.asarray(booster.predict(x, raw_score=objective == "binary"))
    got = shap_row_values(booster, x, booster.feature_name(), sample=31)
    assert got is not None, "the declared SHAP dependency must explain this supported fixture"
    values, idx = got
    assert values.shape == (31, 3) and len(np.unique(idx)) == 31
    native = np.asarray(booster.predict(x[idx], pred_contrib=True))
    scale = 1.0 / n_trees if boosting == "rf" and objective == "regression" else 1.0
    assert shap_score_scale(booster) == scale
    np.testing.assert_allclose(native.sum(axis=1) * scale, predictions[idx], atol=1e-12)
    np.testing.assert_allclose(values, native[:, :-1] * scale, atol=1e-12)
    np.testing.assert_allclose(values.sum(axis=1) + native[:, -1] * scale, predictions[idx], atol=1e-12)
    assert np.array_equal(predictions, np.asarray(booster.predict(x, raw_score=objective == "binary")))
    if objective == "binary":
        assert not np.allclose(predictions, np.asarray(booster.predict(x))), "classification SHAP explains raw margins, not probabilities"
    print(f"SANITY: {boosting}/{objective}, {n_trees} trees: sampled SHAP plus base reconstructs model scores; scale={scale:g}.")


def test_categorical_pdp_uses_only_observed_valid_codes() -> None:
    class Scorer:
        def predict(self, values: np.ndarray) -> np.ndarray:
            assert set(values[:, 0]) <= {2, 7}
            return values[:, 0] + values[:, 1]

    matrix = np.array([[-1.0, 3.0], [2.0, 4.0], [7.0, 5.0]])
    original = matrix.copy()
    grid, means = partial_dependence(Scorer(), matrix, 0, categorical=True)
    np.testing.assert_array_equal(grid, [2, 7])
    np.testing.assert_allclose(means, [6, 11])
    np.testing.assert_array_equal(matrix, original)
    print("SANITY: categorical PDP uses observed integer codes, excludes missing -1 and leaves the background unchanged.")


@pytest.mark.parametrize("categorical,rare_count,sample", [(False, 10, 0), (False, 10, 20), (True, 1, 20)])
def test_pdp_preserves_rare_observed_states(categorical: bool, rare_count: int, sample: int) -> None:
    matrix = np.zeros((1000, 1))
    matrix[-rare_count:, 0] = 1
    original = matrix.copy()

    def predict(values: np.ndarray) -> np.ndarray:
        assert len(values) == (sample or len(matrix))
        return values[:, 0]

    grid, means = partial_dependence(SimpleNamespace(predict=predict), matrix, 0, sample=sample, categorical=categorical)
    np.testing.assert_array_equal(grid, [0, 1])
    np.testing.assert_array_equal(means, [0, 1])
    np.testing.assert_array_equal(matrix, original)
    print(
        f"SANITY: categorical={categorical}, rare rows={rare_count}, background={sample or len(matrix)}: both states retain their known score effect 1."
    )


@pytest.mark.parametrize("categorical,grid_points,expected", [(False, 4, [-7, -1, 0, 3]), (True, 1, [0, 3])])
def test_pdp_discrete_domain_filters_only_invalid_states(categorical: bool, grid_points: int, expected: list[int]) -> None:
    matrix = np.array([[-7.0], [-1.0], [0.0], [3.0], [np.nan], [np.inf], [-np.inf]])
    original = matrix.copy()

    def predict(values: np.ndarray) -> np.ndarray:
        return values[:, 0]

    grid, means = partial_dependence(SimpleNamespace(predict=predict), matrix, 0, grid_points=grid_points, categorical=categorical)
    np.testing.assert_array_equal(grid, expected)
    np.testing.assert_array_equal(means, expected)
    np.testing.assert_array_equal(matrix, original)
    print(f"SANITY: categorical={categorical}: observed grid {expected} excludes nonfinite values and preserves numeric negatives only.")


@pytest.mark.parametrize("categorical,values", [(False, [np.nan, np.inf, -np.inf]), (True, [-7, -1, np.nan, np.inf]), (False, []), (True, [])])
def test_pdp_empty_valid_domain_skips_prediction(categorical: bool, values: list[float]) -> None:
    matrix = np.asarray(values, dtype=float).reshape(-1, 1)
    original = matrix.copy()

    def predict(values: np.ndarray) -> np.ndarray:
        raise AssertionError("an empty valid domain must not be scored")

    assert partial_dependence(SimpleNamespace(predict=predict), matrix, 0, categorical=categorical) == (None, None)
    np.testing.assert_array_equal(matrix, original)
    print(f"SANITY: categorical={categorical}, rows={len(values)}: empty valid domain returns (None, None) without scoring or input mutation.")


@pytest.mark.parametrize("sample,seed", [(0, 42), (17, 42), (17, 7)])
def test_pdp_continuous_grid_preserves_background_and_score_units(sample: int, seed: int) -> None:
    feature = np.concatenate([np.linspace(-10, 10, 101), [np.nan, np.inf, -np.inf]])
    matrix = np.column_stack([np.arange(len(feature)), feature])
    original = matrix.copy()
    idx = np.random.default_rng(seed).choice(len(matrix), sample, replace=False) if sample else np.arange(len(matrix))
    background = matrix[idx, 0].copy()
    calls = 0

    def predict(values: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        np.testing.assert_array_equal(values[:, 0], background)
        return 7 + 3 * values[:, 1] + values[:, 0]

    grid, means = partial_dependence(SimpleNamespace(predict=predict), matrix, 1, grid_points=8, sample=sample, seed=seed)
    expected = np.quantile(feature[np.isfinite(feature)], np.linspace(0.02, 0.98, 8))
    np.testing.assert_allclose(grid, expected)
    np.testing.assert_allclose(means, 7 + 3 * expected + background.mean())
    assert calls == 8
    np.testing.assert_array_equal(matrix, original)
    print(
        f"SANITY: continuous PDP retains the full-domain 2-98% grid and native score units; seed={seed}, background={len(background)}, input unchanged."
    )
