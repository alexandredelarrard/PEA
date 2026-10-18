"""Known-truth support and conditional means for SHAP dependence summaries."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.modelling.transformers import assessment_diagnostics
from src.modelling.transformers.assessment_diagnostics import _binned_dependence
from src.modelling.transformers.linear_regression import LinearRegression


def test_saved_linear_coefficients_reconstruct_signed_effects_and_pdp(tmp_path: Path) -> None:
    member = object.__new__(LinearRegression)
    member.features = ["positive", "negative", "zero", "constant", "missing"]
    member.categoricals, member.family, member.horizon = [], "elasticnet", 60
    member.coef_ = np.array([2.0, -3.0, 0.0, 5.0, 0.5])
    member.mean_, member.std_ = np.array([1.0, 5.0, 0.0, 7.0, 0.0]), np.array([2.0, 4.0, 1.0, 1.0, 1.0])
    member.intercept_, member.model = 0.25, "fitted"
    panel = pd.DataFrame(
        {
            "date": pd.bdate_range("2022-01-03", periods=4),
            "ticker": list("ABCD"),
            "positive": [1.0, 3.0, np.nan, 5.0],
            "negative": [1.0, 5.0, 9.0, np.nan],
            "zero": [1.0, 2.0, 3.0, 4.0],
            "constant": 7.0,
            "missing": np.nan,
        }
    )
    config = OmegaConf.create({"seed": 7, "model": {"diagnostics": {"dependence_sample": 4, "pdp_grid": 30, "pdp_background": 4}}})
    result = assessment_diagnostics.explain_linear(member, panel, tmp_path, config, logging.getLogger(__name__), pdp=True)
    values = pd.read_parquet(tmp_path / "contributions.parquet")
    expected = np.nan_to_num((panel[member.features].to_numpy() - member.mean_) / member.std_, nan=0.0) * member.coef_
    np.testing.assert_allclose(values[member.features], expected)
    np.testing.assert_allclose(values[member.features].sum(axis=1) + values.base, member.predict(panel), atol=1e-12)
    for index, feature in enumerate(member.features):
        curve = result["features"][feature]
        assert curve["importance"] == abs(member.coef_[index])
        assert curve["coefficient"] == member.coef_[index]
        if curve["pdp"]["x"] is not None:
            for grid, mean in zip(curve["pdp"]["x"], curve["pdp"]["mean"], strict=True):
                assert mean == pytest.approx(member.predict(panel.assign(**{feature: grid})).mean(), abs=1e-12)
    assert result["kind"] == "linear" and result["additivity_max_error"] < 1e-12
    assert result["features"]["missing"]["pdp"]["x"] is None
    assert result["features"]["constant"]["binned"]["mean"] == pytest.approx([0.0])
    print(
        "SANITY: saved standardized beta, intercept and scaling exactly reconstruct positive/negative/zero/missing/constant contributions and direct-predict PDP; importance is abs(beta)."
    )


@pytest.mark.parametrize(
    ("states", "counts"),
    [
        ([0.0, 1.0], [990, 10]),
        ([-3.0, 0.0, 4.0], [990, 8, 2]),
        ([7.0], [20]),
        (list(np.arange(10.0)), [991, *([1] * 9)]),
    ],
    ids=["rare-binary", "negative-multistate", "constant", "ten-state-cap"],
)
def test_binned_dependence_preserves_observed_numeric_states(states: list[float], counts: list[int]) -> None:
    x = np.repeat(states, counts)[::-1]
    phi = 2 * x
    original_x, original_phi = x.copy(), phi.copy()

    result = _binned_dependence(x, phi, categorical=False)

    assert set(result) == {"x", "mean", "count"}
    np.testing.assert_array_equal(result["x"], states)
    np.testing.assert_allclose(result["mean"], 2 * np.asarray(states))
    np.testing.assert_array_equal(result["count"], counts)
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(phi, original_phi)
    print(f"SANITY: all {len(states)} observed numeric states retain exact SHAP means and counts, including rare states; inputs are unchanged.")


def test_binned_dependence_groups_unordered_codes_and_excludes_missing_sentinels() -> None:
    x = np.asarray([8.0, -1.0, 2.0, 8.0, 2.0, -2.0, np.nan, np.inf, -np.inf])
    phi = np.asarray([3.0, 100.0, 2.0, 5.0, 6.0, 100.0, 100.0, 100.0, 100.0])
    original_x, original_phi = x.copy(), phi.copy()

    result = _binned_dependence(x, phi, categorical=True)

    np.testing.assert_array_equal(result["x"], [2.0, 8.0])
    np.testing.assert_allclose(result["mean"], [4.0, 4.0])
    np.testing.assert_array_equal(result["count"], [2, 2])
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(phi, original_phi)
    print("SANITY: unordered categories keep their observed codes and arithmetic SHAP means; missing sentinels and nonfinite values are excluded.")


@pytest.mark.parametrize(
    ("values", "categorical"),
    [([], False), ([np.nan, np.inf, -np.inf], False), ([-1.0, -2.0, np.nan, np.inf, -np.inf], True)],
    ids=["empty", "nonfinite-numeric", "missing-categorical"],
)
def test_binned_dependence_returns_empty_artifact_without_valid_states(values: list[float], categorical: bool) -> None:
    x = np.asarray(values, dtype=float)

    assert _binned_dependence(x, np.zeros_like(x), categorical) == {"x": [], "mean": [], "count": []}
    print("SANITY: no valid observed states yields the existing empty x/mean/count artifact.")


def test_binned_dependence_filters_nonfinite_numeric_values_but_keeps_negative_states() -> None:
    x = np.asarray([np.nan, np.inf, -np.inf, -2.0, -2.0, 0.0, 3.0])
    phi = np.asarray([99.0, 99.0, 99.0, 1.0, 3.0, 4.0, 8.0])

    result = _binned_dependence(x, phi, categorical=False)

    np.testing.assert_array_equal(result["x"], [-2.0, 0.0, 3.0])
    np.testing.assert_allclose(result["mean"], [2.0, 4.0, 8.0])
    np.testing.assert_array_equal(result["count"], [2, 1, 1])
    print("SANITY: numeric negatives remain observed states; nonfinite x values cannot contaminate exact conditional means or support counts.")


def test_binned_dependence_retains_ten_quantile_bins_for_continuous_values() -> None:
    x = np.arange(101.0)[::-1]
    phi = x**2 - 3 * x
    original_x, original_phi = x.copy(), phi.copy()
    expected_groups = [np.arange(11.0), *(np.arange(start, start + 10.0) for start in range(11, 101, 10))]

    result = _binned_dependence(x, phi, categorical=False)

    np.testing.assert_allclose(result["x"], [np.median(group) for group in expected_groups])
    np.testing.assert_allclose(result["mean"], [(group**2 - 3 * group).mean() for group in expected_groups])
    np.testing.assert_array_equal(result["count"], [11, *([10] * 9)])
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(phi, original_phi)
    print("SANITY: 101 continuous states retain the original ten quantile groups, median x, mean SHAP and full support; inputs are unchanged.")
