"""Known-truth assessment arithmetic: fixed populations, pruning and date-balanced samples."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modelling.utils.assessment import balanced_sample_positions
from src.modelling.utils.metrics import daily_ic, summarize_ic


def test_pooled_ir_uses_all_dates_instead_of_mean_fold_ir() -> None:
    values = pd.Series([0.1, 0.2, -0.2, 0.1, 0.3])
    result = summarize_ic(values, 60)
    expected = values.mean() / values.to_numpy().std() * np.sqrt(252 / 60)
    assert result["ic_ir"] == pytest.approx(expected)
    assert result["n_days"] == 5
    print("SANITY: pooled IC-IR uses the full daily series with population dispersion and h60 adjustment.")


@pytest.mark.parametrize("horizon", [-3, 0, 1, 60])
def test_daily_ic_preserves_custom_annualization_and_horizon_conversion(horizon: int) -> None:
    panel = pd.DataFrame({"date": np.repeat(pd.bdate_range("2020-01-02", periods=3), 4), "y": [0.0, 1.0, 2.0, 3.0] * 3})
    predictions = pd.Series([0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0, 0.0, 2.0, 1.0, 3.0])
    known_ic = pd.Series([1.0, -1.0, 0.8])
    result = daily_ic(panel, predictions, horizon=horizon, trading_days_per_year=360)
    assert result["mean_ic"] == pytest.approx(known_ic.mean())
    assert result["ic_std"] == pytest.approx(known_ic.to_numpy().std())
    assert result["ic_ir"] == pytest.approx(known_ic.mean() / known_ic.to_numpy().std() * np.sqrt(360 / max(1, int(horizon))))
    assert result["n_days"] == 3
    assert summarize_ic(known_ic, horizon, trading_days_per_year=360) == pytest.approx(result)
    print("SANITY: known daily ranks retain custom 360-session annualization and the existing minimum-one-session horizon conversion.")


@pytest.mark.parametrize("values, n_days", [([], 0), ([np.nan, np.inf, -np.inf], 0), ([0.2, np.nan, np.inf, 0.2], 2)])
def test_ic_summary_ignores_nonfinite_and_keeps_degenerate_statistics_undefined(values: list[float], n_days: int) -> None:
    result = summarize_ic(pd.Series(values, dtype=float), 60)
    assert result["n_days"] == n_days
    assert np.isnan(result["ic_ir"])
    if n_days:
        assert result["mean_ic"] == 0.2 and result["ic_std"] == 0
    else:
        assert np.isnan(result["mean_ic"]) and np.isnan(result["ic_std"])
    print("SANITY: only finite ICs count; empty and constant supported series retain undefined IC-IR without warnings.")


def test_date_balanced_sample_is_reproducible_and_feature_independent() -> None:
    panel = pd.DataFrame({"date": np.repeat(pd.bdate_range("2000-01-03", periods=5), [10, 20, 40, 70, 100])})
    positions = balanced_sample_positions(panel, 17, 4325)
    counts = panel.iloc[positions].groupby("date").size()
    assert len(positions) == 17 and counts.max() - counts.min() <= 1
    assert np.array_equal(positions, balanced_sample_positions(panel.assign(unused=np.nan), 17, 4325))
    assert len(balanced_sample_positions(panel, 0, 4325)) == len(panel)
    print("SANITY: the same 17 row keys are sampled across models, balanced by date despite changing breadth.")
