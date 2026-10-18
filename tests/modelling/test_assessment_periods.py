"""Known-truth controls for retrospective weak-period diagnosis, never model selection."""

import numpy as np
import pandas as pd
import pytest

from src.modelling.steps.step_assess_long_short import StepAssessLongShort
from src.modelling.transformers.assessment_diagnostics import period_evidence


def test_fixed_cohort_and_earlier_model_comparisons_preserve_the_population() -> None:
    rows = []
    for date in pd.to_datetime(["2018-01-02", "2018-01-03", "2019-01-02", "2019-01-03"]):
        for index, ticker in enumerate(["A", "B", "C", "D"]):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "fold": 4 if date.year == 2018 else 5,
                    "sector": index // 2,
                    "y": float(index),
                    "lgbm": float(index) if date.year == 2018 else float(3 - index),
                }
            )
    predictions = pd.DataFrame(rows)
    predictions.loc[(predictions.date.dt.year == 2019) & (predictions.ticker == "D"), "y"] = np.nan
    prior = predictions.loc[predictions.date.dt.year == 2019, ["date", "ticker", "lgbm"]].copy()
    prior["lgbm"] *= -1
    result = period_evidence(predictions, ["lgbm"], {"2019": ["2019-01-01", "2019-12-31"]}, [2018, 2019], prior)
    common = result["fixed_cohort"]
    assert common["tickers"] == ["A", "B", "C"] and common["n_dates"] == 4
    assert common["years"]["2018"]["lgbm"]["mean_ic"] == pytest.approx(1)
    assert common["years"]["2019"]["lgbm"]["mean_ic"] == pytest.approx(-1)
    assert result["earlier_model"]["lgbm"]["mean_ic"] == pytest.approx(1)
    assert result["periods"]["2019"]["lgbm"]["mean_ic"] == pytest.approx(-1)
    assert result["year_population"]["2019"]["label_rows"] == 6
    assert predictions.y.isna().sum() == 2
    print(
        "SANITY: a fixed diagnostic cohort preserves all compared dates; an earlier fitted model isolates refit sensitivity without replacing original predictions."
    )


def test_quality_distributions_preserve_missing_support_and_unordered_categories() -> None:
    frame = pd.DataFrame({"date": pd.to_datetime(["2019-01-02"] * 3), "sector": [0, 0, 0], "x": [1.0, 3.0, np.nan]})
    step = object.__new__(StepAssessLongShort)
    quality = step._coverage(frame, ["x", "sector"], ["sector"]).set_index("feature")
    assert quality.loc["x", "rows"] == 3 and quality.loc["x", "observed"] == 2
    assert quality.loc["x", "mean"] == 2 and quality.loc["x", "median"] == 2
    assert quality.loc["x", "std"] == pytest.approx(np.sqrt(2))
    assert pd.isna(quality.loc["sector", "mean"])
    assert quality.loc["sector", "observed"] == 3
    print(
        "SANITY: yearly sector distributions retain missing denominators and exact numeric moments; unordered category codes have no numeric trend statistics."
    )
