"""Known-truth temporal and ex-ante cohort assessment contracts."""

import numpy as np
import pandas as pd
import pytest

from src.modelling.utils.assessment_metrics import formation_cohorts, forward_returns, matched_cohort_returns, trend_summary


def test_cohorts_are_fixed_before_future_outcomes() -> None:
    scores = pd.DataFrame({"date": [pd.Timestamp("2022-01-03")] * 11, "ticker": [f"T{i:02}" for i in range(11)], "lgbm": np.arange(11.0)})
    chosen = formation_cohorts(scores, ["lgbm"])
    assert set(chosen.ticker) == {"T09", "T10"}
    assert chosen.weight.tolist() == [0.5, 0.5]
    outcomes = scores[["date", "ticker"]].assign(realized_return=0.1, valid=True)
    outcomes.loc[outcomes.ticker == "T10", ["realized_return", "valid"]] = [np.nan, False]
    market = pd.DataFrame({"date": [scores.date.iloc[0]], "market_return": [0.02], "market_valid": [True]})
    result = matched_cohort_returns(chosen, outcomes, market)
    assert pd.isna(result.cohort_return.iloc[0])
    assert result.selected.iloc[0] == 2 and result.supported.iloc[0] == 1
    assert result.known_weight.iloc[0] == 0.5 and result.known_contribution.iloc[0] == pytest.approx(0.05)
    assert set(chosen.ticker) == {"T09", "T10"}
    print("SANITY: ceil(N/10) predictions define fixed equal weights; a missing selected outcome cannot replace or reweight a stock.")


def test_positive_endpoints_cannot_hide_an_interior_masked_return() -> None:
    calendar = pd.bdate_range("2022-01-03", periods=6)
    prices = pd.DataFrame({"date": calendar, "ticker": "A", "close_total": 100 * 1.01 ** np.arange(6), "ret": [0.01] * 6})
    valid = forward_returns(prices, calendar, 3)
    first = valid.iloc[0]
    assert first.valid and first.realized_return == pytest.approx(1.01**3 - 1)
    prices.loc[2, "ret"] = np.nan
    rejected = forward_returns(prices, calendar, 3).iloc[0]
    assert not rejected.valid and rejected.reason == "incomplete_path"
    assert pd.isna(rejected.realized_return)
    prices.loc[2, "ret"] = 0.20
    seam = forward_returns(prices, calendar, 3).iloc[0]
    assert not seam.valid and seam.reason == "endpoint_path_mismatch"
    print("SANITY: every interior bar is required; masked seams and price/return disagreement fail despite positive endpoints.")


def test_trend_assessment_distinguishes_flat_and_reversing_curves() -> None:
    assert trend_summary(np.arange(5), np.zeros(5))["shape"] == "flat"
    assert trend_summary(np.arange(5), np.arange(5))["shape"] == "increasing"
    assert trend_summary(np.arange(5), -np.arange(5))["shape"] == "decreasing"
    assert trend_summary(np.arange(5), [0, 2, 3, 2, 0])["shape"] == "nonmonotonic"
    print("SANITY: measured flat, directional and reversing PDP shapes have separate interpretations; direction alone is not causal evidence.")
