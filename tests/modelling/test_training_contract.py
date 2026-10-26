"""Known-truth checks for fold inputs, score population and final iteration selection."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.modelling.steps.step_long_short import StepLongShort
from src.modelling.utils.metrics import daily_ic
from tests.modelling.model_fixtures import make_config


@pytest.mark.parametrize("counts, expected", [([1, 2, 5, 10, 13], 12), ([2, 10, 13], 12)])
def test_final_rounds_use_last_two_folds(counts: list[int], expected: int) -> None:
    step: Any = object.__new__(StepLongShort)
    step.families = ["lgbm"]
    captured: list[int] = []
    step._fit_members = lambda h, panel, valid, rounds: captured.append(rounds)
    step._fit_final(60, pd.DataFrame(), counts)
    assert captured == [expected]
    print("SANITY: final rounds use the last two chronological folds, rounding halves upward, for three or five folds.")


def test_fold_uses_all_training_rows_and_external_test() -> None:
    step: Any = object.__new__(StepLongShort)
    step._config = make_config(model={"cv": {"n_splits": 3, "embargo": 1}})
    step.families = ["lgbm"]
    step.label_column = "y"
    step._run_path = None
    step._context = None
    step._log = logging.getLogger("training-contract")
    step._calendar = pd.bdate_range("2020-01-01", periods=48)
    panel = pd.DataFrame([(date, str(k), k, k) for date in step._calendar for k in range(5)], columns=["date", "ticker", "x", "y"])
    panel["label_end"] = panel.date + pd.offsets.BDay(1)
    panel.loc[panel.ticker == "4", "y"] = np.nan
    fits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    scored_rows: list[int] = []

    class Member:
        model: Any = None

        def __init__(self, *args: Any) -> None:
            self.model = self
            self.best_iteration = 7
            self.training_iterations = 8

        def current_iteration(self) -> int:
            return 7

        def fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> Member:
            fits.append((train.copy(), valid.copy()))
            return self

        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            scored_rows.append(len(frame))
            return frame.x.to_numpy()

    step._classes = {"lgbm": Member}
    result = step._cross_validate(1, panel)
    assert len(fits) == 3
    for (train, valid), audit, rows in zip(fits, result.folds, scored_rows, strict=True):
        expected = panel.loc[(panel.date < valid.date.min() - pd.offsets.BDay(1)) & panel.y.notna()]
        pd.testing.assert_frame_equal(train, expected)
        assert train.label_end.max() < valid.date.min()
        assert rows > len(valid)
        assert audit["train_start"] == str(train.date.min().date())
        assert audit["test_scoring_rows"] == rows
        assert audit["test_paired_rows"] == len(valid)
        assert audit["early_stopping_sample"] == "external_test"
        assert "inner_train_end" not in audit
    print("SANITY: all admissible training rows fit; only labelled external-test rows select RMSE; missing-target titles still get scores.")


def test_daily_ic_uses_only_finite_prediction_outcome_pairs() -> None:
    panel = pd.DataFrame({"date": [pd.Timestamp("2020-01-02")] * 6, "y": [1, 2, 3, 4, np.nan, 6]})
    prediction = pd.Series([4, 3, 2, 1, 100, np.inf])
    result = daily_ic(panel, prediction, horizon=60)
    assert result["mean_ic"] == pytest.approx(-1)
    assert result["n_days"] == 1
    print("SANITY: finite prediction/outcome pairs alone enter IC; missing targets and infinite predictions cannot invalidate a supported day.")
