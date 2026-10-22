"""Scoring tests verify the exact mathematical blend, never economic performance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.portfolio.transformers.signals import score_panel
from src.portfolio.utils.calendar import is_due
from src.portfolio.utils.contracts import Settings


class Member:
    def __init__(self, column: str) -> None:
        self.column = column

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        return panel[self.column].to_numpy()


def test_training_horizon_weights_and_complete_daily_cross_sections() -> None:
    panel = pd.DataFrame(
        {
            "date": np.repeat(pd.to_datetime(["2022-01-03", "2022-01-04"]), 4),
            "ticker": list("ABCD") * 2,
            "x": [1, 2, 3, 4, 4, 3, 2, 1],
            "other": [4, 2, 1, 3, 1, 8, 4, 2],
            "future_label": 100,
        }
    )
    models = {10: {"a": Member("x")}, 60: {"b": Member("other")}}
    meta = {"train_ic_ir": {"10": 2, "60": -1}}
    full = score_panel(panel, models, meta)
    prefix = score_panel(panel.iloc[:4], models, meta)
    pd.testing.assert_frame_equal(full.iloc[:4], prefix)
    changed = panel.copy()
    changed.loc[4:, ["x", "other", "future_label"]] = 999
    pd.testing.assert_frame_equal(full.iloc[:4], score_panel(changed, models, meta).iloc[:4])
    assert full.iloc[:4].score.tolist() == pytest.approx(((panel.iloc[:4].x - panel.iloc[:4].x.mean()) / panel.iloc[:4].x.std()).tolist())
    print("SANITY: negative training IR receives zero weight; changing future labels or cross-sections leaves the past identical.")


def test_actual_calendar_and_invalid_allocations() -> None:
    assert is_due(pd.Timestamp("2022-01-14"), pd.Timestamp("2022-01-18"), "weekly")
    assert not is_due(pd.Timestamp("2022-01-18"), pd.Timestamp("2022-01-19"), "weekly")
    assert is_due(pd.Timestamp("2022-01-31"), pd.Timestamp("2022-02-01"), "monthly")
    config = OmegaConf.load("configs/portfolio.yml")
    Settings.from_config(config)
    config.portfolio.allocation.cash = -0.1
    with pytest.raises(ValueError, match="allocations"):
        Settings.from_config(config)
    print("SANITY: calendar weeks/months survive holidays; invalid allocations fail instead of being silently normalized.")
