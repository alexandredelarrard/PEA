"""Scoring tests verify the exact mathematical blend, never economic performance."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.portfolio.transformers import signals
from src.portfolio.transformers.signals import score_panel
from src.portfolio.utils.calendar import due_flags
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
    assert due_flags(pd.DatetimeIndex(["2022-01-14", "2022-01-18"]), "weekly")[1]
    assert not due_flags(pd.DatetimeIndex(["2022-01-18", "2022-01-19"]), "weekly")[1]
    assert due_flags(pd.DatetimeIndex(["2022-01-31", "2022-02-01"]), "monthly")[1]
    config = OmegaConf.load("configs/portfolio.yml")
    config.portfolio.start = "2022-01-03"
    Settings.from_config(config)
    config.portfolio.allocation.cash = -0.1
    with pytest.raises(ValueError, match="allocations"):
        Settings.from_config(config)
    print("SANITY: calendar weeks/months survive holidays; invalid allocations fail instead of being silently normalized.")


@pytest.mark.parametrize(
    "location,key",
    [
        ("metadata", "training_as_of"),
        ("metadata", "selection_as_of"),
        ("metadata", "train_end"),
        ("fit", "feature_max"),
        ("fit", "label_end_max"),
        ("fit", "training_as_of"),
        ("fold", "train_label_end_max"),
        ("fold", "test_label_end_max"),
    ],
)
@pytest.mark.parametrize("invalid_date", ["2022-01-01", None])
def test_invalid_training_outcomes_fail_before_model_consumption(
    location: str, key: str, invalid_date: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = {
        "run_id": "exact",
        "horizons": [60],
        "model_types": ["linear"],
        "feature_cols": ["x"],
        "categorical_cols": [],
        "target_type": "zscore",
        "label_column": "y",
        "training_as_of": "2021-12-31",
        "folds": {"60": [1]},
    }
    audit = {"feature_max": "2021-12-01", "label_end_max": "2021-12-31", "training_as_of": "2021-12-31"}
    fold = {
        "train_end": "2017-01-01",
        "train_label_end_max": "2017-06-01",
        "test_start": "2018-01-01",
        "test_end": "2018-06-01",
        "test_label_end_max": "2018-12-01",
    }
    meta = {
        **deepcopy(state),
        "role": "cutoff",
        "full_history": False,
        "member_files": {"model": {}},
        "train_end": "2021-12-31",
        "selection_as_of": "2021-12-31",
        "fit_audit": {"60": audit},
        "selection": {"60": {"folds": [fold]}},
    }
    monkeypatch.setattr(signals, "select_run", lambda *args, **kwargs: (Path("exact"), state))
    monkeypatch.setattr(signals, "read_metadata", lambda _: meta)
    config = OmegaConf.create({"portfolio": {"start": "2022-01-03"}, "train": {"end_date": "2021-12-31"}})
    signals.select_cutoff(SimpleNamespace(), config, "exact")
    {"metadata": meta, "fit": audit, "fold": fold}[location][key] = invalid_date
    with pytest.raises(ValueError, match="later than"):
        signals.select_cutoff(SimpleNamespace(), config, "exact")
    print(f"SANITY: invalid {location}.{key}={invalid_date!r} is rejected before any model member can be loaded.")
