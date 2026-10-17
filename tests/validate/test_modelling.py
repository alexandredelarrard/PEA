"""Independent known-truth model assessment checks; no source database or trained model needed."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.context import Context
from src.validate.checks.modelling import check_modelling, independent_daily, independent_paths


def test_independent_rank_ic_skips_undefined_dates() -> None:
    dates = pd.to_datetime(["2022-01-03"] * 4 + ["2022-01-04"] * 4)
    rows = pd.DataFrame({"date": dates, "ticker": list("ABCD") * 2, "y": [4, 3, 2, 1] * 2, "lgbm": [1, 2, 3, 4, 5, 5, 5, 5]})
    got = independent_daily(rows, "lgbm")
    assert len(got) == 1 and got.iloc[0] == -1
    print("SANITY: independent SciPy rank IC gives -1 on a reverse ordering and excludes a constant-score date.")


def test_independent_return_path_requires_all_future_bars() -> None:
    calendar = pd.bdate_range("2022-01-03", periods=5)
    returns = np.array([0.0, 0.1, 0.2, -0.1, 0.05])
    prices = pd.DataFrame({"date": calendar, "ticker": "A", "ret": returns, "close_total": 100 * np.cumprod(1 + returns)})
    prices.loc[0, "ret"] = np.nan  # Today's return is not one of the three future bars.
    first = independent_paths(prices, calendar, 3).iloc[0]
    assert first.valid and abs(first.realized_return - 0.188) < 1e-12
    prices.loc[2, "ret"] = np.nan
    masked = independent_paths(prices, calendar, 3).iloc[0]
    assert not masked.valid and pd.isna(masked.realized_return)
    prices.loc[2, "ret"] = 0.5
    assert not independent_paths(prices, calendar, 3).iloc[0].valid
    print("SANITY: independent prefix replay matches the hand-computed 18.8% and rejects an interior hole or inconsistent seam.")


def test_incomplete_assessment_abstains_without_store_access(tmp_path: Path) -> None:
    class NoRead:
        def __getattr__(self, name: str) -> None:
            raise AssertionError(f"Unexpected store read {name}")

    context = cast(Context, SimpleNamespace(store=NoRead()))
    config = cast(DictConfig, OmegaConf.create({"model": {"ensemble": ["lgbm"]}}))
    result = check_modelling(context, out=tmp_path, config=config)
    assert result.status == "abstain" and result.scope["missing"]
    print("SANITY: incomplete modelling evidence abstains explicitly; it cannot produce a false pass or read the database.")
