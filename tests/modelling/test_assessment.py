"""Known-truth assessment arithmetic: fixed populations, pruning and date-balanced samples."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modelling.utils.assessment import (
    balanced_sample_positions,
    frame_digest,
    summarize_ic,
)


def test_pooled_ir_uses_all_dates_instead_of_mean_fold_ir() -> None:
    values = pd.Series([0.1, 0.2, -0.2, 0.1, 0.3])
    result = summarize_ic(values, 60)
    expected = values.mean() / values.to_numpy().std() * np.sqrt(252 / 60)
    assert result["ic_ir"] == pytest.approx(expected)
    assert result["n_days"] == 5
    print("SANITY: pooled IC-IR uses the full daily series with population dispersion and h60 adjustment.")


def _population() -> pd.DataFrame:
    return pd.DataFrame(
        {"date": np.repeat(pd.bdate_range("2000-01-03", periods=3), 6), "ticker": list("ABCDEF") * 3, "y": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] * 3}
    )


def test_date_balanced_sample_is_reproducible_and_feature_independent() -> None:
    panel = pd.DataFrame({"date": np.repeat(pd.bdate_range("2000-01-03", periods=5), [10, 20, 40, 70, 100])})
    positions = balanced_sample_positions(panel, 17, 4325)
    counts = panel.iloc[positions].groupby("date").size()
    assert len(positions) == 17 and counts.max() - counts.min() <= 1
    assert np.array_equal(positions, balanced_sample_positions(panel.assign(unused=np.nan), 17, 4325))
    assert len(balanced_sample_positions(panel, 0, 4325)) == len(panel)
    print("SANITY: the same 17 row keys are sampled across models, balanced by date despite changing breadth.")


def test_input_identity_is_row_order_independent_and_value_sensitive() -> None:
    panel = _population()
    panel["a"] = np.arange(len(panel), dtype="float32")
    before = frame_digest(panel, ["a", "y"])
    assert frame_digest(panel.iloc[::-1], ["y", "a"]) == before
    panel.loc[0, "a"] = 1.0
    assert frame_digest(panel, ["a", "y"]) != before
    print("SANITY: normalized keyed input identity ignores row/column order and rejects changed values.")
