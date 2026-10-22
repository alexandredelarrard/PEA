"""Exact quotas and invariance of decisions to future observations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.transformers.construction import construct, risk_panels
from src.portfolio.utils.contracts import IncompleteScenario
from tests.portfolio.test_simulation import market, settings


def test_sector_quota_neutrality_and_future_invariance() -> None:
    data, options = market(), settings()
    variance, momentum = risk_panels(data, options)
    day = pd.Timestamp("2022-01-03")
    weights = construct(data, day, 2, variance, momentum)
    assert (weights > 0).sum() == (weights < 0).sum() == 2
    assert weights.groupby(data.sectors).sum().abs().max() < 1e-12
    assert weights.abs().sum() == pytest.approx(1)
    data.returns.loc[data.returns.index > day] *= 100
    data.scores.loc[data.scores.index > day] *= -1
    changed_variance, changed_momentum = risk_panels(data, options)
    pd.testing.assert_series_equal(weights, construct(data, day, 2, changed_variance, changed_momentum))
    with pytest.raises(IncompleteScenario, match="capacity"):
        construct(data, day, 10, variance, momentum)
    print("SANITY: per-sector dollars cancel, N is total per side, missing capacity refuses; future changes cannot alter past decisions.")


def test_tilts_are_bounded_and_inverse_variance_is_within_each_leg() -> None:
    data = market()
    date = pd.Timestamp("2022-01-03")
    variance, momentum = risk_panels(data, settings())
    momentum.loc[date, ["Energy", "Tech"]] = [-0.2, -0.1]
    weights = construct(data, date, 2, variance, momentum)
    sector_gross = weights.abs().groupby(data.sectors).sum()
    assert sector_gross.Tech == pytest.approx(2 * sector_gross.Energy)
    assert np.isclose(weights.sum(), 0)
    print("SANITY: less negative sector momentum receives twice the gross budget; the positive tilt remains bounded.")
