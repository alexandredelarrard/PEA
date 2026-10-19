"""Tree family feature selection, shared types, and horizon-wide monotonic priors."""

from __future__ import annotations

import logging

import pytest

from src.modelling.transformers import model_class
from tests.modelling.model_fixtures import ctx, make_config, signal_panel


@pytest.mark.parametrize("family", ["lgbm", "random_forest"])
@pytest.mark.parametrize("horizon", [30, 60, 90])
def test_shared_category_types_and_monotonic_directions_reach_each_horizon(family: str, horizon: int, caplog: pytest.LogCaptureFixture) -> None:
    pools = {30: ["f0", "sector"], 60: ["f2"], 90: ["f0", "sector", "f2"]}
    config = make_config(**{family: {"columns_by_horizon": pools, "num_boost_round": 3}})
    member = model_class(family)(ctx(), config, family, horizon)
    with caplog.at_level(logging.WARNING):
        member.fit(signal_panel(n_days=10, n_tickers=20))
    assert member.categoricals == (["sector"] if horizon != 60 else [])
    assert member.features == [name for name in pools[horizon] if name != "sector"] + member.categoricals
    assert list(member.model.dump_model()["monotone_constraints"]) == [{"f0": 1, "f2": -1}.get(name, 0) for name in member.features]
    assert not [record for record in caplog.records if "typo" in record.getMessage()]
    print(f"SANITY: {family} h{horizon} uses only selected inputs, shared categorical types and correctly aligned monotonic constraints.")


def test_category_registry_does_not_select_inputs_or_use_rf_types() -> None:
    config = make_config(random_forest={"columns": ["f0", "sector"], "categoricals": ["f0"], "columns_by_horizon": {60: []}})
    family = model_class("random_forest")
    assert family.configured_categoricals(config, 30) == ["sector"]
    assert family.configured_categoricals(config, 60) == []
    assert model_class("lgbm").configured_categoricals(make_config(lgbm={"columns": ["f0"]}), 30) == []
    print("SANITY: lgbm.categoricals alone assigns types; unselected registry entries do not add model inputs.")
