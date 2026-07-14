"""Sanity checks for LightGBM monotone constraint wiring."""
from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.modelling.utils_model.model import (
    build_monotone_constraints,
    parse_monotone_feature_map,
    train_ranker,
)


def test_parse_monotone_feature_map_list_and_dict_formats():
    list_cfg = OmegaConf.create({
        "features": [
            {"f_employee_growth_xs": 1},
            {"f_sga_growth_xs": -1},
            {"f_fwd_eps_yield_xs": 1},
        ]
    })
    dict_cfg = OmegaConf.create({
        "features": {
            "f_employee_growth_xs": 1,
            "f_sga_growth_xs": -1,
            "f_fwd_eps_yield_xs": 1,
        }
    })

    expected = {
        "f_employee_growth_xs": 1,
        "f_sga_growth_xs": -1,
        "f_fwd_eps_yield_xs": 1,
    }
    assert parse_monotone_feature_map(list_cfg.features) == expected
    assert parse_monotone_feature_map(dict_cfg.features) == expected

    feats = ["mom_12_1", "f_employee_growth_xs", "f_sga_growth_xs", "f_fwd_eps_yield_xs"]
    constraints = build_monotone_constraints(feats, expected)
    assert constraints == [0, 1, -1, 1]

    print("\n=== SANITY CHECK: monotone constraint parsing ===")
    print(f"  feature order={feats}")
    print(f"  constraints  ={constraints}")
    print("  +1 on employee growth & fwd EPS yield, -1 on SG&A growth, 0 elsewhere.")


def test_train_ranker_passes_monotone_constraints_to_lightgbm():
    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    rows = []
    for d in dates:
        for i, ticker in enumerate(["AAA", "BBB", "CCC"]):
            rows.append({
                "date": d,
                "ticker": ticker,
                "f_employee_growth_xs": float(i),
                "f_sga_growth_xs": float(2 - i),
                "f_fwd_eps_yield_xs": float(i) / 10.0,
                "y": float(i) / 2.0,
            })
    panel = pd.DataFrame(rows)
    feats = ["f_employee_growth_xs", "f_sga_growth_xs", "f_fwd_eps_yield_xs"]
    constraints = [1, -1, 1]

    booster = train_ranker(
        panel,
        feats,
        params={"monotone_constraints": constraints},
        num_boost_round=5,
        valid_panel=None,
    )

    assert booster.params["monotone_constraints"] == constraints

    mono = booster.dump_model()["monotone_constraints"]
    assert mono == constraints

    print("\n=== SANITY CHECK: LightGBM monotone_constraints ===")
    print(f"  requested={constraints}")
    print(f"  booster.params={booster.params['monotone_constraints']}")
    print(f"  dump_model ={mono}")
    print("  Constraints flow through train_ranker into the fitted booster.")
