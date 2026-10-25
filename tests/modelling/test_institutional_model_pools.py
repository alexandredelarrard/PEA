"""Institutional input configuration and production schema preparation, without fitting."""

from __future__ import annotations

import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.data_store.schema import Tables
from src.modelling.steps.step_long_short import StepLongShort, union_all_columns
from src.utils.config import read_config


def test_institutional_29_names_selected_without_fitting(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    names = [
        "f_ic_insider_buy_value_mcap_60d",
        "f_ic_insider_buy_value_mcap_180d",
        "f_ic_insider_distinct_buyers_120d",
        "f_ic_insider_cluster_buy_120d",
        "f_ic_insider_ceo_buy_mcap_180d",
        "f_ic_insider_cfo_buy_mcap_180d",
        "f_ic_insider_director_buy_mcap_180d",
        "f_ic_insider_purchase_pct_prior",
        "f_ic_insider_owner_surprise_120d",
        "f_ic_insider_net_buy_ratio_180d",
        "f_ic_insider_discretionary_sell_mcap_60d",
        "f_ic_insider_planned_sell_mcap_60d",
        "f_ic_insider_management_position_change_60d",
        "f_ic_shortvol_ratio_5d",
        "f_ic_shortvol_ratio_20d",
        "f_ic_shortvol_ratio_60d",
        "f_ic_shortvol_acceleration",
        "f_ic_shortvol_turnover_20d",
        "f_ic_shortvol_high_x_weak_price",
        "f_ic_shortvol_high_x_strong_price",
        "f_ic_shortvol_market_coverage",
        "f_ic_ftd_to_adv20",
        "f_ic_ftd_persistence_30d",
        "f_ic_act_initial_13d",
        "f_ic_act_amendment_intensity",
        "f_ic_act_repeat_activist",
        "f_ic_bo_new_holder",
        "f_ic_bo_escalation_13g_to_13d",
        "f_ic_bo_de_escalation_13d_to_13g",
    ]
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    config = read_config(str(config_dir))
    for key in ("linear", "lgbm", "random_forest"):
        columns = list(config[key].columns)
        assert len(columns) == len(set(columns))
        assert [column for column in columns if column.startswith("f_ic_")] == names

    dates = pd.bdate_range("2021-12-28", periods=4)
    cube = pd.DataFrame(1.0, index=range(len(dates)), columns=union_all_columns(config))
    cube[names] = np.nan  # Schema selection must retain explicitly present, all-null inputs.
    cube["date"], cube["ticker"] = dates, "F010"
    for horizon in (30, 60, 90):
        cube[f"target_zscore_h{horizon}"] = 0.0
    sqlite_store.replace(Tables.cube, cube)
    sqlite_store.replace(Tables.cube_part_prices, cube[["date", "ticker"]])
    context: Any = SimpleNamespace(store=sqlite_store, config_dir=config_dir, paths={"ROOT": tmp_path})
    step = StepLongShort(context=context, config=config)
    trips: list[str] = []

    def forbidden(*args: Any, **kwargs: Any) -> None:
        trips.append("forbidden side effect")
        pytest.fail("Schema preparation must not fit, read feature payloads, connect, or write")

    for name in ("run_train", "_open_run", "_fit_members", "_cross_validate", "_fit_final"):
        monkeypatch.setattr(StepLongShort, name, forbidden)
    for path in (
        "src.modelling.transformers.base.BaseModel.fit",
        "src.modelling.transformers.lightgbm_model.LightGBMModel.fit_fixed_rounds",
        "src.modelling.transformers.lightgbm_model.train_booster",
        "src.modelling.steps.step_long_short.save_member",
        "src.modelling.steps.step_long_short.write_metadata",
        "src.modelling.steps.step_long_short.publish_generation",
        "src.modelling.steps.step_long_short.write_json",
        "src.context.get_config_context",
    ):
        monkeypatch.setattr(path, forbidden)
    for name in ("save", "bulk_seed", "append_tail", "delete", "replace", "ensure_columns", "drop", "load", "iter_load"):
        monkeypatch.setattr(sqlite_store, name, forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)

    schema = step._resolve_schema()
    prepared, prior = step._prepare_training()
    assert prepared == schema and prior is None and schema.horizons == [30, 60, 90]
    for family, key in (("linear", "linear"), ("lgbm", "lgbm"), ("rf", "random_forest")):
        assert schema.family_cols[family] == list(config[key].columns)
        for horizon in schema.horizons:
            assert schema.family_cols_by_h[family][horizon] == list(config[key].columns)
    assert set(names) <= set(schema.load_cols) and not any(column.startswith("target_") for column in schema.load_cols)
    assert cube[names].isna().all().all() and not trips
    assert step._training_as_of == pd.Timestamp("2021-12-31")
    assert not list(tmp_path.iterdir())
    counts = {family: len(columns) for family, columns in schema.family_cols.items()}
    print(f"SANITY: all29 institutional inputs selected once at h30/h60/h90; pools={counts}; all29 stay null; zero fit/write/network trips.")
