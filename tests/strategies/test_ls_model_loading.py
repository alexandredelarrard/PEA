"""`ls_model._load_models` (src/strategies/utils/ls_model.py) loads only the horizons the current
`metadata.json` was trained on: a member file left by an older run for another configured horizon
is never unpickled."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.modelling.utils.artifacts import member_inventory, member_path, publish_generation, save_member, write_metadata
from src.modelling.utils.metrics import per_day_zscore
from src.strategies.utils import ls_model
from src.strategies.utils.ls_model import _load_models


class _Member:
    def __init__(self, tag: str) -> None:
        self.tag = tag


def test_only_metadata_horizons_are_loaded(tmp_path: Path) -> None:
    for h in (30, 60):
        save_member(_Member(f"enet{h}"), member_path(tmp_path, h, "elasticnet"))
    write_metadata(tmp_path, {"horizons": [30, 60], "model_types": ["elasticnet"], "target_type": "rank"})
    # an older run's h90 member (here: bytes that are not even a pickle -- loading them would raise)
    member_path(tmp_path, 90, "elasticnet").write_bytes(b"stale, unreadable member of a previous run")

    config = OmegaConf.create({"model": {"models_dir": str(tmp_path)}, "build_cube": {"targets": {"horizons": [30, 60, 90]}}, "strategy_ls": {}})
    context = SimpleNamespace(paths={"DATA_STORE": tmp_path, "MODELS_DIR": tmp_path})
    meta, models, target_type, horizons = _load_models(context, config)  # type: ignore[arg-type]

    assert horizons == [30, 60] and list(models) == [30, 60]
    assert models[30]["elasticnet"].tag == "enet30" and target_type == "rank"
    print("\n=== SANITY CHECK: ls_model loads the trained horizons only ===")
    print(
        f"  config horizons [30, 60, 90], metadata horizons {meta['horizons']} -> loaded {list(models)}; the stale h90 file is never read. Validated."
    )


def test_strategy_loads_the_pinned_generation_instead_of_legacy_root_files(tmp_path: Path) -> None:
    snapshot, production = tmp_path / "snapshot", tmp_path / "production"
    save_member(_Member("current60"), member_path(snapshot, 60, "elasticnet"))
    write_metadata(
        snapshot,
        {
            "horizons": [60],
            "model_types": ["elasticnet"],
            "target_type": "zscore",
            "run_id": "20261018_test",
            "full_history": True,
            "member_files": member_inventory(snapshot, [60], ["elasticnet"]),
        },
    )
    publish_generation(production, snapshot, "20261018_test", "a" * 32)
    save_member(_Member("stale-root"), member_path(production, 60, "elasticnet"))
    config = OmegaConf.create({"model": {"models_dir": str(production)}, "build_cube": {"targets": {"horizons": [60, 90]}}, "strategy_ls": {}})
    context = SimpleNamespace(paths={"DATA_STORE": tmp_path, "MODELS_DIR": production})
    metadata, models, target_type, horizons = _load_models(context, config)  # type: ignore[arg-type]
    assert metadata["member_directory"] and horizons == [60] and target_type == "zscore"
    assert models[60]["elasticnet"].tag == "current60"
    print("SANITY: the real sleeve loader pins generation metadata and ignores a stale flat-root member at the same horizon.")


def test_signal_scores_all_features_before_any_future_target_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    cube = pd.DataFrame(
        {"date": np.repeat(pd.to_datetime(["2022-01-03", "2022-01-04"]), 6), "ticker": list("ABCDEF") * 2, "x": [0, 1, 2, 3, 100, np.nan] * 2}
    )
    projections = []

    def load(table: Any, *, columns: list[str], **kwargs: Any) -> pd.DataFrame:
        projections.append(columns)
        return cube[columns].copy()

    class Member:
        def predict(self, panel: pd.DataFrame) -> pd.Series:
            return panel.x

    context: Any = SimpleNamespace(store=SimpleNamespace(columns=lambda table: list(cube), load=load))
    config = OmegaConf.create({"build_cube": {}, "strategy_ls": {"blend": "equal"}})
    meta = {"train_end": "2021-12-31", "feature_cols": ["x"], "train_ic_ir": {"60": 1.0}}
    monkeypatch.setattr(ls_model, "_load_models", lambda *args: (meta, {60: {"elasticnet": Member()}}, "zscore", [60]))
    returns = cube.assign(value=0).pivot(index="date", columns="ticker", values="value")
    monkeypatch.setattr(ls_model, "_returns", lambda *args: (returns + 100, returns, returns.A))
    first = ls_model.build_signal(context, config).signal
    expected = (
        cube[["date", "ticker"]]
        .assign(score=per_day_zscore(cube.x.to_numpy(), cube.date.to_numpy()))
        .pivot(index="date", columns="ticker", values="score")
    )
    np.testing.assert_allclose(first, expected, atol=1e-12, equal_nan=True)
    cube["target_zscore_h60"] = [0, 1, 2, 3, np.nan, 5] * 2
    second = ls_model.build_signal(context, config).signal
    pd.testing.assert_frame_equal(first, second)
    assert all(columns == ["date", "ticker", "x"] for columns in projections)
    assert first.E.notna().all() and first.F.isna().all()
    print(
        "SANITY: backtest normalizes all finite predictions, including missing-target titles; changing future target availability cannot change rankings."
    )
