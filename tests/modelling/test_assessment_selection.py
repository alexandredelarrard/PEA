"""Known-truth search stopping, baseline retention and isolation behavior."""

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from src.context import Context
from src.modelling.steps.step_assess_long_short import StepAssessLongShort
from src.modelling.utils.assessment import write_json


@pytest.mark.parametrize("case", ["baseline", "tie", "cap"])
def test_search_retains_best_and_obeys_stops(case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    step = object.__new__(StepAssessLongShort)
    step._output = tmp_path
    step._options = OmegaConf.create({"max_rounds": 10, "patience": 2, "prune_fraction": 0.2})
    step._log = logging.getLogger(__name__)
    features = [f"x{i:03}" for i in range(100)]
    calls = []
    objectives = [0.1, 0.09, 0.08] if case == "baseline" else [0.2, 0.21, 0.21, 0.21] if case == "tie" else [0.1 + i / 100 for i in range(11)]

    def candidate(identifier: str, family: str, names: list[str], cats: list[str], panel: pd.DataFrame, expected: pd.DataFrame) -> dict[str, Any]:
        objective = objectives[len(calls)]
        calls.append(identifier)
        write_json(tmp_path / "_out" / identifier / "importance.json", {name: i + 1 for i, name in enumerate(names)})
        return {
            "id": identifier,
            "family": family,
            "features": names,
            "categoricals": [],
            "metrics": {"objective": objective, "ic_ir": objective, "mean_ic": 0.01, "eligible": True},
        }

    monkeypatch.setattr(step, "_candidate", candidate)
    baseline = {
        "metrics": {"lgbm": {"objective": 0.5 if case == "baseline" else 0.01, "eligible": True}},
        "features": {"lgbm": features[:2]},
        "categoricals": {"lgbm": []},
    }
    result = step._search("lgbm", pd.DataFrame(), pd.DataFrame(), features, [], baseline)
    assert len(calls) == {"baseline": 3, "tie": 4, "cap": 11}[case]
    assert result["best"]["id"] == {"baseline": "b0", "tie": "l3", "cap": "l10"}[case]
    assert result["non_improving_rounds"] == (0 if case == "cap" else 2)
    print(f"SANITY: {case}: best measured pool retained, deterministic smaller tie, two-failure stop or ten-pruning-round cap honored.")


def test_run_restores_original_context_objects_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = {"ROOT": tmp_path, "OUTPUT_DIR": tmp_path / "active"}
    store = object()
    context = SimpleNamespace(paths=paths, store=store, config_dir=tmp_path)
    config = cast(
        DictConfig,
        OmegaConf.create(
            {
                "model": {
                    "target_type": "zscore",
                    "assessment": {"horizon": 60, "output_dir": str(tmp_path / "reports" / "validate" / "run"), "stage": "development"},
                }
            }
        ),
    )
    step = StepAssessLongShort(cast(Context, context), config)

    def fail() -> None:
        assert context.paths is not paths
        with pytest.raises(RuntimeError, match="refused"):
            _ = context.store.upsert
        raise ValueError("known candidate failure")

    monkeypatch.setattr(step, "_development", fail)
    with pytest.raises(ValueError, match="known candidate failure"):
        step.run()
    assert context.paths is paths and context.store is store
    assert paths["OUTPUT_DIR"] == tmp_path / "active"
    print("SANITY: failure restores the original path/store objects and refuses application mutation before it can occur.")
