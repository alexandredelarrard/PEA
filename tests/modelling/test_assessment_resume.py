"""Known-truth frozen development resume and artifact provenance checks."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.modelling.steps.step_assess_long_short import StepAssessLongShort
from src.modelling.utils.artifacts import member_path, save_member
from src.modelling.utils.assessment import frame_digest, write_json


@pytest.fixture
def completed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[StepAssessLongShort, dict[str, Any], pd.DataFrame, list[dict]]:
    step = object.__new__(StepAssessLongShort)
    step._output = tmp_path
    step._config = OmegaConf.create({"seed": 7, "model": {"label_column": "y"}})
    step._options = OmegaConf.create({"resume": True, "test_windows": [["2000-01-01", "2000-12-31"]]})
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2000-01-03", "2000-01-04"]),
            "ticker": ["A", "B"],
            "x": [1.0, 2.0],
            "sector": [0, 1],
            "y": [0.1, -0.1],
            "label_end": pd.to_datetime(["2000-03-28", "2000-03-29"]),
        }
    )
    columns = list(panel.columns)
    selected = {
        family: {
            "id": identifier,
            "family": family,
            "features": ["x", "sector"] if family != "elasticnet" else ["x"],
            "categoricals": ["sector"] if family != "elasticnet" else [],
        }
        for family, identifier in [("lgbm", "l1"), ("random_forest", "r1"), ("elasticnet", "e1")]
    }
    ledger = {
        "input_digest": frame_digest(panel, ["x", "sector", "y", "label_end"]),
        "final_input_digest": frame_digest(panel, ["x", "sector", "y", "label_end"]),
        "features": ["x", "sector"],
        "categoricals": ["sector"],
        "families": {family: {"best": copy.deepcopy(candidate)} for family, candidate in selected.items()},
        "selected": selected,
        "frozen": True,
        "frozen_at": "2026-10-17 09:00:00",
        "fixed_parameters": {},
        "controls": {},
    }
    output = tmp_path / "_out"
    write_json(output / "selection.json", ledger)
    write_json(output / "eligibility.json", {"features": [{"feature": "x", "role": "numeric"}, {"feature": "sector", "role": "categorical"}]})
    write_json(
        output / "b0" / "result.json",
        {"source_hashes": {}, "seed": 7, "audit": {"input_digest_columns": columns, "input_digest": frame_digest(panel, columns)}},
    )
    final = output / "final"
    for family, candidate in selected.items():
        save_member(
            SimpleNamespace(family=family, horizon=60, features=candidate["features"], categoricals=candidate["categoricals"]),
            member_path(final / "m", 60, family),
        )
    write_json(
        final / "result.json",
        {
            "input_digest": ledger["final_input_digest"],
            "selection_frozen_at": ledger["frozen_at"],
            "artifacts": {str(path.relative_to(final)): hashlib.sha256(path.read_bytes()).hexdigest() for path in final.rglob("*.pkl")},
        },
    )
    config_writes: list[dict] = []
    monkeypatch.setattr(step, "_fit_source_identity", lambda: {})
    monkeypatch.setattr(step, "_fixed_parameters", lambda: {})
    monkeypatch.setattr(step, "_controls", lambda: {})
    monkeypatch.setattr(step, "_check_baseline_controls", lambda _: None)
    monkeypatch.setattr(step, "_development_panel", lambda *_: (SimpleNamespace(label_column="y"), panel))
    monkeypatch.setattr(step, "_write_selected_config", lambda values: config_writes.append(copy.deepcopy(values)))
    monkeypatch.setattr(step, "_search", lambda *_: pytest.fail("Frozen resume must not repeat selection"))
    monkeypatch.setattr(step, "_runtime_config", lambda *_: pytest.fail("Frozen resume must not refit"))
    monkeypatch.setattr(step, "_audit", lambda: pytest.fail("Frozen resume must not repeat the feature audit"))
    monkeypatch.setattr(step, "_baseline", lambda: pytest.fail("Frozen resume must not refit the baseline"))
    return step, ledger, panel, config_writes


def test_completed_development_resume_preserves_freeze(completed_run: tuple[StepAssessLongShort, dict[str, Any], pd.DataFrame, list[dict]]) -> None:
    step, ledger, _, config_writes = completed_run
    before = {path: path.read_bytes() for path in (step._output / "_out").rglob("*") if path.is_file()}
    step._development()
    step._development()
    assert json.loads((step._output / "_out" / "selection.json").read_text(encoding="utf-8"))["frozen_at"] == ledger["frozen_at"]
    assert {path: path.read_bytes() for path in (step._output / "_out").rglob("*") if path.is_file()} == before
    assert not config_writes
    print("SANITY: two completed resumes preserve frozen state and final provenance byte for byte, without selection, config writes or fitting.")


@pytest.mark.parametrize(
    "changed",
    [
        "selected_pool",
        "both_pools",
        "categorical_outside_pool",
        "categoricals",
        "input",
        "final_input",
        "freeze",
        "artifact",
        "missing_artifact",
        "missing_final",
        "missing_family",
        "missing_audit",
        "missing_baseline",
    ],
)
def test_frozen_resume_rejects_changed_or_incomplete_state(
    changed: str, completed_run: tuple[StepAssessLongShort, dict[str, Any], pd.DataFrame, list[dict]]
) -> None:
    step, ledger, panel, config_writes = completed_run
    output = step._output / "_out"
    final_result = output / "final" / "result.json"
    if changed in {"selected_pool", "both_pools", "categorical_outside_pool"}:
        features = ["sector"] if changed == "both_pools" else ["x"]
        ledger["selected"]["lgbm"]["features"] = features
        if changed != "selected_pool":
            ledger["families"]["lgbm"]["best"]["features"] = features
    elif changed == "categoricals":
        ledger["selected"]["lgbm"]["categoricals"] = []
        ledger["families"]["lgbm"]["best"]["categoricals"] = []
    elif changed == "input":
        panel.loc[0, "x"] = 3.0
    elif changed == "final_input":
        ledger["final_input_digest"] = "changed"
    elif changed == "freeze":
        result = json.loads(final_result.read_text(encoding="utf-8"))
        result["selection_frozen_at"] = "changed"
        write_json(final_result, result)
    elif changed in {"artifact", "missing_artifact"}:
        artifact = member_path(output / "final" / "m", 60, "lgbm")
        if changed == "artifact":
            artifact.write_bytes(b"changed")
        else:
            artifact.unlink()
    elif changed == "missing_final":
        final_result.unlink()
    elif changed == "missing_family":
        del ledger["families"]["lgbm"]
    elif changed == "missing_audit":
        (output / "eligibility.json").unlink()
    elif changed == "missing_baseline":
        (output / "b0" / "result.json").unlink()
    write_json(output / "selection.json", ledger)
    before = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="identity|artifact|Frozen|inputs differ"):
        step._development()
    assert {path: path.read_bytes() for path in output.rglob("*") if path.is_file()} == before
    assert not config_writes
    print(f"SANITY: frozen resume refuses {changed} before changing persisted state or fitting.")
