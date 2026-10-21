"""Selection by successful training completion, with explicit-run refusal and no estimator fits."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.modelling.utils.assessment import write_json
from src.utils.modelling_runs import read_run, select_run
from tests.modelling.model_fixtures import completed_run


def test_latest_uses_successful_completion_not_name_or_assessment_timestamp(tmp_path: Path) -> None:
    older, config = completed_run(tmp_path, "z_older", "2026-10-20T10:00:00+00:00")
    newer, _ = completed_run(tmp_path, "a_newer", "2026-10-21T10:00:00+00:00")
    failed, _ = completed_run(tmp_path, "failed", "2026-10-21T12:00:00+00:00")
    state = json.loads((failed / "metadata.json").read_text())
    state["training_complete"] = False
    write_json(failed / "metadata.json", state)
    context = SimpleNamespace(paths={"DATA_STORE": tmp_path})
    assert select_run(context, config)[0] == newer
    write_json(older / "assessment/metadata.json", {"run_id": older.name, "complete": True})
    (older / "report.html").write_text("previous complete report", encoding="utf-8")
    assert select_run(context, config, assessed=True)[0] == older
    assert select_run(context, config, older.name)[0] == older
    with pytest.raises(ValueError, match="incomplete"):
        select_run(context, config, failed.name)
    with pytest.raises(FileNotFoundError):
        select_run(context, config, "absent")
    print("SANITY: newest successful training wins by completion time; assessed selection and explicit pinning cannot fall back to another run.")


@pytest.mark.parametrize("run_id", ["../escape", "nested/run", "CON", "", "run."])
def test_explicit_unsafe_run_is_rejected(tmp_path: Path, run_id: str) -> None:
    _, config = completed_run(tmp_path, "complete", "2026-10-21T10:00:00+00:00")
    with pytest.raises(ValueError):
        select_run(SimpleNamespace(paths={"DATA_STORE": tmp_path}), config, run_id)
    print("SANITY: unsafe explicit identities are refused before any alternate-run lookup.")


def test_declared_five_fold_multihorizon_inventory_without_fits(tmp_path: Path) -> None:
    run, _ = completed_run(tmp_path, "normal", "2026-10-21T10:00:00+00:00", folds=5, horizons=[30, 60, 90])
    state = read_run(run)
    assert state["horizons"] == [30, 60, 90]
    assert all(folds == [1, 2, 3, 4, 5] for folds in state["folds"].values())
    state["folds"]["90"] = [1, 2, 3]
    write_json(run / "metadata.json", state)
    with pytest.raises(ValueError, match="fold inventory"):
        read_run(run)
    print("SANITY: normal five-fold/multihorizon inventory is supported without fitting, and a missing horizon's folds cannot pass readiness.")
