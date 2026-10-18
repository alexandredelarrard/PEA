"""Known-truth partial evaluation replay guards; no outcomes, database or fitting."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.context import Context
from src.data_store.schema import Tables
from src.modelling.steps import step_assess_long_short as assessment
from src.modelling.steps.step_assess_long_short import StepAssessLongShort
from src.modelling.utils.assessment import write_json


class _OutcomeBoundaryError(RuntimeError):
    pass


@pytest.fixture
def evaluation_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[StepAssessLongShort, pd.DataFrame, SimpleNamespace]:
    step = object.__new__(StepAssessLongShort)
    step._output = tmp_path
    step._config = OmegaConf.create({"seed": 7, "model": {"ensemble": ["lgbm"], "diagnostics": {"shap_sample": 4}}})
    step._options = OmegaConf.create(
        {
            "resume": False,
            "test_windows": [["2021-09-01", "2021-12-31"]],
            "evaluation_start": "2022-01-03",
            "comparison_years": [2021],
            "read_chunksize": 20,
        }
    )
    step._log = logging.getLogger(__name__)
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-09-01", "2021-09-01", "2022-01-03", "2022-01-03"]),
            "ticker": ["A", "B", "A", "B"],
            "sector": [0, 1, 0, 1],
            "x": [1.0, 2.0, 3.0, 4.0],
        }
    )
    state = SimpleNamespace(calendar=pd.bdate_range("2021-09-01", periods=160), reads=0, outcomes=0)

    def distinct(table: Any, column: str, **kwargs: Any) -> list[pd.Timestamp]:
        assert column == "date"
        assert table in {Tables.cube_part_prices, Tables.cube}
        return list(state.calendar) if table == Tables.cube_part_prices else [state.calendar[-1]]

    def features(store: Any, table: Any, **kwargs: Any) -> pd.DataFrame:
        assert table == Tables.cube
        assert kwargs["columns"] == ["date", "ticker", "sector", "x"]
        state.reads += 1
        return panel.loc[panel.date.between(kwargs["since"], kwargs["until"])].copy().reset_index(drop=True)

    def predict(members: dict, frame: pd.DataFrame) -> tuple[pd.Series, dict[str, pd.Series]]:
        return frame.x, {"lgbm": frame.x}

    def stop_before_outcomes(*args: Any) -> None:
        state.outcomes += 1
        raise _OutcomeBoundaryError("Fixture stops before reading outcomes")

    step._context = cast(Context, SimpleNamespace(store=SimpleNamespace(distinct=distinct)))
    monkeypatch.setattr(step, "_controls", lambda: {})
    monkeypatch.setattr(step, "_fixed_parameters", lambda: {})
    monkeypatch.setattr(step, "_attach_outcomes", stop_before_outcomes)
    monkeypatch.setattr(step, "_fit_selected", lambda *_: pytest.fail("Evaluation must never fit"))
    monkeypatch.setattr(step, "_search", lambda *_: pytest.fail("Evaluation must never select"))
    monkeypatch.setattr(assessment, "load_member", lambda _: SimpleNamespace())
    monkeypatch.setattr(assessment, "load_frame", features)
    monkeypatch.setattr(assessment, "ensemble_predict", predict)
    write_json(
        tmp_path / "_out" / "selection.json",
        {
            "frozen": True,
            "frozen_at": "2026-10-17 09:00:00",
            "controls": {},
            "fixed_parameters": {},
            "outcome_cutoff": "2021-12-31",
            "selected": {"lgbm": {"features": ["x"], "categoricals": []}},
        },
    )
    final = tmp_path / "_out" / "final"
    final.mkdir()
    (final / "frozen.pkl").write_bytes(b"known-truth-model-identity")
    write_json(
        final / "result.json",
        {
            "selection_frozen_at": "2026-10-17 09:00:00",
            "artifacts": {"frozen.pkl": hashlib.sha256((final / "frozen.pkl").read_bytes()).hexdigest()},
        },
    )
    return step, panel, state


def _snapshot(step: StepAssessLongShort) -> dict[str, bytes]:
    output = step._output / "_out" / "eval"
    return {str(path.relative_to(output)): path.read_bytes() for path in output.iterdir() if path.is_file()}


def test_partial_evaluation_requires_explicit_resume(evaluation_state: tuple[StepAssessLongShort, pd.DataFrame, SimpleNamespace]) -> None:
    step, _, state = evaluation_state
    with pytest.raises(_OutcomeBoundaryError):
        step._evaluation()
    before, reads = _snapshot(step), state.reads
    with pytest.raises(ValueError, match="explicit --resume"):
        step._evaluation()
    assert _snapshot(step) == before and state.reads == reads and state.outcomes == 1
    print("SANITY: interrupted evaluation requires explicit resume before any source reread or artifact overwrite.")


@pytest.mark.parametrize("committed", [False, True])
def test_partial_evaluation_retry_preserves_first_seal(
    committed: bool, evaluation_state: tuple[StepAssessLongShort, pd.DataFrame, SimpleNamespace], monkeypatch: pytest.MonkeyPatch
) -> None:
    step, _, state = evaluation_state
    if not committed:

        def stop_before_features(*args: Any) -> None:
            raise _OutcomeBoundaryError()

        with monkeypatch.context() as patch:
            patch.setattr(step, "_score_universes", stop_before_features)
            with pytest.raises(_OutcomeBoundaryError):
                step._evaluation()
    else:
        with pytest.raises(_OutcomeBoundaryError):
            step._evaluation()
    seal = step._output / "_out" / "eval" / "started.json"
    before, seal_time = _snapshot(step), seal.stat().st_mtime_ns
    step._options.resume = True
    with pytest.raises(_OutcomeBoundaryError):
        step._evaluation()
    after = _snapshot(step)
    assert all(after[name] == content for name, content in before.items())
    assert seal.stat().st_mtime_ns == seal_time
    assert state.outcomes == (2 if committed else 1)
    print(f"SANITY: unchanged explicit retry with committed={committed} preserves first seal bytes/timestamp and existing feature evidence.")


@pytest.mark.parametrize("changed", ["freeze", "models", "current_freeze", "current_models"])
def test_partial_evaluation_rejects_changed_seal_identity(
    changed: str, evaluation_state: tuple[StepAssessLongShort, pd.DataFrame, SimpleNamespace]
) -> None:
    step, _, state = evaluation_state
    with pytest.raises(_OutcomeBoundaryError):
        step._evaluation()
    output = step._output / "_out"
    path = output / "eval" / "started.json"
    seal = json.loads(path.read_text(encoding="utf-8"))
    if changed in {"freeze", "models"}:
        seal["frozen_at" if changed == "freeze" else "models"] = "changed"
        write_json(path, seal)
    elif changed == "current_freeze":
        for path, key in [(output / "selection.json", "frozen_at"), (output / "final" / "result.json", "selection_frozen_at")]:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload[key] = "changed"
            write_json(path, payload)
    else:
        model = output / "final" / "frozen.pkl"
        model.write_bytes(b"replacement-model")
        path = output / "final" / "result.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["artifacts"]["frozen.pkl"] = hashlib.sha256(model.read_bytes()).hexdigest()
        write_json(path, payload)
    before, reads = _snapshot(step), state.reads
    step._options.resume = True
    with pytest.raises(ValueError, match="identity mismatch"):
        step._evaluation()
    assert _snapshot(step) == before and state.reads == reads and state.outcomes == 1
    print(f"SANITY: changed {changed} identity refuses partial replay before source reads and preserves committed artifacts.")


@pytest.mark.parametrize("changed", ["calendar", "window", "feature", "later_feature", "ticker", "sector", "missing_calendar_digest"])
def test_partial_evaluation_rejects_changed_feature_snapshot(
    changed: str, evaluation_state: tuple[StepAssessLongShort, pd.DataFrame, SimpleNamespace]
) -> None:
    step, panel, state = evaluation_state
    with pytest.raises(_OutcomeBoundaryError):
        step._evaluation()
    if changed == "calendar":
        state.calendar = state.calendar.delete(10)
    elif changed == "window":
        step._options.evaluation_start = "2022-01-04"
    elif changed in {"feature", "later_feature"}:
        panel.loc[0 if changed == "feature" else 2, "x"] = 2.0 if changed == "feature" else 4.0
    elif changed in {"ticker", "sector"}:
        panel.loc[0, changed] = "C" if changed == "ticker" else 2
    else:
        path = step._output / "_out" / "eval" / "feature_prediction_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("calendar_digest", None)
        write_json(path, payload)
    before = _snapshot(step)
    step._options.resume = True
    with pytest.raises(ValueError, match="snapshot.*mismatch"):
        step._evaluation()
    assert _snapshot(step) == before and state.outcomes == 1
    print(f"SANITY: changed {changed} refuses replay before replacing that fold's samples/coverage or committed scores/cohorts.")


@pytest.mark.parametrize("resume", [False, True])
def test_completed_evaluation_resume_behavior_is_preserved(
    resume: bool, evaluation_state: tuple[StepAssessLongShort, pd.DataFrame, SimpleNamespace]
) -> None:
    step, _, state = evaluation_state
    write_json(step._output / "_out" / "eval" / "result.json", {"complete": True})
    before = _snapshot(step)
    step._options.resume = resume
    if resume:
        step._evaluation()
    else:
        with pytest.raises(ValueError, match="explicit --resume"):
            step._evaluation()
    assert _snapshot(step) == before and state.reads == 0 and state.outcomes == 0
    print(f"SANITY: completed evaluation with resume={resume} retains its existing gate without new scoring or outcome reads.")
