"""Refresh state and publication failure checks without fitting any estimator."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.modelling.steps import step_assess_long_short as assessment_module
from src.modelling.steps.step_assess_long_short import StepAssessLongShort
from src.modelling.utils import assessment as assessment_utils
from src.modelling.utils.assessment import assessment_workspace, publish_assessment
from tests.modelling.model_fixtures import completed_run


@pytest.mark.parametrize("age_hours, expected_warning", [(24, False), (24.01, True)])
def test_assessment_always_refreshes_and_accepts_current_config_and_calendar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, age_hours: float, expected_warning: bool
) -> None:
    now = datetime(2026, 10, 21, tzinfo=UTC)
    run, config = completed_run(tmp_path, "older", (now - timedelta(hours=age_hours)).isoformat())
    completed_run(tmp_path, "newer", now.isoformat())
    config.seed = 123
    config.lgbm.num_boost_round = 10
    config.model.horizons = [30, 90]
    config.model.diagnostics.pdp_grid = 31
    context: Any = SimpleNamespace(
        config=config, config_dir=Path("configs"), paths={"DATA_STORE": tmp_path}, save=True, log=logging.getLogger("fresh-assessment")
    )
    calendar = pd.bdate_range("2020-01-01", "2026-10-20").tolist()
    context.store = SimpleNamespace(
        columns=lambda table: ["date", "ticker", "x"],
        distinct=lambda table, *args, **kwargs: calendar[-1:] if kwargs.get("order") == "desc" else calendar,
    )

    class Clock(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:
            return now

    monkeypatch.setattr(assessment_module, "datetime", Clock)
    calls: list[int] = []

    def analyse(self: Any, horizon: int, training: dict, cutoff: dict, calendar: pd.DatetimeIndex, state: dict) -> None:
        calls.append(horizon)
        assert self._output == run
        assert calendar[-1] == pd.Timestamp("2026-10-20")
        assert not (self._analysis_path / "scores.txt").exists()
        (self._analysis_path / "scores.txt").write_text(str(len(calls)), encoding="utf-8")
        state["horizons"][str(horizon)] = {"effective_fold_test_end": "2021-10-06"}

    def render(directory: Path, *, assessment_directory: Path, report_path: Path) -> Path:
        report_path.write_text("new report " + (assessment_directory / "scores.txt").read_text(), encoding="utf-8")
        return report_path

    monkeypatch.setattr(StepAssessLongShort, "_assess_horizon", analyse)
    monkeypatch.setattr(assessment_module, "render_assessment", render)
    with caplog.at_level(logging.INFO):
        for _ in range(2):
            assert StepAssessLongShort(context, config, run_id="older").run() == run / "report.html"
    assert calls == [60, 60]
    assert (run / "report.html").read_text() == "new report 2"
    assert json.loads((run / "assessment/metadata.json").read_text())["diagnostics"]["pdp_grid"] == 31
    assert any(record.levelno == logging.WARNING for record in caplog.records) == expected_warning
    assert "older" in caplog.text and "2026-10" in caplog.text
    assert not (tmp_path / "newer/assessment").exists()
    print("SANITY: each explicitly selected assessment recalculates; fitting-config/calendar drift is allowed; age warning uses strict >24h.")


@pytest.mark.parametrize("during_publication", [False, True])
def test_failed_refresh_preserves_previous_analysis_and_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, during_publication: bool) -> None:
    run = tmp_path / "run"
    logger = logging.getLogger("fresh-assessment")
    (run / "assessment").mkdir(parents=True)
    (run / "assessment/scores.txt").write_text("previous", encoding="utf-8")
    (run / "report.html").write_text("previous report", encoding="utf-8")
    replace = Path.replace

    def fail_swap(self: Path, target: Any) -> Path:
        if self.name == "report.html" and self.parent.name.startswith(".assessment-") and "backup" not in self.parent.name:
            raise OSError("injected publication error")
        return replace(self, target)

    with pytest.raises(OSError, match="injected"):
        with assessment_workspace(run, logger) as workspace:
            (workspace / "scores.txt").write_text("new", encoding="utf-8")
            (workspace / "report.html").write_text("new report", encoding="utf-8")
            if not during_publication:
                raise OSError("injected computation error")
            monkeypatch.setattr(Path, "replace", fail_swap)
            publish_assessment(run, workspace, {"run_id": "run"}, logger)
    assert (run / "assessment/scores.txt").read_text() == "previous"
    assert (run / "report.html").read_text() == "previous report"
    assert not (run / ".assessment.lock").exists()
    print("SANITY: computation and publication failures preserve the previous complete report and scores.")


def test_same_run_refresh_is_serialized(tmp_path: Path) -> None:
    logger = logging.getLogger("fresh-assessment")
    with assessment_workspace(tmp_path, logger):
        with pytest.raises(FileExistsError):
            with assessment_workspace(tmp_path, logger):
                pytest.fail("Concurrent refresh must not enter the workspace")
    assert not (tmp_path / ".assessment.lock").exists()
    print("SANITY: simultaneous refreshes of the same run are refused before competing publication.")


def test_backup_cleanup_failure_preserves_successful_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    run = tmp_path / "run"
    logger = logging.getLogger("fresh-assessment")
    (run / "assessment").mkdir(parents=True)
    (run / "assessment/scores.txt").write_text("previous", encoding="utf-8")
    (run / "report.html").write_text("previous report", encoding="utf-8")
    original = assessment_utils.shutil.rmtree

    def failed_cleanup(path: Any) -> None:
        if Path(path).name.startswith(".assessment-backup-"):
            raise OSError("injected Windows directory cleanup failure")
        original(path)

    monkeypatch.setattr(assessment_utils.shutil, "rmtree", failed_cleanup)
    with assessment_workspace(run, logger) as workspace:
        (workspace / "report.html").write_text("new report", encoding="utf-8")
        (workspace / "scores.txt").write_text("new scores", encoding="utf-8")
        assert publish_assessment(run, workspace, {"run_id": "run"}, logger) == run / "report.html"
    assert (run / "report.html").read_text() == "new report"
    assert json.loads((run / "assessment/metadata.json").read_text())["complete"]
    assert not (run / ".assessment.lock").exists()
    assert "cleanup failed" in caplog.text
    assert any(record.name == logger.name and "cleanup failed" in record.message for record in caplog.records)
    print("SANITY: Windows backup-cleanup failure retains the completed new assessment, logs its retained scratch path and releases the lock.")


def test_transient_windows_directory_lock_is_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.replace
    logger = logging.getLogger("fresh-assessment")
    attempts = []

    def transient(self: Path, target: Any) -> Path:
        if self.is_dir() and Path(target).name == "assessment":
            attempts.append(self)
            if len(attempts) < 3:
                raise PermissionError("injected transient scanner lock")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", transient)
    with assessment_workspace(tmp_path, logger) as workspace:
        (workspace / "report.html").write_text("complete", encoding="utf-8")
        publish_assessment(tmp_path, workspace, {"run_id": tmp_path.name}, logger)
    assert len(attempts) == 3 and (tmp_path / "assessment/metadata.json").is_file()
    print("SANITY: temporary Windows directory locks receive a bounded retry before successful publication.")
