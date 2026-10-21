"""Assessment launcher contracts through the top-level dynamic plugin loader."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import click
import pytest
from click.testing import CliRunner
from omegaconf import DictConfig, OmegaConf

from src import context as context_module
from src import portfolio as portfolio_module
from src.cli import CLI, main
from src.context import Context
from src.modelling.steps import step_assess_long_short, step_long_short
from src.modelling.utils.artifacts import member_inventory, member_path, save_member
from src.modelling.utils.assessment import write_json
from tests.modelling.model_fixtures import completed_run


@pytest.mark.parametrize("command,selected", [("train", None), ("full-train", None), ("full-train", "chosen")])
def test_training_cli_uses_automatic_identity_and_optional_full_history_selection(
    command: str, selected: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = cast(DictConfig, OmegaConf.create({"model": {"assessment": {}}, "train": {"start_date": "1999-01-01", "end_date": "2022-01-01"}}))
    context = SimpleNamespace(log=Mock())
    factory = Mock(return_value=(config, context))
    fitted = Mock()
    fitted.run_train.return_value = SimpleNamespace(run_id="generated")
    constructor = Mock(return_value=fitted)
    monkeypatch.setattr(context_module, "get_config_context", factory)
    monkeypatch.setattr(step_long_short, "StepLongShort", constructor)
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    args = ["modelling", command, "-c", "./configs"] + (["--run-id", selected] if selected else [])
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, repr(result.exception)
    fitted.run_train.assert_called_once_with(**({"full_history": True, "run_id": selected} if command == "full-train" else {}))
    if command == "train":
        assert result.output.splitlines()[-1] == "generated"
    assert "run_id" not in config.model.assessment and "resume" not in config.model.assessment
    print(
        "SANITY: train returns its automatic ID as final stdout; full-history forwards optional exact selection without persistent identity/resume controls."
    )


def test_failed_training_emits_no_successful_run_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    config = OmegaConf.create({"model": {"assessment": {}}, "train": {"start_date": "1999-01-01", "end_date": "2022-01-01"}})
    monkeypatch.setattr(context_module, "get_config_context", Mock(return_value=(config, SimpleNamespace(log=Mock()))))
    fitted = Mock()
    fitted.run_train.side_effect = ValueError("injected fit failure")
    monkeypatch.setattr(step_long_short, "StepLongShort", Mock(return_value=fitted))
    result = CliRunner().invoke(main, ["modelling", "train"])
    assert result.exit_code != 0 and not result.output.strip()
    print("SANITY: failed training cannot emit the successful final stdout record consumed by DAG XCom.")


@pytest.mark.parametrize("layout", ["worktree-relative", "worktree-absolute", "explicit-root", "checkout"])
def test_assess_dynamic_loader_preserves_root_and_selection(layout: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    monkeypatch.setattr(context_module, "__file__", str(source_root / "src" / "context.py"))
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    monkeypatch.setenv("ROOT_PATH", "")
    monkeypatch.delenv("ROOT_PATH")
    expected_root = None
    if layout != "checkout":
        original_root = tmp_path / "original"
        git_dir = original_root / ".git" / "worktrees" / "assessment"
        pointer = os.path.relpath(git_dir, source_root) if layout == "worktree-relative" else str(git_dir)
        (source_root / ".git").write_text(f"gitdir: {pointer}\n", encoding="utf-8")
        expected_root = str(original_root)
        if layout == "explicit-root":
            expected_root = str(tmp_path / "explicit" / ".." / "custom")
            monkeypatch.setenv("ROOT_PATH", expected_root)
    else:
        (source_root / ".git").mkdir()

    config = cast(DictConfig, OmegaConf.create({"model": {"assessment": {"output_dir": str(tmp_path / "configured_runs")}}}))
    context = SimpleNamespace(log=Mock())
    context_calls = []

    def create_context(config_path: str, use_cache: bool, save: bool) -> tuple[DictConfig, Context]:
        context_calls.append((config_path, use_cache, save, os.environ.get("ROOT_PATH")))
        return config, cast(Context, context)

    step = SimpleNamespace(run=Mock(return_value="assessment result"))
    constructor = Mock(return_value=step)
    monkeypatch.setattr(context_module, "get_config_context", create_context)
    monkeypatch.setattr(step_assess_long_short, "StepAssessLongShort", constructor)
    plugin = cast(click.Group, cast(CLI, main).get_command(click.Context(main), "modelling"))
    callback = plugin.commands["assess"].callback
    assert callback is not None and "__file__" not in callback.__globals__

    args = ["modelling", "assess", "-c", "./configs", "--run-id", "20261018_42"]
    result = CliRunner().invoke(main, args)

    assert result.exit_code == 0, repr(result.exception)
    assert context_calls == [("./configs", False, False, expected_root)]
    assert os.environ.get("ROOT_PATH") == expected_root
    assert config.model.assessment.output_dir == str(tmp_path / "configured_runs")
    assert "run_id" not in config.model.assessment
    constructor.assert_called_once_with(context=context, config=config, run_id="20261018_42")
    step.run.assert_called_once_with()
    print(f"SANITY: dynamic loader keeps {layout} routing and forwards the exact constructor selection with save=False.")


@pytest.mark.parametrize(
    "args",
    [
        ["assess", "--report-only"],
        ["assess", "--resume"],
        ["train", "--run-id", "manual"],
        ["train", "--resume"],
        ["full-train", "--resume"],
        ["assess", "--run-id", "CON"],
    ],
)
def test_retired_modes_and_unsafe_ids_fail_before_context(args: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    factory = Mock(side_effect=AssertionError("Invalid controls must fail before opening context"))
    monkeypatch.setattr(context_module, "get_config_context", factory)
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    result = CliRunner().invoke(main, ["modelling", *args])
    assert result.exit_code == 2, repr(result.exception)
    factory.assert_not_called()
    print(f"SANITY: invalid or retired CLI controls {args} cannot open data or invoke fitting.")


@pytest.mark.parametrize("state", ["complete", "incomplete", "missing-configuration", "malformed-configuration"])
@pytest.mark.parametrize("selected", [None, "selected"])
def test_portfolio_cli_uses_selected_or_latest_assessed_cutoff(
    state: str, selected: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, config = completed_run(tmp_path / "runs", "selected", "2026-10-21T10:00:00+00:00")
    completed_run(tmp_path / "runs", "newer_unassessed", "2026-10-21T11:00:00+00:00")
    cutoff = run / "cutoff"
    for family in config.model.ensemble:
        save_member(family, member_path(cutoff, 60, family))
    metadata = json.loads((cutoff / "metadata.json").read_text(encoding="utf-8"))
    metadata["member_files"] = member_inventory(cutoff, [60], config.model.ensemble)
    write_json(cutoff / "metadata.json", metadata)
    config.model.models_dir = "output/models"
    config.portfolio = {"start": "2022-06-01", "fee_bps": 2.0}
    write_json(run / "assessment/metadata.json", {"run_id": run.name, "complete": state != "incomplete"})
    (run / "report.html").write_text("assessed", encoding="utf-8")
    if state == "missing-configuration":
        (run / "training/configuration.json").unlink()
    elif state == "malformed-configuration":
        write_json(run / "training/configuration.json", {"seed": 8})
    context = SimpleNamespace(log=Mock(), paths={"DATA_STORE": tmp_path, "OUTPUT_DIR": tmp_path / "output"})
    constructor = Mock(return_value=SimpleNamespace(run=Mock()))
    monkeypatch.setattr(context_module, "get_config_context", Mock(return_value=(config, context)))
    monkeypatch.setattr(portfolio_module, "StepPortfolio", constructor)
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    args = ["portfolio", "backtest", "-c", "./configs"] + (["--run-id", selected] if selected else [])
    result = CliRunner().invoke(main, args)
    if state == "complete":
        assert result.exit_code == 0, repr(result.exception)
        assert config.model.models_dir == str(run / "cutoff") and context.paths["OUTPUT_DIR"] == run / "portfolio"
        assert config.portfolio["start"] == "2022-06-01" and config.portfolio["fee_bps"] == 2.0
        constructor.assert_called_once_with(context=context, config=config)
    else:
        assert result.exit_code != 0
        constructor.assert_not_called()
    print(f"SANITY: portfolio state={state} is checked before construction; latest unassessed training cannot change the cutoff or dates/costs.")


@pytest.mark.parametrize("state", ["missing-inventory", "null-inventory", "wrong-run", "full-history"])
def test_portfolio_cli_refuses_invalid_cutoff_before_consuming_members(state: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, config = completed_run(tmp_path / "runs", "selected", "2026-10-21T10:00:00+00:00")
    config.model.models_dir = "output/models"
    write_json(run / "assessment/metadata.json", {"run_id": run.name, "complete": True})
    (run / "report.html").write_text("assessed", encoding="utf-8")
    path = run / "cutoff/metadata.json"
    cutoff = json.loads(path.read_text(encoding="utf-8"))
    if state == "null-inventory":
        cutoff["member_files"] = None
    elif state == "wrong-run":
        cutoff["run_id"] = "another"
    elif state == "full-history":
        cutoff["full_history"] = True
    write_json(path, cutoff)
    context = SimpleNamespace(log=Mock(), paths={"DATA_STORE": tmp_path, "OUTPUT_DIR": tmp_path / "output"})
    constructor = Mock(side_effect=AssertionError("Invalid cutoff reached StepPortfolio and its member consumers"))
    monkeypatch.setattr(context_module, "get_config_context", Mock(return_value=(config, context)))
    monkeypatch.setattr(portfolio_module, "StepPortfolio", constructor)
    monkeypatch.chdir(Path(__file__).resolve().parents[2])

    result = CliRunner().invoke(main, ["portfolio", "backtest", "--run-id", run.name])

    assert isinstance(result.exception, ValueError), repr(result.exception)
    message = (
        "Portfolio backtest requires the named cutoff models" if state in {"wrong-run", "full-history"} else "Missing cutoff model member inventory"
    )
    assert str(result.exception) == message
    constructor.assert_not_called()
    assert config.model.models_dir == "output/models" and context.paths["OUTPUT_DIR"] == tmp_path / "output"
    print(f"SANITY: {state} is refused before portfolio construction, member loading, or output routing changes; no fitting or data access occurs.")
