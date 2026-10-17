"""Assessment launcher contracts through the top-level dynamic plugin loader."""

from __future__ import annotations

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
from src.cli import CLI, main
from src.context import Context
from src.modelling.steps import step_assess_long_short


@pytest.mark.parametrize(
    "layout,stage,resume",
    [
        ("worktree-relative", "development", True),
        ("worktree-absolute", "evaluation", False),
        ("explicit-root", "report", True),
        ("checkout", None, False),
        ("checkout", "all", True),
    ],
)
def test_assess_dynamic_loader_preserves_root_and_controls(
    layout: str, stage: str | None, resume: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    config = cast(DictConfig, OmegaConf.create({"model": {"assessment": {}}}))
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

    output = tmp_path / "run"
    args = ["modelling", "assess", "-c", "./configs", "-o", str(output)]
    if stage is not None:
        args.extend(["--stage", stage])
    if resume:
        args.append("--resume")
    result = CliRunner().invoke(main, args)

    assert result.exit_code == 0, repr(result.exception)
    assert context_calls == [("./configs", False, False, expected_root)]
    assert os.environ.get("ROOT_PATH") == expected_root
    assert config.model.assessment.output_dir == str(output.resolve())
    assert config.model.assessment.stage == (stage or "development")
    assert config.model.assessment.resume is resume
    constructor.assert_called_once_with(context=context, config=config)
    step.run.assert_called_once_with()
    print(f"SANITY: dynamic loader without __file__ reaches assessment; {layout}, stage={stage or 'development'}, resume={resume}, save=False.")
