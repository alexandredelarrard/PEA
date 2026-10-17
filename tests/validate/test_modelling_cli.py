"""Modelling validator contracts through the top-level dynamic plugin loader."""

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
from src.validate import io as validation_io
from src.validate.checks import modelling as modelling_check
from src.validate.result import CheckResult


@pytest.mark.parametrize("layout", ["worktree-relative", "worktree-absolute", "explicit-root", "explicit-empty", "checkout"])
@pytest.mark.parametrize("status,exit_code", [("pass", 0), ("fail", 1), ("abstain", 3)])
def test_modelling_dynamic_loader_preserves_root_and_result(
    layout: str, status: str, exit_code: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
        git_dir = original_root / ".git" / "worktrees" / "validator"
        pointer = os.path.relpath(git_dir, source_root) if layout == "worktree-relative" else str(git_dir)
        (source_root / ".git").write_text(f"gitdir: {pointer}\n", encoding="utf-8")
        expected_root = str(original_root)
        if layout in {"explicit-root", "explicit-empty"}:
            expected_root = str(tmp_path / "explicit" / ".." / "custom") if layout == "explicit-root" else ""
            monkeypatch.setenv("ROOT_PATH", expected_root)
    else:
        (source_root / ".git").mkdir()

    config = cast(DictConfig, OmegaConf.create({}))
    context = SimpleNamespace()
    context_calls = []

    def create_context(config_path: str, use_cache: bool, save: bool) -> tuple[DictConfig, Context]:
        context_calls.append((config_path, use_cache, save, os.environ.get("ROOT_PATH")))
        return config, cast(Context, context)

    checked = CheckResult("modelling", "assessment", status, reason="known outcome")
    checker = Mock(return_value=checked)
    output = str(tmp_path / "run")
    result_path = Path(output) / "_out" / "modelling.json"
    writer = Mock(return_value=result_path)
    monkeypatch.setattr(context_module, "get_config_context", create_context)
    monkeypatch.setattr(modelling_check, "check_modelling", checker)
    monkeypatch.setattr(validation_io, "write_result", writer)
    plugin = cast(click.Group, cast(CLI, main).get_command(click.Context(main), "validate"))
    callback = plugin.commands["modelling"].callback
    assert callback is not None and "__file__" not in callback.__globals__

    result = CliRunner().invoke(main, ["validate", "modelling", "-c", "./configs", "-o", output])

    assert result.exit_code == exit_code, repr(result.exception)
    assert context_calls == [("./configs", False, False, expected_root)]
    assert os.environ.get("ROOT_PATH") == expected_root
    checker.assert_called_once_with(context, out=output, config=config)
    writer.assert_called_once_with(output, checked)
    assert checked.summary() in result.output
    assert str(result_path) in result.output
    print(f"SANITY: dynamic validator without __file__ preserves {layout}, save=False, output, and {status} exit {exit_code}.")
