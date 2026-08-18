"""
Shared fixtures for the definition-of-done hook tests.

`dod_lib` lives in `.claude/hooks/`, which is deliberately NOT an importable package: the hooks
run under `python -S -E` with only the stdlib and must never depend on repo imports. So the
tests load it by path, the same way the hook entry points do.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOKS_DIR = REPO_ROOT / ".claude" / "hooks"


def _load(name: str):
    path = HOOKS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"dod_test_{name}", path)
    assert spec and spec.loader, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def dod_lib():
    return _load("dod_lib")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


# --------------------------------------------------------------------------- #
# Transcript builders -- the shape a real Claude Code JSONL transcript has     #
# --------------------------------------------------------------------------- #
def tool_line(name: str, **inp) -> str:
    return json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "tool_use", "id": "toolu_x", "name": name, "input": inp}]}})


def text_line(text: str) -> str:
    return json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "text", "text": text}]}})


def result_line(content: str = "ok") -> str:
    return json.dumps({"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_x", "content": content}]}})


def write_transcript(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "transcript.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def make_transcript(tmp_path):
    def _make(lines: list[str]) -> Path:
        return write_transcript(tmp_path, lines)
    return _make
