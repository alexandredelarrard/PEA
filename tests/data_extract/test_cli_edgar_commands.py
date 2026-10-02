"""The five per-ticker EDGAR CLI commands dispatch their `EdgarFetch` declaration to the shared
driver with the requested tickers, window and `--full` flag; the steps and the Airflow DAG that
call them still import and parse. Offline: `run_edgar_fetch` is replaced by a recorder.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from click.testing import CliRunner

import src.data_extract.cli as cli_mod
from src.data_extract.transformers.step_extract_institutionals import StepExtractInstitutionals
from src.data_extract.transformers.step_extract_structure import StepExtractStructure
from src.data_extract.utils.common.edgar_driver import EdgarFetch
from src.data_extract.utils.institutionals.fetch_8k_edgar import SEC_8K_FETCH
from src.data_extract.utils.institutionals.fetch_13d_edgar import SEC_13D_FETCH
from src.data_extract.utils.institutionals.fetch_13g_edgar import SEC_13G_FETCH
from src.data_extract.utils.structure.fetch_def14a_edgar import DEF14A_EDGAR_FETCH
from src.data_extract.utils.structure.fetch_filing_text import FILING_TEXT_FETCH

DAG_FILE = Path(__file__).resolve().parents[2] / "src" / "dags" / "dag_data_extraction.py"

#: command name -> (its fetch declaration, whether the command takes -F/--full)
COMMANDS: dict[str, tuple[EdgarFetch, bool]] = {
    "sec-8k-items": (SEC_8K_FETCH, True),
    "sec-13d": (SEC_13D_FETCH, True),
    "sec-13g": (SEC_13G_FETCH, True),
    "filing-text": (FILING_TEXT_FETCH, False),
    "def14a-edgar": (DEF14A_EDGAR_FETCH, False),
}


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    config = SimpleNamespace(data_extract=SimpleNamespace(years_history=15))
    monkeypatch.setattr(cli_mod, "_ctx", lambda path: (config, SimpleNamespace(name="ctx")))
    monkeypatch.setattr(cli_mod, "_tickers", lambda ctx, names: [t.strip().upper() for t in names.split(",")])
    monkeypatch.setattr(cli_mod, "run_edgar_fetch", lambda context, **kwargs: calls.append({"context": context, **kwargs}))
    return calls


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_edgar_command_dispatches_its_fetch_declaration(command: str, recorded: list[dict[str, Any]]) -> None:
    fetch, takes_full = COMMANDS[command]
    args = [command, "-t", "aapl,msft", "-y", "7", *(["-F"] if takes_full else [])]

    result = CliRunner().invoke(cli_mod.cli, args)

    assert result.exit_code == 0, result.output
    assert len(recorded) == 1
    call = recorded[0]
    assert call["fetch"] is fetch
    assert call["tickers"] == ["AAPL", "MSFT"]
    assert call["years_history"] == 7
    assert call.get("full", False) is takes_full
    print(f"\n=== SANITY: `{command}` dispatch ===")
    print(f"  {args} -> run_edgar_fetch(fetch={fetch.desc!r}, tickers={call['tickers']}, years=7, full={call.get('full', False)}).")


def test_edgar_command_without_years_uses_the_configured_window(recorded: list[dict[str, Any]]) -> None:
    result = CliRunner().invoke(cli_mod.cli, ["sec-13g", "-t", "JNJ"])

    assert result.exit_code == 0, result.output
    assert recorded[0]["years_history"] == 15
    assert recorded[0]["full"] is False
    print("\n=== SANITY: default window ===")
    print("  no -y -> data_extract.years_history (15) reaches the driver; no -F -> full=False.")


def test_steps_import_and_dag_parses() -> None:
    tree = ast.parse(DAG_FILE.read_text(encoding="utf-8"))
    dag_commands = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "fetch" and node.args and isinstance(node.args[0], ast.Constant)
    }

    assert callable(StepExtractInstitutionals.run) and callable(StepExtractStructure.run)
    assert set(COMMANDS) <= dag_commands, f"DAG no longer schedules {sorted(set(COMMANDS) - dag_commands)}"
    assert set(COMMANDS) <= set(cli_mod.cli.commands)
    print("\n=== SANITY: step imports + DAG ===")
    print(f"  both steps import; the DAG parses and still schedules all {len(COMMANDS)} EDGAR commands by their unchanged CLI names.")
