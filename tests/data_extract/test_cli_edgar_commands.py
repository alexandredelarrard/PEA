"""The five per-ticker EDGAR CLI commands dispatch their `EdgarFetch` declaration to the shared
driver with the requested tickers, window, `--full`, `--as-of` and `--no-cap`; `resume-plan` and
`markers` report per table; the steps and the Airflow DAG that call them still import and parse.
Offline: `run_edgar_fetch` is replaced by a recorder, `markers` runs on SQLite.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from click.testing import CliRunner

import src.data_extract.cli as cli_mod
from src.data_extract.transformers.step_extract_institutionals import StepExtractInstitutionals
from src.data_extract.transformers.step_extract_structure import StepExtractStructure
from src.data_extract.utils.common.edgar_driver import EdgarFetch, FilingStamp, marker_row
from src.data_extract.utils.common.resume import DocumentWork
from src.data_extract.utils.institutionals.fetch_8k_edgar import SEC_8K_FETCH
from src.data_extract.utils.institutionals.fetch_13d_edgar import SEC_13D_FETCH
from src.data_extract.utils.institutionals.fetch_13g_edgar import SEC_13G_FETCH
from src.data_extract.utils.structure.fetch_def14a_edgar import DEF14A_EDGAR_FETCH
from src.data_extract.utils.structure.fetch_filing_text import FILING_TEXT_FETCH
from src.data_store.schema import Tables
from tests.data_extract.edgar_fixtures import fake_filing

DAG_FILE = Path(__file__).resolve().parents[2] / "src" / "dags" / "dag_data_extraction.py"

#: command name -> (its fetch declaration, whether the command takes -F/--full)
COMMANDS: dict[str, tuple[EdgarFetch, bool]] = {
    "sec-8k-items": (SEC_8K_FETCH, True),
    "sec-13d": (SEC_13D_FETCH, True),
    "sec-13g": (SEC_13G_FETCH, True),
    "filing-text": (FILING_TEXT_FETCH, True),
    "def14a-edgar": (DEF14A_EDGAR_FETCH, True),
}


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    config = SimpleNamespace(data_extract=SimpleNamespace(years_history=15))
    monkeypatch.setattr(cli_mod, "get_config_context", lambda path, **kwargs: (config, SimpleNamespace(name="ctx")))
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


def test_as_of_and_no_cap_reach_the_driver(recorded: list[dict[str, Any]]) -> None:
    result = CliRunner().invoke(cli_mod.cli, ["sec-8k-items", "-t", "AAPL", "--as-of", "2026-09-30", "--no-cap"])

    assert result.exit_code == 0, result.output
    assert recorded[0]["as_of"] == pd.Timestamp("2026-09-30") and recorded[0]["no_cap"] is True
    print("\n=== SANITY: --as-of / --no-cap ===")
    print("  the injected run date and the lifted cap reach run_edgar_fetch.")


def test_markers_counts_and_deletes_by_sentinel(monkeypatch: pytest.MonkeyPatch, sqlite_store: Any) -> None:
    stamp = FilingStamp.of(fake_filing("0001-empty", 1, "2026-09-10"), "0000000001")
    real = pd.DataFrame({"ticker": ["AAA"], "accession_number": ["0001-real"], "item": ["8.01"], "filing_date": [pd.Timestamp("2026-09-09")]})
    sqlite_store.save(Tables.sec_8k, pd.concat([real, marker_row(Tables.sec_8k, "AAA", stamp)], ignore_index=True))
    monkeypatch.setattr(cli_mod, "get_config_context", lambda path, **kwargs: (None, SimpleNamespace(store=sqlite_store)))

    counted = CliRunner().invoke(cli_mod.cli, ["markers", "--table", "sec_8k"])
    deleted = CliRunner().invoke(cli_mod.cli, ["markers", "--table", "sec_8k", "--delete"])

    assert counted.exit_code == 0 and '"markers": 1' in counted.output and '"deleted": 0' in counted.output
    assert deleted.exit_code == 0 and '"deleted": 1' in deleted.output
    assert sqlite_store.load(Tables.sec_8k, markers=True)["accession_number"].tolist() == ["0001-real"]
    print("\n=== SANITY: markers command ===")
    print(f"  count -> {counted.output.strip()}; --delete removed the marker and kept the real row.")


def test_resume_plan_prints_one_line_per_document_table(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(data_extract=SimpleNamespace(years_history=15))
    monkeypatch.setattr(cli_mod, "get_config_context", lambda path, **kwargs: (config, SimpleNamespace(name="ctx")))
    monkeypatch.setattr(cli_mod, "_tickers", lambda ctx, names: ["AAA"])
    monkeypatch.setattr(cli_mod, "load_cik_mapping", lambda ctx, names: pd.DataFrame({"ticker": ["AAA"], "cik": ["0000000001"]}))
    monkeypatch.setattr(cli_mod, "_document_fetches", lambda ctx, names: {"sec_8k": SEC_8K_FETCH, "sec_13g": SEC_13G_FETCH})
    monkeypatch.setattr(cli_mod, "load_edgar_scope", lambda ctx, identity_aware: None)
    work = DocumentWork(units={"AAA": pd.DataFrame({"accession": ["x"]})}, key_class={"AAA": "established"}, counts={"forward": 1}, uncapped=1)
    work.timings = {"index_read": 0.1, "done_read": 0.2, "total": 0.3}
    monkeypatch.setattr(cli_mod, "plan_fetch", lambda *args, **kwargs: work)

    result = CliRunner().invoke(cli_mod.cli, ["resume-plan", "--timings", "--as-of", "2026-09-30"])

    lines = result.output.strip().splitlines()
    assert result.exit_code == 0 and len(lines) == 2 and '"table": "sec_8k"' in lines[0] and '"timings_s"' in lines[0]
    print("\n=== SANITY: resume-plan ===")
    print(f"  {lines[0]}")


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


@pytest.fixture
def insider_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    config = SimpleNamespace(data_extract=SimpleNamespace(years_history=15))
    monkeypatch.setattr(cli_mod, "get_config_context", lambda path, **kwargs: (config, SimpleNamespace(name="ctx")))
    monkeypatch.setattr(cli_mod, "_tickers", lambda ctx, names: [t.strip().upper() for t in names.split(",")] if names else ["ALL"])
    monkeypatch.setattr(cli_mod, "fetch_insider_transactions", lambda context, **kwargs: calls.append(("bulk", kwargs)))
    monkeypatch.setattr(cli_mod, "fetch_insider_edgar", lambda context, **kwargs: calls.append(("edgar", kwargs)))
    return calls


def test_the_insider_bulk_parse_and_its_live_edgar_tail_run_as_separate_commands(insider_calls: list[tuple[str, dict[str, Any]]]) -> None:
    """D-8: `--bulk-only` parses the zips without the EDGAR walk, `insider-edgar` walks only; the bare command still does both."""
    runner = CliRunner()
    invocations = (
        ["insider-transactions", "--bulk-only", "-t", "aapl"],
        ["insider-edgar", "-t", "aapl", "-F"],
        ["insider-transactions", "-t", "aapl"],
    )
    for args in invocations:
        result = runner.invoke(cli_mod.cli, args)
        assert result.exit_code == 0, result.output

    assert [kind for kind, _ in insider_calls] == ["bulk", "edgar", "bulk", "edgar"]
    assert {k: insider_calls[1][1][k] for k in ("tickers", "years_history", "full", "no_cap")} == {
        "tickers": ["AAPL"],
        "years_history": 15,
        "full": True,
        "no_cap": False,
    }
    assert insider_calls[3][1]["full"] is False and insider_calls[0][1]["tickers"] == ["AAPL"]
    print("\n=== SANITY: insider commands (D-8) ===")
    print("  insider-transactions --bulk-only -> bulk parse only; insider-edgar -F -> live walk only (full=True);")
    print("  insider-transactions -> both, as before. OK: the DAG can schedule the walk in its own pool.")
