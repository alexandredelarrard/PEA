"""Shared extraction options parse into the intended source operations without source reads or writes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
from click.testing import CliRunner

from src.data_extract import cli as cli_mod
from src.data_extract.transformers import step_extract_fundamentals as step_fundamentals
from src.data_extract.transformers import step_extract_identity as step_identity
from src.data_extract.transformers import step_extract_zips as step_zips
from tests.data_extract.edgar_fixtures import stub_edgar_step
from tests.support.propagation import stub_propagation
from tests.support.universe import seed_universe


def test_cli_tickers_split_csv_before_normalizing(sqlite_store: Any) -> None:
    context = SimpleNamespace(universe=seed_universe(sqlite_store, ["AAPL", "BRK-B", "MSFT"]))

    assert cli_mod._tickers(context, " aapl ,msft, ,AAPL,brk.b,brk/b ") == ["AAPL", "MSFT", "BRK-B"]
    print("SANITY: -t CSV splits on commas and normalizes each symbol once to the yfinance spelling (brk.b, brk/b -> BRK-B).")


@pytest.mark.parametrize("names,message", [("GOOG", "GOOG (alias of GOOGL)"), ("AAA,FOO", "FOO")])
def test_cli_refuses_names_outside_the_universe_before_any_fetch(
    names: str, message: str, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SimpleNamespace(data_extract=SimpleNamespace(years_history=31))
    context = SimpleNamespace(config=config, log=Mock(), universe=seed_universe(sqlite_store, ["AAA", "GOOGL"], aliases={"GOOGL": ["GOOG"]}))
    fetch = Mock(side_effect=AssertionError("fetched before the subset check"))
    monkeypatch.setattr(cli_mod, "get_config_context", lambda path, **kwargs: (config, context))
    monkeypatch.setattr(cli_mod, "fetch_short_interest", fetch)

    result = CliRunner().invoke(cli_mod.cli, ["short-interest", "-t", names])
    assert result.exit_code != 0 and message in str(result.exception) and not fetch.called
    print(f"SANITY: -t {names} exits non-zero naming {message!r}; the fetcher was never called.")


@pytest.mark.parametrize(
    "only,expected",
    [
        ((), ["sec-tickers", "sharadar-tickers", "sharadar-actions", "sec-submissions", "sync"]),
        (("--only", "sync"), ["sync"]),
        (("--only", "sharadar-tickers", "--only", "sec-tickers"), ["sharadar-tickers", "sec-tickers"]),
        (("--only", "sec-submissions", "--only", "sharadar-actions"), ["sec-submissions", "sharadar-actions"]),
    ],
)
def test_cli_universe_runs_the_named_parts_in_order(only: tuple[str, ...], expected: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[str] = []
    step = SimpleNamespace(
        run_sec_tickers=lambda: ran.append("sec-tickers"),
        run_sharadar_tickers=lambda: ran.append("sharadar-tickers"),
        run_sharadar_actions=lambda: ran.append("sharadar-actions"),
        run_sec_submissions=lambda: ran.append("sec-submissions"),
        run_sync=lambda: ran.append("sync"),
    )
    monkeypatch.setattr(cli_mod, "get_config_context", lambda path, **kwargs: (None, None))
    monkeypatch.setattr(cli_mod, "StepExtractUniverse", lambda context, config, as_of: step)

    result = CliRunner().invoke(cli_mod.cli, ["universe", *only])
    assert result.exit_code == 0, result.exception
    assert ran == expected
    print(f"SANITY: `universe {' '.join(only)}` ran {ran}.")


def test_cli_default_is_the_universe_list(sqlite_store: Any) -> None:
    context = SimpleNamespace(universe=seed_universe(sqlite_store, ["AAPL", "MSFT"]))

    assert cli_mod._tickers(context, None) == ["AAPL", "MSFT"]
    print("SANITY: omitted -t returns the universe list unchanged.")


def test_cli_short_interest_passes_the_normalized_subset(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(data_extract=SimpleNamespace(years_history=31))
    context = SimpleNamespace(config=config, log=Mock(), universe=seed_universe(sqlite_store, ["AAA", "BBB"]))
    fetch = Mock()
    monkeypatch.setattr(cli_mod, "get_config_context", lambda path, **kwargs: (config, context))
    monkeypatch.setattr(cli_mod, "fetch_short_interest", fetch)

    result = CliRunner().invoke(cli_mod.cli, ["short-interest", "-t", " aaa ,AAA", "-F"])
    assert result.exit_code == 0, result.exception
    assert fetch.call_args.kwargs["tickers"] == ["AAA"] and fetch.call_args.kwargs["full"] is True
    print("SANITY: short-interest passes the normalized CSV subset and the full flag to the source.")


def stub_zips_step(monkeypatch: pytest.MonkeyPatch, config: SimpleNamespace) -> None:
    """Make `StepExtractZips` constructible on a stub config: caches and the Notes landing scrape stubbed."""
    config.local = SimpleNamespace(paths=SimpleNamespace(insider_transactions="insider", financial_notes="notes", fails_deliver="ftd"))
    monkeypatch.setattr(step_zips, "cache_dir", lambda ctx, name: Path(name))
    monkeypatch.setattr(step_zips, "notes_periods", lambda ctx, years, today: ["2026q1"])


@pytest.mark.parametrize("command,target", [(["zips-ingest", "--only", "insider"], "fetch_insider_transactions")])
def test_cli_subset_reaches_the_source(command: list[str], target: str, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(data_extract=SimpleNamespace(years_history=31))
    context = SimpleNamespace(config=config, log=Mock(), config_dir="configs", universe=seed_universe(sqlite_store, ["AAA", "BBB"]))
    fetch = Mock()
    monkeypatch.setattr(cli_mod, "get_config_context", lambda path, **kwargs: (config, context))
    stub_zips_step(monkeypatch, config)
    monkeypatch.setattr(step_zips, target, fetch)

    result = CliRunner().invoke(cli_mod.cli, [*command, "-t", " aaa ,AAA", "-F"])
    assert result.exit_code == 0, result.exception
    assert fetch.call_args.args[1] == ["AAA"] and "all_tickers" not in fetch.call_args.kwargs
    print(f"SANITY: {command} passes the normalized CSV subset; the source judges the scope against context.universe.")


def test_cli_backfill_keeps_the_missing_subset_distinct(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(data_extract=SimpleNamespace(years_history=31))
    context = SimpleNamespace(config=config, log=Mock(), universe=seed_universe(sqlite_store, ["AAA", "BBB"]))
    fetch = Mock()
    monkeypatch.setattr(cli_mod, "get_config_context", lambda path, **kwargs: (config, context))
    monkeypatch.setattr(cli_mod, "fetch_13f_backfill", fetch)
    runner = CliRunner()

    for flags, expected in (([], None), (["-t", " aaa ,bbb"], ["AAA", "BBB"])):
        result = runner.invoke(cli_mod.cli, ["thirteen-f-backfill", *flags])
        assert result.exit_code == 0, result.exception
        assert fetch.call_args.kwargs["tickers"] == expected and "all_tickers" not in fetch.call_args.kwargs
    print("SANITY: 13F backfill retains None for omitted -t (new names); explicit CSV reaches the source as a list.")


def stub_identity_step(monkeypatch: pytest.MonkeyPatch, config: SimpleNamespace, context: SimpleNamespace) -> None:
    """Make `StepExtractIdentity` constructible on a stub config and context: caches, universe and resolver stubbed."""
    config.local = SimpleNamespace(paths=SimpleNamespace(insider_transactions="insider", financial_notes="notes", fails_deliver="ftd"))
    context.config_dir = "configs"
    monkeypatch.setattr(step_identity, "cache_dir", lambda ctx, name: Path(name))
    stub_propagation(monkeypatch, step_identity)


@pytest.fixture
def context(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    config = SimpleNamespace(data_extract=SimpleNamespace(years_history=31))
    context = SimpleNamespace(config=config, log=Mock(), store=Mock(), ensure_edgar_identity=Mock(), universe=seed_universe(sqlite_store, ["AAA"]))
    context.store.row_count.return_value = 1
    monkeypatch.setattr(cli_mod, "get_config_context", lambda path, **kwargs: (config, context))
    monkeypatch.setattr(cli_mod, "_tickers", lambda ctx, names: ["AAA"])
    return context


@pytest.mark.parametrize(
    ("command", "target", "keyword"),
    [
        ("zips-ingest --only notes", "ingest_financial_notes", "full"),
        ("zips-ingest --only insider", "fetch_insider_transactions", "reparse"),
        ("sec-fundamentals-history", "build_fundamentals_history", "rebuild_history"),
        ("identity --only propagate", "recheck_tickers", "every"),
        ("edgar-index", "refresh", "rebuild"),
    ],
)
def test_full_reaches_the_existing_source_operation(
    command: str, target: str, keyword: str, context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    edgar_calls: list[dict[str, Any]] = []

    def record(*args: Any, **kwargs: Any) -> SimpleNamespace:
        # the notes core takes its `ZipScope` positionally: its `full` and run date are the options it carries
        scope = next((arg for arg in args if hasattr(arg, "full") and hasattr(arg, "run_date")), None)
        calls.append(kwargs if scope is None else {"full": scope.full, "as_of": scope.run_date})
        return SimpleNamespace(removals=pd.DataFrame())

    if command.startswith("identity "):
        stub_identity_step(monkeypatch, context.config, context)
        monkeypatch.setattr(step_identity, target, record)
    elif command.startswith("zips-ingest "):
        stub_zips_step(monkeypatch, context.config)
        monkeypatch.setattr(step_zips, target, record)
    elif command == "edgar-index":
        monkeypatch.setattr(cli_mod.edgar_index, target, record)
        monkeypatch.setattr(cli_mod.edgar_index, "quarter_counts", lambda ctx: {})
    elif command == "sec-fundamentals-history":
        monkeypatch.setattr(step_fundamentals, target, record)
    else:
        monkeypatch.setattr(cli_mod, target, record)
    context.config_dir = "configs"
    stub_edgar_step(monkeypatch, lambda ctx, **kwargs: edgar_calls.append(kwargs))
    runner = CliRunner()
    for flags in ([], ["-F"], ["--full"]):
        calls.clear()
        edgar_calls.clear()
        date_flags = [] if command == "sec-fundamentals-history" else ["--as-of", "2026-09-29"]
        result = runner.invoke(cli_mod.cli, [*command.split(), *date_flags, *flags])
        assert result.exit_code == 0, result.output
        assert len(calls) == 1 and calls[0][keyword] is bool(flags)
        assert not edgar_calls, "no command of this list walks EDGAR"
        if calls and "as_of" in calls[0]:
            assert calls[0]["as_of"] == pd.Timestamp("2026-09-29")
    context.store.delete.assert_not_called()
    print(f"SANITY: {command} resumes by default; -F and --full select the complete source operation; supported run dates reach the source.")


def test_fundamentals_check_runs_the_check_on_the_ticker_subset(context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []
    context.config.data_extract.fundamentals_check = SimpleNamespace(gap_pct=0.03, field_share=0.05, top_tickers=3)
    context.config_dir = "configs"
    monkeypatch.setattr(step_fundamentals, "check_fundamentals_sources", lambda ctx, tickers, **kwargs: calls.append((tickers, kwargs)))
    monkeypatch.setattr(step_fundamentals, "build_fundamentals_history", Mock(side_effect=AssertionError("the check builds no history")))

    result = CliRunner().invoke(cli_mod.cli, ["fundamentals-check", "-t", "AAA"])

    assert result.exit_code == 0, result.output
    assert calls == [(["AAA"], {"gap_pct": 0.03, "field_share": 0.05, "top_tickers": 3})], calls
    context.store.save.assert_not_called()
    context.store.delete.assert_not_called()
    print(f"SANITY: fundamentals-check -> check only, tickers {calls[0][0]}, thresholds {calls[0][1]}; no history build, no store write.")


def test_help_lists_the_split_fundamentals_commands_and_none_of_the_removed_ones() -> None:
    """AC-010: one command per producer stage, plus the check; the combined `fundamentals` / `fundamentals-facts` are gone."""
    result = CliRunner().invoke(cli_mod.cli, ["--help"])
    listed = {line.split()[0] for line in result.output.splitlines() if line.startswith("  ") and line.split()}
    current = {"sec-fundamentals-facts", "sec-fundamentals-history", "fundamentals-check", "fundamentals-sharadar", "fundamentals-employees"}

    assert result.exit_code == 0, result.output
    assert current <= listed, sorted(current - listed)
    assert not {"fundamentals", "fundamentals-facts"} & listed and not {"fundamentals", "fundamentals-facts"} & set(cli_mod.cli.commands)
    print(f"SANITY: --help lists {sorted(current)}; the removed fundamentals / fundamentals-facts commands are absent.")


@pytest.mark.parametrize(
    "command,exit_code,called",
    [
        ("identity --dry-run", 2, []),
        ("identity --only cik-history --dry-run", 2, []),
        ("identity --only propagate --dry-run", 0, ["run_propagate"]),
        ("identity --only cik-history", 0, ["run_cik_history"]),
        ("identity --approve-rekey A:B", 2, []),
        ("zips-ingest --repair-availability", 2, []),
        ("zips-ingest --only notes --only ftd --repair-availability", 2, []),
        ("zips-ingest --only notes --repair-availability", 0, ["run_ingest_financial_notes"]),
    ],
)
def test_a_flag_scoped_to_one_stage_is_refused_elsewhere(
    command: str, exit_code: int, called: list[str], context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    step = Mock()
    step.run_propagate.return_value.removals = pd.DataFrame()
    monkeypatch.setattr(cli_mod, "StepExtractIdentity", Mock(return_value=step))
    monkeypatch.setattr(cli_mod, "StepExtractZips", Mock(return_value=step))
    result = CliRunner().invoke(cli_mod.cli, command.split())
    assert result.exit_code == exit_code, result.output
    assert [name for name, _, _ in step.mock_calls if "." not in name] == called
    print(f"SANITY: `{command}` exits {exit_code} and runs {called or 'nothing'}; a scoped flag never widens the run.")
