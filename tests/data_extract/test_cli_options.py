"""Shared extraction options parse into the intended source operations without source reads or writes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, call

import pandas as pd
import pytest
from click.testing import CliRunner

from src.data_extract import cli as cli_mod
from src.data_extract.transformers import step_extract_identity as step_identity
from src.data_extract.transformers import step_extract_zips as step_zips
from src.data_store.schema import Tables
from tests.data_extract.edgar_fixtures import stub_edgar_step


def test_cli_tickers_split_csv_before_normalizing(monkeypatch: pytest.MonkeyPatch) -> None:
    context = SimpleNamespace()
    load = Mock(side_effect=AssertionError("Explicit CLI tickers must not load the universe."))
    monkeypatch.setattr(cli_mod, "load_universe_tickers", load)

    assert cli_mod._tickers(context, " aapl ,msft, ,AAPL,brk.b,brk/b ") == ["AAPL", "MSFT", "BRK.B", "BRK/B"]
    load.assert_not_called()
    print("SANITY: -t CSV splits on commas, normalizes each symbol once, and preserves share-class separators.")


def test_cli_default_reuses_the_normalized_universe_list(monkeypatch: pytest.MonkeyPatch) -> None:
    context = SimpleNamespace()
    universe = ["AAPL", "MSFT"]
    load = Mock(return_value=universe)
    monkeypatch.setattr(cli_mod, "load_universe_tickers", load)
    monkeypatch.setattr(cli_mod, "normalise_ticker", Mock(side_effect=AssertionError("Universe tickers are already normalized.")))

    assert cli_mod._tickers(context, None) is universe
    load.assert_called_once_with(context)
    load.reset_mock()
    selected, complete = cli_mod._ticker_scope(context, None)
    assert selected is universe and complete is universe
    load.assert_called_once_with(context)
    print("SANITY: omitted -t returns the universe list unchanged; scoped-write guards share its single read.")


def test_cli_short_interest_passes_the_normalized_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(data_extract=SimpleNamespace(years_history=31))
    context = SimpleNamespace(config=config, log=Mock())
    fetch = Mock()
    monkeypatch.setattr(cli_mod, "get_config_context", lambda path, **kwargs: (config, context))
    monkeypatch.setattr(cli_mod, "load_universe_tickers", Mock(return_value=["AAA", "BBB"]))
    monkeypatch.setattr(cli_mod, "fetch_short_interest", fetch)

    result = CliRunner().invoke(cli_mod.cli, ["short-interest", "-t", " aaa ,AAA", "-F"])
    assert result.exit_code == 0, result.exception
    assert fetch.call_args.kwargs["tickers"] == ["AAA"] and fetch.call_args.kwargs["full"] is True
    print("SANITY: short-interest passes the normalized CSV subset and the full flag to the source.")


def stub_zips_step(monkeypatch: pytest.MonkeyPatch, config: SimpleNamespace, load: Any) -> None:
    """Make `StepExtractZips` constructible on a stub config: caches, universe and the Notes landing scrape stubbed."""
    config.local = SimpleNamespace(
        paths=SimpleNamespace(insider_transactions="insider", financial_notes="notes", financial_statements="statements", fails_deliver="ftd")
    )
    monkeypatch.setattr(step_zips, "cache_dir", lambda ctx, name: Path(name))
    monkeypatch.setattr(step_zips, "load_universe_tickers", load)
    monkeypatch.setattr(step_zips, "notes_periods", lambda ctx, years, today: ["2026q1"])


@pytest.mark.parametrize("command,target", [(["zips-ingest", "--only", "insider"], "fetch_insider_transactions")])
def test_cli_subset_and_complete_universe_reach_the_source(command: list[str], target: str, monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(data_extract=SimpleNamespace(years_history=31))
    context = SimpleNamespace(config=config, log=Mock(), config_dir="configs")
    universe = ["AAA", "BBB"]
    load = Mock(return_value=universe)
    fetch = Mock()
    monkeypatch.setattr(cli_mod, "get_config_context", lambda path, **kwargs: (config, context))
    monkeypatch.setattr(cli_mod, "load_universe_tickers", load)
    stub_zips_step(monkeypatch, config, load)
    monkeypatch.setattr(step_zips, target, fetch)

    result = CliRunner().invoke(cli_mod.cli, [*command, "-t", " aaa ,AAA", "-F"])
    assert result.exit_code == 0, result.exception
    assert fetch.call_args.args[1] == ["AAA"]
    assert fetch.call_args.kwargs["all_tickers"] is universe
    load.assert_called_once_with(context)
    print(f"SANITY: {command} passes the normalized CSV subset and one complete universe snapshot to protect other rows.")


def test_cli_backfill_keeps_the_missing_subset_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(data_extract=SimpleNamespace(years_history=31))
    context = SimpleNamespace(config=config, log=Mock())
    fetch = Mock()
    monkeypatch.setattr(cli_mod, "get_config_context", lambda path, **kwargs: (config, context))
    monkeypatch.setattr(cli_mod, "load_universe_tickers", lambda ctx: ["AAA", "BBB"])
    monkeypatch.setattr(cli_mod, "fetch_13f_backfill", fetch)
    runner = CliRunner()

    for flags, expected in (([], None), (["-t", " aaa ,bbb"], ["AAA", "BBB"])):
        result = runner.invoke(cli_mod.cli, ["thirteen-f-backfill", *flags])
        assert result.exit_code == 0, result.exception
        assert fetch.call_args.kwargs["tickers"] == expected
        assert fetch.call_args.kwargs["all_tickers"] == ["AAA", "BBB"]
    print("SANITY: 13F backfill retains None for omitted -t (new names); explicit CSV reaches the source as a list.")


def stub_identity_step(monkeypatch: pytest.MonkeyPatch, config: SimpleNamespace, context: SimpleNamespace) -> None:
    """Make `StepExtractIdentity` constructible on a stub config and context: caches, universe and resolver stubbed."""
    config.local = SimpleNamespace(paths=SimpleNamespace(insider_transactions="insider", financial_notes="notes", fails_deliver="ftd"))
    config.data_extract.redundant_ticks = []
    context.config_dir = "configs"
    monkeypatch.setattr(step_identity, "cache_dir", lambda ctx, name: Path(name))
    monkeypatch.setattr(step_identity, "load_universe_tickers", lambda ctx: ["AAA", "BBB"])
    monkeypatch.setattr(step_identity, "load_identity", lambda ctx, refresh=False: object())


@pytest.fixture
def context(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    config = SimpleNamespace(data_extract=SimpleNamespace(years_history=31))
    context = SimpleNamespace(config=config, log=Mock(), store=Mock(), ensure_edgar_identity=Mock())
    context.store.row_count.return_value = 1
    monkeypatch.setattr(cli_mod, "get_config_context", lambda path, **kwargs: (config, context))
    monkeypatch.setattr(cli_mod, "_tickers", lambda ctx, names: ["AAA"])
    monkeypatch.setattr(cli_mod, "load_universe_tickers", lambda ctx: ["AAA"])
    monkeypatch.setattr(cli_mod, "unverified_ciks", lambda ctx: [])
    return context


@pytest.mark.parametrize(
    ("command", "target", "keyword"),
    [
        ("seed-universe", "get_sp500_tickers", None),
        ("zips-ingest --only notes", "ingest_financial_notes", "full"),
        ("zips-ingest --only statements", "fetch_financial_statements", "reparse"),
        ("zips-ingest --only insider", "fetch_insider_transactions", "reparse"),
        ("fundamentals-history-sec", "build_fundamentals_history", "rebuild_history"),
        ("identity --only propagate", "propagate_identity", "every_ticker"),
        ("edgar-index", "refresh", "rebuild"),
    ],
)
def test_full_reaches_the_existing_source_operation(
    command: str, target: str, keyword: str | None, context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
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
        stub_zips_step(monkeypatch, context.config, lambda ctx: ["AAA"])
        monkeypatch.setattr(step_zips, target, record)
    elif command == "edgar-index":
        monkeypatch.setattr(cli_mod.edgar_index, target, record)
        monkeypatch.setattr(cli_mod.edgar_index, "quarter_counts", lambda ctx: {})
    else:
        monkeypatch.setattr(cli_mod, target, record)
    context.config_dir = "configs"
    stub_edgar_step(monkeypatch, lambda ctx, **kwargs: edgar_calls.append(kwargs))
    runner = CliRunner()
    for flags in ([], ["-F"], ["--full"]):
        calls.clear()
        edgar_calls.clear()
        date_flags = [] if command == "fundamentals-history-sec" else ["--as-of", "2026-09-29"]
        result = runner.invoke(cli_mod.cli, [*command.split(), *date_flags, *flags])
        assert result.exit_code == 0, result.output
        if keyword is None:
            assert len(calls) == int(bool(flags)), "An existing roster is scraped only in full mode."
        else:
            assert len(calls) == 1 and calls[0][keyword] is bool(flags)
        assert not edgar_calls, "no command of this list walks EDGAR"
        if calls and "as_of" in calls[0]:
            assert calls[0]["as_of"] == pd.Timestamp("2026-09-29")
    context.store.delete.assert_not_called()
    print(f"SANITY: {command} resumes by default; -F and --full select the complete source operation; supported run dates reach the source.")


@pytest.mark.parametrize(("flags", "deletes"), [([], False), (["--full"], False), (["--rebuild"], True)])
def test_full_and_destructive_fundamentals_rebuild_remain_distinct(
    flags: list[str], deletes: bool, context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetch = Mock(return_value=SimpleNamespace())
    history = Mock()
    monkeypatch.setattr(cli_mod, "fetch_fundamentals_sec", fetch)
    monkeypatch.setattr(cli_mod, "fetched_filing_dates", lambda summary: {"AAA": [pd.Timestamp("2026-09-29")]})
    monkeypatch.setattr(cli_mod, "build_fundamentals_history", history)

    result = CliRunner().invoke(cli_mod.cli, ["fundamentals", "-t", "AAA", *flags])

    assert result.exit_code == 0, result.output
    assert fetch.call_args.kwargs["full"] is bool(flags)
    assert history.call_args.kwargs["full_fetch"] is bool(flags)
    assert history.call_args.kwargs["rebuild_history"] is deletes
    assert history.call_args.kwargs["fetched"] == {"AAA": [pd.Timestamp("2026-09-29")]}
    assert context.store.delete.call_args_list == (
        [call(table, {"ticker": "AAA"}) for table in (Tables.fundamentals_facts, Tables.fundamentals_history_sec, Tables.fundamentals_reason_codes)]
        if deletes
        else []
    )
    print(f"SANITY: fundamentals {flags or 'default'} reaches the expected replay mode; only --rebuild deletes the three SEC layers.")


def test_gap_check_uses_the_shared_report_option_and_keeps_its_default(context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    report = Mock()
    monkeypatch.setattr(cli_mod, "run_gap_check", report)
    runner = CliRunner()
    for flags, expected in (([], cli_mod.GAP_REPORT_PATH), (["--report", "custom.md"], "custom.md")):
        result = runner.invoke(cli_mod.cli, ["sharadar-gap-check", *flags])
        assert result.exit_code == 0, result.output
        assert report.call_args.kwargs["report_path"] == expected
        assert report.call_args.kwargs["tickers"] is None
    print("SANITY: --report selects the gap-check output path; an omitted path retains the existing report destination.")
