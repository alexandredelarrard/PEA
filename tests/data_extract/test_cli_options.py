"""Shared extraction options parse into the intended source operations without source reads or writes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, call

import pandas as pd
import pytest
from click.testing import CliRunner

from src.data_extract import cli as cli_mod
from src.data_store.schema import Tables


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
        ("financial-notes", "fetch_financial_notes", "reparse"),
        ("financial-statements", "fetch_financial_statements", "reparse"),
        ("insider-zip", "fetch_insider_transactions", "reparse"),
        ("insider-transactions", "fetch_insider_transactions", "reparse"),
        ("fundamentals-history-sec", "build_fundamentals_history", "rebuild_history"),
        ("identity-propagate", "propagate_identity", "every_ticker"),
        ("edgar-index", "refresh", "rebuild"),
    ],
)
def test_full_reaches_the_existing_source_operation(
    command: str, target: str, keyword: str | None, context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    edgar_calls: list[dict[str, Any]] = []

    def record(*args: Any, **kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(removals=pd.DataFrame())

    if command == "edgar-index":
        monkeypatch.setattr(cli_mod.edgar_index, target, record)
        monkeypatch.setattr(cli_mod.edgar_index, "quarter_counts", lambda ctx: {})
    else:
        monkeypatch.setattr(cli_mod, target, record)
    monkeypatch.setattr(cli_mod, "fetch_insider_edgar", lambda ctx, **kwargs: edgar_calls.append(kwargs))
    runner = CliRunner()
    for flags in ([], ["-F"], ["--full"]):
        calls.clear()
        edgar_calls.clear()
        date_flags = [] if command == "fundamentals-history-sec" else ["--as-of", "2026-09-29"]
        result = runner.invoke(cli_mod.cli, [command, *date_flags, *flags])
        assert result.exit_code == 0, result.output
        if keyword is None:
            assert len(calls) == int(bool(flags)), "An existing roster is scraped only in full mode."
        else:
            assert len(calls) == 1 and calls[0][keyword] is bool(flags)
        if command == "insider-transactions":
            assert len(edgar_calls) == 1 and edgar_calls[0]["full"] is bool(flags)
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
    print("SANITY: --report selects the gap-check output path; an omitted path retains the existing report destination.")
