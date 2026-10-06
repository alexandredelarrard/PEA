"""Schema-driven extraction freshness report (per-key share, exit 0) and recent-tail replay."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from click.testing import CliRunner

from src.data_extract import cli as extraction_cli
from src.data_extract.utils.prices import fetch_prices as prices_fetcher
from src.data_store.schema import Resume, Tables


def _context(store) -> Any:
    return SimpleNamespace(
        store=store,
        log=logging.getLogger("test.extraction_freshness"),
        config=SimpleNamespace(data_extract=SimpleNamespace(redundant_ticks=[], prediction_fresh_share=0.95)),
    )


def _prices(store: Any, last_by_ticker: dict[str, str]) -> None:
    """A universe of the given tickers, each with one `prices` row on its own last date."""
    store.replace(Tables.sp500_tickers, pd.DataFrame({"ticker": list(last_by_ticker)}))
    store.replace(
        Tables.prices,
        pd.DataFrame({"ticker": list(last_by_ticker), "date": pd.to_datetime(list(last_by_ticker.values())), "close_split": 10.0}),
    )


def test_gate_reads_table_cadence_and_date_column_from_schema(sqlite_store, monkeypatch):
    _prices(sqlite_store, {"AAA": "2026-09-25"})
    monkeypatch.setattr(extraction_cli, "freshness_tables", lambda: (Tables.prices,))

    green = extraction_cli._extraction_status_report(cast(Any, _context(sqlite_store)), as_of=pd.Timestamp("2026-09-27"))
    status = cast(dict[str, Any], cast(dict[str, object], green["tables"])[Tables.prices.name])
    assert green["ok"]
    assert status["date_column"] == Tables.prices.freshness_col
    assert status["cadence"] == Tables.prices.freshness

    _prices(sqlite_store, {"AAA": "2026-09-20"})
    red = extraction_cli._extraction_status_report(cast(Any, _context(sqlite_store)), as_of=pd.Timestamp("2026-09-27"))
    assert not red["ok"] and red["behind"] == [Tables.prices.name]

    print("\n=== SANITY CHECK: schema-driven extraction gate ===")
    print("  prices at 2026-09-25 -> GREEN; prices at 2026-09-20 -> RED")
    print("  OK: the gate reads its table, date column and cadence directly from schema.py")


def test_notes_tables_declare_monthly_archive_availability():
    for table in (Tables.notes_num, Tables.notes_text):
        assert table.date_col == "ddate"
        assert table.freshness == "monthly"
        assert table.freshness_col == "available_at"
        assert "available_at" in table.date_type_cols

    print("\n=== SANITY CHECK: financial-notes registry contract ===")
    print("  notes_num and notes_text retain ddate as data time and use date-typed available_at for monthly freshness. Validated.")


def test_dividends_replay_the_same_recent_tail_as_prices(sqlite_store, monkeypatch):
    as_of = pd.Timestamp("2026-09-30")
    last = pd.Timestamp("2026-09-28")
    sqlite_store.replace(Tables.prices, pd.DataFrame({"ticker": ["AAA"], "date": [last], "close_split": [1.0]}))
    sqlite_store.replace(Tables.dividends, pd.DataFrame({"ticker": ["AAA"], "date": [last], "dividends": [0.0]}))
    captured: list[pd.Timestamp] = []

    def _download(tickers, since, until, *args, **kwargs):
        del tickers, until, args, kwargs
        captured.append(pd.Timestamp(since))
        return pd.DataFrame()

    monkeypatch.setattr(prices_fetcher, "download_ohlcv", _download)
    prices_fetcher.fetch_prices_and_actions(_context(sqlite_store), ["AAA"], years_history=15, as_of=as_of, pause=0.0)

    overlap = pd.Timedelta(days=cast(Resume, Tables.dividends.resume).overlap_days)
    assert captured == [last - overlap]

    print("\n=== SANITY CHECK: dividend repair window ===")
    print(f"  prices and dividends at {last.date()} -> one download from {(last - overlap).date()} (the {overlap.days}-day overlap)")
    print("  OK: a recent interior dividend-grid hole can self-heal on the next run")


def test_report_is_green_when_every_key_is_fresh_and_red_with_exit_0_when_one_key_lags(sqlite_store, monkeypatch, caplog):
    context = _context(sqlite_store)
    monkeypatch.setattr(extraction_cli, "freshness_tables", lambda: (Tables.prices, Tables.sec_13d))
    monkeypatch.setattr(extraction_cli, "get_config_context", lambda *args, **kwargs: (None, context))
    sqlite_store.replace(
        Tables.sec_13d, pd.DataFrame({"ticker": ["AAA"], "filing_date": pd.to_datetime(["2026-09-26"]), "accession_number": ["a1"], "rp_seq": [0]})
    )

    def _run() -> dict[str, Any]:
        result = CliRunner().invoke(extraction_cli.cli, ["extraction-status", "-c", "configs", "--as-of", "2026-09-27"])
        assert result.exit_code == 0, result.output
        return json.loads(result.output.strip().splitlines()[-1])

    _prices(sqlite_store, {"AAA": "2026-09-25", "BBB": "2026-09-25"})
    green = _run()
    assert green["ok"] and green["tables"]["prices"]["fresh_share"] == 1.0
    assert green["tables"]["sec_13d"]["fresh_share"] == 0.5, "an event table reports its share and stays GREEN on its table-wide age"

    _prices(sqlite_store, {"AAA": "2026-09-25", "BBB": "2026-09-10"})  # the table-wide max is still fresh
    with caplog.at_level(logging.WARNING, logger="test.extraction_freshness"):
        red = _run()
    prices = red["tables"]["prices"]
    assert not red["ok"] and red["behind"] == ["prices"]
    assert prices["age_days"] == 2 and prices["fresh_share"] == 0.5
    assert any("Extraction freshness RED: prices" in record.getMessage() for record in caplog.records)

    print("\n=== SANITY CHECK: per-key freshness report (AC-008) ===")
    print("  AAA and BBB fresh -> GREEN, share 1.0; BBB 17 days old while the table max is 2 days old -> RED, share 0.5")
    print("  the command still exits 0, prints the JSON report and logs one WARNING per RED table. Validated.")


@pytest.mark.parametrize("share, ok", [(0.95, True), (0.9, False)])
def test_prediction_inputs_turn_red_below_the_configured_share(sqlite_store, monkeypatch, share, ok):
    last = {f"T{i:02d}": "2026-09-25" for i in range(20)}
    last["T00"] = "2026-09-01"
    if share < 0.95:
        last["T01"] = "2026-09-01"
    _prices(sqlite_store, last)
    monkeypatch.setattr(extraction_cli, "freshness_tables", lambda: (Tables.prices,))
    report = extraction_cli._extraction_status_report(cast(Any, _context(sqlite_store)), as_of=pd.Timestamp("2026-09-27"))
    status = cast(dict[str, Any], cast(dict[str, object], report["tables"])[Tables.prices.name])
    assert status["fresh_share"] == share and report["ok"] is ok

    print(f"\n=== SANITY CHECK: prediction-input share {share} ===")
    print(f"  prediction_fresh_share 0.95 -> {'GREEN' if ok else 'RED'} (the threshold is inclusive). Validated.")
