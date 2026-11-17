"""Schema-driven extraction freshness report (per-key share, exit 0) and recent-tail replay."""

from __future__ import annotations

import json
import logging
from datetime import date
from types import SimpleNamespace
from typing import Any, Literal, cast

import pandas as pd
import pytest
from click.testing import CliRunner

from src.data_extract import cli as extraction_cli
from src.data_extract.utils.prices import fetch_prices as prices_fetcher
from src.data_store.schema import Tables
from src.utils.freshness import fresh_share, table_freshness
from src.utils.tickers_universe.universe import Universe
from src.utils.window import Window, WindowRun
from tests.data_extract.fake_context import extract_config
from tests.support.store_identity import store_identity
from tests.support.universe import seed_universe, universe_config


class _Context(SimpleNamespace):
    """A stub context whose universe is the stored `universe_tickers`, read on each access."""

    @property
    def universe(self) -> Universe:
        tickers = self.store.load(Tables.universe_tickers, columns=["ticker"])["ticker"].tolist()
        return Universe(self.store, universe_config(tickers))


def _context(store) -> Any:
    return _Context(
        store=store,
        log=logging.getLogger("test.extraction_freshness"),
        config=extract_config(),
        run=WindowRun("incremental", "freshness sample", pd.Timestamp("2026-09-27")),
    )


def _prices(store: Any, last_by_ticker: dict[str, str]) -> None:
    """A universe of the given tickers, each with one `prices` row on its own last date."""
    if store.exists(Tables.universe_tickers):
        store.drop(Tables.universe_tickers)
    seed_universe(store, list(last_by_ticker))
    store.replace(
        Tables.prices,
        pd.DataFrame({"ticker": list(last_by_ticker), "date": pd.to_datetime(list(last_by_ticker.values())), "close_split": 10.0}),
    )


def test_gate_reads_table_cadence_and_date_column_from_config(sqlite_store, monkeypatch):
    _prices(sqlite_store, {"AAA": "2026-09-25"})
    monkeypatch.setattr(extraction_cli, "freshness_tables", lambda config: (Tables.prices,))

    green = extraction_cli._extraction_status_report(cast(Any, _context(sqlite_store)))
    status = cast(dict[str, Any], cast(dict[str, object], green["tables"])[Tables.prices.name])
    assert green["ok"]
    assert status["date_column"] == "date"
    assert status["cadence"] == "daily"

    _prices(sqlite_store, {"AAA": "2026-09-20"})
    red = extraction_cli._extraction_status_report(cast(Any, _context(sqlite_store)))
    assert not red["ok"] and red["behind"] == [Tables.prices.name]

    print("\n=== SANITY CHECK: schema-driven extraction gate ===")
    print("  prices at 2026-09-25 -> GREEN; prices at 2026-09-20 -> RED")
    print("  OK: the gate reads its table, date column and cadence directly from data.yml")


def test_notes_tables_declare_monthly_archive_availability():
    for table in (Tables.notes_num, Tables.notes_text):
        assert table.date_col == "ddate"
        assert extract_config().data.tables[table.name].freshness.cadence == "monthly"
        assert extract_config().data.tables[table.name].freshness.clock == "available_at"
        assert "available_at" in table.date_type_cols

    print("\n=== SANITY CHECK: notes registry contract ===")
    print("  notes_num and notes_text retain ddate as data time and use date-typed available_at for monthly freshness. Validated.")


@pytest.mark.parametrize("unit", ["us", "ns"])
@pytest.mark.parametrize(
    "latest, expected_age, expected_ok",
    [(date.min, 739908, False), (date.max, -2912150, True), (date(3006, 9, 30), -357917, True), (None, None, False)],
    ids=["ancient", "future", "observed-13f-date", "missing"],
)
def test_status_handles_full_calendar_date_range(
    sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, unit: Literal["us", "ns"], latest: date | None, expected_age: int | None, expected_ok: bool
) -> None:
    _prices(sqlite_store, {"AAA": "2026-10-19"})
    table = Tables.sec13f_hr
    if latest is not None:
        sqlite_store.replace(
            table,
            pd.DataFrame({"cik": ["0000000001"], "period": [latest], "accession_number": ["a1"], "ticker": ["AAA"], "cusip": ["123456789"]}),
        )
    monkeypatch.setattr(extraction_cli, "freshness_tables", lambda config: (table,))
    as_of = pd.Timestamp("2026-10-20").as_unit(unit)

    context = _context(sqlite_store)
    context.run = WindowRun("incremental", "full calendar dates", as_of)
    report = extraction_cli._extraction_status_report(context)
    status = cast(dict[str, Any], cast(dict[str, object], report["tables"])[table.name])
    assert status["max_date"] == (None if latest is None else latest.isoformat())
    assert status["age_days"] == expected_age and report["ok"] is expected_ok
    assert status["fresh_share"] == (None if latest is None else float(expected_ok))
    assert table_freshness(context, table, as_of) == expected_age
    if latest is not None:
        assert fresh_share({"AAA": pd.Timestamp(latest)}, ["AAA"], as_of, status["max_age_days"]) == float(expected_ok)
    json.dumps(report)

    print(f"\nSANITY CHECK: latest={latest}, as_of resolution={unit} -> age={expected_age}; status serializes without overflow.")


def test_dividends_replay_the_same_recent_tail_as_prices(sqlite_store, monkeypatch):
    as_of = pd.Timestamp("2026-09-30")
    last = pd.Timestamp("2026-09-28")
    sqlite_store.replace(Tables.prices, pd.DataFrame({"ticker": ["AAA"], "date": [last], "close_split": [1.0]}))
    sqlite_store.replace(Tables.dividends, pd.DataFrame({"ticker": ["AAA"], "date": [last], "dividends": [0.0]}))
    seed_universe(sqlite_store, ["AAA"])
    captured: list[pd.Timestamp] = []

    def _download(tickers, window: Window, *args, **kwargs):
        assert kwargs["actions"] is True
        del tickers, args, kwargs
        captured.append(window.start)
        return pd.DataFrame()

    monkeypatch.setattr(prices_fetcher, "download_ohlcv", _download)
    # Price anchor checks have dedicated controls in test_price_restatement_anchor.py.
    monkeypatch.setattr(prices_fetcher, "anchor_restated_keys", lambda *args, **kwargs: [])
    context = _context(sqlite_store)
    context._identity_resolver = store_identity(sqlite_store)
    context.run = WindowRun("incremental", "dividend overlap", as_of)
    prices_fetcher.fetch_prices(context, prices_fetcher.YahooSession(pause=0.0), ["AAA"])

    overlap = pd.Timedelta(days=context.config.data.tables.prices_dividends.reprocess.overlap.value)
    assert captured == [last - overlap]

    print("\n=== SANITY CHECK: dividend repair window ===")
    print(f"  prices and dividends at {last.date()} -> one download from {(last - overlap).date()} (the {overlap.days}-day overlap)")
    print("  OK: a recent interior dividend-grid hole can self-heal on the next run")


def test_report_is_green_when_every_key_is_fresh_and_red_with_exit_0_when_one_key_lags(sqlite_store, monkeypatch, caplog):
    context = _context(sqlite_store)
    monkeypatch.setattr(extraction_cli, "freshness_tables", lambda config: (Tables.prices, Tables.sec_13d))
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
    monkeypatch.setattr(extraction_cli, "freshness_tables", lambda config: (Tables.prices,))
    report = extraction_cli._extraction_status_report(cast(Any, _context(sqlite_store)))
    status = cast(dict[str, Any], cast(dict[str, object], report["tables"])[Tables.prices.name])
    assert status["fresh_share"] == share and report["ok"] is ok

    print(f"\n=== SANITY CHECK: prediction-input share {share} ===")
    print(f"  prediction_fresh_share 0.95 -> {'GREEN' if ok else 'RED'} (the threshold is inclusive). Validated.")


def test_configured_freshness_clock_and_tolerance_reach_actual_status(sqlite_store: Any) -> None:
    _prices(sqlite_store, {"AAA": "2026-09-20"})
    context = _context(sqlite_store)
    item = context.config.data.tables.prices
    initial = extraction_cli._table_status(context, Tables.prices, ["AAA"], context.run.as_of, 0.95)
    assert not initial["ok"] and initial["max_age_days"] == 4
    context.config.data.freshness.max_age_days.daily = 10
    changed = extraction_cli._table_status(context, Tables.prices, ["AAA"], context.run.as_of, 0.95)
    assert changed["ok"] and changed["max_age_days"] == 10
    table = Tables.sec13f_hr
    item = context.config.data.tables[table.name]
    sqlite_store.replace(
        table,
        pd.DataFrame(
            {
                "cik": ["0000000001"],
                "period": [pd.Timestamp("2026-01-01")],
                "accession_number": ["a1"],
                "ticker": ["AAA"],
                "cusip": ["123456789"],
                "filing_date": [pd.Timestamp("2026-09-26")],
            }
        ),
    )
    assert extraction_cli._table_status(context, table, ["AAA"], context.run.as_of, 0.95)["age_days"] == 269
    item.freshness.clock = "filing_date"
    shifted = extraction_cli._table_status(context, table, ["AAA"], context.run.as_of, 0.95)
    assert shifted["age_days"] == 1 and shifted["date_column"] == "filing_date"
    print("SANITY: configured freshness clock and tolerance change the actual report without a schema/default override.")
