"""13F ticker scope limits filing work while preserving complete source books."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from click.testing import CliRunner

import src.data_extract.cli as extract_cli
from src.data_extract.utils.institutionals import fetch_13f as f13
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.tickers_universe.universe import UniverseError
from src.utils.window import Window, WindowRun
from src.validate.checks.institutionals import check_institutionals
from tests.data_extract.institutionals.test_13f_low_watermark import _AS_OF, _ctx, _filing, _patch, _seed_hr
from tests.data_extract.institutionals.test_13f_one_walk import CMAP, _FakeFiling, _line


@pytest.mark.parametrize(("options", "expected"), [(["-t", "aapl"], ["AAPL"]), ([], ["AAPL", "MSFT"]), (["-t", "aapl", "-F"], ["AAPL"])])
def test_cli_restores_validated_ticker_scope(
    tmp_path: Path, filing_store: DataStore, monkeypatch: pytest.MonkeyPatch, options: list[str], expected: list[str]
) -> None:
    ctx = _ctx(tmp_path, filing_store)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(extract_cli, "get_config_context", lambda *args, **kwargs: (ctx.config, ctx))
    monkeypatch.setattr(extract_cli, "fetch_13f", lambda context, **kwargs: calls.append(kwargs))

    result = CliRunner().invoke(extract_cli.cli, ["thirteen-f", *options])

    assert result.exit_code == 0, result.output
    assert calls == [{"tickers": expected, "filing_window": None}]
    print("SANITY: thirteen-f accepts -t and passes the normalized, universe-validated production scope.")


@pytest.mark.parametrize("full", [False, True])
def test_subset_stops_at_stored_filing_frontier(tmp_path: Path, filing_store: DataStore, monkeypatch: pytest.MonkeyPatch, full: bool) -> None:
    ctx = _ctx(tmp_path, filing_store)
    ctx.run = WindowRun("full" if full else "incremental", "13F scope regression", _AS_OF)
    _seed_hr(filing_store, "0000000009", "2026-09-20")
    inside, after = _filing(1, "2026-09-19"), _filing(2, "2026-09-25")
    windows, _ = _patch(monkeypatch, [inside, after])

    f13.fetch_13f(ctx, tickers=["AAPL"])

    assert windows == [Window(pd.Timestamp("1999-07-01"), pd.Timestamp("2026-09-20"))]
    assert inside.reads == [inside.accession_number] and not after.reads
    print("SANITY: a ticker subset keeps the shared configured start and stops filing work at the stored frontier.")


def test_cli_rejects_unknown_tickers_before_source_work(tmp_path: Path, filing_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx(tmp_path, filing_store)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(extract_cli, "get_config_context", lambda *args, **kwargs: (ctx.config, ctx))
    monkeypatch.setattr(extract_cli, "fetch_13f", lambda context, **kwargs: calls.append(kwargs))

    result = CliRunner().invoke(extract_cli.cli, ["thirteen-f", "-t", "UNKNOWN"])

    assert result.exit_code != 0 and isinstance(result.exception, UniverseError)
    assert calls == []
    print("SANITY: an unknown ticker fails universe validation before the fetcher or a provider is called.")


@pytest.mark.parametrize("tickers", [None, ["AAPL", "MSFT"]])
def test_full_universe_can_advance_beyond_the_stored_frontier(
    tmp_path: Path, filing_store: DataStore, monkeypatch: pytest.MonkeyPatch, tickers: list[str] | None
) -> None:
    ctx = _ctx(tmp_path, filing_store)
    _seed_hr(filing_store, "0000000009", "2026-09-20")
    recent = _filing(1, "2026-09-25")
    windows, _ = _patch(monkeypatch, [recent])

    f13.fetch_13f(ctx, tickers=tickers)

    assert windows == [Window(pd.Timestamp("1999-07-01"), _AS_OF)] and recent.reads == [recent.accession_number]
    print("SANITY: omitted tickers and an explicit full universe both retain the configured run-date filing range.")


def test_cold_subset_uses_the_bounded_run_date(tmp_path: Path, filing_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx(tmp_path, filing_store)
    ctx.config.data.tables.sec13f_hr.start_date = "2026-09-18"
    ctx.config.data.tables.sec13f_hr.end_date = "2026-10-02"
    recent = _filing(1, "2026-09-29")
    windows, _ = _patch(monkeypatch, [recent])

    f13.fetch_13f(ctx, tickers=["AAPL"])

    assert windows == [Window(pd.Timestamp("2026-09-18"), _AS_OF)] and recent.reads == [recent.accession_number]
    print("SANITY: a cold subset has no stored ceiling and reaches the root run date inside its fixed bounds.")


def test_inactive_subset_returns_before_store_or_provider_io(tmp_path: Path, filing_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx(tmp_path, filing_store)
    ctx.config.data.tables.sec13f_hr.start_date = "2027-01-01"

    def unexpected_io(*args: Any, **kwargs: Any) -> None:
        pytest.fail("an inactive 13F window must not query a frontier or initialize provider work")

    monkeypatch.setattr(filing_store, "max_date", unexpected_io)
    monkeypatch.setattr(ctx, "ensure_edgar_identity", unexpected_io)
    monkeypatch.setattr(f13, "configure", unexpected_io)

    assert f13.fetch_13f(ctx, tickers=["AAPL"]) == 0
    print("SANITY: a configured start after the root date returns before any frontier or provider I/O.")


@pytest.mark.parametrize(
    "case",
    [
        ("2026-09-20", "2026-09-28", "2026-09-30", "2026-09-20"),
        ("2026-09-29", "2026-09-24", "2026-09-30", "2026-09-24"),
        ("2026-09-29", "2026-09-28", "2026-09-26", "2026-09-26"),
        ("2026-09-17", "2026-09-28", "2026-09-30", None),
    ],
)
def test_subset_ceiling_respects_fixed_bounds_and_empty_intersection(
    tmp_path: Path, filing_store: DataStore, monkeypatch: pytest.MonkeyPatch, case: tuple[str, str, str, str | None]
) -> None:
    stored, end, as_of, expected = case
    ctx = _ctx(tmp_path, filing_store)
    ctx.run = WindowRun("incremental", "bounded 13F scope", pd.Timestamp(as_of))
    ctx.config.data.tables.sec13f_hr.start_date = "2026-09-18"
    ctx.config.data.tables.sec13f_hr.end_date = end
    _seed_hr(filing_store, "0000000009", stored)
    before, inside, after = (_filing(n, day) for n, day in enumerate(["2026-09-17", "2026-09-19", "2026-09-29"], 1))
    windows, _ = _patch(monkeypatch, [before, inside, after])

    f13.fetch_13f(ctx, tickers=["AAPL"])

    assert windows == ([] if expected is None else [Window(pd.Timestamp("2026-09-18"), pd.Timestamp(expected))])
    assert not before.reads and not after.reads
    assert bool(inside.reads) == (expected is not None)
    print("SANITY: the stored ceiling, configured end and root date only narrow scope; a ceiling below the fixed start does no source work.")


@pytest.mark.parametrize("full", [False, True])
def test_explicit_filing_window_bypasses_the_subset_ceiling(
    tmp_path: Path, filing_store: DataStore, monkeypatch: pytest.MonkeyPatch, full: bool
) -> None:
    ctx = _ctx(tmp_path, filing_store)
    ctx.run = WindowRun("full" if full else "incremental", "explicit 13F recovery", _AS_OF)
    _seed_hr(filing_store, "0000000009", "2026-09-20")
    recent = _filing(1, "2026-09-25")
    windows, _ = _patch(monkeypatch, [recent])
    requested = Window(pd.Timestamp("2026-09-24"), pd.Timestamp("2026-09-26"))

    f13.fetch_13f(ctx, tickers=["AAPL"], filing_window=requested)

    assert windows == [requested] and recent.reads == [recent.accession_number]
    print("SANITY: an explicit filing window bypasses the stored subset ceiling in normal and full replay modes.")


@pytest.mark.parametrize(
    ("requested", "configured_end"),
    [
        (Window(pd.Timestamp("2026-09-17"), _AS_OF), "2026-10-02"),
        (Window(pd.Timestamp("2026-09-18"), pd.Timestamp("2026-10-01")), "2026-10-02"),
        (Window(pd.Timestamp("2026-09-18"), pd.Timestamp("2026-09-28")), "2026-09-25"),
    ],
)
def test_explicit_subset_window_cannot_cross_config_or_root_bounds(
    tmp_path: Path, filing_store: DataStore, monkeypatch: pytest.MonkeyPatch, requested: Window, configured_end: str
) -> None:
    ctx = _ctx(tmp_path, filing_store)
    ctx.config.data.tables.sec13f_hr.start_date = "2026-09-18"
    ctx.config.data.tables.sec13f_hr.end_date = configured_end
    windows, _ = _patch(monkeypatch, [])

    with pytest.raises(ValueError, match="configured hard bounds and the run date"):
        f13.fetch_13f(ctx, tickers=["AAPL"], filing_window=requested)

    assert windows == []
    print("SANITY: explicit subset recovery rejects a window outside the configured bounds or root date before listing.")


def test_subset_retains_the_complete_book_and_maps_every_cusip(tmp_path: Path, filing_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx(tmp_path, filing_store)
    _seed_hr(filing_store, "0000000009", "2026-09-20")
    filing = _FakeFiling(
        "0000000001",
        "2026-09-19",
        "2026-06-30",
        [_line("037833100", "APPLE INC", 1000.0, 10), _line("594918104", "MICROSOFT", 2000.0, 20), _line("G0450A105", "NON SP500", 3000.0, 30)],
        accession="0000000001-26-000001",
    )
    _patch(monkeypatch, [filing])
    lookups: list[set[str]] = []

    def mapped(context: Any, cusips: list[str], **kwargs: Any) -> pd.DataFrame:
        lookups.append(set(cusips))
        return CMAP

    monkeypatch.setattr(f13, "build_cusip_ticker_map", mapped)
    f13.fetch_13f(ctx, tickers=["AAPL"])

    stored = filing_store.load(Tables.sec13f_hr, columns=["cusip"], where={"cik": filing.cik})
    expected = {"037833100", "594918104", "G0450A105"}
    assert stored is not None and set(stored["cusip"]) == expected and lookups == [expected]
    validation = check_institutionals(
        ctx,
        ciks=[filing.cik],
        start_period="2026-06-30",
        end_period="2026-06-30",
        filed_through="2026-09-30",
        trading_index=pd.bdate_range("2026-06-01", "2026-10-01"),
    )
    assert validation.status == "pass", validation.summary()
    assert validation.metrics["accessions"] == 1 and validation.metrics["holding_rows"] == 3
    print("SANITY: -t AAPL keeps AAPL, MSFT and the non-universe security, maps all three CUSIPs and passes the read-only source validator.")
