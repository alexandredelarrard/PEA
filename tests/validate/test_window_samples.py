"""Permanent scoped timing samples and observable persistence boundary checks."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pandas as pd
import pytest
from click.testing import CliRunner

from src.constants.constants_price import MACRO_ALL_SERIES
from src.context import Context
from src.data_aggregate.utils.common.incremental import write_part
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore
from src.utils.config import read_config
from src.utils.window import Window, WindowPlan, WindowRun
from src.validate.checks.window_sample import SampleSelection, capture_window_selection, check_window_sample, timing_sample_tables
from src.validate.cli import cli
from tests.canonical_schema import create_canonical
from tests.support.universe import universe_of
from tests.support.window_samples import WindowSamples


def _scoped_context(store: DataStore, tickers: list[str]) -> Context:
    return cast(
        Context,
        SimpleNamespace(
            store=store,
            config=read_config("configs"),
            universe=universe_of(tickers),
            run=WindowRun("incremental", "scoped sample storage fixture", pd.Timestamp("2026-11-16")),
        ),
    )


@pytest.mark.parametrize("table", timing_sample_tables(), ids=lambda table: table.name)
def test_pinned_table_cells(window_samples: WindowSamples, table: Table) -> None:
    result = window_samples.measure(table)
    if window_samples.manifest["tables"].get(table.name, {}).get("applicability") == "inactive":
        pytest.skip(result.reason)
    assert result.status == "pass", result.summary()
    print(f"SANITY: {table.name}: exact scoped PK/clock/value/null capture; producer economics remain separately measured.")


def test_sample_requires_explicit_scope(sqlite_store: DataStore) -> None:
    # Seed a known-truth storage fixture; this tests selection safety, not economics.
    df_rows = pd.DataFrame({"ticker": ["AAPL"], "date": pd.to_datetime(["2026-11-09"]), "feature": [1.0]})
    sqlite_store.replace(Tables.cube_part_text, df_rows)
    context = cast(Context, SimpleNamespace(store=sqlite_store))
    selection = SampleSelection(tuple(df_rows.columns), {})
    with pytest.raises(ValueError, match="explicit key/partition scope"):
        check_window_sample(context, Tables.cube_part_text, selection=selection)
    print("SANITY: an unscoped sample is refused before reading table rows.")


@pytest.mark.parametrize("table", [Tables.cube_signal, Tables.predictions, Tables.predictions_latest, Tables.trend_asset_returns, Tables.strategy])
def test_model_outputs_are_excluded_before_any_store_access(table: Table, tmp_path: Path) -> None:
    context = cast(Context, SimpleNamespace())
    selection = SampleSelection(tuple(table.pk), {table.pk[0]: "excluded"})
    with pytest.raises(ValueError, match="outputs are excluded"):
        check_window_sample(context, table, selection=selection)
    run = WindowSamples(context, tmp_path, tmp_path, time.perf_counter(), 300, True, {"tables": {}})
    with pytest.raises(ValueError, match="outputs are excluded"):
        run.measure(table)
    assert run.manifest["tables"] == {} and run.results == {}
    print(f"SANITY: {table.name} is excluded before schema, clock, value or artifact access and produces no model-validation result.")


def test_capture_selects_five_tickers_in_2026(sqlite_store: DataStore, tmp_path: Path) -> None:
    tickers = ["AAPL", "AMZN", "JPM", "MSFT", "XOM"]
    rows = pd.DataFrame(
        [
            {"ticker": ticker, "date": pd.Timestamp(day), "feature": float(index)}
            for index, ticker in enumerate(tickers)
            for day in ("2025-11-10", "2026-07-01")
        ]
    )
    sqlite_store.replace(Tables.cube_part_text, rows)
    context = cast(
        Context,
        SimpleNamespace(
            store=sqlite_store,
            config=read_config("configs"),
            universe=universe_of(tickers),
            run=WindowRun("incremental", "five-key sample contract", pd.Timestamp("2026-11-16")),
        ),
    )
    baseline, out = tmp_path / "capture", tmp_path / "out"
    baseline.mkdir()
    out.mkdir()
    run = WindowSamples(context, baseline, out, time.perf_counter(), 300, True, {"tables": {}})

    result = run.measure(Tables.cube_part_text)

    assert result.status == "pass", result.summary()
    captured = pd.DataFrame(result.metrics["records"])
    assert set(captured.ticker) == set(tickers)
    assert pd.to_datetime(captured.date).dt.year.eq(2026).all()
    assert len(captured) == 5
    print("SANITY: one repeatable2026 key per requested ticker is captured; older rows cannot satisfy the five-ticker baseline.")


def test_undated_companion_sample_uses_2026_parent_filing_dates(sqlite_store: DataStore, tmp_path: Path) -> None:
    tickers = ["AAPL", "AMZN", "JPM", "MSFT", "XOM"]
    create_canonical(sqlite_store.engine, [Tables.insider_transactions, Tables.insider_footnotes])
    filings = [
        {
            "accession_number": f"{ticker}-2026",
            "security_type": "nonderiv",
            "row_sequence": 1,
            "ticker": ticker,
            "filing_date": pd.Timestamp("2026-07-01"),
        }
        for ticker in tickers
    ]
    filings.append({**filings[0], "accession_number": "000-old", "filing_date": pd.Timestamp("2025-11-10")})
    sqlite_store.save(Tables.insider_transactions, pd.DataFrame(filings))
    notes = pd.DataFrame(
        [{"accession_number": row["accession_number"], "footnote_id": "F1", "footnote_text": "known source note"} for row in filings]
    )
    sqlite_store.save(Tables.insider_footnotes, notes)
    baseline, out = tmp_path / "baseline", tmp_path / "out"
    baseline.mkdir()
    out.mkdir()
    run = WindowSamples(_scoped_context(sqlite_store, tickers), baseline, out, time.perf_counter(), 300, True, {"tables": {}})

    result = run.measure(Tables.insider_footnotes)

    assert result.status == "pass", result.summary()
    captured = pd.DataFrame(result.metrics["records"])
    assert set(captured.accession_number) == {f"{ticker}-2026" for ticker in tickers}
    scope = run.manifest["tables"][Tables.insider_footnotes.name]["sample_scope"]
    assert scope["source_table"] == Tables.insider_transactions.name
    assert pd.to_datetime(scope["source_dates"]).year.tolist() == [2026] * 5
    print("SANITY: five undated companion units are sampled through actual2026 parent filings; older accessions cannot fill the quota.")


def test_macro_sample_uses_series_keys_instead_of_equity_tickers(sqlite_store: DataStore) -> None:
    series = list(MACRO_ALL_SERIES[:5])
    sqlite_store.replace(Tables.prices_macro, pd.DataFrame({"ticker": series, "date": pd.Timestamp("2026-07-01"), "close": 1.0}))
    entry = capture_window_selection(_scoped_context(sqlite_store, ["AAPL"]), Tables.prices_macro)
    assert "selection" in entry, entry
    assert set(entry["sample_scope"]["groups"]) == set(series)
    print("SANITY: macro samples select five actual series in 2026 independently of the equity ticker roster.")


def test_capture_reports_a_shortfall_without_smaller_or_older_substitutes(sqlite_store: DataStore, tmp_path: Path) -> None:
    rows = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "date": pd.to_datetime(["2026-07-01", "2025-11-10"]), "feature": [1.0, 2.0]})
    sqlite_store.replace(Tables.cube_part_text, rows)
    baseline, out = tmp_path / "baseline", tmp_path / "out"
    baseline.mkdir()
    out.mkdir()
    run = WindowSamples(
        _scoped_context(sqlite_store, ["AAPL", "AMZN", "JPM", "MSFT", "XOM"]), baseline, out, time.perf_counter(), 300, True, {"tables": {}}
    )

    result = run.measure(Tables.cube_part_text)

    assert result.status == "abstain" and "1 of 5" in result.reason
    entry = run.manifest["tables"][Tables.cube_part_text.name]
    assert "selection" not in entry and entry["sample_scope"]["groups"] == ["AAPL"]
    print("SANITY: insufficient2026 coverage is explicit; neither fewer tickers nor prior-year rows become a passing sample.")


def test_capture_cli_pins_once_and_refuses_to_overwrite(sqlite_store: DataStore, tmp_path: Path) -> None:
    tickers = ["AAPL", "AMZN", "JPM", "MSFT", "XOM"]
    rows = pd.DataFrame({"ticker": tickers, "date": pd.Timestamp("2026-07-01"), "feature": 1.0})
    sqlite_store.replace(Tables.cube_part_text, rows)
    context = _scoped_context(sqlite_store, tickers)
    args = ["window-capture", "-T", Tables.cube_part_text.name, "-o", str(tmp_path / "capture")]
    with patch("src.validate.cli.get_config_context", return_value=(context.config, context)):
        captured = CliRunner().invoke(cli, args)
    assert captured.exit_code == 0, captured.output
    target = tmp_path / "capture" / "_cache" / f"window-{Tables.cube_part_text.name}.json"
    before = target.read_bytes()
    with patch("src.validate.cli.get_config_context", side_effect=AssertionError("An existing pin cannot read new source data")):
        refused = CliRunner().invoke(cli, args)
    assert refused.exit_code == 1 and "already exists" in refused.output
    assert target.read_bytes() == before
    print("SANITY: the reusable capture CLI writes an immutable five-key pin and refuses replacement before reopening the database.")


def test_sample_preserves_values_and_nulls(sqlite_store: DataStore) -> None:
    # Seed exact parser/storage truth, including an observed zero and an unknown value.
    df_rows = pd.DataFrame({"ticker": ["AAPL", "AAPL"], "date": pd.to_datetime(["2026-11-09", "2026-11-10"]), "feature": [0.0, float("nan")]})
    sqlite_store.replace(Tables.cube_part_text, df_rows)
    context = cast(Context, SimpleNamespace(store=sqlite_store))
    selection = SampleSelection(tuple(df_rows.columns), {"ticker": "AAPL"}, "2026-11-09", "2026-11-10")
    result = check_window_sample(context, Tables.cube_part_text, selection=selection)
    assert result.status == "pass", result.summary()
    assert result.metrics["nulls"]["feature"] == 1
    assert result.metrics["records"][0]["feature"] == 0.0
    assert result.metrics["records"][1]["feature"] is None
    print("SANITY: scoped sample retains observed zero, unknown NULL, exact primary keys and inclusive date bounds.")


def test_source_identity_distinguishes_null_and_both_infinities(sqlite_store: DataStore) -> None:
    # Hash exact storage representations, even when display JSON uses null for nonfinite values.
    context = cast(Context, SimpleNamespace(store=sqlite_store))
    selection = SampleSelection(("ticker", "date", "feature"), {"ticker": "AAPL"})
    digests = set()
    for value in (float("nan"), float("inf"), float("-inf")):
        df_rows = pd.DataFrame({"ticker": ["AAPL"], "date": pd.to_datetime(["2026-11-09"]), "feature": [value]})
        sqlite_store.replace(Tables.cube_part_text, df_rows)
        result = check_window_sample(context, Tables.cube_part_text, selection=selection)
        assert result.status == "pass", result.summary()
        digests.add(result.metrics["sha256"])
    assert len(digests) == 3
    print("SANITY: NULL, positive infinity and negative infinity have three distinct pinned source identities.")


def test_changed_source_abstains_without_rewriting_baseline(sqlite_store: DataStore, tmp_path: Path) -> None:
    # Pin exact storage truth, then change one source cell through the normal store.
    tickers = ["AAPL", "AMZN", "JPM", "MSFT", "XOM"]
    df_rows = pd.DataFrame({"ticker": tickers, "date": pd.Timestamp("2026-11-09"), "feature": 1.0})
    sqlite_store.replace(Tables.cube_part_text, df_rows)
    context = cast(
        Context,
        SimpleNamespace(
            store=sqlite_store,
            config=read_config("configs"),
            universe=universe_of(tickers),
            run=WindowRun("incremental", "immutable grouped sample", pd.Timestamp("2026-11-16")),
        ),
    )
    baseline, out = tmp_path / "baseline", tmp_path / "out"
    baseline.mkdir()
    out.mkdir()
    run = WindowSamples(context, baseline, out, time.perf_counter(), 300, True, {"tables": {}})
    assert run.measure(Tables.cube_part_text).status == "pass"
    source = baseline / "manifest.json"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = json.loads(source.read_text(encoding="utf-8"))
    sqlite_store.replace(Tables.cube_part_text, df_rows.assign(feature=2.0))

    # A replay cannot quietly establish a new baseline or certify changed source cells.
    replay = WindowSamples(context, baseline, out, time.perf_counter(), 300, False, manifest)
    result = replay.measure(Tables.cube_part_text)
    assert result.status == "abstain", result.summary()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    print("SANITY: changed source values abstain; the pinned baseline remains byte-for-byte unchanged.")


def test_missing_replay_entry_never_discovers_new_source_keys(sqlite_store: DataStore, tmp_path: Path) -> None:
    # An incomplete manifest stays incomplete even when the source currently contains rows.
    df_rows = pd.DataFrame({"ticker": ["AAPL"], "date": pd.to_datetime(["2026-11-09"]), "feature": [1.0]})
    sqlite_store.replace(Tables.cube_part_text, df_rows)
    context = cast(Context, SimpleNamespace(store=sqlite_store))
    replay = WindowSamples(context, tmp_path, tmp_path, time.perf_counter(), 300, False, {"tables": {}})
    result = replay.measure(Tables.cube_part_text)
    assert result.status == "abstain", result.summary()
    assert replay.manifest["tables"] == {}
    assert not (tmp_path / "manifest.json").exists()
    print("SANITY: final replay refuses an unpinned table instead of discovering convenient replacement rows.")


@pytest.mark.parametrize("has_entry", [True, False])
def test_cli_missing_sample_publishes_abstention(tmp_path: Path, has_entry: bool) -> None:
    # A missing source is evidence, including when no database context can be opened.
    table = Tables.notes_embedding
    manifest = tmp_path / "manifest.json"
    entries = {table.name: {"reason": "Source table absent"}} if has_entry else {}
    manifest.write_text(json.dumps({"tables": entries}), encoding="utf-8")
    out = tmp_path / "out"
    with patch("src.validate.cli.get_config_context", side_effect=AssertionError("Absent sample must not open a database")):
        result = CliRunner().invoke(cli, ["window-sample", "-T", table.name, "--manifest", str(manifest), "-o", str(out)])
    assert result.exit_code == 3, result.output
    report = json.loads((out / "_out/window_sample.json").read_text(encoding="utf-8"))
    assert report["status"] == "abstain"
    print("SANITY: missing manifest/source sample publishes ABSTAIN with exit3 and does not open a database.")


def test_finite_part_rewrite_preserves_suffix_and_other_ticker(sqlite_store: DataStore) -> None:
    # Freeze an independent persistence oracle before replacing the legacy writer.
    dates = pd.to_datetime(["2026-11-06", "2026-11-09", "2026-11-13", "2026-11-16"])
    df_rows = pd.DataFrame({"date": dates.repeat(2), "ticker": ["AAPL", "MSFT"] * len(dates), "feature": range(8)})
    sqlite_store.replace(Tables.cube_part_text, df_rows)
    df_write = df_rows.loc[df_rows.ticker.eq("AAPL") & df_rows.date.between("2026-11-09", "2026-11-13")].assign(feature=10)
    requested = Window(dates[0], dates[2])
    window = WindowPlan(
        requested,
        requested,
        Window(dates[1], dates[2]),
        WindowRun("incremental", "finite AAPL correction", requested.end),
        dates[2],
        (("ticker", ("AAPL",)),),
    )

    # Run the actual current public writer; P0 records its expected safety failure.
    write_part(sqlite_store, Tables.cube_part_text, df_write, window)
    df_actual = sqlite_store.load(Tables.cube_part_text)
    assert df_actual is not None
    df_protected = df_rows.loc[~(df_rows.ticker.eq("AAPL") & df_rows.date.between("2026-11-09", "2026-11-13"))]
    keys = ["date", "ticker"]
    df_actual = df_actual.set_index(keys).sort_index()
    df_protected = df_protected.set_index(keys).sort_index()
    print(
        f"SANITY: independently protected {len(df_protected)} rows; actual writer retained {int(df_actual.index.isin(df_protected.index).sum())} of them."
    )
    pd.testing.assert_frame_equal(df_actual.reindex(df_protected.index), df_protected, check_dtype=False)
    print("SANITY: finite AAPL rewrite protects both suffix/prefix dates and all MSFT rows.")
