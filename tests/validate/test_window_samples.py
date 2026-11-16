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

from src.context import Context
from src.data_aggregate.utils.common.incremental import PartWindow, write_part
from src.data_store.schema import ALL, Table, Tables
from src.data_store.store import DataStore
from src.validate.checks.window_sample import SampleSelection, check_window_sample
from src.validate.cli import cli
from tests.support.window_samples import WindowSamples


@pytest.mark.parametrize("table", ALL, ids=lambda table: table.name)
def test_pinned_table_cells(window_samples: WindowSamples, table: Table) -> None:
    result = window_samples.measure(table)
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
    df_rows = pd.DataFrame({"ticker": ["AAPL"], "date": pd.to_datetime(["2026-11-09"]), "feature": [1.0]})
    sqlite_store.replace(Tables.cube_part_text, df_rows)
    context = cast(Context, SimpleNamespace(store=sqlite_store))
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
    table = Tables.predictions_latest
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
    window = PartWindow(pd.Timestamp("2026-11-13"), pd.Timestamp("2026-11-06"), pd.Timestamp("2026-11-09"))

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
