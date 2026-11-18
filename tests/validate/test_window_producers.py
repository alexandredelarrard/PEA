"""Known-truth adapter controls; financial acceptance uses pinned native cases."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.db import get_engine
from src.utils.window import Window
from src.validate.result import CheckResult
from tests.support.config import fixture_config
from tests.support.window_producers import ProducerStore, compare_producer_rows, measure_window_producer, producer_cases, reference_config
from tests.support.window_samples import WindowSamples


def test_reference_changes_only_feature_input_history() -> None:
    config = fixture_config()
    before = OmegaConf.to_container(config, resolve=True)
    full = reference_config(config, (Tables.cube_part_text, Tables.cube))
    assert set(full.data.tables.cube_part_text.warmup.values()) == {None}
    assert dict(full.data.tables.cube.warmup) == dict(config.data.tables.cube.warmup)
    assert set(full.data.tables.cube.warmup.values()) == {0}
    assert full.data_aggregate == config.data_aggregate
    assert full.data.tables.cube_part_text.overlap == config.data.tables.cube_part_text.overlap
    assert OmegaConf.to_container(config, resolve=True) == before
    print(
        "SANITY: full reference extends declared feature inputs only; native science, assembly zero budgets and accepted candidate remain unchanged."
    )


def test_output_route_preserves_other_keys_and_rolls_back(tmp_path: Path) -> None:
    output = DataStore(get_engine("sqlite:///" + str((tmp_path / "adapter.sqlite").resolve()).replace("\\", "/")))
    table = Tables.cube_part_text
    original = pd.DataFrame(
        {"date": pd.to_datetime(["2025-01-02", "2026-01-02", "2026-01-02"]), "ticker": ["A", "A", "B"], "feature": [10.0, 20.0, 30.0]}
    )
    output.replace(table, original)
    record: dict[str, Any] = {"reads": [], "writes": []}
    route = ProducerStore(SimpleNamespace(), output, (table,), {table: ["A"]}, record)
    window = Window(pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-02"))
    candidate = original.loc[window.mask(original.date)].copy()
    candidate["feature"] += 100
    route.replace_window(table, candidate, window, {"ticker": ["A", "B"]})
    saved = output.load(table).sort_values(["date", "ticker"]).reset_index(drop=True)
    assert saved.feature.tolist() == [10.0, 120.0, 30.0]
    with pytest.raises(RuntimeError, match="source mutation"):
        route.replace_window(Tables.earnings_surprises, candidate, window, {})
    with pytest.raises(AttributeError):
        route.save(table, candidate)
    with pytest.raises(RuntimeError, match="rollback probe"):
        with route.transaction() as transaction:
            transaction.replace_window(table, original.loc[window.mask(original.date)], window, {"ticker": ["A"]})
            raise RuntimeError("rollback probe")
    pd.testing.assert_frame_equal(saved, output.load(table).sort_values(["date", "ticker"]).reset_index(drop=True))
    assert record["writes"][0]["observed_keys"] == ["A"]
    print(
        "SANITY: real disposable replacement repairs its selected key, preserves older and unrelated rows, refuses source writes, and rolls back a failed transaction."
    )


def test_comparison_checks_first_row_nulls_and_keys() -> None:
    expected = pd.DataFrame({"date": pd.to_datetime(["2025-12-31", "2026-01-02"]), "ticker": ["A", "A"], "f_value": [2.0, float("nan")]})
    assert compare_producer_rows(expected, expected.copy(), Tables.cube_part_prices)["finite_cells"] == 1
    for changed in (expected.assign(f_value=[3.0, float("nan")]), expected.assign(f_value=[2.0, 0.0]), expected.assign(ticker=["B", "A"])):
        with pytest.raises(AssertionError):
            compare_producer_rows(expected, changed, Tables.cube_part_prices)
    print("SANITY: a wrong first boundary value, changed unknown mask and changed identity all fail despite an unchanged later row.")


def test_existing_case_evidence_is_not_overwritten(tmp_path: Path) -> None:
    directory = tmp_path / "producer-prices"
    directory.mkdir()
    path = directory / "producer.json"
    payload = '{"status":"original evidence"}'
    path.write_text(payload, encoding="utf-8")
    run = SimpleNamespace(context=SimpleNamespace(), capturing=False, identity={}, out=tmp_path, producers={}, publish=lambda: None)
    result = measure_window_producer(run, "prices")
    assert result.status == "abstain"
    assert path.read_text(encoding="utf-8") == payload
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "original evidence"
    print("SANITY: repeated invocation names its unavailable fresh case directory and preserves prior evidence byte-for-byte.")


def test_case_selector_uses_the_existing_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from tests.support import run_window_samples as worker

    seen = []
    run = SimpleNamespace()

    @contextmanager
    def session(path: Path, out: Path, capturing: bool, started: float) -> Any:
        assert path == tmp_path / "baseline" and out == tmp_path / "out" and not capturing and started > 0
        yield run

    def measure(actual: Any, case: str) -> CheckResult:
        assert actual is run
        seen.append(case)
        return CheckResult.abstained("window_producer", case, "Required real cache absent")

    monkeypatch.setattr(worker, "window_sample_session", session)
    monkeypatch.setattr(worker, "measure_window_producer", measure)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_window_samples",
            "--window-baseline",
            str(tmp_path / "baseline"),
            "--window-out",
            str(tmp_path / "out"),
            "--window-producer",
            "fundamentals",
        ],
    )
    assert worker.main() == 3 and seen == ["fundamentals"]
    print("SANITY: the existing canonical session/deadline receives one producer case and propagates missing-cache ABSTAIN as exit3.")


def test_comparison_preserves_mixed_numeric_and_categorical_payloads() -> None:
    expected = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "ticker": ["A", "A"],
            "f_value": [2.0, float("nan")],
            "peers": [{"B": 0.5}, None],
            "gics_sector": ["Technology", "Technology"],
            "eligible": [True, None],
            "is_primary": [True, False],
        }
    )
    measured = compare_producer_rows(expected, expected.copy(), Tables.cube)
    assert measured["finite_cells"] == 1
    for changed in (
        expected.assign(f_value=[3.0, float("nan")]),
        expected.assign(peers=[{"B": 0.7}, None]),
        expected.assign(gics_sector=["Financials", "Technology"]),
        expected.assign(eligible=[False, None]),
        expected.assign(peers=[{"B": 0.5}, {}]),
    ):
        with pytest.raises(AssertionError):
            compare_producer_rows(expected, changed, Tables.cube)
    print("SANITY: mixed cube numeric values use tolerance while JSON, categories, booleans and their null masks remain exact.")


@pytest.mark.parametrize("case", producer_cases())
def test_pinned_cube_producer(window_samples: WindowSamples, case: str) -> None:
    result = measure_window_producer(window_samples, case)
    assert result.status == "pass", result.summary()
    print(f"SANITY: {case}: actual native full/bounded first-write and fixed2026 output, overwrite restoration and exact unchanged rerun.")
