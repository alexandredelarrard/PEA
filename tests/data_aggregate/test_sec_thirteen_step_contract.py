"""Independent broad and elite 13F merge ownership."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import pandas as pd
import pytest

from src.data_aggregate.transformers import step_cube_sec_thirteen as step_module
from src.data_aggregate.transformers.step_cube_sec_thirteen import StepCubeSecThirteen
from src.data_aggregate.utils.common import inputs as capital_inputs
from src.data_aggregate.utils.common.incremental import PartWindow
from src.data_aggregate.utils.sec_thirteen.inputs import load_filing_vintages
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from tests.data_aggregate.test_institutional_step_contract import context, seed_prices


def test_sec_thirteen_executes_broad_and_elite_once(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    days = seed_prices(sqlite_store)
    ctx = context(sqlite_store, tmp_path)
    step = StepCubeSecThirteen(ctx, ctx.config)
    monkeypatch.setattr(capital_inputs, "load_peers_or_raise", lambda ctx: {})
    broad = Mock(
        return_value=pd.DataFrame({"period": [pd.Timestamp("2023-09-30")], "as_of": [days[0]], "ticker": ["AAA"], "ic_inst_concentration": [0.4]})
    )
    elite = Mock(return_value={"ic_super_holders": pd.DataFrame({"AAA": [0.5]}, index=days[:1])})
    monkeypatch.setattr(step, "_institutional_features", broad)
    monkeypatch.setattr(step, "_superinvestor_levels", elite)
    result = step.build_panel()
    assert step._store is sqlite_store and step._part.table == Tables.cube_part_sec_thirteen
    assert broad.call_count == elite.call_count == 1
    assert set(result.columns) == {"date", "ticker", "f_ic_inst_concentration", "f_ic_super_holders"}
    assert result.f_ic_inst_concentration.dropna().eq(pd.Series([0.4], dtype="float32").iloc[0]).all()
    assert result.f_ic_super_holders.dropna().eq(0.5).all()
    assert not result.duplicated(["date", "ticker"]).any()
    print("SANITY: SEC thirteen activates broad and elite 13F exactly once with a dedicated part owner.")


def test_completed_accession_markers_and_positions_share_one_read() -> None:
    calls = []
    table = Tables.sec13f_hr
    assert table.empty_marker is not None
    manifest = {
        "accession_number": "original",
        "cik": "0000000001",
        "period": "2023-12-31",
        "filing_date": "2024-02-14",
        "parse_status": "complete",
        "cusip": table.empty_marker[1],
    }
    position = {**manifest, "parse_status": "position", "cusip": "123456789", "shares": 10.0, "value_usd": 100.0}

    class Store:
        def columns(self, table: Any) -> list[str]:
            return [
                "accession_number",
                "cik",
                "period",
                "filing_date",
                "accepted_at",
                "amendment_type",
                "amendment_number",
                "report_type",
                "restricted",
                "parsed_row_count",
                "parse_status",
            ]

        def iter_load(self, table: Any, *, columns: list[str], **kwargs: Any) -> Any:
            calls.append((table, kwargs))
            return iter([pd.DataFrame([manifest, position]).reindex(columns=columns)])

    result = load_filing_vintages(cast(DataStore, Store()), pd.Timestamp("2024-02-20"), ["1"])
    assert result is not None
    filings, holdings = result
    assert len(filings) == len(holdings) == 1 and holdings.iloc[0]["shares"] == 10.0
    assert len(calls) == 1 and calls[0][1]["markers"] is True
    assert calls[0][1]["where"] == {"parse_status": ["complete", "position"], "cik": ["0000000001"]}
    print("SANITY: completed empty-book evidence and whole-book positions come from one projected marker-inclusive stream.")


def test_manager_accessions_use_one_projected_load() -> None:
    table = Tables.sec13f_manager_holdings
    assert table.empty_marker is not None
    required = [
        "accession_number",
        "cik",
        "period",
        "filing_date",
        "accepted_at",
        "amendment_type",
        "amendment_number",
        "report_type",
        "restricted",
        "parsed_row_count",
        "parse_status",
    ]
    rows = pd.DataFrame(
        [
            {
                "accession_number": "original",
                "cik": "0000000001",
                "period": "2023-12-31",
                "filing_date": "2024-02-14",
                "parse_status": "complete",
                "cusip": table.empty_marker[1],
            },
            {
                "accession_number": "original",
                "cik": "0000000001",
                "period": "2023-12-31",
                "filing_date": "2024-02-14",
                "parse_status": "position",
                "cusip": "123456789",
                "shares": 10.0,
                "value_usd": 100.0,
            },
        ]
    )
    calls = []

    class Store:
        def columns(self, actual: Any) -> list[str]:
            assert actual == table
            return required

        def load(self, actual: Any, *, columns: list[str], **kwargs: Any) -> pd.DataFrame:
            calls.append((actual, columns, kwargs))
            return rows.reindex(columns=columns)

        def iter_load(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("The bounded manager table should use ordinary load, not chunk iteration")

    result = load_filing_vintages(cast(DataStore, Store()), pd.Timestamp("2024-02-20"), ["1"], table=table)
    assert result is not None
    filings, holdings = result
    assert len(filings) == len(holdings) == 1 and holdings.shares.iloc[0] == 10.0
    assert len(calls) == 1 and calls[0][0] == table
    assert calls[0][2]["markers"] is True
    assert calls[0][2]["where"] == {"parse_status": ["complete", "position"], "cik": ["0000000001"]}
    assert calls[0][2]["date_col"] == "filing_date" and calls[0][2]["until"] == pd.Timestamp("2024-02-20")
    print("SANITY: manager markers and positions use one scoped projected ordinary load, with identical public-cutoff filtering.")


def test_schema_change_reuses_the_computed_panel(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    days = seed_prices(sqlite_store)
    ctx = context(sqlite_store, tmp_path)
    step = StepCubeSecThirteen(ctx, ctx.config)
    previous = pd.DataFrame({"date": days, "ticker": "AAA", "f_previous": 1.0})
    sqlite_store.save(Tables.cube_part_sec_thirteen, previous)
    panel = previous.drop(columns="f_previous").assign(f_current=2.0)
    calls = []

    def build(window: PartWindow | None = None, *, existing_columns: Sequence[str] | None = None) -> pd.DataFrame:
        assert window is not None and existing_columns is not None
        calls.append(window.is_full)
        return panel

    monkeypatch.setattr(step, "build_panel", build)
    monkeypatch.setattr(step_module, "load_trading_calendar", lambda store: days)
    step.run()
    assert calls == [False]
    persisted = sqlite_store.load(Tables.cube_part_sec_thirteen)
    assert persisted is not None
    pd.testing.assert_frame_equal(persisted.sort_values(["date", "ticker"]).reset_index(drop=True), panel.reset_index(drop=True), check_dtype=False)
    print("SANITY: a changed feature schema replaces the part once using the already computed full-history panel.")
