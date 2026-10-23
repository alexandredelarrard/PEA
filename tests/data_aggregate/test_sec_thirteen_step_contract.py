"""Independent broad and elite 13F merge ownership."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.data_aggregate.transformers.step_cube_sec_thirteen import StepCubeSecThirteen
from src.data_aggregate.utils.common import inputs as capital_inputs
from src.data_aggregate.utils.sec_thirteen.inputs import load_filing_vintages
from src.data_store.schema import Tables
from tests.data_aggregate.test_institutional_step_contract import context, seed_prices


def test_sec_thirteen_executes_broad_and_elite_once(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    days = seed_prices(sqlite_store)
    ctx = context(sqlite_store, tmp_path)
    step = StepCubeSecThirteen(ctx, ctx.config)
    monkeypatch.setattr(capital_inputs, "load_peers_or_raise", lambda ctx: {})
    calls = []

    def panel(name: str) -> pd.DataFrame:
        calls.append(name)
        return pd.DataFrame({"date": days, "ticker": "AAA", "f_" + name: 1.0})

    monkeypatch.setattr(step, "_institutional_panel", lambda *args, **kwargs: panel("ic_inst_test"))
    monkeypatch.setattr(step, "_superinvestor_panel", lambda *args, **kwargs: panel("ic_super_test"))
    result = step.build_panel()
    assert step._store is sqlite_store and step._part.table == Tables.cube_part_sec_thirteen
    assert calls == ["ic_inst_test", "ic_super_test"]
    assert set(result.columns) == {"date", "ticker", "f_ic_inst_test", "f_ic_super_test"}
    assert not result.duplicated(["date", "ticker"]).any()
    print("SANITY: SEC thirteen activates broad and elite 13F exactly once with a dedicated part owner.")


def test_completed_accession_markers_and_positions_share_one_read() -> None:
    calls = []
    table = Tables.sec13f_hr
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

    result = load_filing_vintages(Store(), pd.Timestamp("2024-02-20"), ["1"])
    assert result is not None
    filings, holdings = result
    assert len(filings) == len(holdings) == 1 and holdings.iloc[0]["shares"] == 10.0
    assert len(calls) == 1 and calls[0][1]["markers"] is True
    assert calls[0][1]["where"] == {"parse_status": ["complete", "position"], "cik": ["0000000001"]}
    print("SANITY: completed empty-book evidence and whole-book positions come from one projected marker-inclusive stream.")
