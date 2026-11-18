"""Known-truth governance schema support through the native bounded writer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pandas as pd
import pytest

from src.context import Context
from src.data_aggregate.transformers import step_cube_governance as module
from src.data_aggregate.transformers.step_cube_governance import StepCubeGovernance
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.window import Window, WindowRun
from tests.support.config import fixture_config
from tests.support.universe import universe_of
from tests.support.windows import window_plan


def _owner(store: DataStore, monkeypatch: pytest.MonkeyPatch) -> tuple[StepCubeGovernance, list[pd.DataFrame]]:
    # Older source support expires before the deliberately bounded calculation grid.
    days = pd.DatetimeIndex(["2019-01-02", "2019-01-03", "2020-01-02", "2021-01-04", "2026-01-02", "2026-11-16"], name="date")
    store.replace(
        Tables.cube_part_prices,
        pd.DataFrame({"date": days, "ticker": "AAA", "close_split": 100.0, "close_total": 100.0, "shares_outstanding_pit": 1e8}),
    )
    store.save(
        Tables.def14a_llm,
        pd.DataFrame(
            {
                "ticker": ["AAA"] * 4,
                "accession_number": ["historical", "carry", "expired", "current"],
                "as_of": pd.to_datetime(["2019-01-01", "2020-01-01", "2021-01-01", "2026-01-01"]),
                "period": pd.to_datetime(["2018-12-31", "2019-12-31", "2020-12-31", "2025-12-31"]),
                "board_size": [9.0, 9.0, 9.0, 12.0],
                "n_directors": [9.0, 9.0, 9.0, 12.0],
                "pct_female_directors": [0.4, None, None, None],
            }
        ),
    )
    config = fixture_config()
    context = SimpleNamespace(
        store=store,
        config=config,
        config_dir=Path("configs"),
        universe=universe_of(["AAA"]),
        run=WindowRun("full", "known-truth schema support", days[-1]),
    )
    step = StepCubeGovernance(cast(Context, context), config)
    monkeypatch.setattr(module, "load_peers_or_raise", lambda *_args: {})
    monkeypatch.setattr(
        module,
        "plan_window",
        lambda context, *_args: window_plan(
            Window(days[0], days[-1]),
            scope=("AAA",),
            read_start=days[0] if context.run.mode == "full" else days[3],
            write_start=days[0] if context.run.mode == "full" else days[4],
            frontier=None if context.run.mode == "full" else days[4],
            mode=context.run.mode,
        ),
    )

    # Observe physical columns from the unchanged native domain builder, including support probes.
    emitted: list[pd.DataFrame] = []
    native = module.build_governance_feature_panel

    def observe(*args: Any, **kwargs: Any) -> tuple[pd.DataFrame, dict[str, int]]:
        panel, tally = native(*args, **kwargs)
        emitted.append(panel.copy())
        return panel, tally

    monkeypatch.setattr(module, "build_governance_feature_panel", observe)
    step.run()
    context.run = WindowRun("incremental", "known-truth schema support", days[-1])
    return step, emitted


def test_historical_only_column_survives_bounded_governance_write(filing_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    step, emitted = _owner(filing_store, monkeypatch)
    table = Tables.cube_part_governance
    before = filing_store.load(table)
    assert before is not None
    expected_columns = set(emitted[0])
    assert {"date", "ticker", "f_board_size", "f_pct_female_directors"} <= expected_columns
    support = emitted[0].loc[emitted[0].f_pct_female_directors.notna(), "date"]
    assert support.tolist() == list(pd.to_datetime(["2019-01-02", "2019-01-03", "2020-01-02"]))

    # The finite writer repairs an incorrect current cell while preserving an actual old-row witness.
    damaged = before.copy()
    damaged.loc[damaged.date.eq(pd.Timestamp("2026-11-16")), "f_board_size"] = 3.0
    filing_store.replace(table, damaged)
    clean = Mock(wraps=module.clean_governance_inputs)
    monkeypatch.setattr(module, "clean_governance_inputs", clean)
    step.run()
    assert clean.call_count == 1
    assert "f_pct_female_directors" not in emitted[1]
    after = filing_store.load(table)
    assert after is not None and set(after) == expected_columns
    pd.testing.assert_frame_equal(after, before, check_dtype=False)
    assert after.loc[after.date.ge(pd.Timestamp("2026-01-02")), "f_pct_female_directors"].isna().all()
    print(
        "SANITY: native FULL emits f_pct_female_directors on 2019-01-02..2020-01-02; the 2021+ native panel omits it. "
        "The actual bounded writer retains that column as unknown, restores the current board cell, and preserves the historical rows."
    )
    print(f"SANITY: exact FULL columns={sorted(map(str, expected_columns))}; bounded native columns={sorted(map(str, emitted[1]))}.")


@pytest.mark.parametrize("drift", ["unknown", "removed", "new", "source-loss"])
def test_unproven_governance_schema_drift_preserves_saved_rows(filing_store: DataStore, monkeypatch: pytest.MonkeyPatch, drift: str) -> None:
    step, _ = _owner(filing_store, monkeypatch)
    table = Tables.cube_part_governance
    previous = filing_store.load(table)
    assert previous is not None
    if drift == "unknown":
        filing_store.replace(table, previous.assign(f_unknown=7.0))
    if drift == "removed":
        native = module.build_governance_feature_panel

        def removed(*args: Any, **kwargs: Any) -> tuple[pd.DataFrame, dict[str, int]]:
            panel, tally = native(*args, **kwargs)
            return panel.drop(columns="f_pct_female_directors", errors="ignore"), tally

        monkeypatch.setattr(module, "build_governance_feature_panel", removed)
    if drift in {"new", "source-loss"}:
        sources = filing_store.load(Tables.def14a_llm)
        assert sources is not None
        field, value = ("ceo_is_founder", 1.0) if drift == "new" else ("pct_female_directors", float("nan"))
        sources[field] = value
        filing_store.save(Tables.def14a_llm, sources)
    before = filing_store.load(table)
    assert before is not None
    with pytest.raises(ValueError, match="stored columns differ|column set changed|explicit.*rebuild"):
        step.run()
    after = filing_store.load(table)
    assert after is not None
    pd.testing.assert_frame_equal(after, before)
    print(f"SANITY: {drift} schema drift requires an explicit rebuild; the strict writer refuses and every saved row survives.")
