"""Known-truth source-publication and preserved filling checks through the governance producer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from src.context import Context
from src.data_aggregate.transformers import step_cube_governance as module
from src.data_aggregate.transformers.step_cube_governance import StepCubeGovernance
from src.data_aggregate.utils.price.frames import PriceFrames
from src.data_store.schema import Tables
from src.utils.window import Window, WindowPlan, WindowRun
from tests.canonical_schema import create_canonical
from tests.support.config import fixture_config
from tests.support.universe import universe_of
from tests.support.windows import window_plan


class _ReadyInputsError(RuntimeError):
    pass


def _fixture(store: Any) -> tuple[StepCubeGovernance, PriceFrames, WindowPlan]:
    config = fixture_config()
    create_canonical(store.engine, [Tables.def14a_llm, Tables.def14a_executive_comp, Tables.def14a_directors])
    store.save(
        Tables.def14a_llm,
        pd.DataFrame(
            {
                "ticker": ["AAA", "AAA"],
                "accession_number": ["early", "late"],
                "as_of": pd.to_datetime(["2026-05-04", "2026-10-05"]),
                "period": pd.to_datetime(["2025-12-31", "2025-12-31"]),
                "fiscal_year_extract": [2025, 2025],
                "board_size": [10.0, 10.0],
                "n_directors": [10.0, 10.0],
                "ceo_name_proxy": ["Jane Chief", "Jane Chief"],
                "ceo_total_comp": [None, 2_000_000.0],
                "ceo_salary": [None, 1_000_000.0],
                "ceo_since_year": [None, 2010.0],
                "ceo_is_founder": [None, 1.0],
                "ceo_age": [None, 56.0],
            }
        ),
    )
    store.save(
        Tables.def14a_executive_comp,
        pd.DataFrame(
            {
                "ticker": ["AAA"],
                "accession_number": ["late"],
                "as_of": pd.to_datetime(["2026-10-05"]),
                "name": ["Jane Chief"],
                "fiscal_year": [2025],
                "total": [2_000_000.0],
                "salary": [1_000_000.0],
            }
        ),
    )
    store.save(
        Tables.def14a_directors,
        pd.DataFrame(
            {
                "ticker": ["AAA", "AAA"],
                "accession_number": ["early", "late"],
                "name": ["Pat Director", "Pat Director"],
                "as_of": pd.to_datetime(["2026-05-04", "2026-10-05"]),
                "age": [None, 50.0],
                "is_independent": [True, True],
            }
        ),
    )
    sessions = pd.DatetimeIndex(["2026-05-01", "2026-05-04", "2026-05-05", "2026-05-06", "2026-10-05", "2026-10-06", "2026-11-16"], name="date")
    values = pd.DataFrame({"AAA": 100.0}, index=sessions)
    frames = PriceFrames(sessions, ("AAA",), {}, close_split=values, close_total=values, shares_outstanding_pit=values * 1_000_000.0)
    context = cast(
        Context,
        SimpleNamespace(
            store=store,
            config=config,
            config_dir=Path("configs"),
            universe=universe_of(["AAA"]),
            run=WindowRun("full", "controlled2026 governance clock", sessions[-1]),
        ),
    )
    plan = window_plan(Window(sessions[0], sessions[-1]), scope=("AAA",))
    return StepCubeGovernance(context, config), frames, plan


def _inputs(step: StepCubeGovernance, frames: PriceFrames, plan: WindowPlan, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    observed: dict[str, Any] = {}

    def capture(*args: Any, **kwargs: Any) -> None:
        observed.update(def14a=args[0], **kwargs)
        raise _ReadyInputsError

    monkeypatch.setattr(module, "build_governance_feature_panel", capture)
    with pytest.raises(_ReadyInputsError):
        step._governance_panel(frames, plan)
    return observed


def test_governance_prepares_next_session_inputs_once(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    step, frames, plan = _fixture(sqlite_store)
    prepared = _inputs(step, frames, plan, monkeypatch)
    assert pd.to_datetime(prepared["def14a"].as_of).tolist() == list(pd.to_datetime(["2026-05-05", "2026-10-06"]))
    assert pd.to_datetime(prepared["exec_comp"].as_of).tolist() == [pd.Timestamp("2026-10-06")]
    print(
        "SANITY: the real governance producer supplies parent and compensation inputs only on the next actual session, applying their inherited buffer once."
    )


@pytest.mark.parametrize("field", ["ceo_total_comp", "ceo_salary", "ceo_since_year", "ceo_is_founder", "ceo_age"])
def test_later_proxy_still_fills_earlier_feature_inputs(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    step, frames, plan = _fixture(sqlite_store)
    prepared = _inputs(step, frames, plan, monkeypatch)
    observations = prepared["def14a"].set_index("accession_number")
    assert pd.notna(observations.loc["early", field]), field
    assert pd.notna(observations.loc["late", field]), field
    print(f"SANITY: {field} retains the existing later-filing fill, while destination filing rows receive their configured publication buffer.")


def test_director_age_keeps_existing_later_disclosure_fill(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    step, frames, plan = _fixture(sqlite_store)
    prepared = _inputs(step, frames, plan, monkeypatch)
    observations = prepared["directors"].set_index("accession_number")
    assert observations.loc["early", "age"] == 50.0
    assert observations.loc["late", "age"] == 50.0
    print("SANITY: the existing per-person birth-year fill retains the earlier director age, as explicitly requested.")


@pytest.mark.parametrize("child", ["exec_comp", "directors", "director_comp"])
def test_proxy_children_keep_donors_without_scalar_parents(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, child: str) -> None:
    step, frames, plan = _fixture(sqlite_store)
    tables = (Tables.def14a_executive_comp, Tables.def14a_directors, Tables.def14a_director_comp)
    create_canonical(sqlite_store.engine, [Tables.def14a_director_comp])
    monkeypatch.setattr(step._context, "universe", universe_of(["AAA", "BBB"]))
    values = pd.DataFrame({"AAA": 100.0, "BBB": 100.0}, index=frames.trading_index)
    frames = PriceFrames(
        frames.trading_index, ("AAA", "BBB"), {}, close_split=values, close_total=values, shares_outstanding_pit=values * 1_000_000.0
    )
    for table in (Tables.def14a_llm, *tables):
        item = step._context.config.data.tables[table.name]
        item.start_date, item.source_start, item.end_date = "2026-01-01", "2026-01-01", "2026-10-31"

    # A completed child remains useful when its parent carries only extraction metadata.
    sqlite_store.delete(Tables.def14a_llm, where={"accession_number": "late"})
    sqlite_store.save(
        Tables.def14a_llm,
        pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "accession_number": ["late", "child-only"],
                "as_of": pd.to_datetime(["2026-10-05", "2026-10-05"]),
                "retry": [2, 2],
            }
        ),
    )
    df_children = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "AAA", "AAA", "AAA", "OTHER"],
            "accession_number": ["late", "child-only", "before-source", "after-source", "future", "foreign-scope"],
            "as_of": pd.to_datetime(["2026-10-05", "2026-10-05", "2025-12-31", "2026-11-02", "2026-12-01", "2026-10-05"]),
        }
    )
    sqlite_store.save(Tables.def14a_executive_comp, df_children.assign(name="Jane Chief", fiscal_year=2025, total=2_000_000.0, salary=1_000_000.0))
    sqlite_store.save(Tables.def14a_directors, df_children.assign(name="Pat Director", age=50.0, is_independent=True))
    sqlite_store.save(Tables.def14a_director_comp, df_children.assign(name="Pat Director", total=250_000.0, fees_earned=250_000.0))

    prepared = _inputs(step, frames, plan, monkeypatch)

    df_parent = prepared["def14a"].set_index("accession_number")
    assert df_parent.index.tolist() == ["early"], "metadata-only parents must remain excluded from scalar features"
    df_child = prepared[child]
    assert df_child is not None, f"{child} lost valid child history because the scalar parent has no evidence"
    expected = {("AAA", "late"), ("BBB", "child-only")}
    if child == "directors":
        expected.add(("AAA", "early"))
    assert set(zip(df_child.ticker, df_child.accession_number, strict=True)) == expected
    assert set(pd.to_datetime(df_child.as_of)) <= {pd.Timestamp("2026-05-05"), pd.Timestamp("2026-10-06")}
    if child == "exec_comp":
        assert df_parent.loc["early", "ceo_total_comp"] == 2_000_000.0
        assert df_parent.loc["early", "ceo_salary"] == 1_000_000.0
    elif child == "directors":
        assert df_child.set_index(["ticker", "accession_number"]).loc[("AAA", "early"), "age"] == 50.0
    else:
        assert df_child.total.eq(250_000.0).all()
    print(
        f"SANITY: {child} keeps same-ticker later donors and Universe child-only events without scalar parents; "
        "source hard dates and ticker scope exclude old/future/foreign rows, and filing buffers still apply once."
    )
