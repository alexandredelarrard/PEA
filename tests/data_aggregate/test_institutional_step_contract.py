from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import src.data_aggregate.transformers.step_cube_institutionals as step_module
from src.data_aggregate.transformers.step_cube_institutionals import StepCubeInstitutionals
from src.data_aggregate.utils.common.incremental import COLUMNS_CHANGED, PartWindow
from src.data_aggregate.utils.institutionals.sink import ConditioningSink
from src.data_store.schema import Tables


def _bare_step() -> StepCubeInstitutionals:
    step = object.__new__(StepCubeInstitutionals)
    step._log = logging.getLogger(__name__)
    return step


def test_input_loaders_keep_full_price_calendar_and_exact_share_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    step = _bare_step()
    config = object()
    context = object()
    peers = {"AAA": {"BBB": 1.0}}
    price_frames = object()
    shares = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "as_of": [pd.Timestamp("2026-01-02")],
            "sharesOutstanding": [100.0],
            "sharesOutstandingPit": [90.0],
        }
    )
    calls: dict[str, Any] = {}

    class _Store:
        def load(self, table: object, columns: list[str], *, optional: bool) -> pd.DataFrame:
            calls["shares"] = {"table": table, "columns": columns, "optional": optional}
            return shares

    store = _Store()
    step._context = context
    step._config = config
    step._store = store

    def load_peers(actual_context: object, actual_config: object) -> dict[str, dict[str, float]]:
        calls["peers"] = {"context": actual_context, "config": actual_config}
        return peers

    def load_prices(actual_store: object, *, peers: object, fields: object, since: object) -> object:
        calls["prices"] = {"store": actual_store, "peers": peers, "fields": fields, "since": since}
        return price_frames

    monkeypatch.setattr(step_module, "load_peers_or_raise", load_peers)
    monkeypatch.setattr(step_module, "load_price_frames", load_prices)

    assert step._load_frames() is price_frames
    assert step._load_shares_out() is shares
    assert calls == {
        "peers": {"context": context, "config": config},
        "prices": {
            "store": store,
            "peers": peers,
            "fields": ("close_split", "close_total", "volume", "level_factor", "sector_ret", "ret"),
            "since": None,
        },
        "shares": {
            "table": Tables.fundamentals_history,
            "columns": ["ticker", "as_of", "sharesOutstanding", "sharesOutstandingPit"],
            "optional": True,
        },
    }
    print("SANITY: price loading kept the full calendar and six exact fields; shares loaded both required bases with an optional read.")


def test_grid_restriction_is_on_exact_date_ticker_pairs(caplog: pytest.LogCaptureFixture) -> None:
    step = _bare_step()
    first, second = pd.to_datetime(["2026-01-02", "2026-01-05"])
    long = pd.DataFrame(
        {
            "date": [first, first, second, first],
            "ticker": ["AAA", "BBB", "AAA", "ZZZ"],
            "_grid": [1.0, 1.0, pd.NA, pd.NA],
            "feature": [1.5, pd.NA, 7.0, 9.0],
        }
    )

    with caplog.at_level(logging.WARNING):
        got = step._restrict_to_grid(long)

    assert list(got.columns) == ["date", "ticker", "feature"]
    assert list(got[["date", "ticker"]].itertuples(index=False, name=None)) == [
        (first, "AAA"),
        (first, "BBB"),
    ]
    assert got["feature"].iloc[0] == 1.5
    assert pd.isna(got["feature"].iloc[1])
    assert "dropped 2 row(s) off the price grid" in caplog.text
    assert "1 of them appear nowhere" in caplog.text
    print("SANITY: grid restriction kept exact date/ticker pairs and removed both an off-date pair and an unknown ticker.")


def test_build_panel_preserves_order_sink_and_output_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    step = _bare_step()
    store = object()
    step._store = store
    step._cfg = {}
    step._part = SimpleNamespace(warmup_trading_days=390)

    first, second = pd.to_datetime(["2026-01-02", "2026-01-05"])
    trading_index = pd.DatetimeIndex([first, second])
    grid = pd.DataFrame({"date": [first, second], "ticker": ["AAA", "BBB"]})

    class _Frames:
        universe = ["AAA", "BBB"]

        @staticmethod
        def skeleton() -> pd.DataFrame:
            return grid.copy()

    frames = _Frames()
    shares = pd.DataFrame({"ticker": ["AAA"], "as_of": [first]})
    splits = pd.DataFrame({"ticker": ["AAA"], "date": [first], "ratio": [2.0]})
    window = PartWindow(last=first, since=first, refresh_from=second)

    panels = {
        "institutional": pd.DataFrame(
            {
                "date": [first, second],
                "ticker": ["AAA", "BBB"],
                "f_inst": pd.Series([1.25, pd.NA], dtype="Float32"),
            }
        ),
        "superinvestor": pd.DataFrame(
            {
                "date": [first, second],
                "ticker": ["AAA", "BBB"],
                "f_super": pd.Series([0, pd.NA], dtype="Int64"),
            }
        ),
        "insider": pd.DataFrame({"date": [first], "ticker": ["AAA"], "f_insider": [0.0]}),
        "short": pd.DataFrame({"date": [second], "ticker": ["BBB"], "f_short": [2.5]}),
        "ownership": pd.DataFrame({"date": [first, second], "ticker": ["AAA", "AAA"], "f_owner": [3.5, 99.0]}),
        "conditioning": pd.DataFrame({"date": [first], "ticker": ["AAA"], "f_condition": [4.5]}),
        "cross": pd.DataFrame({"date": [second], "ticker": ["BBB"], "f_cross": [5.5]}),
    }

    calls: list[str] = []
    sinks: list[ConditioningSink] = []
    loads = {"frames": 0, "shares": 0, "splits": 0, "calendar": 0}
    planned: dict[str, Any] = {}

    def load_calendar(actual_store: object) -> pd.DatetimeIndex:
        assert actual_store is store
        loads["calendar"] += 1
        return trading_index

    def plan(actual_store: object, table: object, **kwargs: Any) -> PartWindow:
        planned.update(store=actual_store, table=table, **kwargs)
        return window

    def load_frames() -> _Frames:
        loads["frames"] += 1
        return frames

    def load_shares() -> pd.DataFrame:
        loads["shares"] += 1
        return shares

    def load_source(table: object, universe: object = None) -> pd.DataFrame:
        assert table is Tables.prices_splits
        assert universe is None
        loads["splits"] += 1
        return splits

    def emit(name: str, sink: ConditioningSink | None = None) -> pd.DataFrame:
        calls.append(name)
        if sink is not None:
            sinks.append(sink)
        return panels[name]

    monkeypatch.setattr(step_module, "load_trading_calendar", load_calendar)
    monkeypatch.setattr(step_module, "plan_window", plan)
    monkeypatch.setattr(step, "_load_frames", load_frames)
    monkeypatch.setattr(step, "_load_shares_out", load_shares)
    monkeypatch.setattr(step, "_load_source", load_source)
    monkeypatch.setattr(
        step,
        "_institutional_panel",
        lambda actual_frames, actual_shares, actual_splits: (
            emit("institutional")
            if (actual_frames is frames and actual_shares is shares and actual_splits is splits)
            else pytest.fail("institutional inputs changed")
        ),
    )
    monkeypatch.setattr(
        step,
        "_superinvestor_panel",
        lambda actual_frames, actual_shares, actual_splits, sink: (
            emit("superinvestor", sink)
            if (actual_frames is frames and actual_shares is shares and actual_splits is splits)
            else pytest.fail("superinvestor inputs changed")
        ),
    )
    monkeypatch.setattr(step, "_insider_panel", lambda actual_frames, actual_shares, sink: emit("insider", sink))
    monkeypatch.setattr(
        step,
        "_short_flow_panel",
        lambda actual_frames, actual_shares, actual_splits, sink: emit("short", sink),
    )
    monkeypatch.setattr(step, "_ownership_panel", lambda actual_frames, sink: emit("ownership", sink))
    monkeypatch.setattr(step, "_conditioning_panel", lambda actual_frames, actual_splits, sink: emit("conditioning", sink))
    monkeypatch.setattr(step, "_cross_source_panel", lambda actual_frames, sink: emit("cross", sink))

    got, got_window = step.build_panel(full=True)

    assert planned == {
        "store": store,
        "table": Tables.cube_part_institutionals,
        "full": True,
        "warmup": 390,
        "trading_index": trading_index,
    }
    assert got_window is window
    assert loads == {"frames": 1, "shares": 1, "splits": 1, "calendar": 1}
    assert calls == ["institutional", "superinvestor", "insider", "short", "ownership", "conditioning", "cross"]
    assert len(sinks) == 6 and all(sink is sinks[0] for sink in sinks)
    assert isinstance(sinks[0], ConditioningSink)

    assert list(got.columns) == [
        "date",
        "ticker",
        "f_inst",
        "f_super",
        "f_insider",
        "f_short",
        "f_owner",
        "f_condition",
        "f_cross",
    ]
    assert list(got[["date", "ticker"]].itertuples(index=False, name=None)) == [(first, "AAA"), (second, "BBB")]
    assert not got.duplicated(["date", "ticker"]).any()
    assert str(got["f_inst"].dtype) == "Float32"
    assert str(got["f_super"].dtype) == "Int64"
    assert got.loc[0, "f_inst"] == 1.25
    assert got.loc[0, "f_super"] == 0
    assert got.loc[0, "f_insider"] == 0.0
    assert pd.isna(got.loc[1, "f_inst"])
    assert pd.isna(got.loc[1, "f_super"])
    assert got.loc[1, "f_short"] == 2.5
    assert got.loc[0, "f_owner"] == 3.5
    assert second not in got.loc[got["ticker"] == "AAA", "date"].tolist()
    print("SANITY: build_panel kept order, one sink, one input load, exact output schema/dtypes/nulls, and the planned window.")


def test_run_requests_a_full_rerun_when_columns_change(monkeypatch: pytest.MonkeyPatch) -> None:
    step = _bare_step()
    store = object()
    step._store = store
    panel = pd.DataFrame({"date": [pd.Timestamp("2026-01-02")], "ticker": ["AAA"], "f": [1.0]})
    incremental = PartWindow(last=pd.Timestamp("2026-01-01"), since=pd.Timestamp("2025-01-01"))
    full = PartWindow(last=None, since=None)
    build_calls: list[bool] = []
    writes: list[tuple[object, object, pd.DataFrame, PartWindow, bool]] = []
    results = iter([COLUMNS_CHANGED, 1])

    def build_panel(*, full: bool = False) -> tuple[pd.DataFrame, PartWindow]:
        build_calls.append(full)
        return panel, full_window if full else incremental

    def write_part(store_arg: object, table: object, rows: pd.DataFrame, window: PartWindow, *, drop_empty: bool) -> int:
        writes.append((store_arg, table, rows, window, drop_empty))
        return next(results)

    full_window = full
    monkeypatch.setattr(step, "build_panel", build_panel)
    monkeypatch.setattr(step_module, "write_part", write_part)

    assert step.run(full=False) is None
    assert build_calls == [False, True]
    assert [call[3] for call in writes] == [incremental, full]
    assert all(call[0] is store for call in writes)
    assert all(call[1] is Tables.cube_part_institutionals for call in writes)
    assert all(call[2] is panel for call in writes)
    assert all(call[4] is True for call in writes)
    print("SANITY: COLUMNS_CHANGED requested a full rebuild and both writes retained the table, panel, window, and drop-empty contract.")
