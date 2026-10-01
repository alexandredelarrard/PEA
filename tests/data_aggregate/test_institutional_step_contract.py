# pyright: reportMissingImports=false
from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

import src.data_aggregate.transformers.step_cube_institutionals as step_module
from scripts.prove_insider_outliers import _panel as build_proof_panel
from src.data_aggregate.transformers.step_cube_institutionals import StepCubeInstitutionals
from src.data_aggregate.utils.common.incremental import COLUMNS_CHANGED, PartWindow, write_part
from src.data_aggregate.utils.institutionals.cross_source_features import EMISSION as CROSS_SOURCE_EMISSION
from src.data_aggregate.utils.institutionals.insider_features import EMISSION as INSIDER_EMISSION
from src.data_aggregate.utils.institutionals.institutional_features import EMISSION as INSTITUTIONAL_EMISSION
from src.data_aggregate.utils.institutionals.ownership_features import EMISSION as OWNERSHIP_EMISSION
from src.data_aggregate.utils.institutionals.short_flow_features import EMISSION as SHORT_FLOW_EMISSION
from src.data_aggregate.utils.institutionals.signal_conditioning import EMISSION as CONDITIONING_EMISSION
from src.data_aggregate.utils.institutionals.sink import ConditioningSink
from src.data_aggregate.utils.institutionals.superinvestor_features import EMISSION as SUPERINVESTOR_EMISSION
from src.data_store.schema import Tables


def _bare_step() -> StepCubeInstitutionals:
    step = object.__new__(StepCubeInstitutionals)
    cast(Any, step)._log = logging.getLogger(__name__)
    return step


def test_final_taxonomy_is_raw_only_without_unproved_peer_legs() -> None:
    emissions = (
        CROSS_SOURCE_EMISSION,
        INSIDER_EMISSION,
        INSTITUTIONAL_EMISSION,
        OWNERSHIP_EMISSION,
        SHORT_FLOW_EMISSION,
        CONDITIONING_EMISSION,
        SUPERINVESTOR_EMISSION,
    )
    declared = {name: mode for family in emissions for name, mode in family.items()}
    peer_features = {name for name, mode in declared.items() if mode == "raw+peers"}
    removed = {
        "ic_act_campaign_age_days",
        "ic_act_purpose_board",
        "ic_act_purpose_strategic",
        "ic_bo_holder_count",
        "ic_ftd_pct_so",
        "ic_ftd_z252",
        "ic_insider_buy_shares_so_180d",
        "ic_inst_flow_to_mcap",
        "ic_shortvol_ratio_z252",
        "ic_super_flow_to_mcap",
        "ic_super_exit_after_top10",
        "ic_xs_bearish_family_ratio",
        "ic_xs_bullish_actor_count",
        "ic_xs_bullish_family_ratio",
        "ic_xs_conflict_ratio",
    }

    assert len(declared) == sum(map(len, emissions)), "feature names must be unique across institutional families"
    assert set(declared.values()) == {"raw"}
    assert peer_features == set()
    assert removed.isdisjoint(declared)
    assert len(declared) == 71
    assert sum(1 if mode == "raw" else 2 for mode in declared.values()) == 71
    print(
        "SANITY: the final schema declares 71 unique characteristics / 71 legs, all raw; "
        "both provisional peer legs were removed because target/OOS evidence was unavailable."
    )


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
    fake_step = cast(Any, step)
    fake_step._context = context
    fake_step._config = config
    fake_step._store = store

    def load_peers(actual_context: object) -> dict[str, dict[str, float]]:
        calls["peers"] = {"context": actual_context}
        return peers

    def load_prices(actual_store: object, *, peers: object, fields: object, since: object) -> object:
        calls["prices"] = {"store": actual_store, "peers": peers, "fields": fields, "since": since}
        return price_frames

    monkeypatch.setattr(step_module.institutional_inputs, "load_peers_or_raise", load_peers)
    monkeypatch.setattr(step_module.institutional_inputs, "load_price_frames", load_prices)

    assert step._load_frames() is price_frames
    assert step._load_shares_out() is shares
    assert calls == {
        "peers": {"context": context},
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


def test_symbol_lineage_loader_projects_current_issuers() -> None:
    calls: list[dict[str, Any]] = []

    class _Store:
        def load(self, table: object, **kwargs: Any) -> pd.DataFrame:
            calls.append({"table": table, **kwargs})
            if table is Tables.sp500_tickers:
                return pd.DataFrame({"ticker": ["FISV"], "cik": [798354]})
            return pd.DataFrame(
                {
                    "symbol": ["FISV", "FI"],
                    "issuer_cik": ["0000798354", "0000798354"],
                    "valid_from": ["2006-01-03", "2023-06-07"],
                    "valid_to": ["2023-06-07", "2025-11-11"],
                }
            )

    tenure, roster = step_module.institutional_inputs.load_symbol_lineage(cast(Any, _Store()), logging.getLogger(__name__), ["FISV"])

    assert tenure is not None and roster is not None
    assert calls[0]["where"] == {"ticker": ["FISV"]}
    assert calls[1]["where"] == {"issuer_cik": ["0000798354"]}
    print("SANITY: the input layer projected FISV's current CIK and its FI/FISV lineage without relabelling canonical source rows twice.")


def test_insider_outlier_proof_uses_the_current_step_contract() -> None:
    insider = pd.DataFrame({"ticker": ["AAA", "BBB"], "value_usd": [10.0, 20.0]})
    scoped_insider = insider.loc[insider["ticker"].eq("AAA")].copy()
    passthrough = pd.DataFrame({"ticker": ["BBB"]})
    panel = pd.DataFrame({"date": [pd.Timestamp("2026-01-02")], "ticker": ["AAA"]})
    calls: list[tuple[object, object]] = []

    def original_load(table: object, universe: object = None) -> pd.DataFrame:
        calls.append((table, universe))
        return passthrough

    step = SimpleNamespace(_load_source=original_load)
    frames = object()
    shares = pd.DataFrame({"ticker": ["AAA"]})

    def build(actual_frames: object, actual_shares: pd.DataFrame, sink: ConditioningSink) -> pd.DataFrame:
        assert actual_frames is frames
        assert actual_shares is shares
        assert isinstance(sink, ConditioningSink)
        substituted = step._load_source(Tables.insider_transactions, ["AAA"])
        pd.testing.assert_frame_equal(substituted, scoped_insider)
        assert substituted is not insider
        assert step._load_source(Tables.insider_transactions_live, ["AAA"]) is None
        assert step._load_source(Tables.short_interest, ["AAA"]) is passthrough
        return panel

    step._insider_panel = build
    got = build_proof_panel(cast(Any, step), insider, cast(Any, frames), shares)

    assert got is panel
    assert step._load_source is original_load
    assert calls == [(Tables.short_interest, ["AAA"])]
    print("SANITY: the insider outlier proof supplies a fresh sink, disables live overlay, forwards universe scope, and restores the loader.")


def test_build_panel_preserves_order_sink_and_output_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    step = _bare_step()
    store = object()
    fake_step = cast(Any, step)
    fake_step._store = store
    fake_step._cfg = {}
    fake_step._part = SimpleNamespace(warmup_trading_days=390)

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

    incremental_panel, incremental_window = step.build_panel(full=False)
    pd.testing.assert_frame_equal(got, incremental_panel)

    class _SeededStore:
        def __init__(self, rows: pd.DataFrame) -> None:
            self.rows = rows.copy()

        def columns(self, _table: object) -> list[str]:
            return list(self.rows.columns)

        def append_tail(self, _table: object, tail: pd.DataFrame, cutoff: pd.Timestamp, *, inclusive: bool) -> int:
            keep = self.rows["date"] < cutoff if inclusive else self.rows["date"] <= cutoff
            self.rows = pd.concat([self.rows.loc[keep], tail], ignore_index=True)
            return len(tail)

    seeded = _SeededStore(got[got["date"] < second])
    write_part(cast(Any, seeded), Tables.cube_part_institutionals, incremental_panel, incremental_window, drop_empty=True)
    first_increment = seeded.rows.sort_values(["date", "ticker"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, first_increment)
    write_part(cast(Any, seeded), Tables.cube_part_institutionals, incremental_panel, incremental_window, drop_empty=True)
    pd.testing.assert_frame_equal(first_increment, seeded.rows.sort_values(["date", "ticker"]).reset_index(drop=True))
    assert not seeded.rows.duplicated(["date", "ticker"]).any()
    print(
        "SANITY: StepCubeInstitutionals full and seeded-incremental builds match on keys, "
        "dtypes, null masks and values; the identical second update is idempotent."
    )


def test_run_requests_a_full_rerun_when_columns_change(monkeypatch: pytest.MonkeyPatch) -> None:
    step = _bare_step()
    store = object()
    cast(Any, step)._store = store
    panel = pd.DataFrame({"date": [pd.Timestamp("2026-01-02")], "ticker": ["AAA"], "f": [1.0]})
    incremental = PartWindow(
        last=cast(pd.Timestamp, pd.Timestamp("2026-01-01")),
        since=cast(pd.Timestamp, pd.Timestamp("2025-01-01")),
    )
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
