"""Aggregate-owned contracts for the fundamentals cube writer.

These tests stay narrow: source reads are universe-scoped before any transform, the final
panel cannot escape the persisted price skeleton, and incremental writes always rewrite the
shared trailing repair window.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.transformers.step_cube_fundamentals import StepCubeFundamentals
from src.data_aggregate.utils.common.incremental import PART_REFRESH_TRADING_DAYS
from src.data_store.schema import Tables, name_of


class _StopAfterPlanError(RuntimeError):
    pass


def test_fundamentals_requests_the_shared_tail_refresh(monkeypatch):
    captured: dict[str, object] = {}

    def _plan(*args, **kwargs):
        captured.update(kwargs)
        raise _StopAfterPlanError

    monkeypatch.setattr(
        "src.data_aggregate.transformers.step_cube_fundamentals.load_trading_calendar",
        lambda store: pd.bdate_range("2026-08-01", periods=10),
    )
    monkeypatch.setattr("src.data_aggregate.transformers.step_cube_fundamentals.plan_window", _plan)
    step = StepCubeFundamentals.__new__(StepCubeFundamentals)
    step._store = object()
    step._warmup = lambda: 1320

    with pytest.raises(_StopAfterPlanError):
        step.run(full=False)

    assert captured["refresh"] == PART_REFRESH_TRADING_DAYS
    print("\n=== SANITY CHECK: fundamentals incremental repair window ===")
    print(
        f"  plan_window received refresh={PART_REFRESH_TRADING_DAYS}; the stored maximum " "date is rewritten instead of strict-appended. Validated."
    )


class _SourceStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def load(self, table, **kwargs):
        self.calls.append((name_of(table), kwargs))
        return pd.DataFrame(
            {
                "ticker": ["AAA", "GHOST"],
                "as_of": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            }
        )


def test_every_direct_source_read_is_pushed_down_to_the_price_universe(monkeypatch):
    store = _SourceStore()
    step = StepCubeFundamentals.__new__(StepCubeFundamentals)
    step._context = SimpleNamespace(store=store)
    step._log = logging.getLogger("test")
    monkeypatch.setattr(
        "src.data_aggregate.transformers.step_cube_fundamentals.add_cube_time_growth",
        lambda frame: frame,
    )
    monkeypatch.setattr(
        "src.data_aggregate.transformers.step_cube_fundamentals.attach_gics_columns",
        lambda frame, context, log: frame,
    )

    step._load_fundamentals(("AAA", "BBB"))
    step._load_optional(
        Tables.earnings_surprises,
        "earnings-surprise history",
        "fetch_earnings_surprises",
        ("AAA", "BBB"),
    )
    step._load_optional(
        Tables.dividends,
        "dividend history",
        "fetch_price_history",
        ("AAA", "BBB"),
    )

    assert len(store.calls) == 3
    for table, kwargs in store.calls:
        assert kwargs["where"] == {"ticker": ["AAA", "BBB"]}, (table, kwargs)
    print("\n=== SANITY CHECK: fundamentals source universe pushdown ===")
    print("  fundamentals, earnings, and dividends all carry the same ticker predicate before " "any peer/cross-sectional transform. Validated.")


def test_final_fundamentals_panel_is_contained_by_the_price_skeleton():
    date = pd.Timestamp("2026-09-04")
    skeleton = pd.DataFrame({"date": [date, date], "ticker": ["AAA", "BBB"]})
    panel = pd.DataFrame(
        {
            "date": [date, date, date, date - pd.Timedelta(days=1)],
            "ticker": ["AAA", "BBB", "GHOST", "AAA"],
            "f_value": [1.0, 2.0, 99.0, 3.0],
        }
    )

    bounded = StepCubeFundamentals._restrict_to_skeleton(panel, skeleton)

    assert list(zip(bounded["date"], bounded["ticker"], strict=False)) == [
        (date, "AAA"),
        (date, "BBB"),
    ]
    print("\n=== SANITY CHECK: final fundamentals key containment ===")
    print("  a ghost ticker and an off-grid date were removed; every persisted row is an exact " "price-skeleton key. Validated.")


def test_real_fundamentals_full_and_incremental_tail_parity(real_frames):
    """A warm-up-bounded real rebuild must reproduce the full panel's written tail."""
    from src.data_aggregate.utils.fundamentals.fundamental_features import (
        build_fundamental_feature_panel,
    )
    from src.data_peers.utils.sector_peers import build_peer_dict
    from src.data_store.store import DataStore
    from src.utils.db import get_engine

    tickers = list(real_frames["stock_close"].columns[:4])
    close = real_frames["stock_close"][tickers]
    returns = real_frames["stock_ret"][tickers]
    fundamentals = DataStore(get_engine()).load(
        Tables.fundamentals_history,
        where={"ticker": tickers},
    )
    peers = build_peer_dict(returns, top_k=10, weighting="corr", min_obs=120)
    edge_start = pd.Timestamp("2026-08-03")
    edge_end = pd.Timestamp("2026-09-04")
    # A bounded real-history oracle: 2019 supplies more than the declared five-year
    # maximum look-back while avoiding a redundant 1995-onward feature build in every phase.
    calendar = close.index[(close.index >= pd.Timestamp("2019-01-01")) & (close.index <= edge_end)]
    tail_dates = calendar[(calendar >= edge_start) & (calendar <= edge_end)]
    assert len(tail_dates) >= 20, "accepted 2026-08-03..2026-09-04 real tail is unavailable"
    start_pos = calendar.get_indexer([tail_dates[0]])[0]
    warmup_start = calendar[max(0, start_pos - 1320)]
    window = calendar[calendar >= warmup_start]

    full = build_fundamental_feature_panel(
        fundamentals,
        peers,
        calendar,
        stock_close=close.reindex(calendar),
    )

    incremental = build_fundamental_feature_panel(
        fundamentals,
        peers,
        window,
        stock_close=close.reindex(window),
    )

    keys = ["date", "ticker"]
    expected = full[full["date"].isin(tail_dates)].sort_values(keys).reset_index(drop=True)
    actual = incremental[incremental["date"].isin(tail_dates)].sort_values(keys).reset_index(drop=True)
    assert list(actual.columns) == list(expected.columns)
    pd.testing.assert_frame_equal(actual[keys], expected[keys], check_dtype=False)
    value_cols = [column for column in expected.columns if column not in keys]
    assert actual[value_cols].isna().equals(expected[value_cols].isna())
    actual_values = actual[value_cols].to_numpy(dtype=float)
    expected_values = expected[value_cols].to_numpy(dtype=float)
    np.testing.assert_allclose(
        actual_values,
        expected_values,
        atol=5e-7,
        rtol=2e-6,
        equal_nan=True,
    )
    print("\n=== SANITY CHECK: real fundamentals full/incremental tail parity ===")
    print(
        f"  {len(tail_dates)} sessions from {tail_dates.min().date()} through "
        f"{tail_dates.max().date()}, {len(tickers)} real tickers, {len(actual)} keys, "
        f"{len(value_cols)} features: ordered keys and null mask match exactly; persisted "
        "float32 values match within atol=5e-7 and rtol=2e-6. Validated."
    )
