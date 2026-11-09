"""Aggregate-owned contracts for the fundamentals cube writer.

These tests stay narrow: source reads are universe-scoped before any transform, the final
panel cannot escape the persisted price skeleton, and incremental writes always rewrite the
shared trailing repair window.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest

from src.constants.constants import FUNDAMENTALS_REFRESH_TRADING_DAYS
from src.data_aggregate.transformers.step_cube_fundamentals import StepCubeFundamentals
from src.data_aggregate.utils.common.incremental import PART_REFRESH_TRADING_DAYS
from src.data_aggregate.utils.fundamentals.inputs import FundamentalHistory
from src.data_aggregate.utils.price.frames import PriceFrames
from src.data_store.schema import Tables, name_of
from tests.data_aggregate.fundamental_pit_reference import known_no_share_events, legacy_capital_frames


class _StopAfterPlanError(RuntimeError):
    pass


def test_fundamentals_requests_its_45_session_tail_refresh(monkeypatch):
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

    assert captured["refresh"] == FUNDAMENTALS_REFRESH_TRADING_DAYS == 45
    assert PART_REFRESH_TRADING_DAYS == 7
    print("\n=== SANITY CHECK: fundamentals incremental repair window ===")
    print("  fundamentals requests 45 sessions while the shared sibling-part refresh is 7. Validated.")


class _SourceStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def load(self, table, **kwargs):
        self.calls.append((name_of(table), kwargs))
        frame = pd.DataFrame(
            {
                "ticker": ["AAA", "GHOST"],
                "as_of": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            }
        )
        return frame.reindex(columns=kwargs.get("columns") or frame.columns)

    @staticmethod
    def columns(table) -> list[str]:
        return ["ticker", "tag", "ddate", "qtrs", "value", "filed", "available_at"]


def test_every_direct_source_read_is_pushed_down_to_the_price_universe(monkeypatch):
    store = _SourceStore()
    step = StepCubeFundamentals.__new__(StepCubeFundamentals)
    step._context = SimpleNamespace(store=store, universe=SimpleNamespace(attach_gics=lambda frame: frame))
    step._log = logging.getLogger("test")
    monkeypatch.setattr(
        "src.data_aggregate.transformers.step_cube_fundamentals.add_cube_time_growth",
        lambda frame: frame,
    )

    source_since = pd.Timestamp("2020-09-04")
    step._load_fundamentals(("AAA", "BBB"), since=source_since)
    step._load_optional(
        Tables.earnings_surprises,
        "earnings-surprise history",
        "fetch_earnings_surprises",
        ("AAA", "BBB"),
    )
    step._load_optional(
        Tables.dividends,
        "dividend history",
        "fetch_dividends",
        ("AAA", "BBB"),
    )

    assert [table for table, _ in store.calls] == [
        "sharadar_fundamentals_history",
        "sec_fundamentals_history",
        "earnings_surprises",
        "prices_dividends",
    ]
    for table, kwargs in store.calls:
        assert kwargs["where"] == {"ticker": ["AAA", "BBB"]}, (table, kwargs)
    assert store.calls[0][1]["since"] == source_since
    assert store.calls[0][1]["project"] is True
    assert store.calls[1][1]["since"] == source_since
    assert store.calls[1][1]["columns"] == ["ticker", "as_of", "minorityInterest", "operatingLeaseLiability", "financeLeaseLiability"]
    print("\n=== SANITY CHECK: fundamentals source universe pushdown ===")
    print("  both histories are date-bounded and projected; all four sources carry the price-universe predicate. Validated.")


_CARRIED = ("minorityInterest", "operatingLeaseLiability", "financeLeaseLiability")


class _HistoryStore:
    """Sharadar spine + an SEC history (or none) keyed on `(ticker, as_of)`."""

    def __init__(self, sec: pd.DataFrame | None) -> None:
        self._sec = sec
        self.calls: list[tuple[str, dict]] = []

    def load(self, table, **kwargs):
        self.calls.append((name_of(table), kwargs))
        if name_of(table) == "sharadar_fundamentals_history":
            return pd.DataFrame(
                {
                    "ticker": ["AAA", "AAA", "BBB"],
                    "as_of": pd.to_datetime(["2024-02-01", "2024-05-01", "2024-02-10"]),
                    "fiscal_end": pd.to_datetime(["2023-12-31", "2024-03-31", "2023-12-31"]),
                    "totalRevenue": [10.0, 11.0, 20.0],
                }
            )
        return None if self._sec is None else self._sec[kwargs["columns"]]


def _history_step(store: _HistoryStore) -> StepCubeFundamentals:
    step = StepCubeFundamentals.__new__(StepCubeFundamentals)
    step._context = SimpleNamespace(store=store, universe=SimpleNamespace(attach_gics=lambda frame: frame))
    step._log = logging.getLogger("test.cube_fundamentals")
    return step


def test_the_cube_left_joins_three_sec_columns_onto_the_sharadar_spine(monkeypatch):
    monkeypatch.setattr("src.data_aggregate.transformers.step_cube_fundamentals.add_cube_time_growth", lambda frame: frame)
    sec = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "as_of": pd.to_datetime(["2024-02-01", "2024-02-09", "2024-02-01"]),  # BBB one day off, CCC not in Sharadar
            "minorityInterest": [5.0, 6.0, 7.0],
            "operatingLeaseLiability": [1.0, 2.0, 3.0],
            "financeLeaseLiability": [np.nan, 4.0, 5.0],
            "totalRevenue": [99.0, 99.0, 99.0],  # an SEC column the cube must not read
        }
    )
    store = _HistoryStore(sec)

    frame = _history_step(store)._load_fundamentals(("AAA", "BBB"))

    keys = list(zip(frame["ticker"], frame["as_of"].dt.strftime("%Y-%m-%d"), strict=True))
    assert keys == [("AAA", "2024-02-01"), ("AAA", "2024-05-01"), ("BBB", "2024-02-10")]
    assert frame["totalRevenue"].tolist() == [10.0, 11.0, 20.0]
    assert not set(_CARRIED) & set(frame.columns), "a bare SEC name would change meaning in fundamentals_to_daily"
    matched = frame.iloc[0]
    assert (matched["sec_minorityInterest"], matched["sec_operatingLeaseLiability"]) == (5.0, 1.0)
    assert np.isnan(matched["sec_financeLeaseLiability"])
    assert frame.loc[1:, [f"sec_{c}" for c in _CARRIED]].isna().all().all()
    sec_kwargs = store.calls[1][1]
    assert sec_kwargs["columns"] == ["ticker", "as_of", *_CARRIED] and sec_kwargs["optional"] is True
    print("\n=== SANITY CHECK: Sharadar spine, exact SEC key ===")
    print(
        f"  {len(frame)} Sharadar rows kept of 3; 1 exact match carries sec_minorityInterest=5, sec_operatingLeaseLiability=1; "
        "the 1-day-off BBB row and the Sharadar-less CCC row add nothing; no bare SEC name in the frame. Validated."
    )


def test_a_missing_sec_history_only_warns(monkeypatch, caplog):
    monkeypatch.setattr("src.data_aggregate.transformers.step_cube_fundamentals.add_cube_time_growth", lambda frame: frame)

    with caplog.at_level(logging.WARNING, logger="test.cube_fundamentals"):
        frame = _history_step(_HistoryStore(None))._load_fundamentals(("AAA", "BBB"))

    assert len(frame) == 3 and frame["totalRevenue"].tolist() == [10.0, 11.0, 20.0]
    assert frame[[f"sec_{c}" for c in _CARRIED]].isna().all().all()
    assert not set(_CARRIED) & set(frame.columns)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("sec_fundamentals_history" in message for message in warnings), warnings
    print("\n=== SANITY CHECK: missing SEC history ===")
    print(f"  3 Sharadar rows kept, the 3 sec_* columns empty, one warning: {warnings[0]!r}. Validated.")


def test_sec_fact_reads_are_ticker_and_date_bounded():
    from src.constants.constants import SEC_NET_PENSION_LIABILITY_TAGS
    from src.data_aggregate.utils.fundamentals.fundamental_features import (
        _FN_PBO_TAG,
        _FN_PLAN_ASSETS_TAG,
        load_notes_num_scoped,
    )

    store = _SourceStore()
    context = SimpleNamespace(store=store)
    source_since = pd.Timestamp("2020-09-04")

    load_notes_num_scoped(
        context,
        tickers=("AAA", "BBB"),
        since=source_since,
    )

    assert [table for table, _ in store.calls] == ["notes_num"]  # one read serves every pension leg
    for table, kwargs in store.calls:
        assert kwargs["where"]["ticker"] == ["AAA", "BBB"], (table, kwargs)
        assert kwargs["where"]["tag"], (table, kwargs)
        assert kwargs["since"] == source_since, (table, kwargs)
    assert "available_at" in store.calls[0][1]["columns"]
    assert "adsh" in store.calls[0][1]["columns"]
    assert set(store.calls[0][1]["where"]["tag"]) == {_FN_PBO_TAG, _FN_PLAN_ASSETS_TAG, *SEC_NET_PENSION_LIABILITY_TAGS}
    print("\n=== SANITY CHECK: SEC fact source bounds ===")
    print("  the single notes_num read pushes ticker universe, tag projection, and six-year date bound into the store. Validated.")


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
    print("  a ghost ticker and an off-grid date were removed; every persisted row is an exact price-skeleton key. Validated.")


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
        Tables.sharadar_fundamentals_history,
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
        FundamentalHistory(raw_history=fundamentals, share_basis=known_no_share_events()),
        replace(cast(PriceFrames, legacy_capital_frames(fundamentals, close.reindex(calendar))), trading_index=calendar, peers=peers),
    )

    incremental = build_fundamental_feature_panel(
        FundamentalHistory(raw_history=fundamentals, share_basis=known_no_share_events()),
        replace(cast(PriceFrames, legacy_capital_frames(fundamentals, close.reindex(window))), trading_index=window, peers=peers),
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
