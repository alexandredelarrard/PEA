"""Actual store, publication, source-join and skeleton contracts for fundamentals."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from omegaconf.errors import ConfigAttributeError

from src.constants.constants import SEC_NET_PENSION_LIABILITY_TAGS
from src.context import Context
from src.data_aggregate.transformers.step_cube_fundamentals import StepCubeFundamentals
from src.data_aggregate.utils.common.inputs import input_windows
from src.data_aggregate.utils.fundamentals.fundamental_features import (
    _FN_PBO_TAG,
    _FN_PLAN_ASSETS_TAG,
    build_fundamental_feature_panel,
    load_notes_num_scoped,
)
from src.data_aggregate.utils.fundamentals.inputs import FundamentalHistory
from src.data_aggregate.utils.price.frames import PriceFrames
from src.data_peers.utils.sector_peers import build_peer_dict
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.db import get_engine
from src.utils.window import Window, WindowRun
from tests.data_aggregate.fundamental_pit_reference import known_no_share_events, legacy_capital_frames
from tests.support.config import fixture_config
from tests.support.universe import universe_of

_CARRIED = ("minorityInterest", "operatingLeaseLiability", "financeLeaseLiability")
_CALENDAR = pd.DatetimeIndex(["2024-02-01", "2024-02-02", "2024-02-09", "2024-02-12", "2024-05-01", "2024-05-02", "2026-11-16"])


def _context(store: DataStore) -> Context:
    config = fixture_config()
    return cast(
        Context,
        SimpleNamespace(
            store=store,
            config=config,
            config_dir=Path("configs"),
            universe=universe_of(["AAA", "BBB"]),
            run=WindowRun("full", "known-truth financial source join", _CALENDAR[-1]),
        ),
    )


def _history_step(store: DataStore, sec: pd.DataFrame | None) -> StepCubeFundamentals:
    store.save(
        Tables.sharadar_fundamentals_history,
        pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "BBB"],
                "as_of": pd.to_datetime(["2024-02-01", "2024-05-01", "2024-02-10"]),
                "fiscal_end": pd.to_datetime(["2023-12-31", "2024-03-31", "2023-12-31"]),
                "totalRevenue": [10.0, 11.0, 20.0],
            }
        ),
    )
    if sec is not None:
        store.save(Tables.sec_fundamentals_history, sec)
    context = _context(store)
    return StepCubeFundamentals(context, context.config)


def test_fundamental_settings_preserve_the_configured_controls(sqlite_store: DataStore) -> None:
    context = _context(sqlite_store)
    configured = context.config.data_aggregate

    settings = StepCubeFundamentals(context, context.config)._feature_settings()

    assert settings.history_window == int(configured.hist.window)
    assert settings.history_min_periods == int(configured.hist.min_periods)
    assert settings.earnings_scale_quarters == int(configured.fundamentals_features.earnings_scale_quarters)
    assert all(isinstance(value, int) for value in (settings.history_window, settings.history_min_periods, settings.earnings_scale_quarters))
    assert settings.intrinsic_cfg is configured.intrinsic
    print("SANITY: the production settings factory preserves the configured integer controls and the existing intrinsic configuration.")


@pytest.mark.parametrize(
    "missing",
    ["hist", "hist.window", "hist.min_periods", "fundamentals_features", "fundamentals_features.earnings_scale_quarters", "intrinsic"],
)
def test_fundamental_settings_refuse_missing_required_config(sqlite_store: DataStore, missing: str) -> None:
    context = _context(sqlite_store)
    owner = context.config.data_aggregate
    path = missing.split(".")
    for key in path[:-1]:
        owner = owner[key]
    del owner[path[-1]]

    step = StepCubeFundamentals(context, context.config)
    with pytest.raises(ConfigAttributeError, match=path[-1]):
        step._feature_settings()
    print(f"SANITY: the production settings factory refuses missing required {missing} instead of choosing a hidden default.")


def test_the_cube_left_joins_three_sec_columns_onto_the_sharadar_spine(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    sec = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "as_of": pd.to_datetime(["2024-02-01", "2024-02-09", "2024-02-01"]),
            "minorityInterest": [5.0, 6.0, 7.0],
            "operatingLeaseLiability": [1.0, 2.0, 3.0],
            "financeLeaseLiability": [np.nan, 4.0, 5.0],
            "totalRevenue": [99.0, 99.0, 99.0],
        }
    )
    step = _history_step(sqlite_store, sec)
    calls = []
    original = sqlite_store.iter_load

    def capture(table: Any, **kwargs: Any) -> Any:
        calls.append((table, kwargs))
        return original(table, **kwargs)

    monkeypatch.setattr(sqlite_store, "iter_load", capture)
    windows = input_windows(step._context, Tables.cube_part_fundamentals, Window(_CALENDAR[0], _CALENDAR[-1]))
    frame = step._load_fundamentals(("AAA", "BBB"), _CALENDAR, windows)
    assert list(zip(frame.ticker, frame.as_of.dt.strftime("%Y-%m-%d"), strict=True)) == [
        ("AAA", "2024-02-02"),
        ("AAA", "2024-05-02"),
        ("BBB", "2024-02-12"),
    ]
    assert frame.totalRevenue.tolist() == [10.0, 11.0, 20.0]
    assert not set(_CARRIED) & set(frame)
    assert frame.iloc[0].sec_minorityInterest == 5.0 and frame.iloc[0].sec_operatingLeaseLiability == 1.0
    assert frame.loc[1:, [f"sec_{field}" for field in _CARRIED]].isna().all().all()
    assert [table for table, _ in calls] == [Tables.sharadar_fundamentals_history, Tables.sec_fundamentals_history]
    for table, query in calls:
        assert query["where"]["ticker"] == ["AAA", "BBB"] and query["date_col"] == "as_of"
        assert query["since"] == windows[table].start and query["until"] == windows[table].end
    assert calls[1][1]["columns"] == ["ticker", "as_of", *_CARRIED]
    print(
        "SANITY: the real source reads are projected and scoped once; exact raw filing keys join SEC fields before one configured market-session eligibility roll."
    )


def test_a_missing_sec_history_only_warns(sqlite_store: DataStore, caplog: pytest.LogCaptureFixture) -> None:
    step = _history_step(sqlite_store, None)
    windows = input_windows(step._context, Tables.cube_part_fundamentals, Window(_CALENDAR[0], _CALENDAR[-1]))
    with caplog.at_level(logging.WARNING):
        frame = step._load_fundamentals(("AAA", "BBB"), _CALENDAR, windows)
    assert len(frame) == 3 and frame.totalRevenue.tolist() == [10.0, 11.0, 20.0]
    assert frame[[f"sec_{field}" for field in _CARRIED]].isna().all().all()
    assert any(Tables.sec_fundamentals_history.name in record.message for record in caplog.records)
    print("SANITY: an unavailable SEC companion retains every Sharadar row with explicit null carried fields and a warning.")


def test_sec_fact_reads_are_ticker_tag_and_filing_date_bounded(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    row = {
        "ticker": "AAA",
        "adsh": "aaa",
        "tag": _FN_PBO_TAG,
        "ddate": pd.Timestamp("2023-12-31"),
        "qtrs": 0,
        "value": 10.0,
        "filed": pd.Timestamp("2024-02-01"),
        "available_at": pd.Timestamp("2024-02-01"),
        "period": "2023q4",
    }
    sqlite_store.save(Tables.notes_num, pd.DataFrame([row, {**row, "ticker": "OUTSIDE", "adsh": "outside"}]))
    context = _context(sqlite_store)
    calls = []
    original = sqlite_store.load

    def capture(table: Any, **kwargs: Any) -> Any:
        calls.append((table, kwargs))
        return original(table, **kwargs)

    monkeypatch.setattr(sqlite_store, "load", capture)
    rows = load_notes_num_scoped(context, _CALENDAR, tickers=("AAA", "BBB"), window=Window(pd.Timestamp("2020-09-04"), context.run.as_of))
    assert rows is not None and rows.ticker.tolist() == ["AAA"]
    assert rows.available_at.iloc[0] == pd.Timestamp("2024-02-01")
    assert pd.Timestamp(rows._source_available_at.iloc[0]) == pd.Timestamp("2024-02-01")
    assert len(calls) == 1 and calls[0][0] == Tables.notes_num
    query = calls[0][1]
    assert query["where"]["period"] == ["2023q4"]
    assert set(query["where"]["tag"]) == {_FN_PBO_TAG, _FN_PLAN_ASSETS_TAG, *SEC_NET_PENSION_LIABILITY_TAGS}
    assert {"period", "available_at", "adsh"}.issubset(query["columns"])
    print("SANITY: one notes payload read uses normalized archive-period bounds, ticker/tag scope and period+12 eligibility; raw metadata remains.")


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
    tickers = list(real_frames["stock_close"].columns[:5])
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
    warmup_start = tail_dates[0] - pd.Timedelta(days=int(fixture_config().data.tables.cube_part_fundamentals.warmup.cube_part_prices))
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
