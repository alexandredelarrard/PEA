"""Every reader of `prices` filters on the universe (Q2d, D-Q2-6).

`prices` also holds the companies' current secondary share classes (BRK-A, GOOG, ...). Each reader is run
on two stores, one of them with a non-universe ticker added on an extra later session and wild prices; the
reader's output must be identical.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from src.constants.constants_price import MACRO_MARKET_SERIES
from src.data_aggregate.transformers.step_cube_prices import StepCubePrices
from src.data_peers.step_deduce_peers import StepDeducePeers
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.strategies import step_super_investors
from src.strategies.step_super_investors import SuperInvestorsStrategy
from src.strategies.utils import ls_model
from src.utils.freshness import StaleInputsError, check_prediction_inputs
from src.utils.universe import unverified_ciks
from src.validate.checks import check_coverage, check_leakage
from src.validate.utils import prices as vprices

UNIVERSE = ["AAA", "BBB"]
EXTRA = "BRK-A"
SESSIONS = pd.bdate_range("2024-01-01", periods=40)


def _prices(tickers: list[str], sessions: pd.DatetimeIndex, level: float) -> pd.DataFrame:
    rows = [
        {
            "date": day,
            "ticker": ticker,
            "open": level + i,
            "high": level + i + 1,
            "low": level + i - 1,
            "close_split": level + i * (1 + k),
            "close_total": level + i * (1 + k),
            "volume": 1000.0 + i,
        }
        for k, ticker in enumerate(tickers)
        for i, day in enumerate(sessions)
    ]
    return pd.DataFrame(rows)


def _config() -> DictConfig:
    loaded = cast(DictConfig, OmegaConf.load("configs/validate.yml"))
    config = cast(DictConfig, OmegaConf.create({"validate": loaded["validate"], "data_extract": {"redundant_ticks": []}}))
    config.validate.defaults["universe_expected"] = len(UNIVERSE)
    return config


def _store(extra: bool) -> DataStore:
    store = DataStore(create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}))
    store.save(Tables.sp500_tickers, pd.DataFrame({"ticker": UNIVERSE, "cik": ["0000000001", "0000000002"]}))
    frames = [_prices(UNIVERSE, SESSIONS, 100.0)]
    if extra:
        frames.append(_prices([EXTRA], SESSIONS.append(pd.DatetimeIndex([SESSIONS[-1] + pd.offsets.BDay(1)])), 600_000.0))
    store.save(Tables.prices, pd.concat(frames, ignore_index=True))
    return store


def _context(store: DataStore) -> Any:
    return SimpleNamespace(store=store, config=_config(), log=logging.getLogger("test.prices_readers"))


def _same(read: Callable[[Any], Any], name: str) -> Any:
    """Run `read` on the store without and with the non-universe ticker; both outputs must be equal."""
    base, mixed = read(_context(_store(False))), read(_context(_store(True)))
    _assert_equal(base, mixed)
    print(f"\n=== SANITY CHECK: {name} ===\n  a non-universe {EXTRA} in `prices` (extra session, 600k prices) changes nothing")
    return mixed


def _assert_equal(a: Any, b: Any) -> None:
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    elif isinstance(a, pd.Series):
        pd.testing.assert_series_equal(a, b)
    elif isinstance(a, tuple | list):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            _assert_equal(x, y)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_equal(a[key], b[key])
    else:
        assert a == b, (a, b)


class _StopError(Exception):
    pass


def test_cube_prices_reads_the_universe_only():
    def read(context: Any) -> pd.DataFrame:
        step = StepCubePrices.__new__(StepCubePrices)
        step._store, step._tickers, step._log, step._bugfix = context.store, list(UNIVERSE), context.log, {}
        captured: dict[str, pd.DataFrame] = {}
        step._plan_window = lambda full: SimpleNamespace(since=None)  # type: ignore[method-assign]

        def pivot(raw: pd.DataFrame) -> None:
            captured["raw"] = raw
            raise _StopError

        step._pivot_fields = pivot  # type: ignore[method-assign]
        with pytest.raises(_StopError):
            step.run()
        return captured["raw"].sort_values(["ticker", "date"])

    assert set(_same(read, "step_cube_prices")["ticker"]) == set(UNIVERSE)


def test_super_investors_mark_the_universe_only(monkeypatch):
    monkeypatch.setattr(step_super_investors, "roster_as_of", lambda context: ["0000000009"])

    def read(context: Any) -> tuple:
        filings = pd.DataFrame(
            {
                "cik": ["0000000009"],
                "ticker": ["AAA"],
                "cusip": ["000000001"],
                "filing_date": [SESSIONS[5]],
                "period": [SESSIONS[0]],
                "shares": [10.0],
                "value_usd": [1000.0],
            }
        )
        context.store.save(Tables.sec13f_hr, filings)
        step = SuperInvestorsStrategy.__new__(SuperInvestorsStrategy)
        step._context = context
        _, prices, end = step.load_raw()
        return prices.sort_values(["ticker", "date"]), end

    _same(read, "step_super_investors.load_raw")


def test_ls_sleeve_returns_cover_the_universe_only():
    def read(context: Any) -> tuple:
        macro = pd.DataFrame({"date": SESSIONS.append(pd.DatetimeIndex([SESSIONS[-1] + pd.offsets.BDay(1)])), "ticker": MACRO_MARKET_SERIES})
        context.store.save(Tables.prices_macro, macro.assign(close=np.linspace(400.0, 440.0, len(macro))))
        return ls_model._returns(context, cast(Any, None), cast(Any, None), cast(Any, {"beta_window": 5, "vol_window": 5}), SESSIONS[20])

    close, _, _ = _same(read, "ls_model._returns")
    assert list(close.columns) == UNIVERSE


def test_peer_deduction_reads_the_universe_only():
    def read(context: Any) -> pd.DataFrame:
        step = StepDeducePeers.__new__(StepDeducePeers)
        step._context, step._log = context, context.log
        step.load_prices()
        return step.prices_long.sort_values(["ticker", "date"])

    _same(read, "step_deduce_peers.load_prices")


def test_unverified_ciks_is_unchanged():
    _same(unverified_ciks, "utils.universe.unverified_ciks")


def test_validate_price_invariants_read_the_universe_only(monkeypatch):
    monkeypatch.setattr(vprices, "_repair_registered", lambda context, prices, vendor: prices)
    monkeypatch.setattr(vprices, "_level_factor_for", lambda context, panel, where: 1.0)

    def read(context: Any) -> tuple:
        vendor = pd.DataFrame(
            {
                "ticker": UNIVERSE,
                "date": [SESSIONS[10]] * 2,
                "dimension": "ARQ",
                "reportperiod": [SESSIONS[5]] * 2,
                "price": [110.0, 120.0],
                "sharesbas": 5.0,
                "marketcap": 1.0,
            }
        )
        context.store.save(Tables.sharadar_fundamentals, vendor)
        context.store.save(Tables.fundamentals_history, pd.DataFrame({"ticker": UNIVERSE, "as_of": [SESSIONS[10]] * 2, "sharesOutstanding": 5.0}))
        panel = vprices.load_panel(context).sort_values(["ticker", "date"])
        return panel, vprices.invariant_spike_revert(context), vprices.invariant_day_coverage(context)

    _same(read, "validate.utils.prices load_panel / spike_revert / day_coverage")


def test_coverage_reference_grid_is_the_universe_only():
    def read(context: Any) -> dict:
        part = _prices(UNIVERSE, SESSIONS, 1.0)[["date", "ticker"]].assign(f_x=1.0)
        context.store.save(Tables.cube_part_momentum, part)
        return check_coverage(context, Tables.cube_part_momentum, config=context.config).metrics

    _same(read, "validate coverage (reference = prices)")


def test_leakage_last_price_session_is_the_universe_only():
    def read(context: Any) -> Any:
        targets = _prices(UNIVERSE, SESSIONS, 1.0)[["date", "ticker"]].assign(f_ret_h30=1.0)
        context.store.save(Tables.cube_part_targets, targets)
        return check_leakage(context, Tables.cube_part_targets, config=context.config).metrics.get("last_reference_session")

    assert _same(read, "validate leakage (reference = prices)") == SESSIONS[-1]


def test_prediction_freshness_guard_reads_the_universe_frontier_only():
    """The cube-vs-prices check: a secondary class with a later bar must not make the universe's cube look stale."""

    def read(context: Any) -> str:
        context.store.save(Tables.cube, pd.DataFrame({"date": [SESSIONS[-1]] * 2, "ticker": UNIVERSE}))
        try:
            check_prediction_inputs(context.store, UNIVERSE, SESSIONS[-1], 0.0)
        except StaleInputsError as exc:
            return f"stale: {exc}"
        return "fresh"

    assert _same(read, "utils.freshness.check_prediction_inputs (cube vs prices)") == "fresh"
