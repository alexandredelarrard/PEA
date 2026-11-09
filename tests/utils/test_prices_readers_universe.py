"""Every reader of `prices` filters on the universe (Q2d, D-Q2-6).

`prices` also holds the companies' current secondary share classes (BRK-A, GOOG, ...). Each reader is run
on two stores, one of them with a non-universe ticker added on an extra later session and wild prices; the
reader's output must be identical.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from src.constants.constants_price import MACRO_MARKET_SERIES
from src.data_aggregate.transformers.step_cube_prices import StepCubePrices
from src.data_peers.step_deduce_peers import StepDeducePeers
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.freshness import StaleInputsError, check_prediction_inputs
from src.utils.tickers_universe.universe import Universe
from src.validate.checks import check_coverage, check_leakage
from src.validate.utils import prices as vprices
from tests.support.universe import seed_universe, universe_config

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
    config = cast(DictConfig, OmegaConf.create({"validate": loaded["validate"]}))
    config.validate.defaults["universe_expected"] = len(UNIVERSE)
    return config


def _store(extra: bool) -> DataStore:
    store = DataStore(create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}))
    seed_universe(store, UNIVERSE)
    frames = [_prices(UNIVERSE, SESSIONS, 100.0)]
    if extra:
        frames.append(_prices([EXTRA], SESSIONS.append(pd.DatetimeIndex([SESSIONS[-1] + pd.offsets.BDay(1)])), 600_000.0))
    store.save(Tables.prices, pd.concat(frames, ignore_index=True))
    return store


def _context(store: DataStore) -> Any:
    universe = Universe(store, universe_config(UNIVERSE))
    return SimpleNamespace(store=store, config=_config(), log=logging.getLogger("test.prices_readers"), universe=universe)


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


def test_cube_prices_reads_the_universe_only(tmp_path: Path) -> None:
    def read(context: Any) -> pd.DataFrame:
        context.store.save(Tables.prices_macro, pd.DataFrame({"ticker": MACRO_MARKET_SERIES, "date": SESSIONS, "close": 100.0}))
        peer_path = tmp_path / "peers.json"
        peer_path.write_text(json.dumps({ticker: {ticker: 1.0} for ticker in UNIVERSE}), encoding="utf-8")
        context.paths, context.config_dir = {"SECTOR_PEERS_PATH": peer_path}, tmp_path
        context.config.build_cube = {}
        StepCubePrices(context, context.config).run(full=True)
        return context.store.load(Tables.cube_part_prices).sort_values(["ticker", "date"])

    assert set(_same(read, "step_cube_prices")["ticker"]) == set(UNIVERSE)


def test_peer_deduction_reads_the_universe_only():
    def read(context: Any) -> pd.DataFrame:
        step = StepDeducePeers.__new__(StepDeducePeers)
        step._context, step._log = context, context.log
        step.load_prices()
        return step.prices_long.sort_values(["ticker", "date"])

    _same(read, "step_deduce_peers.load_prices")


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
        context.store.save(Tables.sharadar_fundamentals_facts, vendor)
        context.store.save(
            Tables.sharadar_fundamentals_history, pd.DataFrame({"ticker": UNIVERSE, "as_of": [SESSIONS[10]] * 2, "sharesOutstanding": 5.0})
        )
        panel = vprices.load_panel(context).sort_values(["ticker", "date"])
        return panel, vprices.invariant_spike_revert(context), vprices.invariant_day_coverage(context)

    _same(read, "validate.utils.prices load_panel / spike_revert / day_coverage")


def test_coverage_reference_grid_is_the_universe_only():
    def read(context: Any) -> dict:
        part = _prices(UNIVERSE, SESSIONS, 1.0)[["date", "ticker"]].assign(f_x=1.0)
        context.store.save(Tables.cube_part_momentum, part)
        return check_coverage(context, Tables.cube_part_momentum, config=context.config).metrics

    _same(read, "validate coverage (reference = prices)")


def test_f108_a_former_member_with_prices_is_a_roster_that_moved_on_not_a_row_from_nowhere():
    context = _context(_store(True))
    part = _prices([*UNIVERSE, EXTRA, "ZZZ"], SESSIONS, 1.0)[["date", "ticker"]].assign(f_x=1.0)
    context.store.save(Tables.cube_part_momentum, part)
    findings = check_coverage(context, Tables.cube_part_momentum, config=context.config).findings
    scores = {f.ticker: f.score for f in findings if f.ticker in (EXTRA, "ZZZ")}
    assert scores == {EXTRA: 4, "ZZZ": 7}, scores
    print("\n=== SANITY CHECK: F-108 coverage unknown tickers ===")
    print(f"  {EXTRA} (in `prices`, not in the universe) scores 4 (roster moved on); ZZZ (no `prices` row) scores 7 (row from nowhere)")


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
