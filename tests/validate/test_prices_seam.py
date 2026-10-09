"""Invariant 5 of the prices validator: an unrebased split seam in `close_split`.

Known-truth synthetic series: a flat quote with a one-bar step on a chosen day, beside a
`prices_splits` row at a chosen offset. The gate must block only on a RECENT, UNDECLARED seam.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.data_store.schema import Tables
from src.validate.utils import prices as prices_mod
from src.validate.utils.prices import PricesReport, gate, invariant_unrebased_seam, run_prices_validation
from tests.conftest import FakeStore

SESSIONS = 120
RATIO = 6.665
DATES = pd.bdate_range("2026-04-01", periods=SESSIONS)


def _context(
    tmp_path: Path,
    seam_at: int | None,
    step_up: float = RATIO,
    split_offset_days: int | None = 2,
    split_ratio: float = RATIO,
    register: dict | None = None,
) -> Any:
    """One ticker `T`: quote 100 after the seam bar, 100 x `step_up` before it."""
    close = [100.0] * SESSIONS
    if seam_at is not None:
        close[:seam_at] = [100.0 * step_up] * seam_at
    tables: dict[Any, pd.DataFrame] = {
        Tables.prices: pd.DataFrame({"ticker": "T", "date": DATES, "close_split": close}),
        Tables.prices_splits: pd.DataFrame({"ticker": [], "date": [], "ratio": []}),
    }
    if split_offset_days is not None and seam_at is not None:
        when = DATES[seam_at] - pd.Timedelta(days=split_offset_days)
        tables[Tables.prices_splits] = pd.DataFrame({"ticker": ["T"], "date": [when], "ratio": [split_ratio]})
    elif split_offset_days is not None:
        tables[Tables.prices_splits] = pd.DataFrame({"ticker": ["T"], "date": [DATES[60]], "ratio": [split_ratio]})
    if register is not None:
        path = tmp_path / "prices" / "yf_price_bugfix.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"_APPROVED": {"by": "test"}, **register}), encoding="utf-8")
    return SimpleNamespace(store=FakeStore(tables), config_dir=str(tmp_path))


def _verdict(context: Any) -> tuple[Any, bool, str]:
    result = invariant_unrebased_seam(context, tickers=["T"])
    ok, reason = gate(PricesReport(invariants=[result]))
    return result, ok, reason


def test_recent_undeclared_seam_is_flagged_and_blocks(tmp_path: Path) -> None:
    result, ok, reason = _verdict(_context(tmp_path, seam_at=100))
    assert result.failed == 1 and set(result.failing_tickers) == {"T"}, result.detail
    assert result.detail[0]["recent"] and not result.detail[0]["declared"]
    assert not ok and "unrebased split seam" in reason
    print(f"\nSANITY recent undeclared seam: flagged {result.detail[0]}, gate blocks -> {reason[:60]}")


def test_clean_series_passes(tmp_path: Path) -> None:
    result, ok, _ = _verdict(_context(tmp_path, seam_at=None))
    assert result.failed == 0 and not result.detail and ok
    print("\nSANITY clean continuous series beside a split row: nothing flagged, gate passes")


def test_genuine_move_without_a_matching_split_is_not_flagged(tmp_path: Path) -> None:
    mrna_like = 2.77
    uncorroborated, ok_a, _ = _verdict(_context(tmp_path, seam_at=100, step_up=1 / mrna_like, split_offset_days=None))
    other_ratio, ok_b, _ = _verdict(_context(tmp_path, seam_at=100, step_up=1 / mrna_like, split_offset_days=2, split_ratio=20.0))
    assert uncorroborated.failed == 0 and other_ratio.failed == 0 and ok_a and ok_b
    print("\nSANITY +177% move: no split row, and a 20x split row, both leave it unflagged")


def test_declared_register_entry_is_reported_not_blocking(tmp_path: Path) -> None:
    seam_day = DATES[100].strftime("%Y-%m-%d")
    result, ok, _ = _verdict(_context(tmp_path, seam_at=100, register={"return_seams": {"T": [{"date": seam_day}]}}))
    assert len(result.detail) == 1 and result.detail[0]["declared"] and result.failed == 0 and ok
    print(f"\nSANITY seam {seam_day} declared in the register: listed as declared, gate passes")


def test_old_seam_is_reported_not_blocking(tmp_path: Path) -> None:
    result, ok, _ = _verdict(_context(tmp_path, seam_at=20))
    assert result.failed == 1 and not result.detail[0]["recent"] and ok
    print("\nSANITY seam 100 sessions back: reported in failing_tickers, gate passes")


def test_late_registered_event_twenty_days_from_the_seam_is_flagged(tmp_path: Path) -> None:
    near, ok_near, _ = _verdict(_context(tmp_path, seam_at=100, split_offset_days=20))
    far, ok_far, _ = _verdict(_context(tmp_path, seam_at=100, split_offset_days=45))
    inverse, ok_inv, _ = _verdict(_context(tmp_path, seam_at=100, step_up=1 / RATIO, split_offset_days=20))
    assert near.failed == 1 and not ok_near
    assert far.failed == 0 and ok_far
    assert inverse.failed == 1 and not ok_inv
    print("\nSANITY split row 20 days away: flagged (also the inverse step); 45 days away: outside the window")


def test_run_prices_validation_includes_the_seam_invariant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub = lambda *a, **k: prices_mod.InvariantResult(name="stub")  # noqa: E731
    monkeypatch.setattr(prices_mod, "load_panel", stub)
    for name in ("invariant_market_cap", "invariant_price_vintage", "invariant_day_coverage", "invariant_spike_revert"):
        monkeypatch.setattr(prices_mod, name, stub)
    report = run_prices_validation(_context(tmp_path, seam_at=100), tickers=["T"])
    names = [r.name for r in report.invariants]
    ok, _ = gate(report)
    assert "unrebased_seam" in names and not ok
    print(f"\nSANITY run_prices_validation reports {names}; the gate blocks on the seam")


# --------------------------------------------------------------------------- #
# small split ratios (spinoff price factors within 10% of 1)                  #
# --------------------------------------------------------------------------- #
SMALL = 1.057  # SPGI's 2026-07-01 spinoff factor
SEAM_DAY = 100


def _noisy(sigma: float, seed: int) -> list[float]:
    """A random walk with daily log-return dispersion `sigma`, starting at 100."""
    rng = np.random.default_rng(seed)
    return list(100.0 * np.exp(np.cumsum(rng.normal(0.0, sigma, SESSIONS))))


def _series_context(tmp_path: Path, closes: dict[str, list[float]], split_ratio: float, split_ticker: str = "T") -> Any:
    """Several tickers' closes on `DATES`; one `prices_splits` row for `split_ticker` two days before `SEAM_DAY`."""
    prices = pd.concat([pd.DataFrame({"ticker": t, "date": DATES, "close_split": c}) for t, c in closes.items()], ignore_index=True)
    splits = pd.DataFrame({"ticker": [split_ticker], "date": [DATES[SEAM_DAY] - pd.Timedelta(days=2)], "ratio": [split_ratio]})
    return SimpleNamespace(store=FakeStore({Tables.prices: prices, Tables.prices_splits: splits}), config_dir=str(tmp_path))


def _with_step(close: list[float], step: float, at: int = SEAM_DAY) -> list[float]:
    """`close` with every bar from `at` on multiplied by `step` (a permanent one-bar step)."""
    return [c * (step if i >= at else 1.0) for i, c in enumerate(close)]


def _small_verdict(context: Any, tickers: list[str]) -> Any:
    return invariant_unrebased_seam(context, tickers=tickers)


def test_small_ratio_unrebased_seam_on_a_calm_series_is_flagged(tmp_path: Path) -> None:
    close = _with_step(_noisy(0.004, seed=1), 1 / SMALL)
    result = _small_verdict(_series_context(tmp_path, {"T": close}, SMALL), ["T"])
    assert result.failed == 1 and result.detail[0]["date"] == DATES[SEAM_DAY].strftime("%Y-%m-%d"), result.detail
    ok, _ = gate(PricesReport(invariants=[result]))
    assert not ok
    print(f"\nSANITY ratio {SMALL} with a {1 / SMALL - 1:+.1%} step on a 0.4%-sigma series: flagged {result.detail[0]}, gate blocks")


def test_small_ratio_ordinary_move_on_a_volatile_series_is_not_flagged(tmp_path: Path) -> None:
    close = _with_step(_noisy(0.04, seed=2), 1 / SMALL)
    result = _small_verdict(_series_context(tmp_path, {"T": close}, SMALL), ["T"])
    assert result.failed == 0 and not result.detail, result.detail
    print(f"\nSANITY ratio {SMALL}, the same {1 / SMALL - 1:+.1%} step on a 4%-sigma series: an ordinary day, not flagged")


def test_small_ratio_step_against_the_stale_direction_is_not_flagged(tmp_path: Path) -> None:
    close = _with_step(_noisy(0.004, seed=1), SMALL)
    result = _small_verdict(_series_context(tmp_path, {"T": close}, SMALL), ["T"])
    assert result.failed == 0, result.detail
    print(f"\nSANITY ratio {SMALL} with a {SMALL - 1:+.1%} step: a stale history steps by 1/ratio, so not flagged")


def test_tiny_ratio_below_the_noise_floor_is_not_flagged(tmp_path: Path) -> None:
    tiny = 1.0017  # OXY 2016-02-25
    close = _with_step([100.0] * SESSIONS, 1 / tiny)
    result = _small_verdict(_series_context(tmp_path, {"T": close}, tiny), ["T"])
    assert result.failed == 0, result.detail
    print(f"\nSANITY ratio {tiny} on a flat series: below the noise floor, not scanned")


def test_small_ratio_step_shared_by_the_market_is_not_flagged(tmp_path: Path) -> None:
    peers = {f"P{i:02d}": _noisy(0.004, seed=10 + i) for i in range(60)}
    alone = {"T": _with_step(_noisy(0.004, seed=1), 1 / SMALL), **peers}
    shared = {"T": alone["T"], **{t: _with_step(c, 1 / SMALL) for t, c in peers.items()}}
    flagged = _small_verdict(_series_context(tmp_path, alone, SMALL), list(alone))
    market = _small_verdict(_series_context(tmp_path, shared, SMALL), list(shared))
    assert flagged.failed == 1 and market.failed == 0, (flagged.detail, market.detail)
    print(f"\nSANITY 61 tickers: T alone steps {1 / SMALL - 1:+.1%} -> flagged; every ticker steps the same day -> market move, not flagged")


def test_reviewed_genuine_move_is_declared_without_a_data_repair(tmp_path: Path) -> None:
    close = _with_step(_noisy(0.004, seed=1), 1 / SMALL)
    context = _series_context(tmp_path, {"T": close}, SMALL)
    seam_day = DATES[SEAM_DAY].strftime("%Y-%m-%d")
    path = tmp_path / "prices" / "yf_price_bugfix.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"_APPROVED": {"by": "test"}, "seam_reviewed": {"T": [{"date": seam_day, "why": "earnings"}]}}), encoding="utf-8")
    result = _small_verdict(context, ["T"])
    ok, _ = gate(PricesReport(invariants=[result]))
    assert len(result.detail) == 1 and result.detail[0]["declared"] and result.failed == 0 and ok, result.detail
    print(f"\nSANITY {seam_day} listed under seam_reviewed: reported as declared, gate passes")
