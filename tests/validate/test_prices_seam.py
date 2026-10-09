"""Invariant 5 of the prices validator: an unrebased split seam in `close_split`.

Known-truth synthetic series: a flat quote with a one-bar step on a chosen day, beside a
`prices_splits` row at a chosen offset. The gate must block only on a RECENT, UNDECLARED seam.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
