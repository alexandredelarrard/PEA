"""Volume-ledger check on known-truth frames: FINRA / saved volume must stay flat across every Yahoo ledger event."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.data_store.schema import Tables
from src.validate.checks.volume_ledger import REVIEWED, STEP_COLUMNS, STEP_TOLERANCE, check_volume_ledger, step_findings, volume_ledger_steps

EVENT = pd.Timestamp("2024-06-03")
DAYS = pd.DatetimeIndex(pd.bdate_range("2023-10-02", "2025-01-31"), name="date")
RAW = 1_000_000.0
TURNOVER = 0.4  # FINRA total volume as a share of raw traded shares


def _frames(ticker: str, tape_before: float, raw_after: float, finra_before: float, finra_after: float, ratio: float, actual: float | None):
    before = DAYS < EVENT
    volume = pd.DataFrame({"ticker": ticker, "date": DAYS, "volume": np.where(before, tape_before, raw_after)})
    finra = pd.DataFrame({"ticker": ticker, "date": DAYS, "total_volume": np.where(before, finra_before, finra_after)})
    ledger = pd.DataFrame({"ticker": [ticker], "date": [EVENT], "ratio": [ratio]})
    genuine = pd.DataFrame({"ticker": [ticker], "date": [EVENT], "value": [actual]})
    genuine = genuine if actual is not None else genuine.iloc[:0].astype({"ticker": "object", "value": "float64"})
    return volume, finra, ledger, genuine


def _spinoff(ratio: float, repaired: bool):
    """A pure spin-off: raw traded shares R on both sides; Yahoo's tape multiplies the earlier bars by the ledger ratio unless repaired."""
    return _frames("CTVA", RAW if repaired else RAW * ratio, RAW, TURNOVER * RAW, TURNOVER * RAW, ratio, None)


def test_an_unrepaired_spinoff_tape_steps_by_the_ledger_ratio_and_a_repaired_one_is_flat() -> None:
    broken = volume_ledger_steps(*_spinoff(6.665, repaired=False))
    fixed = volume_ledger_steps(*_spinoff(6.665, repaired=True))

    assert broken["step"].iloc[0] == pytest.approx(6.665, rel=1e-12)
    assert fixed["step"].iloc[0] == pytest.approx(1.0, rel=1e-12)
    assert [f.ticker for f in step_findings(broken)] == ["CTVA"] and step_findings(fixed) == []
    print(
        f"\nSANITY: CTVA-shaped spin-off 6.665: unrepaired step {broken['step'].iloc[0]:.4f} is flagged; repaired step {fixed['step'].iloc[0]:.4f} passes."
    )


def test_a_genuine_split_is_flat_once_finra_is_carried_to_the_terminal_basis() -> None:
    # 2:1 split: raw shares R before and 2R after for the same turnover; the tape is in post-split shares (2R before).
    volume, finra, ledger, genuine = _frames("SPLT", 2 * RAW, 2 * RAW, TURNOVER * RAW, TURNOVER * 2 * RAW, 2.0, 2.0)

    steps = volume_ledger_steps(volume, finra, ledger, genuine)

    assert steps["step"].iloc[0] == pytest.approx(1.0, rel=1e-12) and steps["actual_ratio"].iloc[0] == 2.0
    assert step_findings(steps) == []
    print("\nSANITY: a genuine 2:1 split with a restated tape and FINRA carried by the split gives step 1.0; no finding.")


@pytest.mark.parametrize(("repaired", "flagged"), [(False, True), (True, False)])
def test_a_spinoff_codated_with_a_reverse_split_is_judged_against_the_split(repaired: bool, flagged: bool) -> None:
    # HON-shaped: ledger 0.9535, genuine 1-for-2 (0.5). Terminal tape before = raw x 0.5; Yahoo's unrepaired tape = raw x 0.9535.
    volume, finra, ledger, genuine = _frames("HON", RAW * (0.5 if repaired else 0.9535), RAW * 0.5, TURNOVER * RAW, TURNOVER * RAW * 0.5, 0.9535, 0.5)

    steps = volume_ledger_steps(volume, finra, ledger, genuine)

    assert steps["step"].iloc[0] == pytest.approx(1.0 if repaired else 0.9535 / 0.5, rel=1e-12)
    assert bool(step_findings(steps)) is flagged
    print(f"\nSANITY: HON-shaped event, repaired={repaired}: step {steps['step'].iloc[0]:.4f} -> flagged={flagged} (tolerance x{STEP_TOLERANCE}).")


def test_events_without_enough_bars_or_before_finra_start_are_not_a_pass() -> None:
    volume, finra, ledger, genuine = _spinoff(6.665, repaired=False)
    short = finra[finra["date"] < EVENT + pd.Timedelta(days=10)]
    old = ledger.assign(date=pd.Timestamp("2017-03-01"))

    unverified = volume_ledger_steps(volume, short, ledger, genuine)
    ignored = volume_ledger_steps(volume, finra, old, genuine)

    assert np.isnan(unverified["step"].iloc[0]) and unverified["n_post"].iloc[0] < 20 and step_findings(unverified) == []
    assert ignored.empty
    print(
        f"\nSANITY: {int(unverified['n_post'].iloc[0])} matched bars after the event give no step (unverified, never a pass); an event before 2018-08 is out of scope."
    )


def test_the_inputs_are_not_modified() -> None:
    frames = _spinoff(6.665, repaired=False)
    copies = [frame.copy() for frame in frames]

    volume_ledger_steps(*frames)

    for frame, copy in zip(frames, copies, strict=True):
        pd.testing.assert_frame_equal(frame, copy)
    print("\nSANITY: volume, FINRA, ledger and genuine-event frames are bit-identical after the call.")


class _Store:
    """Just enough of `DataStore.load` over in-memory frames."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self._frames = frames

    def exists(self, table: Any) -> bool:
        return table.name in self._frames

    def load(self, table: Any, columns: list[str], where: dict[str, Any], optional: bool = False, **_: Any) -> pd.DataFrame | None:
        frame = self._frames[table.name]
        key, wanted = next(iter(where.items()))
        rows = frame[frame[key].isin(wanted)][columns]
        return rows if len(rows) else None


@pytest.mark.parametrize(("repaired", "status"), [(False, "fail"), (True, "pass")])
def test_the_wrapper_reads_the_store_and_reports_the_flagged_ticker(repaired: bool, status: str) -> None:
    volume, finra, ledger, _ = _spinoff(6.665, repaired=repaired)
    store = _Store(
        {
            Tables.cube_part_prices.name: volume,
            Tables.prices_splits.name: ledger,
            Tables.sec_short_volume_security.name: finra.rename(columns={"ticker": "source_symbol"}),
        }
    )
    context = SimpleNamespace(store=store, universe=SimpleNamespace(tickers=["CTVA"]), log=logging.getLogger("test"))

    result = check_volume_ledger(context, tickers=None)  # type: ignore[arg-type]

    assert result.status == status
    assert result.metrics["events_measured"] == 1
    assert [f.ticker for f in result.findings] == (["CTVA"] if not repaired else [])
    print(f"\nSANITY: wrapper on the {'repaired' if repaired else 'unrepaired'} CTVA-shaped part -> {result.status}: {result.summary()}")


def test_a_reviewed_event_is_not_judged_but_the_same_step_elsewhere_is() -> None:
    date = pd.Timestamp("2021-08-02")
    rows = [
        {"ticker": t, "date": date, "yahoo_ratio": 0.125, "actual_ratio": 0.125, "n_pre": 60, "n_post": 60, "pre": 0.40, "post": 0.24, "step": 0.6}
        for t in ("GE", "ZZZ")
    ]
    steps = pd.DataFrame(rows, columns=list(STEP_COLUMNS))

    findings = step_findings(steps)

    assert ("GE", "2021-08-02") in REVIEWED and [f.ticker for f in findings] == ["ZZZ"]
    print("\nSANITY: a x0.60 step is a finding for ZZZ and not for the reviewed GE 2021-08-02 reverse split.")
