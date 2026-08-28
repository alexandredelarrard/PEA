"""Tests `replay_equality.py` ITSELF: `compare` must actually detect the three classes of
silent change Phases 1-4 could introduce -- a changed cell, an all-null dtype drift, and a
dropped reason code. A gate nobody has tested is not a gate.

Synthetic throughout: the question here is "does the comparison notice a planted defect",
which is a known-truth question, not an economic one (see docs/testing.md's parsing/
derivation exception).
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data_extract.utils.fundamentals.build_history import TickerHistory, build_ticker
from tests.data_extract.fundamentals import replay_equality as harness

_TICKER = "TST"
_FLOW = "totalRevenue"


def _fact(**kwargs) -> dict:
    """One `fundamentals_facts` row with every column `build_ticker` reads defaulted."""
    row = {"ticker": _TICKER, "accession_number": "a-1", "field": _FLOW,
           "fiscal_year": 2023, "fiscal_period": "Q1", "duration_type": "quarterly",
           "form": "10-Q", "filing_date": "2023-05-01", "is_amendment": False,
           "period_of_report": "2023-03-31", "regime": "industrial",
           "period_start": "2023-01-01", "period_end": "2023-03-31", "period_days": 89,
           "value": 100.0, "unit": "USD", "source_concept": "us-gaap:Revenues",
           "dc_code": None, "adjustment": None}
    row.update(kwargs)
    return row


def _four_quarters() -> pd.DataFrame:
    """Four filed calendar quarters -- enough for one TTM `totalRevenue` event."""
    windows = [("2023-01-01", "2023-03-31", "Q1", "2023-05-01"),
              ("2023-04-01", "2023-06-30", "Q2", "2023-08-01"),
              ("2023-07-01", "2023-09-30", "Q3", "2023-11-01"),
              ("2023-10-01", "2023-12-31", "Q4", "2024-02-15")]
    rows = [_fact(accession_number=f"acc-{i}", fiscal_period=label,
                  period_start=start, period_end=end, filing_date=filed,
                  period_of_report=end, value=100.0 + i,
                  form="10-K" if label == "Q4" else "10-Q")
           for i, (start, end, label, filed) in enumerate(windows)]
    return pd.DataFrame(rows)


def _built() -> TickerHistory:
    return build_ticker(_TICKER, _four_quarters())


def test_harness_detects_a_planted_change(tmp_path):
    built = _built()
    before_dir, after_dir = tmp_path / "before", tmp_path / "after"
    harness.snapshot({_TICKER: built}, before_dir)

    mutated = built.history.copy()
    last = mutated.index[-1]                    # the only event with a full TTM window
    original = float(mutated.loc[last, _FLOW])
    mutated.loc[last, _FLOW] = original + 1.0
    harness.snapshot({_TICKER: TickerHistory(mutated, built.reason_codes)}, after_dir)

    report = harness.compare(before_dir, after_dir)

    print("\n=== SANITY CHECK: replay_equality.compare detects a planted cell change ===")
    print(f"  planted: {_FLOW} {original} -> {original + 1.0} on 1 row")
    print(f"  detected: {report.cells_differing[_TICKER]} differing cell(s), "
         f"first={report.first_10_diffs[0] if report.first_10_diffs else None}")
    assert report.cells_differing[_TICKER] == 1
    assert report.first_10_diffs[0][2] == _FLOW
    assert not report.ok
    print("  -> Gate validated: exactly the planted cell was reported.")


def test_harness_detects_a_dtype_change(tmp_path):
    built = _built()
    before_dir, after_dir = tmp_path / "before", tmp_path / "after"
    harness.snapshot({_TICKER: built}, before_dir)

    null_columns = [c for c in built.history.columns if built.history[c].isna().all()]
    assert null_columns, "fixture needs >= 1 all-null column to plant the TEXT-drift bug on"
    target = null_columns[0]
    mutated = built.history.copy()
    mutated[target] = mutated[target].astype(object)
    harness.snapshot({_TICKER: TickerHistory(mutated, built.reason_codes)}, after_dir)

    print("\n=== SANITY CHECK: replay_equality.compare detects an all-null dtype drift ===")
    print(f"  planted: {target} float64 -> object (the VRT/APA TEXT-column bug)")
    with pytest.raises(AssertionError, match="dtype drift"):
        harness.compare(before_dir, after_dir)
    print("  -> Gate validated: the comparison refuses a column that changed TYPE, not just "
         "value.")


def test_harness_detects_a_missing_reason_code(tmp_path):
    """A dropped reason-code row with the history frame otherwise UNCHANGED -- the
    comparison must catch it even though every cell still matches."""
    history = pd.DataFrame({"ticker": [_TICKER], "as_of": [pd.Timestamp("2023-05-01")]})
    codes = pd.DataFrame({
        "ticker": [_TICKER, _TICKER],
        "as_of": [pd.Timestamp("2023-05-01")] * 2,
        "field": ["totalRevenue", "totalAssets"],
        "dc_code": ["stale_ttm", "not_disclosed"],
        "combined_into": [None, None],
        "rejected_value": [float("nan"), float("nan")],
    })
    before_dir, after_dir = tmp_path / "before", tmp_path / "after"
    harness.snapshot({_TICKER: TickerHistory(history, codes)}, before_dir)
    harness.snapshot(
        {_TICKER: TickerHistory(history, codes.iloc[[0]].reset_index(drop=True))}, after_dir)

    report = harness.compare(before_dir, after_dir)

    print("\n=== SANITY CHECK: replay_equality.compare detects a dropped reason code ===")
    print("  planted: 1 code row dropped (totalAssets/not_disclosed), history unchanged")
    print(f"  detected: removed={report.codes_removed[_TICKER]}, "
         f"added={report.codes_added[_TICKER]}, cells_differing="
         f"{report.cells_differing[_TICKER]}")
    assert report.cells_differing[_TICKER] == 0
    assert report.codes_removed[_TICKER] == 1
    assert report.codes_added[_TICKER] == 0
    assert not report.ok
    print("  -> Gate validated: the missing code was caught even though the history frame "
         "matched.")
