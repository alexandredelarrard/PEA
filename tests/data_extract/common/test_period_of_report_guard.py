"""`period_of_report` must never abort a walk.

This is a regression test for the defect the register EXPOSED rather than caused: once
`resolve_registrant_filings` started walking predecessor archives, the 8-K walk reached
2004-era submissions whose EDGAR homepage metadata does not parse, and edgartools' own
`Filing.period_of_report` property raised `TypeError` from inside itself. `TypeError` is in
`PROGRAMMING_ERRORS`, which `edgar_driver._worker` re-raises deliberately, so one unparseable
filing aborted a 16-ticker run after BKR (237 recovered predecessor filings) and VTRS (292)
had already resolved.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.data_extract.utils.common.edgar_driver import PROGRAMMING_ERRORS, period_of_report


class _Raising:
    """A filing whose `period_of_report` raises, exactly as edgartools' does."""

    accession_number = "0000950129-04-000328"
    form = "8-K"
    filing_date = pd.Timestamp("2004-01-30").date()
    cik = 808362

    @property
    def period_of_report(self):
        _, _, period = None                             # what attachments.py:1170 really does
        return period


def test_the_raising_property_is_a_programming_error_and_would_abort_the_run():
    """The premise. If this ever stops holding, the guard below is no longer load-bearing."""
    with pytest.raises(PROGRAMMING_ERRORS):
        _ = _Raising().period_of_report
    print("\n=== SANITY: the premise ===")
    print("  filing.period_of_report raises TypeError, which _worker re-raises by design.")


def test_getattr_with_a_default_does_not_guard_it():
    """The trap: three call sites used `getattr(filing, "period_of_report", None)` and looked
    guarded. `getattr`'s default only answers AttributeError -- a raising property passes
    straight through it."""
    with pytest.raises(TypeError):
        getattr(_Raising(), "period_of_report", None)
    print("\n=== SANITY: why getattr is not enough ===")
    print("  getattr(..., None) re-raises the TypeError; only try/except swallows it.")


def test_the_guard_returns_none_instead():
    assert period_of_report(_Raising()) is None
    print("\n=== SANITY: the guard ===")
    print("  period_of_report(filing) -> None, so the walk keeps its other 15 tickers.")


def test_the_guard_is_transparent_when_the_property_works():
    good = SimpleNamespace(period_of_report="2024-12-31")
    assert period_of_report(good) == "2024-12-31"
    assert period_of_report(SimpleNamespace()) is None       # absent attribute, not an error
