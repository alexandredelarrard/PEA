"""
`filing_rows`' two error classes: OUR defect propagates, the FILER's is a parse failure.

The reason a one-word `NameError` survived 10.6 h of production is the error path, not the
typo: every per-filing and per-ticker handler on the way up reported it exactly as it reports
a malformed submission. So the split tested here is the fix -- and the asymmetry is
deliberate, because `filing.xbrl()` (edgartools parsing the filer's XBRL) must keep
treating everything as the filer's problem, including the classes that mean "our bug" when
they come out of our own resolver.

Synthetic: which exception class reaches which handler is a known-truth question about
control flow (wiki/guides/testing.md's parsing exception).
"""

from __future__ import annotations

import types
from typing import Any, cast

import pytest

from src.data_extract.utils.common.edgar_driver import FilingStamp
from src.data_extract.utils.common.parallel_fetch import PROGRAMMING_ERRORS
from src.data_extract.utils.common.sec_io import ParseFailureError
from src.data_extract.utils.fundamentals import fetch_fundamentals_sec as fetcher

_ACCESSION = "0000320193-24-000123"


def _filing(*, xbrl_error: Exception | None = None):
    """A filing stand-in: the attributes `FilingStamp.of` reads, plus the `.xbrl()` that
    `filing_rows` calls before handing the parse to `rows_from_xbrl`."""

    def xbrl():
        if xbrl_error is not None:
            raise xbrl_error
        return object()  # opaque: the patched resolver never reads it

    return types.SimpleNamespace(accession_number=_ACCESSION, form="10-K", filing_date="2024-11-01", xbrl=xbrl)


def _stamp(*, xbrl_error: Exception | None = None) -> FilingStamp:
    return FilingStamp.of(_filing(xbrl_error=xbrl_error), "1164727")


def test_a_programming_error_from_the_resolver_propagates(monkeypatch):
    """The NEM/MO/AIZ case, at the layer where it started."""

    def boom(*args, **kwargs):
        raise NameError("name 'cols' is not defined")

    monkeypatch.setattr(fetcher, "rows_from_xbrl", boom)

    with pytest.raises(NameError, match="cols"):
        fetcher.filing_rows("NEM", _stamp(), catalogue=cast(Any, None), gics=None)

    print("\n=== SANITY CHECK: filing_rows re-raises a resolver NameError ===")
    print("  NameError propagated -> not reported as an unreadable filing; the run fails.")


def test_a_data_error_from_the_resolver_is_a_parse_failure(monkeypatch):
    """One malformed filing must not cost a ticker its other 68: it becomes a logged marker."""

    def boom(*args, **kwargs):
        raise ValueError("period_end 0000-00-00 is not a date")

    monkeypatch.setattr(fetcher, "rows_from_xbrl", boom)

    with pytest.raises(ParseFailureError, match=_ACCESSION):
        fetcher.filing_rows("NEM", _stamp(), catalogue=cast(Any, None), gics=None)

    print("\n=== SANITY CHECK: filing_rows classifies a data failure ===")
    print("  ValueError -> ParseFailureError naming the accession -> one logged empty-filing marker.")


@pytest.mark.parametrize("error", [AttributeError("'NoneType' has no attribute 'facts'"), KeyError("ContextRef"), ValueError("bad xml")])
def test_an_unreadable_filing_is_always_a_parse_failure_whatever_the_class(error):
    """`filing.xbrl()` is the LIBRARY boundary: a malformed submission can raise any class
    at all out of edgartools, so the classes that mean "our bug" out of our own resolver
    still mean "unreadable filing" here."""
    with pytest.raises(ParseFailureError, match=_ACCESSION):
        fetcher.filing_rows("NEM", _stamp(xbrl_error=error), catalogue=cast(Any, None), gics=None)
    print(f"\n  {type(error).__name__} from filing.xbrl() -> ParseFailureError (a marker, never a crash)")


def test_the_programming_error_classes_are_the_ones_the_driver_uses():
    """One list, so the per-filing and per-ticker handlers cannot drift apart."""
    assert NameError in PROGRAMMING_ERRORS and KeyError in PROGRAMMING_ERRORS
    print("\n=== SANITY CHECK: shared class list ===")
    print(f"  PROGRAMMING_ERRORS = {tuple(e.__name__ for e in PROGRAMMING_ERRORS)}")
