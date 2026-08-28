"""
`filing_rows`' two error classes: OUR defect propagates, the FILER's is counted.

The reason a one-word `NameError` survived 10.6 h of production is the error path, not the
typo: every per-filing and per-ticker handler on the way up reported it exactly as it reports
a malformed submission. So the split tested here is the fix -- and the asymmetry is
deliberate, because `filing.xbrl()` (edgartools parsing the filer's XBRL) must keep
swallowing everything, including the classes that mean "our bug" when they come out of our
own resolver.

Synthetic: which exception class reaches which handler is a known-truth question about
control flow (docs/testing.md's parsing exception).
"""
from __future__ import annotations

import types

import pytest

from src.data_extract.utils.common.edgar_driver import PROGRAMMING_ERRORS
from src.data_extract.utils.fundamentals import fetch_fundamentals_sec as fetcher

_ACCESSION = "0000320193-24-000123"


def _filing(*, xbrl_error: Exception | None = None):
    """A filing stand-in. `filing_rows` touches only `.xbrl()` and `.accession_number`
    before handing the parse to `rows_from_xbrl`."""
    def xbrl():
        if xbrl_error is not None:
            raise xbrl_error
        return object()                     # opaque: the patched resolver never reads it
    return types.SimpleNamespace(accession_number=_ACCESSION, xbrl=xbrl)


def test_a_programming_error_from_the_resolver_propagates(monkeypatch):
    """The NEM/MO/AIZ case, at the layer where it started."""
    def boom(*args, **kwargs):
        raise NameError("name 'cols' is not defined")

    monkeypatch.setattr(fetcher, "rows_from_xbrl", boom)
    failures: list[tuple[str, str]] = []

    with pytest.raises(NameError, match="cols"):
        fetcher.filing_rows("NEM", "1164727", _filing(), catalogue=None, gics=None,
                            failures=failures)

    assert failures == [], "a repo defect is not a filing failure and must not be counted"

    print("\n=== SANITY CHECK: filing_rows re-raises a resolver NameError ===")
    print(f"  NameError propagated; failures recorded: {len(failures)}")
    print("  -> Not reported as an unreadable filing.")


def test_a_data_error_from_the_resolver_is_counted_and_swallowed(monkeypatch):
    """One malformed filing must not cost a ticker its other 68 -- the convention that
    stays, now with the count that makes it visible."""
    def boom(*args, **kwargs):
        raise ValueError("period_end 0000-00-00 is not a date")

    monkeypatch.setattr(fetcher, "rows_from_xbrl", boom)
    failures: list[tuple[str, str]] = []

    rows = fetcher.filing_rows("NEM", "1164727", _filing(), catalogue=None, gics=None,
                               failures=failures)

    assert rows == []
    assert [acc for acc, _ in failures] == [_ACCESSION]

    print("\n=== SANITY CHECK: filing_rows counts a data failure ===")
    print(f"  ValueError swallowed, rows={len(rows)}, failures={failures}")
    print("  -> The gap is COUNTED, not inferred from a hole in the accession set.")


@pytest.mark.parametrize("error", [AttributeError("'NoneType' has no attribute 'facts'"),
                                   KeyError("ContextRef"), ValueError("bad xml")])
def test_an_unreadable_filing_is_always_swallowed_whatever_the_class(monkeypatch, error):
    """`filing.xbrl()` is the LIBRARY boundary: a malformed submission can raise any class
    at all out of edgartools, so the classes that mean "our bug" out of our own resolver
    still mean "unreadable filing" here. Asserted, because narrowing this `except` to match
    the other one would turn a bad filing into an aborted 490-ticker run."""
    failures: list[tuple[str, str]] = []

    rows = fetcher.filing_rows("NEM", "1164727", _filing(xbrl_error=error),
                               catalogue=None, gics=None, failures=failures)

    assert rows == []
    assert len(failures) == 1
    print(f"\n  {type(error).__name__} from filing.xbrl() -> swallowed and counted "
          f"({failures[0][1][:40]})")


def test_the_programming_error_classes_are_the_ones_the_driver_uses():
    """One list, so the per-filing and per-ticker handlers cannot drift apart."""
    assert NameError in PROGRAMMING_ERRORS and KeyError in PROGRAMMING_ERRORS
    print(f"\n=== SANITY CHECK: shared class list ===")
    print(f"  PROGRAMMING_ERRORS = "
          f"{tuple(e.__name__ for e in PROGRAMMING_ERRORS)}")
