"""The issuer/filer guard on 13D/13G accepts EVERY CIK of the ticker's filing scope, not just the roster's.

The schedule parse (`schedule_rows.schedule_ticker_rows`) keeps a schedule only when the issuer CIK read
off the filing is one of the ticker's scope CIKs. A single-CIK comparison against a multi-CIK listing
rejects exactly the pre-boundary schedules the lineage exists to recover -- measured as `sec-13d`
storing +0 rows after resolving MDT 28, BLK 50, VTRS 16+6 and ICE 10 predecessor filings.
"""

from __future__ import annotations

from src.data_extract.utils.common.edgar_driver import EdgarScope
from tests.data_extract.common.scope_fixtures import SENTINEL, dated_identity

#: VTRS as the register holds it: Mylan Inc -> Mylan N.V. -> Viatris.
_VTRS = EdgarScope(
    dated_identity(
        [
            ("VTRS", "0000069499", "cik_window", SENTINEL, "2015-05-01"),
            ("VTRS", "0001623613", "cik_window", "2015-05-01", "2020-11-07"),
            ("VTRS", "0001792044", "cik_window", "2020-11-07", None),
        ],
        {"VTRS": "0001792044", "AAPL": "0000320193"},
    )
)


def _subjects(ticker: str, cik: str) -> frozenset[str]:
    return frozenset(_VTRS.filing_scope(ticker, cik).event_ciks)


def test_every_scope_cik_identifies_the_ticker_as_issuer():
    got = _subjects("VTRS", "0001792044")
    assert got == {"0000069499", "0001623613", "0001792044"}
    print("\n=== SANITY: the widened guard ===")
    print(f"  VTRS accepts {len(got)} issuer CIKs, one per window: {sorted(got)}")


def test_a_predecessor_schedule_is_not_rejected():
    """A 2012 schedule about Mylan Inc carries issuer CIK 0000069499, not the roster's 0001792044."""
    assert "0000069499" in _subjects("VTRS", "0001792044")
    print("\n=== SANITY: a predecessor issuer CIK is in the subject set ===")


def test_an_unrelated_issuer_is_still_rejected():
    """A schedule VTRS FILED about Apple keeps Apple's issuer CIK and must still be dropped."""
    assert "0000320193" not in _subjects("VTRS", "0001792044")
    print("\n=== SANITY: an unrelated issuer stays outside ===")


def test_a_single_cik_ticker_keeps_exactly_its_roster_cik():
    assert _subjects("AAPL", "0000320193") == {"0000320193"}
    assert EdgarScope().filing_scope("AAPL", "320193").event_ciks == ("0000320193",)
    print("\n=== SANITY: a single-CIK scope (with or without identity) is its roster CIK ===")
