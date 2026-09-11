"""The issuer/filer guard on 13D/13G must accept EVERY segment CIK, not just the roster's.

Regression test for a defect the register introduced rather than exposed. `build_ticker_13g_edgar`
keeps a schedule only when the issuer CIK read off the filing matches the ticker's own. Before the
register that was a single-CIK test against a single-CIK listing, and consistent. Widening the
listing to every segment without widening the comparison rejects exactly the pre-boundary
schedules the register exists to recover -- measured as `sec-13d` storing +0 rows after resolving
MDT 28, BLK 50, VTRS 16+6 and ICE 10 predecessor filings.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data_extract.utils.common.registrant import (
    Registrant, SCHEDULE_SEGMENT_CAP, Segment, issuer_ciks,
)


def _reg() -> dict[str, Registrant]:
    """VTRS as the register holds it: a three-segment chain."""
    segs = (Segment(cik="0000069499", valid_from=None,
                    valid_to=pd.Timestamp("2015-05-01"), evidence="Mylan Inc"),
            Segment(cik="0001623613", valid_from=pd.Timestamp("2015-05-01"),
                    valid_to=pd.Timestamp("2020-11-07"), evidence="Mylan N.V."),
            Segment(cik="0001792044", valid_from=pd.Timestamp("2020-11-07"),
                    valid_to=None, evidence="Viatris"))
    return {"VTRS": Registrant(ticker="VTRS", kind="reorganisation", segments=segs)}


def test_every_segment_cik_identifies_the_ticker_as_issuer():
    got = issuer_ciks("VTRS", "0001792044", _reg())
    assert got == {"0000069499", "0001623613", "0001792044"}
    print("\n=== SANITY: the widened guard ===")
    print(f"  VTRS accepts {len(got)} issuer CIKs, one per segment: {sorted(got)}")


def test_a_predecessor_schedule_is_no_longer_rejected():
    """The exact rejection that produced +0. A 2012 schedule about Mylan Inc carries issuer CIK
    0000069499; the old test compared it to the roster's 0001792044 and dropped it."""
    accepted = issuer_ciks("VTRS", "0001792044", _reg())
    predecessor_issuer = "0000069499"
    assert predecessor_issuer != "0001792044"           # the old single-CIK test
    assert predecessor_issuer in accepted               # the new one


def test_an_unrelated_issuer_is_still_rejected():
    """The guard must not become a pass-through: a schedule VTRS FILED about Apple keeps
    Apple's issuer CIK and must still be dropped."""
    assert "0000320193" not in issuer_ciks("VTRS", "0001792044", _reg())


def test_a_ticker_with_no_register_entry_keeps_exactly_its_roster_cik():
    assert issuer_ciks("AAPL", "0000320193", _reg()) == {"0000320193"}


def test_the_roster_cik_is_kept_even_when_it_is_not_a_segment():
    """XOM's roster CIK was the holdco while the register named the predecessor. Dropping the
    roster CIK would have discarded rows the pipeline already resolved correctly."""
    reg = _reg()
    assert issuer_ciks("VTRS", "0009999999", reg) == {
        "0000069499", "0001623613", "0001792044", "0009999999"}


@pytest.mark.parametrize("n,skipped", [(109, False), (2_000, False), (40_070, True)])
def test_the_cap_sits_between_the_real_and_the_pathological(n, skipped):
    """Measured 2026-09-10: issuer-side segment contributions ran 12-109 across all 16 register
    tickers; BLK's predecessor CIK offered 40,070 because BlackRock Inc is one of the largest
    13G FILERS in existence."""
    assert (n > SCHEDULE_SEGMENT_CAP) is skipped
