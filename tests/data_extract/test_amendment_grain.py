"""What an AMENDMENT does to `fundamentals_history`, and what it must not do.

Rebuild plan §5.0 rule 2 and decision 34. Three properties, each of which was violated by the
period-grain table this replaces:

  1. **The original row is FROZEN.** A restatement filed in 2024 must not change what the table
     said in 2023 -- that is the whole no-leakage property, and on a `(ticker, fiscal_end)`
     grain it was impossible: the restated value simply overwrote the period.
  2. **A restatement propagates across the TTM WINDOW, not just its own cell.** The table
     stores trailing-twelve LEVELS, so restating Q1 moves the TTM at Q1, Q2, Q3 and Q4. This
     is expected to be re-litigated as a bug during maintenance; it is not one.
  3. **>365 days late means no row at all.** A stock has long since passed a two-year-old
     restatement, so there is nothing there for a cross-sectional model to learn -- and the
     row costs a snapshot rebuild.

Synthetic, deliberately: "which rows changed and which did not" is a known-truth question about
one number, and no real filing isolates it. The real-filer half of §5.8 (SMCI / ADM) lives in
`test_build_history.py`'s ledger-backed tests and in the phase report.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data_extract.utils.fundamentals.build_history import (
    MAX_AMENDMENT_LAG_DAYS, build_ticker_history)
from src.data_extract.utils.fundamentals.kpi_catalogue import load_catalogue

CATALOGUE = load_catalogue("./configs")
FIELD = "totalRevenue"

#: A calendar-year filer's four quarters, each in its own 10-Q/10-K, with a distinct value so
#: a restatement is visible in the arithmetic rather than only in the provenance columns.
_WINDOWS = (("2023-01-01", "2023-03-31", "Q1", "2023-05-01", "10-Q", 100.0),
            ("2023-04-01", "2023-06-30", "Q2", "2023-08-01", "10-Q", 200.0),
            ("2023-07-01", "2023-09-30", "Q3", "2023-11-01", "10-Q", 300.0),
            ("2023-10-01", "2023-12-31", "Q4", "2024-02-15", "10-K", 400.0))


def _originals() -> list[dict]:
    return [{"ticker": "TST", "accession_number": f"orig-{label}", "field": FIELD,
             "fiscal_year": 2023, "fiscal_period": label, "duration_type": "quarterly",
             "form": form, "filing_date": filed, "is_amendment": False,
             "period_of_report": end, "regime": "industrial", "period_start": start,
             "period_end": end, "period_days": 89, "value": value, "unit": "USD",
             "source_concept": "us-gaap:Revenues", "dc_code": None, "adjustment": None}
            for start, end, label, filed, form, value in _WINDOWS]


def _restatement(*, filed: str, value: float = 150.0) -> dict:
    """A 10-Q/A restating Q1-2023 revenue from 100 to `value`, filed on `filed`."""
    row = dict(_originals()[0])
    row.update({"accession_number": f"amd-{filed}", "form": "10-Q/A", "is_amendment": True,
                "filing_date": filed, "value": value})
    return row


def _history(rows: list[dict]) -> pd.DataFrame:
    return build_ticker_history("TST", pd.DataFrame(rows), catalogue=CATALOGUE)


def test_a_restating_amendment_inside_a_year_emits_its_own_row_and_freezes_the_earlier_ones():
    """The ≤365-day twin of the cutoff test, and the round-trip §5.8 asks to print."""
    plain = _history(_originals())
    # 2024-04-01 is 336 days after the original Q1 filing -- inside the window.
    amended = _history([*_originals(), _restatement(filed="2024-04-01")])

    assert len(amended) == len(plain) + 1, "the restatement did not emit a row"
    new = amended.iloc[-1]
    assert pd.Timestamp(new["as_of"]) == pd.Timestamp("2024-04-01")
    assert new["is_amendment"] and new["publication_form"] == "10-Q/A"
    assert new["amended_fields"] == FIELD
    assert pd.Timestamp(new["amended_fiscal_end"]) == pd.Timestamp("2023-03-31")

    # Property 1: every row that already existed is byte-identical.
    shared = amended.iloc[:len(plain)].reset_index(drop=True)
    pd.testing.assert_frame_equal(shared, plain.reset_index(drop=True))

    # Property 2: the TTM at the amendment's own as_of is rebuilt with the restated quarter.
    assert plain[FIELD].iloc[-1] == 1000.0
    assert new[FIELD] == 1050.0, "the TTM was not recomputed with the restated quarter"

    # Rule: `fiscal_end` stays the LATEST KNOWN period, not the restated one.
    assert pd.Timestamp(new["fiscal_end"]) == pd.Timestamp("2023-12-31")

    print("\n=== SANITY CHECK: amendment round-trip (restated Q1, filed +336d) ===")
    print(amended[["as_of", "fiscal_end", "publication_form", "is_amendment",
                   "amended_fiscal_end", "amended_fields", FIELD]].to_string(index=False))
    print(f"  original 2023-05-01 row unchanged; the TTM moves 1000 -> 1050 only at the "
          f"amendment's own as_of; fiscal_end stays 2023-12-31 while amended_fiscal_end "
          f"carries 2023-03-31. Validated.")


def test_an_amendment_more_than_a_year_late_emits_nothing():
    """Decision 34. 2024-06-01 is 397 days after the original Q1 filing."""
    late = _history([*_originals(), _restatement(filed="2024-06-01")])
    assert len(late) == 4, f"a >365-day amendment emitted a row ({len(late)} rows)"
    assert not late["is_amendment"].any()
    assert late[FIELD].iloc[-1] == 1000.0, "a refused amendment still moved the TTM"

    # And the boundary is where it says it is: one day inside emits, one day outside does not.
    inside = _history([*_originals(), _restatement(filed="2024-05-01")])   # +366d... check
    lag_inside = (pd.Timestamp("2024-05-01") - pd.Timestamp("2023-05-01")).days
    assert lag_inside == 366 and len(inside) == 4, "the 366-day case should be refused"
    edge = _history([*_originals(), _restatement(filed="2024-04-30")])
    assert (pd.Timestamp("2024-04-30") - pd.Timestamp("2023-05-01")).days == 365
    assert len(edge) == 5, "the exactly-365-day case should be admitted"

    print("\n=== SANITY CHECK: the >365-day cutoff ===")
    print(f"  MAX_AMENDMENT_LAG_DAYS = {MAX_AMENDMENT_LAG_DAYS}")
    print(f"  +365d -> 5 rows (admitted) | +366d -> 4 rows | +397d -> 4 rows (refused, and "
          "the TTM stays 1000). The lag is measured from the ORIGINAL filing of the restated "
          "period, not from the period end. Validated.")


def test_an_amendment_that_changes_no_value_emits_nothing_however_many_facts_it_carries():
    """The value test, not a fact-count threshold: 88 of 246 real amendments carry <10 facts
    and a count rule both admits some of those and rejects a one-number restatement."""
    echo = [dict(row, accession_number="amd-echo", form="10-Q/A", is_amendment=True,
                 filing_date="2024-03-01") for row in _originals()]
    assert len(_history([*_originals(), *echo])) == 4

    one_number = _history([*_originals(), _restatement(filed="2024-03-01", value=100.5)])
    assert len(one_number) == 5, "a 0.5% one-number restatement was missed"

    print("\n=== SANITY CHECK: value test vs fact count ===")
    print("  a 4-fact amendment re-tagging every value identically -> 0 rows; a 1-fact "
          "amendment moving one number by 0.5% -> 1 row. Validated.")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
