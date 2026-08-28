"""Tests for employee headcount as a `fundamentals_facts` field
(src/data_extract/utils/fundamentals/fundamentals_employees.py).

Headcount used to be its own fetcher writing its own `employees_history` table.
It is now parsed out of the SAME 10-K the fundamentals walk already opens and
appended as an ordinary instant fact, so what needs proving is the JOIN between
the two halves -- the parser itself is covered by test_employee_extract.py /
test_employee_extract_audit.py and is unchanged:

  1. the fact row is shaped so the period engine's `instant_stock` accepts it and
     lands it on the Q4 (fiscal-year-end) snapshot, like a balance-sheet level;
  2. a 10-Q is never opened for it (the body-text download is the expensive part);
  3. the continuity guard still fires, now seeded from `fundamentals_facts`;
  4. `employees` reaches `fundamentals_history_sec` as a real column, carried across
     the interim quarters rather than populating only the fiscal-year-end row.

Checks 1 and 4 consume the period engine (`periods.instant_stock`) and the history build
(`build_history.carry_latest_known`). Both are imported at TOP LEVEL, deliberately: they
were once pinned with `importorskip` on a dotted module string while those modules were
being rebuilt, which turns a rename into a SKIP -- a test that asserts nothing while
reporting green. The modules have landed, so a missing symbol must fail at collection.

All network access is faked -- no EDGAR calls.
"""
from __future__ import annotations

import pandas as pd

from src.data_extract.utils.fundamentals.periods import instant_stock

from src.data_extract.utils.fundamentals.build_history import carry_latest_known
from src.data_extract.utils.fundamentals.fundamentals_employees import (
    EMPLOYEES_FIELD, employee_fact_frame, history_by_ticker, is_headcount_form,
)

_TEXT = ("Item 1. Business. As of December 31, 2020, we had approximately "
         "21,400 employees worldwide.")


class _FakeFiling:
    """The three attributes `employee_fact_frame` touches, plus the body-text
    accessor. `html()` returning None forces the `.text()` fallback path."""

    def __init__(self, form="10-K", period="2020-12-31", body=_TEXT, as_html=True):
        self.form = form
        self.period_of_report = period
        self.accession_number = "0000000000-20-000001"
        self._body = body
        self._as_html = as_html
        self.html_calls = 0

    def html(self):
        self.html_calls += 1
        return f"<html><body><p>{self._body}</p></body></html>" if self._as_html else None

    def text(self):
        return self._body


# --------------------------------------------------------------------------- #
# 1. Fact-row shape: what `instant_stock` needs to accept it                    #
# --------------------------------------------------------------------------- #
def test_employee_fact_row_is_a_year_end_instant():
    row = employee_fact_frame(_FakeFiling()).iloc[0]

    assert row["field"] == EMPLOYEES_FIELD and row["value"] == 21_400.0
    assert row["period_type"] == "instant"
    assert row["period_end"] == pd.Timestamp("2020-12-31")
    # NO period_start is the load-bearing part: it is what tells instant_stock this
    # is a year-end SNAPSHOT (rename 'FY' -> 'Q4') rather than a duration measure
    # that legitimately has both an FY and a Q4 flavour (e.g. basicShares).
    assert pd.isna(row["period_start"])
    # left blank on purpose -> backfilled from the filing's tagged duration facts
    assert row["fiscal_year"] is None and row["fiscal_period"] is None

    # ... and now prove instant_stock actually does that with it.
    facts = pd.DataFrame([{
        "fiscal_year": 2020, "fiscal_period": "FY", "value": row["value"],
        "filing_date": pd.Timestamp("2021-02-18"), "accession_number": "acc-1",
        "form": "10-K", "period_start": row["period_start"],
        "period_end": row["period_end"], "source_tag": row["source_tag"],
        "is_amendment": 0.0, "fiscal_period_source": "native",
    }])
    out = instant_stock(facts)
    assert len(out) == 1 and out.iloc[0]["fiscal_period"] == "Q4"
    assert out.iloc[0]["value"] == 21_400.0

    print("\n=== SANITY CHECK: employee fact row shape ===")
    print(f"  parsed {row['value']:,.0f} employees -> instant fact, period_end "
          f"{row['period_end'].date()}, no period_start")
    print(f"  instant_stock lands it on fiscal_period='{out.iloc[0]['fiscal_period']}' "
          "(the fiscal-year-end snapshot, same grid as every balance-sheet level). Validated.")


# --------------------------------------------------------------------------- #
# 2. Only the 10-K is ever opened                                              #
# --------------------------------------------------------------------------- #
def test_only_annual_reports_are_downloaded():
    assert is_headcount_form("10-K") and is_headcount_form("10-K/A")
    assert not is_headcount_form("10-Q") and not is_headcount_form("10-Q/A")
    assert not is_headcount_form(None)

    # a 10-Q must SHORT-CIRCUIT before the body text is fetched -- that download is
    # the whole cost of this feature, and ~75% of the filings walked are 10-Qs
    tenq = _FakeFiling(form="10-Q", period="2020-09-30")
    assert employee_fact_frame(tenq) is None
    assert tenq.html_calls == 0, "10-Q body text was downloaded -- pure waste"

    tenk = _FakeFiling()
    assert employee_fact_frame(tenk) is not None
    assert tenk.html_calls == 1

    print("\n=== SANITY CHECK: download only where a headcount can exist ===")
    print("  10-K/10-K/A -> parsed; 10-Q/10-Q/A -> skipped with ZERO body-text "
          "fetches. Validated.")


def test_plain_text_submission_fallback():
    """Pre-2001 filings have no HTML rendition; `.text()` must still be parsed."""
    frame = employee_fact_frame(_FakeFiling(as_html=False))
    assert frame is not None and frame.iloc[0]["value"] == 21_400.0
    print("\n=== SANITY CHECK: .txt submission fallback ===")
    print("  html() -> None falls back to text() and still parses 21,400. Validated.")


# --------------------------------------------------------------------------- #
# 3. Continuity guard, seeded from fundamentals_facts                          #
# --------------------------------------------------------------------------- #
def test_continuity_guard_drops_parse_artifact():
    """The CSGP failure: a filing that states no headcount at all, where the parser
    picks up an unrelated "2.3 million people" phrase in a workforce-shaped sentence.
    It scores WELL (`as of` + `approximately` + a workforce noun), so nothing inside
    the document betrays it -- only the ticker's own history does."""
    text = ("As of December 31, 2020, approximately 2,300,000 people used our "
            "marketplace.")
    assert employee_fact_frame(_FakeFiling(body=text), [1000, 1100, 1155, 1200]) is None
    # ... and with no history to anchor on (a ticker's FIRST filing) it is accepted:
    # the guard only ever REJECTS against evidence, it never invents a value
    first = employee_fact_frame(_FakeFiling(body=text), [])
    assert first is not None and first.iloc[0]["value"] == 2_300_000.0

    # the seed comes off `fundamentals_facts` rows, in filing-date order
    stored = pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB"],
        "filing_date": ["2022-02-01", "2021-02-01", "2020-02-01"],
        "value": [1200.0, 1000.0, 50.0],
    })
    assert history_by_ticker(stored) == {"AAA": [1000, 1200], "BBB": [50]}

    print("\n=== SANITY CHECK: continuity guard on the new fact path ===")
    print(f"  2,300,000 against a stored median of 1,155 -> DROPPED; same value with "
          "no history -> kept (nothing to contradict it)")
    print("  guard seeded from fundamentals_facts (ticker/filing_date/value), "
          "filing-date ordered. Validated.")


# --------------------------------------------------------------------------- #
# 4. It reaches fundamentals_history_sec as a column                               #
# --------------------------------------------------------------------------- #
def test_employees_is_carried_forward_into_the_interim_quarters():
    """A headcount is disclosed ONCE A YEAR, in the 10-K, so `employees` must reach
    `fundamentals_history_sec` under an as-of (ffill) alignment rather than populating only
    the fiscal-year-end row and leaving the three interim quarters blank.

    This is the one property of the field that lives in the history build rather than in
    the parser, and it is easy to lose when the column list is rewritten."""
    # one annual headcount (FY2019, filed with the 10-K) against a quarterly grid
    ends = pd.DatetimeIndex(["2019-12-31", "2020-03-31", "2020-06-30", "2020-09-30"])
    facts = pd.DataFrame({
        "ticker": "AAA",
        "field": EMPLOYEES_FIELD,
        "period_end": [ends[0]],
        "filing_date": [pd.Timestamp("2020-02-14")],
        "value": [21_400.0],
    })
    out = carry_latest_known(facts, ends, field=EMPLOYEES_FIELD)

    assert EMPLOYEES_FIELD in out.columns, "employees never became a column"
    assert (out[EMPLOYEES_FIELD] == 21_400.0).all(), \
        "the annual headcount did not carry into the interim quarters"

    print("\n=== SANITY CHECK: employees reaches fundamentals_history_sec ===")
    print(f"  ONE annual disclosure (2019-12-31: 21,400) -> populated on all "
          f"{len(out)} quarter rows {list(out[EMPLOYEES_FIELD].astype(int))}")
    print("  as-of (ffill) alignment, so interim quarters are not blank. Validated.")
