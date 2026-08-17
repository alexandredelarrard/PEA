"""
test_fundamentals_fiscal_calendar.py
-------------------------------------
`fundamentals_periods.resolve_fiscal_year_by_filing_calendar` + the
`period_of_report` / retry hardening in `fetch_fundamentals_edgar`.

Every fixture below reproduces a CONFIRMED live-data failure (accession numbers
in the docstrings), all of which shared one signature: the stored fiscal_year or
period_of_report was a perfectly well-formed value that was simply the WRONG one,
so the bad rows either collided with a real fiscal year (destroying both) or the
filing vanished from `fundamentals_facts` with no log line.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data_extract.utils.fundamentals.fetch_fundamentals_edgar import (
    _cover_fiscal_year, _cover_page_shares_fallback, _filings_to_parse,
    _resolve_period_of_report,
)
from src.data_extract.utils.fundamentals.fundamentals_periods import (
    _fiscal_year_index, resolve_fiscal_year_by_filing_calendar,
)
from src.data_extract.utils.fundamentals.fundamentals_tags import (
    COVER_PAGE_SHARES_MAX_LAG_DAYS, COVER_PAGE_SHARES_TAG, DILUTED_SHARES_TAGS,
    FISCAL_YEAR_CONTEXT_DAYS, SHARES_OUTSTANDING_FIELD, SHARES_TAGS,
)


# --- fixtures ---------------------------------------------------------------

def _fact(form: str, period_start, period_end, fiscal_year, cover_fiscal_year=None) -> dict:
    return {"form": form, "period_start": pd.Timestamp(period_start) if period_start else pd.NaT,
            "period_end": pd.Timestamp(period_end), "fiscal_year": fiscal_year,
            "cover_fiscal_year": cover_fiscal_year}


def _cisco_frame() -> pd.DataFrame:
    """Cisco's real fiscal calendar (July FYE, 52/53-week, so the year end drifts
    2011-07-30 ... 2017-07-29), with the confirmed defect: the fiscal-2016 10-K
    (accession 0000858877-16-000117) tags its OWN current-period facts
    `fiscal_year=2017`, colliding with the genuine fiscal-2017 10-K."""
    fye = {2011: "2011-07-30", 2012: "2012-07-28", 2013: "2013-07-27", 2014: "2014-07-26",
           2015: "2015-07-25", 2016: "2016-07-30", 2017: "2017-07-29"}
    rows = []
    for fy, end in fye.items():
        native = 2017 if fy == 2016 else fy          # <- the bug, on that one 10-K
        start = pd.Timestamp(end) - pd.Timedelta(days=364)
        rows.append(_fact("10-K", start, end, native, cover_fiscal_year=fy))
    # a Q1 10-Q of each following fiscal year, ~91d after the prior year end
    for fy, end in fye.items():
        if fy + 1 not in fye:
            continue
        q1_end = pd.Timestamp(end) + pd.Timedelta(days=91)
        rows.append(_fact("10-Q", q1_end - pd.Timedelta(days=91), q1_end, fy + 1,
                          cover_fiscal_year=fy + 1))
    return pd.DataFrame(rows)


def _smucker_frame() -> pd.DataFrame:
    """J.M. Smucker (April FYE) with the MIRROR defect: the fiscal-2015 Q1 10-Q
    (accession 0001193125-14-323665, period_end 2014-07-31) has a COVER PAGE
    reading 2014 -- one year early -- while edgartools' per-fact label (2015) is
    right. Proves "prefer the cover page" is not a fix either."""
    rows = []
    for fy in (2012, 2013, 2014, 2015, 2016):
        end = pd.Timestamp(f"{fy}-04-30")
        rows.append(_fact("10-K", end - pd.Timedelta(days=364), end, fy, cover_fiscal_year=fy))
        for n, offset in enumerate((92, 184, 276), start=1):
            q_end = end + pd.Timedelta(days=offset)
            cover = fy + 1 if not (fy == 2014 and n == 1) else 2014   # <- the bug
            rows.append(_fact("10-Q", q_end - pd.Timedelta(days=92), q_end, fy + 1,
                              cover_fiscal_year=cover))
    return pd.DataFrame(rows)


# --- resolve_fiscal_year_by_filing_calendar ---------------------------------

def test_native_fiscal_year_typo_is_outvoted_by_the_calendar():
    """CSCO: the fiscal-2016 10-K's own facts claim 2017 -- re-keyed to 2016, and
    the real fiscal-2017 10-K keeps 2017, so the two no longer collide."""
    out = resolve_fiscal_year_by_filing_calendar(_cisco_frame())
    tenk = out[out["form"] == "10-K"]
    got = dict(zip(tenk["period_end"].dt.strftime("%Y-%m-%d"), tenk["fiscal_year"]))
    assert got["2016-07-30"] == 2016, got
    assert got["2017-07-29"] == 2017, got
    assert sorted(tenk["fiscal_year"]) == [2011, 2012, 2013, 2014, 2015, 2016, 2017]


def test_cover_page_fiscal_year_typo_is_also_outvoted():
    """SJM: the cover page is the wrong source on that one 10-Q -- the calendar
    still places period_end 2014-07-31 in fiscal 2015, with no duplicate Q1."""
    out = resolve_fiscal_year_by_filing_calendar(_smucker_frame())
    q1_2015 = out[out["period_end"] == pd.Timestamp("2014-07-31")]
    assert list(q1_2015["fiscal_year"]) == [2015]
    tenq = out[out["form"] == "10-Q"]
    assert tenq.groupby("fiscal_year").size().max() == 3, "at most Q1-Q3 per fiscal year"


def test_quarters_map_to_the_fiscal_year_that_ends_after_them():
    """A 52/53-week filer's quarter must follow the ACTUAL year end, not a fixed
    anniversary: Cisco's fiscal-2017 Q1 ends 2016-10-29, AFTER the 2016-07-30 year
    end but BEFORE 2017-07-29."""
    out = resolve_fiscal_year_by_filing_calendar(_cisco_frame())
    tenq = out[out["form"] == "10-Q"].sort_values("period_end")
    assert list(tenq["fiscal_year"]) == [2012, 2013, 2014, 2015, 2016, 2017]


def test_in_progress_fiscal_year_is_extrapolated_one_step():
    """The current fiscal year has no 10-K yet, so its quarters sit beyond every
    observed year end and must extrapolate to +1 -- not be clamped to the last."""
    frame = _cisco_frame()
    q1_2018 = _fact("10-Q", "2017-07-30", "2017-10-28", None, cover_fiscal_year=None)
    out = resolve_fiscal_year_by_filing_calendar(
        pd.concat([frame, pd.DataFrame([q1_2018])], ignore_index=True))
    assert out.iloc[-1]["fiscal_year"] == 2018


def test_quarter_before_the_earliest_10k_resolves_backwards():
    """A 53-week year is 371d, so one backward step must still be exactly -1."""
    fye = [pd.Timestamp("2016-07-30"), pd.Timestamp("2017-07-29")]
    assert _fiscal_year_index(pd.Timestamp("2016-04-30"), fye) == 0     # fiscal 2016 Q3
    assert _fiscal_year_index(pd.Timestamp("2015-07-25"), fye) == -1    # prior year end
    assert _fiscal_year_index(pd.Timestamp("2015-04-25"), fye) == -1    # fiscal 2015 Q3
    assert _fiscal_year_index(pd.Timestamp("2017-10-28"), fye) == 2     # fiscal 2018 Q1


def test_untouched_without_an_annual_anchor_or_enough_votes():
    """"Null, never guess wrong": no 10-K to anchor the calendar, or too few
    labels to outvote anything, leaves fiscal_year exactly as it was."""
    only_10q = pd.DataFrame([_fact("10-Q", "2015-01-01", "2015-03-31", 2015, 2015)])
    assert list(resolve_fiscal_year_by_filing_calendar(only_10q)["fiscal_year"]) == [2015]
    unlabelled = pd.DataFrame([_fact("10-K", "2015-01-01", "2015-12-31", None, None)])
    assert resolve_fiscal_year_by_filing_calendar(unlabelled)["fiscal_year"].isna().all()


def test_missing_columns_and_empty_frames_pass_through():
    assert resolve_fiscal_year_by_filing_calendar(pd.DataFrame()).empty
    bare = pd.DataFrame({"fiscal_year": [2020]})
    assert list(resolve_fiscal_year_by_filing_calendar(bare)["fiscal_year"]) == [2020]


# --- period_of_report resolution -------------------------------------------

class _Filing:
    def __init__(self, header, *, raises: bool = False):
        self._header, self._raises = header, raises
        self.form, self.accession_number, self.filing_date = "10-Q", "x", "2013-05-01"

    @property
    def period_of_report(self):
        if self._raises:
            raise TypeError("cannot unpack non-iterable NoneType object")
        return self._header


class _Xbrl:
    def __init__(self, info):
        self.entity_info = info


@pytest.mark.parametrize("header,cover,want", [
    # KeyCorp 0001193125-13-192157: header is a mid-March typo
    ("2013-03-15", "2013-03-31", "2013-03-31"),
    # Packaging Corp 0000075677-17-000004: header is the FILING date
    ("2017-02-28", "2016-12-31", "2016-12-31"),
    # Baker Hughes 0001701605-18-000052: header is one year early
    ("2017-03-31", "2018-03-31", "2018-03-31"),
    # no cover page tag -> the header is all there is
    ("2013-03-31", None, "2013-03-31"),
])
def test_cover_page_period_of_report_wins_over_the_header(header, cover, want):
    got = _resolve_period_of_report(_Filing(header), _Xbrl({"document_period_end_date": cover}))
    assert got == pd.Timestamp(want)


def test_raising_header_does_not_propagate():
    """`filing.period_of_report` lazily fetches the homepage and has been seen
    raising; unguarded it aborted the ENTIRE ticker, not just this filing."""
    assert _resolve_period_of_report(_Filing(None, raises=True), _Xbrl({})) is None
    assert _resolve_period_of_report(
        _Filing(None, raises=True),
        _Xbrl({"document_period_end_date": "2018-03-31"})) == pd.Timestamp("2018-03-31")


def test_cover_fiscal_year_is_coerced_or_none():
    assert _cover_fiscal_year(_Xbrl({"fiscal_year": "2016"})) == 2016
    assert _cover_fiscal_year(_Xbrl({"fiscal_year": None})) is None
    assert _cover_fiscal_year(_Xbrl({})) is None


# --- fiscal-year context re-fetch ------------------------------------------

class _Listed:
    def __init__(self, acc, filing_date):
        self.accession_number, self.filing_date = acc, filing_date


def _quarterly_filings(n: int) -> list[_Listed]:
    start = pd.Timestamp("2013-02-01")
    return [_Listed(f"a{i}", (start + pd.DateOffset(months=3 * i)).date()) for i in range(n)]


def test_context_refetch_pulls_in_the_new_filings_own_fiscal_year():
    """A new filing alone can never yield a derived Q4 (FY - Q1 - Q2 - Q3), since
    its siblings are already in the DB and so absent from the in-memory frame."""
    filings = _quarterly_filings(20)
    done = frozenset(f.accession_number for f in filings[:-1])
    picked = _filings_to_parse(filings, done)
    assert filings[-1] in picked
    window = pd.Timedelta(days=FISCAL_YEAR_CONTEXT_DAYS)
    newest = pd.Timestamp(filings[-1].filing_date)
    assert all(abs(pd.Timestamp(f.filing_date) - newest) <= window for f in picked)
    assert len(picked) >= 5, "the new filing plus at least its own fiscal year"


def test_nothing_new_parses_nothing_and_a_cold_start_parses_everything():
    filings = _quarterly_filings(8)
    assert _filings_to_parse(filings, frozenset(f.accession_number for f in filings)) == []
    assert _filings_to_parse(filings, frozenset()) == filings


# --- sharesOutstanding: point-in-time only ---------------------------------

def test_the_diluted_period_average_is_no_longer_a_shares_outstanding_candidate():
    """It is a period AVERAGE on a DILUTED basis, and already its own field. Being a
    DURATION fact it also escaped the priority coalesce (which groups by period_start,
    NaT for the instant counts), so both were emitted and one was picked by frame
    ordering -- 2,452 rows table-wide resolved to the average, 2,056 to a true count."""
    candidates = SHARES_TAGS[SHARES_OUTSTANDING_FIELD]
    assert DILUTED_SHARES_TAGS == ["WeightedAverageNumberOfDilutedSharesOutstanding"]
    assert not any("WeightedAverage" in t for t in candidates)
    # Every candidate must be a POINT-IN-TIME count of shares outstanding -- the property
    # this test exists to protect. `SharesOutstandingAsConvertedBasis` (added ahead of the
    # two below for the multi-class filers that publish an already-converted total) is one:
    # an instant fact dated at period end, not a period average.
    assert candidates == ["SharesOutstandingAsConvertedBasis",
                          "CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"]


def _shares_row(tag: str, period_end: str, value: float) -> dict:
    return {"field": SHARES_OUTSTANDING_FIELD, "value": value, "period_start": pd.NaT,
            "period_end": pd.Timestamp(period_end), "source_tag": f"dei:{tag}"}


_POR = pd.Timestamp("2025-12-31")


def test_filing_dated_cover_page_count_is_recovered_onto_the_period():
    """GPC/CB/JBHT all state the cover-page count weeks AFTER the period of report
    (2026-02-17 vs 2025-12-31), so the current-period filter dropped it and the field
    fell through to the diluted average. Re-stamped to the period of report so it
    keys like every other instant fact."""
    all_periods = pd.DataFrame([_shares_row(COVER_PAGE_SHARES_TAG, "2026-02-17", 137_622_108)])
    out = _cover_page_shares_fallback(all_periods, all_periods.iloc[0:0], _POR)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 137_622_108
    assert out.iloc[0]["period_end"] == _POR


def test_an_as_reported_balance_sheet_count_is_never_overridden():
    """Fill-only -- otherwise the two concepts both produce a row for one filing and
    the arbitrary pick is back."""
    current = pd.DataFrame([_shares_row("CommonStockSharesOutstanding", "2025-12-31", 137_617_832)])
    all_periods = pd.concat(
        [current, pd.DataFrame([_shares_row(COVER_PAGE_SHARES_TAG, "2026-02-17", 137_622_108)])],
        ignore_index=True)
    out = _cover_page_shares_fallback(all_periods, current, _POR)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 137_617_832


def test_the_earliest_qualifying_cover_date_wins_and_stale_ones_are_ignored():
    """A cover page can be re-filed; and a fact from an unrelated later period must
    not be pulled in (the window is forward-only and bounded)."""
    rows = pd.DataFrame([
        _shares_row(COVER_PAGE_SHARES_TAG, "2026-04-30", 1_000),   # a later re-filing
        _shares_row(COVER_PAGE_SHARES_TAG, "2026-02-17", 2_000),   # the real one
    ])
    assert _cover_page_shares_fallback(rows, rows.iloc[0:0], _POR).iloc[0]["value"] == 2_000

    too_late = pd.DataFrame([_shares_row(
        COVER_PAGE_SHARES_TAG,
        (_POR + pd.Timedelta(days=COVER_PAGE_SHARES_MAX_LAG_DAYS + 1)).strftime("%Y-%m-%d"), 3_000)])
    assert _cover_page_shares_fallback(too_late, too_late.iloc[0:0], _POR).empty

    before = pd.DataFrame([_shares_row(COVER_PAGE_SHARES_TAG, "2025-09-30", 4_000)])
    assert _cover_page_shares_fallback(before, before.iloc[0:0], _POR).empty
