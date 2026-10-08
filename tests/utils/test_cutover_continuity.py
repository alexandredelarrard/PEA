"""Cutover continuity (src/utils/cutover_continuity.py): the classes of a missing vendor quarter, the recorded
exceptions in `configs/sec/vendor_coverage_exceptions.json`, and the predecessor-series replacement inside a
register window (D-Q2-8). Known-truth fixtures, no database.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.utils import cutover_continuity as cc

CONFIG_DIR = Path("./configs")
SEAM = pd.Timestamp("2020-01-01")
OLD, NEW = "0000000001", "0000000002"
CHAIN = (cc.CikWindow(OLD, None, SEAM), cc.CikWindow(NEW, SEAM, None))
LABEL = "Sharadar/source coverage gap; SEC filings present"
RECORD = cc.VendorException("TST", "2019Q2", "2019-06-30", OLD, f"{OLD}-19-000001", "2019-08-01", LABEL)
FILING_COLUMNS = ["cik", "accession_number", "form", "filing_date"]


def _stored(cik: str, filed: str) -> pd.DataFrame:
    return pd.DataFrame([[cik, f"{cik}-19-000001", "10-Q", pd.Timestamp(filed)]], columns=FILING_COLUMNS)


def _arq(ticker: str, quarters: list[str], *, assets: float = 100.0, shift_days: int = 0, filed_lag: int = 40) -> pd.DataFrame:
    """ARQ rows for `quarters` (`2019Q2` labels): calendardate at the quarter end, reportperiod `shift_days` before it."""
    ends = [pd.Period(q, freq="Q").end_time.normalize() for q in quarters]
    return pd.DataFrame(
        {
            "ticker": ticker,
            "dimension": "ARQ",
            "calendardate": ends,
            "reportperiod": [e - pd.Timedelta(days=shift_days) for e in ends],
            "date": [e + pd.Timedelta(days=filed_lag) for e in ends],
            "assets": assets,
        }
    )


def test_cutover_classification_rules() -> None:
    """Known-truth chain (old CIK to 2020-01-01, new CIK after): each class and the record checks."""
    empty = pd.DataFrame(columns=FILING_COLUMNS)
    cases = {
        "old CIK files inside its window": (_stored(OLD, "2019-08-01"), RECORD, cc.VENDOR_COVERAGE_GAP, True),
        "new CIK inside the 31-day seam margin": (_stored(NEW, "2019-12-15"), RECORD, cc.VENDOR_COVERAGE_GAP, True),
        "new CIK long before its window": (_stored(NEW, "2019-08-01"), RECORD, cc.INCORRECT_CIK_WINDOW, False),
        "a CIK outside the chain": (_stored("0000000009", "2019-08-01"), RECORD, cc.INCORRECT_CIK_WINDOW, False),
        "vendor gap with no exception row": (_stored(OLD, "2019-08-01"), None, cc.VENDOR_COVERAGE_GAP, False),
        "nothing stored, no record": (empty, None, cc.MISSING_SEC_FILING, False),
        "nothing stored, record admitted": (empty, RECORD, cc.VENDOR_COVERAGE_GAP, True),
        "record for another quarter": (empty, replace(RECORD, period_end="2019-09-30"), cc.VENDOR_COVERAGE_GAP, False),
        "record CIK outside its window": (empty, replace(RECORD, filed="2020-06-01"), cc.INCORRECT_CIK_WINDOW, False),
    }
    print("\n=== SANITY CHECK: cutover discontinuity classes on a known-truth chain ===")
    for name, (filings, rec, want_cls, want_ok) in cases.items():
        cls, ok, evidence = cc.classify(CHAIN, "2019Q2", filings, rec)
        print(f"  {name:40s} -> {cls:20s} explained={ok}")
        assert (cls, ok) == (want_cls, want_ok), f"{name}: got {cls}/{ok}, expected {want_cls}/{want_ok} ({evidence})"
    print(f"  OK: all {len(cases)} cases classify as defined; only a recorded, admitted vendor gap is explained.")


def test_the_exceptions_config_holds_the_4_accession_exact_rows() -> None:
    """DOW's 2018Q1-Q2 rows left with TDCC (its accessions are TDCC's; DOW starts with Dow Inc. on 2019-04-01)."""
    rows = cc.load_vendor_exceptions(CONFIG_DIR)
    keys = [(r.ticker, r.quarter) for r in rows]
    assert len(rows) == 4 and len(set(keys)) == 4
    assert {r.ticker for r in rows} == {"BKR", "VMC"}
    assert all(r.accession and r.cik and r.label == LABEL and cc.quarter_of(r.period_end) == r.quarter for r in rows)
    assert ("BKR", "2017Q2", "0000808362-17-000034") in [(r.ticker, r.quarter, r.accession) for r in rows]
    print("\n=== SANITY CHECK: configs/sec/vendor_coverage_exceptions.json ===")
    print(f"  {len(rows)} rows, one per (ticker, quarter), each with its accession and the vendor-gap label")


def test_assess_continuity_explains_records_and_names_healed_rows() -> None:
    """A recorded gap is explained; a recorded quarter present again is healed; an unrecorded one is not explained."""
    windows = {"TST": CHAIN}
    arq = _arq("TST", ["2019Q1", "2019Q3", "2019Q4", "2020Q1", "2020Q3"])  # 2019Q2 and 2020Q2 missing
    filings = cc.prepare_filings(
        pd.DataFrame([["TST", OLD, f"{OLD}-19-000001", "10-Q", pd.Timestamp("2019-08-01"), "2019-06-30"]], columns=list(cc.FILING_COLUMNS))
    )
    healed_row = replace(RECORD, quarter="2019Q4", period_end="2019-12-31", accession=f"{OLD}-20-000002", filed="2020-02-10")
    report = cc.assess_continuity(arq, windows, filings, (RECORD, healed_row))
    table = report.table.set_index("quarter")
    print("\n=== SANITY CHECK: continuity assessment on a fixture ===")
    print(report.table[["ticker", "quarter", "class", "explained", "accession"]].to_string(index=False))
    assert table.loc["2019Q2", "explained"] and table.loc["2019Q2", "accession"] == RECORD.accession
    assert table.loc["2020Q2", "class"] == cc.MISSING_SEC_FILING and not table.loc["2020Q2", "explained"]
    assert [r.quarter for r in report.healed] == ["2019Q4"]
    assert report.unobserved == () and report.duplicated == ()
    dup = cc.assess_continuity(arq, windows, filings, (RECORD, RECORD))
    assert dup.duplicated == ("TST 2019Q2",)
    print("  OK: recorded gap explained with its accession; 2020Q2 unexplained; the filled 2019Q4 record is named healed.")


def test_register_windows_and_predecessor_series_are_derived_from_rows() -> None:
    lineage = pd.DataFrame(
        [
            {
                "canonical_ticker": "PLD",
                "cik": "899881",
                "role": "cik_window",
                "valid_from": "1900-01-01",
                "valid_to": "2011-06-03",
                "sources": "register",
            },
            {"canonical_ticker": "PLD", "cik": "1045609", "role": "cik_window", "valid_from": "2011-06-03", "valid_to": None, "sources": "register"},
            {
                "canonical_ticker": "XOM",
                "cik": "34088",
                "role": "cik_window",
                "valid_from": "1900-01-01",
                "valid_to": "2026-07-01",
                "sources": "register",
            },
            {"canonical_ticker": "XOM", "cik": "2115436", "role": "cik_window", "valid_from": "2026-07-01", "valid_to": None, "sources": "register"},
            {"canonical_ticker": "AAA", "cik": "5", "role": "cik_window", "valid_from": "1900-01-01", "valid_to": None, "sources": "roster"},
            {
                "canonical_ticker": "MRK",
                "cik": "64978",
                "role": "cik_window",
                "valid_from": "1900-01-01",
                "valid_to": "2009-11-03",
                "sources": "register",
            },
        ]
    )
    windows = cc.register_windows(lineage)
    assert set(windows) == {"PLD", "XOM", "MRK"}
    assert windows["PLD"][0] == cc.CikWindow("0000899881", None, pd.Timestamp("2011-06-03"))
    assert cc.boundaries(windows["PLD"]) == (pd.Timestamp("2011-06-03"),)
    url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={}&type=&dateb=&owner=include&count=40"
    vendor = pd.DataFrame(
        [
            {"ticker": "PLD1", "secfilings": url.format("0000899881"), "lastquarter": "2011-03-31"},
            {"ticker": "PLD", "secfilings": url.format("0001045609"), "lastquarter": "2026-06-30"},
            {"ticker": "XOM", "secfilings": url.format("0000034088"), "lastquarter": "2026-06-30"},
            {"ticker": "SGP1", "secfilings": url.format("0000310158"), "lastquarter": "2009-09-30"},
        ]
    )
    series = cc.predecessor_series(vendor, windows, ["PLD", "XOM", "MRK"])
    print("\n=== SANITY CHECK: predecessor vendor tickers derived from sharadar_tickers + register windows ===")
    for s in series:
        print(f"  {s.ticker}: {s.vendor_ticker} (CIK {s.cik}) owns ..{s.valid_to.date() if s.valid_to else 'open'}")
    assert series == (cc.PredecessorSeries("PLD", "PLD1", "0000899881", None, pd.Timestamp("2011-06-03")),)
    print("  OK: PLD1 only -- XOM's vendor ticker is a universe ticker, SGP1's CIK is a current CIK, MRK's predecessor has none.")


def test_predecessor_series_replace_the_canonical_rows_inside_the_window_only() -> None:
    """D-Q2-8: inside the window the owner's rows replace the canonical ones (relabelled); outside nothing moves;
    a quarter only the owner has is filled; each quarter is logged once."""
    seam = pd.Timestamp("2011-06-03")
    series = (cc.PredecessorSeries("PLD", "PLD1", "0000899881", None, seam),)
    canonical = _arq("PLD", ["2010Q3", "2010Q4", "2011Q2", "2011Q3"], assets=7.0)  # AMB's, 2011Q1 missing
    owner = _arq("PLD1", ["2010Q2", "2010Q3", "2010Q4", "2011Q1"], assets=20.0)
    outside = _arq("PLD1", ["2011Q2"], assets=99.0)  # a stray owner row after the seam: never inserted (E28)
    merged, events = cc.apply_predecessor_series(canonical, pd.concat([owner, outside], ignore_index=True), series)
    merged = merged.assign(quarter=[cc.quarter_of(d) for d in merged["calendardate"]]).set_index("quarter").sort_index()
    print("\n=== SANITY CHECK: predecessor series inside the window (D-Q2-8) ===")
    print(events.to_string(index=False))
    assert merged["ticker"].eq("PLD").all()
    assert merged.loc[["2010Q2", "2010Q3", "2010Q4", "2011Q1"], "assets"].eq(20.0).all()
    assert merged.loc[["2011Q2", "2011Q3"], "assets"].eq(7.0).all(), "a row outside the window moved"
    assert dict(zip(events["quarter"], events["event"], strict=True)) == {
        "2010Q2": "filled",
        "2010Q3": "replaced",
        "2010Q4": "replaced",
        "2011Q1": "filled",
    }
    assert merged.index.is_unique
    print("  OK: 2 replaced, 2 filled (2011Q1 heals the hole), nothing outside the window touched, one row per quarter.")


def test_e28_present_or_outside_quarters_are_never_inserted_twice() -> None:
    series = (cc.PredecessorSeries("STE", "STE1", "0000815065", None, pd.Timestamp("2015-10-31")),)
    canonical = _arq("STE", ["2014Q1", "2015Q4", "2016Q1"], assets=5.0)
    owner = _arq("STE1", ["2014Q1", "2015Q4"], assets=6.0)  # 2015Q4 ends after the window
    merged, events = cc.apply_predecessor_series(canonical, owner, series)
    labels = [cc.quarter_of(d) for d in merged["calendardate"]]
    assert sorted(labels) == ["2014Q1", "2015Q4", "2016Q1"]
    assert merged.loc[[q == "2015Q4" for q in labels], "assets"].tolist() == [5.0]
    assert events[["quarter", "event"]].values.tolist() == [["2014Q1", "replaced"]]
    unchanged, none = cc.apply_predecessor_series(canonical, owner.iloc[:0], series)
    assert none.empty and unchanged.equals(canonical)
    print("\n=== SANITY CHECK: E28 ===")
    print("  OK: an owner quarter outside the window is not inserted; a present quarter is replaced once; no owner rows -> no change.")


def test_e29_a_duplicated_owner_quarter_counts_once_and_is_reported() -> None:
    series = (cc.PredecessorSeries("STE", "STE1", "0000815065", None, pd.Timestamp("2015-10-31")),)
    canonical = _arq("STE", ["2014Q1", "2015Q4"], assets=5.0)
    owner = pd.concat(
        [_arq("STE1", ["2014Q1"], assets=6.0)] + [_arq("STE1", ["2014Q2"], assets=6.0, filed_lag=40 + i) for i in range(4)], ignore_index=True
    )
    merged, events = cc.apply_predecessor_series(canonical, owner, series)
    row = events.set_index("quarter").loc["2014Q2"]
    print("\n=== SANITY CHECK: E29 ===")
    print(events.to_string(index=False))
    assert row["event"] == "filled" and int(row["predecessor_rows"]) == 4
    assert int(events["event"].eq("filled").sum()) == 1
    report = cc.assess_continuity(
        merged,
        {"STE": (cc.CikWindow("0000815065", None, pd.Timestamp("2015-10-31")),) + ((cc.CikWindow("0001624899", pd.Timestamp("2015-10-31"), None)),)},
        cc.prepare_filings(None),
        (),
    )
    assert "2014Q2" not in set(report.table["quarter"])
    print("  OK: four filings of 2014Q2 are one filled quarter (predecessor_rows=4), not four fills.")


def test_other_company_quarters_compare_assets_inside_the_window() -> None:
    series = cc.PredecessorSeries("DD", "DD1", "0000030554", None, pd.Timestamp("2017-08-31"))
    canonical = pd.concat([_arq("DD", ["2017Q1"], assets=80.8e9), _arq("DD", ["2017Q2"], assets=41.0e9), _arq("DD", ["2017Q3"], assets=190e9)])
    owner = _arq("DD1", ["2017Q1", "2017Q2"], assets=41.0e9)
    out = cc.other_company_quarters(canonical, owner, series)
    assert out["quarter"].tolist() == ["2017Q1"]
    print("\n=== SANITY CHECK: vendor series of another company ===")
    print(out.to_string(index=False))
    print("  OK: 2017Q1 ($80.8bn vs $41.0bn) is another company's; 2017Q2 agrees; 2017Q3 is outside the window.")
