"""Unit tests for the SEC Financial Statement & Notes extractor
(src/data_extract/utils/fundamentals/fetch_financial_notes.py).

Covers, with NO network:
  * the pure parse/join (num + text) incl. the dimn==0 / coreg / universe filters,
  * the full zip IO path via an in-memory synthetic notes zip,
  * the rolling quarterly<->monthly period logic + year window.
A separate script (scripts-style) exercises a REAL quarterly zip end-to-end.
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

import src.data_extract.utils.fundamentals.fetch_financial_notes as fn


# --------------------------------------------------------------------------- #
# Synthetic in-memory notes zip                                                  #
# --------------------------------------------------------------------------- #
_SUB_COLS = ["adsh", "cik", "name", "form", "period", "fy", "fp", "filed"]
_NUM_COLS = ["adsh", "tag", "version", "ddate", "qtrs", "uom", "dimh", "iprx",
             "value", "footnote", "footlen", "dimn", "coreg", "durp", "datp", "dcml"]
_TXT_COLS = ["adsh", "tag", "version", "ddate", "qtrs", "iprx", "lang", "dcml",
             "durp", "datp", "dimh", "dimn", "coreg", "escaped", "srclen",
             "txtlen", "footnote", "footlen", "context", "value"]

AAPL = "0000320193-24-000001"      # universe filer (AAPL 10-K)
OTHER = "0000000001-24-000009"     # NOT in universe


def _row(cols: list[str], **kw) -> str:
    return "\t".join(str(kw.get(c, "")) for c in cols)


def _synthetic_zip() -> io.BytesIO:
    sub = [
        "\t".join(_SUB_COLS),
        _row(_SUB_COLS, adsh=AAPL, cik="320193", name="APPLE INC", form="10-K",
             period="20240930", fy="2024", fp="FY", filed="20241101"),
        _row(_SUB_COLS, adsh=OTHER, cik="1", name="NOT IN UNIVERSE", form="10-K",
             period="20241231", fy="2024", fp="FY", filed="20250201"),
    ]
    num = [
        "\t".join(_NUM_COLS),
        # KEEP: undimensioned consolidated PBO + plan assets for AAPL
        _row(_NUM_COLS, adsh=AAPL, tag="DefinedBenefitPlanBenefitObligation",
             ddate="20240930", qtrs="0", uom="USD", dimn="0", value="1000000"),
        _row(_NUM_COLS, adsh=AAPL, tag="DefinedBenefitPlanFairValueOfPlanAssets",
             ddate="20240930", qtrs="0", uom="USD", dimn="0", value="800000"),
        _row(_NUM_COLS, adsh=AAPL, tag="DefinedBenefitPlanServiceCost",
             ddate="20240930", qtrs="4", uom="USD", dimn="0", value="50000"),
        # DROP: dimensioned (pension-vs-OPEB breakdown) -> dimn>0
        _row(_NUM_COLS, adsh=AAPL, tag="DefinedBenefitPlanBenefitObligation",
             ddate="20240930", qtrs="0", uom="USD", dimn="1", value="600000"),
        # DROP: co-registrant subsidiary line
        _row(_NUM_COLS, adsh=AAPL, tag="DefinedBenefitPlanFairValueOfPlanAssets",
             ddate="20240930", qtrs="0", uom="USD", dimn="0", coreg="SUBSID", value="1"),
        # DROP: not a pension tag
        _row(_NUM_COLS, adsh=AAPL, tag="Assets", ddate="20240930", qtrs="0",
             uom="USD", dimn="0", value="999999"),
        # DROP: non-universe filer
        _row(_NUM_COLS, adsh=OTHER, tag="DefinedBenefitPlanBenefitObligation",
             ddate="20241231", qtrs="0", uom="USD", dimn="0", value="123"),
    ]
    txt = [
        "\t".join(_TXT_COLS),
        # KEEP: high-signal pension note text (undimensioned)
        _row(_TXT_COLS, adsh=AAPL, tag="PensionAndOtherPostretirementBenefitPlansFullDisclosureTextBlock",
             ddate="20240930", qtrs="0", dimn="0", escaped="1", txtlen="42",
             value="The Company sponsors defined benefit plans."),
        # KEEP: revenue recognition policy text
        _row(_TXT_COLS, adsh=AAPL, tag="RevenueRecognitionPolicyTextBlock",
             ddate="20240930", qtrs="4", dimn="0", txtlen="20", value="Revenue is recognized."),
        # DROP: not a high-signal tag
        _row(_TXT_COLS, adsh=AAPL, tag="SomeRandomTextBlock", ddate="20240930",
             qtrs="0", dimn="0", txtlen="5", value="noise"),
        # DROP: dimensioned text
        _row(_TXT_COLS, adsh=AAPL, tag="SegmentReportingDisclosureTextBlock",
             ddate="20240930", qtrs="0", dimn="2", txtlen="5", value="dim"),
        # DROP: non-universe filer
        _row(_TXT_COLS, adsh=OTHER, tag="SegmentReportingDisclosureTextBlock",
             ddate="20241231", qtrs="0", dimn="0", txtlen="5", value="other"),
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("sub.tsv", "\n".join(sub))
        z.writestr("num.tsv", "\n".join(num))
        z.writestr("txt.tsv", "\n".join(txt))
    buf.seek(0)
    return buf


@pytest.fixture
def zip_path(tmp_path):
    p = tmp_path / "2024q4_notes.zip"
    p.write_bytes(_synthetic_zip().getvalue())
    return p


# --------------------------------------------------------------------------- #
# Parse / IO                                                                     #
# --------------------------------------------------------------------------- #
def test_read_notes_filters_and_joins(zip_path):
    cik2tkr = {"0000320193": "AAPL"}
    num, txt = fn._read_notes(zip_path, cik2tkr, {"AAPL"})

    # NUM: exactly the 3 undimensioned, consolidated, pension-tag, universe rows
    assert set(num["tag"]) == {
        "DefinedBenefitPlanBenefitObligation",
        "DefinedBenefitPlanFairValueOfPlanAssets",
        "DefinedBenefitPlanServiceCost",
    }
    assert len(num) == 3
    assert (num["ticker"] == "AAPL").all()
    assert (num["cik"] == "0000320193").all()
    pbo = num.loc[num["tag"] == "DefinedBenefitPlanBenefitObligation", "value"].iloc[0]
    assert pbo == 1_000_000                                   # the dimn==0 total, NOT the 600k member
    assert pd.api.types.is_datetime64_any_dtype(num["ddate"])

    # TXT: only the 2 high-signal undimensioned blocks
    assert set(txt["tag"]) == {
        "PensionAndOtherPostretirementBenefitPlansFullDisclosureTextBlock",
        "RevenueRecognitionPolicyTextBlock",
    }
    assert (txt["ticker"] == "AAPL").all()
    assert txt["value"].str.len().gt(0).all()

    print("\n=== SANITY: notes zip parse ===")
    print(f"  num rows kept = {len(num)} (dropped: dimn>0, coreg, non-pension, non-universe)")
    print(f"  PBO total = {pbo:,.0f} (undimensioned, not the 600k pension-only member)")
    print(f"  text tags kept = {sorted(txt['tag'])}")
    print("  -> dimn==0 / coreg / tag / universe filters all correct.")


def test_read_notes_empty_when_universe_disjoint(zip_path):
    num, txt = fn._read_notes(zip_path, {"0000320193": "AAPL"}, {"MSFT"})
    assert num.empty and txt.empty


def test_join_num_drops_nan_value():
    sub_meta = pd.DataFrame({"adsh": [AAPL], "cik": ["0000320193"], "ticker": ["AAPL"],
                             "form": ["10-K"], "fy": ["2024"], "fp": ["FY"],
                             "filed": [pd.Timestamp("2024-11-01")]})
    num = pd.DataFrame({"adsh": [AAPL, AAPL], "tag": ["DefinedBenefitPlanBenefitObligation"] * 2,
                        "ddate": ["20240930", "20240930"], "qtrs": ["0", "0"],
                        "uom": ["USD", "USD"], "value": ["1000000", ""], "footnote": ["", ""]})
    out = fn._join_notes_num(num, sub_meta)
    assert len(out) == 1 and out["value"].iloc[0] == 1_000_000


# --------------------------------------------------------------------------- #
# Period logic (rolling quarterly <-> monthly)                                   #
# --------------------------------------------------------------------------- #
def test_generate_periods_has_quarterly_and_recent_monthly():
    today = pd.Timestamp("2026-07-19")
    periods = fn._generate_periods(years_history=15, today=today)
    assert "2024q1" in periods and "2012q3" in periods          # quarterly era
    assert "2026_06" in periods and "2025_07" in periods        # recent monthly
    assert all(fn._period_year(p) >= 2011 for p in periods)     # year window respected


def test_notes_periods_uses_scrape_when_available(monkeypatch):
    fake = ["2009q1", "2020q3", "2025_07", "2026_06"]
    monkeypatch.setattr(fn, "_scrape_available_periods", lambda context: fake)
    got = fn._notes_periods(None, years_history=3, today=pd.Timestamp("2026-07-19"))
    assert got == ["2025_07", "2026_06"]                        # 2009/2020 outside 3y window


def test_notes_periods_falls_back_to_generator(monkeypatch):
    monkeypatch.setattr(fn, "_scrape_available_periods", lambda context: None)
    got = fn._notes_periods(None, years_history=2, today=pd.Timestamp("2026-07-19"))
    assert got and all(fn._period_year(p) >= 2024 for p in got)
