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
import os
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

import src.data_extract.utils.fundamentals.fetch_financial_notes as fn
from src.constants.constants import MARKET_TIMEZONE
from src.data_extract.utils.common import bulk_cache

# --------------------------------------------------------------------------- #
# Synthetic in-memory notes zip                                                  #
# --------------------------------------------------------------------------- #
_SUB_COLS = ["adsh", "cik", "name", "form", "period", "fy", "fp", "filed"]
_NUM_COLS = [
    "adsh",
    "tag",
    "version",
    "ddate",
    "qtrs",
    "uom",
    "dimh",
    "iprx",
    "value",
    "footnote",
    "footlen",
    "dimn",
    "coreg",
    "durp",
    "datp",
    "dcml",
]
_TXT_COLS = [
    "adsh",
    "tag",
    "version",
    "ddate",
    "qtrs",
    "iprx",
    "lang",
    "dcml",
    "durp",
    "datp",
    "dimh",
    "dimn",
    "coreg",
    "escaped",
    "srclen",
    "txtlen",
    "footnote",
    "footlen",
    "context",
    "value",
]

AAPL = "0000320193-24-000001"  # universe filer (AAPL 10-K)
OTHER = "0000000001-24-000009"  # NOT in universe


def _row(cols: list[str], **kw) -> str:
    return "\t".join(str(kw.get(c, "")) for c in cols)


def _synthetic_zip() -> io.BytesIO:
    sub = [
        "\t".join(_SUB_COLS),
        _row(_SUB_COLS, adsh=AAPL, cik="320193", name="APPLE INC", form="10-K", period="20240930", fy="2024", fp="FY", filed="20241101"),
        _row(_SUB_COLS, adsh=OTHER, cik="1", name="NOT IN UNIVERSE", form="10-K", period="20241231", fy="2024", fp="FY", filed="20250201"),
    ]
    num = [
        "\t".join(_NUM_COLS),
        # KEEP: undimensioned consolidated PBO + plan assets for AAPL
        _row(_NUM_COLS, adsh=AAPL, tag="DefinedBenefitPlanBenefitObligation", ddate="20240930", qtrs="0", uom="USD", dimn="0", value="1000000"),
        _row(_NUM_COLS, adsh=AAPL, tag="DefinedBenefitPlanFairValueOfPlanAssets", ddate="20240930", qtrs="0", uom="USD", dimn="0", value="800000"),
        _row(_NUM_COLS, adsh=AAPL, tag="DefinedBenefitPlanServiceCost", ddate="20240930", qtrs="4", uom="USD", dimn="0", value="50000"),
        # DROP: dimensioned (pension-vs-OPEB breakdown) -> dimn>0
        _row(_NUM_COLS, adsh=AAPL, tag="DefinedBenefitPlanBenefitObligation", ddate="20240930", qtrs="0", uom="USD", dimn="1", value="600000"),
        # DROP: co-registrant subsidiary line
        _row(
            _NUM_COLS,
            adsh=AAPL,
            tag="DefinedBenefitPlanFairValueOfPlanAssets",
            ddate="20240930",
            qtrs="0",
            uom="USD",
            dimn="0",
            coreg="SUBSID",
            value="1",
        ),
        # DROP: not a pension tag
        _row(_NUM_COLS, adsh=AAPL, tag="Assets", ddate="20240930", qtrs="0", uom="USD", dimn="0", value="999999"),
        # DROP: non-universe filer
        _row(_NUM_COLS, adsh=OTHER, tag="DefinedBenefitPlanBenefitObligation", ddate="20241231", qtrs="0", uom="USD", dimn="0", value="123"),
    ]
    txt = [
        "\t".join(_TXT_COLS),
        # KEEP: high-signal pension note text (undimensioned)
        _row(
            _TXT_COLS,
            adsh=AAPL,
            tag="PensionAndOtherPostretirementBenefitPlansFullDisclosureTextBlock",
            ddate="20240930",
            qtrs="0",
            dimn="0",
            escaped="1",
            txtlen="42",
            value="The Company sponsors defined benefit plans.",
        ),
        # KEEP: revenue recognition policy text
        _row(
            _TXT_COLS,
            adsh=AAPL,
            tag="RevenueRecognitionPolicyTextBlock",
            ddate="20240930",
            qtrs="4",
            dimn="0",
            txtlen="20",
            value="Revenue is recognized.",
        ),
        # DROP: not a high-signal tag
        _row(_TXT_COLS, adsh=AAPL, tag="SomeRandomTextBlock", ddate="20240930", qtrs="0", dimn="0", txtlen="5", value="noise"),
        # DROP: dimensioned text
        _row(_TXT_COLS, adsh=AAPL, tag="SegmentReportingDisclosureTextBlock", ddate="20240930", qtrs="0", dimn="2", txtlen="5", value="dim"),
        # DROP: non-universe filer
        _row(_TXT_COLS, adsh=OTHER, tag="SegmentReportingDisclosureTextBlock", ddate="20241231", qtrs="0", dimn="0", txtlen="5", value="other"),
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
    num, txt = fn._read_notes(zip_path, cik2tkr, {"AAPL"}, {})

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
    assert pbo == 1_000_000  # the dimn==0 total, NOT the 600k member
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
    num, txt = fn._read_notes(zip_path, {"0000320193": "AAPL"}, {"MSFT"}, {})
    assert num.empty and txt.empty


def test_join_num_drops_nan_value():
    sub_meta = pd.DataFrame(
        {
            "adsh": [AAPL],
            "cik": ["0000320193"],
            "ticker": ["AAPL"],
            "form": ["10-K"],
            "fy": ["2024"],
            "fp": ["FY"],
            "filed": [pd.Timestamp("2024-11-01")],
        }
    )
    num = pd.DataFrame(
        {
            "adsh": [AAPL, AAPL],
            "tag": ["DefinedBenefitPlanBenefitObligation"] * 2,
            "ddate": ["20240930", "20240930"],
            "qtrs": ["0", "0"],
            "uom": ["USD", "USD"],
            "value": ["1000000", ""],
            "footnote": ["", ""],
        }
    )
    out = fn._join_notes_num(num, sub_meta)
    assert len(out) == 1 and out["value"].iloc[0] == 1_000_000


# --------------------------------------------------------------------------- #
# Period logic (rolling quarterly <-> monthly)                                   #
# --------------------------------------------------------------------------- #
def test_generate_periods_has_quarterly_and_recent_monthly():
    today = pd.Timestamp("2026-07-19")
    periods = fn._generate_periods(years_history=15, today=today)
    assert "2024q1" in periods and "2012q3" in periods  # quarterly era
    assert "2026_06" in periods and "2025_07" in periods  # recent monthly
    assert all(fn._period_year(p) >= 2011 for p in periods)  # year window respected


def test_notes_periods_uses_scrape_when_available(monkeypatch):
    fake = ["2009q1", "2020q3", "2025_07", "2026_06"]
    monkeypatch.setattr(fn, "_scrape_available_periods", lambda context: fake)
    got = fn._notes_periods(cast(Any, None), years_history=3, today=pd.Timestamp("2026-07-19"))
    assert got == ["2025_07", "2026_06"]  # 2009/2020 outside 3y window


def test_notes_periods_falls_back_to_generator(monkeypatch):
    monkeypatch.setattr(fn, "_scrape_available_periods", lambda context: None)
    got = fn._notes_periods(cast(Any, None), years_history=2, today=pd.Timestamp("2026-07-19"))
    assert got and all(fn._period_year(p) >= 2024 for p in got)


@pytest.mark.parametrize(
    ("period", "expected"),
    [
        ("2021_08", date(2021, 9, 13)),  # September 12 was Sunday.
        ("2013q4", date(2014, 1, 13)),  # January 12 was Sunday.
        ("2025_08", date(2025, 9, 12)),
        ("2026_08", date(2026, 9, 14)),  # September 12 is Saturday.
    ],
)
def test_historical_archive_available_at_is_next_month_twelfth_or_monday(period: str, expected: date, tmp_path: Path) -> None:
    assert (
        bulk_cache.archive_available_at(
            bulk_cache.period_end(period), tmp_path / f"{period}_notes.zip", observed_from=fn._OBSERVED_FROM, downloaded=False
        )
        == expected
    )
    print("\n=== SANITY CHECK: estimated historical notes availability ===")
    print(f"  {period} becomes available on {expected}; weekend twelfths move to Monday. Validated.")


def test_future_archive_available_at_is_cached_download_date(tmp_path: Path) -> None:
    path = tmp_path / "2026_09_notes.zip"
    path.write_bytes(b"cached")
    downloaded = datetime(2026, 10, 16, 0, 30, tzinfo=UTC)  # Still October 15 in New York.
    os.utime(path, (downloaded.timestamp(), downloaded.timestamp()))

    assert bulk_cache.archive_available_at(bulk_cache.period_end("2026_09"), path, observed_from=fn._OBSERVED_FROM, downloaded=False) == date(
        2026, 10, 15
    )
    assert (
        bulk_cache.archive_available_at(bulk_cache.period_end("2026_09"), tmp_path / "missing.zip", observed_from=fn._OBSERVED_FROM, downloaded=False)
        is None
    )
    print("\n=== SANITY CHECK: observed future notes availability ===")
    print("  the first New York download date, even after the twelfth, is used; missing archives have no clock. Validated.")


def test_historical_availability_repair_writes_only_primary_keys_and_clock():
    class Store:
        def __init__(self) -> None:
            self.saved: list[tuple[object, pd.DataFrame]] = []

        @staticmethod
        def columns(table) -> list[str]:
            return [*table.pk, "period"]

        @staticmethod
        def load(table, *, columns, where, optional):
            assert columns == [*table.pk, "period"]
            assert where == {"period": "2026_08"}
            assert optional is True
            return pd.DataFrame(
                {
                    "adsh": [AAPL],
                    "tag": ["DefinedBenefitPlanBenefitObligation"],
                    "ddate": [pd.Timestamp("2024-09-30")],
                    "qtrs": [0],
                    "period": ["2026_08"],
                }
            )

        def save(self, table, frame: pd.DataFrame) -> int:
            self.saved.append((table, frame.copy()))
            return len(frame)

    store = Store()
    repaired = fn._repair_period_available_at(SimpleNamespace(store=store), "2026_08", date(2026, 9, 14))

    assert repaired == 2
    assert [table for table, _ in store.saved] == [fn.Tables.notes_num, fn.Tables.notes_text]
    for table, frame in store.saved:
        assert list(frame.columns) == [*table.pk, "available_at"]
        assert frame["available_at"].eq(date(2026, 9, 14)).all()

    print("\n=== SANITY CHECK: historical notes metadata repair ===")
    print("  both notes tables receive only their primary key plus available_at; payload columns are not rewritten. Validated.")


def test_historical_availability_repair_replaces_wrong_nonnull_clock() -> None:
    saved: list[pd.DataFrame] = []

    class Store:
        @staticmethod
        def columns(table) -> list[str]:
            return [*table.pk, "period", "available_at"]

        @staticmethod
        def load(table, *, columns, where, optional):
            assert columns == [*table.pk, "period"]
            assert where == {"period": "2021_08"}
            return pd.DataFrame(
                {
                    "adsh": [AAPL],
                    "tag": ["DefinedBenefitPlanBenefitObligation"],
                    "ddate": [pd.Timestamp("2021-08-31")],
                    "qtrs": [0],
                    "period": ["2021_08"],
                }
            )

        @staticmethod
        def save(table, frame: pd.DataFrame) -> int:
            saved.append(frame.copy())
            return len(frame)

    assert fn._repair_period_available_at(SimpleNamespace(store=Store()), "2021_08", date(2021, 9, 13), overwrite=True) == 2
    assert len(saved) == 2
    assert all(list(frame.columns) == [*fn.Tables.notes_num.pk, "available_at"] for frame in saved)
    assert all(frame["available_at"].eq(date(2021, 9, 13)).all() for frame in saved)
    print("\n=== SANITY CHECK: existing notes clock correction ===")
    print("  the metadata-only repair overwrites a wrong non-null clock without rereading ZIP payloads. Validated.")


def test_fetch_repairs_converged_historical_clock_without_reparsing_zip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    period = "2021_08"
    repaired: list[tuple[str, date, bool]] = []
    context = SimpleNamespace(
        config_dir="configs",
        store=SimpleNamespace(),
        config=SimpleNamespace(local=SimpleNamespace(paths=SimpleNamespace(financial_notes="unused"))),
    )
    monkeypatch.setattr(fn, "load_cik_mapping", lambda context: pd.DataFrame())
    monkeypatch.setattr(fn, "cik_to_ticker", lambda mapping, config_dir: {})
    monkeypatch.setattr(fn, "cache_dir", lambda context, key: tmp_path)
    monkeypatch.setattr(fn, "stored_values", lambda context, tables, column: frozenset({period}))
    monkeypatch.setattr(fn, "_periods_missing_available_at", lambda context: set())
    monkeypatch.setattr(
        fn, "_repair_period_available_at", lambda context, period, available_at, *, overwrite: repaired.append((period, available_at, overwrite)) or 1
    )
    monkeypatch.setattr(fn, "pending_periods", lambda *args, **kwargs: pytest.fail("metadata repair must not enter extraction"))

    assert fn.fetch_financial_notes(context, ["AAPL"], repair_availability=True) == 0
    assert repaired == [(period, date(2021, 9, 13), True)]
    print("\n=== SANITY CHECK: converged notes metadata repair ===")
    print("  an already-ingested August 2021 period is relabelled September 13 without a ZIP reparse. Validated.")


def test_fetch_validates_clock_before_converged_period_fast_path(tmp_path, monkeypatch):
    period = "2026_08"
    context = SimpleNamespace(
        config_dir="configs",
        store=SimpleNamespace(),
        config=SimpleNamespace(local=SimpleNamespace(paths=SimpleNamespace(financial_notes="unused"))),
    )

    monkeypatch.setattr(fn, "load_cik_mapping", lambda context: pd.DataFrame())
    monkeypatch.setattr(fn, "cik_to_ticker", lambda mapping, config_dir: {})
    monkeypatch.setattr(fn, "cache_dir", lambda context, key: tmp_path)
    monkeypatch.setattr(fn, "pending_periods", lambda *args, **kwargs: [])
    monkeypatch.setattr(fn, "_periods_missing_available_at", lambda context: set())
    monkeypatch.setattr(fn, "_notes_periods", lambda context, years_history: [period])
    monkeypatch.setattr(fn, "record_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        fn,
        "stored_period_clock",
        lambda context, tables, period: (_ for _ in ()).throw(ValueError("conflicting stored available_at values")),
    )

    with pytest.raises(ValueError, match="conflicting stored available_at values"):
        fn.fetch_financial_notes(context, ["AAPL"])

    print("\n=== SANITY CHECK: converged notes archive clock ===")
    print("  a fully ingested period still validates its immutable stored archive clock before the fast-path skip. Validated.")


def test_fetch_stamps_one_archive_clock_on_numeric_and_text_rows(tmp_path, monkeypatch):
    period = "2026_08"
    archive_date = date(2026, 9, 14)
    path = tmp_path / f"{period}_notes.zip"
    path.write_bytes(b"fixture")
    saved: list[tuple[object, pd.DataFrame]] = []

    class Store:
        @staticmethod
        def save(table, frame: pd.DataFrame) -> int:
            saved.append((table, frame.copy()))
            return len(frame)

    context = SimpleNamespace(
        config_dir="configs",
        store=Store(),
        config=SimpleNamespace(local=SimpleNamespace(paths=SimpleNamespace(financial_notes="unused"))),
    )
    num = pd.DataFrame(
        [
            {
                "cik": "0000320193",
                "ticker": "AAPL",
                "adsh": AAPL,
                "tag": "DefinedBenefitPlanBenefitObligation",
                "ddate": pd.Timestamp("2024-09-30"),
                "qtrs": 0,
                "value": 1.0,
                "filed": pd.Timestamp("2024-11-01"),
            }
        ]
    )
    txt = pd.DataFrame(
        [
            {
                "cik": "0000320193",
                "ticker": "AAPL",
                "adsh": AAPL,
                "tag": "DefinedBenefitPlanDisclosureTextBlock",
                "ddate": pd.Timestamp("2024-09-30"),
                "qtrs": 0,
                "value": "text",
                "filed": pd.Timestamp("2024-11-01"),
            }
        ]
    )

    monkeypatch.setattr(fn, "load_cik_mapping", lambda context: pd.DataFrame())
    monkeypatch.setattr(fn, "cik_to_ticker", lambda mapping, config_dir: {})
    monkeypatch.setattr(fn, "cache_dir", lambda context, key: tmp_path)
    monkeypatch.setattr(fn, "pending_periods", lambda context, cache, tables, periods, scope, reparse: list(periods))
    monkeypatch.setattr(fn, "_periods_missing_available_at", lambda context: set())
    monkeypatch.setattr(fn, "_notes_periods", lambda context, years_history: [period])
    monkeypatch.setattr(fn, "ensure_zip", lambda *args, **kwargs: path)
    monkeypatch.setattr(fn, "stored_period_clock", lambda context, tables, period: None)
    monkeypatch.setattr(fn, "_read_notes", lambda path, cik2tkr, universe, registrants: (num.copy(), txt.copy()))
    monkeypatch.setattr(fn, "mark_processed", lambda *args, **kwargs: None)
    monkeypatch.setattr(fn, "record_run", lambda *args, **kwargs: None)

    assert fn.fetch_financial_notes(context, ["AAPL"]) == 2
    assert [table for table, _ in saved] == [fn.Tables.notes_num, fn.Tables.notes_text]
    assert all(frame["available_at"].eq(archive_date).all() for _, frame in saved)

    print("\n=== SANITY CHECK: one archive clock stamps both outputs ===")
    print("  numeric and text rows from 2026_08 carry the same available_at date. Validated.")


def test_fetch_stamps_new_zip_with_successful_download_date(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    period = "2026_09"
    path = tmp_path / f"{period}_notes.zip"
    saved: list[pd.DataFrame] = []

    class Store:
        @staticmethod
        def save(table, frame: pd.DataFrame) -> int:
            saved.append(frame.copy())
            return len(frame)

    class Clock:
        @staticmethod
        def now(tz):
            assert tz == MARKET_TIMEZONE
            return datetime(2026, 10, 15, 17, 0, tzinfo=tz)

    def download(*args, **kwargs) -> Path:
        path.write_bytes(b"fixture")
        stale = datetime(2026, 10, 1, tzinfo=UTC).timestamp()
        os.utime(path, (stale, stale))
        return path

    num = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "adsh": AAPL,
                "tag": "DefinedBenefitPlanBenefitObligation",
                "ddate": pd.Timestamp("2026-08-31"),
                "qtrs": 0,
                "value": 1.0,
                "filed": pd.Timestamp("2026-09-30"),
            }
        ]
    )
    context = SimpleNamespace(
        store=Store(), config_dir="configs", config=SimpleNamespace(local=SimpleNamespace(paths=SimpleNamespace(financial_notes="unused")))
    )
    monkeypatch.setattr(bulk_cache, "datetime", Clock)
    monkeypatch.setattr(fn, "load_cik_mapping", lambda context: pd.DataFrame())
    monkeypatch.setattr(fn, "cik_to_ticker", lambda mapping, config_dir: {})
    monkeypatch.setattr(fn, "cache_dir", lambda context, key: tmp_path)
    monkeypatch.setattr(fn, "pending_periods", lambda context, cache, tables, periods, scope, reparse: list(periods))
    monkeypatch.setattr(fn, "_periods_missing_available_at", lambda context: set())
    monkeypatch.setattr(fn, "_notes_periods", lambda context, years_history: [period])
    monkeypatch.setattr(fn, "stored_period_clock", lambda context, tables, period: None)
    monkeypatch.setattr(fn, "ensure_zip", download)
    monkeypatch.setattr(fn, "_read_notes", lambda path, cik2tkr, universe, registrants: (num.copy(), pd.DataFrame()))
    monkeypatch.setattr(fn, "mark_processed", lambda *args, **kwargs: None)
    monkeypatch.setattr(fn, "record_run", lambda *args, **kwargs: None)

    assert fn.fetch_financial_notes(context, ["AAPL"]) == 1
    assert saved[0]["available_at"].iloc[0] == date(2026, 10, 15)
    print("\n=== SANITY CHECK: newly downloaded notes ZIP ===")
    print("  the completed download day in New York wins over the ZIP cache's older file timestamp. Validated.")
