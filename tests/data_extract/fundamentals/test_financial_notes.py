"""Unit tests for the SEC Financial Statement & Notes extractor
(src/data_extract/utils/zips/fetch_financial_notes.py).

Covers, with NO network:
  * the pure parse/join (num + text) incl. the dimn==0 / coreg / universe filters,
  * the full zip IO path via an in-memory synthetic notes zip,
  * fixed quarterly/monthly source-unit dates and configured publication clocks.
A separate script (scripts-style) exercises a REAL quarterly zip end-to-end.
"""

from __future__ import annotations

import io
import os
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import src.data_extract.utils.zips.fetch_financial_notes as fn
import src.data_extract.utils.zips.zip_archive as zip_archive
from src.constants.constants import MARKET_TIMEZONE, SEC_NET_PENSION_LIABILITY_TAGS
from src.data_extract.utils.common import bulk_cache
from src.data_extract.utils.common.resume import ArchiveWork
from src.data_store.schema import Tables
from src.utils.window import Window, WindowRun
from tests.canonical_schema import create_canonical
from tests.data_extract.common.scope_fixtures import SENTINEL, dated_identity
from tests.data_extract.fake_context import extract_config

AAPL_TICKERS = ["AAPL"]


def _notes_context(store: Any = None, *, start: str = "2009-01-01", as_of: str = "2026-11-16") -> Any:
    config = extract_config()
    for table in (Tables.notes_num, Tables.notes_text):
        config.data.tables[table.name].start_date = start
    return SimpleNamespace(
        store=store, config=config, config_dir="configs", run=WindowRun("incremental", "fixed notes source fixture", pd.Timestamp(as_of))
    )


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
#: AAPL holds one open consolidating window on its roster CIK.
IDENTITY = dated_identity([("AAPL", "0000320193", "cik_window", SENTINEL, None)], {"AAPL": "0000320193"})
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
    num, txt = fn._read_notes(zip_path, IDENTITY, {"AAPL"})

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
    num, txt = fn._read_notes(zip_path, IDENTITY, {"MSFT"})
    assert num.empty and txt.empty


_PRIMARY, _VARIANT = SEC_NET_PENSION_LIABILITY_TAGS
_PBO = "DefinedBenefitPlanBenefitObligation"
_PRE_COLS = ["adsh", "report", "line", "stmt", "inpth", "tag", "version", "prole", "plabel", "negating"]
HWM = "0000004281-22-000004"  # variant on the balance sheet, primary only on a footnote (OPEB-only)
IBM = "0000051143-20-001334"  # primary on the balance sheet, variant only on a footnote


def _net_liability_zip(path: Path, pre: list[tuple[str, str, str]] | None) -> Path:
    """Notes zip with undimensioned net-liability and PBO rows on two accessions; `pre` = (adsh, tag, stmt) rows, None = no pre.tsv."""
    sub = [
        "\t".join(_SUB_COLS),
        *(_row(_SUB_COLS, adsh=a, cik="320193", form="10-K", period="20211231", fy="2021", fp="FY", filed="20220201") for a in (HWM, IBM)),
    ]
    num = ["\t".join(_NUM_COLS)]
    for adsh in (HWM, IBM):
        for tag, value in ((_PRIMARY, 153), (_VARIANT, 771), (_PBO, 5000)):
            num.append(_row(_NUM_COLS, adsh=adsh, tag=tag, ddate="20211231", qtrs="0", uom="USD", dimn="0", value=value))
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("sub.tsv", "\n".join(sub))
        z.writestr("num.tsv", "\n".join(num))
        z.writestr("txt.tsv", "\t".join(_TXT_COLS))
        if pre is not None:
            rows = ["\t".join(_PRE_COLS)] + [_row(_PRE_COLS, adsh=a, tag=t, stmt=s) for a, t, s in pre]
            z.writestr("pre.tsv", "\n".join(rows))
    return path


def test_net_liability_tags_kept_only_when_on_the_balance_sheet(tmp_path):
    pre = [
        (HWM, _VARIANT, "BS"),  # HWM: variant is the balance-sheet line
        (HWM, _PRIMARY, ""),  # primary is an OPEB-only footnote line
        (HWM, _PBO, ""),
        (IBM, _PRIMARY, "BS"),  # IBM: primary is the balance-sheet line
        (IBM, _VARIANT, "CF"),
    ]
    num, _ = fn._read_notes(_net_liability_zip(tmp_path / "2022q1_notes.zip", pre), IDENTITY, {"AAPL"})
    kept = {(r.adsh, r.tag) for r in num.itertuples()}
    assert kept == {(HWM, _VARIANT), (HWM, _PBO), (IBM, _PRIMARY), (IBM, _PBO)}
    print("\n=== SANITY: net-liability tags need a balance-sheet presentation ===")
    print(f"  kept = {sorted(kept)}")
    print("  -> HWM footnote-only primary dropped, IBM balance-sheet primary kept, PBO untouched.")


def test_net_liability_tags_dropped_with_warning_when_pre_is_missing(tmp_path, caplog):
    with caplog.at_level("WARNING", logger=fn.logger.name):
        num, _ = fn._read_notes(_net_liability_zip(tmp_path / "2022q1_notes.zip", None), IDENTITY, {"AAPL"})
    assert set(num["tag"]) == {_PBO} and len(num) == 2
    assert "pre.tsv" in caplog.text and "2022q1_notes.zip" in caplog.text
    print("\n=== SANITY: pre.tsv missing ===")
    print(f"  tags kept = {sorted(set(num['tag']))}; warning logged = {'pre.tsv' in caplog.text}")


def test_on_balance_sheet_accepts_a_column_less_empty_num() -> None:
    out = fn._on_balance_sheet(pd.DataFrame(), pd.DataFrame(), "2022q1_notes.zip")
    assert out.empty
    print("\n=== SANITY: header-only num.tsv ===")
    print("  a column-less empty num frame passes through without a KeyError. Validated.")


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
    periods = fn._generate_periods(Window(pd.Timestamp("2011-01-01"), today))
    assert "2024q1" in periods and "2012q3" in periods  # quarterly era
    assert "2026_06" in periods and "2025_07" in periods  # recent monthly
    assert all(int(period[:4]) >= 2011 for period in periods)
    print("SANITY: quarterly and monthly archive candidates use a fixed 2011 source floor.")


def test_notes_periods_uses_scrape_when_available(monkeypatch):
    fake = ["2009q1", "2020q3", "2025_07", "2026_06"]
    monkeypatch.setattr(fn, "_scrape_available_periods", lambda context: fake)
    got = fn.notes_periods(_notes_context(start="2023-01-01", as_of="2026-07-19"))
    assert got == ["2025_07", "2026_06"]  # 2009/2020 precede the fixed source start
    print("SANITY: published notes units are filtered by configured source dates.")


#: Landing-page links as the SEC serves them: the republished 2010q1-q3 files carry a `_N` suffix.
_LANDING_LINKS = "".join(
    f'<a href="/files/dera/data/financial-statement-notes-data-sets/{name}" download>x</a>'
    for name in ("2009q4_notes.zip", "2010q1_notes_1.zip", "2010q2_notes_0.zip", "2010q3_notes_0.zip", "2010q4_notes.zip")
)


def test_a_missing_republished_period_is_listed_and_downloaded(monkeypatch, tmp_path):
    """A notes zip absent from the cache is downloaded, also when the SEC links it under a suffixed file name."""
    monkeypatch.setattr(fn, "sec_get", lambda context, url, timeout: SimpleNamespace(text=_LANDING_LINKS))
    requested: list[str] = []

    def fake_zip(context: Any, path: Path, url: str, **kwargs: Any) -> Path:
        if not bulk_cache.is_cached(path):  # the real `ensure_zip` contract: download only what the cache lacks
            requested.append(url)
            path.write_bytes(_synthetic_zip().getvalue())
        return path

    monkeypatch.setattr(fn, "ensure_zip", fake_zip)
    for cached in ("2009q4", "2010q3", "2010q4"):
        (tmp_path / f"{cached}_notes.zip").write_bytes(_synthetic_zip().getvalue())

    context = _notes_context(as_of="2026-11-11")
    periods = fn.notes_periods(context)
    available = fn.download_notes_periods(context, tmp_path, periods)

    assert periods == ["2009q4", "2010q1", "2010q2", "2010q3", "2010q4"], periods
    assert requested == [fn.SEC_FINNOTES_URL_TEMPLATE.format(period=p) for p in ("2010q1", "2010q2")], requested
    assert available == 5 and fn.cached_notes_periods(tmp_path) == periods
    print("\n=== SANITY CHECK: missing notes zip triggers a download ===")
    print(f"  landing page lists {periods} (2010q1-q3 under suffixed names); cache missed 2010q1, 2010q2")
    print(
        f"  downloaded {len(requested)} zip(s): {[u.rsplit('/', 1)[1] for u in requested]}; cached ones untouched; {available}/5 available. Validated."
    )


def test_notes_periods_falls_back_to_generator(monkeypatch):
    monkeypatch.setattr(fn, "_scrape_available_periods", lambda context: None)
    got = fn.notes_periods(_notes_context(start="2024-01-01", as_of="2026-07-19"))
    assert got and all(int(period[:4]) >= 2024 for period in got)
    print("SANITY: unavailable landing-page metadata uses bounded source-unit candidates, with no moving-year fallback.")


@pytest.mark.parametrize(
    ("period", "expected"),
    [("2021_08", date(2021, 9, 12)), ("2013q4", date(2014, 1, 21)), ("2025_08", date(2025, 9, 12)), ("2026_08", date(2026, 9, 12))],
)
def test_archive_release_clock_uses_monthly_day_or_quarterly_lag(period: str, expected: date, tmp_path: Path) -> None:
    assert bulk_cache.archive_available_at(period, tmp_path / f"{period}_notes.zip", downloaded=False) == expected
    print(f"SANITY: raw {period} release clock is {expected}; final market-session eligibility is resolved by the shared publication policy.")


def test_future_archive_available_at_is_cached_download_date(tmp_path: Path) -> None:
    path = tmp_path / "2026_09_notes.zip"
    path.write_bytes(b"cached")
    downloaded = datetime(2026, 10, 16, 0, 30, tzinfo=UTC)  # Still October 15 in New York.
    os.utime(path, (downloaded.timestamp(), downloaded.timestamp()))

    assert bulk_cache.archive_available_at(
        "2026_09",
        path,
        downloaded=False,
    ) == date(2026, 10, 15)
    assert (
        bulk_cache.archive_available_at(
            "2026_09",
            tmp_path / "missing.zip",
            downloaded=False,
        )
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
            assert where == {"period": "2026_08", "ticker": AAPL_TICKERS}
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
    repaired = fn._repair_period_available_at(_notes_context(store), "2026_08", date(2026, 9, 14), AAPL_TICKERS)

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
            assert where == {"period": "2021_08", "ticker": AAPL_TICKERS}
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

    assert fn._repair_period_available_at(_notes_context(Store()), "2021_08", date(2021, 9, 13), AAPL_TICKERS) == 2
    assert len(saved) == 2
    assert all(list(frame.columns) == [*fn.Tables.notes_num.pk, "available_at"] for frame in saved)
    assert all(frame["available_at"].eq(date(2021, 9, 13)).all() for frame in saved)
    print("\n=== SANITY CHECK: existing notes clock correction ===")
    print("  the metadata-only repair overwrites a wrong non-null clock without rereading ZIP payloads. Validated.")


def _work(pending: list[str]) -> ArchiveWork:
    """A notes work list parsing `pending` for AAPL."""
    return ArchiveWork(listed=pending, pending=pending, rescan=[], keys=["AAPL"], screen=["AAPL"], scoped=False)


@pytest.mark.parametrize(("period", "later"), [("2026_08", "2026-09-16"), ("2026q1", "2026-04-30")])
def test_explicit_release_repair_preserves_later_recorded_availability(sqlite_store: Any, tmp_path: Path, period: str, later: str) -> None:
    create_canonical(sqlite_store.engine, [Tables.notes_num, Tables.notes_text])
    for table in (Tables.notes_num, Tables.notes_text):
        value = 1.0 if table is Tables.notes_num else "recorded text"
        row = {
            "ticker": "AAPL",
            "adsh": AAPL,
            "tag": "T",
            "ddate": pd.Timestamp("2025-12-31"),
            "qtrs": 0,
            "period": period,
            "available_at": pd.Timestamp(later),
            "value": value,
        }
        sqlite_store.save(table, pd.DataFrame([row]))
    context = _notes_context(sqlite_store)

    fn.repair_notes_availability(context, tmp_path, AAPL_TICKERS)

    for table in (Tables.notes_num, Tables.notes_text):
        saved = sqlite_store.load(table, columns=["available_at", "value"])
        assert pd.Timestamp(saved.available_at.iloc[0]) == pd.Timestamp(later)
        assert saved.value.iloc[0] == (1.0 if table is Tables.notes_num else "recorded text")
    print(f"SANITY: explicit {period} release repair preserves the later {later} clock and both payloads.")


def test_fetch_repairs_converged_historical_clock_without_reparsing_zip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    period = "2021_08"
    repaired: list[tuple[str, date]] = []
    context = _notes_context(SimpleNamespace())
    monkeypatch.setattr(fn, "load_identity", lambda context: IDENTITY)
    monkeypatch.setattr(fn, "stored_values", lambda context, tables, column, **kwargs: frozenset({period}))
    monkeypatch.setattr(fn, "stored_period_clock", lambda context, tables, period: None)
    monkeypatch.setattr(
        fn, "_repair_period_available_at", lambda context, period, available_at, tickers: repaired.append((period, available_at)) or 1
    )
    monkeypatch.setattr(fn, "archive_worklist", lambda *args, **kwargs: pytest.fail("metadata repair must not enter extraction"))

    assert fn.repair_notes_availability(context, tmp_path, AAPL_TICKERS) == 0
    assert repaired == [(period, date(2021, 9, 12))]
    print("\n=== SANITY CHECK: converged notes metadata repair ===")
    print("  an already-ingested August 2021 period is relabelled with the configured September 12 raw estimate without a ZIP reparse. Validated.")


def test_completed_period_is_skipped_before_identity_payload_or_clock_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    period = "2026_08"
    context = _notes_context()
    monkeypatch.setattr(fn, "archive_worklist", lambda *args, **kwargs: _work([]))
    monkeypatch.setattr(fn, "load_identity", lambda *_: pytest.fail("completed unit read identity"))
    monkeypatch.setattr(fn, "stored_period_clock", lambda *_: pytest.fail("completed unit re-read clocks"))
    assert fn.ingest_financial_notes(context, tmp_path, AAPL_TICKERS, [period]) == 0
    print("SANITY: a completed notes unit needs no identity, payload or clock read on a normal run.")


def test_fetch_stamps_one_archive_clock_on_numeric_and_text_rows(tmp_path, monkeypatch):
    period = "2026_08"
    archive_date = date(2026, 9, 12)
    path = tmp_path / f"{period}_notes.zip"
    path.write_bytes(_synthetic_zip().getvalue())
    saved: list[tuple[object, pd.DataFrame]] = []

    class Store:
        @staticmethod
        def save(table, frame: pd.DataFrame) -> int:
            saved.append((table, frame.copy()))
            return len(frame)

    context = _notes_context(Store())
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

    monkeypatch.setattr(fn, "load_identity", lambda context: IDENTITY)
    monkeypatch.setattr(fn, "archive_worklist", lambda context, tables, periods, keys: _work(list(periods)))
    monkeypatch.setattr(zip_archive, "ensure_zip", lambda *args, **kwargs: path)
    monkeypatch.setattr(fn, "stored_period_clock", lambda context, tables, period: None)
    monkeypatch.setattr(fn, "_read_notes", lambda path, identity, universe: (num.copy(), txt.copy()))

    assert fn.ingest_financial_notes(context, tmp_path, AAPL_TICKERS, [period]) == 2
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
        path.write_bytes(_synthetic_zip().getvalue())
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
    context = _notes_context(Store())
    monkeypatch.setattr(bulk_cache, "datetime", Clock)
    monkeypatch.setattr(fn, "load_identity", lambda context: IDENTITY)
    monkeypatch.setattr(fn, "archive_worklist", lambda context, tables, periods, keys: _work(list(periods)))
    monkeypatch.setattr(fn, "stored_period_clock", lambda context, tables, period: None)
    monkeypatch.setattr(zip_archive, "ensure_zip", download)
    monkeypatch.setattr(fn, "_read_notes", lambda path, identity, universe: (num.copy(), pd.DataFrame()))

    assert fn.ingest_financial_notes(context, tmp_path, AAPL_TICKERS, [period]) == 1
    assert saved[0]["available_at"].iloc[0] == date(2026, 10, 15)
    print("\n=== SANITY CHECK: newly downloaded notes ZIP ===")
    print("  the completed download day in New York wins over the ZIP cache's older file timestamp. Validated.")


def test_net_pension_liability_tags_kept_only_undimensioned_and_without_coreg() -> None:
    liability_tags = list(SEC_NET_PENSION_LIABILITY_TAGS)
    cols = ["adsh", "tag", "dimn", "coreg"]
    rows = [
        ("A", liability_tags[0], 0, ""),  # keep
        ("A", liability_tags[1], 0, ""),  # keep
        ("A", liability_tags[0], 1, ""),  # drop: dimension
        ("A", liability_tags[1], 2, ""),  # drop: dimension
        ("A", liability_tags[0], 0, "SUBSID"),  # drop: co-registrant
        ("A", liability_tags[1], 0, "SUBSID"),  # drop: co-registrant
        ("B", liability_tags[0], 0, ""),  # drop: non-universe filing
    ]
    chunk = pd.DataFrame(rows, columns=cols)
    kept = fn._consolidated_rows(chunk, {"A"}, fn._NOTES_NUM_TAGS)
    assert kept.tolist() == [True, True, False, False, False, False, False]
    assert set(chunk.loc[kept, "tag"]) == set(liability_tags)

    print("\n=== SANITY: net pension liability tags ===")
    print(f"  kept rows = {int(kept.sum())} of {len(chunk)} (dimn==0, no coreg, universe filing only)")
    print("  -> both recognized net-liability tags are curated; dimension/coreg/non-universe rows are dropped.")
