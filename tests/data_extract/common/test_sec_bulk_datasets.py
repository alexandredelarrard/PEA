"""Tests for the SEC bulk quarterly extractors (insider transactions + pension
facts from the Financial Statement Data Sets).

The parse/join/filter functions are PURE and tested on both hand-built inputs and
the REAL cached 2024q1 zips (skipped if not downloaded). The incremental-state
query is tested against a throwaway SQLite DB.
"""

from __future__ import annotations

import os
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from click.testing import CliRunner
from sqlalchemy import create_engine

import src.data_extract.cli as cli_mod
from src.constants.constants import MARKET_TIMEZONE
from src.data_extract.utils.common import bulk_cache
from src.data_extract.utils.common.identity import build_identity
from src.data_extract.utils.fundamentals import fetch_financial_statements as fin
from src.data_extract.utils.institutionals import fetch_insider_transactions as ins
from src.data_extract.utils.institutionals.insider_common import BULK_DATE_FORMATS, build_insider_frame, screen_insider_rows
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from tests.data_extract.common.scope_fixtures import SENTINEL, dated_identity

# Repo root = the first ancestor holding pyproject.toml, NOT a fixed `parents[N]`.
# This file moved down one directory level once already (into the mirrored
# tests/data_extract/<area>/ layout) and the hard index silently repointed this at
# tests/configs/ -- a hard index would silently point the fixture ZIPs at a
# tests/data/ that does not exist, and the tests would skip rather than fail.
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
REPO = _ROOT
INSIDER_ZIP = REPO / "data" / "sec_insider_transactions" / "2024q1_form345.zip"
#: BDX holds one open consolidating window on its roster CIK.
BDX_IDENTITY = dated_identity([("BDX", "0000010795", "cik_window", SENTINEL, None)], {"BDX": "0000010795"})
FINSTMT_ZIP = REPO / "data" / "sec_financial_statements" / "2024q1.zip"


# --------------------------------------------------------------------------- #
# Insider transactions                                                          #
# --------------------------------------------------------------------------- #
def test_bulk_quarter_periods_are_deterministic_and_bounded():
    """`quarter_periods` is the SHARED period generator (utils/common/bulk_cache.py).
    The insider and financial-statement fetchers each had their own identical `_quarters`
    before the refactor; both now call this one, so the first-year bound is enforced in a
    single place."""
    from src.data_extract.utils.common.bulk_cache import quarter_periods

    qs = quarter_periods(3, ins.SEC_INSIDER_FIRST_YEAR, today=pd.Timestamp("2024-05-01"))
    assert qs == [
        "2021q1",
        "2021q2",
        "2021q3",
        "2021q4",
        "2022q1",
        "2022q2",
        "2022q3",
        "2022q4",
        "2023q1",
        "2023q2",
        "2023q3",
        "2023q4",
        "2024q1",
        "2024q2",
        "2024q3",
        "2024q4",
    ]
    # never emits before the data set exists, whichever data set asks
    for first_year in (ins.SEC_INSIDER_FIRST_YEAR, fin.SEC_FINSTMT_FIRST_YEAR):
        assert all(int(q[:4]) >= first_year for q in quarter_periods(50, first_year, today=pd.Timestamp("2024-05-01")))
    print("\n=== SANITY: shared quarter_periods bounded to each data-set era ===")
    print(
        f"  years_history=3 @2024 -> {len(qs)} quarters 2021q1..2024q4; "
        f"first-year floor honoured for insider ({ins.SEC_INSIDER_FIRST_YEAR}) and "
        f"finstmt ({fin.SEC_FINSTMT_FIRST_YEAR}). Validated."
    )


def test_insider_parse_and_universe_filter_synthetic():
    sub = pd.DataFrame(
        {
            "ACCESSION_NUMBER": ["a1", "a2"],
            "ISSUERCIK": ["320193", "999999"],
            "ISSUERNAME": ["APPLE INC", "OFFUNIVERSE CO"],
            "ISSUERTRADINGSYMBOL": ["AAPL", "ZZZZ"],
            "DOCUMENT_TYPE": ["4", "4"],
            "FILING_DATE": ["31-JAN-2024", "31-JAN-2024"],
            "PERIOD_OF_REPORT": ["29-JAN-2024", "29-JAN-2024"],
        }
    )
    own = pd.DataFrame(
        {
            "ACCESSION_NUMBER": ["a1", "a2"],
            "RPTOWNERCIK": ["111", "222"],
            "RPTOWNERNAME": ["COOK TIMOTHY", "DOE JOHN"],
            "RPTOWNER_RELATIONSHIP": ["Officer", "Director"],
            "RPTOWNER_TITLE": ["CEO", ""],
        }
    )
    nd = pd.DataFrame(
        {
            "ACCESSION_NUMBER": ["a1", "a2"],
            "NONDERIV_TRANS_SK": ["1", "1"],
            "SECURITY_TITLE": ["Common", "Common"],
            "TRANS_DATE": ["29-JAN-2024", "29-JAN-2024"],
            "TRANS_CODE": ["P", "S"],
            "TRANS_SHARES": ["1000", "500"],
            "TRANS_PRICEPERSHARE": ["150", "20"],
            "TRANS_ACQUIRED_DISP_CD": ["A", "D"],
            "SHRS_OWND_FOLWNG_TRANS": ["5000", "100"],
            "DIRECT_INDIRECT_OWNERSHIP": ["D", "D"],
        }
    )
    out = build_insider_frame(*ins.extract_bulk_strings(sub, own, nd, pd.DataFrame()), date_formats=BULK_DATE_FORMATS)
    assert set(out["accession_number"]) == {"a1", "a2"}
    a1 = out[out["accession_number"] == "a1"].iloc[0]
    assert a1["ticker"] == "AAPL" and a1["is_officer"] == 1.0 and a1["transaction_code"] == "P"
    assert abs(a1["value_usd"] - 150000.0) < 1e-6  # 1000 * 150

    # `ZZZZ`'s issuer is nobody's entity, so it is neither kept nor quarantined: quarantine
    # is scoped to rows that CLAIMED a universe ticker, or ~50M unrelated filers' rows would
    # be stored as evidence of nothing.
    identity = build_identity(
        lineage=pd.DataFrame([{"cik": "0000320193", "entity_id": "E0000320193", "source": "roster", "confidence": None, "evidence": "test"}]),
        tenure=pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "issuer_cik": "0000320193",
                    "valid_from": pd.Timestamp("2006-01-03"),
                    "valid_to": None,
                    "n_filings": 900,
                    "source": "form345",
                    "evidence": "",
                }
            ]
        ),
        roster=pd.DataFrame([{"ticker": "AAPL", "cik": "0000320193"}]),
    )
    filt, rejected = screen_insider_rows(out, ["AAPL"], identity)
    assert set(filt["ticker"]) == {"AAPL"}
    assert rejected.empty
    print("\n=== SANITY: insider parse + universe filter ===")
    print("  a1 AAPL officer PURCHASE 1000@150 = $150k; CIK-first kept AAPL, dropped the unrelated ZZZZ filer without quarantining it. Validated.")


@pytest.mark.skipif(not INSIDER_ZIP.exists(), reason="cached insider 2024q1 zip absent")
def test_insider_parse_real_zip():
    tables = ins._read_tables(INSIDER_ZIP)
    assert tables is not None
    df = build_insider_frame(*ins.extract_bulk_strings(*tables[:4]), date_formats=BULK_DATE_FORMATS)
    assert not df.empty and df["row_sequence"].ge(1).all()
    assert set(df["security_type"]) <= {"nonderiv", "deriv"}
    codes = df["transaction_code"].value_counts()
    aapl = df[df["ticker"] == "AAPL"]
    print("\n=== SANITY: insider REAL 2024q1 zip ===")
    print(f"  {len(df):,} transactions, {df['ticker'].nunique():,} issuers; top codes: {codes.head(4).to_dict()}")
    print(f"  AAPL rows={len(aapl)}; sample value_usd nonnull={aapl['value_usd'].notna().mean():.0%}. Validated.")
    assert len(df) > 10000 and df["ticker"].nunique() > 1000


def test_insider_work_list_comes_from_the_stored_quarters_alone(tmp_path):
    """Quarter-skip comes from the DB: a stored quarter is skipped, and a new ticker with no zip-covered
    row gets every cached quarter re-parsed for itself only; once it holds one nothing is re-parsed."""
    from src.data_extract.utils.common.resume import archive_worklist

    ds = DataStore(create_engine(f"sqlite:///{tmp_path / 't.db'}"))
    ds.save(
        "insider_transactions",
        pd.DataFrame([{"accession_number": "a1", "security_type": "nonderiv", "row_sequence": 1, "ticker": "AAPL", "quarter": "2024q1"}]),
    )
    ds.save(
        "sp500_tickers", pd.DataFrame({"ticker": ["AAPL", "MSFT", "NVDA"], "added_on": pd.to_datetime(["2000-01-01", "2000-01-01", "2024-06-25"])})
    )
    context: Any = SimpleNamespace(store=ds, config=SimpleNamespace(data_extract=SimpleNamespace(redundant_ticks=[])))
    quarters, as_of = ["2024q1", "2024q2"], pd.Timestamp("2024-07-01")

    work = archive_worklist(context, (Tables.insider_transactions,), quarters, {"2024q1"}, ["AAPL", "MSFT", "NVDA"], as_of)
    assert dict(work.units()) == {"2024q1": ["NVDA"], "2024q2": ["AAPL", "MSFT", "NVDA"]}

    ds.save(
        "insider_transactions",
        pd.DataFrame(
            [{"accession_number": "e1", "security_type": "nonderiv", "row_sequence": 1, "ticker": "NVDA", "source": "edgar", "quarter": None}]
        ),
    )
    edgar_only = archive_worklist(context, (Tables.insider_transactions,), quarters, {"2024q1"}, ["AAPL", "MSFT", "NVDA"], as_of)
    assert dict(edgar_only.units()) == {"2024q1": ["NVDA"], "2024q2": ["AAPL", "MSFT", "NVDA"]}, "an EDGAR row carries no quarter"

    ds.save(
        "insider_transactions",
        pd.DataFrame([{"accession_number": "a2", "security_type": "nonderiv", "row_sequence": 1, "ticker": "NVDA", "quarter": "2024q1"}]),
    )
    again = archive_worklist(context, (Tables.insider_transactions,), quarters, {"2024q1"}, ["AAPL", "MSFT", "NVDA"], as_of)
    assert dict(again.units()) == {"2024q2": ["AAPL", "MSFT", "NVDA"]}

    print("\n=== SANITY: insider work list from the DB ===")
    print("  2024q1 stored -> skipped for established keys; new NVDA re-reads cached 2024q1 alone, still after an EDGAR row (no quarter);")
    print("  once NVDA has a zip-covered row only 2024q2 is left. Validated.")


def test_insider_download_caches_every_quarter_from_the_first_year_and_parses_nothing(tmp_path, monkeypatch):
    """The download step the identity build depends on: every quarter since `SEC_INSIDER_FIRST_YEAR`, old and new URL
    templates, an unpublished quarter skipped, no store access and no identity load."""
    requested: list[tuple[str, tuple[str, str]]] = []
    spans: list[tuple[int, int]] = []

    def _ensure(context, path, url, **kwargs):
        requested.append((path.name, url))
        if path.stem == "2026q3":
            return None  # not published yet
        path.write_bytes(b"zip")
        return path

    def _quarters(span, first_year):
        spans.append((span, first_year))
        return ["2006q1", "2026q2", "2026q3"]

    monkeypatch.setattr(ins, "cache_dir", lambda context, key: tmp_path)
    monkeypatch.setattr(ins, "ensure_zip", _ensure)
    monkeypatch.setattr(ins, "quarter_periods", _quarters)
    monkeypatch.setattr(ins, "load_identity", lambda context: pytest.fail("the download step must not load identity"))
    context = SimpleNamespace(config=SimpleNamespace(local=SimpleNamespace(paths=SimpleNamespace(insider_transactions="unused"))))

    assert ins.download_insider_transactions(context) == ["2006q1", "2026q2"]
    assert [name for name, _ in requested] == ["2006q1.zip", "2026q2.zip", "2026q3.zip"]
    assert spans == [(pd.Timestamp.today().year - ins.SEC_INSIDER_FIRST_YEAR + 1, ins.SEC_INSIDER_FIRST_YEAR)]
    assert requested[0][1] == ins.zip_urls("2006q1") and requested[0][1][0] == ins.SEC_INSIDER_URL_TEMPLATE.format(quarter="2006q1")
    assert requested[1][1] == ins.zip_urls("2026q2") and requested[1][1][0] == ins.SEC_INSIDER_URL_NEW_TEMPLATE.format(quarter="2026q2")

    print("\n=== SANITY CHECK: insider download step ===")
    print(
        f"  {len(requested)} quarter(s) requested from {ins.SEC_INSIDER_FIRST_YEAR}, both SEC paths each, likelier first; 2 cached, the unpublished one skipped; no parse. Validated."
    )


def test_insider_download_cli_runs_only_the_download(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(cli_mod, "get_config_context", lambda path, **kwargs: (None, SimpleNamespace(name="ctx")))
    monkeypatch.setattr(cli_mod, "download_insider_transactions", lambda context: calls.append("download") or [])
    monkeypatch.setattr(cli_mod, "fetch_insider_transactions", lambda *a, **k: calls.append("parse"))
    monkeypatch.setattr(cli_mod, "fetch_insider_edgar", lambda *a, **k: calls.append("edgar"))

    result = CliRunner().invoke(cli_mod.cli, ["insider-download"])

    assert result.exit_code == 0, result.output
    assert calls == ["download"]
    print("\n=== SANITY CHECK: insider-download CLI ===")
    print("  the command caches the Form 3/4/5 zips only; parsing stays in insider-transactions. Validated.")


# --------------------------------------------------------------------------- #
# Pension facts (Financial Statement Data Sets)                                  #
# --------------------------------------------------------------------------- #
def test_pension_join_filters_segments_and_coreg():
    num = pd.DataFrame(
        {
            "adsh": ["x", "x", "x", "x"],
            "tag": ["PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent"] * 3 + ["SomethingElse"],
            "ddate": ["20231231", "20231231", "20231231", "20231231"],
            "qtrs": ["0", "0", "0", "0"],
            "uom": ["USD"] * 4,
            "segments": ["", "PlanNameAxis=USPlan", "", ""],  # 2nd row = dimensional member
            "coreg": ["", "", "SubCo", ""],  # 3rd row = co-registrant
            "value": ["1000", "600", "400", "50"],
        }
    )
    sub = pd.DataFrame({"adsh": ["x"], "cik": ["320193"], "form": ["10-K"], "fy": ["2023"], "fp": ["FY"], "filed": ["20240201"]})
    out = fin._join_pension(num, sub)
    # only the consolidated pension row survives (segment + coreg + non-pension dropped)
    assert len(out) == 1
    r = out.iloc[0]
    assert r["tag"].endswith("LiabilitiesNoncurrent") and r["value"] == 1000.0
    assert r["cik"] == "0000320193" and r["qtrs"] == 0.0
    print("\n=== SANITY: pension join (consolidated only) ===")
    print("  kept the 1 consolidated pension fact ($1000), dropped the plan-segment / co-registrant / non-pension rows. Validated.")


@pytest.mark.parametrize(
    ("quarter", "expected"),
    [
        ("2025q2", date(2025, 7, 14)),  # July 12 was Saturday.
        ("2026q1", date(2026, 4, 13)),  # April 12 was Sunday.
        ("2026q2", date(2026, 7, 13)),  # July 12 was Sunday.
        ("2025q3", date(2025, 10, 13)),
    ],
)
def test_pension_historical_zip_available_after_quarter_end_plus_twelve_days(quarter: str, expected: date, tmp_path: Path) -> None:
    assert (
        bulk_cache.archive_available_at(
            bulk_cache.period_end(quarter), tmp_path / f"{quarter}.zip", observed_from=fin._OBSERVED_FROM, downloaded=False
        )
        == expected
    )
    print("\n=== SANITY CHECK: estimated quarterly pension availability ===")
    print(f"  {quarter} is available on {expected}; weekend dates move to Monday. Validated.")


def test_pension_future_cached_zip_uses_new_york_file_date(tmp_path: Path) -> None:
    path = tmp_path / "2026q3.zip"
    path.write_bytes(b"cached")
    downloaded = datetime(2026, 10, 16, 0, 30, tzinfo=UTC)  # Still October 15 in New York.
    os.utime(path, (downloaded.timestamp(), downloaded.timestamp()))

    assert bulk_cache.archive_available_at(bulk_cache.period_end("2026q3"), path, observed_from=fin._OBSERVED_FROM, downloaded=False) == date(
        2026, 10, 15
    )
    assert (
        bulk_cache.archive_available_at(bulk_cache.period_end("2026q3"), tmp_path / "missing.zip", observed_from=fin._OBSERVED_FROM, downloaded=False)
        is None
    )
    print("\n=== SANITY CHECK: cached future pension ZIP clock ===")
    print("  the cached file uses its New York modification date only as a fallback; a missing file has no clock. Validated.")


def test_pension_fetch_preserves_two_zip_vintages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = DataStore(create_engine("sqlite:///:memory:"))
    context = SimpleNamespace(
        store=store, config_dir="configs", config=SimpleNamespace(local=SimpleNamespace(paths=SimpleNamespace(financial_statements="unused")))
    )
    tag = "PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent"

    def fact(path: Path) -> pd.DataFrame:
        q1 = path.stem == "2026q1"
        return pd.DataFrame(
            [
                {
                    "cik": "0000010795",
                    "tag": tag,
                    "ddate": pd.Timestamp("2025-09-30"),
                    "qtrs": 0,
                    "uom": "USD",
                    "value": 1_069_000_000.0 if q1 else 1_027_000_000.0,
                    "adsh": "0000010795-26-000005" if q1 else "0000010795-26-000026",
                    "filed": pd.Timestamp("2026-02-09" if q1 else "2026-05-07"),
                    "form": "10-K" if q1 else "10-Q",
                    "fy": "2025",
                    "fp": "FY" if q1 else "Q2",
                }
            ]
        )

    monkeypatch.setattr(fin, "load_identity", lambda context: BDX_IDENTITY)
    monkeypatch.setattr(fin, "cache_dir", lambda context, key: tmp_path)
    monkeypatch.setattr(fin, "quarter_periods", lambda *args: ["2026q1", "2026q2"])
    monkeypatch.setattr(fin, "ensure_zip", lambda context, path, url, **kwargs: path)
    monkeypatch.setattr(fin, "_read_pension_facts", fact)

    assert fin.fetch_financial_statements(context, ["BDX"], years_history=15) == 2
    assert fin.fetch_financial_statements(context, ["BDX"], years_history=15, reparse=True) == 2
    rows = store.load(fin.Tables.pension_facts, columns=["cik", "tag", "ddate", "qtrs", "quarter", "value", "available_at"])
    assert rows is not None and len(rows) == 2
    rows = rows.sort_values("quarter")
    assert rows["quarter"].tolist() == ["2026q1", "2026q2"]
    assert rows["value"].tolist() == [1_069_000_000.0, 1_027_000_000.0]
    assert pd.to_datetime(rows["available_at"]).dt.date.tolist() == [date(2026, 4, 13), date(2026, 7, 13)]
    print("\n=== SANITY CHECK: two pension ZIP vintages ===")
    print("  BDX's Q1 and Q2 values coexist at separate availability dates; reparse remains idempotent. Validated.")


def test_pension_new_zip_uses_successful_download_day_and_preserves_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = DataStore(create_engine("sqlite:///:memory:"))
    context = SimpleNamespace(
        store=store, config_dir="configs", config=SimpleNamespace(local=SimpleNamespace(paths=SimpleNamespace(financial_statements="unused")))
    )
    path = tmp_path / "2026q3.zip"

    class Clock:
        @staticmethod
        def now(tz):
            assert tz == MARKET_TIMEZONE
            return datetime(2026, 10, 15, 17, 0, tzinfo=tz)

        @staticmethod
        def fromtimestamp(value, tz):
            return datetime.fromtimestamp(value, tz=tz)

    def download(context, path: Path, url: str, **kwargs) -> Path:
        if not path.exists():
            path.write_bytes(b"fixture")
            stale = datetime(2026, 10, 1, tzinfo=UTC).timestamp()
            os.utime(path, (stale, stale))
        return path

    monkeypatch.setattr(bulk_cache, "datetime", Clock)
    monkeypatch.setattr(fin, "load_identity", lambda context: BDX_IDENTITY)
    monkeypatch.setattr(fin, "cache_dir", lambda context, key: tmp_path)
    monkeypatch.setattr(fin, "quarter_periods", lambda *args: ["2026q3"])
    monkeypatch.setattr(fin, "ensure_zip", download)
    monkeypatch.setattr(
        fin,
        "_read_pension_facts",
        lambda path: pd.DataFrame(
            [
                {
                    "cik": "0000010795",
                    "tag": "PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent",
                    "ddate": pd.Timestamp("2026-06-30"),
                    "qtrs": 0,
                    "uom": "USD",
                    "value": 1.0,
                    "adsh": "new",
                    "filed": pd.Timestamp("2026-08-01"),
                    "form": "10-Q",
                    "fy": "2026",
                    "fp": "Q3",
                }
            ]
        ),
    )

    assert fin.fetch_financial_statements(context, ["BDX"], years_history=15) == 1
    future_mtime = datetime(2026, 10, 25, tzinfo=UTC).timestamp()
    os.utime(path, (future_mtime, future_mtime))
    assert fin.fetch_financial_statements(context, ["BDX"], years_history=15, reparse=True) == 1
    rows = store.load(fin.Tables.pension_facts, columns=["quarter", "available_at"])
    assert rows is not None and pd.to_datetime(rows["available_at"]).dt.date.tolist() == [date(2026, 10, 15)]
    print("\n=== SANITY CHECK: first successful future ZIP download ===")
    print("  new Q3 uses the New York completion day, and cached reparse cannot change its stored clock. Validated.")


def _pension_zip(path: Path, rows: list[tuple[str, str, str, str]]) -> Path:
    """A Financial Statement data-set ZIP with one `sub.txt` and one `num.txt` row per (adsh, cik, value, filed)."""
    import zipfile

    tag = "DefinedBenefitPlanBenefitObligation"
    sub = "adsh\tcik\tname\tform\tperiod\tfy\tfp\tfiled\n" + "".join(f"{a}\t{c}\tCo\t10-K\t20251231\t2025\tFY\t{f}\n" for a, c, _, f in rows)
    num = "adsh\ttag\tversion\tcoreg\tddate\tqtrs\tuom\tsegments\tvalue\n" + "".join(
        f"{a}\t{tag}\tus-gaap/2025\t\t20251231\t0\tUSD\t\t{v}\n" for a, _, v, _ in rows
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sub.txt", sub)
        archive.writestr("num.txt", num)
    return path


def test_pension_new_key_re_reads_a_cached_stored_quarter_for_itself_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = DataStore(create_engine(f"sqlite:///{tmp_path / 'p.db'}"))
    context = SimpleNamespace(
        store=store,
        config_dir="configs",
        config=SimpleNamespace(
            local=SimpleNamespace(paths=SimpleNamespace(financial_statements="unused")), data_extract=SimpleNamespace(redundant_ticks=[])
        ),
    )
    _pension_zip(tmp_path / "2026q1.zip", [("adsh-a", "1", "999", "20260210"), ("adsh-n", "2", "500", "20260211")])
    store.save("sp500_tickers", pd.DataFrame({"ticker": ["AAA", "NEW"], "added_on": pd.to_datetime(["2000-01-01", "2026-09-28"])}))
    stored = pd.DataFrame(
        [
            {
                "cik": "0000000001",
                "ticker": "AAA",
                "tag": "DefinedBenefitPlanBenefitObligation",
                "ddate": pd.Timestamp("2025-12-31"),
                "qtrs": 0,
                "value": 111.0,
                "quarter": "2026q1",
            }
        ]
    )
    store.save(Tables.pension_facts, stored)
    identity = dated_identity(
        [("AAA", "0000000001", "cik_window", SENTINEL, None), ("NEW", "0000000002", "cik_window", SENTINEL, None)],
        {"AAA": "0000000001", "NEW": "0000000002"},
    )
    monkeypatch.setattr(fin, "load_identity", lambda context: identity)
    monkeypatch.setattr(fin, "cache_dir", lambda context, key: tmp_path)
    monkeypatch.setattr(fin, "quarter_periods", lambda *args: ["2026q1"])

    saved = fin.fetch_financial_statements(context, ["AAA", "NEW"], years_history=15, as_of=pd.Timestamp("2026-09-30"))
    rows = store.load(Tables.pension_facts, columns=["ticker", "value"]).set_index("ticker")["value"]

    assert saved == 1 and rows.to_dict() == {"AAA": 111.0, "NEW": 500.0}
    assert fin.fetch_financial_statements(context, ["AAA", "NEW"], years_history=15, as_of=pd.Timestamp("2026-10-01")) == 0
    print("\n=== SANITY CHECK: pension new-key re-read from a fixture ZIP ===")
    print("  NEW (added 2026-09-28) gets its 500 from the cached 2026q1 ZIP; AAA keeps its stored 111 (not the ZIP's 999);")
    print("  the next night NEW holds a row, so nothing is re-read. Validated.")


@pytest.mark.skipif(not FINSTMT_ZIP.exists(), reason="cached financial-statement 2024q1 zip absent")
def test_pension_parse_real_zip():
    facts = fin._read_pension_facts(FINSTMT_ZIP)
    assert facts is not None and not facts.empty
    net_liab = facts[facts["tag"] == "PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent"]
    assert not net_liab.empty
    assert (facts["value"] > 0).mean() > 0.5  # liabilities are positive
    print("\n=== SANITY: pension REAL 2024q1 zip ===")
    print(
        f"  {len(facts):,} pension facts, {facts['cik'].nunique():,} companies; "
        f"net-liability rows={len(net_liab):,}, median ${net_liab['value'].median():,.0f}"
    )
    print(f"  tags: {facts['tag'].value_counts().head(5).to_dict()}. Validated.")
    assert net_liab["cik"].nunique() > 100  # ~244 filers report a net DB deficit in 2024q1
