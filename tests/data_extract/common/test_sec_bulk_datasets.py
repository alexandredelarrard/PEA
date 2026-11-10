"""Tests for the SEC bulk quarterly extractors (insider transactions) and the shared quarterly archive clock.

The parse/join/filter functions are PURE and tested on both hand-built inputs and
the REAL cached 2024q1 zip (skipped if not downloaded). The incremental-state
query is tested against a throwaway SQLite DB.
"""

from __future__ import annotations

import os
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from click.testing import CliRunner
from sqlalchemy import create_engine

import src.data_extract.cli as cli_mod
from src.data_extract.transformers import step_extract_edgar as step_edgar
from src.data_extract.transformers import step_extract_zips as step_zips
from src.data_extract.utils.common import bulk_cache, resume
from src.data_extract.utils.common.identity import build_identity
from src.data_extract.utils.edgar_extract.insider_common import BULK_DATE_FORMATS, build_insider_frame, screen_insider_rows
from src.data_extract.utils.zips import fetch_insider_transactions as ins
from src.data_store.schema import Resume, Tables
from src.data_store.store import DataStore
from tests.canonical_schema import create_canonical
from tests.data_extract.test_cli_options import stub_zips_step
from tests.support.universe import as_roster, seed_universe

# Repo root = the first ancestor holding pyproject.toml, NOT a fixed `parents[N]`.
# This file moved down one directory level once already (into the mirrored
# tests/data_extract/<area>/ layout) and the hard index silently repointed this at
# tests/configs/ -- a hard index would silently point the fixture ZIPs at a
# tests/data/ that does not exist, and the tests would skip rather than fail.
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
REPO = _ROOT
INSIDER_ZIP = REPO / "data" / "sec_insider_transactions" / "2024q1_form345.zip"


# --------------------------------------------------------------------------- #
# Insider transactions                                                          #
# --------------------------------------------------------------------------- #
def test_bulk_quarter_periods_are_deterministic_and_bounded():
    """`quarter_periods` is the SHARED period generator (utils/common/bulk_cache.py).
    The insider and Notes fetchers share it, so the first-year bound is enforced in a single place."""
    from src.data_extract.utils.common.bulk_cache import quarter_periods

    insider_first_year = pd.Timestamp(cast(str, cast(Resume, Tables.insider_transactions.resume).source_start)).year
    notes_first_year = pd.Timestamp(cast(str, cast(Resume, Tables.notes_num.resume).source_start)).year
    qs = quarter_periods(3, insider_first_year, today=pd.Timestamp("2024-05-01"))
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
    for first_year in (insider_first_year, notes_first_year):
        assert all(int(q[:4]) >= first_year for q in quarter_periods(50, first_year, today=pd.Timestamp("2024-05-01")))
    print("\n=== SANITY: shared quarter_periods bounded to each data-set era ===")
    print(
        f"  years_history=3 @2024 -> {len(qs)} quarters 2021q1..2024q4; "
        f"first-year floor honoured for insider ({insider_first_year}) and "
        f"notes ({notes_first_year}). Validated."
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
        lineage=pd.DataFrame([{"cik": "0000320193", "company_id": "E0000320193", "source": "roster", "confidence": None, "evidence": "test"}]),
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
        roster_cik=as_roster(pd.DataFrame([{"ticker": "AAPL", "cik": "0000320193"}])),
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
    """Quarter-skip comes from the DB: a stored quarter is skipped; NVDA, added after the newest stored quarter ended
    and holding no zip-covered row of its window CIK, gets every cached stored quarter re-parsed for that CIK alone, but
    only on a run whose new quarter is cached; once it holds one nothing is re-parsed."""
    from src.data_extract.utils.common.resume import archive_worklist

    ds = DataStore(create_engine(f"sqlite:///{tmp_path / 't.db'}"))
    create_canonical(ds.engine, [Tables.insider_transactions])
    ds.save(
        "insider_transactions",
        pd.DataFrame([{"accession_number": "a1", "security_type": "nonderiv", "row_sequence": 1, "ticker": "AAPL", "quarter": "2024q1"}]),
    )
    universe = seed_universe(ds, ["AAPL", "MSFT", "NVDA"], rows={"NVDA": {"universe_added_on": pd.Timestamp("2024-06-25")}})
    lineage = {
        "company_id": "E3",
        "canonical_ticker": "NVDA",
        "cik": "0000000003",
        "role": "cik_window",
        "symbol": "",
        "valid_from": pd.Timestamp("1900-01-01"),
    }
    lineage_rows = pd.DataFrame([lineage | {"scope_changed_at": pd.Timestamp("2020-01-01")}])
    context: Any = SimpleNamespace(store=ds, universe=universe)
    quarters, as_of, keys = ["2024q1", "2024q2"], pd.Timestamp("2024-07-01"), ["AAPL", "MSFT", "NVDA"]

    uncached = archive_worklist(context, (Tables.insider_transactions,), quarters, {"2024q1"}, keys, as_of, lineage=lineage_rows)
    work = archive_worklist(context, (Tables.insider_transactions,), quarters, {"2024q1", "2024q2"}, keys, as_of, lineage=lineage_rows)
    assert dict(uncached.units()) == {"2024q2": keys}, "the new quarter is not cached: no run stores it, no rescan"
    assert dict(work.units()) == {"2024q1": ["NVDA"], "2024q2": keys} and work.rescan_ciks == {"NVDA": frozenset({"0000000003"})}

    edgar = {"accession_number": "e1", "security_type": "nonderiv", "row_sequence": 1, "ticker": "NVDA", "issuer_cik": "3", "source": "edgar"}
    ds.save("insider_transactions", pd.DataFrame([edgar | {"quarter": None}]))
    edgar_only = archive_worklist(context, (Tables.insider_transactions,), quarters, {"2024q1", "2024q2"}, keys, as_of, lineage=lineage_rows)
    assert dict(edgar_only.units()) == {"2024q1": ["NVDA"], "2024q2": keys}, "an EDGAR row carries no quarter"

    ds.save("insider_transactions", pd.DataFrame([edgar | {"accession_number": "a2", "source": "zip", "quarter": "2024q1"}]))
    again = archive_worklist(context, (Tables.insider_transactions,), quarters, {"2024q1", "2024q2"}, keys, as_of, lineage=lineage_rows)
    assert dict(again.units()) == {"2024q2": keys}

    print("\n=== SANITY: insider work list from the DB ===")
    print("  2024q1 stored -> skipped for established keys; new NVDA re-reads cached 2024q1 for CIK 3 alone, only once 2024q2 is cached,")
    print("  still after an EDGAR row (no quarter); once NVDA has a zip-covered row only 2024q2 is left. Validated.")


def test_insider_download_caches_every_quarter_from_the_first_year_and_parses_nothing(tmp_path, monkeypatch):
    """The download step the identity build depends on: every quarter since the table's source start, old and new URL
    templates, an unpublished quarter skipped, no store access and no identity load."""
    requested: list[tuple[str, tuple[str, str]]] = []
    spans: list[tuple[int, int]] = []
    insider_first_year = pd.Timestamp(cast(str, cast(Resume, Tables.insider_transactions.resume).source_start)).year

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
    assert spans == [(pd.Timestamp.today().year - insider_first_year + 1, insider_first_year)]
    assert requested[0][1] == ins.zip_urls("2006q1") and requested[0][1][0] == ins.SEC_INSIDER_URL_TEMPLATE.format(quarter="2006q1")
    assert requested[1][1] == ins.zip_urls("2026q2") and requested[1][1][0] == ins.SEC_INSIDER_URL_NEW_TEMPLATE.format(quarter="2026q2")

    print("\n=== SANITY CHECK: insider download step ===")
    print(
        f"  {len(requested)} quarter(s) requested from {insider_first_year}, both SEC paths each, likelier first; 2 cached, the unpublished one skipped; no parse. Validated."
    )


def test_insider_download_cli_runs_only_the_download(monkeypatch):
    """`zips-download --only insider` caches the Form 3/4/5 zips and nothing else."""
    calls: list[str] = []
    config = SimpleNamespace(data_extract=SimpleNamespace(years_history=15))
    context = SimpleNamespace(name="ctx", config=config, config_dir="configs")
    monkeypatch.setattr(cli_mod, "get_config_context", lambda path, **kwargs: (config, context))
    stub_zips_step(monkeypatch, config)
    monkeypatch.setattr(step_zips, "download_insider_transactions", lambda context: calls.append("download") or [])
    monkeypatch.setattr(step_zips, "fetch_insider_transactions", lambda *a, **k: calls.append("parse"))
    monkeypatch.setattr(step_edgar, "run_edgar_fetch", lambda *a, **k: calls.append("edgar"))

    result = CliRunner().invoke(cli_mod.cli, ["zips-download", "--only", "insider"])

    assert result.exit_code == 0, result.output
    assert calls == ["download"]
    print("\n=== SANITY CHECK: zips-download --only insider CLI ===")
    print("  the command caches the Form 3/4/5 zips only; parsing stays in zips-ingest --only insider. Validated.")


# --------------------------------------------------------------------------- #
# Quarterly bulk archive clock (`bulk_cache.archive_available_at`)               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("quarter", "expected"),
    [
        ("2025q2", date(2025, 7, 14)),  # July 12 was Saturday.
        ("2026q1", date(2026, 4, 13)),  # April 12 was Sunday.
        ("2026q2", date(2026, 7, 13)),  # July 12 was Sunday.
        ("2025q3", date(2025, 10, 13)),
    ],
)
def test_quarterly_zip_available_after_quarter_end_plus_twelve_days(quarter: str, expected: date, tmp_path: Path) -> None:
    assert (
        bulk_cache.archive_available_at(
            resume.period_end(quarter),
            tmp_path / f"{quarter}.zip",
            observed_from=date.fromisoformat(cast(str, cast(Resume, Tables.notes_num.resume).observed_from)),
            downloaded=False,
        )
        == expected
    )
    print("\n=== SANITY CHECK: estimated quarterly zip availability ===")
    print(f"  {quarter} is available on {expected}; weekend dates move to Monday. Validated.")


def test_future_cached_zip_uses_new_york_file_date(tmp_path: Path) -> None:
    path = tmp_path / "2026q3.zip"
    path.write_bytes(b"cached")
    downloaded = datetime(2026, 10, 16, 0, 30, tzinfo=UTC)  # Still October 15 in New York.
    os.utime(path, (downloaded.timestamp(), downloaded.timestamp()))

    assert bulk_cache.archive_available_at(
        resume.period_end("2026q3"),
        path,
        observed_from=date.fromisoformat(cast(str, cast(Resume, Tables.notes_num.resume).observed_from)),
        downloaded=False,
    ) == date(2026, 10, 15)
    assert (
        bulk_cache.archive_available_at(
            resume.period_end("2026q3"),
            tmp_path / "missing.zip",
            observed_from=date.fromisoformat(cast(str, cast(Resume, Tables.notes_num.resume).observed_from)),
            downloaded=False,
        )
        is None
    )
    print("\n=== SANITY CHECK: cached future ZIP clock ===")
    print("  the cached file uses its New York modification date only as a fallback; a missing file has no clock. Validated.")
