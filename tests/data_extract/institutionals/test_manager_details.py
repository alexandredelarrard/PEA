"""
test_manager_details.py (tests/data_extract/institutionals/test_manager_details.py)
----------------------------------------------------------------------------------
The `sec_13f_manager_details` extract (`fetch_13f_managers`, `manager_rows`): rows read from real cover excerpts of the
cached SEC 13F data sets (2023q2-2023q4 and the folder-nested `01jun2025-31aug2025` window, rewritten as small ZIPs in
`tmp_path`), real EDGAR master-index rows joined to marker rows for the pre-2013 periods, and a check of the two real
cached ZIPs when they exist. Known-truth edits (a nameless filing, a filing dated before its period) cover the
cleaning rules only. The store is an in-memory SQLite database; no SEC request and no PostgreSQL write is made.
"""

from __future__ import annotations

import logging
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common import sec_io
from src.data_extract.utils.common.bulk_cache import ZipRead, read_zip_tables
from src.data_extract.utils.institutionals import fetch_13f_managers
from src.data_extract.utils.institutionals.fetch_13f_managers import fetch_manager_details
from src.data_extract.utils.institutionals.fetch_13f_zips import zip_submissions
from src.data_extract.utils.institutionals.manager_rows import (
    EXTRACT_COLUMNS,
    INDEX_SOURCE,
    candidate_rows,
    rows_to_save,
    zip_manager_filings,
)
from src.data_store.schema import Tables
from tests.canonical_schema import create_canonical
from tests.data_extract.edgar_fixtures import fake_context, seed_index

HOLDINGS, COMBINATION, NOTICE = "13F HOLDINGS REPORT", "13F COMBINATION REPORT", "13F NOTICE"

# Real cover excerpts: (accession, filed, form, cik, period, name as filed, state, file number, report type).
_ABACUS = ("0001978011-23-000003", "31-MAY-2023", "13F-HR", "0001978011", "31-MAR-2021", "Abacus Wealth Partners, LLC", "CA", "801-62557", HOLDINGS)
_PIONEER_Q3 = ("0001006435-23-000009", "12-JUL-2023", "13F-HR", "0001006435", "30-JUN-2023", "PIONEER TRUST BANK N A/OR", "OR", None, HOLDINGS)
_NICHOLAS = ("0000903949-23-000004", "08-AUG-2023", "13F-HR", "0000903949", "30-JUN-2023", "NICHOLAS COMPANY, INC.", "WI", "801-4658", HOLDINGS)
_PIONEER = ("0001006435-23-000011", "30-OCT-2023", "13F-HR", "0001006435", "30-SEP-2023", "PIONEER TRUST BANK N A/OR", "OR", None, HOLDINGS)
_PIONEER_AMENDED = ("0001006435-23-000013", "31-OCT-2023", "13F-HR/A", "0001006435", "30-SEP-2023", "PIONEER TRUST BANK N A/OR", "OR", None, HOLDINGS)
_NICHOLAS_AMENDED = (
    "0000903949-23-000005",
    "31-OCT-2023",
    "13F-HR/A",
    "0000903949",
    "30-JUN-2023",
    "NICHOLAS COMPANY, INC.",
    "WI",
    "801-4658",
    HOLDINGS,
)
_NOTICE = ("0001178913-23-003468", "31-OCT-2023", "13F-NT", "0001831222", "30-SEP-2023", "Analyst I.M.S. Mutual Funds Management", "L3", None, NOTICE)
_CAPITAL = (
    "0001085146-23-003989",
    "31-OCT-2023",
    "13F-HR",
    "0001730456",
    "30-SEP-2023",
    "CAPITAL WEALTH MANAGEMENT, LLC",
    "RI",
    "801-76539",
    COMBINATION,
)
_FOGEL = (
    "0001415889-25-020537",
    "30-JUL-2025",
    "13F-HR",
    "0002011427",
    "30-JUN-2025",
    " FOGEL CAPITAL MANAGEMENT, INC.",
    "FL",
    "801-491135",
    HOLDINGS,
)
_AVALON = (
    "0001415889-25-020289",
    "25-JUL-2025",
    "13F-HR",
    "0002026666",
    "30-JUN-2025",
    " AVALON VENTURES MANAGEMENT, LLC ",
    "CA",
    "802-75016",
    HOLDINGS,
)

_Q2, _Q3, _Q4 = "2023q2_form13f.zip", "2023q3_form13f.zip", "2023q4_form13f.zip"
_NESTED = "01jun2025-31aug2025_form13f.zip"

# Real EDGAR master-index rows (cik, company, form, filed, accession) of 2008Q4 with their 13F markers' period.
_WACHOVIA = [
    (36995, "WACHOVIA CORP NEW", "13F-HR", "2008-11-13", "0001074683-08-000636"),
    (36995, "WACHOVIA CORP NEW", "13F-HR/A", "2008-12-04", "0001074683-08-000638"),
]
_HUNTINGTON = [
    (49205, "HUNTINGTON NATIONAL BANK", "13F-HR", "2008-10-21", "0000049205-08-000016"),
    (49205, "HUNTINGTON NATIONAL BANK", "13F-HR", "2008-10-22", "0000049205-08-000018"),
    (49205, "HUNTINGTON NATIONAL BANK", "13F-HR/A", "2008-10-31", "0000049205-08-000020"),
]
_RBC = [(50916, "RBC Capital Markets CORP", "13F-HR/A", "2008-11-14", "0001047469-08-012330")]


def _write_data_set(cache: Path, filename: str, filings: Iterable[tuple], *, folder: str | None = None) -> None:
    """A data-set ZIP holding the SUBMISSION and COVERPAGE members (upper-case headers, the SEC's names) of `filings`."""
    rows = list(filings)
    submission = pd.DataFrame(
        [{"ACCESSION_NUMBER": a, "FILING_DATE": f, "SUBMISSIONTYPE": form, "CIK": c, "PERIODOFREPORT": p} for a, f, form, c, p, *_ in rows]
    )
    cover = pd.DataFrame(
        [
            {
                "ACCESSION_NUMBER": a,
                "REPORTCALENDARORQUARTER": p,
                "FILINGMANAGER_NAME": name,
                "FILINGMANAGER_STATEORCOUNTRY": state,
                "REPORTTYPE": kind,
                "SECFILENUMBER": number,
            }
            for a, _, _, _, p, name, state, number, kind in rows
        ]
    )
    prefix = f"{folder}/" if folder else ""
    with zipfile.ZipFile(cache / filename, "w") as archive:
        archive.writestr(f"{prefix}SUBMISSION.tsv", submission.to_csv(sep="\t", index=False))
        archive.writestr(f"{prefix}COVERPAGE.tsv", cover.to_csv(sep="\t", index=False))


def _marker(cik: str, accession: str, filed: str, form: str, period: str) -> dict[str, Any]:
    return {
        "cik": cik,
        "period": pd.Timestamp(period),
        "accession_number": accession,
        "ticker": "_empty",
        "cusip": "_empty",
        "filing_date": pd.Timestamp(filed),
        "form": form,
        "parse_status": "complete",
    }


@pytest.fixture
def store(sqlite_store: Any) -> Any:
    create_canonical(sqlite_store.engine, [Tables.sec13f_hr, Tables.sec_13f_manager_details])
    return sqlite_store


@pytest.fixture
def context(tmp_path: Path, store: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A context over the SQLite store whose cache folders are empty; any SEC request fails the test."""

    def _no_request(*args: object, **kwargs: object) -> None:
        raise AssertionError("the manager extract made an SEC request")

    monkeypatch.setattr(sec_io, "download", _no_request)
    monkeypatch.setattr(sec_io, "sec_get", _no_request)
    ctx = fake_context(tmp_path, store, [])
    (tmp_path / str(ctx.config.local.paths.sec_13f_datasets)).mkdir(parents=True, exist_ok=True)
    return ctx


def _cache(context: Any) -> Path:
    return context.paths["DATA_STORE"] / str(context.config.local.paths.sec_13f_datasets)


def _rows(store: Any, **where: object) -> pd.DataFrame:
    df = store.load(Tables.sec_13f_manager_details, where=where or None, optional=True)
    assert df is not None
    df["period"] = pd.to_datetime(df["period"])
    df["first_filed_date"] = pd.to_datetime(df["first_filed_date"])
    return df.sort_values(["cik", "period"], ignore_index=True)


def _spy_zip_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The names of the ZIPs the extract opens."""
    opened: list[str] = []

    def _read(path: Path, specs: Any, *, on_corrupt: Any) -> Any:
        opened.append(path.name)
        return read_zip_tables(path, specs, on_corrupt=on_corrupt)

    monkeypatch.setattr(fetch_13f_managers, "read_zip_tables", _read)
    return opened


def test_zip_cover_rows_are_stripped_typed_and_exclude_notices(context: Any, store: Any) -> None:
    """Real 2023q4 covers: a combination cover, a notice (13F-NT) and a holdings filing, read from a ZIP."""
    _write_data_set(_cache(context), _Q4, [_PIONEER, _CAPITAL, _NOTICE])

    changed = fetch_manager_details(context, full=False, as_of=pd.Timestamp("2023-12-31"))

    df = _rows(store)
    assert changed == 2 and len(df) == 2
    assert _NOTICE[0] not in set(df["accession_number"]) and "0001831222" not in set(df["cik"]), "a 13F-NT accession is never stored"
    by_cik = df.set_index("cik")
    capital = by_cik.loc["0001730456"]
    assert (capital["report_type"], capital["sec_file_number"], capital["state_or_country"]) == ("combination", "801-76539", "RI")
    pioneer = by_cik.loc["0001006435"]
    assert str(pioneer["report_type"]) == "holdings", "a holdings cover reads holdings"
    assert bool(pd.isna(pioneer["sec_file_number"])), "a blank cover field is NULL, not an empty string"
    assert set(df["source"]) == {_Q4}
    print("\n=== SANITY CHECK: ZIP cover rows ===")
    print(df[["cik", "period", "manager_name", "report_type", "source"]].to_string(index=False))
    print("The notice is absent, the combination cover reads `combination`, a blank file number is NULL. Validated.")


def test_nested_zip_names_are_stripped(context: Any, store: Any) -> None:
    """The rolling window whose members sit in a folder reads, and names lose the whitespace the filers typed."""
    _write_data_set(_cache(context), _NESTED, [_FOGEL, _AVALON], folder="01JUN2025-31AUG2025_form13f")

    fetch_manager_details(context, full=False, as_of=pd.Timestamp("2025-09-30"))

    df = _rows(store).set_index("cik")
    assert df.loc["0002011427", "manager_name"] == "FOGEL CAPITAL MANAGEMENT, INC."
    assert df.loc["0002026666", "manager_name"] == "AVALON VENTURES MANAGEMENT, LLC"
    assert df["manager_name"].notna().all() and (df["manager_name"] == df["manager_name"].str.strip()).all()
    assert set(df["source"]) == {_NESTED}
    print("\n=== SANITY CHECK: folder-nested ZIP ===")
    print(df["manager_name"].tolist())
    print("The nested window is read and every name is stripped and non-null. Validated.")


def test_original_accession_and_first_filing_date_over_an_amendment(context: Any, store: Any) -> None:
    """The row carries the original 13F-HR; `first_filed_date` is the earliest of the original and its amendment."""
    _write_data_set(_cache(context), _Q4, [_PIONEER_AMENDED, _PIONEER])

    fetch_manager_details(context, full=False, as_of=pd.Timestamp("2023-12-31"))

    row = _rows(store).iloc[0]
    assert row["accession_number"] == _PIONEER[0] and row["first_filed_date"] == pd.Timestamp("2023-10-30")
    assert row["period"] <= row["first_filed_date"]
    print("\n=== SANITY CHECK: original vs amendment ===")
    print(f"accession {row['accession_number']} first filed {row['first_filed_date'].date()}")
    print("The original is kept (amendments are ignored) and availability is the first filing. Validated.")


def test_pre_2013_rows_join_index_names_to_markers(context: Any, store: Any) -> None:
    """Real 2008Q4 index rows joined to markers: the name is the index company, the accession the earliest original."""
    seed_index(context, _WACHOVIA + _HUNTINGTON + _RBC)
    markers = [_marker(f"{cik:010d}", accession, filed, form, "2008-09-30") for cik, _, form, filed, accession in _WACHOVIA + _HUNTINGTON + _RBC]
    store.save(Tables.sec13f_hr, pd.DataFrame(markers))

    fetch_manager_details(context, full=False, as_of=pd.Timestamp("2008-12-31"))

    df = _rows(store).set_index("cik")
    assert len(df) == 3
    assert df.loc["0000036995", "accession_number"] == "0001074683-08-000636" and df.loc["0000036995", "first_filed_date"] == pd.Timestamp(
        "2008-11-13"
    )
    assert df.loc["0000049205", "accession_number"] == "0000049205-08-000016", "the earliest of two originals"
    assert df.loc["0000049205", "first_filed_date"] == pd.Timestamp("2008-10-21")
    assert df.loc["0000050916", "accession_number"] == "0001047469-08-012330", "no original stored: the earliest accession"
    assert df["manager_name"].to_dict() == {
        "0000036995": "WACHOVIA CORP NEW",
        "0000049205": "HUNTINGTON NATIONAL BANK",
        "0000050916": "RBC Capital Markets CORP",
    }
    assert set(df["source"]) == {INDEX_SOURCE}
    assert df[["state_or_country", "sec_file_number", "report_type"]].isna().all().all(), "the index carries no cover fields"
    print("\n=== SANITY CHECK: pre-2013 index + marker rows ===")
    print(df[["manager_name", "accession_number", "first_filed_date"]].to_string())
    print("Names come from the index, original accessions win, cover fields stay NULL. Validated.")


def test_zip_cover_takes_precedence_over_the_index_for_the_same_accession(context: Any, store: Any) -> None:
    """A filing present as marker + index row and as a ZIP cover reads the ZIP (name, state, source)."""
    seed_index(context, [(1006435, "PIONEER TRUST BANK NA INDEX SPELLING", "13F-HR", "2023-10-30", _PIONEER[0])])
    store.save(Tables.sec13f_hr, pd.DataFrame([_marker("0001006435", _PIONEER[0], "2023-10-30", "13F-HR", "2023-09-30")]))
    _write_data_set(_cache(context), _Q4, [_PIONEER])

    fetch_manager_details(context, full=False, as_of=pd.Timestamp("2023-12-31"))

    row = _rows(store).iloc[0]
    assert row["source"] == _Q4 and row["manager_name"] == "PIONEER TRUST BANK N A/OR" and row["state_or_country"] == "OR"
    print("\n=== SANITY CHECK: ZIP precedence ===")
    print(row[["manager_name", "state_or_country", "source"]].to_dict())
    print("The ZIP cover wins over the index spelling. Validated.")


def test_steady_state_opens_one_zip_plus_overlap_and_changes_nothing(
    context: Any, store: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A second run reads the newest ZIP and the previous window only, makes no SEC request and writes 0 rows."""
    _write_data_set(_cache(context), _Q2, [_ABACUS])
    _write_data_set(_cache(context), _Q3, [_PIONEER_Q3, _NICHOLAS])
    _write_data_set(_cache(context), _Q4, [_PIONEER, _PIONEER_AMENDED, _NICHOLAS_AMENDED, _CAPITAL, _NOTICE])
    as_of = pd.Timestamp("2023-12-31")
    first = fetch_manager_details(context, full=False, as_of=as_of)
    before = _rows(store)

    opened = _spy_zip_reads(monkeypatch)
    with caplog.at_level(logging.INFO):
        second = fetch_manager_details(context, full=False, as_of=as_of)

    assert first == len(before) == 5 and second == 0
    assert opened == [_Q3, _Q4], "the newest window plus the previous one; older ZIPs stay closed"
    assert any(_Q3 in message and _Q4 in message for message in caplog.messages), "the log names the windows read"
    pd.testing.assert_frame_equal(before, _rows(store))
    print("\n=== SANITY CHECK: steady state ===")
    print(f"first run {first} rows; second run opened {opened} and changed {second} row(s)")
    print("Two windows opened (newest + overlap), nothing rewritten, no SEC request. Validated.")


def test_amendment_only_window_keeps_the_original_and_never_moves_first_filed_date_later(context: Any, store: Any) -> None:
    """Only the window holding an amendment is cached: the stored original row and its earlier date survive."""
    original = pd.DataFrame(
        [
            {
                "cik": "0000903949",
                "period": pd.Timestamp("2023-06-30"),
                "manager_name": "NICHOLAS COMPANY, INC.",
                "state_or_country": "WI",
                "report_type": "holdings",
                "sec_file_number": "801-4658",
                "accession_number": _NICHOLAS[0],
                "first_filed_date": pd.Timestamp("2023-08-08"),
                "source": _Q3,
            }
        ]
    )
    store.save(Tables.sec_13f_manager_details, original)
    _write_data_set(_cache(context), _Q4, [_NICHOLAS_AMENDED, _PIONEER])

    changed = fetch_manager_details(context, full=False, as_of=pd.Timestamp("2023-12-31"))

    df = _rows(store).set_index("cik")
    assert changed == 1, "only the new Pioneer row is written; the amendment-only Nicholas row is unchanged"
    assert df.loc["0000903949", "accession_number"] == _NICHOLAS[0] and df.loc["0000903949", "first_filed_date"] == pd.Timestamp("2023-08-08")
    assert (df["period"] <= df["first_filed_date"]).all()
    print("\n=== SANITY CHECK: amendment-only window ===")
    print(df[["accession_number", "first_filed_date"]].to_string())
    print("The original accession and its earlier first_filed_date survive an amendment-only window. Validated.")


def test_extract_leaves_profile_columns_untouched(context: Any, store: Any) -> None:
    """A table whose profile columns are populated keeps them byte-identical when the extract refreshes the row."""
    seeded = pd.DataFrame(
        [
            {
                "cik": "0001006435",
                "period": pd.Timestamp("2023-09-30"),
                "manager_name": "STALE NAME",
                "accession_number": _PIONEER[0],
                "first_filed_date": pd.Timestamp("2023-11-15"),
                "source": INDEX_SOURCE,
                "category": "bank",
                "hhi": 0.31,
                "pct_cap_large": 0.7,
                "n_quarters_used": 3,
            }
        ]
    )
    store.save(Tables.sec_13f_manager_details, seeded)
    _write_data_set(_cache(context), _Q4, [_PIONEER])

    fetch_manager_details(context, full=False, as_of=pd.Timestamp("2023-12-31"))

    row = _rows(store).iloc[0]
    assert row["manager_name"] == "PIONEER TRUST BANK N A/OR" and row["source"] == _Q4 and row["first_filed_date"] == pd.Timestamp("2023-10-30")
    assert (row["category"], row["hhi"], row["pct_cap_large"], row["n_quarters_used"]) == ("bank", 0.31, 0.7, 3)
    print("\n=== SANITY CHECK: writer separation ===")
    print(row[["manager_name", "first_filed_date", "category", "hhi", "n_quarters_used"]].to_dict())
    print("Extract columns refreshed, profile columns identical. Validated.")


def test_filings_dated_before_their_period_or_without_a_name_are_dropped() -> None:
    """Known-truth edits of two real rows: a filing dated before its period end and a nameless filing yield no row."""
    sub = pd.DataFrame(
        [
            {"ACCESSION_NUMBER": a, "FILING_DATE": f, "SUBMISSIONTYPE": form, "CIK": c, "PERIODOFREPORT": p}
            for a, f, form, c, p, *_ in (_PIONEER, _NICHOLAS)
        ]
    )
    cover = pd.DataFrame(
        [
            {"ACCESSION_NUMBER": _PIONEER[0], "REPORTCALENDARORQUARTER": _PIONEER[4], "FILINGMANAGER_NAME": "  ", "REPORTTYPE": HOLDINGS},
            {
                "ACCESSION_NUMBER": _NICHOLAS[0],
                "REPORTCALENDARORQUARTER": "31-DEC-2023",
                "FILINGMANAGER_NAME": "NICHOLAS COMPANY, INC.",
                "REPORTTYPE": HOLDINGS,
            },
        ]
    )
    for column in ("FILINGMANAGER_STATEORCOUNTRY", "SECFILENUMBER"):
        cover[column] = None
    filings = zip_manager_filings(zip_submissions(sub, cover), cover, _Q4)

    rows = candidate_rows(filings, pd.Timestamp("1999-06-30"), pd.Timestamp("2023-12-31"))

    assert rows.empty
    kept = candidate_rows(filings.assign(manager_name="NAMED"), pd.Timestamp("1999-06-30"), pd.Timestamp("2023-12-31"))
    assert kept["cik"].tolist() == ["0001006435"], "only the filing dated before its period end is dropped once named"
    print("\n=== SANITY CHECK: row cleaning ===")
    print("A nameless filing and a filing dated before its period end give no row; the named, well-dated one stays. Validated.")


def test_rows_to_save_keeps_stored_zip_evidence_against_index_evidence() -> None:
    """Index evidence never replaces a stored ZIP row's cover fields, but an earlier date still lowers `first_filed_date`."""
    stored = pd.DataFrame(
        [
            {
                "cik": "0001006435",
                "period": pd.Timestamp("2023-09-30"),
                "manager_name": "PIONEER TRUST BANK N A/OR",
                "state_or_country": "OR",
                "report_type": "holdings",
                "sec_file_number": None,
                "accession_number": _PIONEER[0],
                "first_filed_date": pd.Timestamp("2023-10-31"),
                "source": _Q4,
            }
        ]
    )
    candidate = stored.assign(manager_name="INDEX SPELLING", state_or_country=None, report_type=None, source=INDEX_SOURCE, original=True)
    candidate["first_filed_date"] = pd.Timestamp("2023-10-30")

    out = rows_to_save(candidate, stored)

    assert len(out) == 1 and out.iloc[0]["manager_name"] == "PIONEER TRUST BANK N A/OR" and out.iloc[0]["state_or_country"] == "OR"
    assert out.iloc[0]["first_filed_date"] == pd.Timestamp("2023-10-30") and out.iloc[0]["source"] == _Q4
    assert list(out.columns) == list(EXTRACT_COLUMNS)
    print("\n=== SANITY CHECK: evidence precedence ===")
    print("Stored ZIP fields are kept; only the earlier first_filed_date moves. Validated.")


def _real_data_sets() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "data" / "sec_13f_datasets").is_dir():
            return parent / "data" / "sec_13f_datasets"
    return None


@pytest.mark.skipif(_real_data_sets() is None, reason="the cached SEC 13F data sets are not available")
@pytest.mark.parametrize("filename", ["2023q4_form13f.zip", _NESTED])
def test_real_cached_zip_rows_equal_the_covers(filename: str) -> None:
    """On the real 2023q4 and nested ZIPs: one row per filed (cik, period), no notice, cover fields as filed (stripped)."""
    directory = _real_data_sets()
    assert directory is not None
    specs = {"SUBMISSION.tsv": ZipRead(upper=True), "COVERPAGE.tsv": ZipRead(upper=True)}
    tables = read_zip_tables(directory / filename, specs, on_corrupt="skip")
    assert tables
    df_sub, df_cover = tables["SUBMISSION.tsv"], tables["COVERPAGE.tsv"]
    filings = zip_manager_filings(zip_submissions(df_sub, df_cover), df_cover, filename)

    rows = candidate_rows(filings, pd.Timestamp("1999-06-30"), pd.Timestamp("2026-12-31"))

    submitted = zip_submissions(df_sub, df_cover)
    assert len(rows) == len(submitted.drop_duplicates(["cik", "report_period"]))
    assert not set(rows["accession_number"]) & set(df_sub.loc[df_sub["SUBMISSIONTYPE"].str.startswith("13F-NT"), "ACCESSION_NUMBER"])
    assert rows["manager_name"].notna().all() and (rows["manager_name"] == rows["manager_name"].str.strip()).all()
    assert set(rows["report_type"].dropna()) <= {"holdings", "combination"} and (rows["period"] <= rows["first_filed_date"]).all()
    raw = df_cover.drop_duplicates("ACCESSION_NUMBER").set_index("ACCESSION_NUMBER")
    sample = rows.sample(20, random_state=0)
    for row in sample.itertuples():
        cover = raw.loc[row.accession_number]
        assert row.manager_name == str(cover["FILINGMANAGER_NAME"]).strip()
        assert (row.state_or_country or None) == (
            None if bool(pd.isna(cover["FILINGMANAGER_STATEORCOUNTRY"])) else str(cover["FILINGMANAGER_STATEORCOUNTRY"]).strip()
        )
    print(f"\n=== SANITY CHECK: real {filename} ===")
    print(f"{len(rows)} rows from {len(submitted)} 13F-HR(/A) accessions; 20 sampled covers equal the rows.")
    print("Rows follow the cover as filed and carry no notice filer. Validated.")


def test_window_stays_bounded_with_many_old_index_rows_and_issues_no_key_list(
    context: Any, store: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """100k old rows still lacking cover evidence neither pin the window at the first data set nor trigger a key IN-list."""
    n_old = 100_000
    old = pd.DataFrame(
        {
            "cik": [f"{i + 1_000_000:010d}" for i in range(n_old)],
            "period": pd.Timestamp("2013-03-31"),
            "manager_name": "OLD INDEX ROW",
            "accession_number": [f"0000000000-13-{i:06d}" for i in range(n_old)],
            "first_filed_date": pd.Timestamp("2013-04-15"),
            "source": INDEX_SOURCE,
        }
    )
    store.insert(Tables.sec_13f_manager_details, old)
    _write_data_set(_cache(context), _Q2, [_ABACUS])
    _write_data_set(_cache(context), _Q3, [_PIONEER_Q3, _NICHOLAS])
    _write_data_set(_cache(context), _Q4, [_PIONEER, _CAPITAL])
    as_of = pd.Timestamp("2023-12-31")
    calls: list[Any] = []
    real_load = store.load

    unbounded: list[Any] = []

    def _recording_load(table: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs.get("where"))
        if getattr(table, "name", table) == Tables.sec_13f_manager_details.name and not (kwargs.get("where") or kwargs.get("since")):
            unbounded.append(kwargs)
        return real_load(table, *args, **kwargs)

    monkeypatch.setattr(store, "load", _recording_load)
    opened = _spy_zip_reads(monkeypatch)

    with caplog.at_level(logging.INFO):
        first = fetch_manager_details(context, full=False, as_of=as_of)
        opened_first = list(opened)
        second = fetch_manager_details(context, full=False, as_of=as_of)

    assert opened_first == [_Q3, _Q4], "the newest two data sets only, never back to the first one"
    assert any("filings from 2023-07-01" in message for message in caplog.messages)
    assert all(not isinstance(where, pd.DataFrame) and not (isinstance(where, dict) and "cik" in where) for where in calls), "no key list is issued"
    assert not unbounded, f"manager-table loads without a where or since bound: {unbounded}"
    assert first == 4 and second == 0, "the second run is idempotent"
    print("\n=== SANITY CHECK: bounded window ===")
    print(f"{n_old} old index rows; opened {opened_first}; first run {first} row(s), second {second}")
    print("The window starts at the second-newest data set, no key IN-list is used, a re-run writes nothing. Validated.")
