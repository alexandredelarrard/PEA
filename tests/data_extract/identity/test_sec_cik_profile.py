"""`sec_cik_profile` from the bulk SEC `submissions.zip`: an exact parse of a tiny known-truth zip, and the table replace.

The zip is built here with three CIKs shaped like SEC's files: Exxon Mobil 34088 (dated former name, one supplementary
page of older filings, Form 25-NSE and 15-15D), ExxonMobil Holdings 2115436 (current XOM listing, 8-K12B) and Paramount
Global 813828 (three dated former names, Form 15-12G, a 25 amendment); an orphan page and a non-JSON member are ignored.
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.identity import sec_profile
from src.data_store import ddl
from src.data_store.schema import Tables
from src.data_store.store import DataStore

_SCHEMA_SQL = Path(__file__).resolve().parents[3] / "sql" / "schema.sql"
_RUN = pd.Timestamp("2026-11-04")


def _filings(rows: list[tuple[str, str, str]]) -> dict[str, list[str]]:
    """SEC's parallel filing columns from `(accession, filed, form)` rows, plus a column the parser ignores."""
    return {
        "accessionNumber": [r[0] for r in rows],
        "filingDate": [r[1] for r in rows],
        "form": [r[2] for r in rows],
        "primaryDocument": ["doc.htm" for _ in rows],
    }


def _main(cik: str, name: str, *, tickers: list[str], exchanges: list[str | None], former: list[dict[str, str]], recent: list, files: list) -> dict:
    return {
        "cik": cik.lstrip("0"),
        "entityType": "operating",
        "sic": "2911",
        "sicDescription": "Petroleum Refining",
        "name": name,
        "tickers": tickers,
        "exchanges": exchanges,
        "ein": "135409005",
        "stateOfIncorporation": "NJ",
        "formerNames": former,
        "filings": {"recent": _filings(recent), "files": files},
    }


def _build_zip(path: Path) -> Path:
    xom = _main(
        "0000034088",
        "EXXON MOBIL CORP",
        tickers=[],
        exchanges=[],
        former=[{"name": "EXXON CORP", "from": "1994-05-11T00:00:00.000Z", "to": "1999-11-30T00:00:00.000Z"}],
        recent=[
            ("0000034088-26-000050", "2026-09-25", "10-Q"),
            ("0000034088-26-000041", "2026-07-02", "25-NSE"),
            ("0000034088-24-000020", "2024-06-26", "25-NSE"),
            ("0000034088-10-000005", "2010-03-01", "15-15D"),
        ],
        files=[{"name": "CIK0000034088-submissions-001.json", "filingCount": 2, "filingFrom": "1994-03-04", "filingTo": "2001-05-01"}],
    )
    xom_page = _filings([("0000034088-01-000003", "2001-05-01", "25"), ("0000034088-94-000001", "1994-03-04", "10-K")])
    holdco = {
        **_main(
            "0002115436",
            "ExxonMobil Holdings Corp",
            tickers=["XOM"],
            exchanges=["NYSE"],
            former=[],
            recent=[("0002115436-26-000002", "2026-07-10", "8-K"), ("0002115436-26-000001", "2026-07-01", "8-K12B")],
            files=[],
        ),
        "ein": "414104094",
        "stateOfIncorporation": "TX",
    }
    paramount = {
        **_main(
            "0000813828",
            "Paramount Global",
            tickers=[],
            exchanges=[],
            former=[
                {"name": "ViacomCBS Inc.", "from": "2019-12-04T00:00:00.000Z", "to": "2022-02-15T00:00:00.000Z"},
                {"name": "CBS CORP", "from": "2006-01-09T00:00:00.000Z", "to": "2019-12-04T00:00:00.000Z"},
                {"name": "VIACOM INC", "from": "1994-01-11T00:00:00.000Z", "to": "2006-01-05T00:00:00.000Z"},
            ],
            recent=[
                ("0000813828-25-000090", "2025-08-18", "15-12G"),
                ("0000813828-25-000080", "2025-08-08", "25/A"),
                ("0000813828-25-000010", "2025-02-26", "10-K"),
            ],
            files=[],
        ),
        "sic": "4833",
        "ein": "042949533",
        "stateOfIncorporation": "DE",
        "entityType": "operating",
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("CIK0000034088.json", json.dumps(xom))
        archive.writestr("CIK0000034088-submissions-001.json", json.dumps(xom_page))
        archive.writestr("CIK0002115436.json", json.dumps(holdco))
        archive.writestr("CIK0000813828.json", json.dumps(paramount))
        archive.writestr("CIK0000000099-submissions-001.json", json.dumps(_filings([])))  # a page with no main JSON: skipped
        archive.writestr("README.txt", "not a CIK member")
    return path


def test_the_submissions_zip_parses_to_one_exact_row_per_cik(tmp_path: Path) -> None:
    path = _build_zip(tmp_path / "submissions.zip")

    parsed, skipped = sec_profile.parse_submissions(path)
    profile = parsed.set_index("cik")

    assert list(profile.index) == ["0000034088", "0000813828", "0002115436"] and skipped == 1
    assert list(profile.reset_index().columns) == list(Tables.sec_cik_profile.read_columns)
    xom, holdco, paramount = profile.loc["0000034088"], profile.loc["0002115436"], profile.loc["0000813828"]
    assert json.loads(xom["former_names"]) == [{"name": "EXXON CORP", "from": "1994-05-11", "to": "1999-11-30"}]
    assert (xom["first_filed"], xom["last_filed"], xom["n_filings"]) == (pd.Timestamp("1994-03-04"), pd.Timestamp("2026-09-25"), 6)
    assert json.loads(xom["form_25"]) == [
        {"form": "25", "filed": "2001-05-01", "accession": "0000034088-01-000003"},
        {"form": "25-NSE", "filed": "2024-06-26", "accession": "0000034088-24-000020"},
        {"form": "25-NSE", "filed": "2026-07-02", "accession": "0000034088-26-000041"},
    ]
    assert json.loads(xom["form_15"]) == [{"form": "15-15D", "filed": "2010-03-01", "accession": "0000034088-10-000005"}]
    assert pd.isna(xom["form_8k12b"]) and pd.isna(xom["tickers"]) and pd.isna(xom["exchanges"])
    assert (xom["name"], xom["sic"], xom["ein"], xom["state_of_incorporation"], xom["entity_type"]) == (
        "EXXON MOBIL CORP",
        "2911",
        "135409005",
        "NJ",
        "operating",
    )
    assert json.loads(holdco["tickers"]) == ["XOM"] and json.loads(holdco["exchanges"]) == ["NYSE"]
    assert json.loads(holdco["form_8k12b"]) == [{"form": "8-K12B", "filed": "2026-07-01", "accession": "0002115436-26-000001"}]
    assert pd.isna(holdco["former_names"]) and pd.isna(holdco["form_25"]) and holdco["n_filings"] == 2
    assert [entry["name"] for entry in json.loads(paramount["former_names"])] == ["ViacomCBS Inc.", "CBS CORP", "VIACOM INC"]
    assert json.loads(paramount["former_names"])[1] == {"name": "CBS CORP", "from": "2006-01-09", "to": "2019-12-04"}
    assert json.loads(paramount["form_15"]) == [{"form": "15-12G", "filed": "2025-08-18", "accession": "0000813828-25-000090"}]
    assert [entry["form"] for entry in json.loads(paramount["form_25"])] == ["25/A"]
    assert profile["form_8k12g3"].isna().all()
    assert profile["fetched_at"].nunique() == 1

    print("\n=== SANITY CHECK: submissions.zip -> sec_cik_profile (exact) ===")
    print("  3 CIKs -> 3 rows; XOM 34088: EXXON CORP 1994-05-11..1999-11-30, 6 filings 1994-03-04..2026-09-25 incl. the supplementary page,")
    print(
        "  25 (2001) + 25-NSE x2, 15-15D 2010; holdco 2115436: XOM/NYSE, 8-K12B 2026-07-01; Paramount 813828: 3 dated former names, 15-12G. Validated."
    )


def test_the_profile_table_is_registered_and_spliced_into_schema_sql() -> None:
    table = Tables.sec_cik_profile
    blocks = ddl.existing_blocks(_SCHEMA_SQL.read_text(encoding="utf-8"))

    persisted = blocks[table.name]

    assert table.pk == ("cik",) and table.ticker_col is None and set(table.date_type_cols) == {"first_filed", "last_filed"}
    assert 'PRIMARY KEY ("cik")' in persisted
    for column in table.read_columns:
        assert f'"{column}" ' in persisted, column
    assert '"first_filed" DATE' in persisted and '"n_filings" BIGINT' in persisted and '"fetched_at" TIMESTAMP' in persisted
    assert "sec_company_tickers" in blocks and "ix_universe_cik_symbol_company_id" in _SCHEMA_SQL.read_text(encoding="utf-8")

    print("\n=== SANITY CHECK: sec_cik_profile DDL ===")
    print(f"  registered with PK (cik) and {len(table.read_columns)} columns, hand-spliced into sql/schema.sql; neighbouring blocks kept. Validated.")


def test_a_refresh_downloads_once_per_run_date_and_replaces_the_table(
    tmp_path: Path, sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _build_zip(tmp_path / "source.zip")
    cache = tmp_path / "sec_submissions"
    cache.mkdir()
    downloads: list[str] = []

    def fake_download(context: Any, url: str, path: Path, *, timeout: int = 300) -> int:
        downloads.append(url)
        path.write_bytes(source.read_bytes())
        return 200

    monkeypatch.setattr(sec_profile, "download", fake_download)
    infos: list[str] = []
    context = SimpleNamespace(store=sqlite_store, log=SimpleNamespace(info=lambda msg, *a: infos.append(msg % a)))
    stale = sec_profile.parse_submissions(source)[0].head(1).assign(cik="0000000001", name="STALE CO")
    sqlite_store.save(Tables.sec_cik_profile, stale)

    written = sec_profile.refresh_sec_cik_profile(context, cache, pd.Timestamp.now().normalize())
    again = sec_profile.refresh_sec_cik_profile(context, cache, pd.Timestamp.now().normalize())
    stored = sqlite_store.load(Tables.sec_cik_profile)

    assert written == again == 3 and downloads == [sec_profile.SUBMISSIONS_URL]
    assert sorted(stored["cik"]) == ["0000034088", "0000813828", "0002115436"], "the stale CIK is gone"
    assert stored.set_index("cik").loc["0000034088", "first_filed"] == pd.Timestamp("1994-03-04").date()

    os.utime(cache / "submissions.zip", (0, 0))
    sec_profile.refresh_sec_cik_profile(context, cache, _RUN)
    assert len(downloads) == 2, "a zip fetched on another day is downloaded again"
    assert any("submissions.zip" in line for line in infos)

    print("\n=== SANITY CHECK: sec_cik_profile refresh ===")
    print("  one download per run date (a retry reuses today's zip); the replace drops the stale CIK 0000000001 and writes 3 rows;")
    print("  an older cached zip is downloaded again. Validated.")
