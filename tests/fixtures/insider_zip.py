"""Real insider fixtures: the ownership XMLs in `tests/fixtures/insider/` and their rows in the cached
2026q2 SEC insider zip.

The zip is looked up under `data/` of the working directory or of any parent, so a git worktree
inside the repository reads the main tree's cache. It is opened read-only; a corrupt archive is
skipped, never deleted.
"""

from __future__ import annotations

import logging
from functools import cache
from pathlib import Path

import pandas as pd

from src.data_extract.utils.common.bulk_cache import read_zip_tables
from src.data_extract.utils.institutionals import fetch_insider_transactions as ins
from src.data_extract.utils.institutionals.insider_common import BULK_DATE_FORMATS, LIVE_DATE_FORMATS, build_insider_frame
from src.data_extract.utils.institutionals.insider_edgar_parser import extract_xml_strings

FIXTURE_DIR = Path(__file__).resolve().parent / "insider"
ZIP_MEMBER = Path("data") / "sec_insider_transactions" / "2026q2.zip"


def fixture_accessions() -> list[str]:
    """Accession numbers of every stored ownership XML fixture, sorted."""
    return sorted(path.stem for path in FIXTURE_DIR.glob("*.xml"))


def zip_path() -> Path | None:
    """The cached 2026q2 insider zip under the working directory or a parent, else None."""
    cwd = Path.cwd().resolve()
    return next((base / ZIP_MEMBER for base in (cwd, *cwd.parents) if (base / ZIP_MEMBER).exists()), None)


@cache
def zip_members(path: Path) -> tuple[pd.DataFrame, ...]:
    """(SUBMISSION, REPORTINGOWNER, NONDERIV_TRANS, DERIV_TRANS) of `path`, cut to the fixture accessions."""
    tables = read_zip_tables(path, ins._ZIP_SPECS, on_corrupt="skip", log=logging.getLogger(__name__))
    assert tables, f"{path} has no SUBMISSION member"
    accessions = set(fixture_accessions())
    sub, own, nonderiv, deriv, _ = (ins._accession_rows(df, accessions) for df in tables.values())
    return sub, own, nonderiv, deriv


def zip_frame(path: Path) -> pd.DataFrame:
    """The zip path's typed frame for the fixture accessions."""
    return build_insider_frame(*ins.extract_bulk_strings(*zip_members(path)), date_formats=BULK_DATE_FORMATS)


def xml_frame(accession: str) -> pd.DataFrame:
    """The EDGAR path's typed frame for one fixture accession."""
    df_str, df_owners, _ = extract_xml_strings((FIXTURE_DIR / f"{accession}.xml").read_text(encoding="utf-8"), accession)
    return build_insider_frame(df_str, df_owners, date_formats=LIVE_DATE_FORMATS)
