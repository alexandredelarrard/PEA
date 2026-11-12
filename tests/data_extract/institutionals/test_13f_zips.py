"""
test_13f_zips.py (tests/data_extract/institutionals/test_13f_zips.py)
---------------------------------------------------------------------
The 13F data-set ZIP readers (`fetch_13f_zips`) on small fixture ZIPs written to `tmp_path` (no download).
"""

from __future__ import annotations

import zipfile
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.institutionals.fetch_13f_zips import cached_filing_vintages

# (cik, accession, form, filed, period, [(cusip, issuer, value as filed, shares, putcall)])
_FILING = ("1", "0000000001-24-000004", "13F-HR", "2024-08-14", "2024-06-30", [("111111111", "NEWCO", 4_000_000, 25_000, "")])


def _dmy(day: str) -> str:
    return pd.Timestamp(day).strftime("%d-%b-%Y").upper()


def write_filings_zip(path: Any, filings: list[tuple], *, confidential: str | None = "N", omit_confidential: bool = False) -> None:
    """A data-set ZIP with the SEC member names and upper-case headers."""
    sub = pd.DataFrame(
        [
            {"ACCESSION_NUMBER": a, "FILING_DATE": _dmy(f), "SUBMISSIONTYPE": form, "CIK": c, "PERIODOFREPORT": _dmy(p)}
            for c, a, form, f, p, _ in filings
        ]
    )
    cover = pd.DataFrame(
        [
            {
                "ACCESSION_NUMBER": a,
                "REPORTCALENDARORQUARTER": _dmy(p),
                "ISAMENDMENT": "Y" if form.endswith("/A") else "N",
                "AMENDMENTTYPE": "RESTATEMENT" if form.endswith("/A") else "",
                "AMENDMENTNO": 1 if form.endswith("/A") else 0,
                "REPORTTYPE": "13F HOLDINGS REPORT",
            }
            for _, a, form, _, p, _ in filings
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "ACCESSION_NUMBER": a,
                "TABLEENTRYTOTAL": len(lines),
                "TABLEVALUETOTAL": sum(line[2] for line in lines),
                "ISCONFIDENTIALOMITTED": confidential,
            }
            for _, a, _, _, _, lines in filings
        ]
    )
    if omit_confidential:
        summary = summary.drop(columns="ISCONFIDENTIALOMITTED")
    info = pd.DataFrame(
        [
            {
                "ACCESSION_NUMBER": a,
                "INFOTABLE_SK": i,
                "NAMEOFISSUER": name,
                "TITLEOFCLASS": "COM",
                "CUSIP": cusip,
                "VALUE": value,
                "SSHPRNAMT": shares,
                "SSHPRNAMTTYPE": "SH",
                "PUTCALL": putcall,
                "INVESTMENTDISCRETION": "SOLE",
            }
            for _, a, _, _, _, lines in filings
            for i, (cusip, name, value, shares, putcall) in enumerate(lines)
        ]
    )
    with zipfile.ZipFile(path, "w") as archive:
        for name, frame in (("SUBMISSION.tsv", sub), ("COVERPAGE.tsv", cover), ("SUMMARYPAGE.tsv", summary), ("INFOTABLE.tsv", info)):
            archive.writestr(name, frame.to_csv(sep="\t", index=False))


@pytest.mark.parametrize(("flag", "restricted"), [(None, False), ("N", False), ("false", False), ("Y", True), ("true", True)])
@pytest.mark.parametrize("omit_column", [False, True])
def test_optional_zip_confidentiality_requires_affirmative_evidence(tmp_path: Any, flag: str | None, restricted: bool, omit_column: bool) -> None:
    path = tmp_path / "source.zip"
    write_filings_zip(path, [_FILING], confidential=flag, omit_confidential=omit_column)
    packets = cached_filing_vintages(path, [_FILING[1]])
    assert len(packets) == 1
    manifest, holdings = packets[0]
    assert bool(manifest.restricted.iloc[0]) is (restricted and not omit_column)
    assert holdings.shares.iloc[0] == 25000 and holdings.value_usd.iloc[0] == 4000000
    print(f"SANITY: ZIP flag {flag!r}, absent column={omit_column}; only affirmative metadata restricts a complete book.")
