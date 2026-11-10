"""`FORM_POLICY` -- which CIKs a ticker's filings are listed from, and how they combine.

Listing reads only the ticker's `FilingScope` (`universe_cik_symbol`): insider forms UNION every event CIK,
every other form (8-Ks and schedules too, P35) SPLITs by the CIK windows, widened 31 days at a seam. No symbol, current or
historical, ever reaches `edgar.Company`: a reused symbol resolves to whoever holds it TODAY, which is
how ~1,750 foreign 8-Ks reached `sec_8k`.

Offline, no network.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pandas as pd
import pytest

from src.data_extract.utils.common.identity import build_identity
from src.data_extract.utils.common.registrant import FORM_POLICY, Combine, combine_for, resolve_registrant_entries
from src.data_extract.utils.identity import cik_history as cik_module
from tests.support.lineage_build import derive_lineage
from tests.support.universe import as_roster

REPO = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------- #
# The policy table                                                             #
# --------------------------------------------------------------------------- #
def test_an_undeclared_form_raises_and_the_message_names_it():
    """Fail closed: a new form family must not inherit a combination rule silently."""
    with pytest.raises(ValueError, match="NT 10-K"):
        combine_for(["NT 10-K"])
    print("\n=== SANITY CHECK: an undeclared form raises ===")
    print("  'NT 10-K' has no FORM_POLICY entry -> ValueError naming it. Validated.")


def test_a_mixed_policy_list_raises_at_the_call_site():
    with pytest.raises(ValueError, match="mix"):
        combine_for(["4", "8-K"])
    print("\n=== SANITY CHECK: a mixed forms list raises ===")
    print("  ['4', '8-K'] mixes union and split -> refused. Validated.")


@pytest.mark.parametrize(
    ("forms", "expected"),
    [
        (["8-K", "8-K/A"], Combine.SPLIT),
        (["SC 13D", "SCHEDULE 13D/A"], Combine.SPLIT),
        (["SC 13G", "SC 13G/A", "SCHEDULE 13G", "SCHEDULE 13G/A"], Combine.SPLIT),
        (["3", "4", "5"], Combine.UNION),
        (["10-K", "10-K/A", "10-K405"], Combine.SPLIT),
        (["10-K", "10-Q"], Combine.SPLIT),
        (["DEF 14A", "DEF 14C", "DEFC14A"], Combine.SPLIT),
    ],
)
def test_every_fetched_form_family_has_the_policy_its_pipeline_needs(forms, expected):
    assert combine_for(forms) is expected
    print(f"\n=== SANITY CHECK: {forms[0]}... -> {expected.value} ===")


def test_both_spellings_of_the_renamed_schedules_are_declared():
    for pair in (("SC 13D", "SCHEDULE 13D"), ("SC 13D/A", "SCHEDULE 13D/A"), ("SC 13G", "SCHEDULE 13G"), ("SC 13G/A", "SCHEDULE 13G/A")):
        assert all(f in FORM_POLICY for f in pair), pair
    print("\n=== SANITY CHECK: both form-string eras are declared ===")
    print("  SC 13D/G and SCHEDULE 13D/G, base and /A, all present. Validated.")


# --------------------------------------------------------------------------- #
# AC-001: CIK only, never a symbol                                             #
# --------------------------------------------------------------------------- #
#: Non-issuer `Company(...)` call sites allowed a string: 13F managers are listed by their own CIK, not a symbol (00-run N1).
_COMPANY_ALLOWLIST = {"src/data_extract/utils/institutionals/fetch_13f_managers.py"}


def test_no_issuer_listing_builds_company_from_anything_but_an_int_cik():
    """AC-001: every issuer `Company(...)` listing takes `int(...)`; dated symbol evidence never becomes a listing argument."""
    offenders: list[str] = []
    for path in sorted((REPO / "src" / "data_extract").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
            if name != "Company" or rel in _COMPANY_ALLOWLIST:
                continue
            arg = node.args[0] if node.args else None
            if not (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "int"):
                offenders.append(f"{rel}:{node.lineno}")
    assert offenders == [], f"Company() built from a non-int argument: {offenders}"
    print("\n=== SANITY CHECK: AC-001 source search ===")
    print(f"  every Company() call under src/data_extract takes int(cik); allowlisted (manager CIKs): {sorted(_COMPANY_ALLOWLIST)}. Validated.")


# --------------------------------------------------------------------------- #
# AC-006 (amended): uncurated multi-CIK consolidating scope                     #
# --------------------------------------------------------------------------- #
def test_an_uncurated_extra_cik_warns_lists_the_roster_cik_and_is_backlogged(tmp_path, monkeypatch, caplog):
    """No register entry, D3 off: the lineage build warns and backlogs the extra CIK; consolidating
    forms list the roster CIK alone, insider forms both, and 8-Ks (P35) the roster CIK alone."""
    (tmp_path / "configs" / "sec").mkdir(parents=True)
    tenure = pd.DataFrame(
        [
            {
                "symbol": "ABC",
                "issuer_cik": "0000000100",
                "valid_from": pd.Timestamp("2006-01-05"),
                "valid_to": pd.Timestamp("2015-07-01"),
                "n_filings": 300,
            },
            {"symbol": "ABC", "issuer_cik": "0000000200", "valid_from": pd.Timestamp("2015-07-01"), "valid_to": pd.NaT, "n_filings": 100},
        ]
    ).assign(source="form345", evidence="fixture")
    dei = tenure.assign(
        source="dei",
        valid_from=[pd.Timestamp("2010-03-01"), pd.Timestamp("2015-07-10")],
        valid_to=[pd.Timestamp("2015-06-16"), pd.Timestamp("2024-11-02")],
    )
    roster = pd.DataFrame([{"ticker": "ABC", "cik": "0000000200"}])
    monkeypatch.setattr(cik_module, "AUTO_WINDOWS_ENABLED", False)
    with caplog.at_level(logging.DEBUG):  # the backlog detail is DEBUG since the phase-5 logging policy (X58)
        build = derive_lineage(tenure, as_roster(roster), pd.DataFrame(columns=["issuer_cik", "owner_cik_raw"]), str(tmp_path / "configs"), dei=dei)
    backlog = build.backlog[build.backlog["kind"].eq("multi_cik_no_window")]
    assert len(backlog) == 1 and backlog["cik"].iloc[0] == "0000000100"
    assert "curation backlog" in caplog.text and "multi_cik_no_window" in caplog.text

    scope = build_identity(lineage=build.rows, tenure=tenure, roster_cik=as_roster(roster)).filing_scope("ABC")
    rows = pd.DataFrame(
        [
            {"cik": cik, "company": "Fixture", "form": form, "filed": pd.Timestamp(filed), "accession": f"{name}-{form}"}
            for cik, name, filed in (("0000000100", "pred", "2012-02-01"), ("0000000200", "succ", "2020-02-01"))
            for form in ("10-K", "8-K", "4")
        ]
    )

    def listed(form: str) -> list[str]:
        return resolve_registrant_entries(scope, rows, [form])["accession"].tolist()

    assert listed("10-K") == ["succ-10-K"]
    assert listed("8-K") == ["succ-8-K"]
    assert listed("4") == ["pred-4", "succ-4"]
    print("\n=== SANITY CHECK: AC-006 uncurated multi-CIK ===")
    print(f"  build WARNING + backlog row ({backlog['detail'].iloc[0][:70]}...); 10-K and 8-K index rows of roster CIK 200 only; Form 4 both")
