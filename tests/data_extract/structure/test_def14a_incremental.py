"""Legacy evidence/flatten compatibility and the unchanged worker contract.

Daily section retry integration lives in test_def14a_retry_daily.py and test_sec_def14a_llm.py; the proxy listing and
the subject guard belong to the DEF 14A text walk (test_edgar_text_walks.py).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from src.context import Context
from src.data_extract.utils.structure.def14a import flatten as flatten_mod
from src.data_extract.utils.structure.def14a.def14a_schema import BeneficialOwner as _BeneficialOwner
from src.data_extract.utils.structure.def14a.def14a_schema import Def14AExtract as _Def14AExtract
from src.data_extract.utils.structure.def14a.def14a_schema import DirectorCompensation as _DirectorCompensation
from src.data_extract.utils.structure.def14a.def14a_schema import DirectorInfo as _DirectorInfo
from src.data_extract.utils.structure.def14a.def14a_schema import ExecutiveCompensation as _ExecutiveCompensation
from src.data_extract.utils.structure.def14a.def14a_schema import GovernanceProfile as _GovernanceProfile
from src.data_extract.utils.structure.def14a.flatten import _flatten, _result_frames
from src.data_store.schema import Tables
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.gpt_extract.utils.schemas_gpt import LlmResult, LlmTask
from tests.data_extract.fake_context import extract_config

BeneficialOwner: Any = _BeneficialOwner
Def14AExtract: Any = _Def14AExtract
DirectorCompensation: Any = _DirectorCompensation
DirectorInfo: Any = _DirectorInfo
ExecutiveCompensation: Any = _ExecutiveCompensation
GovernanceProfile: Any = _GovernanceProfile


@pytest.mark.parametrize(
    "label, extract, expected",
    [
        ("empty", Def14AExtract(), False),
        ("metadata", Def14AExtract(company_name="ACME", fiscal_year=2025), False),
        (
            "inferred false defaults",
            Def14AExtract(governance=GovernanceProfile(classified_board=False, dual_class_shares=False)),
            False,
        ),
        ("classified true", Def14AExtract(governance=GovernanceProfile(classified_board=True)), True),
        ("dual class true", Def14AExtract(governance=GovernanceProfile(dual_class_shares=True)), True),
        ("explicit founder false", Def14AExtract(ceo_is_founder=False), True),
        (
            "explicit majority false",
            Def14AExtract(governance=GovernanceProfile(majority_voting_for_directors=False)),
            True,
        ),
        (
            "numeric zero",
            Def14AExtract(governance=GovernanceProfile(insider_ownership_pct=0.0)),
            True,
        ),
        (
            "ownership only",
            Def14AExtract(ownership_holders=[BeneficialOwner(holder_name="Example Fund", holder_type="5pct_holder")]),
            True,
        ),
        ("director only", Def14AExtract(directors=[DirectorInfo(name="Jane Director")]), True),
        (
            "executive compensation only",
            Def14AExtract(compensation=[ExecutiveCompensation(name="Jane CEO", title="CEO", fiscal_year=2025)]),
            True,
        ),
        (
            "director compensation only",
            Def14AExtract(director_compensation=[DirectorCompensation(name="Jane Director", fiscal_year=2025)]),
            True,
        ),
        ("ceo only", Def14AExtract(ceo_name="Jane CEO"), True),
        ("auditor only", Def14AExtract(governance=GovernanceProfile(auditor_name="Example Audit LLP")), True),
        ("board only", Def14AExtract(governance=GovernanceProfile(board_size=8)), True),
        ("blank scalar", Def14AExtract(ceo_name="   "), False),
    ],
)
def test_evidence_gate_parsed_stored_and_persistence_agree(label, extract, expected):
    filing = pd.Series(
        {
            "accession_number": f"acc-{label}",
            "filing_date": pd.Timestamp("2025-04-01"),
            "period_of_report": "2024-12-31",
        }
    )
    row = _flatten("ZZ", filing, extract)
    projected = {column: row.get(column) for column in flatten_mod._DEF14A_EVIDENCE_COLUMNS}
    task = LlmTask(seq=0, payload="proxy", schema=Def14AExtract, table=Tables.def14a_llm, meta={"ticker": "ZZ", "filing": filing})
    frames = _result_frames(LlmResult(seq=0, task=task, parsed=extract))

    parent = frames[Tables.def14a_llm].iloc[0]
    assert flatten_mod._has_extract_evidence(extract) is expected, label
    assert flatten_mod._has_parent_evidence(projected) is expected, label
    assert (parent["def14a_json"] != "_empty") is expected, label
    if not expected:
        assert set(frames) == {Tables.def14a_llm} and list(frames[Tables.def14a_llm].columns) == [
            "ticker",
            "accession_number",
            "as_of",
            "def14a_json",
        ]

    print(f"\n=== SANITY: DEF 14A evidence gate — {label} ===")
    print(f"  parsed={expected}, stored={expected}, {'parent row' if expected else 'empty-filing marker, no child row'}. Validated.")


def test_flatten_surfaces_the_auditor_block():
    """`n_technology_directors` / `technology_committee` were REMOVED -- they were an opinion,
    not an extraction (mean |delta| of 1.06 directors between consecutive filings of the same
    company, only 38.8% unchanged, and wrong by 7x on HUBB 2022 whose own matrix states
    "Cybersecurity and Technology 78%" of 9 directors).

    What replaced them is the auditor block, which is the opposite kind of field: the firm name
    is present in 98% of documents and was the WORST column in the retired edgar table at 2.05%
    fill."""
    extract = Def14AExtract(
        company_name="ACME",
        fiscal_year=2024,
        governance=GovernanceProfile(
            board_size=10,
            auditor_name="Ernst & Young LLP",
            auditor_since_year=1934,
            auditor_fees_usd=12_000_000.0,
            audit_fees_audit_usd=9_000_000.0,
            audit_fees_audit_related_usd=1_000_000.0,
            audit_fees_tax_usd=1_500_000.0,
            audit_fees_other_usd=500_000.0,
            auditor_fees_prior_usd=11_000_000.0,
        ),
    )
    filing = pd.Series({"filing_date": pd.Timestamp("2024-04-01"), "period_of_report": "2023-12-31", "accession_number": "a1"})
    row = _flatten("ACME", filing, extract)
    assert row["auditor_name"] == "Ernst & Young LLP"
    assert row["auditor_since_year"] == 1934
    assert row["auditor_fees"] == 12_000_000.0
    assert row["audit_fees_audit"] == 9_000_000.0
    assert row["audit_fees_tax"] == 1_500_000.0
    assert row["auditor_fees_prior"] == 11_000_000.0
    # the dropped fields must NOT come back
    for gone in ("n_technology_directors", "pct_technology_directors", "technology_committee"):
        assert gone not in row, f"{gone} reappeared in the flatten"
    # absent -> null (not a false 0)
    empty = _flatten("X", filing, Def14AExtract(governance=GovernanceProfile(board_size=8)))
    assert empty["auditor_name"] is None and empty["auditor_fees"] is None
    assert empty["audit_fees_audit"] is None

    print("\n=== SANITY: auditor block flattens; technology fields are gone ===")
    print("  auditor_name='Ernst & Young LLP', since=1934, fees=12,000,000 split 9.0M/1.0M/1.5M/0.5M, prior=11,000,000; absent -> null.")
    print("  n_technology_directors / pct_technology_directors / technology_committee are")
    print("  absent from the flatten -- they were an opinion, not an extraction. Validated.")


def test_section_extractors_run_two_wide_whatever_config_gpt_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data_extract.utils.structure.def14a import fetch as mod

    config = extract_config(data_extract={"years_history": 15}, gpt={"threads": 7})
    context = cast(Context, SimpleNamespace(store=None, log=logging.getLogger("test.def14a.workers"), config_dir="configs", config=config))
    built: list[int] = []

    class _RecordingExtractor(LLMExtractor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.append(self.threads)

    monkeypatch.setattr(mod, "LLMExtractor", _RecordingExtractor)
    extractors = mod.section_extractors(context, config)

    assert built == [2] * 5 and sorted(extractors) == sorted(section.key for section in mod.SECTIONS)
    print("\n=== SANITY: DEF 14A LLM workers ===")
    print(f"  each of the 5 section extractors runs {built[0]}-wide although config.gpt.threads is 7. Validated.")
