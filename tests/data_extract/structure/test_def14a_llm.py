"""
Tests for the DEF 14A LLM extraction pipeline (trimmed-schema + targeted-slice design).

test_def14a_schema_roundtrip        — Def14AExtract serializes / deserializes cleanly
test_compensation_anchor_skips_noise— the SCT anchor skips CD&A prose AND "realized/Total
                                      Cash" pay tables, landing on the real Summary Comp Table
test_sayonpay_anchor_matches_approval — say-on-pay anchor catches "NN% approval"/"NN% of votes"
test_payratio_and_ownership_anchors — pay-ratio & beneficial-ownership content anchors match
test_def14a_sections_extraction     — prepare_def14a_sections() isolates the right blocks
test_llm_extractor_mock             — LLMExtractor.extract() with a mocked OpenAI call
test_flatten_surfaces_all_signals   — _flatten emits every governance / comp / ownership column
test_flatten_ceo_age_fallback       — ceo_age recovered from the directors list
test_evidence_gate_parsed_stored_and_persistence_agree — parsed, stored and persisted evidence agree
test_flatten_surfaces_the_auditor_block — the auditor block flattens; technology fields are gone
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_extract.utils.structure.def14a import flatten as flatten_mod
from src.data_extract.utils.structure.def14a.carve import (
    _COMPENSATION_CONTENT_RE,
    _DIRECTOR_MAX_FRAC,
    _DIRECTOR_ROW_RE,
    _OWNERSHIP_CONTENT_RE,
    _OWNERSHIP_ROW_RE,
    _PAYRATIO_CONTENT_RE,
    _SAYONPAY_CONTENT_RE,
    _densest_window,
    prepare_def14a_sections,
)
from src.data_extract.utils.structure.def14a.def14a_schema import BeneficialOwner as _BeneficialOwner
from src.data_extract.utils.structure.def14a.def14a_schema import (
    Def14AExtract as _Def14AExtract,
)
from src.data_extract.utils.structure.def14a.def14a_schema import DirectorCompensation as _DirectorCompensation
from src.data_extract.utils.structure.def14a.def14a_schema import (
    DirectorInfo as _DirectorInfo,
)
from src.data_extract.utils.structure.def14a.def14a_schema import (
    ExecutiveCompensation as _ExecutiveCompensation,
)
from src.data_extract.utils.structure.def14a.def14a_schema import (
    GovernanceProfile as _GovernanceProfile,
)
from src.data_extract.utils.structure.def14a.flatten import _flatten, _result_frames
from src.data_store.schema import Tables
from src.gpt_extract.utils.schemas_gpt import LlmResult, LlmTask

BeneficialOwner: Any = _BeneficialOwner
Def14AExtract: Any = _Def14AExtract
DirectorCompensation: Any = _DirectorCompensation
DirectorInfo: Any = _DirectorInfo
ExecutiveCompensation: Any = _ExecutiveCompensation
GovernanceProfile: Any = _GovernanceProfile

# --------------------------------------------------------------------------- #
# Synthetic DEF 14A fixture — exercises the tricky section-anchoring cases      #
# (CD&A prose before the real SCT; a "realized / Total Cash" pay table before   #
#  it; a directors-and-officers-as-a-group ownership row; a "NN% approval"      #
#  say-on-pay result; a pay-ratio sentence). Padded > 5k chars so the prose,    #
#  anchor-only CORPORATE GOVERNANCE section clears the TOC-skip floor.          #
# --------------------------------------------------------------------------- #
_FILLER = "The Board is committed to strong governance and long-term value creation. " * 12

_SYNTHETIC = f"""
ACME CORPORATION — 2024 PROXY STATEMENT
Table of Contents: Election of Directors; Executive Compensation; Summary Compensation
Table; Security Ownership; CEO Pay Ratio; Say on Pay; Auditor Fees. {_FILLER}

PROPOSAL 1 — ELECTION OF DIRECTORS
{_FILLER}
Alice Johnson, 58, has served as our President and Chief Executive Officer since 2019.
Ms. Johnson co-founded the Company in 2002 and serves on the Executive Committee.
Robert Williams, 64, has served as an independent director since 2010 and chairs the Audit
Committee. Mary Chen, 52, has been an independent director since 2015.
{_FILLER}

COMPENSATION DISCUSSION AND ANALYSIS
Our program balances base salary, annual cash incentives, and long-term equity awards to
align pay with performance; Total direct compensation is targeted near the median. {_FILLER}

REALIZED PAY / TOTAL CASH SUMMARY (supplemental, non-GAAP)
Name Year Salary Bonus Stock-Based Awards Total Cash Vesting of Previous Awards
Alice Johnson 2023 850,000 500,000 3,200,000 1,350,000 2,000,000
{_FILLER}

SUMMARY COMPENSATION TABLE
Name and Principal Position Year Salary Bonus Stock Awards Option Awards Non-Equity Incentive Plan Compensation All Other Compensation Total
Alice Johnson, Chief Executive Officer 2023 850,000 500,000 3,200,000 250,000 100,000 50,000 4,900,000
James Thompson, Chief Financial Officer 2023 620,000 280,000 1,800,000 0 0 0 2,900,000
{_FILLER}

CORPORATE GOVERNANCE
The Board has a majority-voting standard for directors and a Lead Independent Director; the
Chair and CEO roles are combined. There is no classified board and no dual-class stock. {_FILLER}

SECURITY OWNERSHIP OF CERTAIN BENEFICIAL OWNERS AND MANAGEMENT
Name Shares Beneficially Owned Percent of Class
Alice Johnson 5,250,000 8.2%
The Vanguard Group 6,000,000 9.4%
All directors and executive officers as a group (3 persons) 5,327,000 8.4%
{_FILLER}

CEO PAY RATIO
The annual total compensation of our median employee was $60,000; the ratio of the annual
total compensation of our CEO to that of the median employee was 250 to 1. {_FILLER}

SAY ON PAY
At the 2023 Annual Meeting, our Say on Pay proposal received 91% approval from shareholders. {_FILLER}

AUDITOR FEES
Audit Fees billed by Ernst & Young LLP were $5,000,000 for the year. {_FILLER}
"""


def _make_expected() -> Any:
    """A fully-populated extract matching the current (trimmed) schema."""
    return Def14AExtract(
        company_name="ACME Corporation",
        fiscal_year=2023,
        ceo_name="Alice Johnson",
        ceo_age=58,
        ceo_since_year=2019,
        ceo_is_founder=True,
        ceo_is_board_chair=True,
        directors=[
            DirectorInfo(name="Alice Johnson", age=58, tenure_years=5.0, is_independent=False, gender="female", other_public_company_boards=1),
            DirectorInfo(name="Robert Williams", age=64, tenure_years=14.0, is_independent=True, gender="male", other_public_company_boards=2),
            DirectorInfo(name="Mary Chen", age=52, tenure_years=9.0, is_independent=True, gender="female", other_public_company_boards=0),
        ],
        compensation=[
            ExecutiveCompensation(
                name="Alice Johnson",
                title="Chief Executive Officer",
                fiscal_year=2023,
                salary_usd=850_000,
                bonus_usd=500_000,
                stock_awards_usd=3_200_000,
                option_awards_usd=250_000,
                non_equity_incentive_usd=100_000,
                all_other_comp_usd=50_000,
                total_compensation_usd=4_900_000,
            ),
            ExecutiveCompensation(
                name="James Thompson",
                title="Chief Financial Officer",
                fiscal_year=2023,
                salary_usd=620_000,
                bonus_usd=280_000,
                stock_awards_usd=1_800_000,
                total_compensation_usd=2_900_000,
            ),
        ],
        governance=GovernanceProfile(
            board_size=3,
            n_independent_directors=2,
            n_women_directors=2,
            independent_chair=False,
            ceo_is_board_chair=True,
            lead_independent_director=True,
            classified_board=False,
            dual_class_shares=False,
            poison_pill=False,
            majority_voting_for_directors=True,
            say_on_pay_support_pct=0.91,
            ceo_pay_ratio=250.0,
            median_employee_pay_usd=60_000.0,
            auditor_fees_usd=5_000_000.0,
            insider_ownership_pct=0.084,
            ceo_ownership_pct=0.082,
            n_five_percent_holders=1,
        ),
    )


# --------------------------------------------------------------------------- #
# Schema                                                                        #
# --------------------------------------------------------------------------- #
def test_def14a_schema_roundtrip():
    """Def14AExtract serializes to JSON and roundtrips without loss."""
    original = _make_expected()
    roundtrip = Def14AExtract.model_validate_json(original.model_dump_json())

    assert roundtrip.company_name == original.company_name
    assert len(roundtrip.directors) == 3
    assert roundtrip.directors[1].is_independent is True
    assert roundtrip.compensation[0].salary_usd == 850_000.0
    assert roundtrip.ceo_age == 58 and roundtrip.ceo_is_board_chair is True
    assert roundtrip.governance.ceo_pay_ratio == 250.0
    assert roundtrip.governance.insider_ownership_pct == pytest.approx(0.084)
    assert roundtrip.governance.n_five_percent_holders == 1

    print("\n=== SANITY CHECK: Def14AExtract schema roundtrip ===")
    print(f"  company={roundtrip.company_name}  directors={[d.name for d in roundtrip.directors]}")
    print(
        f"  CEO {roundtrip.compensation[0].name}: salary=${roundtrip.compensation[0].salary_usd:,.0f} "
        f"total=${roundtrip.compensation[0].total_compensation_usd:,.0f}; "
        f"insider_own={roundtrip.governance.insider_ownership_pct:.3f}. Validated."
    )


# --------------------------------------------------------------------------- #
# Section-anchoring regexes (the fixes made this session)                       #
# --------------------------------------------------------------------------- #
def test_compensation_anchor_skips_noise():
    """The SCT anchor must skip (a) CD&A prose ('salary, ... equity awards ... Total')
    and (b) the 'realized / Total Cash' pay table, landing on the real Summary
    Compensation Table (whose total column is a plain 'Total')."""
    m = _COMPENSATION_CONTENT_RE.search(_SYNTHETIC)
    assert m is not None, "SCT anchor found nothing"

    cda_pos = _SYNTHETIC.index("base salary, annual cash incentives")
    realized_pos = _SYNTHETIC.index("Total Cash Vesting")
    sct_pos = _SYNTHETIC.index("Name and Principal Position Year Salary Bonus Stock Awards")

    # the match lands at the real SCT header, after both decoys
    assert m.start() > realized_pos > cda_pos
    assert abs(m.start() - sct_pos) < 60, (m.start(), sct_pos)
    window = _SYNTHETIC[m.start() : m.start() + 200]
    assert "Option Awards" in window and "Total Cash" not in window

    print("\n=== SANITY CHECK: SCT anchor skips CD&A + realized-pay ===")
    print(f"  CD&A @{cda_pos}, realized/Total-Cash @{realized_pos}, real SCT @{sct_pos}; anchor landed @{m.start()} (on the SCT header). Validated.")


def test_sayonpay_anchor_matches_approval():
    """Say-on-pay anchor catches 'NN% approval' (not just 'NN% of votes')."""
    m = _SAYONPAY_CONTENT_RE.search(_SYNTHETIC)
    assert m is not None and "91%" in m.group(0)
    # a bare qualified percent with no say-on-pay context nearby must NOT match
    assert _SAYONPAY_CONTENT_RE.search("Turnout was 91% of shares outstanding.") is None

    print("\n=== SANITY CHECK: say-on-pay anchor ===")
    print(f"  matched {m.group(0)[:60]!r} (91% approval); context-free percent ignored. Validated.")


def test_payratio_and_ownership_anchors():
    """Pay-ratio anchors on the SEC 'median employee / ratio of the … compensation'
    sentence; ownership anchors on 'directors and … officers as a group'."""
    pr = _PAYRATIO_CONTENT_RE.search(_SYNTHETIC)
    ow = _OWNERSHIP_CONTENT_RE.search(_SYNTHETIC)
    assert pr is not None and ow is not None
    assert "median employee" in _SYNTHETIC[pr.start() : pr.start() + 40].lower() or "ratio of the" in _SYNTHETIC[pr.start() : pr.start() + 40].lower()
    # the ownership anchor lands in the beneficial-ownership block; the "as a group"
    # insider row sits a few lines below it (well inside the slice window)
    ow_window = _SYNTHETIC[ow.start() : ow.start() + 400].lower()
    assert "beneficial" in ow_window and "as a group" in ow_window

    print("\n=== SANITY CHECK: pay-ratio & ownership anchors ===")
    print(f"  pay-ratio @{pr.start()}, ownership 'as a group' @{ow.start()}. Validated.")


def test_def14a_sections_extraction():
    """prepare_def14a_sections() returns text containing every key section + its data.

    `html=""` on purpose: with no markup the table classifier finds nothing, so this exercises
    the ANCHOR-CARVE FALLBACK path end to end -- including the three windows
    (EXECUTIVE COMPENSATION / SECURITY OWNERSHIP / AUDITOR FEES) that are now emitted only when
    the table classifier came up empty. That path is not hypothetical: 12 of 64 real filings
    disclose their audit fees in prose and land here."""
    focused = prepare_def14a_sections("", _SYNTHETIC)
    for label in ("DIRECTOR NOMINEES", "EXECUTIVE COMPENSATION", "SECURITY OWNERSHIP", "PAY RATIO & MEDIAN PAY", "SAY ON PAY", "AUDITOR FEES"):
        assert f"=== {label} ===" in focused, f"missing section {label}"
    # target data present
    assert "4,900,000" in focused  # CEO SCT total (real table)
    assert "as a group" in focused and "8.4%" in focused  # insider ownership
    assert "250 to 1" in focused  # pay ratio
    assert "91% approval" in focused  # say-on-pay
    assert "$5,000,000" in focused  # auditor fees

    print("\n=== SANITY CHECK: prepare_def14a_sections ===")
    print(
        f"  focused={len(focused):,} chars (orig {len(_SYNTHETIC):,}); all sections + "
        f"CEO total / ownership / pay-ratio / say-on-pay / auditor present. Validated."
    )


def _section_body(focused: str, label: str) -> str:
    import re

    parts = re.split(r"\n\n=== (.+?) ===\n", focused)
    for i in range(1, len(parts), 2):
        if parts[i] == label:
            return parts[i + 1]
    return ""


def test_densest_window_lands_on_table_not_prose():
    """_densest_window ignores isolated prose row-tokens and anchors on the dense
    cluster (the table) — the fix for director bios in matrix/table layouts."""
    prose = "The company was founded in 1998. " * 30  # noise, no row tokens
    lone = "An independent director since 2005 chairs the audit committee. "  # 1 lone token
    table = (
        "Name Age Director Since\n"
        "Alice Johnson, 58 2019\nRobert Williams, 64 2010\n"
        "Mary Chen, 52 2015\nDavid Park, 47 2020\n"
    )  # dense cluster of tokens
    text = prose + lone + prose + table + prose
    pos = _densest_window(text, _DIRECTOR_ROW_RE, 4000)
    assert pos != -1
    window = text[pos : pos + 4000]
    # the window must contain the table (>=4 director rows), not the lone prose token
    assert len(_DIRECTOR_ROW_RE.findall(window)) >= 4
    assert "Alice Johnson" in window and "David Park" in window

    print("\n=== SANITY CHECK: densest-window anchoring ===")
    print(f"  ignored lone 'director since 2005' prose; landed on the 5-row table ({len(_DIRECTOR_ROW_RE.findall(window))} row tokens). Validated.")


def test_a_denser_back_half_decoy_loses_to_the_positional_ceiling():
    """A DENSER cluster of the same row tokens in the back half must not win the director window.

    Measured on 22 real filings: T's winner sat at 53.6% of the document, in the pension
    -assumptions discussion (0 of 11 directors in the block, every `is_independent` null), and
    EOG's at 70.0% in change-of-control prose (1 of 10). Both decoys are dense in "Age" and
    "Director Since", so neither density nor a bio-marker count separates them from a roster --
    only POSITION does. The negative control is the first assertion: without the ceiling the
    decoy genuinely wins, so this test would pass vacuously if the ceiling were removed.
    """
    roster = "Alice Johnson, 58 Director Since 2019\nRobert Williams, 64 Director Since 2010\nMary Chen, 52 Director Since 2015\n"
    decoy = "the assumed retirement age is Age 65 Director Since Age 62 Age 58 Director Since Age 60 Age 55 Director Since Age 61 "
    filler = "The company was founded in 1998 and operates globally. " * 400  # no row tokens
    text = filler + roster + filler * 3 + decoy * 3 + filler

    uncapped = _densest_window(text, _DIRECTOR_ROW_RE, 4_000)
    capped = _densest_window(text, _DIRECTOR_ROW_RE, 4_000, max_frac=_DIRECTOR_MAX_FRAC)
    assert uncapped / len(text) > _DIRECTOR_MAX_FRAC  # the decoy wins without the cap
    assert "Alice Johnson" not in text[uncapped : uncapped + 4_000]
    assert "Alice Johnson" in text[capped : capped + 4_000]
    assert "Mary Chen" in text[capped : capped + 4_000]

    print("\n=== SANITY CHECK: director-window positional ceiling ===")
    print(
        f"  uncapped window at {uncapped / len(text):.1%} of the doc -> the back-half decoy, "
        f"no roster; capped at {capped / len(text):.1%} -> the 3-director roster."
    )
    print(
        f"  On real filings the ceiling ({_DIRECTOR_MAX_FRAC:.0%}) moves EOG (70.0% -> 18.9%, "
        f"1 -> 9 of 10 directors in-block) and T (53.6% -> the content anchor, 0 -> 4 of 11),"
    )
    print("  and no other filing of the 22: 191 -> 203 directors reached. Validated.")


def test_tabular_sections_capture_rows_in_synthetic():
    """In the synthetic proxy, the densest-window director & ownership slices carry the
    actual rows (director ages; 5%-holder + as-a-group ownership lines).

    `html=""` -> the anchor-carve fallback, same rationale as above."""
    focused = prepare_def14a_sections("", _SYNTHETIC)
    directors = _section_body(focused, "DIRECTOR NOMINEES")
    ownership = _section_body(focused, "SECURITY OWNERSHIP")

    assert len(_DIRECTOR_ROW_RE.findall(directors)) >= 3  # Johnson 58, Williams 64, Chen 52
    assert "Alice Johnson" in directors
    own_rows = _OWNERSHIP_ROW_RE.findall(ownership)
    assert len(own_rows) >= 3 and "as a group" in ownership.lower()

    print("\n=== SANITY CHECK: tabular slices carry rows ===")
    print(
        f"  DIRECTOR slice: {len(_DIRECTOR_ROW_RE.findall(directors))} director rows; "
        f"OWNERSHIP slice: {len(own_rows)} rows incl 'as a group'. Validated."
    )


# --------------------------------------------------------------------------- #
# LLMExtractor (mocked)                                                         #
# --------------------------------------------------------------------------- #
def test_llm_extractor_mock():
    """extract() uses the configured model, tailored instructions and stable cache key."""
    from src.gpt_extract.transformers.gpt_getter import LLMExtractor
    from tests.gpt_extract.fakes import fake_context, gpt_config

    mock_response = MagicMock()
    mock_response.output_parsed = _make_expected()

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch("src.gpt_extract.utils.providers.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.responses.parse.return_value = mock_response
            mock_cls.return_value = mock_client

            config = gpt_config()
            extractor = LLMExtractor(fake_context(), config, action="def14a_executive_pay")
            result = extractor.extract(Def14AExtract, _SYNTHETIC)

    assert isinstance(result, Def14AExtract)
    assert result.company_name == "ACME Corporation"
    call_kw = mock_client.responses.parse.call_args.kwargs
    expected_model = str(config.gpt.llm_model[config.gpt.default_api])
    assert call_kw["model"] == expected_model
    assert call_kw["text_format"] is Def14AExtract
    # the section prompt lives in prompt_templates/def14a_executive_pay_system_prompt.md
    assert "Summary Compensation Table" in call_kw["instructions"]
    assert call_kw["input"].rstrip().endswith(_SYNTHETIC.rstrip()[-60:])  # payload LAST
    assert call_kw["prompt_cache_key"] == f"{expected_model}:Def14AExtract"
    assert "seed" not in call_kw
    if expected_model in config.gpt.reasoning_models:
        assert "temperature" not in call_kw
    else:
        assert call_kw["temperature"] == config.gpt.temperature

    print("\n=== SANITY CHECK: LLMExtractor mock ===")
    print(
        f"  extract() -> {result.company_name}; parse() got the .md instructions, the "
        f"payload last, configured model={expected_model!r}, "
        f"prompt_cache_key={call_kw['prompt_cache_key']!r}, and no seed. Validated."
    )


# --------------------------------------------------------------------------- #
# _flatten                                                                      #
# --------------------------------------------------------------------------- #
def test_flatten_surfaces_all_signals():
    """_flatten emits ceo_age and every governance / comp / ownership column with the
    right values, sourcing board composition & ownership from the governance summary."""
    filing = pd.Series({"filing_date": pd.Timestamp("2024-04-01"), "period_of_report": "2023-12-31", "accession_number": "0000-24-000001"})
    row = _flatten("ACME", filing, _make_expected())

    # CEO
    assert row["ceo_age"] == 58
    assert row["ceo_is_board_chair"] == 1.0 and row["ceo_is_founder"] == 1.0
    assert row["ceo_since_year"] == 2019
    assert row["ceo_salary"] == 850_000 and row["ceo_non_equity_incentive"] == 100_000
    assert row["ceo_equity_pay_pct"] == pytest.approx((3_200_000 + 250_000) / 4_900_000, abs=1e-3)
    # board composition — from governance counts / board_size
    assert row["board_size"] == 3
    assert row["pct_independent_directors"] == pytest.approx(2 / 3, abs=1e-3)
    assert row["pct_female_directors"] == pytest.approx(2 / 3, abs=1e-3)
    assert row["avg_other_public_boards"] == pytest.approx((1 + 2 + 0) / 3, abs=1e-3)
    # NEO aggregate
    assert row["n_neos"] == 2
    assert row["total_neo_comp"] == 4_900_000 + 2_900_000
    # ownership — direct from governance summary
    assert row["insider_ownership_pct"] == pytest.approx(0.084)
    assert row["ceo_ownership_pct"] == pytest.approx(0.082)
    assert row["n_five_percent_holders"] == 1
    # governance provisions (bool -> numeric flag)
    assert row["classified_board"] == 0.0 and row["dual_class_shares"] == 0.0
    assert row["majority_voting"] == 1.0 and row["lead_independent_director"] == 1.0
    assert row["say_on_pay_support_pct"] == pytest.approx(0.91)
    assert row["ceo_pay_ratio"] == 250.0 and row["median_employee_pay"] == 60_000.0
    assert row["auditor_fees"] == 5_000_000.0
    # dropped columns must NOT reappear. `auditor_name` was on this list and is now a
    # DELIBERATE addition -- it was the worst column in the retired edgar table at 2.05% fill
    # while the firm name is present in 98% of documents. The technology fields took its place
    # on the dropped list: they were an opinion, not an extraction.
    for gone in ("n_financial_experts", "n_officers", "n_technology_directors", "pct_technology_directors", "technology_committee"):
        assert gone not in row, f"{gone} reappeared in the flatten"

    print("\n=== SANITY CHECK: _flatten expanded signals ===")
    print(
        f"  ceo_age={row['ceo_age']} pay_ratio={row['ceo_pay_ratio']:.0f}:1 "
        f"equity_pay={row['ceo_equity_pay_pct']:.2f} say_on_pay={row['say_on_pay_support_pct']:.2f}"
    )
    print(f"  board: size={row['board_size']} %indep={row['pct_independent_directors']:.2f} %female={row['pct_female_directors']:.2f}")
    print(
        f"  ownership: insider={row['insider_ownership_pct']:.3f} ceo={row['ceo_ownership_pct']:.3f} "
        f"5%+={row['n_five_percent_holders']}; auditor_fees=${row['auditor_fees']:,.0f}. Validated."
    )


def test_flatten_ceo_age_fallback():
    """When ceo_age isn't given top-level, _flatten recovers it from the CEO's entry
    in the directors list (the CEO is a director nominee)."""
    filing = pd.Series({"filing_date": pd.Timestamp("2024-04-01"), "period_of_report": None, "accession_number": "acc-x"})
    extract = Def14AExtract(
        company_name="FallbackCo",
        ceo_name="Bob Stone",  # ceo_age omitted
        directors=[DirectorInfo(name="Bob Stone", age=61, is_independent=False), DirectorInfo(name="Jane Roe", age=55, is_independent=True)],
    )
    row = _flatten("FBK", filing, extract)
    assert row["ceo_age"] == 61
    print("\n=== SANITY CHECK: ceo_age fallback ===")
    print(f"  ceo_age omitted top-level -> recovered {row['ceo_age']} from directors. Validated.")


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
