"""Known-truth units, reporting-year, partial-answer and archive replay checks."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data_extract.utils.schemas.def14a_schema import (
    AuditExtract,
    Def14AExtract,
    DirectorPayExtract,
    ExecutivePayExtract,
    GovernanceExtract,
    GovernanceProfile,
    OwnershipExtract,
)
from src.data_extract.utils.structure.def14a.postprocess import (
    append_answer,
    archived_answers,
    postprocess_section,
    refresh_archive,
)
from src.data_extract.utils.structure.def14a.retry import SECTIONS
from src.data_store.schema import Tables


@pytest.fixture
def stamp() -> dict[str, object]:
    return {"ticker": "AAA", "cik": "1", "accession_number": "a", "as_of": pd.Timestamp("2025-04-01")}


@pytest.mark.parametrize("scale,factor", [("units", 1), ("thousands", 1000), ("millions", 1000000)])
def test_audit_units_are_explicit_and_current_prior_stay_separate(stamp: dict[str, object], scale: str, factor: int) -> None:
    section = next(section for section in SECTIONS if section.key == "audit")
    answer = AuditExtract.model_validate(
        {
            "currency": "USD",
            "amount_scale": scale,
            "auditor_name": "Example LLP",
            "audit_fees_audit": 50,
            "audit_fees_audit_related": 5,
            "audit_fees_tax": 2,
            "audit_fees_other": 0.6,
            "auditor_fees_prior": 41,
        }
    )
    result = postprocess_section(section, answer, stamp, None, {})
    row = result.frames[Tables.def14a_llm].iloc[0]
    assert row["auditor_fees"] == pytest.approx(57.6 * factor)
    assert row["auditor_fees_prior"] == 41 * factor
    assert row["audit_fees_tax"] == 2 * factor
    again = postprocess_section(section, answer, stamp, None, {})
    pd.testing.assert_frame_equal(result.frames[Tables.def14a_llm], again.frames[Tables.def14a_llm])
    print(f"SANITY: {scale} converts once; all six fee fields keep their category/year and replay is identical.")


def test_partial_fee_categories_do_not_invent_total_or_convert_foreign_money(stamp: dict[str, object]) -> None:
    section = next(section for section in SECTIONS if section.key == "audit")
    partial = AuditExtract(currency="USD", amount_scale="units", audit_fees_audit=50, audit_fees_tax=2)
    row = postprocess_section(section, partial, stamp, None, {}).frames[Tables.def14a_llm].iloc[0]
    assert "auditor_fees" not in row
    foreign = AuditExtract(currency="EUR", amount_scale="millions", auditor_name="Example LLP", auditor_fees=10)
    row = postprocess_section(section, foreign, stamp, None, {}).frames[Tables.def14a_llm].iloc[0]
    assert "auditor_fees" not in row and row["auditor_name"] == "Example LLP"
    print("SANITY: incomplete categories do not create totals, and unsupported currency remains unknown while identity survives.")


def test_pay_keeps_every_year_zero_negative_and_latest_year_summaries(stamp: dict[str, object]) -> None:
    section = next(section for section in SECTIONS if section.key == "executive_pay")
    answer = ExecutivePayExtract.model_validate(
        {
            "currency": "USD",
            "amount_scale": "thousands",
            "compensation": [
                {"name": "Jane Roe", "title": "CEO", "fiscal_year": 2024, "salary": 1000, "bonus": 0, "pension_change": -20, "total": 1980},
                {"name": "Jane Roe", "title": "CEO", "fiscal_year": 2023, "salary": 900, "total": 1800},
            ],
        }
    )
    result = postprocess_section(section, answer, stamp, None, {"ceo_name_proxy": "Jane Roe"})
    df_pay = result.frames[Tables.def14a_executive_comp].set_index("fiscal_year")
    assert set(df_pay.index) == {2023, 2024}
    assert df_pay.loc[2024, "bonus"] == 0 and df_pay.loc[2024, "pension_change"] == -20_000
    parent = result.frames[Tables.def14a_llm].iloc[0]
    assert parent["ceo_total_comp"] == 1_980_000 and parent["n_neos"] == 1
    assert parent["total_neo_comp"] == 1_980_000
    assert "retry" not in parent
    print("SANITY: all pay years survive; explicit zero/negative values remain real, and CEO/NEO summaries use only the latest year.")


def test_option_quantity_does_not_become_dollar_awards(stamp: dict[str, object]) -> None:
    section = next(section for section in SECTIONS if section.key == "executive_pay")
    answer = ExecutivePayExtract.model_validate(
        {
            "currency": "USD",
            "amount_scale": "units",
            "compensation": [{"name": "Jane Roe", "title": "CEO", "fiscal_year": 1995, "salary": 100000, "option_count": 25000}],
        }
    )
    result = postprocess_section(section, answer, stamp, None, {})
    row = result.frames[Tables.def14a_executive_comp].iloc[0]
    assert pd.isna(row["option_awards"]) and pd.isna(row["total"])
    print("SANITY: a historical option count is never priced as a monetary award or a fabricated modern total.")


def test_ownership_normalizes_percentages_and_keeps_voting_separate(stamp: dict[str, object]) -> None:
    section = next(section for section in SECTIONS if section.key == "ownership")
    answer = OwnershipExtract.model_validate(
        {
            "percent_unit": "percent",
            "shares_scale": "thousands",
            "insider_shares": 25,
            "insider_ownership_pct": 3.5,
            "insider_voting_pct": 55,
            "holders": [
                {"holder_name": "Jane Roe", "holder_type": "director_officer", "shares": 10, "percent_of_class": 9.63, "percent_of_voting_power": 60}
            ],
        }
    )
    result = postprocess_section(section, answer, stamp, None, {"ceo_name_proxy": "Jane Roe"})
    holder = result.frames[Tables.def14a_ownership].iloc[0]
    parent = result.frames[Tables.def14a_llm].iloc[0]
    assert holder["shares"] == 10_000 and holder["percent_of_class"] == pytest.approx(0.0963)
    assert holder["percent_of_voting_power"] == pytest.approx(0.6)
    assert parent["insider_ownership_pct"] == pytest.approx(0.035)
    assert parent["insider_voting_pct"] == pytest.approx(0.55) and parent["insider_shares"] == 25_000
    assert parent["ceo_ownership_pct"] == pytest.approx(0.0963)
    print("SANITY: source shares become counts and percentages fractions; economic and voting legs stay distinct.")


def test_board_ceo_age_tenure_and_pay_ratio_use_only_this_filing(stamp: dict[str, object]) -> None:
    section = next(section for section in SECTIONS if section.key == "governance")
    answer = GovernanceExtract.model_validate(
        {
            "currency": "USD",
            "amount_scale": "thousands",
            "percent_unit": "percent",
            "ceo_name": "Jane Roe",
            "directors": [{"name": "Jane Roe", "age": 61, "director_since_year": 2010}],
            "ceo_pay_ratio": 200,
            "median_employee_pay": 50,
            "say_on_pay_support_pct": 31,
        }
    )
    result = postprocess_section(section, answer, stamp, None, {})
    row = result.frames[Tables.def14a_directors].iloc[0]
    parent = result.frames[Tables.def14a_llm].iloc[0]
    assert row["tenure_years"] == 15 and parent["ceo_age"] == 61
    assert parent["median_employee_pay"] == 50_000 and parent["ceo_pay_ratio"] == 200
    assert parent["say_on_pay_support_pct"] == pytest.approx(0.31)
    assert "retry" not in parent
    print("SANITY: CEO age falls back to its own roster, tenure uses this filing date, and a real failed vote/pay-ratio disclosure is preserved.")


def test_free_replay_keeps_valid_cells_people_and_retry_two(stamp: dict[str, object]) -> None:
    section = next(section for section in SECTIONS if section.key == "director_pay")
    saved = pd.DataFrame(
        [
            {**stamp, "name": "Anne Smith", "fees_earned": None, "total": 90000, "retry": 2},
            {**stamp, "name": "Bob Jones", "fees_earned": 50000, "total": 50000, "retry": 2},
        ]
    )
    answer = DirectorPayExtract.model_validate(
        {
            "currency": "USD",
            "amount_scale": "thousands",
            "compensation": [
                {"name": "Anne Smith", "fiscal_year": 2024, "fees_earned": 60},
                {"name": "Carol Roe", "fiscal_year": 2024, "fees_earned": 70},
            ],
        }
    )
    result = postprocess_section(section, answer, stamp, saved, {})
    df_pay = result.frames[Tables.def14a_director_comp].set_index("name")
    assert set(df_pay.index) == {"Anne Smith", "Bob Jones", "Carol Roe"}
    assert df_pay.loc["Anne Smith", "total"] == 90000 and df_pay.loc["Anne Smith", "fees_earned"] == 60000
    assert df_pay.loc["Bob Jones", "total"] == 50000 and set(df_pay["retry"]) == {2}
    assert result.frames[Tables.def14a_llm].iloc[0]["n_director_comp_rows"] == 3
    print("SANITY: partial repair fills missing cells and adds people without losing valid rows; all retained/new rows preserve retry=2.")


def test_paid_archive_keeps_original_raw_answers_and_legacy_facade(stamp: dict[str, object]) -> None:
    section = next(section for section in SECTIONS if section.key == "audit")
    original = Def14AExtract(ceo_name="Jane Roe", ceo_age=61, governance=GovernanceProfile(auditor_name="Old LLP")).model_dump_json()
    answer = AuditExtract(currency="USD", amount_scale="millions", auditor_fees=57.6, auditor_fees_prior=41)
    raw = append_answer(original, section, answer, "Audit fees (in millions): 57.6; prior 41")
    processed = postprocess_section(section, answer, stamp, None, {})
    final = refresh_archive(raw, section, processed.legacy)
    document = json.loads(final)
    assert document["_def14a_split"]["legacy_json"] == original
    assert document["_def14a_split"]["responses"]["audit"][0]["response"]["auditor_fees"] == 57.6
    facade = Def14AExtract.model_validate_json(final)
    assert facade.ceo_age == 61 and facade.ceo_name == "Jane Roe"
    assert facade.governance is not None and facade.governance.auditor_fees_usd == 57_600_000
    recovered = archived_answers(final)
    assert len(recovered["audit"]) == 2 and len(recovered["governance"]) == 1
    assert recovered["audit"][-1].model_dump() == answer.model_dump()
    print("SANITY: the original paid JSON stays byte-exact, new source-scale answers remain raw, and old JSON readers still see a usable facade.")


def test_partial_archive_facade_does_not_replace_valid_existing_fees(stamp: dict[str, object]) -> None:
    section = next(section for section in SECTIONS if section.key == "audit")
    original = Def14AExtract(governance=GovernanceProfile(auditor_fees_usd=12_000_000)).model_dump_json()
    answer = AuditExtract(currency="USD", amount_scale="millions", auditor_fees=57.6, auditor_fees_prior=41)
    raw = append_answer(original, section, answer, "reported fee repair")
    processed = postprocess_section(section, answer, stamp, None, {"auditor_fees": 12_000_000})
    assert processed.frames[Tables.def14a_llm].iloc[0]["auditor_fees"] == 12_000_000
    facade = Def14AExtract.model_validate_json(refresh_archive(raw, section, processed.legacy))
    assert facade.governance is not None and facade.governance.auditor_fees_usd == 12_000_000
    assert facade.governance.auditor_fees_prior_usd == 41_000_000
    print("SANITY: a partial repair fills the missing prior fee while both the saved parent and compatibility facade retain its valid current fee.")
