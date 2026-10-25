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
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore


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
    result = postprocess_section(section, answer, stamp, {}, {})
    row = result.frames[Tables.def14a_llm].iloc[0]
    assert row["auditor_fees"] == pytest.approx(57.6 * factor)
    assert row["auditor_fees_prior"] == 41 * factor
    assert row["audit_fees_tax"] == 2 * factor
    again = postprocess_section(section, answer, stamp, {}, {})
    pd.testing.assert_frame_equal(result.frames[Tables.def14a_llm], again.frames[Tables.def14a_llm])
    print(f"SANITY: {scale} converts once; all six fee fields keep their category/year and replay is identical.")


def test_partial_fee_categories_do_not_invent_total_or_convert_foreign_money(stamp: dict[str, object]) -> None:
    section = next(section for section in SECTIONS if section.key == "audit")
    partial = AuditExtract(currency="USD", amount_scale="units", audit_fees_audit=50, audit_fees_tax=2)
    row = postprocess_section(section, partial, stamp, {}, {}).frames[Tables.def14a_llm].iloc[0]
    assert "auditor_fees" not in row
    foreign = AuditExtract(currency="EUR", amount_scale="millions", auditor_name="Example LLP", auditor_fees=10)
    row = postprocess_section(section, foreign, stamp, {}, {}).frames[Tables.def14a_llm].iloc[0]
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
    result = postprocess_section(section, answer, stamp, {}, {"ceo_name_proxy": "Jane Roe"})
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
    result = postprocess_section(section, answer, stamp, {}, {})
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
    result = postprocess_section(section, answer, stamp, {}, {"ceo_name_proxy": "Jane Roe"})
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
    result = postprocess_section(section, answer, stamp, {}, {})
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
    result = postprocess_section(section, answer, stamp, {section.table: saved}, {})
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
    processed = postprocess_section(section, answer, stamp, {}, {})
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
    processed = postprocess_section(section, answer, stamp, {}, {"auditor_fees": 12_000_000})
    assert processed.frames[Tables.def14a_llm].iloc[0]["auditor_fees"] == 12_000_000
    facade = Def14AExtract.model_validate_json(refresh_archive(raw, section, processed.legacy))
    assert facade.governance is not None and facade.governance.auditor_fees_usd == 12_000_000
    assert facade.governance.auditor_fees_prior_usd == 41_000_000
    print("SANITY: a partial repair fills the missing prior fee while both the saved parent and compatibility facade retain its valid current fee.")


@pytest.fixture
def ceo_facts(stamp: dict[str, object]) -> dict[Table, pd.DataFrame]:
    components = dict(bonus=0, stock_awards=25, option_awards=25, non_equity_incentive=0, pension_change=0, other_compensation=0)
    return {
        Tables.def14a_executive_comp: pd.DataFrame(
            [
                {**stamp, **components, "name": "Alice Adams", "title": "CEO", "fiscal_year": 2024, "salary": 50, "total": 100, "retry": 2},
                {
                    **stamp,
                    "name": "Bob Brown",
                    "title": "CEO",
                    "fiscal_year": 2024,
                    "salary": 200,
                    "bonus": None,
                    "stock_awards": None,
                    "option_awards": None,
                    "non_equity_incentive": None,
                    "pension_change": None,
                    "other_compensation": None,
                    "total": None,
                    "retry": 2,
                },
            ]
        ),
        Tables.def14a_ownership: pd.DataFrame(
            [
                {
                    **stamp,
                    "holder_name": "Alice Adams",
                    "holder_type": "director_officer",
                    "shares": 100,
                    "percent_of_class": 0.1,
                    "percent_of_voting_power": 0.15,
                    "retry": 2,
                },
                {
                    **stamp,
                    "holder_name": "Bob Brown",
                    "holder_type": "director_officer",
                    "shares": 200,
                    "percent_of_class": None,
                    "percent_of_voting_power": 0,
                    "retry": 2,
                },
            ]
        ),
    }


def test_matching_partial_ceo_clears_other_person_copies_and_ignores_future_rows(
    stamp: dict[str, object], ceo_facts: dict[Table, pd.DataFrame]
) -> None:
    section = next(section for section in SECTIONS if section.key == "governance")
    future = (
        ceo_facts[Tables.def14a_executive_comp].iloc[[1]].assign(accession_number="future", as_of=pd.Timestamp("2026-04-01"), salary=999, total=999)
    )
    ceo_facts[Tables.def14a_executive_comp] = pd.concat([ceo_facts[Tables.def14a_executive_comp], future], ignore_index=True)
    before = {table: frame.copy(deep=True) for table, frame in ceo_facts.items()}
    parent = dict(ceo_total_comp=100, ceo_equity_pay_pct=0.5, ceo_ownership_pct=0.1, ceo_voting_pct=0.15, auditor_fees=777, retry=2)
    answer = GovernanceExtract.model_validate(dict(ceo_name="Bob Brown", percent_unit="fraction"))
    processed = postprocess_section(section, answer, stamp, ceo_facts, parent)
    row = processed.frames[Tables.def14a_llm].iloc[0]
    assert row["ceo_name_proxy"] == "Bob Brown" and row["ceo_salary"] == 200
    assert pd.isna(row["ceo_total_comp"]) and pd.isna(row["ceo_equity_pay_pct"]) and pd.isna(row["ceo_ownership_pct"])
    assert row["ceo_voting_pct"] == 0
    assert "auditor_fees" not in row and "retry" not in row
    assert set(processed.frames) == {Tables.def14a_directors, Tables.def14a_llm}
    for table, frame in before.items():
        pd.testing.assert_frame_equal(ceo_facts[table], frame)
    print(
        "SANITY: a matching partial CEO clears unsupported other-person copies, retains a real zero, ignores future facts and leaves child rows/counters untouched."
    )


@pytest.mark.parametrize("legacy_replay", [False, True])
def test_same_ceo_partial_children_preserve_valid_copies_and_refresh_supported_values(
    stamp: dict[str, object], ceo_facts: dict[Table, pd.DataFrame], sqlite_store: DataStore, legacy_replay: bool
) -> None:
    section = next(section for section in SECTIONS if section.key == "governance")
    ceo_facts[Tables.def14a_executive_comp].loc[1, "bonus"] = 0
    before = {table: frame.copy(deep=True) for table, frame in ceo_facts.items()}
    parent = dict(
        ceo_name_proxy="BOB BROWN",
        ceo_salary=None,
        ceo_bonus=12,
        ceo_total_comp=200,
        ceo_equity_pay_pct=0.5,
        ceo_ownership_pct=0.2,
        ceo_voting_pct=0.25,
        auditor_fees=777,
        retry=2,
        def14a_json='{"original":"paid response"}',
    )
    previous = parent.copy()
    keys = {column: stamp[column] for column in (*Tables.def14a_llm.pk, "as_of")}
    sqlite_store.save(Tables.def14a_llm, pd.DataFrame([{**keys, **parent}]))
    answer = Def14AExtract(ceo_name="Bob Brown") if legacy_replay else GovernanceExtract(percent_unit="fraction")
    raw = answer.model_dump_json()
    processed = postprocess_section(section, answer, stamp, ceo_facts, parent)
    row = processed.frames[Tables.def14a_llm].iloc[0]
    assert row["ceo_total_comp"] == 200 and row["ceo_equity_pay_pct"] == 0.5 and row["ceo_ownership_pct"] == 0.2
    assert row["ceo_salary"] == 200 and row["ceo_bonus"] == 0 and row["ceo_voting_pct"] == 0
    assert pd.isna(row["ceo_stock_awards"])
    assert "auditor_fees" not in row and "retry" not in row and "def14a_json" not in row
    assert set(processed.frames) == {Tables.def14a_directors, Tables.def14a_llm}
    sqlite_store.save(Tables.def14a_llm, processed.frames[Tables.def14a_llm])
    stored = sqlite_store.load(Tables.def14a_llm, where=dict(ticker=stamp["ticker"], accession_number=stamp["accession_number"])).iloc[0]
    assert (stored["ceo_total_comp"], stored["ceo_equity_pay_pct"], stored["ceo_ownership_pct"]) == (200, 0.5, 0.2)
    assert stored["ceo_salary"] == 200 and stored["ceo_bonus"] == 0 and stored["ceo_voting_pct"] == 0
    assert stored["as_of"] == stamp["as_of"] and stored["auditor_fees"] == 777 and stored["retry"] == 2
    assert stored["def14a_json"] == parent["def14a_json"] and parent == previous and answer.model_dump_json() == raw
    for table, frame in before.items():
        pd.testing.assert_frame_equal(ceo_facts[table], frame)
    print(
        "SANITY: partial facts for the retained same CEO preserve valid copies through a real save, fill holes and refresh zeros without changing archives/counters."
    )


def test_conflicting_core_answer_derives_against_the_retained_ceo(stamp: dict[str, object], ceo_facts: dict[Table, pd.DataFrame]) -> None:
    section = next(section for section in SECTIONS if section.key == "governance")
    parent = dict(ceo_name_proxy="Alice Adams", ceo_age=61, ceo_total_comp=100, ceo_equity_pay_pct=0.5)
    answer = GovernanceExtract.model_validate(dict(ceo_name="Bob Brown", ceo_age=50, percent_unit="fraction"))
    row = postprocess_section(section, answer, stamp, ceo_facts, parent).frames[Tables.def14a_llm].iloc[0]
    assert row["ceo_name_proxy"] == "Alice Adams" and row["ceo_age"] == 61
    assert (row["ceo_salary"], row["ceo_total_comp"], row["ceo_equity_pay_pct"]) == (50, 100, 0.5)
    assert (row["ceo_ownership_pct"], row["ceo_voting_pct"]) == (0.1, 0.15)
    print("SANITY: a conflicting later raw name cannot reattribute pay/ownership away from the CEO retained by the parent merge.")


@pytest.mark.parametrize("state", ["absent", "marker", "other_filing", "other_date", "unmatched"])
def test_missing_or_unmatched_child_facts_do_not_clear_existing_fallbacks(stamp: dict[str, object], state: str) -> None:
    section = next(section for section in SECTIONS if section.key == "governance")
    pay = {**stamp, "name": "Bob Brown", "title": "CEO", "fiscal_year": 2024, "salary": 999, "total": 999, "retry": 2}
    holder = {**stamp, "holder_name": "Bob Brown", "holder_type": "director_officer", "percent_of_class": 0.9, "retry": 2}
    if state == "marker":
        pay.update(name="_empty", fiscal_year=-1)
        holder.update(holder_name="_empty", holder_type="_empty")
    if state == "other_filing":
        pay["accession_number"] = holder["accession_number"] = "different"
    if state == "other_date":
        pay["as_of"] = holder["as_of"] = pd.Timestamp("2026-04-01")
    if state == "unmatched":
        pay["name"] = holder["holder_name"] = "Alice Adams"
    saved = {} if state == "absent" else {Tables.def14a_executive_comp: pd.DataFrame([pay]), Tables.def14a_ownership: pd.DataFrame([holder])}
    parent = dict(ceo_total_comp=100, ceo_ownership_pct=0.1)
    answer = GovernanceExtract.model_validate(dict(ceo_name="Bob Brown", percent_unit="fraction"))
    row = postprocess_section(section, answer, stamp, saved, parent).frames[Tables.def14a_llm].iloc[0]
    assert "ceo_total_comp" not in row and "ceo_ownership_pct" not in row and "ceo_voting_pct" not in row
    print(f"SANITY: {state} child facts do not justify clearing a saved fallback; only matching facts at this filing clock can refresh it.")
