"""Recorded filing layouts exercise source-backed daily repair without a provider."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.structure.def14a import fetch
from src.data_extract.utils.structure.def14a.retry import SECTIONS
from src.data_extract.utils.structure.def14a.work import FilingWork, requests_for
from src.data_store.schema import Tables
from tests.data_extract.structure.test_def14a_retry_daily import PAYLOADS, _attempts, _complete_answers
from tests.data_extract.structure.test_def14a_retry_daily import daily as daily

# Exact header/row excerpts from benchmark50/calls/*.req.json, identified in the F-008 diagnosis.
SCTS = {
    "AAPL1996": (
        "Gilbert F. Amelio",
        1996,
        655061,
        "SUMMARY COMPENSATION TABLE\nNAME AND FISCAL OTHER ANNUAL RESTRICTED ALL OTHER\n"
        "PRINCIPAL POSITION YEAR SALARY($) BONUS($) COMPENSATION($) STOCK($) OPTIONS(#) COMPENSATION($)\n"
        "Gilbert F. Amelio........ 1996 655,061 2,334,000(2) -- 3,830,580(3) 1,000,000 3,060(4)",
    ),
    "GE1996": (
        "John F. Welch, Jr.",
        1995,
        2000000,
        "SUMMARY COMPENSATION TABLE\nOTHER ANNUAL TOTAL ANNUAL\n"
        "NAME AND PRINCIPAL POSITION YEAR SALARY BONUS COMPENSATION COMPENSATION\n"
        "JOHN F. WELCH, JR. 1995 $2,000,000 $3,250,000 $71,228 $5,321,228",
    ),
    "EOG1996": (
        "Forrest E. Hoglund",
        1995,
        570000,
        "SUMMARY COMPENSATION TABLE\nNAME & PRINCIPAL SALARY BONUS COMPENSATION AWARDS OPTIONS/ PAYOUTS\n"
        "POSITION YEAR ($) ($) ($)(2) ($)(3) SARS (#) ($) ($)(4)\n"
        "Forrest E. Hoglund..... 1995 $570,000 $ 69,721 $ 793",
    ),
    "JPM1996": (
        "Walter V. Shipley",
        1995,
        850000,
        "SUMMARY COMPENSATION TABLE(1)\nName and Principal Position Year Salary($)(2) Bonus($)(2) ($)(3) (#) ($)(4)\n"
        "Walter V. Shipley 1995 $850,000 $2,400,000 $ 946,250 120,000 $1,267,500",
    ),
    "CAT1996": (
        "Donald V. Fites",
        1995,
        1000000,
        "Summary Compensation Table\nName and Securities LTIP\nPrincipal Other Annual Underlying Payouts All Other\n"
        "Position Year Salary Bonus/2/ Compensation Options/3/ ($)/4/ Compensation/1/\n"
        "D. V. Fites, 1995 $1,000,000 $564,000 $4,420 75,000 $900,000 $48,000",
    ),
    "JNJ1996": (
        "Ralph S. Larsen",
        1995,
        1005000,
        "SUMMARY COMPENSATION TABLE\nPRINCIPAL COMPEN- OPTIONS COMPEN-\n"
        "POSITION YEAR SALARY($) BONUS($) SATION($) (#) SATION($)\n(1) (2) (3)\n"
        "R. S. Larsen ....... 1995 $1,005,000 $1,135,931 $534,456 25,500 $ 6,750",
    ),
    "JNJ2007": (
        "William C. Weldon",
        2006,
        1659231,
        "A Name and Principal Position\tB Year\tB Year\tB Year\tC Salary($)\tC Salary($)\tC Salary($)\t"
        "D Stock Awards($)\tD Stock Awards($)\tD Stock Awards($)\n"
        "William C. Weldon Chairman/CEO\t\t2006\t\t$\t1,659,231\t\t$\t2,041,054\t",
    ),
    "PG2015": (
        "A. G. Lafley",
        2015,
        2000000,
        "FY 2014-15 Summary Compensation Table Name and Principal Position\tFY 2014-15 Summary Compensation Table Year\t"
        "FY 2014-15 Summary Compensation Table Salary\tFY 2014-15 Summary Compensation Table Salary\t"
        "FY 2014-15 Summary Compensation Table Bonus\tFY 2014-15 Summary Compensation Table Bonus\n"
        "A. G. Lafley\t2014-15\t\t2,000,000\t\t3,290,000",
    ),
}
AUDIT = "TYPES\nOF FEES (IN\nMILLIONS)\nAUDIT\nAUDIT- RELATED\nTAX\nALL OTHER\nTOTAL\n2024\n$17.9\n$3.0\n$0.0\n$0.0\n$20.9\n2023\n$46.2\n$24.3\n$0.4\n$0.2\n$71.1"
FEES = dict(auditor_fees=20.9, audit_fees_audit=17.9, audit_fees_audit_related=3.0, audit_fees_tax=0.0, audit_fees_other=0.0, auditor_fees_prior=71.1)


def _work(domain: str, rows: list[dict[str, Any]], parent: dict[str, Any] | None = None) -> FilingWork:
    saved = {section.table: pd.DataFrame() for section in SECTIONS}
    section = next(section for section in SECTIONS if section.key == domain)
    saved[section.table] = pd.DataFrame(rows)
    return FilingWork(pd.Series(), {}, saved, parent or {"auditor_name": "KPMG LLP"}, dict.fromkeys((s.key for s in SECTIONS), 1), pending={domain})


@pytest.mark.parametrize("case", list(SCTS))
def test_recorded_salary_layout_selects_only_its_domain(case: str) -> None:
    name, year, _, payload = SCTS[case]
    work = _work("executive_pay", [dict(name=name, fiscal_year=year, salary=None, stock_awards=0)])
    requests = requests_for(work, {"executive_pay": payload})
    assert [section.key for section, _ in requests] == ["executive_pay"]
    work.attempts["executive_pay"] = 2
    assert requests_for(work, {"executive_pay": payload}) == []
    print(f"SANITY: {case} disclosed salary selects executive pay once; exhausted 2 is excluded.")


@pytest.mark.parametrize("field", list(FEES))
def test_recorded_transposed_audit_matches_each_missing_fee(field: str) -> None:
    parent = {"auditor_name": "KPMG LLP", **FEES, field: None}
    work = _work("audit", [], parent)
    assert [section.key for section, _ in requests_for(work, {"audit": AUDIT})] == ["audit"]
    work.parent[field] = FEES[field]
    assert requests_for(work, {"audit": AUDIT}) == []
    print(f"SANITY: transposed matrix identifies {field}; existing disclosed zero and fee values stay complete.")


@pytest.mark.parametrize("case", ["AAPL1996", "JNJ2007", "PG2015", "audit"])
def test_recorded_layout_repairs_once_through_daily_store(daily, monkeypatch, case: str) -> None:
    context, calls, answers, _ = daily
    answers.update(_complete_answers())
    domain = "audit" if case == "audit" else "executive_pay"
    schema = "AuditExtract" if case == "audit" else "ExecutivePayExtract"
    payloads = {**PAYLOADS, domain: AUDIT if case == "audit" else SCTS[case][3]}
    monkeypatch.setattr(fetch, "prepare_def14a_payloads", lambda *_: payloads)
    if case == "audit":
        answers[schema].update({field: None for field in FEES})
    else:
        name, year, salary, _ = SCTS[case]
        answers["GovernanceExtract"]["ceo_name"] = name
        answers[schema]["compensation"][0].update(name=name, fiscal_year=year, salary=None)
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    assert len(calls) == 5 and all(value == [1] for value in _attempts(context).values())
    before = context.store.load(Tables.def14a_llm).iloc[0]
    if case == "audit":
        answers[schema].update(**FEES, amount_scale="millions")
    else:
        answers[schema]["compensation"][0]["salary"] = salary
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    assert [name for name, _, _ in calls[5:]] == [schema]
    owner = next(section.table for section in SECTIONS if section.key == domain)
    assert _attempts(context) == {section.table.name: [2 if section.table is owner else 1] for section in SECTIONS}
    after = context.store.load(Tables.def14a_llm).iloc[0]
    assert after.ceo_age == before.ceo_age == 54
    if case == "audit":
        assert all(after[field] == value * 1_000_000 for field, value in FEES.items())
        assert after.ceo_total_comp == before.ceo_total_comp
    else:
        assert after.ceo_salary == salary and after.auditor_fees == before.auditor_fees
        assert context.store.load(Tables.def14a_executive_comp).iloc[0].total == 100
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"], full=True)
    assert len(calls) == 6
    print(f"SANITY: {case} daily persisted only {domain} 1 -> 2; valid facts survive; later normal/full spend zero.")


@pytest.mark.parametrize("value", ["--", "N/A", "1,000,000 options"])
def test_plain_salary_requires_a_disclosed_amount(value: str) -> None:
    name, year, _, payload = SCTS["AAPL1996"]
    payload = payload.replace("655,061", value)
    work = _work("executive_pay", [dict(name=name, fiscal_year=year, salary=None)])
    assert requests_for(work, {"executive_pay": payload}) == []
    print(f"SANITY: salary {value!r} does not turn option counts, headings or years into dollars.")


@pytest.mark.parametrize("case", ["AAPL1996", "JNJ2007", "PG2015"])
def test_other_person_or_year_does_not_recall_saved_salary(case: str) -> None:
    name, year, salary, payload = SCTS[case]
    rows = [dict(name=name, fiscal_year=year, salary=salary, stock_awards=0, bonus=0), dict(name="Absent Person", fiscal_year=year, salary=None)]
    rows.append(dict(name=name, fiscal_year=year - 1, salary=None))
    assert requests_for(_work("executive_pay", rows), {"executive_pay": payload}) == []
    print(f"SANITY: {case} complete matching salary cannot support a different person's/year's missing salary.")


@pytest.mark.parametrize(
    "payload", [AUDIT.replace("$0.0", "--"), AUDIT.replace("$0.0", "N/A"), AUDIT.split("2024")[0], AUDIT.replace("MILLIONS)", "POLICY)")]
)
def test_transposed_audit_requires_year_units_and_the_missing_category(payload: str) -> None:
    work = _work("audit", [], {"auditor_name": "KPMG LLP", **FEES, "audit_fees_tax": None, "audit_fees_other": None})
    assert requests_for(work, {"audit": payload}) == []
    print("SANITY: audit headings, years, unsupported units and category dashes/N/A do not establish missing dollars.")


def test_transposed_prior_total_absence_does_not_use_current_amounts() -> None:
    work = _work("audit", [], {"auditor_name": "KPMG LLP", **FEES, "auditor_fees_prior": None})
    assert requests_for(work, {"audit": AUDIT.replace("$71.1", "N/A")}) == []
    print("SANITY: a missing prior total cannot borrow current or other prior categories as evidence.")


@pytest.mark.parametrize("value", ["0", "-100"])
def test_plain_disclosed_zero_and_negative_salary_are_values(value: str) -> None:
    name, year, _, payload = SCTS["AAPL1996"]
    work = _work("executive_pay", [dict(name=name, fiscal_year=year, salary=None)])
    assert requests_for(work, {"executive_pay": payload.replace("655,061", value)})
    work.saved[Tables.def14a_executive_comp]["salary"] = int(value)
    assert requests_for(work, {"executive_pay": payload.replace("655,061", value)}) == []
    print(f"SANITY: salary {value} is disclosed; once saved it remains complete.")


def test_distinct_full_names_do_not_share_salary_evidence() -> None:
    payload = SCTS["AAPL1996"][3].replace("Gilbert F. Amelio", "John Smith")
    rows = [dict(name="John Smith", fiscal_year=1996, salary=None), dict(name="Jane Smith", fiscal_year=1996, salary=100)]
    work = _work("executive_pay", rows)
    assert requests_for(work, {"executive_pay": payload})
    work.saved[Tables.def14a_executive_comp]["salary"] = [100, None]
    assert requests_for(work, {"executive_pay": payload}) == []
    print("SANITY: Jane Smith's salary cannot complete John Smith or turn his complete row into her disclosure.")


def test_salary_2000_is_not_a_year_cell() -> None:
    name, year, _, payload = SCTS["AAPL1996"]
    work = _work("executive_pay", [dict(name=name, fiscal_year=year, salary=None)])
    assert requests_for(work, {"executive_pay": payload.replace("655,061", "2000")})
    print("SANITY: 2000 under Salary($) is a monetary salary, independent of its resemblance to a year.")


def test_monetary_year_suffix_is_not_a_fiscal_year_header() -> None:
    payload = "Name\tSalary this year\tBonus\nJane Example\t2000\t0"
    rows = [
        dict(
            name="Jane Example",
            fiscal_year=2024,
            salary=2000,
            bonus=0,
            stock_awards=0,
            option_awards=None,
            non_equity_incentive=0,
            pension_change=0,
            other_compensation=0,
            total=2000,
        )
    ]
    work = _work("executive_pay", rows)
    assert requests_for(work, {"executive_pay": payload}) == []
    payload = "Name\tYear\tSalary this year\tBonus\nJane Example\t2024\t2000\t0"
    assert requests_for(work, {"executive_pay": payload}) == []
    work.saved[Tables.def14a_executive_comp]["salary"] = None
    assert [section.key for section, _ in requests_for(work, {"executive_pay": payload})] == ["executive_pay"]
    print("SANITY: a salary header ending in year cannot invent a fiscal year; a true Year column preserves salary 2000 as disclosed money.")


def test_prefixed_year_does_not_price_option_counts() -> None:
    payload = "Name\tB Year\tOption Awards(#)\nJane Example\t2006\t1,000,000"
    work = _work("executive_pay", [dict(name="Jane Example", fiscal_year=2006, salary=100, option_awards=None)])
    assert requests_for(work, {"executive_pay": payload}) == []
    print("SANITY: recognizing a prefixed year never turns option quantities into monetary awards.")


def test_equivalent_transposed_fee_caption_is_supported() -> None:
    payload = AUDIT.replace("TYPES\nOF FEES", "AUDITOR FEES").replace(
        "AUDIT\nAUDIT- RELATED\nTAX\nALL OTHER\nTOTAL", "AUDIT FEES\nAUDIT - RELATED FEES\nTAX FEES\nALL OTHER FEES\nTOTAL FEES"
    )
    assert requests_for(_work("audit", []), {"audit": payload})
    print("SANITY: an equivalent caption and spaced fee labels preserve the same ordered dollar/year matrix evidence.")


@pytest.mark.parametrize(
    "payload",
    [
        "SUMMARY COMPENSATION TABLE\nName and Principal Position Year\nGilbert F. Amelio 1996",
        "SUMMARY COMPENSATION TABLE\nSalary policy\nGilbert F. Amelio has a salary policy of $1,000,000 for 1996.",
    ],
)
def test_plain_header_year_and_policy_do_not_disclose_salary(payload: str) -> None:
    work = _work("executive_pay", [dict(name="Gilbert F. Amelio", fiscal_year=1996, salary=None)])
    assert requests_for(work, {"executive_pay": payload}) == []
    print("SANITY: a year-only row or salary policy has no disclosed salary column and cannot justify repair.")


@pytest.mark.parametrize("holdings", ["1000", "1000 10"])
def test_plain_salary_evidence_stops_before_a_later_table(holdings: str) -> None:
    payload = f"SUMMARY COMPENSATION TABLE\nName Year Salary Bonus\nJane Example 2025 -- 1234\nBENEFICIAL OWNERSHIP TABLE\nName Year Shares\nJane Example 2025 {holdings}"
    work = _work("executive_pay", [dict(name="Jane Example", fiscal_year=2025, salary=None)])
    assert requests_for(work, {"executive_pay": payload}) == []
    print("SANITY: an ownership share count cannot become salary through a stale compensation header.")


def test_plain_salary_requires_a_following_compensation_cell() -> None:
    payload = "SUMMARY COMPENSATION TABLE\nName Year Salary Bonus\nJane Example 2025    1234"
    work = _work("executive_pay", [dict(name="Jane Example", fiscal_year=2025, salary=None)])
    assert requests_for(work, {"executive_pay": payload}) == []
    print("SANITY: one collapsed number cannot distinguish a disclosed salary from a blank salary and bonus.")


def test_plain_numbered_footnote_prose_still_ends_the_table() -> None:
    name, year, _, payload = SCTS["JNJ1996"]
    payload = payload.replace("(1) (2) (3)\n", "(1) (2) (3)\n(1) Salary policy discussed below.\n")
    work = _work("executive_pay", [dict(name=name, fiscal_year=year, salary=None)])
    assert requests_for(work, {"executive_pay": payload}) == []
    print("SANITY: pure note-number headers may be skipped; actual numbered footnote prose still stops salary evidence.")


def test_plain_salary_stops_at_an_uppercase_section_heading() -> None:
    payload = (
        "SUMMARY COMPENSATION TABLE\nName Year Salary Bonus\nJane Example 2025 -- 1234\nSECURITY OWNERSHIP OF MANAGEMENT\nJane Example 2025 1000 10"
    )
    work = _work("executive_pay", [dict(name="Jane Example", fiscal_year=2025, salary=None)])
    assert requests_for(work, {"executive_pay": payload}) == []
    print("SANITY: a new uppercase source heading ends the SCT even without a repeated Name header.")
