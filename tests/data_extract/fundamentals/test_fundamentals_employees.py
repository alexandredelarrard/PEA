"""Offline known-truth checks for SEC employee extraction: the source guard, the excerpt rebuilt from stored text rows
and one ticker's decisions (`_ticker_employees` over `sec_employees_text` rows parsed by the walk into a SQLite store)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import src.data_extract.utils.edgar_extract.employee_document as document_mod
import src.data_extract.utils.edgar_extract.fetch_employees_text as text_mod
import src.data_extract.utils.fundamentals.employee_guard as guard
import src.data_extract.utils.fundamentals.fundamentals_employees as mod
from src.data_extract.utils.edgar_extract import edgar_driver
from src.data_store.schema import Tables
from tests.data_extract.edgar_fixtures import FakeEmployeesLLM, identity_for, lent, sent_excerpt
from tests.data_extract.edgar_fixtures import employee_answer as answer
from tests.data_extract.fake_context import extract_config

TEXT = Tables.sec_employees_text


class Filing:
    def __init__(self, accession: str, filed: str, text: str, *, form: str = "10-K", report: str | None = "2020-12-31") -> None:
        self.accession_number = accession
        self.filing_date = pd.Timestamp(filed).date()
        self.period_of_report = report
        self.form = form
        self.cik = "0000000001"
        self.body = text
        self.attachments: list[object] = []

    def html(self) -> str:
        return f"<p>{self.body}</p>"

    def text(self) -> str:
        return ""


def text_rows(filings: list[Filing]) -> pd.DataFrame:
    """The `sec_employees_text` rows the walk's parse stores for `filings` (none for a filing with no usable text)."""
    parse = text_mod.EMPLOYEES_TEXT_FETCH.parse
    assert parse is not None
    frames = [parse("AAA", "0000000001", edgar_driver.FilingStamp.of(filing, "0000000001"), identity_for({"AAA": "1"}))[TEXT] for filing in filings]
    return pd.concat([frame for frame in frames if not frame.empty] or [pd.DataFrame(columns=list(TEXT.retrieval_columns))], ignore_index=True)


def build(
    store: Any, filings: list[Filing], answers: dict[str, guard.EmployeeAnswer], *, stored: pd.DataFrame | None = None
) -> mod.EmployeeTickerResult:
    """One ticker's decisions over the stored text rows of `filings`, through the per-ticker worker and a fake LLM."""
    df_text = text_rows(filings)
    if not df_text.empty:
        store.save(TEXT, df_text)
    df_todo, _ = mod._undecided(store.load(TEXT, columns=mod._TEXT_COLUMNS, optional=True), None, {})
    history = pd.DataFrame(columns=["ticker", "as_of", *guard.COUNT_COLUMNS]) if stored is None else stored.assign(ticker="AAA")
    context = SimpleNamespace(
        store=store,
        config=extract_config(data_extract={"llm_text_budget": {"employees": 60000}}),
        log=SimpleNamespace(info=lambda *args: None, warning=lambda *args: None),
    )
    return mod._ticker_employees("AAA", context=context, extractors=lent(FakeEmployeesLLM(answers)), df_todo=df_todo, df_history=history)  # type: ignore[arg-type]


def excerpt(filing: Filing, limit: int) -> str:
    """The excerpt the LLM step rebuilds at `limit` from the row the walk stores for `filing`."""
    return sent_excerpt(next(text_rows([filing]).itertuples(index=False)), limit)


STORED_COLUMNS = [
    "ticker",
    "as_of",
    "cik",
    "accession_number",
    "form",
    "employees_total",
    "employees_full_time",
    "employees_part_time",
    "basis",
    "status",
    "source_document",
    "source_quote",
    "measurement_period",
]


def test_no_roster_and_every_column_is_stored(sqlite_store: Any) -> None:
    """REQ-005 / A1: no roster file or reader; each decided date is one row with provenance, NULL components carry a status."""
    assert not hasattr(mod, "load_manual_roster") and not hasattr(mod, "MANUAL_ROSTER")
    assert not (Path(mod.__file__).parents[4] / "configs" / "sec" / "employees_manual_roster.json").exists()
    low = "Lowe’s employed approximately 161,000 full-time associates and 109,000 part-time associates"
    filings = [
        Filing("low", "2025-03-24", f"{low}.", report="2025-01-31"),
        Filing("quiet", "2024-03-01", "We sell hardware."),
        Filing("image", "2023-03-01", "Workforce totals are shown in the following image."),
        Filing("paraphrase", "2022-03-01", "We had 41,000 employees."),
    ]
    answers = {
        "low": answer(
            None, None, full_time=161000, full_time_quote=low, part_time=109000, part_time_quote=low, measurement_period="January 31, 2025"
        ),
        "quiet": answer(None, None, status="not_disclosed", measurement_period=None),
        "image": answer(None, None, status="image_only", measurement_period=None),
        "paraphrase": answer(41000, "The company employs 41,000 people."),
    }
    frame = build(sqlite_store, filings, answers).frame
    assert frame.columns.tolist() == STORED_COLUMNS == guard.FRAME_COLUMNS
    rows = {row["accession_number"]: row for row in frame.to_dict("records")}
    found = rows["low"]
    assert (found["ticker"], found["as_of"], found["cik"], found["form"], found["status"], found["source_document"]) == (
        "AAA",
        pd.Timestamp("2025-03-24"),
        "0000000001",
        "10-K",
        "found",
        "primary",
    )
    assert (found["employees_total"], found["employees_full_time"], found["employees_part_time"], found["basis"]) == (
        270000,
        161000,
        109000,
        "full_part",
    )
    assert json.loads(found["source_quote"]) == {"full_time": low, "part_time": low} and found["measurement_period"] == "January 31, 2025"
    for accession, status in (("quiet", "not_disclosed"), ("image", "image_only"), ("paraphrase", "unsupported")):
        row = rows[accession]
        assert row["status"] == status and row["cik"] == "0000000001" and row["source_document"] == "primary", row
        assert all(pd.isna(row[column]) for column in ("employees_total", "employees_full_time", "employees_part_time", "basis", "source_quote")), row
    print(
        "\nSANITY: no roster file or reader remains; LOW stores 270,000 = 161,000 FT + 109,000 PT (full_part) with cik, accession, form, "
        "source document and both component quotes; not_disclosed, image_only and unsupported filings are NULL rows carrying their status."
    )


def decide(
    text: str,
    *,
    total: int | None = None,
    full_time: int | None = None,
    part_time: int | None = None,
    quote: str | None = None,
    is_fte: bool = False,
    includes_contractors: bool = False,
) -> guard.EmployeeDecision:
    """The decision on `text` for a fake answer whose every claimed component quotes `quote` (default: all of `text`)."""
    cited = quote or text
    reply = answer(
        total,
        cited if total is not None else None,
        full_time=full_time,
        full_time_quote=cited if full_time is not None else None,
        part_time=part_time,
        part_time_quote=cited if part_time is not None else None,
        is_fte=is_fte,
        includes_contractors=includes_contractors,
    )
    return guard.decide_employee_answer(reply, text)


def supported(count: int | None, text: str, quote: str | None = None) -> int | None:
    """The decided total for a fake answer claiming `count` as the total, quoting all of `text` unless a narrower quote is given."""
    return decide(text, total=count, quote=quote).employees_total


def components(decision: guard.EmployeeDecision) -> tuple[int | None, int | None, int | None, str | None, str]:
    return (decision.employees_total, decision.employees_full_time, decision.employees_part_time, decision.basis, decision.status)


def test_source_guard_resolves_bounds_units_and_ranges():
    exact = "As of December 31, 2023, we employed approximately 74,042 employees."
    assert components(decide(exact, total=74042)) == (74042, None, None, "total", "found")
    # Open bounds move half a unit of the stated precision; a quote trimmed of "over" still reads it.
    assert supported(13000, "At December 31, 2016, we had over 13,000 employees.") == 13_500
    assert supported(300000, "We had over 300,000 employees.") == 305_000
    assert supported(300000, "We had over 300,000 employees.", quote="300,000 employees") == 305_000
    assert supported(2200000, "We employed nearly 2.2 million associates.") == 2_150_000
    assert supported(19800, "With approximately 19,800 employees in more than 30 countries") == 19_800
    # Units, disjoint parts and tables in thousands.
    assert (
        supported(61000, "The number of regular employees was 61 thousand, 62 thousand, and 62 thousand at years ended 2024, 2023, and 2022.")
        == 61_000
    )
    assert supported(826, "As of December 31, 2024, we had 714 non-union employees and 112 union employees.") == 826
    assert supported(97900, "Number of regular employees at year-end (thousands) 97.9") == 97_900
    # A range is not a count, and neither is an unsupported number, a fabricated quote or an image.
    assert supported(50000, "We employ between 50,000 and 100,000 people.") is None
    assert supported(100000, "We employ between 50,000 and 100,000 people.") is None
    assert supported(50000, "We employ 50 to 100 thousand people.") is None
    assert supported(99999, exact) is None
    assert supported(2023, exact) is None
    assert components(decide(exact, total=74042, quote="fabricated quote")) == (None, None, None, None, "unsupported")
    print(
        "\nSANITY: over 13,000 -> 13,500, nearly 2.2 million -> 2,150,000, 61 thousand -> 61,000, disjoint parts sum; "
        "ranges, unsupported counts and a fabricated quote leave every component NULL (status unsupported)."
    )


def test_statuses_without_a_count():
    text = "Workforce totals are shown in the following image."
    assert components(guard.decide_employee_answer(answer(None, None, status="image_only"), text)) == (None, None, None, None, "image_only")
    assert components(guard.decide_employee_answer(answer(None, None, status="not_disclosed"), text))[4] == "not_disclosed"
    assert components(guard.decide_employee_answer(answer(None, None, status="incorporated_by_reference"), text))[4] == "incorporated"
    assert components(guard.decide_employee_answer(answer(None, None, status="ambiguous"), text))[4] == "ambiguous"
    assert components(guard.decide_employee_answer(answer(None, None, status="found"), text))[4] == "ambiguous"
    print("\nSANITY: image-only, not-disclosed, incorporated-by-reference and ambiguous filings decide to NULL components with their own status.")


DLTR_2005 = "We employed approximately 11,040 full-time and 19,115 part-time associates on January 29, 2005."
# MAA FY2019 and FY2020 10-Ks (0001564590-20-005582, 0001564590-21-006666); benchmark totals 2,513 and 2,530.
MAA_2019 = (
    "MAA and MAALP were formed in Tennessee in 1993. As of December 31, 2019, we had 2,476 full-time employees and 37 part-time employees.\n3\n"
    "Business Objectives"
)
MAA_2020 = (
    "Human Capi tal\nAs of December 31, 2020, we employed 2,530 associates. Our associates’ time, energy, creativity and passion are essential."
)


def test_full_and_part_time_components_and_basis():
    # D7: when only FT and PT are stated, the total is their sum with basis full_part.
    assert components(decide(DLTR_2005, full_time=11040, part_time=19115)) == (30155, 11040, 19115, "full_part", "found")
    # A model total that is only the full + part-time sum is recorded as those components.
    assert components(decide(DLTR_2005, total=30155)) == (30155, 11040, 19115, "full_part", "found")
    # A component is supported only by a number its own shift labels.
    assert components(decide(DLTR_2005, full_time=19115)) == (None, None, None, None, "unsupported")
    assert components(decide(DLTR_2005, part_time=19115)) == (None, None, None, None, "unsupported")
    # A full-time count is not a total: stated full-time only.
    assert components(decide("The Company had over 6,250 full-time employees.", total=6250)) == (None, 6255, None, "full_time_only", "found")
    assert components(decide("Medtronic has 95,000+ full-time employees", full_time=95000)) == (None, 95500, None, "full_time_only", "found")
    assert supported(230800, "We employed approximately 230,800 full-time and part-time employees.") == 230_800
    # MAA: the stated total is kept, never converted to its full-time row.
    maa_2019_quote = "As of December 31, 2019, we had 2,476 full-time employees and 37 part-time employees."
    assert components(decide(MAA_2019, total=2513, quote=maa_2019_quote)) == (2513, 2476, 37, "full_part", "found")
    maa_2020_quote = "As of December 31, 2020, we employed 2,530 associates."
    assert components(decide(MAA_2020, total=2530, quote=maa_2020_quote)) == (2530, None, None, "total", "found")
    print("\nSANITY: FT+PT give a full_part total (DLTR 30,155; MAA 2019 2,513 = benchmark); MAA 2020 keeps its 2,530 total; no FT conversion.")


# ADM FY2019 10-K 0000007084-20-000009: the shift is named before the number, in the same phrase.
ADM_2019 = (
    "Number of Employees The number of full-time employees of the Company was approximately 38,100 at December 31, 2019 and 31,600 at "
    "December 31, 2018 . The net increase in the number of full-time employees is primarily related to acquisitions."
)


def test_shift_named_before_the_number_is_a_full_time_count():
    quote = "The number of full-time employees of the Company was approximately 38,100 at December 31, 2019"
    # The model's total is a full-time count, so it is recorded as full-time only, never as a total.
    assert components(decide(ADM_2019, total=38100, quote=quote)) == (None, 38100, None, "full_time_only", "found")
    assert components(decide(ADM_2019, full_time=38100, quote=quote)) == (None, 38100, None, "full_time_only", "found")
    # A shift that labels an earlier number does not label the next one (C 2001): the sum stays a total.
    c_2001 = (
        "the Company had approximately 138,000 full-time and 9,000 part-time employees in the United States and "
        "approximately 95,000 employees outside of the United States."
    )
    assert components(decide(c_2001, total=242000)) == (242000, None, None, "total", "found")
    # A combined "full- and part-time" label before the number stays a total.
    assert components(decide("The number of full- and part-time employees was 5,400 at year end.", total=5400)) == (
        5400,
        None,
        None,
        "total",
        "found",
    )
    print("\nSANITY: ADM 2019 'number of full-time employees ... was approximately 38,100' is full_time_only 38,100, not a 38,100 total.")


# MRK FY2020/FY2024 10-Ks: 0000310158-21-000004 and 0001628280-25-007732.
@pytest.mark.parametrize("total,label", [(74000, "full time-employees"), (75000, "full-time employees")])
def test_mrk_possessive_employee_sentence_keeps_full_time_count(total: int, label: str) -> None:
    total_quote = f"the Company had approximately {total:,} employees worldwide"
    full_quote = f"Approximately 73,000 of the Company’s employees are {label}."
    text = f"As of December 31, 2024, {total_quote}. {full_quote}"
    decision = guard.decide_employee_answer(answer(total, total_quote, full_time=73000, full_time_quote=full_quote), text)
    assert components(decision) == (total, 73000, None, "full_part", "found")
    assert not decision.rejected
    print(f"\nSANITY: MRK's explicit {total:,} total and 73,000 full-time employees both survive verbatim quote validation.")


@pytest.mark.parametrize("value", [90000, 100000, 110000])
@pytest.mark.parametrize("quote", [None, "Our workforce was approximately this many people."])
def test_missing_or_paraphrased_quote_recovers_source_count_within_ten_percent(value: int, quote: str | None) -> None:
    text = f"We employed around {value:,} employees worldwide."
    decision = guard.decide_employee_answer(answer(value, quote), text, previous={"total": 100000})
    assert decision.employees_total == value and decision.status == "found" and not decision.rejected
    assert dict(decision.quotes)["total"] in text
    print(f"\nSANITY: {value:,} is within inclusive +/-10% of 100,000; a broken quote is replaced with actual filing evidence.")


@pytest.mark.parametrize(
    "text,value,previous",
    [
        ("We employed 89,999 employees.", 89999, {"total": 100000}),
        ("We employed 110,001 employees.", 110001, {"total": 100000}),
        ("We employed 100,000 employees.", 100000, {}),
        ("We employed 100,000 employees.", 100000, {"total": 0}),
        ("We employed 100,000 employees.", 101000, {"total": 100000}),
        ("Revenue was 100,000 dollars. We employed 500 people.", 100000, {"total": 100000}),
        ("Revenue was 100,000 dollars.We employed 500 people.", 100000, {"total": 100000}),
        ("We employed between 90,000 and 100,000 employees.", 100000, {"total": 100000}),
        ("We employed 100,000 employees and contractors.", 100000, {"total": 100000}),
    ],
)
def test_quote_recovery_retains_numeric_and_workforce_guards(text: str, value: int, previous: dict[str, int]) -> None:
    decision = guard.decide_employee_answer(answer(value, None), text, previous=previous)
    assert decision.status == "unsupported" and decision.employees_total is None
    print("\nSANITY: quote recovery rejects out-of-band, unanchored, absent, non-workforce, ranged and unflagged contractor counts.")


@pytest.mark.parametrize("qualifier", ["approximately", "around", "about", "roughly"])
def test_approximate_counts_keep_the_stated_value(qualifier: str) -> None:
    text = f"We employed {qualifier} 75,000 employees worldwide."
    assert decide(text, total=75000).employees_total == 75000
    print(f"\nSANITY: '{qualifier} 75,000' retains the stated 75,000 employees.")


def test_quote_recovery_keeps_component_labels() -> None:
    text = "Approximately 73,000 of the Company’s employees are full-time employees."
    reply = answer(None, None, full_time=73000)
    decision = guard.decide_employee_answer(reply, text, previous={"full_time": 72000})
    assert components(decision) == (None, 73000, None, "full_time_only", "found")
    assert dict(decision.quotes)["full_time"] in text
    assert guard.decide_employee_answer(reply, "We had 73,000 employees.", previous={"full_time": 72000}).status == "unsupported"
    assert guard.decide_employee_answer(reply, text, previous={"total": 74000}).status == "unsupported"
    fte = "Approximately 73,000 of the Company's employees were full-time equivalent employees."
    assert decide(fte, total=73000).basis == "fte"
    assert decide(fte, full_time=73000).status == "unsupported"
    print("\nSANITY: a missing full-time quote recovers its explicit label; a total or an earlier total cannot stand in for full-time evidence.")


def test_quote_recovery_uses_stored_and_new_counts_only_from_earlier_dates(sqlite_store: Any) -> None:
    stored = pd.DataFrame(
        [
            {"as_of": pd.Timestamp("2019-03-01"), "employees_total": 100000, "employees_full_time": None, "employees_part_time": None},
            {"as_of": pd.Timestamp("2025-03-01"), "employees_total": 95000, "employees_full_time": None, "employees_part_time": None},
        ]
    )
    filings = [
        Filing("before-history", "2018-03-01", "We employed 95,000 employees worldwide."),
        Filing("stored-prior", "2020-03-01", "We employed around 110,000 employees worldwide."),
        Filing("new-prior", "2021-03-01", "We employed about 121,000 employees worldwide."),
        Filing("large-change", "2022-03-01", "We employed 170,000 employees worldwide."),
    ]
    answers = {
        "before-history": answer(95000, None),
        "stored-prior": answer(110000, None),
        "new-prior": answer(121000, "The company has approximately 121,000 staff."),
        "large-change": answer(170000, "We employed 170,000 employees worldwide."),
    }
    result = build(sqlite_store, filings, answers, stored=stored)
    rows = result.frame.set_index("accession_number")
    assert rows.loc["before-history", "status"] == "unsupported"
    assert rows.loc[["stored-prior", "new-prior", "large-change"], "employees_total"].tolist() == [110000, 121000, 170000]
    print(
        "\nSANITY: recovery uses stored and newly validated earlier counts; future rows cannot justify history, and quoted large changes remain valid."
    )


# ADM FY2023 10-K 0000007084-24-000009: type-by-gender table whose earlier cells are percentages.
ADM_2023_TABLE = (
    "Number of Employees by Type and Gender Male % Female % Total % Full-time 30,497 76 % 9,716 24 % 40,213 100 % "
    "Part-time 603 38 % 986 62 % 1,589 100 % Total 31,100 74 % 10,702 26 % 41,802 100 %"
)


def test_table_rows_with_percentage_cells_keep_their_shift_label():
    reply = answer(
        41802,
        "Total 31,100 74 % 10,702 26 % 41,802",
        full_time=40213,
        full_time_quote="Full-time 30,497 76 % 9,716 24 % 40,213",
        part_time=1589,
        part_time_quote="Part-time 603 38 % 986 62 % 1,589",
    )
    assert components(guard.decide_employee_answer(reply, ADM_2023_TABLE)) == (41802, 40213, 1589, "full_part", "found")
    # The Total row's number is still unlabelled, so it cannot stand in for the full-time row.
    wrong_row = answer(None, None, full_time=41802, full_time_quote="Total 31,100 74 % 10,702 26 % 41,802")
    assert components(guard.decide_employee_answer(wrong_row, ADM_2023_TABLE)) == (None, None, None, None, "unsupported")
    print("\nSANITY: ADM FY2023 rows with % cells store 41,802 / 40,213 / 1,589 (full_part); a Total-row number is not a full-time count.")


# DLTR FY2020 10-K 0000935703-21-000014 (Human Capital table).
DLTR_2021 = (
    "for 11 Table of Contents progress on key DEI objectives. As of January 30, 2021, we employed more than 199,300 associates, as follows: "
    "Store and Distribution Center Associates Dollar Tree Family Dollar Store Support Center Associates Total Full-time Associates 27,952 "
    "29,862 2,403 60,217 Part-time Associates 97,913 41,184 13 139,110 Total 125,865 71,046 2,416 199,327 Part-time associates work an "
    "average of less than 30 hours per week and the number of part-time associates fluctuates depending on seasonal needs."
)
# DLTR FY2024 10-K 0000935703-25-000015: the count includes the held-for-sale Family Dollar business.
DLTR_2025 = (
    "The number of associates we employed as of February 1, 2025 is as follows, including associates within the held for sale Family Dollar "
    "business: Store and Distribution Center Associates Dollar Tree Family Dollar Store Support Center Associates Total Full-time Associates "
    "33,233 28,712 2,489 64,434 Part-time Associates 110,050 40,224 2 150,276 Total 143,283 68,936 2,491 214,710 Part-time associates work an "
    "average of less than 30 hours per week."
)


def test_dltr_table_rows_are_components():
    reply = answer(
        199327,
        "As of January 30, 2021, we employed more than 199,300 associates ... Total 125,865 71,046 2,416 199,327",
        full_time=60217,
        full_time_quote="Total Full-time Associates 27,952 29,862 2,403 60,217",
        part_time=139110,
        part_time_quote="Part-time Associates 97,913 41,184 13 139,110",
    )
    assert components(guard.decide_employee_answer(reply, DLTR_2021)) == (199327, 60217, 139110, "full_part", "found")
    # The full-time count quoted with the Total row only is not supported; the total stays.
    wrong_row = answer(199327, "Total 125,865 71,046 2,416 199,327", full_time=60217, full_time_quote="Total 125,865 71,046 2,416 199,327")
    decision = guard.decide_employee_answer(wrong_row, DLTR_2021)
    assert components(decision) == (199327, None, None, "total", "found") and decision.rejected == ("full_time: not_stated",)
    # A bounded total is stored as stated, never combined with an exact component.
    bounded = answer(
        199300, "we employed more than 199,300 associates", full_time=60217, full_time_quote="Full-time Associates 27,952 29,862 2,403 60,217"
    )
    assert components(guard.decide_employee_answer(bounded, DLTR_2021)) == (199350, 60217, None, "full_part", "found")
    held_for_sale = answer(
        214710, "Total 143,283 68,936 2,491 214,710", full_time=64434, full_time_quote="Full-time Associates 33,233 28,712 2,489 64,434"
    )
    assert components(guard.decide_employee_answer(held_for_sale, DLTR_2025))[:2] == (214710, 64434)
    print(
        "\nSANITY: DLTR 2021 stores total 199,327 / FT 60,217 / PT 139,110 from their own rows; a wrong-row FT is rejected; 2025 keeps held-for-sale staff."
    )


# LOW FY2024 10-K 0000060667-25-000049.
LOW_2025 = (
    "4 Table of Contents Our People As of January 31, 2025, Lowe’s employed approximately 161,000 full-time associates and 109,000 part-time "
    "associates, primarily in the United States and India. During the spring season, we temporarily expand our workforce by hiring associates "
    "in part-time and full-time positions to meet the elevated levels of demand."
)
# KR FY2024 10-K 0001558370-25-004267.
KR_2025 = (
    "HUMAN CAPITAL MANAGEMENT Our People We want Kroger to be a place where our customers love to shop and associates love to work. "
    "As of February 1, 2025, Kroger employed over 409,000 full- and part-time employees. Our people are essential to our success."
)


def test_low_split_and_kr_combined_total():
    low_quote = "Lowe’s employed approximately 161,000 full-time associates and 109,000 part-time associates"
    assert components(decide(LOW_2025, full_time=161000, part_time=109000, quote=low_quote)) == (270000, 161000, 109000, "full_part", "found")
    kr_quote = "As of February 1, 2025, Kroger employed over 409,000 full- and part-time employees."
    assert components(decide(KR_2025, total=409000, quote=kr_quote)) == (409500, None, None, "total", "found")
    assert components(decide(KR_2025, full_time=409000, quote=kr_quote))[4] == "unsupported"
    print(
        "\nSANITY: LOW 2025 is 161,000 FT + 109,000 PT = 270,000 (full_part); KR's 'over 409,000 full- and part-time' is a 409,500 total, not a full-time count."
    )


# VTRS (Mylan) FY2016 10-K 0001623613-17-000007; AAPL FY2001 10-K405 0000912057-01-544436.
VTRS_2017 = (
    "Employees As of December 31, 2016 , Mylan �s global workforce included more than 35,000 employees and external contractors. Certain production"
)
AAPL_2001 = (
    "Employees\nAs of September 29, 2001, Apple and its subsidiaries worldwide had 9,603 employees and an additional 1,831 temporary "
    "employees and contractors.\nItem 2. Properties"
)


def test_contractor_mixed_total_needs_its_flag():
    vtrs_quote = "Mylan �s global workforce included more than 35,000 employees and external contractors."
    rejected = decide(VTRS_2017, total=35000, quote=vtrs_quote)
    assert components(rejected) == (None, None, None, None, "unsupported") and rejected.rejected == ("total: contractors",)
    flagged = decide(VTRS_2017, total=35000, quote=vtrs_quote, includes_contractors=True)
    assert components(flagged) == (35500, None, None, "total_incl_contractors", "found")
    aapl_quote = "Apple and its subsidiaries worldwide had 9,603 employees and an additional 1,831 temporary employees and contractors."
    assert components(decide(AAPL_2001, total=9603, quote=aapl_quote)) == (9603, None, None, "total", "found")
    print(
        "\nSANITY: VTRS 'more than 35,000 employees and external contractors' is rejected unless flagged total_incl_contractors; AAPL's separate 1,831 contractors keep 9,603."
    )


# ACN FY2024 10-K 0001467373-24-000278; AMP FY2023 10-K 0000820027-24-000015 (page footer inside the sentence).
ACN_2024 = (
    "Our workforce, the majority of which serves our clients, increased to approximately 774,000 as of August 31, 2024, compared to "
    "approximately 733,000 as of August 31, 2023. $64.9B in revenues. We employed approximately 774,000 people as of August 31, 2024."
)
AMP_2024 = "This includes approximately 13,800 global 8 Index Ameriprise Financial, Inc. employees, including our corporate employees and employee financial advisors."


def test_paraphrase_rejected_and_ellipsis_join_located():
    paraphrase = decide(ACN_2024, total=774000, quote="Our workforce increased to approximately 774,000 as of August 31, 2024")
    assert components(paraphrase) == (None, None, None, None, "unsupported") and paraphrase.rejected == ("total: not_located",)
    assert supported(774000, ACN_2024, "We employed approximately 774,000 people as of August 31, 2024.") == 774_000
    assert supported(774000, ACN_2024, "Our workforce, ... increased to approximately 774,000 as of August 31, 2024") == 774_000
    assert supported(13800, AMP_2024, "approximately 13,800 global ... employees") == 13_800
    print(
        "\nSANITY: ACN's word-dropping paraphrase is rejected; its verbatim sentence and a ` ... `-joined quote across AMP's page footer are located."
    )


def test_source_guard_locates_quotes_through_filing_text_noise():
    assert (
        supported(
            53368, "Employees We employed 53,368 persons at December 31, 2018 . Environmental", "We employed 53,368 persons at December 31, 2018."
        )
        == 53_368
    )
    assert (
        supported(
            5700,
            "EMPLOYEES\nA s of February 7, 2014, we had approximately 5,700 employees.",
            "As of February 7, 2014, we had approximately 5,700 employees.",
        )
        == 5_700
    )
    msft = "the Company employed approximately 47,600 people on a full-\ntime basis, 33,000 in the United States"
    msft_quote = "the Company employed approximately 47,600 people on a full-time basis"
    assert components(decide(msft, full_time=47600, quote=msft_quote)) == (None, 47600, None, "full_time_only", "found")
    mcd = "The Company’s number of employees worldwide was approximately 440,000 as of year-end 2012 ."
    assert supported(440000, mcd, "The Company�s number of employees worldwide was approximately 440,000 as of year-end 2012.") == 440_000
    afl = (
        "Aflac Japan had 3,860 employees and Aflac U.S. had 4,089 employees. We consider our relations excellent. Other operations had 293 employees."
    )
    assert (
        supported(8242, afl, "Aflac Japan had 3,860 employees and Aflac U.S. had 4,089 employees. ... Other operations had 293 employees.") == 8_242
    )
    assert (
        supported(
            39000, "ADM is a global company of approximately 39,000 employees.", "The Company is a global company of approximately 39,000 employees."
        )
        is None
    )
    print("\nSANITY: line-break hyphens, split letters, spaced full stops, bad apostrophes and `...` joins locate; a paraphrase does not.")


def test_source_guard_recovers_punctuation_and_anchored_table_quote():
    apple_source = "Employees As of September 28, 2019 , the Company had approximately 137,000 full-time equivalent employees."
    apple_quote = "As of September 28, 2019, the Company had approximately 137,000 full-time equivalent employees."
    heading = (
        "The following tables set forth information about the Company's employees as of December 31, 2020. Number of Employees by Contract and Region"
    )
    total = "Total 17,163 20,412 1,513 39,088"
    table_quote = f"{heading} ... {total}"
    table_source = f"{heading} North America 8,196 10,227 270 18,693 EMEA 4,586 4,847 564 9,997 {total}"
    assert components(decide(apple_source, total=137000, quote=apple_quote, is_fte=True)) == (137000, None, None, "fte", "found")
    assert decide(apple_source, total=137000, quote=apple_quote).basis == "fte"
    assert supported(39088, table_source, table_quote) == 39088
    intro = heading.split(" Number of Employees", 1)[0]
    intro_quote = f"{intro} ... {total}"
    assert supported(39088, table_source, intro_quote) == 39088
    assert supported(39089, table_source, table_quote) is None
    assert supported(39088, total, table_quote) is None
    assert supported(39088, f"{heading} {'other data ' * 300} {total}", table_quote) is None
    assert supported(39088, f"{heading} [... filing gap ...] {total}", table_quote) is None
    assert supported(39088, f"{heading} [... filing gap ...] {total}", f"{heading} [... filing gap ...] {total}") is None
    assert supported(39088, f"{heading} As of December 31, 2019, employees {total}", table_quote) is None
    assert supported(39088, f"{intro} Number of Employees by Type Number of Employees by Region {total}", intro_quote) is None
    assert supported(44043, "ADM employed approximately 44,000 people.") is None
    print(
        "\nSANITY: punctuation noise and a nearby real table heading/total recover supported counts; an FTE-labelled total is basis fte; "
        "missing, distant, cross-gap, conflicting or imprecise claims abstain."
    )


class NoPrimaryDocumentFiling(Filing):
    """edgartools 5.51 `Filing.html()` reads `homepage.primary_html_document.empty` without a
    None check, and `text()` goes through `html()`; both raise AttributeError from the library."""

    def __init__(self, accession: str, filed: str, submission: str) -> None:
        super().__init__(accession, filed, "")
        self.submission = submission

    def html(self) -> str:
        return None.empty  # type: ignore[attr-defined]

    def text(self) -> str:
        return self.html()

    def full_text_submission(self) -> str:
        return self.submission


def test_missing_primary_document_reads_full_submission(sqlite_store: Any) -> None:
    submission = "<SEC-DOCUMENT><TEXT>As of December 31, 1999, we had 1,234 employees.</TEXT></SEC-DOCUMENT>"
    filing = NoPrimaryDocumentFiling("no-primary", "2000-03-01", submission)
    assert document_mod.filing_body_text(filing) == "As of December 31, 1999, we had 1,234 employees."
    unreadable = NoPrimaryDocumentFiling("unreadable", "2000-03-01", "")
    assert text_rows([unreadable]).empty
    result = build(sqlite_store, [filing, unreadable], {"no-primary": answer(1234, "we had 1,234 employees.")})
    assert result.frame["employees_total"].tolist() == [1234.0] and [outcome["accession_number"] for outcome in result.outcomes] == ["no-primary"]
    print(
        "\nSANITY: a filing whose index lists no primary document no longer aborts the run with AttributeError; "
        "its full submission supplies the source text, and a truly empty one stores no text row, so it is never sent."
    )


FILLER = "The Company designs and sells measurement instruments to laboratories in many markets. "  # no workforce word


def test_ix_header_is_stripped_and_the_workforce_sentence_survives(sqlite_store: Any) -> None:
    """Agilent FY2019 shape: a hidden inline-XBRL header precedes the visible text."""
    junk = " ".join(f"<ix:nonNumeric name='us-gaap:Fact{i}' contextRef='c{i}'>0001090872 2019-10-31 {i * 37}</ix:nonNumeric>" for i in range(400))
    header = f"<div style='display:none'><ix:header><ix:hidden>{junk}</ix:hidden><ix:references/></ix:header></div>"
    statement = "As of October 31, 2019, we employed approximately 16,300 people worldwide."
    filing = Filing("a-2019", "2019-12-19", "")
    filing.html = lambda: f"<html><body>{header}<p>{FILLER * 150}</p><p>{statement}</p></body></html>"
    text = document_mod.filing_body_text(filing)
    assert len(junk) > 20_000 and "us-gaap" not in text and "0001090872" not in text
    assert statement in excerpt(filing, 10_000)
    result = build(sqlite_store, [filing], {"a-2019": answer(16300, statement)})
    assert result.frame["employees_total"].tolist() == [16300.0]
    print("\nSANITY: >20k chars of hidden ix:header values are dropped before excerpting; Agilent's 16,300 people sentence is sent and saved.")


def test_workforce_number_windows_come_before_generic_context():
    persons = "As of December 31, 2019, 502 persons were employed by the Company."
    kim = f"{FILLER * 120}{persons} {FILLER * 20}"
    assert persons in excerpt(Filing("kim", "2020-03-01", kim), 12_000)
    noise = "".join(f"Our employees value safety and training, item {i}. {FILLER * 15}" for i in range(130))
    workforce = "At December 31, 2020, we had 41,000 employees worldwide."
    many_hits = f"{noise}{workforce} {FILLER * 10}"
    sent = excerpt(Filing("many", "2021-03-01", many_hits), 60_000)
    assert workforce in sent and len(sent) <= 60_000
    assert len(text_mod._CONTEXT_RE.findall(many_hits)) > 120
    for headcount in ("Headcount 256,981 at year end.", "Our workforce of more than 10,000 people.", "We employ more than 25,000 people."):
        assert headcount in excerpt(Filing("one", "2021-03-01", f"{FILLER * 120}{headcount} {FILLER * 10}"), 9_000)
    print(
        "\nSANITY: '502 persons were employed', 'Headcount 256,981' and 'employ more than 25,000' are excerpted, and a 41,000-employee "
        "sentence after 130 generic workforce mentions stays inside the 60,000-char budget."
    )


class Attachment:
    def __init__(self, document_type: str, content: str, description: str = "") -> None:
        self.document_type = document_type
        self.description = description
        self._content = content

    @property
    def content(self) -> str:
        return self._content

    def is_binary(self) -> bool:
        return False


class UnreadAttachment(Attachment):
    @property
    def content(self) -> str:
        pytest.fail(f"{self.document_type} is not an annual-report exhibit and was read")


def test_exhibit_fallback_reads_the_annual_report_exhibit(sqlite_store: Any) -> None:
    wy = "The company has 44,800 employees, of whom 43,800 are employed in its timber-based businesses."
    primary = (
        f"{FILLER * 30}The following information is included in the 1999 Annual Report to Stockholders and is incorporated "
        "herein by reference: 1. Segment information--Pages 75 and 76. 2. The number of persons employed by the registrant--Page 49."
    )
    filing = Filing("wy-2000", "2000-03-10", primary)
    filing.attachments = [UnreadAttachment("EX-21", ""), Attachment("EX-13", f"<p>{FILLER * 40}</p><p>{wy}</p>")]
    chosen = document_mod.employee_text(filing)
    assert chosen.source_document == "EX-13" and wy in chosen.text
    no_number = Filing("wy-bare", "2000-03-10", f"{FILLER * 30}Employees are described in the Annual Report.")
    no_number.attachments = [Attachment("EX-99", f"<p>{wy}</p>", description="1999 Annual Report to Shareholders")]
    assert document_mod.employee_text(no_number).source_document == "EX-99"
    result = build(sqlite_store, [filing], {"wy-2000": answer(44800, "The company has 44,800 employees")})
    assert result.frame["employees_total"].tolist() == [44800.0] and result.frame["source_document"].tolist() == ["EX-13"]
    assert document_mod.is_annual_report_exhibit("EX-99", "Annual Report to Shareholders") and not document_mod.is_annual_report_exhibit(
        "EX-99.1", "Press release"
    )
    stated = Filing("own", "2025-03-01", "We had 41,000 employees.")
    stated.attachments = [UnreadAttachment("EX-13", "")]
    assert document_mod.employee_text(stated).source_document == "primary"
    bare = Filing("bare", "2025-03-01", f"{FILLER * 30}No exhibit carries a workforce figure.")
    bare.attachments = [Attachment("EX-13", f"<p>{FILLER * 10}</p>")]
    assert document_mod.employee_text(bare).source_document == "primary"
    print(
        "\nSANITY: a primary document that incorporates its employee count by reference sends the EX-13 text (44,800 saved, "
        "source EX-13), as does an EX-99 annual report; a primary that states its own count reads no exhibit; an exhibit with no workforce number is not used."
    )


def test_offline_task_builder_and_decision_for_cached_text():
    """A task from one stored text row (no Filing, no network), then the pure decision."""
    filing = Filing("0000060667-25-000049", "2025-03-24", f"{FILLER * 200}{LOW_2025} {FILLER * 20}", report="2025-01-31")
    row = next(text_rows([filing]).itertuples(index=False))
    task = guard.employee_task(7, "LOW", row, 9_000)
    source_text = str(task.meta["source_text"])
    assert task.seq == 7 and task.schema is guard.EmployeeAnswer and task.meta["source_document"] == "primary"
    assert task.payload.startswith("Ticker: LOW\nFiscal period end: 2025-01-31\nSEC filing date: 2025-03-24\nAccession: 0000060667-25-000049\n")
    assert (
        LOW_2025 in source_text and task.payload.endswith(source_text) and "Source document: primary" in task.payload and len(task.payload) <= 9_000
    )
    quote = "Lowe’s employed approximately 161,000 full-time associates and 109,000 part-time associates"
    reply = answer(None, None, full_time=161000, full_time_quote=quote, part_time=109000, part_time_quote=quote)
    decision = guard.decide_employee_answer(reply, source_text)
    assert (decision.employees_total, decision.basis, decision.quotes) == (270000, "full_part", (("full_time", quote), ("part_time", quote)))
    with pytest.raises(ValueError, match="too small for filing metadata"):
        guard.employee_task(0, "LOW", row, 100)
    print("\nSANITY: a stored LOW text row builds a 9,000-char task offline and its fake answer decides to 270,000 full_part with both quotes kept.")


def test_prompt_states_the_component_rules():
    prompt = (Path(mod.__file__).parents[3] / "gpt_extract" / "prompt_templates" / "employees_system_prompt.md").read_text(encoding="utf-8")
    for rule in ("verbatim", "respectively", "held for sale", "contractors", "has no employees", " ... ", "full_time", "part_time", "is_fte"):
        assert rule in prompt, rule
    print("\nSANITY: the employee prompt carries the verbatim, row, respectively, held-for-sale, contractor, subsidiary-sum and ` ... ` rules.")


# AAPL FY2002 10-K 0001047469-02-007674; CSCO FY2014 10-K 0000858877-14-000029.
AAPL_2002 = (
    "Employees\nAs of September 28, 2002, Apple and its subsidiaries worldwide had 10,211 employees and an additional 2,030 temporary "
    "employees and contractors.\nItem 2. Properties"
)
CSCO_2014 = (
    "Employees Employees are summarized as follows: July 26, 2014 Employees by geography: United States 36,725 Rest of world 37,317 "
    "Total 74,042 Employees by line item on the Consolidated Statements of Operations: Cost of sales (1) 16,348 Research and development "
    "25,837 Sales and marketing 24,740 General and administrative 7,117 Total 74,042 (1) Cost of sales includes manufacturing support."
)


def test_legacy_form_table_total_and_image_null_row(sqlite_store: Any) -> None:
    filings = [
        Filing("aapl-2001", "2001-12-21", AAPL_2001, form="10-K405", report="2001-09-29"),
        Filing("aapl-2002", "2002-12-19", AAPL_2002, report="2002-09-28"),
        Filing("csco-2014", "2014-09-09", CSCO_2014, report="2014-07-26"),
        Filing("peg-2023", "2024-02-26", "Workforce totals are shown in the following image.", report="2023-12-31"),
        Filing("dltr-2021", "2021-03-16", DLTR_2021, report="2021-01-30"),
    ]
    answers = {
        "aapl-2001": answer(9603, "Apple and its subsidiaries worldwide had 9,603 employees", measurement_period="September 29, 2001"),
        "aapl-2002": answer(10211, "Apple and its subsidiaries worldwide had 10,211 employees", measurement_period="September 28, 2002"),
        "csco-2014": answer(74042, "United States 36,725 Rest of world 37,317 Total 74,042", measurement_period="July 26, 2014"),
        "peg-2023": answer(None, None, status="image_only", measurement_period=None),
        "dltr-2021": answer(
            None,
            None,
            full_time=60217,
            full_time_quote="Full-time Associates 27,952 29,862 2,403 60,217",
            part_time=139110,
            part_time_quote="Part-time Associates 97,913 41,184 13 139,110",
        ),
    }
    result = build(sqlite_store, filings, answers)
    employees = result.frame["employees_total"].tolist()
    assert employees[:4] == [9603.0, 10211.0, 74042.0, 199327.0] and pd.isna(employees[4])
    assert result.frame["status"].tolist() == ["found", "found", "found", "found", "image_only"]
    assert pd.to_datetime(result.frame["as_of"]).tolist() == [
        pd.Timestamp(day) for day in ("2001-12-21", "2002-12-19", "2014-09-09", "2021-03-16", "2024-02-26")
    ]
    by_accession = {row["accession_number"]: row for row in result.outcomes}
    assert {key: row["status"] for key, row in by_accession.items()} == {
        "aapl-2001": "found",
        "aapl-2002": "found",
        "csco-2014": "found",
        "peg-2023": "image_only",
        "dltr-2021": "found",
    }
    dltr = by_accession["dltr-2021"]
    assert (dltr["employees_total"], dltr["employees_full_time"], dltr["employees_part_time"], dltr["basis"]) == (199327, 60217, 139110, "full_part")
    assert by_accession["aapl-2001"]["basis"] == "total" and by_accession["peg-2023"]["basis"] is None
    assert by_accession["dltr-2021"]["source_document"] == "primary"
    print(
        "\nSANITY: AAPL 9,603 / 10,211 (separate contractors excluded) and CSCO 74,042 keep their truths at SEC filing dates; image-only PEG "
        "is a NULL row with status image_only; DLTR's FT/PT rows give a 199,327 full_part total."
    )


def test_same_day_amendment_uses_later_supported_count(sqlite_store: Any) -> None:
    filings = [
        Filing("original", "2024-03-01", "We had 40,000 employees.", report="2023-12-31"),
        Filing("amendment", "2024-03-01", "We had 41,000 employees.", form="10-K/A", report="2023-12-31"),
    ]
    result = build(
        sqlite_store,
        filings,
        {
            "original": answer(40000, "We had 40,000 employees."),
            "amendment": answer(41000, "We had 41,000 employees."),
        },
    )
    assert result.frame["employees_total"].tolist() == [41000.0]
    assert [row["status"] for row in result.outcomes] == ["superseded", "found"]
    print("\nSANITY: a same-date amendment replaces the original value at the one-row filing-date grain.")


def test_missing_report_period_keeps_filing_date(sqlite_store: Any) -> None:
    filing = Filing("no-period", "2024-03-01", "We had 40,000 employees.", report=None)
    llm = FakeEmployeesLLM({"no-period": answer(40000, "We had 40,000 employees.")})
    df_text = text_rows([filing])
    sqlite_store.save(TEXT, df_text)
    df_todo, _ = mod._undecided(sqlite_store.load(TEXT, columns=mod._TEXT_COLUMNS), None, {})
    context = SimpleNamespace(
        store=sqlite_store,
        config=extract_config(data_extract={"llm_text_budget": {"employees": 60000}}),
        log=SimpleNamespace(info=lambda *args: None, warning=lambda *args: None),
    )
    history = pd.DataFrame(columns=["ticker", "as_of", *guard.COUNT_COLUMNS])
    result = mod._ticker_employees("AAA", context=context, extractors=lent(llm), df_todo=df_todo, df_history=history)  # type: ignore[arg-type]
    assert pd.to_datetime(result.frame["as_of"]).tolist() == [pd.Timestamp("2024-03-01")]
    assert "Fiscal period end: unknown\n" in llm.tasks[0].payload
    print("\nSANITY: a filing with no report period is stored at its filing date, its prompt saying 'Fiscal period end: unknown'.")


def test_the_walk_scans_a_document_for_workforce_numbers_once(monkeypatch):
    scans: list[int] = []
    spans = document_mod.workforce_number_spans
    monkeypatch.setattr(document_mod, "workforce_number_spans", lambda text: scans.append(len(text)) or spans(text))
    row = next(text_rows([Filing("one", "2025-03-01", "We had 41,000 employees.")]).itertuples(index=False))
    task = guard.employee_task(1, "AAA", row, 60_000)
    assert "41,000 employees" in str(task.meta["source_text"]) and len(scans) == 1
    print("\nSANITY: the stored row is built from one workforce-number scan of the document; the LLM side scans nothing.")
