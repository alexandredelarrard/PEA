"""Offline known-truth checks for SEC employee extraction and safe replay."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from click.testing import CliRunner
from omegaconf import OmegaConf

import src.data_extract.cli as cli_mod
import src.data_extract.utils.fundamentals.fundamentals_employees as mod
from src.data_extract.utils.common import edgar_driver
from src.data_extract.utils.common.identity import FilingScope
from src.data_extract.utils.common.registrant import Combine, combine_for
from src.gpt_extract.utils.schemas_gpt import LlmResult


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


def answer(
    total: int | None,
    quote: str | None,
    *,
    full_time: int | None = None,
    full_time_quote: str | None = None,
    part_time: int | None = None,
    part_time_quote: str | None = None,
    is_fte: bool = False,
    includes_contractors: bool = False,
    status: str = "found",
    measurement_period: str | None = "year end",
) -> mod.EmployeeAnswer:
    """A fake LLM answer: `total` with `quote`, plus optional full-/part-time components with their own quotes."""
    return mod.EmployeeAnswer(
        status=status,
        total=total,
        total_quote=quote,
        full_time=full_time,
        full_time_quote=full_time_quote,
        part_time=part_time,
        part_time_quote=part_time_quote,
        is_fte=is_fte,
        includes_contractors=includes_contractors,
        measurement_period=measurement_period,
        qualifier=None,
        reason="fixture",
    )


def build(
    monkeypatch: pytest.MonkeyPatch,
    filings: list[Filing],
    answers: dict[str, mod.EmployeeAnswer],
    *,
    done_dates: frozenset[pd.Timestamp] = frozenset(),
    manual: dict[str, dict] | None = None,
    scope: mod.EdgarScope | None = None,
) -> mod.EmployeeTickerResult:
    def listing(filing_scope, forms, **kwargs):
        assert filing_scope.ticker == "AAA" and filing_scope.event_ciks == ("0000000001",)
        assert "10-K405" in forms
        return filings

    class FakeLLM:
        def __init__(self, *args, **kwargs):
            self.tasks = []

        def submit(self, task):
            self.tasks.append(task)

        def run(self):
            return [LlmResult(seq=task.seq, task=task, parsed=answers[task.meta["stamp"].accession_number]) for task in self.tasks]

    identity = SimpleNamespace(
        owns=lambda ticker, cik: ticker == "AAA" and str(cik).zfill(10) == "0000000001",
        filing_scope=lambda ticker: FilingScope.roster_only(ticker, "0000000001"),
    )
    monkeypatch.setattr(edgar_driver, "resolve_registrant_filings", listing)
    monkeypatch.setattr(mod, "LLMExtractor", FakeLLM)
    config = OmegaConf.create(
        {
            "gpt": {
                "default_api": "open_ai",
                "llm_model": {"open_ai": "gpt-6-sol", "open_ai_cheap": "gpt-6-luna"},
                "max_chars": {"employees": 60000},
            }
        }
    )
    context = SimpleNamespace(config=config, log=SimpleNamespace(info=lambda *args: None, warning=lambda *args: None))
    return mod.build_ticker_employees(
        context,
        "AAA",
        "0000000001",
        since=None,
        done_dates=done_dates,
        manual=manual or {},
        scope=scope or mod.EdgarScope(identity),
    )


def test_decided_date_skips_filing_before_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    filings = [
        Filing("stored", "2024-03-01", "We had 40,000 employees."),
        Filing("new", "2025-03-01", "We had 41,000 employees."),
    ]
    filings[0].html = lambda: pytest.fail("stored filing text was fetched")
    result = build(
        monkeypatch,
        filings,
        {"new": answer(41000, "We had 41,000 employees.")},
        done_dates=frozenset({pd.Timestamp("2024-03-01")}),
    )
    assert result.frame["as_of"].tolist() == [pd.Timestamp("2025-03-01")]
    assert [outcome["accession_number"] for outcome in result.outcomes] == ["new"]
    print("\nSANITY: a filing date that already has a row (count or NULL) is skipped before filing text and the LLM.")


def test_manual_roster_replaces_the_llm(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    (tmp_path / "sec").mkdir()
    roster = {
        "_README": "fixture",
        "AAA": [
            {"accession_number": "hand", "filing_date": "2023-03-01", "employees": 1234, "status": "saved"},
            {"accession_number": "hand-null", "filing_date": "2024-03-01", "employees": None, "status": "no_headcount"},
        ],
    }
    (tmp_path / mod.MANUAL_ROSTER).write_text(json.dumps(roster), encoding="utf-8")
    manual = mod.load_manual_roster(str(tmp_path))
    assert set(manual) == {"hand", "hand-null"} and manual["hand"]["ticker"] == "AAA"
    assert mod.load_manual_roster(str(tmp_path / "absent")) == {}
    filings = [
        Filing("hand", "2023-03-01", "unused"),
        Filing("hand-null", "2024-03-01", "unused"),
        Filing("new", "2025-03-01", "We had 41,000 employees."),
    ]
    for filing in filings[:2]:
        filing.html = lambda: pytest.fail("a manual roster filing was read or sent to the LLM")
    result = build(monkeypatch, filings, {"new": answer(41000, "We had 41,000 employees.")}, manual=manual)
    employees = result.frame["employees"].tolist()
    assert employees[0] == 1234.0 and pd.isna(employees[1]) and employees[2] == 41000.0
    assert [(outcome["source"], outcome["status"]) for outcome in result.outcomes] == [
        ("manual", "saved"),
        ("manual", "no_headcount"),
        ("llm", "found"),
    ]
    print("\nSANITY: a manual roster filing takes its count (or NULL) from configs/sec without text or LLM; others go to the LLM.")


def decide(
    text: str,
    *,
    total: int | None = None,
    full_time: int | None = None,
    part_time: int | None = None,
    quote: str | None = None,
    is_fte: bool = False,
    includes_contractors: bool = False,
) -> mod.EmployeeDecision:
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
    return mod.decide_employee_answer(reply, text)


def supported(count: int | None, text: str, quote: str | None = None) -> int | None:
    """The decided total for a fake answer claiming `count` as the total, quoting all of `text` unless a narrower quote is given."""
    return decide(text, total=count, quote=quote).employees_total


def components(decision: mod.EmployeeDecision) -> tuple[int | None, int | None, int | None, str | None, str]:
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
    assert components(mod.decide_employee_answer(answer(None, None, status="image_only"), text)) == (None, None, None, None, "image_only")
    assert components(mod.decide_employee_answer(answer(None, None, status="not_disclosed"), text))[4] == "not_disclosed"
    assert components(mod.decide_employee_answer(answer(None, None, status="incorporated_by_reference"), text))[4] == "incorporated"
    assert components(mod.decide_employee_answer(answer(None, None, status="ambiguous"), text))[4] == "ambiguous"
    assert components(mod.decide_employee_answer(answer(None, None, status="found"), text))[4] == "ambiguous"
    assert set(mod.DECIDED_STATUSES) == {"found", "not_disclosed", "image_only", "incorporated", "ambiguous", "unsupported"}
    assert set(mod.BASES) == {"total", "full_part", "full_time_only", "fte", "total_incl_contractors"}
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
    assert components(mod.decide_employee_answer(reply, DLTR_2021)) == (199327, 60217, 139110, "full_part", "found")
    # The full-time count quoted with the Total row only is not supported; the total stays.
    wrong_row = answer(199327, "Total 125,865 71,046 2,416 199,327", full_time=60217, full_time_quote="Total 125,865 71,046 2,416 199,327")
    decision = mod.decide_employee_answer(wrong_row, DLTR_2021)
    assert components(decision) == (199327, None, None, "total", "found") and decision.rejected == ("full_time: not_stated",)
    # A bounded total is stored as stated, never combined with an exact component.
    bounded = answer(
        199300, "we employed more than 199,300 associates", full_time=60217, full_time_quote="Full-time Associates 27,952 29,862 2,403 60,217"
    )
    assert components(mod.decide_employee_answer(bounded, DLTR_2021)) == (199350, 60217, None, "full_part", "found")
    held_for_sale = answer(
        214710, "Total 143,283 68,936 2,491 214,710", full_time=64434, full_time_quote="Full-time Associates 33,233 28,712 2,489 64,434"
    )
    assert components(mod.decide_employee_answer(held_for_sale, DLTR_2025))[:2] == (214710, 64434)
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


def test_missing_primary_document_reads_full_submission(monkeypatch):
    submission = "<SEC-DOCUMENT><TEXT>As of December 31, 1999, we had 1,234 employees.</TEXT></SEC-DOCUMENT>"
    filing = NoPrimaryDocumentFiling("no-primary", "2000-03-01", submission)
    assert mod.filing_body_text(filing) == "As of December 31, 1999, we had 1,234 employees."
    result = build(monkeypatch, [filing], {"no-primary": answer(1234, "we had 1,234 employees.")})
    assert result.frame["employees"].tolist() == [1234.0]
    unreadable = NoPrimaryDocumentFiling("unreadable", "2000-03-01", "")
    with pytest.raises(ValueError, match="filing text unavailable"):
        build(monkeypatch, [unreadable], {})
    print(
        "\nSANITY: a filing whose index lists no primary document no longer aborts the run with AttributeError; "
        "its full submission supplies the source text, and a truly empty one fails only its ticker."
    )


FILLER = "The Company designs and sells measurement instruments to laboratories in many markets. "  # no workforce word


def test_ix_header_is_stripped_and_the_workforce_sentence_survives(monkeypatch):
    """Agilent FY2019 shape: a hidden inline-XBRL header precedes the visible text."""
    junk = " ".join(f"<ix:nonNumeric name='us-gaap:Fact{i}' contextRef='c{i}'>0001090872 2019-10-31 {i * 37}</ix:nonNumeric>" for i in range(400))
    header = f"<div style='display:none'><ix:header><ix:hidden>{junk}</ix:hidden><ix:references/></ix:header></div>"
    statement = "As of October 31, 2019, we employed approximately 16,300 people worldwide."
    filing = Filing("a-2019", "2019-12-19", "")
    filing.html = lambda: f"<html><body>{header}<p>{FILLER * 150}</p><p>{statement}</p></body></html>"
    text = mod.filing_body_text(filing)
    assert len(junk) > 20_000 and "us-gaap" not in text and "0001090872" not in text
    assert statement in mod.employee_excerpt(text, 10_000)
    result = build(monkeypatch, [filing], {"a-2019": answer(16300, statement)})
    assert result.frame["employees"].tolist() == [16300.0]
    print("\nSANITY: >20k chars of hidden ix:header values are dropped before excerpting; Agilent's 16,300 people sentence is sent and saved.")


def test_workforce_number_windows_come_before_generic_context():
    persons = "As of December 31, 2019, 502 persons were employed by the Company."
    kim = f"{FILLER * 120}{persons} {FILLER * 20}"
    assert persons in mod.employee_excerpt(kim, 12_000)
    noise = "".join(f"Our employees value safety and training, item {i}. {FILLER * 15}" for i in range(130))
    workforce = "At December 31, 2020, we had 41,000 employees worldwide."
    many_hits = f"{noise}{workforce} {FILLER * 10}"
    excerpt = mod.employee_excerpt(many_hits, 60_000)
    assert workforce in excerpt and len(excerpt) <= 60_000
    assert len(mod._CONTEXT_RE.findall(many_hits)) > 120
    for headcount in ("Headcount 256,981 at year end.", "Our workforce of more than 10,000 people.", "We employ more than 25,000 people."):
        assert headcount in mod.employee_excerpt(f"{FILLER * 120}{headcount} {FILLER * 10}", 9_000)
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


def test_exhibit_fallback_reads_the_annual_report_exhibit(monkeypatch):
    wy = "The company has 44,800 employees, of whom 43,800 are employed in its timber-based businesses."
    primary = (
        f"{FILLER * 30}The following information is included in the 1999 Annual Report to Stockholders and is incorporated "
        "herein by reference: 1. Segment information--Pages 75 and 76. 2. The number of persons employed by the registrant--Page 49."
    )
    filing = Filing("wy-2000", "2000-03-10", primary)
    filing.attachments = [UnreadAttachment("EX-21", ""), Attachment("EX-13", f"<p>{FILLER * 40}</p><p>{wy}</p>")]
    chosen = mod.employee_text(filing)
    assert chosen.source_document == "EX-13" and wy in chosen.text
    no_number = Filing("wy-bare", "2000-03-10", f"{FILLER * 30}Employees are described in the Annual Report.")
    no_number.attachments = [Attachment("EX-99", f"<p>{wy}</p>", description="1999 Annual Report to Shareholders")]
    assert mod.employee_text(no_number).source_document == "EX-99"
    task = mod._employee_task(0, "AAA", edgar_driver.FilingStamp.of(filing, "0000000001"), 60_000)
    assert task.meta["source_document"] == "EX-13" and wy in str(task.meta["source_text"])
    result = build(monkeypatch, [filing], {"wy-2000": answer(44800, "The company has 44,800 employees")})
    assert result.frame["employees"].tolist() == [44800.0]
    assert mod.is_annual_report_exhibit("EX-99", "Annual Report to Shareholders") and not mod.is_annual_report_exhibit("EX-99.1", "Press release")
    stated = Filing("own", "2025-03-01", "We had 41,000 employees.")
    stated.attachments = [UnreadAttachment("EX-13", "")]
    assert mod.employee_text(stated).source_document == "primary"
    bare = Filing("bare", "2025-03-01", f"{FILLER * 30}No exhibit carries a workforce figure.")
    bare.attachments = [Attachment("EX-13", f"<p>{FILLER * 10}</p>")]
    assert mod.employee_text(bare).source_document == "primary"
    print(
        "\nSANITY: a primary document that incorporates its employee count by reference sends the EX-13 text (44,800 saved, "
        "source EX-13), as does an EX-99 annual report; a primary that states its own count reads no exhibit; an exhibit with no workforce number is not used."
    )


def test_offline_task_builder_and_decision_for_cached_text():
    """The P3 benchmark path: a task from cached document text (no Filing, no network), then the pure decision."""
    document = mod.EmployeeText(f"{FILLER * 200}{LOW_2025} {FILLER * 20}", "primary")
    task = mod.employee_task(
        7, "LOW", document, 9_000, filed=pd.Timestamp("2025-03-24"), accession="0000060667-25-000049", report_date=pd.Timestamp("2025-01-31")
    )
    source_text = str(task.meta["source_text"])
    assert task.seq == 7 and task.schema is mod.EmployeeAnswer and task.meta["source_document"] == "primary"
    assert (
        LOW_2025 in source_text and task.payload.endswith(source_text) and "Source document: primary" in task.payload and len(task.payload) <= 9_000
    )
    quote = "Lowe’s employed approximately 161,000 full-time associates and 109,000 part-time associates"
    reply = answer(None, None, full_time=161000, full_time_quote=quote, part_time=109000, part_time_quote=quote)
    decision = mod.decide_employee_answer(reply, source_text)
    assert (decision.employees_total, decision.basis, decision.quotes) == (270000, "full_part", (("full_time", quote), ("part_time", quote)))
    with pytest.raises(ValueError, match="filing text unavailable"):
        mod.employee_task(0, "LOW", mod.EmployeeText(" ", "primary"), 9_000, filed=pd.Timestamp("2025-03-24"), accession="x", report_date=None)
    print("\nSANITY: a cached LOW text builds a 9,000-char task offline and its fake answer decides to 270,000 full_part with both quotes kept.")


def test_prompt_states_the_component_rules():
    prompt = (Path(mod.__file__).parents[3] / "gpt_extract" / "prompt_templates" / "employees_system_prompt.md").read_text(encoding="utf-8")
    for rule in ("verbatim", "respectively", "held for sale", "contractors", "has no employees", " ... ", "full_time", "part_time", "is_fte"):
        assert rule in prompt, rule
    print("\nSANITY: the employee prompt carries the verbatim, row, respectively, held-for-sale, contractor, subsidiary-sum and ` ... ` rules.")


def test_legacy_annual_form_uses_dated_registrant_scope():
    assert combine_for(mod.HEADCOUNT_FORMS) is Combine.SPLIT
    print("\nSANITY: 10-K405 joins the dated annual registrant scope used with entity lineage and symbol tenure.")


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


def test_legacy_form_table_total_and_image_null_row(monkeypatch):
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
    result = build(monkeypatch, filings, answers)
    employees = result.frame["employees"].tolist()
    assert employees[:4] == [9603.0, 10211.0, 74042.0, 199327.0] and pd.isna(employees[4])
    assert result.frame["as_of"].tolist() == [pd.Timestamp(day) for day in ("2001-12-21", "2002-12-19", "2014-09-09", "2021-03-16", "2024-02-26")]
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
        "is a NULL row; DLTR's FT/PT rows give a 199,327 full_part total, which fills the interim `employees` column."
    )


def test_same_day_amendment_uses_later_supported_count(monkeypatch):
    filings = [
        Filing("original", "2024-03-01", "We had 40,000 employees.", report="2023-12-31"),
        Filing("amendment", "2024-03-01", "We had 41,000 employees.", form="10-K/A", report="2023-12-31"),
    ]
    result = build(
        monkeypatch,
        filings,
        {
            "original": answer(40000, "We had 40,000 employees."),
            "amendment": answer(41000, "We had 41,000 employees."),
        },
    )
    assert result.frame["employees"].tolist() == [41000.0]
    assert [row["status"] for row in result.outcomes] == ["superseded", "found"]
    print("\nSANITY: a same-date amendment replaces the original value at the one-row filing-date grain.")


def test_missing_report_period_keeps_filing_date_and_a_foreign_cik_is_skipped_and_counted(monkeypatch):
    """AC-008 (employees): a listed filing whose CIK the identity layer does not tie to the ticker is skipped and counted, never raised."""
    filing = Filing("no-period", "2024-03-01", "We had 40,000 employees.", report=None)
    result = build(monkeypatch, [filing], {"no-period": answer(40000, "We had 40,000 employees.")})
    assert result.frame["as_of"].tolist() == [pd.Timestamp("2024-03-01")]
    foreign = Filing("foreign", "2025-03-01", "We had 9 employees.")
    foreign.cik = "0000000002"
    foreign.html = lambda: pytest.fail("a foreign filing was read or sent to the LLM")
    identity = SimpleNamespace(owns=lambda ticker, cik: str(cik).zfill(10) == "0000000001", filing_scope=lambda t: FilingScope.roster_only(t, "1"))
    scope = mod.EdgarScope(identity)
    result = build(monkeypatch, [filing, foreign], {"no-period": answer(40000, "We had 40,000 employees.")}, scope=scope)
    assert result.frame["as_of"].tolist() == [pd.Timestamp("2024-03-01")]
    assert scope.guard.skipped == 1
    print("\nSANITY: optional report metadata does not block filing-date storage; a foreign CIK is skipped and counted (1), not raised.")


def run_context(saved: list[pd.DataFrame], stored: pd.DataFrame | None, warnings: list[str] | None = None) -> SimpleNamespace:
    logged = warnings if warnings is not None else []
    return SimpleNamespace(
        store=SimpleNamespace(save=lambda table, frame: saved.append(frame), load=lambda *args, **kwargs: stored),
        config=SimpleNamespace(data_extract=SimpleNamespace(fundamentals_workers=1)),
        config_dir="configs",
        ensure_edgar_identity=lambda: None,
        log=SimpleNamespace(info=lambda *args: None, warning=lambda msg, *args: logged.append(msg % args if args else msg)),
    )


def patch_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "load_cik_mapping", lambda *args: pd.DataFrame([{"ticker": "AAA", "cik": "0000000001"}]))
    monkeypatch.setattr(mod, "load_identity", lambda *args: SimpleNamespace())
    monkeypatch.setattr(mod, "load_manual_roster", lambda *args: {})


def test_table_rows_decide_what_is_done_and_full_rereads_them(monkeypatch):
    # 1030 ambiguous filings once lived only in a side manifest and went back to the LLM nightly.
    saved: list[pd.DataFrame] = []
    decided = pd.DataFrame([{"ticker": "AAA", "as_of": pd.Timestamp("2024-02-26")}])
    patch_run(monkeypatch)
    seen: list[dict] = []

    def fake_build(*args, **kwargs):
        seen.append(kwargs)
        frame = pd.DataFrame([{"ticker": "AAA", "as_of": pd.Timestamp("2025-02-26"), "employees": float("nan")}], columns=mod.FRAME_COLUMNS)
        return mod.EmployeeTickerResult(
            frame, [{"ticker": "AAA", "accession_number": "unclear", "source": "llm", "status": "ambiguous", "count": None}]
        )

    monkeypatch.setattr(mod, "build_ticker_employees", fake_build)
    mod.fetch_fundamentals_employees(run_context(saved, decided), ["AAA"], 15)
    assert seen[0]["done_dates"] == frozenset({pd.Timestamp("2024-02-26")})
    assert seen[0]["since"] == pd.Timestamp.today().normalize() - pd.DateOffset(years=15)
    assert pd.isna(saved[0]["employees"].iloc[0])
    mod.fetch_fundamentals_employees(run_context(saved, decided), ["AAA"], 15, full=True)
    assert seen[1]["done_dates"] == frozenset()
    print("\nSANITY: an ambiguous filing is stored as a NULL row and completes the run; dates with rows are skipped, `--full` re-reads them.")


def test_a_failed_ticker_is_logged_and_the_run_exits_cleanly(monkeypatch):
    saved: list[pd.DataFrame] = []
    warnings: list[str] = []
    patch_run(monkeypatch)
    monkeypatch.setattr(mod, "build_ticker_employees", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("EDGAR down")))

    mod.fetch_fundamentals_employees(run_context(saved, None, warnings), ["AAA"], 15)

    assert saved == []
    assert any("1/1 ticker(s) not read, listed again next run: AAA" in w for w in warnings), warnings
    print(
        "\nSANITY: a failed ticker saves nothing, is named in the coverage log and the run returns (exit 0); it has no rows, so it retries next run."
    )


INVALID_JSON = (
    "ValidationError: 1 validation error for EmployeeAnswer\n  Invalid JSON: expected value at line 1 column 2 "
    "[type=json_invalid, input_value=' ensembling?', input_type=str]"
)


def test_one_invalid_llm_answer_fails_only_its_own_filing_date(monkeypatch):
    filings = [Filing(f"good-{year}", f"{year}-03-01", f"We had {10 * year:,} employees.") for year in range(2001, 2031)]
    filings += [
        Filing("bad", "2031-03-01", "We had 9,999 employees."),
        Filing("bad-day-original", "2032-03-01", "We had 7,000 employees."),
        Filing("bad-day-amendment", "2032-03-01", "We had 7,100 employees.", form="10-K/A"),
    ]
    answers: dict[str, mod.EmployeeAnswer | str] = {
        f"good-{year}": answer(10 * year, f"We had {10 * year:,} employees.") for year in range(2001, 2031)
    }
    answers |= {"bad": INVALID_JSON, "bad-day-original": answer(7000, "We had 7,000 employees."), "bad-day-amendment": INVALID_JSON}
    sent: list[list[str]] = []

    class FakeLLM:
        """No LLM call: each accession gets its fixture answer, or a failed result carrying the error."""

        def __init__(self, *args, **kwargs):
            self.tasks = []

        def submit(self, task):
            self.tasks.append(task)

        def run(self):
            sent.append([task.meta["stamp"].accession_number for task in self.tasks])
            results = []
            for task in self.tasks:
                reply = answers[task.meta["stamp"].accession_number]
                failed = isinstance(reply, str)
                results.append(LlmResult(seq=task.seq, task=task, parsed=None if failed else reply, error=reply if failed else None))
            return results

    stored: list[pd.DataFrame] = []
    warnings: list[str] = []
    context = SimpleNamespace(
        store=SimpleNamespace(
            save=lambda table, frame: stored.append(frame),
            load=lambda *args, **kwargs: pd.concat(stored, ignore_index=True) if stored else None,
        ),
        config=OmegaConf.create(
            {
                "data_extract": {"fundamentals_workers": 1},
                "gpt": {"default_api": "open_ai", "llm_model": {"open_ai": "a", "open_ai_cheap": "b"}, "max_chars": {"employees": 60000}},
            }
        ),
        config_dir="configs",
        ensure_edgar_identity=lambda: None,
        log=SimpleNamespace(info=lambda *args: None, warning=lambda msg, *args: warnings.append(msg % args if args else msg)),
    )
    patch_run(monkeypatch)
    identity = SimpleNamespace(owns=lambda ticker, cik: True, filing_scope=lambda ticker: FilingScope.roster_only(ticker, "0000000001"))
    monkeypatch.setattr(mod, "load_identity", lambda *args: identity)
    monkeypatch.setattr(edgar_driver, "resolve_registrant_filings", lambda *args, **kwargs: filings)
    monkeypatch.setattr(mod, "LLMExtractor", FakeLLM)

    mod.fetch_fundamentals_employees(context, ["AAA"], 50)

    night_one = pd.concat(stored, ignore_index=True) if stored else pd.DataFrame(columns=mod.FRAME_COLUMNS)
    assert len(sent[0]) == 33
    assert night_one["as_of"].tolist() == [pd.Timestamp(f"{year}-03-01") for year in range(2001, 2031)]
    assert night_one["employees"].tolist() == [float(10 * year) for year in range(2001, 2031)]
    assert any("bad" in w and "listed again next run" in w for w in warnings), warnings

    mod.fetch_fundamentals_employees(context, ["AAA"], 50)

    assert sorted(sent[1]) == ["bad", "bad-day-amendment", "bad-day-original"]
    print(
        "\nSANITY: one invalid-JSON answer among 33 keeps the 30 good counts stored and marks nothing; the failed filing "
        "(and the other filing sharing its date) is the only work listed, and re-sent, on the next night."
    )


def test_missing_roster_cik_cannot_certify_coverage(monkeypatch):
    context = SimpleNamespace(ensure_edgar_identity=lambda: None)
    monkeypatch.setattr(mod, "load_cik_mapping", lambda *args: pd.DataFrame(columns=["ticker", "cik"]))
    with pytest.raises(ValueError, match="no roster CIK for AAA"):
        mod.fetch_fundamentals_employees(context, ["AAA"], 15)
    print("\nSANITY: an unresolved requested ticker cannot be reported as fully covered.")


def test_cli_still_dispatches_explicit_full_replay(monkeypatch):
    calls = []
    context = SimpleNamespace()
    config = SimpleNamespace(data_extract=SimpleNamespace(years_history=31))
    monkeypatch.setattr(cli_mod, "get_config_context", lambda path, **kwargs: (config, context))
    monkeypatch.setattr(cli_mod, "_tickers", lambda ctx, names: ["AAA"])
    monkeypatch.setattr(
        cli_mod,
        "fetch_fundamentals_employees",
        lambda ctx, tickers, years_history, full=False: calls.append((tickers, years_history, full)),
    )
    result = CliRunner().invoke(cli_mod.cli, ["fundamentals-employees", "-t", "AAA", "--full"])
    assert result.exit_code == 0, result.output
    assert calls == [(["AAA"], 31, True)]
    print("\nSANITY: the existing CLI runs a ticker-scoped full employee replay when explicitly requested.")
