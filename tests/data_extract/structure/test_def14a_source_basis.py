"""Exact source/category and guarded unit corrections for focused proxy answers."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data_extract.utils.schemas.def14a_schema import Def14AExtract, ExecutivePayExtract, OwnershipExtract
from src.data_extract.utils.structure.def14a.carve import prepare_def14a_payloads
from src.data_extract.utils.structure.def14a.postprocess import (
    append_answer,
    archived_answers,
    postprocess_section,
    refresh_archive,
    source_for_answer,
)
from src.data_extract.utils.structure.def14a.retry import SECTIONS
from src.data_extract.utils.structure.def14a.tables import SCT, classify_table
from src.data_store.schema import Tables
from src.data_store.store import DataStore


def _process(domain: str, rows: list[dict], source: str, saved: list[dict] | None = None) -> pd.DataFrame:
    section = next(item for item in SECTIONS if item.key == domain)
    stamp = {"ticker": "X", "accession_number": "a", "as_of": pd.Timestamp("2025-04-01"), "cik": "1"}
    payload = {"currency": "USD", "amount_scale": "units", "compensation": rows}
    if domain == "ownership":
        payload = {"percent_unit": "fraction", "shares_scale": "units", "holders": rows}
    old = pd.DataFrame([{**stamp, **row, "retry": 2} for row in saved]) if saved else pd.DataFrame()
    answer = section.schema.model_validate(payload)
    result = postprocess_section(section, answer, stamp, {section.table: old}, {}, source)
    return result.frames[section.table]


def test_performance_year_table_is_not_a_regulatory_sct() -> None:
    header = ["Name", "Year", "Annual Compensation (For Performance Year) Salary", "Annual Compensation (For Performance Year) Total"]
    assert SCT not in classify_table(header, [["James Dimon", "2024", "1,500,000", "39,000,000"]])
    assert SCT in classify_table(
        ["Name", "Year", "Salary ($)", "Stock awards ($)", "Total ($)"], [["James Dimon", "2024", "1,500,000", "29,500,000", "37,683,462"]]
    )
    print("SANITY: performance-year decision pay is rejected; an actual SCT remains eligible.")


@pytest.mark.parametrize("beneficial,total", [(430708, 3295477), (1214259, 1996235), (6464764, 7186564)])
def test_ownership_uses_beneficial_column_through_saved_alias(beneficial: int, total: int) -> None:
    source = f"Name\tCommon Stock (#)\tTotal beneficial ownership (#)\tAdditional underlying stock units (#)\tTotal (#)\nJane Roe\t{beneficial:,}\t{beneficial:,}\t{total - beneficial:,}\t{total:,}"
    new = {"holder_name": "Jane Roe", "holder_type": "director_officer", "shares": total}
    old = {**new, "holder_name": "JANE ROE, 123 Main St, Boston MA 02110"}
    result = _process("ownership", [new], source, [old])
    assert len(result) == 1 and result.iloc[0].shares == beneficial and result.iloc[0].retry == 2
    print("SANITY: same-filing canonical holder retains beneficial shares, excludes additional units, and preserves retry2.")


def test_source_confirmed_pay_corrects_saved_nonnull_and_reconciliation() -> None:
    source = "Name\tYear\tSalary ($)\tBonus ($)\tStock awards ($)\tPension value ($)\tAll other compensation ($)\tTotal ($)\nJames Dimon\t2014\t1,500,000\t7,400,000\t18,500,000\t55,816\t245,893\t27,701,709"
    new = {
        "name": "James Dimon",
        "title": "CEO",
        "fiscal_year": 2014,
        "salary": 1500000,
        "bonus": 7400000,
        "stock_awards": 18500000,
        "pension_change": 55816,
        "other_compensation": 245893,
        "total": 27701709,
    }
    result = _process("executive_pay", [new], source, [{**new, "total": 20000000}]).iloc[0]
    assert result.total == 27701709 and result.reconciles == 1
    print("SANITY: exact SCT source permits the correct new total to replace saved CD&A pay; reconciliation refreshes.")


@pytest.mark.parametrize(
    "domain,field,wrong,label",
    [
        ("executive_pay", "bonus", 69721, "Other Annual Compensation ($)"),
        ("executive_pay", "total", 5321228, "Total Annual Compensation ($)"),
        ("director_pay", "option_awards", 232078, "Stock awards Additional accounting expense ($)"),
    ],
)
def test_wrong_category_is_cleared_only_when_the_source_identifies_it(domain: str, field: str, wrong: int, label: str) -> None:
    source = f"Name\tYear\tSalary ($)\tBonus ($)\t{label}\nJane Roe\t1995\t570,000\t\t{wrong:,}"
    new = {"name": "Jane Roe", "fiscal_year": 1995, field: None}
    result = _process(domain, [new], source, [{**new, field: wrong}]).iloc[0]
    assert pd.isna(result[field])
    unchanged = _process(domain, [new], "", [{**new, field: wrong}]).iloc[0]
    assert unchanged[field] == wrong
    print("SANITY: an identified wrong source category clears its old value; source-less answers preserve it.")


@pytest.mark.parametrize("bad", [1500, 1500000000])
def test_thousandfold_pay_scale_needs_same_person_source_median(bad: int) -> None:
    source = "Name\tYear\tSalary ($)\nJane Roe\t2024\t1,500,000\nJane Roe\t2023\t1,500,000\nJane Roe\t2022\t1,500,000"
    rows = [{"name": "Jane Roe", "title": "CEO", "fiscal_year": year, "salary": bad if year == 2024 else 1500000} for year in (2024, 2023, 2022)]
    result = _process("executive_pay", rows, source).set_index("fiscal_year")
    assert result.loc[2024, "salary"] == 1500000
    again = _process("executive_pay", result.reset_index().to_dict("records"), source).set_index("fiscal_year")
    assert again.loc[2024, "salary"] == 1500000
    print("SANITY: source-backed 0.001/1000 unit error rebases once and replay does not double-convert.")


@pytest.mark.parametrize("value", [0, -1500, 1500000000, 3000000])
def test_valid_source_changes_zero_and_negative_are_never_smoothed(value: int) -> None:
    source = f"Name\tYear\tSalary ($)\nJane Roe\t2024\t{value:,}\nJane Roe\t2023\t1,500,000\nJane Roe\t2022\t1,500,000"
    rows = [{"name": "Jane Roe", "fiscal_year": year, "salary": value if year == 2024 else 1500000} for year in (2024, 2023, 2022)]
    result = _process("executive_pay", rows, source).set_index("fiscal_year")
    assert result.loc[2024, "salary"] == value
    print("SANITY: exact source value survives even at a thousandfold jump, zero or negative.")


def test_fixed_width_sct_keeps_the_blank_bonus_column() -> None:
    raw = """SUMMARY COMPENSATION TABLE
<TABLE><CAPTION>
NAME                   YEAR   SALARY     BONUS     OTHER ANNUAL
                                                   COMPENSATION
<S>                    <C>    <C>        <C>       <C>
Forrest E. Hoglund      1995   $570,000              $69,721
</TABLE>"""
    source = prepare_def14a_payloads(raw, raw)["executive_pay"]
    assert "ORIGINAL COLUMN ALIGNMENT" in source and "\t\t" in source
    row = _process(
        "executive_pay",
        [{"name": "Forrest E. Hoglund", "fiscal_year": 1995, "bonus": None}],
        source,
        [{"name": "Forrest E. Hoglund", "fiscal_year": 1995, "bonus": 69721}],
    ).iloc[0]
    assert pd.isna(row.bonus)
    print("SANITY: SGML column starts preserve a blank bonus; other annual compensation cannot slide into it.")


@pytest.mark.parametrize("source_name", ["Jane Roe Jr.", "Jane Roe Sr.", "Jane Roe Advisors LLC"])
def test_source_prefixes_and_person_suffixes_do_not_authorize_correction(source_name: str) -> None:
    source = f"Name\tTotal beneficial ownership\tAdditional underlying stock units\tTotal\n{source_name}\t100\t900\t1000"
    row = {"holder_name": "Jane Roe", "holder_type": "director_officer", "shares": 1000}
    assert _process("ownership", [row], source).iloc[0].shares == 1000
    print("SANITY: a related name, affiliate, Jr. or Sr. cannot supply another holder's corrected quantity.")


@pytest.mark.parametrize(
    "source",
    [
        "Name\tYear\tSalary ($)\nJane Roe\t2023\t1,500,000",
        "Name\tYear\tSalary ($)\nJane Roe\t2024\t1,500,000\nJane Roe\t2024\t1,600,000",
        "Amounts (in thousands)\nName\tYear\tSalary ($)\nJane Roe\t2024\t1,500\nJane Roe\t2023\t1,500",
    ],
)
def test_wrong_year_conflicting_rows_and_unattached_unit_notes_abstain(source: str) -> None:
    new = {"name": "Jane Roe", "fiscal_year": 2024, "salary": 1500000}
    row = _process("executive_pay", [new], source, [{**new, "salary": 3000000}]).iloc[0]
    assert row.salary == 3000000
    print("SANITY: ambiguous rows, a different year, or unattached unit notes preserve existing facts.")


def test_table_local_scale_does_not_change_another_monetary_column() -> None:
    source = "Name\tYear\tSalary ($ in thousands)\tBonus ($)\nJane Roe\t2024\t1,500\t50,000"
    new = {"name": "Jane Roe", "fiscal_year": 2024, "salary": 1500000, "bonus": 50000}
    row = _process("executive_pay", [new], source, [{**new, "salary": 1500}]).iloc[0]
    assert row.salary == 1500000 and row.bonus == 50000
    print("SANITY: a source-labelled thousands column is corrected while a whole-dollar neighbor keeps its own unit.")


def test_foreign_currency_never_replaces_saved_dollar_pay() -> None:
    section = next(item for item in SECTIONS if item.key == "executive_pay")
    stamp = {"ticker": "X", "accession_number": "a", "as_of": pd.Timestamp("2025-04-01"), "cik": "1"}
    source = "Name\tYear\tSalary (EUR)\nJane Roe\t2024\t1,500,000"
    answer = ExecutivePayExtract(currency="EUR", amount_scale="units", compensation=[{"name": "Jane Roe", "fiscal_year": 2024, "salary": 1500000}])
    old = pd.DataFrame([{**stamp, "name": "Jane Roe", "fiscal_year": 2024, "salary": 2000000, "retry": 2}])
    row = postprocess_section(section, answer, stamp, {section.table: old}, {}, source).frames[section.table].iloc[0]
    assert row.salary == 2000000
    print("SANITY: unsupported foreign money cannot overwrite stored dollars through source correction.")


def test_exact_archived_source_replay_persists_a_null_category(sqlite_store: DataStore) -> None:
    section = next(item for item in SECTIONS if item.key == "executive_pay")
    stamp = {"ticker": "X", "accession_number": "a", "as_of": pd.Timestamp("2025-04-01"), "cik": "1"}
    source = "Name\tYear\tSalary ($)\tBonus ($)\tOther Annual Compensation ($)\nJane Roe\t1995\t570,000\t\t69,721"
    answer = ExecutivePayExtract(currency="USD", amount_scale="units", compensation=[{"name": "Jane Roe", "fiscal_year": 1995, "salary": 570000}])
    raw = append_answer(None, section, answer, source)
    original = raw
    saved = pd.DataFrame([{**stamp, "name": "Jane Roe", "fiscal_year": 1995, "salary": 570000, "bonus": 69721, "retry": 2}])
    sqlite_store.save(section.table, saved)
    for _ in range(2):
        replay = archived_answers(raw)[section.key][0]
        assert source_for_answer(raw, section, replay) == source
        saved = sqlite_store.load(section.table, where={"ticker": "X", "accession_number": "a"})
        result = postprocess_section(section, replay, stamp, {section.table: saved}, {}, source_for_answer(raw, section, replay))
        sqlite_store.save(section.table, result.frames[section.table])
        stored = sqlite_store.load(section.table, where={"ticker": "X", "accession_number": "a"})
        assert stored is not None and pd.isna(stored.iloc[0].bonus) and stored.iloc[0].retry == 2
    assert raw == original
    conflicting = append_answer(raw, section, answer, "a different source")
    assert source_for_answer(conflicting, section, answer) == ""
    print("SANITY: free exact-source replay clears and persists only the wrong category, is idempotent at retry2, and retains original raw bytes.")


def test_source_foreign_column_cannot_confirm_an_incorrect_usd_answer() -> None:
    source = "Name\tYear\tSalary (EUR)\nJane Roe\t2024\t1,500,000"
    new = {"name": "Jane Roe", "fiscal_year": 2024, "salary": 1500000}
    row = _process("executive_pay", [new], source, [{**new, "salary": 2000000}]).iloc[0]
    assert row.salary == 2000000
    print("SANITY: a source-labelled foreign amount cannot confirm dollars even when the answer incorrectly declares USD.")


def test_proven_total_clear_reaches_saved_parent_and_archive(sqlite_store: DataStore) -> None:
    section = next(item for item in SECTIONS if item.key == "executive_pay")
    stamp = {"ticker": "X", "accession_number": "a", "as_of": pd.Timestamp("2025-04-01"), "cik": "1"}
    raw_source = """SUMMARY COMPENSATION TABLE
ANNUAL COMPENSATION
<TABLE><CAPTION>
                                                                      OTHER ANNUAL        TOTAL ANNUAL
NAME AND PRINCIPAL POSITION      YEAR         SALARY          BONUS   COMPENSATION        COMPENSATION
<S>                              <C>      <C>            <C>               <C>              <C>
Jane Roe                         1995     $2,000,000     $3,250,000        $71,228          $5,321,228
</TABLE>"""
    source = prepare_def14a_payloads(raw_source, raw_source)["executive_pay"]
    answer = ExecutivePayExtract(currency="USD", amount_scale="units", compensation=[{"name": "Jane Roe", "fiscal_year": 1995, "salary": 2000000}])
    old = pd.DataFrame([{**stamp, "name": "Jane Roe", "fiscal_year": 1995, "salary": 2000000, "total": 5321228, "retry": 2}])
    parent = {**stamp, "ceo_name_proxy": "Jane Roe", "ceo_total_comp": 5321228}
    sqlite_store.save(Tables.def14a_llm, pd.DataFrame([parent]))
    result = postprocess_section(section, answer, stamp, {section.table: old}, parent, source)
    for table, frame in result.frames.items():
        sqlite_store.save(table, frame)
    stored = sqlite_store.load(Tables.def14a_llm, where={"ticker": "X", "accession_number": "a"})
    assert stored is not None and pd.isna(stored.iloc[0].ceo_total_comp)
    original = Def14AExtract.model_validate(
        {"ceo_name": "Jane Roe", "compensation": [{"name": "Jane Roe", "title": "CEO", "fiscal_year": 1995, "total_compensation_usd": 5321228}]}
    ).model_dump_json()
    facade = Def14AExtract.model_validate_json(refresh_archive(original, section, result.legacy, result.corrections))
    assert facade.compensation[0].total_compensation_usd is None
    print("SANITY: a proven annual subtotal clears child, saved CEO total and compatibility facade together.")


def test_proven_insider_basis_updates_saved_parent_and_archive(sqlite_store: DataStore) -> None:
    section = next(item for item in SECTIONS if item.key == "ownership")
    stamp = {"ticker": "X", "accession_number": "a", "as_of": pd.Timestamp("2025-04-01"), "cik": "1"}
    source = "Name\tTotal beneficial ownership\tAdditional underlying stock units\tTotal\nJane Roe\t6,464,764\t721,800\t7,186,564\nAll directors and executive officers as a group\t9,415,420\t3,599,147\t13,014,567"
    answer = OwnershipExtract.model_validate(
        {
            "percent_unit": "fraction",
            "shares_scale": "units",
            "insider_shares": 13014567,
            "holders": [{"holder_name": "Jane Roe", "holder_type": "director_officer", "shares": 7186564}],
        }
    )
    parent = {**stamp, "insider_shares": 13014567}
    result = postprocess_section(section, answer, stamp, {}, parent, source)
    for table, frame in result.frames.items():
        sqlite_store.save(table, frame)
    stored = sqlite_store.load(Tables.def14a_llm, where={"ticker": "X", "accession_number": "a"})
    assert stored is not None and stored.iloc[0].insider_shares == 9415420
    raw = Def14AExtract.model_validate({"governance": {"insider_shares": 13014567}}).model_dump_json()
    facade = json.loads(refresh_archive(raw, section, result.legacy, result.corrections))
    assert facade["governance"]["insider_shares"] == 9415420
    assert json.loads(raw)["governance"]["insider_shares"] == 13014567
    print("SANITY: a source-proven beneficial group total updates child-derived facts, saved parent and facade, preserving original raw JSON.")


def test_source_currency_column_cannot_confirm_usd_overwrite() -> None:
    source = "Name\tYear\tCurrency\tSalary ($)\nJane Roe\t2024\tCAD\t1,500,000"
    new = {"name": "Jane Roe", "fiscal_year": 2024, "salary": 1500000}
    row = _process("executive_pay", [new], source, [{**new, "salary": 2000000}]).iloc[0]
    assert row.salary == 2000000
    print("SANITY: explicit row Currency=CAD cannot validate an answer that incorrectly calls the amount USD.")


def test_ownership_percent_column_never_overwrites_beneficial_share_count() -> None:
    source = "Name\tTotal beneficial ownership Number\tTotal beneficial ownership Percent\tAdditional underlying stock units\tTotal\nJane Roe\t430,708\t0.6\t2,864,769\t3,295,477"
    row = {"holder_name": "Jane Roe", "holder_type": "director_officer", "shares": 3295477}
    result = _process("ownership", [row], source).iloc[0]
    assert result.shares == 430708
    print("SANITY: adjacent beneficial ownership Number and Percent columns cannot turn a percentage into shares.")


@pytest.mark.parametrize("field,label", [("stock_awards", "Stock awards (#)"), ("option_awards", "Option awards (#)")])
def test_award_quantity_is_never_corroborated_as_dollar_pay(field: str, label: str) -> None:
    source = f"Name\tYear\tSalary ($)\t{label}\nJane Roe\t2024\t1,500,000\t1,000"
    new = {"name": "Jane Roe", "fiscal_year": 2024, "salary": 1500000, field: 1000}
    assert pd.isna(_process("executive_pay", [new], source).iloc[0][field])
    old = _process("executive_pay", [new], source, [{**new, field: 500000}]).iloc[0]
    assert old[field] == 500000
    monetary = source.replace(label, label.replace("(#)", "fair value of shares ($)"))
    assert _process("executive_pay", [new], monetary).iloc[0][field] == 1000
    print("SANITY: explicit award counts stay unknown as dollars; a real dollar fair-value column remains valid.")
