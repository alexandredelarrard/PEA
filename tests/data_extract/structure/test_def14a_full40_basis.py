"""Actual full-cohort paid answers expose category, fiscal-year and group-scope regressions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.schemas.def14a_schema import Def14AExtract
from src.data_extract.utils.structure.def14a.basis import correct_basis
from src.data_extract.utils.structure.def14a.postprocess import append_answer, postprocess_section, refresh_archive
from src.data_extract.utils.structure.def14a.retry import SECTIONS
from src.data_store.schema import Tables
from src.data_store.store import DataStore


@pytest.fixture(scope="module")
def full40_packets() -> dict[str, Any]:
    return json.loads((Path(__file__).parent / "fixtures/def14a_full40_regressions.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", ["eog_director_options", "pg_director_year", "pg_group_scope"])
def test_actual_full40_paid_answers_keep_source_categories_year_and_scope(full40_packets: dict[str, Any], case: str, sqlite_store: DataStore) -> None:
    packet = full40_packets[case]
    section = next(item for item in SECTIONS if item.key == packet["domain"])
    stamp = {**packet["meta"], "as_of": pd.Timestamp(packet["meta"]["as_of"])}
    answer = section.schema.model_validate(packet["parsed"])
    original = answer.model_dump()
    raw = append_answer(None, section, answer, packet["source"])
    saved, parent = {}, {}

    # The original paid answer and its exact text pass through actual persistence twice.
    for _ in range(2):
        result = postprocess_section(section, answer, stamp, saved, parent, packet["source"])
        for table, frame in result.frames.items():
            sqlite_store.save(table, frame)
        facade = Def14AExtract.model_validate_json(refresh_archive(raw, section, result.legacy, result.corrections))
        if case == "pg_group_scope":
            stored = sqlite_store.load(Tables.def14a_llm, where={"ticker": "PG", "accession_number": stamp["accession_number"]})
            assert stored is not None and pd.isna(stored.iloc[0].insider_ownership_pct)
            assert facade.governance is not None and facade.governance.insider_ownership_pct is None
            assert stored.iloc[0].insider_shares == 17348734
        else:
            stored = sqlite_store.load(section.table, where={"ticker": stamp["ticker"], "accession_number": stamp["accession_number"]})
            name, year = ("George A. Alcorn", 2006) if case == "eog_director_options" else ("Francis S. Blake", 2020)
            matches = stored.loc[stored.name.eq(name) & stored.fiscal_year.eq(year)]
            assert len(matches) == 1
            row = matches.iloc[0]
            if case == "eog_director_options":
                assert pd.isna(row.stock_awards) and row.option_awards == 233152
            else:
                assert row.fees_earned == 117500 and row.stock_awards == 200000 and row.other_compensation == 0 and row.total == 317500
            legacy = next(item for item in facade.director_compensation if item.name == name)
            assert legacy.fiscal_year == year and row.reconciles == 1 and row.retry == 1
        saved = result.frames
        parent = result.frames[Tables.def14a_llm].iloc[0].to_dict()
    assert answer.model_dump() == original
    print(
        f"SANITY: exact full40 paid {case} persists the source-supported category/year/scope without altering the original answer or spending again."
    )


@pytest.mark.parametrize(
    "guard",
    [
        "multiple_years",
        "conflicting_source",
        "wrong_year",
        "fees",
        "total",
        "nonblank_stock",
        "count",
        "foreign",
        "different_person",
        "unknown_stock",
        "mixed_stock_columns",
    ],
)
def test_option_category_move_abstains_without_unique_person_year_and_monetary_anchors(guard: str) -> None:
    section = next(item for item in SECTIONS if item.key == "director_pay")
    stamp = {"ticker": "EOG", "accession_number": "a", "as_of": pd.Timestamp("2007-03-29"), "cik": "1"}
    row = {"name": "George A. Alcorn", "fiscal_year": 2006, "fees_earned": 80000, "stock_awards": 233152, "total": 313690}
    rows = [row]
    saved = (
        {section.table: pd.DataFrame([{**stamp, **row, "retry": 2}, {**stamp, **row, "fiscal_year": 2005, "retry": 2}])}
        if guard == "multiple_years"
        else {}
    )
    source = "Name\tFees earned ($)\tStock awards ($)\tOption awards ($)\tTotal ($)\nGeorge A. Alcorn\t80,000\t\t233,152\t313,690"
    if guard == "conflicting_source":
        source += "\nGeorge A. Alcorn\t80,000\t\t233,153\t313,690"
    if guard == "wrong_year":
        source = source.replace("Name\t", "Name\tYear\t").replace("Alcorn\t", "Alcorn\t2005\t")
    if guard in ("fees", "total", "nonblank_stock"):
        old, new = {"fees": ("80,000", "80,001"), "total": ("313,690", "313,691"), "nonblank_stock": ("\t\t233,152", "\t100\t233,152")}[guard]
        source = source.replace(old, new)
    if guard == "mixed_stock_columns":
        source = source.replace("Stock awards ($)", "Stock awards ($)\tStock awards ($)(b)").replace("\t\t233,152", "\t\t100\t233,152")
    if guard == "unknown_stock":
        source = source.replace("\t\t233,152", "\tsee footnote\t233,152")
    if guard == "count":
        source = source.replace("Option awards ($)", "Option awards (#)")
    if guard == "foreign":
        source = source.replace("($)", "(CAD)")
    if guard == "different_person":
        source = source.replace("George A. Alcorn", "George A. Alcorn Jr.")
    answer = section.schema.model_validate({"currency": "USD", "amount_scale": "units", "compensation": rows})
    if guard == "multiple_years":
        result, _ = correct_basis(section, saved[section.table], pd.DataFrame(rows), source, None)
    else:
        result = postprocess_section(section, answer, stamp, saved, {}, source).frames[section.table]
    assert result.loc[result.fiscal_year.eq(2006)].iloc[0].stock_awards == 233152
    print(f"SANITY: option category correction abstains for {guard}, retaining the existing year and unsupported amount.")


@pytest.mark.parametrize(
    "guard",
    [
        "nonconsecutive",
        "wrong_domain",
        "intervening_heading",
        "unlabelled_year",
        "different_person",
        "fees",
        "stock",
        "total",
        "currency",
        "existing_year",
        "conflicting_years",
    ],
)
def test_director_missing_year_needs_local_table_authority_and_exact_amounts(guard: str) -> None:
    section = next(item for item in SECTIONS if item.key == "director_pay")
    stamp = {"ticker": "PG", "accession_number": "a", "as_of": pd.Timestamp("2020-08-28"), "cik": "1"}
    row = {
        "name": "Francis S. Blake",
        "fiscal_year": 2019 if guard == "existing_year" else None,
        "fees_earned": 117500,
        "stock_awards": 200000,
        "total": 317500,
    }
    caption = (
        "The following table and footnotes provide information regarding the compensation paid to the Company's non-employee Directors in FY 2019-20."
    )
    table = "Name\tFees earned ($)\tStock awards ($)\tTotal ($)\nFrancis S. Blake\t117,500\t200,000\t317,500"
    source = caption + "\n" + table
    if guard == "nonconsecutive":
        source = source.replace("2019-20", "2019-21")
    if guard == "wrong_domain":
        source = source.replace("Directors", "Named Officers")
    if guard == "intervening_heading":
        source = caption + "\nGRANTS OF PLAN-BASED AWARDS\n" + table
    if guard == "unlabelled_year":
        source = "Proxy filing in 2020 discusses 2019.\n" + table
    if guard == "different_person":
        source = source.replace("Francis S. Blake", "Francis S. Blake Jr.")
    if guard in ("fees", "stock", "total"):
        old, new = {"fees": ("117,500", "117,501"), "stock": ("200,000", "200,001"), "total": ("317,500", "317,501")}[guard]
        source = source.replace(old, new)
    if guard == "conflicting_years":
        source += "\n" + caption.replace("2019-20", "2018-19") + "\n" + table
    answer = section.schema.model_validate({"currency": "CAD" if guard == "currency" else "USD", "amount_scale": "units", "compensation": [row]})
    result = postprocess_section(section, answer, stamp, {}, {}, source).legacy.director_compensation[0]
    assert result.fiscal_year == (2019 if guard == "existing_year" else None)
    print(f"SANITY: missing director year stays conservative for {guard}; no person, amount or annual period is invented.")


@pytest.mark.parametrize("guard", ["mention", "single_class", "zero_preferred", "all_class", "different_percentage"])
def test_common_class_percentage_needs_actual_positive_preferred_group_holdings(guard: str) -> None:
    section = next(item for item in SECTIONS if item.key == "ownership")
    stamp = {"ticker": "PG", "accession_number": "a", "as_of": pd.Timestamp("2020-08-28"), "cik": "1"}
    common = "COMMON STOCK NUMBER OF SHARES/OPTIONS Amount and Nature of Beneficial Ownership Name Total Percent of Class\nDirectors and executive officers, as a group 17,348,734 .70%"
    preferred = "\nSERIES A PREFERRED STOCK Name\tAmount and Nature of Beneficial Ownership\nDirectors and executive officers, as a group\t101,850"
    source = common + preferred
    if guard == "mention":
        source = common + "\nPreferred stock may be issued."
    if guard == "single_class":
        source = common
    if guard == "zero_preferred":
        source = source.replace("101,850", "0")
    if guard == "all_class":
        source += "\nAll classes combined beneficial ownership is 0.7%."
    if guard == "different_percentage":
        source = source.replace(".70%", ".80%")
    answer = section.schema.model_validate(
        {"percent_unit": "percent", "shares_scale": "units", "insider_ownership_pct": 0.7, "insider_voting_pct": 1.25, "holders": []}
    )
    result = postprocess_section(section, answer, stamp, {}, {}, source)
    assert result.frames[Tables.def14a_llm].iloc[0].insider_ownership_pct == pytest.approx(0.007)
    assert result.legacy.governance is not None and result.legacy.governance.insider_voting_pct == pytest.approx(0.0125)
    print(f"SANITY: {guard} does not prove that the reported group percentage is the wrong all-class scalar.")


@pytest.mark.parametrize("saved_percentage", [0.1, 0.007, None])
def test_group_scope_correction_preserves_valid_saved_aggregate_and_clears_only_proven_common_value(
    full40_packets: dict[str, Any], saved_percentage: float | None, sqlite_store: DataStore
) -> None:
    packet = full40_packets["pg_group_scope"]
    section = next(item for item in SECTIONS if item.key == "ownership")
    stamp = {**packet["meta"], "as_of": pd.Timestamp(packet["meta"]["as_of"])}
    values = {**packet["parsed"], "insider_voting_pct": 1.25}
    answer = section.schema.model_validate(values)
    parent = {**stamp, "insider_ownership_pct": saved_percentage, "insider_voting_pct": 0.0125}
    sqlite_store.save(Tables.def14a_llm, pd.DataFrame([parent]))
    raw = Def14AExtract.model_validate({"governance": {"insider_ownership_pct": saved_percentage, "insider_voting_pct": 0.0125}}).model_dump_json()
    raw = append_answer(raw, section, answer, packet["source"])
    original = raw

    # Apply the same paid answer twice, preserving valid retained ownership and separate voting.
    for _ in range(2):
        result = postprocess_section(section, answer, stamp, {}, parent, packet["source"])
        frame = result.frames[Tables.def14a_llm]
        for table, saved in result.frames.items():
            sqlite_store.save(table, saved)
        stored = sqlite_store.load(Tables.def14a_llm, where={"ticker": "PG", "accession_number": stamp["accession_number"]})
        facade = Def14AExtract.model_validate_json(refresh_archive(raw, section, result.legacy, result.corrections))
        assert stored is not None and facade.governance is not None
        if saved_percentage == 0.1:
            assert frame.iloc[0].insider_ownership_pct == stored.iloc[0].insider_ownership_pct == facade.governance.insider_ownership_pct == 0.1
        else:
            assert pd.isna(frame.iloc[0].get("insider_ownership_pct"))
            assert pd.isna(stored.iloc[0].insider_ownership_pct) and facade.governance.insider_ownership_pct is None
        assert stored.iloc[0].insider_voting_pct == facade.governance.insider_voting_pct == 0.0125
        assert result.legacy.governance is not None and result.legacy.governance.insider_ownership_pct is None
        parent = stored.iloc[0].to_dict()
    assert raw == original
    print(
        "SANITY: source-proven common-only percentage clears matching bad values, retains a different valid saved aggregate and voting, and is idempotent in SQLite/facade."
    )


@pytest.mark.parametrize("common_row", ["", "Directors and executive officers, as a group 17,348,734"])
def test_common_group_percentage_cannot_borrow_a_preferred_class_percentage(common_row: str) -> None:
    section = next(item for item in SECTIONS if item.key == "ownership")
    stamp = {"ticker": "PG", "accession_number": "a", "as_of": pd.Timestamp("2020-08-28"), "cik": "1"}
    source = "COMMON STOCK NUMBER OF SHARES/OPTIONS Amount and Nature of Beneficial Ownership Name Total Percent of Class\n" + common_row
    source += "\nSERIES A PREFERRED STOCK Name\tAmount and Nature of Beneficial Ownership\tPercent of Class\nDirectors and executive officers, as a group\t101,850\t.70%"
    answer = section.schema.model_validate({"percent_unit": "percent", "shares_scale": "units", "insider_ownership_pct": 0.7, "holders": []})
    result = postprocess_section(section, answer, stamp, {}, {}, source)
    assert result.frames[Tables.def14a_llm].iloc[0].insider_ownership_pct == pytest.approx(0.007)
    print("SANITY: missing common-class group disclosure cannot borrow the later preferred-class percentage.")
