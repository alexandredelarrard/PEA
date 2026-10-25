"""Frozen paid pilot answers expose missing SAR dollars and foreign-security collisions."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.schemas.def14a_schema import Def14AExtract
from src.data_extract.utils.structure.def14a.postprocess import append_answer, postprocess_section, refresh_archive
from src.data_extract.utils.structure.def14a.retry import SECTIONS
from src.data_store.schema import Tables
from src.data_store.store import DataStore


@pytest.fixture(scope="module")
def pilot_packets() -> dict[str, Any]:
    return json.loads((Path(__file__).parent / "fixtures/def14a_eog_pilot_basis.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("domain", ["executive_pay", "director_pay", "ownership"])
def test_actual_paid_pilot_source_recovers_the_three_regressions(pilot_packets: dict[str, Any], domain: str, sqlite_store: DataStore) -> None:
    packet = pilot_packets[domain]
    section = next(item for item in SECTIONS if item.key == domain)
    stamp = {**packet["meta"], "as_of": pd.Timestamp(packet["meta"]["as_of"])}
    answer = section.schema.model_validate(packet["parsed"])
    before = answer.model_dump()
    target = packet["target"]
    name_column = "holder_name" if domain == "ownership" else "name"
    saved = {}
    parent = {}
    raw = append_answer(None, section, answer, packet["source"])

    # Exact recorded answers and original request text replay through real persistence twice.
    for _ in range(2):
        result = postprocess_section(section, answer, stamp, saved, parent, packet["source"])
        for table, frame in result.frames.items():
            sqlite_store.save(table, frame)
        stored = sqlite_store.load(section.table, where={"ticker": "EOG", "accession_number": stamp["accession_number"]})
        matches = stored.loc[stored[name_column].eq(target["name"])]
        row = matches.iloc[0] if domain == "ownership" else matches.loc[matches.fiscal_year.eq(2014)].iloc[0]
        assert row[target["field"]] == target["expected"]
        assert row.retry == 1
        if domain != "ownership":
            assert row.reconciles == 1
        if domain == "executive_pay":
            assert result.frames[Tables.def14a_llm].iloc[0].ceo_option_awards == target["expected"]
        facade = Def14AExtract.model_validate_json(refresh_archive(raw, section, result.legacy, result.corrections))
        children = getattr(
            facade, {"executive_pay": "compensation", "director_pay": "director_compensation", "ownership": "ownership_holders"}[domain]
        )
        field = "shares" if domain == "ownership" else "option_awards_usd"
        matched = [item for item in children if getattr(item, name_column) == target["name"] and (domain == "ownership" or item.fiscal_year == 2014)]
        assert getattr(matched[0], field) == target["expected"]
        saved = result.frames
        parent = result.frames[Tables.def14a_llm].iloc[0].to_dict()
    assert answer.model_dump() == before
    print(
        f"SANITY: exact paid {domain} answer plus its original source recovers {target['expected']:,} and persists idempotently without a provider call."
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("existing", [None, 52863, 72863, 4822])
def test_explicit_issuer_components_win_only_over_foreign_class(pilot_packets: dict[str, Any], reverse: bool, existing: int | None) -> None:
    packet = pilot_packets["ownership"]
    section = next(item for item in SECTIONS if item.key == "ownership")
    stamp = {**packet["meta"], "as_of": pd.Timestamp(packet["meta"]["as_of"])}
    payload = deepcopy(packet["parsed"])
    payload["holders"] = payload["holders"][::-1] if reverse else payload["holders"]
    saved = (
        pd.DataFrame([{**stamp, "holder_name": "Dennis M. Ulak", "holder_type": "director_officer", "shares": existing, "retry": 2}])
        if existing
        else pd.DataFrame()
    )
    result = postprocess_section(section, section.schema.model_validate(payload), stamp, {section.table: saved}, {}, packet["source"])
    row = result.frames[section.table].loc[lambda frame: frame.holder_name.eq("Dennis M. Ulak")].iloc[0]
    assert row.shares == (52863 if existing == 52863 else 72863)
    print("SANITY: either answer order retains a saved issuer component or the proven disjoint issuer sum; foreign shares cannot win by position.")


@pytest.mark.parametrize("mode", ["unmatched_issuer", "missing_issuer", "overlapping_columns", "ambiguous_issuer_class"])
def test_unproven_security_class_or_component_layout_abstains(pilot_packets: dict[str, Any], mode: str) -> None:
    packet = pilot_packets["ownership"]
    section = next(item for item in SECTIONS if item.key == "ownership")
    stamp = {**packet["meta"], "as_of": pd.Timestamp(packet["meta"]["as_of"])}
    source = packet["source"]
    if mode in ("unmatched_issuer", "missing_issuer"):
        stamp["company_name"] = "Different Company" if mode == "unmatched_issuer" else None
    if mode == "overlapping_columns":
        source = source.replace("AND LIMITED", "AND DISPOSITIVE")
    if mode == "ambiguous_issuer_class":
        source = source.replace("Enron Corp.", "Enron Oil & Gas Company")
    result = postprocess_section(section, section.schema.model_validate(packet["parsed"]), stamp, {}, {}, source)
    assert result.frames[section.table].loc[lambda frame: frame.holder_name.eq("Dennis M. Ulak")].iloc[0].shares == 4822
    print("SANITY: missing or ambiguous issuer identity and overlapping ownership columns cannot authorize a source sum.")


@pytest.mark.parametrize("mode", ["sar_counts", "vector_length_mismatch", "unlabelled_director_year", "ambiguous_director_year"])
def test_sar_rescue_requires_money_and_exact_source_year_binding(pilot_packets: dict[str, Any], mode: str) -> None:
    domain = "director_pay" if "director" in mode else "executive_pay"
    packet = pilot_packets[domain]
    section = next(item for item in SECTIONS if item.key == domain)
    stamp = {**packet["meta"], "as_of": pd.Timestamp(packet["meta"]["as_of"])}
    source = packet["source"]
    if mode == "sar_counts":
        source = source.replace("SAR Awards ($)(c)", "SAR Awards (#)(c)")
    if mode == "vector_length_mismatch":
        source = source.replace("1,978,503 1,763,659 1,355,883", "1,978,503 1,763,659")
    if mode == "unlabelled_director_year":
        source = source.replace("Director Compensation Table for 2014", "Director Compensation Table")
    if mode == "ambiguous_director_year":
        source = source.replace("Director Compensation Table for 2014", "Director Compensation Table for 2014 / Director Compensation Table for 2013")
    result = postprocess_section(section, section.schema.model_validate(packet["parsed"]), stamp, {}, {}, source)
    row = result.frames[section.table].loc[lambda frame: frame.name.eq(packet["target"]["name"]) & frame.fiscal_year.eq(2014)].iloc[0]
    assert pd.isna(row.option_awards)
    print("SANITY: SAR counts, unmatched year vectors and unlabelled or ambiguous director years remain unknown as annual option dollars.")


@pytest.mark.parametrize("mode", ["yearless_grants", "wrong_domain_fee_table", "intervening_heading", "intervening_named_table"])
def test_director_caption_cannot_lend_its_year_to_another_table(pilot_packets: dict[str, Any], mode: str) -> None:
    domain = "executive_pay" if mode in ("yearless_grants", "wrong_domain_fee_table") else "director_pay"
    packet = pilot_packets[domain]
    section = next(item for item in SECTIONS if item.key == domain)
    stamp = {**packet["meta"], "as_of": pd.Timestamp(packet["meta"]["as_of"])}
    source = "Director Compensation Table for 2014\n"
    if mode == "yearless_grants":
        source += "Name\tFees Earned ($)\tSAR Awards ($)\nJanet F. Clark\t140000\t36655\n\n=== GRANTS OF PLAN-BASED AWARDS ===\n"
        source += "Name\tSAR Awards ($)\nWilliam R. Thomas\t999\n"
    if mode == "wrong_domain_fee_table":
        source += "Name\tFees Earned ($)\tSAR Awards ($)\nWilliam R. Thomas\t140000\t999\n"
    if mode == "intervening_heading":
        source += "=== ANOTHER COMPENSATION TABLE ===\nName\tFees Earned ($)\tSAR Awards ($)\nJanet F. Clark\t140000\t999\n"
    if mode == "intervening_named_table":
        source += "Name\tStock Awards ($)\nOther Person\t100\n\nName\tFees Earned ($)\tSAR Awards ($)\nJanet F. Clark\t140000\t999\n"
    result = postprocess_section(section, section.schema.model_validate(packet["parsed"]), stamp, {}, {}, source)
    row = result.frames[section.table].loc[lambda frame: frame.name.eq(packet["target"]["name"]) & frame.fiscal_year.eq(2014)].iloc[0]
    assert pd.isna(row.option_awards)
    if domain == "executive_pay":
        assert pd.isna(result.frames[Tables.def14a_llm].iloc[0].get("ceo_option_awards"))
    print("SANITY: a director year belongs only to its local fee table; another heading, table or pay domain cannot borrow it.")


def test_later_unrelated_caption_does_not_replace_the_owned_director_year(pilot_packets: dict[str, Any]) -> None:
    packet = pilot_packets["director_pay"]
    section = next(item for item in SECTIONS if item.key == "director_pay")
    stamp = {**packet["meta"], "as_of": pd.Timestamp(packet["meta"]["as_of"])}
    source = packet["source"] + "\nDirector Compensation Table for 2013\nName\tFees Earned ($)\tSAR Awards ($)\nDifferent Person\t140000\t999\n"
    result = postprocess_section(section, section.schema.model_validate(packet["parsed"]), stamp, {}, {}, source)
    assert result.frames[section.table].loc[lambda frame: frame.name.eq("Janet F. Clark")].iloc[0].option_awards == 36655
    print("SANITY: a later unrelated caption cannot erase the explicit2014 authority of the earlier director fee table.")
