"""A printed omitted director amount cannot survive repair as an invented zero."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.schemas.def14a_schema import Def14AExtract
from src.data_extract.utils.structure.def14a.postprocess import append_answer, postprocess_section, refresh_archive
from src.data_extract.utils.structure.def14a.retry import SECTIONS
from src.data_store.store import DataStore


@pytest.fixture(scope="module")
def wmt_packet() -> dict[str, Any]:
    return json.loads((Path(__file__).parent / "fixtures/def14a_wmt_unknown_compensation.json").read_text())


@pytest.mark.parametrize("saved_state", ["legacy_saved_conde", "repair_saved_conde"])
@pytest.mark.usefixtures("filing_store")
def test_actual_wmt_paid_unknown_clears_only_the_source_unsupported_legacy_zero(
    wmt_packet: dict[str, Any], saved_state: str, sqlite_store: DataStore
) -> None:
    section = next(item for item in SECTIONS if item.key == "director_pay")
    stamp = {**wmt_packet["meta"], "as_of": pd.Timestamp(wmt_packet["meta"]["as_of"])}
    answer = section.schema.model_validate(wmt_packet["parsed"])
    original = answer.model_dump()
    old = pd.DataFrame([wmt_packet[saved_state]])
    old["as_of"] = pd.to_datetime(old["as_of"])
    sqlite_store.save(section.table, old)
    raw = wmt_packet["legacy_raw_json"]
    assert hashlib.sha256(raw.encode()).hexdigest() == wmt_packet["legacy_raw_sha256"]
    raw = append_answer(raw, section, answer, wmt_packet["source"])
    unchanged = raw

    # Replay the exact paid source and answer twice against the actual legacy row.
    for _ in range(2):
        saved = sqlite_store.load(section.table, where={"ticker": "WMT", "accession_number": stamp["accession_number"]})
        assert saved is not None
        result = postprocess_section(section, answer, stamp, {section.table: saved}, {}, wmt_packet["source"])
        for table, frame in result.frames.items():
            sqlite_store.save(table, frame)
        stored = sqlite_store.load(section.table, where={"ticker": "WMT", "accession_number": stamp["accession_number"]})
        assert stored is not None
        row = stored.loc[stored.name.eq("Cesar Conde")].iloc[0]
        assert row.fiscal_year == 2025 and pd.isna(row.other_compensation)
        assert row.retry == wmt_packet[saved_state]["retry"]
        assert row.fees_earned == 100000 and row.stock_awards == 199973 and row.total == 299973 and row.reconciles == 1
        facade = Def14AExtract.model_validate_json(refresh_archive(raw, section, result.legacy, result.corrections))
        conde = next(item for item in facade.director_compensation if item.name == "Cesar Conde")
        assert conde.fiscal_year == 2025 and conde.all_other_comp_usd is None
    assert raw == unchanged and answer.model_dump() == original
    print(
        "SANITY: exact Walmart paid answer plus literal omitted amount clears the legacy zero in SQLite and facade, preserves retry/facts/raw, and replays idempotently."
    )


@pytest.mark.parametrize(
    "guard",
    [
        "literal_zero_2020",
        "positive_saved",
        "missing_source_year",
        "conflicting_year",
        "different_person",
        "fees",
        "total",
        "unknown_text",
        "foreign",
        "count",
        "new_zero",
        "duplicate_new",
        "conflicting_source",
    ],
)
def test_omitted_director_amount_needs_literal_dash_matching_year_and_both_amount_anchors(guard: str) -> None:
    section = next(item for item in SECTIONS if item.key == "director_pay")
    stamp = {"ticker": "WMT", "accession_number": "a", "as_of": pd.Timestamp("2025-04-24"), "cik": "1"}
    year = 2020 if guard == "literal_zero_2020" else 2025
    row = {"name": "Cesar Conde", "fiscal_year": year, "fees_earned": 100000, "stock_awards": 199973, "other_compensation": None, "total": 299973}
    source = f"Director Compensation for Fiscal {year}\nName\tFees earned ($)\tStock awards ($)\tAll other compensation ($)\tTotal ($)\nCesar Conde\t100,000\t199,973\t\u2014\t299,973"
    old = {**stamp, **row, "other_compensation": 42 if guard == "positive_saved" else 0, "retry": 2}
    if guard == "literal_zero_2020":
        source = source.replace("\u2014", "0")
    if guard == "missing_source_year":
        source = source.replace(f"Director Compensation for Fiscal {year}\n", "")
    if guard == "conflicting_year":
        source = source.replace("Fiscal 2025", "Fiscal 2024")
    if guard == "different_person":
        source = source.replace("Cesar Conde", "Cesar Conde Jr.")
    if guard in ("fees", "total"):
        old_cell, new_cell = {"fees": ("100,000", "100,001"), "total": ("299,973", "299,974")}[guard]
        source = source.replace(old_cell, new_cell)
    if guard == "unknown_text":
        source = source.replace("\u2014", "see note")
    if guard == "foreign":
        source = source.replace("($)", "(CAD)")
    if guard == "count":
        source = source.replace("All other compensation ($)", "All other compensation (#)")
    if guard == "conflicting_source":
        source += "\nCesar Conde\t100,000\t199,973\t1\t299,973"
    if guard == "new_zero":
        row["other_compensation"] = 0
    rows = [row, {**row, "name": "CESAR CONDE"}] if guard == "duplicate_new" else [row]
    answer = section.schema.model_validate({"currency": "USD", "amount_scale": "units", "compensation": rows})
    result = postprocess_section(section, answer, stamp, {section.table: pd.DataFrame([old])}, {}, source).frames[section.table]
    assert result.loc[result.name.eq("Cesar Conde")].iloc[0].other_compensation == old["other_compensation"]
    print(f"SANITY: omitted-amount correction abstains for {guard}; a literal zero or a different supported saved value survives.")
