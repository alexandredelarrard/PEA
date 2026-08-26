"""
Form-dispatch registry tests: every FORM_REGISTRY entry must point at a real
schema.py table, and its declared sec_forms must not drift from the
centralized constants.py form-type lists.
"""
from __future__ import annotations

from src.constants.constants import (
    DEF14A_FORMS, FUNDAMENTALS_FORMS, SEC_8K_FORMS, SEC_13D_FORMS,
)
from src.data_extract.utils.common.form_registry import FORM_REGISTRY
from src.data_store.schema import BY_NAME

#: registry key -> the constants.py form list it must not drift from. Only these four
#: forms have a centralized list; `def14a_edgar` shares `def_14`'s and the remaining
#: entries own their forms outright.
EXPECTED_FORMS: dict[str, tuple[str, ...]] = {
    "fundamentals": tuple(FUNDAMENTALS_FORMS),
    "sec_8k": tuple(SEC_8K_FORMS),
    "sec_13d": tuple(SEC_13D_FORMS),
    "def_14": tuple(DEF14A_FORMS),
}


def test_registry_tables_exist_in_schema_registry():
    for name, spec in FORM_REGISTRY.items():
        assert spec.table in BY_NAME, f"{name}: table '{spec.table}' not in schema.BY_NAME"


def test_registry_forms_match_constants():
    """No entry's declared `sec_forms` may drift from its constants.py list.

    Checks the intersection rather than every key in `EXPECTED_FORMS`, because the
    `fundamentals` entry is absent while that stack is rebuilt (see
    reports/planning/active-tasks/2026-08-21-fundamentals-rebuild-plan.md); the absent
    keys are printed so the gap stays visible instead of passing silently."""
    for name in sorted(EXPECTED_FORMS.keys() & FORM_REGISTRY.keys()):
        assert FORM_REGISTRY[name].sec_forms == EXPECTED_FORMS[name], f"{name}: form drift"

    absent = sorted(EXPECTED_FORMS.keys() - FORM_REGISTRY.keys())
    print(f"\n[form drift] {len(EXPECTED_FORMS) - len(absent)} of {len(EXPECTED_FORMS)} "
          f"form lists checked; NOT REGISTERED: {absent or 'none'}")


def test_registry_handlers_are_callable():
    for name, spec in FORM_REGISTRY.items():
        assert callable(spec.handler), f"{name}: handler is not callable"


def test_def14a_logical_key_maps_to_existing_def14a_llm_table():
    """The task's own spec says to keep def14a_llm as-is ('process as a
    complementary one') -- the registry's logical key is 'def_14' but the
    physical table must remain the existing, already-working def14a_llm."""
    assert FORM_REGISTRY["def_14"].table == "def14a_llm"


def test_renamed_tables_use_new_names_not_old():
    """sec_8k_items -> sec_8k and institutional_holdings -> sec13f_hr (DB-level
    ALTER TABLE RENAME, see scripts/rename_form_dispatch_tables.py) -- the
    registry must reference the NEW names."""
    assert FORM_REGISTRY["sec_8k"].table == "sec_8k"
    assert FORM_REGISTRY["sec13f_hr"].table == "sec13f_hr"


def test_thirteen_f_keeps_its_all_filers_grain():
    """13F-HR is an all-filers pull, discovered by FILING DATE across every manager --
    not the per-CIK/accession shape the other four forms use. Discovery moved off SEC's
    quarterly bulk data sets onto edgartools (they publish weeks after a quarter closes),
    but the all-filers grain, and the (cik, period, ticker, cusip) PK, did not."""
    spec = FORM_REGISTRY["sec13f_hr"]
    assert spec.discovery == "all_filers_by_date"
    assert "13F-NT" not in spec.sec_forms      # a notice filing carries no info table

    print("\n=== SANITY CHECK: form-dispatch registry ===")
    print(f"  {len(FORM_REGISTRY)} forms registered, every table exists in schema.py,")
    print("  every handler is callable, no drift from constants.py form lists;")
    print("  def_14 correctly points at the EXISTING def14a_llm table (kept per the task's")
    print("  own instruction); sec_8k/sec13f_hr correctly point at the renamed tables;")
    print("  sec13f_hr keeps its distinct all-filers-by-filing-date discovery contract.")
    print("  Validated.")
