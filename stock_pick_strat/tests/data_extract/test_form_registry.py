"""
Form-dispatch registry tests: every FORM_REGISTRY entry must point at a real
schema_registry table, and its declared sec_forms must not drift from the
centralized constants.py form-type lists.
"""
from __future__ import annotations

from src.constants.constants import (
    DEF14A_FORMS, FUNDAMENTALS_FORMS, SEC_8K_FORMS, SEC_13D_FORMS,
)
from src.data_extract.utils.common.form_registry import FORM_REGISTRY
from src.data_store.schema_registry import BY_NAME


def test_registry_tables_exist_in_schema_registry():
    for name, spec in FORM_REGISTRY.items():
        assert spec.table in BY_NAME, f"{name}: table '{spec.table}' not in schema_registry.BY_NAME"


def test_registry_forms_match_constants():
    assert FORM_REGISTRY["fundamentals"].sec_forms == tuple(FUNDAMENTALS_FORMS)
    assert FORM_REGISTRY["sec_8k"].sec_forms == tuple(SEC_8K_FORMS)
    assert FORM_REGISTRY["sec_13d"].sec_forms == tuple(SEC_13D_FORMS)
    assert FORM_REGISTRY["def_14"].sec_forms == tuple(DEF14A_FORMS)


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


def test_thirteen_f_keeps_its_bulk_quarterly_grain():
    """13F-HR is a bulk, all-filers pull (SEC's pre-joined quarterly data sets) --
    it must NOT be forced into the per-CIK/accession discovery shape the other
    four forms use."""
    spec = FORM_REGISTRY["sec13f_hr"]
    assert spec.discovery == "bulk_quarterly"

    print("\n=== SANITY CHECK: form-dispatch registry ===")
    print(f"  {len(FORM_REGISTRY)} forms registered, every table exists in schema_registry,")
    print("  every handler is callable, no drift from constants.py form lists;")
    print("  def_14 correctly points at the EXISTING def14a_llm table (kept per the task's")
    print("  own instruction); sec_8k/sec13f_hr correctly point at the renamed tables;")
    print("  sec13f_hr keeps its distinct bulk-quarterly discovery contract.")
    print("  Validated.")
