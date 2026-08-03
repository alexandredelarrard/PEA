"""
Unit tests for the DEF 14A repair layer (def14a_validate.py).

Every case below is a REAL defect observed in the `def14a_edgar*` tables and
reproduced against edgartools 5.44.1 on the live filing -- the values are the
actual ones the library returned, not invented fixtures. Pure-synthetic, no
network, matching the convention in test_fetch_def14a_edgar.py.
"""
from __future__ import annotations

import math

import pandas as pd

from src.data_extract.utils.structure.def14a_validate import (
    clean_person_name, clean_text, repair_director_comp_rows, repair_exec_comp_rows,
    repair_main_row, repair_ownership_rows,
)

_NAN = float("nan")


def _main(**over) -> dict:
    base = {
        "company_name": "COCA COLA CO", "peo_name": "James Quincey",
        "company_selected_measure_name": "Organic revenue growth", "auditor_name": "",
        "audit_fiscal_year_current": 2025.0, "audit_fiscal_year_prior": 2024.0,
        "audit_fees_current": _NAN, "audit_fees_prior": _NAN,
        "audit_related_fees_current": _NAN, "audit_related_fees_prior": _NAN,
        "tax_fees_current": _NAN, "tax_fees_prior": _NAN,
        "other_fees_current": _NAN, "other_fees_prior": _NAN,
        "total_fees_current": _NAN, "total_fees_prior": _NAN,
        "net_income": 13_137_000_000.0,
        "peo_total_comp": 31_208_165.0, "peo_actually_paid_comp": 61_649_669.0,
        "neo_avg_total_comp": 9_524_407.0, "neo_avg_actually_paid_comp": 12_000_000.0,
        "ceo_pay_ratio_ceo_comp": _NAN, "ceo_pay_ratio_median_employee_comp": _NAN,
        "ceo_pay_ratio": _NAN,
    }
    base.update(over)
    return base


# ── text / key normalisation ────────────────────────────────────────────────

def test_clean_text_collapses_source_html_whitespace_runs():
    # GE's real company_selected_measure_name.
    assert clean_text("Free                          cash flow") == "Free cash flow"
    assert clean_text("Return     on Equity") == "Return on Equity"
    assert clean_text("") is None


def test_clean_person_name_strips_footnote_index_so_pk_survives_year_over_year():
    # MSFT keyed this director as "...11" in FY24 and "...10" in FY25 -- two PK rows, one person.
    assert clean_person_name("Emma N. Walmsley11") == "Emma N. Walmsley"
    assert clean_person_name("Emma N. Walmsley10") == "Emma N. Walmsley"
    assert clean_person_name("Carlos A. Rodriguez9") == "Carlos A. Rodriguez"
    assert clean_person_name("Stephen Angel*") == "Stephen Angel"


def test_clean_person_name_strips_glued_title():
    assert clean_person_name("James DimonChairman and CEO") == "James Dimon"
    assert clean_person_name("Deirdre O’Brien Senior Vice") == "Deirdre O’Brien"
    assert clean_person_name("Judson B. Althoff Executive Vice President and") == "Judson B. Althoff"
    assert clean_person_name("H. Lawrence   Culp, Jr.") == "H. Lawrence Culp, Jr."


def test_clean_person_name_consumes_the_title_modifier_word():
    """Without the leading modifier group the split leaves the modifier stuck to the name."""
    assert clean_person_name("Luca Maestri Former Senior Vice President") == "Luca Maestri"
    assert clean_person_name("Bob De Lange Group President") == "Bob De Lange"
    assert clean_person_name("Denise C. Johnson Group President") == "Denise C. Johnson"


# ── main table ──────────────────────────────────────────────────────────────

def test_audit_fee_block_in_thousands_is_rescaled_to_dollars():
    """KO's 2026 proxy: edgartools missed the '(in thousands)' header, so the SAME fee it read as
    32,104,000 from the 2025 proxy came back as 32,104 here."""
    out = repair_main_row(_main(
        audit_fees_current=30_587.0, audit_fees_prior=32_104.0,
        audit_related_fees_current=4_834.0, tax_fees_current=6_760.0,
        other_fees_current=85.0, total_fees_current=42_266.0, total_fees_prior=45_568.0))
    assert out["audit_fees_current"] == 30_587_000.0
    assert out["audit_fees_prior"] == 32_104_000.0          # == KO 2025's audit_fees_current
    assert out["total_fees_prior"] == 45_568_000.0          # == KO 2025's total_fees_current
    assert out["other_fees_current"] == 85_000.0            # whole block moves together


def test_dollar_audit_fee_block_is_left_alone():
    out = repair_main_row(_main(audit_fees_current=32_104_000.0, total_fees_current=45_568_000.0))
    assert out["audit_fees_current"] == 32_104_000.0
    assert out["total_fees_current"] == 45_568_000.0


def test_implausible_net_income_is_nulled_not_guessed():
    """PG's proxy yielded net_income=16.1 where every other issuer yielded whole dollars. 16.1 is
    equally consistent with millions and billions and nothing in the row disambiguates them, so it
    is dropped rather than rescaled into a confidently wrong number."""
    assert math.isnan(repair_main_row(_main(net_income=16.1))["net_income"])
    assert math.isnan(repair_main_row(_main(net_income=15.0))["net_income"])
    assert repair_main_row(_main(net_income=93_736_000_000.0))["net_income"] == 93_736_000_000.0
    assert repair_main_row(_main(net_income=6_556_000_000.0))["net_income"] == 6_556_000_000.0


def test_bogus_fiscal_year_labels_are_rejected():
    out = repair_main_row(_main(audit_fiscal_year_current=13.0, audit_fiscal_year_prior=2024.0))
    assert math.isnan(out["audit_fiscal_year_current"])
    assert out["audit_fiscal_year_prior"] == 2024.0


def test_zero_peo_comp_is_nulled():
    """PG's 2025 proxy returned peo_total_comp=0 / peo_actually_paid_comp=0 -- a failed read."""
    out = repair_main_row(_main(peo_total_comp=0.0, peo_actually_paid_comp=0.0))
    assert math.isnan(out["peo_total_comp"])
    assert math.isnan(out["peo_actually_paid_comp"])


def test_pay_ratio_missing_leg_is_derived():
    """GE disclosed the ratio and the median but edgartools lost the CEO figure."""
    out = repair_main_row(_main(ceo_pay_ratio_median_employee_comp=69_553.0, ceo_pay_ratio=1279.0))
    assert out["ceo_pay_ratio_ceo_comp"] == 69_553.0 * 1279.0

    out = repair_main_row(_main(ceo_pay_ratio_ceo_comp=25_261_296.0,
                                ceo_pay_ratio_median_employee_comp=72_000.0))
    assert round(out["ceo_pay_ratio"], 2) == round(25_261_296.0 / 72_000.0, 2)


def test_pay_ratio_that_does_not_reconcile_is_dropped_entirely():
    out = repair_main_row(_main(ceo_pay_ratio_ceo_comp=28_002_284.0,
                                ceo_pay_ratio_median_employee_comp=14_144.0, ceo_pay_ratio=42.0))
    assert all(math.isnan(out[c]) for c in
               ("ceo_pay_ratio_ceo_comp", "ceo_pay_ratio_median_employee_comp", "ceo_pay_ratio"))


def test_consistent_pay_ratio_triplet_is_preserved():
    out = repair_main_row(_main(ceo_pay_ratio_ceo_comp=28_002_284.0,
                                ceo_pay_ratio_median_employee_comp=14_144.0, ceo_pay_ratio=1980.0))
    assert out["ceo_pay_ratio"] == 1980.0
    assert out["ceo_pay_ratio_ceo_comp"] == 28_002_284.0


# ── executive compensation ──────────────────────────────────────────────────

def test_total_duplicated_into_pension_column_is_corrected():
    """CAT: every SCT row came back with pension_change == total, while the other components
    already summed to total exactly -- so the true pension figure is 0."""
    rows = repair_exec_comp_rows([{
        "name": "D. James Umpleby", "title": "III(7) Chairman and CEO",
        "salary": 1_811_250.0, "bonus": _NAN, "stock_awards": 14_079_975.0,
        "option_awards": 4_125_037.0, "non_equity_incentive": 4_363_800.0,
        "pension_change": 25_261_296.0, "other_compensation": 881_234.0, "total": 25_261_296.0,
    }])
    assert rows[0]["pension_change"] == 0.0
    assert rows[0]["total"] == 25_261_296.0


def test_glued_title_is_recovered_into_the_title_column():
    rows = repair_exec_comp_rows([{
        "name": "James DimonChairman and CEO", "title": None, "salary": 1_500_000.0,
        "bonus": _NAN, "stock_awards": _NAN, "option_awards": _NAN, "non_equity_incentive": _NAN,
        "pension_change": _NAN, "other_compensation": _NAN, "total": 39_000_000.0,
    }])
    assert rows[0]["name"] == "James Dimon"
    assert rows[0]["title"] == "Chairman and CEO"


def test_modifier_stranded_on_the_name_is_reattached_to_the_title():
    """CAT: edgartools split the cell BETWEEN the modifier and its noun, leaving name="Bob De Lange
    Group" / title="President"."""
    rows = repair_exec_comp_rows([{
        "name": "Bob De Lange Group", "title": "President", "salary": 889_950.0, "bonus": _NAN,
        "stock_awards": 5_157_307.0, "option_awards": 924_982.0,
        "non_equity_incentive": 1_299_200.0, "pension_change": 8_556_855.0,
        "other_compensation": 285_416.0, "total": 8_556_855.0,
    }])
    assert rows[0]["name"] == "Bob De Lange"
    assert rows[0]["title"] == "Group President"


def test_generational_suffix_stranded_on_the_title_is_dropped():
    rows = repair_exec_comp_rows([{
        "name": "D. James Umpleby", "title": "III(7) Chairman and CEO", "salary": 1_811_250.0,
        "bonus": _NAN, "stock_awards": _NAN, "option_awards": _NAN, "non_equity_incentive": _NAN,
        "pension_change": _NAN, "other_compensation": _NAN, "total": 25_261_296.0,
    }])
    assert rows[0]["title"] == "Chairman and CEO"


def test_all_null_exec_comp_row_is_dropped():
    """JPM: names parsed, every single value dropped. An all-NULL row carries nothing but still
    squats on its primary key, blocking a later good extraction."""
    rows = repair_exec_comp_rows([{
        "name": "Mary Callahan ErdoesCEO, AWM", "title": None, "salary": _NAN, "bonus": _NAN,
        "stock_awards": _NAN, "option_awards": _NAN, "non_equity_incentive": _NAN,
        "pension_change": _NAN, "other_compensation": _NAN, "total": _NAN,
    }])
    assert rows == []


# ── director compensation ───────────────────────────────────────────────────

def test_single_missing_director_component_is_imputed_from_the_residual():
    """PFE: total 268,764 vs components summing to 99,148 -- the $169,616 stock award was the only
    component the parser dropped, so the residual IS that award."""
    rows = repair_director_comp_rows([{
        "name": "Cyrus Taraporevala", "fees_earned": 79_148.0, "stock_awards": _NAN,
        "option_awards": 0.0, "non_equity_incentive": 0.0, "pension_change": 0.0,
        "other_compensation": 20_000.0, "total": 268_764.0,
    }])
    assert rows[0]["stock_awards"] == 169_616.0


def test_unattributable_director_residual_is_not_fabricated():
    """CAT: four components are NULL, so the $175,033 gap cannot be pinned on one of them.
    `total` stays authoritative and nothing is invented."""
    rows = repair_director_comp_rows([{
        "name": "DEBRA L. REED-KLAGES", "fees_earned": 225_000.0, "stock_awards": _NAN,
        "option_awards": _NAN, "non_equity_incentive": _NAN, "pension_change": _NAN,
        "other_compensation": _NAN, "total": 400_033.0,
    }])
    assert math.isnan(rows[0]["stock_awards"])
    assert rows[0]["total"] == 400_033.0


def test_director_subtotal_row_is_dropped():
    rows = repair_director_comp_rows([
        {"name": "Total", "fees_earned": _NAN, "stock_awards": _NAN, "option_awards": _NAN,
         "non_equity_incentive": _NAN, "pension_change": _NAN, "other_compensation": _NAN,
         "total": 2_468_728.0},
        {"name": "Thomas Horton", "fees_earned": _NAN, "stock_awards": _NAN, "option_awards": _NAN,
         "non_equity_incentive": _NAN, "pension_change": _NAN, "other_compensation": _NAN,
         "total": 393_392.0},
    ])
    assert [r["name"] for r in rows] == ["Thomas Horton"]


# ── beneficial ownership ────────────────────────────────────────────────────

def test_fabricated_half_percent_placeholder_is_nulled():
    """edgartools hardcodes 0.5 for the '*' (= 'less than 1%') footnote, which made Apple's CEO
    and its 12-person director group both read as exactly 0.5%."""
    rows = repair_ownership_rows([
        {"holder_name": "Tim Cook", "holder_type": "director_officer",
         "shares": 3_280_295.0, "percent_of_class": 0.5},
        {"holder_name": "The Vanguard Group", "holder_type": "5pct_holder",
         "shares": 1_415_826_462.0, "percent_of_class": 9.63},
    ], insider_names=set())
    assert math.isnan(rows[0]["percent_of_class"])
    assert rows[0]["shares"] == 3_280_295.0        # the share count is real, only the pct was not
    assert rows[1]["percent_of_class"] == 9.63


def test_address_only_holder_name_is_dropped():
    """JPM: edgartools took the address line as the holder name and lost the share count."""
    rows = repair_ownership_rows([
        {"holder_name": "100 Vanguard Blvd, Malvern, PA 19355", "holder_type": "5pct_holder",
         "shares": _NAN, "percent_of_class": 9.86},
    ], insider_names=set())
    assert rows == []


def test_address_is_stripped_off_institutional_holder_name():
    rows = repair_ownership_rows([
        {"holder_name": "The Vanguard Group 100 Vanguard Blvd. Malvern, PA 19355",
         "holder_type": "5pct_holder", "shares": 2_249_200_352.0, "percent_of_class": 9.60},
        {"holder_name": "BlackRock, Inc. 55 East 52nd Street New York, NY 10055",
         "holder_type": "5pct_holder", "shares": 1_557_622_991.0, "percent_of_class": 6.65},
    ], insider_names=set())
    assert rows[0]["holder_name"] == "The Vanguard Group"
    assert rows[1]["holder_name"] == "BlackRock, Inc."


def test_ceo_mistyped_as_five_percent_holder_is_retyped_from_the_comp_tables():
    """GE's and XOM's CEO were both tagged `5pct_holder`. Nobody is both, and the name appears in
    the same filing's comp tables, which settles it."""
    rows = repair_ownership_rows([
        {"holder_name": "H. Lawrence Culp, Jr.", "holder_type": "5pct_holder",
         "shares": 1_612_480.0, "percent_of_class": _NAN},
    ], insider_names={"H. Lawrence Culp, Jr."})
    assert rows[0]["holder_type"] == "director_officer"


def test_ownership_group_subtotal_row_is_dropped():
    rows = repair_ownership_rows([
        {"holder_name": "All current directors and executive officers as a group (12 persons)",
         "holder_type": "group", "shares": 9_079_765.0, "percent_of_class": 0.5},
        {"holder_name": "Total", "holder_type": "director_officer",
         "shares": 2_468_728.0, "percent_of_class": _NAN},
        {"holder_name": "Rahul Ghai", "holder_type": "director_officer",
         "shares": 140_216.0, "percent_of_class": _NAN},
    ], insider_names=set())
    assert [r["holder_name"] for r in rows] == ["Rahul Ghai"]


def test_repairs_are_pure_and_do_not_mutate_the_input():
    src = _main(net_income=16.1)
    repair_main_row(src)
    assert src["net_income"] == 16.1
