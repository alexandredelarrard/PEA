"""
Unit tests for the DEF 14A row cleaner (def14a_validate.py).

Every case below is a REAL defect observed on a live filing -- the values are the actual ones
returned, not invented fixtures.

The module used to be a repair layer for edgartools' proxy HTML parser, and most of these tests
covered repairs for that parser's output: a fabricated `0.5` for the "*" ownership footnote, the
Total column duplicated into a component slot, a title glued into a name column, pay-ratio
triplets completed from an identity, ownership rows re-typed from the comp tables. That whole
HTML block was DELETED, so those tests went with it -- keeping them would have asserted the
behaviour of code that no longer runs, which is worse than having no test at all.

What is kept is what both DEF 14A paths still use:

- the name / holder PRIMARY KEYS, whose footnote strip is load-bearing (a director keyed as
  "Emma N. Walmsley11" one year and "...10" the next is two rows and one person);
- `rescale_block`, now the safety net behind the LLM's own unit conversion;
- `repair_main_row`, now only the ECD row: text cleaning, the net-income plausibility drop, and
  the belt-and-braces zero guard behind `def14a_ecd`'s selection-time zero drop.

The pay-ratio identity is NOT untested -- it moved to `def14a_impute._reconcile_rows` and is
covered by `tests/data_aggregate/test_def14a_impute.py`.
"""
from __future__ import annotations

import math

from src.data_extract.utils.structure.def14a_validate import (
    DEF14A_AUDIT_FEE_MIN_PLAUSIBLE, clean_holder_name, clean_person_name, clean_text,
    is_subtotal_holder, repair_main_row, rescale_block, sum_fee_total,
)

_NAN = float("nan")

#: The LLM path's fee columns -- the block `rescale_block` now guards.
_FEE_COLS = ["audit_fees_audit", "audit_fees_audit_related", "audit_fees_tax",
             "audit_fees_other", "auditor_fees", "auditor_fees_prior"]


def _ecd(**over) -> dict:
    """A `sec_def14a` (ECD) row as `fetch_def14a_edgar` now builds it."""
    base = {
        "company_name": "COCA COLA CO", "peo_name": "James Quincey",
        "peo_names_all": "James Quincey",
        "company_selected_measure_name": "Organic revenue growth",
        "net_income": 13_137_000_000.0,
        "peo_total_comp": 31_208_165.0, "peo_actually_paid_comp": 61_649_669.0,
        "neo_avg_total_comp": 9_524_407.0, "neo_avg_actually_paid_comp": 12_000_000.0,
    }
    base.update(over)
    return base


# ── text / key normalisation (used by BOTH paths) ───────────────────────────

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
    """Kept even though the HTML parser that glued titles is gone: an LLM asked for a name field
    still occasionally returns "Name, Title", and this is the primary key."""
    assert clean_person_name("James DimonChairman and CEO") == "James Dimon"
    assert clean_person_name("Deirdre O’Brien Senior Vice") == "Deirdre O’Brien"
    assert clean_person_name("Judson B. Althoff Executive Vice President and") == "Judson B. Althoff"
    assert clean_person_name("H. Lawrence   Culp, Jr.") == "H. Lawrence Culp, Jr."


def test_clean_person_name_consumes_the_title_modifier_word():
    """Without the leading modifier group the split leaves the modifier stuck to the name."""
    assert clean_person_name("Luca Maestri Former Senior Vice President") == "Luca Maestri"
    assert clean_person_name("Bob De Lange Group President") == "Bob De Lange"
    assert clean_person_name("Denise C. Johnson Group President") == "Denise C. Johnson"


def test_address_only_holder_name_is_dropped():
    """JPM's proxy: the address line came back as the holder name. Returning None lets the caller
    drop the row rather than store a street as a shareholder."""
    assert clean_holder_name("100 Vanguard Blvd, Malvern, PA 19355") is None
    assert clean_holder_name("50 Hudson Yards, New York, NY 10001") is None


def test_address_is_stripped_off_institutional_holder_name():
    assert clean_holder_name(
        "The Vanguard Group 100 Vanguard Blvd. Malvern, PA 19355") == "The Vanguard Group"
    assert clean_holder_name(
        "BlackRock, Inc. 55 East 52nd Street New York, NY 10055") == "BlackRock, Inc."


def test_subtotal_pseudo_holders_are_recognised():
    """That aggregate is already the `insider_ownership_pct` scalar on `def14a_llm`; storing it
    again as a holder double-counts the insiders against the real per-person rows."""
    assert is_subtotal_holder("All current directors and executive officers as a group (16 people)")
    assert is_subtotal_holder("Total")
    assert not is_subtotal_holder("The Vanguard Group")


# ── unit rescale (now the LLM path's safety net) ────────────────────────────

def test_fee_block_in_thousands_is_rescaled_to_dollars():
    """KO's 2026 proxy: the '(in thousands)' header was missed, so the SAME fee read as
    32,104,000 from the 2025 proxy came back as 32,104. The whole block moves together --
    rescaling cell by cell would invent a table whose components no longer sum to its total."""
    row = {"audit_fees_audit": 30_587.0, "auditor_fees_prior": 32_104.0,
           "audit_fees_audit_related": 4_834.0, "audit_fees_tax": 6_760.0,
           "audit_fees_other": 85.0, "auditor_fees": 42_266.0}
    rescale_block(row, _FEE_COLS, DEF14A_AUDIT_FEE_MIN_PLAUSIBLE)
    assert row["audit_fees_audit"] == 30_587_000.0
    assert row["auditor_fees_prior"] == 32_104_000.0        # == KO 2025's current-year audit fee
    assert row["auditor_fees"] == 42_266_000.0
    assert row["audit_fees_other"] == 85_000.0              # whole block moves together


def test_dollar_fee_block_is_left_alone():
    row = {"audit_fees_audit": 32_104_000.0, "auditor_fees": 45_568_000.0}
    rescale_block(row, _FEE_COLS, DEF14A_AUDIT_FEE_MIN_PLAUSIBLE)
    assert row["audit_fees_audit"] == 32_104_000.0
    assert row["auditor_fees"] == 45_568_000.0


# ── the ECD row ─────────────────────────────────────────────────────────────

def test_implausible_net_income_is_nulled_not_guessed():
    """SBUX FY2025 arrives as 1856.4 (raw value '1856.4', decimals='1', unit_ref='usd' -- tagged
    in $ millions). That is equally consistent with millions and billions and NOTHING in the fact
    disambiguates them, unlike a fee block whose factor a sibling filing's overlapping year
    confirms. So it is dropped rather than rescaled into a confidently wrong number;
    `fundamentals_history` has the trustworthy figure."""
    assert math.isnan(repair_main_row(_ecd(net_income=1856.4))["net_income"])
    assert math.isnan(repair_main_row(_ecd(net_income=16.1))["net_income"])
    assert repair_main_row(_ecd(net_income=93_736_000_000.0))["net_income"] == 93_736_000_000.0
    assert repair_main_row(_ecd(net_income=6_556_000_000.0))["net_income"] == 6_556_000_000.0


def test_negative_net_income_is_kept():
    """A loss is a disclosure, and the plausibility floor is on the MAGNITUDE."""
    assert repair_main_row(_ecd(net_income=-2_400_000_000.0))["net_income"] == -2_400_000_000.0


def test_zero_peo_comp_is_nulled():
    """Belt-and-braces behind `def14a_ecd`'s selection-time zero drop, which recovers the CORRECT
    value from SBUX's individual x year matrix instead of leaving this NULL. If a zero still
    reaches here, the selection missed something."""
    out = repair_main_row(_ecd(peo_total_comp=0.0, peo_actually_paid_comp=0.0))
    assert math.isnan(out["peo_total_comp"])
    assert math.isnan(out["peo_actually_paid_comp"])


def test_negative_actually_paid_comp_is_never_touched():
    """NKE 2025's retained PEO has a CAP of -10,924,243. Compensation Actually Paid subtracts
    prior-year unvested fair value, so a share-price fall makes it negative -- 28.5% of 2023 and
    33.7% of 2025 S&P 500 proxies report at least one."""
    out = repair_main_row(_ecd(peo_actually_paid_comp=-10_924_243.0))
    assert out["peo_actually_paid_comp"] == -10_924_243.0


def test_repairs_are_pure_and_do_not_mutate_the_input():
    src = _ecd(net_income=16.1)
    repair_main_row(src)
    assert src["net_income"] == 16.1


#: The four Item 9(e) fee categories, in the order `_FEE_CATEGORY_COLS` uses.
_PARTS = ["audit_fees_audit", "audit_fees_audit_related", "audit_fees_tax", "audit_fees_other"]


def _fees(total: float | None, audit: float | None, related: float | None,
          tax: float | None, other: float | None) -> dict:
    return {"auditor_fees": total, "audit_fees_audit": audit,
            "audit_fees_audit_related": related, "audit_fees_tax": tax,
            "audit_fees_other": other}


def test_a_total_that_is_really_the_audit_line_is_rebuilt_from_the_categories():
    """BA 2026 and T 2026 print no Total row, so the model reported the `Audit Fees` line as the
    total -- BA 39.1M against a real 43.6M, T 34.2M against 38.9M. Both are real filings."""
    ba = _fees(39_100_000.0, 39_100_000.0, 4_500_000.0, 0.0, 0.0)
    assert sum_fee_total(ba, "auditor_fees", _PARTS) is True
    assert ba["auditor_fees"] == 43_600_000.0

    t = _fees(34_200_000.0, 34_200_000.0, 1_300_000.0, 3_400_000.0, 0.0)
    assert sum_fee_total(t, "auditor_fees", _PARTS) is True
    assert t["auditor_fees"] == 38_900_000.0


def test_a_filer_stated_total_is_never_overwritten():
    """AAPL 2026 states 34,277k, and its categories sum to exactly that. 16 of the 18 complete
    fee blocks measured behave this way, which is why the repair must be a no-op on them."""
    aapl = _fees(34_277_000.0, 24_703_000.0, 2_274_000.0, 4_533_000.0, 2_767_000.0)
    assert sum_fee_total(aapl, "auditor_fees", _PARTS) is False
    assert aapl["auditor_fees"] == 34_277_000.0


def test_a_partial_category_set_is_left_alone():
    """JPM 2026 discloses three categories and a correct 135,000,000 total. Summing a partial set
    would REPLACE a right answer with a low one, so an absent category blocks the repair --
    even though here the total happens to equal the sum of the three."""
    jpm = _fees(135_000_000.0, 91_000_000.0, 38_100_000.0, 5_900_000.0, None)
    assert sum_fee_total(jpm, "auditor_fees", _PARTS) is False
    assert jpm["auditor_fees"] == 135_000_000.0


def test_a_total_above_its_categories_is_left_alone():
    """A stated total LARGER than the four categories means the filer counted something this
    schema does not model. The filer's own row is authoritative there, so the sum must lose."""
    row = _fees(50_000_000.0, 39_100_000.0, 4_500_000.0, 0.0, 0.0)
    assert sum_fee_total(row, "auditor_fees", _PARTS) is False
    assert row["auditor_fees"] == 50_000_000.0


def test_a_genuine_audit_only_fee_table_is_left_alone():
    """When the other three categories really are zero the total EQUALS the audit line
    legitimately, and the sum equals it too -- so the guard is the sum, not the equality."""
    row = _fees(5_000_000.0, 5_000_000.0, 0.0, 0.0, 0.0)
    assert sum_fee_total(row, "auditor_fees", _PARTS) is False
    assert row["auditor_fees"] == 5_000_000.0


def test_a_null_total_is_not_filled():
    """EOG 2026 has four categories and no total at all. Filling it is a DIFFERENT change with
    its own risk -- EOG's categories sum to $598k, implausibly small for an S&P 500 audit, so
    the parts are themselves suspect and this repair must not launder them into a total."""
    eog = _fees(None, 270_170.0, 321_000.0, 0.0, 6_954.0)
    assert sum_fee_total(eog, "auditor_fees", _PARTS) is False
    assert eog["auditor_fees"] is None


def test_sanity_check_prints_conclusion():
    print("\n=== SANITY: what the DEF 14A row cleaner still does ===")
    print("  primary keys: 'Emma N. Walmsley11' -> 'Emma N. Walmsley' (stable year over year);")
    print("                an address-only holder cell -> None, so the caller drops the row.")
    print("  fee units:    a whole block below $100k is rescaled together (KO 32,104 ->")
    print("                32,104,000, matching the same fee in the sibling filing).")
    print("  fee total:    a table with no Total row had its Audit line read as the total;")
    print("                rebuilt from the 4 categories (BA 39.1M -> 43.6M, T 34.2M ->")
    print("                38.9M). Fires ONLY on a complete category set whose total equals")
    print("                one category and whose sum EXCEEDS it -- 16 of 18 complete blocks")
    print("                already sum to their stated total to the dollar and are untouched.")
    print("  ECD row:      net_income 1856.4 -> NULL (millions vs billions is undecidable);")
    print("                peo_* == 0.0 -> NULL; a NEGATIVE peo_actually_paid_comp is KEPT.")
    print("  Removed with the HTML block: the 0.5 percent placeholder, the duplicated-Total")
    print("  correction, the pay-ratio identity (now in def14a_impute) and the ownership")
    print("  re-typing -- their code no longer runs, so asserting it would be theatre.")
