"""Acceptance gate for quarterly ZIP versus daily EDGAR ownership rows."""

from __future__ import annotations

import pandas as pd

from src.validate.checks.insider_parity import (
    InsiderParityThresholds,
    reconcile_feature_panels,
    reconcile_transactions,
)

THRESHOLDS = InsiderParityThresholds(
    accession_coverage=0.99,
    exact_structure=0.99,
    categorical_agreement=0.999,
    share_agreement=0.999,
    economic_within_tolerance=0.99,
    identity_agreement=0.999,
    numeric_relative_tolerance=0.01,
    aggregate_value_relative_tolerance=0.01,
    feature_cell_agreement=0.99,
    edge_rank_correlation=0.99,
)


def _row(accession: str, price: float = 91.67) -> dict[str, object]:
    return {
        "accession_number": accession,
        "security_type": "nonderiv",
        "ticker": "AAA",
        "filing_date": pd.Timestamp("2026-06-30"),
        "transaction_date": pd.Timestamp("2026-06-29"),
        "transaction_code": "S",
        "acquired_disposed": "D",
        "direct_indirect": "D",
        "security_title": "Common Stock",
        "issuer_cik": "1",
        "owner_cik": "2",
        "owner_name": "DOE JANE",
        "is_director": 1.0,
        "is_officer": 0.0,
        "is_ten_pct_owner": 0.0,
        "officer_title": "Director",
        "is_10b5_1": 1.0,
        "shares": 1_000.0,
        "price_per_share": price,
        "value_usd": 1_000.0 * price,
        "shares_owned_after": 9_000.0,
    }


def test_zip_rounding_difference_passes_the_multi_level_gate():
    bulk = pd.DataFrame([_row("A", 91.67), _row("B", 10.00)])
    live = pd.DataFrame([_row("B", 10.00), _row("A", 91.6725)])
    result = reconcile_transactions(bulk, live, THRESHOLDS)
    assert result["passed"]
    assert result["accession_coverage"] == 1.0
    assert result["exact_structure"] == 1.0
    assert result["economic_within_tolerance"] == 1.0
    print(
        "SANITY: reordered EDGAR rows with 91.6725 versus ZIP's 91.67 passed at 100% "
        "coverage and structure; harmless source rounding stayed inside the 1% gate."
    )


def test_small_aggregate_difference_cannot_hide_a_missing_accession():
    bulk = pd.DataFrame([_row(f"A{i}", 10.0) for i in range(100)])
    live = bulk.iloc[:-2].copy()
    result = reconcile_transactions(bulk, live, THRESHOLDS)
    assert not result["passed"]
    assert result["accession_coverage"] == 0.98
    assert result["missing_accessions"] == ["A98", "A99"]
    print("SANITY: losing two of 100 filings failed the 99% accession gate even though the " "remaining transaction economics matched exactly.")


def test_a_missing_cube_input_column_cannot_pass_on_the_remaining_fields():
    bulk = pd.DataFrame([_row("A")])
    live = bulk.drop(columns="owner_cik")
    result = reconcile_transactions(bulk, live, THRESHOLDS)
    assert not result["passed"]
    assert result["missing_live_columns"] == ["owner_cik"]
    print(
        "SANITY: omitting owner_cik fails the source contract explicitly; high agreement on " "the remaining fields cannot hide a missing cube input."
    )


def test_feature_gate_requires_identical_null_masks():
    bulk = pd.DataFrame({"date": [pd.Timestamp("2026-06-30")], "ticker": ["AAA"], "f_signal": [0.0]})
    live = bulk.assign(f_signal=float("nan"))
    result = reconcile_feature_panels(bulk, live, THRESHOLDS)
    assert not result["passed"]
    assert result["null_mask_agreement"] == 0.0
    print(
        "SANITY: a numeric zero versus unavailable NaN failed feature parity; matching source "
        "totals cannot erase a point-in-time availability mismatch."
    )
