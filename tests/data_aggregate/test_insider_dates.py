"""Invalid source dates retain disclosure evidence without certifying execution timing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.institutionals.inputs import QuantityBasis, load_insider_transactions
from src.data_aggregate.utils.institutionals.insider_positions import position_units
from src.data_aggregate.utils.institutionals.insider_quality import clean_transactions, normalize_insider_dates
from src.data_store.schema import Tables
from src.utils.price_adjustments import ShareBasis
from tests.data_aggregate.insider_pit_reference import row, source_frame
from tests.data_aggregate.test_insider_features import _action_cap_inputs, _build_action_cap


@pytest.mark.parametrize("column", ["original_submission_date", "transaction_date"])
def test_ancient_microsecond_dates_reach_actual_cleaner_safely(column: str) -> None:
    df_source = pd.DataFrame([row(document_type="4/A", transaction_code="M" if column == "original_submission_date" else "P")])
    df_source[column] = pd.Series(np.array(["0001-01-01"], dtype="datetime64[us]"))
    df_source = normalize_insider_dates(df_source)
    df_original = df_source.copy(deep=True)
    df_clean, diagnostics = clean_transactions(df_source)
    df_result = diagnostics["other_flows"] if column == "original_submission_date" else df_clean

    assert len(df_result) == 1 and df_result[column].isna().all()
    assert df_result[column + "_raw"].iloc[0].year == 1
    assert diagnostics["invalid_" + column + "s"] == 1
    pd.testing.assert_frame_equal(df_source, df_original)
    print(f"SANITY: actual cleaner rejects typed microsecond year1 {column} before nanosecond keys, preserving raw source and input.")


def test_nanosecond_minimum_is_rejected_before_midnight_normalization() -> None:
    df_source = pd.DataFrame([row()])
    df_source["transaction_date"] = pd.Series([pd.Timestamp.min], dtype="datetime64[ns]")
    df_source = normalize_insider_dates(df_source)
    df_clean, diagnostics = clean_transactions(df_source)
    assert df_clean["transaction_day"].isna().all() and diagnostics["invalid_transaction_dates"] == 1
    assert df_clean["transaction_date_raw"].iloc[0] == pd.Timestamp.min
    assert df_clean["reference_day"].eq(pd.Timestamp("2024-01-10")).all()
    print("SANITY: Timestamp.min is rejected before midnight arithmetic can overflow; its raw value survives and only filing reference is available.")


def test_unusable_execution_dates_have_only_a_marked_filing_reference() -> None:
    actions: list[object] = ["0001-01-01", None, "bad-date", "2024-04-01"]
    df_source = source_frame(
        [row(accession_number=f"bad-{n}", filing_date="2024-02-01", transaction_date=action) for n, action in enumerate(actions)]
    )
    df_clean, diagnostics = clean_transactions(df_source)

    assert len(df_clean) == 4 and df_clean["transaction_date"].isna().all() and df_clean["transaction_day"].isna().all()
    assert str(df_clean["transaction_day"].dtype) == "datetime64[ns]"
    assert df_clean["reference_day"].eq(pd.Timestamp("2024-02-01")).all() and df_clean["reference_day_estimated"].all()
    pd.testing.assert_series_equal(df_clean["transaction_date_raw"], df_source["transaction_date_raw"], check_names=False)
    assert diagnostics["date_fallback_rows"] == 4 and diagnostics["copy_rows_dropped"] == 0
    df_repeat, _ = clean_transactions(df_clean)
    pd.testing.assert_series_equal(df_repeat["transaction_date_raw"], df_source["transaction_date_raw"], check_names=False)
    print("SANITY: ancient, missing, malformed and post-filing execution dates stay unknown; marked filing references never become copy identity.")


@pytest.mark.parametrize("submitted", ["0001-01-01", "2024-03-01"])
def test_invalid_original_submission_cannot_fabricate_an_amendment_link(submitted: str) -> None:
    df_source = source_frame(
        [
            row(accession_number="original", filing_date="2024-01-10"),
            row(accession_number="amendment", filing_date="2024-02-01", original_submission_date=submitted, document_type="4/A", shares=1100.0),
        ]
    )
    df_clean, diagnostics = clean_transactions(df_source)
    amendment = df_clean.loc[df_clean["accession_number"].eq("amendment")]
    assert amendment["original_submission_date"].isna().all() and amendment["original_submission_date_raw"].tolist() == [submitted]
    assert diagnostics["amendments_linked"] == 0 and diagnostics["invalid_original_submission_dates"] == 1
    assert df_clean["visible_until"].isna().all()
    print("SANITY: ancient or post-filing original submission is unknown, preserves raw provenance, and establishes no amendment link.")


def test_future_quotes_do_not_reprice_a_filing_date_fallback() -> None:
    df_source = source_frame([row(filing_date="2024-02-01", transaction_date="2024-04-01", price_per_share=100_000.0)])
    df_quotes = pd.DataFrame({"AAA": [100.0, 1000.0]}, index=pd.to_datetime(["2024-02-01", "2024-04-01"]))
    df_prefix, _ = clean_transactions(df_source, df_valuation_close=df_quotes.iloc[:1])
    df_full, _ = clean_transactions(df_source, df_valuation_close=df_quotes)

    assert df_full["price_clean"].tolist() == [100.0] and df_full["value"].tolist() == [100_000.0]
    assert df_full["price_estimated"].all() and df_full["reference_day_estimated"].all()
    pd.testing.assert_frame_equal(df_prefix, df_full)
    print(
        "SANITY: the available filing-day quote repairs a corrupt price; adding the invalid future execution quote cannot change historical output."
    )


def test_unknown_dates_do_not_identify_packages_or_amendment_cells() -> None:
    df_source = source_frame(
        [
            row(accession_number="package", filing_date="2024-02-01", transaction_date="2024-04-01", transaction_code="M", security_type="deriv"),
            row(accession_number="package", filing_date="2024-02-01", transaction_date="2024-04-01", transaction_code="S"),
            row(accession_number="original", filing_date="2024-01-10", transaction_date=None),
            row(
                accession_number="amendment",
                filing_date="2024-02-01",
                transaction_date=None,
                document_type="4/A",
                original_submission_date="2024-01-10",
            ),
        ]
    )
    df_clean, diagnostics = clean_transactions(df_source)
    assert len(df_clean) == 3 and not df_clean["in_exercise_package"].any()
    assert diagnostics["amendments_linked"] == 1
    assert diagnostics["amendment_cells_identical"] == 0 and diagnostics["amendment_cells_superseded"] == 0
    assert df_clean["visible_until"].isna().all()
    print("SANITY: unknown execution dates cannot prove an exercise package or equal amendment cells even when a genuine filing link exists.")


def test_unknown_non_market_flow_dates_cannot_share_an_amendment_boundary() -> None:
    df_source = source_frame(
        [
            row(accession_number="original", filing_date="2024-01-10", transaction_date=None, transaction_code="G"),
            row(
                accession_number="amendment",
                filing_date="2024-02-01",
                transaction_date=None,
                transaction_code="G",
                document_type="4/A",
                original_submission_date="2024-01-10",
            ),
        ]
    )
    _, diagnostics = clean_transactions(df_source)
    df_flows = diagnostics["other_flows"]
    assert len(df_flows) == 2 and df_flows["transaction_day"].isna().all() and df_flows["visible_until"].isna().all()
    print("SANITY: non-P/S unknown execution dates remain independent account blockers rather than matching amendment cells.")


def _future_versions(code: str) -> list[dict[str, Any]]:
    return [
        row(accession_number="future-original", filing_date="2024-03-08", transaction_date="2024-03-11", transaction_code=code),
        row(
            accession_number="future-amended",
            filing_date="2024-03-12",
            transaction_date="2024-03-11",
            transaction_code=code,
            document_type="4/A",
            original_submission_date="2024-03-08",
        ),
    ]


@pytest.mark.parametrize("code", ["P", "G"])
def test_later_valid_amendment_confirms_a_declared_day_only_prospectively(code: str) -> None:
    df_source = source_frame(_future_versions(code))
    df_original = df_source.copy(deep=True)
    df_clean, diagnostics = clean_transactions(df_source)
    df_versions = (df_clean if code == "P" else diagnostics["other_flows"]).set_index("accession_number")
    df_original_version, df_amended = df_versions.loc[["future-original"]], df_versions.loc[["future-amended"]]

    assert len(df_original_version) == len(df_amended) == 1
    assert df_original_version["transaction_day"].isna().all() and df_original_version["transaction_date_raw"].tolist() == ["2024-03-11"]
    assert df_original_version["reference_day"].eq(pd.Timestamp("2024-03-08")).all() and df_original_version["reference_day_estimated"].all()
    assert df_original_version["visible_until"].eq(pd.Timestamp("2024-03-12")).all()
    assert df_amended["transaction_day"].eq(pd.Timestamp("2024-03-11")).all() and not df_amended["reference_day_estimated"].any()
    assert df_amended["visible_from"].eq(pd.Timestamp("2024-03-12")).all()
    if code == "P":
        assert df_versions["anchor"].eq(pd.Timestamp("2024-03-08")).all()
        assert diagnostics["amendment_cells_identical"] == 0
    df_again, repeated = clean_transactions(df_versions.reset_index())
    df_again = (df_again if code == "P" else repeated["other_flows"]).set_index("accession_number")
    pd.testing.assert_series_equal(df_again["transaction_date_raw"], df_versions["transaction_date_raw"])
    pd.testing.assert_frame_equal(df_source, df_original)
    print(
        f"SANITY: {code=}; later verified timing retires only the old public version, retaining its unknown action/raw provenance and disclosure age."
    )


@pytest.mark.parametrize("code", ["P", "G"])
@pytest.mark.parametrize(
    "case", ["missing", "ancient", "different-day-no-row-proof", "future-amendment", "invalid-submission", "ambiguous", "code", "security"]
)
def test_declared_day_cannot_replace_independent_amendment_proof(code: str, case: str) -> None:
    rows = _future_versions(code)
    if case == "missing":
        rows[0]["transaction_date"] = None
    elif case == "ancient":
        rows[0]["transaction_date"] = "0001-01-01"
    elif case == "different-day-no-row-proof":
        rows[0]["transaction_date"] = "2024-03-10"
        rows[0]["row_sequence"] = None
    elif case == "future-amendment":
        rows[1]["filing_date"] = "2024-03-10"
    elif case == "invalid-submission":
        rows[1]["original_submission_date"] = "0001-01-01"
    elif case == "ambiguous":
        rows.insert(1, rows[0] | {"accession_number": "other-original"})
    elif case == "code":
        rows[1]["transaction_code"] = "S" if code == "P" else "F"
    elif case == "security":
        rows[1]["security_type"] = "deriv"
    df_clean, diagnostics = clean_transactions(source_frame(rows))
    df_original = (df_clean if code == "P" else diagnostics["other_flows"]).set_index("accession_number").loc[["future-original"]]
    assert len(df_original) == 1 and df_original["transaction_day"].isna().all() and df_original["visible_until"].isna().all()
    print(f"SANITY: {code=} {case=}; a raw date cannot establish a filing link or replace a verified matching action cell.")


@pytest.mark.parametrize("code", ["P", "G"])
def test_later_confirmation_does_not_merge_earlier_unknown_versions(code: str) -> None:
    original, confirmed = _future_versions(code)
    pending = confirmed | {"accession_number": "future-pending", "filing_date": "2024-03-10"}
    df_clean, diagnostics = clean_transactions(source_frame([original, pending, confirmed]))
    df_versions = (df_clean if code == "P" else diagnostics["other_flows"]).set_index("accession_number")
    assert df_versions.loc["future-original", "visible_until"] == pd.Timestamp("2024-03-12")
    assert pd.isna(df_versions.loc["future-pending", "visible_until"]) and pd.isna(df_versions.loc["future-pending", "transaction_day"])
    if code == "P":
        assert df_versions.loc["future-pending", "anchor"] == pd.Timestamp("2024-03-10")
    print(f"SANITY: {code=}; confirmation on March 12 cannot equate two earlier unknown action dates or retire either version on March 10.")


@pytest.mark.parametrize("case", ["complete", "missing-row", "owner", "account", "class", "cardinality", "amount", "ambiguous", "missing", "ancient"])
def test_explicit_row_proof_can_correct_only_a_safe_original_declaration(case: str) -> None:
    rows = _future_versions("P")
    rows[1]["transaction_date"] = "2024-03-04"
    if case == "missing-row":
        rows[0]["row_sequence"] = None
    elif case == "owner":
        rows[1].update(owner_cik="2", owner_ciks="2")
    elif case == "account":
        rows[1]["direct_indirect"] = "I"
    elif case == "class":
        rows[1]["security_title"] = "Class B Common Stock"
    elif case == "cardinality":
        rows[1]["n_reporting_owners"] = 2
    elif case == "amount":
        rows[1]["shares"] = 1100.0
    elif case == "ambiguous":
        rows.insert(1, rows[0] | {"accession_number": "other-original"})
    elif case == "missing":
        rows[0]["transaction_date"] = None
    elif case == "ancient":
        rows[0]["transaction_date"] = "0001-01-01"
    df_clean, _ = clean_transactions(source_frame(rows))
    df_versions = df_clean.set_index("accession_number")
    df_original = df_versions.loc[["future-original"]]
    assert df_original["transaction_day"].isna().all()
    if case == "complete":
        assert df_original["visible_until"].eq(pd.Timestamp("2024-03-12")).all()
        assert df_versions["anchor"].eq(pd.Timestamp("2024-03-08")).all()
        assert df_versions.loc[["future-amended"], "transaction_day"].eq(pd.Timestamp("2024-03-04")).all()
    else:
        assert df_original["visible_until"].isna().all()
    print(
        f"SANITY: {case=}; only complete independent row correspondence can prospectively correct a safe future declaration to a different valid day."
    )


def test_later_row_proof_cannot_retire_an_earlier_unknown_amendment() -> None:
    original, confirmed = _future_versions("P")
    pending = confirmed | {"accession_number": "future-pending", "filing_date": "2024-03-10"}
    confirmed["transaction_date"] = "2024-03-04"
    df_clean, _ = clean_transactions(source_frame([original, pending, confirmed]))
    df_versions = df_clean.set_index("accession_number")
    assert df_versions.loc[["future-original"], "visible_until"].eq(pd.Timestamp("2024-03-12")).all()
    assert df_versions.loc[["future-pending"], "visible_until"].isna().all()
    assert df_versions.loc[["future-pending"], "transaction_day"].isna().all()
    print(
        "SANITY: later row proof can retire the original only on March 12; an earlier unknown amendment remains an independent uncertainty barrier."
    )


def test_later_row_confirmation_cannot_borrow_an_earlier_ambiguous_link() -> None:
    original, confirmed = _future_versions("P")
    other = original | {"accession_number": "z-other-original", "owner_cik": "2", "owner_ciks": "2", "transaction_date": "2024-03-20"}
    ambiguous = confirmed | {"accession_number": "earlier-ambiguous", "filing_date": "2024-03-11", "owner_ciks": "1,2", "n_reporting_owners": 2}
    confirmed["transaction_date"] = "2024-03-04"
    df_clean, _ = clean_transactions(source_frame([original, other, ambiguous, confirmed]))
    df_versions = df_clean.set_index("accession_number")
    assert df_versions.loc[["future-original"], "visible_until"].eq(pd.Timestamp("2024-03-12")).all()
    assert df_versions.loc[["earlier-ambiguous"], "anchor"].eq(pd.Timestamp("2024-03-11")).all()
    print("SANITY: proof first available March 12 cannot retroactively make an ambiguous March 11 filing retire the unknown original.")


def test_unknown_execution_keeps_actual_panel_cap_quantity_and_accounts_unsupported() -> None:
    _, df_close, df_history = _action_cap_inputs()
    source = row(filing_date="2024-03-05", transaction_date="2024-03-20", officer_title="CEO", is_officer=1.0)
    df_clean, _ = clean_transactions(source_frame([source]))
    df_empty = pd.DataFrame(columns=["ticker", "date", "value"])
    basis = QuantityBasis.from_context(ShareBasis(df_empty, df_empty, df_empty, {}), pd.DatetimeIndex(df_close.index))
    df_positions = position_units(df_clean, basis)
    df_panel = _build_action_cap([source], df_close, df_history)
    unknown = [
        "f_ic_insider_buy_value_mcap_60d",
        "f_ic_insider_purchase_pct_prior",
        "f_ic_insider_management_position_change_60d",
    ]

    assert not df_positions["units_known"].any() and not df_positions["declared_units_known"].any()
    assert df_panel.reindex(index=[pd.Timestamp("2024-03-07")], columns=unknown).isna().to_numpy().all()
    assert df_panel.loc["2024-03-07", "f_ic_insider_distinct_buyers_120d"] == 1.0
    df_control = _build_action_cap([source | {"transaction_date": "2024-03-04"}], df_close, df_history)
    assert df_control.reindex(index=[pd.Timestamp("2024-03-07")], columns=unknown).gt(0).to_numpy().all()
    print(
        "SANITY: actual panel retains the disclosed buyer but cannot borrow fallback timing for action capital, quantity rebasing or account ratios."
    )


def test_loader_normalizes_before_first_filing_and_role_metadata(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    df_source = pd.DataFrame(
        [
            row(accession_number="ancient", issuer_cik="123", lineage_role="canonical_current"),
            row(accession_number="valid", issuer_cik="123", lineage_role="canonical_current"),
        ]
    )
    sqlite_store.save(Tables.insider_transactions, df_source)
    df_source["filing_date"] = pd.Series(np.array(["0001-01-01", "2024-01-10"], dtype="datetime64[us]"))
    original = sqlite_store.iter_load

    def stream(table: Any, **kwargs: Any) -> Any:
        return iter([df_source]) if table == Tables.insider_transactions else original(table, **kwargs)

    monkeypatch.setattr(sqlite_store, "iter_load", stream)
    loaded = load_insider_transactions(sqlite_store, ["AAA"])
    assert loaded is not None and loaded.first_filed["AAA"] == pd.Timestamp("2024-01-10")
    assert loaded.transactions["filing_date"].isna().tolist() == [True, False]
    assert loaded.transactions["filing_date_raw"].iloc[0].year == 1
    df_clean, diagnostics = clean_transactions(loaded.transactions)
    assert len(df_clean) == 1 and diagnostics["unavailable_filing_dates"] == 1
    print("SANITY: actual loader applies date policy before first_filed and downstream roles; an ancient public date cannot establish support.")
