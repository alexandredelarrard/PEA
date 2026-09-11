"""
test_13f_managers.py (tests/data_extract/institutionals/test_13f_managers.py)
------------------------------------------------------------------------------
`fetch_13f_managers`'s parse. Two invariants carry the whole table:

  * NO UNIVERSE FILTER. `sec13f_hr` keeps the S&P 500 slice of a manager's book, which is why a
    weight computed from it is inflated by a manager-specific 1.0x-7.6x. This table exists to be
    the denominator, so a non-S&P500 CUSIP surviving is not a detail -- it is the feature.
  * `position_type` AGREES WITH `_classify_holdings`. Both read `_holding_masks`, and this test
    is what keeps that true: if someone re-derives either side, the buckets and the label
    disagree while every column still sums correctly.
"""
import pandas as pd

from src.data_extract.utils.institutionals.fetch_13f import (
    POSITION_TYPES, _classify_holdings, position_type)
from src.data_extract.utils.institutionals.fetch_13f_managers import (
    _COLS, _dominant_type, _manager_holdings_frame)


def _infotable(rows: list[dict]) -> pd.DataFrame:
    """An edgartools-shaped info table. `SSHPRNAMTTYPE` is spelled SH/PRN in the bulk TSVs and
    Shares/Principal by edgartools; both spellings are accepted, so both appear here."""
    return pd.DataFrame(rows)


_MIXED = _infotable([
    # a plain common line -- blank type, no put/call, the overwhelmingly common case
    {"CUSIP": "037833100", "NAMEOFISSUER": "APPLE INC", "TITLEOFCLASS": "COM",
     "VALUE": 1_000_000.0, "SSHPRNAMT": 5_000, "SSHPRNAMTTYPE": "SH", "PUTCALL": ""},
    {"CUSIP": "88160R101", "NAMEOFISSUER": "TESLA INC", "TITLEOFCLASS": "COM",
     "VALUE": 400_000.0, "SSHPRNAMT": 2_000, "SSHPRNAMTTYPE": "Shares", "PUTCALL": "CALL"},
    {"CUSIP": "594918104", "NAMEOFISSUER": "MICROSOFT CORP", "TITLEOFCLASS": "COM",
     "VALUE": 250_000.0, "SSHPRNAMT": 700, "SSHPRNAMTTYPE": "SH", "PUTCALL": "PUT"},
    {"CUSIP": "459200101", "NAMEOFISSUER": "IBM CORP", "TITLEOFCLASS": "NOTE 3.5% 2030",
     "VALUE": 900_000.0, "SSHPRNAMT": 900_000, "SSHPRNAMTTYPE": "PRN", "PUTCALL": ""},
    {"CUSIP": "G0450A105", "NAMEOFISSUER": "A NON SP500 NAME", "TITLEOFCLASS": "ORD",
     "VALUE": 123_000.0, "SSHPRNAMT": 1_500, "SSHPRNAMTTYPE": "", "PUTCALL": ""},
])


def test_position_type_agrees_with_the_classified_buckets():
    """The label and the columns must be derived from ONE set of masks. Checked by asserting
    that the value each row carries sits in the column its label names."""
    labels = position_type(_MIXED)
    buckets = _classify_holdings(_MIXED)
    column_for = {"common": "value_usd", "call": "call_value", "put": "put_value",
                  "debt": "debt_value", "other": "other_value"}
    assert list(labels) == ["common", "call", "put", "debt", "common"]
    for i, label in enumerate(labels):
        assert buckets[column_for[label]].iloc[i] == _MIXED["VALUE"].iloc[i], (
            f"row {i} labelled {label} but its value is not in {column_for[label]}")
        others = [c for k, c in column_for.items() if k != label]
        assert (buckets[others].iloc[i] == 0).all(), f"row {i} leaked value into another bucket"
    print("\n=== SANITY: 13F position_type vs buckets ===")
    print(f"  {len(labels)} lines labelled {list(labels)}; each row's VALUE lands in exactly "
          f"the column its label names. One mask set, no drift. Validated.")


def test_a_blank_amount_type_is_common_not_other():
    """The blank type is what older data sets omit, and it is long stock -- reading it as
    'other' would quietly move real equity out of every conviction denominator."""
    assert position_type(_MIXED).iloc[4] == "common"
    assert set(position_type(_MIXED)) <= set(POSITION_TYPES)


def test_no_universe_filter_is_applied():
    """G0450A105 is a non-S&P500 (indeed non-US) CUSIP. `fetch_13f` drops it; this table's whole
    purpose is that it does not."""
    out = _manager_holdings_frame("0001067983", "2026-05-10", "2026-03-31", _MIXED)
    assert "G0450A105" in set(out["cusip"])
    assert len(out) == 5
    assert list(out.columns) == _COLS
    assert "ticker" not in out.columns          # by design -- see the schema docstring
    print("\n=== SANITY: 13F manager book has no universe filter ===")
    print(f"  {len(out)} CUSIPs kept including the non-S&P500 G0450A105. Validated.")


def test_split_sub_account_lines_collapse_to_one_summed_row():
    """A manager files several lines for one security (split sub-accounts). The PK is
    (cik, period, cusip), so they must sum rather than fight over the row."""
    split = _infotable([
        {"CUSIP": "037833100", "NAMEOFISSUER": "APPLE INC", "TITLEOFCLASS": "COM",
         "VALUE": 600_000.0, "SSHPRNAMT": 3_000, "SSHPRNAMTTYPE": "SH", "PUTCALL": ""},
        {"CUSIP": "037833100", "NAMEOFISSUER": "APPLE INC", "TITLEOFCLASS": "COM",
         "VALUE": 400_000.0, "SSHPRNAMT": 2_000, "SSHPRNAMTTYPE": "SH", "PUTCALL": ""},
    ])
    out = _manager_holdings_frame("0001067983", "2026-05-10", "2026-03-31", split)
    assert len(out) == 1
    assert out.iloc[0]["shares"] == 5_000 and out.iloc[0]["value_usd"] == 1_000_000.0
    assert out.iloc[0]["issuer_name"] == "APPLE INC"
    assert out.iloc[0]["position_type"] == "common"


def test_a_cusip_held_as_both_stock_and_calls_keeps_both_legs():
    """One row, both legs in their own columns, and the label says which leg dominates -- the
    columns stay authoritative."""
    both = _infotable([
        {"CUSIP": "037833100", "NAMEOFISSUER": "APPLE INC", "TITLEOFCLASS": "COM",
         "VALUE": 100_000.0, "SSHPRNAMT": 500, "SSHPRNAMTTYPE": "SH", "PUTCALL": ""},
        {"CUSIP": "037833100", "NAMEOFISSUER": "APPLE INC", "TITLEOFCLASS": "COM",
         "VALUE": 900_000.0, "SSHPRNAMT": 4_000, "SSHPRNAMTTYPE": "SH", "PUTCALL": "CALL"},
    ])
    out = _manager_holdings_frame("0001067983", "2026-05-10", "2026-03-31", both)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["value_usd"] == 100_000.0 and row["call_value"] == 900_000.0
    assert row["position_type"] == "call"       # the dominant leg


def test_dominant_type_breaks_ties_towards_common():
    """An all-zero row is a disclosed position with no value, not an 'other'."""
    grouped = pd.DataFrame({"value_usd": [0.0, 5.0], "call_value": [0.0, 0.0],
                            "put_value": [0.0, 0.0], "debt_value": [0.0, 0.0],
                            "other_value": [0.0, 0.0]})
    assert list(_dominant_type(grouped)) == ["common", "common"]


def test_period_and_cik_are_stored_in_their_join_forms():
    out = _manager_holdings_frame("1067983", "2026-05-10", "2026-03-31", _MIXED)
    assert set(out["cik"]) == {"0001067983"}                  # zero-padded, as the PK expects
    assert set(out["period"]) == {pd.Timestamp("2026-03-31")}
    assert set(out["filing_date"]) == {pd.Timestamp("2026-05-10")}


if __name__ == "__main__":
    test_position_type_agrees_with_the_classified_buckets()
    test_no_universe_filter_is_applied()
