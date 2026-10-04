"""The insider row key (`accession_number`, `security_type`, `row_sequence`) is the same from both
sources: the zip ranks SEC's numeric transaction id inside each filing table, the XML counts nodes."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_extract.utils.institutionals.fetch_insider_transactions import extract_bulk_strings
from tests.fixtures.insider_zip import fixture_accessions, xml_frame, zip_frame, zip_path

KEY = ["accession_number", "security_type", "row_sequence"]


def test_zip_row_sequence_ranks_the_numeric_sk_per_filing_table():
    sub = pd.DataFrame({"ACCESSION_NUMBER": ["a", "b"], "ISSUERCIK": ["1", "2"], "FILING_DATE": ["03-FEB-2026", "03-FEB-2026"]})
    nonderiv = pd.DataFrame(
        {"ACCESSION_NUMBER": ["a", "a", "a", "a", "b"], "NONDERIV_TRANS_SK": ["10", "9", None, "30", "5"], "TRANS_CODE": list("PQRST")}
    )
    deriv = pd.DataFrame({"ACCESSION_NUMBER": ["a"], "DERIV_TRANS_SK": ["7"], "TRANS_CODE": ["M"]})
    df_str, _ = extract_bulk_strings(sub, pd.DataFrame(), nonderiv, deriv)
    got = {(row.accession_number, row.security_type, row.row_sequence): row.transaction_code for row in df_str.itertuples()}
    assert got == {("a", "nonderiv", 1): "Q", ("a", "nonderiv", 2): "P", ("a", "nonderiv", 3): "S", ("b", "nonderiv", 1): "T", ("a", "deriv", 1): "M"}
    assert "R" not in set(df_str["transaction_code"]) and "transaction_sk" not in df_str.columns
    print(
        "SANITY: SKs 10, 9, 30 rank numerically to 2, 1, 3 (not lexically), each filing table restarts at 1, "
        "and a line without an SK cannot be keyed and is dropped."
    )


@pytest.mark.skipif(zip_path() is None or not fixture_accessions(), reason="cached 2026q2 insider zip or XML fixtures absent")
def test_zip_and_xml_give_equal_keys_on_every_fixture_filing():
    path = zip_path()
    assert path is not None
    df_zip = zip_frame(path)
    df_xml = pd.concat([xml_frame(accession) for accession in fixture_accessions()], ignore_index=True)
    zip_keys = set(map(tuple, df_zip[KEY].to_numpy()))
    xml_keys = set(map(tuple, df_xml[KEY].to_numpy()))
    df_matched = df_zip.merge(df_xml, on=KEY, suffixes=("_zip", "_xml"))
    zip_date, xml_date = df_matched["transaction_date_zip"], df_matched["transaction_date_xml"]
    zip_shares, xml_shares = df_matched["shares_zip"], df_matched["shares_xml"]
    same_line = (
        df_matched["transaction_code_zip"].eq(df_matched["transaction_code_xml"])
        & (zip_date.eq(xml_date) | (zip_date.isna() & xml_date.isna()))
        & ((zip_shares - xml_shares).abs().le(0.005 + 1e-6) | (zip_shares.isna() & xml_shares.isna()))
    )
    filings = df_xml["accession_number"].nunique()
    print(
        f"\nSANITY: {filings} fixture filings, {len(xml_keys)} XML keys, {len(zip_keys)} zip keys, {len(zip_keys & xml_keys)} shared, "
        f"{len(zip_keys ^ xml_keys)} on one side only; {int(same_line.sum())}/{len(df_matched)} matched keys carry the same code, date and shares, "
        "so the zip SK rank is the XML order."
    )
    assert filings == len(fixture_accessions())
    assert zip_keys == xml_keys
    assert same_line.all()
