"""Known-truth tests for the shared insider contract: the coercers under each path's options,
`build_insider_frame`'s two value rules and role encoding, and one Form 4 read by both adapters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_extract.utils.institutionals.fetch_insider_transactions import extract_bulk_strings
from src.data_extract.utils.institutionals.insider_common import (
    INSIDER_FIELDS,
    build_insider_frame,
    coerce_numeric,
    normalize_flag,
    parse_sec_date,
)
from src.data_extract.utils.institutionals.insider_edgar_parser import extract_xml_strings

FORM4_XML = """<ownershipDocument>
  <documentType>4</documentType><periodOfReport>2026-01-30</periodOfReport><aff10b5One>1</aff10b5One>
  <issuer><issuerCik>0000320193</issuerCik><issuerName>Apple Inc.</issuerName><issuerTradingSymbol>aapl</issuerTradingSymbol></issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerCik>0001234567</rptOwnerCik><rptOwnerName>DOE JANE</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isDirector>1</isDirector><isOfficer>true</isOfficer><isTenPercentOwner>0</isTenPercentOwner>
      <officerTitle>CEO</officerTitle></reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable><nonDerivativeTransaction>
    <securityTitle><value>Common Stock</value></securityTitle><transactionDate><value>2026-01-30</value></transactionDate>
    <transactionCoding><transactionFormType>4</transactionFormType><transactionCode>S</transactionCode><equitySwapInvolved>0</equitySwapInvolved></transactionCoding>
    <transactionAmounts><transactionShares><value>1000</value></transactionShares><transactionPricePerShare><value>250.5</value></transactionPricePerShare>
      <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts>
    <postTransactionAmounts><sharesOwnedFollowingTransaction><value>9000</value></sharesOwnedFollowingTransaction></postTransactionAmounts>
    <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
  </nonDerivativeTransaction></nonDerivativeTable>
</ownershipDocument>"""

#: The same Form 4 as the four bulk TSV members carry it.
BULK_SUBMISSION = {
    "ACCESSION_NUMBER": "0000320193-26-000010",
    "ISSUERCIK": "0000320193",
    "ISSUERNAME": "Apple Inc.",
    "ISSUERTRADINGSYMBOL": "aapl",
    "DOCUMENT_TYPE": "4",
    "FILING_DATE": "03-FEB-2026",
    "PERIOD_OF_REPORT": "2026-01-30",
    "AFF10B5ONE": "1",
}
BULK_OWNER = {
    "ACCESSION_NUMBER": "0000320193-26-000010",
    "RPTOWNERCIK": "0001234567",
    "RPTOWNERNAME": "DOE JANE",
    "RPTOWNER_RELATIONSHIP": "Director,Officer",
    "RPTOWNER_TITLE": "CEO",
}
BULK_NONDERIV = {
    "ACCESSION_NUMBER": "0000320193-26-000010",
    "NONDERIV_TRANS_SK": "1",
    "SECURITY_TITLE": "Common Stock",
    "TRANS_DATE": "2026-01-30",
    "TRANS_FORM_TYPE": "4",
    "TRANS_CODE": "S",
    "EQUITY_SWAP_INVOLVED": "0",
    "TRANS_SHARES": "1000",
    "TRANS_PRICEPERSHARE": "250.5",
    "TRANS_ACQUIRED_DISP_CD": "D",
    "SHRS_OWND_FOLWNG_TRANS": "9000",
    "DIRECT_INDIRECT_OWNERSHIP": "D",
}


def test_coerce_numeric_keeps_each_paths_parse_rule():
    raw = pd.Series(["1,000", "$5", "12.5", "", None, "4.0973523936194694e-06"], dtype="object")
    bulk = coerce_numeric(raw, numeric_rule="to_numeric")
    live = coerce_numeric(raw, numeric_rule="strip_currency_float")
    assert bulk.iloc[:2].isna().all(), "the bulk TSV rule does not strip separators or currency"
    assert list(live.iloc[:3]) == [1000.0, 5.0, 12.5]
    assert live.iloc[3:5].isna().all() and live.dtype == "float64"
    assert live.iloc[5] == float("4.0973523936194694e-06"), "the live rule is Python float, bit for bit"
    print(
        f"SANITY: '1,000'/'$5' -> bulk NaN, live 1000/5; blank/None -> NaN; the long decimal parses to {live.iloc[5]!r} "
        f"live (Python float) vs {bulk.iloc[5]!r} bulk (to_numeric) -- the two rules are kept apart on purpose."
    )


def test_parse_sec_date_tries_formats_in_order():
    raw = pd.Series(["03-FEB-2026", "2026-02-03", "junk", None], dtype="object")
    mixed = parse_sec_date(raw, formats=("mixed",))
    explicit = parse_sec_date(raw, formats=("%d-%b-%Y", "ISO8601"))
    expected = [pd.Timestamp("2026-02-03"), pd.Timestamp("2026-02-03")]
    assert list(mixed.iloc[:2]) == expected and list(explicit.iloc[:2]) == expected
    assert mixed.iloc[2:].isna().all() and explicit.iloc[2:].isna().all()
    print(
        "SANITY: both SEC date shapes parse to 2026-02-03 under 'mixed' and under the explicit %d-%b-%Y -> ISO8601 fallback; junk and None stay NaT."
    )


def test_normalize_flag_maps_yes_no_text_and_keeps_unknown_nan():
    out = normalize_flag(pd.Series(["1", " TRUE ", "y", "0", "false", "N", "", None, "maybe"], dtype="object"))
    assert list(out.iloc[:6]) == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert out.iloc[6:].isna().all() and out.dtype == "float64"
    print("SANITY: yes-text -> 1.0, no-text -> 0.0, blank/None/unrecognised -> NaN (never False).")


def test_build_applies_the_two_value_rules_and_the_role_encoding():
    df_str = pd.DataFrame(
        {
            "shares": ["100", "100", None],
            "price_per_share": ["2", "2", None],
            "total_value": [None, "999", "50"],
            "relationship": ["Director,TenPercentOwner", "", None],
        }
    )
    bulk = build_insider_frame(df_str, value_rule="shares_x_price_first", numeric_rule="to_numeric")
    live = build_insider_frame(df_str, value_rule="stated_total_first", numeric_rule="strip_currency_float")
    assert list(bulk["value_usd"]) == [200.0, 200.0, 50.0]
    assert list(live["value_usd"]) == [200.0, 999.0, 50.0]
    assert list(bulk["is_director"].iloc[:2]) == [1.0, 0.0] and list(bulk["is_ten_pct_owner"].iloc[:2]) == [1.0, 0.0]
    assert bulk.iloc[2][["is_director", "is_officer", "is_ten_pct_owner", "is_other"]].isna().all()
    assert "relationship" not in bulk.columns and "total_value" not in bulk.columns
    print(
        "SANITY: shares x price wins on the bulk rule (200 over a stated 999), the stated total wins on the live rule (999); "
        "'' relationship -> roles 0, unknown relationship -> roles NaN; helper columns dropped."
    )


def test_one_form4_gives_one_canonical_string_frame_on_both_paths():
    df_bulk = extract_bulk_strings(pd.DataFrame([BULK_SUBMISSION]), pd.DataFrame([BULK_OWNER]), pd.DataFrame([BULK_NONDERIV]), pd.DataFrame())
    df_xml, _ = extract_xml_strings(FORM4_XML)
    shared = ["security_type", *(field.name for field in INSIDER_FIELDS if field.bulk and field.xml)]
    left = df_bulk[shared].reset_index(drop=True).astype(object).where(df_bulk[shared].notna().to_numpy(), None)
    right = df_xml[shared].reset_index(drop=True).astype(object).where(df_xml[shared].notna().to_numpy(), None)
    pd.testing.assert_frame_equal(left, right)

    typed_bulk = build_insider_frame(df_bulk, value_rule="shares_x_price_first", numeric_rule="to_numeric")
    typed_xml = build_insider_frame(df_xml, value_rule="stated_total_first", numeric_rule="strip_currency_float")
    assert typed_bulk.iloc[0]["value_usd"] == typed_xml.iloc[0]["value_usd"] == 250_500.0
    assert typed_bulk.iloc[0]["ticker"] == typed_xml.iloc[0]["ticker"] == "AAPL"
    roles = ["is_director", "is_officer", "is_ten_pct_owner", "is_other"]
    assert np.array_equal(typed_bulk.iloc[0][roles].to_numpy(float), typed_xml.iloc[0][roles].to_numpy(float))
    print(
        f"SANITY: the same Form 4 as bulk TSV rows and as ownership XML gives an identical canonical string frame on "
        f"{len(shared)} shared fields (relationship 'Director,Officer' on both), and the same typed value_usd, ticker and roles."
    )
