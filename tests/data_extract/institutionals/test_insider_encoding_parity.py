"""The same filing parsed from its zip rows and from its ownership XML gives the same stored values
(REQ-004) and the same owners (REQ-005): a known-truth joint filing built both ways, then every real
XML fixture against its rows in the cached 2026q2 zip."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_extract.utils.institutionals.fetch_insider_transactions import extract_bulk_strings
from src.data_extract.utils.institutionals.insider_common import BULK_DATE_FORMATS, LIVE_DATE_FORMATS, OWNER_SUMMARY_COLUMNS, build_insider_frame
from src.data_extract.utils.institutionals.insider_edgar_parser import extract_xml_strings
from tests.fixtures.insider_zip import fixture_accessions, xml_frame, zip_frame, zip_path

KEY = ["accession_number", "security_type", "row_sequence"]
ROLES = ["is_director", "is_officer", "is_ten_pct_owner", "is_other"]
FERTITTA = "0001104659-26-068304"
#: The XML fixtures in tests/fixtures/insider/ (fetched from EDGAR, 2026q2 accessions) and why each was chosen.
FIXTURE_ACCESSIONS = {
    FERTITTA: "joint (4 owners, Fertitta, WYNN), derivative rows",
    "0000904548-26-000008": "joint, owners with different roles (ECHO), derivative rows; owner pick differed",
    "0001193125-26-274227": "joint, 8 owners (DELL); owner pick differed",
    "0001309416-26-000005": "joint, Other|Other|TenPercentOwner (LVS); owner pick differed",
    "0001104659-26-077658": "joint, Director|Other|Other (DLTR), 12 derivative rows; owner pick differed",
    "0001193125-26-290621": "joint, 2 owners (DELL), 14 rows; owner pick differed",
    "0001193125-26-161653": "joint, 9 owners (DELL), nonderiv + deriv; owner pick differed",
    "0001193125-26-260117": "joint, 5 owners (DELL), nonderiv + deriv; owner pick differed",
    "0001104659-26-060502": "joint, 2 TenPercentOwners (RSG); owner pick differed",
    "0001193125-26-140097": "joint, 6 owners (INCY); owner pick differed",
    "0001104659-26-039650": "joint, 4 owners (WYNN), 2 derivative rows; owner pick differed",
    "0001193125-26-267826": "joint, 5 owners (DELL); owner pick differed",
    "0002043501-26-000005": "4/A with an original submission date (LIN), nonderiv + deriv",
    "0001193125-26-274337": "4/A with an original submission date (PYPL), nonderiv + deriv",
    "0000320335-26-000149": "4/A with an original submission date (GL), nonderiv + deriv",
    "0000070858-26-000255": "4/A with an original submission date (BAC), 2 owners",
    "0001561550-26-000196": "4/A with an original submission date (DDOG), nonderiv + deriv",
    "0001921955-26-000012": "4/A with an original submission date (TGT), derivative only",
    "0001104659-26-039710": "stated total, no shares or price (derivative)",
    "0001302110-26-000026": "Form 5 (FTNT)",
    "0000310158-26-000139": "plain Form 4 (MRK), 2 deriv",
    "0000897069-26-000776": "plain Form 4 (ULTA), 1 nonderiv + 2 deriv",
    "0001193125-26-274081": "plain Form 4 (EXPD), 3 deriv",
    "0000050863-26-000092": "plain Form 4 (INTC), 1 nonderiv + 1 deriv",
    "0001225208-26-004336": "plain Form 4 (HBAN), 2 nonderiv",
    "0001193125-26-242634": "plain Form 4 (GOOGL), 3 nonderiv",
    "0001140536-26-000136": "plain Form 4 (WTW), 2 nonderiv",
    "0000858877-26-000092": "plain Form 4 (CSCO), 3 nonderiv",
}
#: Numbers the zip rounds to 2 decimals; compared within half a cent.
ROUNDED = ["shares", "price_per_share", "shares_owned_after", "underlying_shares", "exercise_price", "underlying_value"]
ROUNDING = 0.005 + 1e-6
#: Optional per-line text the zip can attach to a different line of the same filing than the XML does.
ZIP_SHIFTED = ["transaction_timeliness", "nature_of_ownership"]

# --------------------------------------------------------------------------- #
# (a) known truth: one joint Form 4/A, built as zip members and as XML            #
# --------------------------------------------------------------------------- #
ACCESSION = "0000000001-26-000123"
KNOWN_XML = """<ownershipDocument>
  <documentType>4/A</documentType><periodOfReport>2026-05-01</periodOfReport><dateOfOriginalSubmission>2026-05-04</dateOfOriginalSubmission>
  <aff10b5One>0</aff10b5One>
  <issuer><issuerCik>0000000001</issuerCik><issuerName>Known Truth Inc</issuerName><issuerTradingSymbol>  </issuerTradingSymbol></issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerCik>0000000300</rptOwnerCik><rptOwnerName>DIRECTOR DAN</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isDirector>1</isDirector></reportingOwnerRelationship>
  </reportingOwner>
  <reportingOwner>
    <reportingOwnerId><rptOwnerCik>0000000200</rptOwnerCik><rptOwnerName>NO ROLE FUND</rptOwnerName></reportingOwnerId>
  </reportingOwner>
  <reportingOwner>
    <reportingOwnerId><rptOwnerCik>0000000100</rptOwnerCik><rptOwnerName>BIG HOLDER LP</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isTenPercentOwner>true</isTenPercentOwner></reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle><transactionDate><value>2026-05-01</value></transactionDate>
      <transactionCoding><transactionFormType>4</transactionFormType><transactionCode>P</transactionCode><equitySwapInvolved>0</equitySwapInvolved></transactionCoding>
      <transactionAmounts><transactionShares><value>1,500</value></transactionShares><transactionPricePerShare><value>$12.25</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode></transactionAmounts>
      <postTransactionAmounts><sharesOwnedFollowingTransaction><value>101500</value></sharesOwnedFollowingTransaction></postTransactionAmounts>
      <ownershipNature><directOrIndirectOwnership><value>I</value></directOrIndirectOwnership><natureOfOwnership><value>By LP</value></natureOfOwnership></ownershipNature>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle><transactionDate><value>2026-05-01</value></transactionDate>
      <transactionCoding><transactionFormType>4</transactionFormType><transactionCode>P</transactionCode><equitySwapInvolved>0</equitySwapInvolved></transactionCoding>
      <transactionAmounts><transactionShares><value>500</value></transactionShares><transactionPricePerShare><value>12.30</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode></transactionAmounts>
      <postTransactionAmounts><sharesOwnedFollowingTransaction><value>102000</value></sharesOwnedFollowingTransaction></postTransactionAmounts>
      <ownershipNature><directOrIndirectOwnership><value>I</value></directOrIndirectOwnership><natureOfOwnership><value>By LP</value></natureOfOwnership></ownershipNature>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
  <derivativeTable><derivativeTransaction>
    <securityTitle><value>Forward Contract</value></securityTitle><transactionDate><value>2026-05-01</value></transactionDate>
    <transactionCoding><transactionFormType>4</transactionFormType><transactionCode>J</transactionCode><equitySwapInvolved>0</equitySwapInvolved></transactionCoding>
    <transactionAmounts><transactionTotalValue><value>965000000</value></transactionTotalValue>
      <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts>
    <expirationDate><value>2027-05-01</value></expirationDate>
    <underlyingSecurity><underlyingSecurityTitle><value>Common Stock</value></underlyingSecurityTitle></underlyingSecurity>
    <postTransactionAmounts><sharesOwnedFollowingTransaction><value>0</value></sharesOwnedFollowingTransaction></postTransactionAmounts>
    <ownershipNature><directOrIndirectOwnership><value>I</value></directOrIndirectOwnership><natureOfOwnership><value>By LP</value></natureOfOwnership></ownershipNature>
  </derivativeTransaction></derivativeTable>
</ownershipDocument>"""

KNOWN_SUBMISSION = {
    "ACCESSION_NUMBER": ACCESSION,
    "FILING_DATE": "06-MAY-2026",
    "PERIOD_OF_REPORT": "01-MAY-2026",
    "DATE_OF_ORIG_SUB": "04-MAY-2026",
    "DOCUMENT_TYPE": "4/A",
    "ISSUERCIK": "0000000001",
    "ISSUERNAME": "Known Truth Inc",
    "ISSUERTRADINGSYMBOL": " ",
    "AFF10B5ONE": "0",
}
#: The zip lists owners in its own order and leaves an absent relationship blank.
KNOWN_OWNERS = [
    {"ACCESSION_NUMBER": ACCESSION, "RPTOWNERCIK": "0000000100", "RPTOWNERNAME": "BIG HOLDER LP", "RPTOWNER_RELATIONSHIP": "TenPercentOwner"},
    {"ACCESSION_NUMBER": ACCESSION, "RPTOWNERCIK": "0000000200", "RPTOWNERNAME": "NO ROLE FUND", "RPTOWNER_RELATIONSHIP": None},
    {"ACCESSION_NUMBER": ACCESSION, "RPTOWNERCIK": "0000000300", "RPTOWNERNAME": "DIRECTOR DAN", "RPTOWNER_RELATIONSHIP": "Director"},
]
_LINE = {
    "ACCESSION_NUMBER": ACCESSION,
    "SECURITY_TITLE": "Common Stock",
    "TRANS_DATE": "01-MAY-2026",
    "TRANS_FORM_TYPE": "4",
    "TRANS_CODE": "P",
    "EQUITY_SWAP_INVOLVED": "0",
    "TRANS_ACQUIRED_DISP_CD": "A",
    "DIRECT_INDIRECT_OWNERSHIP": "I",
    "NATURE_OF_OWNERSHIP": "By LP",
}
KNOWN_NONDERIV = [
    {**_LINE, "NONDERIV_TRANS_SK": "9002", "TRANS_SHARES": "500", "TRANS_PRICEPERSHARE": "12.3", "SHRS_OWND_FOLWNG_TRANS": "102000"},
    {**_LINE, "NONDERIV_TRANS_SK": "9001", "TRANS_SHARES": "1500", "TRANS_PRICEPERSHARE": "12.25", "SHRS_OWND_FOLWNG_TRANS": "101500"},
]
KNOWN_DERIV = [
    {
        **_LINE,
        "DERIV_TRANS_SK": "4001",
        "SECURITY_TITLE": "Forward Contract",
        "TRANS_CODE": "J",
        "TRANS_ACQUIRED_DISP_CD": "D",
        "TRANS_TOTAL_VALUE": "965000000",
        "EXPIRATION_DATE": "01-MAY-2027",
        "UNDLYNG_SEC_TITLE": "Common Stock",
        "SHRS_OWND_FOLWNG_TRANS": "0",
    }
]


def _cells(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """`columns` as plain Python objects with None for every missing value, for exact comparison."""
    out = df[columns].astype(object)
    return out.where(df[columns].notna(), None).reset_index(drop=True)


def test_a_known_joint_filing_stores_identical_values_from_zip_rows_and_from_xml():
    df_str, df_owners = extract_bulk_strings(
        pd.DataFrame([KNOWN_SUBMISSION]), pd.DataFrame(KNOWN_OWNERS), pd.DataFrame(KNOWN_NONDERIV), pd.DataFrame(KNOWN_DERIV)
    )
    df_zip = build_insider_frame(df_str, df_owners, date_formats=BULK_DATE_FORMATS).sort_values(KEY, ignore_index=True)
    xml_str, xml_owners, _ = extract_xml_strings(KNOWN_XML, ACCESSION)
    df_xml = build_insider_frame(xml_str, xml_owners, date_formats=LIVE_DATE_FORMATS).sort_values(KEY, ignore_index=True)

    compared = sorted((set(df_zip.columns) & set(df_xml.columns)) - {"filing_date"})
    pd.testing.assert_frame_equal(_cells(df_zip, compared), _cells(df_xml, compared))
    row = df_xml.iloc[-1]
    assert len(df_xml) == 3, "one row per trade, never per owner"
    assert (row["owner_cik"], row["owner_name"], row["owner_ciks"], row["n_reporting_owners"]) == (
        "0000000300",
        "DIRECTOR DAN",
        "0000000100,0000000200,0000000300",
        3,
    )
    assert row[ROLES].tolist() == [1.0, 0.0, 1.0, 0.0] and pd.isna(row["ticker"])
    assert df_xml["value_usd"].tolist() == pytest.approx([965_000_000.0, 18_375.0, 6_150.0])
    assert set(df_xml["original_submission_date"]) == {pd.Timestamp("2026-05-04")}
    print(
        f"SANITY: a joint 4/A (3 owners, one without a relationship block), a blank ticker, '$'/',' numbers and a stated total, "
        f"parsed from zip rows and from XML, gives identical values on {len(compared)} columns: primary owner the director "
        f"(0000000300), owner_ciks of all 3, roles director+10% (others 0), ticker NULL, value_usd 18,375 / 6,150 / 965m, 3 rows."
    )


# --------------------------------------------------------------------------- #
# (b) real: every XML fixture vs its rows in the cached 2026q2 zip                #
# --------------------------------------------------------------------------- #
def _agreement(df_matched: pd.DataFrame, column: str) -> pd.Series:
    """Per-row agreement of `column` between the `_zip` and `_xml` sides."""
    left, right = df_matched[f"{column}_zip"], df_matched[f"{column}_xml"]
    both_missing = left.isna() & right.isna()
    if column in ROUNDED:
        return both_missing | (left - right).abs().le(ROUNDING)
    if column == "value_usd":
        bound = ROUNDING * (df_matched["shares_xml"].abs() + df_matched["price_per_share_xml"].abs()).fillna(0.0) + 1e-4
        return both_missing | (left - right).abs().le(bound)
    cells = pd.Series(
        [a == b for a, b in zip(_cells(df_matched, [f"{column}_zip"]).iloc[:, 0], _cells(df_matched, [f"{column}_xml"]).iloc[:, 0], strict=True)]
    )
    return cells.set_axis(df_matched.index)


def _same_values_per_filing_table(df_matched: pd.DataFrame, column: str) -> bool:
    """True when each (accession, security_type) holds the same multiset of `column` values on both sides."""
    groups = df_matched.groupby(["accession_number", "security_type"])
    return all(
        sorted(map(str, group[f"{column}_zip"].fillna("<NA>"))) == sorted(map(str, group[f"{column}_xml"].fillna("<NA>"))) for _, group in groups
    )


@pytest.mark.skipif(zip_path() is None or not fixture_accessions(), reason="cached 2026q2 insider zip or XML fixtures absent")
def test_every_real_fixture_matches_its_zip_rows_field_by_field():
    path = zip_path()
    assert path is not None
    assert set(fixture_accessions()) == set(FIXTURE_ACCESSIONS), "the fixture directory and FIXTURE_ACCESSIONS disagree"
    df_zip = zip_frame(path)
    df_xml = pd.concat([xml_frame(accession) for accession in fixture_accessions()], ignore_index=True)
    df_matched = df_zip.merge(df_xml, on=KEY, how="inner", suffixes=("_zip", "_xml"))
    columns = sorted((set(df_zip.columns) & set(df_xml.columns)) - set(KEY) - {"filing_date"})

    table = {column: _agreement(df_matched, column) for column in columns}
    print(f"\n{len(FIXTURE_ACCESSIONS)} filings, {len(df_xml)} XML rows, {len(df_zip)} zip rows, {len(df_matched)} matched on the key")
    for column, agree in table.items():
        print(f"  {column:<26} {int(agree.sum()):>4}/{len(agree)} ({agree.mean():.1%})")

    first = df_matched.drop_duplicates("accession_number")
    joint = first[first["n_reporting_owners_xml"] > 1]
    fertitta = df_matched[df_matched["accession_number"] == FERTITTA]
    shifted = {column: int((~table[column]).sum()) for column in ZIP_SHIFTED}
    print(
        f"SANITY: keys agree on {len(df_matched)}/{len(df_xml)} rows; every field agrees at 100% (numbers within half a cent of the "
        f"zip's 2-decimal rounding) except {shifted} rows where the zip puts an optional line value on another line of the same filing "
        f"(same values per filing table). Owners agree on {len(first)} filings, {len(joint)} of them joint; Fertitta has "
        f"{int(fertitta['n_reporting_owners_xml'].iloc[0])} owners on {len(fertitta)} rows (one per trade), primary {fertitta['owner_cik_xml'].iloc[0]}."
    )
    assert len(df_matched) == len(df_xml) == len(df_zip)
    for column in columns:
        if column in ZIP_SHIFTED:
            assert _same_values_per_filing_table(df_matched, column), column
        else:
            assert table[column].all(), (column, df_matched.loc[~table[column], [f"{column}_zip", f"{column}_xml"]].head())
    assert set(OWNER_SUMMARY_COLUMNS) <= set(columns)
    assert len(joint) >= 12
    assert np.array_equal(fertitta["row_sequence"].sort_values().to_numpy(), np.arange(1, len(fertitta) + 1))
    assert int(fertitta["n_reporting_owners_xml"].iloc[0]) == 4 and fertitta["owner_cik_xml"].iloc[0] == "0001080301"
