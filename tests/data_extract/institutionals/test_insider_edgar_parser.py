"""Known-truth ownership XML tests for the live Forms 3/4/5 adapter."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_extract.utils.institutionals.insider_edgar_parser import parse_ownership_xml

FORM4_XML = """<?xml version="1.0"?>
<ownershipDocument>
  <documentType>4</documentType><periodOfReport>2026-06-30</periodOfReport><aff10b5One>true</aff10b5One>
  <issuer><issuerCik>0000093556</issuerCik><issuerName>Example Inc</issuerName><issuerTradingSymbol>exm</issuerTradingSymbol></issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerCik>0001234567</rptOwnerCik><rptOwnerName>DOE JANE</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isDirector>1</isDirector><isOfficer>1</isOfficer><isTenPercentOwner>0</isTenPercentOwner><isOther>0</isOther><officerTitle>Chief Executive Officer</officerTitle></reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable><nonDerivativeTransaction>
    <securityTitle><value>Common Stock</value></securityTitle><transactionDate><value>2026-06-29</value></transactionDate><deemedExecutionDate><value>2026-06-28</value></deemedExecutionDate>
    <transactionTimeliness><value>E</value></transactionTimeliness><transactionCoding><transactionFormType>4</transactionFormType><transactionCode>S</transactionCode><equitySwapInvolved>0</equitySwapInvolved><footnoteId id="F1"/></transactionCoding>
    <transactionAmounts><transactionShares><value>1000</value></transactionShares><transactionPricePerShare><value>91.6725</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts>
    <postTransactionAmounts><sharesOwnedFollowingTransaction><value>9000</value></sharesOwnedFollowingTransaction></postTransactionAmounts>
    <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership><natureOfOwnership><value>Direct</value></natureOfOwnership></ownershipNature>
  </nonDerivativeTransaction></nonDerivativeTable>
  <derivativeTable><derivativeTransaction>
    <securityTitle><value>Stock Option</value></securityTitle><conversionOrExercisePrice><value>20</value></conversionOrExercisePrice><transactionDate><value>2026-06-29</value></transactionDate>
    <transactionCoding><transactionFormType>4</transactionFormType><transactionCode>M</transactionCode><equitySwapInvolved>0</equitySwapInvolved></transactionCoding>
    <transactionAmounts><transactionShares><value>1000</value></transactionShares><transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts>
    <exerciseDate><value>2020-01-01</value></exerciseDate><expirationDate><value>2030-01-01</value></expirationDate>
    <underlyingSecurity><underlyingSecurityTitle><value>Common Stock</value></underlyingSecurityTitle><underlyingSecurityShares><value>1000</value></underlyingSecurityShares></underlyingSecurity>
    <postTransactionAmounts><sharesOwnedFollowingTransaction><value>0</value></sharesOwnedFollowingTransaction></postTransactionAmounts><ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
  </derivativeTransaction></derivativeTable>
  <footnotes><footnote id="F1">Sold under a Rule 10b5-1 trading plan.</footnote></footnotes>
</ownershipDocument>"""


FORM3_XML = """<ownershipDocument>
  <documentType>3</documentType><periodOfReport>2026-07-01</periodOfReport>
  <issuer><issuerCik>0000093556</issuerCik><issuerName>Example Inc</issuerName><issuerTradingSymbol>EXM</issuerTradingSymbol></issuer>
  <reportingOwner><reportingOwnerId><rptOwnerCik>0001234567</rptOwnerCik><rptOwnerName>DOE JANE</rptOwnerName></reportingOwnerId></reportingOwner>
  <nonDerivativeTable><nonDerivativeHolding><securityTitle><value>Common Stock</value></securityTitle><postTransactionAmounts><sharesOwnedFollowingTransaction><value>9000</value></sharesOwnedFollowingTransaction></postTransactionAmounts></nonDerivativeHolding></nonDerivativeTable>
</ownershipDocument>"""


def test_form4_preserves_every_cube_input_and_the_richer_xml_fields():
    transactions, footnotes = parse_ownership_xml(FORM4_XML)
    assert list(transactions["security_type"]) == ["nonderiv", "deriv"]
    sale = transactions.iloc[0]
    assert sale["ticker"] == "EXM"
    assert sale["owner_cik"] == "0001234567"
    assert sale["transaction_code"] == "S"
    assert sale["price_per_share"] == pytest.approx(91.6725)
    assert sale["value_usd"] == pytest.approx(91_672.5)
    assert sale["shares_owned_after"] == pytest.approx(9_000)
    assert sale["is_10b5_1"] == 1.0
    assert sale["deemed_execution_date"].strftime("%Y-%m-%d") == "2026-06-28"
    assert sale["transaction_timeliness"] == "E"
    assert sale["footnote_ids"] == "F1"
    option = transactions.iloc[1]
    assert option["exercise_price"] == pytest.approx(20.0)
    assert option["underlying_shares"] == pytest.approx(1_000)
    assert footnotes.to_dict("records") == [{"footnote_id": "F1", "footnote_text": "Sold under a Rule 10b5-1 trading plan."}]
    print(
        "SANITY: one Form 4 XML produced one non-derivative sale and one derivative "
        "exercise, retaining price precision, roles, 10b5-1, deemed date, timeliness, "
        "derivative terms, and the referenced footnote."
    )


def test_form3_holdings_are_not_invented_as_transactions():
    transactions, footnotes = parse_ownership_xml(FORM3_XML)
    assert transactions.empty
    assert footnotes.empty
    print(
        "SANITY: a Form 3 containing a holding but no transaction produces zero transaction "
        "rows; holdings need their own future table and cannot contaminate P/S features."
    )


@pytest.mark.parametrize("document_type", ["3", "3/A", "4", "4/A", "5", "5/A"])
def test_all_declared_ownership_form_variants_share_one_xml_contract(document_type):
    xml = FORM4_XML.replace(
        "<documentType>4</documentType>",
        f"<documentType>{document_type}</documentType>",
    )
    transactions, _ = parse_ownership_xml(xml)
    assert set(transactions["document_type"]) == {document_type}
    assert transactions.groupby("security_type")["source_row_sequence"].apply(list).to_dict() == {
        "deriv": [1],
        "nonderiv": [1],
    }
    print(f"SANITY: {document_type} uses the same ownership-XML transaction contract and " "stable per-security-table row sequence.")


def test_missing_ten_b5_flag_remains_unknown_on_an_amendment():
    xml = FORM4_XML.replace(
        "<documentType>4</documentType>",
        "<documentType>4/A</documentType>",
    ).replace("<aff10b5One>true</aff10b5One>", "")
    transactions, _ = parse_ownership_xml(xml)
    assert transactions["is_10b5_1"].isna().all()
    assert transactions["source_row_sequence"].tolist() == [1, 1]
    assert pd.isna(transactions.iloc[0]["is_10b5_1"])
    print(
        "SANITY: an amended filing with no aff10b5One element preserves UNKNOWN rather than "
        "inventing False; row sequences remain stable within each security table."
    )


def test_omitted_relationship_checkboxes_are_false_not_unknown():
    xml = FORM4_XML.replace("<isDirector>1</isDirector>", "").replace("<isTenPercentOwner>0</isTenPercentOwner>", "")
    transactions, _ = parse_ownership_xml(xml)
    assert transactions["is_director"].eq(0.0).all()
    assert transactions["is_ten_pct_owner"].eq(0.0).all()
    assert transactions["is_officer"].eq(1.0).all()
    print("SANITY: omitted relationship checkboxes mean False, while the explicitly checked " "officer role remains True.")


def test_explicit_transaction_total_is_kept_without_shares_or_price():
    xml = FORM4_XML.replace(
        "<transactionShares><value>1000</value></transactionShares>" "<transactionPricePerShare><value>91.6725</value></transactionPricePerShare>",
        "<transactionTotalValue><value>965000000</value></transactionTotalValue>",
        1,
    )
    transactions, _ = parse_ownership_xml(xml)
    sale = transactions.iloc[0]
    assert pd.isna(sale["shares"])
    assert pd.isna(sale["price_per_share"])
    assert sale["value_usd"] == pytest.approx(965_000_000.0)
    print("SANITY: transactionTotalValue survives when a derivative transaction has no " "shares or per-share price.")
