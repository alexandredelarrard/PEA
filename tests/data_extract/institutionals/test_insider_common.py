"""Known-truth tests for the shared insider contract: the one numeric parser, the date and flag
coercers, the one value rule, the owner rule, and one Form 4 read by both adapters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_extract.utils.institutionals.fetch_insider_transactions import extract_bulk_strings
from src.data_extract.utils.institutionals.insider_common import (
    BULK_DATE_FORMATS,
    INSIDER_FIELDS,
    LIVE_DATE_FORMATS,
    OWNER_STRING_COLUMNS,
    build_insider_frame,
    normalize_flag,
    owner_summary,
    parse_number,
    parse_sec_date,
)
from src.data_extract.utils.institutionals.insider_edgar_parser import extract_xml_strings

ROLES = ["is_director", "is_officer", "is_ten_pct_owner", "is_other"]
ACCESSION = "0000320193-26-000010"

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
    "ACCESSION_NUMBER": ACCESSION,
    "ISSUERCIK": "0000320193",
    "ISSUERNAME": "Apple Inc.",
    "ISSUERTRADINGSYMBOL": "aapl",
    "DOCUMENT_TYPE": "4",
    "FILING_DATE": "03-FEB-2026",
    "PERIOD_OF_REPORT": "30-JAN-2026",
    "AFF10B5ONE": "1",
}
BULK_OWNER = {
    "ACCESSION_NUMBER": ACCESSION,
    "RPTOWNERCIK": "0001234567",
    "RPTOWNERNAME": "DOE JANE",
    "RPTOWNER_RELATIONSHIP": "Director,Officer",
    "RPTOWNER_TITLE": "CEO",
}
BULK_NONDERIV = {
    "ACCESSION_NUMBER": ACCESSION,
    "NONDERIV_TRANS_SK": "1",
    "SECURITY_TITLE": "Common Stock",
    "TRANS_DATE": "30-JAN-2026",
    "TRANS_FORM_TYPE": "4",
    "TRANS_CODE": "S",
    "EQUITY_SWAP_INVOLVED": "0",
    "TRANS_SHARES": "1000",
    "TRANS_PRICEPERSHARE": "250.5",
    "TRANS_ACQUIRED_DISP_CD": "D",
    "SHRS_OWND_FOLWNG_TRANS": "9000",
    "DIRECT_INDIRECT_OWNERSHIP": "D",
}


def _owners(rows: list[tuple[str, str | None, str | None, str | None]]) -> pd.DataFrame:
    """Owner string rows (accession, cik, title, relationship) with a name derived from the CIK."""
    return pd.DataFrame(
        [{"accession_number": a, "owner_cik": c, "owner_name": f"OWNER {c}", "officer_title": t, "relationship": r} for a, c, t, r in rows],
        columns=OWNER_STRING_COLUMNS,
    )


def test_parse_number_is_one_rule_for_both_sources():
    raw = pd.Series(["1,000", "$5", "12.5", "", None, "4.0973523936194694e-06", "junk", " 7 "], dtype="object")
    out = parse_number(raw)
    assert list(out.iloc[:3]) == [1000.0, 5.0, 12.5]
    assert out.iloc[[3, 4, 6]].isna().all() and out.dtype == "float64"
    assert out.iloc[5] == float("4.0973523936194694e-06"), "Python float, bit for bit"
    assert out.iloc[7] == 7.0
    print(
        f"SANITY: '1,000'/'$5' -> 1000/5, blank/None/junk -> NaN, the long decimal parses to {out.iloc[5]!r} bit for bit; "
        "one numeric parser now serves the zip and the XML path."
    )


def test_parse_sec_date_tries_formats_in_order():
    raw = pd.Series(["03-FEB-2026", "2026-02-03", "junk", None], dtype="object")
    mixed = parse_sec_date(raw, formats=LIVE_DATE_FORMATS)
    explicit = parse_sec_date(raw, formats=BULK_DATE_FORMATS)
    expected = [pd.Timestamp("2026-02-03"), pd.Timestamp("2026-02-03")]
    assert list(mixed.iloc[:2]) == expected and list(explicit.iloc[:2]) == expected
    assert mixed.iloc[2:].isna().all() and explicit.iloc[2:].isna().all()
    print(
        "SANITY: both SEC date shapes parse to 2026-02-03 under the XML ('mixed') and the zip (%d-%b-%Y -> ISO8601) formats; junk and None stay NaT."
    )


def test_normalize_flag_maps_yes_no_text_and_keeps_unknown_nan():
    out = normalize_flag(pd.Series(["1", " TRUE ", "y", "0", "false", "N", "", None, "maybe"], dtype="object"))
    assert list(out.iloc[:6]) == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert out.iloc[6:].isna().all() and out.dtype == "float64"
    print("SANITY: yes-text -> 1.0, no-text -> 0.0, blank/None/unrecognised -> NaN (never False) for the filing-level flags.")


def test_build_applies_one_value_rule_and_never_null_role_flags():
    df_str = pd.DataFrame(
        {
            "accession_number": ["a", "b", "c", "d"],
            "shares": ["100", "100", None, "10"],
            "price_per_share": ["2", "2", None, None],
            "total_value": [None, "999", "50", None],
            "ticker": [" exm ", "", None, "EXM"],
        }
    )
    owners = _owners([("a", "1", None, "Director,TenPercentOwner"), ("b", "2", None, ""), ("c", "3", None, None)])
    out = build_insider_frame(df_str, owners, date_formats=BULK_DATE_FORMATS)
    assert list(out["value_usd"].iloc[:3]) == [200.0, 200.0, 50.0] and pd.isna(out["value_usd"].iloc[3])
    assert out.loc[0, ROLES].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert (out.loc[1:, ROLES] == 0.0).all().all(), "blank, absent or missing relationship -> 0, never NaN"
    assert out["ticker"].iloc[0] == "EXM" and out["ticker"].iloc[1:3].isna().all()
    assert out["n_reporting_owners"].tolist() == [1, 1, 1, 0] and out["owner_cik"].iloc[:3].tolist() == ["0000000001", "0000000002", "0000000003"]
    assert pd.isna(out["owner_cik"].iloc[3]) and pd.isna(out["owner_ciks"].iloc[3])
    assert "relationship" not in out.columns and "total_value" not in out.columns
    print(
        "SANITY: shares x price wins over a stated 999 (200), the stated total fills a missing product (50); "
        "'' / absent relationship -> roles 0; '' ticker -> NULL; an accession with no owner row -> roles 0, n 0, owner NULL."
    )


def test_owner_summary_picks_the_primary_owner_by_role_rank_then_lowest_cik():
    owners = _owners(
        [
            # joint filing: a 10% owner with the lowest CIK, an officer, a director
            ("joint", "0000000003", None, "TenPercentOwner"),
            ("joint", "0000000009", "CFO", "Officer"),
            ("joint", "0000000005", None, "Director"),
            ("joint", "9", "CFO", "Officer"),  # same owner listed twice, unpadded
            # tie on role: the lowest numeric CIK wins, a missing CIK sorts last
            ("tie", None, None, "Director"),
            ("tie", "0000000200", None, "Director"),
            ("tie", "0000000100", None, "Director"),
            # no role anywhere, no CIK anywhere
            ("bare", None, None, None),
        ]
    )
    out = owner_summary(owners).set_index("accession_number")
    joint, tie, bare = out.loc["joint"], out.loc["tie"], out.loc["bare"]
    assert (joint["owner_cik"], joint["officer_title"]) == ("0000000009", "CFO")
    assert joint["owner_ciks"] == "0000000003,0000000005,0000000009" and joint["n_reporting_owners"] == 3
    assert joint[ROLES].tolist() == [1.0, 1.0, 1.0, 0.0], "flags are OR'ed across owners"
    assert tie["owner_cik"] == "0000000100" and tie["owner_ciks"] == "0000000100,0000000200" and tie["n_reporting_owners"] == 2
    assert pd.isna(bare["owner_cik"]) and pd.isna(bare["owner_ciks"]) and bare["n_reporting_owners"] == 0
    assert bare[ROLES].tolist() == [0.0, 0.0, 0.0, 0.0]
    print(
        "SANITY: the officer outranks a lower-CIK 10% owner and a director (primary 0000000009, title CFO); equal roles go to the lowest CIK "
        "(0000000100, missing CIK last); owner_ciks is sorted, deduplicated and padded; flags are OR'ed; an owner without role or CIK gives n 0."
    )


def test_a_joint_filing_stays_one_row_per_trade():
    df_str = pd.DataFrame({"accession_number": ["joint", "joint"], "security_type": ["nonderiv", "nonderiv"], "row_sequence": [1, 2]})
    owners = _owners([("joint", "1", None, "Director"), ("joint", "2", None, "Officer"), ("joint", "3", None, "Other")])
    out = build_insider_frame(df_str, owners, date_formats=BULK_DATE_FORMATS)
    assert len(out) == 2 and out["row_sequence"].tolist() == [1, 2]
    assert set(out["n_reporting_owners"]) == {3} and set(out["owner_cik"]) == {"0000000002"}
    print("SANITY: 2 trades x 3 owners -> 2 rows (never one per owner), each carrying n_reporting_owners 3 and the officer as primary.")


def test_one_form4_gives_one_canonical_frame_on_both_paths():
    df_bulk, own_bulk = extract_bulk_strings(
        pd.DataFrame([BULK_SUBMISSION]), pd.DataFrame([BULK_OWNER]), pd.DataFrame([BULK_NONDERIV]), pd.DataFrame()
    )
    df_xml, own_xml, _ = extract_xml_strings(FORM4_XML, ACCESSION)
    shared = [
        "accession_number",
        "security_type",
        "row_sequence",
        *(f.name for f in INSIDER_FIELDS if f.bulk and f.xml and f.scope != "owner" and f.name != "total_value"),
    ]
    pd.testing.assert_frame_equal(own_bulk.astype(object), own_xml.astype(object))

    typed_bulk = build_insider_frame(df_bulk, own_bulk, date_formats=BULK_DATE_FORMATS)
    typed_xml = build_insider_frame(df_xml, own_xml, date_formats=LIVE_DATE_FORMATS)
    left = typed_bulk[shared + ["value_usd", *ROLES, "owner_cik", "owner_ciks", "n_reporting_owners"]].astype(object)
    right = typed_xml[left.columns].astype(object)
    pd.testing.assert_frame_equal(left.where(left.notna(), None), right.where(right.notna(), None))
    assert typed_bulk.iloc[0]["value_usd"] == 250_500.0 and typed_bulk.iloc[0]["ticker"] == "AAPL"
    assert np.array_equal(typed_bulk.iloc[0][ROLES].to_numpy(float), [1.0, 1.0, 0.0, 0.0])
    print(
        f"SANITY: the same Form 4 as bulk TSV rows and as ownership XML gives identical owner strings and an identical typed frame on "
        f"{len(left.columns)} fields (value_usd 250,500, ticker AAPL, roles Director+Officer, row_sequence 1)."
    )
