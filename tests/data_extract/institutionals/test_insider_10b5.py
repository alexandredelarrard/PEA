"""
test_insider_10b5.py (tests/data_extract/institutionals/test_insider_10b5.py)
------------------------------------------------------------------------------
The Form 4 enrichment pass: the 10b5-1 flag, the derivative block, and the footnote table.

The one thing that must not slip is the difference between "the insider did not trade under a
plan" and "the source has no such field". `AFF10B5ONE` appears in SUBMISSION.tsv only from
2023q1 (measured across all 81 cached zips), so 68 quarters have no answer at all, and a stored
False there would be an assertion the SEC never made. That is why the column is a float and why
the pre-2023 case has its own test.
"""
import pandas as pd

from src.data_extract.utils.common.bulk_cache import quarter_periods
from src.data_extract.utils.institutionals.fetch_insider_transactions import (
    SEC_INSIDER_FIRST_YEAR, _FOOTNOTE_COLS, _footnotes, _normalize_10b5_1, _parse_insider,
    _transactions)


def _submission(rows: list[dict], *, with_10b5: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not with_10b5 and "AFF10B5ONE" in df.columns:
        df = df.drop(columns=["AFF10B5ONE"])
    return df


# --------------------------------------------------------------------------- #
# AFF10B5ONE normalization                                                      #
# --------------------------------------------------------------------------- #
def test_aff10b5one_normalization_table():
    """The raw column is mixed-encoding within a single quarter. Measured 2026q1:
    '0' 42,435 - 'false' 11,525 - '1' 3,620 - 'true' 1,162 - NaN 10,517."""
    raw = pd.Series(["1", "true", "TRUE", "Y", "0", "false", "FALSE", "N",
                     "", "  ", None, "maybe"], dtype="object")
    out = _normalize_10b5_1(raw)
    assert list(out[:4]) == [1.0, 1.0, 1.0, 1.0]
    assert list(out[4:8]) == [0.0, 0.0, 0.0, 0.0]
    assert out[8:].isna().all(), "an unrecognised value must be NaN, never False"
    assert out.dtype == "float64", "bool dtype would collapse NaN into False"
    print("\n=== SANITY: AFF10B5ONE normalization ===")
    print(f"  '1'/'true'/'TRUE'/'Y' -> 1.0, '0'/'false'/'FALSE'/'N' -> 0.0, "
          f"''/blank/None/unexpected -> NaN ({int(out.isna().sum())} of {len(out)}). "
          f"dtype stays float64. Validated.")


def test_a_submission_without_the_column_parses_and_yields_nan():
    """Demanding `AFF10B5ONE` would kill the 68 quarters that predate it -- 2006q1 through
    2022q4 have no such column at all."""
    sub = _submission([{
        "ACCESSION_NUMBER": "0001-15-000001", "ISSUERCIK": "320193",
        "ISSUERNAME": "APPLE INC", "ISSUERTRADINGSYMBOL": "AAPL",
        "DOCUMENT_TYPE": "4", "FILING_DATE": "15-MAR-2015",
        "PERIOD_OF_REPORT": "13-MAR-2015", "AFF10B5ONE": "1",
    }], with_10b5=False)
    own = pd.DataFrame([{"ACCESSION_NUMBER": "0001-15-000001", "RPTOWNERCIK": "1",
                         "RPTOWNERNAME": "SOMEONE", "RPTOWNER_RELATIONSHIP": "Officer"}])
    nonderiv = pd.DataFrame([{
        "ACCESSION_NUMBER": "0001-15-000001", "NONDERIV_TRANS_SK": "10",
        "SECURITY_TITLE": "Common Stock", "TRANS_DATE": "13-MAR-2015", "TRANS_CODE": "S",
        "TRANS_SHARES": "100", "TRANS_PRICEPERSHARE": "50", "TRANS_ACQUIRED_DISP_CD": "D",
        "TRANS_FORM_TYPE": "4"}])
    out = _parse_insider(sub, own, nonderiv, pd.DataFrame())
    assert len(out) == 1
    assert pd.isna(out.iloc[0]["is_10b5_1"])
    assert out.iloc[0]["transaction_form_type"] == "4"   # present since 2006q1, unlike the flag


def test_the_flag_reaches_the_transaction_rows_when_present():
    sub = _submission([{
        "ACCESSION_NUMBER": "0001-26-000001", "ISSUERCIK": "320193",
        "ISSUERNAME": "APPLE INC", "ISSUERTRADINGSYMBOL": "AAPL",
        "DOCUMENT_TYPE": "4", "FILING_DATE": "05-FEB-2026",
        "PERIOD_OF_REPORT": "03-FEB-2026", "AFF10B5ONE": "true"}])
    own = pd.DataFrame([{"ACCESSION_NUMBER": "0001-26-000001", "RPTOWNERCIK": "1",
                         "RPTOWNERNAME": "SOMEONE", "RPTOWNER_RELATIONSHIP": "Director"}])
    nonderiv = pd.DataFrame([{
        "ACCESSION_NUMBER": "0001-26-000001", "NONDERIV_TRANS_SK": "10",
        "TRANS_DATE": "03-FEB-2026", "TRANS_CODE": "S", "TRANS_SHARES": "100",
        "TRANS_PRICEPERSHARE": "50", "TRANS_ACQUIRED_DISP_CD": "D"}])
    out = _parse_insider(sub, own, nonderiv, pd.DataFrame())
    assert out.iloc[0]["is_10b5_1"] == 1.0


# --------------------------------------------------------------------------- #
# The derivative block                                                          #
# --------------------------------------------------------------------------- #
def test_derivative_block_reads_secs_own_misspelling():
    """SEC spells the column `EXCERCISE_DATE` in every quarter from 2006q1 to 2026q1. Reading
    the correct spelling returns all-NA and nothing anywhere would say so."""
    deriv = pd.DataFrame([{
        "ACCESSION_NUMBER": "0001-26-000002", "DERIV_TRANS_SK": "20",
        "SECURITY_TITLE": "Employee Stock Option", "TRANS_DATE": "03-FEB-2026",
        "TRANS_CODE": "M", "TRANS_SHARES": "1000", "CONV_EXERCISE_PRICE": "42.5",
        "EXCERCISE_DATE": "01-FEB-2020", "EXPIRATION_DATE": "01-FEB-2030",
        "UNDLYNG_SEC_TITLE": "Common Stock", "UNDLYNG_SEC_SHARES": "1000",
        "UNDLYNG_SEC_VALUE": "50000", "TRANS_ACQUIRED_DISP_CD": "A"}])
    out = _transactions(deriv, "DERIV_TRANS_SK", "deriv")
    row = out.iloc[0]
    assert row["exercise_price"] == 42.5
    assert row["exercise_date"] == pd.Timestamp("2020-02-01")
    assert row["expiration_date"] == pd.Timestamp("2030-02-01")
    assert row["underlying_security_title"] == "Common Stock"
    assert row["underlying_shares"] == 1000.0


def test_derivative_columns_are_null_on_nonderivative_rows():
    """NULL by construction, not by omission -- NONDERIV_TRANS.tsv has no such columns."""
    nonderiv = pd.DataFrame([{
        "ACCESSION_NUMBER": "0001-26-000003", "NONDERIV_TRANS_SK": "30",
        "TRANS_DATE": "03-FEB-2026", "TRANS_CODE": "P", "TRANS_SHARES": "10",
        "TRANS_PRICEPERSHARE": "5", "TRANS_ACQUIRED_DISP_CD": "A"}])
    out = _transactions(nonderiv, "NONDERIV_TRANS_SK", "nonderiv")
    for col in ("exercise_price", "exercise_date", "expiration_date",
                "underlying_security_title", "underlying_shares", "underlying_value"):
        assert out[col].isna().all(), f"{col} should be NULL on a non-derivative row"


# --------------------------------------------------------------------------- #
# Footnotes                                                                     #
# --------------------------------------------------------------------------- #
def test_footnotes_key_on_accession_and_id_and_respect_the_universe():
    """The raw file is ~167k rows a quarter for ALL filers. Only accessions the transaction
    parse kept may be stored, or the table grows to ~13M rows about companies nothing reads."""
    notes = pd.DataFrame([
        {"ACCESSION_NUMBER": "A", "FOOTNOTE_ID": "F1", "FOOTNOTE_TXT": "Sold under a Rule "
                                                                      "10b5-1 trading plan."},
        {"ACCESSION_NUMBER": "A", "FOOTNOTE_ID": "F2", "FOOTNOTE_TXT": "Shares withheld for tax."},
        {"ACCESSION_NUMBER": "Z", "FOOTNOTE_ID": "F1", "FOOTNOTE_TXT": "Some other filer."},
    ])
    out = _footnotes(notes, {"A"})
    assert list(out.columns) == _FOOTNOTE_COLS
    assert len(out) == 2 and set(out["accession_number"]) == {"A"}
    assert not out.duplicated(["accession_number", "footnote_id"]).any()
    print("\n=== SANITY: insider footnotes ===")
    print(f"  3 source rows over 2 accessions -> {len(out)} kept for the 1 in-universe "
          f"accession; (accession, footnote_id) unique. Validated.")


def test_footnotes_tolerate_an_absent_or_empty_table():
    assert _footnotes(pd.DataFrame(), {"A"}).empty
    assert list(_footnotes(None, {"A"}).columns) == _FOOTNOTE_COLS


# --------------------------------------------------------------------------- #
# The --reparse window                                                          #
# --------------------------------------------------------------------------- #
def test_reparse_reaches_further_back_than_the_routine_window():
    """A reparse must cover every quarter the SOURCE has, not the routine `years_history` one.

    The table holds rows from 2006-01-03, ingested when that window still reached them, but a
    15-year window in 2026 starts at 2010q1. Re-parsing only the window would leave 2006q1-2009q4
    carrying the OLD column set for ever -- new fields NULL on the oldest data and populated on
    the rest, which is indistinguishable downstream from a real coverage cliff."""
    today_year = pd.Timestamp.today().year
    routine = quarter_periods(15 + 1, SEC_INSIDER_FIRST_YEAR)
    reparse = quarter_periods(today_year - SEC_INSIDER_FIRST_YEAR + 1, SEC_INSIDER_FIRST_YEAR)
    assert reparse[0] == f"{SEC_INSIDER_FIRST_YEAR}q1"
    assert set(routine) <= set(reparse), "the reparse window must contain the routine one"
    missed = sorted(set(reparse) - set(routine))
    assert missed, "expected the reparse window to reach quarters the routine one cannot"
    print("\n=== SANITY: insider reparse window ===")
    print(f"  routine {routine[0]}->{routine[-1]} ({len(routine)}q); "
          f"reparse {reparse[0]}->{reparse[-1]} ({len(reparse)}q); "
          f"{len(missed)} quarters ({missed[0]}-{missed[-1]}) reachable ONLY by --reparse. "
          f"Validated.")


if __name__ == "__main__":
    test_aff10b5one_normalization_table()
    test_footnotes_key_on_accession_and_id_and_respect_the_universe()
