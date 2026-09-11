"""
test_13g_fetch.py (tests/data_extract/institutionals/test_13g_fetch.py)
------------------------------------------------------------------------
`fetch_13g_edgar`'s parse, which has exactly one job beyond shape: never publish a number the
filing did not structurally disclose.

The structured-data cliff (2024-12-17) is what makes that non-trivial. Before it, edgartools
builds a `Schedule13G` from the SGML header alone and every numeric is the class default 0;
after it, the same 0 would be a real disclosure. Both eras are stubbed here rather than fetched,
so the guard is tested without a network call -- the `get_fn`-injection precedent from
`fetch_superinvestors`.
"""
from types import SimpleNamespace

import pandas as pd

from src.data_store.schema import Tables
from src.data_extract.utils.institutionals.fetch_13g_edgar import (
    _COLS, _NUMERIC_COLS, _filing_rows, _norm_entity, _reporting_person_cik,
    build_ticker_13g_edgar)


def _rp(name="FMR LLC", cik="", *, no_cik=False, pct=0.0, agg=0, sole=0, shared=0,
        torp="", citizenship="", member=None, comment=None):
    return SimpleNamespace(
        name=name, cik=cik, no_cik=no_cik, citizenship=citizenship,
        sole_voting_power=sole, shared_voting_power=shared,
        sole_dispositive_power=sole, shared_dispositive_power=shared,
        aggregate_amount=agg, percent_of_class=pct,
        type_of_reporting_person=torp, member_of_group=member, comment=comment)


def _filing(*, form="SC 13G/A", filing_date="2024-11-12", accession="0000315066-24-002743",
            issuer_cik="0001551182", issuer_name="Eaton Corp plc", cusip="",
            event_date="", rule=None, structured=False, persons=None, filers=None):
    obj = SimpleNamespace(
        has_structured_data=structured,
        issuer_info=SimpleNamespace(cik=issuer_cik, name=issuer_name),
        security_info=SimpleNamespace(cusip=cusip, title=""),
        date_of_event=event_date, rule_designation=rule,
        is_amendment="/A" in form, amendment_number=1 if "/A" in form else None,
        reporting_persons=persons if persons is not None else [_rp()])
    header = SimpleNamespace(filers=[
        SimpleNamespace(company_information=SimpleNamespace(cik=c, name=n))
        for c, n in (filers or [])])
    return SimpleNamespace(
        form=form, filing_date=filing_date, accession_number=accession,
        primary_document="doc.htm", cik=issuer_cik, ticker=None, document=None,
        header=header, obj=lambda: obj)


# --------------------------------------------------------------------------- #
# The numeric guard -- the whole reason this module has a test                  #
# --------------------------------------------------------------------------- #
def test_pre_mandate_zeros_become_nan_while_identity_survives():
    """A pre-mandate filing parses with `has_structured_data=False` and 0 in every numeric.
    Publishing the 0 would claim a 0% stake the filer never disclosed."""
    rows = _filing_rows(_filing(structured=False,
                                persons=[_rp(name="FMR LLC", cik="0000315066")]))
    assert len(rows) == 1
    row = rows[0]
    for col in _NUMERIC_COLS:
        assert pd.isna(row[col]), f"{col} leaked a pre-mandate class default"
    # identity is what the header path CAN recover, and it is what the event features need
    assert row["reporting_person_name"] == "FMR LLC"
    assert row["reporting_person_cik"] == "0000315066"
    assert row["has_structured_data"] == 0.0
    assert row["date_of_event"] is None and row["cusip"] is None
    print("\n=== SANITY: 13G pre-mandate guard ===")
    print(f"  6/6 numerics NaN, name+CIK kept ({row['reporting_person_name']}, "
          f"{row['reporting_person_cik']}). A 0% stake is never published. Validated.")


def test_post_mandate_numbers_pass_through_with_rule_designation():
    rows = _filing_rows(_filing(
        form="SCHEDULE 13G", filing_date="2026-04-28", structured=True,
        cusip="G29183103", event_date="03/31/2026", rule="Rule 13d-1(b)",
        persons=[_rp(name="Vanguard Capital Management", cik="", pct=7.48,
                     agg=29042849, sole=3796733, torp="IA")],
        filers=[("0002100119", "VANGUARD CAPITAL MANAGEMENT LLC")]))
    row = rows[0]
    assert row["percent_of_class"] == 7.48
    assert row["aggregate_amount"] == 29042849
    assert row["sole_voting_power"] == 3796733
    assert row["rule_designation"] == "Rule 13d-1(b)"
    assert row["cusip"] == "G29183103"
    assert row["date_of_event"] == pd.Timestamp("2026-03-31")
    print("\n=== SANITY: 13G post-mandate passthrough ===")
    print(f"  pct={row['percent_of_class']} agg={row['aggregate_amount']:,} "
          f"rule={row['rule_designation']!r} event={row['date_of_event'].date()}. Validated.")


def test_event_date_is_parsed_month_first():
    """The cover page writes MM/DD/YYYY. An inferred parse can read 01/02 day-first and move
    the event by ten months, which no downstream check would catch."""
    row = _filing_rows(_filing(structured=True, event_date="01/02/2026"))[0]
    assert row["date_of_event"] == pd.Timestamp("2026-01-02")


# --------------------------------------------------------------------------- #
# The reporting-person CIK backfill -- the escalation join key                  #
# --------------------------------------------------------------------------- #
def test_post_mandate_cik_is_backfilled_from_the_header():
    """edgartools hard-codes `cik=''` on the 13G XML path (`# Not provided in 13G cover page`),
    so without this the escalation key is NULL exactly where the numbers are real."""
    row = _filing_rows(_filing(
        structured=True, persons=[_rp(name="Vanguard Capital Management", cik="")],
        filers=[("0002100119", "VANGUARD CAPITAL MANAGEMENT LLC")]))[0]
    assert row["reporting_person_cik"] == "0002100119"


def test_no_cik_beats_every_fallback():
    """`no_cik` is the filer's own assertion that it HAS no CIK. Inventing one from the header
    would attribute the stake to whichever entity transmitted the filing."""
    row = _filing_rows(_filing(
        structured=True, persons=[_rp(name="A Natural Person", cik="", no_cik=True)],
        filers=[("0000999999", "SOME FILING AGENT LLC")]))[0]
    assert row["reporting_person_cik"] is None


def test_cik_backfill_prefers_name_match_over_position():
    """Two persons, two filers, listed in DIFFERENT orders: position would cross-assign them."""
    persons = [_rp(name="Beta Advisers", cik=""), _rp(name="Alpha Capital", cik="")]
    filers = [("0000000111", "ALPHA CAPITAL LLC"), ("0000000222", "BETA ADVISERS LP")]
    rows = _filing_rows(_filing(structured=True, persons=persons, filers=filers))
    assert [r["reporting_person_cik"] for r in rows] == ["0000000222", "0000000111"]
    assert [r["rp_seq"] for r in rows] == [0, 1]


def test_cik_backfill_declines_rather_than_guessing():
    """Names that match nothing AND a filer list of a different length -> None, not a guess."""
    assert _reporting_person_cik(
        _rp(name="Unrelated Fund", cik=""), 0,
        [("0000000111", "ALPHA CAPITAL LLC"), ("0000000222", "BETA ADVISERS LP")], 1) is None


def test_norm_entity_strips_legal_form_but_not_identity():
    assert _norm_entity("VANGUARD CAPITAL MANAGEMENT LLC") == "vanguard capital"
    assert _norm_entity("Vanguard Capital Management") == "vanguard capital"
    assert _norm_entity(None) == "" and _norm_entity("  ") == ""


# --------------------------------------------------------------------------- #
# Shape                                                                         #
# --------------------------------------------------------------------------- #
def test_multiple_reporting_persons_share_the_filing_level_fields():
    persons = [_rp(name="Fund A", cik="0000000001", pct=3.0),
               _rp(name="Fund B", cik="0000000002", pct=2.5)]
    rows = _filing_rows(_filing(structured=True, persons=persons, cusip="123456789"))
    assert [r["rp_seq"] for r in rows] == [0, 1]
    assert {r["cusip"] for r in rows} == {"123456789"}
    assert {r["accession_number"] for r in rows} == {"0000315066-24-002743"}
    assert [r["percent_of_class"] for r in rows] == [3.0, 2.5]


def test_empty_reporting_persons_yields_one_nan_fallback_row():
    """NaN, not None: an all-None column is inferred as SQL TEXT by `ensure_table` the first
    time a batch of these seeds a cold table."""
    rows = _filing_rows(_filing(structured=True, persons=[]))
    assert len(rows) == 1 and rows[0]["rp_seq"] == 0
    assert rows[0]["reporting_person_name"] is None
    assert all(pd.isna(rows[0][col]) for col in _NUMERIC_COLS)


# --------------------------------------------------------------------------- #
# The issuer/filer guard                                                        #
# --------------------------------------------------------------------------- #
def _patch_new_filings(monkeypatch, filings):
    monkeypatch.setattr(
        "src.data_extract.utils.institutionals.fetch_13g_edgar.new_filings",
        lambda ticker, forms, since, done: filings)


def test_guard_drops_filings_where_the_ticker_is_the_filer(monkeypatch):
    """JNJ's own 13G listing is 159 filings, 60 of them JNJ disclosing stakes in Rallybio, CVRx
    and Rapport Therapeutics. Kept, every field would describe a different company."""
    own = _filing(issuer_cik="0000200406", issuer_name="JOHNSON & JOHNSON")
    other = _filing(issuer_cik="0001739410", issuer_name="Rallybio Corporation",
                    accession="0000904454-26-000233")
    _patch_new_filings(monkeypatch, [own, other])
    frame = build_ticker_13g_edgar("JNJ", "0000200406")[Tables.sec_13g]
    assert len(frame) == 1
    assert frame.iloc[0]["issuer_name"] == "JOHNSON & JOHNSON"
    assert list(frame.columns) == _COLS
    print("\n=== SANITY: 13G issuer/filer guard ===")
    print("  2 listed filings -> 1 kept; the one naming another issuer is dropped. Validated.")


def test_guard_does_not_reject_when_either_cik_is_unresolvable(monkeypatch):
    """An unknown CIK on either side means "unknown", which must not reject -- otherwise a
    header that failed to parse would silently cost the ticker its whole history."""
    _patch_new_filings(monkeypatch, [_filing(issuer_cik="", issuer_name="")])
    assert len(build_ticker_13g_edgar("JNJ", "0000200406")[Tables.sec_13g]) == 1
    _patch_new_filings(monkeypatch, [_filing(issuer_cik="0001739410")])
    assert len(build_ticker_13g_edgar("JNJ", "")[Tables.sec_13g]) == 1


if __name__ == "__main__":
    test_pre_mandate_zeros_become_nan_while_identity_survives()
    test_post_mandate_numbers_pass_through_with_rule_designation()
