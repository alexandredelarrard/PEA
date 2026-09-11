"""`sec_def14a` -- the ECD (Pay-versus-Performance) inline-XBRL path.

These tests run on REAL facts frames captured from live filings into
`fixtures/ecd_*.parquet`. That is deliberate and it is the point of the file: the previous
version of this module was built entirely from `SimpleNamespace` fakes, which is exactly why
every defect the research later found was invisible to CI. A fake cannot carry a dimension the
code forgot to read, so a synthetic test of a dimension bug always passes.

Each fixture is one measured tagging shape:

  ecd_ba_2025    co-PEO year (Ortberg + Calhoun), IndividualAxis ONLY, one CAP at -23,875,735
  ecd_nke_2025   co-PEO year where the SELECTED PEO's CAP is NEGATIVE (-10,924,243)
  ecd_nke_2026   an UNDIMENSIONED duplicate of the dimensioned current-year total
  ecd_sbux_2026  individual x year matrix with 0.0 in every non-applicable cell
  ecd_aapl_2026  26 PeoName facts on BOTH axes, but amounts fully UNDIMENSIONED
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data_extract.utils.structure.def14a.ecd import (
    _CATEGORY_AXIS, _PEO_MEMBER, _PEO_TOTAL, ecd_row, has_ecd_block, latest_period, peo_block,
)
from src.data_extract.utils.structure.fetch_def14a_edgar import build_ticker_def14a_edgar
from src.data_store.schema import Tables

FIXTURES = Path(__file__).parent / "fixtures"


def _facts(name: str) -> pd.DataFrame:
    return pd.read_parquet(FIXTURES / f"ecd_{name}.parquet")


# --------------------------------------------------------------------------- #
# the dimension rule                                                          #
# --------------------------------------------------------------------------- #
def test_a_fixed_peo_member_filter_would_return_nothing():
    """The negative control, and the reason the axis filter is CONDITIONAL.

    The obvious implementation -- always filter `ExecutiveCategoryAxis == 'ecd:PeoMember'` --
    returns an EMPTY frame on BA, NKE and SBUX, because those filers put the executive on
    `IndividualAxis` alone. It would have failed on exactly the co-PEO filings the filter exists
    to fix, and the failure mode is a silent NULL, not an error.
    """
    counts = {}
    for name in ("ba_2025", "nke_2025", "nke_2026", "sbux_2026", "aapl_2026"):
        facts = _facts(name)
        totals = facts[facts["concept"].astype(str) == _PEO_TOTAL]
        with_axis = int(totals[_CATEGORY_AXIS].notna().sum()) if _CATEGORY_AXIS in totals else 0
        fixed = int((totals.get(_CATEGORY_AXIS, pd.Series(dtype=object)) == _PEO_MEMBER).sum())
        counts[name] = (len(totals), with_axis, fixed)

    print("\n=== SANITY: would a fixed 'ecd:PeoMember' filter work? ===")
    print(f"  {'fixture':<14}{'PeoTotalCompAmt':>17}{'with the axis':>15}{'kept by ==':>12}")
    for name, (n, with_axis, fixed) in counts.items():
        print(f"  {name:<14}{n:>17}{with_axis:>15}{fixed:>12}")
    print("  Every fixture keeps 0 -- so the axis is applied only when the concept's own facts")
    print("  actually carry it, and the conditional rule is load-bearing, not defensive.")

    assert all(fixed == 0 for _n, _a, fixed in counts.values())
    assert all(n > 0 for n, _a, _f in counts.values())


def test_co_peo_year_keeps_both_names_and_the_larger_total():
    """BA's 2025 proxy covers FY2024 with two PEOs. `ProxyStatement` keeps whichever comes first
    in document order and silently drops the other."""
    row = peo_block(_facts("ba_2025"))
    assert row["n_peos"] == 2.0
    assert row["peo_total_comp"] == 18_388_629.0            # Ortberg > Calhoun's 15,050,812
    assert row["peo_actually_paid_comp"] == 19_904_513.0    # Ortberg's CAP, not Calhoun's
    assert "Ortberg" in row["peo_names_all"] and "Calhoun" in row["peo_names_all"]
    assert row["ecd_period_end"] == pd.Timestamp("2024-12-31")

    print("\n=== SANITY: co-PEO year (BA 2025 proxy, FY2024) ===")
    print(f"  n_peos={row['n_peos']}, kept {row['peo_name']} at {row['peo_total_comp']:,.0f}, "
          f"CAP {row['peo_actually_paid_comp']:,.0f}")
    print(f"  peo_names_all={row['peo_names_all']} -- the co-PEO is VISIBLE, not halved.")


def test_a_negative_cap_survives_selection():
    """NKE's 2025 proxy: the retained PEO (Donahoe, 28,442,712 -- the larger total) has a CAP of
    -10,924,243. Compensation Actually Paid subtracts prior-year unvested fair value, so a
    share-price fall makes it negative; 28.5% of 2023 and 33.7% of 2025 S&P 500 proxies report
    at least one. An `abs()` anywhere on this path would silently invert a real disclosure."""
    row = peo_block(_facts("nke_2025"))
    assert row["n_peos"] == 2.0
    assert row["peo_total_comp"] == 28_442_712.0
    assert row["peo_actually_paid_comp"] == -10_924_243.0
    print("\n=== SANITY: negative Compensation Actually Paid ===")
    print(f"  {row['peo_name']}: total {row['peo_total_comp']:,.0f}, "
          f"CAP {row['peo_actually_paid_comp']:,.0f} -- sign preserved.")
    print("  Selecting on the largest TOTAL also means the retained name can be the DEPARTING")
    print("  CEO (Donahoe over the incumbent Hill); n_peos=2 is what says so.")


def test_the_zero_matrix_cell_is_never_selected():
    """SBUX tags a full individual x year matrix with 0.0 in every cell where the person was not
    PEO that year. Dropping the zeros BEFORE selecting recovers the correct value; a post-hoc
    "a PEO is never paid $0" guard would only leave a NULL."""
    facts = _facts("sbux_2026")
    row = peo_block(facts)
    period = latest_period(facts)
    raw = facts[(facts["concept"].astype(str) == _PEO_TOTAL)
                & (pd.to_datetime(facts["period_end"], errors="coerce") == period)]
    n_zero = int((pd.to_numeric(raw["value"], errors="coerce") == 0).sum())

    assert row["peo_total_comp"] == 30_992_773.0
    assert "Niccol" in row["peo_name"]
    assert row["n_peos"] == 1.0
    print("\n=== SANITY: the SBUX zero matrix ===")
    print(f"  {len(raw)} PeoTotalCompAmt facts in the covered year, {n_zero} of them exactly 0")
    print(f"  selected {row['peo_name']} at {row['peo_total_comp']:,.0f} (not a NULL, not a 0)")
    assert n_zero >= 3


def test_undimensioned_amounts_still_get_the_right_name():
    """AAPL is the opposite shape: its `ecd:PeoName` facts are dimensioned on BOTH axes (26 of
    them, 21 tagged `NonPeoNeoMember` -- these are the NEOs the library reads undimensioned and
    mistakes for PEOs), while `ecd:PeoTotalCompAmt` carries no dimension at all. So the name and
    the amount cannot be joined on an individual key, and are resolved independently."""
    facts = _facts("aapl_2026")
    row = peo_block(facts)
    names = facts[facts["concept"].astype(str) == "ecd:PeoName"]
    n_neo_tagged = int((names[_CATEGORY_AXIS] != _PEO_MEMBER).sum())

    assert row["peo_total_comp"] == 74_294_811.0
    assert row["n_peos"] == 1.0
    assert "Cook" in row["peo_name"]
    for neo in ("Maestri", "Adams", "Williams", "Parekh", "Khan"):
        assert neo not in (row["peo_name"] or ""), f"{neo} is an NEO, not the PEO"
    print("\n=== SANITY: undimensioned amounts, dimensioned names (AAPL 2026) ===")
    print(f"  {len(names)} ecd:PeoName facts, {n_neo_tagged} of them NonPeoNeoMember")
    print(f"  peo_name={row['peo_name']} at {row['peo_total_comp']:,.0f}; no NEO leaked in.")
    assert n_neo_tagged >= 20


def test_an_undimensioned_duplicate_is_not_a_second_peo():
    """NKE's 2026 proxy tags FY2026's PEO total twice -- once on `nke:ElliottHillMember` and once
    with no dimension, same 36,340,876. Counting both reads as two PEOs, one of them nameless."""
    facts = _facts("nke_2026")
    period = latest_period(facts)
    raw = facts[(facts["concept"].astype(str) == _PEO_TOTAL)
                & (pd.to_datetime(facts["period_end"], errors="coerce") == period)]
    row = peo_block(facts)

    assert len(raw) == 2, "the fixture is meant to contain the duplicate"
    assert row["n_peos"] == 1.0
    assert row["peo_total_comp"] == 36_340_876.0
    assert "Hill" in row["peo_name"]
    print("\n=== SANITY: undimensioned duplicate (NKE 2026) ===")
    print(f"  {len(raw)} facts for the covered year, same value; n_peos={row['n_peos']} "
          f"({row['peo_name']})")


# --------------------------------------------------------------------------- #
# scope: a filing with no ECD facts must not produce a row                    #
# --------------------------------------------------------------------------- #
def test_no_ecd_facts_means_no_row():
    """Item 402(v) applies to fiscal years ending on or after 2022-12-16. AAPL's 2023-01-12
    proxy covers FY2022 (ended 2022-09-24) and `filing.xbrl()` returns nothing at all -- so
    there is no row to write. `has_xbrl` was dropped as a column for exactly this reason: a
    table that only ever holds tagged filings makes the flag degenerate."""
    assert has_ecd_block(None) is False
    assert has_ecd_block(pd.DataFrame()) is False
    assert has_ecd_block(pd.DataFrame({"concept": ["dei:EntityRegistrantName"]})) is False
    assert has_ecd_block(_facts("ba_2025")) is True
    print("\n=== SANITY: pre-402(v) scope ===")
    print("  no ecd: concepts -> no row (the proxy inventory is def14a_llm's job, and it")
    print("  covers all of them).")


def test_ecd_row_carries_every_registered_column():
    row = ecd_row(_facts("ba_2025"))
    for col in ("peo_name", "peo_total_comp", "peo_actually_paid_comp", "n_peos",
                "peo_names_all", "ecd_period_end", "neo_avg_total_comp",
                "neo_avg_actually_paid_comp", "total_shareholder_return", "peer_group_tsr",
                "net_income", "company_selected_measure_name", "company_selected_measure_value",
                "insider_trading_policy_adopted", "award_timing_mnpi_considered",
                "award_dates_predetermined", "mnpi_disclosure_timed_for_comp_value",
                "has_individual_executive_data"):
        assert col in row, f"{col} missing from the ECD row"
    print("\n=== SANITY: ECD row shape ===")
    print(f"  {len(row)} columns, all registered on sec_def14a.")


# --------------------------------------------------------------------------- #
# ticker-level walk (incremental dedup + since cutoff)                        #
# --------------------------------------------------------------------------- #
_FIXTURE = "ba_2025"


def _fake_filing(*, accession: str, filing_date: str, form: str = "DEF 14A"):
    """A filing whose `.xbrl()` yields a REAL facts frame -- so the driver test also exercises
    the ECD path rather than asserting against a mock of it."""
    facts = _facts(_FIXTURE)
    return SimpleNamespace(
        accession_number=accession, accession_no=accession, form=form,
        filing_date=pd.Timestamp(filing_date).date(),
        period_of_report="2024-12-31", company="THE BOEING COMPANY",
        xbrl=lambda: SimpleNamespace(facts=SimpleNamespace(to_dataframe=lambda: facts)),
    )


def test_build_ticker_skips_done_accessions_and_pre_since_filings(monkeypatch):
    filings = [_fake_filing(accession="0001-old", filing_date="2020-01-01"),
               _fake_filing(accession="0001-done", filing_date="2024-01-01"),
               _fake_filing(accession="0001-new", filing_date="2024-06-01")]
    monkeypatch.setattr("edgar.Company",
                        lambda ticker: SimpleNamespace(get_filings=lambda form: filings))

    df = build_ticker_def14a_edgar(
        "BA", "0000012927", since=pd.Timestamp("2024-01-01"),
        done_accessions=frozenset({"0001-done"}))[Tables.def14a_edgar]

    assert set(df["accession_number"]) == {"0001-new"}
    assert df["n_peos"].iloc[0] == 2.0                       # the real frame really was read
    print("\n=== SANITY: incremental walk ===")
    print(f"  3 filings offered, 1 written ({set(df['accession_number'])}); "
          f"n_peos={df['n_peos'].iloc[0]} proves a real facts frame went through.")


def test_a_filing_without_xbrl_is_skipped_not_crashed(monkeypatch):
    """DEF 14C is not in edgartools' `PROXY_FORMS` dispatch, so the old code's
    `hasattr(proxy, "voting_proposals")` guard silently skipped every one of them. Going straight
    to `filing.xbrl()` treats both forms alike -- what decides now is whether the filing carries
    ECD facts, which is the regulatory question and not a library artefact."""
    good = _fake_filing(accession="0001-good", filing_date="2025-03-07")
    empty = _fake_filing(accession="0001-empty", filing_date="2025-03-08", form="DEF 14C")
    empty.xbrl = lambda: None
    raising = _fake_filing(accession="0001-raises", filing_date="2025-03-09")
    raising.xbrl = lambda: (_ for _ in ()).throw(RuntimeError("xbrl parse failed"))

    monkeypatch.setattr("edgar.Company",
                        lambda ticker: SimpleNamespace(
                            get_filings=lambda form: [good, empty, raising]))

    df = build_ticker_def14a_edgar("BA", "0000012927")[Tables.def14a_edgar]
    assert set(df["accession_number"]) == {"0001-good"}
    print("\n=== SANITY: unreadable filings ===")
    print("  xbrl() -> None and xbrl() -> raise both skip the row; neither crashes the walk.")


def test_company_name_falls_back_to_the_filing_index(monkeypatch):
    """edgartools reads `dei:EntityRegistrantName` XBRL-only with NO fallback to the filing
    index, and DEF 14A has no mandatory cover-page iXBRL requirement -- so it was None on every
    pre-2023 filing and `__str__` substituted the literal "Unknown Company"."""
    f = _fake_filing(accession="0001-x", filing_date="2025-03-07")
    facts = _facts(_FIXTURE)
    stripped = facts[facts["concept"].astype(str) != "dei:EntityRegistrantName"]
    f.xbrl = lambda: SimpleNamespace(facts=SimpleNamespace(to_dataframe=lambda: stripped))
    monkeypatch.setattr("edgar.Company",
                        lambda ticker: SimpleNamespace(get_filings=lambda form: [f]))

    df = build_ticker_def14a_edgar("BA", "0000012927")[Tables.def14a_edgar]
    assert df["company_name"].iloc[0] == "THE BOEING COMPANY"
    print("\n=== SANITY: company_name fallback ===")
    print(f"  dei tag removed from the frame -> company_name={df['company_name'].iloc[0]!r} "
          f"from the filing index.")


def test_the_retired_html_columns_are_gone():
    """Every column the HTML parser fed. `ceo_pay_ratio` moved to the LLM path because it is
    narrative-only FOREVER (Rel. 33-9877 created no XBRL tag) and edgartools extracted it with
    three value-inventing repairs; the fee block and the proposal counters went with the parser
    that produced them."""
    row = ecd_row(_facts("ba_2025"))
    gone = [c for c in ("has_xbrl", "auditor_name", "ceo_pay_ratio", "ceo_pay_ratio_ceo_comp",
                        "audit_fees_current", "total_fees_current", "n_voting_proposals",
                        "n_say_on_pay_proposals", "n_board_against_recommendations")
            if c in row]
    print("\n=== SANITY: retired HTML columns ===")
    print(f"  still present on the ECD row: {gone or 'none'}")
    assert gone == []


if __name__ == "__main__":                                   # pragma: no cover
    pytest.main([__file__, "-s"])
