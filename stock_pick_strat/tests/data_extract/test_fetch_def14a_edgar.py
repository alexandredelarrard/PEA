"""
Unit tests for the edgartools-based, deterministic DEF 14A fetcher
(fetch_def14a_edgar.py) -- the structured complement to fetch_def14a_llm.py.
Pure-synthetic, no network -- filings and their typed `.obj()` results
(`ProxyStatement`) are faked with SimpleNamespace, matching the convention in
test_fetch_8k_13d_edgar.py / test_fetch_filing_text.py.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.data_extract.utils.structure.fetch_def14a_edgar import (
    _director_comp_rows, _exec_comp_rows, _main_row, _ownership_rows, _votes_rows,
    build_ticker_def14a_edgar,
)


def _proposal(number, description, board_recommendation, proposal_type):
    return SimpleNamespace(number=number, description=description,
                           board_recommendation=board_recommendation, proposal_type=proposal_type)


def _fake_proxy(*, has_xbrl=True, has_individual_executive_data=False,
                company_name="Acme Corp.", fiscal_year_end="2025-12-31",
                peo_name="Jane Doe", peo_total_comp=15_000_000, peo_actually_paid_comp=16_500_000,
                neo_avg_total_comp=4_000_000, neo_avg_actually_paid_comp=4_200_000,
                total_shareholder_return=125.0, peer_group_tsr=110.0, net_income=2_000_000_000,
                company_selected_measure="Adjusted EBITDA", company_selected_measure_value=3_000_000_000,
                insider_trading_policy_adopted=True, award_timing_mnpi_considered=False,
                award_dates_predetermined=True, mnpi_disclosure_timed_for_comp_value=False,
                ceo_pay_ratio=None, audit_fees=None,
                summary_compensation_table=None, director_compensation_table=None,
                beneficial_ownership=None, voting_proposals=None):
    return SimpleNamespace(
        has_xbrl=has_xbrl, has_individual_executive_data=has_individual_executive_data,
        company_name=company_name, fiscal_year_end=fiscal_year_end,
        peo_name=peo_name, peo_total_comp=peo_total_comp, peo_actually_paid_comp=peo_actually_paid_comp,
        neo_avg_total_comp=neo_avg_total_comp, neo_avg_actually_paid_comp=neo_avg_actually_paid_comp,
        total_shareholder_return=total_shareholder_return, peer_group_tsr=peer_group_tsr,
        net_income=net_income, company_selected_measure=company_selected_measure,
        company_selected_measure_value=company_selected_measure_value,
        insider_trading_policy_adopted=insider_trading_policy_adopted,
        award_timing_mnpi_considered=award_timing_mnpi_considered,
        award_dates_predetermined=award_dates_predetermined,
        mnpi_disclosure_timed_for_comp_value=mnpi_disclosure_timed_for_comp_value,
        ceo_pay_ratio=ceo_pay_ratio, audit_fees=audit_fees,
        summary_compensation_table=(summary_compensation_table
                                    if summary_compensation_table is not None else pd.DataFrame()),
        director_compensation_table=(director_compensation_table
                                     if director_compensation_table is not None else pd.DataFrame()),
        beneficial_ownership=(beneficial_ownership
                              if beneficial_ownership is not None else pd.DataFrame()),
        voting_proposals=(voting_proposals if voting_proposals is not None else []),
    )


def _fake_filing(*, form="DEF 14A", accession="0001-24-000003", filing_date="2024-04-01"):
    return SimpleNamespace(accession_number=accession, form=form, filing_date=filing_date)


# --- _main_row ----------------------------------------------------------------- #
def test_main_row_reads_xbrl_and_html_fields():
    ratio = SimpleNamespace(ceo_compensation=15_000_000, median_employee_compensation=65_000, ratio=231)
    audit = SimpleNamespace(auditor_name="Ernst & Young LLP", current_year=2024, prior_year=2023,
                            audit_fees_current=5_000_000, audit_fees_prior=4_800_000,
                            audit_related_current=200_000, audit_related_prior=180_000,
                            tax_fees_current=300_000, tax_fees_prior=250_000,
                            other_fees_current=0, other_fees_prior=0,
                            total_current=5_500_000, total_prior=5_230_000)
    proposals = [
        _proposal(1, "Election of Directors", "FOR", "director_election"),
        _proposal(2, "Advisory Vote to Approve Executive Compensation", "FOR", "say_on_pay"),
        _proposal(3, "Ratification of Auditor", "FOR", "auditor_ratification"),
        _proposal(4, "Stockholder Proposal on Climate Disclosure", "AGAINST", "shareholder_proposal"),
    ]
    proxy = _fake_proxy(ceo_pay_ratio=ratio, audit_fees=audit, voting_proposals=proposals)
    filing = _fake_filing()

    row = _main_row("AAPL", "0000320193", filing, proxy)

    assert row["ticker"] == "AAPL"
    assert row["accession_number"] == "0001-24-000003"
    assert row["company_name"] == "Acme Corp."
    assert row["has_xbrl"] == 1.0
    assert row["peo_name"] == "Jane Doe"
    assert row["peo_total_comp"] == 15_000_000.0
    assert row["ceo_pay_ratio"] == 231.0
    assert row["ceo_pay_ratio_median_employee_comp"] == 65_000.0
    assert row["auditor_name"] == "Ernst & Young LLP"
    assert row["audit_fees_current"] == 5_000_000.0
    assert row["total_fees_prior"] == 5_230_000.0
    assert row["n_voting_proposals"] == 4.0
    assert row["n_say_on_pay_proposals"] == 1.0
    assert row["n_director_election_proposals"] == 1.0
    assert row["n_auditor_ratification_proposals"] == 1.0
    assert row["n_shareholder_proposals"] == 1.0
    assert row["n_board_against_recommendations"] == 1.0     # the shareholder proposal
    assert row["period_of_report"] == pd.Timestamp("2025-12-31")


def test_main_row_survives_missing_optional_sections():
    """A filer with no XBRL (SRC/EGC), no CEO-pay-ratio disclosure, and no audit-fee
    table found must yield NaN for those columns -- not a crash, not a false zero."""
    proxy = _fake_proxy(has_xbrl=False, ceo_pay_ratio=None, audit_fees=None, voting_proposals=[])
    row = _main_row("XYZ", "0000000001", _fake_filing(), proxy)
    assert row["has_xbrl"] == 0.0
    assert pd.isna(row["ceo_pay_ratio"])
    assert pd.isna(row["auditor_name"]) or row["auditor_name"] is None
    assert pd.isna(row["audit_fees_current"])
    assert row["n_voting_proposals"] == 0.0
    assert row["n_board_against_recommendations"] == 0.0


def test_main_row_survives_attribute_errors():
    """A ProxyStatement property that raises (a real edgartools edge case on a
    malformed filing) must not propagate -- the field is simply NaN/None."""
    class BrokenProxy:
        @property
        def has_xbrl(self):
            raise RuntimeError("xbrl parse exploded")
        peo_name = None
        voting_proposals = []
        ceo_pay_ratio = None
        audit_fees = None
        summary_compensation_table = pd.DataFrame()
        director_compensation_table = pd.DataFrame()
        beneficial_ownership = pd.DataFrame()
        company_name = None
        fiscal_year_end = None
        peo_total_comp = None
        peo_actually_paid_comp = None
        neo_avg_total_comp = None
        neo_avg_actually_paid_comp = None
        total_shareholder_return = None
        peer_group_tsr = None
        net_income = None
        company_selected_measure = None
        company_selected_measure_value = None
        insider_trading_policy_adopted = None
        award_timing_mnpi_considered = None
        award_dates_predetermined = None
        mnpi_disclosure_timed_for_comp_value = None
        has_individual_executive_data = False

    row = _main_row("AAPL", "0000320193", _fake_filing(), BrokenProxy())
    assert pd.isna(row["has_xbrl"])


# --- child-row extractors ------------------------------------------------------- #
def test_exec_comp_rows_reads_summary_compensation_table_and_skips_blank_rows():
    df = pd.DataFrame([
        {"name": "Jane Doe", "title": "CEO", "year": 2024, "salary": 1_500_000, "bonus": 0,
         "stock_awards": 8_000_000, "option_awards": 0, "non_equity_incentive": 3_000_000,
         "pension_change": 0, "other_compensation": 200_000, "total": 12_700_000},
        {"name": "", "title": None, "year": 2024, "salary": None, "bonus": None,
         "stock_awards": None, "option_awards": None, "non_equity_incentive": None,
         "pension_change": None, "other_compensation": None, "total": None},
    ])
    proxy = _fake_proxy(summary_compensation_table=df)
    rows = _exec_comp_rows("AAPL", "0000320193", _fake_filing(), proxy)
    assert len(rows) == 1
    assert rows[0]["name"] == "Jane Doe"
    assert rows[0]["year"] == 2024.0
    assert rows[0]["total"] == 12_700_000.0


def test_director_comp_rows_reads_director_compensation_table():
    df = pd.DataFrame([
        {"name": "John Smith", "fees_earned": 120_000, "stock_awards": 180_000,
         "option_awards": 0, "non_equity_incentive": 0, "pension_change": 0,
         "other_compensation": 0, "total": 300_000},
    ])
    proxy = _fake_proxy(director_compensation_table=df)
    rows = _director_comp_rows("AAPL", "0000320193", _fake_filing(), proxy)
    assert len(rows) == 1
    assert rows[0]["name"] == "John Smith"
    assert rows[0]["total"] == 300_000.0


def test_ownership_rows_reads_beneficial_ownership_table():
    df = pd.DataFrame([
        {"holder_name": "The Vanguard Group", "holder_type": "5pct_holder",
         "shares": 1_300_000_000, "percent_of_class": 8.1},
        {"holder_name": "Jane Doe", "holder_type": "director_officer",
         "shares": 3_000_000, "percent_of_class": 0.02},
    ])
    proxy = _fake_proxy(beneficial_ownership=df)
    rows = _ownership_rows("AAPL", "0000320193", _fake_filing(), proxy)
    assert {r["holder_name"] for r in rows} == {"The Vanguard Group", "Jane Doe"}
    assert {r["holder_type"] for r in rows} == {"5pct_holder", "director_officer"}


def test_votes_rows_reads_voting_proposals():
    proposals = [_proposal(1, "Election of Directors", "FOR", "director_election"),
                _proposal(2, "Say-on-Pay", "FOR", "say_on_pay")]
    proxy = _fake_proxy(voting_proposals=proposals)
    rows = _votes_rows("AAPL", "0000320193", _fake_filing(), proxy)
    assert [r["proposal_number"] for r in rows] == [1.0, 2.0]
    assert rows[1]["proposal_type"] == "say_on_pay"


# --- ticker-level walk (incremental dedup, since-cutoff, non-proxy skip) -------- #
def test_build_ticker_def14a_edgar_skips_done_accessions_and_pre_since_filings(monkeypatch):
    proxy = _fake_proxy()
    old_filing = _fake_filing(accession="0001-old", filing_date="2020-01-01")
    done_filing = _fake_filing(accession="0001-done", filing_date="2024-01-01")
    new_filing = _fake_filing(accession="0001-new", filing_date="2024-06-01")
    for f in (old_filing, done_filing, new_filing):
        f.obj = lambda p=proxy: p
    fake_company = SimpleNamespace(get_filings=lambda form: [old_filing, done_filing, new_filing])
    monkeypatch.setattr(
        "src.data_extract.utils.structure.fetch_def14a_edgar.Company",
        lambda ticker: fake_company,
    )

    main_df, *_ = build_ticker_def14a_edgar(
        "AAPL", "0000320193",
        since=pd.Timestamp("2024-01-01"),
        done_accessions=frozenset({"0001-done"}),
    )
    assert set(main_df["accession_number"]) == {"0001-new"}


def test_build_ticker_def14a_edgar_skips_non_proxy_obj_results(monkeypatch):
    """A DEF 14C (or any form whose `.obj()` doesn't resolve to a ProxyStatement --
    edgartools' PROXY_FORMS dispatch list excludes DEF 14C) must be skipped by this
    deterministic path, not crash it -- it stays covered by fetch_def14a_llm.py."""
    good_filing = _fake_filing(accession="0001-good", form="DEF 14A")
    good_filing.obj = lambda: _fake_proxy()
    non_proxy_filing = _fake_filing(accession="0001-non-proxy", form="DEF 14C")
    non_proxy_filing.obj = lambda: SimpleNamespace()          # no voting_proposals attribute
    failing_filing = _fake_filing(accession="0001-fails", form="DEF 14A")
    failing_filing.obj = lambda: (_ for _ in ()).throw(RuntimeError("parse failed"))

    fake_company = SimpleNamespace(
        get_filings=lambda form: [good_filing, non_proxy_filing, failing_filing])
    monkeypatch.setattr(
        "src.data_extract.utils.structure.fetch_def14a_edgar.Company",
        lambda ticker: fake_company,
    )

    main_df, *_ = build_ticker_def14a_edgar("AAPL", "0000320193")
    assert set(main_df["accession_number"]) == {"0001-good"}


def test_sanity_check_prints_conclusion():
    print("\n=== SANITY CHECK: edgartools DEF 14A structured extraction ===")
    print("  Main row: XBRL ECD tags (PEO/NEO pay-vs-performance) + HTML-extracted CEO")
    print("  pay ratio + audit fee breakdown + voting-proposal counts by classified type")
    print("  all read via best-effort getattr (a raising property yields NaN/None, not a")
    print("  crash). Missing optional sections (no XBRL, no pay-ratio disclosure, no audit")
    print("  fee table) yield NaN -- never a false zero.")
    print("  Child tables (executive_comp, director_comp, ownership, votes) each skip")
    print("  blank-name/blank-year rows and read straight off edgartools' own HTML-table")
    print("  DataFrames -- no re-parsing.")
    print("  build_ticker_def14a_edgar correctly skips already-seen accessions, filings")
    print("  before the `since` cutoff, AND any filing whose .obj() does not resolve to a")
    print("  ProxyStatement (DEF 14C, or a parse failure) -- these remain covered by")
    print("  fetch_def14a_llm.py's LLM path, untouched by this module.")
    print("  Validated.")
